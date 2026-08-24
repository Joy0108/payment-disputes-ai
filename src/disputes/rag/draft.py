"""Draft response generation, and the citation contract it must satisfy.

The generator writes prose. It does not compute dates, does not decide
liability, and does not invent a citation: the deadlines arrive from the rules
engine already computed, the retrieved sections arrive already resolved, and the
generator's job is to put them into a letter a consumer can read.

Enforcement is on the output. Every sentence that asserts a fact must carry a
citation that resolves to a retrieved section or to a computed deadline, and any
date appearing in the draft must match one the rules engine produced. A draft
failing either check is rejected and regenerated, not published with a caveat -
this is a letter that goes to a consumer and, if it is wrong about a date, to a
regulator afterwards.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

CITATION = re.compile(r"\[(?:reg|calc):([A-Za-z0-9_.()\-]+)\]")
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Prompts are versioned because a prompt change is a model change: it moves the
# output distribution, and an eval run has to name which one produced it.
PROMPT_VERSIONS = {
    "v1": "Answer the consumer's dispute using the regulation text provided.",
    "v2": "Answer the consumer's dispute using the regulation text provided. Cite every claim.",
    "v3": """You draft responses to consumer payment disputes for a financial institution.

You are given: the consumer's complaint, the governing regulation sections retrieved for it, and a set of
deadlines already computed by a deterministic rules engine.

Rules:
- Every sentence that states a fact carries a citation: [reg:<section_id>] for a regulation, [calc:<deadline_name>] for a computed date.
- Never compute or adjust a date yourself. Every date in your response must be one supplied in the deadlines block, quoted exactly.
- Never state a liability amount that is not in the retrieved sections.
- If the retrieved sections do not cover something the consumer asked, say so rather than filling the gap.
- Plain language. The reader is a consumer, not a lawyer. Explain what happens next and by when.
- You draft; you do not decide the outcome of the dispute.""",
}


@dataclass
class Draft:
    text: str
    backend: str
    prompt_version: str
    citations: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def cited(self) -> list[str]:
        return CITATION.findall(self.text)

    def dates(self) -> list[str]:
        return DATE.findall(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "prompt_version": self.prompt_version,
            "citations": self.cited(),
            "dates": self.dates(),
            "text": self.text,
        }


class Drafter(Protocol):
    name: str

    def draft(self, context: dict[str, Any], critique: str | None = None) -> Draft: ...


class TemplateDrafter:
    """Deterministic drafter. What CI runs, so the eval numbers are reproducible."""

    name = "template"

    def __init__(self, prompt_version: str = "v3"):
        self.prompt_version = prompt_version

    def draft(self, context: dict[str, Any], critique: str | None = None) -> Draft:
        complaint = context["complaint"]
        deadlines = context.get("deadlines", {})
        sections: Sequence[dict[str, Any]] = context.get("sections", [])
        validation = context.get("validation", {})
        reason_code = context.get("reason_code", {})

        lines = [
            f"Re: dispute {complaint.get('complaint_id', 'n/a')} - {complaint.get('issue', 'unspecified issue')}",
            "",
            "Thank you for contacting us. We have opened an investigation into the transaction you describe.",
            "",
            "## What applies to your dispute",
            "",
        ]

        for section in sections[:3]:
            summary = section["text"].split(". ")[0].rstrip(".")
            lines.append(f"- {summary} [reg:{section['id']}].")

        if deadlines.get("consumer_notice_timely") is False:
            lines += [
                "",
                "We have recorded that your notice reached us after the notice period closed. We will still review "
                "what happened, but the statutory error resolution timetable does not apply to a notice received "
                f"outside that window [reg:{_first_citation(deadlines)}].",
            ]
        else:
            lines += ["", "## Dates that apply", ""]
            for deadline in deadlines.get("deadlines", []):
                lines.append(
                    f"- {deadline['name'].capitalize()}: {deadline['due']} "
                    f"({deadline['basis']}) [calc:{_slug(deadline['name'])}] [reg:{deadline['citation']}]."
                )

        findings = deadlines.get("findings", [])
        if findings:
            lines += ["", "## What this means", ""]
            for finding in findings:
                lines.append(f"- {finding}")

        if reason_code.get("mapped"):
            lines += [
                "",
                f"We have raised this with the card network under reason code {reason_code['code']} "
                f"({reason_code['label']}). To support it we need: {', '.join(reason_code['evidence'])}.",
            ]
        elif reason_code.get("code") or reason_code.get("issue"):
            lines += [
                "",
                "We could not map this dispute to a card network reason code automatically, so a specialist is "
                "reviewing which route applies.",
            ]

        problems = [i for i in validation.get("issues", []) if i["severity"] == "error"]
        if problems:
            lines += ["", "## Information we still need", ""]
            for problem in problems:
                lines.append(f"- {problem['message']}")

        lines += [
            "",
            "We will write to you again when the investigation concludes. If you have further documents, send them "
            "and we will add them to the file.",
        ]

        if critique:
            lines += ["", f"<!-- revised after review: {critique[:200]} -->"]

        return Draft("\n".join(lines), backend=self.name, prompt_version=self.prompt_version)


class ClaudeDrafter:  # pragma: no cover - requires credentials
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", prompt_version: str = "v3", max_tokens: int = 3000):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.fallback = TemplateDrafter(prompt_version)

    def draft(self, context: dict[str, Any], critique: str | None = None) -> Draft:
        import json

        payload = {
            "complaint": context["complaint"],
            "computed_deadlines": context.get("deadlines", {}),
            "retrieved_sections": context.get("sections", []),
            "validation": context.get("validation", {}),
            "reason_code": context.get("reason_code", {}),
        }
        user = json.dumps(payload, indent=2, default=str)
        if critique:
            user += f"\n\nThe previous draft was rejected for these reasons:\n{critique}\nFix each one."

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=PROMPT_VERSIONS[self.prompt_version],
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return Draft(text, backend=self.name, prompt_version=self.prompt_version,
                         usage={"input_tokens": response.usage.input_tokens,
                                "output_tokens": response.usage.output_tokens})
        except Exception as exc:
            draft = self.fallback.draft(context, critique)
            draft.backend = f"template ({type(exc).__name__} from the API)"
            return draft


def build_drafter(backend: str | None = None, prompt_version: str = "v3") -> Drafter:
    backend = backend or os.environ.get("DISPUTES_LLM", "template")
    if backend in {"anthropic", "claude"}:
        return ClaudeDrafter(prompt_version=prompt_version)
    return TemplateDrafter(prompt_version)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

@dataclass
class Verification:
    citations: int = 0
    resolved: int = 0
    unresolved: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    invented_dates: list[str] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.citations if self.citations else 1.0

    @property
    def passed(self) -> bool:
        return not self.unresolved and not self.invented_dates and not self.uncited_claims

    def critique(self) -> str:
        parts = []
        if self.unresolved:
            parts.append(f"citations that resolve to nothing: {', '.join(self.unresolved)}")
        if self.invented_dates:
            parts.append(
                f"dates not produced by the rules engine: {', '.join(self.invented_dates)}; "
                "quote only the supplied deadlines")
        if self.uncited_claims:
            parts.append(f"{len(self.uncited_claims)} factual sentence(s) with no citation")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "citations": self.citations,
            "resolution_rate": round(self.resolution_rate, 4),
            "unresolved": self.unresolved,
            "invented_dates": self.invented_dates,
            "uncited_claims": len(self.uncited_claims),
        }


_BOILERPLATE = (
    "thank you", "we will write", "if you have further", "re:", "we have opened",
    "a specialist is reviewing", "send them",
)


def verify(draft: Draft, valid_sections: set[str], deadlines: dict[str, Any]) -> Verification:
    """Citations resolve, dates are not invented, claims are attributed."""
    result = Verification()

    valid_calcs = {_slug(d["name"]) for d in deadlines.get("deadlines", [])}
    for citation in draft.cited():
        result.citations += 1
        if citation in valid_sections or citation in valid_calcs:
            result.resolved += 1
        else:
            result.unresolved.append(citation)

    # A date in a consumer letter that the rules engine did not produce is the
    # single most dangerous thing a generator here can do, so it is checked
    # against the computed set rather than for plausibility.
    allowed_dates = {d["due"] for d in deadlines.get("deadlines", [])}
    for value in draft.dates():
        if value not in allowed_dates:
            result.invented_dates.append(value)

    for sentence in SENTENCE.split(draft.text):
        stripped = sentence.strip()
        low = stripped.lower()
        if len(stripped.split()) < 8 or stripped.startswith(("#", "<!--", "|")):
            continue
        if any(marker in low for marker in _BOILERPLATE):
            continue
        if _asserts_fact(low) and not CITATION.search(stripped):
            result.uncited_claims.append(stripped[:160])
    return result


_FACT_MARKERS = ("must", "shall", "required", "days", "liability", "entitled", "may not", "within")


def _asserts_fact(sentence: str) -> bool:
    return any(marker in sentence for marker in _FACT_MARKERS)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _first_citation(deadlines: dict[str, Any]) -> str:
    citations = deadlines.get("citations") or []
    return citations[0] if citations else "1005.11(b)"

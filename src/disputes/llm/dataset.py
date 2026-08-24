"""Instruction-tuning dataset, and the curriculum over it.

The supervision target is not "write a nice letter". It is the specific thing a
general model gets wrong on this task: producing a fluent response that quietly
invents a date or cites a section that does not say what the letter claims. So
every training example pairs a complaint with a response whose dates come from
the rules engine and whose citations resolve to retrieved sections - the target
is a *grounded* response, and the grounding is generated, not annotated.

Difficulty, for the curriculum, is defined by what makes an example hard for
this task rather than by length:

* how many distinct regulatory deadlines the response has to carry;
* whether an extension branch applies (new account, point of sale, foreign);
* whether the consumer notice was out of time, which inverts the whole letter;
* whether validation found a problem that the letter must raise;
* how much of the narrative is boilerplate rather than signal.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ARTIFACT_DIR
from ..rag.draft import PROMPT_VERSIONS, TemplateDrafter
from ..rag.index import RegulationIndex
from ..rules import deadlines as rules
from ..rules.validate import map_reason_code, validate_amounts, validate_dates


@dataclass
class Example:
    example_id: str
    instruction: str
    input: str
    output: str
    difficulty: float
    difficulty_parts: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chat(self) -> dict[str, Any]:
        """The chat format most SFT trainers expect."""
        return {
            "messages": [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": self.input},
                {"role": "assistant", "content": self.output},
            ],
            "difficulty": round(self.difficulty, 4),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_difficulty(state: dict[str, Any]) -> tuple[float, dict[str, float]]:
    deadlines = state.get("deadlines", {})
    parts = {
        "deadline_count": min(len(deadlines.get("deadlines", [])) / 6.0, 1.0),
        "extension_branch": 1.0 if any("90" in d.get("basis", "") or "20 business" in d.get("basis", "")
                                       for d in deadlines.get("deadlines", [])) else 0.0,
        "out_of_time": 1.0 if deadlines.get("consumer_notice_timely") is False else 0.0,
        "validation_problem": 1.0 if not state.get("validation", {}).get("ok", True) else 0.0,
        "unmapped_reason_code": 1.0 if not state.get("reason_code", {}).get("mapped", True) else 0.0,
        "narrative_noise": min(len(state["complaint"].get("narrative", "").split()) / 90.0, 1.0),
    }
    weights = {
        "deadline_count": 0.25, "extension_branch": 0.25, "out_of_time": 0.20,
        "validation_problem": 0.15, "unmapped_reason_code": 0.10, "narrative_noise": 0.05,
    }
    return sum(weights[k] * v for k, v in parts.items()), parts


def build_examples(
    complaints: Sequence[dict[str, Any]],
    index: RegulationIndex | None = None,
    limit: int | None = None,
) -> list[Example]:
    index = index if index is not None else RegulationIndex()
    drafter = TemplateDrafter()
    examples: list[Example] = []

    for row in complaints[: limit or len(complaints)]:
        issue = row["issue"]
        regulation = row.get("regulation") or rules.regulation_for_issue(issue)
        complaint = _synthesise_dates(row)

        amounts = validate_amounts(complaint.get("disputed_amount"))
        dates = validate_dates(complaint["transaction_date"], complaint["statement_date"], complaint["notice_date"])
        validation = {
            "ok": amounts.ok and dates.ok,
            "issues": [i.to_dict() for i in amounts.issues + dates.issues],
            "normalised": {**amounts.normalised, **dates.normalised},
        }
        if not dates.ok:
            continue  # an example whose dates do not parse cannot carry a grounded target

        deadline_result = _compute(regulation, complaint, validation)
        sections = index.search(f"{issue} {complaint['narrative'][:300]}", 4, regulation=regulation)
        reason_code = map_reason_code(None, issue)

        state = {"complaint": complaint, "deadlines": deadline_result, "sections": sections,
                 "validation": validation, "reason_code": reason_code, "risk": {}}
        draft = drafter.draft(state)
        difficulty, parts = score_difficulty(state)

        examples.append(Example(
            example_id=row["complaint_id"],
            instruction=PROMPT_VERSIONS["v3"],
            input=json.dumps({
                "complaint": {k: complaint[k] for k in
                              ("complaint_id", "issue", "narrative", "disputed_amount",
                               "transaction_date", "statement_date", "notice_date")},
                "computed_deadlines": deadline_result,
                "retrieved_sections": sections,
                "validation": validation,
                "reason_code": reason_code,
            }, indent=2, default=str),
            output=draft.text,
            difficulty=difficulty,
            difficulty_parts=parts,
            metadata={"issue": issue, "regulation": regulation, "split": row.get("split")},
        ))
    return examples


def _synthesise_dates(row: dict[str, Any]) -> dict[str, Any]:
    """Derive the dispute dates the raw complaint record does not carry.

    The CFPB export has a received date and nothing else; a dispute workflow
    needs a transaction date, a statement date and a notice date. They are
    derived deterministically from the complaint id so the dataset is
    reproducible, and the derivation is recorded in the metadata rather than
    presented as source data.
    """
    from datetime import date, timedelta

    rng = random.Random(row["complaint_id"])
    received = date.fromisoformat(row["date_received"])
    notice = received - timedelta(days=rng.randint(0, 12))
    statement = notice - timedelta(days=rng.choice([5, 12, 20, 35, 58, 64, 80]))
    transaction = statement - timedelta(days=rng.randint(1, 20))
    return {
        **row,
        "transaction_date": transaction.isoformat(),
        "statement_date": statement.isoformat(),
        "notice_date": notice.isoformat(),
        "discovery_date": (notice - timedelta(days=rng.randint(0, 4))).isoformat(),
        "point_of_sale": rng.random() < 0.35,
        "foreign_initiated": rng.random() < 0.08,
        "provisional_credit_given": rng.random() < 0.55,
        "_dates_derived": True,
    }


def _compute(regulation: str, complaint: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    normalised = validation["normalised"]
    try:
        if regulation == "REG_E":
            return rules.regulation_e(
                normalised["notice_date"], normalised["statement_date"],
                point_of_sale=complaint.get("point_of_sale", False),
                foreign_initiated=complaint.get("foreign_initiated", False),
                provisional_credit_given=complaint.get("provisional_credit_given", False),
            ).to_dict()
        if regulation == "REG_Z":
            return rules.regulation_z(normalised["notice_date"], normalised["statement_date"]).to_dict()
        if regulation == "FCRA":
            return rules.fcra(normalised["notice_date"]).to_dict()
        return rules.fdcpa(normalised["notice_date"]).to_dict()
    except KeyError:
        return {"regulation": regulation, "deadlines": [], "findings": [], "citations": []}


def write_dataset(examples: Sequence[Example], path: Path | None = None, curriculum: bool = True) -> Path:
    path = path or (ARTIFACT_DIR / ("sft_curriculum.jsonl" if curriculum else "sft_shuffled.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(examples, key=lambda e: e.difficulty) if curriculum else list(examples)
    if not curriculum:
        random.Random(0).shuffle(ordered)
    with path.open("w", encoding="utf-8") as fh:
        for example in ordered:
            fh.write(json.dumps(example.to_chat(), default=str) + "\n")
    return path


def difficulty_summary(examples: Sequence[Example]) -> dict[str, Any]:
    if not examples:
        return {"examples": 0}
    values = sorted(e.difficulty for e in examples)
    return {
        "examples": len(examples),
        "difficulty_min": round(values[0], 4),
        "difficulty_median": round(values[len(values) // 2], 4),
        "difficulty_max": round(values[-1], 4),
        "hardest_drivers": _top_drivers(examples),
    }


def _top_drivers(examples: Sequence[Example]) -> dict[str, float]:
    hardest = sorted(examples, key=lambda e: -e.difficulty)[: max(1, len(examples) // 10)]
    keys = hardest[0].difficulty_parts.keys()
    return {k: round(sum(e.difficulty_parts[k] for e in hardest) / len(hardest), 4) for k in keys}

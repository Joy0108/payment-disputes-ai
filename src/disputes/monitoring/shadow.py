"""Shadow-mode rollout.

A candidate model runs on live traffic and its predictions are recorded, but
nothing acts on them. The incumbent's output is what reaches the consumer.

The point is not the comparison of offline metrics - that already happened on
the frozen test split. It is the two things the offline split cannot tell you:

* **agreement on live traffic**, which is a different distribution to any
  historical split, and where the candidate disagrees;
* **whether the disagreements are the cases that matter.** A candidate that
  disagrees only on complaints nobody escalates is a free swap. One that
  disagrees on the high-value disputes is a change of policy, however good its
  aggregate score.

A promotion decision made on aggregate agreement alone hides exactly the second
case, so the report segments the disagreements before it reports the total.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ARTIFACT_DIR


@dataclass
class ShadowRecord:
    key: str
    incumbent: Any
    candidate: Any
    agreed: bool
    segment: str
    observed: Any = None
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "incumbent": self.incumbent, "candidate": self.candidate,
            "agreed": self.agreed, "segment": self.segment, "observed": self.observed,
            "recorded_at": self.recorded_at,
        }


@dataclass
class ShadowRun:
    incumbent_version: str
    candidate_version: str
    records: list[ShadowRecord] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return sum(1 for r in self.records if r.agreed) / len(self.records) if self.records else float("nan")

    def by_segment(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for segment in sorted({r.segment for r in self.records}):
            subset = [r for r in self.records if r.segment == segment]
            agreed = sum(1 for r in subset if r.agreed)
            scored = [r for r in subset if r.observed is not None]
            entry: dict[str, Any] = {
                "n": len(subset),
                "agreement": round(agreed / len(subset), 4) if subset else float("nan"),
                "disagreements": len(subset) - agreed,
            }
            if scored:
                entry["incumbent_correct"] = round(sum(1 for r in scored if r.incumbent == r.observed) / len(scored), 4)
                entry["candidate_correct"] = round(sum(1 for r in scored if r.candidate == r.observed) / len(scored), 4)
            out[segment] = entry
        return out

    def report(self) -> dict[str, Any]:
        scored = [r for r in self.records if r.observed is not None]
        segments = self.by_segment()

        # Where the two models disagree and the truth is known, which one was
        # right. This is the number a promotion decision actually turns on.
        contested = [r for r in scored if not r.agreed]
        candidate_wins = sum(1 for r in contested if r.candidate == r.observed)
        incumbent_wins = sum(1 for r in contested if r.incumbent == r.observed)

        return {
            "incumbent_version": self.incumbent_version,
            "candidate_version": self.candidate_version,
            "records": len(self.records),
            "overall_agreement": round(self.agreement, 4),
            "labelled_records": len(scored),
            "contested": {
                "n": len(contested),
                "candidate_correct": candidate_wins,
                "incumbent_correct": incumbent_wins,
                "neither_correct": len(contested) - candidate_wins - incumbent_wins,
                "verdict": (
                    "candidate better on contested cases" if candidate_wins > incumbent_wins
                    else "incumbent better on contested cases" if incumbent_wins > candidate_wins
                    else "no separation on contested cases"
                ),
            },
            "by_segment": segments,
            "high_value_segments_with_disagreement": [
                name for name, entry in segments.items()
                if name.startswith("high") and entry["disagreements"] > 0
            ],
            "recommendation": _recommend(self.agreement, candidate_wins, incumbent_wins, segments),
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or (ARTIFACT_DIR / "shadow_run.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "report": self.report(),
            "records": [r.to_dict() for r in self.records[:500]],
        }, indent=2, default=str), encoding="utf-8", newline="\n")
        return path


def _recommend(agreement: float, candidate_wins: int, incumbent_wins: int, segments: dict[str, dict]) -> str:
    high_risk = [n for n, e in segments.items() if n.startswith("high") and e["disagreements"] > 0]
    if agreement != agreement:  # NaN
        return "no traffic observed; keep the candidate in shadow"
    if candidate_wins > incumbent_wins and not high_risk:
        return "promote: the candidate wins the contested cases and does not disturb the high-value segment"
    if candidate_wins > incumbent_wins and high_risk:
        return (
            "promote only after a human review of the high-value disagreements: the candidate is better on "
            "aggregate but is changing decisions in the segment where a wrong one is expensive"
        )
    if incumbent_wins > candidate_wins:
        return "do not promote: the incumbent is right more often where the two disagree"
    return "hold in shadow: not enough separation to justify a change"


def run_shadow(
    rows: Sequence[dict[str, Any]],
    incumbent: Callable[[dict[str, Any]], Any],
    candidate: Callable[[dict[str, Any]], Any],
    *,
    key: str = "complaint_id",
    label: str | None = None,
    segment: Callable[[dict[str, Any]], str] | None = None,
    incumbent_version: str = "production",
    candidate_version: str = "candidate",
) -> ShadowRun:
    segment = segment or (lambda row: "all")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run = ShadowRun(incumbent_version, candidate_version)
    for row in rows:
        a, b = incumbent(row), candidate(row)
        run.records.append(ShadowRecord(
            key=str(row.get(key, "")), incumbent=a, candidate=b, agreed=a == b,
            segment=segment(row), observed=row.get(label) if label else None, recorded_at=now,
        ))
    return run


def value_segment(row: dict[str, Any], high_threshold: float = 750.0) -> str:
    """Segment by what a wrong answer costs, not by what is convenient to group."""
    amount = row.get("disputed_amount")
    try:
        amount = float(amount) if amount not in (None, "") else 0.0
    except (TypeError, ValueError):
        amount = 0.0
    if amount >= high_threshold:
        return "high_value"
    if row.get("consumer_disputed") in (True, "Yes"):
        return "high_escalation"
    return "standard"


def disagreement_examples(run: ShadowRun, limit: int = 5) -> list[dict[str, Any]]:
    return [r.to_dict() for r in run.records if not r.agreed][:limit]


def summarise_predictions(run: ShadowRun) -> dict[str, Any]:
    return {
        "incumbent": dict(Counter(str(r.incumbent) for r in run.records).most_common(5)),
        "candidate": dict(Counter(str(r.candidate) for r in run.records).most_common(5)),
    }

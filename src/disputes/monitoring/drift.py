"""Drift monitoring on the daily complaint feed.

Two kinds of drift, monitored separately because they mean different things and
call for different responses:

**Data drift** - the inputs move. A new product launches, a channel changes mix,
a marketing campaign brings a different population. The model may still be fine.

**Concept drift** - the relationship between inputs and the label moves. The
model is now wrong and no amount of input monitoring will say so, because the
inputs can look identical. This is what happened to ``keep_alive_session`` in
the fraud data, and it is why prediction distribution and realised outcomes are
monitored alongside the features.

Population Stability Index throughout, with the conventional bands from credit
risk monitoring (below 0.10 stable, 0.10-0.25 investigate, above 0.25 alert),
and per-category contributions so an alert comes with a diagnosis rather than
just a number.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PSI_STABLE = 0.10
PSI_ALERT = 0.25


@dataclass
class Signal:
    name: str
    kind: str  # data | concept | volume
    statistic: str
    value: float
    status: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"signal": self.name, "kind": self.kind, "statistic": self.statistic,
                "value": round(self.value, 4), "status": self.status, "detail": self.detail}


def psi(baseline: Counter, current: Counter, epsilon: float = 1e-6) -> tuple[float, dict[str, float]]:
    categories = set(baseline) | set(current)
    base_total = sum(baseline.values()) or 1
    curr_total = sum(current.values()) or 1
    total = 0.0
    contributions: dict[str, float] = {}
    for category in categories:
        b = max(baseline.get(category, 0) / base_total, epsilon)
        c = max(current.get(category, 0) / curr_total, epsilon)
        contribution = (c - b) * math.log(c / b)
        contributions[str(category)] = round(contribution, 6)
        total += contribution
    return total, dict(sorted(contributions.items(), key=lambda kv: -abs(kv[1])))


def _status(value: float) -> str:
    return "stable" if value < PSI_STABLE else ("investigate" if value < PSI_ALERT else "alert")


def quantile_bins(values: Sequence[float], edges: Sequence[float]) -> Counter:
    counter: Counter = Counter()
    for value in values:
        index = 0
        while index < len(edges) and value > edges[index]:
            index += 1
        counter[str(index)] += 1
    return counter


def edges_from(values: Sequence[float], n_bins: int = 10) -> list[float]:
    """Bin edges taken from the *baseline* only.

    Recomputing bins on the current window is the classic mistake: the bins move
    with the data, the histograms line up again, and PSI reports zero no matter
    what happened.
    """
    if not values:
        return []
    ordered = sorted(values)
    return [ordered[int(len(ordered) * i / n_bins)] for i in range(1, n_bins)]


def categorical_drift(name: str, baseline: Sequence[Any], current: Sequence[Any], kind: str = "data") -> Signal:
    value, contributions = psi(Counter(map(str, baseline)), Counter(map(str, current)))
    return Signal(name, kind, "psi", value, _status(value), {
        "top_contributors": dict(list(contributions.items())[:5]),
        "new_categories": sorted(set(map(str, current)) - set(map(str, baseline)))[:5],
    })


def numeric_drift(name: str, baseline: Sequence[float], current: Sequence[float], kind: str = "data") -> Signal:
    edges = edges_from(list(baseline))
    value, contributions = psi(quantile_bins(baseline, edges), quantile_bins(current, edges))
    return Signal(name, kind, "psi", value, _status(value), {
        "baseline_mean": round(sum(baseline) / len(baseline), 4) if baseline else None,
        "current_mean": round(sum(current) / len(current), 4) if current else None,
        "top_bins": dict(list(contributions.items())[:4]),
    })


def outcome_drift(baseline_labels: Sequence[Any], current_labels: Sequence[Any]) -> Signal:
    """Realised outcomes, which is where concept drift becomes visible.

    Inputs can be identical while the label distribution moves, and that is
    exactly the case a feature monitor cannot see.
    """
    signal = categorical_drift("realised_outcome", baseline_labels, current_labels, kind="concept")
    signal.detail["note"] = (
        "A shift here with stable inputs is concept drift: the same complaints are now resolved differently, "
        "so a model fitted on the old period is predicting a policy that no longer applies."
    )
    return signal


def prediction_drift(baseline_predictions: Sequence[Any], current_predictions: Sequence[Any]) -> Signal:
    signal = categorical_drift("prediction_mix", baseline_predictions, current_predictions, kind="concept")
    signal.detail["note"] = (
        "The model's own output distribution. Moves before the realised outcomes do, because predictions are "
        "available immediately and labels are not."
    )
    return signal


def reference_window(rows: Sequence[dict[str, Any]], date_key: str = "date_received", months: int = 12
                     ) -> list[dict[str, Any]]:
    """The earliest ``months`` of the data, as the drift reference.

    Using the *whole* training set as the baseline is the mistake that makes a
    drift monitor quiet. If the shift happened partway through training, the
    baseline already contains both regimes, the two histograms overlap, and PSI
    reports stable while the model is measurably stale. The reference has to be
    a window the pipeline believes was stable, not everything it was fitted on.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: str(r.get(date_key, "")))
    start = str(ordered[0].get(date_key, ""))[:7]
    year, month = int(start[:4]), int(start[5:7])
    end_month = month + months
    end = f"{year + (end_month - 1) // 12:04d}-{((end_month - 1) % 12) + 1:02d}"
    return [r for r in ordered if str(r.get(date_key, ""))[:7] < end]


def run_monitors(
    baseline: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    categorical: Sequence[str] = ("issue", "product", "submitted_via", "company_size"),
    numeric: Sequence[str] = ("narrative_length",),
    label: str | None = "company_response",
    predictions: tuple[Sequence[Any], Sequence[Any]] | None = None,
) -> dict[str, Any]:
    signals: list[Signal] = []
    for column in categorical:
        if column in (baseline[0] if baseline else {}):
            signals.append(categorical_drift(column, [r.get(column) for r in baseline], [r.get(column) for r in current]))
    for column in numeric:
        if column in (baseline[0] if baseline else {}):
            signals.append(numeric_drift(
                column,
                [float(r[column]) for r in baseline if r.get(column) is not None],
                [float(r[column]) for r in current if r.get(column) is not None],
            ))
    if label and baseline and label in baseline[0]:
        signals.append(outcome_drift([r.get(label) for r in baseline], [r.get(label) for r in current]))
    if predictions:
        signals.append(prediction_drift(*predictions))

    order = {"stable": 0, "investigate": 1, "alert": 2}
    worst = max((s.status for s in signals), key=lambda s: order[s], default="stable")
    concept = [s for s in signals if s.kind == "concept" and s.status != "stable"]

    return {
        "baseline_rows": len(baseline),
        "current_rows": len(current),
        "overall_status": worst,
        "concept_drift_detected": bool(concept),
        "signals": [s.to_dict() for s in signals],
        "action": (
            "hold the model and re-baseline: the input-output relationship has moved, not just the inputs"
            if concept else {
                "stable": "no action",
                "investigate": "review the top contributors before the next promotion",
                "alert": "hold promotion and investigate the inputs",
            }[worst]
        ),
    }

"""Declarative data validation at every layer boundary.

Expectations are declared as data, not written as assertions inside transforms,
for one reason: a transform that validates its own output only fails when
someone runs that transform. A contract attached to the boundary fails when
anything crosses it, including a hand-loaded file or a backfill.

Every expectation carries the consequence of its being violated, because
"expected column complaint_id to be unique" tells an on-call engineer nothing
about whether to page someone.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Expectation:
    name: str
    column: str | None
    check: Callable[[list[dict[str, Any]]], tuple[bool, dict[str, Any]]]
    severity: str = "error"  # error blocks the layer, warning is recorded
    consequence: str = ""


@dataclass
class ValidationReport:
    layer: str
    rows: int
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(r["severity"] == "error" and not r["passed"] for r in self.results)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [r for r in self.results if not r["passed"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rows": self.rows,
            "passed": self.passed,
            "checks": len(self.results),
            "failures": self.failures,
        }

    def raise_if_failed(self) -> ValidationReport:
        if not self.passed:
            blocking = [f for f in self.failures if f["severity"] == "error"]
            lines = "\n".join(f"  - {f['name']}: {f['detail']}  ({f['consequence']})" for f in blocking)
            raise DataContractError(f"{self.layer} failed {len(blocking)} blocking expectation(s):\n{lines}")
        return self


class DataContractError(RuntimeError):
    pass


# --- expectation builders --------------------------------------------------

def not_null(column: str, consequence: str = "", severity: str = "error") -> Expectation:
    def check(rows):
        missing = [r for r in rows if r.get(column) in (None, "")]
        return not missing, {"missing": len(missing), "example": missing[0] if missing else None}

    return Expectation(f"{column} is never null", column, check, severity, consequence)


def unique(column: str, consequence: str = "", severity: str = "error") -> Expectation:
    def check(rows):
        seen, dupes = set(), []
        for r in rows:
            value = r.get(column)
            (dupes.append(value) if value in seen else seen.add(value))
        return not dupes, {"duplicates": len(dupes), "examples": dupes[:5]}

    return Expectation(f"{column} is unique", column, check, severity, consequence)


def in_set(column: str, allowed: Sequence[str], consequence: str = "", severity: str = "error") -> Expectation:
    allowed_set = set(allowed)

    def check(rows):
        bad = sorted({str(r.get(column)) for r in rows if r.get(column) not in allowed_set})
        return not bad, {"unexpected_values": bad[:8], "allowed": sorted(allowed_set)[:8]}

    return Expectation(f"{column} is one of {len(allowed_set)} known values", column, check, severity, consequence)


def matches(column: str, pattern: str, consequence: str = "", severity: str = "error") -> Expectation:
    compiled = re.compile(pattern)

    def check(rows):
        bad = [str(r.get(column)) for r in rows if not compiled.fullmatch(str(r.get(column) or ""))]
        return not bad, {"non_matching": len(bad), "examples": bad[:5]}

    return Expectation(f"{column} matches {pattern}", column, check, severity, consequence)


def between(column: str, low: float, high: float, consequence: str = "", severity: str = "error") -> Expectation:
    def check(rows):
        bad = []
        for r in rows:
            value = r.get(column)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                bad.append(value)
                continue
            if not (low <= number <= high):
                bad.append(number)
        return not bad, {"out_of_range": len(bad), "examples": bad[:5], "bounds": [low, high]}

    return Expectation(f"{column} within [{low}, {high}]", column, check, severity, consequence)


def monotonic(column: str, consequence: str = "", severity: str = "error") -> Expectation:
    """Rows are ordered by this column.

    Not cosmetic. Every split in this project is temporal, and a temporal split
    computed over unordered rows silently becomes a random one.
    """

    def check(rows):
        values = [r.get(column) for r in rows]
        ordered = all(a <= b for a, b in zip(values, values[1:], strict=False))
        first_break = next((i for i, (a, b) in enumerate(zip(values, values[1:], strict=False)) if a > b), None)
        return ordered, {"first_break_at_row": first_break}

    return Expectation(f"{column} is non-decreasing", column, check, severity, consequence)


def class_balance(column: str, min_share: float, consequence: str = "", severity: str = "warning") -> Expectation:
    """Every class holds at least ``min_share`` of the rows."""

    def check(rows):
        counts: dict[str, int] = {}
        for r in rows:
            counts[str(r.get(column))] = counts.get(str(r.get(column)), 0) + 1
        total = len(rows) or 1
        rare = {k: round(v / total, 5) for k, v in counts.items() if v / total < min_share}
        return not rare, {"classes": len(counts), "below_threshold": rare, "min_share": min_share}

    return Expectation(f"{column} classes each hold >= {min_share:.1%}", column, check, severity, consequence)


def row_count(minimum: int, consequence: str = "", severity: str = "error") -> Expectation:
    def check(rows):
        return len(rows) >= minimum, {"rows": len(rows), "minimum": minimum}

    return Expectation(f"at least {minimum} rows", None, check, severity, consequence)


def referential(column: str, valid: Sequence[str], consequence: str = "", severity: str = "error") -> Expectation:
    valid_set = set(valid)

    def check(rows):
        bad = sorted({str(r.get(column)) for r in rows if r.get(column) not in valid_set})
        return not bad, {"unresolvable": bad[:8]}

    return Expectation(f"{column} resolves to a known key", column, check, severity, consequence)


# --- runner ----------------------------------------------------------------

def validate(layer: str, rows: list[dict[str, Any]], expectations: Sequence[Expectation]) -> ValidationReport:
    report = ValidationReport(layer=layer, rows=len(rows))
    for expectation in expectations:
        passed, detail = expectation.check(rows)
        report.results.append({
            "name": expectation.name,
            "column": expectation.column,
            "passed": bool(passed),
            "severity": expectation.severity,
            "detail": detail,
            "consequence": expectation.consequence,
        })
    return report

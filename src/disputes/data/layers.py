"""Bronze, silver, gold - and the contracts between them.

* **bronze** is the raw file, read and nothing else. No parsing, no coercion, no
  dropped rows. It exists so that "the data changed" and "our parsing changed"
  are distinguishable questions.
* **silver** is typed, deduplicated, ordered, and validated. Rows that fail
  parsing are *quarantined*, not dropped, because a silently shrinking row count
  is the hardest data bug to find.
* **gold** is the modelling view: derived features, the label, and the temporal
  split assignment frozen in the table rather than recomputed at train time.

Each layer writes a content hash into the manifest, so a model version can name
the exact bytes it was trained on.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import COMPLAINT_ISSUES, GOLD_DIR, RAW_DIR, RESPONSE_CATEGORIES, SILVER_DIR
from ..rules.deadlines import regulation_for_issue
from . import validate as v
from .manifest import DataManifest


@dataclass
class Layer:
    name: str
    rows: list[dict[str, Any]]
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    report: v.ValidationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.name,
            "rows": len(self.rows),
            "quarantined": len(self.quarantined),
            "validation": self.report.to_dict() if self.report else None,
        }


# ---------------------------------------------------------------------------
# bronze
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def bronze_complaints(raw_dir: Path = RAW_DIR) -> Layer:
    rows = read_csv(raw_dir / "complaints.csv")
    report = v.validate("bronze/complaints", rows, [
        v.row_count(1000, "an empty or truncated extract would train a model on nothing"),
        v.not_null("complaint_id", "rows without an identifier cannot be deduplicated or traced back"),
        v.not_null("date_received", "every split in this project is temporal; an undated row cannot be placed"),
    ])
    return Layer("bronze/complaints", rows, report=report)


def bronze_fraud(raw_dir: Path = RAW_DIR) -> Layer:
    rows = read_csv(raw_dir / "fraud_applications.csv")
    report = v.validate("bronze/fraud", rows, [
        v.row_count(1000, "too few rows to estimate a base rate near one percent"),
        v.not_null("application_id", "rows without an identifier cannot be deduplicated"),
        v.not_null("month", "the month index is the temporal split key"),
    ])
    return Layer("bronze/fraud", rows, report=report)


# ---------------------------------------------------------------------------
# silver
# ---------------------------------------------------------------------------

def silver_complaints(bronze: Layer) -> Layer:
    typed: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in bronze.rows:
        try:
            complaint_id = row["complaint_id"].strip()
            if complaint_id in seen:
                quarantine.append({**row, "_quarantine_reason": "duplicate complaint_id"})
                continue
            amount = row.get("disputed_amount", "")
            typed.append({
                "complaint_id": complaint_id,
                "date_received": row["date_received"].strip()[:10],
                "product": row["product"].strip(),
                "issue": row["issue"].strip(),
                "regulation": row.get("regulation", "").strip() or regulation_for_issue(row["issue"].strip()),
                "company": row["company"].strip(),
                "company_size": row["company_size"].strip(),
                "state": row["state"].strip(),
                "submitted_via": row["submitted_via"].strip(),
                "narrative": row["narrative"].strip(),
                "narrative_length": len(row["narrative"].split()),
                "disputed_amount": float(amount) if amount not in ("", None) else None,
                "company_response": row["company_response"].strip(),
                "timely_response": row["timely_response"].strip() == "Yes",
                "consumer_disputed": row["consumer_disputed"].strip() == "Yes",
            })
            seen.add(complaint_id)
        except (KeyError, ValueError) as exc:
            quarantine.append({**row, "_quarantine_reason": f"{type(exc).__name__}: {exc}"})

    typed.sort(key=lambda r: (r["date_received"], r["complaint_id"]))

    report = v.validate("silver/complaints", typed, [
        v.unique("complaint_id", "a duplicated complaint would appear in both train and test"),
        v.not_null("narrative", "the classifier has no other input"),
        v.in_set("issue", COMPLAINT_ISSUES, "an unknown issue cannot be mapped to a regulation, so no deadline is computable"),
        v.in_set("company_response", RESPONSE_CATEGORIES, "an unknown response category breaks the outcome label"),
        v.in_set("regulation", ["REG_E", "REG_Z", "FCRA", "FDCPA"],
                 "the regulation drives the deadline calculation; an unknown value is a wrong or missing deadline"),
        v.monotonic("date_received", "an unordered table turns the temporal split into a random one"),
        v.between("narrative_length", 3, 400, "a narrative this short or long is a parsing failure, not a complaint"),
        v.class_balance("issue", 0.005,
                        "a class this rare will not have enough test examples for its per-class recall to mean anything",
                        severity="warning"),
    ])
    return Layer("silver/complaints", typed, quarantine, report)


def silver_fraud(bronze: Layer) -> Layer:
    from ..config import FRAUD_FEATURES

    typed: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in bronze.rows:
        try:
            record: dict[str, Any] = {
                "application_id": row["application_id"].strip(),
                "month": int(row["month"]),
                "fraud_bool": int(row["fraud_bool"]),
            }
            for feature in FRAUD_FEATURES:
                record[feature] = float(row[feature])
            typed.append(record)
        except (KeyError, ValueError) as exc:
            quarantine.append({**row, "_quarantine_reason": f"{type(exc).__name__}: {exc}"})

    typed.sort(key=lambda r: (r["month"], r["application_id"]))
    report = v.validate("silver/fraud", typed, [
        v.unique("application_id", "a duplicated application leaks across the temporal boundary"),
        v.in_set("fraud_bool", [0, 1], "the label must be binary"),
        v.between("month", 0, 11, "the month index is the split key and must be in range"),
        v.monotonic("month", "an unordered table turns the temporal split into a random one"),
        v.between("customer_age", 10, 100, "an out-of-range age is a parsing failure"),
    ])
    return Layer("silver/fraud", typed, quarantine, report)


# ---------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------

def gold_complaints(silver: Layer, split_dates: tuple[str, str]) -> Layer:
    """Modelling view with the split assignment frozen into the row.

    Freezing it matters more than it looks. A split recomputed at train time
    moves whenever the data does, so two runs a week apart are not comparable
    and a regression looks like noise.
    """
    train_end, val_end = split_dates
    rows = []
    for row in silver.rows:
        received = row["date_received"]
        split = "train" if received < train_end else ("validation" if received < val_end else "test")
        rows.append({
            **row,
            "split": split,
            # Labels for the four models.
            "label_issue": row["issue"],
            "label_outcome": row["company_response"],
            "label_relief": row["company_response"] in {"Closed with monetary relief", "Closed with non-monetary relief"},
            "label_escalation": row["consumer_disputed"],
        })

    report = v.validate("gold/complaints", rows, [
        v.in_set("split", ["train", "validation", "test"], "an unassigned row would be silently excluded from every fold"),
        v.not_null("label_issue", "the classification target"),
        v.class_balance("split", 0.10, "a fold this small gives a metric with no usable confidence interval",
                        severity="warning"),
    ])
    return Layer("gold/complaints", rows, silver.quarantined, report)


def gold_fraud(silver: Layer, split_months: tuple[int, int]) -> Layer:
    train_end, val_end = split_months
    rows = []
    for row in silver.rows:
        month = row["month"]
        split = "train" if month < train_end else ("validation" if month < val_end else "test")
        rows.append({**row, "split": split})
    report = v.validate("gold/fraud", rows, [
        v.in_set("split", ["train", "validation", "test"], "an unassigned row would be silently excluded"),
        v.class_balance("split", 0.10, "a fold this small cannot estimate a one-percent base rate",
                        severity="warning"),
    ])
    return Layer("gold/fraud", rows, silver.quarantined, report)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def build_all(
    raw_dir: Path = RAW_DIR,
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
    write: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    from ..config import FRAUD_SPLIT_MONTHS, SPLIT_DATES

    manifest = DataManifest()
    layers: dict[str, Layer] = {}

    b_complaints = bronze_complaints(raw_dir)
    b_fraud = bronze_fraud(raw_dir)
    if strict:
        b_complaints.report.raise_if_failed()
        b_fraud.report.raise_if_failed()

    s_complaints = silver_complaints(b_complaints)
    s_fraud = silver_fraud(b_fraud)
    if strict:
        s_complaints.report.raise_if_failed()
        s_fraud.report.raise_if_failed()

    g_complaints = gold_complaints(s_complaints, SPLIT_DATES)
    g_fraud = gold_fraud(s_fraud, FRAUD_SPLIT_MONTHS)
    if strict:
        g_complaints.report.raise_if_failed()
        g_fraud.report.raise_if_failed()

    layers = {
        "bronze/complaints": b_complaints, "bronze/fraud": b_fraud,
        "silver/complaints": s_complaints, "silver/fraud": s_fraud,
        "gold/complaints": g_complaints, "gold/fraud": g_fraud,
    }

    if write:
        silver_dir.mkdir(parents=True, exist_ok=True)
        gold_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(silver_dir / "complaints.jsonl", s_complaints.rows)
        _write_jsonl(silver_dir / "fraud.jsonl", s_fraud.rows)
        _write_jsonl(gold_dir / "complaints.jsonl", g_complaints.rows)
        _write_jsonl(gold_dir / "fraud.jsonl", g_fraud.rows)
        if s_complaints.quarantined or s_fraud.quarantined:
            _write_jsonl(silver_dir / "quarantine.jsonl", s_complaints.quarantined + s_fraud.quarantined)

        for name, path in [
            ("raw/complaints", raw_dir / "complaints.csv"),
            ("raw/fraud", raw_dir / "fraud_applications.csv"),
            ("silver/complaints", silver_dir / "complaints.jsonl"),
            ("silver/fraud", silver_dir / "fraud.jsonl"),
            ("gold/complaints", gold_dir / "complaints.jsonl"),
            ("gold/fraud", gold_dir / "fraud.jsonl"),
        ]:
            manifest.track(name, path)
        manifest.save()

    return {
        "layers": {k: layer.to_dict() for k, layer in layers.items()},
        "manifest": manifest.summary(),
        "splits": {
            "complaints": _split_counts(g_complaints.rows),
            "fraud": _split_counts(g_fraud.rows),
        },
    }


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def load_gold(name: str, gold_dir: Path = GOLD_DIR) -> list[dict[str, Any]]:
    path = gold_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not built; run `disputes build`")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

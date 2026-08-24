"""Regulatory retrieval evaluation.

Recall@k rather than nDCG: a draft cites a handful of sections and the question
is whether the governing one is among them. Two extra checks that a generic
retrieval metric does not cover:

* **regulation routing** - the top hit must be from the regulation that actually
  governs the dispute. Answering a Reg E debit dispute out of Reg Z is wrong on
  the law, not merely off-topic, and it scores as a hit on plain recall.
* **deadline agreement** - questions carrying a ``deadline_check`` are also
  answered by the rules engine, and the two answers must agree. Retrieval that
  finds the right section while the code computes a different number is the
  failure mode that reaches a consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DEFAULT_RAG, GOLDEN_PATH
from ..rag.index import RegulationIndex

# What the rules engine says, for the questions that assert a number.
DEADLINE_ORACLE = {
    "reg_e_initial_business_days": 10,
    "reg_e_new_account_business_days": 20,
    "reg_e_notice_calendar_days": 60,
    "reg_z_notice_calendar_days": 60,
    "reg_z_resolution_cap_days": 90,
    "fcra_reinvestigation_days": 30,
    "fdcpa_validation_notice_days": 5,
    "fdcpa_dispute_window_days": 30,
}


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def evaluate(index: RegulationIndex | None = None, k: int | None = None, route: bool = True) -> dict[str, Any]:
    index = index if index is not None else RegulationIndex()
    k = k or DEFAULT_RAG.top_k
    golden = load_golden()

    rows = []
    for q in golden["questions"]:
        regulation = q.get("regulation") if route else None
        hits = index.search(q["question"], k, regulation=regulation)
        ids = [h["id"] for h in hits]
        primary = set(q["primary"])
        relevant = primary | set(q.get("secondary", []))
        rank = next((i + 1 for i, h in enumerate(ids) if h in primary), None)
        rows.append({
            "qid": q["qid"],
            "regulation": q.get("regulation"),
            "recall": len(primary & set(ids)) / len(primary) if primary else float("nan"),
            "any_primary": 1.0 if primary & set(ids) else 0.0,
            "precision": len(relevant & set(ids)) / k,
            "mrr": 1.0 / rank if rank else 0.0,
            "top_regulation_correct": 1.0 if (hits and hits[0]["regulation"] == q.get("regulation")) else 0.0,
            "hits": ids,
            "missed_primary": sorted(primary - set(ids)),
        })

    deadline_rows = []
    for q in golden["questions"]:
        check = q.get("deadline_check")
        if not check:
            continue
        engine = DEADLINE_ORACLE.get(check)
        deadline_rows.append({
            "qid": q["qid"], "check": check, "golden_expected": q["expected"],
            "rules_engine": engine, "agrees": engine == q["expected"],
        })

    return {
        "k": k,
        "questions": len(rows),
        f"recall@{k}": round(_mean(r["recall"] for r in rows), 4),
        f"any_primary@{k}": round(_mean(r["any_primary"] for r in rows), 4),
        f"precision@{k}": round(_mean(r["precision"] for r in rows), 4),
        "mrr": round(_mean(r["mrr"] for r in rows), 4),
        "regulation_routing_accuracy": round(_mean(r["top_regulation_correct"] for r in rows), 4),
        "deadline_agreement": {
            "checked": len(deadline_rows),
            "agree": sum(1 for r in deadline_rows if r["agrees"]),
            "disagreements": [r for r in deadline_rows if not r["agrees"]],
        },
        "rows": rows,
    }


def ablate() -> list[dict[str, Any]]:
    """Lexical only, dense only, fused, fused with the regulation-aware rerank."""
    from dataclasses import replace

    index = RegulationIndex()
    golden = load_golden()
    k = DEFAULT_RAG.top_k
    out = []

    for name in ("bm25 only", "dense only", "rrf fused", "rrf + regulation rerank"):
        rows = []
        for q in golden["questions"]:
            if name == "bm25 only":
                ids = [s for s, _ in index.bm25.search(q["question"], k)]
            elif name == "dense only":
                ids = [s for s, _ in index.dense.search(q["question"], k)]
            else:
                cfg = replace(DEFAULT_RAG, rerank=(name.endswith("rerank")))
                scoped = RegulationIndex(index.sections, cfg)
                regulation = q.get("regulation") if name.endswith("rerank") else None
                ids = [h["id"] for h in scoped.search(q["question"], k, regulation=regulation)]
            primary = set(q["primary"])
            rows.append({
                "recall": len(primary & set(ids)) / len(primary) if primary else float("nan"),
                "routing": 1.0 if ids and index.by_id[ids[0]].regulation == q.get("regulation") else 0.0,
            })
        out.append({
            "config": name,
            f"recall@{k}": round(_mean(r["recall"] for r in rows), 4),
            "regulation_routing_accuracy": round(_mean(r["routing"] for r in rows), 4),
        })
    return out

"""End-to-end evaluation and the promotion gates.

Stages are measured separately because they fail separately, and because the
gates carry different kinds of authority. The retrieval and model numbers are
quality thresholds - somebody chose them and they are arguable. Deadline
exactness and citation resolution are not: a wrong deadline in a consumer letter
is a regulatory violation and a citation that resolves to nothing is a
fabricated reference, and neither has a level at which some are acceptable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import ARTIFACT_DIR, DEFAULT_GATES, DEFAULT_LLM, DEFAULT_RAG, REPORT_DIR, Gates, ensure_dirs
from ..data.layers import load_gold
from ..data.manifest import DataManifest
from ..rag.index import RegulationIndex
from ..workflow.nodes import Predictors, build_workflow
from . import deadlines as deadline_eval
from . import rag as rag_eval


def evaluate_workflow(
    complaints: list[dict[str, Any]],
    index: RegulationIndex,
    predictors: Predictors | None = None,
    sample: int = 120,
    seed: int = 5,
) -> dict[str, Any]:
    """Run the workflow over a sample and check what it produced."""
    from ..llm.dataset import _synthesise_dates

    machine = build_workflow(index, predictors or Predictors(), DEFAULT_LLM)
    rng = random.Random(seed)
    chosen = rng.sample(complaints, k=min(sample, len(complaints)))

    total_citations = resolved = 0
    invented_dates = 0
    uncited = 0
    passed_first_attempt = 0
    attempts: list[int] = []
    queues: dict[str, int] = {}
    failures: list[dict[str, Any]] = []

    for row in chosen:
        complaint = _synthesise_dates(row)
        state = machine.invoke({"complaint": complaint})
        verification = state.get("verification")
        if verification is None:
            continue
        total_citations += verification.citations
        resolved += verification.resolved
        invented_dates += len(verification.invented_dates)
        uncited += len(verification.uncited_claims)
        attempts.append(state.get("attempts", 1))
        passed_first_attempt += 1 if (verification.passed and state.get("attempts") == 1) else 0
        queue = state.get("outcome", {}).get("queue", "unknown")
        queues[queue] = queues.get(queue, 0) + 1
        if not verification.passed:
            failures.append({"complaint_id": complaint.get("complaint_id"), **verification.to_dict()})

    n = len(attempts) or 1
    return {
        "sampled": len(chosen),
        "citations": total_citations,
        "citation_resolution": round(resolved / total_citations, 4) if total_citations else 1.0,
        "invented_dates": invented_dates,
        "uncited_claims": uncited,
        "verification_pass_rate": round((n - len(failures)) / n, 4),
        "passed_on_first_attempt": round(passed_first_attempt / n, 4),
        "mean_draft_attempts": round(sum(attempts) / n, 3),
        "routing": queues,
        "failures": failures[:5],
    }


def run_full_eval(
    out_dir: Path = REPORT_DIR,
    write: bool = True,
    with_models: bool = True,
    with_curriculum: bool = True,
    sample: int = 120,
) -> dict[str, Any]:
    ensure_dirs()
    index = RegulationIndex()
    complaints = load_gold("complaints")
    test_rows = [r for r in complaints if r["split"] == "test"]

    report: dict[str, Any] = {
        "data": {
            "complaints": len(complaints),
            "splits": _counts(complaints, "split"),
            "manifest_fingerprint": DataManifest().fingerprint(),
        },
        "deadlines": {k: v for k, v in deadline_eval.run().items() if k != "rows"},
        "retrieval": {k: v for k, v in rag_eval.evaluate(index).items() if k != "rows"},
    }

    predictors = None
    if with_models:
        try:
            predictors = Predictors.from_artifacts()
            report["models"] = json.loads((ARTIFACT_DIR / "models.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ImportError):
            report["models"] = {"note": "no trained models found; run `disputes train`"}

    report["workflow"] = evaluate_workflow(test_rows, index, predictors, sample=sample)

    if with_curriculum:
        report["curriculum"] = _curriculum(complaints, index)

    report["gates"] = check_gates(report, DEFAULT_GATES)
    report["config"] = {
        "rag": asdict(DEFAULT_RAG), "llm": asdict(DEFAULT_LLM), "gates": asdict(DEFAULT_GATES),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8", newline="\n")
    return report


def _curriculum(complaints: list[dict[str, Any]], index: RegulationIndex, n: int = 2000) -> dict[str, Any]:
    from ..llm.curriculum import run_experiment
    from ..llm.dataset import build_examples, difficulty_summary

    train = [r for r in complaints if r["split"] == "train"][:n]
    test = [r for r in complaints if r["split"] == "test"][:800]
    examples = build_examples(train, index)
    difficulty = {e.example_id: e.difficulty for e in examples}
    subset = [r for r in train if r["complaint_id"] in difficulty]
    return {
        "dataset": difficulty_summary(examples),
        "experiment": run_experiment(subset, test, difficulty),
    }


def check_gates(report: dict[str, Any], gates: Gates = DEFAULT_GATES) -> dict[str, Any]:
    models = report.get("models") or {}
    actual = {
        "deadline_exactness": report["deadlines"]["exactness"],
        "citation_resolution": report["workflow"]["citation_resolution"],
        "rag_recall_at_k": report["retrieval"].get(f"recall@{DEFAULT_RAG.top_k}"),
        "issue_macro_f1": (models.get("issue_classifier") or {}).get("test_score"),
        "outcome_balanced_accuracy": (models.get("outcome_model") or {}).get("test_score"),
        "escalation_roc_auc": (models.get("escalation_model") or {}).get("test_score"),
        "fraud_recall_at_budget": (models.get("fraud_model") or {}).get("test_score"),
    }

    results = []
    for name, minimum in [
        ("deadline_exactness", gates.deadline_exactness),
        ("citation_resolution", gates.citation_resolution),
        ("rag_recall_at_k", gates.rag_recall_at_k),
        ("issue_macro_f1", gates.issue_macro_f1),
        ("outcome_balanced_accuracy", gates.outcome_balanced_accuracy),
        ("escalation_roc_auc", gates.escalation_roc_auc),
        ("fraud_recall_at_budget", gates.fraud_recall_at_budget),
    ]:
        value = actual.get(name)
        # A gate whose metric was not produced is skipped, not silently passed:
        # a missing number and a passing number are different states.
        status = "skipped" if value is None else ("pass" if value >= minimum else "FAIL")
        results.append({
            "gate": name, "minimum": minimum, "actual": value, "status": status,
            "rationale": gates.thresholds.get(name, ""),
        })

    # An invented date is a hard stop regardless of any rate.
    if report["workflow"]["invented_dates"] > 0:
        results.append({
            "gate": "no_invented_dates", "minimum": 0, "actual": report["workflow"]["invented_dates"],
            "status": "FAIL",
            "rationale": "A date in a consumer letter that the rules engine did not produce is unacceptable at any rate.",
        })

    return {
        "passed": not any(r["status"] == "FAIL" for r in results),
        "results": results,
        "failed": [r for r in results if r["status"] == "FAIL"],
        "skipped": [r["gate"] for r in results if r["status"] == "skipped"],
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row.get(key))] = out.get(str(row.get(key)), 0) + 1
    return out

"""Command line entry point: ``disputes <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import REPORT_DIR


def cmd_build(args) -> int:
    from .data.layers import build_all

    print(json.dumps(build_all(strict=not args.lenient), indent=2, default=str))
    return 0


def cmd_verify(args) -> int:
    from .data.manifest import DataManifest

    report = DataManifest().verify()
    print(json.dumps(report, indent=2))
    return 0 if report["clean"] else 1


def cmd_train(args) -> int:
    from .models.train import train_all

    for name, model in train_all().items():
        d = model.to_dict()
        print(f"{name:20} {d['metric']:30} val={d['validation_score']:.4f} test={d['test_score']:.4f}")
    return 0


def cmd_deadlines(args) -> int:
    from .eval.deadlines import run

    report = run()
    print(f"{report['passed']}/{report['cases']} hand-computed cases pass (exactness {report['exactness']})")
    for row in report["rows"]:
        mark = "ok  " if row["passed"] else "FAIL"
        print(f"  [{mark}] {row['case']}")
        if not row["passed"]:
            print(f"          expected {row['expected']!r}, got {row['actual']!r} {row['error'] or ''}")
            print(f"          {row['why']}")
    return 0 if report["exactness"] == 1.0 else 1


def cmd_dispute(args) -> int:
    from .workflow.graph import audit_trail
    from .workflow.nodes import process

    if args.file:
        complaint = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        complaint = {
            "complaint_id": args.id or "cli-001",
            "issue": args.issue,
            "narrative": args.narrative or "",
            "disputed_amount": args.amount,
            "transaction_date": args.transaction_date,
            "statement_date": args.statement_date,
            "notice_date": args.notice_date,
            "point_of_sale": args.point_of_sale,
            "provisional_credit_given": args.provisional_credit,
        }

    state = process(complaint, engine=getattr(args, "engine", "auto"))
    print(state["draft"].text if state.get("draft") else json.dumps(state.get("outcome"), indent=2))
    print("\n--- verification ---")
    verification = state.get("verification")
    print(json.dumps(verification.to_dict() if verification else None, indent=2))
    print("--- deadlines ---")
    print(json.dumps(state.get("deadlines", {}), indent=2))
    if args.trace:
        print(f"--- audit trail ({state.get('_engine')} engine) ---")
        print(json.dumps(audit_trail(state), indent=2))
    return 0


def cmd_regulations(args) -> int:
    from .rag.index import RegulationIndex

    for hit in RegulationIndex().search(args.question, args.k, regulation=args.regulation):
        print(f"\n[{hit['id']}] ({hit['regulation']}) {hit['title']}")
        print(f"  {hit['text'][:280]}...")
    return 0


def cmd_curriculum(args) -> int:
    from .data.layers import load_gold
    from .llm.curriculum import run_experiment
    from .llm.dataset import build_examples, difficulty_summary, write_dataset
    from .rag.index import RegulationIndex

    rows = load_gold("complaints")
    train = [r for r in rows if r["split"] == "train"][: args.n]
    test = [r for r in rows if r["split"] == "test"][:800]
    examples = build_examples(train, RegulationIndex())
    print(json.dumps(difficulty_summary(examples), indent=2))
    print("curriculum dataset:", write_dataset(examples, curriculum=True))
    print("shuffled dataset:  ", write_dataset(examples, curriculum=False))

    difficulty = {e.example_id: e.difficulty for e in examples}
    subset = [r for r in train if r["complaint_id"] in difficulty]
    print(json.dumps(run_experiment(subset, test, difficulty), indent=2))
    return 0


def cmd_qlora_plan(args) -> int:
    from .llm.qlora import QLoraConfig, plan

    print(json.dumps(plan(args.examples, QLoraConfig()).to_dict(), indent=2))
    return 0


def cmd_shadow(args) -> int:
    from .data.layers import load_gold
    from .models.train import load_estimators
    from .monitoring.shadow import run_shadow, value_segment

    rows = [r for r in load_gold("complaints") if r["split"] == "test"][: args.n]
    incumbent = load_estimators()["issue_classifier"]

    def candidate(row):
        """A stand-in challenger: the same model with a confidence fallback.

        Not a real competitor. It exists so the shadow harness runs against two
        models that genuinely disagree on a minority of cases, which is the
        situation the report is designed to read.
        """
        probabilities = incumbent.predict_proba([row["narrative"]])[0]
        order = probabilities.argsort()[::-1]
        classes = incumbent.classes_
        return classes[order[0]] if probabilities[order[0]] > 0.35 else classes[order[1]]

    run = run_shadow(
        rows,
        incumbent=lambda r: incumbent.predict([r["narrative"]])[0],
        candidate=candidate,
        label="label_issue",
        segment=value_segment,
        incumbent_version="issue_classifier@production",
        candidate_version="issue_classifier@low-confidence-fallback",
    )
    print(json.dumps(run.report(), indent=2))
    print("saved to", run.save())
    return 0


def cmd_drift(args) -> int:
    from .data.layers import load_gold
    from .monitoring.drift import reference_window, run_monitors

    rows = load_gold("complaints")
    train = [r for r in rows if r["split"] == "train"]
    current = [r for r in rows if r["split"] == "test"]
    baseline = train if args.whole_training_set else reference_window(train, months=args.reference_months)
    print(json.dumps({
        "baseline": ("whole training set" if args.whole_training_set
                     else f"first {args.reference_months} months of training"),
        "baseline_rows": len(baseline),
        **run_monitors(baseline, current),
    }, indent=2))
    return 0


def cmd_eval(args) -> int:
    from .eval.run_eval import run_full_eval

    report = run_full_eval(with_curriculum=not args.fast, sample=args.sample)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, default=str))

    gates = report["gates"]
    print("\npromotion gates:")
    for row in gates["results"]:
        actual = "n/a" if row["actual"] is None else f"{row['actual']:.4f}"
        print(f"  [{row['status']:>7}] {row['gate']:<28} {actual} >= {row['minimum']}")
        if row["status"] == "FAIL" and row["rationale"]:
            print(f"            {row['rationale']}")

    if not gates["passed"] and not args.no_gate:
        print("\nPROMOTION GATES FAILED", file=sys.stderr)
        return 1
    print(f"\nreport written to {REPORT_DIR / 'eval_report.json'}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("disputes.service.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_workflow(args) -> int:
    """Render the graph.

    On the LangGraph engine this is LangGraph's own renderer walking the
    compiled graph, so what is printed is the thing that will execute rather
    than a drawing of it maintained alongside.
    """
    from .workflow.nodes import build_workflow

    workflow = build_workflow(engine=getattr(args, "engine", "auto"))
    print(f"%% engine: {workflow.engine}")
    print(workflow.to_mermaid())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="disputes", description="Payment dispute and complaint workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build the bronze, silver and gold layers")
    p.add_argument("--lenient", action="store_true", help="record validation failures without blocking")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify", help="re-hash the tracked data and report what moved")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("train", help="train the four models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("deadlines", help="run the hand-computed regulatory deadline cases")
    p.set_defaults(func=cmd_deadlines)

    p = sub.add_parser("dispute", help="run the eight-step workflow on one dispute")
    p.add_argument("--file", help="JSON file holding the complaint")
    p.add_argument("--id")
    p.add_argument("--issue", default="Unauthorized transactions or other transaction problem")
    p.add_argument("--narrative", default="")
    p.add_argument("--amount", type=float)
    p.add_argument("--transaction-date")
    p.add_argument("--statement-date")
    p.add_argument("--notice-date")
    p.add_argument("--point-of-sale", action="store_true")
    p.add_argument("--provisional-credit", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--engine", choices=["auto", "langgraph", "reference"], default="auto",
                   help="execution engine; auto picks langgraph when installed")
    p.set_defaults(func=cmd_dispute)

    p = sub.add_parser("regulations", help="query the regulation corpus")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--regulation", choices=["REG_E", "REG_Z", "FCRA", "FDCPA", "CIRCULAR"])
    p.set_defaults(func=cmd_regulations)

    p = sub.add_parser("curriculum", help="build the SFT dataset and run the curriculum experiment")
    p.add_argument("-n", type=int, default=2000)
    p.set_defaults(func=cmd_curriculum)

    p = sub.add_parser("qlora-plan", help="print the QLoRA training plan without running it")
    p.add_argument("--examples", type=int, default=2000)
    p.set_defaults(func=cmd_qlora_plan)

    p = sub.add_parser("shadow", help="run a shadow-mode comparison")
    p.add_argument("-n", type=int, default=600)
    p.set_defaults(func=cmd_shadow)

    p = sub.add_parser("drift", help="run the drift monitors against a reference window")
    p.add_argument("--reference-months", type=int, default=12,
                   help="length of the stable reference window taken from the start of training")
    p.add_argument("--whole-training-set", action="store_true",
                   help="use all of training as the baseline, which dilutes any shift that happened inside it")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("eval", help="run the full evaluation and the promotion gates")
    p.add_argument("--fast", action="store_true", help="skip the curriculum experiment")
    p.add_argument("--sample", type=int, default=120)
    p.add_argument("--no-gate", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("workflow", help="print the workflow as mermaid")
    p.add_argument("--engine", choices=["auto", "langgraph", "reference"], default="auto",
                   help="execution engine; auto picks langgraph when installed")
    p.set_defaults(func=cmd_workflow)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

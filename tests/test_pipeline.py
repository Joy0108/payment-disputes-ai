"""Data layers, retrieval, drafting, the workflow, the queue and monitoring."""

from __future__ import annotations

import pytest

from disputes.config import COMPLAINT_ISSUES
from disputes.data import validate as v
from disputes.data.layers import bronze_complaints, load_gold, silver_complaints
from disputes.data.manifest import DataManifest
from disputes.monitoring.drift import edges_from, psi, reference_window, run_monitors
from disputes.monitoring.shadow import ShadowRecord, ShadowRun, run_shadow, value_segment
from disputes.rag.draft import Draft, TemplateDrafter, verify
from disputes.rag.index import RegulationIndex, tokenize
from disputes.rules import deadlines as rules
from disputes.service.queue import DisputeQueue
from disputes.workflow.graph import END, START, StateMachine, WorkflowError, audit_trail
from disputes.workflow.langgraph_engine import langgraph_available
from disputes.workflow.nodes import build_workflow, process, select_engine


@pytest.fixture(scope="session")
def index():
    return RegulationIndex()


@pytest.fixture(scope="session")
def gold():
    try:
        return load_gold("complaints")
    except FileNotFoundError:
        pytest.skip("gold layer not built; run `disputes build`")


@pytest.fixture()
def dispute():
    return {
        "complaint_id": "T-0001",
        "issue": "Unauthorized transactions or other transaction problem",
        "narrative": "there are charges on my account I did not authorize. someone used my debit card without my permission.",
        "disputed_amount": 500.0,
        "transaction_date": "2024-10-28",
        "statement_date": "2024-11-01",
        "notice_date": "2024-11-20",
        "discovery_date": "2024-11-19",
    }


# --- data contracts ---------------------------------------------------------

def test_expectations_carry_a_consequence():
    expectation = v.not_null("complaint_id", "rows without an identifier cannot be traced")
    assert expectation.consequence


def test_a_blocking_failure_names_the_consequence():
    rows = [{"complaint_id": "a"}, {"complaint_id": None}]
    report = v.validate("test", rows, [v.not_null("complaint_id", "untraceable rows")])
    assert not report.passed
    with pytest.raises(v.DataContractError, match="untraceable rows"):
        report.raise_if_failed()


def test_a_warning_does_not_block():
    rows = [{"cls": "a"}] * 99 + [{"cls": "b"}]
    report = v.validate("test", rows, [v.class_balance("cls", 0.05, "rare class", severity="warning")])
    assert report.passed
    assert report.failures


def test_monotonic_catches_an_unordered_table():
    rows = [{"d": "2024-01-02"}, {"d": "2024-01-01"}]
    report = v.validate("test", rows, [v.monotonic("d", "temporal splits become random")])
    assert not report.passed


def test_silver_quarantines_rather_than_drops(tmp_path):
    bronze = bronze_complaints()
    broken = list(bronze.rows[:50])
    broken.append({**broken[0], "disputed_amount": "not-a-number"})
    broken.append(dict(broken[0]))  # duplicate id
    bronze.rows = broken
    silver = silver_complaints(bronze)
    assert len(silver.rows) + len(silver.quarantined) == len(broken)
    assert len(silver.quarantined) == 2
    assert all("_quarantine_reason" in row for row in silver.quarantined)


def test_the_temporal_split_is_frozen_into_the_row(gold):
    assert {r["split"] for r in gold} == {"train", "validation", "test"}
    train_max = max(r["date_received"] for r in gold if r["split"] == "train")
    test_min = min(r["date_received"] for r in gold if r["split"] == "test")
    assert train_max < test_min, "a row in train must not post-date a row in test"


def test_no_complaint_appears_in_two_splits(gold):
    seen = {}
    for row in gold:
        assert seen.setdefault(row["complaint_id"], row["split"]) == row["split"]


def test_the_manifest_detects_a_changed_file(tmp_path):
    target = tmp_path / "data.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = DataManifest(path=tmp_path / "manifest.json")
    manifest.track("thing", target, write_pointer=False)
    before = manifest.fingerprint()
    assert manifest.verify()["clean"]

    target.write_text("a,b\n1,3\n", encoding="utf-8")
    assert not manifest.verify()["clean"]
    manifest.track("thing", target, write_pointer=False)
    assert manifest.fingerprint() != before


# --- retrieval --------------------------------------------------------------

def test_spelled_numbers_and_numerals_share_tokens():
    assert "60" in tokenize("sixty days from transmittal")
    assert "sixty" in tokenize("60 days from transmittal")


def test_a_section_identifier_is_searchable_by_its_parts(index):
    hits = [h["id"] for h in index.search("1005.11 error resolution time limits", 5)]
    assert any(h.startswith("1005.11") for h in hits)


def test_the_governing_regulation_wins_over_a_lexical_match(index):
    """"Unauthorized use of a card" matches Reg E and Reg Z about equally."""
    query = "someone used my card without permission, what is my liability"
    reg_e = [h["regulation"] for h in index.search(query, 3, regulation="REG_E")]
    reg_z = [h["regulation"] for h in index.search(query, 3, regulation="REG_Z")]
    assert reg_e[0] == "REG_E"
    assert reg_z[0] == "REG_Z"


def test_retrieval_meets_its_gate(index):
    from disputes.eval.rag import evaluate

    result = evaluate(index)
    assert result["recall@4"] >= 0.80, [r for r in result["rows"] if r["missed_primary"]]
    assert result["regulation_routing_accuracy"] >= 0.85


def test_retrieval_agrees_with_the_rules_engine_on_every_stated_deadline(index):
    from disputes.eval.rag import evaluate

    agreement = evaluate(index)["deadline_agreement"]
    assert agreement["checked"] > 0
    assert agreement["disagreements"] == []


# --- drafting and verification ---------------------------------------------

def test_a_citation_that_resolves_to_nothing_is_caught():
    draft = Draft("The institution must respond within ten business days [reg:MADE-UP].", "template", "v3")
    result = verify(draft, {"1005.11(c)(1)"}, {"deadlines": []})
    assert not result.passed
    assert result.unresolved == ["MADE-UP"]


def test_a_date_the_rules_engine_did_not_produce_is_caught():
    deadlines = {"deadlines": [{"name": "investigation", "due": "2024-12-05"}]}
    draft = Draft("The investigation must conclude within ten days by 2024-12-31 [reg:1005.11(c)(1)].", "template", "v3")
    result = verify(draft, {"1005.11(c)(1)"}, deadlines)
    assert not result.passed
    assert result.invented_dates == ["2024-12-31"]


def test_a_correctly_grounded_draft_passes():
    deadlines = {"deadlines": [{"name": "investigation", "due": "2024-12-05"}]}
    draft = Draft("The investigation must conclude by 2024-12-05 [calc:investigation] [reg:1005.11(c)(1)].",
                  "template", "v3")
    assert verify(draft, {"1005.11(c)(1)"}, deadlines).passed


def test_the_template_drafter_only_quotes_computed_dates(index, dispute):
    deadlines = rules.regulation_e("2024-11-20", "2024-11-01", provisional_credit_given=True).to_dict()
    draft = TemplateDrafter().draft({
        "complaint": dispute, "deadlines": deadlines,
        "sections": index.search("unauthorized debit provisional credit", 3, regulation="REG_E"),
        "validation": {"ok": True, "issues": []}, "reason_code": {"mapped": False}, "risk": {},
    })
    computed = {d["due"] for d in deadlines["deadlines"]}
    assert set(draft.dates()) <= computed


# --- workflow ---------------------------------------------------------------

def test_the_workflow_runs_all_eight_steps(dispute):
    state = process(dispute)
    assert state["_path"] == [
        "intake", "classify", "validate", "compute_deadlines", "assess_risk",
        "retrieve_regulation", "draft", "verify", "route",
    ]
    assert state["verification"].passed


def test_deadline_computation_is_a_required_stage():
    machine = StateMachine("t")
    machine.add_node("a", lambda s: {})
    machine.add_node("compute_deadlines", lambda s: {}, required=True)
    machine.add_edge(START, "a")
    machine.add_edge("a", END)
    with pytest.raises(WorkflowError, match="required stage"):
        machine.invoke({})


def test_an_uncategorisable_complaint_goes_to_a_human():
    """The escape path skips the required stages, and that is correct.

    "Required" means required on any path that produces a letter. A complaint
    the pipeline cannot categorise produces no letter, so demanding that it
    computed a deadline first would be demanding it guess.
    """
    state = process({"complaint_id": "T-0002", "issue": None, "narrative": ""})
    assert state["outcome"]["queue"] == "manual_review"
    assert state["outcome"]["requires_human_approval"]
    assert state["_path"] == ["intake", "classify", "unclassifiable"]


def test_a_late_notice_changes_the_letter(dispute):
    late = {**dispute, "notice_date": "2025-03-01", "statement_date": "2024-11-01"}
    state = process(late)
    assert state["deadlines"]["consumer_notice_timely"] is False
    assert "after the notice period closed" in state["draft"].text


def test_a_reconciliation_failure_reaches_the_letter(dispute):
    broken = {**dispute, "disputed_amount": 500.0,
              "transactions": [{"id": "a", "amount": 100.0}]}
    state = process(broken)
    assert not state["validation"]["ok"]
    assert "Information we still need" in state["draft"].text
    assert state["outcome"]["requires_human_approval"]


def test_the_audit_trail_records_every_transition(dispute):
    trail = audit_trail(process(dispute))
    assert [c["node"] for c in trail] == [
        "intake", "classify", "validate", "compute_deadlines", "assess_risk",
        "retrieve_regulation", "draft", "verify", "route",
    ]
    assert all("duration_ms" in c for c in trail)


def test_the_issue_drives_which_regulation_is_applied():
    for issue, expected in [
        ("Problem with a purchase shown on your statement", "REG_Z"),
        ("Incorrect information on your report", "FCRA"),
        ("Attempts to collect debt not owed", "FDCPA"),
    ]:
        state = process({
            "complaint_id": "T", "issue": issue, "narrative": "x",
            "transaction_date": "2024-10-01", "statement_date": "2024-10-15", "notice_date": "2024-11-01",
        })
        assert state["regulation"] == expected
        assert state["deadlines"]["regulation"] == expected


def test_every_issue_in_the_taxonomy_produces_a_deadline():
    for issue in COMPLAINT_ISSUES:
        state = process({
            "complaint_id": "T", "issue": issue, "narrative": "x",
            "transaction_date": "2024-10-01", "statement_date": "2024-10-15", "notice_date": "2024-11-01",
        })
        assert state["deadlines"]["deadlines"], issue


# --- queue ------------------------------------------------------------------

def test_the_queue_survives_an_interrupted_run(tmp_path, dispute):
    path = tmp_path / "queue.json"
    queue = DisputeQueue(path)
    job_id = queue.enqueue(dispute)
    queue.jobs[job_id]["status"] = "running"  # simulate the crash
    queue._save()

    recovered = DisputeQueue(path)
    assert recovered.jobs[job_id]["status"] == "queued"
    assert "requeued" in recovered.jobs[job_id]["note"]


def test_a_failing_job_is_retried_then_marked_failed(tmp_path, dispute):
    class Exploding:
        def invoke(self, state):
            raise RuntimeError("boom")

    queue = DisputeQueue(tmp_path / "queue.json")
    job_id = queue.enqueue(dispute)
    result = queue.drain(Exploding())
    assert queue.jobs[job_id]["status"] == "failed"
    assert queue.jobs[job_id]["attempts"] == 3
    assert result["failed"] == 1


def test_a_drained_queue_records_the_result(tmp_path, dispute):
    from disputes.workflow.nodes import build_workflow

    queue = DisputeQueue(tmp_path / "queue.json")
    job_id = queue.enqueue(dispute)
    queue.drain(build_workflow())
    job = queue.get(job_id)
    assert job["status"] == "succeeded"
    assert job["result"]["citations"]
    assert job["result"]["verification"]["passed"]


# --- monitoring -------------------------------------------------------------

def test_psi_is_zero_when_nothing_moved():
    from collections import Counter

    dist = Counter({"a": 50, "b": 30, "c": 20})
    assert psi(dist, dist)[0] == pytest.approx(0.0, abs=1e-9)


def test_bin_edges_come_from_the_baseline_only():
    """Recomputing bins on the current window makes PSI report zero forever."""
    baseline = list(range(100))
    edges = edges_from(baseline)
    shifted = [x + 500 for x in baseline]
    value, _ = psi(_bins(baseline, edges), _bins(shifted, edges))
    assert value > 1.0


def test_a_stable_reference_window_finds_drift_the_whole_training_set_hides(gold):
    """The reference window is the whole point.

    The response policy shifts partway through the training period, so a
    baseline built from all of training already contains both regimes and PSI
    reports stable while the model is measurably stale. A reference window taken
    from a period the pipeline believes was stable sees it.
    """
    train = [r for r in gold if r["split"] == "train"]
    current = [r for r in gold if r["split"] == "test"]

    diluted = run_monitors(train, current)
    focused = run_monitors(reference_window(train, months=12), current)

    assert diluted["overall_status"] == "stable"
    assert focused["overall_status"] in {"investigate", "alert"}
    assert focused["concept_drift_detected"]
    assert "re-baseline" in focused["action"]


def test_shadow_mode_segments_before_it_totals():
    # A candidate that wins on aggregate but changes one high-value decision
    # wrongly. Aggregate agreement alone would wave it through.
    rows = (
        [{"complaint_id": f"s{i}", "disputed_amount": 10, "label": "b"} for i in range(4)]
        + [{"complaint_id": "h1", "disputed_amount": 900, "label": "a"}]
    )
    run = run_shadow(
        rows,
        incumbent=lambda r: "a",
        candidate=lambda r: "b",
        label="label", segment=value_segment,
    )
    report = run.report()
    assert report["contested"]["candidate_correct"] == 4
    assert report["by_segment"]["high_value"]["disagreements"] == 1
    assert "high_value" in report["high_value_segments_with_disagreement"]
    assert "human review" in report["recommendation"]


def test_shadow_recommends_against_a_worse_candidate():
    run = ShadowRun("prod", "cand")
    run.records = [ShadowRecord(str(i), "a", "b", False, "standard", observed="a") for i in range(10)]
    assert "do not promote" in run.report()["recommendation"]


def _bins(values, edges):
    from collections import Counter

    counter = Counter()
    for value in values:
        index = 0
        while index < len(edges) and value > edges[index]:
            index += 1
        counter[str(index)] += 1
    return counter


# --- the two engines --------------------------------------------------------

needs_langgraph = pytest.mark.skipif(not langgraph_available(), reason="langgraph is not installed")


@needs_langgraph
def test_langgraph_is_the_default_engine():
    assert select_engine("auto") == "langgraph"
    assert build_workflow().engine == "langgraph"


@needs_langgraph
def test_both_engines_execute_the_same_graph_identically(dispute):
    """The conformance test.

    One declared topology, two executors. If LangGraph's reducers, conditional
    edges and recursion bound mean what the reference walker means, then the
    path, the audit trail, the letter and the routing decision are the same
    object. Any divergence is a misunderstanding of the framework, and this is
    where it surfaces rather than in production.
    """
    reference = build_workflow(engine="reference").invoke({"complaint": dispute})
    langgraph = build_workflow(engine="langgraph").invoke({"complaint": dispute})

    assert reference["_path"] == langgraph["_path"]
    assert reference["draft"].text == langgraph["draft"].text
    assert reference["outcome"] == langgraph["outcome"]
    assert reference["deadlines"] == langgraph["deadlines"]
    assert reference["verification"].passed == langgraph["verification"].passed

    left, right = audit_trail(reference), audit_trail(langgraph)
    assert [c["node"] for c in left] == [c["node"] for c in right]
    assert [c["next"] for c in left] == [c["next"] for c in right]
    assert [c["added"] for c in left] == [c["added"] for c in right]


@needs_langgraph
def test_the_escape_path_waives_required_stages_on_langgraph_too():
    state = build_workflow(engine="langgraph").invoke(
        {"complaint": {"complaint_id": "T-0002", "issue": None, "narrative": ""}})
    assert state["_path"] == ["intake", "classify", "unclassifiable"]
    assert state["outcome"]["requires_human_approval"]


@needs_langgraph
def test_the_checkpointer_records_every_super_step(dispute):
    workflow = build_workflow(engine="langgraph")
    state = workflow.invoke({"complaint": dispute})

    # The framework's own record, not a list this codebase maintains.
    history = workflow.state_history(state["_thread_id"])
    assert history[-1]["path"] == state["_path"]

    # Every stage appears as pending in some checkpoint before it completes,
    # so the record is a replayable sequence rather than a final summary.
    pending = [n for h in history for n in h["next"]]
    assert "compute_deadlines" in pending and "verify" in pending
    assert [h["completed"] for h in history if h["completed"]][-1] == "route"


@needs_langgraph
def test_the_human_gate_pauses_before_routing_and_resumes(dispute):
    """An interrupt is a durable pause, which is why it needs the checkpointer.

    The approver reads the finished draft and its verification - the thing
    being approved - rather than a queue assignment already made for them.
    """
    workflow = build_workflow(engine="langgraph", human_in_the_loop=True)
    paused = workflow.invoke({"complaint": dispute})

    assert workflow.interrupted(paused["_thread_id"])
    assert "route" not in paused["_path"]
    assert paused["draft"].text                      # the draft exists to be read
    assert "outcome" not in paused                   # but nothing has been routed

    resumed = workflow.resume(paused["_thread_id"])
    assert resumed["_path"][-1] == "route"
    assert resumed["outcome"]["complaint_id"] == "T-0001"


@needs_langgraph
def test_the_verify_cycle_is_bounded_by_the_recursion_limit(dispute, monkeypatch):
    """A verifier that never passes must terminate, not spin."""
    import disputes.workflow.nodes as nodes

    monkeypatch.setattr(nodes, "MAX_DRAFT_ATTEMPTS", 10_000)
    workflow = build_workflow(engine="langgraph")
    workflow.spec.max_steps = 6                      # below one full pass

    with pytest.raises(WorkflowError, match="exceeded max_steps"):
        workflow.invoke({"complaint": dispute})


@needs_langgraph
def test_the_graph_renders_from_langgraph_itself():
    mermaid = build_workflow(engine="langgraph").to_mermaid()
    assert "compute_deadlines" in mermaid and "unclassifiable" in mermaid

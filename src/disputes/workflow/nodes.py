"""The eight-step dispute workflow.

    intake -> classify -> validate -> compute_deadlines -> assess_risk
           -> retrieve_regulation -> draft -> verify (loop) -> route

The split between steps 4 and 7 is the design. Everything that has a right
answer - which regulation applies, what date the investigation must conclude,
which reason code the network expects - is computed deterministically in step 4
and handed to the generator as fact. The generator's contribution is prose, and
step 8 checks that it did not add anything.

Verification loops back into drafting up to a bound. It does not "warn and
continue": a draft that cites a section that does not exist, or states a date
the rules engine never produced, is not published.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import DEFAULT_LLM, DEFAULT_RAG, LlmConfig
from ..rag.draft import build_drafter, verify
from ..rag.index import RegulationIndex
from ..rules import deadlines as rules
from ..rules.validate import map_reason_code, validate_amounts, validate_dates
from .graph import END, START, StateMachine, WorkflowError

MAX_DRAFT_ATTEMPTS = 3


@dataclass
class Predictors:
    """The three complaint models, or ``None`` to run on the stated issue.

    Made optional deliberately. The workflow has to work before the models are
    trained, and an intake agent who has already categorised a complaint should
    not have the classifier's opinion substituted for theirs.
    """

    issue: Any = None
    outcome: Any = None
    escalation: Any = None

    @classmethod
    def from_artifacts(cls) -> Predictors:
        from ..models.train import load_estimators

        estimators = load_estimators()
        return cls(
            issue=estimators.get("issue_classifier"),
            outcome=estimators.get("outcome_model"),
            escalation=estimators.get("escalation_model"),
        )


def build_workflow(
    index: RegulationIndex | None = None,
    predictors: Predictors | None = None,
    llm: LlmConfig = DEFAULT_LLM,
) -> StateMachine:
    index = index if index is not None else RegulationIndex()
    predictors = predictors or Predictors()
    drafter = build_drafter(llm.backend, llm.prompt_version)

    # -- 1 -----------------------------------------------------------------
    def intake(state: dict[str, Any]) -> dict[str, Any]:
        complaint = dict(state["complaint"])
        complaint.setdefault("complaint_id", "unassigned")
        return {"complaint": complaint, "attempts": 0}

    # -- 2 -----------------------------------------------------------------
    def classify(state: dict[str, Any]) -> dict[str, Any]:
        complaint = state["complaint"]
        predicted: dict[str, Any] = {}

        if predictors.issue is not None and complaint.get("narrative"):
            issue = predictors.issue.predict([complaint["narrative"]])[0]
            probabilities = predictors.issue.predict_proba([complaint["narrative"]])[0]
            predicted["issue"] = issue
            predicted["issue_confidence"] = round(float(max(probabilities)), 4)
        issue = complaint.get("issue") or predicted.get("issue")

        if issue is None:
            return {"prediction": predicted, "error": "no issue supplied and no classifier available",
                    "issue": None, "regulation": None}

        return {
            "prediction": predicted,
            "issue": issue,
            "regulation": complaint.get("regulation") or rules.regulation_for_issue(issue),
        }

    # -- 3 -----------------------------------------------------------------
    def validate_fields(state: dict[str, Any]) -> dict[str, Any]:
        complaint = state["complaint"]
        amounts = validate_amounts(complaint.get("disputed_amount"), complaint.get("transactions"))
        dates = validate_dates(
            complaint.get("transaction_date"), complaint.get("statement_date"), complaint.get("notice_date"))
        issues = amounts.issues + dates.issues
        return {
            "validation": {
                "ok": not any(i.severity == "error" for i in issues),
                "issues": [i.to_dict() for i in issues],
                "normalised": {**amounts.normalised, **dates.normalised},
            },
            "reason_code": map_reason_code(complaint.get("reason_code"), state.get("issue")),
        }

    # -- 4 -----------------------------------------------------------------
    def compute_deadlines(state: dict[str, Any]) -> dict[str, Any]:
        """Deterministic. Never a model, never a prompt."""
        complaint = state["complaint"]
        regulation = state.get("regulation")
        normalised = state["validation"]["normalised"]

        try:
            if regulation == "REG_E":
                result = rules.regulation_e(
                    normalised["notice_date"], normalised["statement_date"],
                    account_opened=complaint.get("account_opened"),
                    point_of_sale=bool(complaint.get("point_of_sale")),
                    foreign_initiated=bool(complaint.get("foreign_initiated")),
                    provisional_credit_given=bool(complaint.get("provisional_credit_given")),
                )
                liability = rules.regulation_e_liability(
                    complaint.get("discovery_date") or normalised["notice_date"],
                    normalised["notice_date"], normalised["statement_date"],
                    access_device=bool(complaint.get("access_device", True)),
                )
            elif regulation == "REG_Z":
                result = rules.regulation_z(
                    normalised["notice_date"], normalised["statement_date"],
                    billing_cycle_days=int(complaint.get("billing_cycle_days", 30)))
                liability = None
            elif regulation == "FCRA":
                result = rules.fcra(
                    normalised["notice_date"],
                    additional_information_provided=bool(complaint.get("additional_information_provided")))
                liability = None
            else:
                result = rules.fdcpa(
                    complaint.get("first_contact_date") or normalised["notice_date"],
                    written_dispute_date=complaint.get("written_dispute_date"))
                liability = None
        except KeyError as exc:
            return {"deadlines": {"regulation": regulation, "deadlines": [], "findings": [],
                                  "error": f"missing date field {exc}"}, "liability": None}

        return {"deadlines": result.to_dict(), "liability": liability}

    # -- 5 -----------------------------------------------------------------
    def assess_risk(state: dict[str, Any]) -> dict[str, Any]:
        complaint = state["complaint"]
        risk: dict[str, Any] = {}
        narrative = complaint.get("narrative", "")

        if predictors.outcome is not None and narrative:
            features = (f"{narrative} __issue_{str(state['issue']).replace(' ', '_')} "
                        f"__size_{complaint.get('company_size', 'large')} "
                        f"__via_{complaint.get('submitted_via', 'Web')}")
            risk["predicted_outcome"] = predictors.outcome.predict([features])[0]
        if predictors.escalation is not None and narrative:
            features = (f"{narrative} __issue_{str(state['issue']).replace(' ', '_')} "
                        f"__resp_{risk.get('predicted_outcome', 'Closed with explanation').replace(' ', '_')} "
                        f"__size_{complaint.get('company_size', 'large')} __timely_True")
            risk["escalation_probability"] = round(float(predictors.escalation.predict_proba([features])[0][1]), 4)

        # Priority is a policy expression over the model outputs, not a model
        # output itself, so the rule is inspectable and the threshold is
        # somebody's decision rather than an implicit one.
        escalation = risk.get("escalation_probability", 0.0)
        untimely = state["deadlines"].get("consumer_notice_timely") is False
        risk["priority"] = (
            "high" if (escalation >= 0.65 or not state["validation"]["ok"])
            else "low" if (escalation < 0.35 and not untimely)
            else "standard"
        )
        return {"risk": risk}

    # -- 6 -----------------------------------------------------------------
    def retrieve_regulation(state: dict[str, Any]) -> dict[str, Any]:
        complaint = state["complaint"]
        query = " ".join(filter(None, [
            str(state.get("issue") or ""),
            complaint.get("narrative", "")[:400],
        ]))
        sections = index.search(query, DEFAULT_RAG.top_k, regulation=state.get("regulation"))
        return {"sections": sections}

    # -- 7 -----------------------------------------------------------------
    def draft(state: dict[str, Any]) -> dict[str, Any]:
        result = drafter.draft(
            {
                "complaint": state["complaint"],
                "deadlines": state["deadlines"],
                "sections": state["sections"],
                "validation": state["validation"],
                "reason_code": state["reason_code"],
                "risk": state["risk"],
            },
            critique=state.get("critique"),
        )
        return {"draft": result, "attempts": state.get("attempts", 0) + 1}

    # -- 8 -----------------------------------------------------------------
    def verify_draft(state: dict[str, Any]) -> dict[str, Any]:
        result = verify(state["draft"], set(index.by_id), state["deadlines"])
        return {"verification": result, "critique": result.critique()}

    def route(state: dict[str, Any]) -> dict[str, Any]:
        verification = state["verification"]
        risk = state["risk"]
        return {
            "outcome": {
                "complaint_id": state["complaint"].get("complaint_id"),
                "issue": state.get("issue"),
                "regulation": state.get("regulation"),
                "priority": risk.get("priority"),
                "queue": (
                    "manual_review" if not verification.passed or risk.get("priority") == "high"
                    else "auto_send"
                ),
                "verification_passed": verification.passed,
                "draft_attempts": state.get("attempts"),
                "requires_human_approval": not verification.passed or risk.get("priority") == "high",
            }
        }

    def unclassifiable(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "outcome": {
                "complaint_id": state["complaint"].get("complaint_id"),
                "queue": "manual_review",
                "reason": state.get("error", "the complaint could not be categorised"),
                "requires_human_approval": True,
            },
            "verification": None,
        }

    machine = StateMachine("dispute-workflow", max_steps=24)
    machine.add_node("intake", intake)
    machine.add_node("classify", classify)
    machine.add_node("validate", validate_fields)
    machine.add_node("compute_deadlines", compute_deadlines, required=True)
    machine.add_node("assess_risk", assess_risk)
    machine.add_node("retrieve_regulation", retrieve_regulation)
    machine.add_node("draft", draft)
    machine.add_node("verify", verify_draft, required=True)
    machine.add_node("route", route)
    machine.add_node("unclassifiable", unclassifiable)

    machine.set_escape_nodes({"unclassifiable"})
    machine.add_edge(START, "intake")
    machine.add_edge("intake", "classify")
    machine.add_conditional_edges(
        "classify",
        lambda s: "ok" if s.get("issue") else "unclassifiable",
        {"ok": "validate", "unclassifiable": "unclassifiable"},
    )
    machine.add_edge("validate", "compute_deadlines")
    machine.add_edge("compute_deadlines", "assess_risk")
    machine.add_edge("assess_risk", "retrieve_regulation")
    machine.add_edge("retrieve_regulation", "draft")
    machine.add_edge("draft", "verify")
    machine.add_conditional_edges(
        "verify",
        lambda s: "revise" if (not s["verification"].passed and s["attempts"] < MAX_DRAFT_ATTEMPTS) else "done",
        {"revise": "draft", "done": "route"},
    )
    machine.add_edge("route", END)
    machine.add_edge("unclassifiable", END)
    return machine


def process(
    complaint: dict[str, Any],
    index: RegulationIndex | None = None,
    predictors: Predictors | None = None,
    llm: LlmConfig = DEFAULT_LLM,
) -> dict[str, Any]:
    machine = build_workflow(index, predictors, llm)
    return machine.invoke({"complaint": complaint})


__all__ = ["Predictors", "build_workflow", "process", "WorkflowError", "MAX_DRAFT_ATTEMPTS"]

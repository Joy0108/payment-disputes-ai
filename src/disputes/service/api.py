"""FastAPI service.

Two entry points with different contracts, because disputes arrive both ways:

* ``POST /disputes`` runs the workflow synchronously. For an agent with a
  consumer on the phone.
* ``POST /disputes/batch`` enqueues and returns immediately. For the overnight
  file, where the caller is a scheduler and latency does not matter but
  throughput and restartability do.

Both return the audit trail. A response that says "manual review" without the
checkpoint list is not reviewable, and review is the whole point of routing
something to a human.

Models and the regulation index are loaded once at startup, not per request.
Loading a TF-IDF pipeline per request turns a 20ms call into a 2s one and is a
mistake that only shows up under load.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import DEFAULT_LLM
from ..rag.index import RegulationIndex
from ..workflow.graph import audit_trail
from ..workflow.nodes import Predictors, build_workflow
from .queue import DisputeQueue

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["index"] = RegulationIndex()
    try:
        STATE["predictors"] = Predictors.from_artifacts()
        STATE["models_loaded"] = True
    except (FileNotFoundError, ImportError):
        # The service is useful before the models are trained: the rules engine,
        # validation and retrieval all work on a stated issue. Starting without
        # them and saying so beats refusing to start.
        STATE["predictors"] = Predictors()
        STATE["models_loaded"] = False
    STATE["workflow"] = build_workflow(STATE["index"], STATE["predictors"], DEFAULT_LLM)
    STATE["queue"] = DisputeQueue()
    yield
    STATE.clear()


app = FastAPI(
    title="Payment dispute workflow",
    version="0.1.0",
    description="Intake to cited draft response, with deterministic deadline computation.",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    id: str | None = None
    amount: float


class DisputeRequest(BaseModel):
    complaint_id: str | None = None
    issue: str | None = Field(None, description="If omitted, the classifier assigns one.")
    narrative: str = ""
    disputed_amount: float | None = None
    transactions: list[Transaction] | None = None
    transaction_date: str | None = None
    statement_date: str | None = None
    notice_date: str | None = None
    discovery_date: str | None = None
    first_contact_date: str | None = None
    written_dispute_date: str | None = None
    account_opened: str | None = None
    reason_code: str | None = None
    point_of_sale: bool = False
    foreign_initiated: bool = False
    provisional_credit_given: bool = False
    access_device: bool = True
    billing_cycle_days: int = 30
    company_size: str = "large"
    submitted_via: str = "Web"

    def to_complaint(self) -> dict[str, Any]:
        data = self.model_dump()
        if data.get("transactions"):
            data["transactions"] = [t for t in data["transactions"]]
        return data


class DisputeResponse(BaseModel):
    complaint_id: str | None
    issue: str | None
    regulation: str | None
    deadlines: dict[str, Any]
    liability: dict[str, Any] | None
    validation: dict[str, Any]
    reason_code: dict[str, Any]
    risk: dict[str, Any]
    draft: str
    citations: list[str]
    verification: dict[str, Any] | None
    outcome: dict[str, Any]
    audit_trail: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": STATE.get("models_loaded", False),
        "regulation_sections": len(STATE["index"].by_id) if "index" in STATE else 0,
        "queue_depth": STATE["queue"].depth() if "queue" in STATE else 0,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.post("/disputes", response_model=DisputeResponse)
def create_dispute(request: DisputeRequest) -> DisputeResponse:
    try:
        state = STATE["workflow"].invoke({"complaint": request.to_complaint()})
    except Exception as exc:  # pragma: no cover - surfaced as a 422 rather than a 500
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc

    verification = state.get("verification")
    draft = state.get("draft")
    return DisputeResponse(
        complaint_id=state["complaint"].get("complaint_id"),
        issue=state.get("issue"),
        regulation=state.get("regulation"),
        deadlines=state.get("deadlines", {}),
        liability=state.get("liability"),
        validation=state.get("validation", {}),
        reason_code=state.get("reason_code", {}),
        risk=state.get("risk", {}),
        draft=draft.text if draft else "",
        citations=draft.cited() if draft else [],
        verification=verification.to_dict() if verification else None,
        outcome=state.get("outcome", {}),
        audit_trail=audit_trail(state),
    )


@app.post("/disputes/batch", status_code=202)
def enqueue_batch(requests: list[DisputeRequest], background: BackgroundTasks) -> dict[str, Any]:
    queue: DisputeQueue = STATE["queue"]
    ids = [queue.enqueue(r.to_complaint()) for r in requests]
    background.add_task(queue.drain, STATE["workflow"])
    return {"accepted": len(ids), "job_ids": ids, "queue_depth": queue.depth()}


@app.get("/disputes/batch/{job_id}")
def batch_status(job_id: str) -> dict[str, Any]:
    queue: DisputeQueue = STATE["queue"]
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return job


@app.get("/regulations/search")
def search_regulations(q: str, k: int = 4, regulation: str | None = None) -> dict[str, Any]:
    return {"query": q, "results": STATE["index"].search(q, k, regulation=regulation)}

"""Batch queue with a durable job record.

Deliberately small: a file-backed queue with explicit states, not a broker. What
it demonstrates is the property a real deployment needs, which is that a worker
dying mid-batch leaves a recoverable state rather than a silent gap.

Jobs move ``queued -> running -> succeeded | failed``, and the transition to
``running`` is written before the work starts. A job found in ``running`` at
startup was interrupted, and that is a different thing from a job that failed:
the first should be retried, the second should not be retried blindly.
"""

from __future__ import annotations

import json
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ARTIFACT_DIR

MAX_ATTEMPTS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DisputeQueue:
    def __init__(self, path: Path | None = None):
        self.path = path or (ARTIFACT_DIR / "queue.json")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pending: deque[str] = deque()
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.jobs = payload.get("jobs", {})
        for job_id, job in self.jobs.items():
            if job["status"] == "queued":
                self.pending.append(job_id)
            elif job["status"] == "running":
                # Interrupted, not failed. Requeue it and say so, rather than
                # leaving it stuck in a state no worker will ever revisit.
                job["status"] = "queued"
                job["note"] = "requeued after an interrupted run"
                self.pending.append(job_id)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"jobs": self.jobs}, indent=2, default=str), encoding="utf-8", newline="\n")

    # -- api ---------------------------------------------------------------
    def enqueue(self, complaint: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {
            "job_id": job_id, "status": "queued", "attempts": 0,
            "complaint": complaint, "enqueued_at": _now(), "result": None, "error": None,
        }
        self.pending.append(job_id)
        self._save()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def depth(self) -> int:
        return len(self.pending)

    def drain(self, workflow, limit: int | None = None) -> dict[str, Any]:
        processed = succeeded = failed = 0
        while self.pending and (limit is None or processed < limit):
            job_id = self.pending.popleft()
            job = self.jobs[job_id]

            job["status"] = "running"
            job["attempts"] += 1
            job["started_at"] = _now()
            self._save()  # written before the work, so a crash is detectable

            try:
                state = workflow.invoke({"complaint": job["complaint"]})
                draft = state.get("draft")
                verification = state.get("verification")
                job.update({
                    "status": "succeeded",
                    "finished_at": _now(),
                    "result": {
                        "outcome": state.get("outcome", {}),
                        "deadlines": state.get("deadlines", {}),
                        "citations": draft.cited() if draft else [],
                        "verification": verification.to_dict() if verification else None,
                        "draft": draft.text if draft else "",
                    },
                })
                succeeded += 1
            except Exception as exc:
                job["error"] = {"type": type(exc).__name__, "message": str(exc),
                                "traceback": traceback.format_exc(limit=4)}
                if job["attempts"] < MAX_ATTEMPTS:
                    job["status"] = "queued"
                    self.pending.append(job_id)
                else:
                    job["status"] = "failed"
                    job["finished_at"] = _now()
                    failed += 1
            processed += 1
            self._save()

        return {"processed": processed, "succeeded": succeeded, "failed": failed, "remaining": self.depth()}

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {"total": len(self.jobs), "by_status": counts, "pending": self.depth()}

    def clear(self) -> None:
        self.jobs.clear()
        self.pending.clear()
        self._save()

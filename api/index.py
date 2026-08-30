"""Vercel entrypoint.

Vercel looks for an ASGI ``app`` in a fixed set of paths, and this project's
service lives at ``src/disputes/service/api.py``, which is not one of them. This
module is the adapter: it puts ``src`` on the path, sets the three environment
choices a serverless runtime needs, and re-exports the same ``app`` the
Kubernetes deployment runs.

**Three things differ from a normal deployment, and each is a deliberate
choice rather than a workaround.**

*The filesystem is read-only except ``/tmp``.* The durable job queue writes
``queue.json``, so ``DISPUTES_ARTIFACT_DIR`` points at ``/tmp``. That makes the
queue work per invocation. It does **not** make it durable: serverless
instances are ephemeral and do not share ``/tmp``, so a job enqueued by one
request is not visible to the next. ``POST /disputes`` is the endpoint that
makes sense here; ``POST /disputes/batch`` needs the long-lived worker in
``deploy/`` to mean anything.

*The reference engine, not LangGraph.* Both engines execute the same declared
topology and the conformance test asserts they produce identical output, so
picking the dependency-free walker costs nothing behaviourally and keeps
``langchain-core``, ``langsmith`` and friends out of a bundle that has a size
limit. This is exactly what that walker was kept for.

*No trained models.* The estimator artifacts are build outputs and are not in
the repository. The service is designed to start without them and say so on
``/health``: the rules engine, field validation, retrieval, drafting and the
citation verifier all work from a stated issue. What is unavailable is the
issue classifier and the risk scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# setdefault, not assignment: anything configured in the Vercel dashboard wins.
os.environ.setdefault("DISPUTES_ARTIFACT_DIR", "/tmp/disputes-artifacts")
os.environ.setdefault("DISPUTES_REPORT_DIR", "/tmp/disputes-reports")
os.environ.setdefault("DISPUTES_ENGINE", "reference")

Path(os.environ["DISPUTES_ARTIFACT_DIR"]).mkdir(parents=True, exist_ok=True)

from disputes.service.api import app  # noqa: E402  (path setup must run first)

__all__ = ["app"]

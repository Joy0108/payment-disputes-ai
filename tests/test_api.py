from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="serving extras not installed")
pytest.importorskip("httpx", reason="httpx is needed by the test client")

from fastapi.testclient import TestClient  # noqa: E402

from disputes.service.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


DISPUTE = {
    "complaint_id": "API-1",
    "issue": "Unauthorized transactions or other transaction problem",
    "narrative": "there are charges on my account I did not authorize.",
    "disputed_amount": 250.0,
    "transaction_date": "2024-10-28",
    "statement_date": "2024-11-01",
    "notice_date": "2024-11-20",
    "discovery_date": "2024-11-19",
}


def test_health_reports_whether_models_are_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["regulation_sections"] > 0
    assert "models_loaded" in body


def test_a_dispute_returns_deadlines_citations_and_an_audit_trail(client):
    body = client.post("/disputes", json=DISPUTE).json()
    assert body["regulation"] == "REG_E"
    assert body["deadlines"]["deadlines"]
    assert body["citations"]
    assert body["verification"]["passed"]
    assert [c["node"] for c in body["audit_trail"]][:3] == ["intake", "classify", "validate"]


def test_the_response_never_contains_a_date_the_rules_engine_did_not_compute(client):
    body = client.post("/disputes", json=DISPUTE).json()
    computed = {d["due"] for d in body["deadlines"]["deadlines"]}
    import re

    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", body["draft"]):
        assert value in computed


def test_a_malformed_dispute_is_a_422_not_a_500(client):
    response = client.post("/disputes", json={"issue": "Fraud or scam", "disputed_amount": "not a number"})
    assert response.status_code == 422


def test_a_batch_is_accepted_and_the_jobs_are_trackable(client):
    response = client.post("/disputes/batch", json=[DISPUTE, {**DISPUTE, "complaint_id": "API-2"}])
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] == 2

    # TestClient runs background tasks on the way out of the request, so the
    # jobs are already drained by the time the response is read.
    job = client.get(f"/disputes/batch/{body['job_ids'][0]}").json()
    assert job["status"] in {"succeeded", "queued", "running"}
    if job["status"] == "succeeded":
        assert job["result"]["verification"]["passed"]


def test_an_unknown_job_is_a_404(client):
    assert client.get("/disputes/batch/does-not-exist").status_code == 404


def test_regulation_search_can_be_scoped(client):
    body = client.get("/regulations/search", params={"q": "unauthorized card use liability", "regulation": "REG_E"}).json()
    assert body["results"]
    assert body["results"][0]["regulation"] == "REG_E"

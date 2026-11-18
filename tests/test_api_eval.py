from fastapi.testclient import TestClient

from claims_agent.api import app
from claims_agent.eval import evaluate, to_markdown

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "offline"


def test_assess_endpoint():
    r = client.post("/claims/assess", json={
        "text": "Cambriolage le 12/02/2026 : la porte a été fracturée, bijoux volés. Contrat HOME-FR-200388. 3 400 EUR.",
        "reported_on": "2026-02-13",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "covered"
    assert body["citations"][0]["article_id"] == "HOME-05"
    assert len(body["trace"]) == 5


def test_assess_validates_input():
    assert client.post("/claims/assess", json={"text": "short"}).status_code == 422


def test_evaluation_regression():
    r = evaluate("offline")
    dev, holdout = r["splits"]["dev"], r["splits"]["holdout"]
    assert dev["decision_accuracy"] >= 0.95
    # Safety property: the agent must never approve a claim that should not be approved.
    assert dev["unsafe_approvals"] == [] and holdout["unsafe_approvals"] == []
    assert "Headline metrics" in to_markdown(r)

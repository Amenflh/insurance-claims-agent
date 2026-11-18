from datetime import date

import pytest

from claims_agent.graph import ClaimsAgent
from claims_agent.llm import LLMBackend, OfflineBackend, get_backend
from claims_agent.schemas import Decision


@pytest.fixture(scope="module")
def agent():
    return ClaimsAgent(OfflineBackend())


def steps(assessment):
    return [s.step for s in assessment.trace]


def test_covered_claim_cites_coverage_article(agent):
    a = agent.assess("On 2026-05-10 a stone hit my windscreen and it cracked. Policy AUTO-TN-100587. Quote: 640 TND.",
                     date(2026, 5, 11))
    assert a.decision == Decision.COVERED
    assert [c.article_id for c in a.citations] == ["AUTO-06"]
    assert not a.needs_human
    assert steps(a) == ["extract", "lookup_policy", "retrieve", "decide", "guardrails"]


def test_exclusion_overrides_coverage(agent):
    a = agent.assess("Le 21/02/2026 j'ai heurté un mur. Le contrôle d'alcoolémie était positif. "
                     "Contrat AUTO-TN-100231. Dégâts : 4 800 TND.", date(2026, 2, 22))
    assert a.decision == Decision.EXCLUDED
    assert a.citations[0].article_id == "AUTO-10"


def test_negated_exclusion_does_not_fire(agent):
    a = agent.assess("On 2026-04-22 another driver hit my car. No alcohol was involved. "
                     "Policy AUTO-TN-100587. Repair estimate: 3,450 TND.", date(2026, 4, 23))
    assert a.decision == Decision.COVERED


def test_outside_policy_period(agent):
    a = agent.assess("Accident le 12/01/2026, collision avec un camion. Contrat AUTO-TN-102310. 6 200 TND.",
                     date(2026, 1, 13))
    assert a.decision == Decision.EXCLUDED
    assert a.citations[0].article_id == "AUTO-02"


def test_missing_policy_routes_to_human_and_skips_retrieval(agent):
    a = agent.assess("Water came through the ceiling last weekend, about AED 3,000.", date(2026, 5, 18))
    assert a.decision == Decision.NEEDS_REVIEW and a.needs_human
    assert "policy_number" in a.missing_fields
    assert "retrieve" not in steps(a)


def test_unknown_policy_number(agent):
    a = agent.assess("Fire on 2026-07-08 in my kitchen. Policy HOME-AE-999999. AED 900.", date(2026, 7, 9))
    assert a.decision == Decision.NEEDS_REVIEW
    assert "not found in the register" in a.reasons[0]


def test_amount_above_authority_limit(agent):
    a = agent.assess("On 2026-10-11 an electrical fire destroyed our villa. Policy HOME-AE-200951. AED 240,000.",
                     date(2026, 10, 12))
    assert a.decision == Decision.NEEDS_REVIEW
    assert a.citations[0].article_id == "HOME-14"


class FakeClient:
    """Stands in for an LLM: returns canned JSON so the guardrails can be tested without a network."""

    def __init__(self, decision):
        self.decision = decision

    def complete_json(self, system, user):
        if "extract" in system:
            return {"language": "en", "claim_type": "home_fire", "incident_date": "2026-07-08", "amount": 900,
                    "currency": "AED", "policy_number": "HOME-AE-200951", "third_party": False}
        return self.decision


def test_guardrail_drops_hallucinated_citation():
    llm = LLMBackend(FakeClient({"decision": "covered", "citations": ["HOME-99"], "confidence": 0.95}), "fake")
    a = ClaimsAgent(llm).assess("Fire in the kitchen on 2026-07-08. Policy HOME-AE-200951. AED 900.",
                                date(2026, 7, 9))
    assert a.decision == Decision.NEEDS_REVIEW
    assert a.citations == []
    guard = next(s for s in a.trace if s.step == "guardrails")
    assert "ungrounded_citation" in guard.detail["fired"]


def test_guardrail_low_confidence():
    llm = LLMBackend(FakeClient({"decision": "covered", "citations": ["HOME-03"], "confidence": 0.4}), "fake")
    a = ClaimsAgent(llm).assess("Fire in the kitchen on 2026-07-08. Policy HOME-AE-200951. AED 900.",
                                date(2026, 7, 9))
    assert a.decision == Decision.NEEDS_REVIEW
    assert [c.article_id for c in a.citations] == ["HOME-03"]


def test_llm_backend_falls_back_to_offline_on_error():
    class Broken:
        def complete_json(self, system, user):
            raise ConnectionError("no network")

    a = ClaimsAgent(LLMBackend(Broken(), "broken")).assess(
        "Fire in the kitchen on 2026-07-08. Policy HOME-AE-200951. AED 900.", date(2026, 7, 9))
    assert a.decision == Decision.COVERED
    decide = next(s for s in a.trace if s.step == "decide")
    assert decide.detail["llm_fallback"] == "ConnectionError"


def test_backend_selection(monkeypatch):
    assert get_backend("offline").name == "offline"
    monkeypatch.setenv("CLAIMS_AGENT_BACKEND", "ollama")
    assert get_backend().name == "ollama"
    with pytest.raises(ValueError):
        get_backend("gpt-17")

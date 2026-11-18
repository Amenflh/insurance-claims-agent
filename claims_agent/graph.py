r"""Agentic claim-assessment workflow, orchestrated as a LangGraph ``StateGraph``.

    extract -> lookup_policy --(policy found)--> retrieve -> decide -> guardrails -> END
                             \--(no policy)---------------------------> guardrails -> END

Every node appends a ``TraceStep`` (duration + summary) so each assessment is auditable.
"""
from __future__ import annotations

import time
from datetime import date
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .decision import DecisionContext, DecisionResult
from .guardrails import apply_guardrails
from .llm import Backend, get_backend
from .policies import articles_by_id, load_register
from .retrieval import Retriever
from .schemas import ClaimAssessment, ClaimExtraction, Citation, Decision, Policy, TraceStep

TOP_K = 5


class ClaimState(TypedDict, total=False):
    claim_text: str
    reported_on: date
    extraction: ClaimExtraction
    policy: Policy | None
    retrieved: list[tuple[str, int, float]]
    result: DecisionResult
    guardrails: list[str]
    trace: list[TraceStep]


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


def _traced(name: str, fn):
    def node(state: ClaimState) -> dict[str, Any]:
        t0 = time.perf_counter()
        update, detail = fn(state)
        step = TraceStep(step=name, duration_ms=round((time.perf_counter() - t0) * 1000, 2), detail=detail)
        update["trace"] = state.get("trace", []) + [step]
        return update
    return node


def build_graph(backend: Backend, top_k: int = TOP_K, retrieval_mode: str = "hybrid"):
    def extract(state):
        ex = backend.extract(state["claim_text"])
        return {"extraction": ex}, ex.model_dump(mode="json")

    def lookup_policy(state):
        number = state["extraction"].policy_number
        policy = load_register().get(number) if number else None
        return {"policy": policy}, {"policy_number": number, "found": policy is not None}

    def retrieve(state):
        ex, policy = state["extraction"], state["policy"]
        query = f"{state['claim_text']} {ex.claim_type or ''}"
        hits = _retriever().search(query, product=policy.product, k=top_k, mode=retrieval_mode)
        retrieved = [(h.article_id, h.rank, h.score) for h in hits]
        return {"retrieved": retrieved}, {"mode": retrieval_mode, "hits": [f"{i}#{r}" for i, r, _ in retrieved]}

    def decide(state):
        arts = articles_by_id()
        ctx = DecisionContext(state["claim_text"], state["extraction"], state["policy"],
                              [(arts[i], r) for i, r, _ in state["retrieved"]], state["reported_on"])
        res = backend.decide(ctx)
        return {"result": res}, {"decision": res.decision.value, "citations": res.citations,
                                 "confidence": res.confidence, **res.details}

    def guardrails(state):
        ex = state["extraction"]
        if state.get("policy") is None:
            reason = ("No policy number found in the declaration." if ex.policy_number is None
                      else f"Policy {ex.policy_number} not found in the register.")
            res = DecisionResult(Decision.NEEDS_REVIEW, [], [reason, "Routed to a human claims handler."],
                                 round(max(0.0, 1 - 0.15 * len(ex.missing_fields())), 2))
            return {"result": res, "guardrails": ["missing_information"]}, {"fired": ["missing_information"]}
        arts = articles_by_id()
        ctx = DecisionContext(state["claim_text"], ex, state["policy"],
                              [(arts[i], r) for i, r, _ in state["retrieved"]], state["reported_on"])
        res, fired = apply_guardrails(state["result"], ex, ctx.available_ids)
        return {"result": res, "guardrails": fired}, {"fired": fired, "final": res.decision.value}

    g = StateGraph(ClaimState)
    for name, fn in [("extract", extract), ("lookup_policy", lookup_policy), ("retrieve", retrieve),
                     ("decide", decide), ("guardrails", guardrails)]:
        g.add_node(name, _traced(name, fn))
    g.add_edge(START, "extract")
    g.add_edge("extract", "lookup_policy")
    g.add_conditional_edges("lookup_policy", lambda s: "retrieve" if s.get("policy") else "guardrails",
                            {"retrieve": "retrieve", "guardrails": "guardrails"})
    g.add_edge("retrieve", "decide")
    g.add_edge("decide", "guardrails")
    g.add_edge("guardrails", END)
    return g.compile()


class ClaimsAgent:
    def __init__(self, backend: Backend | None = None, top_k: int = TOP_K, retrieval_mode: str = "hybrid"):
        self.backend = backend or get_backend()
        self.graph = build_graph(self.backend, top_k, retrieval_mode)

    def assess(self, text: str, reported_on: date | None = None) -> ClaimAssessment:
        state = self.graph.invoke({"claim_text": text, "reported_on": reported_on or date.today(), "trace": []})
        res: DecisionResult = state["result"]
        arts = articles_by_id()
        ex: ClaimExtraction = state["extraction"]
        return ClaimAssessment(
            decision=res.decision,
            confidence=res.confidence,
            needs_human=res.decision == Decision.NEEDS_REVIEW,
            citations=[Citation(article_id=c, title=arts[c].title, excerpt=arts[c].text[:240]) for c in res.citations],
            reasons=res.reasons,
            missing_fields=ex.missing_fields(),
            extraction=ex,
            policy=state.get("policy"),
            backend=self.backend.name,
            trace=state["trace"],
        )

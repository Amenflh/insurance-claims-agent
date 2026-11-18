"""Interchangeable reasoning backends.

* ``offline`` (default): deterministic rules + retrieval, no network, used by tests and the evaluation.
* ``openai``: any OpenAI-compatible Chat Completions endpoint (OpenAI, Azure OpenAI proxy, vLLM, LM Studio...).
* ``ollama``: a local Ollama server.

Select with ``CLAIMS_AGENT_BACKEND``. LLM outputs are parsed as JSON and validated with the same Pydantic schemas;
on any failure the step falls back to the offline backend and the fallback is recorded in the trace.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

from .decision import DecisionContext, DecisionResult, rule_based_decision
from .extraction import extract as rule_based_extract
from .schemas import ALL_PERILS, ClaimExtraction, Decision


class Backend(Protocol):
    name: str

    def extract(self, text: str) -> ClaimExtraction: ...

    def decide(self, ctx: DecisionContext) -> DecisionResult: ...


class OfflineBackend:
    name = "offline"

    def extract(self, text: str) -> ClaimExtraction:
        return rule_based_extract(text)

    def decide(self, ctx: DecisionContext) -> DecisionResult:
        return rule_based_decision(ctx)


class ChatClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class OpenAICompatibleClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("CLAIMS_LLM_MODEL", "gpt-4o-mini")

    def complete_json(self, system: str, user: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=60,
        )
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


class OllamaClient:
    def __init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("CLAIMS_LLM_MODEL", "llama3.1:8b")

    def complete_json(self, system: str, user: str) -> dict:
        r = httpx.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])


EXTRACT_PROMPT = f"""You extract structured data from insurance claim declarations written in French or English.
Return a JSON object with keys: language ("fr"|"en"), claim_type (one of {list(ALL_PERILS)} or null),
incident_date (YYYY-MM-DD or null), amount (number or null), currency (ISO code or null),
policy_number (string or null), third_party (boolean). Never guess: use null when the text does not say."""

DECIDE_PROMPT = """You are a claims assessor. Decide whether the claim is "covered", "excluded" or "needs_review",
using ONLY the policy articles provided. Exclusions override coverage. If information is missing or the articles do
not settle the question, answer "needs_review". Return JSON: {"decision": ..., "citations": [article ids],
"reasons": [short strings], "confidence": number between 0 and 1}. Cite only ids that appear in the articles."""


class LLMBackend:
    """Uses a chat model for extraction and decision; falls back to offline rules on any error."""

    def __init__(self, client: ChatClient, name: str) -> None:
        self.client, self.name, self.fallback = client, name, OfflineBackend()

    def extract(self, text: str) -> ClaimExtraction:
        try:
            data = self.client.complete_json(EXTRACT_PROMPT, text)
            data["claim_type_margin"] = 2  # an LLM gives no lexical margin; trust its label
            return ClaimExtraction(**data)
        except Exception:  # noqa: BLE001 - any parsing/network error -> deterministic fallback
            return self.fallback.extract(text)

    def decide(self, ctx: DecisionContext) -> DecisionResult:
        articles = [a for a, _ in ctx.retrieved] + ctx.conditions
        user = json.dumps({
            "claim": ctx.claim_text,
            "extracted": ctx.extraction.model_dump(mode="json"),
            "policy": ctx.policy.model_dump(mode="json"),
            "reported_on": ctx.reported_on.isoformat(),
            "articles": [{"id": a.id, "title": a.title, "text": a.text} for a in articles],
        }, ensure_ascii=False)
        try:
            data = self.client.complete_json(DECIDE_PROMPT, user)
            return DecisionResult(
                decision=Decision(data["decision"]),
                citations=[str(c) for c in data.get("citations", [])],
                reasons=[str(r) for r in data.get("reasons", [])],
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception as exc:  # noqa: BLE001
            res = self.fallback.decide(ctx)
            res.details["llm_fallback"] = type(exc).__name__
            return res


def get_backend(name: str | None = None) -> Backend:
    name = (name or os.environ.get("CLAIMS_AGENT_BACKEND", "offline")).lower()
    if name == "offline":
        return OfflineBackend()
    if name == "openai":
        return LLMBackend(OpenAICompatibleClient(), "openai")
    if name == "ollama":
        return LLMBackend(OllamaClient(), "ollama")
    raise ValueError(f"Unknown backend {name!r} (expected offline, openai or ollama)")

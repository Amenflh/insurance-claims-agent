"""HTTP API.  Run: ``uvicorn claims_agent.api:app --reload``"""
from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .graph import ClaimsAgent
from .policies import load_articles, load_register
from .schemas import ClaimAssessment, ClaimRequest

app = FastAPI(
    title="Insurance Claims Agent",
    version=__version__,
    description="Assess a free-text insurance claim against policy wordings, with cited articles and human-review "
                "routing. All data is synthetic.",
)
agent = ClaimsAgent()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": agent.backend.name, "articles": len(load_articles()),
            "policies": len(load_register())}


@app.post("/claims/assess", response_model=ClaimAssessment)
def assess(req: ClaimRequest) -> ClaimAssessment:
    return agent.assess(req.text, req.reported_on)

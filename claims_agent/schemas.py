"""Typed data contracts shared by every step of the pipeline (validated with Pydantic)."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PERILS: dict[str, tuple[str, ...]] = {
    "auto": ("auto_collision", "auto_theft", "auto_glass", "auto_fire", "auto_natural", "auto_liability"),
    "home": ("home_water", "home_fire", "home_theft", "home_storm"),
    "health": ("health_hospital", "health_outpatient", "health_dental", "health_optical", "health_maternity"),
}
ALL_PERILS = tuple(p for perils in PERILS.values() for p in perils)
REQUIRED_FIELDS = ("policy_number", "incident_date", "claim_type", "amount")


class Decision(str, Enum):
    COVERED = "covered"
    EXCLUDED = "excluded"
    NEEDS_REVIEW = "needs_review"


class ClaimExtraction(BaseModel):
    """Structured view of a free-text first notice of loss (FNOL)."""

    language: Literal["fr", "en"] = "en"
    claim_type: str | None = Field(None, description="Peril code, e.g. auto_theft")
    incident_date: date | None = None
    amount: float | None = Field(None, ge=0)
    currency: str | None = None
    policy_number: str | None = None
    third_party: bool = False
    claim_type_margin: int = Field(0, description="Keyword-score margin between best and second claim type")

    @field_validator("claim_type")
    @classmethod
    def _known_peril(cls, v: str | None) -> str | None:
        if v is not None and v not in ALL_PERILS:
            raise ValueError(f"unknown claim_type {v!r}")
        return v

    @field_validator("policy_number")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) is None]


class Policy(BaseModel):
    policy_number: str
    product: Literal["auto", "home", "health"]
    start: date
    end: date
    currency: str
    review_threshold: float


class Article(BaseModel):
    id: str
    product: str
    title: str
    text: str
    kind: Literal["coverage", "exclusion", "condition", "definition"]
    perils: list[str] = []
    applies_to: list[str] = []
    triggers: list[str] = []
    keywords: list[str] = []
    check: str | None = None
    params: dict[str, str] = {}


class RetrievedArticle(BaseModel):
    article_id: str
    score: float
    rank: int


class Citation(BaseModel):
    article_id: str
    title: str
    excerpt: str


class TraceStep(BaseModel):
    step: str
    duration_ms: float
    detail: dict


class ClaimRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    reported_on: date | None = Field(None, description="Date the claim was notified (defaults to today)")


class ClaimAssessment(BaseModel):
    decision: Decision
    confidence: float = Field(..., ge=0, le=1)
    needs_human: bool
    citations: list[Citation]
    reasons: list[str]
    missing_fields: list[str]
    extraction: ClaimExtraction
    policy: Policy | None = None
    backend: str
    trace: list[TraceStep]

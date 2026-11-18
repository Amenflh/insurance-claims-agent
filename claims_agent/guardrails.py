"""Backend-independent safety checks applied after the decision step."""
from __future__ import annotations

import os

from .decision import DecisionResult
from .schemas import ClaimExtraction, Decision

CONFIDENCE_THRESHOLD = float(os.environ.get("CLAIMS_CONFIDENCE_THRESHOLD", "0.7"))


def apply_guardrails(result: DecisionResult, extraction: ClaimExtraction, available_ids: set[str],
                     threshold: float = CONFIDENCE_THRESHOLD) -> tuple[DecisionResult, list[str]]:
    """Return the (possibly downgraded) result and the list of guardrails that fired."""
    fired: list[str] = []
    reasons = list(result.reasons)

    unknown = [c for c in result.citations if c not in available_ids]
    citations = [c for c in result.citations if c in available_ids]
    if unknown:
        fired.append("ungrounded_citation")
        reasons.append(f"Dropped citations not present in the retrieved policy text: {unknown}.")

    decision = result.decision
    if decision != Decision.NEEDS_REVIEW:
        if unknown or not citations:
            fired.append("no_grounding")
        missing = extraction.missing_fields()
        if decision == Decision.COVERED and missing:
            fired.append("missing_information")
            reasons.append(f"Cannot approve with missing fields: {missing}.")
        if result.confidence < threshold:
            fired.append("low_confidence")
            reasons.append(f"Confidence {result.confidence:.2f} below threshold {threshold:.2f}.")
        if fired:
            decision = Decision.NEEDS_REVIEW
            reasons.append("Routed to a human claims handler.")

    return DecisionResult(decision, citations, reasons, result.confidence, result.details), fired

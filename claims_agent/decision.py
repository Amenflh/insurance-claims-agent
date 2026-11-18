"""Coverage reasoning.

``rule_based_decision`` is the deterministic ``offline`` reasoner. It only sees the articles returned by retrieval
(coverage and exclusion clauses) plus the condition clauses of the product (period, deadlines, thresholds), so a
retrieval miss propagates to the decision exactly as it would with an LLM reading the retrieved context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .policies import condition_article, load_articles
from .schemas import Article, ClaimExtraction, Decision, Policy
from .text import find_phrase, words


@dataclass
class DecisionContext:
    claim_text: str
    extraction: ClaimExtraction
    policy: Policy
    retrieved: list[tuple[Article, int]]  # (article, rank)
    reported_on: date

    @property
    def conditions(self) -> list[Article]:
        return [a for a in load_articles() if a.product == self.policy.product and a.kind == "condition"]

    @property
    def available_ids(self) -> set[str]:
        return {a.id for a, _ in self.retrieved} | {a.id for a in self.conditions}


@dataclass
class DecisionResult:
    decision: Decision
    citations: list[str]
    reasons: list[str]
    confidence: float
    details: dict = field(default_factory=dict)


def triggered_exclusions(ctx: DecisionContext) -> list[tuple[Article, str]]:
    toks = words(ctx.claim_text)
    hits = []
    for article, _ in ctx.retrieved:
        if article.kind != "exclusion":
            continue
        applies = "*" in article.applies_to or ctx.extraction.claim_type in article.applies_to
        trigger = next((t for t in article.triggers if find_phrase(toks, t)), None)
        if applies and trigger:
            hits.append((article, trigger))
    return hits


def _confidence(ctx: DecisionContext, decisive_rank: int | None) -> float:
    ex = ctx.extraction
    conf = 1.0 - 0.15 * len(ex.missing_fields())
    if ex.claim_type and ex.claim_type_margin <= 1:
        conf -= 0.15  # claim type is ambiguous between two perils
    if decisive_rank is not None and decisive_rank > 2:
        conf -= 0.10  # supporting clause was ranked low by retrieval
    return round(max(0.0, min(1.0, conf)), 2)


def rule_based_decision(ctx: DecisionContext) -> DecisionResult:
    ex, policy = ctx.extraction, ctx.policy
    rank_of = {a.id: r for a, r in ctx.retrieved}

    def result(decision: Decision, article: Article | None, reason: str, **details) -> DecisionResult:
        cites = [article.id] if article else []
        rank = rank_of.get(article.id) if article else None
        return DecisionResult(decision, cites, [reason], _confidence(ctx, rank), details)

    if ex.incident_date is None:
        return result(Decision.NEEDS_REVIEW, None, "Incident date is missing: cover period cannot be checked.")
    if not policy.start <= ex.incident_date <= policy.end:
        art = condition_article(policy.product, "policy_period")
        return result(Decision.EXCLUDED, art,
                      f"Incident on {ex.incident_date} is outside the policy period {policy.start} to {policy.end}.")

    exclusions = triggered_exclusions(ctx)
    if exclusions:
        art, trigger = exclusions[0]
        return result(Decision.EXCLUDED, art, f"Exclusion '{art.title}' applies (claim mentions '{trigger}').",
                      triggers=[t for _, t in exclusions])

    if ex.claim_type is None:
        return result(Decision.NEEDS_REVIEW, None, "The type of loss could not be identified from the declaration.")

    waiting = condition_article(policy.product, "waiting_period")
    if waiting and ex.claim_type in waiting.perils:
        days = int(waiting.params.get("days", 0))
        if (ex.incident_date - policy.start).days < days:
            return result(Decision.EXCLUDED, waiting,
                          f"Waiting period of {days} days since {policy.start} not completed.")

    coverage = next((a for a, _ in ctx.retrieved if a.kind == "coverage" and ex.claim_type in a.perils), None)
    if coverage is None:
        return result(Decision.NEEDS_REVIEW, None, f"No retrieved clause covers peril '{ex.claim_type}'.")

    deadline = condition_article(policy.product, "report_deadline")
    if deadline:
        days = int(deadline.params.get(f"days_{ex.claim_type}", deadline.params.get("days", 9999)))
        delay = (ctx.reported_on - ex.incident_date).days
        if delay > days:
            return result(Decision.NEEDS_REVIEW, deadline, f"Reported after {delay} days (limit {days} days).")

    if ex.amount is None:
        return result(Decision.NEEDS_REVIEW, coverage, "Claimed amount is missing.")
    if ex.amount > policy.review_threshold:
        art = condition_article(policy.product, "review_threshold")
        return result(Decision.NEEDS_REVIEW, art,
                      f"Amount {ex.amount:,.2f} exceeds the authority limit {policy.review_threshold:,.0f} {policy.currency}.")

    return result(Decision.COVERED, coverage, f"Peril '{ex.claim_type}' is covered by '{coverage.title}'.")

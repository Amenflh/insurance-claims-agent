"""Reproducible evaluation on the annotated synthetic claims.

    python -m claims_agent.eval [--backend offline] [--k 5] [--out reports/eval.md]

Two splits are scored separately:
* ``dev`` (claims.jsonl): used while writing the extraction lexicons and rules, so its scores are in-sample;
* ``holdout`` (claims_holdout.jsonl): written after the rules were frozen and never used to tune them.

Metrics: decision accuracy, per-class precision/recall, unsafe approvals, citation precision/recall, human
referral rate, field-level extraction F1, and a retrieval ablation (BM25 vs TF-IDF vs hybrid).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path

from .graph import ClaimsAgent
from .llm import get_backend
from .policies import DATA_DIR, articles_by_id, load_register
from .retrieval import Retriever

FIELDS = ("claim_type", "incident_date", "amount", "currency", "policy_number", "third_party")
LABELS = ("covered", "excluded", "needs_review")
SPLITS = {"dev": "claims.jsonl", "holdout": "claims_holdout.jsonl"}


def load_claims(split: str = "dev") -> list[dict]:
    path = DATA_DIR / "claims" / SPLITS[split]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(field: str, value):
    if value is None:
        return None
    if field == "amount":
        return round(float(value), 2)
    return str(value) if field == "incident_date" else value


def score(claims: list[dict], agent: ClaimsAgent) -> dict:
    t0 = time.perf_counter()
    rows = [(c, agent.assess(c["text"], date.fromisoformat(c["reported_on"]))) for c in claims]
    elapsed = time.perf_counter() - t0
    n = len(rows)
    gold = lambda c: c["expected"]["decision"]  # noqa: E731

    per_class = {}
    for label in LABELS:
        tp = sum(a.decision.value == label == gold(c) for c, a in rows)
        pred = sum(a.decision.value == label for _, a in rows)
        sup = sum(gold(c) == label for c, _ in rows)
        per_class[label] = {"precision": tp / pred if pred else None, "recall": tp / sup if sup else None,
                            "support": sup}

    cited = sum(len(a.citations) for _, a in rows)
    gold_cites = sum(len(c["expected"]["articles"]) for c, _ in rows)
    hit = sum(len({x.article_id for x in a.citations} & set(c["expected"]["articles"])) for c, a in rows)

    tp = fp = fn = 0
    field_ok = Counter()
    for c, a in rows:
        pred = a.extraction.model_dump(mode="json")
        for f in FIELDS:
            g, p = _norm(f, c["expected"]["fields"][f]), _norm(f, pred[f])
            field_ok[f] += p == g
            if p is not None and p == g:
                tp += 1
            else:
                fp += p is not None
                fn += g is not None
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    return {
        "n_claims": n,
        "decision_accuracy": sum(a.decision.value == gold(c) for c, a in rows) / n,
        "per_class": per_class,
        "unsafe_approvals": [c["id"] for c, a in rows if a.decision.value == "covered" and gold(c) != "covered"],
        "citation_precision": hit / cited if cited else 0.0,
        "citation_recall": hit / gold_cites if gold_cites else 0.0,
        "human_referral_rate": sum(a.needs_human for _, a in rows) / n,
        "expected_referral_rate": sum(gold(c) == "needs_review" for c, _ in rows) / n,
        "extraction": {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                       "field_accuracy": {f: field_ok[f] / n for f in FIELDS}},
        "errors": [{"id": c["id"], "expected": gold(c), "predicted": a.decision.value,
                    "expected_articles": c["expected"]["articles"], "cited": [x.article_id for x in a.citations],
                    "reason": a.reasons[0] if a.reasons else ""}
                   for c, a in rows if a.decision.value != gold(c)],
        "latency_ms_per_claim": 1000 * elapsed / n,
    }


def _is_retrievable(article_id: str) -> bool:
    """Coverage and exclusion clauses are retrieved; condition clauses are looked up by metadata."""
    return articles_by_id()[article_id].kind in {"coverage", "exclusion"}


def retrieval_ablation(claims: list[dict], k: int) -> dict:
    """Recall@k / MRR of the gold coverage & exclusion clauses, given gold product and claim type."""
    retriever, register = Retriever(), load_register()
    out = {}
    for mode in ("bm25", "tfidf", "hybrid"):
        hits = total = 0
        mrr = 0.0
        for c in claims:
            exp = c["expected"]
            pol = register.get(exp["fields"]["policy_number"] or "")
            targets = [x for x in exp["articles"] if _is_retrievable(x)]
            if not pol or not targets:
                continue
            ids = [r.article_id for r in
                   retriever.search(f"{c['text']} {exp['fields']['claim_type'] or ''}", pol.product, k, mode)]
            for t in targets:
                total += 1
                if t in ids:
                    hits += 1
                    mrr += 1 / (ids.index(t) + 1)
        out[mode] = {"recall_at_k": hits / total if total else 0.0, "mrr": mrr / total if total else 0.0, "n": total}
    return out


def evaluate(backend_name: str = "offline", k: int = 5) -> dict:
    agent = ClaimsAgent(get_backend(backend_name), top_k=k)
    splits = {s: load_claims(s) for s in SPLITS}
    return {
        "backend": backend_name,
        "top_k": k,
        "splits": {s: score(claims, agent) for s, claims in splits.items()},
        "retrieval_ablation": retrieval_ablation(splits["dev"] + splits["holdout"], k),
    }


def to_markdown(r: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    d, h = r["splits"]["dev"], r["splits"]["holdout"]
    both = lambda f: f"| {pct(f(d))} | {pct(f(h))} |"  # noqa: E731
    lines = [
        "# Evaluation report",
        "",
        f"Backend: `{r['backend']}` · top-k: {r['top_k']} · claims: {d['n_claims']} dev + {h['n_claims']} held-out "
        f"(synthetic, hand-annotated) · mean latency: {d['latency_ms_per_claim']:.1f} ms/claim",
        "",
        "Generated by `python -m claims_agent.eval`. Do not edit by hand.",
        "",
        "> The **dev** split was used to write the extraction lexicons and the rules, so its scores are in-sample and",
        "> optimistic. The **held-out** split was written after the rules were frozen and was not used for tuning.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Dev | Held-out |",
        "|---|---|---|",
        "| Decision accuracy (3 classes) " + both(lambda s: s["decision_accuracy"]),
        f"| Unsafe approvals (predicted *covered*, gold not covered) | {len(d['unsafe_approvals'])} | "
        f"{len(h['unsafe_approvals'])} |",
        "| Citation precision " + both(lambda s: s["citation_precision"]),
        "| Citation recall " + both(lambda s: s["citation_recall"]),
        "| Human referral rate (predicted) " + both(lambda s: s["human_referral_rate"]),
        "| Human referral rate (gold) " + both(lambda s: s["expected_referral_rate"]),
        "| Extraction F1 (6 fields, micro) " + both(lambda s: s["extraction"]["f1"]),
        "",
        "## Decision by class (precision / recall)",
        "",
        "| Class | Dev | Held-out |",
        "|---|---|---|",
        *[f"| {lab} | {pct(d['per_class'][lab]['precision'])} / {pct(d['per_class'][lab]['recall'])} "
          f"(n={d['per_class'][lab]['support']}) | {pct(h['per_class'][lab]['precision'])} / "
          f"{pct(h['per_class'][lab]['recall'])} (n={h['per_class'][lab]['support']}) |" for lab in LABELS],
        "",
        "## Extraction accuracy by field",
        "",
        "| Field | Dev | Held-out |",
        "|---|---|---|",
        *[f"| {f} | {pct(d['extraction']['field_accuracy'][f])} | {pct(h['extraction']['field_accuracy'][f])} |"
          for f in FIELDS],
        "",
        "## Retrieval ablation (all claims, gold product and claim type, coverage/exclusion clauses)",
        "",
        "| Retriever | Recall@k | MRR | Targets |",
        "|---|---|---|---|",
        *[f"| {m} | {pct(v['recall_at_k'])} | {v['mrr']:.3f} | {v['n']} |" for m, v in r["retrieval_ablation"].items()],
        "",
        "## Decision errors",
        "",
    ]
    errors = [(s, e) for s in ("dev", "holdout") for e in r["splits"][s]["errors"]]
    if errors:
        lines += ["| Split | Claim | Gold | Predicted | Gold articles | Cited | First reason |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {s} | {e['id']} | {e['expected']} | {e['predicted']} | {', '.join(e['expected_articles']) or '-'}"
                  f" | {', '.join(e['cited']) or '-'} | {e['reason']} |" for s, e in errors]
    else:
        lines.append("No decision errors.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="offline")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="reports/eval.md")
    args = ap.parse_args()
    r = evaluate(args.backend, args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(r)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

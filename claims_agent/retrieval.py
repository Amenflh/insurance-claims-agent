"""Hybrid retrieval over policy articles: BM25 (lexical) + TF-IDF cosine (vector), fused with Reciprocal Rank Fusion.

Pure Python on purpose: the corpus is a few dozen clauses, and no heavy dependency (numpy, torch) is needed.
The ``Retriever`` interface makes it easy to swap the vector side for real embeddings later.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from .policies import load_articles
from .schemas import Article, RetrievedArticle
from .text import tokenize

Mode = Literal["bm25", "tfidf", "hybrid"]


def article_document(a: Article) -> str:
    """Text indexed for an article: title, wording, peril codes and bilingual keyword annotations."""
    return " ".join([a.title, a.text, " ".join(a.perils), " ".join(a.keywords), " ".join(a.triggers)])


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out


class TfidfIndex:
    def __init__(self, docs: list[list[str]]):
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vectors = [self._vec(d) for d in docs]

    def _vec(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(t for t in tokens if t in self.idf)
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = self._vec(query)
        return [sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.vectors]


def _ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


class Retriever:
    def __init__(self, articles: tuple[Article, ...] | None = None, rrf_k: int = 60):
        self.articles = list(articles or load_articles())
        docs = [tokenize(article_document(a)) for a in self.articles]
        self.bm25, self.tfidf, self.rrf_k = BM25(docs), TfidfIndex(docs), rrf_k

    def search(self, query: str, product: str | None = None, k: int = 5, mode: Mode = "hybrid") -> list[RetrievedArticle]:
        q = tokenize(query)
        idx = [i for i, a in enumerate(self.articles) if product is None or a.product == product]
        bm = self.bm25.scores(q)
        tf = self.tfidf.scores(q)
        if mode == "bm25":
            fused = {i: bm[i] for i in idx}
        elif mode == "tfidf":
            fused = {i: tf[i] for i in idx}
        else:
            bm_r = _ranks([bm[i] for i in idx])
            tf_r = _ranks([tf[i] for i in idx])
            fused = {i: 1 / (self.rrf_k + bm_r[j]) + 1 / (self.rrf_k + tf_r[j]) for j, i in enumerate(idx)}
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [RetrievedArticle(article_id=self.articles[i].id, score=round(fused[i], 5), rank=r + 1)
                for r, i in enumerate(best) if bm[i] > 0 or tf[i] > 0]

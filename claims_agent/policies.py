"""Load the (synthetic) policy wordings and the policy register.

Each article in a Markdown wording starts with ``## Art. N — Title`` followed by an HTML comment holding
``key: value`` pairs separated by ``|``. List values are comma-separated.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from .schemas import Article, Policy

DATA_DIR = Path(os.environ.get("CLAIMS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
LIST_KEYS = {"perils", "applies_to", "triggers", "keywords"}
_ARTICLE_RE = re.compile(r"^## (?P<title>.+?)\n<!--(?P<meta>.+?)-->\n(?P<body>.*?)(?=^## |\Z)", re.S | re.M)


def parse_wording(markdown: str, product: str) -> list[Article]:
    articles = []
    for m in _ARTICLE_RE.finditer(markdown):
        meta: dict[str, object] = {}
        params: dict[str, str] = {}
        for part in m.group("meta").split("|"):
            key, _, value = part.partition(":")
            key, value = key.strip(), value.strip()
            if key in LIST_KEYS:
                meta[key] = [v.strip() for v in value.split(",") if v.strip()]
            elif key in {"id", "kind", "check"}:
                meta[key] = value
            elif key:
                params[key] = value
        articles.append(
            Article(product=product, title=m.group("title").strip(), text=" ".join(m.group("body").split()),
                    params=params, **meta)
        )
    return articles


@lru_cache(maxsize=1)
def load_articles() -> tuple[Article, ...]:
    articles: list[Article] = []
    for path in sorted((DATA_DIR / "policies").glob("*.md")):
        product = re.search(r"Product: `(\w+)`", path.read_text(encoding="utf-8")).group(1)
        articles.extend(parse_wording(path.read_text(encoding="utf-8"), product))
    return tuple(articles)


@lru_cache(maxsize=1)
def load_register() -> dict[str, Policy]:
    raw = json.loads((DATA_DIR / "policies" / "register.json").read_text(encoding="utf-8"))
    return {p["policy_number"]: Policy(**p) for p in raw["policies"]}


def articles_by_id() -> dict[str, Article]:
    return {a.id: a for a in load_articles()}


def condition_article(product: str, check: str) -> Article | None:
    return next((a for a in load_articles() if a.product == product and a.check == check), None)

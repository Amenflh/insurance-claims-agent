"""Bilingual (FR/EN) text normalisation shared by extraction, retrieval and rule matching."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = set(
    """a an and are as at be been by for from has have i in is it its my of on or our that the this to was we were
    with will your you au aux avec ce ces dans de des du elle en est et il je la le les leur mais me mon ma mes ne nous
    on ou par pas pour qu que qui sa se ses son sur un une vous ete a l d j n s c y""".split()
)
NEGATIONS = {"no", "not", "never", "without", "pas", "non", "sans", "aucun", "aucune", "jamais", "ni"}
_SUFFIXES = ("ements", "ement", "ations", "ation", "ings", "ing", "ees", "ee", "es", "ed", "er", "s", "e")


def normalize(text: str) -> str:
    """Lowercase, strip accents, unify apostrophes."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("œ", "oe"))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", normalize(text))


def stem(token: str) -> str:
    """Very light suffix stripping, good enough to merge vol/volé/volée or leak/leaks."""
    if "_" in token or token.isdigit():
        return token
    for suf in _SUFFIXES:
        if len(token) - len(suf) >= 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def find_phrase(text_words: list[str], phrase: str, negation_window: int = 3) -> bool:
    """True if ``phrase`` occurs in ``text_words`` and is not negated by a nearby preceding negation word.

    "no alcohol was involved" does not trigger the alcohol exclusion; "alcohol test was positive" does.
    """
    target = words(phrase)
    n = len(target)
    if not n:
        return False
    for i in range(len(text_words) - n + 1):
        if text_words[i : i + n] == target:
            window = text_words[max(0, i - negation_window) : i]
            if not NEGATIONS.intersection(window):
                return True
    return False

import pytest

from claims_agent.policies import load_articles, load_register
from claims_agent.retrieval import Retriever
from claims_agent.text import find_phrase, tokenize, words


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_policies_load():
    articles = load_articles()
    assert len(articles) == 42
    assert {a.product for a in articles} == {"auto", "home", "health"}
    assert all(a.triggers for a in articles if a.kind == "exclusion")
    assert len(load_register()) == 10


@pytest.mark.parametrize("mode", ["bm25", "tfidf", "hybrid"])
def test_theft_query_finds_theft_clause(retriever, mode):
    hits = retriever.search("Ma voiture a été volée cette nuit auto_theft", product="auto", k=3, mode=mode)
    assert hits[0].article_id == "AUTO-05"


def test_cross_lingual_query_uses_bilingual_keywords(retriever):
    # English claim against the French auto wording.
    hits = retriever.search("a stone cracked my windscreen", product="auto", k=3)
    assert "AUTO-06" in [h.article_id for h in hits]


def test_product_filter(retriever):
    hits = retriever.search("fire smoke", product="home", k=10)
    assert hits and all(h.article_id.startswith("HOME-") for h in hits)


def test_tokenize_merges_inflections():
    assert tokenize("volé")[0] == tokenize("volée")[0] == tokenize("vol")[0]


@pytest.mark.parametrize("text,phrase,expected", [
    ("The alcohol test was positive", "alcohol", True),
    ("no alcohol was involved", "alcohol", False),
    ("This was not a slow leak", "slow leak", False),
    ("acte non esthétique", "esthétique", False),
    ("Rhinoplastie à visée esthétique", "esthétique", True),
])
def test_negation_aware_matching(text, phrase, expected):
    assert find_phrase(words(text), phrase) is expected

from datetime import date

import pytest
from pydantic import ValidationError

from claims_agent.extraction import detect_language, extract, extract_amount, extract_dates
from claims_agent.schemas import ClaimExtraction


@pytest.mark.parametrize("text,expected", [
    ("le 14/03/2026 vers 18h", date(2026, 3, 14)),
    ("on 2026-05-10 a stone", date(2026, 5, 10)),
    ("On 3 March 2026 a pipe burst", date(2026, 3, 3)),
    ("le 1er février 2026", date(2026, 2, 1)),
    ("on March 12, 2026", date(2026, 3, 12)),
])
def test_dates(text, expected):
    assert extract_dates(text)[0][1] == expected


def test_invalid_date_is_ignored():
    assert extract_dates("le 31/02/2026") == []


@pytest.mark.parametrize("text,lang,expected", [
    ("Devis : 2 350 TND.", "fr", (2350.0, "TND")),
    ("Montant : 145,500 TND.", "fr", (145.5, "TND")),   # Tunisian millimes, comma decimal
    ("Hospital bill: 2,450 TND.", "en", (2450.0, "TND")),  # English thousands separator
    ("repairs: AED 7,800.", "en", (7800.0, "AED")),
    ("Le devis est de 450 DT.", "fr", (450.0, "TND")),
    ("no amount here", "en", (None, None)),
])
def test_amounts(text, lang, expected):
    assert extract_amount(text, lang) == expected


def test_language_detection():
    assert detect_language("Mon véhicule a été volé dans la nuit") == "fr"
    assert detect_language("My car was stolen in the night") == "en"


def test_full_extraction_fr():
    ex = extract("Le 21/02/2026, j'ai heurté un mur. Contrat AUTO-TN-100231. Dégâts estimés à 4 800 TND.")
    assert ex.claim_type == "auto_collision"
    assert ex.policy_number == "AUTO-TN-100231"
    assert ex.incident_date == date(2026, 2, 21)
    assert ex.amount == 4800 and ex.currency == "TND"
    assert ex.missing_fields() == []


def test_claim_type_restricted_to_policy_product():
    # "fire" exists for auto and home; the policy prefix decides which peril family is allowed.
    assert extract("Fire in the kitchen on 2026-07-08. Policy HOME-AE-200951. AED 500").claim_type == "home_fire"
    assert extract("My car caught fire on 2026-07-08. Policy AUTO-TN-100231. 500 TND").claim_type == "auto_fire"


def test_third_party_detection():
    assert extract("Un autre véhicule m'a percuté le 14/03/2026.").third_party is True
    assert extract("J'ai heurté un mur le 14/03/2026.").third_party is False


def test_missing_fields_reported():
    ex = extract("Hello, I went to the emergency room last night and paid about 600 TND.")
    assert set(ex.missing_fields()) == {"policy_number", "incident_date"}


def test_schema_rejects_unknown_peril():
    with pytest.raises(ValidationError):
        ClaimExtraction(claim_type="alien_invasion")

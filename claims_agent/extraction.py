"""Deterministic FR/EN information extraction from a free-text claim (the ``offline`` backend).

Rules are deliberately transparent: regular expressions for dates, amounts and policy numbers, and weighted
bilingual lexicons for the claim type. An LLM backend can replace this step (see ``llm.py``); its output is
validated against the same ``ClaimExtraction`` schema.
"""
from __future__ import annotations

import re
from datetime import date

from .schemas import PERILS, ClaimExtraction
from .text import find_phrase, normalize, words

POLICY_RE = re.compile(r"\b(AUTO|HOME|SANTE)-[A-Z]{2}-\d{6}\b", re.I)
PRODUCT_OF_PREFIX = {"AUTO": "auto", "HOME": "home", "SANTE": "health"}

MONTHS = {
    "janvier": 1, "january": 1, "jan": 1, "fevrier": 2, "february": 2, "feb": 2, "mars": 3, "march": 3, "mar": 3,
    "avril": 4, "april": 4, "apr": 4, "mai": 5, "may": 5, "juin": 6, "june": 6, "jun": 6, "juillet": 7, "july": 7,
    "jul": 7, "aout": 8, "august": 8, "aug": 8, "septembre": 9, "september": 9, "sep": 9, "octobre": 10,
    "october": 10, "oct": 10, "novembre": 11, "november": 11, "nov": 11, "decembre": 12, "december": 12, "dec": 12,
}
_M = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), ("y", "m", "d")),
    (re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b"), ("d", "m", "y")),
    (re.compile(rf"\b(\d{{1,2}})(?:er)? ({_M})\.? (\d{{4}})\b"), ("d", "mon", "y")),
    (re.compile(rf"\b({_M})\.? (\d{{1,2}}),? (\d{{4}})\b"), ("mon", "d", "y")),
]

CURRENCIES = {"tnd": "TND", "dt": "TND", "dinars": "TND", "dinar": "TND", "aed": "AED", "dhs": "AED",
              "eur": "EUR", "euros": "EUR", "euro": "EUR", "€": "EUR", "usd": "USD", "$": "USD"}
_C = r"(tnd|dt|dinars?|aed|dhs|eur|euros?|€|usd|\$)"
_N = r"(\d{1,3}(?:[   .,]\d{3})+(?:[.,]\d{1,3})?|\d+(?:[.,]\d{1,3})?)"
AMOUNT_AFTER = re.compile(rf"{_N}\s?{_C}(?![a-z])", re.I)
AMOUNT_BEFORE = re.compile(rf"(?<![a-z]){_C}\s?{_N}", re.I)

# Weighted lexicons (normalised, accent-free). Weight 2 = strongly indicative.
LEXICON: dict[str, dict[str, int]] = {
    "auto_collision": {"collision": 2, "collided": 2, "percute": 2, "heurte": 2, "accrochage": 2, "crashed": 2,
                       "hit my car": 2, "accident": 1, "frotte": 2, "endommage": 1, "perdu le controle": 2,
                       "rayee": 1, "aile": 1, "pare-choc": 1, "badly damaged": 1, "constat": 1},
    "auto_theft": {"vol": 2, "vole": 2, "volee": 2, "stolen": 2, "theft": 2, "plainte": 1, "autoradio": 1},
    "auto_glass": {"pare-brise": 2, "windscreen": 2, "windshield": 2, "bris de glace": 2, "vitre": 1, "cracked": 1},
    "auto_fire": {"incendie": 2, "caught fire": 2, "fire": 2, "feu": 1, "court-circuit": 1, "short circuit": 1},
    "auto_natural": {"inondation": 2, "flood": 2, "submerge": 2, "tempete": 2, "storm": 2, "grele": 2, "hail": 2,
                     "pluies": 1},
    "auto_liability": {"reclame": 2, "n'ai pas de degats": 3, "no damage to my car": 3, "voiture d'un voisin": 1},
    "home_water": {"fuite": 2, "leak": 2, "leaking": 2, "pipe burst": 2, "burst": 1, "degat des eaux": 3,
                   "water": 2, "flooded": 1, "machine a laver": 1, "washing machine": 1, "hose": 1, "plafond": 1},
    "home_fire": {"fire": 2, "incendie": 2, "smoke": 2, "fumee": 2, "ignited": 2, "brule": 1},
    "home_theft": {"cambriolage": 3, "cambrioleurs": 3, "burgled": 3, "burglary": 3, "stolen": 2, "taken": 2,
                   "vole": 2, "voles": 2, "forced": 1, "fracturee": 1, "effraction": 1},
    "home_storm": {"tempete": 2, "storm": 2, "hailstorm": 3, "hail": 2, "grele": 2, "toiture": 1, "roof": 1},
    "health_hospital": {"hospitalise": 2, "hospitalisee": 2, "hospitalisation": 2, "hospital": 2, "admitted": 2,
                        "emergency room": 2, "urgences": 2, "chirurgie": 2, "surgery": 2, "rhinoplastie": 2,
                        "intervention": 1, "appendicite": 1},
    "health_outpatient": {"consultation": 2, "medecin": 2, "doctor": 2, "pharmacie": 2, "medicaments": 2,
                          "analyses": 2, "lab tests": 2, "radiographie": 2, "treated": 1},
    "health_dental": {"dentaire": 2, "dentaires": 2, "dental": 2, "dentiste": 2, "carie": 2, "caries": 2,
                      "detartrage": 2, "root canal": 2, "fillings": 2},
    "health_optical": {"lunettes": 2, "verres": 2, "glasses": 2, "lenses": 2, "optique": 2, "optical": 2,
                       "ophtalmologue": 1},
    "health_maternity": {"accouchement": 3, "maternite": 2, "naissance": 2, "childbirth": 3, "gave birth": 3},
}
THIRD_PARTY = ["tiers", "autre vehicule", "un camion", "voisin", "autre conducteur", "another driver", "other driver",
               "third party", "another car", "van owner", "a lorry", "a truck", "upstairs flat", "neighbour"]
_FR_HINTS = {"le", "la", "les", "de", "du", "des", "et", "mon", "ma", "est", "une", "pour", "contrat", "j", "a"}
_EN_HINTS = {"the", "and", "my", "was", "of", "on", "in", "to", "is", "policy", "i", "a", "with"}


def detect_language(text: str) -> str:
    toks = words(text)
    fr = sum(t in _FR_HINTS for t in toks)
    en = sum(t in _EN_HINTS for t in toks)
    return "fr" if fr > en else "en"


def extract_policy_number(text: str) -> str | None:
    m = POLICY_RE.search(text)
    return m.group(0).upper() if m else None


def extract_dates(text: str) -> list[tuple[int, date]]:
    """All valid dates with their character offset, in order of appearance."""
    norm = normalize(text)
    found = []
    for pattern, order in DATE_PATTERNS:
        for m in pattern.finditer(norm):
            parts = dict(zip(order, m.groups()))
            month = MONTHS[parts["mon"]] if "mon" in parts else int(parts["m"])
            try:
                found.append((m.start(), date(int(parts["y"]), month, int(parts["d"]))))
            except ValueError:
                continue
    return sorted(found)


def _parse_number(raw: str, language: str) -> float:
    raw = raw.replace(" ", " ").replace(" ", " ")
    if language == "fr":  # FR: space/dot thousands, comma decimal ("145,500 TND" = 145.5 TND)
        raw = raw.replace(" ", "").replace(".", "").replace(",", ".")
    else:  # EN: comma thousands, dot decimal
        raw = raw.replace(" ", "").replace(",", "")
    return float(raw)


def extract_amount(text: str, language: str) -> tuple[float | None, str | None]:
    candidates = []
    for m in AMOUNT_AFTER.finditer(text):
        candidates.append((m.start(), m.group(1), m.group(2)))
    for m in AMOUNT_BEFORE.finditer(text):
        candidates.append((m.start(), m.group(2), m.group(1)))
    if not candidates:
        return None, None
    parsed = [(_parse_number(n, language), CURRENCIES[c.lower()]) for _, n, c in sorted(candidates)]
    return max(parsed)  # the claimed amount is the largest figure with a currency


def classify_claim_type(text: str, product: str | None) -> tuple[str | None, int]:
    toks = words(text)
    allowed = PERILS[product] if product else tuple(LEXICON)
    scores = {p: sum(w for kw, w in LEXICON[p].items() if find_phrase(toks, kw, negation_window=0)) for p in allowed}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0:
        return None, 0
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
    return ranked[0][0], margin


def extract(text: str) -> ClaimExtraction:
    language = detect_language(text)
    policy_number = extract_policy_number(text)
    product = PRODUCT_OF_PREFIX[policy_number.split("-")[0]] if policy_number else None
    claim_type, margin = classify_claim_type(text, product)
    dates = extract_dates(text)
    amount, currency = extract_amount(text, language)
    toks = words(text)
    return ClaimExtraction(
        language=language,
        claim_type=claim_type,
        claim_type_margin=margin,
        incident_date=dates[0][1] if dates else None,
        amount=amount,
        currency=currency,
        policy_number=policy_number,
        third_party=any(find_phrase(toks, p, negation_window=0) for p in THIRD_PARTY),
    )

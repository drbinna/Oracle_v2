"""
MD&A attribution populator.

Pulls the 10-K's narrative and extracts the gross-margin drivers MANAGEMENT
itself states (the "due primarily to ..." clauses in Item 7). These become the
`any_of` ground truth for the attribution criterion — so attribution is grounded
in the primary source, not an LLM's opinion.

Best-effort and graceful: if the filing can't be fetched/parsed, returns [] and
the rubric's attribution criterion simply stays unpopulated.
"""
from __future__ import annotations
import html as _html
import re
import requests

from quant_firm.data.edgar import HEADERS, company_facts  # reuse headers/CIK

# canonical driver -> detection patterns (substrings, lowercased)
LEXICON: dict[str, list[str]] = {
    "mix": ["mix"],
    "foreign currency": ["foreign currenc", "foreign exchange", " fx ", "currency"],
    "leverage": ["leverage"],
    "product cost": ["product cost", "higher cost", "input cost",
                     "component cost", "commodity", "material cost"],
    "pricing": ["pricing", "price increase", "price action"],
    "net sales / volume": ["net sales", "higher sales", "volume"],
    "freight / logistics": ["freight", "logistics", "shipping cost"],
    "wages / labor": ["wage", "labor cost"],
    "tariffs": ["tariff"],
    "promotions / discounts": ["promotion", "discount", "markdown"],
}

_CAUSAL = ("due primarily to", "due to", "driven by", "primarily due",
           "partially offset", "reflecting", "attributable to", "as a result of")
_CHANGE = ("increased", "decreased", "compared to", "percentage", "improved", "declined")


def _doc_url(cik: str, accession: str) -> str | None:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    rec = r.json()["filings"]["recent"]
    try:
        i = rec["accessionNumber"].index(accession)
    except ValueError:
        return None
    doc = rec["primaryDocument"][i]
    nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{doc}"


def _plain_text(url: str) -> str:
    html = requests.get(url, headers=HEADERS, timeout=30).text
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", _html.unescape(txt))


def _attribution_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.])\s+", text)
    out = []
    for s in sents:
        low = s.lower()
        if "gross margin" in low and any(c in low for c in _CAUSAL) \
                and any(c in low for c in _CHANGE):
            out.append(s)
    return out


def _drivers_from(sentences: list[str]) -> list[str]:
    blob = " ".join(sentences).lower()
    found = [canon for canon, pats in LEXICON.items()
             if any(p in blob for p in pats)]
    return found


def margin_drivers(cik: str, accession: str) -> tuple[list[str], list[str]]:
    """
    Returns (driver_terms, evidence_sentences).
    driver_terms feed the rubric's attribution `any_of`.
    """
    try:
        url = _doc_url(cik, accession)
        if not url:
            return [], []
        text = _plain_text(url)
        sents = _attribution_sentences(text)
        return _drivers_from(sents), sents[:4]
    except Exception:
        return [], []

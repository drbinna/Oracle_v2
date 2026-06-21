"""
EDGAR / XBRL data layer.

Ground truth for the rubric comes from here and ONLY here — authoritative,
machine-readable SEC data. Exa/SixtyFour are the agent's discovery tools and
never touch grading.

SEC requires a descriptive User-Agent with contact info. Replace the email
before you run this for real, or SEC may rate-limit you.
"""
from __future__ import annotations
import functools
import requests

HEADERS = {"User-Agent": "HUD Hackathon quant-firm (you@yourteam.com)"}
SEC = "https://data.sec.gov"


@functools.lru_cache(maxsize=1)
def _ticker_map() -> dict[str, str]:
    """ticker (upper) -> zero-padded 10-digit CIK."""
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


def ticker_to_cik(ticker: str) -> str:
    return _ticker_map()[ticker.upper()]


@functools.lru_cache(maxsize=64)
def company_facts(cik: str) -> dict:
    r = requests.get(f"{SEC}/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _annual(facts: dict, tag: str, year: int):
    """Return the deduped annual fact for calendar/fiscal `year` (frame CY{year})."""
    try:
        units = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return None
    for u in units:
        if u.get("frame") == f"CY{year}":
            return u
    return None


def _first_tag(facts: dict, tags: list[str], year: int):
    """Try tag fallbacks (companies tag the same concept differently)."""
    for t in tags:
        f = _annual(facts, t, year)
        if f is not None:
            return t, f
    return None, None


REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
COGS_TAGS = ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"]


def get_10k_accession(cik: str, fiscal_year: int) -> str | None:
    """Accession of the ORIGINAL 10-K for `fiscal_year` — the citation ground truth."""
    r = requests.get(f"{SEC}/submissions/CIK{cik}.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    rdates = recent.get("reportDate", [])
    accns = recent.get("accessionNumber", [])
    for form, rdate, accn in zip(forms, rdates, accns):
        # report period ends in the fiscal year (handles Sept/Dec year-ends)
        if form == "10-K" and rdate.startswith(str(fiscal_year)):
            return accn
    return None


def gross_margin(ticker: str, year: int) -> dict:
    """
    Verified ground truth for a gross-margin task.
    Returns the numbers, the per-share-free margin %, and the citation accession.
    Raises if the company doesn't tag the needed concepts for that year.
    """
    cik = ticker_to_cik(ticker)
    facts = company_facts(cik)

    rev_tag, rev = _first_tag(facts, REVENUE_TAGS, year)
    cogs_tag, cogs = _first_tag(facts, COGS_TAGS, year)
    gp = _annual(facts, "GrossProfit", year)
    if rev is None or (cogs is None and gp is None):
        raise ValueError(f"{ticker} FY{year}: required XBRL tags not found")

    revenue = rev["val"]
    if gp is not None:
        gross_profit = gp["val"]
        cogs_val = revenue - gross_profit if cogs is None else cogs["val"]
    else:
        cogs_val = cogs["val"]
        gross_profit = revenue - cogs_val

    margin_pct = gross_profit / revenue * 100.0
    accession = get_10k_accession(cik, year) or rev.get("accn")

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "fiscal_year": year,
        "revenue_usd": revenue,
        "cogs_usd": cogs_val,
        "gross_profit_usd": gross_profit,
        "gross_margin_pct": round(margin_pct, 4),
        "accession": accession,
        "period_end": rev.get("end"),
        "source_tags": {"revenue": rev_tag, "cogs": cogs_tag},
    }

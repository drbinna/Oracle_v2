"""
Rubric generator.

Turns verified EDGAR ground truth into a HUD-style WEIGHTED rubric — the format
from Jay's finance-RL slide. Calculation accuracy + citation are fully verified
from the primary source. Attribution/strategy are scaffolded as keyword criteria
(fill `any_of` from the filing's MD&A — management's own stated drivers).

Discipline: an LLM may DRAFT the attribution/strategy criteria, but the numbers
and the citation accession come straight from EDGAR. The primary source is truth.
"""
from __future__ import annotations
from quant_firm.data import edgar
from quant_firm.data import mdna


def build_margin_rubric(ticker: str, year: int, difficulty: int = 1) -> dict:
    gt = edgar.gross_margin(ticker, year)

    # attribution ground truth straight from the 10-K MD&A (management's words)
    drivers, evidence = mdna.margin_drivers(gt["cik"], gt["accession"])

    # tolerance tightens with difficulty -> a lever for the 20-50% reward band
    tol = {1: 0.25, 2: 0.10, 3: 0.05}.get(difficulty, 0.10)  # percentage points

    prompt = (
        f"Using {gt['ticker']}'s FY{year} 10-K, report the company's gross "
        f"margin for fiscal year {year}.\n\n"
        "Return ONLY JSON with this shape:\n"
        '{ "gross_margin_pct": <number>, '
        '"revenue_usd": <number>, '
        '"citations": ["<SEC accession number>"], '
        '"drivers": ["<primary driver of the margin, per the MD&A>"] }'
    )

    criteria = [
        {  # CALCULATION ACCURACY — verified from XBRL
            "id": "calc_margin",
            "section": "Calculation Accuracy",
            "weight": 0.45,
            "type": "numeric",
            "field": "gross_margin_pct",
            "target": gt["gross_margin_pct"],
            "tol": tol,
            "unit": "pp",
        },
        {  # secondary numeric anchor (revenue, in $)
            "id": "calc_revenue",
            "section": "Calculation Accuracy",
            "weight": 0.15,
            "type": "numeric",
            "field": "revenue_usd",
            "target": gt["revenue_usd"],
            "tol": gt["revenue_usd"] * 0.005,  # 0.5%
            "unit": "usd",
        },
        {  # SOURCE CITATIONS — verified accession
            "id": "cite_10k",
            "section": "Source Citations",
            "weight": 0.20,
            "type": "citation",
            "field": "citations",
            "accession": gt["accession"],
        },
        {  # ATTRIBUTION — populated from the 10-K MD&A
            "id": "attr_driver",
            "section": "Attribution Analysis",
            "weight": 0.20,
            "type": "keywords",
            "field": "drivers",
            "any_of": drivers,        # e.g. ["mix", "foreign currency", "leverage"]
            "evidence": evidence,     # the MD&A sentences these came from
        },
    ]

    return {
        "task_id": f"{gt['ticker']}-grossmargin-FY{year}-d{difficulty}",
        "ticker": gt["ticker"],
        "fiscal_year": year,
        "difficulty": difficulty,
        "prompt": prompt,
        "criteria": criteria,
        "ground_truth": gt,   # kept env-side; never shown to the agent
    }

"""
Reward redesign for CONTROLLABLE variance.

The law (see WRITEUP.md): a verifiable reward is trainable only if its variance is
policy-controllable, not exogenous — I(A_policy ; R | X_exogenous) > 0. The Exa env
passed the variance gate but trained flat because the spread was exogenous (whether
Exa surfaced the income-statement line at all).

Fix: hold X_exogenous ~constant by ALWAYS providing the complete data, but provide it
the way it really appears — a distractor-laden statement (quarterly vs annual columns,
prior year, multiple line items, millions units). Then grade the figures the agent
SELECTS and COMPUTES against EDGAR truth. The data is always available, so the residual
reward variance is the agent's parse+arithmetic skill = controllable.

Empirical basis: handed CLEAN numbers, Qwen3-8B saturates (0.8-1.0); handed the SAME
numbers buried among distractors (as in real Exa highlights: "90,146 83,360 394,328
365,817"), it errs on which column is FY net sales. That disambiguation is controllable
and non-saturated — exactly the I(A;R|X)>0 we need.

Ground truth stays EDGAR/XBRL only (quant_firm/data/edgar.py via the rubric).
"""
from __future__ import annotations
import json
import re

from quant_firm.data import edgar


# --------------------------------------------------------------------------- #
# Input provision: a deterministic, complete, distractor-laden statement.
# Always contains the correct FY figures, so "couldn't find it" (exogenous) is
# removed; picking the right cell + computing (controllable) is what's left.
# --------------------------------------------------------------------------- #
def build_noisy_statement(gt: dict, hard: bool = False) -> str:
    y = gt["fiscal_year"]
    rev = round(gt["revenue_usd"] / 1e6)
    cogs = round(gt["cogs_usd"] / 1e6)
    gp = rev - cogs
    # plausible DISTRACTORS with DISTINCT margins (different cogs ratios) so picking
    # the wrong column yields a wrong margin too -> selection discriminates cleanly.
    prior_rev, prior_cogs = round(rev * 0.91), round(cogs * 0.96)
    q4_rev, q4_cogs = round(rev * 0.27), round(cogs * 0.31)
    pq4_rev, pq4_cogs = round(prior_rev * 0.26), round(prior_cogs * 0.29)
    opex = round(gp * 0.34)
    head = (
        f"CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)\n"
        f"                          Three Months Ended      Twelve Months Ended\n"
        f"                          FY{y}      FY{y-1}       FY{y}        FY{y-1}\n"
    )
    if not hard:
        return (
            head +
            f"Net sales                 {q4_rev:>8,} {pq4_rev:>10,}    {rev:>9,}   {prior_rev:>10,}\n"
            f"Cost of sales             {q4_cogs:>8,} {pq4_cogs:>10,}    {cogs:>9,}   {prior_cogs:>10,}\n"
            f"Gross profit              {round(q4_rev-q4_cogs):>8,} {round(pq4_rev-pq4_cogs):>10,}    "
            f"{gp:>9,}   {round(prior_rev-prior_cogs):>10,}\n"
            f"Operating expenses        {round(opex*0.27):>8,} {round(opex*0.26):>10,}    {opex:>9,}   {round(opex*0.92):>10,}\n"
            f"(SEC accession {gt['accession']})\n"
        )
    # HARD: segmented revenue/cost with NO total line -> agent must AGGREGATE the
    # Products + Services rows (controllable multi-step arithmetic), then pick the
    # right column, then compute margin. No "Net sales total" is given.
    p_rev, s_rev = round(rev * 0.79), rev - round(rev * 0.79)
    p_cogs, s_cogs = round(cogs * 0.84), cogs - round(cogs * 0.84)
    q_prev, q_srev = round(q4_rev * 0.79), q4_rev - round(q4_rev * 0.79)
    pr_prev, pr_srev = round(prior_rev * 0.79), prior_rev - round(prior_rev * 0.79)
    q_pcogs, q_scogs = round(q4_cogs * 0.84), q4_cogs - round(q4_cogs * 0.84)
    pr_pcogs, pr_scogs = round(prior_cogs * 0.84), prior_cogs - round(prior_cogs * 0.84)
    return (
        head +
        f"Net sales:\n"
        f"  Products                {q_prev:>8,} {pr_prev:>10,}    {p_rev:>9,}   {pr_prev:>10,}\n"
        f"  Services                {q_srev:>8,} {pr_srev:>10,}    {s_rev:>9,}   {pr_srev:>10,}\n"
        f"Cost of sales:\n"
        f"  Products                {q_pcogs:>8,} {pr_pcogs:>10,}    {p_cogs:>9,}   {pr_pcogs:>10,}\n"
        f"  Services                {q_scogs:>8,} {pr_scogs:>10,}    {s_cogs:>9,}   {pr_scogs:>10,}\n"
        f"Operating expenses        {round(opex*0.27):>8,} {round(opex*0.26):>10,}    {opex:>9,}   {round(opex*0.92):>10,}\n"
        f"(No subtotal line is provided; sum the segments. SEC accession {gt['accession']})\n"
    )


# --------------------------------------------------------------------------- #
# Rubric: weight is on the controllable computation. No citation/attribution
# (those carried exogenous/keyword noise); the accession is in the statement so
# a light citation check stays controllable.
# --------------------------------------------------------------------------- #
def build_compute_rubric(ticker: str, year: int, difficulty: int = 1) -> dict:
    gt = edgar.gross_margin(ticker, year)
    tol = {1: 0.25, 2: 0.10, 3: 0.05}.get(difficulty, 0.10)
    prompt = (
        f"You called pull_income_statement for {gt['ticker']} FY{year}. Net sales and "
        f"Cost of sales are split into Products and Services with NO total line — sum the "
        f"segments. Use the FULL-YEAR (Twelve Months Ended FY{year}) column, not the "
        f"quarterly or prior-year ones. Then compute the gross margin.\n\n"
        "Return ONLY JSON:\n"
        '{ "revenue_usd": <total net sales>, "cogs_usd": <total cost of sales>, '
        '"gross_margin_pct": <number>, "citations": ["<accession>"] }\n'
        "Report revenue_usd and cogs_usd as they appear (millions is fine)."
    )
    return {
        "task_id": f"{gt['ticker']}-compute-FY{year}-d{difficulty}",
        "ticker": gt["ticker"], "fiscal_year": year, "difficulty": difficulty,
        "prompt": prompt,
        "tol_pp": tol,
        "ground_truth": gt,
    }


# --------------------------------------------------------------------------- #
# Grader: scale-robust numeric match (so unit choice — millions vs dollars — is
# NOT a uniform gotcha; the spread comes from SELECTION + arithmetic, which is
# controllable). All checks vs EDGAR truth -> grounded, not hackable.
# --------------------------------------------------------------------------- #
def _coerce(answer):
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return {}
    s = re.sub(r"^```(json)?|```$", "", answer.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    try:
        return json.loads(m.group(0) if m else s)
    except Exception:
        return {}


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", "").strip())
    except Exception:
        return None


def _scale_match(reported, gold, reltol) -> bool:
    """True if reported equals gold at any sensible unit scale (dollars/millions/...)."""
    v = _num(reported)
    if v is None or v == 0:
        return False
    for s in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
        if abs(v * s - gold) <= reltol * abs(gold):
            return True
    return False


WEIGHTS = {"revenue": 0.25, "cogs": 0.25, "margin": 0.45, "citation": 0.05}


def grade_compute(answer, rubric: dict) -> dict:
    ans = _coerce(answer)
    gt = rubric["ground_truth"]
    b = {}
    b["revenue"] = 1.0 if _scale_match(ans.get("revenue_usd"), gt["revenue_usd"], 0.005) else 0.0
    b["cogs"] = 1.0 if _scale_match(ans.get("cogs_usd"), gt["cogs_usd"], 0.005) else 0.0
    m = _num(ans.get("gross_margin_pct"))
    b["margin"] = 1.0 if (m is not None and abs(m - gt["gross_margin_pct"]) <= rubric["tol_pp"]) else 0.0
    cites = ans.get("citations") or []
    if isinstance(cites, str):
        cites = [cites]
    want = re.sub(r"[^0-9]", "", str(gt["accession"]))
    b["citation"] = 1.0 if any(re.sub(r"[^0-9]", "", str(c)) == want for c in cites) else 0.0
    reward = sum(WEIGHTS[k] * b[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    return {"reward": round(reward, 4), "breakdown": b}

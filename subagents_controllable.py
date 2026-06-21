"""
Controllable-variance specialist: provides the (distractor-laden) income statement.

Unlike the Exa discovery tools, this ALWAYS returns the complete statement for the
fiscal year — so retrieval is no longer the exogenous bottleneck. The agent's job is
the controllable part: pick the Twelve-Months-Ended FY columns out of the quarterly /
prior-year distractors and compute the margin. Built from EDGAR (the filing's real
numbers); the margin itself is never handed over.

    python -m quant_firm.subagents_controllable    # serves http://127.0.0.1:8082/mcp
"""
from __future__ import annotations
from fastmcp import FastMCP

from quant_firm.data import edgar
from quant_firm import controllable

tools = FastMCP(name="quant-statements")


@tools.tool
def pull_income_statement(ticker: str, year: int) -> str:
    """Pull the company's consolidated statement of operations (in millions) for the
    fiscal year, exactly as it appears in the 10-K: quarterly and annual columns, the
    current and prior year, multiple line items. Identify the right figures yourself."""
    gt = edgar.gross_margin(ticker, year)
    return controllable.build_noisy_statement(gt, hard=True)


if __name__ == "__main__":
    tools.run(transport="http", host="127.0.0.1", port=8082)

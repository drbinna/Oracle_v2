"""
The firm's specialists — Exa-backed DISCOVERY tools (Option B: aggressive/realistic).

The oracle specialists (pull_financials / audit_gross_margin / read_mdna_drivers)
are gone — they spoon-fed exact XBRL numbers + the accession, which saturated the
agent and produced no GRPO within-group spread. The original is preserved in
subagents_oracle.py.bak.

Now the orchestrator must do the real work: SEARCH for the company's 10-K, READ
the filing, and EXTRACT revenue / cost of sales / margin drivers from the prose
itself, then derive the citation (accession) from the SEC URL. Exa is strictly an
agent-side discovery tool — it NEVER touches grading. Ground truth stays EDGAR/XBRL
(quant_firm/data/edgar.py), used only by the grader, never imported here.

Run the server (so the orchestrator's mcp capability has a live URL):
    python -m quant_firm.subagents          # serves http://127.0.0.1:8080/mcp
"""
from __future__ import annotations
import os
import pathlib

from fastmcp import FastMCP


def _load_exa_key() -> str:
    """EXA_API_KEY from env, falling back to the project .env (the serving process
    is started fresh and may not inherit the launch-time shell env)."""
    key = os.environ.get("EXA_API_KEY")
    if key:
        return key
    for envpath in (pathlib.Path(__file__).resolve().parent.parent / ".env",
                    pathlib.Path.cwd() / ".env"):
        try:
            for line in envpath.read_text().splitlines():
                line = line.strip()
                if line.startswith("EXA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "EXA_API_KEY not set and not found in .env — create /Users/drbinna/Downloads/.env "
        "with a line `EXA_API_KEY=...` (Exa is the agent's discovery tool).")


from exa_py import Exa
_exa = Exa(api_key=_load_exa_key())

tools = FastMCP(name="quant-specialists")


@tools.tool
def web_search(query: str, num_results: int = 6) -> dict:
    """Discovery analyst: web search for primary sources. Use this to FIND a
    company's SEC 10-K filing (e.g. "Apple 10-K fiscal 2022 SEC filing annual
    report net sales"). Returns candidate pages with title, url, and query-relevant
    highlight excerpts. The SEC filing URL contains the accession number you must
    cite. Numbers are NOT provided — read the filing and extract them yourself."""
    try:
        res = _exa.search(query, type="auto", num_results=num_results,
                          contents={"highlights": True})
    except Exception as e:
        return {"error": f"exa search failed: {e!r}"[:200], "results": []}
    out = []
    for r in res.results:
        out.append({
            "title": getattr(r, "title", None),
            "url": getattr(r, "url", None),
            "highlights": getattr(r, "highlights", None) or [],
        })
    return {"query": query, "results": out}


@tools.tool
def read_filing(url: str,
                query: str = "net sales revenue cost of sales gross margin drivers") -> dict:
    """Discovery analyst: read a filing/page you found and pull query-relevant
    passages from it. Pass the SEC 10-K url and a query naming the line items you
    need (e.g. "net sales and cost of sales in millions, fiscal 2022, and the
    management-stated drivers of gross margin"). Returns highlight excerpts plus a
    capped text slice so you can extract the figures and the MD&A drivers yourself.
    Does not compute the margin for you."""
    try:
        res = _exa.get_contents(
            [url],
            highlights=True,
            text={"max_characters": 8000},
        )
    except Exception as e:
        return {"error": f"exa get_contents failed: {e!r}"[:200], "url": url}
    if not res.results:
        return {"error": "no content returned", "url": url}
    r = res.results[0]
    return {
        "url": getattr(r, "url", url),
        "title": getattr(r, "title", None),
        "highlights": getattr(r, "highlights", None) or [],
        "text_excerpt": (getattr(r, "text", None) or "")[:6000],
    }


if __name__ == "__main__":
    tools.run(transport="http", host="127.0.0.1", port=8080)

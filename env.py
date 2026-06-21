"""
HUD environment — the autonomous quant firm (v6 API).

The orchestrator agent gets two capabilities:
  - a shell (env.workspace) to run its own Python and compute margins
  - an `mcp` capability exposing the firm's specialists (subagents.py):
    pull_financials, read_mdna_drivers, audit_gross_margin

Two-yield template: yield #1 = prompt, yield #2 = the verifiable rubric reward.
The grader is our programmatic rubric (no LLM judge).

Run it:
    uv tool install hud-python --python 3.12
    export HUD_API_KEY=...
    python -m quant_firm.subagents &          # serve specialists on :8080
    hud dev quant_firm/env.py                  # local hot-reload
    hud eval quant_firm/env.py claude          # local eval
"""
from __future__ import annotations
from hud import Environment
from hud.capabilities import Capability

from quant_firm.rubric import generate, graders

env = Environment(
    name="autonomous-quant-firm",
    capabilities=[
        # the firm's specialists, served by subagents.py
        Capability.mcp(name="specialists", url="http://127.0.0.1:8080/mcp"),
    ],
)
# shell capability: the agent runs its own Python to compute, no custom calculator
env.workspace("/Users/drbinna/Downloads/qf_workspace")


@env.template(id="analyze_filing",
              description="Analyze a company's gross margin from its 10-K.")
async def analyze_filing(ticker: str = "AAPL", year: int = 2022, difficulty: int = 1):
    rubric = generate.build_margin_rubric(ticker, year, difficulty)
    answer = yield rubric["prompt"]                 # PROMPT
    yield graders.grade(answer, rubric)["reward"]   # verifiable REWARD


# difficulty x ticker spread -> a trainable taskset (tune into the 20-50% band)
TICKERS = ["AAPL", "MSFT", "NVDA", "KO", "WMT"]
tasks = [analyze_filing(ticker=t, year=2022, difficulty=d)
         for t in TICKERS for d in (1, 2, 3)]
for v in tasks:
    v.slug = f"{v.args['ticker']}-gm-d{v.args['difficulty']}"

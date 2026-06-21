"""
HUD environment — autonomous quant firm (controllable-variance, cloud-deployable).

Verifiable rubric, EDGAR ground truth. Reward weight is on the controllable COMPUTATION:
the agent calls pull_income_statement (data always complete) and must select the FY
figures from distractors and compute the margin.

Cloud-correct: the MCP specialist is served IN-PROCESS from @env.initialize (bound to
127.0.0.1), so it deploys with the env instead of needing an external server.

    python -m quant_firm.subagents_controllable &   # local: external server (reused)
    hud eval quant_firm/env_controllable.py claude
    hud deploy quant_firm/                           # build + register on the platform
"""
from __future__ import annotations
import asyncio
import socket

from hud import Environment
from hud.capabilities import Capability

from quant_firm import controllable
from quant_firm.subagents_controllable import tools as _stmt_server

env = Environment(name="autonomous-quant-firm")
env.workspace("workspace")          # relative path -> works in a built image

_started: dict = {"task": None}


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


@env.initialize
async def _up():
    # serve the statement specialist in-process unless an external one is already up
    if _started["task"] is None and not _port_open("127.0.0.1", 8082):
        _started["task"] = asyncio.create_task(
            _stmt_server.run_async(transport="http", host="127.0.0.1", port=8082)
        )
        await asyncio.sleep(1.5)
    env.add_capability(Capability.mcp(name="statements",
                                      url="http://127.0.0.1:8082/mcp"))


@env.shutdown
async def _down():
    if _started["task"] is not None:
        _started["task"].cancel()
        _started["task"] = None


@env.template(id="analyze_compute",
              description="Compute gross margin from a distractor-laden income statement.")
async def analyze_compute(ticker: str = "AAPL", year: int = 2022, difficulty: int = 1):
    rubric = controllable.build_compute_rubric(ticker, year, difficulty)
    answer = yield rubric["prompt"]
    yield controllable.grade_compute(answer, rubric)["reward"]


TICKERS = ["AAPL", "MSFT", "NVDA", "KO", "HD"]


def _mk(ticker: str, difficulty: int):
    v = analyze_compute(ticker=ticker, year=2022, difficulty=difficulty)
    v.slug = f"{ticker}-compute-d{difficulty}"
    return v


tasks = [_mk(t, d) for t in TICKERS for d in (1, 2)]

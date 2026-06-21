"""
Step 1 (agentic) — collect traces from the DEPLOYED TOOL ENV.

Instead of the with-context proxy, this runs the real HUD rollout: the agent
calls the env's tools itself (shell + the specialists from subagents.py) to find
and extract the data, and the env grades the result with the rubric. The traces
are real agentic successes/failures — the right training signal.

`group` runs each task several times (GRPO group / variance), and writes every
run's reward + final answer in the SAME schema as collect.py, so build_sft.py
consumes it unchanged.

PREREQUISITES (this needs the served env — can't run from a bare shell):
    export HUD_API_KEY=...
    python -m quant_firm.subagents &          # specialists on :8080
    # then either let the in-process env serve locally (Docker) or `hud dev` it

    python -m quant_firm.train.collect_env --model claude-sonnet-4-5 \
        --tickers AAPL MSFT NVDA KO --difficulties 1 2 --group 6 --out traces.jsonl
    python -m quant_firm.train.collect_env --dry-run     # wire-check, no rollout
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys

from hud import Taskset
from hud.agents import create_agent

from quant_firm.env import analyze_filing   # the @env.template


def build_taskset(tickers, difficulties, year):
    tasks = []
    for t in tickers:
        for d in difficulties:
            v = analyze_filing(ticker=t, year=year, difficulty=d)
            v.slug = f"{t}-gm-d{d}"
            tasks.append(v)
    return Taskset("quant-firm-gm", tasks)


def extract(run):
    """reward + final answer from a completed run (trace.content is the graded answer)."""
    completion = ""
    try:
        completion = run.trace.content or ""
    except Exception:
        completion = ""
    return {
        "reward": getattr(run, "reward", 0.0),
        "prompt": getattr(run, "prompt_text", ""),
        "completion": completion,
    }


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)

    taskset = build_taskset(args.tickers, args.difficulties, args.year)
    agent = create_agent(args.model)
    n_runs = len(taskset.tasks) * args.group
    print(f"agentic collect: {len(taskset.tasks)} tasks x group {args.group} "
          f"= {n_runs} rollouts  | agent={args.model}")

    if args.dry_run:
        print("  [dry-run] taskset + agent built OK; slugs:", list(taskset.tasks))
        print("  [dry-run] not running — start subagents.py + serve the env, then drop --dry-run")
        return

    job = await taskset.run(agent, group=args.group, max_concurrent=args.concurrency)

    rows = [extract(r) for r in job.runs]
    rows = [r for r in rows if r["completion"]]  # keep runs that produced an answer
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    rewards = [r["reward"] for r in rows]
    hi = sum(1 for x in rewards if x >= args.keep_threshold)
    print(f"wrote {len(rows)} traces -> {args.out}")
    if rewards:
        print(f"  reward: mean={sum(rewards)/len(rewards):.3f} "
              f"max={max(rewards):.2f} | {hi} >= {args.keep_threshold} (SFT-eligible)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "KO"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--keep-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="traces.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

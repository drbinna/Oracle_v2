"""
Phase 1 — agentic variance gate (GO/NO-GO, NO GPU).

Runs the REAL tool-augmented rollout (agent retrieves via the :8080 specialists +
shell workspace; NO filing_context handed to it) and reports the per-task
within-group reward distribution — the GRPO advantage signal.

GO  if a healthy share of tasks show spread (some rollouts succeed, some fail).
NO-GO if tasks floor ~0 (can't retrieve) or saturate ~1 (too easy).

    python -m quant_firm.train.variance_env --model Qwen/Qwen3-8B \
        --tickers AAPL MSFT NVDA KO HD --difficulties 1 2 --group 8 --max-tokens 4096
"""
from __future__ import annotations
import argparse
import asyncio
import os
import statistics
import sys

from hud import Taskset
from hud.agents import create_agent

from quant_firm.env import analyze_filing


def build_taskset(tickers, difficulties, year):
    tasks = []
    for t in tickers:
        for d in difficulties:
            v = analyze_filing(ticker=t, year=year, difficulty=d)
            v.slug = f"{t}-gm-d{d}"
            tasks.append(v)
    return Taskset("quant-firm-gm-variance", tasks)


def classify(rewards):
    if len(rewards) < 2:
        return "no-data", 0.0, 0.0
    mean = statistics.mean(rewards)
    sd = statistics.pstdev(rewards)
    if sd < 0.05:
        kind = ("degenerate-high" if mean > 0.8
                else "degenerate-low" if mean < 0.2 else "degenerate-mid")
    else:
        kind = "SIGNAL"
    return kind, round(mean, 3), round(sd, 3)


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)

    taskset = build_taskset(args.tickers, args.difficulties, args.year)
    agent = create_agent(args.model,
                         completion_kwargs={"max_tokens": args.max_tokens})
    n = len(taskset.tasks) * args.group
    print(f"agentic variance: {len(taskset.tasks)} tasks x group {args.group} "
          f"= {n} rollouts | agent={args.model} max_tokens={args.max_tokens}\n")

    job = await taskset.run(agent, group=args.group, max_concurrent=args.concurrency)

    # bucket rewards by task slug (run.slug maps each rollout to its task)
    buckets: dict[str, list[float]] = {}
    for r in job.runs:
        slug = getattr(r, "slug", None) or "?"
        buckets.setdefault(slug, []).append(getattr(r, "reward", 0.0) or 0.0)

    signal = 0
    slugs = list(taskset.tasks.keys())
    for slug in slugs:
        rewards = buckets.get(slug, [])
        kind, mean, sd = classify(rewards)
        if kind == "SIGNAL":
            signal += 1
        dist = " ".join(f"{x:.2f}" for x in sorted(rewards))
        print(f"  {slug:<16} {kind:<16} mean={mean:<5} sd={sd:<5}  [{dist}]")
    frac = signal / len(slugs) if slugs else 0
    print(f"\n  -> {signal}/{len(slugs)} tasks have GRPO signal ({frac:.0%})")
    print(f"  GATE: {'GO' if frac >= 0.4 else 'NO-GO'} "
          f"(healthy spread share = {frac:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "KO", "HD"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

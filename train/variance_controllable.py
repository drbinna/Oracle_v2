"""
Validate CONTROLLABLE variance on the redesigned reward.

Because the income statement is always complete (X_exogenous held ~constant), any
within-group spread here is policy-controllable (parse + arithmetic), not retrieval
luck. GO = spread exists (trainable). Saturated = computation already mastered even with
distractors (honest: need a harder controllable task, not this one).

    python -m quant_firm.train.variance_controllable --model Qwen/Qwen3-8B \
        --tickers AAPL MSFT NVDA KO HD --difficulties 1 2 --group 8
"""
from __future__ import annotations
import argparse
import asyncio
import os
import statistics
import sys

from hud import Taskset
from hud.agents import create_agent

from quant_firm.env_controllable import analyze_compute


def build_taskset(tickers, difficulties, year):
    tasks = []
    for t in tickers:
        for d in difficulties:
            v = analyze_compute(ticker=t, year=year, difficulty=d)
            v.slug = f"{t}-compute-d{d}"
            tasks.append(v)
    return Taskset("quant-firm-compute-variance", tasks)


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
    agent = create_agent(args.model, completion_kwargs={"max_tokens": args.max_tokens})
    n = len(taskset.tasks) * args.group
    print(f"controllable variance: {len(taskset.tasks)} tasks x group {args.group} "
          f"= {n} rollouts | agent={args.model}\n")

    job = await taskset.run(agent, group=args.group, max_concurrent=args.concurrency)
    buckets: dict[str, list[float]] = {}
    for r in job.runs:
        buckets.setdefault(getattr(r, "slug", "?"), []).append(getattr(r, "reward", 0.0) or 0.0)

    signal = 0
    slugs = list(taskset.tasks.keys())
    for slug in slugs:
        rewards = buckets.get(slug, [])
        kind, mean, sd = classify(rewards)
        if kind == "SIGNAL":
            signal += 1
        dist = " ".join(f"{x:.2f}" for x in sorted(rewards))
        print(f"  {slug:<20} {kind:<16} mean={mean:<5} sd={sd:<5}  [{dist}]")
    frac = signal / len(slugs) if slugs else 0
    print(f"\n  -> {signal}/{len(slugs)} tasks have CONTROLLABLE signal ({frac:.0%})")
    print(f"  {'GO (controllable variance exists)' if frac >= 0.4 else 'SATURATED/NO-GO (computation mastered or unwinnable)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "KO", "HD"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

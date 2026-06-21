"""Agentic held-out eval: base vs trained through the deployed tool env."""
from __future__ import annotations
import argparse
import asyncio
import os
import statistics
import sys

from hud import Taskset
from hud.agents import create_agent

from quant_firm.env import analyze_filing


def make_agent(spec: str):
    provider, _, model = spec.partition(":")
    if provider == "fireworks":
        from hud.agents import OpenAIChatAgent
        from hud.agents.types import OpenAIChatConfig
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            sys.exit("FIREWORKS_API_KEY not set (needed for the trained model)")
        return OpenAIChatAgent(OpenAIChatConfig(
            model=model,
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=key,
        ))
    return create_agent(spec)


def build_taskset(tickers, difficulties, year):
    tasks = []
    for t in tickers:
        for d in difficulties:
            v = analyze_filing(ticker=t, year=year, difficulty=d)
            v.slug = f"{t}-gm-d{d}"
            tasks.append(v)
    return Taskset("quant-firm-heldout", tasks)


async def eval_agent(label, spec, taskset, group, concurrency):
    agent = make_agent(spec)
    job = await taskset.run(agent, group=group, max_concurrent=concurrency)
    rewards = [getattr(r, "reward", None) for r in job.runs]
    rewards = [x for x in rewards if x is not None]
    mean = statistics.mean(rewards) if rewards else 0.0
    print(f"  {label:<8} {spec:<60} mean reward = {mean:.3f}  (n={len(rewards)})")
    return mean, rewards


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)
    taskset = build_taskset(args.tickers, args.difficulties, args.year)
    print(f"TOOL-AUGMENTED held-out eval — tickers {args.tickers} "
          f"({len(taskset.tasks)} tasks x group {args.group})\n")
    if args.dry_run:
        make_agent(args.base)
        if args.trained:
            make_agent(args.trained)
        print("  [dry-run] taskset + agent(s) built OK; slugs:", list(taskset.tasks))
        return
    base_mean, _ = await eval_agent("base", args.base, taskset, args.group, args.concurrency)
    if args.trained:
        tr_mean, _ = await eval_agent("trained", args.trained, taskset, args.group, args.concurrency)
        delta = tr_mean - base_mean
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"\n  TOOL-AUGMENTED TRANSFER {arrow} {delta:+.3f} on held-out "
              f"({base_mean:.3f} -> {tr_mean:.3f})")
    else:
        print("\n  (no --trained yet — this is the base number to beat)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--trained", default=None)
    ap.add_argument("--tickers", nargs="*", default=["WMT", "PG", "CVX"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

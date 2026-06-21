"""
GRPO variance diagnostic — the trainability gate.

GRPO normalizes advantage WITHIN a group of rollouts of the same task:
    advantage_i = (reward_i - mean) / std
If a task's rollouts all score the same (std = 0), every advantage is 0 and the
task contributes NO gradient. Before burning GPU hours, measure which tasks
actually produce within-group spread.

This runs group rollouts through the HUD gateway (sampling temperature on), grades
each with the rubric, and classifies every task:
    SIGNAL          - real within-group spread -> trainable
    degenerate-low  - everyone fails (~0)       -> too hard / mis-specified
    degenerate-high - everyone passes (~1)      -> too easy
    degenerate-mid  - same middling score, no spread

    python -m quant_firm.variance --models openai:gpt-4o-mini \
        --tickers AAPL KO WMT --difficulties 2 --group 8 --temp 0.9
"""
from __future__ import annotations
import argparse
import asyncio
import os
import statistics
import sys

import hud.agents as A
from quant_firm.rubric import graders
from quant_firm.baseline import (build_taskset, filing_context, parse_models,
                                 DEFAULT_TICKERS)


async def _ask(client, model, prompt, temperature):
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        r = await client.chat.completions.create(
            model=model, max_tokens=512, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    if hasattr(client, "messages"):
        r = await client.messages.create(
            model=model, max_tokens=512, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    raise RuntimeError("unsupported client")


async def task_group(client, model, rubric, group, temp, with_context, sem):
    prompt = (filing_context(rubric) + rubric["prompt"]) if with_context else rubric["prompt"]

    async def one():
        async with sem:
            try:
                return graders.grade(await _ask(client, model, prompt, temp), rubric)["reward"]
            except Exception:
                return None

    return [r for r in await asyncio.gather(*[one() for _ in range(group)]) if r is not None]


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
    models = parse_models(args.models) if args.models else [("openai", "gpt-4o-mini")]
    tasks = build_taskset(args.tickers, args.difficulties)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"{len(tasks)} tasks x group {args.group} x {len(models)} model(s) "
          f"@ temp {args.temp}  = {len(tasks)*args.group*len(models)} calls\n")

    for provider, model in models:
        try:
            client = A.build_gateway_client(provider)
        except Exception as e:
            print(f"{provider}:{model}: client error {e}"); continue
        print(f"== {provider}:{model} ==")
        groups = await asyncio.gather(*[
            task_group(client, model, t, args.group, args.temp, args.with_context, sem)
            for t in tasks])
        signal = 0
        for t, rewards in zip(tasks, groups):
            kind, mean, sd = classify(rewards)
            if kind == "SIGNAL":
                signal += 1
            dist = "".join(f"{r:.2f} " for r in sorted(rewards))
            print(f"  {t['task_id']:<28} {kind:<16} mean={mean:<5} sd={sd:<5}  [{dist.strip()}]")
        frac = signal / len(tasks) if tasks else 0
        print(f"  -> {signal}/{len(tasks)} tasks have GRPO signal ({frac:.0%})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--difficulties", nargs="*", type=int, default=[2])
    ap.add_argument("--group", type=int, default=8, help="rollouts per task (GRPO group size)")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--with-context", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

"""
Step 4 of the climb — the money shot: held-out transfer.

Runs the BASE model and the TRAINED checkpoint on tickers NOT in the training
set, grades both with the same rubric, and reports the reward delta. A climb on
held-out tickers is the claim that matters — it shows the model learned the
skill, not the answers.

  base    -> gateway (e.g. tinker:Qwen/Qwen3-8B), runnable now for the pre number
  trained -> Fireworks OpenAI-compatible (fireworks:accounts/<acct>/models/<id>),
             needs FIREWORKS_API_KEY + a deployed checkpoint

    # pre-training baseline on held-out tickers (run now):
    python -m quant_firm.train.eval_transfer --base tinker:Qwen/Qwen3-8B \
        --tickers WMT PG CVX --difficulties 1 2 --runs 3
    # after training, add the trained model for the delta:
    python -m quant_firm.train.eval_transfer --base tinker:Qwen/Qwen3-8B \
        --trained fireworks:accounts/<acct>/models/qwen3-8b-quant-firm \
        --tickers WMT PG CVX --difficulties 1 2 --runs 3
"""
from __future__ import annotations
import argparse
import asyncio
import os
import statistics
import sys

import hud.agents as A
from quant_firm.rubric import generate, graders
from quant_firm.baseline import filing_context


def make_client(spec: str):
    """`fireworks:<model-id>` -> OpenAI client at Fireworks; else gateway provider."""
    provider, _, model = spec.partition(":")
    if provider == "fireworks":
        from openai import AsyncOpenAI
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            sys.exit("FIREWORKS_API_KEY not set (needed for the trained model)")
        return AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1",
                           api_key=key), model
    return A.build_gateway_client(provider), model


async def _ask(client, model, prompt, temp):
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        r = await client.chat.completions.create(
            model=model, max_tokens=2048, temperature=temp,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    r = await client.messages.create(
        model=model, max_tokens=2048, temperature=temp,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


async def eval_model(spec, rubrics, runs, temp, sem):
    client, model = make_client(spec)

    async def one(rubric):
        prompt = filing_context(rubric) + rubric["prompt"]
        async with sem:
            try:
                return graders.grade(await _ask(client, model, prompt, temp), rubric)["reward"]
            except Exception:
                return None

    jobs = [one(r) for r in rubrics for _ in range(runs)]
    rewards = [x for x in await asyncio.gather(*jobs) if x is not None]
    return statistics.mean(rewards) if rewards else 0.0, rewards


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)
    rubrics = []
    for t in args.tickers:
        for d in args.difficulties:
            try:
                rubrics.append(generate.build_margin_rubric(t, args.year, d))
            except Exception as e:
                print(f"  ! skip {t} d{d}: {e}", file=sys.stderr)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"HELD-OUT tickers {args.tickers}  ({len(rubrics)} tasks x {args.runs} runs)\n")

    base_mean, _ = await eval_model(args.base, rubrics, args.runs, args.temp, sem)
    print(f"  base    {args.base:<45} mean reward = {base_mean:.3f}")
    if args.trained:
        tr_mean, _ = await eval_model(args.trained, rubrics, args.runs, args.temp, sem)
        print(f"  trained {args.trained:<45} mean reward = {tr_mean:.3f}")
        delta = tr_mean - base_mean
        arrow = "↑" if delta > 0 else "↓"
        print(f"\n  TRANSFER {arrow} {delta:+.3f} on held-out tickers "
              f"({base_mean:.3f} -> {tr_mean:.3f})")
    else:
        print("\n  (no --trained yet — this is the pre-training baseline to beat)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tinker:Qwen/Qwen3-8B")
    ap.add_argument("--trained", default=None, help="fireworks:accounts/<acct>/models/<id>")
    ap.add_argument("--tickers", nargs="*", default=["WMT", "PG", "CVX"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=6)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

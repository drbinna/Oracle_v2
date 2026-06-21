"""
Step 1 of the climb — collect graded rollouts (expert iteration).

Samples the model (default Qwen3-8B, the one we'll fine-tune) over the taskset
several times at temperature, grades each with the rubric, and writes every
trace + reward to JSONL. Step 2 (build_sft) keeps the high-reward ones.

Uses --with-context by default so the model sees what its tools would fetch —
the high-reward completions then teach the OUTPUT DISCIPLINE (compute margin,
carry the accession, name a real driver, emit clean JSON) we want to distill.

    python -m quant_firm.train.collect --model tinker:Qwen/Qwen3-8B \
        --tickers AAPL MSFT NVDA KO --difficulties 1 2 --group 6 --temp 0.9 \
        --out traces.jsonl
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys

import hud.agents as A
from quant_firm.rubric import generate, graders
from quant_firm.baseline import filing_context


async def _ask(client, model, prompt, temp):
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        r = await client.chat.completions.create(
            model=model, max_tokens=2048, temperature=temp,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    if hasattr(client, "messages"):
        r = await client.messages.create(
            model=model, max_tokens=2048, temperature=temp,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    raise RuntimeError("unsupported client")


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)
    provider, _, model = args.model.partition(":")
    client = A.build_gateway_client(provider)

    rubrics = []
    for t in args.tickers:
        for d in args.difficulties:
            try:
                rubrics.append(generate.build_margin_rubric(t, args.year, d))
            except Exception as e:
                print(f"  ! skip {t} d{d}: {e}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    n_calls = len(rubrics) * args.group
    print(f"collecting {n_calls} traces from {args.model} "
          f"({len(rubrics)} tasks x {args.group}) ...")

    async def one(rubric):
        prompt = (filing_context(rubric) + rubric["prompt"]) if args.with_context \
            else rubric["prompt"]
        async with sem:
            try:
                completion = await _ask(client, model, prompt, args.temp)
                g = graders.grade(completion, rubric)
                return {"task_id": rubric["task_id"], "ticker": rubric["ticker"],
                        "difficulty": rubric["difficulty"], "prompt": prompt,
                        "completion": completion, "reward": g["reward"],
                        "breakdown": g["breakdown"]}
            except Exception as e:
                return {"task_id": rubric["task_id"], "error": repr(e)[:160]}

    jobs = [one(r) for r in rubrics for _ in range(args.group)]
    traces = await asyncio.gather(*jobs)

    ok = [t for t in traces if "error" not in t]
    with open(args.out, "w") as f:
        for t in ok:
            f.write(json.dumps(t) + "\n")
    rewards = [t["reward"] for t in ok]
    hi = sum(1 for r in rewards if r >= args.keep_threshold)
    print(f"wrote {len(ok)} traces -> {args.out}")
    if rewards:
        print(f"  reward: mean={sum(rewards)/len(rewards):.3f} "
              f"min={min(rewards):.2f} max={max(rewards):.2f}")
        print(f"  {hi} traces >= {args.keep_threshold} (SFT-eligible)")
    errs = len(traces) - len(ok)
    if errs:
        print(f"  ! {errs} call error(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinker:Qwen/Qwen3-8B")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "KO"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--with-context", action="store_true", default=True)
    ap.add_argument("--no-context", dest="with_context", action="store_false")
    ap.add_argument("--keep-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="traces.jsonl")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

"""
Baseline runner.

Sweeps the taskset across frontier models through the HUD gateway, grades each
answer with our verifiable rubric, and reports the reward distribution — so you
can see whether the baseline lands in the 20-50% band and tune `difficulty`.

This is the ZERO-TOOL baseline (prompt -> answer -> grade): the model's raw
analytical ability, no BashTool/web. The tool-augmented rollout runs through
`hud eval` on the deployed env. Same grader, so the numbers are comparable.

    export HUD_API_KEY=...            # gateway auth (read from env)
    python -m quant_firm.baseline                      # full sweep (default models)
    python -m quant_firm.baseline --models anthropic:claude-haiku-4-5 openai:gpt-4o-mini \
                                  --tickers AAPL MSFT --difficulties 1 3 --runs 1
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import hud.agents as A
from quant_firm.rubric import generate, graders

# (provider, model_name) — provider selects the gateway client
DEFAULT_MODELS = [
    ("anthropic", "claude-sonnet-4-5"),
    ("openai", "gpt-4o"),
    ("openrouter", "qwen3-max"),
]
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "KO", "WMT"]
DEFAULT_DIFFICULTIES = [1, 2, 3]
YEAR = 2022


def build_taskset(tickers, difficulties):
    """Build each rubric ONCE (MD&A/EDGAR fetched once), reused across models."""
    tasks = []
    for t in tickers:
        for d in difficulties:
            try:
                tasks.append(generate.build_margin_rubric(t, YEAR, d))
            except Exception as e:
                print(f"  ! skip {t} d{d}: {e}", file=sys.stderr)
    return tasks


def filing_context(rubric) -> str:
    """What the agent's read_filing tool would hand it: the raw numbers + accession.
    Deliberately NOT the margin or the MD&A drivers — those stay the thing to derive."""
    gt = rubric["ground_truth"]
    return (
        f"[FILING DATA — {gt['ticker']} FY{gt['fiscal_year']} 10-K, "
        f"accession {gt['accession']}]\n"
        f"Net sales / revenue: ${gt['revenue_usd']:,}\n"
        f"Cost of sales: ${gt['cogs_usd']:,}\n\n"
    )


async def _ask(client, model: str, prompt: str) -> str:
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        r = await client.chat.completions.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    if hasattr(client, "messages"):
        r = await client.messages.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    raise RuntimeError(f"unsupported gateway client: {type(client).__name__}")


async def run_model(provider, model, tasks, runs, sem, with_context=False):
    try:
        client = A.build_gateway_client(provider)
    except Exception as e:
        return {"model": f"{provider}:{model}", "error": repr(e)[:160], "rows": []}

    async def one(rubric, rep):
        async with sem:
            prompt = (filing_context(rubric) + rubric["prompt"]) if with_context \
                else rubric["prompt"]
            try:
                ans = await _ask(client, model, prompt)
                g = graders.grade(ans, rubric)
                return {"task": rubric["task_id"], "difficulty": rubric["difficulty"],
                        "reward": g["reward"], "breakdown": g["breakdown"]}
            except Exception as e:
                return {"task": rubric["task_id"], "difficulty": rubric["difficulty"],
                        "reward": 0.0, "breakdown": [], "error": repr(e)[:120]}

    jobs = [one(t, r) for t in tasks for r in range(runs)]
    rows = await asyncio.gather(*jobs)
    return {"model": f"{provider}:{model}", "rows": rows}


def summarize(res):
    rows = res.get("rows", [])
    if res.get("error") or not rows:
        return {"model": res["model"], "error": res.get("error", "no rows")}
    rewards = [r["reward"] for r in rows]
    mean = statistics.mean(rewards)
    # per-criterion pass rate
    crit = {}
    for r in rows:
        for b in r.get("breakdown", []):
            crit.setdefault(b["id"], []).append(b["score"])
    pass_rate = {k: round(statistics.mean(v), 2) for k, v in crit.items()}
    # per-difficulty mean (shows how tolerance moves the band)
    bydiff = {}
    for r in rows:
        bydiff.setdefault(r["difficulty"], []).append(r["reward"])
    bydiff = {d: round(statistics.mean(v), 3) for d, v in sorted(bydiff.items())}
    errs = sum(1 for r in rows if r.get("error"))
    return {
        "model": res["model"],
        "n": len(rows),
        "mean_reward": round(mean, 3),
        "stdev": round(statistics.pstdev(rewards), 3) if len(rewards) > 1 else 0.0,
        "min": round(min(rewards), 3), "max": round(max(rewards), 3),
        "in_band_20_50": 0.20 <= mean <= 0.50,
        "by_difficulty": bydiff,
        "criterion_pass_rate": pass_rate,
        "errors": errs,
    }


def print_report(summaries):
    print("\n" + "=" * 72)
    print("BASELINE REWARD REPORT  (target band: 0.20 - 0.50)")
    print("=" * 72)
    for s in summaries:
        if s.get("error"):
            print(f"\n{s['model']}: ERROR {s['error']}")
            continue
        band = "IN BAND ✓" if s["in_band_20_50"] else "out of band"
        print(f"\n{s['model']}   mean={s['mean_reward']}  sd={s['stdev']}  "
              f"[{s['min']}–{s['max']}]  n={s['n']}  -> {band}")
        print(f"   by difficulty (tol lever): {s['by_difficulty']}")
        print(f"   criterion pass-rate:       {s['criterion_pass_rate']}")
        if s["errors"]:
            print(f"   ! {s['errors']} call error(s)")
    print()


def parse_models(items):
    out = []
    for it in items:
        prov, _, name = it.partition(":")
        if not name:
            print(f"  ! bad --models entry '{it}', want provider:model", file=sys.stderr)
            continue
        out.append((prov, name))
    return out


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set — export it (gateway auth).", file=sys.stderr)
        sys.exit(1)
    models = parse_models(args.models) if args.models else DEFAULT_MODELS
    print(f"building taskset: {args.tickers} x difficulty {args.difficulties} ...")
    tasks = build_taskset(args.tickers, args.difficulties)
    print(f"  {len(tasks)} tasks x {len(models)} models x {args.runs} run(s) "
          f"= {len(tasks)*len(models)*args.runs} calls")
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    results = await asyncio.gather(
        *[run_model(p, m, tasks, args.runs, sem, args.with_context) for p, m in models])
    summaries = [summarize(r) for r in results]
    print_report(summaries)
    print(f"done in {time.time()-t0:.1f}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summaries": summaries,
                       "raw": [{"model": r["model"], "rows": r.get("rows", [])}
                               for r in results]}, f, indent=2, default=str)
        print(f"wrote {args.out}  (no secrets)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="provider:model ... (default: frontier set)")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--difficulties", nargs="*", type=int, default=DEFAULT_DIFFICULTIES)
    ap.add_argument("--runs", type=int, default=1, help="repeats per task (band ~10)")
    ap.add_argument("--with-context", action="store_true",
                    help="inject filing data (revenue/cogs/accession) as read_filing would")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=None, help="write JSON results here")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

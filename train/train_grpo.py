"""
Phase 2 — hosted GRPO short run (LocalRuntime rollouts + hosted TrainingClient).

Architecture (resolved from docs-hud): the deployed env cannot reach the local
:8080 specialists (127.0.0.1 = container-local), so we DON'T deploy. Rollouts run
locally against the working env + Exa specialists with return_token_ids=True; the
local Runs carry inline token samples, which the HOSTED TrainingClient (Tinker/LoRA)
turns into gradients + a promoted checkpoint. No deploy, no egress/secret risk.

    python -m quant_firm.train.train_grpo \
        --tickers AAPL MSFT NVDA HD --difficulties 1 \
        --group 8 --iters 4 --lr 1e-5 --max-concurrent 6

SHORT + BOUNDED by design. Prints batch mean_reward and checkpoint metrics each
step so you can read the trend. Stop if flat/down.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys

from hud import Taskset, LocalRuntime, TrainingClient
from hud.agents import create_agent
from hud.eval import Job

from quant_firm.env import analyze_filing

MODEL = "quant-firm-rl"


def build_taskset(tickers, difficulties, year):
    tasks = []
    for t in tickers:
        for d in difficulties:
            v = analyze_filing(ticker=t, year=year, difficulty=d)
            v.slug = f"{t}-gm-d{d}"
            tasks.append(v)
    return Taskset("quant-firm-gm-train", tasks)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


async def main_async(args):
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY not set.", file=sys.stderr); sys.exit(1)

    taskset = build_taskset(args.tickers, args.difficulties, args.year)
    # return_token_ids => the gateway returns token ids + logprobs the trainer needs
    agent = create_agent(MODEL, completion_kwargs={
        "max_tokens": args.max_tokens,
        "extra_body": {"return_token_ids": True},
    })
    trainer = TrainingClient(MODEL)
    runtime = LocalRuntime("quant_firm/env.py")

    n_tasks = len(taskset.tasks)
    print(f"GRPO short run | model={MODEL} | {n_tasks} tasks x group {args.group} "
          f"= {n_tasks * args.group} rollouts/iter x {args.iters} iters")
    print(f"  lr={args.lr} group_size={args.group} loss=importance_sampling "
          f"max_tokens={args.max_tokens} concurrency={args.max_concurrent}\n")

    session = await Job.start(MODEL, group=args.group)
    trend = []
    for it in range(args.iters):
        start = len(session.runs)
        await taskset.run(agent, runtime=runtime, group=args.group,
                          max_concurrent=args.max_concurrent, job=session)
        batch = session.runs[start:]
        rewards = [getattr(r, "reward", 0.0) or 0.0 for r in batch]
        batch_mean = _mean(rewards)

        res = await trainer.step(batch, learning_rate=args.lr, group_size=args.group)
        trend.append(batch_mean)

        # checkpoint metrics (hosted spend + learning proxies)
        ck_line = ""
        try:
            cps = await trainer.checkpoints()
            if cps:
                c = cps[-1]
                ck_line = (f"  ckpt={getattr(c,'name',getattr(c,'id','?'))} "
                           f"mean_reward={getattr(c,'mean_reward',None)} "
                           f"tokens={getattr(c,'num_tokens',None)} "
                           f"datums={getattr(c,'num_datums',None)} "
                           f"metrics={getattr(c,'metrics',{})}")
        except Exception as e:
            ck_line = f"  (checkpoints read failed: {e!r})"

        print(f"[iter {it+1}/{args.iters}] batch_mean_reward={batch_mean:.4f} "
              f"n={len(batch)} reward_spread=[{min(rewards):.2f},{max(rewards):.2f}] "
              f"checkpoint_id={getattr(res,'checkpoint_id',None)}")
        print(ck_line + "\n")

    arrow = " -> ".join(f"{x:.3f}" for x in trend)
    delta = (trend[-1] - trend[0]) if len(trend) >= 2 else 0.0
    print(f"TREND batch_mean_reward: {arrow}")
    print(f"  net delta (last-first) = {delta:+.4f}  "
          f"=> {'UP (extend)' if delta > 0.02 else 'FLAT/DOWN (stop & report)'}")
    print("  NOTE: credits not readable via CLI — check hud.ai/project/api-keys; "
          "checkpoint tokens/datums above are the hosted-spend proxy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "HD"])
    ap.add_argument("--difficulties", nargs="*", type=int, default=[1])
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-concurrent", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

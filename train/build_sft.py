"""
Step 2 of the climb — turn high-reward rollouts into a Fireworks SFT dataset.

Rejection sampling: keep only traces whose rubric reward clears the threshold,
then write them in Fireworks' JSONL chat schema ({"messages":[...]}). Qwen3
thinking traces are preserved in the assistant turn — Fireworks supports SFT on
thinking models, so the model learns the reasoning that earned the reward, not
just the final JSON.

    python -m quant_firm.train.build_sft --in traces.jsonl --out sft.jsonl \
        --threshold 0.8 --per-task-cap 4
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict

SYSTEM = ("You are a financial analyst. Compute the requested metric directly "
          "from the filing data, cite the exact SEC accession number, identify "
          "the driver management gives in the MD&A, and return ONLY the JSON "
          "object the task specifies — no prose outside it.")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="traces.jsonl")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--per-task-cap", type=int, default=4,
                    help="max examples per task_id (avoid one task dominating)")
    args = ap.parse_args()

    traces = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    kept, seen, per_task = [], set(), defaultdict(int)
    for t in sorted(traces, key=lambda x: -x.get("reward", 0)):
        if t.get("reward", 0) < args.threshold or "completion" not in t:
            continue
        import re as _re
        _m = _re.search(r"\b([A-Z]{1,5})\b.{0,40}(?:FY|fiscal|10-K|gross margin)", t.get("prompt", ""))
        tid = (t.get("task_id") or t.get("slug") or t.get("trace_id")
               or (_m.group(1) if _m else None) or "task")
        key = (tid, _norm(t["completion"])[:400])
        if key in seen:
            continue
        if per_task[tid] >= args.per_task_cap:
            continue
        seen.add(key)
        per_task[tid] += 1
        kept.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": t["prompt"]},
                {"role": "assistant", "content": t["completion"]},
            ]
        })

    with open(args.out, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"kept {len(kept)} / {len(traces)} traces "
          f"(reward >= {args.threshold}) -> {args.out}")
    print("  per-task examples:", dict(per_task))
    if kept:
        a = kept[0]["messages"][-1]["content"]
        print(f"  sample assistant turn (first 180 chars): {_norm(a)[:180]}")
    if len(kept) < 50:
        print("  NOTE: small set — scale collect.py (more tickers/difficulties/group) "
              "toward a few hundred examples for a stronger LoRA.")


if __name__ == "__main__":
    main()

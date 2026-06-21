"""
Step 3 of the climb — fine-tune Qwen3-8B on Fireworks (LoRA, rejection-sampling SFT).

Dry-run by default: validates the dataset and prints the exact firectl commands.
Pass --run to execute them (needs `firectl` installed + `firectl login`, and
FIREWORKS_API_KEY). Get LoRA twice as cost-efficient per Fireworks; hackathon
credits: HUD-HACK-2026.

    python -m quant_firm.train.fireworks_sft --dataset sft.jsonl            # dry-run
    python -m quant_firm.train.fireworks_sft --dataset sft.jsonl --run      # execute

IMPORTANT (Fireworks, as of Feb 2026): serverless LoRA is NOT supported, so after
training you must create an on-demand DEPLOYMENT before serving (see the printed
deploy command), then point the OpenAI-compatible client at api.fireworks.ai.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys

# Confirm the exact base-model slug with:  firectl list models | grep -i qwen3
BASE_MODEL = "accounts/fireworks/models/qwen3-8b"


def validate(path: str) -> int:
    n = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            assert isinstance(msgs, list) and len(msgs) >= 2, f"line {i}: bad messages"
            roles = [m["role"] for m in msgs]
            assert roles[-1] == "assistant", f"line {i}: last turn must be assistant"
            assert all(m.get("content") for m in msgs), f"line {i}: empty content"
            n += 1
    if n == 0:
        sys.exit("dataset is empty — run collect.py + build_sft.py first")
    return n


def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sft.jsonl")
    ap.add_argument("--dataset-id", default="quant-firm-gm")
    ap.add_argument("--output-model", default="qwen3-8b-quant-firm")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--run", action="store_true", help="actually execute (needs firectl)")
    args = ap.parse_args()

    n = validate(args.dataset)
    print(f"dataset OK: {n} examples in {args.dataset}\n")

    create_ds = ["firectl", "create", "dataset", args.dataset_id, args.dataset]
    create_job = [
        "firectl", "create", "sftj",
        "--base-model", args.base_model,
        "--dataset", args.dataset_id,
        "--output-model", args.output_model,
        "--lora-rank", str(args.lora_rank),
        "--epochs", str(args.epochs),
        "--learning-rate", args.lr,
    ]
    deploy = ["firectl", "create", "deployment",
              f"accounts/fireworks/models/{args.output_model}"]

    print("1) upload dataset:")
    print("  $", " ".join(create_ds))
    print("2) launch LoRA SFT job:")
    print("  $", " ".join(create_job))
    print("3) deploy on-demand (serverless LoRA unsupported), then serve:")
    print("  $", " ".join(deploy))
    print(f"  -> OpenAI-compatible at https://api.fireworks.ai/inference/v1, "
          f"model accounts/<acct>/models/{args.output_model}")
    print("4) measure transfer:")
    print(f"  python -m quant_firm.train.eval_transfer "
          f"--trained fireworks:accounts/<acct>/models/{args.output_model} "
          f"--tickers WMT PG CVX   # HELD-OUT tickers\n")

    if args.run:
        print("executing ...")
        run(create_ds)
        run(create_job)
        print("job submitted. poll with: firectl list sftj   |   then run step 3.")
    else:
        print("(dry-run — re-run with --run to execute, or paste the commands above)")


if __name__ == "__main__":
    main()

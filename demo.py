"""
Standalone proof — no HUD platform / GPUs needed.

    python -m quant_firm.demo

Builds a rubric from LIVE EDGAR data, then grades two answers:
  - a correct one   -> high reward
  - a reward-hacky one (wrong number, no citation) -> low reward
Shows the per-criterion breakdown so you can see the dense, verifiable signal.
"""
from __future__ import annotations
import json
from quant_firm.rubric import generate, graders


def show(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, default=str))


def main():
    rubric = generate.build_margin_rubric("AAPL", 2022, difficulty=2)

    print("PROMPT (what the agent sees):")
    print(rubric["prompt"])
    show("GROUND TRUTH (env-side, hidden from agent)", rubric["ground_truth"])

    gt = rubric["ground_truth"]

    good = {
        "gross_margin_pct": gt["gross_margin_pct"],
        "revenue_usd": gt["revenue_usd"],
        "citations": [gt["accession"]],
        "drivers": ["product mix"],
    }
    hacky = {  # confidently wrong + no real citation
        "gross_margin_pct": 60.0,
        "revenue_usd": 400_000_000_000,
        "citations": ["made-up-filing"],
        "drivers": [],
    }
    partial = {  # right margin, wrong revenue, good citation (dense partial credit)
        "gross_margin_pct": gt["gross_margin_pct"],
        "revenue_usd": 1,
        "citations": [gt["accession"]],
        "drivers": [],
    }

    for name, ans in [("CORRECT answer", good),
                      ("REWARD-HACKY answer", hacky),
                      ("PARTIAL answer", partial)]:
        res = graders.grade(ans, rubric)
        print(f"\n--- {name} -> reward = {res['reward']} ---")
        for b in res["breakdown"]:
            mark = "PASS" if b["score"] else "fail"
            print(f"   [{mark}] {b['section']:<22} (w={b['weight']}) {b['id']}")


if __name__ == "__main__":
    main()

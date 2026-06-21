# Autonomous Quant Firm — a verifiable RL environment + a trainability law

**HUD Frontier/RSI RL Environments Hackathon.** A verifiable-reward HUD environment where an
agent computes a company's gross margin from its SEC 10-K — and the honest RL result that came
out of trying to train on it.

> **Headline (the contribution):** A verifiable RL reward is trainable only if its variance is
> **policy-controllable**, not exogenous: `I(A_policy ; R | X_exogenous) > 0`. Passing a variance
> gate (`Var(R) > 0`) is **not** the same as having spread a gradient can climb.

## Links
- **HUD environment:** https://hud.ai/environments/4a7782ac-8489-4b1c-8dc7-78ab58fa3d4a
- **HUD taskset (`quant-firm-margin`, 10 tasks):** https://hud.ai/tasksets/fe7fab65-050d-4312-8071-77dfbadeb0c1
- **Full writeup:** [`WRITEUP.md`](WRITEUP.md) · **Build inventory:** [`INVENTORY.md`](INVENTORY.md)

## What's here
- **Environment** (`env.py`, `env_controllable.py`) — HUD v6, `@env.template` two-yield tasks,
  a verifiable rubric grader (no LLM judge), ground truth from **SEC EDGAR/XBRL only**
  (`data/edgar.py`). Tools are served as an `mcp` capability (FastMCP); the deployed env serves
  them **in-process** so it runs in the cloud.
- **Reward** (`rubric/`, `controllable.py`) — weighted, programmatic: margin/revenue/citation,
  plus a redesigned **controllable-variance** reward (selection + arithmetic over a
  distractor-laden statement).
- **RFT pipeline** (`train/`) — hosted **GRPO/LoRA** on `Qwen/Qwen3-8B` via HUD's
  `TrainingClient` (Tinker backend), with **LocalRuntime rollouts** feeding the hosted trainer
  (`return_token_ids` → inline token samples). Variance gates as the pre-spend trainability check.

## Integrations
HUD (gateway, `TrainingClient`, models CLI, `LocalRuntime`/`Taskset`/`Job`, deploy) · **Tinker**
(LoRA RL backend) · **Exa** (agent-side web discovery) · **SEC EDGAR/XBRL** (ground truth) ·
**FastMCP** (specialist tool servers).

## The result (honest)
1. **Oracle tools → NO-GO.** Hand-feeding the answers saturated the model (8/10 tasks `sd=0`).
2. **Exa discovery (self-retrieval) → GO.** 9/10 tasks showed within-group spread (90%).
3. **...but GRPO trained flat** (10 steps, mean ≈ 0.333, within noise) — the spread was
   **exogenous** (retrieval luck), so the gradient couldn't climb.
4. **Redesigned the reward for controllable variance** (always-available, distractor-laden
   statement) → Qwen3-8B **saturates** it: the controllable computation is already mastered.

So the task is untrainable for a capable 8B *both ways* — exogenous spread can't climb, mastered
skill has nothing to climb. The reusable takeaway is the **controllable-variance gate**: check
`I(A;R|X)>0` *before* spending compute. A weaker base or a genuinely harder multi-step task is the
path to a positive self-improvement curve.

## Verifiable rewards (RLVR) & related work
Rewards here are **verifiable**: every criterion is a programmatic check against SEC EDGAR/XBRL
ground truth — **no LLM judge anywhere** (`rubric/graders.py`, `controllable.py`). That makes this
a Reinforcement-Learning-with-Verifiable-Rewards (RLVR) environment by construction.

Our flat-training finding connects directly to recent work on GRPO's **low-within-group-variance**
failure: [RC-GRPO (arXiv:2602.03025)](https://arxiv.org/abs/2602.03025) diversifies rollouts with
discrete reward-goal tokens to restore advantage signal, and [Gradient Starvation in Binary-Reward
GRPO (arXiv:2605.07689)](https://arxiv.org/abs/2605.07689) analyzes how group-mean centering
collapses. Our **trainability law** is the *precondition* those fixes assume: reward-token
diversification (or any exploration trick) only helps when the variance is **policy-controllable**.
On this env it isn't — exogenous retrieval luck on one axis, a saturated computation on the other —
so no conditioning creates a climbable gradient. That's why our **controllable-variance gate runs
before training**, not after.

## Quickstart
```bash
export HUD_API_KEY=...                       # HUD gateway + training
python -m quant_firm.subagents_controllable  # local specialist server (:8082)

# trainability gates (no GPU):
python -m quant_firm.train.variance_env          --model Qwen/Qwen3-8B --group 8   # Exa path
python -m quant_firm.train.variance_controllable --model Qwen/Qwen3-8B --group 8   # controllable

# hosted GRPO short run (LocalRuntime rollouts + hosted Tinker/LoRA trainer):
python -m quant_firm.train.train_grpo --tickers AAPL MSFT NVDA HD --group 8 --iters 4 --lr 1e-5
```

Ground truth is EDGAR/XBRL only; Exa is strictly an agent-side discovery tool and never touches
grading. Set `EXA_API_KEY` in `.env` (gitignored) for the Exa discovery path.

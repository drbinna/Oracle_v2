# Project inventory — autonomous quant firm RL

Everything done this session: the RL implemented, all integrations, files, and results.
Companion to `WRITEUP.md` (the narrative + the trainability law).

## 1. What we did (chronological)

| Phase | Action | Result |
|---|---|---|
| 0 | Diagnose HD scoring 0.00 | **Output truncation**, not bad data — Qwen3-8B reasoning blows the 512-token budget; at 4096 HD is stable 0.80. EDGAR gold consistent. |
| 0 | Check `--difficulty` | Only changes the numeric tolerance (`0.25/0.10/0.05` pp); a near no-op for exact arithmetic. |
| 1 | Agentic variance gate, **oracle** specialists | **NO-GO (saturation)** — 2/10 tasks had signal; 8/10 `sd=0.0` (oracles spoon-fed answers). |
| 1 | Rewrite specialists → **Exa discovery** (Option B) | **GO** — 9/10 tasks have GRPO signal (90%), spread 0.0–0.85. |
| 2 | Resolve tool-reachability (docs-hud) | Deployed env **can't** reach local `:8080` (`127.0.0.1` = container-local) → use **LocalRuntime rollouts + hosted trainer**, no deploy. |
| 2 | Fork + hosted GRPO | `quant-firm-rl`; smoke (1 iter) + 4 iters @lr1e-5 + 5/8 iters @lr2e-5 = **10 steps, flat** (mean ≈0.333, within noise). Stopped per flat-trend rule. |
| 3 | Reward redesign for **controllable variance** | Always-available distractor-laden statement; Qwen3-8B **saturates** (≈1.0) → controllable variance ≈0 (skill already mastered). |
| — | Writeup + memory | `WRITEUP.md`, 3 memory files. |

## 2. RL implemented

- **Algorithm:** GRPO — group-relative advantage (`advantage = reward − group_mean`,
  normalized within `group_size=8`), via HUD's hosted `TrainingClient` (Tinker/LoRA backend).
- **Loss:** `importance_sampling` (on-policy policy gradient, rollout-logprob ratio) — the
  default surrogate; GRPO grouping supplies the advantage.
- **Optimizer step:** `trainer.step(batch, learning_rate, group_size=8)` = one
  forward_backward + one optim_step → checkpoint promoted to gateway head.
- **Trajectory plumbing:** local rollouts sampled with `return_token_ids=True` →
  Runs carry inline token samples (`TrajectoryPayload`) → fed straight to the hosted trainer.
- **Policy / base:** `Qwen/Qwen3-8B` (8B open-weights), forked to trainable slug
  `quant-firm-rl` (`3f4c72b9-33d3-4f8f-81f7-662ba5c830ec`); **LoRA** via Tinker.
- **Schedule run:** lr 1e-5 (steps 1–5) then 2e-5 (steps 6–10), group 8, 4 tasks/step
  (32 rollouts/step). 10 promoted checkpoints `step-000001…step-000010`.
- **Reward:** verifiable rubric (no LLM judge) — calc_margin/revenue/citation/attribution,
  graded vs SEC EDGAR/XBRL. Redesigned variant weights the controllable computation.
- **Diagnostics:** variance gate (within-group spread = trainability), checkpoint metrics
  (`reward_std`, `tinker.loss`, `sampling_logprob_mean`), trend vs standard error.

## 3. Integrations

| Integration | Use |
|---|---|
| **HUD gateway** (`inference.beta.hud.ai`) | Model inference for all rollouts (`create_agent`, `OpenAIChatAgent`). |
| **HUD `TrainingClient`** | Hosted GRPO/LoRA training; `step`/`checkpoints`/`head`. |
| **HUD models CLI** | `hud models list` / `fork` / `checkpoints` — trainable model registry. |
| **HUD `LocalRuntime` / `Taskset` / `Job`** | Local rollouts + GRPO grouping feeding the hosted trainer. |
| **HUD `HUDRuntime` + `hud deploy`** | Evaluated and **rejected** for this run (local-tool reachability). |
| **docs-hud MCP** | Source of truth for the v6 API (training, runtime, capabilities, deploy). |
| **Tinker** (via HUD) | The actual RL/LoRA training backend behind `quant-firm-rl`. |
| **Exa** (`exa-py==2.14.0`, REST) | Agent-side **discovery** tools (`web_search`, `read_filing`); key in `.env`. |
| **SEC EDGAR / XBRL** (`data/edgar.py`) | **Ground truth only** (revenue/COGS/accession). Never agent-facing in Option B. |
| **FastMCP** | Specialist tool servers (`:8080` Exa, `:8082` statements) as `mcp` capabilities. |
| **HUD env control channel** | `:8765` env serving the `@env.template` rollouts. |

## 4. Files (this session)

**New:**
- `train/variance_env.py` — agentic variance gate (Exa path).
- `train/train_grpo.py` — hosted GRPO loop (LocalRuntime + TrainingClient).
- `train/variance_controllable.py` — controllable-variance gate.
- `controllable.py` — reward redesign (noisy statement builder + compute rubric + grader).
- `subagents_controllable.py` — `:8082` statement server.
- `env_controllable.py` — controllable env.
- `WRITEUP.md`, `INVENTORY.md` — deliverables.
- `../.env` — `EXA_API_KEY`.
- memory: `trainability-law.md`, `honest-science-discipline.md`, `quant-firm-rl-experiment.md`.

**Modified / preserved:**
- `subagents.py` — oracle specialists → Exa discovery (original at `subagents_oracle.py.bak`).

**Pre-existing, used:** `env.py`, `rubric/generate.py`, `rubric/graders.py`, `data/edgar.py`,
`data/mdna.py`, `baseline.py`, `train/collect_env.py`, `train/eval_transfer.py` (held-out
transfer harness — available, not yet run).

## 5. Headline result
The hosted GRPO loop works end-to-end on an 8B; the **honest result is a negative** that proves
the trainability law both ways: spread that passed the variance gate was **exogenous** (Exa
retrieval luck → flat training), and when we isolated **controllable** variance the 8B had already
**saturated** it (nothing to climb). Gross-margin-from-a-10-K is not a trainable RL target for a
capable 8B. Next lever (not run): a weaker base (`Qwen3.5-4B` / `Llama-3.2-3B`) or a harder
multi-step task, then held-out transfer + placebo to confirm self-improvement.

## 6. Live state
- Trained head: `step-000010` (noise-equivalent, discardable via `hud models head --set`).
- Servers up: `:8080` (Exa specialists), `:8082` (statements), `:8765` (env).
- Exa change reversible; ground truth untouched (EDGAR/XBRL).

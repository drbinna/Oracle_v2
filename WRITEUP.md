# Autonomous Quant Firm — environment, method, and an honest negative result

## TL;DR
We built a verifiable-reward agent environment (compute a company's gross margin from its
SEC 10-K) and tried to improve an 8B open model on it with hosted GRPO. The headline is not a
metric — it's a **method result**:

> A verifiable RL reward is trainable only if its variance is **policy-controllable**, not
> exogenous: `I(A_policy ; R | X_exogenous) > 0`. Passing a variance gate (`Var(R) > 0`) is
> necessary but **not** sufficient.

We observed exactly this. After fixing a saturated environment, the task **passed the variance
gate (90% of tasks showed within-group spread, GO)** — yet **10 GRPO steps were flat**
(mean ≈ 0.333, zero slope, across lr 1e-5 and 2e-5). The spread was real but exogenous
(retrieval luck), so the gradient had nothing to climb.

## The environment
- **Task:** given a ticker/year, report FY gross margin as JSON: `gross_margin_pct`,
  `revenue_usd`, `citations` (SEC accession), `drivers` (MD&A).
- **Verifiable reward (no LLM judge):** weighted rubric — calc_margin 0.45 (numeric tol),
  calc_revenue 0.15, citation 0.20 (accession match), attribution 0.20 (MD&A keyword).
  Ground truth comes **only** from SEC EDGAR/XBRL (`data/edgar.py`).
- **Agent capabilities:** a shell workspace + an `mcp` capability of "specialist" tools.
- **Model:** `Qwen/Qwen3-8B`, hosted GRPO via HUD's Tinker/LoRA `TrainingClient`.

## Method and what each phase measured

### Phase 0 — clean the reward signal
- **HD scored 0.00 even with the numbers handed in** — diagnosed as **output truncation**, not
  bad data. Qwen3-8B is a reasoning model; at `max_tokens=512` it never emits the final JSON.
  At 4096 HD is a stable 0.80. EDGAR gold for HD is internally consistent (33.53%). *Kept HD.*
- **Difficulty** (`tol = {1:0.25, 2:0.10, 3:0.05}` pp) only changes the calc tolerance; for exact
  arithmetic it's a near no-op (identical d1/d2 distributions are expected, not a bug).

### Phase 1 — variance gate (go/no-go, no GPU)
Run each task as a group; require within-group spread (the GRPO advantage).

- **Oracle specialists → NO-GO (saturation).** The original tools (`pull_financials`,
  `audit_gross_margin`, `read_mdna_drivers`) returned exact numbers, the truth margin, and the
  accession. Base Qwen3-8B read answers off the spoon: **2/10 tasks had signal; 8/10 had
  `sd=0.0`** (5 saturated at 1.00, 3 flat at 0.80). Uniform 0.80 is as untrainable as uniform 0.0.
- **Exa discovery (Option B) → GO.** We removed the oracles and gave the agent pure discovery
  tools (`web_search`, `read_filing`, Exa SDK) — it must find the 10-K, extract revenue/COGS from
  prose, and derive the accession from the SEC URL. Grader unchanged (EDGAR only).
  **Result: 9/10 tasks have GRPO signal (90%), spread 0.0–0.85.** Gate GO.

### Phase 2 — hosted GRPO short run
Architecture (resolved from the HUD docs): a deployed env **cannot reach the local `:8080`
specialists** (`127.0.0.1` is container-local), so we did **not** deploy. We used **LocalRuntime
rollouts** (`return_token_ids=True`, so local Runs carry inline token samples) feeding the
**hosted `TrainingClient`** (Tinker/LoRA). Validated end-to-end: fork → rollout → `trainer.step`
→ promoted checkpoint.

## Results (with their noise)

**Gate (base Qwen3-8B, tool-augmented, group 8):**

| Env | Tasks with GRPO signal | Verdict |
|---|---|---|
| Oracle specialists | 2/10 (8 at sd=0.0) | NO-GO (saturated) |
| Exa discovery (Option B) | 9/10 (90%) | GO |

**Training (10 GRPO steps, group_size 8, 4 tasks × 8 rollouts/step):**

```
step:   1     2     3     4     5     6     7     8     9    10
lr:   1e-5  1e-5  1e-5  1e-5  1e-5  2e-5  2e-5  2e-5  2e-5  2e-5
mean: .353  .322  .344  .316  .344  .336  .319  .364  .323  .330
```
Mean ≈ 0.333, **zero slope**. Per-point standard error ≈ 0.15/√32 ≈ **0.027**, which swamps every
step-to-step move. `reward_std` stayed 0.12–0.18 throughout — the signal never collapsed; the
policy simply did not move.

## Honest caveats (read these)
- **The lr comparison is confounded, not controlled.** The 2e-5 arm *continued from the 1e-5
  arm's weights* (`step-000005`), so this is not a clean lr sweep from identical base weights. We
  did not run controlled arms.
- **The "trend" in the first 4-iter run (+0.022) is within one standard error** — we do not claim
  it as learning, and the 10-step view confirms flatness.
- **A recurring filesystem-tool error** (the ssh workspace missing a file) added noise to rollouts;
  rewards were earned despite it, but it was not cleaned up.
- **Credits could not be read programmatically** (`hud` has no usage command). Hosted-spend proxy:
  ~530k training tokens across the steps; check hud.ai/project/api-keys for actual burn.

## Why it stayed flat, and what would fix it
The within-group spread that passed the gate is **largely exogenous**: a rollout's reward depends
more on *which noisy Exa highlights came back* (quarterly vs annual columns, thousands vs millions,
whether COGS surfaced) than on a token-level behavior the gradient can reinforce. That is
`I(A_policy ; R | X_exogenous) ≈ 0` — measured, not assumed.

**The fix is to reward the controllable computation, not retrieval luck:** partial credit for
issuing the right query, calling `read_filing` on the discovered SEC URL, and citing *that*
accession — actions the policy controls — plus larger/cleaner batches so the advantage estimate
clears its noise floor. Re-gate for *controllable* variance before spending again.

## Reward redesign for controllable variance — and what it revealed
We then redesigned the reward to target *controllable* variance directly (`controllable.py`,
`env_controllable.py`, `:8082` statement server). The design holds `X_exogenous` ~constant by
**always providing the complete income statement**, but in realistic, distractor-laden form
(quarterly vs annual columns, prior year, and — in hard mode — segmented Products+Services with no
subtotal, requiring aggregation). Reward weight moves onto the figures the agent **selects and
computes** vs EDGAR. Because the data is always available, any within-group spread is now
policy-controllable, not retrieval luck — no trace capture needed.

**Result: Qwen3-8B saturates (controllable variance ≈ 0).** Base model, group 8 (3 tickers, d1):

| design | per-task rewards | verdict |
|---|---|---|
| distractor columns | AAPL `[0×1, 1×7]`, KO `1.0×8`, HD `1.0×8` | saturated |
| + segment summation (hard) | AAPL `1.0×8`, KO `[0×1, 1×7]`, HD `1.0×8` | saturated |

So **both axes are untrainable for this model on this task**: the exogenous axis (retrieval) is luck
the gradient can't climb; the controllable axis (statement parse + margin arithmetic, even with
segment aggregation) is **already mastered**, so there's nothing to learn. This is the trainability
law cutting both ways — `Var(R)>0` on the Exa env was exogenous; `Var(R)≈0` on the provided-statement
env is saturation. The reward redesign succeeded at *isolating* controllable variance; the honest
measurement is that it's ~zero here.

**Implication:** gross-margin-from-a-10-K is not a useful RL target for a capable 8B. A trainable
controllable target needs either a genuinely harder multi-step reasoning task (one the base model
fails ~30–70% of the time) or a weaker base model — both scope decisions, not silently chosen.
*(Caveat: this controllable sweep is 3 tickers × d1, n=24/design — enough to show saturation, not a
full benchmark; the 5-ticker × d1/d2 sweep is one command away.)*

## Verifiable rewards (RLVR) and related work
Every reward in this environment is **verifiable** — a programmatic check against SEC EDGAR/XBRL,
no LLM judge (`rubric/graders.py`, `controllable.py`). This is an RLVR setup, which is what makes
the trainability question well-posed: the reward is an objective function of the agent's output and
the primary source, not a model's opinion.

Our central finding sits directly alongside recent analyses of GRPO's **low-within-group-variance**
failure mode:
- **RC-GRPO** (arXiv:2602.03025) — when rollouts in a group collapse to all-0/all-1, the
  group-normalized advantage vanishes; it injects discrete reward-goal tokens so the policy emits
  varied-quality trajectories and the group regains spread.
- **Gradient Starvation in Binary-Reward GRPO** (arXiv:2605.07689) — analyzes why group-mean
  centering fails under degenerate binary rewards.

Both propose fixes for *low within-group variance*. Our **trainability law** supplies the
precondition they implicitly assume: those fixes only help when the missing variance is
**policy-controllable** (`I(A_policy;R | X_exogenous) > 0`). When the variance is exogenous
(retrieval luck) or the controllable skill is saturated, reward-token diversification cannot
manufacture a climbable gradient — there is nothing for the policy to steer toward. The
controllable-variance gate is therefore an *upstream* check: run it before reaching for RC-GRPO-style
exploration tricks, so you don't spend compute conditioning a reward whose variance the policy can't
move.

## Reproduction
```bash
# variance gate (no GPU)
python -m quant_firm.train.variance_env --model Qwen/Qwen3-8B \
    --tickers AAPL MSFT NVDA KO HD --difficulties 1 2 --group 8 --max-tokens 4096
# hosted GRPO short run
python -m quant_firm.train.train_grpo --tickers AAPL MSFT NVDA HD \
    --difficulties 1 --group 8 --iters 4 --lr 1e-5
```
Servers: specialists `:8080`, env `:8765`. Exa key in `.env`. Oracle env preserved at
`subagents_oracle.py.bak` (the Exa change is reversible). Fork: `quant-firm-rl`
(`3f4c72b9-33d3-4f8f-81f7-662ba5c830ec`), head at `step-000010` (discardable — noise-equivalent).
```

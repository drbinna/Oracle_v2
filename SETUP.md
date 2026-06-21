# quant_firm — v1 (working)

A verifiable finance-analysis RL environment for HUD. v1 task: gross-margin
analysis graded against real SEC ground truth.

## What works right now
- `data/edgar.py`   — pulls verified financials + citation accession from SEC XBRL
- `rubric/generate.py` — builds a HUD-style weighted rubric from that ground truth
- `rubric/graders.py`  — scores an agent answer (the verifiable reward, 0–1)
- `env.py`          — the HUD two-yield scenario (BashTool + thin custom tools)
- `demo.py`         — proves it end-to-end, no HUD platform needed

## Run the proof (no API key needed)
```bash
pip install requests
python -m quant_firm.demo
```

## Run as a HUD env
```bash
uv tool install hud-python --python 3.12
export HUD_API_KEY=...            # hud.ai/project/api-keys
# set your real contact email in data/edgar.py HEADERS first
hud dev env:env -w quant_firm/env.py
hud eval quant_firm/env.py claude
```

## Next steps (in order)
1. Populate `attr_driver.any_of` from each filing's MD&A (the last human/LLM-drafted bit).
2. Run the baseline across models via the HUD gateway; check the reward band.
3. Tune `difficulty` tolerances so the baseline lands in 20–50%.
4. Roll out → filter high-reward traces → Fireworks SFT → held-out eval.

## Notes
- Ground truth = EDGAR/XBRL ONLY. Exa/SixtyFour are discovery tools, never graders.
- Set a real User-Agent email in `data/edgar.py` or SEC may rate-limit you.

## Baseline runner (`baseline.py`)
Sweeps the taskset across gateway models and reports the reward band.
```bash
export HUD_API_KEY=...                      # gateway auth
# closed-book (no filing given) — the floor:
python -m quant_firm.baseline --models anthropic:claude-sonnet-4-5 openai:gpt-4o
# with the data the tools would supply — the ceiling:
python -m quant_firm.baseline --with-context --models openai:gpt-4o
# full sweep:
python -m quant_firm.baseline --runs 10 --out baseline.json
```

### What the smoke runs showed (and what it means)
- **Closed-book (no tools, no data):** models refuse or guess. Haiku correctly
  refuses to fabricate (reward 0.0); gpt-4o-mini guesses (~0.49) but can NEVER
  produce the citation accession (`cite_10k` = 0.0 for everyone).
- **Full-context (revenue/cogs/accession handed over):** reward saturates at
  0.8–0.95 — too easy, above the band. Numeric tolerance (difficulty 1 vs 3)
  barely moves it: once you have the numbers, computing margin is trivial.
- **Therefore the 20–50% band is NOT controlled by numeric tolerance — it's
  controlled by how hard the RETRIEVAL/EXTRACTION is.** The difficulty has to
  come from the agent having to find the right filing and pull the right XBRL
  line items itself (via BashTool/EDGAR/web), not from tightening tolerances.

### Band levers that actually work (ordered)
1. Make the agent locate the correct filing among many (no accession given).
2. Force correct XBRL tag disambiguation (the messy real-world tag problem).
3. Segment / product-line gross margin, or year-over-year deltas.
4. Companies with non-standard tagging (where naive extraction fails).
Numeric `difficulty` tolerance is a fine-tune on top of these, not the main knob.

> The meaningful band measurement is the TOOL-AUGMENTED rollout (agent calls
> read_filing/BashTool itself) via `hud eval` on the deployed env. The two modes
> above are the floor/ceiling brackets that prove where difficulty must come from.

## The firm: extending the agent (HUD "subagents as tools")
Per docs.hud.ai/v6/advanced/extending — the orchestrator agent is extended with
specialist sub-agents exposed as MCP tools, plus its own shell.

- `subagents.py` — three specialists on a FastMCP server:
  - `pull_financials` (research) — verified revenue/cogs/accession from EDGAR
  - `read_mdna_drivers` (research) — management's stated margin drivers
  - `audit_gross_margin` (risk auditor) — independently re-verifies any claimed
    number against the primary source. This is the anti-reward-hacking seam.
- `env.py` — v6 API: declares the `mcp` capability pointing at that server and a
  served shell (`env.workspace`), so the orchestrator can both delegate to
  specialists and run its own Python.

Run order:
```bash
export HUD_API_KEY=...
python -m quant_firm.subagents &      # serve specialists on :8080
hud dev quant_firm/env.py             # orchestrator sees pull_financials / read_mdna_drivers / audit_gross_margin
hud eval quant_firm/env.py claude
```

Other extension seams from the same doc, mapped to this project:
- **Bring-your-own-harness** (`Agent.__call__` / `OpenAIChatAgent`) — plug the
  trained Fireworks/Qwen checkpoint in for the baseline→trained comparison.
- **Group rollouts** (`taskset.run(agent, group=8)`) — GRPO groups + variance.
- **Multiple capabilities** — add `Capability.cdp` (browser) or Exa/SixtyFour
  MCP for live discovery alongside the shell.

NOTE: `env.py` was corrected from the older `@env.scenario` + `add_tool(BashTool())`
to the documented v6 surface (`@env.template`, `env.workspace`, `Capability.mcp`).

## GRPO variance gate (`variance.py`)
GRPO advantage = (reward - group_mean) / group_std. A task whose rollouts all
score the same (std=0) gives ZERO gradient — dead weight. Run this BEFORE
training to see which tasks actually carry signal.
```bash
python -m quant_firm.variance --models openai:gpt-4o-mini \
    --tickers AAPL KO WMT NVDA --difficulties 2 --group 8 --temp 0.9
```
Closed-book result (gpt-4o-mini, group 8): only 2/4 tasks had within-group
spread — AAPL (mean 0.65, sd 0.09) and KO; WMT/NVDA were degenerate-low (the
model can't recall them, so every rollout scores 0). 

Read: closed-book, only well-memorized large-caps carry gradient. The fix is the
SAME diagnostic against the tool-augmented deployed env (`taskset.run(agent,
group=8)` in env.py) — once the agent can FETCH the numbers, the degenerate-low
tasks (WMT/NVDA) gain spread and become trainable. An ideal task: mean in
0.2–0.8 AND sd >= ~0.08. That's the real selection filter for the training set.

## The climb: training pipeline (`train/`)
Rejection-sampling SFT (expert iteration) on Qwen3-8B via Fireworks.
```bash
# 1. collect graded rollouts from the model we'll train (Qwen3-8B itself)
python -m quant_firm.train.collect --model tinker:Qwen/Qwen3-8B \
    --tickers AAPL MSFT NVDA KO --difficulties 1 2 --group 6 --out traces.jsonl
# 2. keep high-reward traces -> Fireworks chat-JSONL (thinking traces preserved)
python -m quant_firm.train.build_sft --in traces.jsonl --out sft.jsonl --threshold 0.8
# 3. LoRA SFT on Fireworks (dry-run prints firectl cmds; --run executes)
python -m quant_firm.train.fireworks_sft --dataset sft.jsonl          # add --run
# 4. held-out transfer (base vs trained on UNSEEN tickers)
python -m quant_firm.train.eval_transfer --base tinker:Qwen/Qwen3-8B \
    --trained fireworks:accounts/<acct>/models/qwen3-8b-quant-firm --tickers WMT PG CVX
```
Verified-now numbers (small demo runs):
- collect: Qwen3-8B produced **7/16 traces ≥ 0.8** (expert iteration works).
- build_sft: 7 examples written in Fireworks `{"messages":[...]}` schema.
- held-out base: **0.800** on WMT/PG/CVX (the number to beat). `sft.example.jsonl`
  and `traces.example.jsonl` are included as real samples.

### Honest read on headroom (don't overclaim transfer)
With filing data in context, base Qwen3-8B already hits ~0.8 on held-out — it
computes margin and cites fine; it just misses the MD&A driver (caps at 0.8). So
SFT on with-context traces buys a small lift (0.8 -> ~1.0 via attribution). The
BIG, defensible transfer story lives in the harder regimes where base is low:
- **closed-book** (no filing handed over), and
- **tool-augmented retrieval** (agent must find the data itself).
Collect traces there (`--no-context`, or the deployed tool env) for a transfer
curve with real altitude. Measurement integrity note: Qwen3-8B is a THINKING
model — give it ~2048 tokens or it spends the budget reasoning and never emits
JSON (a token-budget 0.0 is not a real 0.0).

## Agentic trace collection (`train/collect_env.py`)
The real version of step 1: instead of the with-context proxy, run the HUD
rollout so the agent calls the env's tools (shell + specialists) to find and
extract the data itself, and the env grades the result. Same output schema as
collect.py, so build_sft.py consumes it unchanged.
```bash
export HUD_API_KEY=...
python -m quant_firm.subagents &                       # specialists on :8080
python -m quant_firm.train.collect_env --model claude-sonnet-4-5 \
    --tickers AAPL MSFT NVDA KO --difficulties 1 2 --group 6 --out traces.jsonl
python -m quant_firm.train.collect_env --dry-run       # wire-check (no rollout)
```
Mechanics (verified against the SDK): mints tasks from `env.analyze_filing`,
`Taskset(...).run(agent, group=N)`, then per run reads `run.reward` and
`run.trace.content` (the graded answer). Needs the SERVED env (subagents up +
local Docker / `hud dev`) — the dry-run builds the taskset+agent and stops there.

Why this beats the with-context proxy for training data: the high-reward traces
here are real agentic successes (the model actually retrieved + computed), so
SFT on them teaches the task, and the base→trained gap is measured on the same
hard footing where base is genuinely low.

## One-shot driver (`run_all.sh`)
Chains the pipeline with preflight guards and stops at the human gates.
```bash
./quant_firm/run_all.sh preflight     # check HUD key, env import, specialists :8080, firectl, FW key
./quant_firm/run_all.sh               # preflight -> collect -> build -> train, then STOP at the gate
./quant_firm/run_all.sh eval          # base-only held-out number (the bar to beat)
TRAINED_MODEL=fireworks:accounts/<acct>/models/qwen3-8b-quant-firm \
    ./quant_firm/run_all.sh eval      # base vs trained delta after deploy
```
Config via env vars (MODEL, TICKERS, HELD_OUT, DIFFICULTIES, GROUP, THRESHOLD,
OUTPUT_MODEL, TRAINED_MODEL). Each stage guards its preconditions and fails fast
with the fix (e.g. "specialists NOT on :8080 — start: python -m quant_firm.subagents &").
The train stage submits the Fireworks job then prints the poll/deploy/eval gate —
training is async and deploy is manual, so it hands control back rather than
pretending to block through it.

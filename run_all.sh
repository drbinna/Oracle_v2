#!/usr/bin/env bash
# run_all.sh — one-shot driver for the quant_firm training pipeline.
#
#   ./quant_firm/run_all.sh                 # preflight -> collect -> build -> train, then STOP at the gate
#   ./quant_firm/run_all.sh preflight       # just check preconditions
#   ./quant_firm/run_all.sh collect|build|train|eval
#   TRAINED_MODEL=fireworks:accounts/<acct>/models/qwen3-8b-quant-firm \
#       ./quant_firm/run_all.sh eval        # held-out transfer once the checkpoint is deployed
#
# Config (override via env):
set -uo pipefail

MODEL="${MODEL:-claude-sonnet-4-5}"                 # agent that generates traces
OUTPUT_MODEL="${OUTPUT_MODEL:-qwen3-8b-quant-firm}"
TICKERS="${TICKERS:-AAPL MSFT NVDA KO}"             # training tickers
HELD_OUT="${HELD_OUT:-WMT PG CVX}"                  # eval tickers (NOT trained on)
DIFFICULTIES="${DIFFICULTIES:-1 2}"
GROUP="${GROUP:-6}"
THRESHOLD="${THRESHOLD:-0.8}"
TRACES="${TRACES:-traces.jsonl}"
SFT="${SFT:-sft.jsonl}"
TRAINED_MODEL="${TRAINED_MODEL:-}"                  # set after deploy to get the delta

# --- resolve to the project root (parent of the quant_firm package) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/quant_firm" ]; then ROOT="$HERE"; else ROOT="$(dirname "$HERE")"; fi
cd "$ROOT"
[ -f .env ] && set -a && . ./.env && set +a
[ -f quant_firm/.env ] && set -a && . ./quant_firm/.env && set +a

green(){ printf "\033[32m%s\033[0m\n" "$*"; }
red(){ printf "\033[31m%s\033[0m\n" "$*"; }
banner(){ echo; green "=== $* ==="; }
die(){ red "FAIL: $*"; exit 1; }

port_up(){ python3 - "$1" <<'PY' 2>/dev/null
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)
PY
}

preflight(){
  banner "preflight"
  [ -n "${HUD_API_KEY:-}" ] && green "HUD_API_KEY set" || die "HUD_API_KEY not set (export it or put it in .env)"
  python3 -c "import quant_firm.env" 2>/dev/null && green "quant_firm.env imports" || die "cannot import quant_firm.env (run from project root, deps installed)"
  if port_up 8080; then green "specialists server up on :8080"; else red "specialists NOT on :8080 — start: python -m quant_firm.subagents &"; fi
  command -v firectl >/dev/null && green "firectl present" || red "firectl missing (needed only for the train stage)"
  [ -n "${FIREWORKS_API_KEY:-}" ] && green "FIREWORKS_API_KEY set" || red "FIREWORKS_API_KEY not set (needed only for train stage)"
}

stage_collect(){
  banner "collect (agentic rollouts -> $TRACES)"
  port_up 8080 || die "specialists not on :8080 — start: python -m quant_firm.subagents &"
  python -m quant_firm.train.collect_env --model "$MODEL" \
    --tickers $TICKERS --difficulties $DIFFICULTIES --group "$GROUP" \
    --keep-threshold "$THRESHOLD" --out "$TRACES" || die "collect failed"
  [ -s "$TRACES" ] || die "no traces written"
}

stage_build(){
  banner "build SFT dataset ($TRACES -> $SFT)"
  [ -s "$TRACES" ] || die "no $TRACES — run the collect stage first"
  python -m quant_firm.train.build_sft --in "$TRACES" --out "$SFT" --threshold "$THRESHOLD" || die "build failed"
  [ -s "$SFT" ] || die "no SFT examples cleared threshold $THRESHOLD — collect more / lower --threshold"
}

stage_train(){
  banner "Fireworks LoRA SFT"
  [ -s "$SFT" ] || die "no $SFT — run the build stage first"
  command -v firectl >/dev/null || die "firectl missing — install + 'firectl login'"
  [ -n "${FIREWORKS_API_KEY:-}" ] || die "FIREWORKS_API_KEY not set"
  python -m quant_firm.train.fireworks_sft --dataset "$SFT" --output-model "$OUTPUT_MODEL" --run || die "train submit failed"
  banner "HUMAN GATE"
  cat <<EOF
  Training is async. Now:
    1) poll:   firectl list sftj
    2) deploy: firectl create deployment accounts/fireworks/models/$OUTPUT_MODEL
    3) eval:   TRAINED_MODEL=fireworks:accounts/<acct>/models/$OUTPUT_MODEL ./quant_firm/run_all.sh eval
EOF
}

stage_eval(){
  banner "held-out transfer (tickers: $HELD_OUT)"
  if [ -n "$TRAINED_MODEL" ]; then
    python -m quant_firm.train.eval_transfer --base "tinker:Qwen/Qwen3-8B" \
      --trained "$TRAINED_MODEL" --tickers $HELD_OUT --difficulties $DIFFICULTIES || die "eval failed"
  else
    red "TRAINED_MODEL not set — running base-only (the number to beat)"
    python -m quant_firm.train.eval_transfer --base "tinker:Qwen/Qwen3-8B" \
      --tickers $HELD_OUT --difficulties $DIFFICULTIES || die "eval failed"
  fi
}

case "${1:-all}" in
  preflight) preflight ;;
  collect)   preflight; stage_collect ;;
  build)     stage_build ;;
  train)     stage_train ;;
  eval)      stage_eval ;;
  all)       preflight; stage_collect; stage_build; stage_train ;;
  *) die "unknown stage '${1}' (preflight|collect|build|train|eval|all)" ;;
esac
green "stage '${1:-all}' done."

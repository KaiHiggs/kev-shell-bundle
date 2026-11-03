#!/usr/bin/env bash
# 0.8B LoRA at the full 2,048-token context. Logs to runs/train.log; checkpoint in runs/kev-shell-0.8b.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
UP="$HERE/upstream"
[ -x "$UP/.venv/bin/python" ] || { echo "ERROR: no upstream environment at $UP. Run scripts/bootstrap.sh first." >&2; exit 1; }
DATA=${1:-$HERE/data/train.jsonl}
EPOCHS=${EPOCHS:-2}
ACCUM=${ACCUM:-8}
export EPOCHS ACCUM
OUT=${OUT:-$HERE/runs/kev-shell-0.8b}
mkdir -p "$HERE/runs"
# The trainer refuses an existing --out. A directory without head.pt is a stub from an aborted attempt: move it aside.
if [ -e "$OUT" ]; then
  if [ -f "$OUT/head.pt" ]; then
    echo "ERROR: $OUT already holds a finished run. Move it, or choose another output: OUT=$OUT.$(date +%s) bash scripts/train.sh" >&2
    exit 1
  fi
  echo "moving the aborted run aside: $OUT -> $OUT.stale-$(date +%s)"
  mv "$OUT" "$OUT.stale-$(date +%s)"
fi
cd "$UP"
export PYTHONPATH="$UP" HF_HUB_DISABLE_TELEMETRY=1
# Fragmented 16 GB card: fewer failed allocations at the ceiling.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# Memory, on a 16 GB card: --dtype bf16 (the default fp32 keeps every activation fp32 even with bf16 weights),
# --shared_prefix 1 (default off for LoRA: the 2k state is otherwise re-run and re-stored per question row),
# and --checkpointing 1 (activation recompute, ~30% slower) which is what makes the 2,048-token rows fit at all
# (measured peak 2.8 GiB with it, >15 GiB and OOM without). CHECKPOINTING=0 is the fast path once it fits.
TRAIN=(.venv/bin/python -m kev.train
  --data "$DATA"
  --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
  --device cuda --weights_dtype bf16 --dtype bf16 --shared_prefix 1 --checkpointing "${CHECKPOINTING:-1}"
  --row_budget "${ROW_BUDGET:-0}"
  --max_state 2048 --lora 16 --batch 1 --accum "$ACCUM" --epochs "$EPOCHS" --lr 2e-4
  --out "$OUT")

if [ -t 1 ] && [ "${PROGRESS:-1}" = 1 ]; then
  # Keep every step line in the log, show only the bar and the non-step output on the terminal.
  bash "$HERE/scripts/progress.sh" & PROGRESS_PID=$!
  trap 'kill "$PROGRESS_PID" 2>/dev/null || true' EXIT
  "${TRAIN[@]}" 2>&1 | tee "$HERE/runs/train.log" | grep --line-buffered -vE '^ep[0-9]+ step '
else
  "${TRAIN[@]}" 2>&1 | tee "$HERE/runs/train.log"
fi

#!/usr/bin/env bash
# 0.8B LoRA at the full 2,048-token context. Logs to runs/train.log; checkpoint in runs/kev-shell-0.8b.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
UP="$HERE/upstream"
[ -x "$UP/.venv/bin/python" ] || { echo "ERROR: no upstream environment at $UP. Run scripts/bootstrap.sh first." >&2; exit 1; }
DATA=${1:-$HERE/data/train.jsonl}
EPOCHS=${EPOCHS:-2}
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
# ROCm: bf16 weights halve memory. If you hit a bf16 kernel crash, append --weights_dtype fp32 --checkpointing 1.
exec .venv/bin/python -m kev.train \
  --data "$DATA" \
  --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --device cuda --weights_dtype bf16 \
  --max_state 2048 --lora 16 --batch 1 --accum 8 --epochs "$EPOCHS" --lr 2e-4 \
  --out "$OUT" 2>&1 | tee "$HERE/runs/train.log"

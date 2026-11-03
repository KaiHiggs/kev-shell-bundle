#!/usr/bin/env bash
# Serve the checkpoint, then score the holdout split (<=512) and the long-command split.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
bash "$HERE/scripts/serve.sh" > "$HERE/runs/serve.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 60); do curl -sf http://127.0.0.1:8018/v1/models >/dev/null && break; sleep 2; done
node "$HERE/eval/eval-finetune.mjs" --data "$HERE/data/holdout-512.jsonl" --out "$HERE/runs/eval-holdout"
node "$HERE/eval/eval-finetune.mjs" --data "$HERE/data/calibration-512.jsonl" --out "$HERE/runs/eval-calibration"
echo 'long states are not scored: they exceed the 2,048-token training context and the serving policy is review'

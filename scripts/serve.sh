#!/usr/bin/env bash
# Serve the fine-tuned checkpoint on loopback for evaluation.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-$HERE/runs/kev-shell-0.8b}
[ -d "$RUN" ] || { echo "ERROR: no checkpoint at $RUN. Run scripts/train.sh first." >&2; exit 1; }
cd "$HERE/upstream"
export KEV_TRUNCATE_STATES=0 HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/python -m kev.serve --run "$RUN" --host 127.0.0.1 --port 8018

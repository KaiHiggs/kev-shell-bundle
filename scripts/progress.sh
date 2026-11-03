#!/usr/bin/env bash
# One-line progress bar for a running kev.train, read from its log. Never touches the training process.
#   bash scripts/progress.sh            # live, redraws in place; Ctrl-C stops only the bar
#   bash scripts/progress.sh --once     # print one frame and exit
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-$HERE/../runs/train.log}
EPOCHS=${EPOCHS:-2}
ACCUM=${ACCUM:-8}
ONCE=0; [ "${1:-}" = "--once" ] && ONCE=1
WIDTH=28

frame() {
  local line ep step steps loss srec done total pct filled empty bar eta
  line=$(grep -E '^ep[0-9]+ step [0-9]+/[0-9]+ ' "$LOG" 2>/dev/null | tail -1)
  if grep -q '^saved ' "$LOG" 2>/dev/null; then
    printf '\r[%-*s] 100%%  saved\n' "$WIDTH" "$(printf '%*s' $WIDTH '' | tr ' ' '#')"
    return 1
  fi
  if [ -z "$line" ]; then
    if grep -qE 'Traceback|CUDA out of memory|HIP error|RuntimeError' "$LOG" 2>/dev/null; then
      printf '\r%-70s\n' "training failed — see the traceback in $LOG"
      return 1
    fi
    printf '\r%s waiting for the first step (weights load, then one compile)…' "$(date +%H:%M:%S)"
    return 0
  fi
  read -r ep step steps loss srec <<<"$(awk '{
      ep=$1; sub(/^ep/,"",ep);
      split($3,a,"/"); step=a[1]; steps=a[2];
      srec="?";
      for(i=1;i<=NF;i++){ if($i ~ /s\/rec$/){ srec=$i; sub(/s\/rec$/,"",srec) } }
      printf "%s %s %s %s %s\n", ep, step, steps, $5, srec
    }' <<<"$line")"
  total=$((EPOCHS * steps))
  done=$((ep * steps + step))
  [ "$total" -le 0 ] && total=1
  pct=$((done * 100 / total)); [ "$pct" -gt 100 ] && pct=100
  filled=$((pct * WIDTH / 100)); empty=$((WIDTH - filled))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '')
  eta=$(awk -v r=$((total - done)) -v a="$ACCUM" -v s="$srec" 'BEGIN{v=r*a*s; if(v<=0){print "—"} else if(v<60){printf "%.0fs",v} else {printf "%.0fm",v/60}}')
  printf '\r[%s] %3d%%  ep%s %s/%s  loss %s  %ss/rec  eta %s   ' \
    "$bar" "$pct" "$ep" "$step" "$steps" "$loss" "$srec" "$eta"
}

if [ "$ONCE" = 1 ]; then frame; echo; exit 0; fi
trap 'printf "\n"; exit 0' INT TERM
while :; do
  frame || break
  sleep 3
done

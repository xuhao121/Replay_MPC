#!/usr/bin/env bash
# Start worker lanes for one owner in a tmux session, one window per lane.
#   bash scripts/launch_tmux.sh m1 "90 38 13 13 13"                               # 5 lanes, budgets in GB
#   bash scripts/launch_tmux.sh m1 "90 38 13 13 13" "--deadline 2026-11-08T20:00"  # extra flags for every lane
# Each number is one lane: it only takes jobs whose estimated memory fits its budget, and within a priority it
# first takes the jobs that the next smaller lane cannot run (90 -> jobs above 38 GB, 38 -> above 13 GB), so
# the big jobs start early. Keep the largest job each lane can take summed below ~170 GB on a B200
# (90 + 38 + 3 x 13 = 167). --deadline: no job is started that would not finish before it (worker.py).
# Logs: results/_logs/<owner>_lane<k>.log. Attach: tmux attach -t rmpc-<owner>. Stop: scripts/stop_workers.sh
set -euo pipefail
OWNER=${1:?owner (m1|m2)}
BUDGETS=${2:-"90 38 13 13 13"}
EXTRA_ALL=${3:-}
cd "$(dirname "$0")/.."
mkdir -p results/_logs
rm -f STOP
S="rmpc-${OWNER}"
tmux has-session -t "$S" 2>/dev/null && { echo "session $S exists (tmux attach -t $S)"; exit 1; }
next_smaller() {          # largest budget strictly below $1 (0 if there is none)
  local m=${1%.*} best=0 b
  for b in $BUDGETS; do
    b=${b%.*}
    if [ "$b" -lt "$m" ] && [ "$b" -gt "$best" ]; then best=$b; fi
  done
  echo "$best"
}
k=0
for MEM in $BUDGETS; do
  P=$(next_smaller "$MEM")
  EXTRA=""
  if [ "$P" -gt 0 ]; then EXTRA="--prefer-mem-above $P"; fi
  CMD="python scripts/worker.py --owner $OWNER --lane $k --mem-gb $MEM $EXTRA $EXTRA_ALL 2>&1 | tee -a results/_logs/${OWNER}_lane${k}.log"
  if [ "$k" -eq 0 ]; then tmux new-session -d -s "$S" -n lane0; else tmux new-window -t "$S" -n "lane$k"; fi
  tmux send-keys -t "$S:lane$k" "$CMD" C-m
  k=$((k + 1))
  sleep 2
done
echo "started $k lanes in tmux session $S with memory budgets: $BUDGETS ${EXTRA_ALL:+(extra: $EXTRA_ALL)}"

#!/usr/bin/env bash
# One owner (m1/m2), several independent GPUs, with the original per-GPU lane budgets.
#   bash scripts/launch_tmux.sh m1 "90 38 13 13 13" "--deadline 2026-11-08T20:00" "0 1 2 3"
# The 4th argument defaults to "0 1 2 3"; use "0" to reproduce the original single-GPU launch.
# Each worker sees only its assigned physical GPU as cuda:0 (via CUDA_VISIBLE_DEVICES).
# Each GPU keeps its own 90/38/13/13/13 GB lanes. Total per GPU = 167 GB, not 668 GB.
# Logs: results/_logs/<owner>_gpu<id>_lane<k>.log; monitor: tmux attach -t rmpc-<owner>.
set -euo pipefail
OWNER=${1:?owner (m1|m2)}
BUDGETS=${2:-"90 38 13 13 13"}
EXTRA_ALL=${3:-}
GPUS=${4:-"0 1 2 3"}
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
for GPU in $GPUS; do
  local_k=0
  for MEM in $BUDGETS; do
    P=$(next_smaller "$MEM")
    EXTRA=""
    if [ "$P" -gt 0 ]; then EXTRA="--prefer-mem-above $P"; fi
    WIN="gpu${GPU}_lane${local_k}"
    # Limit CPU oversubscription: there are 20 Python processes with four GPUs.
    CMD="CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -u scripts/worker.py --owner $OWNER --lane $k --mem-gb $MEM $EXTRA $EXTRA_ALL 2>&1 | tee -a results/_logs/${OWNER}_${WIN}.log"
    # Execute the worker directly instead of sending keys before shell startup.
    RUN="bash -lc $(printf '%q' "set -o pipefail; $CMD; echo WORKER_EXIT_STATUS=\$?; exec bash")"
    if [ "$k" -eq 0 ]; then
      tmux new-session -d -s "$S" -n "$WIN" "$RUN"
    else
      tmux new-window -d -t "$S:" -n "$WIN" "$RUN"
    fi
    k=$((k + 1))
    local_k=$((local_k + 1))
    sleep 1
  done
done
echo "started $k lanes for $OWNER on GPUs: $GPUS (per-GPU budgets: $BUDGETS GB)"
echo "tmux attach -t $S    # stop gracefully: bash scripts/stop_workers.sh"

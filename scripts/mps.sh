#!/usr/bin/env bash
# CUDA MPS lets several worker processes run kernels on one GPU at the same time (assignment 2: 4.4x
# throughput with 8 processes). Pipe and log directories go to $HOME because /var/log/nvidia-mps is usually
# not writable ("Failed to create log directory" otherwise).
#   bash scripts/mps.sh start      # before launch_tmux.sh, in the same shell environment
#   bash scripts/mps.sh stop
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.mps/pipe}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-$HOME/.mps/log}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
case "${1:-start}" in
  start) nvidia-cuda-mps-control -d && echo "MPS started (pipe $CUDA_MPS_PIPE_DIRECTORY)";
         echo "export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY  # add to ~/.bashrc" ;;
  stop)  echo quit | nvidia-cuda-mps-control && echo "MPS stopped" ;;
  *) echo "usage: $0 start|stop"; exit 1 ;;
esac

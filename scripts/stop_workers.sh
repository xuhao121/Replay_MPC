#!/usr/bin/env bash
# Graceful stop: every lane finishes its current job and exits. Remove STOP (or relaunch) to continue.
cd "$(dirname "$0")/.."
touch STOP
echo "STOP written; lanes exit after their current job. Check with: tmux ls / python scripts/status.py"

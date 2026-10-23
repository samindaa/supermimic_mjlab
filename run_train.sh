#!/bin/bash
# Launch SuperMimic training in a tmux session, in the uv environment of this repository.
# Usage: [CPUS=0-23] ./run_train.sh <run_name> [train.py args...]
# CPUS optionally pins the run with taskset.
cd "$(dirname "$0")"
RUN=${1:-lafan1_main}
shift || true
PIN=""
[ -n "${CPUS:-}" ] && PIN="taskset -c $CPUS"
mkdir -p "logs/$RUN"
tmux new-session -d -s "$RUN" \
  "TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor $PIN uv run python supermimic/train.py --run_dir logs/$RUN $* 2>&1 | tee -a logs/$RUN/stdout.log"
echo "started tmux session '$RUN' (attach: tmux attach -t $RUN)"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${1:-full_dynamic_bg_$(date +%Y%m%d_%H%M%S)}"
shift || true

CMD=(
  conda run -n chan python Debug/run_pipeline.py
  --mode full
  --run-id "$RUN_ID"
  --kill-existing-runs-before-start
  --cache-mode resume
  --begin-time 2020-01-01
  --end-time 2026-02-28
  --train-mode gpu
  --num-workers 4
  --feature-symbol-workers 4
  --symbol-workers 8
  --backtest-parallel-mode process
  --backtest-data-cache-size 8
)

# Allow appending extra args from caller.
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

mkdir -p Debug/runs
LAUNCH_LOG="Debug/runs/${RUN_ID}_launcher.log"

nohup "${CMD[@]}" > "$LAUNCH_LOG" 2>&1 &
PID=$!

echo "run_id=$RUN_ID"
echo "pid=$PID"
echo "launcher_log=$LAUNCH_LOG"
echo "train_log=Debug/runs/${RUN_ID}/logs/train.log"
echo "backtest_log=Debug/runs/${RUN_ID}/logs/backtest.log"

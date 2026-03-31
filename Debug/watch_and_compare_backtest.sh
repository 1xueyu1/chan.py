#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <target_pid> <old_run_id> <new_run_id> [artifact_kind]" >&2
  exit 2
fi

TARGET_PID="$1"
OLD_RUN_ID="$2"
NEW_RUN_ID="$3"
ARTIFACT_KIND="${4:-run}"

while ps -p "$TARGET_PID" >/dev/null 2>&1; do
  sleep 60
done

/root/anaconda3/envs/chan/bin/python Debug/compare_backtest_runs.py \
  --old-run-id "$OLD_RUN_ID" \
  --new-run-id "$NEW_RUN_ID" \
  --artifact-kind "$ARTIFACT_KIND"

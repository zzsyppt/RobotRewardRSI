#!/usr/bin/env bash
set -euo pipefail

# Launch the Phase 6 live loop as a background server job.
# This wrapper keeps secrets in the server environment and writes only logs/PID.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/runs/phase_6_multi_round_loop}"
RUN_ID="${1:-${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S_phase6_live)}}"
PROVIDER="${PROVIDER:-deepseek}"
ROUNDS="${ROUNDS:-3}"
SECRET_ENV="${SECRET_ENV:-$DATA_ROOT/secrets/deepseek.env}"

mkdir -p "$RUN_ROOT"
cd "$PROJECT_ROOT"

if [ "$PROVIDER" = "deepseek" ]; then
  if [ ! -f "$SECRET_ENV" ]; then
    echo "Missing secret env file: $SECRET_ENV" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$SECRET_ENV"
  if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "DEEPSEEK_API_KEY is empty after sourcing $SECRET_ENV" >&2
    exit 2
  fi
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

LAUNCH_LOG="$RUN_ROOT/${RUN_ID}.launch.log"
PID_FILE="$RUN_ROOT/${RUN_ID}.pid"

if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE" || true)"
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Phase 6 run already appears active: run_id=$RUN_ID pid=$existing_pid"
    echo "launch_log=$LAUNCH_LOG"
    exit 0
  fi
fi

nohup python3 scripts/phase6_multi_round_loop.py \
  --execute \
  --provider "$PROVIDER" \
  --allow-api-spend \
  --rounds "$ROUNDS" \
  --run-id "$RUN_ID" \
  > "$LAUNCH_LOG" 2>&1 &

pid="$!"
printf '%s\n' "$pid" > "$PID_FILE"

echo "PHASE6_RUN_ID=$RUN_ID"
echo "PHASE6_PID=$pid"
echo "PHASE6_LAUNCH_LOG=$LAUNCH_LOG"

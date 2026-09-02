#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-$DATA_ROOT/runs/phase_-1}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$DATA_ROOT/vendor/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-512}"

mkdir -p "$RUN_DIR"

ISAACLAB_ROOT="$ISAACLAB_ROOT" DEVICE=cuda:0 NUM_ENVS="$NUM_ENVS" MAX_ITERATIONS=1 \
  "$PROJECT_ROOT/scripts/smoke_official_rl.sh" \
  > "$RUN_DIR/smoke_dual_gpu_0.log" 2>&1 &
pid_gpu_0=$!

# Avoid identical timestamp-based experiment directories while keeping both
# jobs overlapped for almost their full startup and training periods.
sleep 2

ISAACLAB_ROOT="$ISAACLAB_ROOT" DEVICE=cuda:1 NUM_ENVS="$NUM_ENVS" MAX_ITERATIONS=1 \
  "$PROJECT_ROOT/scripts/smoke_official_rl.sh" \
  > "$RUN_DIR/smoke_dual_gpu_1.log" 2>&1 &
pid_gpu_1=$!

set +e
wait "$pid_gpu_0"
rc_gpu_0=$?
wait "$pid_gpu_1"
rc_gpu_1=$?
set -e

printf 'gpu0_rc=%s gpu1_rc=%s\n' "$rc_gpu_0" "$rc_gpu_1"

if ((rc_gpu_0 != 0 || rc_gpu_1 != 0)); then
  exit 1
fi

grep -q 'Environment device    : cuda:0' "$RUN_DIR/smoke_dual_gpu_0.log"
grep -q 'Learning iteration 0/1' "$RUN_DIR/smoke_dual_gpu_0.log"
grep -q 'Environment device    : cuda:1' "$RUN_DIR/smoke_dual_gpu_1.log"
grep -q 'Learning iteration 0/1' "$RUN_DIR/smoke_dual_gpu_1.log"

printf 'DUAL_GPU_SMOKE_OK\n'

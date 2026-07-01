#!/usr/bin/env bash
set -euo pipefail

# Phase 3 verifier smoke:
# 1. run pure static self-tests for legal/illegal synthetic patches;
# 2. run a short Isaac random-action rollout and validate reset/finite contracts.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/runs/phase_3_verifier_smoke}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"

mkdir -p "$RUN_DIR"

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/cache/conda}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$DATA_ROOT/vendor/unitree_ros}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC1091
source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

printf 'PHASE3_RUN_DIR=%s\n' "$RUN_DIR"

python "$PROJECT_ROOT/scripts/phase3_verifier_selftest.py" \
  > "$RUN_DIR/selftest_summary.json"

python "$PROJECT_ROOT/scripts/phase3_verify_candidate.py" \
  --repo-root "$PROJECT_ROOT" \
  --mode smoke \
  --run-dir "$RUN_DIR/smoke" \
  --summary-file "$RUN_DIR/verifier_smoke_summary.json"

printf '\nPhase 3 verifier smoke complete: %s\n' "$RUN_DIR"

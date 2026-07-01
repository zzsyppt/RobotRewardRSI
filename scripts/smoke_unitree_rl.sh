#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
UNITREE_ROOT="${UNITREE_ROOT:-$DATA_ROOT/vendor/unitree_rl_lab}"
TASK="${TASK:-Unitree-G1-29dof-Velocity}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
DEVICE="${DEVICE:-cuda:0}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

cd "$UNITREE_ROOT"
python scripts/rsl_rl/train.py \
  --task "$TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS"

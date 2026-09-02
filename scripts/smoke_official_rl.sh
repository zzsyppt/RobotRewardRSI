#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
TASK="${TASK:-Isaac-Ant-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

cd "$ISAACLAB_ROOT"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS"

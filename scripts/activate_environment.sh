#!/usr/bin/env bash

# Source this file; do not execute it in a child shell:
#   source /data/yuyangliu/AGENT/scripts/activate_environment.sh

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/cache/conda}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$DATA_ROOT/vendor/unitree_ros}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

printf 'Activated %s with %s\n' "$CONDA_DEFAULT_ENV" "$(python -V 2>&1)"

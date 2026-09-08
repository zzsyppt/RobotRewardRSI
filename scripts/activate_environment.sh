#!/usr/bin/env bash

# Source this file; do not execute it in a child shell:
#   source /data/yuyangliu/AGENT/scripts/activate_environment.sh

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
ENV_PREFIX="$DATA_ROOT/envs/isaaclab6_validation"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$DATA_ROOT/vendor/unitree_ros}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$ENV_PREFIX/bin/activate"

printf 'Activated %s with %s\n' "$VIRTUAL_ENV" "$(python -V 2>&1)"

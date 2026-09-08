#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/data/yuyangliu/envs/isaaclab6_validation}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-/data/yuyangliu/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/yuyangliu/cache/xdg}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON_BIN" -c '
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, enable_cameras=False)
app = launcher.app
for _ in range(10):
    app.update()
print("ISAAC_HEADLESS_OK", flush=True)
app.close()
'

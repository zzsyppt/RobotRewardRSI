#!/usr/bin/env bash
set -euo pipefail

# Reproducible user-space bootstrap for the validated Ubuntu 22.04 server.
# No sudo is required. Large environments, caches, sources, and logs stay under
# DATA_ROOT instead of /home.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOCK_ROOT="$PROJECT_ROOT/setup/locks"

# shellcheck disable=SC1091
source "$LOCK_ROOT/source_revisions.env"

TOOLS_ROOT="${TOOLS_ROOT:-$DATA_ROOT/tools}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$TOOLS_ROOT/miniconda3}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$DATA_ROOT/vendor/IsaacLab}"
UNITREE_ROOT="${UNITREE_ROOT:-$DATA_ROOT/vendor/unitree_rl_lab}"
UNITREE_ROS_ROOT="${UNITREE_ROS_ROOT:-$DATA_ROOT/vendor/unitree_ros}"
RUN_DIR="${RUN_DIR:-$DATA_ROOT/runs/phase_-1}"
SYSTEM_LIB_ROOT="${SYSTEM_LIB_ROOT:-$DATA_ROOT/vendor/system-libs/root}"
INSTALLER_ROOT="${INSTALLER_ROOT:-$DATA_ROOT/installers}"

PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
NVIDIA_INDEX_URL="${NVIDIA_INDEX_URL:-https://pypi.nvidia.com}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
MINICONDA_SHA256="${MINICONDA_SHA256:-2284bafb7863a23411b19874d216e237964d4b32dd9beb6807fa8b2d84570961}"
MINICONDA_INSTALLER="$INSTALLER_ROOT/Miniconda3-latest-Linux-x86_64.sh"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/cache/conda}"
export PIP_INDEX_URL="$PYPI_INDEX_URL"

mkdir -p \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$CONDA_PKGS_DIRS" \
  "$RUN_DIR" \
  "$SYSTEM_LIB_ROOT" \
  "$INSTALLER_ROOT" \
  "$TOOLS_ROOT" \
  "$DATA_ROOT/envs" \
  "$DATA_ROOT/vendor"

for command_name in curl git patch sha256sum apt-get dpkg-deb; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'ERROR: required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done

install_miniconda() {
  if [ -x "$MINICONDA_ROOT/bin/conda" ]; then
    return
  fi

  if [ ! -f "$MINICONDA_INSTALLER" ]; then
    curl --fail --location --retry 5 --retry-delay 5 \
      "$MINICONDA_URL" --output "$MINICONDA_INSTALLER"
  fi

  printf '%s  %s\n' "$MINICONDA_SHA256" "$MINICONDA_INSTALLER" | sha256sum --check
  bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_ROOT"
}

clone_source_at_commit() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"
  local tmp_destination
  local attempt

  if [ -f "$destination/.source_commit" ] && \
    [ "$(cat "$destination/.source_commit")" = "$commit" ]; then
    return
  fi

  if [ -d "$destination/.git" ] && \
    [ "$(git -C "$destination" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    printf '%s\n' "$commit" > "$destination/.source_commit"
    return
  fi

  if [ -e "$destination" ] && [ -n "$(find "$destination" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    printf 'ERROR: %s exists but is not pinned to %s\n' "$destination" "$commit" >&2
    exit 1
  fi

  rmdir "$destination" 2>/dev/null || true
  tmp_destination="${destination}.tmp.$$"
  for attempt in 1 2 3; do
    rm -rf "$tmp_destination"
    if git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout "$repository_url" "$tmp_destination" && \
      git -C "$tmp_destination" -c http.version=HTTP/1.1 fetch --depth 1 origin "$commit" && \
      git -C "$tmp_destination" checkout --detach "$commit"; then
      printf '%s\n' "$commit" > "$tmp_destination/.source_commit"
      mv "$tmp_destination" "$destination"
      return
    fi
    printf 'WARN: clone attempt %s failed for %s\n' "$attempt" "$repository_url" >&2
    sleep 10
  done
  rm -rf "$tmp_destination"
  printf 'ERROR: failed to clone %s at %s\n' "$repository_url" "$commit" >&2
  exit 1
}

clone_sparse_source_at_commit() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"
  local sparse_path="$4"
  local tmp_destination
  local attempt

  if [ -f "$destination/.source_commit" ] && \
    [ "$(cat "$destination/.source_commit")" = "$commit" ]; then
    return
  fi

  if [ -d "$destination/.git" ] && \
    [ "$(git -C "$destination" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    printf '%s\n' "$commit" > "$destination/.source_commit"
    return
  fi

  if [ -e "$destination" ] && [ -n "$(find "$destination" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    printf 'ERROR: %s exists but is not pinned to %s\n' "$destination" "$commit" >&2
    exit 1
  fi

  rmdir "$destination" 2>/dev/null || true
  tmp_destination="${destination}.tmp.$$"
  for attempt in 1 2 3; do
    rm -rf "$tmp_destination"
    if git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout "$repository_url" "$tmp_destination" && \
      git -C "$tmp_destination" sparse-checkout init --cone && \
      git -C "$tmp_destination" sparse-checkout set "$sparse_path" && \
      git -C "$tmp_destination" -c http.version=HTTP/1.1 fetch --depth 1 origin "$commit" && \
      git -C "$tmp_destination" checkout --detach "$commit"; then
      printf '%s\n' "$commit" > "$tmp_destination/.source_commit"
      mv "$tmp_destination" "$destination"
      return
    fi
    printf 'WARN: sparse clone attempt %s failed for %s\n' "$attempt" "$repository_url" >&2
    sleep 10
  done
  rm -rf "$tmp_destination"
  printf 'ERROR: failed to sparse-clone %s at %s\n' "$repository_url" "$commit" >&2
  exit 1
}

install_miniconda

# shellcheck disable=SC1091
source "$MINICONDA_ROOT/etc/profile.d/conda.sh"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$ENV_PREFIX" \
    -c conda-forge \
    --override-channels \
    "python=$PYTHON_VERSION" \
    pip
fi
conda activate "$ENV_PREFIX"

conda install -y -c conda-forge --override-channels \
  "cmake=4.3.4" \
  "ninja=1.13.2"

clone_source_at_commit \
  https://github.com/isaac-sim/IsaacLab.git \
  "$ISAACLAB_ROOT" \
  "$ISAACLAB_COMMIT"
clone_source_at_commit \
  https://github.com/unitreerobotics/unitree_rl_lab.git \
  "$UNITREE_ROOT" \
  "$UNITREE_RL_LAB_COMMIT"

clone_sparse_source_at_commit \
  https://github.com/unitreerobotics/unitree_ros.git \
  "$UNITREE_ROS_ROOT" \
  "$UNITREE_ROS_COMMIT" \
  robots/g1_description

if [ ! -e "$SYSTEM_LIB_ROOT/usr/lib/x86_64-linux-gnu/libGLU.so.1" ]; then
  mkdir -p "$DATA_ROOT/vendor/system-libs/debs"
  (
    cd "$DATA_ROOT/vendor/system-libs/debs"
    apt-get download libglu1-mesa=9.0.2-1
    for package in ./*.deb; do
      dpkg-deb -x "$package" "$SYSTEM_LIB_ROOT"
    done
  )
fi

export LD_LIBRARY_PATH="$SYSTEM_LIB_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -V
python -m pip --version

python -m pip install \
  "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
  --index-url "$PYPI_INDEX_URL" \
  --extra-index-url "$NVIDIA_INDEX_URL"

python -m pip install \
  --upgrade \
  --force-reinstall \
  "torch==$PYTORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$PYTORCH_INDEX_URL"

python -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  "setuptools==80.9.0" \
  "wheel==0.45.1"
python -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  --no-build-isolation \
  "flatdict==4.0.1"

(
  cd "$ISAACLAB_ROOT"
  ./isaaclab.sh --install rsl_rl
)

python -m pip install -e "$UNITREE_ROOT/source/unitree_rl_lab"
python -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  --requirement "$LOCK_ROOT/direct_requirements.txt"

UNITREE_PATCH="$PROJECT_ROOT/setup/unitree_g1_urdf.patch"
if patch -d "$UNITREE_ROOT" -p1 -R --dry-run < "$UNITREE_PATCH" >/dev/null 2>&1; then
  printf 'Unitree G1 URDF patch already applied.\n'
else
  patch -d "$UNITREE_ROOT" -p1 --forward < "$UNITREE_PATCH"
fi

python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
conda list --explicit > "$RUN_DIR/conda_explicit_linux-64.txt"
python -m pip check | tee "$RUN_DIR/pip_check.log"

printf '\nPHASE_MINUS_1_INSTALL_OK\n'
printf 'Before the first Isaac Sim launch, review the NVIDIA EULA.\n'
printf 'If you agree, launch interactively and answer Yes, or explicitly export ACCEPT_EULA=Y.\n'
printf 'Then run the smoke scripts for CUDA, Isaac headless, official PPO, Unitree PPO, and dual GPU.\n'

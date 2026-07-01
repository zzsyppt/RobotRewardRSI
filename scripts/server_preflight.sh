#!/usr/bin/env bash
set -euo pipefail

MIN_DATA_FREE_GIB="${MIN_DATA_FREE_GIB:-100}"
DATA_ROOT="${DATA_ROOT:-/data}"

section() {
  printf '\n[%s]\n' "$1"
}

section "identity"
date --iso-8601=seconds
printf 'user=%s\n' "$(id -un)"
printf 'host=%s\n' "$(hostname)"
printf 'kernel=%s\n' "$(uname -r)"

section "operating-system"
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -ds
else
  sed -n 's/^PRETTY_NAME=//p' /etc/os-release
fi
ldd --version 2>&1 | sed -n '1p'

section "compute"
lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|Thread|Core|Socket)'
free -h

section "gpu"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'ERROR: nvidia-smi is unavailable\n' >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < 2 )); then
  printf 'ERROR: expected at least 2 GPUs, found %s\n' "$gpu_count" >&2
  exit 1
fi

section "storage"
df -h "$DATA_ROOT" /home
if [[ ! -w "$DATA_ROOT" ]]; then
  printf 'ERROR: %s is not writable\n' "$DATA_ROOT" >&2
  exit 1
fi
data_free_kib="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
data_free_gib="$((data_free_kib / 1024 / 1024))"
if (( data_free_gib < MIN_DATA_FREE_GIB )); then
  printf 'ERROR: %s has only %s GiB free; require %s GiB\n' \
    "$DATA_ROOT" "$data_free_gib" "$MIN_DATA_FREE_GIB" >&2
  exit 1
fi

section "tools"
for tool in git python3 docker tmux curl; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s=%s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%s=missing\n' "$tool"
  fi
done

section "result"
printf 'PRECHECK_OK gpu_count=%s data_free_gib=%s\n' "$gpu_count" "$data_free_gib"

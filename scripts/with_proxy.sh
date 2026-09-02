#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: %s <command> [args...]\n' "$0" >&2
  printf 'Example: %s git clone https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab\n' "$0" >&2
  exit 2
fi

proxy_url="${HUMANOID_PROXY_URL:-http://127.0.0.1:7890}"

exec env \
  http_proxy="$proxy_url" \
  https_proxy="$proxy_url" \
  all_proxy="$proxy_url" \
  HTTP_PROXY="$proxy_url" \
  HTTPS_PROXY="$proxy_url" \
  ALL_PROXY="$proxy_url" \
  "$@"

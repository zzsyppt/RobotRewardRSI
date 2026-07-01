#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: %s <command> [args...]\n' "$0" >&2
  exit 2
fi

if [ "$1" = "git" ]; then
  shift
  exec env \
    -u http_proxy \
    -u https_proxy \
    -u all_proxy \
    -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u ALL_PROXY \
    -u HUMANOID_PROXY_URL \
    git -c http.proxy= -c https.proxy= -c all.proxy= "$@"
fi

exec env \
  -u http_proxy \
  -u https_proxy \
  -u all_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u ALL_PROXY \
  -u HUMANOID_PROXY_URL \
  "$@"

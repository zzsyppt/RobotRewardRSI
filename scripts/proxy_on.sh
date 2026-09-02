#!/usr/bin/env bash
# Source this file in an SSH shell when commands in that shell should use the
# reverse-tunneled local proxy.
#
#   source scripts/proxy_on.sh
#
# The default endpoint is provided by:
#   ssh -R 127.0.0.1:7890:127.0.0.1:7890 humanoid-server

HUMANOID_PROXY_URL="${HUMANOID_PROXY_URL:-http://127.0.0.1:7890}"

export http_proxy="$HUMANOID_PROXY_URL"
export https_proxy="$HUMANOID_PROXY_URL"
export all_proxy="$HUMANOID_PROXY_URL"
export HTTP_PROXY="$HUMANOID_PROXY_URL"
export HTTPS_PROXY="$HUMANOID_PROXY_URL"
export ALL_PROXY="$HUMANOID_PROXY_URL"

printf 'PROXY_ON %s\n' "$HUMANOID_PROXY_URL"
printf 'This affects only the current shell and child processes.\n'

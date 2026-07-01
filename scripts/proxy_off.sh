#!/usr/bin/env bash
# Source this file to remove proxy environment variables from the current SSH
# shell. It does not change global Git, systemd, apt, or network routing config.
#
#   source scripts/proxy_off.sh

unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset HUMANOID_PROXY_URL

printf 'PROXY_OFF current shell proxy variables cleared\n'

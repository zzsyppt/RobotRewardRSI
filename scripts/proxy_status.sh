#!/usr/bin/env bash
set -euo pipefail

proxy_url="${HUMANOID_PROXY_URL:-${https_proxy:-${HTTPS_PROXY:-http://127.0.0.1:7890}}}"
host="${proxy_url#*://}"
host="${host%%/*}"
port="${host##*:}"
host="${host%%:*}"

printf 'Proxy endpoint: %s\n' "$proxy_url"

printf '\nEnvironment variables:\n'
for name in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  value="${!name-}"
  if [ -n "$value" ]; then
    printf '  %s=%s\n' "$name" "$value"
  else
    printf '  %s=<unset>\n' "$name"
  fi
done

printf '\nGlobal Git proxy config:\n'
if git config --global --get-regexp '^(http|https|all)\.proxy$' 2>/dev/null; then
  :
else
  printf '  <none>\n'
fi

printf '\nLocal proxy port:\n'
if python3 - "$host" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket()
sock.settimeout(2.0)
try:
    sock.connect((host, port))
except OSError as exc:
    print(f"  closed: {host}:{port} ({exc})")
    raise SystemExit(1)
else:
    print(f"  open: {host}:{port}")
finally:
    sock.close()
PY
then
  port_ok=0
else
  port_ok=1
fi

if [ "${1:-}" = "--probe" ]; then
  printf '\nProxy HTTPS probe:\n'
  curl -I --proxy "$proxy_url" --connect-timeout 10 --max-time 20 https://github.com
fi

exit "$port_ok"

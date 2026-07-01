# Controlled proxy/VPN setup

Use a controlled user-space proxy first. Do not install a system VPN unless a
later task explicitly requires full-machine routing.

## Current recommended mode

Windows already runs Clash on `127.0.0.1:7890`. Expose it to the server through
an SSH reverse tunnel:

```powershell
Set-Location D:\humanoid_agent
powershell -ExecutionPolicy Bypass -File scripts\start_reverse_proxy_tunnel.ps1
```

Stop it when not needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_reverse_proxy_tunnel.ps1
```

This creates only a loopback endpoint on the server:

```text
http://127.0.0.1:7890
```

It does not change the server default route, DNS, apt, Git, or shell startup
files.

## Choose whether a server command uses the proxy

Single command through proxy:

```bash
cd /data/yuyangliu/AGENT
scripts/with_proxy.sh git ls-remote https://github.com/isaac-sim/IsaacLab.git HEAD
```

Single command without proxy, even if proxy variables exist:

```bash
cd /data/yuyangliu/AGENT
scripts/without_proxy.sh git ls-remote https://github.com/isaac-sim/IsaacLab.git HEAD
```

Current SSH shell through proxy:

```bash
cd /data/yuyangliu/AGENT
source scripts/proxy_on.sh
```

Current SSH shell back to direct networking:

```bash
source scripts/proxy_off.sh
```

Status:

```bash
cd /data/yuyangliu/AGENT
scripts/proxy_status.sh
scripts/proxy_status.sh --probe
```

## Policy

- Do not set `git config --global http.proxy` unless we intentionally want Git
  to always use the proxy.
- Do not write proxy exports into `.bashrc` or `.profile`.
- Do not use `sudo` or alter routing tables for this workflow.
- If a system VPN is required later, use WireGuard/OpenVPN only after confirming
  sudo access, firewall policy, and default-route behavior.

## Server capability notes

- The user account is in the `sudo` group, but sudo currently asks for a
  password.
- `/dev/net/tun` exists, so system VPN is technically possible with sudo.
- `wg`, `wg-quick`, and `openvpn` are not currently installed.
- `ip`, `nft`, `iptables`, `curl`, `git`, `ssh`, `nc`, and `python3` are
  present.

# Server preflight status

Status: Phase -1 complete (2026-06-20).

## Confirmed baseline

- Host: `deyugptserver`
- OS: Ubuntu 22.04.5 LTS
- Kernel: 5.15.0-94-generic
- CPU: 32 logical CPUs, Intel Xeon Gold 6133
- RAM: 283 GiB total
- GPU: 2 x NVIDIA GeForce RTX 5090, 32607 MiB each
- NVIDIA driver: 595.58.03
- GLIBC: 2.35
- Docker: 29.5.0
- `/home` free space: approximately 47 GiB
- `/data` free space after installation: approximately 119 GiB
- `/data` is writable by `yuyangliu`
- GitHub and NVIDIA Python package endpoints are reachable

## Required gates

- [x] SSH key authentication
- [x] Two GPUs visible and idle
- [x] `/data` workspace available
- [x] Python 3.11 environment installed
- [x] Smoke A: CUDA-enabled PyTorch sees both GPUs
- [x] Smoke B: Isaac Sim starts and exits headlessly
- [x] Smoke C: official Isaac Lab RL task trains with GPU physics
- [x] Smoke D: official Unitree G1 task trains with GPU physics
- [x] Development tools: Git LFS, tmux, rsync, GCC, CMake, Ninja
- [x] Later-phase Python stack: TensorBoard, SciPy, pytest, jsonschema, YAML/JSON
- [x] Cloud path: Anthropic reachable without credentials (403 probe)
- [ ] Two simultaneous PPO processes on GPU0/GPU1 (script prepared; remote execution quota blocked this run)

See `development_logs/phase_-1_environment_bootstrap/2026-06-20_phase_-1_completion.md`
for exact versions, commands, evidence, workarounds, and artifact paths.

See `setup/environment_reproduction.md` for the full clean-install procedure.

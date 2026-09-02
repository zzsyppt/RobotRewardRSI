# Phase -1 completion: environment bootstrap and GPU smoke gates

Date: 2026-06-20

Status: complete

Decision: **GO — begin Phase 0/1 development on Isaac Lab + Unitree RL Lab**

## Scope and acceptance criteria

Phase -1 establishes a reproducible server environment and requires all four
gates from the project plan:

1. PyTorch sees both RTX 5090 GPUs and executes finite CUDA work.
2. Isaac Sim/Isaac Lab starts and exits in the headless training configuration.
3. An unchanged official Isaac Lab RL task trains with GPU physics and writes a checkpoint.
4. An official Unitree G1/H1 task loads, resets, steps, trains, and writes a checkpoint.

All four gates passed.

## Server and storage

- Host: `deyugptserver` (`humanoid-server` SSH alias)
- OS: Ubuntu 22.04.5 LTS; kernel `5.15.0-94-generic`; glibc 2.35
- CPU/RAM: Xeon Gold 6133, 32 logical CPUs, 283 GiB RAM
- GPU: 2 x NVIDIA GeForce RTX 5090, 32 GiB each
- Driver: 595.58.03
- `/data` after setup: 597 GiB total, 119 GiB free
- Environment: `/data/yuyangliu/envs/isaaclab` (approximately 19 GiB)
- Caches: `/data/yuyangliu/cache` (approximately 12 GiB)
- Raw smoke logs: `/data/yuyangliu/runs/phase_-1`

Large environments, caches, raw logs, and checkpoints remain outside Git.

## Pinned software

| Component | Tested version/commit |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| TorchVision | 0.22.0+cu128 |
| TorchAudio | 2.7.0 |
| CUDA runtime bundled with PyTorch | 12.8 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab source | v2.3.0, `3c6e67bb5c7ada942a6d1884ab69338f57596f77` |
| `isaaclab` Python package | 0.47.2 |
| `isaaclab_tasks` | 0.11.6 |
| `isaaclab_rl` | 0.4.4 |
| Unitree RL Lab source | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` |
| `unitree_rl_lab` Python package | 0.2.1 |
| Unitree ROS G1 assets | `d6f13aad60320ce1d60a07b82a76b5a553f2a0a9` |

Compatibility pins that must not be casually upgraded:

```text
setuptools==80.9.0
wheel==0.45.1
typing_extensions==4.12.2
click==8.1.7
transformers==4.56.2
huggingface_hub==0.34.4
typer==0.12.5
onnx==1.19.1
onnxscript==0.5.6
psutil==5.9.8
```

Final `pip check`: `No broken requirements found.`

## Validation evidence

### Smoke A — PyTorch and CUDA: PASS

Command:

```bash
/data/yuyangliu/envs/isaaclab/bin/python \
  /data/yuyangliu/AGENT/scripts/smoke_torch_cuda.py
```

Evidence:

- `torch.cuda.is_available() == True`
- two RTX 5090 devices detected, compute capability 12.0
- 2048 x 2048 matrix multiplication completed with finite output on each GPU
- raw result: `/data/yuyangliu/runs/phase_-1/smoke_a_torch_cuda.json`

### Smoke B — Isaac Lab headless startup: PASS

Command:

```bash
/data/yuyangliu/AGENT/scripts/smoke_isaac_headless.sh
```

Evidence:

- AppLauncher selected `cuda:0`
- experience: `apps/isaaclab.python.headless.kit`
- application updated ten frames, printed `ISAAC_HEADLESS_OK`, and exited with code 0
- raw log: `/data/yuyangliu/runs/phase_-1/smoke_b_isaac_headless.log`

### Smoke C — official Isaac Lab PPO: PASS

Command:

```bash
ISAACLAB_ROOT=/data/yuyangliu/vendor/IsaacLab \
MAX_ITERATIONS=2 \
  /data/yuyangliu/AGENT/scripts/smoke_official_rl.sh
```

Task: unchanged official `Isaac-Ant-v0` with 4096 environments.

Evidence:

- environment device explicitly reported as `cuda:0`
- 262,144 total simulation steps
- 37,032 steps/s on iteration 0; 190,602 steps/s on iteration 1
- GPU0 reached about 4.3 GiB and sampled 17% utilization
- no NaN, fatal error, CPU-pipeline fallback, or training crash
- checkpoints: `model_0.pt` and `model_1.pt`, 2.9 MiB each

Artifacts:

```text
/data/yuyangliu/runs/phase_-1/smoke_c_official_rl.log
/data/yuyangliu/runs/phase_-1/smoke_c_gpu_monitor.log
/data/yuyangliu/vendor/IsaacLab/logs/rsl_rl/ant/2026-06-20_12-37-55/
```

### Smoke D — official Unitree G1 PPO: PASS

Command:

```bash
UNITREE_ROS_DIR=/data/yuyangliu/vendor/unitree_ros \
NUM_ENVS=1024 MAX_ITERATIONS=2 \
  /data/yuyangliu/AGENT/scripts/smoke_unitree_rl.sh
```

Task: official `Unitree-G1-29dof-Velocity`.

Evidence:

- G1-29DoF URDF loaded; environment reset and all managers initialized
- environment device explicitly reported as `cuda:0`
- existing official interval event `push_robot` was registered at 5-second intervals
- 49,152 total simulation steps
- 5,423 steps/s on iteration 0; 14,317 steps/s on iteration 1
- GPU0 reached about 4.15 GiB and sampled 32% utilization
- no NaN, fatal error, CPU-pipeline fallback, or training crash
- checkpoints: `model_0.pt` and `model_1.pt`, 9.6 MiB each

Artifacts:

```text
/data/yuyangliu/runs/phase_-1/smoke_d_unitree_g1.log
/data/yuyangliu/runs/phase_-1/smoke_d_gpu_monitor.log
/data/yuyangliu/vendor/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-20_12-45-30/
```

## Failures encountered and fixes

### EULA prompt read EOF

The first Smoke B sent Python source through standard input. Isaac Sim also uses
standard input for its EULA prompt, so it received EOF. The script now uses
`python -c`, leaving standard input attached to the interactive terminal. The
EULA was accepted once and persisted by Isaac Sim.

### Full Isaac Sim experience crashed in the RTX renderer

The generic `SimulationApp({"headless": True})` loaded the full RTX stack. It
first exposed missing `libGLU.so.1`; after that library was supplied, the full
RTX experience still segfaulted in `librtx.scenedb.plugin.so` on this headless
5090 server.

Training does not need cameras or RTX rendering. Using Isaac Lab's intended
`isaaclab.python.headless.kit` avoids the renderer and passed all physics and PPO
tests. This is a valid path for the current proprioceptive MVP. Camera/RTX video
rendering remains unvalidated and must not be assumed to work.

The account has no passwordless sudo. `libglu1-mesa` is therefore downloaded and
extracted under `/data/yuyangliu/vendor/system-libs`; smoke scripts prepend its
library directory to `LD_LIBRARY_PATH`.

### Isaac Lab core dependency build failed

`flatdict==4.0.1` imports `pkg_resources` while building. Newer setuptools
removed that compatibility path. Pinning `setuptools==80.9.0`, installing
`flatdict` without build isolation, and then installing the core editable package
resolved it. Transformers-related packages were also pinned to remain compatible
with Isaac Sim's mandatory `click==8.1.7`.

### Unitree assets were not included in the code repository

Unitree RL Lab ships placeholder asset paths. The documented Hugging Face
`unitree_model` download timed out from the server. The official documentation
recommends the `unitree_ros` URDF path for Isaac Sim 5.x, so only the G1 asset
subtree was sparse-cloned from GitHub and `setup/unitree_g1_urdf.patch` switched
G1-29DoF to `UnitreeUrdfFileCfg`.

## Reproduction and artifact sync

Re-run the environment installer only on a fresh or intentionally repaired
environment:

```bash
bash /data/yuyangliu/AGENT/scripts/install_phase_minus_1.sh
```

Run gates individually with the scripts in `scripts/`.

Download raw Phase -1 logs to Windows without placing them in Git:

```powershell
New-Item -ItemType Directory -Force D:\humanoid_agent_artifacts | Out-Null
scp -r humanoid-server:/data/yuyangliu/runs/phase_-1 `
  D:\humanoid_agent_artifacts\
```

Source and Markdown logs synchronize through Git:

```powershell
Set-Location D:\humanoid_agent
git pull --ff-only origin main
```

## Remaining risks and next action

- RTX/camera rendering is not a supported assumption on this host; the MVP is
  proprioceptive and headless, so this does not block development.
- Vulkan enumerates both GPUs and allocates small context memory on GPU1, while
  simulation/training tensors are explicitly on GPU0. GPU1 remains available for
  the second independent seed.
- The first real task implementation should start from the existing G1 velocity
  environment and inspect its already-registered `push_robot` event before
  defining standing push recovery.

Final decision: **GO**. The server can train the required headless G1 policies.

## Post-completion reproducibility check

The environment was rechecked against the full Phase 0–7 plan on 2026-06-20.
The check added two later-phase dependencies that were not necessary for the
original A–D gates:

- `ninja==1.13.2` for possible Extension C native/PyTorch extension builds;
- `jsonschema==4.25.1` for strict Phase 3 config/patch validation.

The recheck passed with:

- `pip check`: no broken requirements;
- finite CUDA work on both RTX 5090 devices;
- all required source files, G1 assets, and user-space `libGLU` present;
- YAML, JSON, SciPy statistics, pytest, TensorBoard, Hydra/OmegaConf, ONNX, and
  HTTP client packages installed;
- Git LFS, tmux, rsync, GCC 11.4, CMake 4.3.4, and Ninja 1.13.2 available.

Network probes without credentials showed Anthropic reachable (HTTP 403),
NVIDIA PyPI reachable (HTTP 200), and direct OpenAI API access timing out
(`000`). Phase 5 can therefore use Anthropic directly; OpenAI requires an
approved proxy/gateway, the controlled SSH reverse-proxy setup, or local
orchestration.

The environment now covers the planned non-visual MVP and Extensions A/B/D/E,
and contains the build/test tools needed by Extension C. Full RTX/camera
rendering remains unsupported. The two-process `cuda:0`/`cuda:1` PPO smoke
script passed after the initial remote-operation quota block was cleared.

Full clean-install and operational documentation is in
`setup/environment_reproduction.md`.

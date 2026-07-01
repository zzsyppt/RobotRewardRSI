# Phase 0: official baseline workflow

Date: 2026-06-20

Status: complete

Plan requirement: do not change the task environment. Prove that official
baseline training, checkpoint resume, play/evaluation, and reproduction commands
work before implementing the custom standing push-recovery task.

## Planned baseline gates

1. Train unchanged official Isaac Lab `Isaac-Ant-v0`.
2. Resume `Isaac-Ant-v0` training from `model_0.pt`.
3. Run official Ant `play.py` from the checkpoint and export `policy.pt` /
   `policy.onnx`.
4. Train unchanged official Unitree `Unitree-G1-29dof-Velocity`.
5. Resume Unitree G1 training from `model_0.pt`.
6. Run Unitree `play.py` from the checkpoint and export `policy.pt` /
   `policy.onnx`.
7. Save exact reproduction commands and a machine-readable `summary.json`.

## Implementation

The workflow is implemented in:

```bash
scripts/phase0_official_baseline.sh
```

Default smoke-scale parameters:

```text
DEVICE=cuda:0
SEED=42
TRAIN_ITERATIONS=1
RESUME_ITERATIONS=1
ANT_NUM_ENVS=1024
UNITREE_NUM_ENVS=512
PLAY_NUM_ENVS=64
PLAY_TIMEOUT_SECONDS=60
```

The play/evaluation scripts are official infinite-loop play scripts. The Phase 0
wrapper therefore runs them under a timeout and treats timeout exit as success
only if the checkpoint was loaded and exported policy artifacts were written.

## Results

Remote run:

```text
/data/yuyangliu/runs/phase_0_official_baseline/2026-06-20_16-20-12
```

Machine-readable summary:

```text
/data/yuyangliu/runs/phase_0_official_baseline/2026-06-20_16-20-12/summary.json
```

Reproduction commands:

```text
/data/yuyangliu/runs/phase_0_official_baseline/2026-06-20_16-20-12/commands.sh
```

### Official Isaac Lab Ant baseline: PASS

- Task: `Isaac-Ant-v0`
- Train exit: 0
- Resume exit: 0
- Play/eval exit: 124, expected timeout from the official infinite-loop play script
- Training checkpoint:
  `/data/yuyangliu/vendor/IsaacLab/logs/rsl_rl/ant/2026-06-20_16-20-24_phase0_ant_train/model_0.pt`
  (`3,023,639` bytes)
- Resume checkpoint:
  `/data/yuyangliu/vendor/IsaacLab/logs/rsl_rl/ant/2026-06-20_16-20-47_phase0_ant_resume/model_0.pt`
  (`3,023,639` bytes)
- Exported policy artifacts:
  - `exported/policy.pt` (`514,130` bytes)
  - `exported/policy.onnx` (`503,282` bytes)
- Evidence: train log contains learning iteration 0; resume log loaded the
  checkpoint; play produced exported policy artifacts and no fatal error before
  timeout.

### Official Unitree G1 velocity baseline: PASS

- Task: `Unitree-G1-29dof-Velocity`
- Train exit: 0
- Resume exit: 0
- Play/eval exit: 124, expected timeout from the official infinite-loop play script
- Training checkpoint:
  `/data/yuyangliu/vendor/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-20_16-22-13_phase0_unitree_train/model_0.pt`
  (`10,010,839` bytes)
- Resume checkpoint:
  `/data/yuyangliu/vendor/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-20_16-22-44_phase0_unitree_resume/model_0.pt`
  (`10,010,839` bytes)
- Exported policy artifacts:
  - `exported/policy.pt` (`1,669,010` bytes)
  - `exported/policy.onnx` (`1,658,204` bytes)
- Evidence: train log contains learning iteration 0; resume log loaded the
  checkpoint; play produced exported policy artifacts and no fatal error before
  timeout.

## Decision

Phase 0 is complete.

The project now has a reproducible official-baseline workflow for unchanged
training, checkpoint recovery, play/evaluation, and command replay. It is safe
to start Phase 1: implementing the standing push-recovery environment.

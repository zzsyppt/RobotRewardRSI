# 2026-07-02 IsaacLab runtime entrypoints

## Scope

- Added project-local IsaacLab/RSL-RL train and eval entrypoints:
  - `scripts/isaaclab_rsl_rl_train.py`
  - `scripts/isaaclab_rsl_rl_eval.py`
- Added true_RL_Agent-owned push recovery runtime helpers:
  - runtime config defaults and YAML merge
  - IsaacLab environment overlay
  - protected eval failure classifier and diagnostic hints
  - candidate runtime reward adapter
- Updated backend materialization to emit structured Python argv for the server IsaacLab environment.
- Updated eval binding so the eval subprocess receives the runner-produced local checkpoint path while ledger checkpoint identity remains runner-sealed.

## Design Notes

- Runtime scripts do not import the parent `humanoid_agent` package and do not call parent scripts.
- Training loads candidate code only inside the train subprocess and injects guarded reward contributions into RSL-RL VecEnv rewards.
- Trusted eval loads only the checkpoint and never loads candidate code.
- Eval summary exposes seed policy and public seed count, not raw seed ids.
- Runtime activation evidence uses the maximum absolute candidate reward observed across training, so zero-mean but active reward terms are not falsely rejected.

## Verification

- `python -m unittest tests.unit.test_push_recovery_runtime_adapter tests.unit.test_prompt_projection_and_mechanism tests.unit.test_push_recovery_task -v`
- `python -m unittest discover -s tests -v`
- `python -m compileall -q src scripts tests`
- Static scans found no parent project runtime references, shell execution calls, or raw seed JSON fields in the new runtime path.

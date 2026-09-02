# Phase 1: standing push-recovery environment

Date: 2026-06-20

Status: complete; push-effect gate superseded by later audit correction

Correction note: the first push-effect gate in this log used broad pre/post
peak speed windows. A later audit showed that this gate could pass even in the
`no_push` profile. The authoritative corrected validation is:

```text
development_logs/phase_1_push_recovery_environment/2026-06-20_phase_1_gate_audit_fix.md
```

The corrected gate uses a single-step root velocity-vector jump near the
configured push time and compares it against the no-push control.

Plan requirement: implement the Standing Push Recovery environment on top of
Unitree G1, confirm the push mechanism, force standing commands, add binary
success and ordered failure classification, run random-policy smoke, run a short
training smoke, and emit machine-readable JSON for future agent/LLM decisions.

## Implementation plan executed

1. Keep the official Unitree vendor code unchanged.
2. Load `Unitree-G1-29dof-Velocity` and apply a project-local Phase 1 config
   overlay.
3. Set the command distribution to standing:
   `lin_vel_x = 0`, `lin_vel_y = 0`, `ang_vel_z = 0`.
4. Use the existing Unitree/IsaacLab push event:
   `push_by_setting_velocity`, i.e. a base velocity kick, not raw force.
5. Record direct push evidence from root planar velocity before and after the
   configured push window.
6. Keep binary success clean:
   min base height, max roll, max pitch, final xy drift.
7. Keep energy/action-rate style metrics as secondary diagnostics only.
8. Emit richer JSON than the initial plan example so a future LLM can judge not
   only the dominant failure label, but also thresholds, evidence, gates, and
   likely next patch scope.

## Files added

```text
configs/phase1_push_recovery.yaml
src/humanoid_agent/phase1_push_recovery/config.py
src/humanoid_agent/phase1_push_recovery/failure_classifier.py
src/humanoid_agent/phase1_push_recovery/diagnostics.py
src/humanoid_agent/phase1_push_recovery/isaac_overlay.py
scripts/phase1_push_recovery_eval.py
scripts/phase1_push_recovery_train_smoke.py
scripts/phase1_collect_smoke_summary.py
scripts/phase1_push_recovery_smoke.sh
```

## JSON schema direction

The Phase 1 evaluator now emits:

- `schema_version`
- command line, host, seed, device
- config path and sha256
- active profile
- standing command and push config snapshot
- protected success thresholds
- reset / observation finite / reward finite evidence
- push-effect evidence:
  pre-push peak speed, post-push peak speed, observed delta, fraction of envs
  above threshold
- success rate and ordered failure histogram
- secondary metrics:
  episode steps, return, action norm, action-rate norm, min height, pitch, roll,
  drift
- bounded per-episode evidence records
- `diagnostic_hints` for future LLM/rule agents

This is intentionally more informative than only:

```json
{
  "success_rate": 0.42,
  "failure_breakdown": {},
  "secondary_metrics": {}
}
```

The additional fields are necessary because the future agent should be able to
distinguish "push did not happen", "policy under-reacted", "policy over-acted",
"task/eval threshold changed", and "metric noise" without reading raw logs.

## Default Phase 1 smoke command

Remote command:

```bash
cd /data/yuyangliu/AGENT
DEVICE=cuda:0 bash scripts/phase1_push_recovery_smoke.sh
```

Default parameters:

```text
DEVICE=cuda:0
SEED=42
EVAL_NUM_ENVS=64
EVAL_MAX_STEPS=220
TRAIN_NUM_ENVS=256
TRAIN_ITERATIONS=1
TRAIN_TIMEOUT_SECONDS=300
```

## Formal validation run

Remote run:

```text
/data/yuyangliu/runs/phase_1_push_recovery/2026-06-20_16-52-31
```

Machine-readable summary:

```text
/data/yuyangliu/runs/phase_1_push_recovery/2026-06-20_16-52-31/summary.json
```

Reproduction commands:

```text
/data/yuyangliu/runs/phase_1_push_recovery/2026-06-20_16-52-31/commands.sh
```

Result: `PASS`.

### Gates

- `zero_strong_push_status_pass`: pass
- `zero_strong_push_reward_finite`: pass
- `zero_strong_push_observation_finite`: pass
- `zero_strong_push_effect_observed`: pass
  - mechanism: `base_velocity_kick`
  - expected interval: `[1.0, 1.0]` seconds
  - mean pre-push peak planar speed: `0.832243 m/s`
  - mean post-push peak planar speed: `1.327982 m/s`
  - mean observed delta: `0.495739 m/s`
  - env fraction with delta above threshold: `1.0`
- `random_strong_push_status_pass`: pass
- `random_strong_push_reward_finite`: pass
- `random_strong_push_observation_finite`: pass
- `random_policy_mostly_fails_under_strong_push`: pass
  - observed success rate: `0.0`
  - required: `<= 0.35`
- `zero_no_push_status_pass`: pass
- `zero_no_push_reward_finite`: pass
- `zero_no_push_observation_finite`: pass
- `zero_no_push_not_immediate_collapse`: pass
  - observed mean episode steps: `60.953125`
  - required: `>= 50`
- `train_smoke_status_pass`: pass
  - checkpoint:
    `/data/yuyangliu/runs/phase_1_push_recovery/2026-06-20_16-52-31/train_smoke/rsl_rl/model_0.pt`
  - checkpoint size: `9.6M`
  - total timesteps: `6144`

## Interpretation

Phase 1 is complete at the environment/smoke level.

What is proven:

- The project can construct a standing command variant of the Unitree G1 task.
- The disturbance uses the existing IsaacLab/Unitree velocity-kick event.
- Push effect is visible in measured base planar velocity.
- The environment resets and steps under headless GPU Isaac.
- Rewards and observations are finite in the tested profiles.
- Random policy mostly fails under strong push, as expected.
- No-push zero-policy episodes do not collapse immediately, which is enough for
  a Phase 1 runability gate.
- Short PPO training on the Phase 1 overlay starts and writes a checkpoint.
- Failure reports are emitted as rich JSON suitable for later Agent/rule loops.

What is not claimed yet:

- A trained push-recovery policy is good.
- Seed variance is known.
- The task difficulty is calibrated to the target `30%–70%` public baseline
  success range.

Those belong to Phase 2.

## Next phase entry

Phase 2 should train the initial human baseline across multiple seeds, evaluate
on fixed public seeds, compute mean/std/standard error, and tune task difficulty
only if the baseline success rate is outside the intended range.

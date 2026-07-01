# Phase 2 baseline variance completion log

Date: 2026-06-20

## Scope

Phase 2 follows `humanoid_agentic_policy_improvement_plan_v3.md` §6.3:

1. define a hand-written initial reward/curriculum baseline;
2. train five seeds;
3. evaluate each checkpoint on a fixed public seed;
4. report baseline mean/std and single-seed GPU-hours;
5. also run an IID-hidden evaluation for later phases, without using it for tuning;
6. calibrate public difficulty so the mean success rate lands in 30%–70%.

The implementation intentionally keeps the official Unitree G1 velocity task and
RSL-RL training path. Phase 2 changes only project-local overlays:

- push/eval profiles in `configs/phase1_push_recovery.yaml`;
- reward/curriculum/PPO scalar overlay in `configs/phase2_baseline.yaml`;
- checkpoint evaluation and result aggregation scripts.

## Implemented files

- `configs/phase2_baseline.yaml`
  - baseline name: `manual_standing_push_recovery_v1`;
  - train seeds: `101`, `102`, `103`, `104`, `105`;
  - training budget: 512 envs, 500 iterations, one CUDA device;
  - public eval: profile `phase2_public`, seed `1001`, 128 envs, 220 steps;
  - IID-hidden eval: profile `phase2_iid_hidden`, seed `2001`, 128 envs, 220 steps;
  - target public mean success range: `[0.30, 0.70]`.
- `src/humanoid_agent/phase2_baseline/overrides.py`
  - applies reward weights;
  - applies reward params such as standing base height target;
  - disables terrain/velocity curricula when those config objects exist;
  - applies optional PPO fields only when present on the loaded config object.
- `scripts/phase2_baseline_variance.sh`
  - reads Phase 2 YAML into shell variables;
  - trains each seed;
  - extracts the latest checkpoint from training summary;
  - evaluates public and IID-hidden profiles;
  - records `.exit` and `.seconds` files;
  - always runs the collector before exiting.
- `scripts/phase2_collect_baseline_summary.py`
  - aggregates per-seed training/eval summaries;
  - computes mean/std/standard error;
  - records per-seed and total GPU-hour estimates;
  - marks the workflow pass/fail separately from difficulty calibration.
- `scripts/phase1_push_recovery_train_smoke.py`
  - accepts arbitrary config profiles instead of hard-coded Phase 1 choices;
  - accepts `--phase2-config`;
  - records the Phase 2 config hash and override report.
- `scripts/phase1_push_recovery_eval.py`
  - supports `--policy checkpoint --checkpoint ...`;
  - loads RSL-RL checkpoints through `OnPolicyRunner`;
  - uses `extras["time_outs"]` to distinguish successful full-horizon timeouts
    from real fall terminations.
- `scripts/phase1_push_recovery_smoke.sh`
  - fixes `run_logged` so a failed subcommand still writes `.exit` and allows
    final summary collection.

## Corrections made before the formal run

### 1. Push gate evidence was not a useful Phase 2 signal

The Phase 1 single-step push-effect gate could pass even when no-push and push
rollouts both had large velocity changes caused by the uncontrolled zero policy
falling. That gate was already downgraded to a low-level environment evidence
gate, not a baseline-quality metric.

Phase 2 therefore uses learned checkpoint evaluation over full rollouts instead
of the Phase 1 velocity-window gate.

### 2. Failure labels were dominated by uncontrolled pre-baseline falls

Zero/random policies mostly falling forward is expected before a learned
standing policy exists. Phase 2 labels are now interpreted only from trained
checkpoint rollouts.

A second bug was found during checkpoint evaluation: IsaacLab/RSL-RL wrappers
return `dones`, while the timeout signal lives in `extras["time_outs"]`. Treating
all `done=True` as termination mislabeled full-horizon success as
`fall_unknown`. The evaluator now splits timeout from failure explicitly.

### 3. Shell cleanup and profile consistency

The smoke shell runner now restores `set -e` after each logged command and
always writes exit codes. Phase 2 training uses the `phase2_train` no-push
profile, while public/IID evaluation use the same ±0.25 m/s push difficulty
declared in config.

The `push_body` field remains only as an audit label in config. The actual
event uses root velocity setting through `push_by_setting_velocity`; this was
verified at the config/event level and is documented as such.

## Calibration runs

The first 3-seed attempt used an 80-iteration budget and nonzero training push.
All seeds failed because the policy had not learned stable standing; training
terminated around ten steps and orientation penalties dominated. This was not a
valid Phase 2 baseline.

A calibration run trained seed `201` with no push for 500 iterations:

```bash
cd /data/yuyangliu/AGENT
DEVICE=cuda:0 RUN_ROOT=/data/yuyangliu/runs/phase_2_baseline_variance \
  bash scripts/phase2_baseline_variance.sh
```

The useful calibration checkpoint was:

```text
/data/yuyangliu/runs/phase_2_baseline_variance/calibration_no_push_500/seed_201/train/rsl_rl/model_499.pt
```

After fixing timeout handling, it showed:

- no-push eval: 128/128 success;
- ±0.15 m/s push eval: 128/128 success;
- ±0.25 m/s push eval: 85/128 success, success rate 0.664;
- ±0.60 m/s push eval: 1/128 success, success rate 0.008.

Therefore the formal Phase 2 config uses no-push training and ±0.25 m/s public
/ IID-hidden push evaluation.

## Formal run

Command:

```bash
cd /data/yuyangliu/AGENT
DEVICE=cuda:0 RUN_ROOT=/data/yuyangliu/runs/phase_2_baseline_variance_5seed \
  RUN_ID=2026-06-20_5seed_formal \
  bash scripts/phase2_baseline_variance.sh
```

Run directory:

```text
/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal
```

Summary file:

```text
/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json
```

## Formal results

Overall status: `pass`

Workflow complete: `true`

Public difficulty calibration needed: `false`

### Public evaluation

Target mean success range: `[0.30, 0.70]`

| Seed | Public success | Dominant public failure | Checkpoint |
| --- | ---: | --- | --- |
| 101 | 0.492188 | fall_backward | `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/seed_101/train/rsl_rl/model_499.pt` |
| 102 | 0.421875 | fall_backward | `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/seed_102/train/rsl_rl/model_499.pt` |
| 103 | 0.757812 | fall_backward | `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/seed_103/train/rsl_rl/model_499.pt` |
| 104 | 0.414062 | fall_backward | `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/seed_104/train/rsl_rl/model_499.pt` |
| 105 | 0.523438 | fall_backward | `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/seed_105/train/rsl_rl/model_499.pt` |

Aggregate:

- mean: `0.521875`;
- std: `0.1397977753`;
- standard error: `0.0625194657`;
- values: `[0.492188, 0.421875, 0.757812, 0.414062, 0.523438]`;
- target range met by mean: `true`.

Note: seed `103` is individually above the 0.70 upper target, but the plan
requires calibrating public success average. The mean is in range and still
leaves useful improvement headroom.

Non-exclusive public condition flag rates:

- `drift_over_threshold`: `0.4546875`;
- `height_below_threshold`: `0.4640625`;
- `pitch_over_threshold`: `0.4671875`;
- `roll_over_threshold`: `0.4375`;
- `terminated`: `0.0`.

### IID-hidden evaluation

| Seed | IID-hidden success | Dominant IID-hidden failure |
| --- | ---: | --- |
| 101 | 0.414062 | fall_backward |
| 102 | 0.414062 | fall_backward |
| 103 | 0.679688 | fall_backward |
| 104 | 0.359375 | fall_backward |
| 105 | 0.460938 | fall_backward |

Aggregate:

- mean: `0.465625`;
- std: `0.1249514750`;
- standard error: `0.0558799984`;
- values: `[0.414062, 0.414062, 0.679688, 0.359375, 0.460938]`.

Non-exclusive IID-hidden condition flag rates:

- `drift_over_threshold`: `0.5`;
- `height_below_threshold`: `0.5125`;
- `pitch_over_threshold`: `0.503125`;
- `roll_over_threshold`: `0.4890625`;
- `terminated`: `0.0`.

IID-hidden was not used for tuning the baseline. It is stored for Phase 3+
regression checks and later agent-evaluation comparison.

### Compute accounting

Method: wall-clock seconds × one visible GPU per run.

| Seed | Train seconds | Public eval seconds | IID-hidden eval seconds | GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| 101 | 763 | 30 | 30 | 0.228611 |
| 102 | 751 | 29 | 29 | 0.224722 |
| 103 | 754 | 29 | 30 | 0.225833 |
| 104 | 739 | 28 | 30 | 0.221389 |
| 105 | 749 | 29 | 31 | 0.224722 |

Aggregate:

- total GPU-hours estimate: `1.1252777778`;
- mean per seed: `0.2250555556`.

## Acceptance check

| Plan requirement | Status | Evidence |
| --- | --- | --- |
| Hand-written reward/curriculum baseline | pass | `configs/phase2_baseline.yaml` plus override report in each `train_summary.json` |
| Train 5 seeds | pass | seeds `101`, `102`, `103`, `104`, `105`, all `train_status=pass` |
| Fixed public eval | pass | public profile `phase2_public`, seed `1001`, 128 envs |
| Mean/std public summary | pass | mean `0.521875`, std `0.1397977753` |
| Difficulty average 30%–70% | pass | public mean in target range |
| IID-hidden summary | pass | mean `0.465625`, std `0.1249514750` |
| Single-seed GPU-hours | pass | recorded in `summary.json` and table above |

## Decision

Phase 2 is complete.

The learned baseline can stand under no push and has nontrivial, calibrated
failure under ±0.25 m/s pushes. This is suitable for Phase 3 verifier work:
the next phase should protect eval seeds/budgets, verify parse/import/reset/NaN
checks, and make later LLM-generated config changes auditable before training.

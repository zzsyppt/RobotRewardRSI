# Phase 4 hand-coded rule-baseline completion log

Date: 2026-06-20

Follow-up note: `2026-06-20_rule_candidate_002` was superseded by
`2026-06-20_rule_candidate_003` after the audit in
`2026-06-20_phase_4_audit_followup.md`. The change fixed YAML scientific-float
serialization so PyYAML reloads small reward weights as floats rather than
strings.

## Scope

Phase 4 implements the required non-LLM baseline from
`humanoid_agentic_policy_improvement_plan_v3.md`:

1. read a Phase 2/3 evaluation summary;
2. classify the dominant failure and diagnostic signals;
3. generate config-only candidate patches;
4. pass the Phase 3 verifier;
5. prove the generated candidate can start training.

The Phase 4 baseline is intentionally deterministic and limited. It is a control
condition for later LLM-agent runs, not the final policy improvement loop.

## Implementation plan

1. Keep the baseline inside the Phase 3 verifier contract:
   - only generate `configs/phase1_push_recovery.yaml` and
     `configs/phase2_baseline.yaml` candidates;
   - do not change public/IID eval profiles, eval seeds, max steps, training
     budget, metric logic, failure-classifier code, or reward implementations.
2. Implement a transparent failure-to-intervention table:
   - `fall_forward` / `fall_backward`: stronger upright and angular-velocity
     shaping plus pitch-dominant training push curriculum;
   - `fall_side`: stronger roll/slide shaping plus lateral push curriculum;
   - `height_collapse`: stronger base-height and alive shaping;
   - `excessive_drift`: stronger drift/velocity shaping;
   - optional secondary diagnostics for high action rate, high energy, and
     oscillation when future summaries expose those fields.
3. Generate reproducible artifacts:
   - candidate Phase 1 push-recovery config;
   - candidate Phase 2 baseline config;
   - `analysis.json`;
   - `candidate.diff`;
   - `verifier_summary.json`;
   - `run_summary.json`.
4. Add a pure Python self-test that verifies:
   - training push is enabled;
   - public eval config stays unchanged;
   - reward weights are changed;
   - optional oscillation diagnostic path is covered;
   - Phase 3 accepts the candidate.
5. Run the final candidate on the server with a 1-iteration CUDA train-start
   smoke.

## Implemented files

- `configs/phase4_rule_baseline.yaml`
  - thresholds for primary failures and secondary diagnostics;
  - training-only push curriculum presets;
  - bounded reward-weight deltas;
  - conservative PPO hook for future low-success cases.
- `src/humanoid_agent/phase4_rule_baseline/rules.py`
  - summary diagnosis;
  - deterministic intervention selection;
  - config candidate construction;
  - stable JSON/YAML-like serialization;
  - unified diff rendering.
- `scripts/phase4_rule_baseline.py`
  - CLI for generating candidate artifacts from a summary file;
  - invokes Phase 3 static verifier by default.
- `scripts/phase4_rule_baseline_selftest.py`
  - local/server pure Python regression test.

## Final candidate evidence

Final server candidate:

```text
/data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002
```

Source summary:

```text
/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json
```

Key diagnosis from `analysis.json`:

- public success mean: `0.521875`;
- public success values:
  `[0.492188, 0.421875, 0.757812, 0.414062, 0.523438]`;
- primary failure counts:
  `fall_backward=184`, `fall_forward=115`, `fall_side=7`,
  `success=334`;
- primary failure rates:
  `fall_backward=0.2875`, `fall_forward=0.1796875`,
  `fall_side=0.0109375`, `success=0.521875`;
- condition flag rates:
  `pitch_over_threshold=0.467187`,
  `height_below_threshold=0.464062`,
  `drift_over_threshold=0.454688`,
  `roll_over_threshold=0.4375`,
  `terminated=0.0`.

Selected interventions:

1. `fall_backward`
2. `height_collapse`
3. `excessive_drift`

Applied config changes:

- `profiles.phase2_train.push`
  - `interval_range_s: [1.0, 1.0]`
  - `velocity_range_mps.x: [-0.1, 0.1]`
  - `velocity_range_mps.y: [-0.1, 0.1]`
- `reward_weights.flat_orientation_l2: -8.0 -> -10.0`
- `reward_weights.base_angular_velocity: -0.2 -> -0.28`
- `reward_weights.base_height: -12.0 -> -14.0`
- `reward_weights.alive: 0.25 -> 0.3`
- `reward_weights.base_linear_velocity: -2.0 -> -2.5`
- `reward_weights.track_lin_vel_xy: 2.0 -> 2.2`

The real Phase 2 aggregate summary does not currently include energy or
oscillation secondary metrics, so those optional rules did not fire for this
candidate. The generator already supports those fields for later Phase 5/6
summaries.

## Validation

Local pure Python validation:

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m py_compile `
  scripts\phase4_rule_baseline.py `
  scripts\phase4_rule_baseline_selftest.py `
  src\humanoid_agent\phase4_rule_baseline\rules.py `
  src\humanoid_agent\phase4_rule_baseline\__init__.py

C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  scripts\phase4_rule_baseline_selftest.py
```

Result: pass.

Server self-test:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase4_rule_baseline_selftest.py
```

Result: pass.

Final candidate generation:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase4_rule_baseline.py \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002
```

Result:

```json
{
  "status": "pass",
  "verifier_status": "pass"
}
```

Phase 3 verifier summary:

```json
{
  "status": "pass",
  "mode": "static",
  "issues": [],
  "checks": [
    {"name": "diff_policy", "status": "pass"},
    {"name": "protected_yaml_keys", "status": "pass"},
    {"name": "finite_numeric_config", "status": "pass"},
    {"name": "phase2_semantics", "status": "pass"},
    {"name": "python_imports", "status": "skipped"}
  ]
}
```

Final train-start smoke:

```bash
cd /data/yuyangliu/AGENT
source scripts/activate_environment.sh
PYTHONPATH=/data/yuyangliu/AGENT/src python scripts/phase1_push_recovery_train_smoke.py \
  --headless \
  --device cuda:0 \
  --task Unitree-G1-29dof-Velocity \
  --seed 402 \
  --profile phase2_train \
  --num-envs 128 \
  --max-iterations 1 \
  --config /data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002/candidate_phase1_push_recovery.yaml \
  --phase2-config /data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002/candidate_phase2_baseline.yaml \
  --run-dir /data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002/train_start_smoke \
  --summary-file /data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002/train_start_smoke_summary.json
```

Result from `train_start_smoke_summary.json`:

```json
{
  "status": "pass",
  "training": {
    "device": "cuda:0",
    "num_envs": 128,
    "max_iterations": 1,
    "checkpoint_count": 1,
    "latest_checkpoint": "/data/yuyangliu/runs/phase_4_rule_baseline/2026-06-20_rule_candidate_002/train_start_smoke/rsl_rl/model_0.pt"
  }
}
```

The IsaacLab log showed the interval push event active at `(1.0, 1.0)` and the
candidate reward weights loaded by the environment.

## Decision

Phase 4 is complete.

Go criteria met:

- deterministic rule baseline implemented;
- summary-to-config candidate generation works;
- generated candidate stays inside Phase 3 verifier constraints;
- public/IID eval profile and budget protections remain intact;
- final candidate starts CUDA training and writes a checkpoint.

## Remaining risks and carry-forward

- This phase does not claim that the rule candidate improves success rate. It
  only proves the rule baseline can generate a legal candidate and start
  training. Multi-seed training/evaluation belongs to Phase 5/6 accept-reject
  comparisons.
- Current Phase 2 aggregate summaries do not yet aggregate energy or oscillation
  diagnostics. Phase 4 supports those optional fields, but they will only affect
  candidates once future summaries provide them.
- The rule baseline is deliberately simple. If it performs close to or better
  than the LLM agent later, the project conclusion should report that honestly
  and narrow the claim to framework/runability rather than LLM superiority.

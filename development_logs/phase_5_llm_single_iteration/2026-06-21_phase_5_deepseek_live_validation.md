# Phase 5 DeepSeek live validation log

Date: 2026-06-21

Audit note added 2026-06-21: this original live proposal is superseded by
`2026-06-21_phase_5_prompt_leak_audit_fix.md`.  The original prompt included
IID-hidden/eval-seed fields from the raw Phase 2 summary.  Use the corrected
public-only run below for clean Phase 5 claims:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001
```

## Scope

This log records the first live Phase 5 LLM-agent proposal using
`deepseek-v4-pro`, followed by Phase 3 verifier validation and a CUDA
train-start smoke.

The API key was not printed, committed, or copied into project artifacts. The
server key is loaded from:

```text
/data/yuyangliu/secrets/deepseek.env
```

with mode `0600`.

## Commands

Because long SSH command strings were unstable on this host, two short wrappers
were added:

- `scripts/phase5_live_provider.sh`
- `scripts/phase5_train_start_smoke.sh`

Live proposal:

```bash
cd /data/yuyangliu/AGENT
RUN_ID=2026-06-21_deepseek_v4_pro_live_001 \
PHASE5_PROVIDER=deepseek \
bash scripts/phase5_live_provider.sh
```

Train-start smoke:

```bash
cd /data/yuyangliu/AGENT
CANDIDATE_DIR=/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001 \
SEED=501 \
bash scripts/phase5_train_start_smoke.sh
```

## Output directory

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001
```

Important files:

- `analysis.json`
- `candidate.diff`
- `candidate_phase1_push_recovery.yaml`
- `candidate_phase2_baseline.yaml`
- `provider_request_summary.json`
- `raw_response.json`
- `parsed_response.json`
- `verifier_summary.json`
- `run_summary.json`
- `train_start_smoke_summary.json`

## LLM proposal

Provider:

```text
deepseek / deepseek-v4-pro
```

Source public summary:

```text
/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json
```

Public success mean:

```text
0.521875
```

LLM hypothesis:

```text
The policy is undertrained on push disturbances because training pushes are
disabled (velocity zero). Adding moderate bidirectional pushes during training
will expose the controller to recovery situations, reducing fall-backward
failures and improving overall stability.
```

Applied operations:

```text
phase1 profiles.phase2_train.push.velocity_range_mps.x -> [-0.3, 0.3]
phase1 profiles.phase2_train.push.velocity_range_mps.y -> [-0.1, 0.1]
```

No reward, eval, seed, budget, metric, classifier, or verifier edits were
requested.

## Verifier result

`verifier_summary.json`:

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

## Train-start smoke

`train_start_smoke_summary.json`:

```json
{
  "status": "pass",
  "training": {
    "device": "cuda:0",
    "num_envs": 128,
    "max_iterations": 1,
    "checkpoint_count": 1,
    "latest_checkpoint": "/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/train_start_smoke/rsl_rl/model_0.pt"
  }
}
```

Loaded training push:

```json
{"x": [-0.3, 0.3], "y": [-0.1, 0.1]}
```

Loaded reward weights remained the Phase 2 baseline, e.g.:

```text
track_lin_vel_xy = 2.0
flat_orientation_l2 = -8.0
```

## Notes and carry-forward

- This is a valid live LLM proposal and it starts CUDA training.
- It is not yet a formal Phase 5 accept/reject result. The next step is fixed
  budget training/evaluation, then noise-aware comparison against baseline and
  rule baseline.
- The generated `candidate.diff` currently includes YAML reserialization noise
  because candidate files are written from parsed config dictionaries. The
  semantic `analysis.json` and `applied_changes` are clean. A future polish item
  is to preserve YAML formatting or emit an additional operation-only diff for
  LLM feedback.

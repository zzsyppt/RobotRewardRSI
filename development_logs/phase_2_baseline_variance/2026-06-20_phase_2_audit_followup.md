# Phase 2 audit follow-up

Date: 2026-06-20

This note records a post-completion audit of Phase 2 concerns before Phase 3.

## 1. Primary failure labels can hide co-occurring conditions

Status: real issue; code changed.

The protected classifier emits one primary `failure_label` per episode. Its
priority is:

```text
fall_forward -> fall_backward -> fall_side -> height_collapse -> fall_unknown -> excessive_drift -> success
```

This is fine for a stable success contract, but it means conditions such as
`height_below_threshold` or `drift_over_threshold` can be true without becoming
the primary label. In the Phase 2 summaries, public/IID failures are dominated
by pitch/roll labels, so `height_collapse` and `excessive_drift` are not reliable
as sole diagnostic words.

Change:

- `scripts/phase1_push_recovery_eval.py` now records
  `aggregate.condition_flag_breakdown`;
- `src/humanoid_agent/phase1_push_recovery/failure_classifier.py` now exposes
  `summarize_condition_flags`;
- diagnostic hints now warn when height/drift flags are masked by higher-priority
  primary labels.

This keeps binary success and primary failure labels unchanged while giving
future rule/LLM agents non-exclusive evidence.

## 2. Push timing is fixed rather than random in Phase 2 eval

Status: real; no behavior change.

`phase2_public` and `phase2_iid_hidden` use:

```yaml
interval_range_s: [1.0, 1.0]
velocity_range_mps:
  x: [-0.25, 0.25]
  y: [-0.25, 0.25]
```

This is an intentional fixed-eval contract: time is reproducible, direction and
magnitude are random. Later training profiles may introduce randomized push
timing/curriculum, but public/IID eval profiles should be protected by the
Phase 3 verifier.

## 3. Eval disables vendor fall terminations

Status: real; no behavior change.

The evaluator disables vendor fall terminations and runs the protected rollout
window. This avoids vendor auto-reset definitions silently changing the public
success contract. Success still requires staying within protected min/max
height, roll, pitch, and drift thresholds over the rollout, so this does not
inflate success by ignoring falls.

## 4. Three-seed Phase 2 variance was noisy

Status: real; Phase 2 was rerun with five seeds.

The first calibrated Phase 2 run followed the project plan requirement of three
training seeds. The observed public values were:

```text
[0.492188, 0.421875, 0.757812]
```

Mean success was in target range, but standard error was about 10 percentage
points. The final Phase 2 baseline now uses five seeds:

```text
[0.492188, 0.421875, 0.757812, 0.414062, 0.523438]
```

The five-seed public mean is `0.521875`, standard deviation is
`0.1397977753`, and standard error is `0.0625194657`. This is materially better
than the three-seed estimate, but still noisy enough that later acceptance tests
for claiming improvement over the baseline should compare against pooled
standard error instead of using raw mean deltas only.

Carry-forward requirement for Phase 3:

- protect public/IID seeds, push profiles, thresholds, and training budget;
- allow training-profile curriculum changes only when they do not mutate public
  or IID eval contracts;
- require future summaries to expose both primary failure labels and
  non-exclusive condition flags.

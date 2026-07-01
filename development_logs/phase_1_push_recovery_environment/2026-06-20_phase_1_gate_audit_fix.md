# Phase 1 gate audit and correction

Date: 2026-06-20

Status: complete

This audit was triggered after reviewing two issues in the first Phase 1
implementation:

1. The original push-effect gate used the difference between peak planar base
   speeds in broad pre/post windows. This could pass even when the configured
   push magnitude was zero.
2. The original failure distribution was dominated by the uncontrolled zero
   policy falling forward before any Phase 2 baseline existed, so those labels
   were not valid policy-improvement diagnostics.

Both issues were real.

## Evidence that the original push gate was invalid

Original formal run:

```text
/data/yuyangliu/runs/phase_1_push_recovery/2026-06-20_16-52-31
```

Old peak-window evidence:

```text
zero_strong_push:
  mean_pre_push_peak_speed_mps:  0.832243
  mean_post_push_peak_speed_mps: 1.327982
  mean_observed_delta_mps:       0.495739
  gate_pass: true

zero_no_push:
  mean_pre_push_peak_speed_mps:  0.701444
  mean_post_push_peak_speed_mps: 1.312673
  mean_observed_delta_mps:       0.611229
  gate_pass: true
```

The no-push profile had a larger legacy delta than the strong-push profile.
Therefore that gate measured falling/self-acceleration inside a broad window,
not the velocity kick itself.

## Fix

The evaluator now records a single-step planar root velocity-vector jump near
the configured push time:

```text
single_step_window_s: [0.9, 1.1]
mean_single_step_delta_mps
env_fraction_with_single_step_delta
single_step_gate_pass
```

The smoke summary also compares the strong-push single-step delta against the
no-push control:

```text
strong_push_delta >= no_push_delta + 0.08 m/s
```

The legacy broad-window values remain in JSON under
`legacy_peak_speed_window` for auditability, but they no longer drive the gate.

## Failure-classifier correction

Evaluation now disables vendor `base_height` and `bad_orientation` fall
terminations for eval rollouts only. This keeps PPO training smoke behavior
unchanged, but lets the protected classifier observe the trajectory instead of
being polluted by vendor auto-reset timing.

Zero/random policy failures are now recorded as uncontrolled-policy references.
They are not treated as push-recovery policy diagnostics until Phase 2 produces
a trained baseline.

## Corrected validation run

Remote run:

```text
/data/yuyangliu/runs/phase_1_push_recovery_gate_audit/2026-06-20_17-29-46
```

Machine-readable summary:

```text
/data/yuyangliu/runs/phase_1_push_recovery_gate_audit/2026-06-20_17-29-46/summary.json
```

Result: `PASS`.

### Corrected push evidence

```text
zero_strong_push:
  mean_single_step_delta_mps:          0.382979
  max_single_step_delta_mps:           0.817707
  env_fraction_with_single_step_delta: 0.9375
  gate_pass:                           true

zero_no_push:
  mean_single_step_delta_mps:          0.070190
  max_single_step_delta_mps:           0.075205
  env_fraction_with_single_step_delta: 0.0
  gate_pass:                           null
  reason: nonzero push is not configured; push-effect gate intentionally skipped

strong_push_delta_exceeds_no_push_control:
  strong: 0.382979
  no_push: 0.070190
  margin: 0.08
  pass: true
```

The corrected gate now proves that the configured velocity kick is present and
distinguishable from the no-push control.

### Corrected interpretation of failure labels

```text
zero_no_push:
  success_rate: 0.0
  failure_breakdown: fall_forward 64/64

random_strong_push:
  success_rate: 0.0
  failure_breakdown:
    fall_forward 63/64
    fall_side     1/64
```

These labels are true for uncontrolled zero/random policies, but they should
not be used as strategy-improvement evidence before Phase 2.

## Decision

The audit issues were real and were fixed immediately because they would affect
future Phase 2/3 decisions:

- Phase 2 needs trustworthy public eval metrics before measuring seed variance.
- Phase 3 verifier/smoke gates must not accept a fake push-effect proof.
- Future LLM/rule agents need explicit no-push controls and diagnostic notes to
  avoid overfitting to uncontrolled-policy failure labels.

Phase 1 remains complete after this correction, with the corrected gate audit as
the authoritative validation.

# Phase7 Implementation Plan - Prompt-Safe Eval Feedback

## Goal

Upgrade the prompt-facing eval feedback so the next candidate receives enough public evidence to act on rejections, while keeping sealed validation and runner-owned eval controls non-prompt and non-writable.

Phase7 must not weaken the Phase0-Phase6 boundaries:

- Shared layers may only project generic public evidence. Push-recovery semantics stay in the push-recovery task package.
- Prompt payloads must pass the public-safety guard and must not expose host paths, unprojected seed identifiers, sealed rollout internals, hidden profiles, checkpoint policy internals, or eval-control knobs.
- Existing accept gates remain runner-owned. The prompt may explain that gates exist and whether they passed, but it must not imply candidates can modify them.

## Current State

Phase6 already wires `behavior_quality_gate` into push-recovery public metrics and decision consumption. The remaining weakness is prompt feedback:

- `PushRecoveryPromptAdapter._public_eval_contract()` still describes a v1 contract and does not clearly name the full eval evidence structure.
- `public_feedback_summary` only reports scalar scores plus a task failure classifier. It does not summarize current-best comparison, gate regressions, or sealed validation status.
- `public_history` keeps failed gate names, but not comparison summaries, sealed validation summaries, or task-owned dominant-failure feedback.

Old Phase8E provides the useful pattern: next-round prompts need bounded feedback such as latest public-dev performance, dominant public failure, and a best-public pointer, while withholding public-accept-only details and real paths. Codex provides the same architectural lesson in another domain: prompt-facing context is a compressed projection, not raw backend payload.

## Implementation Steps

1. Add task-owned prompt feedback to push-recovery public metrics.
   - Build a `prompt_feedback` object from already-public split aggregates.
   - Include only bounded dominant-failure information: status, split, label, rate.
   - Do not include seed identifiers, per-episode trajectories, host paths, or sealed validation internals.
   - Add the same public-safe prompt feedback to the legacy single-eval builder so old projection paths do not drift.

2. Extend shared public history projection generically.
   - Add `comparison`, `dominant_failure`, and `sealed_validation_gate` to the allowlisted history entry surface.
   - Source `dominant_failure` only from task-owned `metrics.prompt_feedback.dominant_failure`; shared code must not classify push failures.
   - Summarize `comparison` with bounded generic fields: mode, scores, delta, threshold, delta_met, score source, counts, and current-best identifiers.
   - Summarize `sealed_validation_gate` with safe status fields only.

3. Upgrade push-recovery prompt contract and feedback summary.
   - Move contract schema to v2 and explicitly describe `public_dev`, `public_accept`, `sealed_validation_gate`, and `non_prompt_validation`.
   - State runner-owned read-only authority and list eval controls candidates must not request or modify.
   - Add a bounded `eval_feedback` summary containing current-best comparison, failed-gate trend, latest dominant failure, and sealed validation summary.

4. Add tests before and with implementation.
   - Public-result tests for `prompt_feedback` generation and seed/path non-leakage.
   - Prompt adapter tests for the v2 contract, full feedback summary, and public-safety guard.
   - Public history tests for comparison/sealed/dominant summaries and for rejecting arbitrary task-specific leakage from direct public context.

5. Verify and audit.
   - Run the focused Phase7 unit tests.
   - Run the broader eval/prompt safety suite touched by this change.
   - Static-audit shared code for push-recovery semantic leakage.
   - Confirm `PLAN/eval/Phase7` contains only this plan and the implementation record.

## Expected Result

The next-round prompt can explain why a candidate was rejected and where to improve, using enough public evidence to be actionable:

- latest round outcome and public score
- current-best comparison summary
- failed gates and gate trend
- dominant task failure from task-owned public projection
- sealed validation gate status and candidate isolation/checkpoint safety status

It still cannot see or influence eval seeds, holdout details, profiles, checkpoint policy, thresholds, runner-local artifact paths, or sealed rollout internals.

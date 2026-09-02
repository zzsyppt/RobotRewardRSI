# Phase 8 Implementation Plan: Run-Level Audit and Half-Upgrade Protection

Date: 2026-07-07

## Goal

Finish Phase 8 by making the run ledger itself auditable. The audit must prove that train, full eval, sealed validation, public projection safety, required gates, and decision comparison were connected end to end, not merely present as disconnected fields.

## Current Boundary Read

- `run_candidate_round()` already validates projected public eval before `decide_candidate()` and records `eval_incomplete` when projection fails.
- `decide_candidate()` consumes required gates only for policy-eligible candidates. Parameter-only, diagnostic-only, and engineering-only candidates may still be evaluated, but they are not policy accept candidates.
- `run_live_loop()` summary currently reports train/public result counts, but does not distinguish authored, trained, full-eval, sealed-validation, accepted, or eval-incomplete counts.
- `run_loop_audit()` currently checks prompt/public safety and curriculum/train-schedule boundaries, but does not audit full eval completion or decision gate consumption.
- `DryRunAuditRunner.run_eval()` still builds a legacy single-eval result and manually patches gates. That can mask protocol drift, so Phase 8 should route dry-run audit through the same full split public-result builder shape used by real eval.

## Reference Decisions

- From the old Phase8E system: preserve enough feedback for learning, but keep accept-only details and host paths out of prompt-facing history.
- From Codex cloud task UI patterns: summarize rich internal state into bounded public lines/counters instead of passing raw internal artifacts through UI/prompt boundaries.

## Implementation Steps

1. Add a reusable ledger audit helper under `agent_core`, because both live summaries and dry-run audit summaries need the same run-level accounting.
2. Keep the helper task-neutral: inspect public result structure, required gate names, decision summaries, and public projection safety. Do not parse push-specific fall/drift/action-rate metrics in the shared helper.
3. Produce a compact audit object:
   - counts: authored, trainable, trained, full-eval completed, sealed-validation completed, accepted, eval-incomplete.
   - failure buckets: authoring, train, public_eval, sealed_validation, motion_gate, behavior_gate, budget_gate, decision_threshold, non_policy_surface, accepted.
   - checks and violations keyed by round id.
   - compact per-round status only, no raw paths, seed details, or rollout payloads.
4. Wire the helper into `run_live_loop()` summary and `run_loop_audit()` summary.
5. Update dry-run eval so push-recovery owns the synthetic public-dev/public-accept summary builder, and `DryRunAuditRunner` receives that builder by injection. This keeps task-specific rollout metrics out of the runner common layer while still exercising `build_push_public_result_from_summaries()` instead of the legacy single-eval builder.
6. Add focused tests before implementation:
   - accepted record missing sealed validation fails audit.
   - accepted record with sealed validation present but not consumed fails audit.
   - trainable record without full eval is counted and attributed.
   - dry-run audit summary reports Phase 8 counts/checks and accepted records consume sealed/budget/motion/behavior gates.
   - live summary exposes the Phase 8 counters and compact audit.
7. Verify with focused unit/integration tests, then run the relevant broader test set and compile check.

## Non-Goals

- Do not change acceptance thresholds or task scoring.
- Do not expose public-accept failure labels in prompt feedback.
- Do not move push-recovery metric semantics into shared decision or audit code.
- Do not add another patched gate path that bypasses the full public-result builder.

# Phase 8 Implementation Record: Run-Level Audit and Half-Upgrade Protection

Date: 2026-07-07

## Completed

- Added `build_eval_run_audit()` as a shared ledger-derived audit helper.
- Wired the audit into live summaries and dry-run audit summaries.
- Added top-level run counters:
  - `authored_candidate_count`
  - `trainable_candidate_count`
  - `trained_candidate_count`
  - `full_eval_completed_count`
  - `sealed_validation_completed_count`
  - `accepted_count`
  - `eval_incomplete_count`
- Added compact failure concentration buckets:
  - `authoring`
  - `train`
  - `public_eval`
  - `sealed_validation`
  - `motion_gate`
  - `behavior_gate`
  - `budget_gate`
  - `decision_threshold`
  - `non_policy_surface`
  - `accepted`
- Added accepted-record checks that fail when:
  - sealed validation evidence is missing.
  - sealed validation is present but not required by acceptance.
  - sealed validation is present but not consumed by decision.
  - comparison threshold details are missing from policy decisions.
  - decision reason contradicts gate evidence.
- Migrated dry-run audit eval away from legacy single-eval projection. Push-recovery now owns the synthetic full-eval builder and injects it into `DryRunAuditRunner`, keeping task-specific rollout metric semantics out of the runner common layer.
- Updated fake eval fixtures to expose full public-dev/public-accept structure.

## Boundary Decisions

- The audit helper only inspects public ledger structure, public result splits, gate names/statuses, and decision summaries.
- Push-recovery-specific fields such as fall, drift, action rate, recovery signal, and base height remain task-owned.
- Non-policy surfaces are evaluated and summarized, but are not required to consume policy accept gates.
- Audit output is bounded: round ids, booleans, gate statuses, counts, and failure buckets only. It does not include raw seed ids, host paths, checkpoint paths, or rollout payloads.

## Verification

- `python -m unittest tests.unit.test_eval_run_audit`
- `python -m unittest tests.integration.test_dry_run_audit_loop`
- `python -m unittest tests.integration.test_live_loop`
- `python -m unittest tests.unit.test_push_recovery_task tests.unit.test_core_schema tests.unit.test_eval_run_audit`
- `python -m unittest discover -s tests`
- `python -m compileall src scripts tests`

Result: 318 tests passed, compileall passed.

## Post-Review Hardening: 2026-07-08

- Changed accepted decision-consumption audit to check the full `acceptance.required_pass_gates` set, while keeping baseline required gates as a separate evidence check.
- Excluded `verifier.status != "pass"` records from trainable/eval-incomplete accounting; static verifier failures are now attributed to authoring instead of train/public-eval.
- Moved dry-run train evidence behind a task-owned injected train-summary builder, matching the eval-result builder pattern.
- Removed `push_recovery_margin` from common dry-run train evidence and added prompt-level regression coverage that dry-run audit prompts do not contain it.
- Recomputed actual required gate statuses from `public_result.metrics` inside run audit and cross-checked decision-reported gate statuses against that evidence. This catches accepted ledgers where decision claims a required gate passed but eval evidence did not publish or did not pass that gate. Policy threshold requirements now use the recomputed metric-backed gate statuses instead of decision-reported statuses.

Verification:

- `python -m unittest tests.unit.test_eval_run_audit tests.integration.test_dry_run_audit_loop tests.integration.test_live_loop`
- `python -m unittest tests.unit.test_push_recovery_task tests.unit.test_core_schema tests.unit.test_prompt_projection_and_mechanism`
- `python -m compileall src scripts tests`

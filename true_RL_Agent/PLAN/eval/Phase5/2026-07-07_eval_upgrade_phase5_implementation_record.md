# Phase5 Eval Upgrade Implementation Record

## Scope

Phase5 moves policy acceptance away from a scalar `current_best_score` input and onto restored current-best eval statistics. The shared accept/reject layer now compares candidate public-accept score statistics against the latest accepted ledger record, consumes required eval gates explicitly, and records auditable comparison detail in the decision record.

## Implemented Changes

1. Added a generic current-best eval state contract.
   - `CurrentBestEvalStats` restores the latest accepted public eval stats from ledger records.
   - Bootstrap mode is used only when there is no accepted ledger record.
   - Accepted mode requires public-accept mean, standard error, eval protocol hash, and a passing sealed validation gate.
   - Incomplete accepted records fail closed instead of silently falling back to scalar best-score behavior.

2. Migrated shared decision logic from scalar best score to eval stats.
   - `decide_candidate()` now takes `CurrentBestEvalStats`.
   - Candidate acceptance uses `acceptance.score_stats.mean`, not top-level `public_score`.
   - Current-best comparison uses `CurrentBestEvalStats.public_accept_stats`, not candidate-provided `best_score_stats` hooks.
   - The threshold remains noise-aware: max of policy delta, task-projected minimum delta, and pooled standard error when available.

3. Made consumed gate evidence part of the decision record.
   - `DecisionRecord` now includes `comparison`, `consumed_gates`, and `gate_statuses`.
   - Policy acceptance requires `sealed_validation_gate` to be present in `acceptance.required_pass_gates`.
   - Required gates are consumed generically by name/status; task-specific gate semantics remain in task projection.

4. Wired current-best stats through round and loop execution.
   - `RoundInputs.current_best_score` was replaced by `current_best_eval_stats`.
   - `run_live_loop`, replay `AgentLoop`, and audit loop restore current-best eval stats from the ledger before each round.
   - Live summaries keep `final_best_score` for compatibility, but derive it from `CurrentBestEvalStats`.
   - Live summaries also include `final_best_eval_stats`.

5. Attached runner-owned eval protocol hash before decision.
   - `run_candidate_round()` attaches the executed `EvalJobSpec.evaluation_protocol().protocol_hash` to public score stats after task projection and before public validation/decision.
   - Candidate and current-best protocol hashes must match outside bootstrap mode.

6. Kept push-recovery semantics in the task layer.
   - `attach_push_budget_gate()` adds `sealed_validation_gate` to `acceptance.required_pass_gates` only when runner-projected metrics contain that gate.
   - Shared decision code does not inspect push, fall, drift, roll, pitch, base-height, or recovery semantics.
   - Fake and dry-run runners emit full-eval-shaped synthetic metrics only for tests and dry-run audit flows.

## TDD Evidence

Focused tests were added before the production implementation and initially failed for the expected reasons:

- `true_rl_agent.evaluation.current_best` did not exist.
- Live-loop summaries did not expose `final_best_eval_stats`.
- Push-recovery public projection did not require the runner-projected `sealed_validation_gate`.
- Bootstrap acceptance could still drift toward top-level `public_score` instead of score stats.

After implementation, the focused Phase5 suite passed.

## Verification

Commands run:

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest -v tests.unit.test_core_schema.CoreSchemaTests.test_bootstrap_absolute_floor_uses_candidate_score_stats_not_top_level_public_score
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_current_best_eval_stats tests.unit.test_core_schema tests.unit.test_push_recovery_task.PushRecoveryTaskTests.test_push_projection_requires_runner_projected_sealed_validation_gate tests.integration.test_fake_round.FakeRoundTests.test_round_decision_records_comparison_and_consumed_eval_gates tests.integration.test_live_loop.LiveLoopTests.test_live_loop_saves_prompt_payloads_and_ledger_summary
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover tests\unit
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover tests\integration
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q src tests
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\materialize_job.py --round-id phase5-sample --candidate-id sample-candidate --component reward --candidate-path artifact://sample-candidate
rg -n "current_best_score=|current_best_score:" src tests scripts
rg -n "fall|drift|push|motion|roll|pitch|base_height|recovery" src\true_rl_agent\agent_core\decision.py src\true_rl_agent\evaluation\current_best.py
git diff --check
```

Observed results:

- Single bootstrap regression: 1 test passed.
- Focused Phase5 suite: 22 tests passed.
- Unit discovery: 266 tests passed.
- Integration discovery: 33 tests passed.
- Compileall completed without errors.
- Materialized eval spec preserves `public_seed_count: 3`, `public_accept_seed_count: 3`, `sealed_validation_seed_count: 7`, and the fixed eval protocol hash.
- Static probe found no old `current_best_score=` or `current_best_score:` call surface.
- Static boundary probe found no task-specific push/motion/fall/drift/robot terms in shared decision/current-best modules.
- Diff whitespace check passed.

## Boundary Notes

- `sealed_validation_gate` is treated as a shared eval safety gate, not a task behavior semantic.
- Behavior gates such as motion quality remain task-projected required gates; shared decision only consumes their public status by name.
- Candidate summaries cannot supply current-best stats. Current-best stats are restored from accepted public ledger records only.
- Bootstrap mode still requires full candidate eval metadata, passing required gates, eval protocol hash, and the absolute floor.
- Existing compatibility fields such as `final_best_score` remain, but they are derived from current-best eval stats.

## Residual Scope

- Phase6 can broaden behavior gate coverage, but should keep each behavior gate owned by the task projection.
- Future multimodal feedback should attach as typed public feedback artifacts or task-owned gates, while shared decision continues to consume generic gate statuses and score stats.
- Older ledgers with accepted records that lack Phase5 stats will fail closed as `current_best_eval_stats_missing`; if old runs must be reused, they need an explicit migration/rehydration path rather than silent scalar fallback.

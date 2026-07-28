# Phase6 Eval Upgrade Implementation Record

## Scope

Phase6 upgraded push-recovery public eval from score-only feedback into task-owned behavior evidence and required gates. The shared decision layer remains task-neutral: it consumes only declared gate names and statuses from `acceptance.required_pass_gates`.

## Implemented

- Added task-owned `behavior_quality_gate` in `src/true_rl_agent/tasks/push_recovery/metrics.py`.
- Required the gate through `acceptance.required_pass_gates` for full public eval summaries.
- Made legacy `build_push_public_result()` and `attach_push_budget_gate()` fail-closed when behavior evidence is absent, so old single-eval shaped results cannot become accept-capable by only adding budget/seal gates.
- Added strict failure-breakdown parse validation for `rates`, `counts`, and flat payloads.
- Removed the lenient prompt-feedback split aggregation path for malformed failure breakdowns; feedback now uses the same strict parser before extracting labels.
- Added bounded behavior checks for action norm/rate, min base height, max pitch/roll, final xy drift, final planar speed, and recovery-window success rate.
- Added final planar speed and recovery-window success output in `scripts/isaaclab_rsl_rl_eval.py`.
- Added generic compressed `failed_gates` in public history from `decision.gate_statuses`; no push/fall/drift/base-height semantics were added to shared context.
- Added prompt-facing `public_eval_contract.metrics.behavior_quality_gate`.
- Updated dry-run and fake metrics with synthetic passing `behavior_quality_gate`.

## Review Fixes

- P0 legacy builder bypass: fixed by making legacy single-eval results fail-closed with required `behavior_quality_gate`, and by making task projection add a missing failing gate.
- P1 prompt contract gap: fixed by documenting `behavior_quality_gate` in the push task prompt contract.
- P2 recovery signal status: fixed so `recovery_signal_status` is `available` only when required recovery signal metrics are present and valid.
- P3 record missing: this file is the Phase6 implementation record.
- Additional audit fix: mixed valid/malformed `failure_breakdown` entries and metadata-only payloads now fail parseability instead of partially extracting labels.

## Tests Added

- Behavior gate requiredness and required-gate consumption.
- Missing behavior metric rejection.
- Malformed, mixed malformed, and metadata-only failure breakdown rejection.
- Strict split failure-breakdown extraction for prompt feedback.
- Out-of-range recovery success rate and negative nonnegative behavior metric rejection.
- Legacy single-eval projection cannot bypass `behavior_quality_gate`.
- Prompt contract includes `behavior_quality_gate`.
- Eval script emits final planar speed and recovery-window success metrics.
- Public history keeps only compressed failed gate name/status and does not expose task metric details.

## Verification

- `python -m unittest tests.unit.test_push_recovery_task`: 34 tests OK.
- Focused suite: `test_push_recovery_task`, `test_core_schema`, `test_current_best_eval_stats`, `test_memory_context_phase13`, `test_task_prompt_adapter`, `test_prompt_safety_boundaries`, `test_fake_round`, `test_live_loop`, `test_dry_run_audit_loop`: 112 tests OK.
- `python -m unittest discover tests\unit`: 276 tests OK.
- `python -m unittest discover tests\integration`: 33 tests OK. The argparse budget mismatch text is expected from an integration test.
- `python -m compileall -q src scripts tests`: exit 0.
- `git diff --check`: exit 0.
- Shared-layer task-term probe on `agent_core/decision.py`, `evaluation/current_best.py`, and `agent_core/context.py`: no matches.
- `scripts/materialize_job.py --round-id phase6-sample --candidate-id sample-candidate --component reward --candidate-path artifact://sample-candidate`: exit 0, budget remained `49152000`, eval protocol hash remained `b5b7a1e95a8cf528bbfca00153c1b6bf50b7eddcd4166d715407cb068b25c7d5`.

## Boundary Notes

- Push-recovery behavior semantics live only under the push task adapter/script/tests.
- Shared `decision.py` was not changed for push semantics.
- Shared public history exposes only generic failed gate names/statuses.
- Candidates and prompt payloads still cannot tune behavior thresholds or eval protocol counts.
- Phase6 did not change the push-recovery success predicate.

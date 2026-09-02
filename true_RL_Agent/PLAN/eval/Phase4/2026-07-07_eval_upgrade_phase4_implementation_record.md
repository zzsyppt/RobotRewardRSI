# Phase4 Eval Upgrade Implementation Record

## Scope

Phase4 thickens public-accept evidence before acceptance decisions. It keeps task-specific push-recovery semantics in the task layer and keeps shared decision logic limited to generic acceptance metadata.

## Implemented Changes

1. Public-accept budget defaults now converge on 3.
   - `PushRecoveryTaskConfig.public_accept_seed_count`
   - `IsaacLabRslRlBackendConfig.public_accept_seed_count`
   - `EvalJobSpec.public_accept_seed_count`
   - `scripts/isaaclab_rsl_rl_eval.py --public-accept-seeds`
   - `configs/backends/isaac_lab_rsl_rl.yaml`

2. Push-recovery public projection now exposes stronger public-accept statistics.
   - `public_accept.success_rate_stats.count`
   - `public_accept.success_rate_stats.episode_count`
   - `public_accept.success_rate_stats.mean`
   - `public_accept.success_rate_stats.std`
   - `public_accept.success_rate_stats.standard_error`
   - rounded public values only; raw seed ids remain hidden.

3. Push-recovery owns the new `accept_evidence_gate`.
   - Fails with `insufficient_accept_evidence` when observed public-accept count is below the configured minimum.
   - Fails with `accept_score_stats_untrusted` when required score statistics are missing or non-finite.
   - Is included in `acceptance.required_pass_gates` for full public split projections.

4. Shared decision logic consumes only generic metadata.
   - `acceptance.min_score_count`
   - `acceptance.score_stats.count`
   - `acceptance.score_stats.mean`
   - `acceptance.score_stats.standard_error`
   - optional candidate/best `eval_protocol_hash`
   - No push, fall, drift, motion, or robot-specific semantics were added to `agent_core.decision`.

5. Prompt-facing task contract now documents `accept_evidence_gate` as read-only public eval feedback.

## TDD Evidence

Initial focused tests were added before implementation and failed for the expected reasons:

- Low public-accept count still accepted a high public score.
- Fixed eval protocol still reported public-accept count 1.
- Public-accept stats did not include `episode_count`.
- Prompt eval contract did not mention `accept_evidence_gate`.

After implementation, the focused tests passed.

## Verification

Commands run:

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_core_schema tests.unit.test_push_recovery_task tests.unit.test_isaac_lab_backend tests.unit.test_materialize_job_script tests.unit.test_runner_jobs tests.integration.test_dry_run_audit_loop
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover tests\unit
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover tests\integration
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q src scripts tests
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\materialize_job.py --round-id phase4-sample --candidate-id sample-candidate --candidate-path artifact://sample-candidate
git diff --check
```

Observed results:

- Focused suite: 68 tests passed.
- Unit discovery: 258 tests passed.
- Integration discovery: 32 tests passed.
- Compileall completed without errors.
- Diff whitespace check passed.
- Materialized eval spec and eval CLI both include `public_accept_seed_count: 3` / `--public-accept-seeds 3`.

## Boundary Notes

- `current_best_score` remains scalar in Phase4. Full current-best statistical comparison is left for Phase5.
- The eval protocol schema still permits any positive public-accept count as a protocol value, but the push-recovery acceptance projection now requires the configured minimum count before acceptance.
- Legacy `trusted_seed_count` remains only as a constructor compatibility alias and dedicated compatibility test surface; production public dictionaries and generated materialized jobs use `sealed_validation_seed_count`.
- No raw seed ids, hidden eval profiles, checkpoint paths, evidence artifact paths, or runner-private files are exposed through public eval metrics.

## Residual Scope

Phase5 should move current-best comparison from scalar `current_best_score` toward persisted best-run score statistics so the protocol-hash mismatch check can become a normal live acceptance input instead of a forward-compatible guard.

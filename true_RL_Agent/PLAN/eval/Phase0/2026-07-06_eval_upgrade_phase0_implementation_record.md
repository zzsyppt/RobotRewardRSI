# Phase0 Evaluation Protocol Invariant Record

## Goal

Complete Phase0 of the eval upgrade by turning the eval protocol into a fixed, runner-owned/task-owned contract. The main target was protocol drift prevention: task config, `TaskSpec`, backend materialization, `EvalJobSpec`, seal hashing, and materialized job output must agree on the same canonical eval protocol fields before a candidate can enter train/eval.

Phase0 intentionally did not implement full eval rollout, sealed holdout execution, byte-level checkpoint loading, or current-best stats comparison.

## Implementation Plan Used

- Add a task-neutral `EvaluationProtocol` object.
- Add canonical eval protocol fields to push recovery task config.
- Project the same protocol fields into `TaskSpec.fixed_eval_protocol` and `TaskInvariantSpec`.
- Make backend materialization fail if task invariants or fixed eval protocol omit or drift from backend protocol fields.
- Make `EvalJobSpec` public output use canonical `sealed_validation_seed_count`.
- Keep `trusted_seed_count` only as a short-term legacy constructor alias.
- Make eval seal hash the canonical protocol payload instead of a legacy trusted seed field.
- Update materialization and live/audit entrypoints to pass canonical sealed validation count.
- Add focused RED tests for missing protocol module, missing materialized fields, protocol hash drift, and invariant mismatch.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/evaluation/protocol.py`
- `true_RL_Agent/src/true_rl_agent/evaluation/__init__.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/config.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/plugin.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py`
- `true_RL_Agent/src/true_rl_agent/backends/isaac_lab_rsl_rl.py`
- `true_RL_Agent/src/true_rl_agent/runner/specs.py`
- `true_RL_Agent/src/true_rl_agent/safety/eval_seal.py`
- `true_RL_Agent/scripts/materialize_job.py`
- `true_RL_Agent/scripts/run_loop.py`
- `true_RL_Agent/src/true_rl_agent/audit/loop_audit.py`
- `true_RL_Agent/tests/unit/test_evaluation_protocol.py`
- `true_RL_Agent/tests/unit/test_push_recovery_task.py`
- `true_RL_Agent/tests/unit/test_runner_jobs.py`
- `true_RL_Agent/tests/unit/test_materialize_job_script.py`
- `true_RL_Agent/tests/unit/test_isaac_lab_backend.py`

## Red Tests Observed

Before implementation:

- `true_rl_agent.evaluation.protocol` import failed because `EvaluationProtocol` did not exist.
- `scripts/materialize_job.py` output did not contain `eval.sealed_validation_seed_count`.
- `EvalJobSpec.to_public_dict()` did not expose `eval_protocol_hash`.
- backend materialization did not fail when `sealed_validation_seed_count` was missing from task invariants.
- backend materialization did not fail when backend sealed validation count drifted from task-owned invariants.

These failures confirmed the tests were exercising the Phase0 protocol-drift bug rather than only checking field presence.

## Implemented Behavior

- Added `EvaluationProtocol` with:
  - `eval_protocol_version`
  - `public_seed_count`
  - `public_accept_seed_count`
  - `sealed_validation_seed_count`
  - `eval_num_envs`
  - stable `eval_protocol_hash`
- `EvaluationProtocol` hash is derived only from fixed protocol fields, not candidate manifest data or provider JSON.
- `PushRecoveryTaskConfig.evaluation_protocol()` now creates the task-owned protocol object.
- `PushRecoveryTaskPlugin.task_spec` writes the same protocol projection into both fixed eval protocol and invariants.
- `PushRecoveryPromptAdapter` exposes canonical sealed validation naming in the prompt-safe fixed eval summary.
- `IsaacLabRslRlBackendAdapter` checks required protocol fields in both task invariants and fixed eval protocol before job materialization.
- Missing protocol fields now fail with `task_eval_protocol_mismatch:<field>`.
- Drifted protocol fields now fail before train/eval materialization.
- `EvalJobSpec` now stores `sealed_validation_seed_count` as the canonical field.
- `trusted_seed_count` remains only as an `InitVar` compatibility alias in `EvalJobSpec` and `IsaacLabRslRlBackendConfig`.
- `EvalJobSpec.to_public_dict()` emits `sealed_validation_seed_count`, `eval_protocol_version`, and `eval_protocol_hash`; it does not emit `trusted_seed_count`.
- `build_trusted_eval_seal()` hashes the canonical evaluation protocol payload.
- `run_loop.py` accepts `--sealed-validation-seed-count`, with `--trusted-seed-count` kept as a legacy CLI alias.
- `materialize_job.py` now uses the same task config instance for task spec and backend config.

## Boundary Notes

- Public/shared eval protocol remains task-neutral. It does not contain push recovery-specific labels such as fall, drift, base height, roll, pitch, push timing, or recovery windows.
- Task-specific semantics remain in the push recovery task layer and metrics projection.
- Follow-up fixture cleanup moved ordinary tests and fakes to `sealed_validation_seed_count`.
- Only two dedicated compatibility tests now construct objects with `trusted_seed_count`: one for `EvalJobSpec`, one for `IsaacLabRslRlBackendConfig`.
- Prompt-safety negative samples and `assertNotIn("trusted_seed_count", ...)` checks still mention the old token intentionally.
- Phase0 keeps eval CLI `--candidate` behavior unchanged; removing that dependency is Phase1.
- Phase0 does not claim byte-level checkpoint safe-load. The seal is only widened to hash canonical protocol fields; byte-level evidence is Phase2.
- Phase0 does not implement sealed holdout rollout. That is Phase3.
- Phase0 does not replace scalar current-best comparison with full eval stats. That is Phase5.

## Verification

Commands run with bundled Python:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_evaluation_protocol tests.unit.test_push_recovery_task tests.unit.test_runner_jobs tests.unit.test_materialize_job_script tests.unit.test_isaac_lab_backend
```

Result:

- `Ran 41 tests`
- `OK`

Full unit suite:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover tests\unit
```

Result:

- `Ran 219 tests`
- `OK`

Integration suite:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover tests\integration
```

Result:

- `Ran 32 tests`
- `OK`

Materialization smoke check:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\materialize_job.py --round-id phase0-sample --candidate-id sample-candidate --candidate-path artifact://sample-candidate
```

Observed canonical eval output includes:

- `sealed_validation_seed_count: 7`
- `eval_protocol_hash: cedd4f879677c877de748c12d1fdfaa8ad2394eb924f30c41d9c2738df9ed21a`
- no `trusted_seed_count` in the public eval projection

## Remaining Scope

Phase1 should remove or downgrade eval CLI candidate path dependency and introduce an explicit sealed validation path that rejects candidate package input. Phase2 should replace structural checkpoint safe-load with byte-level eval-process checkpoint safety evidence.

# Phase3 Eval Upgrade Implementation Record

## Implemented

- Added task-neutral public projection guards in `true_rl_agent.evaluation.public_projection`.
- Made `CheckpointRef.to_public_dict()` omit metadata by default, with explicit `include_metadata=True` opt-in for internal/debug callers.
- Made `TrainJobResult.to_public_dict()` project raw train summaries into public-safe budget/candidate evidence instead of copying raw summaries.
- Made `PublicEvalResult.to_public_dict()` fail closed if metrics contain raw paths, raw seed markers, hidden/trusted/private markers, or forbidden raw-evidence keys.
- Updated `run_candidate_round()` so train success followed by eval runner failure records `eval_incomplete`, preserves train evidence, and cannot accept.
- Added private artifact refs for candidate package, runner train/eval summaries, runner work dirs, checkpoint output, local checkpoint path, and source checkpoint path.
- Added round-boundary public-section safety assertions for jobs, train result, public result, runner error, and decision.
- Kept push-recovery split semantics in task/eval-script code. Shared code only handles task-neutral projection, artifact refs, and safety checks.

## Boundary Notes

- Raw eval/train artifacts are still available for audit through private artifact refs.
- Public ledger sections no longer carry checkpoint metadata such as `local_path` or `source_latest_checkpoint`.
- Public train summary keeps budget evidence and compact candidate-loaded evidence, but drops `cwd`, `config_path`, `log_dir`, `checkpoint_output`, per-run paths, and raw configuration.
- `artifact://...` checkpoint URIs remain allowed public identifiers; local Windows/POSIX paths remain forbidden in public projections.
- Candidate runtime errors and eval incomplete errors now use the same public runner-error projector.

## Tests Added

- `tests/unit/test_eval_public_projection.py`
  - checkpoint metadata omitted by default with explicit opt-in.
  - train result public projection keeps budget evidence and drops raw runner paths.
  - public eval result rejects raw paths, raw seed markers, and hidden markers.
- `tests/unit/test_push_recovery_task.py`
  - round ledger public sections do not leak runner local checkpoint metadata.
  - private artifact refs retain raw runner checkpoint path.
  - train success plus eval missing records `eval_incomplete` without accept.

## Verification

- `python -m unittest tests.unit.test_eval_public_projection tests.unit.test_push_recovery_task.PushRecoveryTaskTests.test_push_round_rejects_schedule_only_and_accepts_code_with_runtime_evidence tests.unit.test_push_recovery_task.PushRecoveryTaskTests.test_train_success_with_missing_eval_records_eval_incomplete_without_accept`
  - Red before implementation: metadata/path leaks and eval error bubbling.
  - Green after implementation: 5 tests passed.
- `python -m unittest discover tests\unit`
  - 253 tests passed.
- `python -m unittest discover tests\integration`
  - 32 tests passed. Existing CLI budget-mismatch usage text is expected by the suite.
- `git diff --check`
  - Passed.

## Residual Scope

- Phase3 establishes public/private projection and fail-closed eval completion boundaries.
- A future phase can deepen sealed validation from checkpoint/isolation gate evidence into additional independent rollout modalities without changing the shared public/private projection contract.


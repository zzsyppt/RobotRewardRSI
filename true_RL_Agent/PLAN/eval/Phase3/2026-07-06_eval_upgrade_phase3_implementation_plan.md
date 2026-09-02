# Phase3 Eval Upgrade Implementation Plan

## Scope

Implement Phase3 as the public/private eval-output boundary for the current architecture:

- Every verified trainable round must execute eval before any accept decision.
- Runner-owned raw train/eval artifacts may remain available for audit, but only as private artifact refs.
- Ledger and prompt-facing records must use scrubbed public projections.
- Shared code must stay task-neutral. Push-recovery split names, seed policy, success-rate aggregation, and motion gates stay in task/eval-script code.

## Current Architecture Findings

- `run_candidate_round()` already executes train, binds eval to the sealed checkpoint, then executes eval before `decide_candidate()`.
- `scripts/isaaclab_rsl_rl_eval.py` already runs `public_dev` and `public_accept` rollouts and projects them through push-recovery metrics.
- Raw summaries can contain audit-only data: `cwd`, host/platform, config paths, per-episode detail, checkpoint local paths, and runner log paths.
- Current ledger writes still serialize `jobs.eval.checkpoint.metadata` and `train_result.checkpoint.metadata` through default `to_public_dict()` paths.
- `TrainJobResult.to_public_dict()` still copies the whole train summary into the ledger public record.
- `ExperimentRecord` already has a public/private split, but the round ledger needs to feed it cleaner public records and richer private artifact refs.

## Design

1. Add shared task-neutral public projection helpers.
   - Public checkpoint projection omits checkpoint metadata by default.
   - Public train-result projection includes checkpoint identity and compact runner/public-safe summary only.
   - Public eval-result projection validates that projected metrics do not contain raw paths, hidden/trusted/private keys, or raw seed markers.
   - Artifact refs are created from runner metadata and checkpoint metadata, not from task-specific fields.

2. Keep task semantics out of shared projection.
   - Shared code may recognize generic private runner evidence keys such as `summary_path`, `checkpoint_output_path`, `work_dir`, `local_path`, and `source_latest_checkpoint`.
   - Shared code must not know `public_dev`, `public_accept`, success predicates, fall metrics, or push-recovery config.
   - Task metrics remain responsible for converting raw rollout summaries into public metrics.

3. Fail closed on incomplete eval.
   - If train succeeds but eval execution/projection raises a runner execution error, the round records `eval_incomplete` and cannot accept.
   - Candidate runtime failures remain classified as `candidate_runtime_failed`.
   - Infrastructure/eval ingestion failures get a public-safe runner error summary with raw paths scrubbed.

4. Write richer private artifact refs.
   - Always keep the candidate package as a private ref.
   - Add private refs for train/eval summary paths, checkpoint output/local paths, and runner work dirs when present.
   - Keep private refs in ledger private artifact area only; public prompt history sees only redactions/counts.

5. Enforce public safety at the round boundary.
   - Before appending a ledger record, assert public-facing ledger sections are prompt-safe.
   - Reuse the common public safety layer instead of adding task-local deny lists.
   - Any public projection leak is a fail-closed implementation error, not a silent sanitize-at-prompt patch.

## Tests First

1. Add a regression test showing the current ledger leaks runner checkpoint metadata/local paths through `jobs.eval` and `train_result`.
2. Add a regression test showing train success plus eval `RunnerExecutionError` records `eval_incomplete`, preserves train evidence, and cannot accept.
3. Add projection-unit tests for checkpoint metadata stripping and public projection rejection of path/raw-seed/hidden markers.
4. Extend an existing round test to assert private artifact refs contain runner-owned paths while public ledger fields do not.

## Implementation Steps

1. Add task-neutral projection helpers under `true_rl_agent.evaluation` or `true_rl_agent.runner`.
2. Update `CheckpointRef`, `EvalJobSpec`, and `TrainJobResult` public serialization so prompt-facing paths use scrubbed checkpoint projection.
3. Update `run_candidate_round()` to:
   - catch non-candidate `RunnerExecutionError` during eval as `eval_incomplete`;
   - build private artifact refs from train/eval metadata;
   - write only scrubbed public projections for jobs/train/public result;
   - assert public-facing payload safety before ledger append.
4. Keep raw summaries available through existing runner summary files and private refs; do not inline per-episode/raw host/cwd data into public ledger.
5. Create Phase3 implementation record after verification.

## Verification

- Focused unit tests for projection and round ledger behavior.
- Existing subprocess/eval safety focused tests.
- Broader relevant suite covering runner jobs, push-recovery task, memory context, and prompt safety.


# True RL Agent Extension and Boundary Guide

This guide defines the v1 path for extending `true_rl_agent` without weakening the first-version safety and attribution boundaries.

## Add A TaskPlugin

Use this path for new tasks such as redirection, reach, stair climbing, or another locomotion recovery task:

1. Create a task-owned package under `src/true_rl_agent/tasks/<task_id>/`.
2. Add a config object for task invariants and fixed protocol values.
3. Implement `TaskPlugin.task_spec` only. The task plugin declares `TaskSpec`, `TaskInvariantSpec`, `fixed_train_protocol`, `fixed_eval_protocol`, supported policy initializations, and optional future curriculum capability metadata.
4. Add public metric projection and a task-owned failure classifier.
5. Optionally add a `TaskMemoryAdapter` that compresses only public eval metrics into a task-specific public summary.
6. Register the plugin through `TaskRegistry`.
7. Add tests for registration, invariant rejection, public metric projection, and the task-specific memory summary.

Task plugins must not create training commands, run training, choose checkpoints, change eval seeds, make accept/reject decisions, append the ledger, or expose private/trusted eval artifacts to prompt context.

## Add A BackendAdapter

Use this path for a new backend or a new Isaac Lab materialization mode:

1. Add a backend package under `src/true_rl_agent/backends/`.
2. Implement `BackendAdapter.backend_spec` with supported tasks and supported policy initialization modes.
3. Implement `materialize_jobs(...)` so it returns typed `TrainJobSpec` and `EvalJobSpec`.
4. Preserve `TaskInvariantSpec` in the returned job bundle; do not weaken task semantics.
5. Generate CLI job metadata instead of importing or executing training scripts in the adapter.
6. Keep public seed count and trusted seed count in job specs; do not project trusted seed data into public prompt context.
7. Register the adapter through `BackendRegistry`.
8. Add dry-run tests for random initialization, warm_start materialization when supported, unsupported task failure, unsupported initialization failure, and fixed protocol metadata.

Backend adapters materialize jobs. They do not run training, run eval, select accepted candidates, write ledger records, or expose decision authority.

## Baseline Checkpoint Checklist

Baseline checkpoint integration is system-level policy initialization, not a task-specific shortcut.

Checklist:

- Create a `BaselineManifest`.
- Set `baseline_id`, `task_id`, and runner-sealed warm_start `CheckpointRef`.
- Record provenance with `source_run_id`, `source_task_id`, and `created_by`.
- Record compatibility: backend id, robot, observation schema hash, action schema hash, and policy architecture.
- Validate with `validate_baseline_compatibility(...)` before materializing a warm_start job.
- Require `checkpoint.role == "warm_start"`.
- Require `checkpoint.sealed == True`.
- Require a `sha256:` checksum.
- Keep training start checkpoint separate from eval checkpoint.
- Add compatibility failure tests for backend mismatch, robot mismatch, schema mismatch, policy architecture mismatch, unsupported warm_start, unsealed checkpoint, and invalid checksum.

The checkpoint selector is runner-owned and must not enter LLM prompt payloads.

## Random Vs Warm_Start

Use `PolicyInitialization.random()` when the task/backend pair is meant to train from a random policy. The first push recovery delivery path must support this mode.

Use `PolicyInitialization.warm_start(...)` only when a validated baseline manifest exists and the backend supports warm_start. Warm-start loading must be materialized as an explicit train start checkpoint, not as an eval checkpoint and not as candidate-controlled metadata.

If a task supports warm_start but the backend does not, compatibility must fail before training.

## Prompt-Facing Capability Admission

Prompt-facing capabilities are narrower than internal runner contracts.

A capability may be shown as prompt-facing only when all of these are true:

- It has a registered schema and stable name.
- It is wired to runtime evidence or a controlled candidate entrypoint.
- It has a public projection that excludes hidden eval, private seeds, trusted seeds, secrets, and checkpoint selectors.
- It cannot directly change task invariants, eval protocol, seed policy, budget class, checkpoint selection, verifier behavior, or trusted eval logic.
- It is covered by verifier tests and prompt payload leak tests.

Unwired or future capability metadata may exist internally, but the prompt must mark it as unavailable, disabled_by_phase, not_prompt_writable, or not effective now. First-version curriculum schedule proposals are disabled_by_phase and must not enter `TrainJobSpec`.

## Trusted Eval And Runner-Owned Boundary

Trusted eval is runner-owned evidence. It can be used for offline audit, private decision evidence, or final reporting, but it must not become candidate proposal input.

Runner-owned fields include:

- trusted eval results and trusted seed data.
- private seed lists.
- checkpoint selector logic.
- eval protocol and success predicate.
- task invariants such as fall threshold, episode length, push magnitude, push timing, and budget class.
- safe-load and checkpoint sealing.
- backend command construction.
- ledger append and final decision records.

Public prompt context may include public eval scores, public failure breakdowns, public diagnostic summaries, public runtime activation summaries, public curriculum lineage, and task-specific public summaries produced by `TaskMemoryAdapter`.

## Redirection And Reach Task Path

Future redirection or reach tasks should follow the same task path:

- `src/true_rl_agent/tasks/redirection/` or `src/true_rl_agent/tasks/reach/`.
- Task-owned config with invariant target bounds, episode limits, action/observation assumptions, and success predicates.
- Public metrics such as target hit rate, tracking error, stability margin, recovery delay, or contact quality.
- Task-owned failure classifier.
- Optional task-specific public memory summary.
- Fixed eval bins or target sets declared in `TaskSpec.fixed_eval_protocol`.
- Candidate-facing controls limited to wired `RewardTerm`, `DiagnosticTerm`, or another explicitly approved code-level surface.

Do not add task-specific branches to AgentCore. Add task semantics through TaskPlugin, backend materialization through BackendAdapter, and public feedback through public projection or TaskMemoryAdapter.

## Testing Matrix

Required coverage currently maps to these test files:

- `tests/unit/test_project_boundaries.py`: independent package and no parent runtime dependency.
- `tests/unit/test_plugin_registries.py`: TaskRegistry, BackendRegistry, compatibility, and plugin authority boundaries.
- `tests/unit/test_core_schema.py`: core schema, random/warm_start policy initialization, decision outcomes, public result projection, and task invariant checks.
- `tests/unit/test_baseline_management.py`: baseline manifest, warm_start compatibility, safe-load, and failure cases.
- `tests/unit/test_runner_jobs.py`: job specs, artifact layout, checkpoint sealing, fixed budget, isolation policy, and trusted eval seal.
- `tests/unit/test_candidate_package.py`: candidate loader, static verifier, controlled entrypoint, and candidate authority limits.
- `tests/unit/test_prompt_projection_and_mechanism.py`: prompt-facing controls, public reward inputs, mechanism evidence gate, and anti-tuning shadow.
- `tests/unit/test_curriculum_phase9.py`: CurriculumPlan modes, disabled LLM curriculum proposals, capability registry, and schedule-only classification.
- `tests/integration/test_agent_core_loop.py`: replay loop and public context construction.
- `tests/integration/test_fake_round.py`: fake vertical round from candidate to ledger.
- `tests/unit/test_push_recovery_task.py`: push task invariants, push job materialization, random path, schedule-only reject, code-level accept with runtime evidence, and public push metrics.
- `tests/unit/test_memory_context_phase13.py`: public-only next context, prompt payload saving, private artifact separation, and task memory adapter extension.
- `tests/unit/test_stair_extension_guide.py`: stair extension guide/schema without runtime plugin or train command.

New tasks or backends must extend this matrix rather than replacing it.

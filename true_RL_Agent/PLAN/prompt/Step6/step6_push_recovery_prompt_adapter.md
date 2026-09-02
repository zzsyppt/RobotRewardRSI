# Step 6 Push Recovery Prompt Adapter

## Scope

This record completes Step 6 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: implement the task-owned prompt adapter for `unitree_push_recovery`.

Step6 does not change shared accept/reject logic, public score calculation, provider parsing, training budget, eval protocol, checkpoint policy, PPO, seed policy, or candidate package schema. It only fills task-owned prompt sections through `TaskPlugin.prompt_adapter()`.

## Implementation Plan Used

1. Add failing tests to prove `PushRecoveryTaskPlugin.prompt_adapter()` still returned the neutral fallback.
2. Add failing tests that the full prompt payload did not contain push-owned `reward_inputs_contract` fields.
3. Add failing tests that public history/runtime feedback were not summarized by the push task adapter.
4. Add `true_rl_agent.tasks.push_recovery.prompt_adapter.PushRecoveryPromptAdapter`.
5. Override `PushRecoveryTaskPlugin.prompt_adapter()` to return the push adapter.
6. Export `PushRecoveryPromptAdapter` from `true_rl_agent.tasks.push_recovery`.
7. Re-run push task, non-push isolation, prompt assembler, provider, live loop, and full unittest verification.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/plugin.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/__init__.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/metrics.py`
- `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/memory.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/tests/unit/test_push_recovery_task.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- `true_RL_Agent/PLAN/prompt/prompt_upgrade_execution_plan.md`

## Sections Implemented

The push adapter now provides all allowed task-owned sections:

- `task`
- `reward_inputs_contract`
- `public_eval_contract`
- `public_feedback_summary`
- `runtime_feedback_summary`
- `known_good_examples`
- `selection_guidance`

The shared assembler promotes these into canonical top-level prompt sections.

## Boundary Decisions

- Push-specific semantics live only under `tasks/push_recovery`.
- Shared prompt code still does not interpret push recovery success, failure taxonomy, or push RewardInputs fields.
- `push_active` and `push_vec_public` are exposed only in task-owned `reward_inputs_contract`; the shared legacy `reward_inputs` alias still omits them.
- `push_active` and `push_vec_public` are explicitly marked as runner-owned signals with current generic-runtime zero fallback and not guaranteed to be nonzero.
- The fixed train/eval protocol is projected as read-only public task context. Candidate cannot modify budget, eval protocol, success metric, seed policy, checkpoint policy, runner code, PPO, task invariants, or push distribution.
- The prompt does not expose raw seed ids, non-prompt eval details, local/server absolute paths, task-private artifacts, or `checkpoint_selector` control.
- No curriculum/stage interface was added to prompt sections.

## Red Tests Observed

Before implementation:

- `test_push_prompt_adapter_projects_task_specific_sections_without_runner_control` failed because `PushRecoveryTaskPlugin.prompt_adapter()` returned `neutral_fallback`.
- `test_push_full_prompt_payload_uses_task_owned_reward_inputs_contract` failed because the promoted `reward_inputs_contract` had no push fields.
- `test_push_prompt_adapter_summarizes_public_history_and_runtime_feedback` failed because the neutral fallback did not provide latest public feedback/runtime summaries.

These failures matched the Step6 gap.

## Review Follow-up

The post-Step6 review identified three real issues and they were fixed in the task/runtime prompt path:

- Real `build_push_public_result_from_summaries()` metrics store failure breakdowns under split `seed_results`, not top-level `failure_breakdown`; `metrics.extract_push_failure_breakdown()` now supports both shapes and is reused by `classify_push_failure()` and `PushTaskMemoryAdapter`.
- Runtime candidate load evidence from real train summaries is under `summary.candidate.enabled`; `runtime_feedback_summary.candidate_package_loaded` now reads that path while preserving the older top-level fallback flags.
- The OpenAI-compatible provider system prompt now distinguishes non-prompt/private sources from prompt-visible public task/eval summaries. Prompt-visible summaries are explicitly read-only evidence, not candidate-controllable targets.

## Verification

Focused commands run with bundled Python:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_push_recovery_task
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_task_prompt_adapter true_RL_Agent.tests.unit.test_prompt_projection_and_mechanism true_RL_Agent.tests.unit.test_memory_context_phase13
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.integration.test_live_loop true_RL_Agent.tests.unit.test_openai_compatible_provider true_RL_Agent.tests.integration.test_dry_run_audit_loop
```

Focused results:

- `test_push_recovery_task`: `Ran 17 tests`, `OK`
- `test_task_prompt_adapter` + `test_prompt_projection_and_mechanism` + `test_memory_context_phase13`: `Ran 22 tests`, `OK`
- `test_live_loop` + `test_openai_compatible_provider` + `test_dry_run_audit_loop`: `Ran 26 tests`, `OK`

Full-suite and diff hygiene:

- `python -m unittest discover -s true_RL_Agent\tests`: `Ran 150 tests`, `OK`
- `git diff --check`: no whitespace or patch formatting issues

Direct prompt probe:

- serialized first-round push prompt payload: about 23 KiB
- `task.task_id`: `unitree_push_recovery`
- `reward_inputs_contract.fields`: 17 fields
- shared legacy `reward_inputs.fields` is empty and does not re-export task-owned fields such as `push_active`
- task-owned `reward_inputs_contract.fields` includes `push_active`
- first-round `runtime_feedback_summary.status`: `no_prior_runtime_feedback`

Known-good prompt examples are also covered by `test_push_prompt_known_good_examples_verify_as_candidate_packages`, which materializes them as temporary candidate packages and checks them with `StaticCandidateVerifier`:

- RewardTerm example: `pass`, derived change set `reward_callable`
- DiagnosticTerm example: `pass`, derived change set `diagnostic`
- ScheduleTerm example: `pass`, derived change set `reward_weight`

## Remaining Scope

- Step7 must expand the candidate package contract beyond the current `components_py` JSON shape.
- Step8 must improve bounded public history and generic runtime feedback projection across tasks.
- Step10 can remove or downgrade legacy prompt aliases after provider and downstream consumers read canonical sections directly.
- Step12/13 must validate real server-generated prompt artifacts and provider user payload capture.

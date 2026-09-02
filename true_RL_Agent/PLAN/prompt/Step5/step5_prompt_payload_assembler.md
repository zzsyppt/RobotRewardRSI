# Step 5 Prompt Payload Assembler

## Scope

This record completes Step 5 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: replace the thin prompt projection with a stable prompt payload assembler that merges generic prompt contracts and task-owned sections in a readable order.

Step5 does not implement the push recovery prompt adapter content. Push-specific RewardInputs semantics, public eval semantics, failure taxonomy, and task-specific examples remain Step6 work.

## Implementation Plan Used

1. Add failing tests for stable top-level prompt order.
2. Add failing tests that `public_context.task_prompt_sections` is promoted into top-level canonical sections.
3. Add failing tests that repair feedback is promoted to `provider_or_runtime_repair_feedback`.
4. Add failing tests that saved prompt JSON preserves insertion order instead of sorting keys alphabetically.
5. Move prompt assembly into `true_rl_agent.prompting.payload`.
6. Keep `true_rl_agent.candidates.prompt_projection` as a backward-compatible import/export layer.
7. Preserve legacy aliases after canonical sections so current provider and tests keep working while newer code reads canonical sections.
8. Re-run focused tests, audit loop tests, and full unittest discovery.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/prompting/__init__.py`
- `true_RL_Agent/src/true_rl_agent/prompting/payload.py`
- `true_RL_Agent/src/true_rl_agent/prompting/public_safety.py`
- `true_RL_Agent/src/true_rl_agent/prompting/task_sections.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/provider_contract.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/public_safety.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/context.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/src/true_rl_agent/candidates/prompt_projection.py`
- `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- `true_RL_Agent/src/true_rl_agent/tasks/prompt_adapter.py`
- `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- `true_RL_Agent/tests/unit/test_task_prompt_adapter.py`
- `true_RL_Agent/tests/unit/test_memory_context_phase13.py`
- `true_RL_Agent/tests/integration/test_live_loop.py`
- `true_RL_Agent/PLAN/prompt/prompt_upgrade_execution_plan.md`

## Canonical Payload Order

The assembler now emits the canonical sections first:

1. `task`
2. `loop_contract`
3. `public_feedback_summary`
4. `runtime_feedback_summary`
5. `required_candidate_package`
6. `reward_inputs_contract`
7. `public_eval_contract`
8. `control_effectiveness_contract`
9. `public_history`
10. `known_good_examples`
11. `selection_guidance`
12. `provider_or_runtime_repair_feedback` when repair feedback exists
13. `schema_version`

The payload schema is now `true_rl_agent.prompt_payload.v2`.

## Legacy Aliases

The following keys remain after canonical sections for compatibility:

- `public_context`
- `reward_inputs`
- `candidate_allowed_controls`
- `candidate_component_contract`
- `anti_tuning_notice`
- `runner_owned_extensions`

These are transitional aliases. The authoritative task semantics now live in canonical task-owned sections, especially `reward_inputs_contract` and `public_eval_contract`.

Legacy `reward_inputs` exit condition:

- Do not reintroduce a shared task-specific reward input denylist; the deprecated `reward_inputs` alias must remain an empty compatibility pointer to `reward_inputs_contract`, not a copy of task-owned field names.
- New task-specific RewardInputs semantics must be exposed only through task-owned `reward_inputs_contract`.
- After provider/system prompt and downstream consumers read canonical sections directly, Step10 should remove or downgrade the legacy `reward_inputs` alias rather than lengthening the exclusion list.

## Boundary Decisions

- `public_context.task_prompt_sections` is removed after extraction; task-owned sections appear at the top level.
- Shared top-level `reward_inputs` remains a legacy alias only and omits task-owned fields such as `push_active` and `push_vec_public`.
- `reward_inputs_contract` is sourced from the task prompt adapter when present; otherwise it explicitly says the task-owned contract was not provided.
- Shared code provides generic `loop_contract`, `required_candidate_package`, `control_effectiveness_contract`, `public_history`, and fallback summaries.
- Shared code still does not contain push recovery success predicate, push failure taxonomy, or push RewardInputs field semantics.
- `PublicContextBuilder.save_prompt_payload()` no longer uses `sort_keys=True`, so saved artifacts preserve the assembler order.

## Blocking Review Fix

Direct calls to `build_prompt_payload()` now reuse the same task section sanitizer used by `TaskPromptAdapter`.
The sanitizer was moved into `true_rl_agent.prompting.task_sections`, and the public-only guard was moved into
`true_rl_agent.prompting.public_safety`. This prevents the assembler entry point from drifting away from Step4
boundaries.

Covered regressions:

- Task prompt sections passed directly through `public_context.task_prompt_sections` scrub Windows and server absolute paths.
- Task prompt sections passed directly through the assembler reject oversized content with the same byte cap used by adapters.
- `tasks.prompt_adapter` re-exports the same sanitizer and constants instead of maintaining a second implementation.
- Provider `declared_changed_components` now has two explicit layers in `agent_core.provider_contract`: prompt-advertised values only include currently wired candidate components, while the provider materializer keeps a legacy parse allowlist for backward compatibility.
- The provider system prompt no longer hardcodes push recovery semantics; task semantics stay in the user payload.
- `PROMPT_PAYLOAD_SECTION_ORDER` now drives canonical payload assembly instead of being a detached declaration.
- Public forbidden key/value rules now come from `agent_core.public_safety`; prompt-facing imports re-export the same implementation.

## Red Tests Observed

Before implementation:

- Stable top-level order test failed because payload still started with `schema_version`, `public_context`, and `reward_inputs`.
- Task-owned section promotion tests failed with `KeyError: 'task'`.
- Repair feedback promotion failed with `KeyError: 'provider_or_runtime_repair_feedback'`.
- Saved artifact order test failed because `sort_keys=True` made the file start with `anti_tuning_notice`.
- Blocking review regression tests failed because direct `build_prompt_payload()` calls leaked `D:/humanoid_agent/...`
  and `/data/yuyangliu/...` paths and did not reject oversized task sections.
- Provider contract regression first exposed prompt/materializer drift, then a follow-up review found the initial single source was too wide: parser-only legacy values such as `control_logic`, `recovery_logic`, `failure_classifier`, `logging`, and `public_projection` were being advertised to the model.
- System prompt regression failed because the provider system message still contained hardcoded `push recovery`.

These failures matched the Step5 gap.

## Verification

Focused commands run with bundled Python:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_prompt_projection_and_mechanism
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_task_prompt_adapter
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_memory_context_phase13
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.integration.test_live_loop true_RL_Agent.tests.unit.test_openai_compatible_provider true_RL_Agent.tests.integration.test_dry_run_audit_loop
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s true_RL_Agent\tests
```

Focused results:

- `test_prompt_projection_and_mechanism`: `Ran 9 tests`, `OK`
- After provider contract review fix, `test_prompt_projection_and_mechanism`: `Ran 10 tests`, `OK`
- `test_task_prompt_adapter`: `Ran 6 tests`, `OK`
- `test_memory_context_phase13`: `Ran 6 tests`, `OK`
- `test_openai_compatible_provider`: `Ran 12 tests`, `OK`
- `test_prompt_projection_and_mechanism` + `test_openai_compatible_provider` + `test_candidate_package`: `Ran 33 tests`, `OK`
- `test_live_loop` + `test_openai_compatible_provider` + `test_dry_run_audit_loop`: `Ran 25 tests`, `OK`

Full-suite result:

- `unittest discover -s true_RL_Agent\tests`: `Ran 144 tests`, `OK`

## Remaining Scope

- Step6 must implement the real push recovery prompt adapter.
- Step7 must expand candidate package output beyond current `components_py` JSON contract.
- Step8 must upgrade bounded public history and runtime feedback using richer ledger evidence.
- Step10 can remove or narrow legacy aliases once provider/system prompt and downstream consumers read canonical sections directly.

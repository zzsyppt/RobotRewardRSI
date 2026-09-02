# Step 4 Task-Owned Prompt Adapter Interface

## Scope

This record completes Step 4 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: define the task-owned prompt adapter interface and wire it through `TaskPlugin` as the single task semantic entry point.

Step4 does not implement the push recovery prompt adapter content. Push-specific RewardInputs semantics, public eval semantics, failure taxonomy, known-good examples, and selection guidance remain Step6 work under `tasks/push_recovery`.

## Current-State Decisions

- `TaskMemoryAdapter` remains the extension point for public history summary only.
- `TaskPromptAdapter` is a separate task-owned prompt semantics interface.
- `TaskPlugin.prompt_adapter()` is the only registration/lookup entry point for task prompt semantics.
- Tasks without a custom prompt adapter receive a neutral fallback adapter.
- Step4 keeps task prompt sections under `public_context.task_prompt_sections` as a transitional location. Step5 will move/merge these sections into a stable top-level prompt assembler.
- The shared layer validates and sanitizes adapter output but does not interpret task-specific success semantics.

## Implementation Plan Used

1. Add failing tests for the missing `true_rl_agent.tasks.prompt_adapter` module and default `TaskPlugin.prompt_adapter()` entry point.
2. Define `TaskPromptAdapter` protocol and `NeutralTaskPromptAdapter`.
3. Add `sanitize_task_prompt_sections()` to enforce public-safe, bounded adapter output.
4. Add `TaskPlugin.prompt_adapter()` default method returning `NeutralTaskPromptAdapter(self.task_spec)`.
5. Extend `PublicContextBuilder` with optional `task_prompt_adapter`.
6. Insert sanitized task prompt sections into `public_context.task_prompt_sections`.
7. Wire live loop, audit loop, and legacy `AgentLoop` through `task_plugin.prompt_adapter()`.
8. Add regression coverage for fake non-push isolation, sanitization, curriculum-interface rejection, path scrubbing, context-size guard, and real live prompt artifact inclusion.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/tasks/prompt_adapter.py`
- `true_RL_Agent/src/true_rl_agent/tasks/base.py`
- `true_RL_Agent/src/true_rl_agent/tasks/__init__.py`
- `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- `true_RL_Agent/src/true_rl_agent/candidates/prompt_projection.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/loop.py`
- `true_RL_Agent/src/true_rl_agent/audit/loop_audit.py`
- `true_RL_Agent/tests/unit/test_task_prompt_adapter.py`
- `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- `true_RL_Agent/tests/integration/test_live_loop.py`
- `true_RL_Agent/PLAN/prompt/prompt_upgrade_execution_plan.md`

## Implemented Interface

`TaskPromptAdapter` protocol:

- `task_id: str`
- `build_prompt_sections(public_records, public_context) -> Mapping[str, Any]`

Allowed task-owned section keys:

- `task`
- `reward_inputs_contract`
- `public_eval_contract`
- `public_feedback_summary`
- `runtime_feedback_summary`
- `known_good_examples`
- `selection_guidance`

The neutral fallback provides:

- public task identity from `TaskSpec`
- supported policy initializations
- explicit `neutral_fallback` marker
- `not_available` reward/public eval/runtime contracts
- empty `known_good_examples`
- generic candidate-package guidance

The neutral fallback intentionally does not provide push-specific fields, push recovery success semantics, public metric interpretation, or failure taxonomy.

## Safety And Context Bounds

Adapter output is sanitized before prompt insertion:

- Existing public sanitizer removes forbidden private/hidden/trusted/checkpoint-like keys and secret-like values.
- Step4 adds absolute path scrubbing for Windows and Linux/server paths such as `D:/...`, `C:\...`, and `/data/...`.
- Unknown top-level task prompt sections are rejected.
- `curriculum`, `curriculum_design`, and `stage_list` style keys are rejected because curriculum is out of scope for this prompt upgrade step.
- Adapter output is capped at 32 KiB after sanitization to prevent context explosion before Step5/Step8 add full prompt budgeting.

## Boundary Checks

- `TaskPlugin` is the only task prompt semantic entry point.
- No new task registry was introduced.
- `TaskMemoryAdapter` was not expanded to carry task brief, RewardInputs semantics, eval semantics, examples, or selection guidance.
- Shared code does not contain push recovery success predicate, push failure taxonomy, or push RewardInputs field semantics.
- Fake non-push task prompt generation does not include `push_active`, `push_vec_public`, or push recovery text.
- Current live prompt artifacts now include a neutral `public_context.task_prompt_sections` block through `TaskPlugin.prompt_adapter()`.

## Residual Review Fix

After Step4 implementation, a review found that the full prompt payload still exposed push-specific fields through the shared top-level `reward_inputs` projection. The task-owned adapter section was clean, but `candidates.prompt_projection.reward_inputs_projection()` still used all `RewardInputs.field_names()`.

This was fixed in the shared projection as a narrow compatibility patch before Step5:

- Shared top-level `reward_inputs.fields` now omits task-owned fields `push_active` and `push_vec_public`.
- Fake non-push full payload regression now checks the entire serialized payload, not only `task_prompt_sections`.
- The full Step5 assembler remains responsible for replacing this transitional top-level projection with task-owned `reward_inputs_contract` sections.

## Red Tests Observed

Before implementation:

- `true_RL_Agent.tests.unit.test_task_prompt_adapter` failed with `ModuleNotFoundError: No module named 'true_rl_agent.tasks.prompt_adapter'`.
- `true_RL_Agent.tests.integration.test_live_loop.LiveLoopTests.test_live_loop_passes_saved_prompt_payload_to_provider` failed with `KeyError: 'task_prompt_sections'`.
- The path-scrub regression failed because `D:/humanoid_agent/private/raw.txt` and `/data/yuyangliu/secrets/key.env` remained in adapter output.
- The residual full-payload regression failed because shared top-level `reward_inputs.fields` still contained `push_active` and `push_vec_public`.

These failures confirmed the tests covered missing Step4 behavior.

## Verification

Focused commands run with bundled Python:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_task_prompt_adapter
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_memory_context_phase13
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.integration.test_live_loop
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_plugin_registries true_RL_Agent.tests.unit.test_prompt_projection_and_mechanism
```

Focused results:

- `test_task_prompt_adapter`: `Ran 6 tests`, `OK`
- `test_memory_context_phase13`: `Ran 6 tests`, `OK`
- `test_live_loop`: `Ran 7 tests`, `OK`
- `test_plugin_registries` + `test_prompt_projection_and_mechanism`: `Ran 10 tests`, `OK`

Full-suite verification was run after documentation updates.

## Remaining Scope

- Step5 must build the stable top-level prompt payload assembler and consume these task-owned sections outside the transitional `public_context.task_prompt_sections` location.
- Step6 must implement the real push recovery adapter under `tasks/push_recovery`, including task-specific RewardInputs contract and public eval semantics.
- Step8 must upgrade bounded public history and runtime feedback projection.
- Step11 must retain fake non-push prompt regression coverage as the prompt becomes thicker.

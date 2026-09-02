# Step 1 Prompt Baseline Audit

## Scope

This audit completes Step 1 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: establish the current prompt baseline from live code and real run artifacts before implementing prompt upgrades.

The audit uses current local code, server artifacts, and old Phase 8E prompt construction as a reference for prompt richness. It does not propose curriculum controls for the new prompt upgrade.

## System State

- Local branch: `true-rl-agent-refactor`
- Local HEAD: `8c6f3a07b76bba7ebec4c698943625bf10e0f8c7`
- Server branch at `/data/yuyangliu/AGENT/true_RL_Agent`: `true-rl-agent-refactor`
- Server HEAD: `8c6f3a07b76bba7ebec4c698943625bf10e0f8c7`
- Server status: only untracked `outputs/`
- Local `true_RL_Agent/runs`: empty
- Real prompt artifacts inspected on `humanoid-server`

## Artifacts Inspected

### New System Smoke

Path:

`/data/yuyangliu/AGENT/true_RL_Agent/runs/live_smoke_20260702T041556Z/rounds/live-smoke-20260702T041556Z-round-001/prompt_payload.json`

Findings:

- Size: 3145 bytes
- Top-level keys:
  - `anti_tuning_notice`
  - `candidate_allowed_controls`
  - `candidate_component_contract`
  - `public_context`
  - `reward_inputs`
  - `runner_owned_extensions`
  - `schema_version`
- `public_context` keys:
  - `best_public_score`
  - `last_outcome`
  - `previous_rounds`
  - `public_only`
  - `pure_knob_streak`
  - `recent_outcomes`

### New System Live

Primary paths:

`/data/yuyangliu/AGENT/true_RL_Agent/runs/live_15round_20260702T050103Z/rounds/live-15round-20260702T050103Z-round-001/prompt_payload.json`

`/data/yuyangliu/AGENT/true_RL_Agent/runs/live_15round_20260702T050103Z/rounds/live-15round-20260702T050103Z-round-001/prompt_payload_attempt_002.json`

`/data/yuyangliu/AGENT/true_RL_Agent/runs/live_15round_20260702T050103Z/rounds/live-15round-20260702T050103Z-round-010/prompt_payload.json`

Findings:

- Round 001 prompt size: 3145 bytes
- Round 001 attempt 002 prompt size: 5975 bytes
- Round 010 prompt size: 4851 bytes
- Top-level keys remain the same across first attempt, repair attempt, and later round.
- Repair attempt adds `public_context.candidate_repair_feedback`, but still does not add task, loop, eval, output-package, or full history sections.
- Round 010 adds `public_context.task_public_summary`, but this is a compact push metrics summary only.

### Old Phase 8E Reference

Live reference:

`/data/yuyangliu/runs/phase_8_cold_start_autoresearch/llm_curriculum_v1_deepseek_15round_20260701_190639/round_001/provider_candidate/phase8e_prompt_payload.json`

`/data/yuyangliu/runs/phase_8_cold_start_autoresearch/llm_curriculum_v1_deepseek_15round_20260701_190639/round_001/provider_candidate/phase8e_prompt_messages.json`

Smoke reference:

`/data/yuyangliu/runs/phase_8_curriculum_smoke/curriculum_compiler_smoke_20260701_032838/round/provider_candidate/phase8e_prompt_payload.json`

Findings:

- Old live payload size: 49423 bytes
- Old live message artifact size: 53953 bytes
- Old smoke payload size: 44798 bytes
- Old top-level keys:
  - `task`
  - `loop_contract`
  - `public_feedback_summary`
  - `public_curriculum_evidence_bundle`
  - `curriculum_design_contract`
  - `required_candidate_package`
  - `known_good_reward_examples`
  - `current_recipe`
  - `public_history`
  - `source_pack`
  - `source_access_policy`
  - `candidate_self_audit_checklist`
  - `selection_guidance`
  - `schema_version`
- Old system saved both payload and messages; system message plus user payload were auditable as artifacts.
- Old system had richer public context and output contract. Curriculum-specific sections should not be copied into this prompt upgrade, but the structural ideas are useful.

## Current Prompt Construction Path

Current new-system code path:

1. `run_live_loop()` builds an attempt payload in `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`.
2. `_build_attempt_prompt_payload()` calls `PublicContextBuilder.build()` and then `build_prompt_payload(public_context=...)`.
3. `PublicContextBuilder` lives in `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`.
4. Generic prompt projection lives in `true_RL_Agent/src/true_rl_agent/candidates/prompt_projection.py`.
5. The live loop saves the attempt payload to `prompt_payload.json` or `prompt_payload_attempt_002.json`.
6. The live loop then calls `provider.next_candidate(public_context=prompt_payload["public_context"])`.
7. `OpenAICompatibleCandidateProvider.next_candidate()` rebuilds a second prompt payload from `public_context`.
8. `_chat_completion_request()` serializes the provider user content with `json.dumps(..., sort_keys=True)`.

Baseline conclusion:

Saved prompt payloads are real artifacts, but they are not currently guaranteed to be the same payload actually sent to the model. The current provider interface is a P0 blocker for all later prompt-section work.

## Current Top-Level Field Sources

| Prompt field | Source | Current role | Baseline issue |
| --- | --- | --- | --- |
| `schema_version` | `candidates/prompt_projection.py` | Names prompt schema | OK but too thin |
| `public_context` | `agent_core/context.py` plus `TaskMemoryAdapter` | Round history summary | Omits many ledger fields |
| `reward_inputs` | `RewardInputs.field_names()` in shared candidates layer | Lists field names only | No semantics; not task-owned |
| `candidate_allowed_controls` | Static list in shared prompt projection | Says `RewardTerm`, `DiagnosticTerm`, `ScheduleTerm` | No task-specific meaning or effectiveness contract |
| `candidate_component_contract` | Static dict in shared prompt projection | Basic component API and minimal template | Output package contract incomplete |
| `anti_tuning_notice` | Static string | Warns against pure knobs | Useful but insufficient |
| `runner_owned_extensions` | Static dict | Marks some controls not prompt writable | Thin and not tied to actual task/runtime evidence |

## Current RewardInputs Baseline

Actual prompt-visible fields from new live prompt:

- `base_height`
- `roll`
- `pitch`
- `base_pos_xy`
- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `commanded_base_vel`
- `joint_pos`
- `joint_vel`
- `joint_pos_error`
- `last_action`
- `prev_action`
- `action_delta`
- `push_active`
- `push_vec_public`
- `progress_fraction`

Baseline issue:

The prompt only lists names. It does not explain shape, units, coordinate frame, source status, fallback behavior, or which fields are reliable for push recovery. This is currently generated by shared candidate projection rather than by a task-owned prompt adapter.

## Current Public Context Baseline

First-round context:

```json
{
  "best_public_score": null,
  "last_outcome": null,
  "previous_rounds": 0,
  "public_only": true,
  "pure_knob_streak": 0,
  "recent_outcomes": []
}
```

Later-round context example from round 010:

- `previous_rounds`: 9
- `last_outcome`: `reject`
- `recent_outcomes`: last 5 rounds with only `round_id`, `outcome`, and `surface`
- `task_public_summary.success_rate.latest`: 0.0
- `task_public_summary.success_rate.best`: 0.0
- `task_public_summary.latest_failure_mode`: `total_dominated`
- `task_public_summary.latest_failure_breakdown`: fall-forward dominated
- `task_public_summary.latest_motion_quality`: includes action, drift, pitch, roll, base height, return

Baseline issue:

The prompt does not show decision reasons, public scores per recent round, changed components, candidate ids, verifier results, runner errors, runtime activation status, or candidate loaded/eval loaded flags.

## Ledger Information Present But Not Adequately Projected

The live ledger at:

`/data/yuyangliu/AGENT/true_RL_Agent/runs/live_15round_20260702T050103Z/ledger.jsonl`

contains public information that would be useful to the model but is not sufficiently projected into the next prompt.

Examples:

- `decision.reason`
  - `public_score_delta_not_met`
  - `candidate_runtime_failed`
  - `candidate_verifier_failed`
- `candidate_surface.classification`
  - mostly `code_level`
  - one `no_code_change`
- `public_result.metrics.candidate_package_loaded`
  - observed `false` for evaluated rounds
- `public_result.metrics.candidate_code_loaded_in_eval`
  - observed `false` for evaluated rounds
- `runtime_activation_summary.reward_term_calls`
- `runtime_activation_summary.diagnostic_calls`
- `runtime_activation_summary.nonzero_reward_contribution`
- `runtime_activation_summary.diagnostic_outputs`
- `mechanism_attribution.status`
- `mechanism_attribution.reason`
- `runner_error.message`
- `runner_error.details.errors`
- `train_result.summary.candidate.components.effective_hook_counts`

Concrete live summary:

- Completed ledger records: 10
- Accepted rounds: 0
- Decision reasons:
  - `public_score_delta_not_met`: 7
  - `candidate_runtime_failed`: 2
  - `candidate_verifier_failed`: 1
- Surface classifications:
  - `code_level`: 9
  - `no_code_change`: 1
- Evaluated public results reported:
  - `candidate_package_loaded=false`
  - `candidate_code_loaded_in_eval=false`
- Runtime errors included tensor shape/component errors such as:
  - `reward term upright must return shape (8,)`
  - `reward term uprightness must return shape (8,)`

Baseline issue:

The next prompt currently tells the LLM that rounds were rejected but does not clearly tell it whether candidate code loaded, whether it ran in training, whether eval used it, whether reward terms produced nonzero output, or whether the failure was a shape/materialization/runtime issue.

## Provider Request Artifact Baseline

Old Phase 8E saved both:

- `phase8e_prompt_payload.json`
- `phase8e_prompt_messages.json`

New true_RL_Agent live run saves:

- `prompt_payload.json`
- `prompt_payload_attempt_002.json`

No provider request/messages artifact was found in the inspected new live run. Combined with the code path that rebuilds the prompt inside provider from `public_context`, the current system cannot prove from artifacts that the saved prompt equals the actual model input.

Baseline issue:

Provider request and saved prompt artifact are not currently same-source auditable.

## Missing Information That Directly Affects Candidate Quality

### Task Goal And Fixed Protocol

Available in code:

- `PushRecoveryTaskPlugin.task_spec`
- `PushRecoveryTaskConfig`
- `TaskSpec.fixed_train_protocol`
- `TaskSpec.fixed_eval_protocol`

Missing or too thin in prompt:

- Task objective
- Robot identity
- Cold-start/random policy condition
- Fixed train budget: 4096 envs, 500 iterations, 24 steps/iteration
- Eval num envs and seed visibility summary
- Success predicate meaning
- Runner-owned checkpoint and eval boundaries

Impact:

The LLM has no strong task grounding and may write generic upright rewards instead of push-recovery-specific mechanisms.

### Public Eval Contract

Available in code:

- `PublicEvalResult(status, public_score, metrics)`
- `tasks/push_recovery/metrics.py`
- `decision.py`

Missing or too thin in prompt:

- Meaning of `status`
- Meaning of `public_score`
- Meaning of `metrics.success_rate`
- Meaning of failure breakdown and motion gates
- Explanation that acceptance logic is runner-owned and read-only

Impact:

The LLM sees rejection summaries without understanding how to interpret public eval evidence.

### Candidate Package Contract

Available in code:

- `OpenAICompatibleCandidateProvider._system_prompt()`
- `_materialize_candidate()`
- `candidate_component_contract`

Missing or too thin in saved prompt:

- Explicit JSON response shape
- `candidate_id`, `rationale`, `declared_changed_components`, `components_py` requirements in user payload
- Distinction between provider-materialized files and LLM-owned content
- Clear prohibition against JSON prose outside object in saved payload

Impact:

The provider relies on hardcoded system prompt text and materializer assumptions rather than a complete auditable user payload.

### Runtime Activation And Load Evidence

Available in ledger:

- reward/diagnostic calls
- nonzero reward contribution
- diagnostic outputs
- `candidate_package_loaded`
- `candidate_code_loaded_in_eval`
- mechanism attribution

Missing or too thin in prompt:

- Candidate load/eval load flags
- Whether candidate code was active in training
- Whether reward output was finite and nonzero
- Whether diagnostics ran
- Whether rejection was due to no score improvement or no actual candidate effect

Impact:

The LLM may misdiagnose zero success as reward-design failure when the candidate may not have been loaded into eval or failed runtime activation.

### Repair Feedback Scope

Current repair prompt can include:

- runtime failure type
- runtime error
- previous candidate id
- previous components code
- vectorization/tensor guidance

Missing:

- provider JSON parse failures
- schema/materialization failures
- public-safe categorization for provider-side failures
- durable ledger record when provider attempts fail before candidate exists

Impact:

Malformed provider responses can terminate or disappear from useful next-attempt feedback.

## Old-System Features Worth Borrowing

Borrow the structural ideas:

- Full saved payload plus saved messages/request artifact
- `task`
- `loop_contract`
- `required_candidate_package`
- `reward_inputs_contract` with semantics, not just names
- `public_feedback_summary`
- bounded `public_history`
- `known_good_reward_examples`
- `source_access_policy` shape if source-read support is later needed
- `candidate_self_audit_checklist`
- `selection_guidance`
- explicit `not_available_to_candidate`

Do not copy directly:

- curriculum stage design sections for this prompt upgrade
- old `public_dev/public_accept` shape as the new canonical public eval interface
- old Phase 8E task ids or G1-specific strings unless the new push task explicitly owns them
- curriculum-primary guidance

## Step 1 Acceptance Checklist

- [x] Current prompt top-level fields, size, order, and source modules recorded.
- [x] At least one smoke artifact inspected.
- [x] At least one live artifact inspected.
- [x] Old-system prompt artifact inspected for reference.
- [x] Necessary public information present in ledger/task config but missing from prompt recorded.
- [x] Missing information that directly affects candidate generation recorded.
- [x] Curriculum fields explicitly excluded from this round's gap target.
- [x] P0 provider same-source issue confirmed and recorded.

## Step 1 Conclusion

The current new prompt is not information-sufficient. The most severe issue is not only that sections are missing, but that provider wiring would discard them even if they were added today. Therefore Step 0/P0 must be implemented before adding task sections.

After P0, the prompt upgrade should prioritize task-owned prompt adapter, push-recovery RewardInputs semantics, public eval contract, richer bounded public history, and runtime/load evidence projection.

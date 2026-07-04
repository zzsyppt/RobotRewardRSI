# Step 9 OpenAI-Compatible Provider Prompt Role Split

## Scope

本记录完成 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step9：收窄 OpenAI-compatible provider 的 system prompt，让 system prompt 只承担通用 provider 角色、JSON-only 输出、安全边界和 runner-owned 禁止面；任务特殊语义、component API 细节、RewardInputs 语义、public feedback 和 candidate package 细节全部来自 user payload。

Step9 不改变 candidate package schema、provider materializer、verifier、accept/reject 逻辑、任务 success 判定、训练预算、eval protocol、seed policy、checkpoint policy、PPO 或 curriculum 控制。

## Codex 设计参考

参考 `D:\humanoid_agent\codex\AGENTS.md` 的 model-visible context 约束：

- model-visible context 中的注入片段应有明确职责。
- 注入内容必须有结构和硬上限，避免把大量细节散落在隐式 system instruction 中。
- 保持上下文增量和稳定，避免频繁改变会破坏可审计性的隐藏 prompt 片段。

对应到新项目，Step9 将稳定、可审计的任务/组件/输出合同保留在 user payload 的结构化 JSON 中；system prompt 只表达 provider 运行角色和不可越界的通用规则。

## Implementation Plan Used

1. 审计当前 `_system_prompt()`：确认已经没有 hardcoded push recovery 任务说明，但仍把 `RewardInputs`、`RewardTerm`、`DiagnosticTerm`、`ScheduleTerm`、`TrainingComponents` 和 declared component values 写进 system prompt。
2. 先改 `test_system_prompt_is_generic_and_task_semantics_stay_in_user_payload`，要求 system prompt 不包含任务名、push 语义或 component/API 细节；同时确认这些细节仍存在于 user payload。
3. 收窄 `_system_prompt()`：只说明 user payload 是 task-specific context、public evidence、component contract 和 output-field constraints 的唯一来源。
4. 保留 response JSON object、JSON-only、无额外 prose、无额外 keys、安全边界、runner/eval/budget/seed/checkpoint/PPO/curriculum/acceptance 禁止面。
5. 保持 provider request 的 user message 仍为完整 prompt payload，不重建、不排序。

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`

## Boundary Decisions

- System prompt 可以说“遵守 user payload 中的 candidate package 和 component contracts”，但不列出具体任务、RewardInputs 字段、component 类名或 declared component values。
- Provider JSON 输出字段仍由 user payload 的 `required_candidate_package.provider_json_output` 暴露给模型；system prompt 只保留 required key 名称作为通用 JSON contract 提醒。
- Prompt-visible task/eval summaries 只能作为 read-only evidence，不能变成 candidate-controllable target。
- `curriculum` 在当前 prompt 升级范围内仍不可由 candidate 修改；system prompt 明确禁止 proposing curriculum changes。
- 非 push task 的 provider system message 不包含 push recovery 语义，也不包含 task id。

## Red Test Observed

实现前，更新后的 `test_system_prompt_is_generic_and_task_semantics_stay_in_user_payload` 失败：

- system prompt 不包含 `user payload is the only source of task-specific context`。
- system prompt 仍包含 `RewardInputs`、`RewardTerm`、`DiagnosticTerm`、`ScheduleTerm`、`TrainingComponents` 等 component/API 细节。

该失败对应 Step9 的真实职责分裂缺口。

## Verification

Focused red-green command:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent\src'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_openai_compatible_provider
```

Current focused result:

- `test_openai_compatible_provider`: `Ran 13 tests`, `OK`

Full verification is recorded in the final turn report for this step.

## Remaining Scope

- Step10 still owns final payload ordering/readability and saved artifact/provider request/ledger prompt ref consistency cleanup.
- Step11 must run the broader prompt safety/boundary regression, including non-push task isolation.
- Step12/13 must validate real generated prompt artifacts and provider user payload captures.

# Step 10 Prompt 顺序与可读性规范

## Scope

本记录对应 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step10。

目标是让 provider-facing prompt payload 具备稳定、可审计、可读的文本形态：

- provider 实际收到的 user message JSON 与 run artifact 保存的 `prompt_payload.json` 使用同一序列化规范。
- ledger 只记录公开 `PromptPayloadRef`，不复制 prompt 内容，避免 ledger 膨胀。
- 顶层 section 顺序保持 task/loop/feedback/runtime/package 优先，让模型先看到任务、边界、反馈和候选包合同。
- 不新增任务特殊语义到 shared prompt 层；任务字段仍由 task-owned prompt adapter 提供。

## Codex Design Reference

参考 `D:\humanoid_agent\codex\AGENTS.md` 的 model-visible context 原则：

- 模型可见上下文必须结构化、稳定，并避免无界注入。
- 真实 request 应可被测试捕获和结构化断言。
- 共享层应提供上下文载体和序列化边界，而不是承载任务特殊语义。

在新系统中的对应做法是：把 provider-facing prompt JSON 串行化收敛到一个 helper，并让保存路径和 provider request 共用它。

## Implementation Plan Used

1. 审计 `build_prompt_payload()`、`save_prompt_payload()`、`OpenAICompatibleCandidateProvider._chat_completion_request()` 与 `PromptPayloadRef` 的实际链路。
2. 先写红灯测试，使用包含中文任务名的 fake non-push payload，要求 saved prompt artifact 文本与 provider user message 文本完全一致。
3. 在 `true_rl_agent.prompting.payload` 增加 `serialize_prompt_payload()`，作为 provider-facing prompt JSON 的单源规范。
4. 让 `PublicContextBuilder.save_prompt_payload()` 和 OpenAI-compatible provider 都调用该 helper。
5. 保持 `candidates.prompt_projection` 与 `prompting.__init__` 的 backward-compatible export，避免后续调用方重新发明 serializer。
6. 增加可读性回归测试，固定序列化文本前段的 section 顺序。
7. 增加 round integration 回归测试，确认 ledger record 写入 `PromptPayloadRef` 的公开 path/visibility。

## Files Changed

- `true_RL_Agent/src/true_rl_agent/prompting/payload.py`
- `true_RL_Agent/src/true_rl_agent/prompting/__init__.py`
- `true_RL_Agent/src/true_rl_agent/candidates/prompt_projection.py`
- `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- `true_RL_Agent/tests/integration/test_fake_round.py`

## Red Test Observed

实现前新增的测试失败：

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_openai_compatible_provider.OpenAICompatibleProviderTests.test_provider_user_payload_matches_saved_prompt_artifact_text
```

失败原因：

- `save_prompt_payload()` 使用默认 `json.dumps(..., indent=2)`，中文任务名被转义为 `\u....`。
- provider user message 使用 `json.dumps(..., ensure_ascii=False, indent=2)`，保留 UTF-8 文本。
- 两者解析后的对象一致，但文本不一致，不能满足 Step10 的严格 traceability。

## Boundary Decisions

- serializer 只定义 shared prompt JSON 文本规范，不引入任何 push recovery、stair climbing 或其他任务特殊字段。
- provider system prompt 不承担任务语义，任务语义仍只在 user payload 中。
- ledger 记录 `PromptPayloadRef` 即可；prompt 内容由 artifact 文件承载，避免重复写入 ledger。
- `public_context`、`reward_inputs` 等 legacy alias 仍保留在 canonical sections 后面，等待后续 step 继续降级，不在 Step10 扩大范围。

## Verification Targets

Step10 至少需要通过：

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_openai_compatible_provider tests.unit.test_prompt_projection_and_mechanism
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.integration.test_fake_round.FakeRoundTests.test_round_ledger_records_prompt_payload_ref
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover
git diff --check
```

## Remaining Scope

- Step11 继续做安全与边界回归，包括 provider request、saved prompt artifact、ledger prompt ref 的更广覆盖。
- Step12 需要在 smoke 中生成真实 prompt artifact，并验证 provider 实际使用的 payload 与 artifact 完全一致。
- Step13 需要在多轮运行中确认 prompt 不会越跑越薄。

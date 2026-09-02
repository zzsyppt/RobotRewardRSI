# Prompt 升级统一实施计划

> **For agentic workers:** 按本计划逐项执行。后续不再使用 `Step1/implementation_plan.md` 作为执行依据；Step1 的审计证据保留在 `Step1/step1_prompt_baseline_audit.md`。

**Goal:** 升级 `true_RL_Agent` 的 prompt 构造与 provider 发送链路，让模型收到足量的 public task、feedback、runtime evidence 和 candidate-package 合同，同时保持新版架构的任务隔离，避免退化成任务强耦合或 LLM 调参。

**Architecture:** 共享层只负责 payload 组装、public-only sanitize、provider request 包装、artifact traceability、通用 candidate package 合同和通用 accept/reject 框架。任务特殊语义通过 `TaskPlugin` 暴露的 task-owned prompt adapter 提供；push recovery 的 success 计算、RewardInputs 语义、public metric projection 和 failure taxonomy 留在 `tasks/push_recovery` 内。

**Tech Stack:** Python package `true_rl_agent`，OpenAI-compatible chat provider，pytest，服务器路径 `/data/yuyangliu/AGENT/true_RL_Agent`，旧系统参考 `src/humanoid_agent/cold_start_lab/phase8e_prompt.py`。

---

## 当前基线

- 本地与服务器代码基线：branch `true-rl-agent-refactor`，HEAD `8c6f3a07b76bba7ebec4c698943625bf10e0f8c7`。
- 当前新系统 prompt artifact 很薄：真实 `prompt_payload.json` 顶层只有 `schema_version`、`public_context`、`reward_inputs`、`candidate_allowed_controls`、`candidate_component_contract`、`anti_tuning_notice`、`runner_owned_extensions`。
- 当前 provider 链路存在 P0 阻塞：live loop 保存完整 `prompt_payload.json` 后，只把 `prompt_payload["public_context"]` 传给 provider；`OpenAICompatibleCandidateProvider` 再重新构造薄 payload，并用 `sort_keys=True` 序列化。
- 旧 Phase 8E prompt 只能作为结构参考，不能把课程 stage、curriculum primary surface、旧 `public_dev/public_accept` 形状直接搬进新系统。
- Step1 审计已完成，证据文件为 `true_RL_Agent/PLAN/prompt/Step1/step1_prompt_baseline_audit.md`。

## 总体验收原则

- 真实生成的 `prompt_payload.json` 必须作为验收对象，不能只验代码结构。
- provider 实际发送给模型的 user payload 必须与保存的 `prompt_payload.json` 同源，不能保存厚 payload 却发送薄 payload。
- prompt 只能包含 public-safe 信息，不得泄露 hidden/private/trusted/raw seed/API key/server secret。
- push recovery 的成功判定、failure taxonomy、RewardInputs 语义、public metric projection 必须留在任务层，不得写入不同任务共享的判定层。
- round 间 accept/reject 决策框架可以共享，但 task-specific success 计算与语义解释不能进入共享判定层。
- 未真实接线或当前不应由 LLM 控制的能力不得出现在 effective controls 中。
- 课程相关字段、stage 设计、curriculum capability registry 本轮不纳入 prompt 升级范围。
- 每一步完成后都要能独立回滚、独立验证，不能依赖后续步骤才能判断是否正确。

## Non-Goals

- 不重新引入旧 Phase 8E curriculum stage 设计。
- 不把精确 accept threshold、absolute floor、min delta 作为 LLM 可优化目标暴露；如作为 public evidence 暴露，必须明确 read-only 且不可修改。
- 不允许 LLM 修改训练预算、eval protocol、seed policy、PPO、checkpoint selector、runner code 或 success metric。
- 不让共享 accept/reject 代码理解 push-recovery success 细节。
- 不新增第二套 task registry。

## Step 0 / P0: Provider Payload 同源改造

**目标**

先修正 provider 接口与发送路径，确保后续新增的 prompt sections 会真正进入模型上下文。`CandidateProvider.next_candidate()` 应接收完整 `prompt_payload`，live loop、普通 loop、audit loop、replay/fake provider、OpenAI-compatible provider 都应沿用同一个完整 payload 合同。

**文件**

- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/provider.py`
- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`
- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/loop.py`
- 修改 `true_RL_Agent/src/true_rl_agent/audit/loop_audit.py`
- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- 更新 `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- 更新 `true_RL_Agent/tests/integration/test_live_loop.py`
- 更新 `true_RL_Agent/tests/integration/test_agent_core_loop.py`
- 更新 `true_RL_Agent/tests/integration/test_dry_run_audit_loop.py`

**任务**

- [ ] 将 `CandidateProvider.next_candidate()` 改为接收 `prompt_payload: Mapping[str, Any]`。
- [ ] 更新 replay、audit replay、fake、dry-run、live provider，使其都兼容完整 prompt payload。
- [ ] 让 live loop 和 audit loop 传入与保存 artifact 同一次构造出来的 payload。
- [ ] 移除 `OpenAICompatibleCandidateProvider` 内部的 `build_prompt_payload(public_context=...)` 二次构造。
- [ ] provider user content 保留 payload 原始顶层顺序，不使用破坏阅读顺序的 `sort_keys=True`。
- [ ] 增加 provider request capture 或测试钩子，证明 user message 来自保存的同一个 payload。
- [ ] 确认 `prompt_payload.json`、provider user content、ledger prompt reference 能追溯到同一个 round/attempt。

**严格验收**

- 用一个人工额外顶层 section 的测试证明：修复前该 section 到不了 provider request，修复后能到达。
- 全仓库不再存在只接受 `public_context` 的 provider 调用路径。
- saved payload 与 provider user payload 语义完全一致。
- P0 未通过前，不开始新增 task sections、RewardInputs contract 或 history sections。

## Step 1: Prompt 现状基线审计

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step1/step1_prompt_baseline_audit.md`。

**已确认事实**

- 当前 new smoke prompt 约 3 KB，live later-round prompt 约 4 到 6 KB。
- 旧 Phase 8E prompt payload 约 45 到 49 KB，并保存 `phase8e_prompt_messages.json`。
- 新系统 ledger 已有 `runtime_activation_summary`、`mechanism_attribution`、`runner_error`、`public_result.metrics.candidate_package_loaded`、`public_result.metrics.candidate_code_loaded_in_eval` 等 public evidence，但当前 prompt 没有充分投影。
- 当前 prompt 缺 task objective、fixed train/eval protocol、RewardInputs 语义、public eval contract、runtime activation summary、provider request 同源证据。

**后续要求**

- Step1 证据文件作为后续实现验收的基线，不再另开执行计划文件。
- 如果服务器 artifact 更新，需要追加审计记录，而不是覆盖当前证据。

## Step 2: Provider Parse/Schema/Materialize Repair Feedback

**目标**

把 provider response 的 JSON parse、schema/materialize、verifier、runtime 失败统一纳入可审计的 public repair feedback，使下一次 provider attempt 能看到失败原因，而不是直接抛异常终止 round。

**文件**

- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`
- 可新增 `true_RL_Agent/src/true_rl_agent/agent_core/provider_errors.py`
- 更新 `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- 更新 `true_RL_Agent/tests/integration/test_live_loop.py`

**任务**

- [ ] 定义 public-safe provider failure categories：`json_parse_error`、`schema_error`、`candidate_materialize_error`、`verifier_error`、`runtime_error`。
- [ ] 将 provider response parse/materialize 失败包装成 sanitized repair feedback。
- [ ] 防止 raw API text、API key、server path、hidden/private 值进入 repair feedback。
- [ ] 达到 max attempts 时，在 round artifact 或 ledger-safe attempt record 中记录公开失败摘要。
- [ ] repair feedback 不得提示 LLM 修改 runner、eval、budget、seed、checkpoint、PPO 或 accept rules。

**严格验收**

- 测试覆盖 malformed JSON、缺失 `components_py`、非法 `declared_changed_components`、verifier rejection、runtime failure。
- 第二次 attempt prompt 包含第一次失败的 public-safe failure type、简短原因和可行动修正方向。

## Step 3: Prompt 信息足量标准定义

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step3/step3_prompt_information_sufficiency_standard.md`。

**目标**

定义新版 prompt 的最低信息足量标准，用作后续所有开发与验收的统一合同。

**任务**

- [x] 定义首轮 prompt 必须包含的 sections：`task`、`loop_contract`、`required_candidate_package`、`reward_inputs_contract`、`public_eval_contract`、`selection_guidance`。
- [x] 定义后续轮 prompt 必须额外包含的 sections：`public_history`、`public_feedback_summary`、`runtime_feedback_summary`、best/latest 摘要。
- [x] 定义 repair attempt prompt 必须额外包含的 sections：`provider_or_runtime_repair_feedback`、previous attempt failure category、public-safe corrective guidance。
- [x] 标注每类信息来源属于 shared layer 还是 task-owned layer。
- [x] 明确 private/trusted/hidden/raw seed 信息禁止进入 prompt。

**严格验收**

- 标准覆盖 task、loop boundary、candidate controls、RewardInputs contract、public eval contract、public feedback、runtime feedback、candidate package output contract、known-good examples、selection guidance。
- 标准明确 push recovery success 判定不能写入共享层。
- 标准明确课程字段不属于本轮 prompt 缺口。

## Step 4: Task-Owned Prompt Adapter Interface

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step4/step4_task_owned_prompt_adapter_interface.md`。

**目标**

定义 task-owned prompt adapter 的公共合同，让每个任务都能提供自己的 prompt 语义，同时共享层无需理解具体任务。落点应是 `TaskPlugin` 可选暴露或持有 `TaskPromptAdapter`，不是新建独立 task registry；`TaskMemoryAdapter` 继续只负责 public history summary。

**文件**

- 修改 `true_RL_Agent/src/true_rl_agent/tasks/base.py`
- 可新增 `true_RL_Agent/src/true_rl_agent/tasks/prompt_adapter.py`
- 修改 `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- 更新 `true_RL_Agent/tests/unit/test_memory_context_phase13.py`
- 新增或更新 prompt adapter unit test

**任务**

- [x] 定义 `TaskPromptAdapter` protocol，能表达 task brief、fixed train/eval protocol summary、reward input contract、public eval contract、public feedback summary、runtime feedback summary、known-good examples、selection guidance。
- [x] `TaskPlugin` 是 task prompt adapter 的唯一注册入口。
- [x] 没有 task prompt adapter 的任务使用 neutral fallback，且 fallback 不注入 push recovery 语义。
- [x] adapter 输出必须经过现有 public sanitization 和 prompt-public checks。
- [x] adapter 合同不包含 curriculum stage 或 curriculum design 接口。

**严格验收**

- fake non-push task 能生成 prompt，且不含 push-specific 字段。
- 没有 prompt adapter 的任务仍能生成 generic candidate-package guidance。
- 共享 prompt assembly 代码不包含 push recovery success predicate、failure taxonomy、RewardInputs 字段语义。

## Step 5: Prompt Payload Assembler

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step5/step5_prompt_payload_assembler.md`。

**目标**

将当前薄的 `candidates.prompt_projection.build_prompt_payload(public_context=...)` 升级为共享 assembler，按稳定顺序合并 generic sections 与 task-owned sections。

**文件**

- 修改或拆分 `true_RL_Agent/src/true_rl_agent/candidates/prompt_projection.py`
- 修改 `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- 可新增 `true_RL_Agent/src/true_rl_agent/prompting/payload.py`
- 更新 `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- 更新 `true_RL_Agent/tests/unit/test_memory_context_phase13.py`

**任务**

- [x] 定义稳定顶层顺序：`task`、`loop_contract`、`public_feedback_summary`、`runtime_feedback_summary`、`required_candidate_package`、`reward_inputs_contract`、`control_effectiveness_contract`、`public_history`、`known_good_examples`、`selection_guidance`、可选 `provider_or_runtime_repair_feedback`、`schema_version`。
- [x] 将通用 component API shape 保留在 shared prompt code。
- [x] RewardInputs 字段语义从 task adapter 获取，不再从 shared candidates 层直接把 `RewardInputs.field_names()` 作为所有任务的 prompt contract。
- [x] 保留并扩展 `assert_public_only_prompt_payload()`。
- [x] 保存 payload 和 provider user payload 时避免顶层字母序重排。
- [x] provider 输出 JSON 合同由单一 helper 提供，但 `declared_changed_components` 分为两层：prompt 只广告当前组件合同真实支持的值，materializer 可保留 legacy parse allowlist 且不得暴露给模型。
- [x] legacy `reward_inputs` 是过渡 alias；不得继续扩展 task-specific denylist，Step10 应移除或降级该 alias。

**严格验收**

- 真实 `prompt_payload.json` 可按预期顺序阅读。
- JSON 工具可解析 payload。
- public-only checks 仍能拒绝 forbidden key names 和 secret-like values。

## Step 6: Push Recovery Prompt Adapter

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step6/step6_push_recovery_prompt_adapter.md`。

**目标**

为当前 `unitree_push_recovery` 任务实现第一版 task-specific prompt adapter，保证信息足量但不暴露私有评估或 runner-owned 决策细节。

**文件**

- 新增 `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py`
- 修改 `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/plugin.py`
- 可修改 `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/memory.py`
- 更新 `true_RL_Agent/tests/unit/test_push_recovery_task.py`
- 更新 `true_RL_Agent/tests/unit/test_memory_context_phase13.py`

**任务**

- [x] 添加 `unitree_push_recovery` task brief：机器人、cold-start/random policy setting、objective、fixed train budget、fixed eval summary。
- [x] 提供 task-owned RewardInputs contract：字段名、shape、语义、单位/坐标系、source status、fallback behavior。
- [x] 对 placeholder 或弱接线字段明确标注，不夸大 `push_active` 或 `push_vec_public` 的有效性。
- [x] 基于当前 `PublicEvalResult(status, public_score, metrics)` 提供 public eval contract。
- [x] gate/status 字段解释为 read-only evidence，不作为 candidate-controllable target。
- [x] 包含 runtime feedback summary：reward calls、diagnostic calls、nonzero reward、candidate package loaded flags、candidate code loaded in eval flags、mechanism attribution。
- [x] 加入少量 known-good minimal examples 或 source references，但不复制旧系统 curriculum-primary guidance。

**严格验收**

- Push prompt 信息足以让 LLM 写出 valid reward/diagnostic code。
- prompt 不暴露 raw seeds、trusted eval、checkpoint selector control、runner paths、secrets 或 private artifacts。
- 共享代码不理解 push-recovery success semantics。

## Step 7: Candidate Package 与 Provider 输出合同

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step7/step7_candidate_package_provider_output_contract.md`。

**目标**

把 LLM 输出从“随意 JSON + components_py”提升为明确的 candidate package 工程合同，同时保持第一版工作量可控。

**任务**

- [x] prompt 明确 LLM 只输出 candidate package，不输出 runner patch。
- [x] prompt 明确允许和禁止的文件/字段，以及 JSON 外 prose 禁止规则。
- [x] prompt 明确 `components.py`、candidate id、rationale、declared changed components 的要求。
- [x] prompt 说明 `operation_diff` 和 `diagnostics` 的用途，即使第一版仍由 provider materialize。
- [x] provider 和 verifier 对候选包失败给出 public-safe 分类反馈。

**严格验收**

- 不允许 LLM 修改 eval、success metric、budget、seed、checkpoint、PPO、runner code。
- provider 输出合同出现在 user payload 中，而不是只藏在 hardcoded system prompt 或 materializer assumptions 里。

## Step 8: Public History 与 Memory 投影升级

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step8/step8_public_history_memory_projection.md`。

**目标**

扩展下一轮 prompt 可见的 public history，让 LLM 能理解上一轮为什么失败、候选是否生效、是否只是参数/诊断变化。

**文件**

- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/context.py`
- 修改 `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- 修改 `true_RL_Agent/src/true_rl_agent/tasks/push_recovery/memory.py`
- 更新 `true_RL_Agent/tests/unit/test_memory_context_phase13.py`

**任务**

- [x] 投影最近若干轮 `round_id`、`outcome`、`reason`、`surface_classification`、`public_score`、changed components。
- [x] 投影 ledger 中 public runtime activation evidence。
- [x] 投影 `candidate_package_loaded` 与 `candidate_code_loaded_in_eval`。
- [x] 投影 runner errors 与 repair feedback 的 sanitized summary。
- [x] 对 history 做 round count 和 byte budget 限制。

**严格验收**

- 后续轮 prompt 能明显看出 candidate code 是否运行、是否进入 eval、是否 public score 仍为 0。
- 后续轮 prompt 能暴露 shape/materialization/runtime 失败的 public-safe 原因。
- history 不无限增长。

## Step 9: OpenAI-Compatible Provider Prompt Role Split

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step9/step9_provider_prompt_role_split.md`。

**目标**

让 system prompt 只承担通用角色、JSON-only、安全输出和禁止修改 runner 的职责；任务语义全部在 user payload 中。

**文件**

- 修改 `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- 更新 `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`

**任务**

- [x] 移除 system prompt 中 hardcoded push recovery task semantics。（已在 Step5 review fix 中提前完成）
- [x] 保留 output-format、安全边界、JSON-only 规则。
- [x] 确认 task semantics 由 user payload 提供。
- [x] 确认 response format 仍为 machine-parseable JSON object。

**严格验收**

- fake non-push task 不会收到 push recovery system instruction。
- candidate output contract 仍然严格。

## Step 10: Prompt 顺序与可读性规范

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step10/step10_prompt_order_and_traceability.md`。

**目标**

确定 provider-facing payload 的稳定顺序与可读性规范，让 LLM 先读任务和边界，再读反馈和输出合同。

**任务**

- [x] 保存到 run artifact 的 payload 与实际 provider user payload 一致。
- [x] system prompt 和 user payload 职责清晰分离。
- [x] 顶层顺序稳定，避免顶层字母序排列。
- [x] prompt payload 可用普通 JSON 工具审计。

**严格验收**

- provider request、saved prompt artifact、ledger prompt ref 三者一致。
- 真实 payload 首屏能看到 task/loop/feedback/package 的主要信息。

## Step 11: 安全与边界回归

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step11/step11_prompt_safety_boundary_regression.md`。

**目标**

确保 prompt 变厚以后没有引入泄露、越权控制或任务语义污染。

**任务**

- [x] 自动检查 prompt 中不存在 private、hidden、trusted seed、checkpoint selector、API key、secret、raw seed id。
- [x] 检查 prompt 中不出现未接线能力作为 effective controls。
- [x] 检查非 push 任务不会继承 push recovery 的 success predicate、failure taxonomy、push-specific RewardInputs 说明。
- [x] 检查 LLM 不被告知可以修改 eval protocol、budget、seed、checkpoint、runner、PPO。
- [x] 检查 provider request、saved prompt artifact、ledger prompt ref 三者一致。

**严格验收**

- 任何 public-only 检查失败都阻断 smoke。
- fake non-push task regression 必须通过。

## Step 12: 单轮 Smoke 验收

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step12/step12_single_round_smoke.md`。

**目标**

用最小真实流程验证新 prompt 能生成、保存、发送、被 provider 使用，并能进入候选验证流程。

**任务**

- [x] 生成单轮 prompt artifact。
- [x] 捕获或验证 provider user payload。
- [x] 至少一次 provider response 能 materialize candidate package。
- [x] 如果候选失败，repair attempt prompt 包含 public-safe feedback。
- [x] 记录 smoke artifact 路径。

**严格验收**

- `prompt_payload.json` 包含 Step 3 定义的必需 sections。
- provider 使用的 payload 与保存 artifact 完全一致。
- 单轮 smoke 不要求策略成功，但要求 prompt 信息完整、安全检查通过、candidate pipeline 不退化。

## Step 13: 多轮 Live/Audit 验收

**状态**

已完成。记录保留在 `true_RL_Agent/PLAN/prompt/Step13/step13_multiround_live_validation.md`。

**目标**

确认升级后的 prompt 在多轮迭代中持续提供足量反馈，不会越跑越薄或丢失关键语义。

**任务**

- [x] 至少运行一轮多轮 audit 或 live smoke，覆盖首轮、普通后续轮、repair attempt。
- [x] 每轮 prompt 保留 task/loop/package/eval/reward input 基础合同。
- [x] 后续轮正确包含 public history、runtime feedback、best/latest 摘要。
- [x] 如果候选未实际生效，下一轮 prompt 明确暴露 public-safe 生效失败。
- [x] 如果 public success 没提升，prompt 反映真实 reject reason，而不是只给 outcome 名称。

**严格验收**

- 多轮 run 的 prompt 不再只有薄 API contract。
- history 和 repair feedback 受控增长。

## Step 14: 新任务扩展性验收

**目标**

验证 prompt 架构不是 push recovery 专用，可以被后续新任务复用。

**任务**

- [ ] 创建或使用一个最小 dummy/fake task adapter 进行 prompt 生成测试。
- [ ] dummy/fake task prompt 不含 push recovery 专属字段。
- [ ] 共享 prompt assembler 不需要修改即可接入该 task adapter。
- [ ] 共享 accept/reject 逻辑不需要知道该 task 的 success 判定细节。
- [ ] 测试证明任务特殊配置可以分离在 task plugin/adapter 下。

**严格验收**

- fake task regression 通过。
- shared layer 没有 push recovery success predicate 或 RewardInputs 语义泄漏。

## Step 15: 服务器真实产物对照

**目标**

把本地通过的 prompt 升级同步到服务器环境后，用服务器真实产物做最终对照。

**任务**

- [ ] 在 `/data/yuyangliu/AGENT/true_RL_Agent` 生成至少一个 live smoke prompt。
- [ ] 生成至少一个多轮 prompt artifact。
- [ ] 对照 saved `prompt_payload.json` 与 provider user payload capture。
- [ ] 检查每轮 payload、candidate、ledger、public result 的对应关系。
- [ ] 若服务器 smoke 暴露 prompt/schema/repair 问题，回到对应步骤修正后重新验收。

**严格验收**

- 服务器生成的 `prompt_payload.json` 与本地期望 schema 一致。
- 服务器 prompt 不包含 Windows 本地路径、server private path、secret 或 hidden/trusted 信息。
- 后续轮 prompt 能看到上一轮公开失败原因和 runtime activation 摘要。

## Step 16: 文档与维护验收

**目标**

把 prompt 架构约束沉淀成维护文档，避免后续新任务接入时重新把任务语义写进共享层。

**任务**

- [ ] 文档说明 prompt 顶层 sections、共享层职责、任务层职责、安全边界。
- [ ] 文档说明新任务接入时需要实现哪些 task-owned prompt 内容。
- [ ] 文档说明哪些信息禁止进入 prompt。
- [ ] 文档说明课程能力本轮不在范围内，后续若接入必须另开设计。
- [ ] 文档能指导一次新任务 prompt adapter 审计，而不需要阅读旧 Phase 8E 代码。

**严格验收**

- 文档、测试、真实 artifact 三者说法一致。
- 不再存在一份旧计划和一份新计划互相冲突的情况。

## Verification Matrix

**本地测试**

- `python -m pytest true_RL_Agent/tests/unit/test_openai_compatible_provider.py -q`
- `python -m pytest true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py -q`
- `python -m pytest true_RL_Agent/tests/unit/test_memory_context_phase13.py -q`
- `python -m pytest true_RL_Agent/tests/unit/test_push_recovery_task.py -q`
- `python -m pytest true_RL_Agent/tests/integration/test_live_loop.py -q`
- `python -m pytest true_RL_Agent/tests/integration/test_dry_run_audit_loop.py -q`

**服务器 artifact 检查**

- 生成至少一个 live smoke prompt 和一个多轮 prompt。
- 对比保存的 `prompt_payload.json` 与 provider user payload capture。
- 确认 prompt 顶层 sections、public-only checks、runtime activation、candidate load/eval load evidence。

## 完成标准

整个 prompt 升级完成时必须同时满足：

- 新版 prompt 信息足量，LLM 能看懂任务、边界、反馈、候选包要求和失败原因。
- Provider 实际发送内容与保存的 `prompt_payload.json` 同源一致。
- 任务特殊语义分离，push recovery 语义不污染共享层。
- RewardInputs contract 由 task-owned prompt adapter 提供，非 push 任务不会继承 push-specific 字段说明。
- Provider parse/schema/materialize/verifier/runtime 失败能进入 public-safe repair feedback。
- public-only 安全边界没有退化。
- 单轮和多轮 smoke 都能证明 prompt 不再只有薄 API contract。
- 后续新任务能通过 task-owned adapter 接入 prompt，而不是改共享 prompt builder。

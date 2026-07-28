# Phase9: Runner-Owned Curriculum/Environment Schema Preview

## 目标

Phase9 的目标不是开放 LLM 直接改课程或训练环境，而是为未来开放 `curriculum_design` 与 `train_environment_design` 建立统一的 schema preview、readiness gate 和真实 prompt 构建入口。

本阶段必须满足：

- 默认情况下课程与环境能力仍不可写、不可执行、不可 prompt-visible。
- 只有任务显式启用、runner compiler 就绪、runner verifier 就绪三项同时满足时，能力才可以进入 `candidate_capabilities.prompt_visible_capabilities`。
- readiness gate 必须严格要求布尔值 `true`；字符串、数字或其它 truthy 值不能被当作就绪。
- 即使进入 prompt-visible，也只是 schema preview：`candidate_writable=false`、`effective_now=false`。
- provider JSON 输出合同仍不允许写课程、环境、`operation_diff` 或 runner patch。
- schema 只能描述 train-only intent，不能包含 eval protocol、success metric、budget、seed、checkpoint、PPO 或 runner source/code 的可修改入口。
- `final_distribution_policy` 必须机器校验为 `runner_owned_target_distribution_required`，不能只靠文档约定。

## 设计边界

公共层只负责能力注册、readiness gate、公开 schema preview 与公共安全校验。公共层不解释 push recovery 语义，也不决定某个任务是否应该开放课程或环境能力。

任务层未来只能通过显式 readiness 输入表达“本任务允许预览这个能力”；runner 层仍必须提供 compiler、materializer、verifier，并保持最终执行权。

当前主链路默认不开放课程/环境能力。`PublicContextBuilder.build_prompt_payload()`、`build_authoring_prompt_payload()` 与 `CandidateAuthoringLoop` 已支持显式 `runner_capability_readiness` 透传；但 live loop 没有自动构造 ready 状态，因此默认 live prompt 仍是 not-ready reserved 状态。

provider-facing 输出合同保持 candidate package only。当前可写内容仍是：

- `candidate_id`
- `rationale`
- `declared_changed_components`
- `components_py`

任何课程、环境、eval、budget、seed、checkpoint、PPO 修改请求都必须被拒绝。

## 调用流程

显式 readiness 流程为：

```text
PublicContextBuilder.build_prompt_payload(runner_capability_readiness=...)
or build_authoring_prompt_payload(runner_capability_readiness=...)
or CandidateAuthoringLoop(runner_capability_readiness=...)
-> prompting.payload.build_prompt_payload()
-> candidate_authoring_capabilities(runner_capability_readiness=...)
-> CandidateAuthoringCapabilityRegistry.to_prompt_section()
-> runner_capability_readiness_gate()
-> ready: schema_preview_only
-> not ready: reserved_until_ready
```

默认 `runner_capability_readiness=None`，因此两个 runner-owned capability 只出现在 `disabled_or_reserved_capabilities`，不携带完整 `output_schema`。

当 readiness 为 ready 时，registry 会把对应 capability 投影为 prompt-visible schema preview，但不会改变 provider JSON contract、component contract、allowed schedule keys 或 runtime effectiveness。

## 验收

本阶段验收必须覆盖：

- 默认 payload 不出现可写课程或环境 surface。
- 默认 reserved entry 不携带完整 `output_schema`，只携带 schema ref 和 not-ready gate。
- ready 后只出现 schema preview，且仍不可 candidate-writable / effective-now。
- ready 后 payload 不包含 runner source/code、raw seed、trusted eval、checkpoint path 等内部信息。
- provider output allowed keys 不新增课程、环境或 `operation_diff`。
- schema validator 优先拒绝 eval/budget/seed/checkpoint/PPO/runner patch 类越权字段。
- schema validator 必须 fail-closed：任何会被 public scrub 静默删除或替换的 key/value 都不能被接受。
- schema validator 至少校验 root object、required nested fields、array item object、未知 nested field 与 final target distribution policy。
- 显式 readiness 可以经过 `PublicContextBuilder`、authoring prompt helper 与 `CandidateAuthoringLoop` 到达 provider-facing payload。
- 默认 authoring/live 路径仍不开放课程/环境 prompt-visible surface。
- 新增 schema 不降低现有 RewardTerm、DiagnosticTerm、ScheduleTerm prompt 能力。

## 后续开放条件

未来真正开放课程或环境写入前，必须另行完成：

- task plugin 侧显式启用接口。
- runner compiler/materializer/verifier 完整实现。
- public repair feedback 与 lineage 记录。
- 训练侧执行结果的 public-safe 投影。
- 不改变 public eval、success metric、budget、seed policy、checkpoint policy、PPO 配置。

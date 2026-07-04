# History-Best Source 与受控 Authoring Loop 架构计划

## 目标

本计划用于把当前提示词系统从“一次性 prompt 生成 candidate”升级为“受控 authoring agent”：

- 让 LLM 能看到并合理利用当前最佳候选的代码与公共反馈。
- 避免看到 current-best 后退化成只调权重、改常数、复制代码。
- 保持公共层与任务特殊层语义边界清晰。
- 为未来开放课程、环境、更多源码读取权限预留统一扩展口。
- 继续保证 prompt artifact 可复现、public-only、安全可审计。

核心结论：不做完全开放式 Codex，不给 LLM 任意 shell 或任意仓库读取能力；做受控 source catalog、candidate lineage、分步 read-design-write-repair authoring loop、显式上下文帧管理和 capability registry。

## 当前系统状态核实

### 已有结构

当前 prompt 构建主链路是：

```text
ledger records
-> ExperimentRecord.from_ledger_record()
-> PublicContextBuilder.build()
-> TaskMemoryAdapter.build_public_summary()
-> TaskPromptAdapter.build_prompt_sections()
-> prompting.payload.build_prompt_payload()
-> OpenAICompatibleCandidateProvider serializes payload as user message
```

公共层已经有：

- `PublicContextBuilder`：只从 public record 构建 public context。
- `build_prompt_payload()`：统一装配 prompt payload，并用 `PROMPT_PAYLOAD_SECTION_ORDER` 控制顺序。
- `assert_public_only_prompt_payload()` / public safety scrub：限制 prompt-facing 内容。
- `required_candidate_package`、`loop_contract`、`control_effectiveness_contract`：定义候选输出与 runner-owned 边界。
- `public_history`：有轮次压缩与大小上限。
- `provider_or_runtime_repair_feedback`：provider/verifier/runtime 错误可以进入下一次 provider attempt。

任务层已经有：

- `TaskPromptAdapter`：任务拥有 task section、reward inputs contract、public eval contract、known examples、selection guidance。
- push recovery adapter 已经把 `RewardInputs` 字段语义放在任务层，而不是公共层。
- `TaskMemoryAdapter`：任务可以从 public records 压缩任务级反馈。

课程相关已有雏形：

- `CurriculumCapability` / `CapabilityRegistry` 已存在。
- 默认 registry 中已有 `task_default_curriculum`、`reward_weight_scale`、`external_push_curriculum`、`train_only_dr_scale` 等 capability。
- 这些 capability 当前大多不是 prompt-visible，也不是 LLM 可写 surface。
- `control_effectiveness_contract.runner_owned_extensions` 当前明确把 curriculum、observation projection、feature projection 标成不可 prompt writable。

### 当前缺口

当前系统还没有：

- `source_catalog`：没有统一声明哪些源码、接口、示例可以被 LLM 读取。
- `candidate_lineage`：没有把 latest accepted / current best 候选作为一等 prompt section。
- current-best `training_code/components.py` 的安全源码投影。
- 受控 source resolver：没有 logical id -> scrubbed source payload 的公共机制。
- 真正的 authoring loop：现在 live loop 有 provider attempts 和 repair feedback，但不是“source request / draft / static feedback / rewrite / final”的状态机。
- 每个 authoring turn 的上下文管理协议：还没有定义哪些信息必须重复、哪些只保留摘要、哪些 source payload 可以继续携带、何时截断。
- provider 协议当前是 candidate-only：`CandidateProvider.next_candidate()` 只能返回 `CandidatePackageRef`，OpenAI provider 只接受并 materialize `candidate_id/rationale/declared_changed_components/components_py`。因此 `source_request` 不是加字段，而是未来要迁移到 `AuthoringProviderResponse = SourceRequest | CandidateDraft` 这类新协议。
- live loop 当前在 `CandidateRuntimeError` 后会生成 runtime repair feedback 并继续 provider attempt，可能导致同一 round 内再次跑真实 runner。这与“真实 train/eval 只对最终 candidate 执行一次”的目标边界冲突。
- 当前 `_candidate_components_source()` 已经从 candidate path 直接读取 `training_code/components.py` 给 runtime repair feedback，但它没有 logical id、hash、artifact、budget 和 scrub 协议；后续应收敛到统一 SourceResolver，而不是新增第二套读取路径。
- anti-degenerate 检查：目前有 parameter-only / surface classification，但没有专门防“复制 current-best、只改数字、只改名字”的公共反馈机制。
- 泛化的 candidate capability authoring 模型：RewardTerm、DiagnosticTerm、ScheduleTerm 已可用，但课程/环境未来开放时还没有统一的 public schema、materializer、verifier 接入协议。

### 当前需要保持的边界

- `candidate.path`、真实文件系统路径、checkpoint、seed、runner internals 不能直接进入 prompt。
- public history 只应包含 public projection，不能把任务内部私有字段塞进公共层。
- 真实 train/eval 仍应每 round 只对最终 candidate 执行一次，不能在同一轮内让 LLM 通过多次 eval 搜索 public split。
- 当前 runtime failure repair 行为需要改边界：一旦 candidate 已进入真实 runner 并发生 `CandidateRuntimeError`，本 round 应结束；public-safe runtime failure 进入 ledger 和下一 round prompt，而不是同 round 再次 provider attempt。
- 公共层不能理解 push recovery 的专用失败语义；任务层继续提供 task-specific summary 和 guidance。

## 架构原则

### 1. 创新性不做强判定

系统不应该尝试判断“这个设计是否有创造力”。自动创新评分会误伤真实有效的小机制改动，也容易变成新的 LLM 调参层。

公共层只做 anti-degenerate：

- 是否只改数字常量、weight、clip、名字或 rationale。
- 是否基本复制 current-best，没有机制级差异。
- 是否没有有效 callable。
- 是否把 parameter-only 伪装成 code-level policy improvement。

anti-degenerate 结果默认作为 authoring feedback，不直接等价于最终拒绝。最终性能仍由 runner train/eval 和 accept decision 判断。

### 2. 源码可以读，但必须通过 source catalog

允许 LLM 读取源码，但不允许任意文件读取。所有可读内容必须由公共层或任务层注册成 logical source item：

```text
logical_id
role
candidate_use
visibility
inline_by_default
requestable
max_bytes
content_sha256
source_owner
scrub_policy
```

prompt 中只暴露 logical id、用途、hash、大小和 scrubbed content。真实路径只留在 resolver 内部。

短期不依赖 LLM 主动请求 current-best 源码。latest accepted `components.py` 小于硬上限时，应在 candidate draft frame 默认内联；超过上限时提供机制摘要、hash、logical id 和明确 directive。否则会把问题从“没有给 current-best 信息”变成“只给目录但模型没有读”。

### 3. 公共层管 authoring 机制，任务层管任务语义

公共层负责：

- source catalog 与 resolver 协议。
- candidate lineage。
- authoring loop 状态机。
- static feedback 类型。
- anti-degenerate 检查。
- capability registry 与 candidate capability contract。
- prompt payload section 顺序和 safety assertions。

任务层负责：

- 任务目标与 invariant。
- reward inputs 字段语义。
- public eval contract。
- public feedback summary 的任务解释。
- 任务专属 known examples。
- 哪些课程/环境 capability 对这个任务 prompt-visible。

### 4. 未来权限开放必须 capability 化

后续开放课程、环境、观察投影等权限时，不应该继续给 prompt 打补丁。每种可写 surface 都必须是一个 capability：

```text
capability_id
kind
owner
prompt_visible
effective_now
candidate_allowed_actions
source_items
output_schema
materializer_owner
verifier_checks
runtime_effect
forbidden_surfaces
public_feedback_projection
```

LLM 只能通过 capability 输出 schema 表达意图；runner-owned compiler/materializer/verifier 决定如何执行、裁剪或拒绝。

### 5. 上下文是分层帧，不是完整对话转录

加入 source-read 后，authoring 不再是一次 prompt 能解决的问题。系统必须把每个 provider turn 渲染成有边界的 `AuthoringContextFrame`，而不是把之前所有 prompt、源码、错误、候选包无限拼接。

每个 frame 都应由固定层组成：

- `stable_contract_frame`：任务身份、loop/output contract、candidate capability、reward/eval/control contract。变化少，可以稳定复用。
- `round_goal_frame`：本轮 public feedback summary、runtime feedback summary、exploration directive、candidate lineage 摘要。
- `source_catalog_frame`：可读 logical id、用途、大小、是否内联，不包含未请求的大源码。
- `source_payload_frame`：本轮已返回且仍需要的 scrubbed source。只保留少量全文，其余降级为 logical id + hash + 用途摘要。
- `attempt_feedback_frame`：最近一次 materialize/verifier/smoke/anti-degenerate 的 public-safe feedback。
- `draft_context_frame`：需要 repair 时携带最近失败 candidate 的必要片段或摘要，不携带无限历史。

上下文 renderer 必须保证每个 frame 有硬上限，并且每次渲染的内容足以完成当前 turn 的目的：

- source request turn 需要 source catalog 和读取目的，不需要完整 failed candidate。
- candidate draft turn 需要合同、lineage、已读源码和探索指令，不需要全部旧 repair 记录。
- repair turn 需要最近失败 candidate、错误反馈和相关合同，不需要完整 public history。
- rewrite turn 需要 current-best / failed draft 的对照摘要和 exploration directive，不需要全部源码全文。

V1 仍保持 candidate-only provider 接口，因此实际发给 provider 的单轮 prompt 必须等价于 `candidate_draft` frame：带够任务合同、lineage、current-best 摘要或内联源码、public feedback 和探索指令。`initial_authoring` 在 V1 可作为 artifact/frame 分类和未来 V2 入口，但不能让 live provider 只收到 source catalog 而没有可写候选所需信息。

所有 frame 必须写入 prompt artifact 或可由 artifact 中的 hash/source payload 重建，保证实验可复现。

### 6. Source request 是 provider 协议迁移，不阻塞 history-best

`source_request` 需要 provider 输出协议从 candidate-only 升级为 union response。它会影响：

- `CandidateProvider` 返回类型。
- OpenAI provider 响应解析与 materialization 时机。
- provider JSON contract。
- replay/audit provider。
- live loop authoring 状态机。
- tests 和 saved prompt artifact 格式。

因此 source_request 不能作为 history-best 的前置条件。第一阶段应保持现有 provider 接口不变，通过默认内联小型 current-best source、source catalog、candidate lineage 和 context frame 先解决“下一轮看不到当前最佳代码”的核心问题。

## 目标架构

### Initial Authoring Prompt 构建顺序

初始 authoring prompt 只负责让 LLM 明确任务、边界、可读目录和本轮目标；它不应该默认携带所有可读源码全文。建议将初始 payload 顺序调整为：

```text
task
loop_contract
required_candidate_package
candidate_capabilities
reward_inputs_contract
public_eval_contract
control_effectiveness_contract
source_catalog
candidate_lineage
public_feedback_summary
runtime_feedback_summary
public_history
known_good_examples
exploration_directive
selection_guidance
provider_or_runtime_repair_feedback
schema_version
```

顺序理由：

- 先交代任务、角色、输出合同。
- 再交代当前可写 capability 和不可越界边界。
- 再给源码目录与 current-best lineage。
- 再给反馈、历史、示例和本轮探索指令。
- repair feedback 放在靠后位置，作为本次 attempt 的局部修复上下文。

后续 source-read / candidate draft / repair turn 不应重新发送完整初始 payload，而应由 `AuthoringContextFrame` 按 turn 目的选择必要层，并记录省略内容。

### SourceCatalog

新增公共 `SourceCatalog`，由公共层和任务层共同贡献 item，但由公共 resolver 统一校验。

首批 source item：

- `api/candidate_components.py`
  - 类型：公共 API。
  - 来源：`true_rl_agent.candidates.components`。
  - 作用：让 LLM 明确 `RewardTerm`、`DiagnosticTerm`、`ScheduleTerm`、`TrainingComponents` 的签名。
  - 建议默认内联或提供精简接口摘录。

- `current_best/training_code/components.py`
  - 类型：latest accepted candidate source。
  - 来源：raw ledger 中 latest accepted candidate path，经 resolver 读取。
  - 作用：继承、重写、对照当前最佳机制。
  - delivery：小于硬上限时在 candidate draft frame 默认内联；超限时给摘要、hash、logical id 和明确读取/重写 directive。
  - prompt 中不能出现真实路径。

- `attempt/latest_failed/training_code/components.py`
  - 类型：最近一次失败 candidate source。
  - 来源：provider materialized candidate path，经同一 SourceResolver 读取。
  - 作用：static repair 或下一轮 runtime failure 复盘。
  - 说明：替代当前 `_candidate_components_source()` 的直接读文件路径，避免平行 source resolver。

- `examples/known_good_reward_terms.py`
  - 类型：任务或公共示例。
  - 来源：任务 adapter 或公共 examples registry。
  - 作用：提供语法、tensor shape、safe reward pattern 参考。

未来可以增加：

- `api/curriculum_design_schema.json`
- `api/environment_design_schema.json`
- `examples/curriculum_stage_design.json`
- `examples/train_only_environment_variation.json`

### SourceResolver 安全规则

resolver 必须满足：

- logical id 必须来自 closed allowlist。
- 本地路径必须 resolve 到允许根目录。
- 只读明确允许的相对路径，例如候选包内 `training_code/components.py`。
- 单文件和单轮总 bytes 硬上限。
- scrub 绝对路径、secret、raw seed、checkpoint path、private artifact 标记。
- prompt payload 记录 logical id、content、content_sha256、bytes、scrub status。
- 读取失败必须变成 public-safe feedback，不暴露真实路径。
- 现有 `_candidate_components_source()` 应迁移为 SourceResolver 的一个调用点，logical id 使用 `attempt/latest_failed/training_code/components.py`，并记录 hash、bytes、scrub status 和 artifact 引用。

### CandidateLineage

新增 prompt section：

```text
candidate_lineage:
  schema_version
  status: none | bootstrap_scaffold | accepted_current_best
  current_best:
    round_id
    candidate_id
    public_score
    changed_components
    source_logical_id
    source_sha256
    source_summary
    source_excerpt_or_inline_content
    source_delivery: inline | requestable | omitted_over_budget
  guidance:
    - current-best 是可继承材料，不是必须小改的模板
    - 可以继承、重写、组合或反证其机制
    - 若只做微小改动，需要给出机制理由
```

第一轮或没有 accepted 时：

- `status = none` 或 `bootstrap_scaffold`。
- 如果有 bootstrap scaffold，需要明确它不是 performance baseline。

默认策略：

- latest accepted `components.py` 小于硬上限时，在 candidate draft frame 默认内联。这样不依赖 LLM 主动 source_request。
- 超过硬上限时，提供机制摘要、source logical id、hash 和明确 directive，要求模型在需要复用细节时读取该 logical id。
- initial authoring frame 可以只给 lineage 摘要；candidate draft frame 才展开当前 best source，避免所有 provider turn 重复携带源码。

### ExplorationDirective

新增公共 `exploration_directive`，由公共层根据轮次、历史、失败模式和 current-best 状态生成，不写死任务语义。

建议模式：

- `exploit_refine`
  - 在 current-best 上修一个明确机制缺陷。
  - 允许较小改动，但必须说明机制理由。

- `mechanism_rewrite`
  - 保留 API 与安全边界，重写核心 reward/diagnostic 机制。
  - 用于多轮停滞或近重复过多时。

- `recombine`
  - 组合 current-best、known examples 和 public feedback。
  - 鼓励跨信号组合，而不是只改一个权重。

- `diagnostic_probe`
  - 生成 public diagnostic，帮助后续轮次理解失败。
  - 不应伪装成 policy improvement。

- `counterfactual`
  - 针对 current-best 的一个假设做替代实现。
  - 适合 current-best 可能过拟合某个 public failure summary 时。

directive 只引导候选写作，不决定 accept/reject。

### Authoring Loop

将 live loop 中 provider attempts 抽象为公共 authoring loop。这个 loop 分两级演进，避免把 source_request 误认为小字段扩展。

V1 不改变 provider 接口：

```text
build initial/candidate-draft frame
-> 默认内联小型 current-best source
-> provider.next_candidate() returns CandidatePackageRef
-> provider JSON/schema/materialize
-> candidate verifier
-> component smoke check
-> anti-degenerate check
-> static repair/rewrite feedback
-> capped retry through existing candidate-only provider call
-> final candidate enters one train/eval
```

V1 的 repair 只允许 provider/schema/verifier/component-smoke/anti-degenerate 这类静态反馈。candidate 一旦进入真实 runner 并发生 `CandidateRuntimeError`，本 round 结束，runtime failure 进入 ledger 和下一 round public history。

V2 在 provider 协议迁移后支持 source_request：

```text
build authoring frame
-> provider response:
     SourceRequest 或 CandidateDraft
-> if source_request:
     resolver 返回 scrubbed source payload
     append authoring feedback
     render source-read follow-up context
     provider 再 request source 或 draft candidate
-> materialize candidate
-> provider JSON/schema check
-> candidate verifier
-> component smoke check
-> anti-degenerate check
-> repair/rewrite feedback
-> capped retry
-> final candidate enters one train/eval
```

同一 round 内允许多次静态 repair/rewrite，但只允许一次真实 train/eval。source_request 只能在 V2 中出现；V1 通过默认内联 current-best 和 known API 摘要解决 history-best 缺失。

Authoring feedback 类型：

- `source_request_denied`
- `source_payload_returned`
- `provider_json_error`
- `candidate_materialize_error`
- `verifier_error`
- `component_smoke_error`
- `anti_degenerate_warning`
- `forbidden_surface_error`
- `capability_schema_error`
- `runtime_failure_for_next_round`

所有 feedback 必须 public-safe。

### AuthoringContextFrame 与上下文预算

新增公共上下文帧协议，避免每轮对话上下文爆炸。

建议公共数据结构：

```text
AuthoringContextFrame:
  frame_id
  frame_kind:
    initial_authoring | source_followup | candidate_draft | repair | rewrite
  stable_contract_refs
  included_sections
  omitted_sections
  source_payloads
  source_payload_summaries
  feedback_items
  byte_limits
  token_budget_estimate
```

默认预算建议：

- `stable_contract_frame`：可重复发送，但要保持紧凑，避免把 task examples 混入公共合同。
- `public_history`：只保留 bounded summary 和最近少量 outcomes，不在 source/repair turn 里反复全文发送。
- `candidate_lineage`：保留 current-best 摘要；完整 source 只在小于上限且当前 turn 需要时携带。
- `source_payload_frame`：总 bytes 硬上限；超过后只保留最相关 source 的全文，其余保留 hash 和摘要。
- `attempt_feedback_frame`：只保留最近一次 blocking feedback 和少量历史摘要。
- `failed_candidate`：repair turn 可携带最近失败 `components.py`，rewrite turn 可携带摘要或机制 diff，不能累积所有失败版本全文。

上下文管理规则：

- 不保留完整 authoring transcript 给下一 turn；保留结构化状态。
- 每个 source item 都有 logical id、hash、bytes、last_used_turn。
- renderer 根据 `frame_kind` 决定展开哪些内容。
- 若内容被省略，必须显式列在 `omitted_sections`，说明省略原因和可恢复来源。
- source-read turn 不允许顺带加入新的 public eval 反馈。
- 静态 verifier/smoke feedback 可以进入同一 round repair；真实 train/eval feedback 只能进入下一 round public history。

### AntiDegenerateAssessment

新增轻量公共检查，不做创新评分：

```text
anti_degenerate_assessment:
  compares_to: current_best source hash
  classification:
    - identical_or_format_only
    - parameter_only_delta
    - rationale_only_delta
    - mechanism_delta_present
    - unknown
  findings
  feedback_for_author
```

允许策略：

- repair attempt 可以做小修。
- final candidate 如果是 `identical_or_format_only`，应要求 rewrite 或标记为 not policy-improvement eligible。
- `parameter_only_delta` 应与现有 surface policy 一起处理。

## 未来课程与环境权限适配

### 课程开放

课程不应让 LLM 修改 runner 代码。应开放为 capability：

```text
capability_id: curriculum_design
kind: train_protocol_design
candidate_output: operation_diff.curriculum_design 或独立 curriculum_plan
materializer_owner: runner_curriculum_compiler
verifier_checks:
  - allowed capability ids
  - train-only only
  - no eval protocol changes
  - budget invariant
  - seed invariant
runtime_effect:
  - runner compiles/clamps stage schedule
```

Prompt 中只暴露：

- 可用课程 capability id。
- 每个 capability 的作用、参数范围、是否 effective now。
- public feedback 中与课程相关的失败摘要。

不暴露：

- eval seed。
- runner 内部实现。
- checkpoint policy。
- 私有 benchmark 判据。

### 环境开放

环境参数也应 capability 化：

```text
capability_id: train_environment_design
kind: train_distribution
candidate_output: environment_design
materializer_owner: runner_environment_compiler
verifier_checks:
  - train-only distribution
  - allowed parameter ranges
  - no public eval changes
  - no task invariant weakening
runtime_effect:
  - runner applies train-time environment variation
```

允许示例：

- train-only push distribution progression。
- train-only domain randomization within allowed ranges。
- terrain family selection if task allows。

禁止：

- 改 eval protocol。
- 改 success metric。
- 改 seed policy。
- 改 training budget。
- 暴露 raw simulator internals。

### 更多源码读取权限

未来如果需要更多 source-read，仍然通过 source catalog：

- 公共 API 和 schema 可以内联。
- runner 实现默认不暴露源码，只暴露接口、作用和不可变边界。
- 若必须暴露某段 runner-owned helper，必须先做 source item 审计，并给出 logical id、用途和最大字节数。

## 实施计划

### 阶段 0：上下文帧与预算协议

目标：

- 定义 `AuthoringContextFrame` 数据结构。
- 定义 `frame_kind`：`initial_authoring`、`candidate_draft`、`static_repair`、`rewrite`。`source_followup` 暂时保留为未来 V2。
- 定义每类 frame 的必需信息、可选信息、省略信息和硬上限。
- 在 prompt artifact 中记录 frame 内容、omitted sections、source hashes 和 budget metadata。

测试：

- 每个 frame 都能独立通过 public-only assertion。
- source/repair turn 不重复携带完整 public history。
- 省略内容被显式记录，并能从 artifact/hash/source payload 追踪。
- 所有 frame 都有 bytes/token budget 估算和硬上限。

### 阶段 1：SourceCatalog、CandidateLineage 与 provider-compatible history-best

目标：

- 新增 `SourceCatalog` / `PromptSourceArtifact` 数据结构。
- 注册首批 logical source item：candidate API、current-best source slot、known-good examples slot、latest failed attempt source slot。
- 新增 current-best resolver，只读取 latest accepted candidate 的 `training_code/components.py`。
- 新增 `candidate_lineage` section，区分 `accepted_current_best`、`bootstrap_scaffold`、`none`。
- 调整 initial authoring payload 顺序。
- 保持现有 `CandidateProvider.next_candidate() -> CandidatePackageRef` 不变。
- latest accepted `components.py` 小于硬上限时，在 candidate draft frame 默认内联；超限时给摘要、hash、logical id 和 directive。
- 初始 prompt 只暴露 source catalog、必要接口摘要、candidate lineage 摘要，不默认塞入所有可读源码；candidate draft frame 才按需展开 current-best。

测试：

- source catalog 中每个 item 都有 logical id、role、candidate_use、max bytes、hash 或 pending 状态。
- initial authoring prompt 足以让 LLM 理解边界和可读资源。
- candidate draft frame 在 current-best 小于上限时默认包含源码。
- 无 accepted record 时不会失败。
- 有 accepted record 时所有 round 都能看到 current-best lineage 摘要。
- current-best source bytes/hash 稳定。
- public-only assertion 覆盖 source content。
- prompt 不出现真实路径。
- candidate path 不进入 prompt。

### 阶段 2：收敛现有 candidate source 读取路径

目标：

- 把当前 `_candidate_components_source()` 迁移到统一 SourceResolver。
- logical id 使用 `attempt/latest_failed/training_code/components.py`。
- runtime/static repair 读取失败 candidate source 时记录 hash、bytes、scrub status 和 artifact 引用。
- 删除或降级直接 path read 的平行逻辑，避免第二套 source resolver。

测试：

- failed candidate source 仍可用于 static repair feedback。
- repair feedback 不包含真实路径。
- oversized failed candidate source 被截断或摘要化，并记录 omitted reason。
- current-best source 与 failed-attempt source 走同一 scrub/hash/budget 协议。

### 阶段 3：修正 runtime failure 边界

目标：

- 修改 live loop：candidate 一旦进入真实 runner 并发生 `CandidateRuntimeError`，本 round 结束。
- runtime failure 作为 public-safe ledger record 和下一 round prompt feedback，不在同一 round 触发 provider retry。
- 同一 round 的 repair/rewrite 仅限 provider/schema/verifier/component-smoke/anti-degenerate 等静态反馈。

测试：

- CandidateRuntimeError 后不会再次调用 provider 并重新跑 runner。
- runtime failure 进入下一 round public context。
- provider/schema/verifier/static smoke failure 仍可在同一 round 内 repair。
- live summary 和 ledger 能区分 static repair 与 runtime failure for next round。

### 阶段 4：provider-compatible static authoring loop

目标：

- 将 `live_loop.py` 中 provider attempts 抽出为 `CandidateAuthoringLoop`。
- V1 只支持 candidate-only provider response，不支持 source_request。
- 支持 draft、materialize、provider JSON/schema、verifier、component smoke、anti-degenerate、static repair/rewrite。
- 每个 provider turn 使用 `AuthoringContextFrame` 渲染，不使用无限 transcript。
- 保持最终 train/eval 只执行一次。

测试：

- JSON repair。
- verifier repair。
- component smoke repair。
- repair turn 只携带最近失败 candidate 和 blocking feedback。
- max attempts 到达后 public-safe failure record。
- 现有 OpenAI provider、ReplayProvider、audit loop 不需要 source_request 迁移即可继续工作。

### 阶段 5：component smoke 与 anti-degenerate

目标：

- 对 `components.py` 做轻量 smoke：import/build、callable 存在、reward output shape/finite。
- 公共层只负责调度 smoke、记录 public-safe 结果和执行通用边界；task-owned adapter 或 runtime adapter 负责提供 RewardInputs fixture、字段 shape 与任务相关输入假设。
- 对 current-best 做轻量 diff classification。
- anti-degenerate 只给明显退化反馈，不做创新评分。

测试：

- identical source 被标记。
- 只改数字被标记 parameter-only/near-duplicate。
- 新增有效 RewardTerm 被标记 mechanism_delta_present。
- component smoke 不在共享层硬编码 push recovery 字段、shape 或失败语义。
- warning 不泄露路径和私有信息。
- warning 不阻止合理 repair 小修。

### 阶段 6：AuthoringProviderResponse 协议迁移

目标：

- 引入 `AuthoringProviderResponse = SourceRequest | CandidateDraft` 或等价协议。
- 修改 `CandidateProvider` 或新增 `AuthoringCandidateProvider`，避免直接破坏所有 replay/audit provider。
- 修改 OpenAI provider：先解析 provider response mode，再决定 source request 或 materialize candidate。
- 修改 provider JSON contract，明确 candidate draft 与 source request 是互斥模式。
- 迁移 tests、saved prompt artifact 和 audit provider。

测试：

- candidate-only provider 兼容路径仍可用。
- SourceRequest 不会被旧 materializer 当 unsupported key 误报。
- CandidateDraft materialization 与当前 provider 行为一致。
- replay/audit provider 有明确适配层。

### 阶段 7：source request 服务

目标：

- Provider 可以返回受控 `source_request`。
- Runtime 只服务 allowlisted logical id。
- 返回 source payload 后渲染 `source_followup` frame，允许 LLM 继续请求少量源码或进入 candidate draft。
- source-read 次数、每次 logical id 数量和总 bytes 有硬上限。

测试：

- 非 allowlist id 被拒绝。
- 超大小 source 被拒绝或截断并记录。
- source payload 被保存到 prompt artifact。
- source-read follow-up 不重复发送完整初始 payload。
- 多次 source-read 后上下文仍在预算内。

### 阶段 8：capability registry 泛化

目标：

- 将候选可写 surface 抽象为 `CandidateAuthoringCapability`。
- RewardTerm、DiagnosticTerm、ScheduleTerm 先迁移到 capability 表达。
- Curriculum/environment 先以 disabled 或 non-prompt-visible 方式注册。

测试：

- 现有 reward prompt 不退化。
- 未 prompt-visible capability 不进入 payload。
- 任务层可以声明 task-specific capability metadata，但公共层不解释任务语义。

### 阶段 9：课程/环境开放预留

目标：

- 为 curriculum_design / train_environment_design 定义 schema 草案。
- 保持默认 disabled。
- 只有在 task plugin 明确开启且 runner compiler/verifier 就绪后才 prompt-visible。

测试：

- 未开启时 prompt 不出现可写课程/环境 surface。
- 开启时只出现 capability schema，不出现 runner 源码和 eval 内部。
- 任何 eval/budget/seed 变更请求被拒绝。

## 审计清单

每次改动后必须检查：

- prompt payload section 顺序符合逻辑，不按字母排序。
- 所有 round 都生效，不只第一轮。
- 没有把任务特殊字段塞入公共字段。
- prompt 不包含真实路径、raw seed、checkpoint、secret、private artifact。
- current-best source 只来自 latest accepted 或明确 bootstrap scaffold。
- latest accepted `components.py` 小于硬上限时，candidate draft frame 默认内联；不是只给 catalog 目录。
- current-best 超限时有摘要、hash、logical id 和明确 directive。
- source catalog 中每个 item 都有 logical id、role、candidate_use、max bytes、hash。
- `_candidate_components_source()` 这类直接 path read 没有形成第二套 source resolver。
- 每个 authoring turn 的 `AuthoringContextFrame` 都能说明 included/omitted sections。
- source-read / repair turn 没有重复携带完整 public history 和所有旧源码。
- candidate draft turn 带够任务、合同、lineage、已读 source 和探索指令，能支持本 turn 目标。
- V1 provider-facing prompt 等价于 candidate draft frame；不能只发 initial/source catalog frame。
- component smoke 的 task-specific input fixture 由任务层提供，公共层不硬编码任务字段语义。
- authoring loop 的 feedback 是 public-safe。
- 真实 train/eval 没有在同一 round 内被重复用于 LLM 搜索。
- 真实 train/eval feedback 没有进入同一 round 的 source/repair inner loop。
- `CandidateRuntimeError` 后本 round 结束，runtime failure 只进入下一 round public context。
- 在 `AuthoringProviderResponse` 协议迁移前，source_request 不进入当前 candidate-only provider contract。
- 课程/环境 capability 未开启时不可写。
- 新增公共能力不会绑定 push recovery 专用语义。

## 风险与应对

### 风险：current-best 导致保守小改

应对：

- `candidate_lineage` 明确 current-best 可继承、可重写、可反证。
- `exploration_directive` 周期性要求 mechanism rewrite / recombine / counterfactual。
- anti-degenerate 只防明显退化，不做创新评分。

### 风险：只给 requestable catalog，模型仍不读 current-best

应对：

- latest accepted source 小于硬上限时，在 candidate draft frame 默认内联。
- initial frame 只给目录和摘要，candidate draft frame 才展开源码，避免所有 turn 重复携带。
- 超限时给机制摘要、hash、logical id 和明确 directive，要求需要细节时读取。
- 不把 source_request 作为 history-best 的前置条件。

### 风险：source-read 扩大泄露面

应对：

- closed allowlist。
- logical id，不暴露真实路径。
- scrub + hash + bytes cap。
- source payload 持久化到 prompt artifact，保证可复现。

### 风险：source_request 低估 provider 迁移成本

应对：

- V1 保持 `CandidateProvider.next_candidate() -> CandidatePackageRef` 不变。
- V2 单独引入 `AuthoringProviderResponse` 或新 provider protocol。
- OpenAI provider、ReplayProvider、audit loop、provider contract、tests 和 artifact schema 一起迁移。
- source_request 不作为早期 history-best/source inline 的阻塞条件。

### 风险：authoring 对话上下文膨胀

应对：

- 不把完整 authoring transcript 当作下一 turn 输入。
- 用 `AuthoringContextFrame` 渲染当前 turn 所需信息。
- source payload 只保留少量全文，旧源码降级为 logical id + hash + 摘要。
- public history 只在 initial/draft frame 中保留 bounded summary，repair/source-followup 默认不重复展开。
- 每个 frame 都记录 included/omitted sections 和预算。

### 风险：课程/环境开放变成改实验协议

应对：

- capability schema 只允许 train-only design。
- runner compiler/clamp/verifier authoritative。
- eval protocol、seed、budget、success metric 永远 runner-owned。

### 风险：公共层吸收任务语义

应对：

- 公共层只知道 capability、source、loop、feedback 类型。
- 任务目标、失败含义、reward input 字段解释继续由 `TaskPromptAdapter` 和 `TaskMemoryAdapter` 提供。

## 非目标

- 不实现开放式 shell agent。
- 不允许 LLM 任意读取仓库。
- 不把 authoring loop 历史完整转录进每个 provider turn。
- 不在同一 round 内做多次真实 train/eval 搜索。
- 不让公共层理解 push recovery 专用指标含义。
- 不用 LLM 自评替代 verifier、runner、accept decision。

## 推荐下一步

先实施阶段 0 和阶段 1：把 `AuthoringContextFrame`、预算协议、`source_catalog`、`candidate_lineage` 和 provider-compatible history-best 默认内联立住。这个顺序不破坏当前 `CandidateProvider.next_candidate()` 接口，又能先解决“模型看不到 current-best 代码”的核心问题。随后实施阶段 2 和阶段 3，收敛现有 `_candidate_components_source()` 读取路径并修正 runtime failure 边界。`source_request` 等到阶段 6/7 的 provider 协议迁移后再做。

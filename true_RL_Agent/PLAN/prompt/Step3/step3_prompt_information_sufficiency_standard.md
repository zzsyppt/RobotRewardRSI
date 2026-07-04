# Step 3 Prompt 信息足量标准定义

## 范围与状态

本文完成 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step 3。它定义后续 prompt 升级的最低信息足量标准和验收合同，不在本步骤实现新的 task adapter、payload assembler 或 push recovery prompt 内容。

本标准依据当前代码、Step1 基线审计、Step2 provider repair feedback、旧 Phase 8E prompt 结构参考制定。旧系统只作为 prompt 工程合同参考；课程 stage、curriculum capability registry、旧 `public_dev/public_accept` 结构不纳入本轮 prompt 升级。

## Step3 实施计划

1. 复核当前真实链路：`PublicContextBuilder.build_prompt_payload()`、`candidates.prompt_projection.build_prompt_payload()`、provider same-source 修复、Step2 repair feedback 入口。
2. 从 Step1 提取当前 prompt 缺口：任务目标、loop 边界、RewardInputs 语义、public eval contract、runtime/load evidence、candidate package 输出协议、bounded history。
3. 从旧 Phase 8E 提取可复用结构：`task`、`loop_contract`、`required_candidate_package`、`public_feedback_summary`、`public_history`、`known_good_examples`、`selection_guidance`。
4. 定义首轮、后续轮、repair attempt 三类 prompt 的必备 sections。
5. 明确每个 section 的所有权：shared layer、task-owned layer、runner-owned read-only evidence、forbidden/private surface。
6. 制定上下文预算和裁剪优先级，保证信息足量但不会无限增长。
7. 写出后续 Step4/5/6/8/11/12/13 可直接使用的严格验收标准。

## 信息足量的定义

一个 prompt 信息足量，指 LLM 在不访问 private/trusted/hidden 信息、不修改 runner-owned 逻辑的前提下，能明确回答以下问题：

1. 当前任务是什么，策略从什么初始化状态开始，训练与评估协议哪些部分固定不可改。
2. 候选包必须输出什么，允许写哪些 Python 组件，哪些 surface 真实影响训练，哪些只是诊断或只读证据。
3. RewardInputs 每个 prompt-visible 字段的 shape、单位、坐标系、来源状态、fallback 行为和可靠性边界是什么。
4. public eval 的 `status`、`public_score` 和公开 `metrics` 如何解释，哪些只是 read-only evidence。
5. 上一轮或近期多轮为什么失败，候选包是否真的加载、是否进入 eval、reward/diagnostic 是否实际执行、是否产生非零贡献。
6. 如果是 repair attempt，上一 attempt 的失败类别、公开原因和可行动修正方向是什么。

如果 prompt 只能列出 API 类名、字段名和 outcome 名称，但不能解释任务语义、反馈语义、输出合同和运行证据，则不满足 Step3 标准。

## 顶层 Section 标准

后续 payload assembler 应按稳定阅读顺序生成 prompt。推荐目标顺序如下：

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
12. `provider_or_runtime_repair_feedback`
13. `schema_version`

`provider_or_runtime_repair_feedback` 只在 repair attempt 或下一轮需要反馈上次 provider/runtime 失败时出现。`schema_version` 的位置可以由实现保留在首位或末位，但 provider user payload、saved artifact 和 ledger prompt ref 必须同源一致，且不能用字母序重排破坏人工审计顺序。

现有字段可以作为过渡 alias 保留，例如 `candidate_component_contract` 可并入 `required_candidate_package`，`candidate_repair_feedback` 可并入 `provider_or_runtime_repair_feedback`。验收看语义覆盖，不只看字段名。

## Prompt Variant 标准

### 首轮 Prompt

首轮没有历史反馈，但必须包含足够的任务和输出合同：

- `task`
- `loop_contract`
- `required_candidate_package`
- `reward_inputs_contract`
- `public_eval_contract`
- `control_effectiveness_contract`
- `known_good_examples`
- `selection_guidance`
- 空的或显式标注 no prior rounds 的 `public_feedback_summary`

首轮 prompt 不应依赖旧 run artifact 或隐含配置让 LLM 猜任务目标。

### 后续轮 Prompt

后续轮在首轮基础上必须额外包含：

- `public_history`
- `public_feedback_summary`
- `runtime_feedback_summary`
- latest candidate 摘要
- current best 或 accepted candidate 摘要；如果还没有 accepted candidate，必须明确 current recipe 只是 scaffold

后续轮必须能看出 public reject 是来自分数未提升、candidate verifier 失败、candidate runtime 失败、provider repair 失败，还是候选包没有真实生效。

### Repair Attempt Prompt

同一 round 内的 retry 或下一轮 repair 必须在对应基础 prompt 上额外包含：

- `provider_or_runtime_repair_feedback`
- previous attempt failure category
- public-safe reason
- public-safe corrective guidance
- 如有必要，包含上一 attempt 的候选摘要或短代码摘录，但必须受预算限制并通过 public-only 检查

repair feedback 只能指导修复候选响应或 `training_code/components.py`，不得要求修改 runner、eval protocol、budget、seed、checkpoint、PPO 或 accept rules。

## Section 归属标准

### Shared Layer

共享层可以生成或装配以下内容：

- `schema_version`
- `loop_contract` 的通用权限边界
- `required_candidate_package` 的通用 JSON/package/API 形状
- `control_effectiveness_contract` 的通用组件类别：`RewardTerm`、`DiagnosticTerm`、`ScheduleTerm`
- provider parse/schema/materialize/verifier/runtime failure 的通用 public-safe 分类
- public history 的通用压缩、排序、字节预算和 public-only sanitize
- provider request、saved prompt artifact、ledger prompt reference 的 traceability

共享层不得包含 push recovery success predicate、push failure taxonomy、push RewardInputs 字段语义或任何任务特定 public metric 解释。

### Task-Owned Layer

任务层 prompt adapter 必须拥有以下语义：

- `task` 的任务目标、机器人/环境摘要、policy initialization、fixed train/eval protocol summary
- `reward_inputs_contract` 的字段语义、shape、单位、坐标系、source status、fallback 行为
- `public_eval_contract` 的 `status`、`public_score`、公开 metrics 解释
- task-specific `public_feedback_summary`
- task-specific `runtime_feedback_summary` 的语义解释
- task-specific known-good examples
- task-specific selection guidance

对 `unitree_push_recovery` 而言，`push_active`、`push_vec_public` 等字段只能由 push task adapter 暴露和解释；非 push 任务不得从 shared prompt projection 自动继承这些字段。

### Runner-Owned Read-Only Evidence

以下信息可以投影为只读证据，但不能让 LLM 控制：

- accept/reject outcome、公开 reject reason、public score delta 摘要
- candidate package loaded / candidate code loaded in eval
- reward term calls、diagnostic calls、nonzero reward contribution
- mechanism attribution status/reason
- train/eval fixed protocol summary
- current best/latest candidate summary
- provider failure category 与 sanitized reason

如果暴露精确 accept threshold 或 min delta，必须明确标注为 public read-only evidence，且不得暗示候选可以修改它。默认标准是不暴露精确阈值，只暴露 public eval 输出语义和 runner-owned decision 摘要。

### Forbidden Surface

以下信息不得进入 prompt：

- private、hidden、trusted eval、trusted seed、raw seed details
- API key、token、secret、server credential
- checkpoint selector control 或 private checkpoint URI
- 绝对本地/服务器路径，例如 `D:/...`、`C:\...`、`/data/...`、`/home/...`
- eval protocol、success metric、training budget、PPO、runner code 的可写指令
- 未真实接线或当前不能由 LLM 控制的能力
- curriculum stage 设计、curriculum capability registry、runner-owned curriculum execution 细节

prompt 可以使用 logical artifact id、round id、candidate id 和相对的公开 artifact 名称；不得把真实机器路径作为模型可读上下文。

## Section 内容标准

### `task`

最低字段：

- `task_id`
- `name`
- `objective`
- `robot_or_environment`
- `policy_initialization`
- `fixed_train_protocol_summary`
- `fixed_eval_protocol_summary`
- `candidate_must_not_change`

`fixed_train_protocol_summary` 可以包含公开、只读的训练规模摘要，例如 env count、iteration count、steps per iteration；不得把这些写成候选可调参数。

### `loop_contract`

最低内容：

- LLM 只输出 candidate package，不输出 runner patch。
- LLM 只能使用 prompt-visible public inputs。
- runner 拥有训练、eval、checkpoint、seed policy、success metric、accept/reject。
- 候选不得修改 budget、seed、PPO、eval threshold、task invariants、checkpoint policy。
- 候选必须以可审计 code-level 机制为主，parameter-only 改动不能作为 policy improvement accept 的充分条件。

### `required_candidate_package`

最低内容：

- provider 必须返回一个 JSON object，不得在 JSON 外输出 prose。
- JSON 至少包含 `candidate_id`、`rationale`、`declared_changed_components`、`components_py`。
- `components_py` 必须定义 `build_training_components() -> TrainingComponents`。
- allowed import、allowed component classes、builder signature 必须清晰列出。
- `declared_changed_components` 必须与真实候选包语义一致，不能声明未生成或未接线的能力。
- 候选不得依赖文件系统、runner state、environment object、private eval 或 hidden seed。

后续 Step7 可以把 manifest、operation diff、diagnostics 等扩展为完整 candidate package 合同；在那之前，当前 JSON/`components_py` 合同必须在 user payload 中可审计。

### `reward_inputs_contract`

每个字段最低包含：

- `name`
- `shape`
- `dtype_or_tensor_kind`
- `units`
- `coordinate_frame`
- `source_status`
- `fallback_behavior`
- `candidate_usage_guidance`
- `known_limitations`

字段 contract 必须由 task-owned adapter 生成。共享层不得用 `RewardInputs.field_names()` 作为所有任务的完整语义合同。

### `public_eval_contract`

最低内容：

- 当前任务的 `PublicEvalResult(status, public_score, metrics)` 如何解释。
- `status` 的公开含义。
- `public_score` 的方向、尺度、缺失值含义。
- 关键 public metrics 的分母、单位和边界。
- gate/status 字段是 read-only evidence，不是候选可控 target。
- accept/reject 由 runner-owned decision 逻辑完成，candidate 不能修改。

push recovery success 判定和 failure taxonomy 必须留在 push task adapter，不得写进共享判定层。

### `control_effectiveness_contract`

最低内容：

- `RewardTerm`：作用于 prompt-visible RewardInputs，经 guard 后进入训练 reward，可能影响 policy。
- `DiagnosticTerm`：进入公开诊断和后续 prompt，不应单独作为 accept 条件。
- `ScheduleTerm`：只允许当前真实有效的 key；当前唯一 prompt-facing effective key 是 `reward_weight_scale`。
- runner-owned observation/feature/curriculum/DR/push execution 能力不是普通 candidate 可写 surface。

只列 `["RewardTerm", "DiagnosticTerm", "ScheduleTerm"]` 不满足信息足量标准。

### `public_feedback_summary`

最低内容：

- total rounds、accepted/rejected counts
- latest round outcome、reason、public score
- best public round 或 accepted candidate 摘要
- score trend 的短摘要
- task-specific public failure summary
- 明确区分 no candidate、verifier failure、runtime failure、provider failure、score delta not met

该 section 必须是摘要，不应复制完整 ledger。

### `runtime_feedback_summary`

最低内容：

- candidate package 是否加载
- candidate code 是否进入 eval
- reward term call count
- diagnostic call count
- nonzero reward contribution 状态
- diagnostic output 摘要
- mechanism attribution status/reason
- runner error category 与 sanitized reason
- provider failure category 与 sanitized reason

如果 public score 为 0 或没有提升，该 section 必须帮助区分“机制设计无效”和“候选根本未加载/未执行/shape 崩溃”。

### `public_history`

最低内容：

- 最近若干轮的 `round_id`
- candidate id 或 missing candidate reason
- outcome、decision reason、surface classification
- public score
- changed components
- verifier/runtime/provider failure category
- runtime activation flags

只保留 `round_id`、`outcome`、`surface` 不满足后续轮信息足量标准。

### `known_good_examples`

最低内容：

- 1 到 2 个 shape-safe 的 `RewardTerm` 示例。
- 至少一个 `DiagnosticTerm` 示例。
- 如启用 schedule，最多一个合法 `ScheduleTerm(key="reward_weight_scale")` 示例。
- 标明示例覆盖哪些 RewardInputs 字段。
- 明确示例只是 API/shape 参考，不是 reward menu 或推荐模板库。

### `selection_guidance`

最低内容：

- 优先真实 code-level training mechanism。
- 使用 public feedback 中的真实失败原因。
- 保持候选小步、可解释、可回滚。
- 不修改 eval、success metric、budget、seed、checkpoint、runner code、PPO。
- 不把 diagnostic 或 parameter-only 改动包装成 policy improvement。
- 如果可用 evidence 显示候选未加载或未执行，优先修复加载/shape/runtime 问题。

## 上下文预算标准

所有预算按序列化后的 UTF-8 JSON 近似大小计算，provider system message 不计入 user payload。

| Prompt 类型 | 目标大小 | 硬上限 |
| --- | ---: | ---: |
| 首轮 prompt | 16-28 KiB | 40 KiB |
| 后续轮 prompt | 22-36 KiB | 48 KiB |
| Repair attempt prompt | 24-42 KiB | 56 KiB |

Section 预算建议：

| Section | 软上限 |
| --- | ---: |
| `task` | 4 KiB |
| `loop_contract` | 4 KiB |
| `required_candidate_package` | 8 KiB |
| `reward_inputs_contract` | 8 KiB |
| `public_eval_contract` | 5 KiB |
| `control_effectiveness_contract` | 4 KiB |
| `public_feedback_summary` | 8 KiB |
| `runtime_feedback_summary` | 8 KiB |
| `public_history` | 12 KiB |
| `known_good_examples` | 8 KiB |
| `selection_guidance` | 4 KiB |
| `provider_or_runtime_repair_feedback` | 6 KiB |

裁剪优先级：

1. 永不裁掉 `task`、`loop_contract`、`required_candidate_package`、`reward_inputs_contract`、`public_eval_contract` 的最低字段。
2. 优先裁剪 examples、旧轮次 history、长 error details、长代码摘录。
3. `public_history` 默认保留最近 5 轮详细摘要，更早轮次聚合为 counts/trends。
4. 单条 error reason 建议不超过 500 字符；长 details 需要摘要并记录 omitted count。
5. 代码摘录默认不超过 120 行或 8 KiB，且只用于 repair 或 known-good example。
6. 所有被裁剪的 list 或 string 必须留下 `truncated: true`、`omitted_count` 或等价公开标记，避免误导 LLM 以为信息完整。

## 安全与 Traceability 验收

每个后续实现步骤都必须满足：

- saved `prompt_payload.json` 与 provider user payload 同源。
- ledger 或 summary 能追溯 prompt payload ref/path，但 prompt 内容本身不得包含绝对本地/服务器路径。
- payload 能被标准 JSON 工具解析。
- payload 通过 public-only 检查。
- fake non-push task prompt 不包含 push-specific RewardInputs 字段或 push recovery success 语义。
- provider parse/schema/materialize/verifier/runtime 失败能进入 public-safe repair feedback。
- repair feedback 不包含 raw path、secret、private/trusted/hidden 信息。
- prompt 不显示未接线能力为 effective control。
- curriculum 字段不作为本轮 prompt 缺口或验收目标。

## Step3 严格验收清单

- [x] 覆盖 task、loop boundary、candidate controls、RewardInputs contract、public eval contract、public feedback、runtime feedback、candidate package output contract、known-good examples、selection guidance。
- [x] 明确首轮、后续轮、repair attempt 的最低 sections。
- [x] 明确 shared layer 与 task-owned layer 的职责边界。
- [x] 明确 push recovery success 判定不能写入共享层。
- [x] 明确 RewardInputs contract 必须由 task-owned adapter 提供。
- [x] 明确 private/trusted/hidden/raw seed/server path/secret 禁止进入 prompt。
- [x] 明确课程字段不属于本轮 prompt 缺口。
- [x] 给出上下文预算、裁剪优先级和防爆炸规则。
- [x] 给出后续 Step4/5/6/8/11/12/13 可直接执行的验收标准。

## 对后续步骤的约束

- Step4 必须以 `TaskPlugin` 可选暴露或持有的 task prompt adapter 作为唯一 task prompt semantic 入口，不新建第二套 task registry。
- Step5 的 shared assembler 必须消费 task-owned sections，并停止把 shared `RewardInputs.field_names()` 当成所有任务的 prompt contract。
- Step6 的 push adapter 必须把 push-specific 字段、success 语义、public eval 解释和 failure taxonomy 留在 `tasks/push_recovery`。
- Step8 的 history projection 必须补上决策原因、public score、changed components、provider/verifier/runtime failure、runtime activation flags，并做预算限制。
- Step11 必须用 fake non-push task 回归检查任务语义隔离。
- Step12/13 必须验收真实生成 artifact，而不是只验代码结构。

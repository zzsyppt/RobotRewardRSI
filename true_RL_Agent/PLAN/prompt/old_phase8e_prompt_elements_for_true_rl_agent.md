# 旧 Phase 8E Prompt 中值得迁移到 true_RL_Agent 的部分

本文只总结旧系统 prompt 中对新系统仍有价值的通用部分。课程设计、curriculum stage、旧 push-specific capability registry 等内容暂不纳入本轮迁移建议。

## 当前判断

`true_RL_Agent` 已经有正确的大边界：独立项目、LLM 只写受控 candidate package、runner 拥有训练/eval/decision、public context 不泄露 hidden/trusted/private 信息。

但服务器真实 run 里的新系统 `prompt_payload.json` 仍然太薄，主要只有：

- `public_context`
- `reward_inputs`
- `candidate_allowed_controls`
- `candidate_component_contract`
- `anti_tuning_notice`
- `runner_owned_extensions`

相比旧 Phase 8E prompt，新系统缺少“让模型知道该怎么写、为什么写、不能碰什么、上一轮为什么失败、输出必须长什么样”的结构化合同。旧 prompt 的价值不是课程部分，而是它把候选包写作变成了一个明确的工程协议。

## 应迁移的 Prompt 结构

新系统 prompt 建议按逻辑顺序组织，而不是按字母顺序序列化：

1. `task`
2. `loop_contract`
3. `public_feedback_summary`
4. `required_candidate_package`
5. `known_good_examples`
6. `current_recipe`
7. `public_history`
8. `source_access_policy`
9. `candidate_self_audit_checklist`
10. `selection_guidance`

这里的重点是让模型先读任务和边界，再读反馈，再读候选包协议，最后读参考材料和自审规则。当前 provider-facing payload 不应使用 `sort_keys=True` 这类字母序输出。

## 1. 任务合同

旧 prompt 一开始就明确：

- 机器人和任务是什么。
- 优化目标是什么。
- policy initialization 是 random 还是 warm-start。
- reference/baseline 是否只作参考，是否作为训练起点。

新系统也应显式加入 `task` 块，而不是只把这些信息隐含在 config 或 public context 中。建议包含：

- `task_id`
- `robot`
- `objective`
- `policy_initialization`
- `fixed_train_protocol_summary`
- `fixed_eval_protocol_summary`
- `reference_or_baseline_policy_role`

这些字段只描述任务和 runner 固定协议，不给 LLM 控制权。

## 2. Loop 权限边界

旧 prompt 的 `loop_contract` 很有价值。它把 LLM 能做什么、不能做什么、runner 拥有什么权力写清楚。

新系统应保留类似结构：

- LLM 输出 candidate package，不输出 runner patch。
- LLM 只能使用 prompt-visible public inputs。
- runner 拥有训练预算、checkpoint selection、eval protocol、success metric、accept/reject。
- candidate 不允许修改 seed、budget、PPO、eval threshold、task invariant、checkpoint policy。
- candidate 允许的训练面要写成具体有效能力，而不是泛泛的类名。

当前 `candidate_allowed_controls = ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"]` 太粗。应改成更接近旧系统的表达，例如：

- `RewardTerm callables over prompt-visible public RewardInputs`
- `DiagnosticTerm callables over prompt-visible public RewardInputs returning public summaries`
- `ScheduleTerm only for currently effective runner keys: reward_weight_scale`

## 3. Public Feedback Summary

旧系统最大的优点之一是把多轮结果压缩成 prompt-facing feedback，而不是只给最近 outcome。

新系统应保留：

- 总轮数、accepted/rejected 计数。
- latest round、latest public-dev round、best public round。
- 上一轮失败原因。
- public score trend。
- dominant failure，且 `rate` 必须是真比例，不能把 success-only seed 排除出分母。
- motion quality gate / budget gate / safety gate 的公开失败摘要。
- task-specific public summary，例如 push recovery 的 failure mode、failure breakdown、motion quality。

不要把 hidden/trusted/private eval 放进 prompt。只放 runner 投影后的 public evidence。

## 4. RewardInputs 字段语义

旧 prompt 不只是列字段名，还解释每个 RewardInputs 字段的：

- shape
- units
- coordinate frame
- provenance
- fallback behavior
- source status

这部分很重要。新系统当前只列 `fields`，模型很容易误用，比如把 `base_pos_xy` 当世界绝对坐标、误解 `action_delta`、误把 zero fallback 当真实观测。

建议新系统把 `reward_inputs` 升级为结构化 contract：

- `field_order`
- `fields.<name>.shape`
- `fields.<name>.semantics`
- `fields.<name>.units`
- `fields.<name>.coordinate_frame`
- `fields.<name>.provenance`
- `fields.<name>.fallback_behavior`

字段只描述已经真实接线并经过 public projection 的输入。

## 5. 控制有效性合同

旧 prompt 会说明哪些 candidate 控制面“真的会影响训练”，哪些只是记录或诊断。这比简单列类名更有用。

新系统应加入 `control_effectiveness_contract`：

- `RewardTerm`：guard 后加入训练 reward，可能影响 policy。
- `DiagnosticTerm`：只进入 public summary / next-round feedback，不能单独 accept。
- `ScheduleTerm`：只允许已接线 key；当前只应是 `reward_weight_scale`。
- `runner_owned_extensions`：observation/feature projection 等不应被普通 candidate 写入。

这能减少模型伪造不存在能力、滥用 schedule 或把 diagnostic 当 decision metric。

## 6. Candidate Package 输出协议

旧 prompt 的 `required_candidate_package.provider_json_output` 非常有价值。新系统现在 provider system prompt 要求返回 `candidate_id/rationale/declared_changed_components/components_py`，但缺少完整可审计的 package 协议。

建议新系统明确：

- 返回一个 JSON object，不要 JSON 外 prose。
- 文件路径必须是相对路径。
- 必须生成 `candidate_manifest.yaml` 和 `training_code/components.py`。
- 可选但建议保留 `rationale.md`、`operation_diff.json`、`diagnostics.json`。
- `operation_diff.json` 用来说明真实 code-level 改动。
- `diagnostics.json` 用来说明假设、预期机制、失败模式目标。
- optional JSON 坏格式应进入 provider repair，而不是被当作系统内部错误。

这部分比“把 components_py 字符串塞进 JSON”更利于审计、repair、ledger 和后续 memory。

## 7. Known-Good Examples

旧 prompt 提供已验证的 runnable example，而且明确说明 example 不是 reward menu，而是 shape-safe 写法参考。

新系统应保留一个小型 examples 区：

- 1-2 个可以运行的 `RewardTerm` 示例。
- 至少一个使用向量 shape 的例子。
- 一个 `DiagnosticTerm` 示例。
- 一个合法 `ScheduleTerm(reward_weight_scale)` 示例。
- 标明覆盖哪些 public RewardInputs 字段。

不要把 examples 写成“推荐奖励模板库”，否则模型会机械复制；应强调它们只是 API 和 shape 参考。

## 8. Current Recipe 与 Public History

旧 prompt 区分：

- bootstrap scaffold
- current best
- accepted recipe
- recent public failures

新系统应加入类似结构，帮助模型做小步、可比较的改动：

- 如果没有 accepted candidate，明确 current recipe 只是 scaffold。
- 如果有 accepted candidate，给出 prompt-facing manifest、public summary、source ref。
- recent history 应包括 public outcome、failure reason、candidate surface classification、runtime activation summary。
- 不要暴露真实服务器路径、private artifact URI、trusted eval、checkpoint selector。

这能避免每轮从零发散，也能让模型针对真实失败改。

## 9. Source-Read / Source-Pack 协议

旧系统的 source-read 设计值得迁移，但新系统不一定马上实现完整多轮 source-read。

值得保留的原则：

- prompt 里只出现 logical source id，不出现服务器本地路径。
- source id 是 closed allowlist。
- provider 不能直接读文件系统。
- source-read 有明确预算，例如 `max_source_read_turns`。
- 每次 source request 必须说明 reason。
- source payload 由 runner/orchestrator 审计后返回。

第一版可以先做静态 `source_pack` manifest，后续再加真正的 source-read turn。

## 10. Candidate Self-Audit Checklist

旧 prompt 的 self-audit checklist 很实用，但它只是 guidance，不是信任边界。

新系统应加入：

- 输出 shape 是否符合 RewardTerm/DiagnosticTerm contract。
- component name 是否唯一。
- 是否只 import allowed API。
- 是否没有 runner hook / env object / filesystem / checkpoint / eval dependency。
- 是否没有 private eval 或 metric dependency。
- JSON 输出是否匹配 package schema。
- 是否针对 public feedback 中的真实失败模式。

同时保留旧系统原则：candidate 不应返回自证通过；最终由 verifier、runner 和 decision engine 判定。

## 11. Selection Guidance

旧 prompt 的 selection guidance 值得保留为短规则：

- 优先可解释、可验证的训练侧机制。
- 写真实 Python callable，不返回 declarative hint。
- 不修改 eval、success metric、budget、checkpoint、runner code。
- 使用 public feedback 和 recipe history。
- 如果是 schedule，只能用当前有效 key。
- 不能把 parameter-only improvement 当 code-level policy improvement。

这部分应短而明确，避免长篇说教。

## 暂不迁移的内容

本轮不要迁移旧 prompt 的课程设计细节：

- `curriculum_design_contract`
- `external_push_curriculum`
- `train_only_dr_scale`
- curriculum stage list / final consolidation schema
- push-specific capability registry
- 旧系统里围绕 curriculum attribution 的大段规则

新系统当前计划明确：第一版不要让 LLM 优化 curriculum schedule。可以保留“未接线能力不显示为 effective”的通用准入原则，但不要把旧 push curriculum prompt 搬进来。

## 建议落地优先级

1. 先重排 provider-facing payload 顺序，停止字母序输出。
2. 增加 `task` 和 `loop_contract`。
3. 把 `reward_inputs` 从字段名列表升级为字段语义 contract。
4. 增加 `public_feedback_summary`，聚合 public history 和 task memory。
5. 增加 `required_candidate_package.provider_json_output`。
6. 增加 `known_good_examples` 和 `candidate_self_audit_checklist`。
7. 后续再做 source-pack / source-read。

这样新系统 prompt 会先具备旧系统最重要的“工程合同感”，但不会把课程系统和 push-specific curriculum 复杂度过早搬进来。

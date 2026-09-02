# LLM 参与课程设计的架构顺序计划

日期: 2026-07-01

## 0. 总目标

让 LLM 参与课程设计,但不退化成只调 `push_magnitude_scale` / `domain_randomization_scale` 这类系数。

第一版目标不是让 LLM 申请任意新能力,而是在一组 **人类/runner 预批准能力** 上做高自由度课程设计:

```text
公开证据 -> LLM 课程假设 -> LLM 课程结构设计 -> runner 编译/拒绝/执行 -> trusted eval 裁决
```

长期目标才是:

```text
公开证据 -> LLM 能力请求 -> runner 审核 -> registry 扩展 -> LLM 使用新能力设计课程
```

也就是说,第一版先跑通"LLM 设计课程结构";第二版再开放"LLM 申请新能力"。

## 1. 当前架构基线

当前系统应被视为两层:

1. **内部 runner/sandbox contract**
   - runner 内部知道完整训练适配器、沙箱、trusted eval、预算和安全边界。
   - 一些字段或 schedule key 可能在内部 contract 中存在,但如果未真实接线,不能直接暴露给 LLM 当作可用能力。

2. **prompt-facing contract**
   - LLM 只看当前真实可用的 RewardInputs 和 effective schedule keys。
   - 未接线字段,例如 placeholder push signal,不应作为 recipe LLM 的可用输入。
   - LLM 不能因为看到内部计划字段就假设自己能使用它。

这个基线必须保留。课程开放必须经过 registry 和 prompt-facing projection,不能让 recipe LLM 直接看到未接线能力。

## 2. 第一版架构: 固定能力 Registry + LLM 课程设计

第一版不要让 LLM 同时"申请新能力"和"设计课程"。这样复杂度过高,也容易把项目变成不断实现 LLM 新功能请求。

第一版只做:

```text
PublicCurriculumEvidenceBundle
  -> Curriculum Designer LLM
  -> CurriculumHypothesis + CurriculumDesign
  -> Runner Curriculum Compiler
  -> Degenerate Curriculum Rejector
  -> Training Execution
  -> Trusted Evaluation
  -> Ledger / Public Feedback
```

第一版固定一组预批准能力:

- `reward_weight_scale`
- `external_push_curriculum`
- `train_only_dr_scale`
- `push_active` / `push_vec_public` reward-only signal
- `recovery_window_diagnostic`
- `public_push_probe`

这些能力由人类/runner 先批准和实现。LLM 第一版不能申请超出 registry 的新能力。

第一版 LLM 的自由度来自:

- 判断主要失败模式。
- 选择哪些已批准能力该用于课程。
- 设计阶段顺序。
- 设计每个阶段的训练目标。
- 决定 push、DR、recovery diagnostic 如何组合。
- 决定什么时候加难、什么时候巩固。
- 解释为什么这个课程不是单纯调参。

## 3. PublicCurriculumEvidenceBundle

先做 `PublicCurriculumEvidenceBundle`。

它给 LLM 的不是原始日志,而是 runner 投影后的公开证据。证据包应包含:

- 当前任务目标和约束。
- 当前 prompt-facing RewardInputs。
- 当前 effective schedule keys。
- 当前已批准 curriculum capability registry。
- 当前 best recipe 或 bootstrap recipe 的公开摘要。
- 最近若干轮 public history。
- public eval 结果,包括 success rate、failure breakdown、condition flags、motion metrics。
- 训练侧公开 diagnostics 和 reward summaries。
- public dev probes,例如 no-push、push、lateral-push、friction-shift、motor-shift、command-range probe。

证据包必须分层:

```text
public_dev_probes:
  进入 LLM prompt,用于课程设计。

public_accept_probes:
  不进入 LLM prompt,用于正式 accept。

hidden_audit_probes:
  不进入 LLM prompt,只用于最终审计或报告。
```

这样可以让 LLM 有足够证据做课程设计,同时避免它反复过拟合所有 probe。

## 4. LLM 第一产物: CurriculumHypothesis

LLM 第一产物不是代码,也不是数值 schedule,而是 `CurriculumHypothesis`。

它应回答:

- 当前主要失败模式是什么。
- 引用了哪些 public evidence。
- 为什么当前 reward-only recipe 不足。
- 应该使用 registry 中哪类课程能力。
- 预期这种课程如何改善训练。

示例:

- 普通站立成功,但 push probe 失败 -> 使用 external_push_curriculum。
- 同一 profile 成功,参数扰动失败 -> 使用 train-only DR。
- 前后恢复可以,侧向恢复差 -> push 方向分布应加强 lateral recovery。
- 动作稳定但恢复慢 -> 使用 recovery-window diagnostic 观察恢复质量。

注意: `recovery_window_diagnostic` 第一版只能是 report-only public diagnostic,不能自动变成 accept/reject metric。

## 5. LLM 第二产物: CurriculumDesign

LLM 第二产物是 `CurriculumDesign`。

它不是单纯的系数 schedule,而是课程结构:

- 阶段名称。
- 每个阶段的训练目标。
- 每个阶段启用哪些 registry capability。
- 每个阶段关注哪种失败模式。
- 阶段之间如何递进。
- 每个阶段的范围、强度和约束。
- 失败时如何降级。
- 最终如何回到 target distribution。

示例结构:

```text
balance_stabilization
  -> mild_push_recovery
  -> lateral_recovery_focus
  -> target_disturbance_distribution
  -> final_target_consolidation
```

LLM 可以填写数值,但数值只是结构设计的一部分。核心要求是:它必须解释每个阶段解决什么失败模式,以及为什么这些阶段组合能提高最终 trusted eval 表现。

## 6. CurriculumDesign DSL 的硬约束

DSL 必须显式包含 `final_consolidation`。

示例:

```json
{
  "final_consolidation": {
    "min_fraction_of_training": 0.2,
    "distribution": "runner_owned_target_distribution",
    "must_include_target_push": true,
    "must_include_target_dr": true
  }
}
```

runner 编译时强制:

- 最后至少 X% training progress 回到 target difficulty。
- push / DR 不能低于 runner floor。
- target push 事件必须真实出现。
- target DR 强度必须真实覆盖。
- final stage 不能只是象征性困难。

这条是硬规则,不是建议。

## 7. Runner Curriculum Compiler

runner 把 `CurriculumDesign` 编译成训练侧可执行课程。

runner 负责:

- 检查所有 capability 都来自当前 registry。
- 对所有数值做 clamp。
- 强制 target distribution floor。
- 记录 raw proposal。
- 记录 compiled/effective curriculum。
- 记录所有 clamp 和 reject reason。
- 保证 eval 不读取训练课程。
- 保证 budget、seed、metric、checkpoint selector 不变。

LLM 不能直接改 env object。LLM 只能提交课程设计,runner 决定它如何变成训练配置。

## 8. 退化课程拒绝规则

runner 不应只做 clamp,还必须拒绝投机课程。

以下课程直接 reject:

1. 全程 no-push,或 push 长期低于 target floor。
2. final stage 少于 runner 要求的最小训练比例。
3. 课程只降低难度,没有恢复到 target distribution 的阶段。
4. schedule 通过极端 `reward_weight_scale` 掩盖 reward 设计。
5. 使用未进入 registry 的 capability。
6. 数值虽然在 clamp 内,但长期停在 easy 端。
7. 课程改变导致 `planned_env_steps` / `env_step_budget` 不一致。
8. 课程需要改变 eval config、success metric、seed 或 checkpoint selector。
9. recovery-window score 被设计成 accept/reject metric,而不是 report-only diagnostic。

这一步是防止"自由课程设计"退化成"找 public probe 漏洞"的关键。

## 9. Candidate Recipe 生成

课程能力通过 registry 和 compiler 管理。candidate package 仍然只负责当前 prompt-facing contract 允许的内容。

candidate 可以使用:

- RewardTerm。
- 当前 prompt-facing ScheduleTerm。
- DiagnosticTerm。
- 公开 rationale 和 diagnostics。

candidate 不直接拿:

- env object。
- runner object。
- eval config。
- success metric。
- checkpoint selector。
- seed / budget controls。
- 未进入 registry 的能力。

如果某个能力还没进入 prompt-facing surface,recipe LLM 不能假设它可用。

## 10. 训练执行

第一版采用一次训练一次课程提案:

```text
训练开始前固定完整 curriculum
-> runner 编译
-> 训练全程按 effective curriculum 执行
-> 生成 checkpoint
-> trusted eval
```

第一版不做:

- per-step LLM 控制。
- always-online LLM 控制。
- 每个 chunk 动态改课程。

chunk-level 闭环课程放到后续版本。

## 11. Trusted Eval 和接受决策

评测保持当前原则:

- eval checkpoint-only。
- eval 不加载 candidate code。
- eval 不加载课程提案代码。
- eval 不读训练课程 schedule。
- eval config、seed、metric、budget、checkpoint selector 都由 runner 拥有。

课程设计只能影响训练出来的 checkpoint。最终是否有效,由 trusted eval 和 public projection 裁决。

## 12. Accepted Curriculum 不可变快照

accepted recipe snapshot 必须包含:

- reward code snapshot。
- raw curriculum proposal snapshot。
- compiled/effective curriculum snapshot。
- capability registry version/hash。
- curriculum compiler version/hash。
- public evidence bundle id/hash。
- prompt-facing contract version/hash。
- training/eval config hash。

原因:

- registry 更新后,同一个 `CurriculumDesign` 可能编译成不同训练行为。
- prompt-facing contract 更新后,同一个 candidate 的可见能力可能不同。
- 没有快照就无法复现实验 lineage。

## 13. Public Feedback 回流

每轮训练和 trusted eval 结束后,runner 生成新的 public feedback:

- 使用了哪些 capability。
- raw proposal 和 effective curriculum 的公开摘要。
- 哪些 clamp 被触发。
- 哪些 degenerate rule 差点触发或已经触发。
- public dev probes 的变化。
- public accept 的安全投影。
- trusted eval 的 public projection。

这些反馈进入下一轮 `PublicCurriculumEvidenceBundle`。

## 14. 区分课程能力和 recipe 代码能力

需要显式监控系统是否退化成课程超参搜索。

每个 accepted round 记录 improvement attribution:

```text
reward_code_changed: true/false
curriculum_changed: true/false
diagnostics_changed: true/false
capability_changed: true/false
gain_attributable_to: reward_code / curriculum / both / unclear
pure_knob_candidate: true/false
```

如果连续多个 accepted 都是 `pure_knob_candidate=true`,说明系统偏离 code-level autoresearch,变成 curriculum hyperparameter search。

不一定禁止纯课程候选,但必须显式暴露。

## 15. 第二版: 模板化 CapabilityRequest

第一版跑通后,再开放 CapabilityRequest。

CapabilityRequest 不能是自由自然语言。必须使用 typed request templates:

- `reward_only_signal_request`
- `train_only_intervention_request`
- `curriculum_executor_request`
- `public_probe_request`
- `diagnostic_request`

不同类型有不同必填字段。

例如 `train_only_intervention_request` 必须包含:

```yaml
train_only: true
policy_observation_changed: false
eval_config_changed: false
budget_changed: false
requires_clamp: true
requires_target_distribution_floor: true
eval_independence_probe: required
rollback: disable_registry_capability
```

LLM 可以提出新能力,但 runner 只能审核结构化请求。

## 16. 第二版能力分类

第二版审核时使用以下分类:

- A 类: reward-only signal。
- B 类: train-only intervention。
- C 类: curriculum executor。
- D 类: eval、seed、budget、metric、checkpoint selector 相关请求。
- E 类: filesystem、network、process、runner hook 相关请求。
- F 类: public diagnostic。
- G 类: public probe / report-only evaluation。

A/B/C/F/G 可以继续审。D/E 默认拒绝。

F/G 的边界:

- 可以作为 public evidence。
- 默认不能作为 accept/reject metric。
- 如果要进入 accept gate,必须 trusted maintainer 单独批准,并重新定义实验 protocol。

特别注意: `recovery_window_diagnostic` 只能先作为 F 类 report-only diagnostic。它不能由 LLM 请求后自动变成评分 gate。

## 17. 第二版 CapabilityRequest 审核

第二版审核权归 runner,不是 LLM。

审核分层:

1. schema 审核。
2. public-only 审核。
3. typed request template 审核。
4. 能力分类。
5. 安全边界审核。
6. static diff 白名单审核。
7. behavior probe。
8. eval independence probe。
9. 人工或 trusted maintainer gate。

LLM reviewer 可以做语义风险审查,但只能作为 advisory。最终授权必须来自 runner-owned checks 和 human/trusted gate。

## 18. 推荐实施顺序

### P0: 保持 prompt-facing 收窄

目的: 确保 LLM 只看到真实可用能力。

要做:

- 继续保持内部 contract 与 prompt-facing projection 分离。
- 未接线字段不进入 recipe prompt。
- 无效 schedule key 不进入 recipe prompt。

### P1: 人类/runner 固定第一版 Capability Registry

目的: 降低第一版复杂度,但保留足够课程自由度。

要做:

- 固定 `reward_weight_scale`。
- 固定 `external_push_curriculum`。
- 固定 `train_only_dr_scale`。
- 固定 `push_active` / `push_vec_public` reward-only signal。
- 固定 `recovery_window_diagnostic` 为 report-only。
- 固定 `public_push_probe` 为 public dev evidence。

### P2: 建立 PublicCurriculumEvidenceBundle

目的: 给 LLM 课程发现所需证据。

要做:

- 汇总 public eval。
- 汇总 public history。
- 汇总训练公开 diagnostics。
- 加入 public dev probe 摘要。
- 区分 public_dev_probes、public_accept_probes、hidden_audit_probes。
- 明确 forbidden data 不进入证据包。

### P3: 建立 CurriculumHypothesis + CurriculumDesign DSL

目的: 让 LLM 先设计课程结构,不是申请新能力。

要做:

- 支持失败模式假设。
- 支持 evidence refs。
- 支持阶段结构。
- 支持 registry capability 选择。
- 强制 `final_consolidation`。
- 支持 fallback。

### P4: 建立 Runner Curriculum Compiler

目的: 把课程设计变成训练侧可执行对象。

要做:

- registry capability 检查。
- clamp。
- target distribution floor。
- degenerate curriculum rejector。
- raw/effective curriculum 记录。

### P5: 训练执行和 trusted eval 闭环

目的: 验证 LLM 设计课程是否真的带来改进。

要做:

- 一次训练一次课程提案。
- runner 编译执行。
- smoke train。
- trusted eval。
- public feedback 回流。

### P6: accepted curriculum 不可变快照

目的: 保证 lineage 可复现。

要做:

- 快照 reward code。
- 快照 raw/effective curriculum。
- 快照 registry/compiler/evidence/prompt-facing contract hash。

### P7: attribution 监控

目的: 防止系统退化成纯课程调参。

要做:

- 记录 reward/curriculum/diagnostic/capability 是否变化。
- 标记 `pure_knob_candidate`。
- 汇总 accepted round 的 improvement attribution。

### P8: 再开放模板化 CapabilityRequest

目的: 在第一版稳定后提高自主性。

要做:

- typed request templates。
- A/B/C/F/G 可审,D/E 默认拒绝。
- public-only + static diff + behavior probe。
- human/trusted maintainer gate。

### P9: 再考虑 chunk-level 闭环

目的: 在 P1-P8 稳定后提高自适应能力。

要做:

- 每个训练 chunk 产生 public feedback。
- LLM 提出下一段课程。
- runner 审核后执行。
- 全程记录 proposal/effective curriculum。

## 19. 当前阶段不做

当前阶段不做:

- 不让 LLM per-step 控制 env。
- 不让 LLM 直接改 runner。
- 不让 LLM 第一版申请新 capability。
- 不让 LLM 改 eval。
- 不让 LLM 改 seed、budget、success metric。
- 不把未接线字段暴露给 recipe prompt。
- 不把 private logs、public_accept_probes 或 hidden audit 作为 prompt 输入。
- 不让 recovery-window score 自动进入 accept gate。
- 不让 LLM reviewer 成为最终授权者。

## 20. 最终判断

合理第一版不是扩大几个 schedule key,也不是让 LLM 直接写 runner hook。

合理第一版是:

```text
人类/runner 固定少量安全课程能力;
LLM 基于公开证据设计课程结构;
runner 编译、拒绝退化课程并执行;
trusted eval 裁决;
accepted curriculum 做不可变快照。
```

这样自由度足够高: LLM 能自己判断什么时候用 push、什么时候用 DR、什么时候强化 lateral recovery、什么时候只做 final consolidation。与此同时,它不会把系统变成无限 capability 申请器,也不会破坏 Phase 8E 的 sandbox、trusted eval 和 public-only 结果投影边界。

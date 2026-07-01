# LLM Curriculum V1 Provider Audit Dev Log

日期: 2026-07-01

## 触发原因

启动第一次 15 轮服务器试跑后，第 1 轮 provider 已生成候选并进入训练前流程。检查候选 artifacts 发现:

- `operation_diff.json.curriculum_design.description` 是单阶段 constant-weight curriculum。
- `diagnostics.json.curriculum_hypothesis.improvement_attribution_intent` 明确把改进归因到 reward terms，而不是 curriculum change。

这不是越权或作弊，但会把“课程设计”退化成“reward 设计 + 无课程声明”。因此停止该试跑，避免继续消耗训练资源。

## 改动范围

- 新增 V1 curriculum artifact validator:
  - 要求 `operation_diff.json.curriculum_design` 存在。
  - 要求 `diagnostics.json.curriculum_hypothesis` 存在。
  - 要求至少两个 `stage_list` 阶段。
  - 要求阶段使用 `reward_weight_scale` 之外的固定 V1 curriculum capability。
  - 要求 final consolidation 至少 20% 且返回 runner-owned target distribution。
  - 拒绝自称 improvement 不归因于 curriculum change 的 reward-only attribution。
- 在 provider staging 后、训练前执行该审核，失败时作为 repairable provider error，不进入训练。
- 升级 Phase 8E dry-run provider 和 provider ingest selftest fixture，让本地 replay/dry-run 也符合 V1 curriculum contract。
- 将非退化要求写回 prompt-facing `curriculum_design_contract` 和 `selection_guidance`。

## 验证结果

- RED: `scripts\phase8e_provider_ingest_selftest.py` 先失败于 `missing curriculum_design was accepted`。
- GREEN:
  - `scripts\phase8e_provider_ingest_selftest.py` 通过。
  - `scripts\phase8e_selftest.py` 通过。

## 后续步骤

1. 推送新提交到服务器分支。
2. 重新启动 15 轮试跑。
3. 首轮重点确认:
   - prompt 中非退化课程约束存在。
   - provider 如果继续给单阶段/no-op 课程，应在 provider repair loop 被拒并反馈。
   - 若候选通过 provider 审核，`curriculum_design` 应至少是两阶段并使用固定 registry 中的课程能力。

## 二次试跑反馈

重新试跑后，provider 审核成功阻断了退化候选，没有进入训练。但 repair loop 仍反复失败，主要原因是 prompt 没把 artifact shape 写得足够具体:

- `diagnostics.json.curriculum_hypothesis` 必须是 object。
- `operation_diff.json.curriculum_design.stage_list[*].capabilities` 必须列出固定 registry 的 exact id。

补充修复:

- 在 `curriculum_design_contract.artifact_shape` 中明确两个 artifact 的对象类型和 stage_list 位置。
- 在 `curriculum_design_contract.stage_schema` 中要求 `capabilities` 字段，并列出可用 exact id。
- 在 selection guidance 中要求每个 stage 的 `capabilities` 使用 fixed registry 的 exact id。

## 三次试跑反馈

再次试跑后，模型已经开始输出多阶段课程，并在 `stage_list[*].capabilities` 中使用 exact id，如 `external_push_curriculum` / `train_only_dr_scale`。新的阻塞点变成 `final_consolidation` 字段别名:

- 模型使用了 `min_fraction`，而 validator 只接受 `fraction` / `min_fraction_of_training`。
- 模型使用了 `target_push_included` / `target_robustness_included`，而 validator 只接受 `includes_target_push` / `includes_target_robustness_variation`。

补充修复:

- 在 `curriculum_design_contract.final_consolidation_schema` 中写明 exact required fields。
- 在同一 schema 中记录低风险 accepted aliases。
- validator 接受这些同义字段，但仍要求 fraction >= 0.2、distribution 为 `runner_owned_target_distribution`，并且 target push / robustness variation 为 true。

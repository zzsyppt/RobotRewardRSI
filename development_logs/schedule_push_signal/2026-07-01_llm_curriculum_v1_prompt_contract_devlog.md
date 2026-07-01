# LLM Curriculum V1 Prompt Contract Dev Log

日期: 2026-07-01

## 改动范围

- 新增 `src/humanoid_agent/cold_start_lab/curriculum_contract.py`，提供 Phase 8E 第一版课程设计的 prompt-facing contract。
- 在 `build_phase8e_prompt_package()` 中加入两个顶层 payload:
  - `public_curriculum_evidence_bundle`
  - `curriculum_design_contract`
- 在 `scripts/phase8e_selftest.py` 中加入回归约束，确认固定能力 registry、课程 contract、accept probe 隔离和 final consolidation 下限。
- 保持 candidate package 文件面不变，课程假设仍写入 `diagnostics.json`，课程设计仍写入 `operation_diff.json`。

## 设计边界

- 第一版不允许 LLM 申请新能力；只允许它围绕 runner 预批准的固定 registry 设计课程。
- registry 暴露的是课程层能力，不等价于 candidate 可以直接调用 runner hook。
- `reward_weight_scale` 仍是当前 prompt-facing schedule 中唯一已经生效的 candidate recipe 控制键。
- push 课程、train-only robustness variation、reward-only push signal、recovery diagnostic、public push probe 都是 runner-owned 能力；是否真正执行由后续 runner compiler/adapter 接入决定。
- prompt 中不复制完整 `task`，只用 `task_ref` / `objective_ref` 指向顶层任务，避免重复 objective 并保持 prompt 顺序约束。
- prompt 仍禁止暴露未接通的具体字段或无效 key，包括 `push_active`、`push_vec_public`、未生效 push schedule key 和未生效 robustness schedule key。

## 验证结果

- RED: 使用 bundled Python 运行 `scripts\phase8e_selftest.py`，失败于 `prompt must include public_curriculum_evidence_bundle`。
- GREEN: 使用 bundled Python 运行 `scripts\phase8e_selftest.py`，通过。
- 通过的 selftest 覆盖了 inactive prompt token 扫描，说明新课程层没有重新广告 placeholder push 字段或无效 schedule key。

## 后续步骤

1. 增加 runner-side curriculum compiler，读取 `operation_diff.json.curriculum_design` 并做静态审核、clamp、退化拒绝。
2. 将预批准的 push/robustness 课程键接入训练执行，而不是只停留在 accepted/precomputed。
3. 把 reward-only push signal 的真实 instrumentation 接入 RewardInputs 后，再允许 prompt-facing reward 使用。
4. 增加 public dev probe / withheld accept probe 的训练后报告，用于下一轮 LLM 课程设计反馈。

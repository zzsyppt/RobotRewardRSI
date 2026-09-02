# 2026-07-02 Phase 9 CurriculumPlan 与 CapabilityRegistry

## 本阶段目标

按计划完成 Phase 9：保留课程系统的可扩展接口，但第一版不开放 LLM curriculum tuning，不把旧项目 push-specific curriculum compiler 搬入新架构。

## 实现内容

- 新增 `CurriculumCapability` 与 `CapabilityRegistry`，默认 registry 中 future push/DR capability 均为 `supported=true`、`effective_now=false`、`prompt_visible=false`。
- 新增 `CurriculumCompiler`，只做 proposal capability id 校验、lineage hash、mode explanation 和 `disabled_by_phase` 记录，不改写训练 schedule。
- `CurriculumCompiler` 支持读取顶层 `capabilities`、`curriculum_design.stage_list`、`operation_diff.curriculum_design.stage_list` 三种 proposal 形态。
- 扩展 `TaskSpec`，增加固定 train/eval protocol 和默认 `task_default` curriculum。
- 扩展 `BackendSpec`，增加 future curriculum capability 状态查询；Isaac Lab dry-run backend 显式声明 future capability 但不启用。
- 新增 stair schema-only fixture，能表达 fixed baseline train distribution、fixed eval bins、height-map/terrain metadata，但不启用 runtime plugin，不开放 stair height progression。
- 补强 `CandidateSurfacePolicy`，把 `reward_weight_scale`、`height_range`、`schedule`、`single_stage_numeric_schedule` 归为 parameter-only。

## Review 结果

- 确认 compiler 输出的 `effective_training_config` 始终等于 task default，不会把 LLM stage schedule 接进 TrainJobSpec。
- 确认 registry 中未接线 capability 不会出现在 prompt effective capability 列表。
- 确认 schedule-only、height-range-only、reward_weight_scale-only 不可能被分类成 code-level improvement。

## 验证

- `python -m unittest tests.unit.test_curriculum_phase9 -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 9 已完成。课程系统现在有可追踪、可扩展的 schema/registry/lineage 接口，但第一版仍保持 runner-owned/task-default 训练配置，不允许 LLM 调课程或凭数值 schedule 进入 accept 路径。

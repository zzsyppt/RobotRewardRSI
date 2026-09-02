# 2026-07-02 Phase 12 Stair Climbing 扩展性验收

## 本阶段目标

按计划完成 Phase 12：不实现 stair runtime，只验证当前架构未来能以小改动接入 stair climbing。

## 实现内容

- 新增 `docs/task_extension_guides/stair_climbing.md`，说明未来 stair task 需要新增的 TaskPlugin、task config、baseline manifest、failure classifier、metrics 和 BackendAdapter capability。
- 更新 `configs/tasks/stair_extension_schema.json`：
  - 明确 `runtime_plugin_enabled=false`、`training_command_enabled=false`。
  - 增加 future baseline：Isaac/Unitree G1 rough locomotion policy trained on stair_height 0.05-0.20 m。
  - 增加固定 eval bins：0.10、0.15、0.20、0.25、0.30、0.35、0.40 m。
  - 增加 future fixed train protocol、failure taxonomy、height-map/terrain metadata。
- 补强 `CandidateSurfacePolicy`，把 future `stair_height_range` 归为 parameter-only。
- 新增 Phase 12 测试，确认没有创建 stair runtime plugin，没有 stair training command，height-map/terrain projection 属于 runner-owned approved extension。

## Review 结果

- 确认 Phase 12 没有新增 `tasks/stair_climbing` runtime plugin。
- 确认文档明确未来 stair 只应新增任务插件、任务配置、baseline manifest、failure classifier、metrics，必要时扩展 BackendAdapter capability。
- 确认 AgentCore、Runner、DecisionEngine、ledger 不需要因 stair 任务语义修改。
- 确认普通 candidate 不允许任意改 obs/action space、height-map projection 或 terrain metadata。
- 确认只扩大 stair_height range、只调 reward weight 或只改 schedule 不会成为 code-level accepted improvement。

## 验证

- `python -m unittest tests.unit.test_stair_extension_guide -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 12 已完成。Stair climbing 目前只作为 schema/documentation-level extension proof 存在，不启用 runtime、不生成训练命令、不运行实验。

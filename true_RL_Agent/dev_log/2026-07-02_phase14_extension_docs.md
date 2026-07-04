# 2026-07-02 Phase 14 文档、测试与任务扩展规范

## 本阶段目标

按计划完成 Phase 14：让独立项目具备可持续扩展任务和后端的工程规范，并把相关测试矩阵固定下来。

## 实现内容

- 新增 `docs/extension_and_boundary_guide.md`：
  - 文档化新增 TaskPlugin 的路径。
  - 文档化新增 BackendAdapter 的路径。
  - 文档化 baseline checkpoint 接入 checklist。
  - 文档化 random vs warm_start 的选择和兼容性失败路径。
  - 文档化 prompt-facing capability 准入原则。
  - 文档化 trusted eval 与 runner-owned boundary。
  - 明确 future redirection/reach task 的扩展路径。
  - 汇总当前测试矩阵。
- 新增 `tests/unit/test_phase14_extension_docs.py`：
  - 验证 guide 覆盖 Phase 14 要求的路径和边界。
  - 验证 guide 明确阻止 TaskPlugin/BackendAdapter/AgentCore authority leak。
  - 验证测试矩阵引用当前已有测试文件。
  - 验证现有测试覆盖注册、job materialization、policy initialization、public projection、trusted eval、prompt boundary 和 safe boundary。

## Review 结果

- 确认 Phase 14 只新增文档和文档测试，没有改变运行路径。
- 确认新增 redirection/reach 有明确 TaskPlugin/BackendAdapter/PublicContext 扩展路径。
- 确认 baseline checkpoint 接入 checklist 不把 warm_start 临时塞进任务插件。
- 确认 prompt-facing capability 仍比内部 runner contract 更窄，未接线能力必须 disabled/not prompt writable。
- 确认 trusted eval、private seeds、checkpoint selector 继续属于 runner-owned/private 边界。

## 验证

- `python -m unittest tests.unit.test_phase14_extension_docs -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 14 已完成。项目现在有面向新任务、新后端、baseline checkpoint、prompt capability 和 trusted eval 边界的扩展规范，并且测试矩阵被单元测试固定。

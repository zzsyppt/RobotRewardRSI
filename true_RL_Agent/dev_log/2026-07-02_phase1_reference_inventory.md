# Phase 1 参考资产盘点与复制策略

## 目标

按照计划的阶段 1，明确父项目 Phase8E、candidate verifier、safe checkpoint、ledger、public projection、training adapter，以及本地 Codex 源码中哪些内容可以复用思想、哪些需要重写、哪些不能复制到通用核心。

## 本次实现

- 扩展 `docs/reference_inventory.md`，形成 extraction map。
- 标注 candidate schema、verifier、safe checkpoint、ledger、public projection、training adapter 的来源、复用方式和 copy target。
- 单独列出 push-specific 内容，禁止提前塞进 AgentCore 或通用 runner。
- 记录旧路径 random-only 限制，并明确 warm-start 只作为后续训练起点契约。
- 记录 Codex 源码只作为 design-only reference，不能成为服务器运行依赖。
- 新增 `tests/unit/test_reference_inventory.py`，确保 Phase 1 文档覆盖计划要求的盘点项和运行依赖边界。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_reference_inventory -v
```

结果：2 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：5 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

## Review 结论

- Phase 1 只提交盘点文档和文档覆盖测试，没有混入 Phase 2/3/4 的代码实现。
- 文档明确区分 copy target、design-only reference 和 push-specific exclusion。
- 没有增加父项目运行时依赖。
- 没有开放 LLM curriculum tuning 或 parameter-only acceptance。

## 下一步

进入阶段 2：定义核心数据契约。阶段 2 只实现 TaskSpec、BackendSpec、PolicyInitialization、BaselinePolicyRef、CandidatePackageRef、CurriculumPlan、TrainJobSpec、EvalJobSpec、CheckpointRef、PublicEvalResult、DecisionOutcome、TaskInvariantSpec 等稳定对象，不实现 registry 或 fake round。


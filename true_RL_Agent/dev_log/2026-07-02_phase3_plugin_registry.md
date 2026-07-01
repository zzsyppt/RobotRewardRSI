# Phase 3 TaskPlugin 与 BackendAdapter 注册机制

## 目标

按照计划阶段 3，建立任务层和后端层的插件边界。当前阶段只实现接口、registry 和 compatibility check，不实现 fake runner、不 append ledger、不执行训练。

## 本次实现

- 在 `tasks/base.py` 中新增 `TaskPlugin` 抽象接口。
- 在 `backends/base.py` 中新增 `BackendAdapter` 抽象接口和 `BackendJobBundle`。
- 新增 `TaskRegistry`，支持注册、查找、重复注册拒绝和未知 task 报错。
- 新增 `BackendRegistry`，支持注册、查找、重复注册拒绝和未知 backend 报错。
- 新增 `CompatibilityReport` 与 `check_task_backend_compatibility()`，联合检查 task/backend 对 task_id 与 policy initialization 的支持。
- 新增 `tests/unit/test_plugin_registries.py`，覆盖注册机制、compatibility check、边界权限和 backend materialize_jobs 接口不携带 decision。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_plugin_registries -v
```

结果：5 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：17 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

## Review 结论

- `TaskPlugin` 只描述任务语义，不暴露训练命令、run_training 或 decision API。
- `BackendAdapter` 只 materialize job specs，不暴露 accept/reject/ledger 决策。
- registry 只解决插件/后端查找，不启动训练。
- compatibility check 是显式 preflight，不把 unsupported warm_start 暗中降级为 random。
- 本阶段没有引入 Phase 4 fake runner 或 AgentCore round 闭环。

## 下一步

进入阶段 4：FakeTask + FakeBackend + FakeRunner 最小纵向闭环。阶段 4 才允许新增 `tests/fakes/`、runner fake、单轮 ledger 和 decision shape 验证。


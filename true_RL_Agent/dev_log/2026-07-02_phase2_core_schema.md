# Phase 2 核心数据契约

## 目标

按照计划阶段 2，只定义系统内部稳定数据契约，为后续 AgentCore、TaskPlugin、BackendAdapter、Runner 打基础。不实现 registry、不跑 fake round、不接真实 Isaac Lab 后端。

## 本次实现

- 新增 `TaskSpec` 与 `TaskInvariantSpec`，用于表达任务身份、支持的 policy initialization 和 runner-owned invariant。
- 新增 `BackendSpec`，用于表达后端支持的任务与初始化模式。
- 新增 `PolicyInitialization` 与 `BaselinePolicyRef`，支持 `random` 和 `warm_start`。
- 新增 `CheckpointRef`、`TrainJobSpec`、`EvalJobSpec`，明确 warm-start checkpoint 是训练起点，eval checkpoint 是评估对象，二者不混用。
- 新增 `CandidatePackageRef`、`CandidateChangeSet`、`CodeChangeManifest`、`CandidateSurfacePolicy`，第一层区分 code-level、parameter-only、diagnostic-only、engineering-only。
- 新增 `CurriculumPlan`，第一版只允许 `frozen` / `runner_owned` / `task_default`，禁止 LLM-controlled curriculum。
- 新增 `PublicEvalResult` 与 `DecisionOutcome`，固定 public result 和四类 decision outcome 名称。
- 新增 `tests/unit/test_core_schema.py`，覆盖阶段 2 验收点。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_core_schema -v
```

结果：7 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：12 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

## Review 结论

- 本阶段没有实现 TaskRegistry、BackendRegistry、BackendAdapter 接口、Runner 执行或 fake round，未越界到阶段 3/4。
- 没有引入父项目运行时依赖。
- `CurriculumPlan` 显式 fail closed：第一版不能让 LLM 控制 curriculum schedule。
- `CandidateSurfacePolicy` 只做数据契约级分类，不把 parameter-only candidate 放进 policy accept 路径。
- `PublicEvalResult.to_public_dict()` 不投影 trusted/private seed 信息。

## 下一步

进入阶段 3：实现 TaskPlugin 与 BackendAdapter 注册机制。阶段 3 只建立插件/后端边界和 compatibility check，不启动训练，不 append ledger，不实现 fake runner 闭环。


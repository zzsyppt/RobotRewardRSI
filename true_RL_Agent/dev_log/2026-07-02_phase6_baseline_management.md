# Phase 6 PolicyInitialization 与 Baseline 管理

## 目标

按照计划阶段 6，正式补齐 warm-start baseline policy 契约，同时保持第一版 push 路径只要求 random 可执行。当前阶段只做 manifest、compatibility、safe-load 和 dry-run materialization，不接真实 baseline 训练。

## 本次实现

- 新增 `policy_baselines.py`。
- 定义 `CheckpointProvenance`，记录 baseline checkpoint 来源。
- 定义 `BaselineCompatibilitySpec`，记录 backend、robot、obs/action schema、policy architecture 兼容信息。
- 定义 `BaselineManifest`，支持 `baseline_id`、`task_id`、checkpoint ref、provenance 与 compatibility。
- 支持 `BaselineManifest.to_policy_initialization()`，生成 `PolicyInitialization.warm_start(BaselinePolicyRef)`。
- 定义 `BaselineCompatibilityReport` 与 `validate_baseline_compatibility()`。
- 检查 backend compatibility、task warm_start 支持、robot、obs/action schema、policy architecture 和 warm-start checkpoint safe-load。
- 新增 `tests/unit/test_baseline_management.py`，覆盖 Phase 6 验收标准。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_baseline_management -v
```

结果：5 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：36 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

```powershell
rg -n "subprocess\\.(run|Popen|call)|os\\.system|Start-Process|torchrun|python .*train\\.py" true_RL_Agent\src true_RL_Agent\tests
```

结果：无匹配。

## Review 结论

- random initialization 仍不需要 checkpoint。
- warm_start 必须来自 BaselineManifest 和 BaselinePolicyRef。
- warm-start checkpoint 必须 role=`warm_start`、runner sealed 且带 sha256 checksum。
- backend/task/robot/obs/action/policy architecture 不兼容会在训练前报告失败。
- warm-start train dry-run command 显式包含 `--warm-start`，并与 eval checkpoint 分离。
- 本阶段没有接入真实 warm-start 训练 smoke；计划要求这个 smoke 在真正接入第一个 warm-start 任务前补，不阻塞第一版 push random 交付。

## 下一步

进入阶段 7：实现 IsaacLabRslRlBackendAdapter dry-run。阶段 7 应 materialize random train command、warm-start train command spec 和 checkpoint eval command，并声明外部依赖；仍先 dry-run，不直接启动真实训练。


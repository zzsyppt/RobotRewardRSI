# P3 开发日志

## 2026-08-16 至 2026-08-17

### 目标与边界

- 重读双卡总计划，将本阶段限定为固定三 phase eval、Stair selector、receipt-bound assembly 和成本合同。
- 只借鉴 Codex 中适用的显式任务边界、不可变输入、owned process、typed receipt 与可重放状态；没有引入通用 Agent、DAG、Graph 数据库或动态调度框架。
- 未启动 GPU、provider API、服务器同步、shadow pilot 或正式实验。
- 入场 HEAD：`d38385c922d3594b443508abdcd99d33e8290cf8`；保留用户已有 outputs、临时包和 `uv.lock`，未清理或纳入 P3。

### 初始实现

1. 新增 task-neutral `EvaluationPhase`、`EvaluationPhaseResult` 和 `EvalPhaseJobSpec`。
2. runner/backend 新增可选 phase 能力；Isaac backend 从完整 eval plan 派生单 split argv。
3. SubprocessRunner 复用 P1 attempt、GPU lease、owned process、checkpoint 和环境校验。
4. TrialReceiptJournal 新增 create-once phase receipt。
5. Stair 单体 eval 与 phased eval 共用同一 summary builder。
6. 新增 Stair public-dev selector、task adapter 和 23 seed-unit成本合同。

### 审计后的源头修正

1. 删除“只有超过 `1.5×SE` 才选择”的弃权门。现在对 guard-eligible sibling 始终按 objective 排序，paired SE只形成噪声带标签。
2. selector加入同域 current-parent public-dev guard reference；越过冻结 5% guard 硬回退上限的候选不可成为 winner。没有复制完整 acceptance。
3. screen/phase evidence显式保存 `execution_device` 与 P1 `resource_binding_hash`，合同声明它们为跨卡允许差异。
4. `project_assembled_eval_summary` 不再接受任意 mapping；只接受绑定 selection和三份 phase receipt 的 `EvaluationAssembly`，runner会重新读取和验证 receipt。
5. phase-result/spec 字段校验收敛成一个 task-neutral validator，避免 receipt loader 与 runner各维护一份真相。
6. 删除未接线且弱于现有 v1 的 generation final-confirmation模块与测试；后续只能版本化扩展现有 Stair终局合同。
7. 删除按 import 名称子串判断 sealed/private边界的弱测试，改为 typed public-dev输入、phase验证和行为拒绝测试。

### 验收

```text
.venv\Scripts\python.exe -m unittest \
  tests.unit.test_stair_phased_evaluation \
  tests.unit.test_subprocess_runner \
  tests.unit.test_isaac_lab_backend \
  tests.unit.test_stair_entrypoint_materialization \
  tests.unit.test_stair_execution_source_preflight \
  tests.unit.test_runner_p1_reliability \
  tests.unit.test_runner_jobs \
  tests.unit.test_stair_final_confirmation \
  tests.integration.test_trial_executor \
  tests.integration.test_fake_round

Ran 190 tests in 29.014s
OK (skipped=2)
```

```text
.venv\Scripts\python.exe -m unittest \
  tests.unit.test_acceptance_contract \
  tests.unit.test_acceptance_protocol \
  tests.unit.test_acceptance_replay \
  tests.unit.test_stair_acceptance_protocol \
  tests.unit.test_stair_final_confirmation \
  tests.unit.test_stair_final_confirmation_entrypoint \
  tests.unit.test_plugin_registries

Ran 63 tests in 1.931s
OK
```

```text
.venv\Scripts\python.exe -m unittest

Ran 1209 tests in 238.763s
OK (skipped=21)
```

另完成 `compileall`、`git diff --check` 和旧规则/弱副本残留扫描，均通过。

### 后续入口

- 用户验收 P3 后才可开始 P4。
- P4必须复用这里的 phase contract、selector、guard reference、typed assembly 和runner receipt重验，不能在 coordinator中复制 Stair分数、guard或 sealed语义。
- P6前不得声称已验证真实双卡效率、真实 Isaac phase等价或正式 selector效果。

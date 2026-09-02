# Phase 5 Runner Job 抽象与执行安全骨架

## 目标

按照计划阶段 5，让执行层只接受统一 job spec，并补上 runner-owned artifact layout、budget、checkpoint sealing、safe-load、isolation placeholder、TaskInvariantSpec preflight 和 trusted eval seal。当前阶段仍不接真实 Isaac Lab 后端，不启动真实训练。

## 本次实现

- 新增 `runner/artifacts.py`，定义 session/round/train/eval/checkpoints 路径布局。
- 新增 `runner/budget.py`，用 `enforce_fixed_budget()` 拦截预算漂移。
- 扩展 `CheckpointRef`，新增 `sealed` 与 `metadata` 字段。
- 新增 `runner/checkpoint.py`，实现 runner-owned `seal_checkpoint()` 和 `safe_load_checkpoint()`。
- 新增 `runner/jobs.py`，提供 train/eval dry-run command projection，并区分 random 与 warm-start train job。
- 新增 `runner/isolation.py`，预留 dry_run/subprocess/rootfs 隔离策略。
- 新增 `safety/invariants.py`，集中执行 TaskInvariantSpec 检查。
- 新增 `safety/eval_seal.py`，记录 eval config hash、checkpoint hash、task invariant hash，并显式声明不加载 candidate code。
- 扩展 `BackendJobBundle`，可携带 materialized task invariants。
- 扩展 `run_candidate_round()`，在 materialize 后、train 前检查 task invariants，违反时 reject 且不进入 train/eval。
- 更新 FakeRunner，使 fake train output 通过 runner seal 变成 eval checkpoint。

## Review 中发现并修复的问题

初版只提供了 invariant helper，但 fake round 还没有在训练前使用它。已新增集成测试模拟 backend 篡改 task invariant，并修复 round：发现 invariant mismatch 后写入 `task_invariant_violation` reject，runner train call 为 0。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_runner_jobs -v
```

结果：8 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：31 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

## Review 结论

- Runner dry-run command 只是字符串投影，不调用 subprocess 或训练脚本。
- random 与 warm-start train job 在 dry-run projection 中有明确差异。
- checkpoint sealing 是 runner-owned；trusted eval seal 要求 sealed eval checkpoint。
- materialized task invariant 违反会在训练前 reject，不能进入 eval/accept。
- trusted eval seal 记录 config/checkpoint/invariant hash，并声明 `candidate_code_loaded=false`。
- 本阶段没有接入真实 Isaac Lab、RSL-RL 或服务器训练命令。

## 下一步

进入阶段 6：PolicyInitialization 与 Baseline 管理。阶段 6 应补 baseline manifest、baseline_id、checkpoint provenance、backend/robot/obs/action/policy compatibility 与 safe-load dry-run；不能跳到真实 push 后端。


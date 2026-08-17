# P1 开发日志

> 本页前半段记录 P1 初次实现；末尾“Prompt 出厂字节真相与启动前 preflight 纠偏”取代前文关于 Prompt 已完全验收的旧结论。Runner/GPU/attempt 结论不变。

## 2026-08-16

### 工作边界

本阶段依照 `dual_gpu_generation_upgrade_plan_zh.md` 和 P1 详细实施计划，只实现 Runner、Prompt、attempt retry 与 GPU lease 的基础能力。

明确未实施：GenerationCoordinator、A/B author、selector、generation ledger、探索价值图、phase eval、P5 continuation inspector，以及任何真实 GPU/API/服务器实验。

### 实施前核对

- 工作分支：`true-rl-agent-refactor`
- 实施基线：`81565dd970a6726fda6a219704bb9a18651e6bb8`
- 先用仓库 CodeGraph 核对 `SubprocessRunner`、job specs、Prompt adapter、task plugin 和现有 process lock 调用链；
- 再用 Codex 仓库 CodeGraph/源码核对 process-group ownership、Linux parent-death signal、Windows Job Object、terminate/grace/kill/reap 模式；
- 只借鉴上述进程生命周期原则，没有引入 Codex 的 Rust、PTY、远程控制、事件总线或通用 Agent 架构。

### 实现记录

#### 1. Runner 进程生命周期

- 新增 `runner/process_handle.py`：流式日志、bounded tail、Windows/Posix process-tree 清理、graceful terminate、force kill、wait/reap；
- 新增 `runner/process_guard.py`：Linux 正式绝对路径 executable 的极小 parent-death guard，设置死亡信号后立即 exec；
- `SubprocessRunner` 从 `subprocess.run(capture_output=True)` 切换到该原语；
- 保留 argv direct execution 和 `shell=False`，未扩成通用执行框架。

#### 2. 失败分类与 attempt receipts

- 在 `runner/base.py` 增加封闭 `RunnerFailureKind`；
- 只有 launch transient、timeout、summary missing 被标为 retryable；
- 新增 `runner/job_attempts.py`，实现 create-once Reservation/Result、前驱 hash、输入/环境/设备/资源/预算绑定；
- `JobAttemptScope` 作为 private operational 字段接入 train/eval/baseline spec；
- 没有 scope 的旧路径保持一次执行；已有 reservation 不自动重跑，要求后续恢复 inspector 判定。

#### 3. GPU lease 与单设备约束

- 新增 `runner/gpu_lease.py`；
- lease key 使用 `machine_id + physical_id`，而不是 ordinal；
- lease 覆盖 logical job 的全部 attempt；
- attempt scope、lease、job spec 和 summary 的 device identity 必须一致；
- 公共 argv validator 拒绝 distributed/multi-GPU/local-rank/world-size 开关。

#### 4. Prompt minimum core

- 新增 task-neutral `prompting/capacity.py`，负责 canonical hash、bytes、projection id、omission receipt；
- 公共 task-section sanitizer 重新验证移除 receipt 后的 provider-visible projection；
- Stair adapter 在合同内压力输入下保留 task/reward/eval contract、current-best/latest 最小证据和 bootstrap sentinel；
- Push adapter 使用自己的公开字段和最小核心；
- Neutral 正常小对象不增加无意义分支；
- 非法输入、未知 section、tamper 和 minimum-core 自身超限仍拒绝。

### 源头与任务边界复核

- common Runner/Prompt 新模块未引入 Stair 的 tread、torso、height-scan、失败分类或选择语义；
- sealed/public-accept 相关既有 Runner 逻辑未改语义；
- ledger、acceptance、current-best、Genesis、final-confirmation 未修改；
- GPU lease 的活跃真相仍是 OS lock，JSON 仅为说明性 receipt；
- attempt journal 不推断科学价值；
- 没有新增 Graph、第二 ledger、外部数据库或常驻调度服务。

### 验证记录

#### 全量 Windows 回归

```text
.venv\Scripts\python.exe -m unittest discover -s tests
Ran 1179 tests in 188.733s
OK (skipped=21)
```

#### Linux/WSL P1 进程回归

```text
wsl -e bash -lc "cd /mnt/d/humanoid_agent/true_RL_Agent && \
  PYTHONPATH=src python3 -m unittest tests.unit.test_runner_p1_reliability"
Ran 16 tests in 1.970s
OK
```

#### 静态与兼容检查

```text
.venv\Scripts\python.exe -m compileall -q src tests
PASS

Python 3.10 AST grammar: 12/12 production files PASS
git diff --check: PASS
```

覆盖内容包括：timeout 子孙清理、Linux parent-death、cleanup 首错保留、真实 summary-missing 一次 retry、不可重试分类、receipt 篡改/未知字段、GPU physical alias 冲突、跨进程锁释放、distributed argv 拒绝、Stair/Push minimum-core 和公共 receipt tamper。

#### P1 文档冻结哈希

| 文件 | bytes | SHA256 |
|---|---:|---|
| `P1_IMPLEMENTATION_PLAN_ZH.md` | 11,466 | `6cb7edc688263d4b932426c98590609eedd1cc5002b0954f4c7ad102685b96bd` |
| `P1_ACCEPTANCE_REPORT_ZH.md` | 9,536 | `4d56f6626610acda83ec4a98e9729edc962fd5c371a88adc9746f8e6aafdfa94` |

这些哈希冻结本次 handoff 内容；后续若修改对应文件，应重新生成并留下新日志记录。

### 当前限制与后续 handoff

- 本地 fake GPU identity 只证明互斥合同；真实 UUID、两进程 SM/显存/RAM/CPU 争用留给 P6-A；
- P1 不实现 hard-crash 后 reservation 的自动结论；P5 必须通过可信 inspector 闭合或继续；
- P1 不定义 future lane intent 或 B exclusion Prompt schema；该真相属于 P4 generation contract；
- 当前 width=1 没有 opt-in scope，因此不会因 P1 自动获得 retry；未来 generation coordinator 必须显式提供 scope、预算和 lease manager；
- 本阶段未 commit、未 push、未同步服务器、未启动实验。

### 出口

P1 本地实现与严格回归 PASS，等待用户检查后再决定是否进入 P2。

## 2026-08-16：Prompt 出厂字节真相与启动前 preflight 纠偏

### 用户复审发现

复核后确认三项都真实：

1. 初次 P1 没有在 `LiveInvocationJournal.start()` 前运行 task worst-case preflight；
2. Stair 压缩函数在 `feedback_entries` 为空时提前返回，因此非-feedback 段超限不会进入 minimum-core；
3. adapter 的 task sections 与最终 provider 顶层七段不是同一对象：shared evidence guidance 被追加进 task-owned `selection_guidance`，历史 R01 的事后七段聚合因此可超过 adapter 的 32 KiB 口径；projection receipt 的 bytes 又明确不含 receipt 自身，不能用作出厂总量。

### 源头收敛

- 删除 Stair 的空 `feedback_entries` 提前返回；是否降级现在只由完整 task sections 的 canonical bytes 决定；
- shared evidence rules 移到 `loop_contract.evidence_discipline`，`selection_guidance` 只保留 task adapter 输出；
- 新增固定 provider task-section key 投影。candidate-draft 下该对象与 adapter canonical sections 逐字段相等；
- `authoring_context_frame.v2` 的 budget metadata 记录实际发出对象的唯一 `provider_task_sections` receipt；
- projection receipt 升为 v2，并把字段改名为 `projected_content_without_receipt_hash/bytes`，不再容易被误读为出厂总量；
- Stair、Push、Neutral 分别提供 task-owned capacity preflight shape；公共 `preflight_task_prompt_capacity()` 只调用、构建真实 payload、重算 receipt，不理解任何任务字段；
- live loop 复用同一个已 preflight 的 adapter，并在创建 invocation journal 前执行。preflight 失败时 `live_invocations/` 不存在、ledger 为空。

当前 Stair 最坏 minimum-core 的实际 provider-facing receipt：

```text
serialized_bytes = 26,182
max_bytes        = 32,768
headroom_bytes   = 6,586
content_hash     = 8023acbf100a3c007a65319a01e1fdd083fb80fe711f7a2ef230a83f9689c24b
```

这取代旧的 adapter 预装配估算和事后手工七段求和。历史 payload 不被重写；新 `authoring_context_frame.v2` 才具有该 receipt。

### 最小必要测试

- `feedback_entries=[]`、超限来自 runtime 段：进入 Stair minimum-core；
- candidate-draft 最终 provider task sections 与 adapter canonical object 完全相等；
- 最终 receipt 重算一致，shared guidance 仍可见但不再算入 task section；
- Stair 最坏形状 preflight 两次结果一致且保留至少 512 字节 headroom；
- preflight 强制失败时 invocation journal 未创建、ledger 无记录；
- cold-start、Push、Neutral 和历史 authoring 规则仍可见。

### 最终验证

```text
.venv\Scripts\python.exe -m unittest discover -s tests
Ran 1183 tests in 180.986s
OK (skipped=21)

PYTHONPATH=src python3 -m unittest tests.unit.test_prompt_capacity_projection
Ran 9 tests in 0.551s
OK

Python 3.10 AST grammar: 18/18 production files PASS
compileall: PASS
git diff --check: PASS
```

没有运行 GPU/API/服务器实验，没有修改 ledger、acceptance、eval 或任务成功语义，也没有提前实现 P4 的 lane intent/B exclusion。P4 增加这些 task-owned 字段时，必须同时进入 Stair minimum-core 与 preflight shape；公共层不增加字段解释。

### 纠偏后出口

Prompt 三项缺口已闭合，P1 再次达到本地 PASS；仍等待用户检查，不进入 P2。

纠偏后文档冻结：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `P1_IMPLEMENTATION_PLAN_ZH.md` | 13,065 | `bc683572a9d9c72af659890e12f9d0e00c758abcf7857118ca12224fe9f0eef6` |
| `P1_ACCEPTANCE_REPORT_ZH.md` | 11,551 | `5f3771b3f2f21488dad7b3d1ca163e81cafb239db524f88a3762375d56c4f306` |

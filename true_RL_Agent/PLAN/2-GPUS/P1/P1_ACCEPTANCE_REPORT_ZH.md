# P1 验收报告：Runner、Prompt 可靠性与双卡资源治理

## 1. 验收结论

**P1 本地实现与回归验收 PASS。**

本阶段建立了后续双候选系统需要的四项底座：

1. Runner 现在对长时间子进程、孙进程、日志、超时和回收负有明确责任；
2. logical job 可选择启用 create-once attempt 链，并且只对封闭分类中的基础设施故障执行至多一次、输入完全相同的重试；
3. 两条未来 lane 可通过可信物理 GPU identity 获得进程持有的互斥租约；
4. Stair 与 Push 对合同内合法 Prompt 输入都有 task-owned 的确定性 minimum-core 降级，公共层只负责容量收据、字节合同和安全校验。

本次没有启动 GPU/API/服务器实验，没有修改 ledger、acceptance、current-best、eval 语义或旧 width=1 正式性合同，也没有实现 P2 之后的 GenerationCoordinator、selector、generation ledger、phase eval 或探索前沿。

实施基线：

```text
branch = true-rl-agent-refactor
HEAD   = 81565dd970a6726fda6a219704bb9a18651e6bb8
```

## 2. 实际落地结构

### 2.1 小型、专用的进程执行句柄

新增 `runner/process_handle.py` 和 Linux 专用的极小 `runner/process_guard.py`：

- argv 仍直接执行，`shell=False`，没有引入 PTY、远程执行或通用 workflow；
- stdout/stderr 流式写入 attempt 独立文件，错误只读取有界尾部，不再把长训练日志全部留在内存；
- Windows 使用新进程组，优先绑定 kill-on-close Job Object，并对明确的宿主嵌套 Job Access Denied 做确定性进程树清理降级；
- POSIX 使用独立 session/process group，按 TERM、grace、KILL、wait/reap 的顺序收口；
- Linux 正式绝对路径 executable 经过一个只负责 parent-death signal 后立即 `exec` 的小 guard，防止 coordinator 硬退出后训练继续占卡；
- cleanup 失败不会覆盖原始异常，仍兼容 Python 3.10。

这里只借鉴 Codex 中成熟的三条原则：执行句柄拥有进程组、父进程死亡必须收口、终止后仍要 reap。没有复制 Codex 的 Rust、PTY、远端会话、事件总线或通用 Agent 框架。

### 2.2 封闭失败分类和一次同参重试

`RunnerExecutionError` 现在携带封闭的 `RunnerFailureKind`。第一版仅允许以下类别重试：

- 预注册的暂时性 launch failure；
- timeout；
- 子进程结束但约定 summary 未生成。

合同/binding/hash/device/summary 格式错误、candidate failure、scientific failure、operator interrupt、executable 不存在和未分类异常均不可重试。

新增 `runner/job_attempts.py`：

```text
Reservation(0) -> Result(0, retryable infra failure)
               -> Reservation(1, previous_result_hash)
               -> Result(1, terminal outcome)
```

每个收据都绑定 session/Genesis、generation contract、generation/lane、invocation segment、logical job、job spec/input/environment/device/resource hash、attempt index 和预算。跳号、第二次 retry、缺前驱、跨 scope 复用、输入漂移、预算少记或篡改均 fail closed。

`JobAttemptScope` 是 job spec 的 private operational 字段，不进入 public job dict。没有 scope 的旧 job 仍只执行一次，因此当前 width=1 和 Push 默认路径没有被静默改成自动重试。

P1 不猜测 hard crash 后未闭合 reservation 的结果，也不自动重新执行已有 reservation；这些恢复判定明确留给 P5 的 continuation inspector。

### 2.3 双卡资源治理底座

新增 `runner/gpu_lease.py`：

- 由可信调用方提供 `execution_device + machine_id + physical_id`；
- 锁键取物理 identity hash，而不是 `cuda:0/1` 字符串，防止两个 ordinal 指向同一物理卡时重复占用；
- 使用现有 OS-held `ProcessFileLock`，metadata 文件不是活跃租约真相；
- lease 覆盖一个 logical job 的全部 attempt；
- attempt scope、lease identity、job device 和现有 summary device 校验必须一致；
- 不同物理 identity 可并行，同一物理 identity 跨进程互斥。

同时在公共 argv 边界统一拒绝 distributed/multi-GPU/local-rank/world-size 开关。每个 job 仍只能使用一个 canonical `cpu` 或 `cuda:n`，P1 没有把 DDP 引进候选训练。

真实服务器 GPU UUID、两卡并行 SM/RAM/CPU 争用仍需 P6-A 真机验证；P1 的 fake identity 测试不能替代该证据。

### 2.4 Prompt 可靠性

新增 task-neutral `prompting/capacity.py`：

- 记录原对象和 minimum-core 在不含自身 receipt 时的 canonical hash/字节数，字段名显式写明 `without_receipt`；
- 记录固定 projection id 和有序 omission reason；
- 由公共 sanitizer 移除内部 receipt 后重新计算可见投影 hash/字节数；
- unknown field、receipt tamper、内容 tamper、cap+1 和 public-safety 违规仍拒绝。

保留内容由任务 adapter 自己决定：

- Stair 保留固定 task/reward-input/eval contract，以及 current-best、latest 的最小公开证据；bootstrap 使用明确的 `no_closed_outcome`；
- Push 只保留 Push 自己已有的公开摘要和最小核心，没有引入 Stair 字段；
- Neutral 现有静态小对象已经是最小核心，不增加多余分支。

Stair 的压缩入口不再依赖 `feedback_entries` 非空；即使超限来自 runtime 或未来其他非-feedback 段，也会进入同一 minimum-core 构造。公共 evidence rules 已从 task-owned `selection_guidance` 移回 `loop_contract.evidence_discipline`，因此最终 provider payload 中固定七个 task sections 与 adapter 的 canonical 对象逐字段一致。

最终 `authoring_context_frame.v2` 从实际 provider payload 提取这一唯一 task 对象，记录 `provider_task_sections={serializer, content_hash, serialized_bytes, max_serialized_bytes, headroom_bytes}`。当前 Stair task-owned 最坏形状启动前实测为：

```text
serialized_bytes = 26,182
max_bytes        = 32,768
headroom_bytes   = 6,586
```

每个 in-tree task adapter 都提供 task-owned preflight shape；live loop 在 `LiveInvocationJournal.start()` 之前构建两次、核对确定性并完成一次真实 provider payload assembly。若固定合同增长导致 minimum-core 放不下，系统在 invocation journal 和科学 ledger 之前拒绝启动。非法或 minimum-core 自身超限仍 fail closed，公共层没有理解 Stair 的目标、失败类别或反馈优先级。

## 3. 单一真相与兼容边界

- process outcome 的唯一分类来自 Runner；attempt journal 只封存分类和绑定，不另行推断科学结论；
- GPU 锁的活跃真相来自 OS-held lock，不来自遗留 JSON；
- task adapter 是 minimum-core 内容真相；最终 provider task receipt 只测量实际出厂对象，不再重组任务语义；
- current round-v7、ledger replay、acceptance、sealed、Session Genesis、final-confirmation 均未修改；
- Stair、Push、Neutral 的任务语义仍留在各自插件；
- 当前正式 width=1 job 未携带 `JobAttemptScope`，行为仍为一次执行；
- P1 没有新建 Graph、第二份 ledger、通用调度器或外部状态服务。

## 4. 验收证据

### 4.1 全量回归

```text
.venv\Scripts\python.exe -m unittest discover -s tests
Ran 1183 tests in 180.986s
OK (skipped=21)
```

21 个 skip 为平台/可选运行时条件；测试输出中的 SimulationApp close、summary publication、backend/source drift 文本来自既有故障注入，最终退出码为 0。

### 4.2 Linux/WSL 进程边界

```text
wsl -e bash -lc "cd /mnt/d/humanoid_agent/true_RL_Agent && \
  PYTHONPATH=src python3 -m unittest tests.unit.test_runner_p1_reliability"

Ran 16 tests in 1.970s
OK
```

该集合在 Linux 实际覆盖 parent-death guard；Windows 全量测试中对应 Linux-only 用例按条件 skip。

Prompt preflight 的 Linux/WSL 定向集合：

```text
PYTHONPATH=src python3 -m unittest tests.unit.test_prompt_capacity_projection
Ran 9 tests in 0.551s
OK
```

### 4.3 语法、格式和 Python 3.10 合同

```text
.venv\Scripts\python.exe -m compileall -q src tests
PASS

ast.parse(..., feature_version=(3, 10))
18 个本次生产改动文件全部通过

git diff --check
PASS
```

### 4.4 关键行为覆盖

- 正常、非零、launch failure、timeout、cleanup failure；
- timeout 后子进程和孙进程清理；Linux 父进程硬退出后目标进程退出；
- stdout/stderr 流式文件与有界尾部；
- summary missing 真实子进程只启动两次，产生两份闭合 result；
- contract/candidate/nonretryable failure 不重试；
- receipt 篡改、未知字段、跨 scope/segment、预算和 hash 漂移拒绝；
- 同一物理 GPU identity 互斥、不同 identity 并行、跨进程锁与异常释放；
- distributed/multi-GPU argv 拒绝；
- Stair bootstrap、current-best/latest 压力降级、空 feedback/non-feedback 压力、Push 独立最小核心；
- preflight failure 发生在 invocation journal 之前，ledger 保持空白；
- 最终 provider task 对象逐字段等于 adapter canonical 对象，出厂 receipt 可重算；
- projection/final receipt tamper、未知 section 和非法输入继续拒绝。

## 5. 主要生产文件 SHA256

| 文件 | SHA256 |
|---|---|
| `runner/process_handle.py` | `8ac37f9be540054f2ffa779e1024d392c378deb73ea86b1b0884214b25257359` |
| `runner/process_guard.py` | `df4755a6c8c397af6b4665a5201123f9afd0da16a59f6e0967811bbb18cfd653` |
| `runner/job_attempts.py` | `7c12341a4c65a7862ad89a5631e83b66c1ad2ef0c8809b67a0b2b13829cb4593` |
| `runner/gpu_lease.py` | `9ab522486852972a4c9790409cebbdcad4896ba4db9a82e91d5f4f0ac5f779ac` |
| `runner/subprocess_runner.py` | `9d13126fea4a4ad587c997990f74d8ce0a3c0ac9d96a23b330db756ad17badbe` |
| `prompting/capacity.py` | `8a40cf0601de3c9209e91fa04b1a6503acadac344041c0cb9cb0bd85d720c009` |
| `prompting/task_sections.py` | `3f2dd876821e175cae8ccadded257e60d32fc753ea2374b24c04618d92624aa9` |
| `prompting/payload.py` | `4de9247f76b647e597cbe844f28d9dd3cd123e4e7da489908109d4891cd92339` |
| `prompting/authoring_frame.py` | `b4a27f8d6d3117b19226dc8117a2c819289e1f861e43781200a33fd78196d579` |
| `memory/context_builder.py` | `83a6c05ecc974dbff9f96c669e95541967d37fc389d12b35d8ee6d9581f9586c` |
| `agent_core/live_loop.py` | `97d5e49e3be1476fbb0ccae0e1579302018f3e224904a2654a2a925667bb434a` |
| `tasks/stair_climbing/prompt_adapter.py` | `b92d95afa67f33ba5d0aba0de4d561f5c094eda12d5272e9a25e542e35281e36` |
| `tasks/push_recovery/prompt_adapter.py` | `971d9118e858f1d49c7cf5fa4a09b6c5a51168fa529b1aa7147fb689210b5add` |

这些哈希对应本报告验收时尚未提交的当前工作树；若代码继续修改，必须重新计算，不能把本表当作未来 commit 的哈希。

## 6. 明确留给后续 phase 的工作

- P2：把现有 round 执行拆成无 ledger 副作用的 trial/phase executor，并复用 P1 process/attempt 原语；
- P3：分阶段 eval receipt/assembler；
- P4：GenerationCoordinator、A-first/B-second authoring、两 lane resource plan 和 generation ledger；
- P5：跨 lifecycle segment 的 continuation chain、hard-crash/未闭合 reservation inspector；
- P6-A：真实双卡 UUID 映射、显存/RAM/CPU/PCIe 争用和墙钟试验。

P1 的接口为这些工作提供底座，但本报告不把接口存在夸大为后续能力已经实现。

## 7. 出口状态

P1 在本地代码与非 GPU 回归层面验收完成，等待用户检查。未经用户明确批准：

- 不进入 P2；
- 不 commit/push；
- 不同步服务器；
- 不启动 GPU/API/正式实验。

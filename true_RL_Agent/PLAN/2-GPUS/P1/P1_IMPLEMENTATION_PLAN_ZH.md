# P1 详细实施计划：Runner、Prompt 可靠性与双卡资源治理

## 1. 目标与本阶段边界

P1 只建设后续双候选系统需要的四类底座：

1. Runner 对长时间子进程及其后代拥有明确的启动、等待、终止和回收责任；
2. 基础设施故障具有封闭分类，并可在预留预算内执行至多一次、输入完全相同的重试；
3. 两个进程不能同时占用同一块物理 GPU，租约与 job 的设备证据可相互核对；
4. 对合同内合法 Prompt 输入，任务 adapter 必须能确定性降级到自己的最小核心，不再在运行中因字节上限意外终止。

本阶段明确不做：

- 不实现 `GenerationCoordinator`、A/B author、selector、generation ledger 或探索前沿；
- 不拆 train/eval phase，不改变现有 public-dev/public-accept/sealed 语义；
- 不修改 round-v7、acceptance、Session Genesis 或旧 width=1 正式性合同；
- 不把 Stair 的目标、失败类型、反馈优先级等任务语义放入 Runner 或 Prompt 公共层；
- 不引入 Ray、Temporal、psutil、外部锁服务或通用工作流框架；
- 不在 P1 自动启动 GPU、API 或服务器实验。

## 2. 当前基线与已确认缺口

实施基线为本地 `81565dd970a6726fda6a219704bb9a18651e6bb8`。P0 已完成机制重复比较的源头修复；P1 不重复建设该能力。

当前真实缺口：

- `SubprocessRunner` 仍使用阻塞 `subprocess.run(capture_output=True)`；超时只杀直接子进程，且长输出常驻内存；
- `RunnerExecutionError` 没有封闭的失败类别，调用方不能安全区分可重试基础设施故障与不可重试的候选、科学或绑定失败；
- train/eval logical job 没有 create-once attempt reservation/result，也没有成本不遗漏的唯一重试链；
- `execution_device` 已绑定 argv 和 summary，但没有跨进程 GPU 互斥租约；
- 当前 argv 没有显式分布式训练入口，但也没有公共校验保证 `distributed/multi_gpu` 开关始终缺席；
- Stair Prompt 有多级压缩，但在 current-best/latest 仍放不下时最终抛异常；Push 已有内部有界投影，Neutral 为静态小对象；
- P5 所需的跨 lifecycle segment continuation、hard-crash inspector 和 `abandoned_unclean` 闭合尚未实现，P1 只提供它们以后可验证的 attempt 基础。

## 3. 采用的最小架构

### 3.1 进程所有权

新增一个 task-neutral 的小型进程执行原语，保持 argv 直接执行、`shell=False`：

- stdout/stderr 直接写入每个 attempt 的独立日志文件，只在错误中读取有界尾部；
- POSIX 使用独立 session/process group，结束时按 group 温和终止、等待、必要时强杀；Linux 对正式绝对路径 executable 再经过一个极小的 runner-owned parent-death guard，设置父进程死亡信号后立即 `exec` 原命令，避免 Runner 硬退出后训练仍占卡；
- Windows 使用新 process group，并优先放入 kill-on-close Job Object；若宿主已有不可嵌套 Job Object 且明确返回 Access Denied，则保留新进程组并使用系统进程树清理作为确定性降级；其他 Job Object 建立错误仍使启动失败并清理已启动进程；
- timeout、`KeyboardInterrupt`、`SIGTERM` 传播路径均先清理再保留原始异常；
- 正常主进程退出后也清理仍存活的后代并执行最终 `wait/reap`。

这里仅借鉴 Codex 的“进程组属于执行句柄、父进程死亡需收口、终止后仍要回收输出和子进程”三条原则，不搬运 Codex 的 Rust/PTY/远端执行/通用 Agent 框架。

### 3.2 封闭失败分类与一次重试

扩展 `RunnerExecutionError`，由 Runner 唯一给出封闭 `failure_kind`。第一版只把以下类别视为可重试基础设施故障：

- 预注册的暂时性 launch `OSError`；
- runner timeout；
- 进程结束但约定 summary 未生成。

以下情况永不重试：

- executable 不存在、权限/argv/路径错误；
- candidate-owned runtime failure；
- structured summary 已存在但格式、身份、hash、binding、contract 或 device 不匹配；
- 正常得到的科学失败/拒绝；
- operator interrupt；
- 未分类异常。

有 retry 能力的 job 必须显式携带 `JobAttemptScope`。旧 job 没有 scope 时仍运行一次，因此旧 width=1/Push 的默认行为和 ledger 字节不变。

每个 scope 最多产生：

```text
AttemptReservation(0) -> AttemptResult(0, retryable infra failure)
                       -> AttemptReservation(1, previous_result_hash)
                       -> AttemptResult(1, terminal outcome)
```

Reservation/Result 均 create-once，并绑定同一 session Genesis、generation contract、generation/lane、invocation segment、logical job、job input/spec/environment/device/resource hash、attempt index 和预算。第二次重试、跳号、输入漂移、跨 scope 复用或成本少记均拒绝。

P1 不尝试替 hard crash 补写 Result；reservation 无 result 保持未闭合，留给 P5 的 inspector。

### 3.3 GPU 资源租约

新增 task-neutral `GpuLeaseManager`：

- 调用方提供可信的 `GpuDeviceIdentity(execution_device, machine_id, physical_id)`；公共层不自行解释 Stair 或训练任务；
- 锁文件键使用物理身份 hash，而不是仅用 `cuda:0/1` 字符串，避免别名重复占用；
- 使用现有 OS-held `ProcessFileLock`，进程退出后锁自动释放，遗留 metadata 文件不代表活跃租约；
- lease receipt 绑定 holder/session/logical job/PID/device identity；
- `SubprocessRunner` 在整个 logical job（含唯一 retry）期间持有同一个 lease；
- 既有 summary 的 `execution_device` 校验继续保留，并要求 attempt scope 的 resource binding 与 lease 一致。

当前 width=1 composition 不自动启用新的双卡 manager；P4 的 generation resource plan 必须显式提供它。P1 用 fake identity 验证互斥与释放，不伪称已完成真机 UUID 验证。

### 3.4 单设备执行约束

在现有 task-neutral argv validator 中统一拒绝分布式/multi-GPU 启动开关。设备分配仍只能来自 `TrainJobSpec/EvalJobSpec.execution_device` 的一个 canonical `cpu` 或 `cuda:n`。

该校验只约束执行方式，不包含任何任务语义，也不修改 Isaac backend 的软件环境 manifest。

### 3.5 Prompt 最小核心

公共层只新增可复用的容量投影收据原语：

- canonical 原对象 hash/字节数；
- minimum-core 在不含自身收据时的内容 hash/字节数，字段名显式写明 `without_receipt`，不得冒充最终出厂总量；
- 固定 projection id 与有序 omission reason；
- 最终仍由现有 public-safety、section key 和 byte-cap validator 校验。

最终 provider payload 再从实际发出的固定 task section key 集合提取唯一 canonical 对象，并在 `authoring_context_frame.budget_metadata.provider_task_sections` 记录其 serializer、hash、bytes 和 headroom。task-owned `selection_guidance` 不再混入 shared evidence rules；后者归回公共 `loop_contract.evidence_discipline`。因此 adapter 对象和最终 provider-facing task 对象可逐字段相等，32 KiB 不再有“预装配”与“出厂后”两个口径。

任务 adapter 对“保留什么”负责：

- Stair：保留固定 task/reward-input/eval contract，保留 current-best 与 latest 的有界最小结果；bootstrap 明确写 `no_closed_outcome`；runtime 细节在容量压力下变为 task-owned omission summary；
- Push：保留现有 task-owned bounded summary；增加最终 minimum-core 兜底但不改变 Push 指标语义；
- Neutral：静态最小核心即现有输出。

最小核心只处理合同内已受长度/数量约束的输入。非法、未受界、tampered、未知 section 或 minimum-core 自身超过上限时仍 fail closed。不会靠静默截断字符串来“保证成功”。

P4 尚未定义 lane intent/B exclusion 的生产 schema，因此 P1 只为其预留公共容量投影接口，不提前发明字段或把 placeholder 写进现有 Prompt。

每个 in-tree task adapter 另外提供 task-owned 的最坏 minimum-core preflight shape。live loop 在 `LiveInvocationJournal.start()` 之前构建两次并核对确定性，再走一次真实 provider payload assembly 和最终 task receipt 校验；固定合同增长导致 minimum-core 无法容纳时，启动前停止。旧的外部 adapter 若尚未实现该 hook，兼容回退只检查其 bootstrap shape，不改变旧插件调用合同。

## 4. 预计文件改面

### 新增

- `src/true_rl_agent/runner/process_handle.py`
- `src/true_rl_agent/runner/job_attempts.py`
- `src/true_rl_agent/runner/gpu_lease.py`
- `src/true_rl_agent/prompting/capacity.py`
- 对应的独立 unit/integration tests

### 修改

- `runner/base.py`：封闭 Runner failure kind；
- `runner/specs.py`：可选、private operational `JobAttemptScope`；
- `runner/subprocess_runner.py`：复用进程句柄、可选 attempt/lease wrapper；
- `runner/job_argv.py`：单设备启动约束；
- `tasks/stair_climbing/prompt_adapter.py`：task-owned minimum core；
- `tasks/push_recovery/prompt_adapter.py`：最终有界兜底；
- `tasks/prompt_adapter.py` / `prompting/task_sections.py`：preflight hook、唯一 provider-facing task receipt 与 task-neutral 容量合同；
- `memory/context_builder.py` / `agent_core/live_loop.py`：真实 payload preflight，并保证发生在 invocation journal 之前；
- `prompting/payload.py` / `prompting/authoring_frame.py`：shared guidance 归属和最终出厂 task receipt；
- 必要的既有测试断言，不改 ledger/decision/task protocol。

若实现审计表明某个新模块只剩极少代码，会合并进相邻单一真相模块，避免为了目录结构制造抽象。

## 5. 实施顺序

1. 先实现并独立验证进程句柄；
2. 再建立 failure kind，保持现有错误文本兼容；
3. 实现 create-once attempt chain，并以 opt-in scope 接入 train/eval；
4. 实现 GPU lease，并接入同一 logical-job wrapper；
5. 收敛单设备 argv 校验；
6. 实现公共容量收据与 Stair/Push/Neutral task-owned minimum core；
7. 运行聚焦测试、旧插件回归、项目边界测试、完整测试；
8. 记录最终文件 hash、测试命令、已实现/未实现边界和 P2 所需接口。

## 6. 严格验收矩阵

### Runner 生命周期

- 正常/非零/timeout/interrupt/launch failure 均关闭日志句柄并 reap；
- timeout 和 interrupt 后，测试子进程及孙进程均不存在；
- cleanup 自身失败不得覆盖原始异常；
- 错误只带有界、脱敏日志尾部；
- Windows 本地测试与 POSIX 条件测试分别覆盖其进程组路径。

### Attempt/retry

- 首次 `summary_missing` 或 timeout 在显式 scope + 预算足够时只重试一次；
- candidate/scientific/contract/binding/device/tamper/interrupt 不重试；
- reservation/result create-once，hash 可独立重算；
- attempt 跳号、缺前驱、第二次 retry、scope/job/device/env 漂移、跨 session/segment 复用、预算少记均拒绝；
- reservation 后 hard crash 模型保持未闭合，不由 P1 猜测结果；
- 无 scope 的现有 job 仍精确执行一次。

### GPU 治理

- 同一物理 identity 同时只能有一个 lease；
- 不同物理 identity 可并行持有；
- 正常释放和持锁进程异常退出后都可重新获得；
- ordinal 与物理 identity 不匹配拒绝；
- attempt scope/resource receipt/summary 的 device 不能漂移。

### Prompt

- Stair bootstrap 有 `no_closed_outcome`；
- `feedback_entries=[]` 且超限来自非-feedback 段时仍必须进入 minimum-core；
- accepted current-best、latest rejected、最大合法 candidate/component identity、最大合法历史形状都能得到完整或 minimum-core；
- live 启动前最坏 task shape preflight 两次结果一致，失败时没有 invocation journal/ledger 行；
- 最终 provider-facing task sections <= 32 KiB，其 canonical bytes/hash/headroom 来自实际出厂对象；
- omission receipt 与投影内容 tamper 可检测；
- minimum-core 保留项为零缺失；
- cap+1、非法字符串、未知 key、public-safety 违规仍拒绝；
- Push/Neutral 使用各自 task-owned最小核心，且原有正常形状输出不变。

### 兼容与边界

- 原 train/eval argv、summary/device/binding/checkpoint 语义不变；
- current round-v7、Push、Stair、Neutral、dry-run tests 不退化；
- Python 3.10 语法/stdlib 兼容；
- `git diff --check`、compile、聚焦测试和全量 suite 全部通过；
- 不产生 commit/push/server sync/GPU/API 调用。

## 7. 停止条件

出现以下任一情况立即停在 P1，不向后续 phase 扩展：

- 为实现 retry 被迫改变 ledger/acceptance 或 current-best 语义；
- Prompt minimum-core 需要在公共层理解 Stair 指标；
- GPU lease 必须依赖外部常驻服务；
- 旧 width=1/Push 只能通过行为放宽才能兼容；
- 全量回归出现无法与 P1 改动隔离的真实阻断。

# P6-A 实施计划：双卡资源 Shadow Pilot

> 状态：实现、Linux 双卡完整矩阵和独立重算均已完成；G2 资源门 GO，等待用户验收
> 实现链：`4b8473e` → `c9e77b4` → `811b789` → `0c46b68` → `4b2aefb`
> 边界：只验证资源、进程、时间和清理；不验证候选能力，不进入 P6-B。

## 1. 目标与不做什么

本阶段只回答一个问题：在同一台 Linux 服务器上，两条真实 Stair/Isaac/RSL-RL 训练 lane 固定绑定两张物理 GPU 后，能否以可接受的墙钟、allocated-active GPU-hours、RAM、swap、CPU、IO 和磁盘代价稳定并行，并能在预注册的 active 后 SIGTERM 中断时独立收口。

P6-A 明确保持以下计数为零：

- provider 请求；
- public-dev、public-accept、sealed 和 final-confirmation 查询；
- primary round/generation ledger 写入；
- selector、acceptance、current-best/head transition；
- ExplorationEvidenceCard、frontier 或机制结论。

因此不调用 `execute_fixed_width_generation()`，也不扩建通用 DAG、Agent runtime 或动态资源调度器。

## 2. 已有基座与复用原则

直接复用：

- `GpuLeaseManager`：物理 GPU 的进程持有锁；
- `run_owned_process`：Linux 独立 process group、TERM/KILL、wait/reap；
- Isaac execution-environment manifest：两 lane 的软件执行身份；
- `generation_resource_plan_hash()`：冻结宿主物理资源标签
  `lane-a -> cuda:0`、`lane-b -> cuda:1`；
- Stair 唯一 `_execute_stair_training()` 训练体；
- 既有 canonical JSON/hash/create-once 文件原语。

只借鉴 Codex 的资源治理思想：先获得资源再 spawn、取消后等待真正清理、以进程组而非单 PID 为所有权边界、排队/启动/active/full-wall 分开计时。不会照搬 Codex 的通用工具注册、gRPC、沙箱、TTL 或任意宽度调度器。

## 3. 对总计划的一处有理收窄

总计划原文写“固定、预验证的 candidate fixture”。P6-A 不测机制效果，候选身份与 authoring authorization 只会增加无关失败面。因此本阶段改为使用仓库内 tracked、已资格化的官方 Stair reference complete-reward program，统称 **preverified program fixture**：

- 与候选走相同的 complete-reward dispatcher、Isaac env、PPO 和 checkpoint 热路径；
- 不依赖历史 output 或 provider；
- fixture source/qualification/protocol hash 全部进入 shadow plan；
- 不把浮点差异解释为机制效果。

正式 generation 仍使用候选制品，本收窄不改变 P4/P5 或旧 Push 插件。

## 4. 固定实验合同

### 4.1 工作负载

新增一个 Stair-owned、不可由 CLI 任意调参的 `resource_shadow_v1` preset：

- `num_envs = 4096`；
- `num_steps_per_env = 24`；
- `max_iterations = 64`；
- `budget_steps = 6,291,456`；
- `checkpoint_save_interval = 50`；
- `cuda_visibility_binding = single_physical_gpu_uuid`；runner 只对子进程设置
  `CUDA_VISIBLE_DEVICES=<计划 GPU 的完整 UUID>`，两条子进程内部的
  `process_execution_device` 都固定为逻辑 `cuda:0`；
- `renderer_multi_gpu = false`、`renderer_multi_gpu_auto_enable = false`、
  `renderer_multi_gpu_max_gpu_count = 1`；Kit 的 `renderer/activeGpu=N` 仍使用
  宿主/Vulkan 物理 ordinal，physics/RSL-RL 使用子进程内逻辑 `cuda:0`；
- seed、agent configuration、训练分布、reward program 与正式训练同源；
- `acceptance_authority = false`、`ledger_authority = false`。

它保留正式峰值 env/step 形状，只缩短迭代数，约为正式单候选训练的
`3.2%`。按既有 2,000 iterations / 59 分钟的量级估算，64 iterations 可提供
约两分钟 active window；每条 measured lane 还必须达到至少 60 秒 active 时长和
至少 45 个与该 lane PID/物理 GPU 对应的 1 Hz 样本，否则 pilot 不完整。数值在
create-once plan 中冻结，pilot 开始后不得按结果调整。

### 4.2 运行矩阵

固定顺序：

1. GPU0 single：1 warm-up + 3 measured；
2. GPU1 single：1 warm-up + 3 measured；
3. paired：1 warm-up + 3 measured；
4. interruption pair：两 lane active 后终止 lane-a，lane-b 必须正常完成。

warm-up 不进入统计。首版不做选择性 replacement；任一 measured 样本无效即把 P6-A 标为 incomplete/no-go，保留完整失败 receipt。

### 4.3 身份链

启动任何 job 前 create-once 冻结：

- 当前 commit；
- 同一个版本化 Session Genesis hash；
- 同一个 Isaac execution-environment manifest hash；
- fixture、workload 和资源计划 hash；
- `/etc/machine-id`、`cuda ordinal -> GPU UUID` 与 PCI bus 诊断；
- 每条 job 的 `physical_execution_device`、`physical_gpu_uuid`、
  `process_execution_device=cuda:0` 与 `renderer_active_gpu_ordinal`；
- 完整运行矩阵、采样口径和 go/no-go 阈值。

硬件/driver 只进入 resource receipt，不污染 software execution manifest。两张卡必须具有不同 GPU UUID/physical-resource hash；不能只检查含 ordinal 的 identity hash。

### 4.4 实施中发现并冻结的资源绑定修正

前两次诊断运行都在首个 GPU0 warm-up 后按原阻断门停止：

- `c9e77b4` 输出
  `/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_c9e77b4_20260818T131516Z`：
  同一目标 PID 在非计划 GPU1 建立约 540 MiB compute context；
- `811b789` 输出
  `/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_811b789_20260818T133055Z`：
  `multi_gpu=false` 后，PID `234532` 仍在 GPU1 保留约 500 MiB compute context；
  150 个有效样本中 GPU0 `nvidia-smi utilization.gpu` median/max 为 31%/44%，
  GPU1 为 0%/0%，
  child summary、初始状态和 checkpoint 均有效，故失败只归因于跨物理 GPU
  compute-context 门，而不是训练失败。

因此不放宽“目标 PID 只允许出现在计划 GPU UUID”这条门，也不引入被动 peer
context 白名单。根修是将宿主物理资源与子进程 CUDA ordinal 显式分层：计划、lease
和外部采样仍认宿主 `cuda:0/cuda:1 + UUID`；每条子进程的 CUDA compute 可见性
只包含自己的完整 GPU UUID，并在内部统一使用逻辑 `cuda:0`。Kit/Vulkan 仍可能枚举
宿主其它 GPU；GPU0、GPU1 的最小 AppLauncher full-UUID 探针证明的是目标 PID 只在
绑定 UUID 建立 compute context，且退出后无残留。这些探针只证明绑定可行，不计入
正式 18-job 矩阵或 G2 结论。

第三次诊断运行使用 `0c46b68`，输出为
`/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_0c46b68_20260818T141210Z`。
它完成了全部性能矩阵，但 fault ready 错把 active marker 前的启动样本计入，且用
`returncode != 0` 判断 SIGTERM 是否生效。本次 Isaac 进程收到 SIGTERM 后完成清理并返回 `0`，
因此该输出保持 no-go。`4b2aefb` 将权威证据改为两 lane 各自 marker 后的精确
PID+UUID 样本、signal/target/marker/monotonic 绑定、lane-a 无正常 summary 且已清理、
lane-b 完整 pass；return code 只作诊断。resource-plan/workload/fixture、7 项资源阈值和
每 lane 3 个 matched samples 均未改变；旧输出保持 no-go，并以新 schema、新 plan 完整
重跑 13-case 矩阵。没有重投影旧输出或跨运行拼接证据。

## 5. 采样与统计唯一口径

- `full_job_wall`：owned process spawn 前至 wait/reap 完成；
- `active_interval`：Isaac App 已启动后、进入唯一 Stair 训练体前，到训练体返回并关闭 env 后；用同机 monotonic ns；
- `overlap_ratio = intersection(active_A, active_B) / min(active_A, active_B)`；
- `allocated_gpu_hours = sum(active_seconds) / 3600`；
- paired repeat `i` 与同序号 single repeat `i` 匹配，再计算 lane slowdown 和 allocated-active GPU-hours ratio；
- 每秒采样 GPU UUID/index、`nvidia-smi utilization.gpu`、显存、功率、compute PID，以及 `/proc` 的 RAM、swap、CPU、IO 和输出目录增量。

`nvidia-smi utilization.gpu` 是 kernel busy-time 采样，不是 SM occupancy。它只能作为诊断，不能替代 allocated-active GPU-hours 门，也不能直接推导同卡可并发的 lane 数量。

## 6. 最小职责分层

1. `runner/resource_shadow.py`
   - task-neutral GPU discovery/resource plan；
   - create-once shadow plan/receipt；
   - nvidia-smi 与 `/proc` 采样；
   - overlap/slowdown/allocated-active GPU-hours 的纯计算和 gate 汇总。
2. `tasks/stair_climbing/resource_shadow.py`
   - 唯一固定 Stair workload；
   - official fixture 身份与训练调用；
   - task-owned child summary。
3. `scripts/isaaclab_stair_resource_shadow_train.py`
   - 单 lane Isaac 子进程；
   - 校验 plan、Genesis、environment、fixture、物理 GPU UUID、进程内逻辑 device
     与 task-owned Kit ordinal；
   - 写 active marker 和闭合 summary。
4. `scripts/run_stair_resource_shadow_pilot.py`
   - preflight/create-once plan；
   - single/paired/fault 编排；
   - lease、sampling、receipt、统计与清理。

`run_owned_process` 只增加一次 spawn 范围内的环境覆盖：复制父环境后为该 child
写入计划 GPU 的完整 UUID，并删除 `NVIDIA_VISIBLE_DEVICES`；不得修改父进程环境或
把这条 P6-A 映射扩成通用动态调度器。active marker 与 summary 同时记录并复核物理
device、UUID、进程内逻辑 device 和 renderer 物理 ordinal。

不得复制 P4 selector、P2 trial executor 或 Stair reward/eval 语义到 common 层。

按 `86e5f48..b3162ee` 的 tracked diff 口径，本阶段共新增 5,771 行：implementation
code/scripts 3,935、tests 1,307、阶段文档与主计划 529。这个规模对一次 shadow 偏重，
但既有共享热路径只有 `process_handle.py` 净增 11 行，其余均为 opt-in shadow harness、
没有 P4/P5/正式执行调用者。后续冻结该诊断脚手架；P6-B 只提炼
`GpuProcessBinding + lease + child-scoped env` 原语，不把 shadow coordinator、采样器或
报告器升级成通用 production 调度系统。

## 7. 启动门

任何 GPU job 前必须同时满足：

- local/origin/server HEAD 相等、服务器 tracked worktree 干净；
- Linux 与绝对 Isaac Python；
- coordinator 父环境不得预设 `CUDA_VISIBLE_DEVICES` 或
  `NVIDIA_VISIBLE_DEVICES`；只有 runner-owned spawn 可按冻结计划向每个 child
  注入完整 GPU UUID，并删除 `NVIDIA_VISIBLE_DEVICES`；
- 两张目标 GPU UUID 与 plan 一致，且无未知 compute PID；
- 预注册本 pilot 峰值输出保守估计为 `8 GiB`，`/data` 可用空间大于
  `max(30 GiB, 3 × 8 GiB) = 30 GiB`；运行后只据真实峰值制定未来门，
  不回改本次门；
- `MemAvailable >= 64 GiB`（仅作为本 pilot 的保守启动门，不宣称是正式常量）；
- plan 中 provider/sealed/ledger/acceptance authority 全为 false。

## 8. Go/No-Go 验收

阻断门：

- 每种模式恰有 3 个有效 measured repeat，warm-up 未混入；
- 两 lane fixture、初始 policy hash、预算、environment manifest 相同；
- 目标 PID 只出现在计划绑定 GPU UUID；无未知 compute PID；两条 child 都报告
  `process_execution_device=cuda:0`，且物理 device、UUID、逻辑 device、renderer
  ordinal 在 plan、marker、training evidence 与 summary 中一致；
- paired overlap 中位数 `>= 0.90`；
- 每 lane paired/single active slowdown 中位数 `<= 1.15`，任一 repeat `<= 1.25`；
- paired allocated-active GPU-hours / matched single sum 中位数 `<= 1.15`；相对一个 single 基线 `<= 2.30`，后者是前者乘 2 的代数冗余展示门，不是独立证据；
- 无 OOM；swap used 增量 `<= 64 MiB`；
- fault pair 中 lane-a 终止不杀 lane-b，两个 process group、compute PID 和 GPU lease 最终全部清空；
- provider/sealed/ledger/current-best/selector/acceptance 计数均为 0；
- 定向测试、全量回归、Linux process-group 真测试、compile/diff-check 通过。

P6-A 报告只给出 G2 资源 go/no-go 和 P6-B 的 RAM/swap/CPU/IO/磁盘建议门。完成后停止，等待用户验收，不自动进入 P6-B。

## 9. 实施顺序

1. 冻结 task-neutral plan/receipt/统计合同；
2. 在唯一 Stair 训练体上增加固定 shadow preset；
3. 完成 child 与 coordinator，先用 fake process 验证 lease、并发和清理；
4. 跑定向及全量本地回归；
5. 精确 commit/push，并验证 local/origin/server HEAD；
6. 服务器启动门审计；
7. 执行完整 shadow 矩阵与 interruption probe；
8. 独立重算指标、检查残留、记录报告和 DEV_LOG。

## 10. 最终完成状态

正式输出为：

`/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_4b2aefb_20260818T151328Z`

最终 `13/13 cases`、`18/18 jobs` 闭合，report v2 `go=true`。主要结果：

- overlap median `0.9999694815`；
- lane-a/lane-b slowdown median `1.0294992890 / 1.0341294103`，最大 `1.0524981051`；
- paired/matched allocated-active GPU-hours ratio `1.0349108098`；
- paired/one-single allocated-active GPU-hours ratio `2.0698216196`（代数冗余展示门，不是独立证据）；
- fault 两 lane 在 signal 前各有 3 个 post-active exact PID+UUID 样本，随后只终止
  lane-a，lane-b 完整完成；
- provider/eval/sealed/ledger/selector/acceptance/exploration authority 均为 0；
- 运行结束后 compute PID、active lease holder、managed process 和 tmux 均为 0。

详细证据、资源门和文件 hash 见
[`P6A_ACCEPTANCE_REPORT_ZH.md`](P6A_ACCEPTANCE_REPORT_ZH.md)，开发过程见
[`DEV_LOG.md`](DEV_LOG.md)。本阶段在此停止；未经用户另行批准，不进入 P6-A2/P6-B。

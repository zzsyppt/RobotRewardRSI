# P6-A 验收报告：双卡资源 Shadow Pilot

> 验收结论：**G2 资源门 GO**
>
> 正式输出时间：2026-08-18 UTC；验收闭合时间：2026-08-19 CST
>
> 边界：本结论只证明固定双卡训练模式的资源、进程、墙钟、成本与故障隔离可行；不授权 P6-A2、P6-B、provider、评测、sealed、primary ledger 或科学 generation。

## 1. 本阶段实际回答的问题

P6-A 使用真实 Stair/Isaac/RSL-RL 热路径和固定 reference complete-reward program，验证以下事实：

- 宿主 `cuda:0/cuda:1` 能否分别绑定两个物理 GPU UUID；
- 每条 child 的 CUDA compute 可见性只包含自己的物理 GPU，并在进程内统一使用逻辑 `cuda:0`；Kit/Vulkan 日志仍可能枚举宿主其它卡，权威运行证明是目标 PID 只在绑定 UUID 建立 compute context；
- 两条 4096-env 训练 lane 能否稳定重叠，而不产生不可接受的 slowdown 或 allocated-active GPU-hours 放大；
- RAM、swap、CPU、IO、显存与磁盘增长是否在可控范围；
- 定向终止一条 lane 后，另一条 lane 是否继续完成，进程组、compute PID 与 lease 是否全部收口。

本阶段没有调用或写入任何科学决策面。以下是 plan-bound 声明计数，并经运行入口/调用面审计确认均为零；它们不是另建的一套动态 telemetry 计数器：

| 权威动作 | 实际计数 |
|---|---:|
| provider requests | 0 |
| public eval queries | 0 |
| sealed/final-confirmation queries | 0 |
| primary ledger writes | 0 |
| selector/acceptance/head transitions | 0 |
| exploration updates | 0 |

## 2. 冻结身份与权威制品

### 2.1 代码与运行身份

| 项目 | 值 |
|---|---|
| 分支 | `true-rl-agent-refactor` |
| P6-A 前置基线 | `86e5f48` |
| 正式运行 source commit | `4b2aefbe0bb0a22226dc9055a02eb3f83cea0a04` |
| Session Genesis hash | `744cc76d31704ebf2f5571c9bc8a7261242471d42a604fd86b3b05c0e9a2059b` |
| execution-environment manifest hash | `3e314c3aa47a414cb84bafcb55d7bafe7bf64f3cb7f775933600dfb05cb6aeda` |
| fixture hash | `340f6e3782c93cb2dca75549924638f947a6bfe391cea5ff638edcabcf8fbb1b` |
| workload hash | `13c61e8c8c9ecdaf6641edff746c43da28a5454cae1e3bb297a07aaeb1d76017` |
| resource-plan hash | `dea2c1484b99c918f89fd22433a50bcfaab75ede4c0ed47a8922a97b79da8afa` |
| plan hash | `ee9d5d9f11bbc92937f2d93791cdbe930812907d983ca6b4c4fa8f3228d38cf8` |
| GPU0 | `GPU-06661cdd-1a0e-904e-a2e3-705eda5fb65b` |
| GPU1 | `GPU-8cd998f2-d42e-394e-7aae-a6701bb9b187` |
| GPU/driver | 2 × RTX 5090，driver `595.58.03` |

运行前与运行后均验证 local/origin/server HEAD 一致；服务器 tracked worktree 干净。

### 2.2 正式输出

正式输出目录：

`/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_4b2aefb_20260818T151328Z`

| 制品 | SHA-256 |
|---|---|
| `resource_shadow_plan.json` 文件 | `20e346b5d9929c516a5aadc041bf61dcded093035e37ba5f827f26257a8377b2` |
| `preflight.json` | `1e09b7baa1ca8a6a1c2a775d887aa8134cffea020c24f15d2f4c0521f7c71d64` |
| `resource_shadow_report.json` | `50c998a9217ec764de5294f5dba7098c11788f0bc76aefbf102bfaf08ad0dec5` |
| core summary canonical hash | `9fe411de0e18ad271deb564461a03b29c1e661980d633f1c8b282c76a62954cc` |

resource-shadow coordinator 退出码为 `0`，report schema 为 `true_rl_agent.p6a_resource_shadow_report.v2`，`go=true`。

## 3. 固定矩阵与数据完整性

固定 workload 为：

- `4096 envs × 24 steps × 64 iterations = 6,291,456 env-steps/job`；
- checkpoint interval `50`；
- single GPU0：1 warm-up + 3 measured；
- single GPU1：1 warm-up + 3 measured；
- paired：1 warm-up + 3 measured；
- interruption pair：2 jobs。

最终闭合 `13/13 cases`、`18/18 jobs`、`17` 份正常 summary。故障 lane-a 被预注册信号终止，因此按合同不生成正常 summary。82 个 JSON 均为 canonical JSON；plan、preflight、case、snapshot、marker、summary 与 checkpoint 的引用/hash 全部匹配。18 份 lease receipt 的 holder、physical identity 和 binding hash 均与 plan 一致，运行后 cleanup-check 能重新取得并释放两张 lease。没有 replacement、选择性重跑或跨输出拼接。

## 4. 性能 Go/No-Go

所有 paired repeat 都与相同序号、相同物理卡的 single repeat 配对。结果如下：

| 指标 | 冻结门 | 实测 | 结论 |
|---|---:|---:|---|
| overlap ratio median | `>= 0.90` | `0.9999694815` | PASS |
| lane-a active slowdown median | `<= 1.15` | `1.0294992890` | PASS |
| lane-b active slowdown median | `<= 1.15` | `1.0341294103` | PASS |
| 任一 repeat 最大 slowdown | `<= 1.25` | `1.0524981051` | PASS |
| paired / matched singles allocated-active GPU-hours median | `<= 1.15` | `1.0349108098` | PASS |
| paired / one-single allocated-active GPU-hours median | `<= 2.30` | `2.0698216196` | PASS |

12 条 measured lane 的精确 PID+UUID 样本数为 `149–158`，均显著高于冻结的 45 样本门；active 时长也全部高于 60 秒。双卡并发带来的中位 slowdown 约为 `2.95% / 3.41%`，allocated-active GPU-hours 相对 matched single 总和增加约 `3.49%`。这里的 GPU-hour 是独占物理 GPU 的 active capacity-time，即 `sum(active_seconds) / 3600`；它不是 utilization 加权算力、FLOPs、能耗或金额。`paired / one-single = 2 × paired / matched-singles` 是代数冗余的展示门，不是第二项独立证据。

同序号两条 single full-job wall 之和与 `max(同次 paired 两 lane full_job_wall)` 的比值分别为 `1.92365 / 1.93016 / 1.90847`，中位数 `1.92365×`。这只证明本次固定 64-iteration **训练 shadow** 的吞吐提升；authoring、分阶段 eval、selector、sealed 和 generation barrier 都不在矩阵内，不能外推成完整 generation 或正式 2,000-iteration 训练的端到端加速。

## 5. 故障隔离证据

旧实现把 active marker 前的启动样本误计入 fault ready，并用 `returncode != 0` 判断中断。本次 Isaac 进程收到 SIGTERM 后完成清理并以 `0` 退出，直接证明 return code 不是可靠的中断事实。`4b2aefb` 从根上改为：

- 两条 lane 必须各自在自己的 active marker 之后取得至少 3 个精确 PID+UUID 样本；
- receipt 绑定 target job、PID、PGID、marker hash、signal 与 monotonic 时间；
- lane-a 必须无正常 summary、进程组已清理；
- lane-b 必须生成完整 pass summary 并完成 workload；
- return code 只作诊断，不作中断真相。

正式运行的独立重算结果：

- lane-a post-active 精确样本：`3`，分别约在 active 后 `0.766 / 1.764 / 2.779s`；
- lane-b post-active 精确样本：`3`，分别约在 active 后 `0.623 / 1.621 / 2.636s`；
- SIGTERM 在 lane-a/lane-b active 后约 `2.923 / 2.779s` 发出；
- target `PID=PGID=821223`，marker canonical hash 与 receipt 精确匹配；
- lane-a `returncode=0`，但无 summary，signal 后约 `0.359s` 闭合；
- lane-b 正常完成，`151` 个 active samples、`151.304s`，summary/checkpoint/hash 均通过；
- 两 lane 均无 cross-device、OOM 或 orphan；case 后 compute apps、未知 PID、sampler errors 均为空。

因此 `fault_isolation_and_cleanup=true` 有原始证据支持，不依赖错误的 return-code 假设。这是跨独立诊断 run 的工程合同纠错，不是把同一次 pilot 的结果回溯改判：resource-plan/workload/fixture、7 项资源阈值和每 lane `3` 个 matched samples 均未改变；旧 `0c46b68` 永久保持 no-go，case/report/core 升 schema 后以新 commit、新 plan 完整重跑 13-case 矩阵。

fault case 外层 `valid=true` 表示“预注册的中断 disposition 已按合同闭合”，lane-a 内层 `measurement.valid=false` 表示“该 job 没有正常完成并生成 pass summary”；两个字段描述不同层级，不是两套相互冲突的完成真相。

## 6. 资源结果与后续建议门

独立审计重算了 `2,195` 个资源 snapshots：

| 项目 | 实测 |
|---|---:|
| 最大 `MemAvailable` 下降 | `12,049,000 KiB`（约 11.49 GiB） |
| 最低 `MemAvailable` | `273,959,744 KiB` |
| swap 增长 | `0 KiB` |
| aggregate CPU busy fraction | `0.12688681`（约 12.69%） |
| disk read/write sector delta | `36,432 / 4,773,312` |
| 输出峰值/最终大小 | `539,318,858 B`（约 514.3 MiB） |
| GPU0/GPU1 峰值显存 | `5,930 / 5,898 MiB` |
| GPU0/GPU1 `nvidia-smi utilization.gpu` 峰值 | `44% / 45%` |
| GPU0/GPU1 峰值功耗 | `177.64 / 153.64 W` |
| measured process-resident busy-time 均值（single / paired） | `24.05% / 23.27%` |
| measured workload-active busy-time 均值（single / paired） | `25.91% / 24.97%` |

数据驱动的后续可用 RAM 门为：

`1.2 × 12,049,000 KiB + 10 GiB = 24,944,560 KiB`，约 `23.79 GiB`。

实施时建议取不低于 `25 GiB MemAvailable` 的整数门；当前 30 GiB 磁盘启动门继续保留，不因本次仅约 514 MiB 的 shadow 输出而下调。swap 仍保持 `<=64 MiB` 的阻断门。CPU 与显存观察到充足余量；IO 未出现阻断，但 system-wide sector delta 只作诊断，不能解释为设备饱和度、iowait 或延迟证明。本阶段不额外发明动态节流器。

上述均值是 measured cases 中、目标 exact PID+UUID 出现在 compute-apps 时对应物理 GPU 的全局 `utilization.gpu` pooled sample mean：process-resident 窗口为 `972 / 1005` 个 single/paired lane-samples；再限定到 workload active interval 后为 `902 / 936` 个。它不是 PID 专属利用率。`nvidia-smi utilization.gpu` 表示采样期内至少一个 kernel 在执行的时间比例，不是 SM occupancy 或可直接相加的“算力占用”。约 25% 的 busy-time 说明同卡并发值得单独测量，但不能据此断言尚有 75% SM 容量、两套 Isaac/CUDA context 不会竞争，或 width=4 已可行。

## 7. 诊断性 No-Go 与根修链

以下输出永久保留为诊断证据，不被改写成通过，也不与正式 GO 结果拼接：

1. `p6a_resource_shadow_c9e77b4_20260818T131516Z`：首个 GPU0 warm-up 在 GPU1 建立约 540 MiB compute context；
2. `p6a_resource_shadow_811b789_20260818T133055Z`：`multi_gpu=false` 后仍有约 500 MiB peer context，证明仅关闭 renderer MGPU 不足；
3. `p6a_resource_shadow_0c46b68_20260818T141210Z`：完整性能矩阵通过，但 fault ready 错把 active 前样本计入，且中断判定错误依赖非零 return code，所以 report 正确保持 no-go。

对应根修是完整 GPU UUID 的 child CUDA compute 可见性与 compute-context 隔离，以及 `4b2aefb` 的 post-active fault proof。Kit/Vulkan 仍可枚举宿主 GPU；没有把“枚举”冒充 compute 执行，也没有通过放宽 cross-device 门、给 peer context 加白名单或重投影旧报告来获得 GO。

## 8. 回归、兼容与文件身份

验收命令结果：

- 本地最终全量：`Ran 1270 tests in 222.066s; OK (skipped=21)`；
- 服务器 Linux 聚焦：`Ran 31 tests in 3.281s; OK`，Linux-only process-group 用例未 skip；
- `python -m compileall -q src scripts tests`：通过；
- `git diff --check`：通过；
- 正式运行后：0 compute PID、0 active lease holder、0 tmux/managed process，服务器 tracked worktree clean。

P6-A 没有修改 selector、acceptance、evaluation、ledger、prompt/memory、P4 generation 或 P5 exploration 语义。任务特殊 workload、fixture、Kit 参数与 summary 都留在 Stair-owned 模块；common runner 只获得一次 spawn 范围内的通用环境覆盖和 task-neutral resource plan/统计。

关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `scripts/isaaclab_stair_resource_shadow_train.py` | `bbc9cc69bee1068800d529f3af5bf2f86b3d9f980c773a3d8db1914f0e3d9d8b` |
| `scripts/run_stair_resource_shadow_pilot.py` | `eeeb0b4ccc482f1c60bfd691c7314a73d0afaebdab996f217d77252432896aa3` |
| `src/true_rl_agent/runner/process_handle.py` | `12a25c15580ebd3e1974540f44ca92c4bdd6aaf8c25acb673ac26b13e0d6589f` |
| `src/true_rl_agent/runner/resource_shadow.py` | `6b0b6625fa8b03bb7e89743099fe51471bb90d5b1cf947fcacf8710c7a010c39` |
| `src/true_rl_agent/tasks/stair_climbing/resource_shadow.py` | `006ef1b29db0ed2d1c7a5abe66f5c559fe91a5ae7a1d3ad3a659c9d41bb795d2` |
| `tests/unit/test_resource_shadow.py` | `0357ec0f077c380da8c2fc97516a1e0ccdd913f51d8d52eceb4254cef1c4c46a` |
| `tests/unit/test_runner_p1_reliability.py` | `6a63640e5a6e580770152f118e5a9804baf4ea9b97b3366567e6835304c2c016` |
| `tests/unit/test_stair_resource_shadow.py` | `f5b707c63f6b90317472dd0186c5dd9646af69581afe73698661153ba7b5d792` |
| `tests/unit/test_stair_resource_shadow_coordinator.py` | `8c1a9b903083887d8260eb8a04a2dbf11586df2a40516d87cb8d78688dfd52ac` |

## 9. 独立审计结论

三路只读审计分别覆盖：

- live 运行、case 推进、GPU/PID/lease/tmux 清理；
- 2,195 snapshots、13 cases、18 jobs、82 JSON 的性能/资源/hash 独立重算；
- fault marker、post-active 样本、signal、PID/PGID、summary 与 schema 的独立重算。

三路结论一致：core report 与外部 PID/lease/tmux/工作树清理审计共同支持 `report.go=true`，不是仅凭 coordinator 内部的 `coordinator_closed` 字段判定；P6-A 可判 G2 资源 GO，无剩余 P6-A 阻断。

## 10. 结论与下一门

P6-A 证明：在本次固定 64-iteration shadow 窗口和 3 个 paired measured repeats 内，width=2 的真实 Stair 训练在本机两张 RTX 5090 上达到约 100% active overlap、约 3% 单 lane slowdown 和约 3.5% matched allocated-active GPU-hours 增量；资源余量充足，预注册的 active 后 SIGTERM 中断能独立收口。该结论不外推为 OOM、SIGKILL、hard crash、同一 GPU 上两条 lane 的可行性，或 2,000-iteration 正式训练的长期稳定性证明。

但它没有证明正式 scientific generation 已可启动。当前 UUID 隔离与 task-owned Kit 配置只在 shadow 入口验证；正式 `SubprocessRunner`、Stair train/eval entrypoint、Genesis reference/baseline 和 generation parent public-dev 尚未消费该绑定。P6-B 的第一个阻断交付必须把宿主 physical device、完整 GPU UUID、child logical `cuda:0`、renderer physical ordinal 和实际 summary/receipt 串成单一绑定，并用 Kit 日志和外部 PID→UUID 采样复验，不能从 attempt scope 回抄“实际执行”或放宽 cross-device 门。当前直接在父进程内运行的 session-end final-confirmation 另列为正式 15-generation 终局前的 runner-owned 接线门；完成后只能声称 generation-vNext scientific eval 与 session-end final-confirmation 已接入，不得把继续保留运行兼容的 legacy width=1 路径也笼统包含进去。

本阶段在此停止。下一步只能先由用户批准并验收 P6-A2；只有 P6-A2 达到 `>=10/15 valid_structurally_distinct` 且等价漏放为 0，之后才可另行申请 P6-B 批准。不得把本报告的 GO 解释成任何自动启动授权。

# Stair Phase D 实施计划：固定 2000 的 R0 能力确认与 Final V2 冻结

## 0. 当前状态

- 本文是 Phase D 的唯一当前实施说明，取代旧文中的 500 选择结论，以及六档预算搜索、曲线收敛门、尾部 checkpoint 稳定门和三训练 seed cohort 设计。
- 2026-07-30 已由操作者固定 `max_iterations=2000`。该决定不声明 2000 最优，也不声明训练曲线已经收敛。
- 旧 500-iteration run 和 1000-iteration 诊断只保留为历史诊断，不参与正式协议、Final V2 lineage 或后续候选比较。
- 正式 pilot 已在 clean committed source `9c2b926` 上从随机初始化完成 2000 iterations；iteration 1999 checkpoint 的五 seed、七桶 `public_accept` 宏平均成功率为 `0.2392857143`，平均最大进度为 `3.6171964735m`，预冻结低阶能力门通过。
- 完整运行已重放为内容寻址 artifact `c786ed32...5816517e`；伴随 manifest 绑定 54 个文件、`333994266` bytes、逐文件 hash/size 和 `b727c156...07d22` report hash。
- Final V2 已用 schema v2 payload、lineage v4、完整 manifest 和新 trust anchor 原子迁移；正式协议 hash 为 `d08200cc...09bc009d`。
- termination capture 修复前的 R1/R2 checkpoint、曲线和报告受错误 `base_contact` 语义污染，不参与训练分布或预算决策。

## 1. 阶段目标与边界

Phase D 只确认：官方开源完整奖励 R0 从随机初始化训练 2000 iterations 后，其最终 checkpoint 是否通过预先冻结的最低上楼能力门，并把可重放证据冻结进 Final V2。

本阶段必须满足：

1. Pilot 只训练一次，固定一个与 formal reference/candidate 互斥的 private training seed。
2. `max_iterations=2000` 只来自 draft 协议；训练命令、预算核算、报告和 freeze 工具只能消费，不得各自提供默认值。
3. TensorBoard 完整性、末段斜率和波动仍记录，但仅用于诊断，不具有跳过评估、选择预算或阻止能力确认的权限。
4. 只评估 iteration 1999 的最终 checkpoint；能力门、成功定义、七桶和评估 seed 语义保持不变。
5. Pilot 不进入 Genesis、acceptance、current-best、candidate ledger 或 provider prompt。
6. Push Recovery、共享 Runner、共享 candidate 合同和默认 registry 不承载 Stair 特殊语义。

## 2. 固定预算合同

| 字段 | 冻结值 |
|---|---:|
| `max_iterations` | 2000 |
| `num_envs` | 4096 |
| `steps_per_env` | 24 |
| `budget_steps` | 196,608,000 |
| policy initialization | random |
| warm start | forbidden |
| checkpoint selector | final iteration |
| pilot training seed count | 1 |

预算策略为 `operator_selected_fixed_budget_without_optimality_or_convergence_claim`。报告必须同时声明：

- `budget_optimality_claimed=false`；
- `curve_convergence_claimed=false`；
- `capability_confirmation_checkpoint=final_iteration`。

任何重新引入 allowed tiers、首个通过档位、convergence pass/fail 或 checkpoint stability 选择权的代码都属于协议回退。

## 3. 证据与执行架构

执行链为：

`trusted R0 source -> qualified reward program -> one 2000-iteration Stair train process -> diagnostic curve summary -> final checkpoint byte verification -> one isolated Stair eval process -> fixed capability gate -> canonical report -> content-addressed artifact -> artifact replay -> Final V2 candidate -> trust anchor`

主要所有权：

- `protocol_v2.py` / `config.py`：固定预算和 Final V2 fail-closed 读取；
- `reference_pilot.py`：唯一 pilot seed 与 formal seed 的隔离绑定；
- `pilot_artifacts.py`：训练、曲线诊断、最终 checkpoint 能力证据和报告重放；
- `run_stair_phase_d_pilot.py`：单次训练、单次评估和 lifecycle 调度；
- `phase_d_run.py`：从原始 checkpoint、TensorBoard 和 episode outcomes 重建报告；
- `pilot_artifact_store.py`：内容寻址 artifact 与完整 manifest；
- `protocol_v2_freeze.py`：从通过重放的 report 和 artifact manifest 派生 Final V2；
- `isaaclab_execution.py`：每个 checkpoint 一个独立评估进程和一个 Simulator 生命周期。

评估进程内部复用同一个只读评估环境运行五个 `public_accept` seeds；每个 seed 都显式 reset seed 和 Stair task history。按 seed 重启 Simulator 会显著增加启动时间，并不提供额外任务语义隔离。GPU rollout 中 done 和 finite 状态在设备端累积，批量复制到 host 后再解码；首回合 tracker 仍只消费每个 env 的第一条 terminal transition。

## 4. 环境源码闭包

训练分布 evidence 必须绑定实际安装的 Isaac Lab 2.3.0 文件 bytes、size、hash、import package 和被覆盖符号。当前闭包至少包括：

- `TerrainImporter.update_env_origins`，覆盖升级、降级、最低等级裁剪和最高等级随机回绕；
- `terrain_levels_vel`，覆盖课程 move-up/move-down 判定和对 importer 的真实调用；
- `CurriculumCfg.terrain_levels`，覆盖环境配置到课程函数的连接；
- `G1RoughEnvCfg`，覆盖上游 G1 rough 配置来源；
- `G1RoughPPORunnerCfg`，覆盖上游 agent 配置来源。

验证器按 `isaaclab` 与 `isaaclab_tasks` 分别解析唯一 package root，拒绝 symlink、越界路径、bytes 漂移和缺失符号。该合同只存在于 Stair task 层。

## 5. Final V2 原子发布

Final V2 必须在同一提交中包含：

1. `configs/tasks/stair_protocol_v2.json` 的 V2 payload；
2. 新 `protocol_hash` trust anchor；
3. draft/report/pilot evidence/fixed budget/training/evaluation/artifact lineage；
4. 受信任的完整 artifact manifest 伴随配置，包含 URI、逐文件 hash/size 和 report/draft lineage；
5. 默认 composition 和 reader 对该 payload 的成功加载；
6. Final V2、entrypoint、composition、prompt preflight 和防伪测试；
7. 由新正式协议重新生成的 prompt preflight report。

Freeze 工具验证 candidate 的全部结构和内部 hash，但在生成候选时不拿旧 trust anchor 比较。正式 reader 始终要求新 trust anchor。这样避免“生成新 payload 必须先信任旧 payload”的循环依赖，同时不放宽生产读取。

## 6. 严格验收与停止条件

必须通过：

- clean committed source 上的一次 2000-iteration CUDA 训练；
- iteration 1999 checkpoint 的真实 bytes/hash/size 校验；
- 未放宽语义的五 seed `public_accept` capability evaluation；
- checkpoint、TensorBoard、raw outcomes、event log 和 artifact manifest 的内容寻址重放；
- Final V2 原子迁移后的本地完整测试与 Push Recovery 回归；
- source、origin 和服务器最终提交一致性检查。

以下任一情况发生时不得生成 Final V2：

1. 最终 checkpoint 未通过预先冻结的 reference capability gate；
2. 训练未精确执行 2000 iterations，或并非从随机初始化开始；
3. pilot/formal seed 或 TrainingBinding 重叠；
4. 真实执行源码、训练分布、PPO、checkpoint 或环境证据无法闭包；
5. artifact 不能从交付位置独立重放；
6. 为使 R0 通过而修改评估语义、能力阈值或成功定义。

曲线仍在增长、尾部 checkpoint 不稳定或单训练 seed 无法估计训练方差，不再是 Phase D 的冻结阻断项；这些只能作为后续 R0/R1 失败分析信息，不得重新获得预算决策权。

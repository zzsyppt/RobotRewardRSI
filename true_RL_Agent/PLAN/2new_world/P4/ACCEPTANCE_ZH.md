# P4 验收：Stair 与 Push 的最小共享编排

2026-09-10。**P4 复审工程 GO。** 最终代码本地全量 1599 项零失败、19 项环境跳过；目标全量 1599 项零失败、零跳过，本地跳过项在目标逐项成功。 当前正式宿主已按任务 × 机器人 cell 装配。当前结果、接口修复、102 文件同步和原实验保护检查见 [本轮复审](AUDIT_FOLLOWUP_ZH.md)；下方第 3–5 节保留首次交付的分批记录，不能当作本轮最终代码全量。完整建设目标中的 P5、Go2 Push 物理资格化（P6）和真实搜索（P7）仍未完成。

进场提交为 `cd48221384c7b658eac18c275298a393e1d7b381`。源码已同步至 `/data/yuyangliu/AGENT/true_RL_Agent`，本次未创建 Git commit。实现范围、阶段记录和 Codex 参考点见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)；当前运行参数见 [runbook](../../../docs/stair_fresh_15round_runbook.md)。

## 1. 实际边界与交付

唯一正式执行入口为 `scripts/run_generation_session.py`，要求显式 task/cell，固定 15 generations × width 2。两个旧 Stair 命令文件仅转发同一入口。没有增加工作流语言、另一套调度器或 Push 专用 session 状态机。

| 所有者 | 当前职责 |
|---|---|
| `agent_core/generation_application.py`、`generation_session.py` | 准备后的执行、两 lane 代循环、暂停/恢复、终态及报告 |
| `generation_preparation.py`、`generation_cost.py` 及已有 receipt/operation 模块 | 不可变合同、实际拓扑、预约、实际计费、剩余额度与回执 |
| `evaluation/phased_adapter.py`、`phase_assembly.py` | 带身份的 phase、分数和选择接口；不解释具体任务指标 |
| Stair task | 题池、分层评分、reference 资格 mask、反馈投影、独立 final study |
| Push task | 真实 Push 指标/guard、dev 选择、reference-program Genesis 与 final cohort 解释 |
| RobotProfile、RuntimeAdapter、cell binding | 本体事实、执行适配及任务与机器人的薄绑定 |

三个真实 cell 经过同一 CLI 准备、不可变基础文件恢复，以及 final checkpoint/GPU/seed 参数物化：`stair.g1_37dof.unified_v3`、`stair.go2_12dof.unified_v3`、`push.g1_29dof.reference_v1`。第三格/新本体的接入没有修改共享状态机。Push 工程 cell 沿用现有 G1-29 任务条件，不能据此声明 Go2 Push 合格。

新成本合同按显式 workload/source/operation 数据工作，不以 Stair V3/V4/V5 继承身份猜测执行拓扑。实际拓扑是两个 train/dev、仅 selected 候选 accept/sealed；保守 reserve、实际计费和未使用额度分别核验。旧合同的数字与 hash 仍由旧 reader 原义解释。

## 2. 最小验收与扩展检查

| 要求 | 已执行的验证与证据 |
|---|---|
| 双任务正式路径 | 两任务实际 composition、TaskPlugin、任务 phase 投影及 `SubprocessRunner` 回执组装，覆盖 accepted、rejected、no-selection、authoring incomplete、候选 runtime failure、基础设施 timeout；见 [目标全量日志](target_full_results.log) 及 [holdout 修复复核](target_holdout_followup.json) |
| reference 仍为 head | 两任务各走完 15 代、final、独立只读 audit；分别核对 214、215 文件不变，见 [Push](shared_push_reference_stage3_results.json)、[Stair](shared_stair_reference_stage4_results.json) |
| 真进程崩溃与恢复 | 退出码 91 → 恢复至第 3 代暂停 → 15 代 → final/audit；每任务训练次数仅为 3（reference + 两候选），ledger 前缀不变，见 [Push](stage7_push_recovery_results.json)、[Stair](stage7_stair_recovery_results.json) |
| 恢复身份 | 三格 CLI 拒绝不同/缺失 manifest、cost、continuation、session contract/final plan 和错误 GPU UUID；不会补造或覆盖既有冻结身份 |
| 确定性早拒 | public-accept 与完整接受判据共用原 guard、headroom、配对噪声和相等边界算术；sealed 依赖的结果仍运行 sealed；私有早拒证明不进入 prompt |
| 缺失阶段诚实表示 | 写 `not_run_due_to_early_reject`，不生成虚假的完整评测/成功门；关闭代后自然释放未消费的 sealed slot，不回写历史成本 |
| final 语义 | 明确 `phase`、`seed_count_per_job`、`total_eval_seed_units`；保留两项原 final 合同断言，并扩展 checkpoint/GPU 绑定到三格，见 [final 复核](target_finalization_followup.json) |
| 历史回放 | 删除旧宿主后，同一冻结 15 代报告重建值相同，251 文件 size/mtime/SHA256 不变，见 [只读复核](retired_host_history_replay.json) |
| 共享边界与删除 | 当前 23 个共享 generation 模块无 task 层导入；复审新增的 `generation_closure.py` 统一当前与历史 closure 证据格式。首次交付的删除核对见 [源码审计](final_source_audit.json)，当前接口归属见 [复审记录](AUDIT_FOLLOWUP_ZH.md) |

以上会话用固定 provider/runner 执行工程验收，不运行新的 RL 训练或外部模型请求。fixture 的候选 callable、任务投影、attempt/receipt、ledger、成本和 lifecycle 仍走生产实现。持久化崩溃场景另在 Linux 执行，未用 Windows skip 代替。

## 3. 首次交付的验证过程与修复

| 执行批次 | 结果 |
|---|---|
| [本地首次全量](local_full_results.json) | 1590 项，0 failure、1 error、19 skip |
| [目标首次全量](target_full_results.json) | 1590 项，0 failure、2 error、0 skip；2620.230 秒 |
| 修正后的完整单元回归 | [本地](local_unit_after_migration.json) 1499 项通过、15 skip；[目标](target_unit_after_migration.json) 1499 项通过、0 skip |
| CLI/provider/final 补充 | [本地](local_cli_provider_followup.json) 和 [目标](target_cli_provider_followup.json) 各 62 项通过 |
| holdout 创建器及原失败分支 | [本地](local_holdout_followup.json) 26 项通过；[目标](target_holdout_followup.json) 27 项通过，包含 Stair runtime/infrastructure 两个子场景 |
| 原 final 合同断言保留复核 | [本地](local_finalization_followup.json) 和 [目标](target_finalization_followup.json) 各 4 项通过 |

逐项核对当前测试目录与这些成功日志：**1594 个当前测试（1503 unit + 91 integration）全部有目标环境成功记录，未覆盖项为 0**。这是全量后针对变更与失败的分批闭合，不把多批次数简单相加；首轮 FAIL 回执原样保留。对应测试 ID 清单摘要及日志 hash 见 [覆盖核对](verification_coverage.json)。目标环境实际执行了本地受限的 Isaac 配置/recorder、Linux 进程及符号链接检查；本地 tensor/video 检查未被缺依赖跳过。

两项首轮错误均已处理：

1. 原测试还调用已迁移的私有 `memory._exploration_projector()`。断言改用现有 `exploration_projector()`，没有给产品添加旧别名，也没有削弱资格 mask 断言。
2. Stair final 创建器先写下随机 nonce，再检查几何隔离。冻结偏移由有限的 SHA256 字节派生，确有 nonce 使 `height_ood_level_3` 与已有 calibration 第 5 档完全相同；固定反例见 [复现记录](holdout_collision_reproduction.json)。现在由任务创建器在首次提交前排除几何重合，再写入有效 nonce。只对这个结构性冲突重新抽样，其他错误继续报错；不查询成功率、reward 或候选结果。既有 holdout、resume 与 replication 不重抽。两本体使用相同已提交 nonce 时，完整题池和所有 cohort 身份与修复前逐值一致，见 [冻结解释核对](holdout_frozen_mapping_check.json)。

额外入口复审补回了正式 CLI 的本地凭据检查，位置在昂贵的 reference 工作之前；使用原 provider 的配置与凭据解析，不新增网络探测。prepare-only 和不需要 provider 的终态审计/恢复仍可离线执行。

## 4. 首次交付的性能对照、旧分支和数据保留

同一旧 15 代 fixture、同一 v2 profiler 的独立报告重放：wall time **327.911 → 81.651 秒**；`project_public_history` **18 → 1**，cost `load_chain` **21 → 3**，`read_bytes` **1385 → 684**，`glob` **91 → 73**，`rglob` 保持 22。报告值不变，新的 audit 仍重新验证输入，不缓存跨变更结论。见 [基线](baseline_report_audit_measurement.json)、[优化后](shared_report_audit_stage2_measurement.json)。该耗时含 profiler，不能和未加 profiler 的 session wall time 混比。

删除了原 9479 行 Stair session、旧 launch/readiness 生成栈及对应 pilot/validation writer 支路；共同能力迁到实际所有者。旧报告仅保留 `generation_history.py`、`launch_history.py` 和必要的版本成本 reader，没有复制旧执行器到 archive/tests。有效测试迁移与退役理由见 [测试清单](retirement_test_inventory.json)、[符号归属](retirement_symbol_owners.json)。计入新增文件后，`src/scripts` 净减少约 4500 行；具体统计见源码审计。

源码同步后重查原完整 Go2 15 代：**71 个 checkpoint 均保留并核对 SHA256**；884 份权威产物和 48 个中间 checkpoint 对应的 P0 所选 932 文件、517212889 字节未变。该清单不包含目录里原本就有的历史日志/锁文件；没有将这些额外文件误当作新污染删除。见 [同步后保护检查](protected_history_after_deploy.json)。

## 5. 首次部署记录与后续边界

同步前服务器主检出 tracked tree 干净、HEAD 与进场提交相同；仅写入核验过的 98 个源码/配置/测试文件并删除明确列出的 11 个旧文件。当前共享 CLI 和两个转发命令在服务器 `--help` 均通过。写入/删除集合不含 outputs 或 checkpoint。见 [最终源码快照](final_source_snapshot.json)、[源码同步回执](source_deployment_receipt.json)。文档和验收证据另按清单同步。

P4 验收进程已全部退出；原有 GPU 进程未停止，见 [资源收尾](resource_closeout.json)。没有为本阶段请求额外训练或模型预算。

- P5 可基于此共享通路继续 authoring、prompt、memory 与机制证据工作。
- P6 必须用真实 Go2 Push 完成施扰、恢复语义、reference 资格与视频一致性；P4 的 G1-29 Push 工程通过不能替代这些门。
- Push P4 final 复用已有 sealed cohort，明确不声明新 seed/新几何泛化；Stair final 在研究开始前承诺私有实例，不能根据已审阅的 accept/sealed/final 表现改题求通过。
- 新物理研究仍需冻结当前源码 commit、合格 reference 及来自实际测量的成本合同；P4 fixture 的预算值不得用于真实训练。

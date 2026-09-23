# P2 审计修订验收（2026-09-24）

**结论：本次必要修复已完成，并通过 CPU、真实模拟器开关对照、旧档案回读和来源核验。** 原审计逐项裁定见 [AUDIT_RESPONSE_ZH.md](AUDIT_RESPONSE_ZH.md)；机器汇总见 [validation_summary.json](validation_summary.json)。P3 尚未实施。

## 真实数据补上了什么

| 新 Stair v2 | 每侧回合 | 有效控制帧 | 禁触终止帧 | 其中瞬时力为零、历史力触发 | 历史判据不一致 |
|---|---:|---:|---:|---:|---:|
| Go2 parent | 864 | 276,259 | 72 | 26 | 0 |
| G1 reference | 864 | 300,247 | 197 | 11 | 0 |

在 576,506 个有效控制帧上，保存的历史力按实际 `norm → history max → threshold → any forbidden body` 重算，与当时 `base_contact` 完全一致。269 个禁触终止帧中，补齐了原瞬时力无法表达的 37 个。两种角色映射分别实测：Go2 四足 + base，G1 两 ankle-roll + torso。新历史没有混入终止后的 padding。

原 decoder 的瞬时 support 判据保持原义。历史与瞬时 support 判据分别有 13,295 / 1,105,036（Go2）、11,240 / 600,494（G1）个有效 body-frame 不一致，说明 P3 必须标清采用哪一种操作数，不能把它们当同义字段。传感器配置周期 0.005 s、控制步 0.02 s、lazy 为 true；没有每槽时间戳，不能把三槽强行解释成三个等距物理子步。证据见 [Go2 history](stair_parent_contact_history.json) / [G1 history](stair_g1_contact_history.json)。

Push 新 numeric v2 在 G1 force、Go2 velocity 共 384 个首回合保存有符号世界系 `vx/vy/dx/dy`。六份新 seed 档案的向量范数与原 pose 范数一致，逐步稳定条件、dwell、恢复时序及终局重算通过。方向不自动等于迈步、碰撞对象或失败因果。

## 新版本开关对照

| 配对 | 每侧回合 | on 数值档案字节 | 采集 rollout/phase 时间差 | 结论 |
|---|---:|---:|---:|---|
| [Stair Go2 parent](results/stair_parent_verification.json) | 864 | 75,258,064 | −1.37% | PASS |
| [Stair G1](results/stair_g1_verification.json) | 864 | 80,557,814 | +1.19% | PASS |
| [Push G1 force v3](results/push_g1_v3_verification.json) | 192 | 3,256,456 | +4.24% | PASS |
| [Push Go2 velocity v2](results/push_go2_v2_verification.json) | 192 | 2,494,709 | +8.02% | PASS |

四对均使用同一 checkpoint/程序/协议/seed/cohort，既有任务主统计、终局与分母保持一致。Stair 还逐 episode 和完整 summary 重算相等。共新采集 2,112 回合，on/off 合计 4,224 回合。单对 wall 有噪声，负差不是加速结论；最大正差 8.02% 小于预定 25%。本次不重跑 Gen8 或另外两种 Push 组合；其原 v1 真实覆盖保留，当前 reader 回归另列。

| 短训练配对 | 每侧真实 transitions | 已采控制步 | on 档案字节 | 结论 |
|---|---:|---:|---:|---|
| [Stair Go2](results/train_stair_go2_verification.json) | 128 | 8 | 53,603 | PASS |
| [Stair G1](results/train_stair_g1_verification.json) | 128 | 8 | 87,534 | PASS |
| [Push Go2](results/train_push_go2_verification.json) | 384 | 24 | 115,444 | PASS |
| [Push G1](results/train_push_g1_verification.json) | 384 | 24 | 115,556 | PASS |

四对的初始策略、完整模型和优化器状态均精确相等；每对检查 17 个模型 tensor、51 个优化器 tensor。保存的 raw/weighted/total 算术、来源、控制步、三个阶段和环境分母均通过；Stair 排除一个 preflight 控制步。长计划最多 24 × 16，首点固定 1、其余分散：48000 步为 `1,3000,…15000 / 17000,…31000 / 33000,…47000`。第 100 步提前失败仍保留启动原始样本的边界通过 CPU 归档及算术检查。

这些短运行验证接线、样本保存和训练不变性，不证明完整长训练表现、全程贡献均值、精确首次非零或失败瞬间，也未新观测 PPO post-manager reward。没有重训历史策略。

## 兼容、来源与资源

- [三份旧 Stair](stair_v1_recompute_regression.json) 共 2,592 回合由新 reader 重算，summary 相等，历史力明确 `not_recorded`。[旧 Push](push_v1_recompute_regression.json) 12 份、768 回合，除新增方向缺失声明外与原派生 JSON 逐字段相等。[旧四组训练](legacy_training_recheck_anchored/summary.json) 用最终锚点版 checker 回读全部 PASS，原引用字节不变。
- [最终源码核验](source_equivalence_audit.json) 检查 385 份文件：当前生产源码与最终训练快照逐字节一致。评估快照 `9b411380…` 与训练快照 `c0d0f1a2…` 仅有 sampler 与训练 smoke checker 两个文件不同；全部评估生产文件相同。训练实际全部使用最终锚点版，不把较早快照冒充最终实现。完整文件清单见 [评估快照](source_snapshot.json) / [训练快照](source_training_snapshot.json)。
- 远端隔离根为 `/data/yuyangliu/AGENT/feedback_p2_audit_20260924`，生产快照不含 PLAN。原项目输出、权重、原审计和封存回执保持只读；本地删除已核对上传的两个临时 ZIP，仅保留指纹，远端不可变 ZIP/源码/档案保留。
- 最大档案 80,557,814 B，最大 Stair chunk 原始数组 7,839,360 B；所有报告均通过 2 GiB/档案、64 MiB/chunk 门槛。所有八对中最大正向 RSS 差 22,401,024 B，小于 512 MiB。RSS 指 GNU time 的进程高水位及 waited children 代理，非同时进程树总量或采集器独占分配。
- [新运行总账](results/run_inventory.json) 共 16 次计时启动，全部成功、无缺测；共 1391.60 秒，即 **0.38656 GPU 容量小时**。原 P2 + 本次合计 **1.21875 小时**，仍在原 8 小时分配内。模型/API 请求 0。完成后已检查无 GPU 计算进程残留。
- [旧总账来源更正](historical_metadata_check.json) 只补实读来源和文字口径，原 27 次运行成本与 full-report SHA 不改。两条手工 on 的 argv 缺失仍明确保留。

## CPU 与工程验收

以下是有针对性的集合，存在重叠，不累计成一个夸大的总测试数；未因本次修复重复原审计 1,790 项全套。

| 范围 | 结果与日志 |
|---|---|
| Stair recorder / 两形态 / reader / phase / replay | [56 项通过](verification_stair_contact_history.log) |
| Push numeric / force runtime / physical protocol | [32 项通过](push_direction_unit_tests.log) |
| 最终训练 sampler / driver（含长计划与提前失败） | [21 项通过](verification_training_sampling_v2_anchor.log) |
| binding / 祖先准入 / runtime 投影 / 记忆 | [41 项通过](verification_binding_memory.log) |
| 完整 generation：formal 有/无、失败回执复用 | [2 项通过](verification_generation_receipt_reuse.log) |
| 独立验收器 | [Stair 4 项](verification_stair_pair_regression.log)、[contact history 1 项](verification_contact_history_probe.log)、[training pair 5 项](verification_training_pair_v2.log)通过 |

哈希/源差分、Python 编译、改动 whitespace 和新增文档链接检查通过。生产语义留在任务模块；共享层不解释禁触、跌倒、恢复等任务规则。主计划更新了操作数、缺失、启动锚点和 P3 词条所有权边界；没有顺带建立视觉、检索、聚合或存储平台。

## 交付边界

本次已关闭两个影响 P3 的信息缺口，并清理必要的重复构造/读取。新数组仍存私有档案；作者当前增加的是有界采样覆盖事实，正式长期记忆仍为 14 KiB 的短状态。高层姿态词条、视觉输入、主动回查和改进收益需后续阶段单独实现与验证。旧 width-one 重复覆盖簿记、全程周期聚合、全身体接触、int32 和关节降采样均按逐项裁定延期。

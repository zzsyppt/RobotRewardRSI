# Phase 3R 计划审计处置：训练支持、OOD 目标与迁移边界

> 日期：2026-08-27
> 状态：**PLAN_AUDIT_RESOLVED**
> 范围：核对当前源码、受版本控制配置、P3A 记录与本地 eval-v5 ledger；本报告不是 Phase 3R 产品实现或 GO 回执
> 权威实施计划：[PHASE3R_CORRECTION_PLAN_ZH.md](PHASE3R_CORRECTION_PLAN_ZH.md)

> **后续勘误**：本报告中的双边“非平凡”筛选是 Phase 3R 当时的审计决定，已在任何 Phase 4
> reference 运行前由 family-level floor-only mask 取代。当前权威是 active suite 配置与总计划；
> ceiling family 不再剔除，空 mask 必须形成具名诊断失败。

## 1. 总结论

“地形降为任务内数据标签、统一训练分布、OOD 优先且 ID 受保护”的架构方向成立，不重启系统。原修正版计划仍有五个源头缺口：

1. 没有真正决定二维训练支持，现有 C0/C1 仍是两条一维曲线；
2. 没有决定课程，散文中的 stationary 与当前 `terrain_levels_vel` 冲突；
3. 把 `J_OOD` 当唯一收益会丢弃历史上已经证明很大的 ID 改进；
4. 没有把单 guard、双 ABI、双 binding 和旧 seed domain 收敛成唯一执行协议；
5. effective task hash 与部分 active consumer 仍把 G1-37 schema 当作 Stair 任务协议事实。

这些缺口已在主计划与 Phase 3R 计划的源头设计中解决。本轮没有半迁移产品代码；旧 active path 仍按当前字节运行，正式修改必须完整执行 Phase 3R 并通过其验收门。

## 2. 逐条审计结论

### A · 训练配置

| 项 | 判定 | 当前证据 | 处置 |
|---|---|---|---|
| A1 C0/C1 是两条一维曲线 | **真实，高** | `irregular_training.py::resolve_irregular_training_sequence` 用同一 difficulty 计算 riser↑ 与 tread↓；C0 只有 height range + 固定 0.40 m tread | C1 改为 10 riser rows × 3 patterns × 3 独立 tread bands；不得再从同一 difficulty 推导两轴 |
| A2 直接 union 吞掉 tread-OOD | **真实，高** | 旧 C1 配置 `[0.33,0.48] ± 0.018`，实际支持约 `[0.312,0.498]`，距 0.50 m 仅 2 mm | C1 draft 实际 tread 支持收窄为 `[0.352,0.448]`；0.30/0.50 m 各保留 52 mm 外推间隔 |
| A3 all-but-one height 硬约束 | **真实，中高** | `_validate_revised_height_range()` 强制 upper=倒数第二个 eval height，且只能有一个 bin 越界 | 从 unified active path 删除；训练支持与 suite 同次冻结 |
| A4 课程未定且 raw-distance 与 tread 耦合 | **真实，高** | 当前 C0 使用 Isaac Lab `terrain_levels_vel`；其 move-up 基于绝对位移/terrain size，不是任务完成比例 | 保留 level curriculum，但用 task-owned `goal_progress_fraction` 0.80/0.40 阈值；不采用 stationary 默认 |
| A5 initial level 5/1 | **不是性能缺陷，但有实现约束** | 两套预算同为 196,608,000 steps；复审上游发现 `TerrainImporter` 只有一个全局初始上限并默认按 columns 等分 env | 保留 5/1；task-owned seeded initializer 只写上游已有 `terrain_types/terrain_levels` buffers，同时实现 50/50 allocation，不新建 curriculum 状态机 |

### B · Evaluation 与 OOD

| 项 | 判定 | 批判性说明 | 处置 |
|---|---|---|---|
| B1 多个 strata 接近饱和 | **定性成立** | audit 中列出的逐点值并非全部来自同一个“已接受 current-best”状态，不能当一条单一曲线；但 eval-v5 确实显示多个 ID/tread 点接近饱和 | 先用 official reference 做 `[0.10,0.90]` family 非平凡资格；保留 `target_maximum_success_rate=0.85`，不再固定五层等权 |
| B2 旧 C1 18 题不是标量 OOD | **真实，高** | 在旧实际 support 内只能叫 unseen combinations | 新支持与 suite 同时冻结；旧 18 题逐题重分类。C0 anchors 承担全局 scalar OOD，C1 承担 composition/conditional frontier |
| B3 OOD-only 丢弃 ID 增益 | **真实，高** | 同 eval-v5 域，reference height-ID macro 约 0.634，已接受 round-3 约 0.978，ID 不是已饱和常量 | 选择最小加权方案：`J_primary=(2*J_OOD+J_ID)/3`，并增加 OOD/C0-ID/C1-ID 三个非劣 guards；不增加分阶段 reference epoch |
| B4 score 未定义 | **真实，高** | success 在困难层可能贴地板，而 progress 有连续信号 | `S=0.5*success+0.5*goal_progress_fraction`；固定 episode→cell→seed aggregation 顺序 |
| B5 缺非平凡上限 | **真实，中高** | 旧 C1 calibration 已有 0.85 ceiling，新计划遗漏 | 以 official reference 资格化 objective family；饱和/贴地板层只报告，不进 objective |
| B6 `stair_010` 反常 | **异常真实，根因未证实** | 非单调 success/progress 值存在，但“更可能是 classifier”仍是假设，不能直接固化 | 做一次 0.10/0.15/0.20 同 checkpoint/seed raw diagnostic；闭合前从 guard/资格化隔离 |
| B7 0.02 guard 小于噪声 | **历史缺陷真实，当前已部分修复** | eval-v5 ledger 使用的旧 v4 是固定 0.02；当前产品 v5 已实现 `min(cap, margin+k*paired_SE)` | v6 对每个 named guard 复用当前 bounded paired-noise 公式，不退回固定阈值 |
| B8 schema 只有一个 guard | **真实，高** | `AcceptanceProtocol` 只有单数 `guard_score_source/...` 字段 | 最小升版到 generic named-guards list；任务词汇只在 Stair score producer，不进共享 decision/replay |

### C · 身份链与迁移

| 项 | 判定 | 处置 |
|---|---|---|
| C1 双 alias 身份风险 | **真实** | 不持久化第二个 `active_cell_protocol_id`；继续用 `cell_variant_id` 作为唯一协议 epoch 字段 |
| C2 seed 派生会改变 | **真实** | unified protocol 无条件绑定 task protocol hash，因此新建 seed domain；legacy replay 只指旧身份下 decision/receipt 复算，不承诺新协议重跑同 seeds |
| C3 unified reward ABI 未指定 | **真实，高** | 新 Stair 全部使用 VNext；legacy 42-field V2 仅历史 reader。旧 candidate 重新执行必须重新资格化为新 subject |
| C4 legacy TrainingBindingV1 必须退休 | **真实** | unified Stair 无条件 TrainingBindingV2；只保留 G1-29 Push/历史所需的 generic legacy grant |
| C5 两套 reference gate 不一致 | **真实** | 退休 C1 `success>=0.20` gate；统一 readiness 对 easy C0/easy C1 分别用 success>=0.05 且 progress>=0.50，OOD 不设成功地板 |
| C6 draft/active 只有散文纪律 | **真实** | catalog 增加 `draft/active/archived` 单字段与 launcher 断言；不引入 workflow engine |
| C7 task protocol 仍混入 G1 schema | **真实，高** | `effective_stair_task_protocol_hash()` 哈希整个 `StairProtocolV2`，其中 `complete_reward_runtime` 固定 37-DoF shapes；evaluation semantics/profile 又嵌有受 V1 robot schema 污染的 `base_protocol_hash`；composition/evaluation profile/legacy adapter 仍读取 V1 `policy_compatibility` | P3R 直接 canonicalize Stair 权威字段，不包裹旧 V1/V2 hashes；robot/runtime 具体 schema 只进入 resolved cell hashes。旧 V1/V2 不改写、仅历史读取 |

### D · 分支收敛

| 项 | 判定 | 处置 |
|---|---|---|
| D1 43 处都需换来源 | **真实** | active `src/`/`scripts/` 中 `is_irregular_cell` 最终必须 0 命中；执行路径改读逐样本 `terrain_regime_id`，历史 reader 不把该 bool 暴露给新运行 |
| D2 模块位置与职责判据冲突 | **真实** | 只按职责：generator/query/decoder/task summary-score/public display 可读 regime；catalog/lifecycle/current-best 不可读。共享 decision 只消费 opaque named scores |

### E · 计划证据

| 项 | 判定 | 处置 |
|---|---|---|
| E1 “candidate public score 0.132 证明 C0 未优化” | **不成立** | 该记录是已拒绝候选且属于不同 eval-v6 语义，不能与 eval-v5 reference/best 比较 | 已从主计划删除；只引用同 eval-v5 域的 reference→accepted-best ID 提升，并明确 eval-v6/v7 当前无本地同协议结论 |

## 3. 收敛后的最小架构

最终没有新增 terrain axis、第二套 runner、分布 DSL 或自动课程搜索。新增的必要结构只有：

- task-owned 二维 C1 generator 配置；
- task-owned normalized curriculum；
- task-owned score/family 汇总；
- 由既有权威字段生成的 canonical Stair task projection；
- shared acceptance 的 named-guards 列表；
- catalog 的 draft/active/archived 单字段。

其余工作都是删除旧 terrain-cell 分支、统一 ABI/binding，以及重签一个新的协议 epoch。

## 4. 同类问题复审

本轮顺着相同根因继续检查到以下必须进入 Phase 3R 的调用面：

1. `reference_program.py` 目前按 `is_irregular_cell` 选择 legacy V2/VNext；
2. `robot_binding.py` 同时按该 bool 选择 environment descriptor、reference gate、reward schema 和 legacy grant；
3. `config.py::formal_training_binding()` 只对 irregular 绑定 task protocol hash；
4. acceptance decision、replay、public projection 与 prompt 目前都假设一个 guard，且 v5 `public_score_binding` 绑定 guard mean；v6 必须把 public score 固定为 `J_primary`、guards 独立成 vector；
5. selection/prompt/memory 虽位于 TaskPlugin 目录，仍有 lifecycle 分支，不能因为目录归属而豁免；
6. `_validate_revised_height_range()` 与旧 C1 gate 会在新分布启用前直接冲突；
7. `effective_stair_task_protocol_hash()` 仍传递性绑定 V2 的 G1-37 reward schema，evaluation semantics/profile 仍嵌旧 V1 base hash，`composition.py`、`evaluation_profile.py` 与 `legacy_reward_adapter.py` 仍从 V1 `policy_compatibility` 读取 robot/control facts；新协议必须直接投影语义内容并改读 canonical task projection + resolved projection/execution profile。

这些没有另开补丁 Phase，全部归入 P3R-W1–W4 的源头迁移。除 `stair_010` 的一次定向诊断和一次 ≤5% official-reference smoke 外，不增加额外 GPU 实验或防御性测试矩阵。

# Phase 3R 实际实施计划：Stair 单格身份、二维训练支持与统一验收

> 日期：2026-08-28
> 状态：**GO（工程边界完成；统一协议保持 draft）**
> 上游：Phase 0、Phase 1、Phase 2 已 GO；Phase 3A 的 joint groups、15-term reward、C1 generator/query、ordered suite 与身份强制可复用
> 取代：Phase 3A 中“C1 irregular 是独立 active cell，并需 fresh 完整资格化”的路线
> 下游：Phase 3B C2 OOD 校准；Phase 4 G1 unified reference 与 reward LOOP
> 权威总计划：[STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md](../STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md)

> **评分规则勘误（Phase 4 pre-reference）**：本文记录的 `[0.10,0.90]`、逐 stratum 资格化与
> `target_maximum_success_rate` 是当时方案，已被现行 suite 的 **family-level floor-only** 规则取代。
> 当前只排除 `S_family < 0.10`；ceiling family 保留，空 mask 产出诊断失败。实现与后续验收不得再以
> 本文旧规则为权威。

实施结果与可复算证据见 [开发日志](DEVELOPMENT_LOG_ZH.md)、[验收报告](PHASE3R_ACCEPTANCE_REPORT_ZH.md)、[契约快照](PHASE3R_CONTRACT_SNAPSHOT.json)和[真实运行时回执](PHASE3R_RUNTIME_SMOKE_RECEIPT.json)。本阶段没有执行完整 reference 训练或 candidate LOOP，也不作策略改善声明。

## 1. 本阶段要纠正什么

Phase 3R 不重建系统，也不重复 P3A。它只完成一次必要的协议迁移：

1. 将 G1-37 × Stair 收敛为一个 robot-task cell、一个活动协议 epoch 和一条 Genesis/ledger/current-best 谱系；
2. 将 C0/C1/C2 降为 Stair TaskPlugin 内的样本 regime，不再参与系统生命周期路由；
3. 把现有一维 C1 difficulty 曲线改成真正的 riser × tread 二维训练支持；
4. 用 task-normalized progress 课程替代与 tread 耦合的 raw-distance 晋级；
5. 统一使用 VNext reward ABI 与 TrainingBindingV2；
6. 冻结 ID/OOD score、非平凡 family、OOD 优先的主目标和三个独立 guards；
7. 保留旧协议的历史决策回放，但禁止把旧 seed、candidate 或 checkpoint 冒充新协议证据。

在本阶段 GO 和 Phase 3B 激活协议前，不启动旧 `stair.g1_37dof.irregular_v1` 的 fresh 196,608,000-step reference qualification，也不启动 Phase 4 candidate LOOP。

## 2. 效果差的因果诊断

旧 C1 reference 失败不是“官方策略无法爬 C1”，而是一个混淆实验：

- reward 从官方 15 项被压成 12 项，joint group 作用域被改成 all-joints，single-stance air-time 门控被删除；同状态对照中 climbing 相对 standing 的净收益约下降 92%，而 `-200` termination penalty 未变，站立/少动成为合理局部最优；
- C1 课程入口陡且难度跨度窄，旧 `terrain_levels_vel` 又以固定绝对位移晋级；在 8 m terrain 上 4 m 阈值对应 tread=0.48 m 时约 62% 完成、tread=0.33 m 时约 95% 完成，使不同踏深受到不同晋级标准；
- 既有 clean checkpoint 在同一批 C1 cells 上达到约 0.81–0.996 mean progress，证明 C1 geometry 可行；修正 tread 方向也只把失败训练 progress 从约 0.031 提到 0.061，进一步说明 terrain 不是主要瓶颈。

P3A 已修复 reward 的 15-term/关节组等价性。本 Phase 不再为独立 C1 cell 重训，而是在统一协议中同时解决训练支持和课程根因。

## 3. 冻结边界与设计决定

### 3.1 保留、退休与禁止

| 对象 | Phase 3R 处置 |
|---|---|
| G1 v2 profile-owned joint groups、官方等价 15-term reward | 保留 |
| C1 generator、IrregularFlatStairSurfaceQuery、ordered summary | 保留并改成 regime/data 驱动 |
| manifest、child reload、bindings、checkpoint lineage | 保留，改绑唯一协议 |
| C1 calibration 与旧失败实验 | 只读历史证据 |
| clean/irregular 两个 active cell、各自 reference/ledger/current-best | 退休 |
| `is_irregular_cell` 配置属性及 active 产品读取 | 删除 |
| Stair 的 legacy V2 reward ABI 与 legacy TrainingBindingV1 grant | 只留历史 reader/replay；新 Stair 不可使用 |
| G1-29 × Push legacy grant/训练/评测/回放 | 保持不变 |

LLM 权限仍只有完整 reward program。terrain、training distribution、curriculum、seed、预算、PPO、evaluation、acceptance 与 checkpoint authority 均由系统持有。Stair 特殊语义不得进入 AgentCore、共享 runner、RobotProfile 或通用 composition。

### 3.2 唯一身份与机器化状态

- 沿用 `cell_variant_id` 作为唯一持久化协议 epoch 字段，值不得包含 clean/irregular/rough；
- 不新增第二个持久化 `active_cell_protocol_id`；若有可读属性，只能由 `cell_variant_id` 派生；
- catalog 仅增加一个轻量状态字段：`draft / active / archived`；
- draft 只允许 compile、contract tests 和具名 calibration smoke，正式 Genesis/reference/candidate launcher 必须拒绝；
- Phase 3B 签发最终 C2 suite、acceptance 和 manifest hashes 后，draft 才能转为 active；旧 C0/C1 entries 转为 archived，只能由历史 reader 解析。

协议必须绑定 training distribution、evaluation suite、acceptance v6、reference gate、VNext reward schema/capability、task protocol、seed/budget/PPO 与 execution profile hashes。

其中 `task_protocol_hash` 的唯一权威是一份由既有字段组成的 canonical Stair task projection：只包含训练分布、terrain/outcome、evaluation/score、acceptance/reference gate 与 candidate capability，不包含 G1/Go2 名称、body/joint、固定关节数、action/observation shape、sensor/actuator 或 control dt。episode horizon 以秒属于任务；control dt/decimation 属于 execution profile，materializer 在 resolved manifest 下换算并校验控制步数。projection 必须直接哈希筛选后的语义内容，禁止用旧 `base_protocol_hash/task_protocol_hash` 或整个 V2 `protocol_hash` 作间接输入；新 suite/semantics/profile 统一改绑 projection hash，避免把 V1 robot schema 经旧 hash 重新带回。

VNext 的逻辑字段、单位和角色需求由 Stair 定义，具体 cardinality/shape 由 RobotProfile + RuntimeProjection 解析，并以独立 reward schema/capability/IO hashes 进入 manifest。`StairProtocolV1/V2` 保持字节不变，只供历史 reader；active Stair 不再从 V1 `policy_compatibility` 或 V2 固定 G1-37 reward schema 取得机器人事实。这里不增加通用协议 DSL，也不复制现有任务字段。

### 3.3 统一 D_train

| regime | 固定 allocation | riser | tread | 结构 |
|---|---:|---|---|---|
| C0 regular | 50% | 0.05–0.35 m | 0.40 m | 既有 regular height curriculum |
| C1 irregular-flat | 50% | base 0.08–0.20 m；±0.012 m 后实际数值包络 0.068–0.212 m | base 0.37/0.40/0.43 m；±0.018 m 后实际数值包络 0.352–0.448 m | 10 riser rows × 3 patterns × 3 tread bands |

实现必须让 tread band 独立于 riser difficulty。统一 grid 为 10 rows × 10 columns：1 个 C0 column，加 9 个 `pattern × tread_band` C1 columns；10 rows 保留当前 C0 curriculum 分辨率，使 initial max 5 仍落在较易的前半区。环境总数和 PPO budget 不增加。

训练与评测必须解析同一份 pattern family 定义。当前两侧不共享任何 family：训练为 `train_alt_v1 / train_pair_v1 / train_wave_v1`（乘子 |m| ≤ 1.0），评测为 `alternating / front_loaded / back_loaded`（乘子 |m| ≤ 2），名称交集为空且形状不同。若保持现状，phase/composition-OOD 实际是整类未见 family，而不是已知 family 内的组合泛化，与总计划 §3.3 第 1 条冲突。W2 必须收敛为一份 family 定义，训练覆盖每个 family 并只暴露预注册的一部分 pattern-phase 组合，评测留出其余组合。

逐级起伏幅度是独立于 riser/tread 的第三个几何因子，必须显式对齐或显式分层。当前训练幅度为 riser ±0.012 m、tread ±0.018 m，评测为 riser ±0.020 m、tread ±0.030 m，两轴均为 1.67 倍，使每一道 C1 评测题都在一根未命名的轴上越出训练支持。W2 只能二选一：把评测幅度对齐到训练支持内，或把 amplitude 作为具名因子写入 §3.5 分层表并单独 claim；不允许它继续隐含在 pattern 定义里。

Isaac Lab 2.3.0 的 `TerrainImporter` 只有一个全局 `max_init_terrain_level`，且默认按 columns 等分 env。Stair 用冻结 seed 在初始化时改写其已有 `terrain_types/terrain_levels` buffers：50% env 指向 C0 且 level∈[0,5]，其余 50% 在 9 个 C1 columns 间等分且 level∈[0,1]；之后继续复用上游 `update_env_origins`。不新增第二套 level 状态或修改共享 Isaac Lab。

以下旧约束必须从 active path 删除：

- `resolve_irregular_training_sequence()` 以同一 difficulty 同时 riser↑、tread↓；
- `_validate_revised_height_range()` 强制训练覆盖 all-but-one evaluation height bin；
- 旧 C1 `[0.33,0.48] ± 0.018 m` 的 0.312–0.498 m tread 支持。

这些数值是 candidate 前的 draft 冻结目标，不再整体推给“以后标定”。只有 §3.4 的短 smoke 能在协议激活前否决它们；任何调整必须更新配置、计划和 hash，candidate 出现后禁止移动边界。

### 3.4 Task-owned normalized curriculum

不采用 stationary from-scratch 作为默认，也不继续使用 raw 4 m displacement 作为 Stair 晋级语义。最小 curriculum 定义为：

- episode success 或 `goal_progress_fraction >= 0.80`：level +1；
- `goal_progress_fraction < 0.40`：level -1；
- 其余保持 level；
- `goal_progress_fraction` 由该样本 geometry 的实际 start/goal 有向距离计算，不得误用 episode 时间 `progress_fraction`；
- C0/C1 `initial_max_terrain_level=5/1` 保留；
- reference 与所有 candidates 使用完全相同的 schedule、seed 与 support。

允许一次 official-reference draft smoke，预算不超过完整训练的 5%（不超过 9,830,400 env steps），只判断两种 regime 是否均有覆盖、level 是否可移动、VNext reward 是否非退化、是否出现结构性零学习。smoke checkpoint 不进入 reference/current-best；失败保持 NO-GO，不自动加预算或引入 teacher。

### 3.5 Evaluation、score 与 claim

基础分层为：

| stratum | 冻结含义 |
|---|---|
| C0-ID | regular，height 0.05–0.35 m、tread 0.40 m |
| C1-ID | irregular，实际 h/t 和训练过的 pattern-phase 组合都在支持内 |
| height-OOD | regular，height 0.40 m、tread 0.40 m；全局高度外推 50 mm |
| tread-OOD | regular，ID height、tread 0.30/0.50 m；相对 C1 实际支持各外推 52 mm |
| joint-OOD | regular，height 0.40 m 且 tread 0.30/0.50 m |
| phase/composition-OOD | 已知 C1 family 内未训练的 pattern-phase 组合 |
| C1 conditional frontier | 越出 C1 支持但数值可能在 C0 出现，只作条件式 claim |
| amplitude frontier | C1，h/t 与 pattern-phase 在支持内但逐级起伏幅度越出训练支持；仅在 §3.3 选择“显式分层”时存在 |
| C2 mechanism OOD | Phase 3B 加入，PPO exposure 必须为 0 |

按新 C1 支持用生产 geometry 公式（`riser_base + m × riser_offset_unit`，phase 旋转后逐级展开）对现有 18 题逐题重算：只有 5 题的实际 riser 与 tread 序列完全落在支持内（`c1_irregular_02 / 05 / 10 / 11 / 14`），按 cohort 为 public_accept 1、sealed_validation 2、calibration 1、public_dev 1；若再要求 pattern-phase 组合也在训练集合内，则在 §3.3 的 family 收敛完成前该数为 0。因此 `C1-ID` 不能靠重新分类得到，W3 必须按新支持重新出题。

每个进入 objective 或 guard 的 stratum，在每个使用它的 cohort 内至少需要 3 个 cell；不满足的 stratum 不得资格化，也不得作为任一 named guard 的唯一来源。该下限与 `S_stratum in [0.10,0.90]` 同为冻结的资格规则，一并进入 suite/acceptance hash 与 `objective_family_mask_hash`。

旧 18-cell C1 suite 必须随新支持重新分类；区间内未抽样点只能叫 interpolation/unseen combination。C0 grid 承担无歧义的 task-wide scalar extrapolation，C1 承担 known-family composition/conditional frontier，C2 承担 surface mechanism shift。

每个 cell/seed 先平均 episodes，再按 cell 等权形成 stratum，最后跨 paired seeds。Stair task-owned score 为：

`S_stratum = 0.5 × macro_success_rate + 0.5 × macro_mean_goal_progress_fraction`

研究 family 为：

- `F_scalar = macro(qualified height, tread, joint)`；
- `F_c1 = macro(qualified phase/composition, C1 conditional frontier)`；
- `F_c2 = qualified C2 mechanism`。

official reference 对每个 stratum 应用 `S_stratum in [0.10,0.90]`，且预注册代表题不全部超过 `target_maximum_success_rate=0.85`；family 只汇总通过的 strata，并至少有一个 stratum 才能进入 `J_OOD`。suite/acceptance hash 冻结完整 cell pool 与资格规则；Phase 4 reference receipt 再机械地产生唯一 `objective_family_mask_hash`，并由 Genesis、candidate EvaluationBinding、decision/replay 绑定。未资格化 strata/family 继续报告但不进入 objective；没有任何非平凡 OOD family 时不得启动 candidate，每个 research claim 至少需要一个对应 family。

`stair_010` 的持续非单调低分只证明异常存在，尚不能归因 classifier。使用同 checkpoint/seed 对 0.10/0.15/0.20 m 做一次定向诊断，核对 geometry digest、goal/horizon、raw progress 和 terminal reasons；闭合前从 C0-ID guard 和非平凡性选择中隔离。

### 3.6 主目标与 acceptance v6

采用：

`J_ID = 0.5 × S_C0-ID + 0.5 × S_C1-ID`

`J_OOD = macro(qualified F_scalar, F_c1, F_c2)`

`J_primary = (2 × J_OOD + J_ID) / 3`

接受条件是 `J_primary` 达到 noise-aware 改善，并同时通过三个 named guards：

1. `J_OOD` aggregate 非劣；
2. `S_C0-ID` 非劣；
3. `S_C1-ID` 非劣。

`acceptance.protocol.v6` 只把当前单 guard 字段升级为通用 named-guards 列表；共享 decision/replay 统一应用 `min(0.05, 0.02 + 1.5 × paired_SE)` 初始容差与各 guard 的 absolute floor，不读取任何 Stair 名称。Stair TaskPlugin 负责 score 名称与语义。

`public_score`、current-best 排序和 objective delta 固定使用 `J_primary`；三个 guard 作为独立 vector 记录和判定，不能再把任一 guard 分数复用成 public score。

历史 eval-v5 round 4 的 objective 从约 0.929 提至 0.949、guard 下降约 0.025，而 paired SE 约 0.023；固定 0.02 的旧 v4 门因此拒绝。当前 v5 已有 bounded paired-noise 公式，Phase 3R 的工作是把它推广到 named guards，而不是重新发明一套统计框架。

### 3.7 Reference readiness 与历史边界

- unified official reference 从随机初始化、以 VNext 15-term official reward 和统一 D_train 训练一次；
- easy C0-ID 与 easy seen-C1-ID 分别满足 `macro success >= 0.05` 且 `macro goal_progress_fraction >= 0.50`；
- OOD 只要求 finite、完整、非结构性退化 telemetry，随后按 §3.5 判断是否非平凡；旧 C1 `success >= 0.20` 独立 gate 退休；
- 新 Stair 无条件使用 TrainingBindingV2 并绑定 task protocol hash；seed domain 因此是新的协议 epoch；
- legacy replay 只承诺旧 frozen manifest/seed resolver/ABI 下的 decision/receipt 复算，不承诺在新协议重跑出相同 seeds；
- legacy V2 与 VNext 必须在同一组物理状态上证明官方 15 terms 逐项数值等价；旧 candidate 要执行必须作为新 subject 重新资格化。

## 4. 实施工作包与工作量

### 4.0 本次实际执行冻结（2026-08-27）

结合当前产品调用链与 P0–P3A 证据，本次实现固定以下选择，后续代码、配置、测试和回执必须一致：

- 新协议 epoch 的唯一 `cell_variant_id` 为 `stair.g1_37dof.unified_v1`，不再把 clean/irregular/rough 写进活动身份；
- 新增 task-owned `stair_unified_protocol_v1.json` 作为 active Stair 权威；旧 `stair_protocol_v1.json`、`stair_protocol_v2.json` 与两个旧 cell 配置保持字节不变，只供历史工具读取；
- C1 的三种 pattern family 使用同一份 task-owned 定义；训练暴露每个 family 的冻结 phase 子集，评测只把未训练 phase/composition 归入对应 OOD；
- C1 评测的 riser/tread offset 与训练统一为 `±0.012 m / ±0.018 m`。本 epoch 不设置 amplitude frontier，避免把未命名的第三几何轴混进所有 C1 题；
- 统一训练 terrain 为 10×10：C0 占 1 列但由 seeded initializer 固定分配 50% env，C1 的 `3 family × 3 tread band` 占 9 列并分配其余 50%；curriculum 只移动 row，不移动 terrain type；
- catalog 只增加 `protocol_state` 单字段。manifest 继续使用既有 schema，但通过 `protocol_state_hash` contract 绑定状态；launcher 从 catalog 读取明文状态并拒绝 draft/archived 正式启动，避免再造 manifest/工作流版本；
- shared acceptance 只增加 generic named guards；Stair 的 `J_primary/J_OOD/C0-ID/C1-ID` 语义全部留在 TaskPlugin；
- Phase 3R 只允许具名 draft smoke，完整 reference 与 candidate LOOP 继续由 Phase 3B 激活后的 Phase 4 承担。

实际执行顺序为 W1 身份/投影/ABI → W2 分布/课程 → W3 suite/score/acceptance → W4 生命周期收敛 → W5 验收与证据。每个工作包只在其责任层解决问题，不为旧双-cell 路线保留新的兼容分支。

### P3R-W1：身份、ABI 与 binding 收敛（1.5–2.0 人日）

**目标**：建立唯一 `cell_variant_id`、机器化 protocol state、canonical Stair task projection、VNext-only Stair 和 TrainingBindingV2；让机器人 schema 只来自 RobotProfile/RuntimeProjection，并冻结旧 manifest/seed/ABI 的历史读取边界。

### P3R-W2：二维 C1 generator 与课程（2.0–3.0 人日）

**目标**：实现 §3.3 的独立 h/t 支持、50/50 family allocation 和 §3.4 的 normalized curriculum；把训练与评测收敛为同一份 pattern family 定义，并对齐或显式分层逐级起伏幅度；删除旧 all-but-one 高度约束。

### P3R-W3：分层 score 与 acceptance v6（2.0–3.0 人日）

**目标**：实现 task-owned stratum/family score、非平凡资格与每 stratum 最小 cell 数、`J_primary` 和三个 generic named guards；按新 C1 支持重新出题，完成旧 18-cell 逐题处置与 `stair_010` 定向诊断入口。

### P3R-W4：生命周期分支收敛（2.0–3.0 人日）

**目标**：让 terrain regime 只决定生成、surface query、evidence、task-owned summary/score 和 public display；删除 composition、Genesis、reference、prompt session routing、selection、ledger/current-best 中的 terrain-cell 分支。

允许读取 `terrain_regime_id` 的职责只有：terrain generator、surface query、transition/evidence decoder、task-owned ordered summary/score，以及 public prompt 中对已公开 regime 的展示。共享 decision 只读取 opaque named scores。模块目录不作为判据，调用职责才是判据。

### P3R-W5：迁移、短 smoke 与回执（1.5–2.0 人日 + 短 GPU）

**目标**：证明统一 draft 可以复用 P3A 能力、旧协议不能混入新运行，并为 Phase 3B 提供可复现入口。本包不执行完整 reference training。

## 5. 严格验收流程

### P3R-G1：身份与启动门

1. registry 对 G1-37 × Stair 只返回一个 draft protocol；manifest/ledger 不出现第二个协议身份字段；
2. draft/archived 无法启动正式 Genesis、reference 或 candidate；
3. 旧 clean/C1 manifest 只能走历史 reader；
4. child reload、TrainingBindingV2、EvaluationBinding、checkpoint lineage 均拒绝旧 identity。
5. 修改 Stair distribution/outcome/suite/gate 会改变 task protocol hash；只修改 RobotProfile/RuntimeProjection 的 body/joint/shape，或只修改旧 V1 `policy_compatibility` 并一致重算其历史 hash，都会保持新 task protocol hash、同时使对应 legacy/cell-local identity 按职责变化；该验收不得通过手填预期 hash 自证。

### P3R-G2：训练支持与课程

1. training distribution hash 可从 canonical config 重算；
2. 固定 seeds 下 unified grid 为 10×10、C0/C1 env 为 50/50、初始 levels 为 5/1，C1 h/t 二维组合、实际几何边界和 pattern 覆盖与 §3.3 一致；
3. riser 变化不再隐式改变 tread，tread band 变化不再隐式改变 riser；
4. curriculum 只按 success/goal progress 移动，等完成比例在不同 tread 上得到相同判定；
5. C2、sealed、final 的 PPO exposure 为 0；reference/candidate 解析同一 D_train；
6. ≤5% draft smoke 满足 §3.4；失败不签 GO；
7. 训练与评测解析到同一份 pattern family 定义，训练覆盖每个 family 且只暴露预注册的 pattern-phase 子集；
8. 逐级起伏幅度或已对齐到训练支持内，或已作为具名因子进入 §3.5 分层；不存在未命名的越界几何轴。

### P3R-G3：评测、objective 与 acceptance

1. 所有 stratum 由实际训练支持判定，旧 18-cell C1 suite 重分类无误；
2. success/progress/S_stratum、family、J_ID/J_OOD/J_primary 从 raw summaries 可复算；
3. 非平凡 family mask 只由 official reference 和冻结规则机械产生；mask hash 在 candidate 前绑定且整个 epoch 不变；
4. acceptance v6 对三个 guards 分别计算 paired tolerance、absolute floor 与原因；删除任一 guard 或篡改 samples 都被 decision replay 拒绝；
5. shared acceptance/decision/replay 源码不出现 C0/C1/C2、riser、tread、Stair metric 名；
6. public_dev/public_accept/sealed/final 可见性和成本上限闭合；
7. 每个资格化 stratum 在其使用 cohort 内 cell 数 ≥ 3；不满足者被机械排除，任一 named guard 不得由单个 cell 决定；
8. 旧 18-cell suite 的处置逐题可复算：在支持内、C1 conditional frontier、amplitude 越界、pattern 未训练四类各自有明确归属，不得笼统标注为标量 OOD。

### P3R-G4：ABI、reference gate 与历史回放

1. unified Stair 的 reference/candidate/prompt/runtime 均只解析同一个 VNext schema；
2. active Stair 的 TaskSpec、evaluation materialization、composition 与 reward runtime 不读取 V1 `policy_compatibility` 或 V2 固定 G1-37 schema；joint/body/action/observation/sensor/control facts 分别来自 resolved projection 与 execution profile；
3. official 15-term reward 的 legacy V2↔VNext 同状态逐项等价在冻结容差内；
4. unified Stair 不获得 legacy TrainingBindingV1 grant，G1-29 Push 仍按原 grant 运行；
5. easy C0/easy C1 readiness 与 OOD telemetry 规则按 §3.7 实现；旧 C1 gate 不可被新 catalog 解析；
6. 旧 receipt/decision 在旧身份下可复算，新协议重跑使用新 seeds/subject 并被明确标记。

### P3R-G5：代码所有权与分支归零

1. active `src/`、`scripts/` 中 `is_irregular_cell` 命中为 0；
2. terrain regime 不影响 catalog、composition、launcher、Genesis、reference lineage、candidate admission、selection/current-best 或 ledger routing；
3. TaskPlugin 与 canonical Stair task projection 不出现 G1 body/joint/USD、固定 joint count 或 policy shape；RobotProfile/RuntimeAdapter 不出现 Stair geometry/split/success/failure taxonomy；
4. 不用 exception list 或第二套 helper 保留旧分支。

### P3R-G6：定向异常与回归

1. `stair_010` 诊断给出 raw evidence；无法闭合时继续隔离，不能写成 classifier 已证实；
2. P3A joint groups、15-term reward、C1 mesh/query、ordered summary 与 identity enforcement 定向测试通过；
3. P2 recorder/decoder parity、旧 C0 decision replay、G1-29 Push materialization/共享 LOOP evidence 不退化；
4. source closure、生成回执和验收器在 clean checkout 可复算，不依赖 Git HEAD 或行尾；
5. 必要单元/集成回归和 `git diff --check` 通过。

### P3R-G7：成本与 GO

1. 只运行 contract/CPU suite、`stair_010` 定向诊断和一次 ≤5% draft smoke；
2. 不运行旧 C1 完整资格化或 candidate；
3. Phase 4 reference/pilot/validation 预算在 candidate 前冻结；
4. G1 draft 保持不可正式启动，直到 Phase 3B 签发 C2 最终 hashes；
5. G1–G6 全通过后签发 Phase 3R GO、验收报告、自动验收器与开发日志。

## 6. Phase 3B / Phase 4 交接

- Phase 3B 只加入 C2 generator/query/evidence 与 C2 OOD family，不创建新 cell；
- Phase 3B 冻结含 C2 的完整 cell pool、family 资格规则和最终 suite/acceptance/manifest hashes，并把 protocol 从 draft 转 active；
- Phase 4 只在 active unified protocol 上训练一次 official reference，由其回执生成 objective family mask；mask 有效且至少一个 OOD family 非平凡后才运行固定 pilot/validation；
- Phase 3B 修改 draft hashes 只替换未激活 draft；激活后的改变必须开启新 `cell_variant_id` epoch；
- P3A checkpoint/receipt 只作为历史/calibration evidence，不进入 Phase 4 reference/current-best lineage。

## 7. 停止与返回规则

- draft smoke 连 easy C0/C1 都结构性不学习：保持 NO-GO，返回 W2/W4 查训练支持、课程或 runtime，不增加完整训练；
- acceptance v6 需要 Stair 术语进入共享 decision：边界错误，返回 W3；
- 删除 `is_irregular_cell` 后仍需 terrain 决定生命周期：身份尚未收敛，返回 W1/W4；
- C2 被训练 sampler 采到：Phase 3R/3B 失败；
- 仅因 OOD 分数低而想恢复独立 terrain reference：拒绝；低但有效的 OOD 是统一 LOOP 的测量对象；
- 现有 catalog、manifest、TaskPlugin 和 generic named guards 已能表达需求时，不新增 DSL、workflow engine 或第二套 runner。

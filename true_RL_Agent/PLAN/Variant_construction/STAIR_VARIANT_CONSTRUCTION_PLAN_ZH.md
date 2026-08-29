# 机器人 × 任务矩阵与统一 Stair 分布自进化建设计划

> 文档状态：**架构修正版，当前权威计划**
> 更新日期：2026-08-29
> 当前进度：Phase 0、Phase 1、Phase 2、Phase 3R、Phase 3B、Phase 4 已 GO；Phase 3A 的可复用能力已并入统一 Stair 单格，独立 C1 生命周期已退休
> 下一工程门：**Phase 5：Go2 机器人行与统一 Stair 资格化**
> 总目标：在同一套受控 LLM LOOP 中扩展机器人 × 任务矩阵，而不是为每个机器人、任务或地形复制系统

---

## 0. 本次架构纠正

此前计划虽然把机器人和任务定义成两根轴，却又让 clean、irregular、rough 各自拥有 manifest、Genesis、reference、ledger、current-best 和 LOOP，事实上形成了第三根“地形轴”。这会带来三类问题：

1. 资格化成本按机器人 × 任务 × 地形重复；
2. 同一任务的地形难度被误写成独立系统身份，产生大量 is_irregular_cell 一类生命周期分支；
3. LLM 在各自小分布上优化，不能自然形成“统一训练分布上学习、固定 OOD 套卷上泛化”的研究问题。

本计划从源头改正为：

- 矩阵只有 **机器人轴** 与 **任务轴**；
- 每个机器人 × 任务格子在一个协议 epoch 内只有一份训练分布、一份评测协议、一条 Genesis/ledger/current-best 谱系；
- C0、C1、C2 是 Stair TaskPlugin 拥有的 terrain regimes 与数据划分，不是 cell；
- G1 × Stair 的训练分布包含 C0 和部分 C1；剩余 C1 与全部 C2 用于 OOD 评测；
- LLM 当前只修改完整 reward program；训练分布、课程、terrain、seed、预算、PPO、评测与验收继续由系统掌握；
- 未来若开放课程权限，只新增一个受限能力，不改变评测所有权。

这不是重启一套新系统。Phase 0–2、Phase 3A 中已验证的通用能力和任务侧地形能力继续复用；只撤销错误的地形生命周期身份，并用 Phase 3R 收敛到正确主线。

作出修正的直接证据是：

- 既有 C0 checkpoint 在同一批 C1 calibration cells 上已有约 0.81–0.996 mean progress 和 0.083–0.54 success，证明 C1 能提供有意义但未饱和的迁移信号；
- P3A 的失败训练同时改变了 terrain、reward ABI 表达与课程：错误的 12-term reward 把官方 15-term reward 的关节组作用域压平并删除 single-stance 门控，运动相对站立的净激励约下降 92%；raw-distance 课程又让初始 C1 分布缺少起步台阶。因此该失败不能归因于 C1 地形本身；
- 同一 eval-v5 比较域内，official reference 的历史 C0 height-ID macro 约为 0.634，已接受的 round-3 current-best 约为 0.978；这证明 LOOP 可同时带来显著 ID 与 OOD 增益，不能把 ID 永久降成只判退化、不计收益的 guard；
- 地形独立 cell 会把 G1/Go2 的核心资格化从 6 次放大到 18 次，却没有增加矩阵行或列；
- 当前 C1 训练用同一个 difficulty 同时增大 riser、减小 tread，实际只是一条一维斜线；其实际 tread 支持约为 0.312–0.498 m，几乎吞掉旧 0.30/0.50 m 前沿，不能直接并入统一训练分布；
- 当前源码直接扫描确认 is_irregular_cell 在 16 个产品模块出现 43 次，问题已经从 terrain 生成扩散到 reference、prompt、selection 等生命周期层。

其中 eval-v5 历史结果只用于说明目标设计，不能与 eval-v6/v7 分数混排；本地目前没有足以声明 eval-v6/v7 reference/current-best 水平的同协议证据。

---

## 1. 一句话目标与完成定义

### 1.1 一句话目标

让不同机器人通过少量 RobotProfile、RuntimeAdapter 和 RobotTaskBinding 适配接入不同 TaskPlugin；每个机器人 × 任务格子都能在系统冻结的训练/评测条件下，由同一个 LLM LOOP 持续提出完整 reward program，并由 runner 独占训练、评测和接受决策。

### 1.2 Core V1 矩阵

| 机器人 | Stair Climbing | Push Recovery |
|---|---|---|
| G1-37DoF | 当前主格子：统一 C0+C1 训练，C1/C2 OOD，运行 LLM LOOP | 非 Core；形态可复用验证由现有 G1-29 legacy 任务承担 |
| G1-29DoF | 非 Core；不强行把两个不同关节体合并为一个 profile | 已有 legacy 格子，必须保持不退化 |
| Go2 | 新机器人行：复用同一 Stair 任务语义、分布协议和 LOOP | Core 后强复用证明 |
| 六足等 | Core 后新增机器人行 | 按任务能力决定 |

Core V1 只要求三个真实格子：

1. G1-37DoF × Stair；
2. Go2 × Stair；
3. G1-29DoF × Push Recovery。

clean、irregular、rough 不计为额外格子，也不分别要求 reference 或 LOOP。

### 1.3 工程完成定义

Core V1 工程 GO 必须同时满足：

1. 机器人事实、任务语义和两者绑定有明确所有权，TaskPlugin 源码不出现具体机器人关节名或 USD 路径；
2. G1-37DoF 与 Go2 各只有一个 active Stair protocol，C0/C1/C2 不参与 session、Genesis、ledger、checkpoint 或 current-best 的路由；
3. 每个 Stair 格子只有一个系统冻结的训练分布，包含 C0 与部分 C1，训练记录证明 C2 暴露为零；
4. ID、连续变量 OOD、类别/组合 OOD 与 C2 mechanism-shift OOD 有不同且诚实的声明；
5. candidate acceptance 使用 OOD 优先但不丢弃 ID 增益的冻结主目标，并分别保护 OOD aggregate、C0-ID 与 seen-C1-ID 不退化；
6. G1 与 Go2 都通过同一 LOOP 入口的真实 runtime、reference 和多代闭环验证；
7. G1-29 × Push 的 legacy 训练、评测、回放和共享 LOOP 证据不退化；
8. improvement_observed 如实报告，但不是工程 GO 的必要条件。

### 1.4 明确不做

- 不建立 TerrainPlugin、第三个矩阵轴、数据库或第二套 runner/ledger；
- 不为 C0、C1、C2 分别维护 reference、Genesis、current-best 或多代 campaign；
- 不让 LLM 修改 terrain、训练分布、课程、seed、预算、PPO、评测或验收；
- 不把 Stair 的几何、成功、失败分类、clearance、support 等语义放入共享层；
- 不枚举机器人 × terrain 参数 × pattern × phase × roughness × friction × seed 的完整笛卡尔积；
- 不为了“防御未来一切错误”增加通用策略语言、规则引擎或大量安全脚手架；
- 不把一次 held-out pattern 的结果表述成对任意类别分布的普遍泛化。

---

## 2. 身份、版本与所有权边界

### 2.1 两级身份

**稳定格子身份**

robot_task_cell_id = exact RobotProfile + TaskPlugin

示例：

- stair.g1_37dof
- stair.go2
- push_recovery.g1_29dof

**活动协议身份**

`cell_variant_id = robot_task_cell_id + protocol_epoch`

沿用现有 `cell_variant_id` 作为 **唯一持久化字段**，但把它的语义收敛为 robot-task cell 的活动协议 epoch。它绑定当前训练分布、评测套卷、acceptance、reference gate、reward ABI、seed/budget/PPO 和 task capability，用于建立新的可比较域，但不增加矩阵维度；其值禁止编码 clean、irregular 或 rough。

不再额外持久化 `active_cell_protocol_id`，避免同一事实出现两个可漂移来源。若代码为了可读性提供 `active_protocol_id` 属性，它只能由 `cell_variant_id` 派生且不得进入 manifest/ledger 的第二字段；字段正式改名留给未来单独 schema 迁移，不在 Core V1 内展开。

### 2.2 Terrain 只作为任务内样本事实

每个 Stair 样本可携带：

- terrain_regime_id：C0 regular、C1 irregular-flat、C2 bounded-rough；
- split_role：train、public_dev、public_accept、sealed_validation，Phase 7 才新增 final_holdout；
- geometry digest、realization seed 与必要 factor metadata。

这些事实用于 terrain 生成、证据解码、分层汇总和 prompt 投影，但不得决定：

- 选择哪条 LOOP；
- 创建哪份 Genesis 或 ledger；
- 选择 reference/current-best/checkpoint lineage；
- 选择机器人 profile；
- candidate 是否拥有训练或评测权限。

### 2.3 协议 epoch 规则

下列变化必须新建 `cell_variant_id`，并开始新的 Genesis/ledger 比较域：

- 训练支持集或采样概率发生实质变化；
- 评测套卷、objective、guard 或阈值改变；
- reward 输入 ABI、observation/action schema 或成功语义改变；
- seed、预算、PPO 或 reference 资格化协议改变。

旧协议的 ledger 与 checkpoint 保留为历史证据；旧分数不能直接和新协议分数混排。这里的“legacy replay 不退化”只指在旧冻结 manifest、旧 seed resolver 和旧 ABI 下复算既有 decision/receipt，不承诺用统一协议重新生成相同随机样本。旧 checkpoint 可在新协议下重新评测，身份必须标为 calibration_anchor 或 transferred_anchor，不能伪装成新协议 reference；旧 candidate 若要重新执行，也必须作为新协议下的新 subject 重新资格化。

### 2.4 各层唯一职责

| 层 | 拥有 | 禁止拥有 |
|---|---|---|
| RobotProfile | body/joint 名、默认姿态、尺度、执行器、传感器挂载、角色候选、关节组 | Stair 高度、pattern、成功、失败分类、评测题 |
| RuntimeAdapter | simulator 原始张量到形态事实的读取 | reward、任务阈值、acceptance |
| TaskPlugin | 任务成功/失败、terrain families、训练分布、评测套卷、objective/guards、prompt/memory 语义 | 具体机器人关节名、USD 路径 |
| RobotTaskBinding | 角色目标、少量尺度/映射、运行时来源选择 | 复制 RobotProfile 或 TaskPlugin，定义第二份任务 |
| ResolvedCellManifest | 冻结组合结果和所有协议 hash | 隐式默认、运行时自由拼装 |
| AgentCore / runner | 权限、预算、seed、训练、评测、决策、ledger | Stair、Go2、roughness 等任务/机器人特例 |
| LLM candidate | 完整 reward program | terrain、课程、PPO、预算、seed、eval、acceptance、checkpoint |

共享角色词汇只表达 base_kinematics、support_effector、forbidden_contact 和 joint_group.* 等形态中立概念。base_kinematics 的物理来源必须由 binding 明确为 articulation root 或具体角色，不再用 legacy/non-legacy 分支猜测。

### 2.5 唯一运行时编译产物

ResolvedCellManifest 必须至少绑定：

- exact RobotProfile、TaskPlugin 与 RobotTaskBinding；
- 唯一 `cell_variant_id` 与协议 catalog 状态；
- training_distribution_hash；
- evaluation_suite_hash；
- acceptance_protocol_hash；
- reference_capability_gate_hash；
- reward input/capability hash；
- observation/action/sensor schema hash；
- seed、budget、PPO 与执行 profile hash；
- resolved role projection 与 task protocol hash。

train/eval 子进程必须重载 exact manifest；TrainingBinding、EvaluationBinding、checkpoint lineage 和 runtime activation 必须比较同一协议身份。Genesis 记录该身份，但执行强制仍由子进程 manifest、binding 和 checkpoint lineage 完成。

task protocol hash 的权威输入必须先收敛为一份 **Stair task projection**：只投影训练分布、terrain/task outcome、评测与计分、acceptance/reference gate、candidate capability 等任务事实。它不得包含具体 body/joint 名、固定关节数、action/observation shape、传感器挂载、actuator 或控制时序。任务只冻结 episode horizon 的时间语义；control dt/decimation 由 execution profile 提供，materializer 在 resolved manifest 下换算并校验步数。该投影必须直接 canonicalize 被选中的语义内容，禁止把仍引用 V1 的 `base_protocol_hash/task_protocol_hash` 或整个 V2 `protocol_hash` 再包一层；统一 epoch 的 suite/semantics/profile 需改绑新 task projection hash。

VNext reward ABI 的逻辑字段、单位与角色需求由 Stair 拥有；随机器人变化的具体 cardinality/shape 由 RobotProfile + RuntimeProjection 解析，形成独立的 cell-local reward schema/capability hash 并写入 manifest。修改机器人事实必须改变这些 cell-local hashes，但不得改变相同 Stair 权威事实的 task projection hash；修改 Stair 分布、outcome、suite 或门则必须改变 task protocol hash。

现有 `StairProtocolV1/V2` 不原地改写，继续作为旧会话 reader 的冻结输入。Phase 3R 后 active Stair 代码不得从它们的 `policy_compatibility` 或 G1-37 `complete_reward_runtime` 取得 robot schema；这不是新增协议 DSL，而是把已经存在的任务字段与机器人字段各归唯一权威源。

统一 Stair 协议只使用 VNext reward ABI，并无条件使用 TrainingBindingV2、绑定 task protocol hash。legacy 42-field V2 ABI 与 Stair 的 `grant_legacy_training_binding_v1` 只保留在历史 reader/replay 边界；G1-29 × Push 的既有 legacy grant 不受影响。迁移门必须用同一组物理状态证明官方 15-term reward 在 legacy V2 与 VNext 投影上的逐项数值等价。

catalog 状态只保留 `draft / active / archived` 三值并机器强制：draft 只能做 compile、contract test 和具名 calibration smoke，不能创建正式 Genesis/reference/candidate session；只有 Phase 3B 签发最终 suite/acceptance/manifest hashes 后才能原子地转为 active；旧 C0/C1 协议进入 archived，不能作为新运行入口。该状态门是一个 catalog 字段和启动断言，不扩展成工作流引擎。

---

## 3. Stair 统一训练分布与 OOD 设计

### 3.1 三类 terrain regime

| regime | 几何定义 | 训练暴露 | 主要用途 |
|---|---|---:|---|
| C0 regular | 等高、等深、平面踏面 | 是 | 稳定起步、legacy 锚点、ID guard |
| C1 irregular-flat | 逐级高度与踏深不等，踏面仍为平面 | 部分 | 训练复杂性与标量/组合 OOD |
| C2 bounded-rough | 在 C1 踏面内部叠加有界微起伏，不改 riser/edge/目标区拓扑 | 否 | policy-level mechanism-shift OOD |

一个 active protocol 只有一个冻结训练分布 D_train。G1 × Stair 的 D_train 采用固定、可复算的 C0+C1 分层采样；其支持集和采样概率在 Genesis 前冻结，对 reference 与所有 candidates 完全相同。

Phase 3R 不再把最关键的支持集决定推迟为“之后标定”，先冻结以下 **draft 架构值**：

| 训练层 | 固定占比 | riser 支持 | tread 支持 | 生成结构 |
|---|---:|---|---|---|
| C0 regular | 50% | 0.05–0.35 m | 固定 0.40 m | 沿既有高度 curriculum 的 regular stairs |
| C1 irregular-flat | 50% | base 0.08–0.20 m，叠加 ±0.012 m 后实际数值包络 0.068–0.212 m | base 为 0.37/0.40/0.43 m，叠加 ±0.018 m 后实际数值包络 0.352–0.448 m | 10 个 riser rows × 3 pattern × 3 tread bands |

C1 的 tread band 必须成为独立子地形因子，不能再由同一个 difficulty 同时决定 riser 和 tread。统一 terrain grid 使用 10 rows × 10 columns：1 个 C0 column，加 9 个 `pattern × tread_band` C1 columns。10 rows 保留当前 C0 高度 curriculum 的分辨率，使初始上限 5 仍只覆盖较易的前半区，而不是在 6 rows 下从一开始暴露 0.35 m；环境总数与训练预算不增加，只增加一小块静态 terrain mesh。

Isaac Lab 2.3.0 原生 `TerrainImporter` 对所有 columns 只提供一个 `max_init_terrain_level`，并按 column 等分 env，不能仅靠配置同时得到 C0/C1=50/50 与 initial level 5/1。Stair 因此只在 env 初始化时以冻结 seed 写入其已有 `terrain_types/terrain_levels` buffers：50% env 指向 C0 column 且 level∈[0,5]，其余 50% 在 9 个 C1 columns 间等分且 level∈[0,1]；后续仍调用上游 `update_env_origins`。这不是第二套 curriculum state，也不修改共享 Isaac Lab。三种已知 pattern 都进入训练，但只开放预注册的一部分 pattern-phase 组合。

当前 `resolve_irregular_training_sequence()` 的 riser↑/tread↓单标量映射和旧 `[0.33, 0.48] ± 0.018 m` tread 支持不得直接并入新协议。上述较窄二维 C1 数值包络特意为 0.30/0.50 m tread 前沿各留下 52 mm 的实际几何间隔，同时避免训练只落在一条坡度曲线上。包络内未采到的离散值仍只算 interpolation/unseen value，不伪称训练样本。

训练保留课程，但不再使用 raw displacement 的 `terrain_levels_vel` 作为 Stair 晋级量。Stair TaskPlugin 提供唯一的 task-owned normalized curriculum：episode success 或 `goal_progress_fraction >= 0.80` 时升级，`goal_progress_fraction < 0.40` 时降级，中间为 dead band；`terrain_types` 固定，因此 family allocation 不变，课程只更新已有 per-env level。这里的 `goal_progress_fraction` 必须由每个样本 geometry 的实际 start/goal 有向距离计算，不是 episode 时间 `progress_fraction`，所以不同 tread 上同完成比例得到相同判定。C0/C1 的初始上限 5/1 保留，它们是合理的不同起点，不是预算冲突。

在 protocol 仍为 draft 且没有任何 LLM candidate 结果时，允许一次不超过完整 196,608,000-step 预算 5% 的 official-reference smoke，只验证两种 regime 均被采样、课程能双向移动、VNext reward 非退化且没有从零学习的结构性停滞。失败即保持 Phase 3R NO-GO 并修改 draft hash；不得扩大 smoke、引入 teacher、按候选表现调支持集或把 smoke checkpoint 当 reference。

### 3.2 连续变量的 train/OOD 切分

高度和踏深是标量维度，必须显式区分：

| 分层 | 定义 | 可声明的泛化 |
|---|---|---|
| C0-ID | C0，高度位于 0.05–0.35 m、tread=0.40 m | 训练分布内 |
| C1-ID | C1，实际连续参数位于上述 C1 支持，且 pattern-phase 组合在训练集合内 | 训练分布内 |
| height-OOD | C0 高度严格高于 0.35 m、tread=0.40 m；当前三 cohort 预注册 0.37–0.44 m 的不同锚点 | task-wide 单轴高度外推 |
| tread-OOD | C0 高度在支持内、tread 严格位于 C1 实际包络 `[0.352,0.448] m` 外；当前锚点落在 0.28–0.34 m 或 0.46–0.54 m | task-wide 单轴踏深外推 |
| joint-OOD | C0 高度严格高于 0.35 m，且 tread 同时位于上述包络外 | task-wide 双轴联合外推 |
| C1 conditional frontier | C1 的某一连续因子越出 C1 支持，但该数值可能曾在 C0 出现 | 只声明 irregular 条件下的 factor frontier，不宣称全局标量外推 |
| amplitude frontier | C1 的 h/t 与 pattern-phase 都在支持内，但逐级起伏幅度越出训练支持 | 只在 Phase 3R 选择保留幅度差时存在；只声明起伏幅度外推 |

位于训练数值区间内但未被抽到的具体点，只能称为 unseen interpolation 或 unseen combination，不能称为标量 OOD。

height/tread/joint OOD 使用 C0 regular anchors，是因为统一 D_train 的全局高度/踏深边界在这里可无歧义地控制；C1 主要承担已知 family 内的组合/相位泛化与条件前沿，C2 承担表面机制变化。旧 C1 18-cell suite 必须按新支持重新分层：位于支持内但组合未训练的题只能叫 unseen combination；高 riser 等越出 C1 支持的题只能叫 C1 conditional frontier；不得把整套 18 题继续笼统称为标量 OOD。

上述数值、占比和 claim 是 Phase 3R 的默认冻结目标。唯一可改变它们的时点是 candidate 前的 draft smoke；变更必须更新本计划、配置与 hash 后重新验收，不能在看到 candidate 结果后移动边界。

### 3.3 类别 pattern 与 phase 的切分

alternating、front_loaded、back_loaded 等是不具备自然顺序的类别。Core V1 采用以下低复杂度规则：

1. 所有已知 pattern family 都在训练中出现，避免把“未知类别发现”与控制能力混在一起。当前资产尚不满足该条：训练用 `train_alt_v1 / train_pair_v1 / train_wave_v1`，评测用 `alternating / front_loaded / back_loaded`，两侧名称交集为空、形状不同、乘子幅度相差 1.67 倍；Phase 3R-W2 必须先把两侧收敛为同一份 family 定义，否则本条 claim 不成立；
2. 不让所有 pattern × phase 组合进入训练，评测留出预注册的相位旋转和组合；
3. 这部分结果只声明 **已知 family 内的 phase/composition 泛化**；
4. “整类 pattern 完全留出”只作为 Core 后独立研究消融，且只能声明对被留出的该 family 有效。

这样既保留类别泛化问题，又不会用一个任意 held-out 类别承担整个生产验收。

相位是否 OOD 必须由展开后的 riser/tread 序列判定，不能只检查 `phase_steps` 是否出现在 held-out 列表。周期为 2 的 alternating 只保留能产生新序列的 phase 1；front/back-loaded 可使用彼此不同且相对训练 phase 0 真正改变序列的 phase 2/4/6。任何 held-out 标签若展开后与训练序列相同，配置加载即失败。

### 3.4 C2 的声明

C2 相对 C0/C1 是表面机制变化，不是高度或踏深的连续外推。所有 C2 样本在 PPO 训练中暴露必须为零，因此它是 policy-level OOD。

如果 public_dev 的 C2 结果进入下一代 LLM prompt，它不再是系统级完全未见数据；文档必须称为“策略训练未见、LLM 可见的开发 OOD”。只有 sealed_validation 与未来 final_holdout 中从未进入 prompt、memory 或候选结果驱动调参的数据，才可用于 hidden/generalization claim。

C2 必须由真实 mesh/height samples 和 task-owned surface query 激活。配置里存在 roughness 字段或 prompt 提到 roughness，不算实现证据。

### 3.5 评测分层与成本

正式套卷采用小规模覆盖设计，不做完整笛卡尔积：

- public_dev：给 LLM 的公开反馈，覆盖各 OOD stratum 的代表题；
- public_accept：决定 candidate 是否进入 current-best，不泄漏 sealed 细节；
- sealed_validation：系统拥有的复核门，题面几何与 public_dev/public_accept 均不重合；
- final_holdout：当前尚未实现，仅在 Phase 7 协议、候选和停止决策冻结后新增并一次性使用，用于最终研究报告。

Phase 3R 冻结时的三个 formal cohort 各有 `7 strata × 3 cells = 21` 道不同几何，完整正式题池为 63 道；Phase 3B 随后在同一 task protocol 内加入 C2，并收敛为每 cohort 24 道、每 cell/seed 3 episodes。两个数字分别是历史基线与当前协议，不是两套并存的 active suite。cohort 内不得重复，cohort 间 geometry digest 必须两两不交；geometry digest 只描述 mesh/目标几何，不包含 cell/cohort 标签，cohort 身份另由 cohort hash 绑定。

三套 formal cohort 的几何虽不重合，但每个 stratum 必须覆盖相邻的难度阶梯，不能让 sealed_validation 在所有外推轴上系统性更远。高度与踏深分别按相对训练支持的有符号轴和绝对外推距离配平；joint-OOD 同时配平两轴；C1 conditional frontier 固定为高度上界、踏深上界、踏深下界各一题；C0-ID 对齐平均高度，C1-ID 与 phase-OOD 对齐高度/踏深边际集合后再换组合保持题面不同。phase 是无序类别，不把 `phase_steps` 数值大小解释成难度距离，而用平衡不完全区组覆盖 held-out phase，并把行为难度交给 reference calibration 实测。`max_public_to_sealed_score_drop` 只有在该几何可比前提成立时才解释为泛化一致性门。reference 的支持性筛选按 public_accept 上的 OOD family 聚合执行，不是逐题剔除；因此也要求 public_accept 不得系统性偏易。这里冻结的是 task-owned 套卷设计，不把“毫米距离”等 Stair 特殊语义放进共享 acceptance。

每个 cell/seed 先对 episode 求均值，再在一个 stratum 内按 cell 等权，最后跨 paired seeds 汇总。Stair task-owned score 明确定义为：

`S_stratum = 0.5 × macro_success_rate + 0.5 × macro_mean_goal_progress_fraction`

其中 `goal_progress_fraction` 由冻结起点到目标的有向进度裁剪到 `[0,1]`。成功率保持任务完成语义，progress 在困难层提供连续信号；两者都由 Stair recorder 产生，共享 acceptance 只消费已命名的标量和 paired samples。

不再把五个 strata 无条件等权。先形成三个研究 family：

- `F_scalar = macro(height-OOD, tread-OOD, joint-OOD)`；
- `F_c1 = macro(phase/composition-OOD, C1 conditional frontier)`；
- `F_c2 = C2 mechanism-shift OOD`。

每个 family 的成员 strata 由 suite 的 `score_contract` 预先冻结，不能在看到 reference 后逐层挑题；每个 stratum 在三个 cohort 中均至少有 3 个 cell，并始终完整报告。official reference 只在 **family 聚合层**生成一次 floor-only mask：`S_family >= 0.10` 的 family 进入 `J_OOD`，低于 0.10 的 family 作为 reference 尚未建立的能力退出本 epoch 的优化声明。上界不再用于剔除：reference 已经做好的 ceiling family 必须继续进入 `J_OOD`，防止候选把既有能力破坏到零而不受惩罚。

family 等权表达研究 claim 的优先级，不是按样本量或方差加权。若 `F_scalar/F_c1/F_c2` 均通过资格，单层 `F_c2` 占 `J_OOD` 的 `1/3`、占 `J_primary` 的 `2/9`；若 mask 只留下更少 family，其机械权重会相应上升，Phase 4 必须连同 mask 一起报告。该取舍不通过事后降权修正；统计不确定性由固定 paired seeds、连续 progress 分量和 acceptance 的 paired delta 单独处理。

为避免“Phase 3B 先签 suite、Phase 4 才有 official reference”的循环，evaluation/acceptance hash 冻结的是 **floor-only 资格规则和完整 cell pool**，不是提前猜出的 family 清单。Phase 4 reference receipt 机械地产生唯一 `objective_family_mask_hash`；Genesis、candidate 的 task-owned evaluation evidence/selection、decision 与 replay 必须绑定该 hash，整个 epoch 不再改变。mask 不写入通用 `EvaluationBindingV1`：它是 reference 结果派生的评分投影，不改变已冻结的 rollout 协议，把它写入通用 binding 反而会令 reference 与 candidate 伪装成两套评测。由此接受一个明确取舍：接近饱和的 family 在本 epoch 内继续保留，局部灵敏度可能较低，但既有能力受目标/guard 保护；只能在下一协议 epoch 依据预注册规则重标定，不能在看到 candidate 后移除。若所有 OOD family 都低于 floor，系统必须产出具名 qualification failure 与各 family 测量后停在诊断，不能在 eval 深处抛异常，也不能启动 candidate。每个研究 claim 至少有一个对应的 reference-supported family，否则不得作该 claim。

不得把 cell/episode 全部池化，因为样本多的简单层会吞掉困难层。当前 3 episodes 使单 stratum/seed 的二元 success 分量较粗，但正式 `S_stratum` 还有等权的连续 progress，candidate/head 又按 seed 配对；sealed 一致性门消费聚合 `J_primary`，不是单个 C2 success rate。因此不为追求表面上的 0.05 粒度临时扩大套卷或改阈值。覆盖题数、episodes、seed 和墙钟上限已在 Phase 3R/3B 的 candidate 前 smoke 后冻结；预算变化必须开启新协议 epoch。

### 3.6 Candidate acceptance

OOD-only 会丢弃当前仍然很大的 ID 改进空间，而先做 ID epoch、再切 OOD epoch 会额外训练 reference 并割裂 LOOP。Core V1 因此采用一个低复杂度的固定加权目标：

`J_ID = 0.5 × S_C0-ID + 0.5 × S_C1-ID`

`J_primary = (2 × J_OOD + J_ID) / 3`

候选接受规则为：

- **优化目标**：相对 current-best 提升 `J_primary`，并超过冻结的 noise-aware 最小有效差；
- **Guard 1**：`J_OOD` aggregate 不退化；
- **Guard 2**：`S_C0-ID` 不退化；
- **Guard 3**：`S_C1-ID` 不退化；
- **硬有效性**：训练、评测、manifest、checkpoint、seed/budget/PPO、reward authority 与 sealed evidence 全部有效。

三个 guard 分别判定，禁止池化。Phase 3R 将 acceptance 升为最小的 `acceptance.protocol.v6`：共享 schema 只新增通用的 named-guards 列表和统一判定，不出现 C0/C1/C2、riser 或 tread；各 score 的含义、生成与命名由 Stair TaskPlugin 拥有。

新协议中的 `public_score`、current-best 排序和 objective delta 一律绑定 `J_primary`；`J_OOD` 与两个 ID 分数只保存在具名 guard vector 中，不再任选某一个 guard 冒充 public score。

每个 guard 复用当前 v5 已实现的有上限 paired-noise 规则，初值冻结为 `tolerance = min(0.05, 0.02 + 1.5 × paired_SE)`，并保留 task-owned absolute floor。历史 eval-v5 round 4 的 objective 提升而 guard 下降 0.025、paired SE 约 0.023，说明固定 0.02 会把统计噪声当退化；这条历史证据支持沿用 bounded noise-aware 规则，但不授权 candidate 后改参数。

candidate acceptance 需要观察到受控目标改善；但 Phase 工程 GO 只要求 LOOP 被真实执行和决策可复算。若固定预算内没有候选改善，记录 improvement_observed=false，不得结果驱动增加 generation、降低阈值或修改 OOD 套卷。

### 3.7 Reference 与 calibration

每个机器人 × Stair active protocol 只有一个 official reference：

- 使用该协议冻结的统一 D_train，从随机初始化训练；
- 使用形态正确、语义等价的官方 locomotion reward 作为 reference program；
- 在同一套 ID/OOD 题上评测，成为 candidate 比较锚。

reference readiness 使用统一的低门槛 ID learnability 结构：easy C0-ID 与 easy seen-C1-ID 必须分别达到 `macro success >= 0.05` 且 `macro goal_progress_fraction >= 0.50`，并完成所有绑定/运行时有效性门；旧 C1 `success >= 0.20` 的独立资格门退休。OOD 要求 finite、完整、非结构性退化 telemetry，并至少有一个 family 达到冻结的 0.10 reference floor；不要求每个 stratum 单独成功。低 OOD 表现可以是 LOOP 起点，不应再次触发“每种地形先单独资格化”。

历史 `stair_010` 的非单调低分是真实异常，但目前没有证据证明由 classifier 导致。Phase 3R 只做一次同 checkpoint、同 seeds 的 0.10/0.15/0.20 m 定向诊断，核对 geometry digest、目标/episode horizon、raw progress 与 terminal reasons；原因闭合前，`stair_010` 不进入 C0-ID guard 或 reference 支持性判断。该诊断不扩展为新的评测框架。

现有 C0 checkpoint 只作为 terrain feasibility 和迁移 calibration anchor。它在新协议上重新评测后可进入报告，但不能冒充 unified reference。

---

## 4. 源码与运行产物管理

### 4.1 建议目录边界

Stair TaskPlugin 内保留：

- terrain_regimes：C0、C1、C2 的生成器和 surface query；
- distributions：统一训练分布与 sampler；
- evaluation_suites：ID/OOD 分层、公开/密封套卷；
- runtime/evidence：任务原始量到任务语义的解码；
- reward_programs：官方 reference 与 LLM candidate ABI；
- acceptance：OOD objective、ID guards 和 sealed 判定。

RobotProfile、RobotTaskBinding 与 manifests 继续使用共享 registry。共享 AgentCore、runner 和 composition 只看通用协议 hash，不读取 C0/C1/C2、pattern、riser、tread 或 roughness。

旧 environment_variants 目录若只服务于独立 cell 生命周期，应在 Phase 3R 合并到上述任务目录；有真实 terrain 生成作用的文件保留，纯路由/复制层删除。

### 4.2 单一操作入口

操作者只选择：

- robot_task_cell_id；
- 受版本控制的唯一 cell_variant_id（协议 epoch）；
- 运行模式与已批准预算。

不得在命令行自由拼 robot、terrain、eval 或 acceptance。terrain regime 由训练分布/评测套卷中的样本事实决定。

### 4.3 证据状态

以下四级是 **证据成熟度**，与 catalog 的 draft/active/archived 生命周期字段相互独立：

1. declared：静态组合和 manifest 可解析；
2. runtime_qualified：真实 simulator、reward ABI、训练/评测路径和 checkpoint lineage 已验证；
3. loop_verified：至少一代 candidate 完成 source→train→eval→decision→ledger；
4. formal_evidence：预注册多代、paired seeds、sealed/final 报告完成。

cell_loop_ready 是派生资格：协议冻结、runtime_qualified、reference readiness、有效 ID/OOD telemetry 和 runner-owned 权限链全部成立。它不要求 reference 在所有 OOD 成功，也不等于 LLM 已改善策略。

---

## 5. 已完成 Phase 与历史处置

### Phase 0：最新基线与二维边界冻结（已 GO）

**保留目标**：冻结真实产品入口、上游版本、机器人资产/runtime facts、legacy Stair/Push 契约与已知 hardcode 债务。

**有效结论**：

- G1-37、G1-29、Go2 资产与运行事实已有来源交叉验证；
- 产品树身份由关键源码 hash/contract 验证，Git HEAD 只记录 provenance；
- legacy Stair 与 Push 的协议、评测、reward ABI 和运行入口已有冻结证据；
- 后续提交导致旧 snapshot 按设计失效时，必须由当前 Phase 迁移报告解释，不能把一次性 P0 verifier 当永久门。

### Phase 1：组合契约、共享角色词汇与两个 legacy G1 profiles（已 GO）

**保留目标**：建立 RobotProfile、RobotTaskBinding、ResolvedCellManifest、共享角色词汇与两个真实 legacy G1 profile。

**有效结论**：

- 机器人事实由 profile 外部来源校验，不依赖自指 hash；
- role target 可承担非互斥的多个职责；仅互斥角色组合被拒；
- Push 不声明未消费的 support_effector；
- TaskPlugin 通过角色词汇访问形态，不出现关节名/USD 路径；
- legacy V1 fallback 只能由受控 legacy catalog 触发；
- normalized/canonical evidence 免疫 checkout 行尾变化。

### Phase 2：Stair 任务语义与机器人形态解耦（已 GO）

**保留目标**：使 Stair reward/evidence/evaluation 消费任务逻辑量和 runtime projection，而不是 G1 具体刚体名。

**有效结论**：

- base_kinematics 与 scanner/contact 目标分离，G1 新旧 recorder 物理量一致；
- failure/evaluation 描述使用任务逻辑角色；
- projection identity 只由投影内容决定，不因 manifest 获取方式改变；
- profile 实际 body/joint 集合参与通用层所有权扫描；
- VNext reward schema/capability 进入 manifest 强制链；
- 通用 transition decoder 已不再从 `StairProtocolV1.policy_compatibility.robot_schema` 取 joint count。

**未完成边界**：现行 effective task hash 仍传递性包含 G1-37 的 `StairProtocolV2.complete_reward_runtime`，composition/evaluation materialization/legacy adapter 仍读取 V1 `policy_compatibility`。因此“StairProtocol 不再作为机器人 schema 第二来源”不能记作 Phase 2 已完成结论；该债务由 Phase 3R 的 task projection 迁移一次收敛。

### Phase 3A：C1 irregular 与首个 non-legacy cell（历史实现，已被 Phase 3R 取代）

**已实现且继续复用**：

- G1-37 v2 profile 的七个 profile-owned joint groups；
- 与 Isaac Lab 官方语义等价的 15 项 G1 reference reward；
- C1 不等高/不等深平面踏面生成器与 IrregularFlatStairSurfaceQuery；
- 非笛卡尔 ordered-cell suite、分层 summary、公开/密封投影；
- manifest、TaskSpec/EvaluationBinding/TrainingBindingV2、子进程和 checkpoint lineage 的强制链；
- C1 几何可行性、成本与旧错误 reward/课程的根因证据。

**不再继续的内容**：

- stair.g1_37dof.irregular_v1 作为独立活动 cell；
- 独立 C1 reference、Genesis、ledger、current-best、LOOP 与 fresh 196,608,000-step 资格化；
- is_irregular_cell 决定 launcher、prompt、selection、reference 或证据生命周期；
- 把旧 attempt1/attempt2 的失败归因于 C1 地形。

旧回执保持只读历史证据，不重算为新协议 PASS。Phase 3A 的当前处置见 [P3A 验收报告](P3A/PHASE3A_ACCEPTANCE_REPORT_ZH.md)；迁移工作由 [Phase 3R 修正计划](P3R/PHASE3R_CORRECTION_PLAN_ZH.md) 接管。

---

## 6. 后续 Phase 计划与严格验收

人日只计工程工作，不含 GPU 排队和实验墙钟。每个 Phase 只设置能证明边界与结果有效的必要门，不扩展为通用安全框架。

### Phase 3R：Stair 单格身份收敛、统一训练分布与 OOD 协议（已 GO）

**目标**：撤销地形的 cell 生命周期，把 P3A 可复用能力并入唯一 G1-37 × Stair protocol draft（Phase 3B 补齐 C2 后再激活）；在任何新 candidate 前完成二维 C1 支持、归一化课程、task/robot 协议投影分离、VNext 单一 ABI、具名多 guard acceptance 和历史迁移规则。

逐条审计判定与证据见 [Phase 3R 计划审计处置](P3R/PHASE3R_PLAN_AUDIT_RESOLUTION_ZH.md)、[开发日志](P3R/DEVELOPMENT_LOG_ZH.md)与[验收报告](P3R/PHASE3R_ACCEPTANCE_REPORT_ZH.md)。Phase 3R 的 GO 是工程与协议边界 GO；统一协议仍为 `draft`，不代表 reference 已资格化、LLM LOOP 已运行或策略性能已改善。

**交付**：

- 唯一 `cell_variant_id` 的 G1 Stair draft manifest 与机器化 draft/active/archived 门；
- 50/50 C0+C1 统一训练分布、二维 C1 generator、goal-progress curriculum 与 zero-C2 exposure contract；
- 不含 G1 schema 的 canonical Stair task projection，以及由 RobotProfile/RuntimeProjection 解析的 cell-local reward/IO hashes；
- VNext-only reward ABI、Stair TrainingBindingV2 与旧协议只读迁移边界；
- 具名 `acceptance.protocol.v6`、ID/OOD ordered suites、`J_primary`、三个 guards、reference readiness 与成本预算；
- terrain-regime 数据分派，删除全部 active-product `is_irregular_cell` 读取；
- P3A 历史 manifest/receipts 的迁移处置报告。

**严格验收**：

1. catalog 只解析一个 G1-37 × Stair draft protocol，正式 launcher 对 draft/archived fail-closed；C0/C1/C2 不再形成独立 Genesis、ledger、current-best 或 checkpoint lineage；
2. manifest 只持久化一个协议身份字段，并同时绑定 training distribution、evaluation suite、acceptance v6、reference gate、VNext ABI 与 task protocol；train/eval child 均强制比较；canonical task projection 不含 robot schema，也不包裹旧 V1/V2 protocol hashes，机器人事实只改变 cell-local projection/IO/reward hashes；
3. C1 generator 的 riser 与 tread 可独立取值，固定 seeds 证明 unified 10×10 grid、C1 10×9 支持、50/50 family allocation、5/1 初始 levels 和实际几何边界可复算；旧单 difficulty 映射及“训练覆盖 all-but-one height bin”的硬约束退出 active path；
4. task-owned curriculum 使用 goal progress 而不是 4 m raw displacement，C0/C1 初始 levels 为 5/1；不超过完整预算 5% 的 draft smoke 证明两类 terrain 有覆盖、课程可移动且无结构性停滞；
5. global scalar OOD、C1 conditional/composition OOD、C2 mechanism OOD 与 ID 的分层和 claim 完全匹配；旧 C1 18 题经新支持重新分类，区间内未采样点不伪称标量 OOD；
   phase OOD 必须以展开序列相对训练相位确实改变为准；三个 formal cohort 各 21 道、总计 63 道几何两两隔离，但仍只执行 1260 episodes；
6. score、family floor-only 资格、`J_ID/J_OOD/J_primary` 和三个 named guards 可独立复算；ceiling family 不被剔除，空 mask 形成诊断失败；v6 共享决策层不读取任何 Stair 术语，所有阈值在 candidate 前冻结；
7. unified Stair 只运行 VNext 与 TrainingBindingV2；active path 不读取 `StairProtocolV1.policy_compatibility` 或 V2 的固定 G1-37 schema，legacy V2 决策回放仍在旧冻结身份成立，旧 candidate 不可直接在新协议重执行；
8. `rg` 在 active `src/` 与 `scripts/` 中对 `is_irregular_cell` 的命中为 0；regime 只被 terrain generator/surface query/evidence decoder/task-owned summary/score 和 public display 消费，不参与生命周期路由；
9. `stair_010` 完成定向诊断或继续从 guard/资格化中隔离；不把未经证明的 classifier 猜测写成结论；
10. P3A joint groups、15-term reward、C1 generator/query、ordered summary 和 lineage 强制继续通过；P2 parity、旧 C0 receipt replay 与 G1-29 Push 回归不退化；
11. 不执行已废弃的独立 C1 完整资格训练；输出可复算验收器、迁移清单、成本上限和开发日志后才 GO。

### Phase 3B：C2 roughness OOD 与 failure-feedback 校准（已 GO）

**目标**：实现有界 C2 表面机制变化，并证明其 geometry、surface query、评测传感器物理信号和失败分类可靠；C2 只进入统一 Stair 协议的 OOD 套卷。Core V1 将其解释为训练未见的表面/接触机制迁移，不声称策略会利用毫米级 height-scan 信号。

实施边界、运行证据与逐门结论见 [实际实施计划](P3B/PHASE3B_IMPLEMENTATION_PLAN_ZH.md)、[开发日志](P3B/DEVELOPMENT_LOG_ZH.md)和[验收报告](P3B/PHASE3B_ACCEPTANCE_REPORT_ZH.md)。本 Phase 的 GO 只表示 C2 考卷、观测、反馈、成本和单格身份已闭合；不表示 reference 已资格化、LLM LOOP 已运行或策略性能已改善。

**交付**：

- C2 task-owned generator 与 runtime surface query；
- C2 public_dev/public_accept/sealed cells；
- roughness activation、scanner observability、clearance/penetration 分类漂移报告；
- 更新后的 unified evaluation_suite_hash。

**严格验收**：

1. max_abs_height_offset 的硬上限为 0.015 m；实际 levels 可在该上限内按观测与分类标定收紧，起点、riser/edge 拓扑、名义 tread 和顶端目标区不被重定义；
2. mesh oracle、surface query 与 evidence 在冻结采样点/容差内一致；
3. roughness 幅值与 insufficient_clearance、penetration 阈值的量级关系有显式标定，C1→C2 对照不会把表面起伏机械地误报成策略失败；
4. 关闭评测 observation corruption 后，height scanner 对目标空间尺度提供可测物理信号；该门只证明 mesh→sensor 链有效。训练期 height-scan 仍继承冻结的 `UniformNoiseCfg(±0.100 m)`，而本轮最大 C1/C2 ray 差约为 `4.52 mm`，所以不得把它表述为策略视觉显著性；C2 的有效性由 Phase 4 reference 行为资格化，`F_c2` 未达到 reference floor 时退出 objective 且不得作 C2 泛化 claim；
5. C2 不出现在训练 generator/sampler/课程或 PPO rollout 中，静态契约与真实训练 smoke 均报告 exposure=0；
6. C2 仅增加 terrain/eval/evidence 资产，不新增 manifest、reference、Genesis、ledger 或 LOOP 分支；
7. public_dev 与 sealed/final 信息边界通过，C0/C1 summary 与 legacy Push 不退化；
8. 套卷成本在 Phase 3R 冻结上限内；Phase 3B 签发含 C2 cell pool 与 family 资格规则的最终 suite/acceptance/manifest hashes 前，Phase 4 不得启动 reference 或 candidate 训练。

**完成结果**：统一协议已转为 `active`；训练 smoke 的 C0/C1 分配为 8/8、C2 暴露为 0；三个 formal cohort 各 24 道且几何互异，正式成本为 1080 episodes；9 个 C2 mesh 均为 watertight，mesh/query 最大误差为 `3.33e-16 m`；height scanner 有 120 条 ray 观测到至少 1 mm 的 C2 物理信号，但不据此声称策略显著感知；surface-relative failure 分类完全一致，固定 world-z 的 82 次边界翻转全部位于 roughness 幅值带内。prompt 的 strata、claim kind、目标和 guards 已改为从 active suite/profile 派生并由 P3B verifier 交叉验收。

### Phase 4：G1-37 × 统一 Stair reference 与 reward LOOP（8–12 人日 + GPU）

**目标**：在唯一 G1 Stair active protocol 上训练 official reference，并证明 LLM 只改 reward 时能够通过统一 ID/OOD 流程持续迭代。

**交付**：

- unified official reference checkpoint/receipt；
- 1-generation pilot；
- pilot 成本通过后的 fresh 3-generation validation；
- `J_primary`、三个 named guards、sealed decision、ledger/replay 与 lineage 报告。

**严格验收**：

1. reference 从随机初始化、在统一 D_train 和冻结预算内训练；官方 15 项 reward 的结构/数值等价证据仍有效；
2. reference 在 C0-ID/C1-ID 达到冻结 learnability 门，所有 OOD strata 有有限、可复算、非结构性退化 telemetry；不要求每层成功；
3. reference evaluation 按冻结的 floor-only 规则生成唯一 objective_family_mask_hash；至少一个 OOD family 达到 reference floor，ceiling family 保留；若全部低于 floor，产出诊断失败并停止。后续 Genesis、candidate task evaluation/selection、decision 与 replay 全部绑定非空 mask；通用 EvaluationBinding 保持同一冻结 rollout 身份；
4. 旧 C0 checkpoint 只以 calibration/transferred anchor 身份出现，不能进入 reference lineage；
5. candidate 只提交完整 reward program；terrain、distribution、curriculum、seed、budget、PPO、eval 和 acceptance 均与 reference 相同；
6. pilot 完整闭合 source→verify→train→eval→decision→ledger，成本未超上限后才授权 validation_3；
7. candidate acceptance 只由 `J_primary` 改善、J_OOD/C0-ID/C1-ID 三个 guards 和 sealed 有效性决定；
8. prompt/memory 只含 public_dev；public_accept 不作为下一代逐题提示，sealed/final 不泄漏；
9. checkpoint、manifest、binding、reward source/effective identity 和 runner artifacts 可从 ledger/replay 复算；
10. Phase GO 要求 loop_verified；improvement_observed 单独报告，未改善不得增加 generation 或改门。

**完成结果（2026-08-29）**：official 15-term reference 从随机初始化训练并通过 C0/C1 readiness；
`F_scalar/F_c1/F_c2` 全部进入冻结 mask。1 代 pilot 与 fresh 3 代 validation 均真实闭合；validation
outcomes 为 accepted / formal_rejected / no_selection，第 1 代候选通过 paired objective、三个 guards
与 sealed gate 后成为 current-best。`loop_verified=true`、`improvement_observed=true`；未增加代数，
未改变训练分布、课程、seed、PPO、评测或 acceptance。原始 ledger、cost、lineage、prompt 泄漏扫描与
GPU closure 均由 `PLAN/Variant_construction/P4/verify_phase4.py` 复核。

### Phase 5：Go2 机器人行与统一 Stair 资格化（8–11 人日 + GPU）

**目标**：只增加 Go2 机器人行的 profile、runtime adapter 和薄 binding，复用相同 Stair 任务语义、terrain regimes、分层方法与 LOOP。

**交付**：

- exact Go2 RobotProfile/RuntimeAdapter；
- Go2 × Stair binding 与单一 active protocol；
- faithful official Go2 rough reward reference；
- Go2 unified D_train、ID/OOD suite 与 runtime/reference receipts。

**严格验收**：

1. Go2 资产、body/joint、action/observation、PD/actuator、默认姿态、尺度和 sensor facts 从当前 Isaac Lab runtime 交叉验证；
2. Stair TaskPlugin、C0/C1/C2 generator、surface/evaluation/acceptance 不出现 Go2 关节名或 USD 路径；
3. Go2 binding 只声明任务需要的角色/组与少量尺度，thigh/calf 等接触目标按 profile 能力选择，不复制任务；
4. Go2 只有一个 Stair active protocol，不执行 clean→irregular→rough 三次顺序资格化；
5. official Go2 reward 按上游父配置与 overrides 得到完整 effective terms，并通过 source/数值等价检查；
6. 统一 D_train 保持与 G1 相同的任务级分层原则，物理尺度边界可由 Go2 profile/binding 合法解析，但不能更改评测语义；
7. G1/Go2 都保留一小组相同绝对几何 anchors 供跨机器人描述，其余 robot-scaled cells 只用于各自能力评测；不得把不同尺度套卷的分数直接宣称为机器人优劣；
8. reference 从随机初始化训练，ID learnability、OOD telemetry、lineage 和真实 runtime 全部有效；
9. G1 Stair 与 G1-29 Push 回归不退化；新增机器人只新增机器人 profile/runtime adapter、薄 binding/registry 项和该行的协议配置，可通过显式 component refs 复用统一 Stair loader；不得修改 Stair TaskPlugin、统一协议加载器或共享 LOOP/launcher。

### Phase 6：Go2 × 统一 Stair LOOP 与矩阵闭环（6–9 人日 + GPU）

**目标**：在 Go2 行上复用 Phase 4 的受控 reward LOOP，并证明机器人轴扩展没有复制系统。

**交付**：

- Go2 1-generation pilot 与 fresh validation_3；
- G1/Go2 跨机器人矩阵覆盖、成本和决策对照；
- 共用入口、不同 profile/binding/reference lineage 的证据。

**严格验收**：

1. Go2 通过同一 generation 入口、runner、verifier、evaluation 和 ledger；不存在 Go2 专用 loop/launcher；
2. LLM 权限与 Phase 4 相同，仅 reward program 可变；
3. Go2 candidate 使用自己的 current-best/reference lineage，不与 G1 checkpoint 或分数混用；
4. `J_primary` 与三个 named guards 的定义相同，数值阈值和 reference-supported family 按 Go2 active protocol 独立冻结；
5. pilot 成本通过后才进行固定 3-generation validation；能力失败不扩预算；
6. loop_verified、sealed decision、checkpoint checksum 和 replay 可复算；
7. improvement_observed 单独报告，不作为工程 GO 强制门。

### Phase 7：矩阵报告、final holdout 与 formal readiness（3–5 人日 + 条件式 GPU）

**目标**：给出不夸大 claim 的最终工程/研究证据，并把后续增加机器人行或任务列的工作量收敛为小适配层。

**交付**：

- 三个 Core 格子的 matrix coverage report；
- G1/Go2 Stair 的 ID、各 OOD strata、sealed/final 结果；
- Push legacy 不退化报告；
- 新机器人/新任务接入模板和实际改动面统计。

**严格验收**：

1. Phase 7 新增 task-owned `final_holdout` 配置；其几何与当前 72 道 formal pool 不重合，并在协议、候选和停止决策冻结后一次性运行，未进入 prompt/memory；当前 active P3B 配置中不存在该 cohort 不构成缺陷；
2. 报告分别标注 ID、单轴标量外推、双轴外推、已知 family 内组合泛化与 C2 mechanism shift；
3. public-dev 可见 OOD 与真正 hidden holdout 分开，不使用“完全未见”混称；
4. 三个 active protocols 的 manifest、training/eval/acceptance hashes、reference/candidate lineage 和决策可复算；
5. G1-37/Go2 Stair 达到 loop_verified，G1-29 Push 保持 legacy parity；
6. 增加机器人行不修改 TaskPlugin/LOOP，增加任务列不修改 RobotProfile/AgentCore；若未满足则不得宣称矩阵扩展完成；
7. 不因未观察到改善扩展 generation；结果报告同时给出 improvement_observed 与不确定性。

---

## 7. Core V1 后的受控扩展

### Extension A：Go2 × Push Recovery

这是“新增机器人行可跨任务复用”的强证明。复用 Go2 RobotProfile/RuntimeAdapter，只新增 Push-owned capability 要求与薄 binding。Push 不得为了 Go2 引入逐足支撑等其不消费的语义。

### Extension B：六足机器人行

先完成 asset/control/contact/observation feasibility，再接入统一 Stair active protocol。六足只扩展 profile/adapter/binding 和必要的任务角色基数，不复制 C0/C1/C2、LOOP 或评测系统。

### Extension C：受限 LLM curriculum capability

只有 reward-only Core V1 完成后才单独授权。candidate 可提交版本化的训练分布 delta，但总预算、可选参数范围、evaluation suite、acceptance、seed 和 runner 仍由系统控制；旧 reward-only 结果与新权限结果分 protocol epoch 报告。

### Extension D：整类 pattern 留出研究

把一个预注册 pattern family 完全排除于训练，并只对该 family 声明类别迁移。它是研究消融，不是 Core 生产门，也不用于证明任意未知类别泛化。

---

## 8. 研究与工程依据

| 依据 | 采用的思想 | 本计划中的约束 |
|---|---|---|
| [Isaac Lab v2.3.0 Terrain Generator](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.terrains.html) | terrain config、seed、random/curriculum 由环境侧统一管理 | 分布由 Stair TaskPlugin/manifest 冻结，LLM 不可修改 |
| [Isaac Lab v2.3.0 TerrainImporter source](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/terrains/terrain_importer.py) | 上游持有 per-env terrain_types/terrain_levels，但初始 max 是全局值、env 默认按 columns 分配 | Stair 只做一次 task-owned seeded buffer 初始化，以最小改动实现 50/50 regime 与 5/1 初始 levels |
| [Isaac Lab v2.3.0 terrain_levels_vel source](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py) | 上游晋级量按相对 terrain origin 的绝对位移与 terrain 长度判断 | Stair 的不同 tread 会让同一 4 m 位移代表不同完成比例，因此只复用 level update 机制，不复用 raw-distance 任务语义 |
| [Isaac Lab v2.3.0 G1/Go2 locomotion configs](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config) | robot-specific asset/action/sensor/reward overrides 位于配置边界 | reference reward 忠实解析 effective terms，不把关节组压平 |
| [Cobbe et al., Quantifying Generalization in Reinforcement Learning](https://proceedings.mlr.press/v97/cobbe19a.html) | 训练与测试环境集合分离，报告 held-out 泛化 | 明确 D_train、OOD strata 与 hidden final |
| [Procgen Benchmark](https://proceedings.mlr.press/v119/cobbe20a.html) | 分布覆盖与 held-out level 能区分记忆和泛化 | exact unseen、插值和 OOD 不混称 |
| [NIST Covering Arrays](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software) | 小规模组合覆盖代替完整笛卡尔积 | ordered suites 覆盖主要二元交互并限制成本 |
| [Agarwal et al., Deep RL at the Edge of the Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | 多 seed、paired 对照和不确定性比单点分数可靠 | 固定 paired seeds、macro strata 和非劣 guards |

采用这些思想不意味着引入新框架。Core V1 只实现当前 Stair/Go2 所需的最小结构：静态 catalog、不可变 manifest、task-owned distribution/suite 和现有 runner 的证据闭包。

---

## 9. 总体验收与停止规则

### 9.1 Core V1 工程 GO

只有以下条件全部满足才能 GO：

- Phase 0、1、2 的既有能力保持有效；
- Phase 3R、3B、4、5、6 全部通过各自严格验收；
- G1-37 与 Go2 各只有一个 Stair active protocol，且均达到 loop_verified；
- C2 真实激活、分类可靠、训练 exposure=0，但不存在 C2 独立 cell/reference/campaign；
- G1-29 × Push legacy 行为不退化；
- matrix report、ledger/replay、checkpoint lineage 与 final claim 在 Phase 7 闭合；
- production 只有一套 AgentCore、runner、evaluation/decision 主线。

### 9.2 返回上一 Phase

- 发现机器人关节名/USD 路径进入 TaskPlugin：返回 profile/binding 边界；
- 发现 Stair 几何/失败分类进入共享层：返回 TaskPlugin 边界；
- terrain regime 再次决定 Genesis/reference/ledger/current-best：返回 Phase 3R；
- C2 出现在训练 rollout：返回 Phase 3R/3B distribution contract；
- failure classifier 被 roughness 量级污染：返回 Phase 3B；
- reference reward 与官方 effective terms 不等价：返回对应机器人 profile/binding；
- `J_primary`、family 资格规则/mask 或任一 named guard 在 candidate 结果后改变：该协议 epoch 作废并重新冻结；
- checkpoint、manifest、binding 或 source identity 不闭合：不得进入 decision。

### 9.3 结果型停止

以下情况不视为工程失败：

- 固定预算内没有 candidate 被接受；
- improvement_observed=false；
- 某一 OOD stratum 表现低，但 telemetry 有效且协议未被破坏。

这些结果必须原样记录。禁止为了得到正结果临时增加 generation、降低 guard、扩大训练支持、泄漏 sealed/final 或挑选更有利的 seeds。

以下情况应停止扩大投入并回到诊断：

- reference 连 ID learnability 都无法达到；
- OOD telemetry 因地形、decoder、sensor 或分类错误而结构性退化；
- 单候选成本超过预注册上限；
- 新机器人必须修改共享 LOOP 或复制 TaskPlugin 才能运行。

### 9.4 成功后的新增规范

完成 Core V1 后，新增机器人行只允许提交 RobotProfile、RuntimeAdapter、少量 RobotTaskBinding 和该机器人 reference；新增任务列只允许提交 TaskPlugin、任务分布/评测/acceptance 和薄 bindings。任何要求复制 runner、ledger、LOOP 或按 terrain 建 cell 的方案，默认视为架构回退，必须重新论证。

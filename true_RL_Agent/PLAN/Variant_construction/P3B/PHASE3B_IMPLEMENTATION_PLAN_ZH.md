# Phase 3B 实际实施计划：C2 roughness OOD 与 failure-feedback 校准

> 实施状态：**已 GO（2026-08-28）**
> 权威证据：[验收报告](PHASE3B_ACCEPTANCE_REPORT_ZH.md)、[运行回执](PHASE3B_RUNTIME_RECEIPT.json)、[训练排除回执](PHASE3B_TRAINING_EXCLUSION_RUNTIME_RECEIPT.json)、[可复算快照](PHASE3B_CONTRACT_SNAPSHOT.json)

## 1. 本 Phase 的完成定义

Phase 3B 不新增地形轴、cell、训练分支或 LOOP 生命周期。它只在现有
`stair.g1_37dof.unified_v1` 单格内，把 C2 作为 Stair 任务拥有的、训练时从未出现的
表面机制 OOD，接入同一套评测、证据和提示反馈链。机器人形态、奖励候选权限和共享
runner 均不因 C2 改变。

完成后必须同时成立：

1. C2 是真实碰撞网格，不是只在标签、query 或 prompt 中存在的伪地形；
2. 网格、任务侧 surface query、评测 cell 身份三者描述同一个表面；
3. height scanner 在关闭评测 corruption 的真实 Isaac Lab 场景中能测到 C1/C2 物理差异；该证据不等同于策略会利用该信号；
4. `insufficient_clearance`、fall surface fallback 与 success clearance 使用 C2 局部表面，
   且不会因为换了 query 就机械地产生失败分类漂移；
5. C2 只存在于评测套卷，训练 generator、sampler、curriculum 和 PPO rollout 的暴露量均为 0；
6. public_dev、public_accept、sealed_validation 各有 3 道 C2 异题，难度阶梯等价；
7. 正式套卷成本不超过 P3R 的 1260 episode 上限；
8. 统一协议在真实运行时证据闭合后由 `draft` 升为 `active`，不另建 C2 manifest、reference 或 ledger。

## 2. 冻结边界

### 2.1 系统拥有且本 Phase 可改

- Stair 任务内的 C2 表面描述、网格生成、surface query、评测套卷和失败反馈聚合；
- 统一评测 suite / task protocol 的身份哈希；
- 为保持物理成本上限而调整同一 suite 的 episode 重复数；
- P3B 的验证脚本、回执和开发日志。

### 2.2 明确不改

- LLM 权限仍仅为完整 reward program；
- C0/C1 训练列、50/50 分布、课程、seed、PPO 超参、训练预算；
- RobotProfile、角色词汇、矩阵公用层、Push 插件；
- acceptance 阈值和 failure taxonomy 的语义阈值；
- 新建 terrain cell、C2 专属 Genesis、C2 专属 LOOP 或额外资格化流程。

### 2.3 所有权原则

C2 的 `surface_profile`、插值、mesh 和 failure-feedback 都位于
`tasks/stair_climbing`。公用矩阵层只继续看到一个 Stair task protocol hash，禁止出现
roughness、踏面、scanner 或 clearance 语义。

## 3. 冻结的 C2 设计

### 3.1 表面机制

- C2 复用 C1 的不等高、不等深台阶骨架；start platform、riser 边、每级 nominal tread、
  top platform 和 progress goal 不变。
- 每块踏面内部使用确定性的有界分片线性高度场；四周边界 offset 为 0，避免改变 riser
  高度和相邻拓扑。
- 表面由少量、可哈希的 profile 参数定义，mesh 顶点与 query 使用同一冻结三角剖分规则。
- 幅值阶梯冻结为 0.004 / 0.006 / 0.008 m，硬上限仍为 0.015 m；任何校准只允许向下
  收紧，不得为了制造差异向上放大。
- 空间尺度不得细于 height scanner 的 0.1 m 采样能力；本版使用固定的纵向/横向采样网格
  和低频 profile，不引入随机噪声纹理。

### 3.2 套卷与成本

- 新增 `c2_mechanism_ood`，每 cohort 3 个 cell；三个 cohort 分别使用不同 C1 骨架/相位，
  但都恰好包含一次 4/6/8 mm 幅值和同一组频率档位。
- 三个 formal cohort 仍保证 geometry digest 两两不交叠；public_dev 只暴露自己的 cell 和结果，
  sealed 几何不得进入 prompt。
- strata 从 7 增至 8；每 cell 的重复数从 4 降至 3。因此每 seed 从 84 降至 72 episodes，
  全部 15 个 formal seeds 共 1080 episodes，低于 P3R 的 1260 上限。
- 保留每 stratum 3 道独立题，不用降低最小题数、复用几何或引入可变 episode 权重。

## 4. 实施工作包

### W1：身份与表面模型

目标：让 flat/C2 表面都进入 ordered geometry 身份，且 C0/C1 行为不变。

产物：task-owned `surface_profile` 契约、C2 分片线性采样函数、geometry schema 升版、
全套 cell/hash 重算。

验收：字段缺失、越界幅值、非法采样网格、digest 篡改均被拒；flat query 的既有边界测试通过。

### W2：真实 mesh 与 surface query

目标：生成有符号微起伏的闭合 C2 碰撞 mesh，并由 decoder 自动选择对应 query。

产物：C2 terrain cfg/mesh、C2 surface query、统一的 Stair task-owned query factory。

验收：冻结采样点上 mesh oracle 与 query 最大误差不超过 1e-5 m；offset 绝对值不超过
profile 幅值；所有 tread 边界 offset 为 0；start/top 平台和 nominal goal 不变。

### W3：统一套卷与反馈

目标：把 C2 纳入同一 objective family，并给 LLM 精确但仅限 public_dev 的失败反馈；prompt 的 strata、claim kind、目标和 guards 必须从 active suite/profile 派生，不能再手抄一份。

产物：8-strata suite、F_c2 评分、按 stratum 聚合的 primary failure breakdown、最终 suite/protocol hash。

验收：三个 cohort 的 C2 幅值与频率多重集相同、geometry 全异；1080 formal episodes；
prompt 中可见 public_dev 的 C2 stratum 分数与失败分布，但不可见 public_accept/sealed 题面；P3B verifier 逐字段比较 prompt 投影与 suite/profile，删除 C2 stratum 或把它标成 future 时必须失败。

### W4：分类与可观测性校准

目标：证明 C2 是真实的表面/接触机制变化，而非证据投影错误。scanner 门只验证 mesh→sensor 物理链，不证明训练后的策略会注意毫米级信号。

产物：paired C1/C2 分类漂移报告、真实 Isaac scanner/mesh smoke 回执。

验收：

- 对保持相同“相对局部表面 clearance”的成对轨迹，C1/C2 的 primary reason 与 diagnostic flags
  必须完全一致；
- 对保持 world-z 不变的边界轨迹，只允许距离 0.03 m clearance 阈值不超过 0.008 m 的样本
  发生状态变化，并如实记录，不修改阈值；
- 真实 eval（关闭 observation corruption）中，固定机器人姿态下 C1/C2 height scan 至少 8 条 ray
  的差异达到 0.001 m，最大差异至少 0.002 m；
- 训练期 height-scan 噪声为 `UniformNoiseCfg(±0.100 m)`，相对实测最大 ray 差 `4.52 mm`
  不具备足以声称策略显著感知的信噪比；Core V1 的 C2 claim 因而限定为 surface/contact
  mechanism shift，是否达到 reference 支持地板完全交给 Phase 4 family-level 资格化；
- C2 mesh 在真实场景加载，transition decoder 能输出有限、surface-relative 的 clearance。

### W5：回归、冻结与清理

目标：以新主线替代 P3R draft，不保留第二套 C2 生命周期或开发偏脚手架。

产物：P3B verifier、自测、contract snapshot、runtime receipt、acceptance report、development log。

验收：目标 unit/integration suite、完整非 GPU 测试、真实 Isaac smoke、P3B verifier 均通过；
C0/C1 和 Push 回归通过；静态与运行时训练暴露均为 0；协议 `active` 后哈希闭合。

## 5. 严格 GO / NO-GO

只有 W1-W5 全部通过才可将 Phase 3B 标为 GO。以下任一情况均为 NO-GO：

- C2 进入训练路径，或产生第二个 Stair cell / terrain lifecycle；
- mesh 与 query 的表面不一致；
- C2 在关闭评测 corruption 的 scanner 物理链上达不到预注册可测门；
- 分类漂移只能通过放宽 task 阈值消除；
- public/sealed 几何或反馈边界泄漏；
- formal episode 成本超过 1260；
- 统一协议仍为 draft，或 active 身份链不能从配置重新计算。

`improvement_observed` 不属于本 Phase 的 GO 门；策略是否在 C2 上达到 reference 支持地板、是否提升由 Phase 4
的 reference 和 LLM LOOP 如实测量。若 `F_c2` 贴地板、饱和或不稳定而未通过冻结资格规则，它退出
`J_OOD` 且本 epoch 不作 C2 泛化 claim，但这不撤销 mesh/query/failure-feedback 的工程 GO。

## 6. 完工后审计处置

- prompt 二源是真实缺陷：旧 payload 少 `c2_mechanism_ood`，同时把已启用 C2 写成 future。现已由
  public_dev suite cells 派生 strata/claim kinds，由 score contract 派生 objective/guards，由 training
  profile 派生 C2 exposure；任务 prompt schema 升为 v4，public-eval prompt schema 升为 v2。
- GPU runtime receipt 原先错误地把纯 prompt 与 post-rollout scoring 文件纳入 mesh/scanner 证据
  闭包。两类文件均由静态 contract/snapshot 验收，运行回执只绑定真正影响 runtime probe 的 11 个
  源文件，并在 RTX 5090 上按同一 seed 重跑。
- `F_c2` 单 stratum 且 family 等权是真实取舍，不是实现错误。三 family 均合格时它占 `J_primary`
  的 `2/9`；Phase 4 必须报告最终 mask 与 paired uncertainty，不按候选结果事后降权。
- 3 episodes 使单 seed 的 success 统计较粗，但 `S_stratum` 还包含连续 progress，正式比较按 seed
  配对，sealed 门消费聚合目标；因此不突破 1260 成本上限，也不在看到结果前改阈值。
- 三套 C2 使用相同 surface-profile 难度多重集是有意的考卷对齐。本阶段只 claim “flat 训练→rough
  表面机制迁移”，不 claim 对未见 roughness generator/profile 的泛化。

## 7. 完工摘要

W1-W5 已全部通过。最终实现保持一个 `stair.g1_37dof.unified_v1` active cell；C2 只新增 Stair
任务内的表面、评测与反馈资产，没有形成第三根轴。正式套卷为 8 strata × 3 cells × 3 cohorts，
每 cell/seed 3 episodes，总成本 1080；训练仍为 C0/C1 50:50，真实 128-step smoke 证明 C2
rollout 暴露为 0。真实 Isaac 回执证明 9 个 C2 mesh 全部 watertight、scanner 物理链可测、decoder
有限且 failure-feedback 漂移受幅值约束；不据此声称策略会利用 height scan。最终协议、suite、
manifest、回执、prompt 投影和产品源码闭包由 `verify_phase3b.py` 机械复算。

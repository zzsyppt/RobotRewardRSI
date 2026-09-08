# Phase 5 开发日志：Go2 机器人行与统一 Stair 资格化

> 日期：2026-08-29～2026-08-30
>
> 最终状态：`GO / cell_loop_ready=true`
>
> 约束：没有新建 Go2 专用系统；最终重评复用既有 checkpoint，训练与 LLM/API 调用均为 0。

## 1. 机器人行与任务边界

Phase 5 只在既有“机器人 × 任务”矩阵中增加 Go2 × Stair 行：

- Go2 joint/body、默认姿态、执行器、资产闭包、形态尺度与角色候选归 RobotProfile；
- Isaac Lab env/runner、PPO 有效配置和上游源码闭包归机器人侧 RuntimeAdapter；
- C0/C1/C2、训练/OOD 划分、成功/失败、score、guards 与 readiness 归 Stair；
- RobotTaskBinding 只选角色、kinematic-base 来源和必要尺度；
- 通用 runner、Genesis、ledger 与 LOOP 不读取 Go2 或 roughness；
- LLM 权限仍仅为提交完整 reward program。

Go2 复用 Rough observation/action/PPO 与 Flat reward raw functions，显式声明
`track_lin_vel_xy_exp 1.5→3.5`、`lin_vel_z_l2 -2.0→0.0` 两项 Stair-context adaptation；
authorship 为 `human_authored`，不伪称原封不动的 official reward。`3.5` 是本阶段人工选取的经验权重，
没有唯一的“command-distribution normalization”推导；冻结 source 中的历史 rationale 措辞不作为方法学依据。
因其字节已进入 checkpoint/source lineage，本阶段不为改一句说明伪造新的训练身份；后续新 reward 版本必须使用
上述诚实表述。

## 2. 工程收敛

本阶段完成的底层收敛包括：

1. P0 Go2 真 Isaac probe 落为 exact profile；profile 不以自身 hash 自证；
2. G1/Go2 env cfg、runner cfg、agent configuration 与源码身份由机器人侧 locomotion runtime spec 统一解析；
3. 显式 row catalog + 每行薄 binding 取代从 G1 legacy 临时改写 binding 的开发路径；
4. Stair env/execution 根据 resolved adapter 实例化，不含 `if Go2`、Go2 关节名或 USD 路径；
5. VNext 为 Go2 增加形态中立的 `support_last_air_time` 与 `support_first_contact`，G1 ABI 不变；
6. prompt 从真实 reward-input contract 派生，Go2 不再收到二足左右脚或 legacy contact 字段；
7. production generation CLI 已能显式选择 cell 行，内部与 CLI 不再一边支持、一边固定 G1；
8. Stair/Push 的 RSL-RL agent configuration 均从机器人侧统一解析；
9. 验收器检查稳定身份与产品契约，不再用 row catalog 原始字节阻止未来增加机器人行。

## 3. 分布与真实 runtime 证据

- D_train：C0/C1 各 50%，C2 零暴露；
- Go2 垂直尺度 `0.589087`，水平尺度 `1.0`；
- 三个 formal cohort 各 24 题，共 72 个互异几何、1080 episodes；
- 三卷逐 stratum 难度阶梯对齐；G1/Go2 只共享 C0-ID 绝对 anchor；
- 72 个真 env × 40 control steps 共 2880 样本，reward raw/weighted/total 最大差为 0；
- 真 schema：12 actions、235 policy observations、187 height rays、19 bodies、4 supports；
- 128-step PPO smoke、C0/C1 assignment、reward activation、checkpoint 与预算记账通过；
- 上游 Rough checkpoint 的独立课程探针触发 C0/C1 升降级，证明课程状态机可工作；该探针不冒充 reference 成绩。

## 4. 训练与两类历史失败

必须区分两个 checkpoint：

- 早期 `sha256:2725…` baseline 在起始平台即失稳，属于真实训练/奖励能力不足；它的 NO-GO 结论保留为历史诊断，不参与最终 reference lineage；
- 后续派生 reference `sha256:fca6…` 完成 2000 iterations / 196,608,000 env steps，实际能到顶，但旧评测要求到顶后继续稳定停留，因此把已完成攀登反向记成失败。

同一 `fca6…` checkpoint、冻结 V1 协议下完成了一次 72-episode `public_dev` 反事实运行：宏成功率为 0，
task-owned state machine 的 primary outcomes 为 `fall=22 / forbidden_contact=50`，视频导出器没有重分类。
这份全量汇总与 checkpoint/protocol/trajectory/video hash 已进入 `verify_phase5.py`，不再只是可视化旁证。
其中被可视化跟踪的 3 个 C0 episode 均为 `progress_fraction=1.0`，primary outcome 却是
fall/forbidden-contact；视频由精确 task-owned pre-reset Isaac trajectory 离线渲染，没有重算物理。
72-episode 运行本身不是 acceptance score，也不声称所有 episode 都到顶；3 条完整轨迹用于证明旧规则确实能
反向否决已完成攀登。正式 GO 仍只由 V2 三相 runner receipt 决定。

用户目标只关注能否完成上楼，不要求到顶后在指定位置停下。对 `fca6…` 来说，能力低分的直接 blocker 因而是
评测 completion semantics，而不是训练、接线、PPO、课程或地形。无需再次训练。

## 5. 从源头修正评测语义

任务成功语义收敛为：

- `completion_mode=first_valid_top_arrival_latched`；
- 首次满足 progress、base/top bounds、orientation 与真实 support-on-top（含接触表面容差）即关闭 outcome；
- 只保留到顶前 `forbidden_contact`、`off_course` 阻断 latch；
- 到顶后的 fall/slip/off-course 不再抹掉已经完成的攀登；
- `top_platform_stability_requirement` 从 active completion contract 删除。

该变化同时进入 episode evidence、ordered summary、prompt 描述、protocol validation 与测试，不是在 score 聚合后
改标签。G1/Go2 都使用相同 V2 任务语义。

### 5.1 删除伪 `fall` 的依据

V2 一度把旧 `fall` 改名为 `low_base_clearance` 并降级为诊断，但继续保留它仍不合理：该量只是运动学 base
相对“最低可行支撑面”的高度，不能证明刚体倒地或任务失败。Go2 在旧 `0.265 m` 阈值下，public_accept
有 `353/360` 个 episode 触发；将阈值按视频轨迹改为 `0.15 m` 后，同一 checkpoint、同一套卷只剩
`1/360` 个触发。标签变化主要由人工阈值决定，而非能力变化；它还可与成功同时出现，并会把本已由“未到顶”
确定的失败包装成误导 LLM 的伪原因。

因此 active V2 从 signal、diagnostic flags、failure priority、primary reasons 和 prompt 中完整删除
`low_base_clearance`。同源的 `minimum_kinematic_base_clearance_above_top_m` 也从 active success contract 删除；
到顶由进度/平台边界、姿态和真实顶层支撑共同证明。支撑面解析所需的 penetration tolerance 迁入
`support_surface_projection`。V1 文件与历史回执不改，旧 `fall` 路径仅用于冻结协议回放。
first-arrival V2 validator 会显式拒绝任何重新加入的 base 净离地字段；源码中仍出现的旧字段只服务 inactive V1
历史重放，不是 active V2 的可选开关。

## 6. 训练 identity 与评测 identity 分离

直接把 V2 当新训练 cell 会错误要求重训。为此新增唯一、显式的 evaluation-only successor lineage：

- V1 训练 cell 继续拥有原 training profile、seed、PPO、reward、runtime dynamics 与 TrainingBinding；
- V2 拥有新的 task/manifest/EvaluationBinding 身份；
- runner 只接受当前 manifest 或协议冻结的唯一 predecessor manifest；
- train source preflight 可识别该 predecessor，eval source preflight 仍必须是当前 V2；
- checkpoint/source/capability/training binding 仍逐项精确匹配。

这不是放宽兼容门，而是把“训练来源”和“当前评测语义”两种不同事实分别建模。G1 同样使用该机制，未引入
Go2 特例。

V1/V2 的重复字段在此处是有意的不可变快照，而不是继续维护的两份真相：row catalog 已将两条 V1 行设为
`active=False`，并钉住各自 protocol/binding hash；产品加载与 `test_archived_stair_training_predecessors_are_identity_frozen`
都会拒绝漂移。今后的变更只能进入新版本。此边界下再引入继承/overlay 会增加隐藏合并语义，收益不足，故不做。

## 7. 不重训重评与资格化

Go2 复用 checkpoint：

`sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`

三相重评结果：

| robot / cohort | episodes | J_primary | J_ID | J_OOD |
| --- | ---: | ---: | ---: | ---: |
| Go2 / public_dev | 216 | 0.934438 | 1.000000 | 0.901658 |
| Go2 / public_accept | 360 | 0.933915 | 1.000000 | 0.900873 |
| Go2 / sealed_validation | 504 | 0.932641 | 1.000000 | 0.898962 |
| G1 / public_dev | 216 | 0.686291 | 0.699153 | 0.679860 |
| G1 / public_accept | 360 | 0.694165 | 0.740130 | 0.671183 |
| G1 / sealed_validation | 504 | 0.719514 | 0.787743 | 0.685399 |

Go2 public_accept 的 `F_scalar/F_c1/F_c2=0.702619/1.0/1.0`，G1 为
`0.714160/0.678748/0.620641`；两行三个 OOD family 全部 qualified。Go2 public→sealed 只下降
`0.001274`，G1 sealed 比 public_accept 高 `0.025349`。任务自有 qualification 均返回 `pass`，Go2/G1
requalification receipt 分别为 `a9e11c09…23fe99ae` 与 `4f195716…1365c0db`。

本轮 `training_executed=false`、`training_reused=true`、provider call=0。没有改 reward、PPO、训练地形、课程、
seed、预算、评测题目或门槛。

## 8. 标准 Genesis 与回归

两行三相 receipt 经产品正式 `prepare_session_genesis` 生成：

- Go2 Genesis/artifact SHA-256：`d2000f41…fad5a820` / `ef21cee5…4190e78e`，Genesis hash
  `db0d7a17…bfdb50b6`；
- G1 Genesis/artifact SHA-256：`53891211…e08ad8a` / `b9ad0029…7a3cd9a`，Genesis hash
  `080154a8…4be41fcc`。

随后使用产品 `load-only` 入口逐行复核：status=pass，Genesis checksum/mtime 前后不变，runner/provider=0，
ledger 不存在。successor import 也明确记录训练与评测执行均为 0。P5 唯一验收器对两行重新聚合 raw phase
receipt、重算 mask/qualification，并交叉核对 Genesis；不信任 GO 回执自报。

最终回归：

- P5 + Phase5 contract 定向回归 14/14 通过；
- 本地 Stair 回归 496 tests 通过、20 skipped；
- 服务器正常 Torch/Isaac 环境全量 1567 tests 通过、3 skipped，997.823s；
- 两行各 1080 episodes 的资格化与标准 Genesis/load-only 均通过。

因此 Phase 5 终态为 `GO / cell_loop_ready=true`。Phase 6 尚未执行。

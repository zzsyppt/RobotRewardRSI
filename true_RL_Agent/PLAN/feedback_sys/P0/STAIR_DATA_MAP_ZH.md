# Phase 0：Stair 数据地图与双策略行为基线

更新：2026-09-23，按第四次审计原件补核。首个开发域为 Go2 Stair v3。现在已有 P3 reference 与 formal Gen8-A 的同 v3 public-dev facts，可做双策略描述差分；时序差分留给 Phase 2。

**统一边界。** 本页区分源码能力、已保存实测与未来采集。回合级 ever 标志只表示出现过，最大归一化进度只表示达到过的距离；两者分别缺少事件时间和碰撞位置。刚体参考点与接触合力不提供脚尖、接触点或碰撞面。当前双策略资料来自不同源码/运行环境版本，且不是历史 parent→child；用于开发画像，不据此归因 reward 或反馈升级收益。分析使用 public-dev 数值与谱系身份，作者输入仍经 task 公共投影。本次修订仅离线处理，没有新增 GPU、训练、回放或 provider 调用。

## 1. 可复核材料与运行结果

- 新增 [stair_behavior_baseline.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_behavior_baseline.py) 与 [结果 JSON](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_behavior_baseline.json)：逐项检查 2×864 条真实 facts，重算全部 primary、全部诊断、cell/stratum 结构、闭合步与最大进度分布，并记录两策略身份和实际祖先。当前 summary validator、逐 cell 结局/标志计数闭合、checkpoint 训练回执绑定均 PASS。
- 复核命令：`.venv/Scripts/python.exe PLAN/feedback_sys/P0/stair_behavior_baseline.py`。输出 `policy_count=2`、`real_episode_facts_checked=1728`、`historical_parent_child=false`。
- 原 P0 字段/ABI解析见 [stair_current_identity.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_current_identity.json)、[源码快照](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_source_snapshot.json)、[原审计脚本](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_p0_audit.py)。快照 HEAD 为 `5cef484dfa5319130faa02d7e9bbcdd9e9b4a84e`，完整依据包含当时 dirty 文件 hash。
- 原 78 项 CPU unittest 全通过、0 skip、4.887 s；精确命令及覆盖见 [receipt](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_cpu_regression_receipt.json) 与 [日志](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_cpu_regression.log)。覆盖 pre-reset/终止掩码、primary 与诊断、first-arrival、staging、形态映射、summary 重算、公共投影；这是原合同回归。本次只改离线分析和文档，未把旧测试记为新采集验收。
- 第四次审计的原件实际在 P0 上一级：[p0_audit_checks.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/p0_audit_checks.py)、[JSON](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/p0_audit_checks.json)、[Gen8-A replay 原件目录中的 summary](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/audit4_replay/gen008a_v3/public_dev_summary.json)。

## 2. 当前协议、机器人与作者接口

| 项目 | Go2 v3 | G1 v3 |
|---|---|---|
| variant | `stair.go2_12dof.unified_v3` | `stair.g1_37dof.unified_v3` |
| robot profile | `unitree_go2_12dof_isaaclab_v1` | `unitree_g1_37dof_isaaclab_v2` |
| binding | `stair_go2_12dof_unified_binding_v3` | `stair_g1_37dof_unified_binding_v3` |
| base / forbidden_contact 角色 | `base` / `base` | `torso_link` / `torso_link` |
| kinematic base 来源 | articulation root | articulation root |
| support 顺序 | FL_foot、FR_foot、RL_foot、RR_foot | left_ankle_roll_link、right_ankle_roll_link |
| joints / actions / reward 字段 | 12 / 12 / 39 | 37 / 37 / 45 |
| force history / height rays | 3 / 187 | 3 / 187 |
| 低净空阈值 | 0.018 m | 0.03 m |

完整 schema、shape/unit 和各项 hash 存在 `stair_current_identity.json`。角色与 ABI 来源：[robot_binding.py:136](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/robot_binding.py:136)、[runtime_projection.py:284](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/runtime_projection.py:284)。G1 的运动学 root 与禁触 torso 是两个不同观测对象。

v3 成功为 `first_valid_top_arrival_latched`：首次合法到顶即闭合 outcome，检查位置、支撑范围、向上力、姿态和阻断条件。primary 互斥优先级为 `forbidden_contact → off_course → base_orientation_instability → slip → insufficient_clearance → timeout`。诊断标志重叠且保留已恢复症状。`episode_steps` 是 outcome 闭合步，成功可早于环境 done，horizon 也可合成 timeout。

orientation 规则为 abs roll/pitch >0.65 rad 连续10步；slip 为接触且平面速度 >0.5 m/s 连续5步，接触阈值1 N；低净空为非接触 support clearance 低于上表阈值连续3步。旧 slip/clearance 先跨脚 `any` 再累积全局 streak，不同脚可以交替贡献。来源：[episode_evidence.py:362](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:362)、[429](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:429)、[589](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:589)、[Go2 协议:74](D:/humanoid_agent/true_RL_Agent/configs/tasks/stair_go2_12dof_unified_protocol_v3.json:74)。

作者可编写既有授权范围内的 reward 程序，ABI 提供字段定义与 reward-time 实值访问。反馈中的实际 rollout 数组是另一条链。诊断 contact 使用当前力范数，reward contact 使用3步 history 最大范数，时间语义分别由 [transition_decoder.py:551](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/transition_decoder.py:551) 与 [isaaclab_reward_runtime.py:231](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:231) 定义。反馈升级沿用当前 reward/PPO/课程/评估的授权划分。

## 3. 字段从采集到作者的实际去向

链路：pre-reset recorder → CPU host history → 每步 transition → first-episode tracker → outcome → ordered/unified summary → task public-dev feedback → prompt adapter → 作者。GPU staging 按块搬运，`host_history` 仍累计整段；最终返回 outcomes 和 vector steps。[采集:97](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py:97)、[历史与返回:321](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/evaluation_runtime.py:321)。以下“未存”指这条评估链的持久产物，“作者未收到”指运行实值。

### 3.1 原始字段：已采，逐步数组未存

| 原始字段 | 采集与后续处理 |
|---|---|
| `base_pos_w` | root 世界坐标m；派生局部 progress/lateral/height |
| `base_quat_w` | wxyz；解算 signed roll/pitch，transition只留abs |
| `joint_pos` | resolved joint顺序、rad；decoder没有放入transition |
| `support_pos_w` | 支撑刚体参考点、m；派生局部位置/投影地形层/clearance |
| `support_lin_vel_w` | 世界速度m/s；transition只留xy范数 |
| `support_contact_forces_w` | 世界合力N；派生contact bool及向上分量 |
| `forbidden_contact_forces_w` | 当前binding角色的逐刚体合力；Go2只有base一项 |
| `episode_length` | 控制步；最终留下闭合步 |
| `terrain_env_origins_w`、`terrain_types` | 场景坐标及分配信息；用于解码/分配检查 |
| `termination_term_order`、`termination_masks`、`done_mask` | 本步捕获term与并集，供pre-reset核验 |
| `runtime_projection_hash`、`reward_input_schema_hash_vnext` | 运行映射与ABI身份；静态版本在manifest保存 |
| `support_count`、`forbidden_contact_count`、`schema_version` | 轴长度/格式身份；当前Go2为4/1、G1为2/1 |

来源：[isaaclab_runtime_env.py:114](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py:114)、[transition_decoder.py:77](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/transition_decoder.py:77)。

### 3.2 Transition 的20字段

| 字段 | 持久化与公开反馈 |
|---|---|
| `env_index` | 槽位未入四字段fact；按cell/repeat保留列表分组 |
| `bin_id`、`evaluation_cell_id` | cell身份由summary分组键保存，可投影有界cell/stratum统计 |
| `episode_step` | 只保存outcome的`episode_steps` |
| `progress_m` | root局部x；压为回合最大`progress_fraction`，作者有平均值 |
| `lateral_m`、`base_height_m` | root局部y/z；曲线未存 |
| `base_clearance_m` | root高度减支撑所对应面高度；曲线未存 |
| `abs_roll_rad`、`abs_pitch_rad` | 绝对姿态；只贡献规则标志 |
| `forbidden_contact` | manager mapped term的本步bool；最终参与primary |
| `support_contacts` | 当前合力判定；序列未存 |
| `support_progress_m`、`support_lateral_m` | 逐support局部x/y；序列未存 |
| `support_surface_levels` | 水平投影查询的地形层；序列未存 |
| `support_planar_speeds_mps` | 世界xy速度范数；只贡献规则 |
| `support_surface_clearances_m` | reference z减同x/y的surface高度；只贡献规则 |
| `support_upward_forces_n` | 世界z合力；用于合法支撑 |
| `done`、`termination_terms` | 当步term并集/完整term集合；四字段fact没有保存 |

字段定义：[episode_evidence.py:27](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:27)；派生：[transition_decoder.py:515](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/transition_decoder.py:515)。

### 3.3 已保存、已投影和仍缺的内容

四字段fact固定为 `primary_reason`、`episode_steps`、`progress_fraction`、`diagnostic_flags`。outcome另有 `max_progress_m` 与 `active_terminal_terms`，四字段fact没有保留。标准作者反馈给 primary计数/主失败、诊断边际率、平均进度、cell/stratum/因子分数；逐回合闭合步、primary×flag共现和时序曲线尚未进入标准投影。来源：[ordered_summary.py:51](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/ordered_summary.py:51)、[public-dev投影:434](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/ordered_summary.py:434)、[unified_scoring.py:453](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/unified_scoring.py:453)、[prompt裁剪:877](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py:877)。

当前 eval recorder 还缺 root线/角速度、joint_vel/torque/acc、support quaternion、全身link pose、toe keypoint、接触点/对象/法线及同步视频。前几项在别处可访问，是否追加由具体问题决定。signed姿态可从已有quaternion派生，首个问题先保存已有必要状态。

训练诊断是另一种摘要：reward 首步及每24步抽样、`last_mean/max_abs/sample_count`，父reward在相同输入上的即时输出差；它没有早中晚轨迹，也不是父策略回放。相关来源：[reward_program.py:489](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/reward_program.py:489)、[602](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/reward_program.py:602)、[training_diagnostics.py:15](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/training_diagnostics.py:15)、[prompt_adapter.py:3454](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py:3454)。

## 4. v2训练策略如何成为可比较的v3数据

formal v2 manifest与当前Go2 v3的policy IO、观测、动作、agent配置、training profile、sensor、binding facts、reward schema与capability等13项contract hash相同；5项不同的是suite、evaluation protocol、ordered cell set、acceptance protocol与protocol state。两份训练回执同为v2训练manifest `193fa903…`、training binding `c0112a87…`。因此“历史v2分数不直接比较”和“已有策略可在统一v3重新评估”应同时成立。

现有 [stair_dev_regression.py:51](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/stair_dev_regression.py:51) 检查训练预算、训练binding和checkpoint字节，保留训练manifest身份，再运行当前public-dev。CPU preflight仅检查计划/回执，`checkpoint_bytes_checked=false`；Gen8-A真实run则为true，actor/critic加载成功、只执行public-dev。旧RSL-RL3模型到当前模型的推理映射由 [rsl_rl_checkpoint.py:18](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/backends/rsl_rl_checkpoint.py:18) 提供。

| 身份 | P3 reference | formal Gen8-A新v3回放 |
|---|---|---|
| checkpoint SHA-256 | `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c` | `bcd8de3a9d0e6e6a6c84b8c7833dc4a1b0ed697758ddee6f9c9eb31416a2f93c` |
| 当前manifest | 两者均`7e8fa0c45109bbfc0162a90cf68d4186a0a3f96a0df9cce1e9b1b09f50aa6de9` | 同左 |
| public-dev cohort | 两者均`85c91d968e78e7fb4d31450b1d33b84bf1928942ccc16a73ff42420df7c19cec` | 同左 |
| eval source manifest | `86fe5909…` | `dc4f199c…` |
| execution environment | `c9f3f098…` | `382b4a42…` |
| 保存的评估资格 | P3 reference资格PASS | 公开诊断run PASS，未执行reference资格 |

formal第8代ledger的真实parent是 `fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`，并非P3 reference。当前二者可提供 **cross-policy facts Diff**；历史真正parent→candidate试验应另外使用ledger指定祖先。这次修订撤回旧版“只能单策略、Diff一律missing”的范围限制，保留缺少时序与不同运行版本的标记。完整身份与选取的ledger字段见新JSON `comparison`；本分析未读取旧v2分数参与差分。

本地库存的24份formal训练summary中23份pass、1份fail；23条checkpoint记录对应22个不同hash，24份的训练binding/manifest均与P3训练回执一致。ledger有15代、30个lane记录，Gen1–8的parent为`fca6a59c…`，Gen9–15为`bcd8de3a…`。这是可继续抽取的历史池；代数、lane数、有效checkpoint数分别登记。

## 5. 两策略真实失败结构

每策略3 seeds×24 cells×12回合=864。下表由新脚本独立重算，全部cell及8个strata的分母/标志保留在JSON。

| 项目 | P3 reference | Gen8-A v3 replay |
|---|---:|---:|
| J_primary（v3） | 0.746697 | 0.924427 |
| success / forbidden / timeout / orientation / off_course | 689 / 113 / 36 / 22 / 4 | 822 / 34 / 4 / 2 / 2 |
| success组orientation ever | 11/689（1.60%） | 8/822（0.97%） |
| forbidden组orientation ever | 47/113（41.59%） | 2/34（5.88%） |
| success组slip ever | 560/689（81.28%） | 221/822（26.89%） |
| forbidden组slip ever | 103/113（91.15%） | 25/34（73.53%） |
| success组clearance ever | 50/689（7.26%） | 32/822（3.89%） |
| forbidden组clearance ever | 8/113（7.08%） | 6/34（17.65%） |
| forbidden闭合步：min / median / max | 90 / 291 / 633 | 105 / 289.5 / 439 |
| forbidden最大归一化进度：min / median / max | 0.232 / 0.597 / 0.904 | 0.195 / 0.786 / 0.901 |
| 有失败cell数 | 14/24 | 8/24 |
| 完全无成功样本的cell数（合并seeds） | 0/24 | 0/24 |
| 有失败但无成功的seed×cell组数 | 2/72 | 0/72 |

reference最集中的禁触cell为`c1_conditional_frontier_level_5`：30禁触、2成功，其次`height_ood_level_5`为19禁触、9成功。Gen8-A最集中于`c2_mechanism_ood_level_5`：13禁触、22成功。reference仅seed2/3的frontier level5组没有成功；跨seed仍有同cell对照候选。

这些事实支持三件具体工作：按cell/stratum描述失败迁移；比较不同策略的规则标志分布；把orientation与slip/clearance一并作为下一步时序核验对象。旧规则保留为可复核基线，整体边际率不足以决定删除规则或宣称它“没有信息”。例如Gen8-A的slip在成功/禁触组为26.9%/73.5%，审计“只在策略间有信息”的断言过强。

`progress_fraction`的公式是`clip((max root_x − progress_start)/(goal − progress_start), 0, 1)`；goal包含起步平台、全部treads及顶面offset。[episode_evidence.py:382](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:382)、[668](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:668)、[ordered_evaluation.py:161](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py:161)。reference禁触的十分位箱2–9正确含义为最大进度区间0.2–1.0，既不是物理2–9级，也不是终止时所在层。ever orientation同样只能写“禁触回合伴随41.6%”，先后关系需逐步档案。

G1仍作兼容资料：既有864回合中618成功、197禁触、28超时、21偏航；原 [共现表](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/stair_episode_cooccurrence.csv) 保留全部primary。此次最小双策略分析聚焦Go2。

## 6. Phase 2最小配对、分层与身体观测

**首批使用真实parent与Gen8-A。** 选取formal历史parent `fca6a59c…` 与Gen8-A `bcd8de3a…`，在同版runner/recorder、同v3 public-dev下评估；P3 reference可作为第3条策略。每策略capture off/on各一次、首批共4次启动是可行建议，取消原6次启动硬上限。在8 GPUh预算内先度量新增落盘成本和主统计一致性，再按实测扩到2–4条策略。22条checkpoint保留为候选资料池；实测约5分钟的旧单次run不作为新writer成本保证。上文已保存的P3↔Gen8-A分析仍是离线开发基线，与这组未来真实parent→child采集分别标注。

首问保持：“禁触回合内发生过哪些事件、何时发生；成功回合在可比阶段是否也有？”保存稳定episode身份、step/dt、root position/quaternion、support参考位置/速度/力、base力、捕获term masks、地形query/mapping身份及覆盖信息；旧ever统计需要完整低维控制步覆盖。关节位置已采，可按需要保留小窗口。第一版使用float32即可。

对照次序保持简单：先同策略同cell；不足时跨seed但仍分层登记；再用另一策略同cell做双策略对照。跨cell同物理阶段可以作为**辅助类比**，同时列出geometry/条件差异，独立显示同条件对照不足。cell名称中的level编号、归一化进度箱都不能代替实际地形边界。

| 情况 | 最小处理 |
|---|---|
| 成功/失败均有目标边界crossing | 以kinematic root跨登记边界为锚，事先固定首次向前规则和误差 |
| 失败在目标边界前闭合 | `not_reached/no_anchor`；保留总分母，展示终止前窗口 |
| 反复跨越或数据缺失 | 区分`ambiguous`与`missing`，分别计数 |
| 后窗过短 | 记录实际覆盖与censor mask；按outcome闭合边界处理 |
| 没有同条件成功对照 | 明示缺失；辅助跨cell类比单列，不替代匹配结果 |

**可选15-body合力范数。** [robot profile](D:/humanoid_agent/true_RL_Agent/configs/robots/unitree_go2_12dof_isaaclab_v1.json:1) 提供15个非足部body（包括base）；[v3 binding](D:/humanoid_agent/true_RL_Agent/configs/tasks/stair_go2_12dof_unified_binding_v3.json:1) 的contact sensor覆盖`Robot/.*`，任务禁触角色收窄为base。当前recorder只切出support/base，termination也使用该绑定：[采集:132](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py:132)、[终止:552](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py:552)。增加具名body的当前合力范数可定位“哪个刚体承受接触载荷”，复用现有sensor并核验运行时body映射；保持原base终止规则。它仍不提供接触点/对象或riser/tread面，也不能把其余14个body称为当前任务的禁触终止角色。

容量只作实现前估算：按288 env×640步×3 seeds，61个float32（含重复origin）+2个int64+3 bool/步，紧凑数组约145 MB/策略，额外14个body范数约31 MB。格式、重复metadata与writer会影响实占空间，应先测两策略。仅将原base xyz力改范数约节约3%，不是减半；float16的阈值误差验证可留待出现实际空间压力后。

Go2 基线调研
一、G1 那把尺子（先确认基线"合适"的判准）
程序化解析确认，G1-37 的基线是 Isaac-Velocity-Rough-G1-v0 → G1RoughEnvCfg，effective terms = 父类 RewardsCfg 11 项 + G1Rewards 新增 6 项 − undesired_contacts(删除) − lin_vel_z_l2(权重置 0) = 15 个有效项——与仓库里那份 reference 程序逐项吻合。

它之所以"合适"，是四个条件同时成立：忠实（257 样本逐项最大绝对差 0.0）、可学（C0-ID progress 0.943 / success 0.667 过门）、留有余量（J_primary 仅 0.645，三个 family 0.55–0.68，第 1 代候选就提到 0.942）、可用任务 ABI 表达。

二、Go2 的真实候选集只有三个
Isaac Lab 里 Go1 / A1 的 rough 覆盖与 Go2 逐行相同（同一份 Unitree 四足配方），ANYmal-D 直接用裸父类，Spot 是另一台机器且只有 flat。所以：

候选	有效项	关键特征
B1 Isaac-Velocity-Rough-Unitree-Go2-v0（计划当前选择）	8	dof_pos_limits=0、flat_orientation_l2=0、undesired_contacts 被删除
B2 Isaac-Velocity-Flat-Unitree-Go2-v0	9	把 flat_orientation_l2 加回 −2.5，但 height_scanner=None，与楼梯 187 维 height_scan 观测契约结构性冲突
B3 unitree_rl_lab Go2 velocity_env_cfg	16	全部非零；含 feet_slide、undesired_contacts、flat_orientation_l2 −2.5、dof_pos_limits −10.0、bad_orientation 终止
B2 直接出局。

三、B1 与 G1 基线的实质差距
楼梯协议评分的 7 个失败原因里，B1 对其中 4 个完全没有奖励信号：

楼梯失败原因	G1 基线	B1 Isaac Lab Go2 rough	B3 unitree_rl_lab Go2
fall / 终止	termination_penalty −200	无	无
base_orientation_instability	flat_orientation_l2 −1.0	0.0（失效）	−2.5
slip	feet_slide −0.1	无	−0.1
forbidden_contact	终止于 torso	undesired_contacts 被删除	−1（Head/hip/thigh/calf）
（关节限位）	dof_pos_limits −1.0	0.0（失效）	−10.0
需要澄清一点：这不影响可测量性——7 个失败原因由任务自有信号判定，与 reference 奖励无关。缺的是基线策略避免这些失败的动机，即起点策略会更差。

还有一处方向性冲突值得单独指出：lin_vel_z_l2 在 G1 里被显式置 0，在 B1 里是 −2.0（有效）。它惩罚基座垂直速度平方——而爬楼梯必须产生垂直位移。G1 的作者正是为此把它归零。以 0.8 m/s 前进、0.4 m 踏面、0.16 m 竖板估算，垂直速度约 0.32 m/s，每步惩罚约 −0.2，而 track_lin_vel_xy_exp 满分才 1.5。不是压倒性的，但方向不利。

四、被低估的两个耦合风险（比选基线更关键）
1. 几何：G1 的楼梯 Go2 物理上走不了。 这是我这次调研最重要的发现。

G1-37 unified suite	Go2 实际能力
C0-ID 竖板	0.12 / 0.22 / 0.32 m	站立高度 0.40 m
height_ood 竖板	0.375 / 0.40 / 0.425 m	Unitree 标称最大跨步 ≈ 0.16 m
训练 riser 范围	0.05–0.35 m	—
height_ood 的 0.425 m 高于 Go2 整个身高。若沿用 G1 几何，无论选哪个奖励基线，reference 都会全线接近 0 分并卡在 ID learnability 门上。计划第 540 行的"robot-scaled cells"必须先落地，量级大约是按站立高度比 0.54 缩放（Go2 的 C0-ID 约 0.065 / 0.12 / 0.17 m，与 0.16 m 标称跨步吻合）。共享绝对锚点取 0.10–0.12 m 比较合理：G1 轻松、Go2 吃力，正好用于跨机器人描述。

2. 执行器模型不同。 Go2 用 DCMotorCfg（力矩-转速饱和的非线性模型，effort/saturation 23.5、velocity_limit 30、stiffness 25、damping 0.5），G1 用 ImplicitActuator。这直接影响 ABI 里 joint_torque 的取值范围——也是为什么 Go2 的 dof_torques_l2 权重是 −2e-4 而 G1 是 −1.5e-7（相差约 1300 倍）。RobotProfile 需要如实记录，reward 等价性检查也要在该模型下做。

五、ABI 可表达性（两个候选都过关）
楼梯 RewardInputs 是按角色/组参数化的，Go2 需解析为 12 DoF、4 个 support member（G1 是 2）、无 ankle/arm/finger 组。

B1 的 8 项全部可表达，无需新增字段。唯一需注意：Go2 用的是四足版 feet_air_time（compute_first_contact + last_air_time 求和），与 G1 的 feet_air_time_positive_biped（单支撑 min(air,contact)）是不同函数，需按四足语义重写，可由 support_air_time / support_contact_time / support_contact 导出。
B3 需额外代价：移植 3 个 unitree_rl_lab 自有函数（energy、joint_position_penalty、air_time_variance_penalty），且 undesired_contacts 覆盖 Head/hip/thigh/calf，而 ABI 目前是 forbidden_contact_force|1,3（单体），需要扩成多体接触组。另外它上游把 height_scan 观测注释掉了。
还要提醒一点：本项目对 unitree_rl_lab 的既有先例（G1-29）只覆盖 RobotProfile 资产事实，不含 reward 等价性验证——Push 明确是 push_recovery_has_no_reference_program_genesis。所以 B3 会是本项目第一次对非 Isaac Lab 上游做 reward 等价性收据。

我的建议
先按计划用 B1，但把几何先定下来，并在烧 GPU 做完整资格化之前加一次廉价探针。

理由：

B1 与 G1 方法论对称——同一上游、同一"父配置 + overrides 解析 effective terms"流程。矩阵的核心主张是"加一行只要薄适配器"，用不同上游家族会削弱这个主张。计划验收项 5 写的也是这个。
试错成本低：reference 训练一个 job ≈ 1.2 GPU-h（8.5377 GPU-h / 7 jobs，含评测摊销）。B1 失败的代价是一个多 GPU 小时，不是灾难。
真正的风险在几何不在奖励。几何选错，B1 和 B3 都会失败；几何选对，B1 的 8 项有没有可学性才是有意义的问题。
顺带一个正面理由：B1 更薄意味着改进余量更大，而 floor-only 资格规则只要 family 分数 ≥ 0.1 就通过——弱基线反而更容易资格化。真正的门槛是 C0/C1 的 ID learnability。
执行顺序建议：先定 Go2 缩放几何 → 用现成的 scripts/isaaclab_stair_phase0_probe.py 做一次短程探针（确认 Go2 能在 C0 几何上迈上第一级）→ 再提交完整 reference 训练。若 B1 卡在 ID learnability，B3 作为已记录的备选，代价是 3 个函数移植 + 多体接触组 + 首次非 Isaac Lab 等价性收据。

一个明确不建议的做法：不要为了让 Go2 好看而给 B1 手工补 feet_slide / flat_orientation_l2。那就不再是 "faithful official reference"，会同时破坏基线合法性和"LLM 改进"的可归因性——补的那部分本来正是应该留给 LLM 去发现的。
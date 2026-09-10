# P3 实施与事前验收约定

2026-09-08。前置修复已通过 [PRE_P3_ACCEPTANCE_ZH.md](PRE_P3_ACCEPTANCE_ZH.md)。以下约定在新题池的 anchor 测量和 candidate 搜索之前写入；本阶段不启动 LLM 搜索。

## 1. 目标与归属

继续使用机器人 × 任务矩阵。C0 规则楼梯、C1 不规则楼梯、C2 楼梯表面 roughness（几何粗糙度）变化是 Stair 的样本分布，不成为新任务、第三维或独立生命周期。训练保持已有 C0/C1、单策略、固定 PPO 与 2000 iterations。按用户最新目标，继续现有 C2 微起伏机制，不引入摩擦材料迁移。

修改集中在已有 Stair 的 suite、ordered summary、scoring、terrain adapter 和 final confirmation。RobotProfile 只提供形态事实，binding 提供薄投影；共享 acceptance 继续解释任务声明的具名 guards，不写机器人或楼梯分支。参考 Codex 的窄配置视图与单一有效配置思想（`codex-rs/rollout/src/config.rs`、`codex-rs/config/src/state.rs`），不引入它的配置层框架。

## 2. 实施顺序

1. 固定指标与标定规则，补充 outcome 原始时间事实和可复算摘要。保留历史 writer/reader 的版本含义，不重写旧实验。
2. 核验现有 roughness 网格与证据表面投影的一致性，补充尺度记录，再跑机器人。
3. 用固定 G1、Go2 anchor 测公开标定池，按下述规则一次选择正式题层。新 reference 必须在当前后端资格化；旧 checkpoint 只作诊断，不能冒充新后端随机初始化训练。
4. 签发新 suite / cell / acceptance，并接入已有评测、reference qualification 和 final 预承诺。新协议只有资格通过后才成为活动入口。
5. 有限反例、真实 G1/Go2、历史重放、完整相关回归与资源收尾。按实际结果给 GO 或 NO-GO；不得临时降低标准刷过。

## 3. 冻结评分

每个 episode 保留合法成功 S∈{0,1}、截断到 [0,1] 的最大目标进度 P、实际闭合步数 t、题目截止步数 H。成功仍是 V2 的 first-valid-top-arrival，提前违规到达不算成功；短时诊断可与成功共存。

合法完成效率 E = S × (H − t) / H。所有失败以 H 右删失，E=0，不能把提前摔倒解释成快速完成。主分 q = 0.8S + 0.1P + 0.1E。合法成功的分数不低于 0.9，失败不高于 0.1；对全部样本报告成功 n/N、进度、删失完成时间和效率，不仅报告成功子集。

聚合仍为 episode → cell → stratum → paired seed。J_ID 为 C0-ID、C1-ID 等权；F_scalar 为 height/tread/joint 等权，F_c1 为 phase/frontier 等权，F_c2 为 roughness；J_OOD 为三个 family 等权，J_primary=(2J_OOD+J_ID)/3。旧 0.5S+0.5P 同时作为标定诊断，不根据 candidate 胜负选择公式。

保留 j_ood、c0_id、c1_id 能力守卫，增加 joint_ood 独立非劣守卫。沿用既有配对 seed 标准误与 headroom 判决：delta≥max(0.2×reference headroom, 1.5×paired SE)，具名 guard 最大回退 0.05、配对回退余量 0.02；绝对 floor 及 public→sealed 规则承接当前 acceptance 的明确值。诊断项不另加门。

新 summary 保存逐 episode 的整数步数、原因和进度事实，可独立重建聚合。缺失、重复、非有限、非法成功/原因组合是无效证据，不当零分；正常失败和超时是有效样本。

## 4. 尺度与真实 C2

使用 profile 已有 `nominal_root_to_support_body_center_z_m` 作为支撑高度 L，记录来源、米制值及 h/L、d/L、v/√(gL)、t√(g/L)。这里的 L 是明确的支撑高度代理，踏深也以同一明确长度归一化，不伪称测到了身体长度；Froude 量仅为描述，不建立跨机器人总排名或动态相似性结论。不同机器人仍各自校准、冻结、比较，不混池。

C2 在 C1 的 ID 几何上添加既有 bounded-piecewise-linear 微起伏。网格与任务证据共用 `rough_surface.py`；材料保持原值，不复制地形实现。保留现有最大振幅 0.015 m、1–3 个空间周期及三角网格采样约束。训练生成器只加载 C0/C1，验证 C2 实际训练暴露为零。

## 5. 公开标定与预算

固定参考程序：当前各 row 指定的 official reference。Go2 固定 P2 完整新后端 checkpoint `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`；它可复用的前提是训练 profile、预算、PPO、资产与 reference 源码全部相同。G1 在当前后端最多新增一次完整随机初始化训练；旧 G1 仅作诊断。每个模型的身份、源码与训练证据在测量前记录。禁止 Gen8 或新 candidate 参与选题。

每个机器人 8 strata × 6 强度 × 3 seeds × 3 episodes = 432 个标定 episode；同一配置最多执行一次完整标定，基础设施故障可原身份恢复。seed 与旧正式池分离。六档强度由训练支撑区间及 L 产生：ID 使用区间分位 [0.15,0.31,0.47,0.63,0.79,0.95]；高度外推增量/L 为 [0.005,0.01,0.02,0.035,0.05,0.07]；踏深缩短量/L 为 [0.02,0.05,0.10,0.15,0.22,0.30]；joint 配对相应两者，不做笛卡尔积；C1 phase 使用训练外相位，frontier 使用相同高度增量；C2 振幅/L=[0.004,0.006,0.009,0.013,0.019,0.020]，周期按 [(1,1),(1,2),(2,1),(2,2),(3,2),(3,3)] 配对。未来 profile 导出的振幅越过既有物理上界时标定不支持，不能静默截断。

每个 stratum 依次选容易锚点、最接近成功率 0.5 的可学层和较难锚点；并列按预先声明的强度顺序。可学层必须至少 1 次合法成功，均分≥0.1，且有失败或合法完成效率跨 episode 的范围≥0.02。每个优化 family 必须至少有一个这样的层，ID 也执行同一资格检查；不能仅因新公式的理论上限留下虚假的可学习余量。每个 family 的三层/各 stratum 记录 floor、ceiling、variance 和真实时间差。无层满足时该机器人 NO-GO，不通过动态 mask 丢掉难 family。

正式每个 stratum 3 个实例。Dev/accept/sealed/新 final 使用同一选定强度邻域内不同几何（高度偏移不超过 0.002L、保持原 ID/OOD 类型），seed 独立。公开 calibration 题不直接充当 formal 题。不同 cohort 的难度由声明的邻域规则保证，并报告实测偏移；不因 sealed 的结果调题。

总新增资源上限 14 GPU 小时：G1 完整训练至多 10 小时；标定、正式资格、roughness/录像验证合计至多 4 小时。保留 Go2 已合格训练，不无故重复 2000 iterations。以既有 lease / process receipt 计费，停止条件为预算耗尽、完整证据显示任一必需 family 无可学余量、或无法解决的语义接线失败。NO-GO 后不自动扩训练分布、追加 seed 或降低阈值。

## 6. 新 final 与兼容边界

新研究在 Genesis 前生成 study 身份和隐藏 final 实例/seed，公开仅保存承诺 hash；resume 读取既有文件，不能重抽。不同研究生成不同 cohort；显式 replication 带被复用 study/cohort 身份并声明不再“首次使用”。final 只在冻结候选后一次消费，无 acceptance、反馈或选优权。复用已有 Stair final plan/reservation，不增加共享生命周期；P4 再做任务中立编排提取。

旧 protocol V1/V2、rough C2、summary V1–V3、final V1/V2 仅按原语义读取；完整 15 rounds 的全部 checkpoint 与证据保留。新版本只为实际改变的格式增加，不能换版本号重编码全部历史。

## 7. 验收证据

必须包含：合法成功/违规到达/短时诊断后成功/提前终止/超时的评分反例；独立重建整数计数与效率；joint 回退即使其他层变好仍拒绝；family 全零/全满不能资格化；roughness 网格/证据投影一致；训练 C2 零暴露；G1/Go2 实测校准表与固定预算；cohort 几何和 seed 隔离、final 恢复与 replication；历史结果不变；实际评测入口和原生录像的题面身份一致。相关检查通过后只跑一次完整回归，失败引起的修改再定向重验。

研究依据：Rudin 等的 [并行课程训练](https://proceedings.mlr.press/v164/rudin22a.html) 支持训练/难度区分，不替代本任务实测；Agarwal 等的 [有限样本统计](https://arxiv.org/abs/2108.13264) 提醒按配对 seed 报不确定性，不能把同一 checkpoint 的 episode 数冒充训练复现。尺度参考 [Alexander 与 Jayes](https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/j.1469-7998.1983.tb04266.x)，不由归一化自动推导跨机器人性能可比。

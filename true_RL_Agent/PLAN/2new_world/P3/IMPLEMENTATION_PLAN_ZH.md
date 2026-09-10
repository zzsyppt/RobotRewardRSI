# P3：地形优先的指标与跨机器人评测实施计划

2026-09-09 修订。公开标定及资源边界见 [复审计划](AUDIT_FOLLOWUP_PLAN_ZH.md)；后续比例、默认行与目标全量修正见 [回归复审](REGRESSION_FOLLOWUP_ZH.md)，当前测量结论见 [验收](ACCEPTANCE_ZH.md)。2026-09-08 的题池、资格和记录保留在 results 与冻结源码包内，不能替代本次新题池的验收。前置 P2 九项复核见 [PRE_P3_AUDIT_ZH.md](PRE_P3_AUDIT_ZH.md) 和 [PRE_P3_ACCEPTANCE_ZH.md](PRE_P3_ACCEPTANCE_ZH.md)。

## 所有权与主线

矩阵只有任务 × 机器人。C0 规则、C1 不规则、C2 表面起伏是 Stair 内部分布。RobotProfile 提供资产与形态事实；Stair 解释尺度、生成分布、标定难度和定义结果；binding 薄投影；共享 runner 与接受引擎只消费冻结事实，不按机器人名称生成地形。

成功率饱和优先修改题池。合法成功仍为 first-valid-top-arrival，cell score 保留 0.5 success + 0.5 progress。完成时间只保留原始 episode 步数，不参与目标或资格，也不维护无消费者的效率统计。

本次当前研究的主目标为 J_OOD，scalar、C1 frontier、C2 三个 family 等权；ID 与 phase/null control 独立守卫。每个目标 stratum 都必须实测有成功率转折，不能靠 family 平均掩盖全满/全零层。未来机器人 ID 是否仍有优化空间须重新判断；本研究的守卫安排不是永久任务定理。

## 先生成范围，再实测定位

L 来自 RobotProfile 的 nominal_root_to_support_body_center_z_m（G1 0.717680 m，Go2 0.422776 m），只用于初始尺度，不等于足长、水平身体长度或可达步长。训练的绝对米制条件继续属于训练 profile。本轮复用已完整训练的 reference，训练分布、PPO、随机初始化和预算不变。

工厂直接使用机器人事实、训练 profile、任务的八种 stratum 声明，不要求先造旧版 72 个几何对象。高度从训练上界向外探测，踏深按 L 从各自训练 d0 有界收窄。C1 保持 pattern 覆盖；C2 先选 reference 可稳定通过的平面基底，再单独改变表面，并在每份正式题池中保留完全匹配的平面对照。

公开粗标定、一次细化、冻结与独立 qualification 依次进行。每 cell 每 seed 12 episodes，粗/细标定每 cell 共 36；正式每 cohort 24 cells、288 envs。具体梯度、选题、cohort 位移、失败修正和停止规则只在复审计划中列出，实际 suite 为机器权威。增加 rollout 不等于增加独立训练 seed。

## 留出、身份与历史

dev/accept/sealed 使用有实际尺度差异的几何实例。Study final 沿用原 final plan、reservation 和 receipt；私有 nonce 生成独立几何并承诺。resume 复用原文件，replication 显式声明，final 不进入作者反馈或选优。本阶段只做验收探针，不声称完成了正式多代 final comparison。

私有 suite 是普通、可 round-trip 的真实 suite：序列化、cell、geometry 与 cohort hash 一致；source_suite_hash 只表示其公开冻结来源。公开 seed 和资格 mask 沿用来源身份，私有 final 的 seed 绑定实际几何 cohort，不复制另一套 suite 视图。

V3 的 evaluation-only lineage 复用原 TrainingBinding 和训练 seed，checkpoint provenance 指向直接 predecessor。V1/V2 reader、原协议、旧 ledger 与完整 15 rounds checkpoint 保留；新评分不回写旧证据。三个新运行入口统一从 catalog 默认 V3，历史回放须显式选择原 cell。

## 验收顺序

1. 必要反例覆盖目标与能力守卫、原始事实复算、几何范围/覆盖、匹配控制、private round-trip 和默认入口。
2. 两台机器人使用已有完整 checkpoint 做公开标定；冻结正式套卷后走生产阶段作业与资格门。
3. 验证原生视频、实际网格与表面查询、独立 final、恢复及训练身份。
4. 完整本地回归、开启 Isaac 动态用例的目标环境全量、旧证据复算、源码发布校验及资源/受保护历史收尾；三个 generation session 场景必须实际执行，专项不能替代全量。

只以实测签发资格；新机器人可复用的是方法，不能凭合成尺度测试或两台机器人结果声称普遍物理泛化。P4 的通用编排与成本工作保持原边界，不因本次地形修订另造系统。

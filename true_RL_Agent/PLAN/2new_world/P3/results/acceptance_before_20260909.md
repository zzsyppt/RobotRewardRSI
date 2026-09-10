# P3 验收：地形区分力、跨本体标定与证据闭合

2026-09-08。**P3 工程与 G1/Go2 reference 资格均通过。** 本轮没有运行 LLM/candidate 搜索，没有形成新的 accepted 改进结论。P4 的共享编排与新后端成本合同仍须完成，之后才能开始新的正式研究。

## 1. 最终改变

- 主分保持 `q = 0.5 success + 0.5 progress`。撤回完成时间入分及“耗时有差即可资格化”的未签发支线；保留可复算的 episode 事实与完成时间诊断。
- 同一 Stair 参数工厂从 RobotProfile 事实生成高度、踏深、不规则变化和 roughness 梯度，再以各本体固定 reference 标定。没有在公共层新增 G1/Go2 分支。
- C2 使用已验证的训练内平面基底，振幅与波长同时按尺度处理；不再被强制绑定到低矮 C1 基底。实际训练 C2 暴露为零。
- 每个 OOD family 都须存在 20%–80% 成功率的 cell；ID 保留能力底线与主分权重。joint-OOD 增加独立非劣守卫。
- V3 复用直接 V2 来源 checkpoint，同时保留最初的训练 seed / TrainingBinding；所有 evaluation-only 谱系环节都核对科学训练条件。身份验证不再依赖摘要格式开关。
- Study final 在 Genesis 前承诺，恢复原实例，新研究生成新实例，主动复现记录 replication_of；接线只位于 Stair final 配置视图与入口，未改共享状态机。

详细参数与限制见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)。P2 后九项问题的判断、修复与验收另见 [前置审计](PRE_P3_AUDIT_ZH.md)和[前置验收](PRE_P3_ACCEPTANCE_ZH.md)，不重复抄录。

## 2. 真实训练与独立资格结果

G1 本轮从随机初始化完成 4096 environments × 24 steps × 2000 iterations = **196608000 steps**。Go2 复用 P2 相同预算的新后端 reference。两者推理加载均为原生 RSL-RL 5，先核验已构造 learner，再加载策略；未加载 optimizer 或训练 iteration。

| 本体 | 完整 reference checkpoint SHA256 |
|---|---|
| Go2 | `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c` |
| G1-37 | `4f62db0e487cfc39c3cef61777bef9e4160b64f91f92fcca1cc795585ab8236c` |

G1 训练的冻结 checkout、patch、预算与路径见 [训练回执](results/g1_reference_training.json)及[原始摘要](results/g1_reference_training_summary.json)。Go2 来源见 [P2 后实测核验](results/preflight_actual_evaluations.json)。这些是完整训练 checkpoint，不能用旧短 smoke checkpoint 替代。

| 本体 | 最终公开标定 | public-dev | public-accept | sealed-validation | public-accept J_primary | reference gate |
|---|---:|---:|---:|---:|---:|---|
| Go2 | 321/432，74.31% | 164/216，75.93% | 264/360，73.33% | 363/504，72.02% | 0.847361 | pass |
| G1-37 | 343/432，79.40% | 151/216，69.91% | 258/360，71.67% | 374/504，74.21% | 0.827842 | pass |

表中成功率与 J_primary 分开记录。题池已改变，不能将新旧协议的 J_primary 相减，宣称训练改进或退化；也不能按这两行的总分排名本体。

最终标定的 C2 六档成功率，按难度参数顺序：

- Go2：9/9、8/9、9/9、9/9、9/9、6/9。
- G1：9/9、9/9、5/9、6/9、4/9、2/9。

选择规则冻结后，独立 public-accept 的 C2 三档分别为 Go2 **15/15、11/15、12/15**，G1 **13/15、9/15、4/15**。并非所有层都已脱离饱和；C0/C1-ID 和 phase 层仍可能接近满分，但每个 OOD family 都有可测成功转折。样本量较小，曲线不要求严格单调，也不据一次波动调整标准。

- 标定原始结果：[Go2](results/go2_normalized_surface_calibration_04.json)、[G1](results/g1_normalized_surface_calibration_02.json)。
- 完整生产入口、phase 证据与任务资格：[Go2](results/go2_formal_qualification_01.json)、[G1](results/g1_formal_qualification_01.json)。
- 正式 suite：Go2 `449630db2cfc26f42870dd5e5a63ea78be94dc323870aa4c5b344b1e37b2f23a`；G1 `149a8c6f572c4477f914a03681df46738c0c6ef7f9c27e70e75a05f61a71f4f2`。

## 3. 验收矩阵

| 验收项 | 结果与证据 |
|---|---|
| 原始 episode → 统计复算 | 新摘要的计数、进度、诊断、分层和分数可重建；篡改原始事实会破坏一致性。时间变化不改变分数或资格。 |
| 独立 joint 非劣 | 使用实际 V7 接受合同构造反例：总分 0.5→0.7、joint 0.5→0.3 时拒绝；joint 保持 0.5 时可接受。 |
| 零分/满分题池 | 全零、全满及只改变成功耗时均 NO-GO；困难 family 不可被移出目标后资格化。 |
| 同工厂跨本体 | G1/Go2 均实测 432 episodes，采用同一尺度解释、基底选择、六档规则和资格判定。L 及实际米制参数进入冻结 suite。 |
| 真实 roughness 网格 | 每个 C2 标定 cell 核对所有顶面三角形中心：Go2 10752、G1 4096；网格/查询最大高度差分别低于 3e-15 m / 6e-15 m。训练构造器 C2 数为 0。 |
| checkpoint 身份 | 生产 runner 接受直接 predecessor provenance；训练 binding 与训练 seed 不变。重新计算协议 hash 后改变训练命令仍会被谱系检查拒绝。 |
| final 实例与入口 | 两本体均通过绑定私有实例的真实 final 入口；研究间 seed/cohort 不同、resume 不重画、replication 显式、承诺篡改拒绝。此处为验收探针，不是 15 代终态比较。 |
| 真实原生录像 | 两本体各 3 段，均为正式 public-dev 的 C2 困难层。6 段实际解码帧数与旁车计数一致，几何 digest 与正式 cell 一致；没有另一个物理后端或录像侧重判。见 [运行复算](results/runtime_acceptance_replay.json)。 |
| 历史复算 | 原 Go2 V2 摘要逐字段复算相同，J_primary 仍为 0.883938290781893。原协议、ledger 和 checkpoint 不重写。 |
| 本地完整回归 | **1625 项，0 failures/errors，19 项环境跳过**；[回执](results/full_regression.json)。跳过不作为真实 Isaac 测试通过依据。 |
| 目标 Python 专项 | **42 项，0 跳过**，包含动态 recorder、469 帧条码编码、checkpoint 张量、标定、study final 与正式 argv 接线；[回执](results/target_final_tests.json)。 |
| 文档核对 | 更新 README、系统概览、Stair 指南、runbook；7 项文档测试通过。旧版本按历史含义保留。 |
| 资源与历史保护 | **71 个正式 15-round checkpoints** 保留；48 个中间文件 stat、23 个权威 checkpoint SHA256、884 个产物 SHA256 无变化。GPU 无计算进程，4 个 lease 锁均可获得；[回执](results/resource_history_closure.json)。 |

相对 GPU 测量源码，三处产品收尾差异为训练谱系核验边界、V3 catalog 开放和内部 episode-facts 命名；它们不改变题面或算术。目标 Python 专项和全部运行回执在最终代码上再次验证。测量源码为 [formal_acceptance_source.json](results/formal_acceptance_source.json)，最终代码为 [final_source.json](results/final_source.json)，两者差异显式列出，未把旧进程冒充最终源码的重新运行。

## 4. 资源、发布与保留范围

P3 计划内工作的 GPU 占用保守合计 **2.043 小时**：G1 完整训练约 **1.205 小时**，其余约 **0.838 小时**，含撤回方案的标定、最终资格、录像和 final 探针，均在 10+4 小时分项上限内。数字来自训练/phase 的实际 lease 时长以及标定进程起止，不用推测 FPS 代替成本。

新代码与配置在本地和服务器执行副本完成验证，随后同步到服务器主项目 `/data/yuyangliu/AGENT/true_RL_Agent`。85 个源码、配置、测试、文档和必要结果文件逐个核对部署前后 SHA256；主项目再执行 13 项参数工厂、计分与 study 身份测试，全部通过。同步没有修改训练 checkout、历史实验目录或创建 Git 提交，见 [部署回执](results/deployment.json)。

两条 V3 行已注册开放，新增研究应显式选择；V2 保留已有会话兼容，历史默认与正式成本参数由 P4 一并更新。本轮不开始新的 15-generation 搜索。

必要源码包、短结果、原始日志与 6 段录像留在本轮工作目录；重复 Kit 日志、视频、训练权重不进入 Git。被撤回的完成时间提案只保留在测量源码包中，没有增加产品兼容 reader。没有进行与本阶段无关的批量删除。

## 5. 泛化结论的限度

本轮证明的是：同一参数工厂与标定流程能为这两个本体找到有区分力的 Stair 题池，且可以沿同一后端和身份链执行。它不是所有机器人等难度的证明，不是多训练 seed 的性能提升实验，也不是从两个本体推出无限机器人泛化。

L 是已有支撑高度代理；水平身体长度、足部几何、关节能力未被一个标量完整编码，因此新机器人必须重新标定，不能照抄已选档位。粗糙振幅与波长都要保存，不能仅比较 A 的厘米数。

正式 cohorts 与 study final 当前是选定难度的几何近邻；虽有独立实例/seed 身份，但微小几何偏移本身不证明更远参数组合泛化。新的研究仍须在其宣称的几何范围内重新预注册；不得把本次入口探针或近邻结果包装成新的大跨度 OOD 发现。

Push 的尺度与恢复指标需要在其任务内、基于实际施扰机制和 reference 资格化完成；P3 没有把 Stair 的 L、地形规则或成功条件塞进 Push/公共层，Go2 × Push 闭合继续按 P4/P6 边界推进。

# Phase 3A 历史实施计划：C1 irregular、cell-local 门与首个复杂闭环前置

> 日期：2026-08-27
> 架构状态：**SUPERSEDED_BY_PHASE3R**
> 历史范围：当时在同一 runner 与 LLM LOOP 内新增 `stair.g1_37dof.irregular_v1`；其中可复用能力继续保留，但“C1 是独立 active cell、需独立 reference/LOOP”的路线不再执行。
> 当前入口：[Phase 3R 修正计划](../P3R/PHASE3R_CORRECTION_PLAN_ZH.md)

## 1. 本阶段目标与不变量

本阶段只解决首个 non-legacy Stair cell 的四个必要问题：

1. 让 Stair VNext 通过 task-owned、显式有序的 cell suite 消费非笛卡尔评测题目，彻底停止把新评测集合解释成 `height × tread` 网格；
2. 落地踏面保持平面、但逐级高度与踏深均不相等的 C1 生成器，并让 terrain、解析 surface query 与 episode evidence 使用同一份逐级几何事实；
3. 让 resolved cell hash 成为该 cell 的 effective TaskSpec、EvaluationBinding 与 TrainingBindingV2 的共同比较域，同时冻结 cell-local acceptance、reference gate、评测 cohort 与 VNext reward ABI；
4. 在任何 LLM LOOP 前完成 G1-37 C1 reference 的真实启动、训练/评测资格化和成本记录，使 Phase 4 可以直接运行一条受控的多代 reward LOOP。

以下边界不可改变：

- C0 clean 的 V5 35-cell、历史 TaskSpec/EvaluationBinding/TrainingBindingV1、官方 G1 reward ABI 和 replay 行为保持原路径；
- 不把 Stair 的逐级几何、成功、失败分类、课程或评测 cell 语义放进 AgentCore、共享 runner、RobotProfile 或公共 composition；
- LLM 仍只能修改完整 reward program；terrain、cell、seed、预算、评测、acceptance、checkpoint 与 PPO 继续由我们掌握；
- 不新增通用 TerrainPlugin、数据库、服务、第二套 ledger、自动 teacher 或大范围安全框架。

## 2. 实施包与工作量

### P3A-W1：有序 cell-suite 与身份契约（1.0–1.5 人日）

**目标**：新增 Stair-owned `OrderedCellSuite`，显式冻结最多 18 个唯一 C1 cells、factor metadata、split 归属、episodes、horizon、cell-set hash 与 pairwise coverage；legacy `StairEvaluationProfile V5` 保持独立。

**验收**：

- 非笛卡尔套卷完整通过 strict JSON 读取，未知字段、重复 cell、split 重叠、环境数不闭合或 hash 漂移均拒绝；
- calibration/public_dev/public_accept/sealed_acceptance/unseen_final_holdout 的 cell 集合互斥，正式三 split 每次不超过 32 env，总 suite 不超过 18 cells，final holdout 不超过 4 cells；
- 所有声明的可行二元 factor 组合覆盖率为 100%，报告由显式 cell 列表重算而不是相信配置中的百分比；
- C0 V5 的 35 cells、280 env、hash 与现有单测结果不变。

### P3A-W2：C1 几何、解析 surface 与 runtime 证据（1.0–1.5 人日）

**目标**：实现一套 task-owned 不等高/不等深平面楼梯生成器；训练按 row difficulty 在 low irregular 与 target irregular 间单调插值，评测按 cell 的显式序列生成。

**验收**：

- 每个 C1 cell 的逐级 riser、逐级 tread、累计边界、累计高度、总 run、总高度、顶台长度和 geometry digest 可独立重算；
- 同 manifest + seed 在两个独立进程得到相同 digest；难度增加时 riser 不减、tread 不增，最大 grade 单调增加；easy 端提供起步台阶，hard 端覆盖评测最大 grade；
- mesh 顶面与独立累计几何 oracle 在每级中心、边界两侧、起点和顶台采样一致，误差不超过 `1e-6 m`；
- decoder、clearance、support level、top height 与 episode success 使用同一 `IrregularFlatStairSurfaceQuery`；C1 不复用等高近似；
- 真实 Isaac smoke 回传实际 terrain type、origin、runtime projection、geometry digest，且结束后无遗留进程/GPU 占用。

### P3A-W3：非网格 evidence、公开投影与 prompt（1.0–1.5 人日）

**目标**：新增 ordered-cell summary，按 cell 等权汇总成功率、失败分类、诊断与进度；public_dev 只投影公开 cells，sealed 与 unseen cells 不进入 prompt/memory。

**验收**：

- config → terrain assignment → recorder → decoder → episode outcome → ordered summary → public projection → prompt 的 cell ID 顺序和 count 闭合；
- 新路径不存在矩阵切片、虚构 height/tread 网格或历史 G1 robot schema fallback；
- public_dev prompt 提供紧凑 cell/factor/失败计数，public_accept 仅用于 acceptance，sealed/unseen 的 cell 几何、seed 与逐 cell 结果不泄漏；
- 历史 grid prompt、public projection、memory/replay 单测保持通过。

### P3A-W4：resolved composition、VNext reward 与 cell-local 门（1.0–1.5 人日）

**目标**：选择 cell 后一次性形成最终执行描述；由版本化 RobotProfile 提供形态拥有的关节组，由 Stair 投影为 task-owned reward 输入，并为 C1 冻结 VNext reward contract、G1 irregular reference program、acceptance protocol、reference capability gate 与 cohort hashes。

**验收**：

- `stair.g1_37dof.irregular_v1` 的 TaskSpec、EvaluationBinding、TrainingBindingV2 三者 task protocol hash 全等于 manifest 的 `resolved_cell_task_protocol_hash`；clean cell 三者仍走历史身份；
- manifest 的 acceptance hash、reference gate hash、reference-gate cell-set hash、public_accept cohort hash、ordered-suite hash、geometry profile hash和 VNext reward hashes 均与运行时主动对象相等；
- train/eval 子进程先重载 manifest，再按 cell 构造 config；跨 cell、旧 ABI reward、V1 training binding、base task hash 或 manifest 篡改在 AppLauncher/PPO 前拒绝；
- candidate admission、smoke fixture、prompt reward contract 与真实 runtime 使用同一个 VNext input schema；C0 仍使用冻结 42-field V2 ABI；
- G1 irregular reference 的 15 项名称、权重、single-stance 门控和关节组作用域与官方基线逐项等价；具体关节名只存在于 RobotProfile，不进入 Stair 通用执行代码；
- reference program 与 candidate 使用同一训练分布、budget、seed 域和评测绑定。

### P3A-W5：标定、真实资格化与 Phase 4 预算（历史方案，现已取消）

**历史目标**：只用 calibration namespace 判断预注册的 easy/target/hard 定义是否“可学习且非平凡”，随后冻结 cell-local thresholds，并在互斥的 public_accept 与 sealed_acceptance 上资格化 reference。该工作包不再执行；其成本与失败记录仅作为 Phase 3R/4 的输入。

**验收**：

- calibration 之前 acceptance/reference gate 配置处于明确的未冻结状态，不能进入 LOOP；冻结后 hash 进入 manifest，且看到候选结果后不得改 target cells 或 threshold；
- reference 在 easy 有能力、target 非饱和、hard 更难；只有 reward 官方等价、训练课程覆盖和 runtime activation 已分别证明后，能力失败才可形成 reference NO-GO，不以放宽门或增加 generation 掩盖；
- 独立 public_accept 与 sealed_acceptance 都通过冻结门；qualification receipt 回传的 gate、cell-set、cohort 与 manifest 一致；
- 记录单 split 与 full phased eval 的 env 数、vector steps、steps/s、墙钟、GPU、输出量；Phase 4 预注册 `pilot_1`、`validation_3` 的训练/评测/失败重跑预算，不宣称预算最优；
- `loop_verified` 是 Phase 4 的工程门，`improvement_observed` 只报告，不作为本阶段或 Phase 4 的强制 GO 条件。

### P3A-W6：回归、独立复算与冻结回执（0.5–1.0 人日）

**目标**：以最终产品字节生成可重复验收器、机器快照、开发日志和验收报告。

**验收**：

- C1 纯契约/几何/身份/negative tests、重点 Stair 集成测试、旧 task plugin 测试及全仓回归全部通过；
- 验收器从产品源与 strict configs 重算 suite/coverage/geometry/contracts/manifests/task/eval/training identity，不用回执自有 hash 自证；
- verifier 不钉死 Git HEAD，不哈希行尾不稳定的原始文本，不导入测试私有方法，也不包含会在 Phase 3B/5 必然变假的字符串黑名单；
- 开发日志明确区分 implemented、local verified、server runtime verified、calibrated、reference qualified 与未完成研究结论。

## 3. 历史 GO / NO-GO（不再作为当前入口门）

按原方案，Phase 3A 只有在 W1–W6 全部通过、真实 C1 reference qualification 已冻结且 C0/旧插件无退化时才会 GO。该门现在已撤销：不会再为独立 C1 cell 执行 W5 的完整资格训练，也不会签发独立 C1 `cell_loop_ready`。P3A 的实现质量由历史证据与 Phase 3R 迁移回归共同确认。

以下任一项直接 **NO-GO**：C1 仍依赖 7×5 网格；实际 mesh 与解析 query 不一致；三种 binding hash 不一致；cell-local 门由候选结果反推；VNext prompt/runtime schema 不同；sealed/unseen 信息进入 LLM；或为了过门放宽旧失败阈值、增加 generation、复制一套 runner。

## 4. 当前实施结论（2026-08-27 架构修正后）

W1–W3 已完成；W4 的原 VNext reward 实现经审计发现不具备官方等价性，因此旧 W5 reference qualification 是“地形与奖励同时改变”的混淆实验。旧 `NO-GO` 仅作为旧字节的失败记录保留，不再代表当前 C1 地形或官方 reference 的能力。

P3A 的 reward 源头修复已完成：C1 使用物理事实与 legacy v1 相同、但含七个关节组角色的 G1-37 v2 profile；VNext 暴露八个按组输入并忠实恢复官方 15 项 reward。P3A 的中间训练 profile 只增加了 easy 入口并覆盖评测最大 grade，仍存在 h/t 单 difficulty 耦合和 raw-distance curriculum，不能直接成为 unified D_train；该部分由 Phase 3R 接管。ordered suite、预算、PPO 和 LLM 权限没有放宽，legacy clean manifest 未改变。

当前状态为 **SUPERSEDED_BY_PHASE3R**，不是待 GPU 的 PENDING。静态等价、定向测试、C1 几何与历史 runtime 证据继续有效；独立 C1 manifest 的 fresh calibration、PPO smoke 和完整 reference qualification 全部取消。Phase 3R 先把这些能力并入统一 G1 Stair protocol draft，Phase 3B 补齐 C2 并激活后，Phase 4 才能启动。详细处置见 [验收报告](PHASE3A_ACCEPTANCE_REPORT_ZH.md)、[开发日志](DEVELOPMENT_LOG_ZH.md) 与 [Phase 3R 修正计划](../P3R/PHASE3R_CORRECTION_PLAN_ZH.md)。

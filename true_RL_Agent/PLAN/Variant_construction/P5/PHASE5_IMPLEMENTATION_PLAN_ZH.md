# Phase 5 实际实施计划：Go2 机器人行与统一 Stair 资格化

> 状态：P5-A～E 已完成；评测语义修正后复用既有 checkpoint 取得 GO
> 基线：`c68460446c4ed52a7e2807bf89b2df2a9e6a0b6f`
> 总目标：在既有“机器人 × 任务”矩阵与统一 Stair LOOP 上增加 Go2 行；不新建 Go2 专用系统，不改变 LLM 仅可修改完整 reward program 的权限边界。

## 1. 本阶段冻结的边界

- RobotProfile 拥有 Go2 的精确 joint/body、默认姿态、执行器、资产闭包、形态尺度、角色候选和关节组。
- RuntimeAdapter/机器人运行时规格拥有 Isaac Lab articulation/env/runner 的上游入口、PPO 有效配置和源码身份；Stair harness 只按 resolved adapter 取工厂，不出现 `if Go2`。
- Stair TaskPlugin 继续独占 terrain family、训练/评测分布、成功/失败、objective/guards 与反馈语义；不得出现 Go2 关节名或 USD 路径。
- RobotTaskBinding 只选择角色、运行时来源及少量物理尺度；不复制 RobotProfile 或任务协议。
- 每个机器人 × Stair 只有一个 unified cell/active protocol；C0/C1/C2 是该任务内部的训练/ID/OOD 数据标签，不再成为额外 cell。
- Phase 5 只做真实 reference 资格化，不调用 LLM、不运行候选 generation；Go2 LOOP 留给 Phase 6。

## 2. 已核实的基线与受控修正

机器人运行时继续选择 Isaac Lab 2.3.0 `UnitreeGo2RoughEnvCfg` 与
`UnitreeGo2RoughPPORunnerCfg`：

- 12 joints、19 bodies、4 support feet；默认 root height `0.400000 m`，root-to-support-center `0.422776 m`；
- `DCMotor`：effort/saturation `23.5`、velocity `30.0`、stiffness `25.0`、damping `0.5`；
- action 为 12 维 joint-position、scale `0.25`；控制周期 `0.02 s`；policy observation 含 187-ray height scan；
- Rough effective reward 为 8 个非零项，但 `flat_orientation_l2=0`；固定预算实测在起始平台即发生姿态/接触失效，没有检验到楼梯几何；
- Flat reward 恢复 `flat_orientation_l2=-2.5`，但仍保留 `lin_vel_z_l2=-2.0` 与较弱的前进跟踪；固定预算实测能稳定站立，却始终停在 level 0；
- 上游 Rough checkpoint 在同一 Stair 环境取得 `J_primary=0.6130`，并在当前训练环境真实触发 C0/C1 课程升级，证明资产、接线、几何和课程更新链可工作；
- Flat 环境本身移除 height scanner，不满足 Stair 观测契约，因此只复用其 reward 函数与权重，观测、动作、PPO 和 runner 仍使用 Rough 配置。

最终 reference 明确命名为 `go2-stair-official-flat-derived-v1`，不冒充原封不动的官方基线。它只做两项可审计的 Stair-context reward 改动：

- `track_lin_vel_xy_exp: 1.5 -> 3.5`；
- `lin_vel_z_l2: -2.0 -> 0.0`。

两项在完整作业前冻结，不声称是唯一或最优权重；reference 仍没有显式 Stair progress、clearance、success、
termination penalty 或分阶段 shaping，保留 Phase 6 reward-only LOOP 的真实改进空间。

同时把先前错误保留为绝对值的 C1 riser offset 从 `0.012 m` 按 Go2 垂直尺度收敛为
`0.007069 m`。课程状态机、0.8/0.4 阈值、C0/C1=50/50、PPO、runner-owned seed 规则、预算和评测门均不改；
协议身份重签后训练 seed 按同一规则重新派生，数值由 `1100090806` 变为 `1262835026`，C1 评测几何也随 profile 重签。
因此本阶段验收的是“派生 reference + 形态一致几何”的最终行配置，不把跨版本结果夸大为 reward-only 因果实验。

Pre-P5 对“可直接用当前 VNext ABI 表达”的判断需要修正：官方四足 air-time 使用 `last_air_time × first_contact`，现有 ABI 只有 current air/contact time。本阶段必须补两个形态中立输入并做真实数值等价，禁止用近似函数冒充官方 reward。

## 3. 工作包与验收

### P5-A：机器人事实与运行时唯一来源

目标：将 P0 已捕获且在当前服务器再次核对的 Go2 事实落为 exact RobotProfile，并建立机器人侧 Isaac Lab locomotion runtime spec。

交付：Go2 profile、runtime adapter、robot registry 项；G1/Go2 的 env cfg、runner cfg、agent configuration 与上游源码入口由同一个机器人侧规格表解析。

验收：

1. profile hash、runtime facts hash、两层 USD closure、12/19/4 schema 与当前 Isaac runtime 一致；
2. profile 的 `forbidden_contact` 提供 base/head/hip/thigh/calf 候选，binding 只选本任务实际禁止的目标；
3. 通用 Stair runtime/execution 不再直接 import G1 env/runner；错误 adapter 必须在建 env 前明确拒绝；
4. G1 原 agent hash、环境类和现有 manifest 身份保持可解释且回归通过。

### P5-B：行 catalog 与薄 binding

目标：删除“从 legacy G1 binding 临时改写出 unified binding”的开发过渡路径，改为显式行 catalog + 每行一份直接 binding。

交付：G1 unified binding、Go2 unified binding、行 catalog；composition 由 `cell_variant_id` 唯一解析 protocol/binding，不复制 launcher。

验收：

1. G1 与 Go2 都能编译为 supported ResolvedCellManifest；profile/adapter/binding/cell/protocol 交叉身份一致；
2. Go2 binding 选 `base`、4 feet、12 actions、articulation root，并声明垂直尺度 `0.589087`、水平尺度 `1.0`；
3. Stair TaskPlugin 和共享 AgentCore/runner/launcher 无机器人特例；
4. 删除 `_unified_binding_from_clean` 等只服务开发迁移的代码。

### P5-C：统一 Stair 分布、评测套卷与 reference

目标：复用相同 C0/C1/C2、8 strata、score、guards 和 acceptance 语义，为 Go2 配置合法的物理难度与显式派生 reference。

设计：

- 依据 `0.422776 / 0.717680 = 0.589087` 缩放 riser 基准包络、C1 riser offset、C2 amplitude 及与高度直接相关的失败阈值；
- tread、水平 patch、forward command 与 slip speed 保持任务尺度不变，避免把“较短踏面”误当成四足必然更容易；
- C0-ID 保留与 G1 完全相同的绝对几何 anchors；其余 cell 使用 Go2 纵向尺度，分数只在本行内用于资格化；
- 三套正式 cohort 仍各 24 cells，几何互不重复、每 stratum 难度阶梯对齐；C2 仍训练零暴露。

交付：Go2 training profile、evaluation suite、VNext reward authority、8-term derived reference program、single active protocol；原 unified loader 通过 component refs 直接加载，不新增分支。

验收：

1. ID cells 落在 Go2 D_train，scalar/joint/phase/C2 OOD 声明均由实际几何关系证明；
2. public_dev/public_accept/sealed_validation 题面不重复且逐 stratum 难度对齐；
3. G1/Go2 绝对 anchor 集合非空并机器可复算；
4. reference 的 7 个未改 active 权重、2 个显式 weight adaptations（其中 `lin_vel_z_l2` 进入 zero-weight）、2 个 zero-weight 和 1 个 disabled term 与 authority 一致；同 transition 逐项 raw/weighted/total 在 `1e-6` 容差内等价；
5. authority 与 runtime projection 的 ABI hash、reward source checksum、manifest lineage 完整闭合。

### P5-D：通用 harness 与真实运行时

目标：让同一 Stair train/eval harness 根据 resolved runtime adapter 实例化 G1 或 Go2，所有任务 overlay 仍只实现一次。

交付：通用 env/runner 工厂接入、source preflight 绑定 task-owned 源码与行级上游源码、Go2 runtime smoke receipt。

验收：

1. Go2 真 Isaac 场景验证 action/observation/sensor/body role、termination、transition decoder、C0/C1/C2 mesh→sensor→evidence 链；
2. train/eval 子进程重载 exact manifest，projection/TrainingBinding/EvaluationBinding/checkpoint 均为同一 Go2 cell 身份；
3. 同一入口切回 G1 仍通过 runtime smoke；不存在第二 launcher、第二 LOOP 或 Go2 专用 evaluator。

### P5-E：reference 资格化与最终回归

目标：从随机初始化完成一次固定预算 reference 训练和分相评测，证明 Go2 行可学习且统一 OOD telemetry 有效。

流程：先做 1-iteration runtime smoke 与真状态 reward 数值 oracle，确认 schema、运行链、reward activation 和预算记账后，直接执行冻结的 2000-iteration reference。单独的短预算趋势训练不能证明最终 learnability、也不会授权调参，因此不再额外烧一份 checkpoint；不得因中途曲线或最终结果不佳临时增加 budget、改 reward、改 PPO 或放宽门。

严格 GO：

1. reference random-init lineage、最终 checkpoint checksum、runtime activation、source manifest 全部可复算；
2. public_accept 的 C0-ID/C1-ID 同时满足 success `>=0.05` 与 goal progress `>=0.50`；
3. OOD 三个 family 给出有限、完整且非结构退化 telemetry；
4. G1 unified Stair、G1 legacy Stair、G1-29 Push 的本地全回归通过；服务器正常环境回归通过；
5. Go2 增量没有引入 Go2 专用 TaskPlugin、protocol loader、LOOP 或 launcher；原固定 G1 的入口只做一次性 row-catalog/adapter 参数化，原 legacy 二足提示词只在 Stair-owned 层改为从真实 V2/VNext contract 派生；任务语义扫描、机器人事实扫描和 G1 parity 均通过；
6. GPU/Isaac 进程清理完毕，local/origin/server 同一提交，P5 receipt、验收报告和开发日志可独立复核。

若 reference readiness 未通过，Phase 5 必须以能力诊断 NO-GO 结束，不伪造 active readiness；只有修正经证据证明属于任务目标定义错误的评测语义，才允许在训练合同完全相同且 lineage 显式冻结时复用 checkpoint 重评，不能用该机制掩盖真实训练失败。

## 4. 实际结果（2026-08-30）

P5-A～D 的机器人事实、薄 binding、统一分布/套卷、reward oracle 与通用 harness 全部通过。更早的
`sha256:2725…` baseline 确实能力不足；后续 `sha256:fca6…` 派生 reference 则已经能完成攀登，低分来自旧
completion contract 额外要求到顶后继续稳定停留，与“首次合法到顶即完成”的任务目标不一致。

修正任务自有 evaluation semantics 后，V1 训练 identity 作为 V2 evaluation-only successor 的唯一前驱；
training profile、seed、PPO、reward、runtime dynamics 与 TrainingBinding 均保持不变。本轮没有重新训练，
只按每行 72 geometry、3/5/7 seeds 和 1080-episode 成本重评既有 checkpoint。active V2 同时删除了
阈值敏感、不能证明物理跌倒的伪 `fall/low_base_clearance` 语义及同源 base 到顶净离地门；真实到顶仍由进度、
平台边界、姿态与顶层支撑共同证明：

- Go2 public_dev/public_accept/sealed：`0.934438/0.933915/0.932641`；
- G1-V2 public_dev/public_accept/sealed：`0.686291/0.694165/0.719514`；
- Go2 `F_scalar/F_c1/F_c2=0.702619/1.0/1.0`，G1 为 `0.714160/0.678748/0.620641`，
  两行三个 family 全部 qualified；
- 两条标准 Session Genesis、successor import 与 `load-only` 重放通过，均 `cell_loop_ready=true`。

G1-V2 与 Phase 4 的 G1-V1 属于不同评测语义身份，分数不作直接纵向比较。服务器正常 Torch/Isaac 环境
全量 `1567 tests` 通过、`3 skipped`（997.823s）。未调用 LLM/API，Phase 6 尚未执行。

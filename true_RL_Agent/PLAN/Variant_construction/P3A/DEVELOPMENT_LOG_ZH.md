# Phase 3A 历史开发日志：C1 irregular、cell-local 门与复杂闭环前置

> 日期：2026-08-27
> 历史范围：`stair.g1_37dof.irregular_v1`；不包含 C2 roughness、Go2 或 Phase 4 多代实验
> 当前状态：**SUPERSEDED_BY_PHASE3R**；保留技术与实验事实，取消独立 C1 fresh 资格化
> 当前入口：[Phase 3R 修正计划](../P3R/PHASE3R_CORRECTION_PLAN_ZH.md)

## 1. 重新冻结的边界

本阶段当时沿用同一条 `AgentCore → authoring → verifier → runner → evaluation → decision → ledger` 主线，并增加了一个可组合 C1 cell。机器人事实继续来自 `RobotProfile/RuntimeAdapter`，Stair 的地形、进度、成功/失败、评测、门和 prompt/memory 仍由 Stair TaskPlugin 拥有。后续审计确认“地形独立 cell”会形成第三轴，因此这一生命周期选择已撤销；task-owned C1 能力保留并在 Phase 3R 并入统一 G1 Stair protocol draft，最终由 Phase 3B 激活。

LLM 权限没有扩大：当前 cell 只开放完整 reward program；PPO、预算、seed、terrain、evaluation、checkpoint、acceptance 和 reference gate 都由 runner/task 固定。C0 的 35-cell V5、旧 reader/adapter、历史 replay 与 Push 插件不迁移到新套卷。

## 2. 实施记录

1. 新增 18-cell 显式有序 C1 suite。cell 自带 cohort、四类 factor、逐级 riser/tread、几何 digest、目标距离与 episode horizon；不再从 `height × tread` 推导环境数。calibration、public_dev、public_accept、sealed_validation 和 unseen holdout 互斥，正式 split 均为 `4 cells × 8 episodes = 32 env`。
2. 新增 task-owned 不等高/不等深、踏面保持平面的训练与评测生成器。评测 mesh、解析 `IrregularFlatStairSurfaceQuery`、recorder/decoder 和 success/progress 使用同一份逐级累计几何。
3. 新增 ordered summary。每个 seed 先对显式 cells 等权，再跨 seed 汇总；public_dev 才进入 prompt/memory，public_accept 只进入 acceptance，sealed/unseen 不公开逐格事实。
4. non-legacy composition 将 manifest 的 `resolved_cell_task_protocol_hash` 同时写入 TaskSpec、EvaluationBinding 与 TrainingBindingV2；legacy cell 继续使用原 base comparison domain。
5. 冻结 C1 acceptance、reference gate、calibration、ordered suite、training profile、VNext reward input/capability 与 cohort hashes，并写入 resolved manifest。
6. 初版尝试将 official G1 rough reward 投影到 VNext；后续审计证明该投影缺少关节组和 single-stance 门控，不能称为官方等价。该问题及源头修复记录在 3.5。
7. train/eval 子进程先重载 exact manifest；runtime projection、reward program、checkpoint lineage、TrainingBindingV2 与 EvaluationBinding 对 cell identity 做真实强制。
8. 新增三段 reference assembly。reference 不伪装 candidate selection，`selection_kind=reference_program_qualification` 且无 acceptance authority；每个 split 仍由 runner 创建一次 phase receipt 后再组装。
9. 新增 calibration/reference/runtime capture 和 Phase 3A snapshot verifier。该 verifier 在 Phase 3A 当时使用 normalized-LF source closure、canonical receipts 与实时重算的 suite/geometry/contracts/manifests；协议退役后收敛为历史证据完整性门，当前 unified 语义改由 P3B/P4/P5 重算。

上述第 4 项只说明同一历史 cell 内四条身份链一致。后续复审发现其 base hash 仍传递性包含 V2 固定 G1-37 reward schema，且部分 materializer 继续读取 V1 `policy_compatibility`；它不再被解释为 task/robot 协议已完全解耦，源头迁移列入 Phase 3R-W1。

## 3. 实施中发现并根治的问题

### 3.1 有序汇总的浮点 reduction tree 不一致

首次 attempt3 calibration 已完成全部真实 rollout，但 builder 直接平均 24 个 episode 浮点，validator 则先按 3 个 seed 求均值；数学等价、二进制浮点不保证 exact equality，最终相差约 `5e-17`。修复为 builder 与 validator 共用“每 seed cell macro → 跨 seed macro”的唯一 reduction tree，并增加非结合浮点回归。失败日志原样保留，未修改阈值或重跑更多 seed。

### 3.2 calibration 的 hard 角色最初不是稳定更难

预注册的 high-riser/long/alternating 组合在真实锚点上进度接近饱和，不能作为 hard。只在 calibration namespace 内、冻结正式门之前，改用 `p2 + target + long + back_loaded` 的 compound cell；原 high/long cell保留为 coverage probe。第三次标定在 easy/target/hard 三项关系上通过后，suite/gate hashes 才冻结。

### 3.3 训练 profile 与真实 Isaac 生成器曾使用相反的 tread 方向

配置和 `StairIrregularTrainingProfile.sequence()` 都声明 tread 随 difficulty 增加，但真实 Isaac 生成器使用 `high - difficulty × range`。因此第一次正式训练的 level 0 实际为 long-tread，训练 summary 却报告由另一公式生成的 short-tread endpoint；这既改变难度，也使声明地形与实际 mesh 不同。

根修是把 riser/tread 插值收敛为 `resolve_irregular_training_sequence()` 一个 task-owned 纯函数，profile receipt 与 Isaac mesh 同时调用；服务器用真实 `trimesh` 端点复核，单测固定生成器必须调用该 resolver。评测 suite、门、reward 与预算均未改变。受污染的第一次资格回执保留为 attempt1，修复后使用全新 session/root，禁止复用旧 train/eval receipts。

### 3.4 证据脚本问题与产品能力分开处理

第一次完整资格运行完成 2000 次迭代和三段评测后，最终回执因调用不存在的 `ReferenceProgramManifest.to_dict()` 失败；改为真实 `to_public_dict()` 后复用已封存回执，未重跑 GPU。训练耗时从 RSL-RL 最后一条 `Time elapsed` 恢复，避免复用后丢失成本。评测成本位于 assembly 顶层，最终脚本将从该权威位置记录，而不是从 metrics 得到 `None`。

最终从仓库根目录复跑验收命令时，发现 verifier 自测只在 `P3A/` 作为当前目录时能导入同目录脚本。修复仅把测试导入根锚定到测试文件所在目录，使文档命令位置无关；产品源码、产品树和冻结 snapshot 均未改变。

### 3.5 C1 reference 奖励并非官方等价，旧失败实验存在混淆变量

审计逐项对照 Isaac Lab v2.3.0 `G1Rewards` 后确认：初版 C1 reference 只有 12 项，官方为 15 项；`dof_pos_limits / dof_acc_l2 / dof_torques_l2 / joint_deviation_*` 的关节作用域被压成全部 37 关节，且 `feet_air_time_positive_biped` 的 single-stance 门控丢失。根因不是 reward 文件少写几行，而是 VNext 当时只暴露 `joint_group.all`，无法表达官方按组语义。

修复没有在 Stair 通用代码里加入 G1 组名或关节名。新增版本化 `unitree_g1_37dof_isaaclab_v2`，物理 runtime facts 与 legacy v1 完全相同，只增加七个 profile-owned `joint_group.*`；C1 RobotTaskBinding 声明八个 `(role, source)` 小映射，通用 runtime projection 只解释该声明并生成 VNext 输入。C1 reference 因而恢复官方 15 项、原权重、原作用域和 single-stance 门控；未来 profile 新增如 `joint_group.thigh` 时无需修改投影代码。legacy clean 继续使用 v1，manifest hash 保持 `af7590ce...282d7`。

为避免再次发生“名字/权重相同但函数语义漂移”，定向测试将 15 个函数 AST 与仓库内已验证的 official baseline 逐项比较，唯一允许的变化是 role 字段重命名。服务器真 torch 对 128 组合成状态的逐项最大绝对差为 `0.0`；双足同时着地的 air-time reward 为 `0`，单支撑为 `0.4`。服务器 Isaac Lab 的 `rough_env_cfg.py` SHA-256 为 `484db0fe...bd6ce`，与冻结上游证据一致。

因此 attempt1/attempt2 的低表现不能再归因于 C1 地形或“官方奖励能力不足”。旧训练/评测回执保留为旧错误字节下的历史失败，不参与当前 qualification。

### 3.6 课程入口过陡且不覆盖评测上界

旧 C1 课程 riser 约为 `0.12–0.23 m`，跨度窄、最易端仍显著陡于 C0，并且训练最大 grade 低于评测上界。P3A 中间 profile 使用 riser `0.05–0.25 m`、tread `0.48→0.33 m`、6 rows、初始最大 level 1；它补齐了 grade 端点覆盖，但仍让 h/t 由同一 difficulty 完全耦合，并沿用 raw-distance curriculum。Phase 3R 因而不会把这份 profile 直接 union，而会改成二维支持和 normalized progress 晋级。

## 4. 历史 Calibration 结果

正式 calibration receipt 为 `PHASE3A_CALIBRATION_RECEIPT.json`，receipt hash `330f9b52ac1ccb5d7cb6a3873946986f897aeee2fb4e8e9461437f5a76480ca4`：

| 角色 | cell | success | mean progress |
|---|---|---:|---:|
| easy | `c1_irregular_03` | 0.5417 | 0.9493 |
| target | `c1_irregular_10` | 0.2500 | 0.9507 |
| hard | `c1_irregular_13` | 0.0833 | 0.8053 |
| coverage probe | `c1_irregular_08` | 0.0833 | 0.9961 |

suite hash 为 `9c3db7e3fdd75fa1260909ba21b9048c1b2022ea34cf27757bdc67fbdfd98a38`，cell-set hash 为 `aa2a57b0b0fe4b1ea26a9079e766e6d32f73cf757ebefc5cbf157002a34b4034`，所有声明的可行二元 factor 组合覆盖率为 100%。这些结果证明同一 evaluation 地形可由 clean anchor 攀登，但 receipt 只属于历史 C1 manifest；它不会 fresh 重签为独立 cell，只作为 unified suite 设计与 terrain feasibility 的输入。

## 5. 旧字节的 Reference qualification 记录

attempt1 在修复前真实执行 `4096 × 24 × 2000 = 196,608,000` env steps，三段 evaluation 也全部完成；结果为 public-accept success `0`、mean progress `0.0314`，reference/sealed gate 均失败。由于训练 mesh 与 profile receipt 的 tread 方向相反，该次结果只能作为失败审计，不能成为最终能力结论。

attempt2 使用相同 suite、门、reward、预算和 seed policy，只修正唯一几何函数并使用新 session/root。它真实完成 196,608,000 env steps，训练循环 3508 s；public_dev/public_accept/sealed 分别耗时 96.934 / 167.142 / 234.229 s。public_accept 成功率仍为 `0`，平均进度从 attempt1 的 `0.0314` 上升到 `0.06064`，但远低于冻结 reference floor `success ≥ 0.20` 且 `progress ≥ 0.35`。`reference_capability_gate` 与 `sealed_validation_gate` 失败，其余 required gates 全部通过，因此最终 canonical receipt hash 为 `cee369bd2f4bc77ff0e33a50b928ed54da1ab561848f6aed3831defd0251befe`，`cell_loop_ready=false`。

训练日志显示 episode 能稳定 timeout，但整个 2000-iteration 课程最终保持 level 0；这与 public-accept 的低进度互相印证。它是旧 reward/课程的失败诊断，不再是当前 C1 reference 的能力结论。当前字节没有降低 train/eval terrain、reference/acceptance floor，也没有增加 generation 或改用 clean checkpoint warm-start。

## 6. 回归与当前证据状态

- 旧字节曾完成 C1 contract/ordered/phased 32/32 与全仓 1522 tests、29 skipped；这些只证明当时工程闭合，不证明奖励语义正确；
- 根修后服务器全量单元回归：`1438 passed, 2 skipped, 1140 subtests passed`，211.82 s；新增 reward AST 等价与 profile-group 定向测试通过；
- shared 层没有 C1/riser/tread/roughness 语义；唯一共享改动只选择 legacy base hash 或 non-legacy resolved hash；
- 旧 evaluation/PPO/runtime receipts 与 snapshot 保留为历史 NO-GO 闭包，不因后续合法产品演进变红；当前能力由后继 Phase 的活契约门判定，且不得重算旧自有 hash 使其冒充当前证据；
- 独立 C1 的 fresh calibration、evaluation scene、PPO smoke 与完整 reference qualification 已因架构修正取消；这不是资源阻塞；
- `git diff --check` 最终复核通过。

## 7. 迁移处置与 Phase 4 边界

- P3A 旧完整训练的 walltime、env steps 与评测成本只作为 unified protocol 的预算输入，不再授权旧 C1 session；
- 不执行独立 C1 pilot、validation 或 reference；旧 `cell_loop_ready` 目标永久关闭；
- Phase 3R 必须先冻结统一 C0+C1 训练分布、ID/OOD suite、OOD objective 与双 ID guards；
- Phase 3B 必须把 C2 作为同一协议的 OOD 校准完成，且证明训练 exposure=0；
- Phase 4 只为 unified G1 Stair protocol 训练一次 official reference，再按 pilot_1 → validation_3 运行 reward LOOP；
- 候选预算、失败重跑和停止条件仍在 candidate 前冻结，能力失败不增加 generation。

因此本日志的旧预算数字只供估算，不再是有效授权。新的权威预算由 Phase 3R 验收回执产生。

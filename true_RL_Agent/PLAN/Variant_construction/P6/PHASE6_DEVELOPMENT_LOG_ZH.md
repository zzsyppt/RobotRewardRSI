# Phase 6 开发日志：Go2 × 统一 Stair LOOP

> 状态：**已完成，GO**；最终提交已同步，服务器独立验收与全量回归均通过
> 实验对象：`stair.go2_12dof.unified_v2`
> fresh validation 源提交：`7e5fa25b607ac118c64424d3e137fa6ce713b2c0`

## 1. 本阶段保持不变的边界

Phase 6 没有新建 Go2 专用 LOOP。Go2 继续使用共享的 generation session、candidate verifier、
runner、phased evaluation、acceptance、ledger 与 current-best；新增机器人的差异只来自既有
RobotProfile、RuntimeAdapter、RobotTaskBinding、reference 与 cell manifest。

LLM 的唯一写权限仍是完整 reward program。训练分布、C0/C1/C2 地形、课程、seed、PPO、预算、
评测套卷、family mask、目标函数、guards 与接受决定均由系统冻结。本轮没有因候选结果改题、调课程、
放宽 guard 或增加 generation。

## 2. 实验谱系

| 用途 | session | source | 结果 |
|---|---|---|---|
| 单代 pilot | `p6-go2-pilot-6508354` | `6508354852138c5828f5b9c80f3d104383479f56` | 完成一代，`formal_rejected` |
| 排除的开发诊断 | `p6-go2-validation3-d8932b9` | `d8932b9de70bbb08375b597bf0952a00f7780b88` | prompt/feedback 语义被后续修复取代，不进入结论 |
| 排除的基础设施诊断 | `p6-go2-validation3-ea089cc` | `ea089cce1ba90e4f7aa553012f8e0da581ea8f67` | generation 0 的历史成本 reserve 误作训练 timeout，不进入结论 |
| fresh validation | `p6-go2-validation3-v4-successor` | `7e5fa25b607ac118c64424d3e137fa6ce713b2c0` | 三代 `accepted / no_selection / no_selection` |

pilot 与 fresh validation 都从 Phase 5 的 Go2 reference 独立创建新 Genesis、ledger、current-best 和
generation lifecycle；validation 没有继承 pilot 的候选、provider history 或 current-best。

共同的科学身份包括：

- reference checkpoint：`sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`；
- initial policy state：`a32148c6a89fbbefbacc5336c95979aa844e4b888b8ed4932af09ce327910744`；
- resolved cell manifest：`193fa9038a6f46539fb71a7e73edc44bdc024be2b899785394b8f82113c91f37`；
- active task protocol：`fb0a30071b27af5b600a7dba65dd03d8a1184ab195c9fd987f770e5eeb667429`；
- reference manifest：`ece5ad653267f11e5b791403208472f42622e44661fdbc7162776799a370dbd4`；
- training binding：`c0112a87a7fb5750ef11d55a6bb50cfdeb0be662380c1387b2d8cab8c977de98`；
- evaluation binding：`62f9d2c837f8ece3dc7dcc7e74742a5aa2089dc4e0aa1db0980cf55e4f7fa36b`；
- acceptance protocol：`d18b5345c8b3112546af401a8dd1560e6c6e840ad8c5ef4d102bd3e1ea5b4c24`。

V2 是 evaluation-only successor，因此 active evaluation protocol 与训练 predecessor protocol 不同是显式
lineage 事实，不是漂移。

## 3. pilot 与 validation 的实际执行

pilot 完成 width 2 的一代真实闭环。被选 lane 的正式分数从 reference `0.933915` 提升到
`0.943657`，三条 named guards 均通过，但 `+0.009742` 小于冻结的最小有效差 `0.013217`，所以
`formal_rejected` 合法成立。该结果证明生产入口、API、双 lane train/eval、正式判决与关账可运行，
不被解释成策略改善。

fresh validation 固定运行三代、每代 width 2。六次训练均为 2,000 iterations、
`196,608,000` env steps，使用相同 seed、初始策略、PPO、binding、terrain/curriculum 和预算；六个 trial
之间唯一被授权的变化都是 `reward_callable`。

第 1 代 lane A `go2_stair_wb_prog_gait_v1` 被接受：

- source checksum：`sha256:4d80d063b9e565637507a208e3e9ab920fe8a7db75693b819a0a999dfdb1bb00`；
- checkpoint：`sha256:2c9e47051de8eb8a70ca58b696689d887d03a01a3ecbaf2836a8feb3077038c8`；
- public_accept `J_primary`：`0.933915 -> 0.974711`；
- public_accept `J_OOD`：`0.900873 -> 0.962066`；
- C0-ID / C1-ID：均保持 `1.0`；
- sealed `J_primary`：`0.975503`，gate pass。

第 2、3 代均为 `no_selection`。第 3 代两条候选虽然 public_dev `J_primary` 分别为
`0.933949/0.936224`，但 `J_OOD` guard regression 分别为 `0.067851/0.064439`，超过冻结上限
`0.05`，因此没有候选进入 sealed 或 current-best。第 1 代 current-best 在后两代保持不变。

## 4. 运行中 blocker 的处理原则

第一次 validation 暴露的不是算法能力失败，而是 imported-reference session 仍保留历史 reference/P6B
operation reserve，并把该 reserve 误当成 candidate train 的第二条 timeout。修复只收敛 runner-owned
成本拓扑与准入：导入 reference 不再预留本 session 不会执行的 reference 工作，candidate train 仍服从冻结的
`7,200 s` 协议 timeout；训练分布、预算步数、PPO、评测、acceptance 与 reward 权限均未改变。失败 session
保持不可变，约 1,400 iteration 的中间 checkpoint 没有被续训或混入 fresh 证据。

fresh run 结束后，跨 Python 审计发现 `statistics.stdev` 派生的 `std/standard_error` 可差 1 ULP。
原 ordered-summary 校验用 Python 字典浮点完全相等，导致同一份原始 episode/cell/seed 数据在该层被拒。
修复仅对这三个派生统计量使用绝对 `1e-12` 的有限数容差；count、protocol hash、seed scores、
cell/stratum/family、objective 与 guards 仍逐项校验。测试同时证明 1 ULP 可通过而 `2e-12` 漂移会被拒绝。
这只关闭 ordered-summary 语义校验，不声称所有 receipt identity hash 已跨 Python 版本规范化；P6 权威复验固定
使用项目 IsaacLab 解释器，系统 Python 3.10 的剩余 1 ULP hash 漂移记录在终态 follow-up。

独立 P6 验收器还补齐两处证据缺口：

1. 从 selected Gen1 的原始 train、public_dev、public_accept、sealed receipt 重新组装 task-owned summary，
   再与 ledger decision、acceptance replay 和 current-best 对照；
2. 隐私扫描覆盖 RoundAuthoringJournal 的全部 10 次 provider attempts，而不是只扫描最终进入 ledger 的
   6 个 prompt。

这些都是在声明的项目解释器下进行的事后可复算能力，不改变实验源码、策略或判决。

## 5. Prompt、记忆与隐私实测

validation 共发生 10 次 provider attempts：9 次通过、1 次 provider failure；4 次 retry 全部发生在
generation 1 lane B 的创作/静态验证/component smoke 阶段，没有 retry train/eval。

- 全部 10 个 prompt payload：`682,434 B`；
- 最终进入 6 条 trial ledger 引用的 prompt：`412,951 B`；
- 第 3 代两条 prompt：`69,241/70,086 B`；task section 都是 `minimum_core`，仍保留 current-best、
  两轮公开历史、family/stratum 与 selector 反馈；
- 公开 prompt 对 private public_accept/sealed 的精确 cell、分数、checkpoint、receipt、服务器路径等扫描均为
  0 命中。

多轮信息没有无限增长，但审计发现若干不影响本轮有效性的表达/投影缺口，已只记录到
`PHASE6_FOLLOWUP_IMPROVEMENTS_ZH.md`。

## 6. 成本与资源闭包

validation 的权威成本为：

- train jobs：`6`；train steps：`1,179,648,000`；
- eval seed units：`55`；job attempts：`19`；provider attempts：`10`；
- simulated control steps：`1,182,443,904 / 1,183,555,008`；
- GPU capacity：`11.736158 / 13.8 h`；
- observed output：`1,766,863,208 B / 549 files`；
- accounted output：`1,767,911,784 / 3,758,096,384 B`。

19/19 attempts 已关闭，open reservations 为 0，双 GPU lease 可同时重新获取；目标 train/eval/provider
进程和 GPU compute apps 均为空。审计期间一条因 PowerShell 提前展开变量而阻塞的只读 `tail|grep` SSH
进程已按精确 PID/PGID 清理并复核消失，它从未占用 GPU 或参与实验。

## 7. 最终复验命令

服务器最终同步后执行：

```bash
/data/yuyangliu/envs/isaaclab/bin/python PLAN/Variant_construction/P6/verify_phase6.py \
  --pilot-output-dir /data/yuyangliu/AGENT/true_RL_Agent/outputs/p6_go2_pilot_6508354 \
  --validation-output-dir /data/yuyangliu/AGENT/true_RL_Agent/outputs/p6_go2_validation3_v4_successor \
  --require-go
```

同时执行 Phase 5 `--require-go`、P6 verifier 自测、全量测试、Push legacy 定向回归，并核对 local、origin、
server 同一最终提交。实验 source 保持 `7e5fa25...`；后续文档/验收器提交只允许把该提交作为 ancestor，
不得把新 HEAD 伪写成训练时 source。

最终结果：P6 verifier `status=pass, phase6_go=true`，P5 `--require-go` 通过；服务器全量
`Ran 1583, OK (skipped=3)`，本地全量 `Ran 1583, OK (skipped=29)`。Push legacy 与共享矩阵定向命令为：

```bash
python -m unittest \
  tests.unit.test_push_recovery_task \
  tests.unit.test_push_recovery_seed_policy \
  tests.unit.test_push_recovery_runtime_adapter \
  tests.unit.test_composition \
  tests.unit.test_robot_matrix_contract
```

当前最终提交在本地与服务器均为 `105/105` 通过。全量回归之后的提交只修改 P6 文档，产品源码与测试字节未变。

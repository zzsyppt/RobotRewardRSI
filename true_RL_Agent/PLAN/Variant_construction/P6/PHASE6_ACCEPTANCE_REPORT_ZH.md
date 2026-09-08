# Phase 6 验收报告：Go2 × 统一 Stair LOOP

> 结论：**GO**
> `loop_verified = true`
> `improvement_observed = true`
> 结论对象：`stair.go2_12dof.unified_v2`

## 1. 结论边界

Phase 6 已证明：Go2 机器人行可以通过与 G1 相同的生产 generation 入口、candidate verifier、runner、
三相评测、acceptance、ledger 与 current-best，在系统冻结的统一 Stair 训练/评测协议下完成固定三代
reward-only LLM LOOP，并得到一次正式接受的策略改善。

本结论不等于整个矩阵已经完成。它没有证明 active G1-V2 已完成独立 LOOP，也没有证明 Go2 × Push、
六足行、final holdout 或任务无关的通用 paired-generation harness；这些继续属于 Phase 7/Core 收口或后续
Extension。Phase 4 的 G1-V1 数字不能冒充 active G1-V2，也不能与 Go2-V2 做跨协议性能排名。

本阶段还系统检查了 G1 常量直接迁移到 Go2 的力阈值、速度/时间、禁碰目标、课程初始等级和四足几何输入，
并识别出若干非阻断风险；缺少控制消融或运行 telemetry 的项目仍保留为观察证据，不宣称这些继承项已经全面
最优或全部关闭，也不在冻结实验中临时改变它们。

## 2. 权威实验身份

fresh validation 使用：

- session：`p6-go2-validation3-v4-successor`；
- source：`7e5fa25b607ac118c64424d3e137fa6ce713b2c0`；
- Genesis：`7fe303cbac06e7f398641227f80b6ac2a23e4407427de049f1a907b46854f0a4`；
- generation contract：`0e7fc4df16833cf59f37fe413ade0745f9e69a12c50575aff3544c46cf71b52e`；
- cost contract：`20ebd3879580f50208acbb3730e10e4950ab3fe6be25814925984624c076d554`；
- source provenance：typed source-checkout receipt，实验 checkout 在运行时为 clean；
- Phase 5 reference checkpoint：
  `sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。

三个 generation 的 manifest、training/evaluation binding、terrain/curriculum、PPO、seed schedule、
预算、objective family mask、acceptance 与 reference 身份均未漂移。六个候选训练都从同一个
initial policy state `a32148c6...` 开始，各自恰好使用 `196,608,000` steps；候选唯一被授权的变化是完整
`reward_callable`。

## 3. pilot 门

单代 pilot `p6-go2-pilot-6508354` 先于三代 validation 完成完整生产闭环：

| 项目 | 结果 |
|---|---:|
| generation | `1`，width `2` |
| 正式 outcome | `formal_rejected` |
| reference `J_primary` | `0.933915` |
| selected candidate `J_primary` | `0.943657` |
| paired delta | `+0.009742` |
| 冻结 materiality threshold | `0.013217` |
| named guards | 全部通过 |

拒绝原因是改善幅度未达到冻结阈值，不是工程错误。pilot 实际消耗 `2` 次训练、`393,216,000`
train steps、`23` eval seed units、`3.938852 GPU h`，全部成本门通过。因此才授权独立的 fresh
validation；pilot candidate、ledger 和 current-best 没有进入 validation。

## 4. 三代 validation 结果

| Generation | lane A public_dev `J_primary/J_OOD` | lane B public_dev `J_primary/J_OOD` | 正式结果 |
|---|---:|---:|---|
| 1 | `0.979183 / 0.968774` | `0.928302 / 0.892452` | lane A accepted |
| 2 | `0.889943 / 0.834914` | `0.879405 / 0.819108` | `no_selection` |
| 3 | `0.933949 / 0.900923` | `0.936224 / 0.904335` | `no_selection` |

Gen1 被接受候选为 `go2_stair_wb_prog_gait_v1`：

| 正式指标 | Phase 5 reference | candidate | 变化 |
|---|---:|---:|---:|
| public_accept `J_primary` | `0.9339153011` | `0.9747107152` | `+0.0407954141` |
| public_accept `J_OOD` | `0.9008729516` | `0.9620660728` | `+0.0611931211` |
| `F_scalar` | `0.7026188549` | `0.8861982183` | `+0.1835793634` |
| `F_c1` | `1.0` | `1.0` | `0` |
| `F_c2` | `1.0` | `1.0` | `0` |
| C0-ID guard | `1.0` | `1.0` | `0` |
| C1-ID guard | `1.0` | `1.0` | `0` |
| sealed `J_primary` | `0.9326411055` | `0.9755029903` | `+0.0428618847` |

正式 paired delta `0.0407954141` 大于冻结阈值 `0.0132169398`；三条 named guards、runtime
activation、budget accounting、Stair behavior、accept evidence 与 sealed gate 全部 pass。因此
`policy_improvement_accept` 与 current-best 更新成立。

被接受实体为：

- source：`sha256:4d80d063b9e565637507a208e3e9ab920fe8a7db75693b819a0a999dfdb1bb00`；
- checkpoint：`sha256:2c9e47051de8eb8a70ca58b696689d887d03a01a3ecbaf2836a8feb3077038c8`；
- resulting current-best hash：`18bb456ae4cea8b2c6304542ace4f63972f09bf91c0c2ec9a754c1abb0cf63bb`。

Gen2/Gen3 的 public_accept 也真实执行，但候选均未满足 frozen selector guard。Gen3 两条候选的
`J_OOD` regression 分别为 `0.067851/0.064439 > 0.05`；系统没有为了追求更多改善把它们送入 sealed，
也没有追加第 4 代。三代终态为 `accepted / no_selection / no_selection`，current-best 在 Gen1 后保持不变。

## 5. 训练、评测与权限证据

六次训练均完成 `2,000` iterations，candidate delta term 有非零 runtime activation。13 次评测使用同一
evaluation/acceptance/seed-cohort 身份，checkpoint hash 均与对应 train receipt 匹配；评测侧
`candidate_code_loaded_in_eval=false`，候选代码没有进入考官进程。

独立验收器在项目声明的 IsaacLab Python 环境下运行，不信任 ledger/terminal report 中的结果摘要，而是对正式 Gen1：

1. 重载 train receipt 与 public_dev/public_accept/sealed 三相 receipt；
2. 重建 EvalJobSpec 和 task-owned phased assembly；
3. 经 Stair TaskPlugin 重新投影 public result；
4. 对照 trial execution decision、acceptance replay、formal projection 与 current-best transition；
5. 校验 source/checkpoint/manifest/binding/protocol/hash 链。

重组得到唯一 formal generation，public result、decision 与 current-best hash 均与冻结证据一致。
Gen2/Gen3 的 selector 与 no-selection 也从两 lane 的原始 public_dev/guard 证据重放。

共享主线的旧任务不退化另由独立定向回归证明。当前本地与服务器分别执行
`test_push_recovery_task`、`test_push_recovery_seed_policy`、`test_push_recovery_runtime_adapter`、
`test_composition`、`test_robot_matrix_contract`，均为 `105/105` 通过。服务器产品树全量回归为
`Ran 1583, OK (skipped=3)`，即 `1580` 项实际通过；本地为 `Ran 1583, OK (skipped=29)`，即 `1554`
项实际通过。全量回归后的提交只修改 P6 文档，最终提交上再次通过上述 `105` 项与 P6 独立验收器，因此
G1-29 Push legacy、共享 composition 与
机器人矩阵契约没有因 Go2 LOOP 退化。

终态所有权扫描同样通过：P5 verifier 从当前 RobotProfile registry 派生 `134` 个具体 body/joint 名并扫描
`83` 个通用任务文件；另外在共享 `agent_core/runner/evaluation/memory/prompting` 中扫描
`go2/g1_37/c0/c1/c2/riser/tread/ankle/torso_link` 等机器人或 Stair 具体标记为 `0` 命中，仓库也不存在
Go2 专用 LOOP/generation/launcher 文件。Go2 差异仍收敛在 row/profile/binding/reference 与 Stair-owned adapter。

## 6. Prompt、反馈与多轮记忆

validation 共 10 次 provider attempts、6 条最终候选 trial。全部 10 个 RoundAuthoringJournal prompt 都被
纳入独立扫描，不只检查最终 ledger 引用：

- prompt payload 总字节：`682,434 B`；
- private phase summaries：`9`；
- private cell-id tokens：`48`；
- private-only 高精度数值 tokens：`755`；
- public_accept/sealed 精确反馈泄漏：`0`。

Gen3 的真实 prompt 仍保留 current-best、两轮公开历史、family/stratum、三个 guards 和 selector 失败原因，
没有退化成只剩全局标量；历史长度保持有界。当前也存在会影响未来改进效率、但不改变本轮有效性的命名、
时间语义、latest-failed source 与人类机制摘要缺口，已按要求只写入 follow-up。

本轮只证明接受候选的 reward delta 与固定协议下的改善相关。虽然 source 结构清楚地记录了 reward 变化，
但没有等预算消融，因此不把 LLM 自述的机制名称当成因果证明。

## 7. 成本与资源验收

| 资源 | 实际 | 上限 | 状态 |
|---|---:|---:|---|
| train jobs | `6` | `6` | pass |
| train steps | `1,179,648,000` | 固定六次完整预算 | pass |
| eval seed units | `55` | contract cap | pass |
| simulated control steps | `1,182,443,904` | `1,183,555,008` | pass |
| GPU capacity hours | `11.736158` | `13.8` | pass |
| job attempts | `19` | contract cap | pass |
| provider attempts | `10` | operator cap | pass |
| accounted output bytes | `1,767,911,784` | `3,758,096,384` | pass |

19/19 attempts 关闭，open reservations 为 0，segment 与 generation lifecycle 都在第 3 代终止；双 GPU
lease simultaneous reacquire 通过，实验 train/eval/provider PID 和 GPU compute apps 均为空。provider usage
和历史 GPU 时刻观测属于带 hash 链的 attestation，验收器只验证 schema 与内部一致性，不夸称能在事后重建
外部账单或过去瞬时状态。

## 8. GO 门逐项结论

| 门 | 结论 |
|---|---|
| Phase 5 `cell_loop_ready` 与 reference 身份未漂移 | pass |
| pilot 先通过成本/闭环门，validation 后独立启动 | pass |
| Go2 使用共享 LOOP，无专用 launcher/系统 | pass |
| LLM 始终 reward-only，runner 拥有训练/评测/接受 | pass |
| 三代固定完成，未按结果扩预算 | pass |
| decision、sealed、checkpoint、ledger、成本、资源可复算 | pass |
| 全 authoring attempts 无私有评测泄漏 | pass |
| G1-29 Push legacy、共享 composition 与矩阵契约不退化 | pass，当前本地/服务器均 `105/105` |
| 机器人/地形具体语义未进入共享主线，无 Go2 专用 LOOP | pass，P5 ownership + 终态静态扫描 |
| `loop_verified` | `true` |
| `improvement_observed` | `true`，仅如实报告 |

因此 Phase 6 判定为 **GO**。下一工程阶段是 Phase 7：补 active G1-V2 的独立 LOOP 证据、一次性
final holdout、Push legacy parity 与不夸大的矩阵报告；不得把本轮 Go2 改善外推成所有机器人或所有任务均已完成。

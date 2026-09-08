# Phase 5 验收报告：Go2 机器人行与统一 Stair 资格化

> 验收日期：2026-08-30
>
> 当前状态：**GO；Go2 与 G1-37 active V2 均已资格化，`cell_loop_ready=true`**
>
> 范围：复用既有 Go2 checkpoint，只重跑评测；未重新训练，未调用 LLM/API，未进入 Phase 6。

## 1. 结论

Go2 派生 reference checkpoint 本身已经具备稳定上楼能力。上一版低分的直接原因，是成功条件仍要求
“到顶后继续稳定停留”；这与“首次合法到达顶部即完成”的任务目标不一致。随后审计又证明旧 `fall`/临时
`low_base_clearance` 只是阈值敏感的运动学净离地标签，不足以宣告物理跌倒。active V2 因此从失败、诊断、
prompt 和到顶条件中完整删除该伪语义。

在最终身份 `0a3f08a…` 上，Go2 与 G1-37 分别复用既有 checkpoint 完成三相正式重评，均通过能力门、OOD
门、sealed 门、标准 Genesis 和 `load-only` 门。两条行都没有重新训练，也没有调用 provider/LLM/API。

同 checkpoint、冻结 V1 协议下的 72-episode `public_dev` 运行现已作为机器验收的一等反事实证据：
`macro_success_rate=0`，task-owned state machine 给出 `fall=22 / forbidden_contact=50`，且视频导出器没有
重分类 outcome。该运行不是 acceptance score，也不声称 72 个 episode 都到过顶部；其中同一 C0 题的 3 个
被跟踪 episode 都达到 `progress_fraction=1.0`，却分别被后续 fall/forbidden-contact 记为失败，直接确认了
“先到顶、后被旧规则否决”的时间顺序。对应视频只重放精确 Isaac state，不重新计算物理。证据见
[72-episode V1 反事实汇总](visualization/go2_derived_c0_id_01/capture_result.json)与
[Go2 C0 旧语义回放](visualization/go2_derived_c0_id_01/go2-derived-c0-id-01.mp4)。`verify_phase5.py`
同时钉住 72-episode 汇总、checkpoint/V1 protocol hash、trajectory 与 video hash，不依赖文档自报。

这里不把更早的 `sha256:2725…` 失败 checkpoint 重新解释为评测问题：它属于早期 baseline 失败。此次结论只
适用于训练已完成的 `sha256:fca6…` 派生 reference。对该 checkpoint，训练、PPO、课程、地形分布和 reward
program 均未改变。

## 2. 修正边界

- 任务成功：首次合法到顶立即锁存并结束 outcome；不要求在顶部平台继续停稳 20 步，也不要求停在特定位置；
- 到顶前仍保留 `forbidden_contact` 与 `off_course` 两条阻断 latch；到顶后的跌倒、滑动或离开平台不再反向抹掉已经完成的攀登；
- V2 已删除 `low_base_clearance` signal、diagnostic flag 与 primary reason：它只是“base 相对推断支撑面低于人工阈值”，不能证明机器人跌倒，且会随阈值选择从几乎全触发变为几乎不触发；
- 真正的姿态异常由 `base_orientation_instability_detected` 表达，真实禁碰撞由 `forbidden_contact` 表达；合法到顶只要求进度/平台边界、姿态有效和至少一个支撑足真实落在顶层，不再设置 base 净离地门；V1 冻结快照继续使用历史 `fall` 名称；
- 支撑面解析仍需要的 penetration tolerance 已迁入任务自有 `support_surface_projection`，不再借用失败语义承载运行参数；
- V1 仍是训练 lineage，V2 是 evaluation-only successor；两者训练 profile、训练 seed、PPO、reward authority、训练 runtime dynamics 与 `TrainingBinding` 完全相同；
- V2 只改变 completion/evidence/evaluation 语义，新的 manifest、resolved task hash 与 EvaluationBinding 独立冻结；
- checkpoint 继承只允许协议声明的唯一 V1 前驱，不能用宽松兼容绕过 manifest、source、capability 或 binding 检查；
- 两条 V1 行在可执行 row catalog 中均为 `active=False`，并分别钉住 protocol hash 与 binding hash；它们只作为历史训练/回执前驱保留，任何字节所代表的协议身份漂移都会在加载时失败。今后的共享语义变化必须新建版本，不能继续修改 V1；因此不增加继承/overlay 合并层；
- Stair TaskPlugin 继续独占地形、成功/失败、objective、guards 与 acceptance；机器人侧仍只拥有 Go2 事实、runtime adapter 与薄 binding。

## 3. 两条 active V2 行的身份与训练复用

| 项目 | Go2 | G1-37 |
| --- | --- | --- |
| 训练 cell | `stair.go2_12dof.unified_v1` | `stair.g1_37dof.unified_v1` |
| 评测 cell | `stair.go2_12dof.unified_v2` | `stair.g1_37dof.unified_v2` |
| V1 manifest | `54cd180c…a3ab1c2` | `cb9c93c1…9ccdae2` |
| V2 manifest | `193fa903…c91f37` | `dbd79e00…0c79a1` |
| V1 resolved task | `1dc2cca0…e77667c00` | `3328f660…181d3b42e` |
| V2 resolved task | `fb0a3007…eb667429` | `32d7f3c9…f3abc0332` |
| TrainingBinding | `c0112a87…c977de98` | `d7aea85d…e045e14c` |
| checkpoint | `sha256:fca6a59c…7fe729` | `sha256:3fbbf80d…810b2` |
| 原训练预算 | `196,608,000 env steps` | `196,608,000 env steps` |
| 原训练 seed | `1262835026` | `1275486543` |
| 本轮训练 / provider 调用 | `0 / 0` | `0 / 0` |

训练兼容回执严格列出允许变化的 evaluation-only 字段，并证明全部训练合同相等。旧 checkpoint 经正式 runner
preflight 进入 V2 评测；其他旧 manifest、错误 binding 或错误 source 仍会被拒绝。

## 4. 最终 V2 三套卷结果

三套卷继续使用 72 个互异 geometry、`3/5/7` 个 seed、每 cell/seed 3 episodes，总计 1080 episodes；
C2 在训练中仍为零暴露。没有修改题目、seed、成本或门槛。

| robot / cohort | episodes | J_primary | J_ID | J_OOD |
| --- | ---: | ---: | ---: | ---: |
| Go2 / public_dev | 216 | 0.934438 | 1.000000 | 0.901658 |
| Go2 / public_accept | 360 | 0.933915 | 1.000000 | 0.900873 |
| Go2 / sealed_validation | 504 | 0.932641 | 1.000000 | 0.898962 |
| G1 / public_dev | 216 | 0.686291 | 0.699153 | 0.679860 |
| G1 / public_accept | 360 | 0.694165 | 0.740130 | 0.671183 |
| G1 / sealed_validation | 504 | 0.719514 | 0.787743 | 0.685399 |

关键验收事实：

- Go2 public_accept 的 `F_scalar/F_c1/F_c2=0.702619/1.0/1.0`；G1 为
  `0.714160/0.678748/0.620641`；两行三个 OOD family 都进入冻结 mask；
- Go2 public_accept/sealed 的 J_primary 标准误为 `0.001565/0.004217`，两卷差 `0.001274`；
- G1 public_accept/sealed 的 J_primary 标准误为 `0.014067/0.009194`，sealed 反而高 `0.025349`；
- Go2 三卷成功数为 `187/310/429`，G1 为 `116/189/274`；两行都没有 `insufficient_clearance` 失败；
- acceptance、budget、runtime activation、Stair behavior、sealed isolation、reference capability 全部通过。

Go2 的 `F_c1/F_c2` 已饱和，真实改进空间主要在 `F_scalar`。以 public_accept `J_primary=0.933915` 计算，
不计 paired SE 的名义 materiality threshold 约为 `0.2*(1-0.933915)=0.013217`；若其他量不变，
`J_primary` 对 `F_scalar` 的系数为 `2/9`，因此约需 `F_scalar +0.0595`。当前 `F_scalar=0.702619`，
数学上仍可达但余量明显较窄。该事实作为 Phase 6 风险输入；不为了制造改进空间而改套卷、reference 或
已冻结 guard，ID=1.0 的 guard 继续只承担非退化约束。
G1-V2 是新评测身份，因此不能把本表与 Phase 4 的 G1-V1 分数当作同一语义下的纵向提升。GO 只表示两条
reference 足以作为各自行 LOOP 的合法锚，不表示能力饱和。

## 5. 标准产品闭包

- Go2：requalification `a9e11c09…23fe99ae`，Genesis hash `db0d7a17…bfdb50b6`，Genesis/file
  SHA-256 `d2000f41…fad5a820`，reference artifact SHA-256 `ef21cee5…4190e78e`；
- G1：requalification `4f195716…1365c0db`，Genesis hash `080154a8…4be41fcc`，Genesis/file
  SHA-256 `53891211…e08ad8a`，reference artifact SHA-256 `b9ad0029…7a3cd9a`；
- 两条 successor import 都只重放 1 份训练 summary 与 3 份评测 receipt，`training_executed=false`、
  `evaluation_executed=false`、provider=0、unexpected execution=0；
- 两条产品 `load-only` 都返回 `status=pass`，Genesis 前后 checksum/mtime 不变，runner/provider=0，ledger 不存在。

机器验收不信任 GO 回执自报：它重新解析当前 V2 cell、V1 predecessor、原训练 summary、三份 phase receipt，
重新聚合 score、生成 family mask、调用任务自有 qualification，再交叉核对 Genesis 与私有 eval artifact；
同一入口还独立复核上述 72-episode V1 反事实及其轨迹/视频字节闭包。

## 6. 回归与最终门

- P5 + Phase5 contract 定向验收：14/14 通过，包含 runtime receipt 与 phase receipt 篡改否定测试；
- 本地 Stair 回归：496 tests 通过、20 skipped；
- 服务器正常 Torch/Isaac 环境全量：1567 tests 通过、3 skipped，997.823s；
- Go2/G1 三相资格化：各 1080 episodes，均 GO；
- 两条标准 Genesis、successor import 与 `load-only`：通过；
- 未重新训练、未追加 generation、未修改冻结预算与套卷、未调用 provider/LLM。

最终机器入口：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe PLAN\Variant_construction\P5\verify_phase5.py --require-go
.\.venv\Scripts\python.exe -m unittest PLAN.Variant_construction.P5.test_verify_phase5
```

两条命令都必须通过后，Phase 5 才保持 `GO / cell_loop_ready=true`。本报告没有执行 Phase 6。

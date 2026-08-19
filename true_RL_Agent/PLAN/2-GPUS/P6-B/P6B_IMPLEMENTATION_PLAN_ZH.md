# P6-B 实施计划：生产 GPU Binding 与单代 Generation Pilot

> 状态：实施中；基线 `e063741366e9ea037b279eb58026b1acb4c56767`
> 授权：用户已批准按 P6-B → P6-C → P6-D 顺序推进；后续阶段只在前一阶段严格 GO、提交/同步和独立审计闭合后生效。
> 边界：本阶段只完成生产接线和一个 fresh development generation；不拼接 P6-A/P6-A2 结果，不运行终局 confirmation，不扩大到同卡 width=4。

## 1. 要回答的问题

P6-B 必须用真实生产链回答：

1. P6-A 证明的两张物理 GPU UUID 隔离能否进入 reference、candidate train 和三阶段 eval，而不是只存在于 shadow；
2. P4/P5 的固定 width=2 coordinator、attempt/lifecycle、selector、ledger 和 exploration 证据能否闭合一个真实 generation；
3. fresh Genesis、parent bootstrap、generation 和 retry reserve 能否在同一个不可重置的 session 成本合同内结算；
4. 旧 Push、legacy width=1、历史 Genesis/ledger/artifact replay 是否保持原行为。

接受或拒绝本代候选都可以；P6-B 的 GO 是工程和证据闭合，不要求产生科学改善。

## 2. 当前系统边界

当前已经有：

- P1 owned process group、parent-death、TERM/KILL/reap、attempt reservation/result 和 GPU lease；
- P2 无 ledger 副作用 trial executor；
- P3 public-dev → selection → public-accept → winner sealed 固定评测拓扑；
- P4 A-first/B-second exclusion、fixed-width generation 和 single commit；
- P5 lifecycle、continuation、cost chain、exploration cards/frontier；
- P6-A full UUID child visibility 与双卡资源实证；
- P6-A2 `deepseek-v4-flash + thinking enabled + reasoning_effort=max + max_tokens=32768` author-only GO。

当前真实缺口是：没有 production generation CLI；host GPU 与 child device 仍共用一个 `execution_device`；正式 runner 未注入 UUID 环境；实际 result 多处从 spec/scope 回抄；reference/private artifact 和 phase 链尚不能携带新 binding；正式 Stair 入口未复用 P6-A 的 renderer/env 约束。

## 3. 设计取舍

### 3.1 唯一公共模型

新增可选、不可变、task-neutral：

```text
GpuProcessBinding(
    host_execution_device,
    process_execution_device,
    resource_binding_hash,
)
```

- host device 只用于资源计划和 lease；
- process device 只用于 child argv；P6-B 两 lane 都是逻辑 `cuda:0`；
- resource hash 与可信 `GpuDeviceIdentity` 交叉验证；
- UUID/physical hash 仍由 `GpuLeaseManager` 唯一拥有，不复制进 JobSpec；
- renderer ordinal、Kit/MGPU 和 Stair env 校验不进入 common。

所有旧 spec/result 在 binding 为 `None` 时完全省略字段，保留旧 canonical bytes/hash。Generation manifest 显式增加 v2；v1 继续只读/回放，v2 才允许两个 child device 相同并以 `binding.host_execution_device + resource_binding_hash` 判两张物理卡不同。禁止 mixed bound/unbound manifest。

### 3.2 执行证据只能来自运行时

`JobAttemptScope`、spec 和 manifest 是预期值，不是执行事实。Bound job 的唯一闭合顺序是：

1. runner 按 host device 从 manager 取得 identity；
2. 验 resource hash、scope 和 binding 一致；
3. spawn 前持有 lease；
4. 只给 child 设置完整 UUID 的 `CUDA_VISIBLE_DEVICES`，删除 `NVIDIA_VISIBLE_DEVICES`；
5. child 在 AppLauncher 前复核环境、逻辑 device 和 task-owned renderer；
6. summary 写实际验证过的 binding；
7. runner 复核 summary 后才形成 TrainJobResult、checkpoint metadata 或 phase result；
8. lease 一直持有到 retry/reap/summary 闭合。

禁止 thread-local、临时修改父 `os.environ`、从 scope 回抄 summary，或把 shadow coordinator搬进生产。

### 3.3 Stair-owned 资源语义

Bound Stair train/eval 使用：

- child `--device cuda:0`；
- `CUDA_VISIBLE_DEVICES=<计划完整 GPU UUID>`；
- host `cuda:N` 派生 renderer physical ordinal N；
- `multi_gpu=false`；
- Kit args 固定 `renderer/multiGpu/autoEnable=false`、`maxGpuCount=1`、`activeGpu=N`；
- summary 同时保存 task-neutral binding 和 task-owned renderer/UUID/PID 证据。

Unbound legacy 入口不自动 remap、不强制 lease、不改变 argv/summary。

### 3.4 Codex 思想的适用边界

只借鉴：副作用前 reservation、单一 owner、整组取消后等待清理、恢复必须沿原身份追加。现有 process/attempt/lifecycle 已经比 Codex 的通用会话语义更适合科学实验，不引入 Tokio/RPC/event bus、TTL、Agent tree、通用 GPU scheduler、DAG 或 rollout DB。

## 4. 唯一传播链

Binding 必须沿以下现有链原样继承，不能局部重建：

```text
generation lane assignment
→ Train/EvalJobSpec
→ attempt reservation/scope
→ SubprocessRunner verified launch context
→ child summary
→ TrainJobResult/checkpoint metadata
→ bind_eval_spec_to_checkpoint
→ materialize_eval_phase
→ public-dev/public-accept/sealed phase result
→ TrialReceiptJournal / generation assembly
```

Reference 路径另要求：

```text
fresh reference train/eval
→ private reference eval artifact payload
→ exact artifact reload
→ Genesis parent-public-dev
```

reload 后 binding 必须字节等价。Generation-vNext CUDA job 缺 binding 在任何 spawn/provider之后的科学副作用前 fail closed；legacy unbound job 保持兼容。

## 5. Session 与成本合同

P6-B 创建 fresh Session Genesis，绑定：

- `session_mode=development_generation_pilot`；
- `max_generations=1`；
- final-confirmation 明确 disabled；
- provider、selector、eval、resource、retry、continuation、cost policy hash；
- fresh空 generation ledger epoch和唯一 output root。

Fresh reference train/eval 发生在 final `session_genesis_hash` 产生之前，不能伪造普通
`JobAttemptScope.session_genesis_hash`。因此只为这条前置执行边界新增窄、task-neutral
`PreGenesisExecutionScope`：绑定 `session_id`、Genesis identity snapshot hash、generation
session/cost contract hash、logical job、预算和 resource binding；Genesis 成功后仍由既有
private reference artifact 把真实结果传递到 final Genesis。普通 `JobAttemptScope`、旧
lease/Genesis/reference artifact 的 schema 与 canonical bytes 不改。

若 coordinator hard-crash 留下 pre-Genesis open reservation，不能直接重跑。Session Genesis
私有 recovery transaction 必须先在同一 creation OS lock 下证明旧 owner incarnation 和
process group 已消失，create-once 发布 recovery receipt，再把精确 open attempt 封为
`abandoned_unclean` 并按最坏预留计成本；仅 reference train 可消费预注册的一次 retry，
reference eval 无 retry 时保持 incomplete。该能力不进入 generation lifecycle 的通用身份，
也不新增 workflow/database；旧 creation-lease v2 partial state继续 fail closed。

该恢复合同持久化的是 coordinator creation owner 与 attempt reservation；child 在 summary
发布前的 PID/PGID 不是 durable receipt。P6-B 的正常单代运行不把这一竞态冒充为“完整孤儿
清理已证明”。若真机在 child 运行中发生 hard crash，续行前还必须由外部只读检查确认旧
coordinator/group 已消失且两张目标 GPU compute-idle；任何无法排除的无 context 子进程都使
session 保持 incomplete。完整 child-launch receipt 级恢复证明留作后续阶段验收，本阶段不为
一个未触发竞态引入常驻 sampler 或第二套进程账本。

Provider 固定 P6-A2 组合：DeepSeek official `/chat/completions`、`deepseek-v4-flash`、thinking enabled、reasoning_effort=max、temperature omitted、JSON object、non-stream、`O=32768`、timeout 300s。每 lane 最多5次，generation最多10次。保守 input reserve `I=300000` normalized units/attempt；A 首请求前原子预留本代 `10I + 10O`，权威 response usage逐次结算。用户已豁免 P6-B–P6-D 金额阻断，但 attempts/units、job、steps、GPU capacity-time和磁盘都不豁免。

完整 fresh-session 预算分栏：

| 部分 | Train | Train steps | Eval seed-units | Episodes |
|---|---:|---:|---:|---:|
| Reference qualification | 1 | 196,608,000 | 15 | 4,200 |
| Parent public-dev bootstrap | 0 | 0 | 3 | 840 |
| One generation | 2 | 393,216,000 | 23 | 6,440 |
| 成功总量 | 3 | 589,824,000 | 41 | 11,480 |
| Retry reserve | 1 | 196,608,000 | 7 | 1,960 |
| 物理 hard cap | 4 | 786,432,000 | 48 | 13,440 |

任何 GPU/API job 前，从服务器既有同 profile receipt 和 P6-A 证据预注册数值化 train/eval timeout、allocated-active GPU-hours、control steps 与 disk reserve；不得看到本代结果后放宽。人工制造 retry 不污染唯一科学 generation：正式前只做一次 production-shaped Linux attempt-chain fault probe；本代只有真实 typed infra failure 才消费 reserve。

正式运行前冻结的非金额资源上限为：

| 资源 | create-once hard cap | 事前依据 |
|---|---:|---|
| allocated-active GPU capacity-time | `8.0 GPU-hours` | P6-A 最慢 paired full-job wall `171.72s / 64 iterations` 按 `2000/64` 保守线性外推，4 次 train 计 `5.96h`；2026-08-13 同 profile reference full eval 的 15 个 seed split wall 合计 `827.526s`，48 seed-units 外推 `0.736h`；余下约 `1.30h` 覆盖 App/summary/phase 启停与 profile 漂移。该口径是 bound operation 的独占物理 GPU capacity-time，不是 `utilization.gpu` 加权算力。 |
| simulated control steps | `795,974,400` | 4 次 train 的 `786,432,000` env-steps，加 48 seed-units × `280 envs × 710 steps = 9,542,400`；按冻结 protocol 精确推导，不留给额外 scientific job。 |
| output bytes | `8,589,934,592`（8 GiB） | P6-A 正式输出约 514 MiB；2026-08-13 fresh reference + 15 rounds（16 train/eval 对）总输出约 5.32 GB。本阶段仅 3 次成功 train、41 seed-units及固定 retry reserve；8 GiB 是 session hard cap，服务器启动门仍要求至少 `max(30GiB, 3×预计峰值)` 可用空间。 |

这些数字在 fresh output 的 session cost contract 中 create-once；resume 只加载并逐字节复核，不接受 CLI 改值。若实际 profile 在硬上限内无法闭合，P6-B 保持 incomplete，不以新 output 或放宽合同续跑。

## 6. 实施顺序

### 6.1 数据模型与兼容

1. `GpuProcessBinding`、JobSpec/result可选字段；
2. generation manifest/resource-plan v2；
3. old v1 exact load/hash、Push和width=1 golden不变；
4. checkpoint/eval/phase/reference artifact exact propagation。

### 6.2 Runner 根接线

1. bound launch context在lease前一次性验证；
2. host lease、child-only env override、parent env不变；
3. summary-derived binding；
4. result/checkpoint/phase receipt只接受validated summary；
5. retry覆盖同一lease/binding直到reap。

### 6.3 Stair 入口

1. 共用一个 task-owned binding helper给train/eval；
2. bound argv/env/renderer/summary；
3. backend materialization与reference artifact reload；
4. unbound旧路径回归。

### 6.4 正式单代入口

新增薄 Stair-owned generation pilot CLI，复用现有 coordinator/lifecycle/provider/Genesis/ledger/frontier，不复制科学状态机。它负责：可信inventory与lease manager、create-once session/resource/cost合同、fresh Genesis/reference资格化、parent public-dev、A/B bindings与scope、一个generation、segment/cost closure、资源诊断和最终report。

CLI 不调用session-end final-confirmation；不改旧 `run_stair_live.py`；不创建通用workflow engine。

## 7. 本地验收

最小高价值测试：

1. host cuda0/cuda1、child均cuda0的manifest v2有效，host或hash相同拒绝；legacy v1 bytes/hash不变；
2. bound runner在spawn前持有正确lease，child env是完整UUID且父env不变，scope/hash/manager漂移均零spawn；
3. summary binding缺失/伪造拒绝，TrainJobResult/checkpoint/三phase来自summary；
4. binding贯穿checkpoint bind、phase物化、TrialReceipt、reference artifact reload；
5. Stair train/eval bound env与renderer0/1正确，unbound argv/行为不变；
6. generation CUDA缺binding fail closed，Push与legacy width=1无binding继续通过；
7. session成本包含3/41成功总量和4/48 physical cap，continuation不重置；
8. coordinator用fake backend完成一代且只有一条ledger，无第二writer。

随后运行受影响合集、全量suite、compileall和`git diff --check`。不增加动态拓扑、renderer组合或大规模故障矩阵。

## 8. 服务器运行门

提交、push、server `ff-only` 后才创建fresh output。启动前同时证明：

- local/origin/server HEAD一致、server tracked clean；
- Linux、绝对Isaac Python、execution environment/Genesis inputs一致；
- coordinator父环境无CUDA/NVIDIA remap；
- inventory恰为计划两张不同UUID，无未知compute PID；
- shared lease root可取得两卡；
- MemAvailable不少于25GiB、swap增长门64MiB、磁盘满足`max(30GiB,3×预计单代峰值)`；
- provider secret只从既有env边界读取；
- create-once合同已冻结provider/resource/cost/retry/continuation；
- 新output、空generation ledger，未启用final-confirmation。

正式运行在独立POSIX session/process group中，stdout/stderr/exitcode写output旁的唯一新文件。监控只读，不主动干预。

## 9. 真实验收与独立重算

运行闭合后至少验证：

- fresh reference、parent bootstrap和一个generation全部使用binding；
- A/B host cuda0/cuda1、UUID不同，child均cuda0；PID只在绑定UUID有compute context；
- generation active wall不超过matched width1 baseline 1.35倍；lane slowdown≤1.25、overlap≥0.90、训练allocated-active GPU-hours ratio≤2.30；fresh Genesis另报完整session wall；
- A→B exclusion和paired-ready barrier可重算；
- public-dev两份证据、selection-before-accept、public-accept、winner sealed、formal decision、唯一ledger row可独立重算；
- loser card/frontier按规则形成或明确不形成；
- provider/attempt/train/eval/steps/GPU-hours/disk成本链未越4/48 physical cap；
- 0未闭合reservation、0孤儿、0 compute PID、0 active lease holder、0 tmux/managed process；
- authority和private/public边界无泄漏；
- historical replay、Push、legacy width=1回归通过。

由独立审计直接从raw receipt重算，不只读取aggregate自报。若本代科学reject但以上全闭合，仍可工程GO。

## 10. Stop/Go

以下任一项立即停止P6-B且不进入P6-C：

- binding不能从summary证明，或跨phase/reference artifact漂移；
- cross-device/unknown PID、lease或进程清理失败；
- duplicate provider/train/eval/ledger side effect；
- selection/decision/ledger不可独立重算；
- session成本超过create-once physical cap；
- 为修复结果必须改变provider、selector、eval、阈值或输入集合；
- 旧Push/width=1/历史replay退化。

只有严格GO、阶段报告/DEV_LOG、精确commit/push/server同步和二次审计都闭合后，才以全新output/Genesis/ledger进入P6-C。

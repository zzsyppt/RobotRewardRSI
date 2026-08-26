# P6-B 实施计划：生产 GPU Binding 与单代 Generation Pilot

> 状态：已闭合（2026-08-22）；工程与证据 GO、科学结果 rejected。正式科学运行提交
> `5f2d7557ca51b73f1f16911e117a9401a103c171`，未来正式启动治理根修基线
> `ba71c5d308f5b498196d53164411f6d7fc0cd542`；历史 prelaunch readiness waiver 保持明示，
> 不补造也不要求重跑约 3.7 GPU-hour 科学链。
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
- P6-A2 `deepseek-v4-flash + thinking enabled + reasoning_effort=max + max_tokens=32768` author-only GO；其 raw receipt 同时暴露出 `2/15` 次 final content 为空而 reasoning 非空的 channel-routing 失败，reasoning 草稿不具候选权威性。

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

### 3.5 离线 preflight 与私有 manifest 依赖

正式 output 创建前必须先以同一 commit、参数、UUID 和 provider env 边界运行
`--preflight-only`。它只允许读 source/inventory/compute-idle/credential，并在自动删除的
隔离 scratch 中构造真实 Stair composition/合同并计算 Genesis identity hash；materialize bound
reference train/eval spec 与合成 checkpoint/result，让 exact reference-v3 private payload、
manifest v2、trial train 和 phase receipt 分别完成 canonical 落盘、严格重载及
hash/argv/checkpoint byte 比对。相邻 baseline/checkpoint 与全部 public/private binding 字段集合
由同 commit 的 S2 projection contract 回归覆盖。preflight 不得取得 GPU lease、启动
runner/Isaac job、发 provider 请求，亦不得创建正式 output、Genesis、attempt、segment、ledger、
cost reservation 或 report；source/GPU 只读探测进程除外。通过只证明装配边界可执行，不是 scientific GO。

当前 `GenerationManifest.manifest_hash` 有意绑定 private lane `to_dict()`，因此包含
`host_execution_device`；public ledger 只保存安全的 `to_public_dict()`。所以 public ledger
本身不能重算该 hash，`validate_generation_ledger_record` 必须加载 ledger 中 checksum-bound
private manifest receipt。P6-B 不改既有承诺，也不把 host topology 泄入 public record；归档、
迁移和纯回放必须把该 private receipt 与 checksum 作为同一证据包保存。若未来需要 public-only
自证，应新增独立 public-projection hash，不能重定义当前 `manifest_hash`。

### 3.6 Provider 账户 readiness 与离线回放边界

`--preflight-only` 的零 API 承诺保持不变；credential 格式正确不能证明账户当前可用。正式
fresh run 在 source checkout 校验之后、任何正式 output/foundation/cost reservation/lease/Genesis
或 runner job 之前，必须通过 task-owned DeepSeek 账户门：使用同一固定 secret 边界向官方
`GET /user/balance` 发一次 authenticated、bounded、拒绝 redirect 的非生成请求，并严格要求
`is_available is true`。HTTP 401、402、redirect、TLS/network/timeout、无效 JSON/schema 或
`is_available=false` 全部 fail closed，不在正式进程内自动重试；其中 402 按 DeepSeek 官方语义
归为余额不足。该 GET 不计 authoring attempt/provider token/cost reservation，不保存 credential、
原始响应或任何余额数值，也不进入公共 ledger、prompt、provider policy 或 acceptance。

CLI 另提供 `--provider-readiness-only`，让同 commit/参数在正式 output 不存在时独立保存脱敏
启动证据；正式入口仍会再次强制检查，不能用旧 readiness 结果代替当下账户状态。已有最终 report
或已有合法非空 generation ledger 的 settle/replay-only resume 跳过网络门，保持闭合 artifact 的
离线可读性；ledger 为空、仍可能产生新 GPU/provider side effect 的 incomplete resume 必须重查，
失败时不得修改既有 formal output。chat transport 原有 402 fail-closed 仍保留，以覆盖 readiness
检查和实际 authoring 请求之间的状态变化。

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

Provider 固定为 P6-A2 的 endpoint/model/输出合同，并把 reasoning mode 冻结为显式 adaptive：每个 lane 首次请求仍是 thinking enabled、reasoning_effort=max、temperature omitted；持久 receipt 同时证明 final content 为 0 bytes、reasoning 非空且不是 `finish_reason=length` 时，下一 attempt 改为 thinking disabled。既有的 length+inline-source repair 也保持 thinking disabled。所有 thinking-disabled repair 都显式移除 reasoning_effort，并固定 `temperature=0.2`；enabled/disabled 两种真实请求形状均进入 provider policy hash。reasoning 内容永不解析、物化或作为候选。JSON object、non-stream、`O=32768`、timeout 300s 不变。每 lane 最多5次，generation最多10次。保守 input reserve `I=300000` normalized units/attempt；A 首请求前原子预留本代 `10I + 10O`，权威 response usage逐次结算。该修复不能免掉第一次 channel-routing 失败的成本，只阻止同类失败在同 lane 重复。用户已豁免 P6-B–P6-D 金额阻断，但 attempts/units、job、steps、GPU capacity-time和磁盘都不豁免。

完整 fresh-session 预算分栏：

| 部分 | Train | Train steps | Eval seed-units | Episodes |
|---|---:|---:|---:|---:|
| Reference qualification | 1 | 196,608,000 | 15 | 4,200 |
| Parent public-dev bootstrap | 0 | 0 | 3 | 840 |
| One generation | 2 | 393,216,000 | 23 | 6,440 |
| 成功总量 | 3 | 589,824,000 | 41 | 11,480 |
| Retry reserve | 1 | 196,608,000 | 7 | 1,960 |
| 物理 hard cap | 4 | 786,432,000 | 48 | 13,440 |

任何 GPU/API job 前，从服务器既有同 profile receipt 和 P6-A 证据预注册数值化 train/eval timeout、allocated-active GPU-hours、control steps 与 disk reserve；不得看到本代结果后放宽。人工制造 retry 不污染唯一科学 generation：正式前只做一次 production-shaped Linux attempt-chain fault probe，并把命令、唯一 scratch、两个 create-once attempt result、retry 链头和零遗留进程保存为启动门证据；本代只有真实 typed infra failure 才消费 reserve。

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

CLI 同时提供完全无 GPU/API 副作用的 `--preflight-only` 和不发生成请求、不触碰 GPU/正式
状态的 `--provider-readiness-only`；正式 child 只能由 tracked
`launch_stair_generation_session.py` 启动，使每次物理 attempt 用独占创建的新目录、log 和 exit
receipt；launcher 复用 runner 已有的整棵进程树 ownership/回收原语，不另造进程协议。CLI 不调用session-end final-confirmation；不改旧 `run_stair_live.py`；不创建通用workflow engine。

## 7. 本地验收

最小高价值测试：

1. host cuda0/cuda1、child均cuda0的manifest v2有效，host或hash相同拒绝；legacy v1 bytes/hash不变；
2. bound runner在spawn前持有正确lease，child env是完整UUID且父env不变，scope/hash/manager漂移均零spawn；
3. summary binding缺失/伪造拒绝，TrainJobResult/checkpoint/三phase来自summary；
4. binding贯穿checkpoint bind、phase物化、TrialReceipt、reference artifact reload；
5. Stair train/eval bound env与renderer0/1正确，unbound argv/行为不变；
6. generation CUDA缺binding fail closed，Push与legacy width=1无binding继续通过；
7. session成本包含3/41成功总量和4/48 physical cap，continuation不重置；
8. coordinator用fake backend完成一代且只有一条ledger，无第二writer；
9. `--preflight-only` 对正式 output 零写入/零 lease/零 spawn/零 provider，并让所有 binding 私有投影完成 JSON 严格往返；
10. 每种持久化 binding 明确断言 public 两字段或 private 三字段，segment exception exit 持久化 path/secret-scrubbed 类型与消息，历史 exit v1 继续回放；
11. launcher 连续两次运行形成两个独占 attempt/log/exit receipt，第一次字节不变；失败 child 的真实 Python traceback 与 returncode 可恢复；Linux active-child SIGTERM 后 child+grandchild 均无 live PGID 且 receipt hash 稳定；cleanup 无法证明时必须记 `cleanup_incomplete` 且不发布 child log hash。
12. provider balance GET 使用通用 bounded/no-redirect transport，但不扩宽 `JsonTransport`/
    `CandidateProvider` 插件协议；`is_available=true/false` 和 HTTP 402 分类、脱敏投影、fresh
    failure 早于首个 formal write，以及 closed-ledger/final-report resume 跳过网络分别回归。

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
- 同参数 `--preflight-only` 已通过且正式 output 仍不存在；
- 同参数 `--provider-readiness-only` 已通过并保存独立脱敏证据，官方 balance response 的
  `is_available=true`；不得保存余额数值或把该 GET 计入 scientific provider attempt；
- production-shaped Linux attempt-chain fault probe 已通过并保存独立证据；
- 当前 `run_owned_process` 必须再用一次固定 128-step 的真实 Stair/Isaac smoke 证明正常退出后
  PGID、目标 GPU compute PID 和 shared lease 全部清零；
- create-once合同已冻结provider/resource/cost/retry/continuation；
- 新output、空generation ledger，未启用final-confirmation。

上述两个 probe 使用固定、单一职责的 task-owned 入口：

- `scripts/run_p6b_attempt_chain_fault_probe.py` 只走现有 `SubprocessRunner` 的真实 retry/owner
  路径，留下两次 owned child/grandchild、`summary_missing -> retry -> terminal` create-once 链、
  retry 前序 hash、terminal 链头和零遗留 PGID；它不碰 GPU、lease、provider 或 Isaac。
- `scripts/run_stair_train_teardown_probe.py` 必须由与正式运行相同的绝对 Isaac Python 启动，
  显式传入单张 host device、完整 GPU UUID、同一 `nvidia-smi` 和同一 shared lease root；它是
  现有 `isaaclab_stair_rsl_rl_smoke.py` 的薄 owner，运行固定
  `16 env x 8 step x 1 iteration = 128` workload，不复制训练主体、不修改正式 196M-step 合同。

双卡 inventory/compute-idle、shared lease 同时 acquire/release/reacquire、MemAvailable、磁盘和
prelaunch swap 窗口仍作为独立 launch-readiness 证据保存，不扩张 attempt-chain driver 的合同。
正式 launcher 运行期间另以独立周期采样保留
`swpd`/GPU/PID 证据，并相对该次启动基线执行 64 MiB swap 增长门；任何时点越界即停止本次
session，不把最终时刻的回落值冒充全过程通过。两条 probe 和正式 launcher 的完整 argv、
source commit、host/boot identity、关键源码及产物 checksum 都保存在各自 fresh evidence root。

正式运行以 `setsid` 启动 tracked Stair launcher，并显式传入预注册的 session `--timeout-s`；
launcher 为每次物理 attempt 在 output 旁的 `<output-name>.launcher-attempts/` 中独占创建新
attempt 子目录，分开保存 launcher traceback、child stdout 和 child stderr。已有 runner
process-owner 在正常退出、timeout、Ctrl-C 或 launcher 异常时先回收该 child 的完整进程树，
随后才计算三个 log checksum 并以 `O_EXCL` 发布含 returncode/PID 的 `exit.json`。禁止用 `>`
复用固定 `.launcher.log` 或覆盖旧 attempt 证据。该 receipt 保证覆盖 child-active 的受控退出，
不把 SIGKILL、掉电或恰落在 receipt 自身发布窗口的 hard crash 误报为可闭合。监控只读，不主动干预。

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
- 归档证据包保留 checksum-bound private generation manifest receipt；不得把 public-only ledger 误报为可独立重算 `manifest_hash`；
- historical replay、Push、legacy width=1回归通过。

由独立审计直接从raw receipt重算，不只读取aggregate自报。若本代科学reject但以上全闭合，仍可工程GO。

## 10. Stop/Go

以下任一项立即停止P6-B且不进入P6-C：

- binding不能从summary证明，或跨phase/reference artifact漂移；
- cross-device/unknown PID、lease或进程清理失败；
- duplicate provider/train/eval/ledger side effect；
- selection/decision/ledger不可独立重算；
- session成本超过create-once physical cap；
- preflight 失败、正式 output 被 preflight 污染、launcher attempt log/exit receipt 缺失或覆盖、或出现 `cleanup_incomplete`/未哈希 child log；
- provider readiness 未通过、HTTP 402/`is_available=false`，或 readiness 失败后仍创建 fresh
  formal output/foundation/reservation/lease/runner side effect；
- 为修复结果必须改变provider、selector、eval、阈值或输入集合；
- 旧Push/width=1/历史replay退化。

只有严格GO、阶段报告/DEV_LOG、精确commit/push/server同步和二次审计都闭合后，才以全新output/Genesis/ledger进入P6-C。

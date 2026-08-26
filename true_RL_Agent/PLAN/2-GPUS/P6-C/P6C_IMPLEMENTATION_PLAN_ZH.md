# P6-C：3-generation 有界验证实施计划

> 状态：已完成。最终 P6-C formal session 在提交 `415bfad` 上以 3 条连续 generation、4 个
> lifecycle segment 闭合；计划 active TERM、唯一恢复、边界暂停、终态 replay、成本与资源清理均
> 通过独立 G4 审计，结论为 `GO_FOR_P6D`。前两个失败 session 继续原样保留为 incomplete 证据，
> 没有跨提交迁移或拼接。最终证据见 [`P6C_ACCEPTANCE_REPORT_ZH.md`](P6C_ACCEPTANCE_REPORT_ZH.md)。
> 下文保留实施时冻结的科学合同、四段编排与验收条款，不用最终结果倒改历史计划。
>
> 上位约束：以 `PLAN/2-GPUS/dual_gpu_generation_upgrade_plan_zh.md` 为唯一总计划。本文件不能放宽其会话合同、双 lane、验收、资源、成本、故障处理或用户门控要求；发生冲突时以上位计划为准。
>
> 实现基线：P6-B 启动证据完整根修提交 `ba71c5d308f5b498196d53164411f6d7fc0cd542`；P6-C 不在更早提交上开工。

## 1. 阶段目标与结论

P6-C 只回答一个问题：在 P6-B 单代已闭合的科学协议上，固定宽度 2 的 generation 是否能在同一新会话中连续运行 3 代，并保持 parent/head、预算、探索、恢复和资源证据可重算。

P6-C 是 development validation，不是正式终局实验：

- 创建全新的 development Genesis、session contract、cost contract、ledger 和 output root；
- 固定 `max_generations=3`，不得继承、拼接或续写 P6-B ledger；
- 每代仍复用已经冻结的 generation-v1 双 lane 科学协议；
- 禁止运行 final qualification/final confirmation，禁止声称正式终局资格；
- 三代期间不得根据观察修改 selector、screen/public-accept/sealed 协议、value/frontier 规则、temperature、attempt cap、阈值或成本合同；
- 任何需要改变冻结语义的发现都结束本 session，经重新审计后创建新 Genesis。

当前公共 generation、ledger replay、lifecycle、exploration projector 和 fixed-width coordinator 已具备单代原语。P6-C 的主要实现工作必须留在 Stair 任务层：把当前硬编码的一代 P6-B session 驱动扩展为一个固定三代驱动。不得另造公共 DAG、工作流引擎、runner、selector、frontier 数据库或第二份科学真相源。

## 2. 范围边界

### 2.1 本阶段允许修改

- Stair generation session/cost/report 的版本化合同与任务层驱动；
- Stair 的 parent guard 回放、generation-scoped 输出路径和 session-wide 计数；
- P6-C 所需的 resume、generation-boundary pause、hard-crash recovery 接线；
- task-owned、launcher 内只观察的 tracked runtime monitor 和 fresh launch-readiness 证据；
- 针对三代会话的定向测试、历史 P6-B v1 回放测试和实施日志。

### 2.2 本阶段明确不修改

- 公共 `execute_fixed_width_generation` 的一代科学语义；
- generation-v1 ledger record、既有 acceptance replay、Stair selector、eval/sealed 协议和阈值；
- A-first/B-second、每 lane 最多 5 次 authoring attempt、B exclusion、both-candidates-ready barrier；
- Push、legacy width=1、round-v7 writer 和旧任务插件；
- P6-B 已闭合 artifact、ledger、cost/report schema 或其字节；
- final-confirmation gate；
- 任何通用 scheduler、DAG、daemon、分布式 runner 或独立 frontier store。

公共层只有在定向测试证明存在真正任务中立的缺口时才允许最小修改；不得把 Stair 的三代策略、priority/high 服务规则或 P6-C 成本常量放入公共层。

## 3. 版本与唯一权威

P6-C 冻结以下版本边界：

| 对象 | P6-C 版本策略 | 说明 |
|---|---|---|
| generation session contract | 保持 `true_rl_agent.generation_session_contract.v3`，新建 create-once 实例 | 固定 `session_mode=development_generation_validation`、`max_generations=3`、`max_trials=6`、`max_train_jobs=6`；其余 selector/eval/provider/cost/retry/resource/continuation/final-plan 继续由 bound contract hashes 冻结，不升级公共 schema |
| generation cost contract | 新增 `true_rl_agent.stair.generation_session_cost.v2` | 显式拆分 successful budget、session-wide retry reserve 和 physical hard cap |
| P6-B cost contract | 严格保留 v1 loader/projector | 必须能按旧字段集、旧 hash 和旧语义回放；不得用 v2 默认值“兼容”旧制品 |
| generation ledger | 保持 generation-v1 | 每代恰好零或一次 head transition，不复制另一份 winner/head |
| invocation segment receipts | 保持现有 generation lifecycle 版本 | 复用 start/exit/recovery chain，不发明 P6-C 私有生命周期 |
| launch readiness | 保留 P6-B v1 严格 loader；P6-C 如需升级则新增 v2 | v2 只增加 `fresh_absent / resume_existing` launch subject：fresh 绑定 output 不存在，非终局 resume 绑定既有 Genesis/ledger/lifecycle/cost heads；不得放宽 v1 |
| provider policy | 绑定已批准的 provider policy v2 及其 hash | 三代内不修改 temperature、model、attempt cap 或计量边界 |
| terminal report | 新增 P6-C report v3 | 按三代列表与 session 聚合报告；保留 P6-B report v2 严格回放 |
| final qualification | disabled | P6-C 不创建、不调用、不模拟终局确认 |

唯一权威关系：

- session contract 是会话级冻结事实；
- generation manifest 只引用 session contract，并绑定本代 parent、base context 和 lane intent；
- ledger 是闭合科学事实和 current-best head 的唯一权威；
- attempt/resource/cost/lifecycle journals 是 operational 权威，不进入科学 ledger；
- EvidenceCard、value band、frontier、统计报告均由上述权威制品重算，不持久化第二份可写状态。

## 4. 冻结成本合同 v2

### 4.1 成功路径、retry reserve 与物理硬门

以下数值写入 P6-C create-once cost contract v2，不得在 continuation segment 中重置：

| 计量项 | 成功 session budget | session-wide retry reserve | physical hard cap |
|---|---:|---:|---:|
| generations | 3 | 0 | 3 |
| train jobs | 7 | 1 | 8 |
| train env-steps | 1,376,256,000 | 196,608,000 | 1,572,864,000 |
| eval seed-units | 87 | 7 | 94 |
| eval episodes | 24,360 | 1,960 | 26,320 |
| simulated control steps | 1,393,551,600 | 197,999,600 | 1,591,551,200 |
| allocated-active GPU-hours | 11.0 h | 1.5 h | 12.5 h |
| output bytes | 3,221,225,472 B（3 GiB） | 536,870,912 B（512 MiB） | 3,758,096,384 B（3.5 GiB） |

成功路径组成固定为：

- reference qualification：`1 train job / 196,608,000 train env-steps / 15 eval seed-units`；
- parent bootstrap：`3 eval seed-units`；
- 三代迭代：`6 train jobs / 1,179,648,000 train env-steps / 69 eval seed-units / 19,320 episodes`；
- final confirmation：禁用。

retry reserve 固定为 `R_train_jobs=1`、`R_eval_seed_units=7`，即一次与 P6-B 相同策略的 reference-train 同参数 retry 加一次 7-seed sealed retry。它是整个 session 共用的执行 reserve，不是每代、每 lane 或每个 sealed scope 各自获得一份；“每个 logical job 最多一次”只是全局上界，不会给 P6-B 冻结策略中 retry eligibility 为零的 public-dev/public-accept/candidate-train operation 新增重试资格。

GPU/disk 数值来自 P6-B `5f2d755` 的同类实测并加简单 headroom：成功路径投影 `8.64662003057 GPU-h`，乘 `1.25` 后为 `10.8082750382 GPU-h`，冻结为 `11.0 GPU-h`；retry 原始最坏 `1.15902337304 GPU-h`，乘 `1.25` 后为 `1.44877921630 GPU-h`，冻结为 `1.5 GPU-h`。成功输出投影 `2,341,181,253 B`，乘 `1.25` 后低于 `3 GiB`；retry 输出 `326,023,821 B`，乘 `1.25` 后低于 `512 MiB`。

正式 launcher 的 session `--timeout-s` 对每个物理 launch segment 预注册为 `43,200` 秒（12 小时），所有 fresh/resume/recovery launch 使用同一数值，不得看到运行结果后放宽。依据是同 profile P6-B 成功正式 attempt 的 create-once `exit.json` 已绑定 `timeout_s=43200`，完整一代实测墙钟约 2 小时 38 分；P6-C 每个计划 segment 至多推进一个尚未闭合的 generation。该 session timeout 不是额外 GPU、retry 或成本额度，内层 job timeout 与本节累计 hard cap 继续独立 fail closed。

证据根：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_5f2d755_20260822T025737Z`

关键证据：

- `generation_actual_cost/0002-segment-12f94bfc119645f2b9675790daaa1b09-settlement.json`；
- `job_attempts/*/attempt_00.result.json`；
- `runner/*`。

### 4.2 Provider 硬门

P6-C 沿用已批准的 `I=300,000`、`O=32,768`，冻结：

- provider attempts：`<=30`；
- request/input units：`<=30I=9,000,000`；
- response/output units：`<=30O=983,040`；
- total units：`<=30(I+O)=9,983,040`。

上位计划已经记录 `money_governor=waived_by_user_20260820`，因此 P6-C 不另造虚假金额门；该 waiver 只覆盖金额，不豁免 attempts、provider units、train/eval、steps、GPU-hours、磁盘或资源硬门。

### 4.3 计费与预留规则

- 每个 train、screen、public-accept、sealed operation 启动前预留完整最坏成本；
- A 首次 provider 请求前，原子检查本代 A+B 共 10 次 attempt 的完整 provider 余额和 session 总余额；
- failed、partial、retry 均累计；无法可信测量的 partial 按该 operation 的完整最坏预留计入；
- 每个 logical job 最多一次同参数 typed-infra retry，同时受 session-wide `1 train / 7 eval` reserve 限制；
- retry entitlement 必须从 create-once cost contract、attempt predecessor chain 和累计 settlement 重算，不得由当前 generation 或 scope 局部生成；
- continuation segment 只继承同一余额；不足时保持 incomplete，不得借用未来预算、增加第三次 attempt 或为了跑满三代超限；
- 每代结束和每次 segment exit 都写入累计 cost-chain head，并从全部 generation/attempt artifacts 重算实际消费。

## 5. Stair 任务层三代驱动

不得复制一个独立的 P6-C runner。应把当前 Stair 一代 session 驱动收敛为一个固定模式驱动：P6-B 历史制品仍按旧版本回放，新建 P6-C session 只允许 `development_generation_validation / max_generations=3`。

每次进入或恢复 session 时执行：

1. 加载并严格验证 create-once session/cost/provider/resource contracts；调用参数不得覆盖已冻结值。
2. replay lifecycle 与 generation ledger，构造任务中立的 continuation cursor，得到 `current_best`、`next_generation_index`、ledger height 和 terminal 状态。
3. 初始 guard 只来自 Genesis parent bootstrap；顺序回放已有 generation：accepted 时用现有 Stair next-generation guard loader 加载已验证 winner public-dev，rejected/zero-train 时保留上一 guard。
4. 从 ledger 和版本化 projector 重建 EvidenceCard/frontier；不得从 report 或可写缓存恢复。
5. 在 provider/GPU 副作用前完成本代最坏 admission；不满足时写 operational pause/incomplete 证据，零 provider、零 GPU 副作用。
6. 为本代生成冻结的 A/B lane intent 和共同 base context，调用一次现有 fixed-width generation coordinator。
7. reload/replay ledger，要求高度只增加一、generation ID 唯一、index 连续、frozen parent 仍是 current head；不允许半条 record。
8. 完成本代 acceptance replay、resource closure、attempt-chain closure、cost settlement 和 exploration/report projection。
9. 任一硬门失败立即退出本代循环并写受控 pause/failure；不得开始下一代。
10. 若存在 boundary-pause 请求，写 segment `paused/operator_pause`；否则继续下一代。
11. 只有 ledger 恰好闭合 3 代且所有硬门通过时，写 segment `completed/normal` 与 terminal report v3。

### 5.1 ID、目录和 segment 归属

- generation ID 固定由 session identity 与 `001..003` 推导；不得硬编码只有 `generation-001`；
- provider candidates、candidate artifacts、lane journals、authoring evidence 和 operational metrics 必须按 generation ID 隔离，避免三代覆盖或错误复用；
- 新启动的 logical job 绑定当前 invocation segment ID；不得继续使用“首个 generation segment ID”作为新 job scope；
- 已闭合并被合法复用的旧 attempt 保留原 segment/attempt identity，不重写归属；
- parent bootstrap 在 fresh session 只运行一次，resume 不重复生成。

## 6. Resume 与恢复语义

### 6.1 正常 resume

- 只有显式 `--resume-existing` 才允许进入已有 output；fresh 模式要求目标 formal output 不存在；
- terminal report 存在时只做离线严格复验，不检查 provider readiness，不触发 provider/GPU；
- ledger 高度为 0、1、2 时，从 ledger/lifecycle/cost/attempt chain 重建状态；只要后续可能 author，就必须重新通过 fresh provider/source/credential readiness；
- 同代 A 已闭合而 B 未闭合时，复用 A evidence 并继续 B；不得重 author A、跳过 B 或降为 width=1；
- A/B 已闭合而 trial 部分完成时，只复用 hash/receipt/attempt chain 完整的结果；不得重复 closed provider/train/eval 或追加第二条科学选择；
- provider actuals、prompt bytes、attempt counts 和 cost 必须遍历全部 generation，不得只扫描第一代目录；
- 恢复后每次 ledger append 仍需同锁 replay 与 parent CAS。

### 6.2 Generation-boundary pause

P6-C 的 clean user-controlled pause 明确定义为 generation 边界暂停：

- 启动时只允许指定固定的 `pause_after_generation=N`；本阶段不增加运行时请求文件、控制 socket 或第二条暂停通道；
- active generation 不因该请求被截成科学 partial；当前 generation 完成 closure/replay/resource/cost 检查后停止；
- 确认所有 runner-owned process group 已安静、GPU lease 已释放后，写 create-once segment exit：`status=paused, reason=operator_pause`；
- resume 必须是显式的新 launch，并沿同一 session/lifecycle/cost chain 重新通过当次启动门；若该 boundary pause/resume 已预注册在同一冻结阶段方案内，则沿用本阶段条件授权，不再增加人工轮询或重复审批通道；
- continuation policy 冻结 `max_segments=4`，余额和 generation 上限均不重置；超过上限保持 incomplete。

### 6.3 活跃 TERM 与 hard crash

对正在 author/train/eval 的 session 发送 TERM 不是 clean pause。P6-C 冻结以下分类：

- launcher/parent-death/process-group 机制负责尽最大可能终止并 reap 当前树；
- 若无法由 session 正常完成 generation boundary 和 create-once pause exit，则该次按 hard crash/no-exit 处理；
- 新 segment 必须取得 exclusive lease、证明旧进程均已死亡并发布唯一 recovery receipt，才可恢复；
- partial trial 不进入科学 ledger，旧 start/attempt 不删除、不覆盖、不补造；
- chain、成本或进程归属无法证明时 session 保持 incomplete。

本阶段不实现通用可抢占 runner，也不得把 TERM 捕获后简单改写成 `operator_pause`。若未来要求“活跃训练中立即且干净暂停”，必须作为新的任务中立取消协议单独审计，不能在 P6-C 启动修复中顺手扩权。

## 7. 每代自动硬门

每个 closed generation 后、开始下一代前，至少自动验证：

- ledger replay、generation ID/index、parent CAS、head transition 和 record hash；
- A/B parent、initialization、seed schedule、evaluation/resource binding 完全一致；
- selection receipt 先于任何 public-accept，sealed subject 只等于冻结 winner；
- attempt predecessor chain、logical retry 次数和 session-wide reserve；
- runner-owned process group 均关闭，目标 GPU 无本 session 遗留计算 PID，lease 可证明释放；
- train/eval/steps/GPU-hours/output/provider usage 与累计 cost-chain head；
- public projection 不含 private path、host binding、raw response、sealed 或 secret；
- EvidenceCard/frontier 可从权威 ledger/receipt 重算，infra/partial 未生成科学卡；
- verifier-equivalent negative 未再次进入训练；
- report/projection 与 authoritative public-dev/public-accept 事实一致。

任一项失败立即暂停；不得在同一 session 修改阈值、输入或合同后“修好继续”。

## 8. P6-C report v3

报告包含 3 代明细和 session 聚合，但不成为新权威。所有字段必须能由 ledger、authoring/attempt/resource/cost receipt 和 prompt refs 重算。

### 8.1 必报探索与 selector 指标

- priority/eligible 各自 arrival、served、service failure、expiry、当前 backlog/deferred；
- `high_card_recall`：符合预注册 high 条件且应建卡的 qualified loser 中，正确建卡比例，硬门 `100%`；
- `revisit_lane_service`：generation 开始时存在兼容 priority/high item 时，服务值必须为 `1`；medium/eligible 未自动服务不计失败；
- `revisit_yield`：同时报告分子与分母，不设置小样本显著性门；
- 完整负向 verifier-equivalent trained repeat：硬门为 `0`；同时报告被 verifier 阻断的 equivalent attempt 数，不能把正确阻断算作训练重复；
- 同输入负向 advisory 命中率；
- public-dev/public-accept 一致率；
- immutable prompt payload bytes 和相对冻结计划的额外 eval 成本；
- 每代 selector/material reversal：若至少两代具有完整双公开证据，且 winner 按冻结 materiality 明显落后 loser，或 winner 公开门失败而 loser 通过，则阻断 P6-D；不得回写已闭合 winner/acceptance。

### 8.2 必报 width=2 操作指标

- A/B authoring 总墙钟；只接受 authoring journal 中的 create-once timing facts，禁止用文件 mtime 推断；若现有 journal 缺少必要事实，只新增窄的 task-owned operational timing receipt，不创建控制状态；
- B exclusion repair rate；
- paired-ready rate；
- same-delta 与 semantic-equivalent-delta 阻断数；
- zero-train authoring-incomplete generation 数；
- 每代 train/eval/provider 实际工作量、retry 消费、GPU-hours 与输出增长。

这些指标只用于判断串行双选是否带来有效 width=2 和 P6-D 是否可进入，不允许在本 session 内调参。

## 9. Tracked runtime monitor 与真实启动门

### 9.1 Tracked runtime monitor

真实 P6-C 不允许依赖临时 shell、未跟踪脚本或人工观察作为唯一运行证据。实施阶段新增仓库内 tracked、测试覆盖的 Stair-owned `runtime_monitor.py`，由正式 launcher 在自身进程内以只读线程运行；不新增 daemon、tmux、socket、HTTP heartbeat 或第二套进程清理协议。顺序固定为：验证 readiness → 写 attempt `start.json` → monitor 首样本 → spawn formal child → 绑定 child PID → 周期采样 → child/process-group 清理 → post-exit 样本 → create-once `stop.json` → launcher `exit.json`。

- 证据只写入现有 launch attempt 下的 `runtime_monitor/`：create-once `start.json`、`session.json`、`stop.json` 与单 writer、连续 sequence、逐行落盘的 `samples.jsonl`；
- 记录 launch identity、launcher/session PID/PGID/SID/incarnation、目标 GPU 原始 snapshot、compute PID ancestry、owned/unknown PID，以及 lifecycle/ledger/attempt/cost 最新权威文件的 path/hash/size；
- monitor 只观察，不取得 lease，不发送 signal，不拥有 kill/retry/resume/cost/science authority，不按进程名白名单判断归属，也不创建第二套 GPU-hour、成本或 hard-cap 真相源；
- 不实现易误报的“无进展 watchdog”，也不由只读 observer 宣称 stale lease；monitor 异常不改写科学结论，但证据不完整时 launcher 对本次正式启动 fail closed。

通用 `run_owned_process` 如需接线，只允许新增默认关闭、任务中立的 `on_started(pid)` 回调；Stair 字段和证据语义不得进入公共 runner。monitor 必须先经过无 GPU fixture 和 production-shaped Linux process probe，再随实现 commit 一同 push/sync。没有完整 tracked monitor 证据的真实 P6-C 启动一律阻断。

### 9.2 每次 launch-readiness

每一次可能继续触发 provider/GPU 副作用的真实 launch（fresh 或非终局 resume/recovery）都必须在服务器、在准确的待运行 commit 上重新生成 launch-attempt-scoped readiness evidence；terminal report 已存在且只做离线复验时除外。P6-B、fresh attempt 或任一较早 P6-C attempt 的 readiness 均不可复用：gen1 后 resume、计划 hard-crash 后 recovery、gen2 后 resume 的 child argv/hash 与既有 subject head 各不相同，必须分别生成新 receipt。该证据先写入 formal output 之外的独立 launch-evidence root；fresh 模式绑定“预期 output 不存在”，非终局 resume/recovery 模式绑定既有 Genesis/ledger/lifecycle/cost heads，绝不能为了证明 readiness 而预创建或改写 formal output。readiness 至少绑定：

- 服务器 commit identity、关键执行文件 hash、tracked monitor hash 和实际 argv；本地/origin/服务器三方 commit 一致性由同一启动证据集中的独立同步检查证明，不伪装成 readiness receipt 自身字段；
- fresh output root 不存在；resume 时则绑定既有 Genesis/ledger/lifecycle/cost heads；
- `--preflight-only` 的真实实现路径全部通过，formal output、lease、Genesis 和 cost reservation 在 preflight 前后均无副作用；
- provider source/credential 只读 readiness 通过，secret 未进入 argv/log/receipt；
- 两张目标 GPU compute-idle、无未知 PID、UUID 与 resource plan 一致，coordinator 父环境未预 remap；
- `MemAvailable >=25 GiB`、swap 增长硬门 `<=64 MiB`；
- 剩余磁盘使用唯一 readiness 合同 `max(30 GiB, 3 * max_output_bytes)`；不得在计划、launcher 和 receipt 中维护第二套等价阈值；
- production-shaped Linux attempt-chain fault probe 已通过并保存独立证据；
- 一个短的真实 Stair train 已验证 Isaac/Kit 正常 teardown、process group 安静、GPU lease/PID 无残留；
- runtime monitor 已在 formal child spawn 前写入首样本，并确认能观察本次 launch identity。

readiness 只授权该 commit、该 output、该 launch attempt。任何代码、合同、GPU identity、输出根或启动参数变化都使它失效，必须重做。本阶段已有的条件授权只在全部启动门为绿且无重大问题时生效；若条件不再满足，GPU/provider 启动立即阻断并报告。

## 10. 实现文件计划

### 10.1 必改

- `src/true_rl_agent/tasks/stair_climbing/generation_session_cost.py`
  - 增加 v2 success/retry/physical 三层合同和严格重算；
  - 保留 v1 字段集与语义的独立严格 loader/projector。
- `src/true_rl_agent/tasks/stair_climbing/generation_session.py`
  - 把一代 pilot driver 泛化为固定 1/3 代 task-owned session loop；
  - 增加 guard/cursor replay、generation-scoped path、session-wide provider/retry、每代硬门、resume、boundary pause 和 report v3；
  - 修正 provider actuals 与新 job segment identity。
- `src/true_rl_agent/tasks/stair_climbing/generation_bootstrap.py`
  - 仅在能减少重复且仍为 Stair-owned 时增加一个小的 guard-chain replay helper。
- `scripts/run_stair_generation_session.py`
  - 暴露固定 P6-C validation mode、显式 resume 和 generation-boundary pause；不得允许任意调用者改 `max_generations`。
- `scripts/launch_stair_generation_session.py`
  - 绑定 fresh readiness、attempt-scoped log、tracked monitor 和 boundary-pause argv；不得把 active TERM 冒充 clean pause。
- `src/true_rl_agent/tasks/stair_climbing/runtime_monitor.py`
  - 新增 launcher 内运行、只观察的 tracked monitor；不得拥有 session lease、科学状态、retry 或 kill 权限。
- `src/true_rl_agent/runner/process_handle.py`
  - 仅在需要时增加默认关闭、任务中立的 `on_started(pid)` 回调；默认行为与旧任务路径不变。

### 10.2 测试与文档

- `tests/unit/test_stair_generation_session_cost.py`；
- `tests/unit/test_stair_generation_session.py`；
- 新增一个小型 production-shaped 三代 integration test，使用 fake provider/runner，但经过真实 JSON、hash、落盘、重载、ledger、attempt、cost 和 lifecycle 边界；
- monitor/launch-readiness 的 Linux process、PID ancestry 与日志不可覆盖测试；
- `PLAN/2-GPUS/P6-C/DEV_LOG.md` 记录每次实现、测试、审计和服务器证据；
- 只有 P6-C GO 后才更新总计划状态。

### 10.3 删除或替换的开发偏假设

- 删除新执行路径中的 `_run_one_generation` 单代状态机假设；
- 删除所有新执行路径中的硬编码 `generation-001`；
- 删除 `_first_generation_segment_id` 作为新 job scope 的用法；
- 替换 pilot-only terminal report builder；旧 report v2 只保留严格回放；
- 删除会与 generation-scoped 目录冲突的 `provider_candidates/<lane>`、`candidate_artifacts/<lane>` 等单代写法；
- 不新增 `run_p6c_generation_session.py` 克隆，不保留两套会逐渐漂移的 Stair session driver。

## 11. 最小验收矩阵

| 组别 | 必须通过的证据 | 失败时 |
|---|---|---|
| Cost schema | P6-B v1 strict replay；P6-C v2 exact round-trip/tamper；本计划数值逐项断言 | 阻断实现 GO |
| Fresh 三代 | 新 Genesis/ledger；3 条连续 generation；bootstrap 一次；无 final confirmation | 阻断 P6-C |
| Parent/head | accepted/rejected/zero-train 组合下 guard 与 current-best 正确；append 时 parent CAS | 立即暂停 |
| Resume | ledger 高度 0/1/2；A 已闭合/B 未闭合；trial partial；无重复 provider/train/eval/append | 保持 incomplete |
| Segment identity | 新 job 绑定当前 segment；复用 attempt 保留原 segment；recovery chain 唯一 | 立即暂停 |
| Provider | 每代 A 前能覆盖 10 attempts；三代累计不超过 30/I/O/total；缺 usage fail closed | 零新副作用或暂停 |
| Retry | 整个 session 只共享 1 train/7 eval；partial 最坏计费；耗尽后后代不得重新获得 | 保持 incomplete |
| Admission | 任一 operation、本代或 session hard cap 不足时，在 provider/GPU 前停止 | 零新副作用 |
| 每代 closure | ledger/acceptance/resource/process/cost/exploration 自动复验；下一代只在全绿后开始 | 立即暂停 |
| Exploration | high-card recall=100%；有兼容 priority/high 时 service=1；negative equivalent trained repeat=0 | 阻断 P6-D/按硬门暂停 |
| Report v3 | 三代指标可独立重算；tamper 被拒；selector reversal 规则精确 | 阻断 P6-D |
| Boundary pause | gen1/gen2 后 clean pause、资源安静、resume 无重复 | 阻断真实启动 |
| Active TERM | 独立验证 child group 清理；无 pause exit 时归类 hard crash；recovery 后才续跑 | 证据不足则 incomplete |
| Monitor/log | tracked monitor 只读；每 attempt 独立日志；旧 traceback 不覆盖 | 阻断真实启动 |
| Legacy | P6-B v1 cost/report/artifact、Push、width=1、round-v7 全部不退化 | 阻断实现 GO |
| Server gate | fresh readiness、attempt-chain probe、短 Stair teardown、commit/hash/argv 对齐 | 禁止 GPU/provider |

测试应覆盖真实序列化与重载边界，不得通过 mock 掉被验函数来宣称 preflight、binding round-trip、process cleanup 或 resume 已通过。新增测试保持聚焦；不为三代穷举无意义组合，不扩展成通用工作流测试框架。

## 12. 实施顺序与用户门控

1. **合同先行**：实现 cost v2、report v3 结构，并保留 P6-B cost v1 与 report v2 的严格 legacy loader；先通过 exact/tamper 测试。
2. **任务驱动**：实现三代 loop、guard/cursor replay、generation-scoped path、current segment identity。
3. **累计治理**：实现 session-wide provider/retry/cost admission 与每代自动 closure。
4. **恢复语义**：实现高度 0/1/2 resume、部分 generation 重入、boundary pause 和 hard-crash recovery 分类。
5. **运行证据**：实现 tracked monitor、attempt-scoped logs 和 fresh launch-readiness。
6. **定向验证**：运行本计划验收矩阵中的无 GPU/fake-runner/production-shaped Linux 测试。
7. **全量回归与 legacy replay**：记录 candidate baseline、base commit、测试/skip 数、受影响文件 hash。
8. **三视角独立审计**：authority/evaluation、runner/resource、prompt/memory；确认无第二真相源、无任务语义下沉、旧插件不退化。
9. **提交与同步证明**：工作树、origin、服务器 commit 和文件 hash 一致。
10. **真实启动前门**：服务器 fresh readiness、tracked monitor、attempt-chain probe、短 Stair teardown 全绿并保存独立证据。
11. **条件授权核对**：记录 2026-08-22 已给出的“无重大问题且启动门全绿即可进入后续 Phase”授权；满足前述条件即可启动真实 P6-C。若发现重大问题或需要改变冻结语义，授权条件不成立，必须停止并报告。

## 13. 阶段完成定义

P6-C 实现 GO 需要同时满足：

- 本文冻结合同已由 create-once artifact 表达且可重算；
- 三代 task-owned driver、resume、session-wide retry/provider 和每代自动硬门通过验收；
- clean pause 只发生在 generation boundary，active TERM 被诚实归为 hard-crash recovery；
- tracked monitor 与 fresh launch-readiness 是运行 commit 的已跟踪、已测试组成部分；
- P6-B v1、Push、width=1、round-v7 严格不退化；
- 无 final qualification、无新公共 DAG/runner、无第二科学真相源；
- 本地、origin、服务器身份一致，真实启动门证据齐备；
- 已有条件授权仍适用，且实现/服务器审计未发现重大问题、全部真实启动门均通过。

真实 P6-C 即使闭合三代也不自动授权 P6-D。若 priority/high 服务、negative equivalent、selector reversal、资源/成本或报告重算触发本文硬门，P6-C 必须停在 development evidence，不得通过改写闭合结果或会话内调参绕过。

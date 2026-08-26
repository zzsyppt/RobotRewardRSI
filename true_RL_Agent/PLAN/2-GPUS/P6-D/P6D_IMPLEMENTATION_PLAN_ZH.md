# P6-D：正式 15-generation 会话实施计划

> 状态：P6-B/P6-C 独立复核触发的主线减法、科学反馈修正和本地全量回归已经完成；当前停在新一轮
> 用户静态审计前，禁止继续
> 真机实验。实现父基线为 `415bfad6eb6e85c6cf0cf94851c6b39d1067116b`。此前基于
> `16175c152430018469c22d1866d8e1ca7d4149db` 的一次预启动在 reference train 期间被人工停止，未产生
> generation ledger、provider attempt 或科学 head；其 output 不得在本轮源代码变化后 resume。
>
> 入场依据：P6-C 源提交 `415bfad6eb6e85c6cf0cf94851c6b39d1067116b`；独立 G4
> 审计结论 `GO_FOR_P6D`，审计 SHA-256
> `ad691299c0ca5b08383f984a68f514d987c1831cdd3b24d367a9ca34d7266a93`。
>
> 上位约束：`PLAN/2-GPUS/dual_gpu_generation_upgrade_plan_zh.md`。本文件只能收紧或落实
> 上位约束，不能放宽科学协议、权限、成本、证据或用户门控。

## 1. 本阶段回答的问题

P6-D 只回答：在 P6-C 已验证的固定 width=2 科学协议不变时，系统能否用一个新正式会话连续闭合
15 个 generation，并在终态用预承诺、与迭代 split 不相交的同一 7-seed cohort，对 Genesis 与最终
current-best 各做一次 runner-owned 配对评测，形成可独立重算但不改变科学 head 的诊断结论。

P6-D 不是新的搜索算法，也不借 15 代扩大候选权限。以下内容整个会话冻结：

- 两 lane authoring、B exclusion、static verifier、one-component policy；
- public-dev selector、parent-relative guard、public-accept、winner-only sealed 和正式 acceptance；
- exploration card/frontier 投影及其 prompt 权限；
- provider policy v2、每 lane 最多 5 次 attempt；
- Stair train/eval 协议、seed namespace、阈值、runner 与 GPU binding；
- 成本合同、生命周期拓扑、final-confirmation Plan 和源/环境身份。

观察到任何科学结果都不能在本 session 内改规则。需要修改上述语义时，本 session 保持
incomplete 或闭合为原结果，改动只能进入新的 Genesis。

## 2. 明确不做

- 不新增通用 DAG、scheduler、daemon、RPC、数据库或第二份 frontier/ledger；
- 不复制一个独立的 15 代 driver；P6-C 与 P6-D 共享同一 task-owned bounded-session engine；
- 不让 parent 直接启动 Isaac final eval；所有正式 train/eval 仍由现有 trusted runner 创建子进程、
  绑定 lease、单 GPU 可见性、process group 和 receipt；
- 不修改 Push、legacy width=1、round-v7、P6-B cost/report 或 P6-C cost/report 的既有字节语义；
- 不把 final confirmation 变成 acceptance、pass/fail 或下一代 prompt；
- 不为了穷举故障组合建立新的测试框架，也不主动制造额外 GPU 故障来“用完”恢复余量。

## 3. 新会话身份与单一真相

正式启动必须使用：

- 新 output root、新 session ID、新 Genesis、新 ledger epoch 与空 generation ledger；
- `session_mode=formal_generation_session`、`max_generations=15`；
- 运行 commit 的 tracked、clean source checkout；
- 两个运行时发现并冻结的物理 GPU UUID；child 均使用 logical `cuda:0`；
- Genesis 前 create-once 的 final-confirmation Plan v2；
- create-once cost v3、continuation policy v2、generation session contract 和资源计划。

ledger 仍是唯一科学权威。attempt/result、trial receipt、lifecycle、cost settlement、exploration
projection 和 final-confirmation artifact 只能证明或诊断 ledger，不得自行推进 current-best。

## 4. 冻结成功路径与物理 hard cap

### 4.1 成功科学路径

| 阶段 | Train jobs | Train steps | Eval seed-units | Episodes |
|---|---:|---:|---:|---:|
| Reference qualification | 1 | 196,608,000 | 15 | 4,200 |
| Parent public-dev bootstrap | 0 | 0 | 3 | 840 |
| 15 generations | 30 | 5,898,240,000 | 345 | 96,600 |
| Final confirmation | 0 | 0 | 14 | 3,920 |
| **成功 session 总量** | **31** | **6,094,848,000** | **377** | **105,560** |

每代最大 2 train、2×3 public-dev seed-units、2×5 public-accept seed-units和 1×7 winner sealed
seed-units。零训练或无 winner 的 generation 只减少实际使用量，不能把余额改作额外 trial。

### 4.2 Retry reserve 与总物理上限

P6-C 的全部 17 个 job 均为 attempt 0；它没有给出真实 retry 消耗，因此 P6-D 延续已验证的最小
session-wide reserve，而不按 15 代线性放大：

- 最多 1 个同参数 train retry：1 job / 196,608,000 train steps；
- train/eval 共用治理下，最多 1 个 7-seed eval-phase retry：7 seed-units / 1,960 episodes；
- 任一 logical job 仍最多一次 typed infra retry；session-wide reserve 先耗尽者阻断其余 retry；
- failed/partial execution 按实测计费，无法可信测量时按该 operation 的完整预留保守收费。

因此物理 hard cap 为：

| 项目 | 成功 cap | Retry reserve | 物理 cap |
|---|---:|---:|---:|
| Train jobs | 31 | 1 | 32 |
| Train steps | 6,094,848,000 | 196,608,000 | 6,291,456,000 |
| Eval seed-units | 377 | 7 | 384 |
| Episodes | 105,560 | 1,960 | 107,520 |
| Simulated control steps | 6,169,795,600 | 197,999,600 | 6,367,795,200 |
| Allocated GPU capacity-hours | 49.0 | 1.5 | 50.5 |
| Output bytes | 13 GiB | 512 MiB | 13.5 GiB |

control-step 上限使用冻结 profile：每 eval seed-unit 198,800 control steps，并与 train steps 相加。

### 4.3 Provider cap

provider policy v2 的 `I=300,000`、`O=32,768` 不变：

- attempts `<=150`；
- request units `<=45,000,000`；
- response units `<=4,915,200`；
- total units `<=49,915,200`；
- money governor 继续记录 `waived_by_user_20260820`，但 attempts、units、reasoning usage 与所有
  GPU/step/disk cap 均不豁免。

每代 lane A 首次 provider call 前仍原子预留本代 A+B 最坏 10 attempts/units；额度不足时零新请求。
continuation、恢复或 finalization 都不得重置累计余额。

### 4.4 GPU 与磁盘数值来源

49.0 GPU-hours 成功 cap 与 1.5 GPU-hours retry reserve 来自 P6-C 真实 attempt 中各 operation
最大 resource-lease 时长乘 1.25，再按 P6-D operation multiplicity 汇总；不是把 P6-C 总时长简单乘 5。

13 GiB 成功输出 cap 来自 P6-C 的 runner job、attempt、public-dev、public-accept、sealed 和非 runner
证据逐类最大目录增长乘 1.25 后，按 P6-D multiplicity 汇总并向上取整；retry 独立保留 512 MiB。
cost v3 的 source evidence 必须绑定 P6-C session、提交、G4 审计 hash 与逐 operation 实测表。

## 5. 生命周期与恢复

### 5.1 固定正常拓扑

正式会话不接受任意 `max_generations` 或任意暂停点。正常路径固定为五段：

| Segment | Generation | 正常结束 |
|---|---|---|
| 0 | 1–3 | `paused/operator_pause` |
| 1 | 4–6 | `paused/operator_pause` |
| 2 | 7–9 | `paused/operator_pause` |
| 3 | 10–12 | `paused/operator_pause` |
| 4 | 13–15 | generation lifecycle `completed/normal` |

continuation policy v2 create-once 冻结暂停边界 `(3, 6, 9, 12)`，`max_segments=7`。五个正常段之外
只提供两段 generation recovery 余量；它们不是额外 generation 或预算，也不要求主动制造故障。
超过余量、链分叉、segment 并发或 identity/head/cost 不一致时会话保持 incomplete。

### 5.2 Recovery readiness 的含义

`recovery readiness` 是 resume 前的只读启动门，不是训练阶段。它必须证明：

- 请求继续的是同一 output、session、Genesis、contract、final Plan、ledger epoch、commit、环境与
  GPU resource plan；
- 最后闭合 generation、ledger head、lifecycle head、actual-cost head 与全部 create-once job receipt
  一致，无 open reservation、并发 owner 或不可解释的活进程；
- 上一 unclean segment 只能被唯一恢复一次，已完成 job 只复用，不能重跑或另开 attempt；
- 续跑 argv/hash 与下一固定边界一致，累计 provider/retry/cost 余额足以覆盖下一原子 operation；
- 两张目标 GPU 无未知 compute PID，RAM/swap 与剩余磁盘门通过。

它只决定“可否安全续跑”，不修改 ledger、receipt、预算或科学结果。

### 5.3 终态 finalization 恢复

第 15 代闭合后 generation lifecycle 永久 terminal，任何恢复都不得重开 generation 或 append ledger。
若 final confirmation 尚未形成唯一 Result，formal resume 只允许进入 finalization：复用 create-once
Reservation 与已有 attempt receipt，补齐未完成的 Genesis/final eval 或 Result。该路径仍获取同一
session owner/lease并重新做 readiness，但不创建新的 generation segment。

Result 与最终报告已经闭合后，只允许 `--terminal-audit-only`；普通 resume 必须拒绝。

## 6. Final confirmation v2

### 6.1 Plan

Plan v2 必须在 Genesis 前 create-once，并由 Genesis/session contract hash 绑定：

- 仅适用于 `formal_generation_session` 与 15 个闭合 generation；
- 冻结与迭代 public-dev/public-accept/sealed 均不相交的 final 7-seed namespace/cohort；
- 冻结按 `Genesis checkpoint -> final current-best checkpoint` 的顺序；
- 冻结 terminal Reservation schema、result schema和 diagnostic-only 权限；
- 不读取未来 head，因此不能预先猜测最终 checkpoint。

P6-C 与旧 15-round live 的 final-confirmation v1 loader/bytes 保持不变。

### 6.2 Reservation

仅当第 15 代、ledger replay、lifecycle、exploration、resource 与 generation cost terminal audit 全绿时，
create-once Reservation v2。它至少绑定：

- Plan/session/Genesis/ledger checksum 与 generation contract hash；
- 实际 terminal ledger/head/lifecycle/cost heads；
- Genesis 与最终 current-best 的 checkpoint、checksum、source/training/capability lineage；
- final cohort、两个 immutable phase spec hash、完整私有 GPU process binding 与 output paths；
- source/environment manifests 和 single-use result anchor。

Reservation 创建后 checkpoint 或 terminal head 漂移必须 fail closed；不能重新创建一个“修正后”版本。

### 6.3 Runner-owned execution 与 Result

两次 eval 在同一物理 GPU 上顺序运行，child 仍为 logical `cuda:0`，并复用现有
`SubprocessRunner.run_eval_phase`、lease、process group、attempt/result 和 task evidence：

1. Genesis checkpoint，final namespace 的 7 seeds；
2. final current-best checkpoint，同一 7 seeds。

Stair eval 入口只增加 task-owned 的 final-confirmation purpose/namespace 选择；generic runner 不认识
Stair seed 语义。两个 job ID 保持 sealed-validation 计费类别，但 Result v2 明确它们不是 generation
acceptance sealed query。

Result v2 只保存完整 canonical summaries、每 seed paired delta、mean、standard error、identity/hash 和
资源收口证据。它没有 `accepted`、`passed` 或 head transition 字段；不写 generation ledger，不更新
current-best，不进入本 session prompt。

## 7. 版本化实现边界

### 7.1 Task-owned 改动

- `generation_session.py`
  - 抽取一个内部 bounded-session engine；保留 P6-C wrapper，新增固定 formal config/wrapper；
  - formal 固定 15 代、边界 3/6/9/12、report v5 与 terminal finalizer；
  - 不把 Stair 科学策略移入公共层。
- `generation_session_cost.py`
  - 新增严格 cost v3 与 formal operation admission/replay；cost v1/v2 loader 不变；
  - actual usage 继续复用可重算的 v2 usage schema，加入 final execution receipt root。
- `final_confirmation.py`
  - 新增 Plan/Reservation/Result v2 及严格 loader；v1 路径原样保留。
- `phased_evaluation.py` 与 Stair eval script
  - 以默认参数保持原 formal namespace；只有明确 final purpose 时使用 final namespace。
- `launch_readiness.py`
  - 新增 formal readiness v3；严格区分 fresh、nonterminal resume、terminal finalization 与
    terminal audit-only。
- run/launch scripts
  - 增加互斥的 `--formal-session`；不开放自由 15/边界/segment 参数；attempt log 继续独立落盘。

### 7.2 唯一允许的公共层改动

`GenerationContinuationPolicy` 新增 v2 的任务中立字段
`operator_pause_generation_boundaries`，表达“只可在预承诺计数边界 clean pause”。P6-C 继续读取 v1；
公共层不包含 Stair、15、3/6/9/12 或 P6-D 成本常量。

若实现发现不需要其他公共修改，不新增。任何额外公共改动必须先证明旧任务也需要同一不变量，并有默认
关闭的兼容测试。

实现审计补充：final current-best 在进程重启后必须从已验证 generation record 指向的私有 train
receipt 恢复完整 checkpoint lineage。为避免 Stair 复制公共 record/private-receipt 的严格绑定逻辑，
公共层增加了一个与既有 eval-phase loader 对称的只读 train-receipt loader。它不增加持久化字段、
不改变 record 校验或旧字节、不认识 Stair/final/15 代语义；任务层仍须先完成 generation chain replay
并自行解释该 receipt。该额外 API 改动列为用户审计重点，而不是静默放宽原计划。

## 8. Preflight、launch readiness 与资源门

### 8.1 Preflight-only

formal preflight 在 scratch 中真实执行：source checkout/GPU inventory/compute-idle/credential 只读门、
contract/Genesis identity、Plan v2、cost v3、reference-v3 私有制品、bound train/eval spec、generation
manifest、trial train result 与 eval-phase result 的 JSON 落盘重载。它不得创建 formal output、ledger、
lease、attempt、provider 请求或 cost reservation；测试至少一条执行真实 GPU compute-idle 检查，
不 mock 掉被验函数。terminal Reservation/Result 与 readiness subject 的严格校验由各自的 schema、
terminal finalization/readiness 测试覆盖，不属于 `--preflight-only` 的执行边界。

### 8.2 磁盘门

formal fresh/resume 使用：

`required_free = max(30 GiB, ceil(1.20 * remaining_worst_case_output) + 10 GiB)`

其中 `remaining_worst_case_output = max(0, 13.5 GiB - current_accounted_output_bytes)`。
fresh 时为 26.2 GiB，因 30 GiB 下限实际要求 30 GiB；resume 随已可信落盘证据递减，但不低于 30 GiB。
不得删除用户产物来满足门槛。

### 8.3 其他门

- 两个准确目标 GPU UUID compute-idle，无未知 PID；
- `MemAvailable >=25 GiB`，swap 增长/占用门保持 `<=64 MiB`；
- source commit、tracked cleanliness、origin/server identity 与 argv hash 一致；
- provider credential 只做存在性/可读性检查，不记录 secret；
- readiness receipt 必须新鲜、单次消费并绑定准确 launch command；fresh 与每次 resume 都重新生成。

## 9. Report v5 与 G5

report v5 必须从原始 ledger/receipt/journal 重算，而不是复制运行时累计值，至少分栏报告：

- reference、parent bootstrap、15 generation、final confirmation、retry/partial 的 job、seed、episode、
  control-step、GPU-hour、provider 和 output 成本；
- 每代 outcome、winner/formal/head transition 与运行激活门；
- lifecycle 五段正常拓扑、所有 recovery、attempt复用、terminal/finalization heads；
- high/medium loser 到达数、被重访数、重访后公开改善与接受数；
- verifier-equivalent avoidance、trial-specific exact implementation审计；
- priority/eligible arrival/serve/failure/expiry/backlog、revisit yield；
- selector reversal、public-dev/public-accept 一致性；
- authoring wall time、B exclusion repair、paired-ready、prompt tier/bytes和 provider usage；
- final paired per-seed deltas、mean/SE，并明确 diagnostic-only。
- 纯 ledger 派生的 generation outcome 计数、有效科学代数、accepted 数、head transition 数和起终 head
  是否不同；这些事实与 `engineering_status`、`acceptance_authority` 分栏。

独立 G5 只读审计必须证明 formal output 审计前后 hash 不变、三条权威链可重算、终态资源为空，且
明确区分：工程/证据是否 GO、科学 head是什么、Genesis-vs-final 诊断结果是什么。曲线或平均 reward
不能替代 acceptance/sealed/head 证据。

## 10. 最小测试与审计矩阵

| 组别 | 最小必须证据 |
|---|---|
| Schema | cost v3、continuation v2、Plan/Reservation/Result v2、report v5 exact round-trip 与关键 tamper |
| Legacy | P6-B v1、P6-C v2/v3、final v1、Push/width=1/round-v7 replay 不变 |
| 15-gen integration | fake provider/runner 走真实 JSON/hash/create-once/private binding/ledger/cost/lifecycle，共 15 条连续 ledger |
| Boundary resume | 3/6/9/12 只在闭合代暂停；resume 不重复 provider/train/eval/append |
| Recovery | 一个 unclean segment 唯一恢复；第 3 个 recovery 拒绝；已闭合 job只复用 |
| Finalization | terminal 前拒绝 Reservation；terminal 后两次 runner eval；Result single-use；中断后复用 attempt |
| Authority | final result 不改变 ledger/head/prompt；parent guard 与 accepted/rejected/zero-train组合正确 |
| Cost | operation 前原子 admission；shared retry；失败/partial保守计费；final cost 链可重算 |
| Readiness | fresh/nonterminal/terminal-finalize/audit-only；真实 idle checker；剩余磁盘公式 |
| Resource | 两物理 UUID、child logical cuda0、process-group teardown、lease/compute PID 清空 |
| Server | Linux 全量、真实短 Stair teardown、preflight、fresh readiness、准确提交/argv/hash |

三视角静态审计分别检查：

1. authority/evaluation：谁能选择、accept、推进 head，final 是否纯诊断；
2. runner/resource：谁能创建进程、绑定 GPU、重试、清理与记成本；
3. prompt/memory：private/sealed/terminal diagnostic 是否泄入 public context，frontier 是否越权。

不建立 15 代所有 outcome 的组合爆炸测试；选择覆盖会改变状态机分支的少数组合。

## 11. 实施与真实运行顺序

1. 提交本冻结计划与 P6-C 闭环文档；
2. 实现版本化合同、共享 bounded-session engine 与 formal CLI；
3. 实现 runner-owned final confirmation、formal readiness/report；
4. 定向测试、全量回归、legacy replay、三视角审计；
5. 提交/push，证明 local/origin/server 精确同步；
6. 服务器 Linux 全量与短真实 Stair teardown；
7. 对准确 P6-D output/argv 运行 preflight-only 与 fresh readiness，确认 formal output 仍不存在；
8. 创建新 formal session，按 Gen3/6/9/12 边界暂停；每次先 terminal audit，再生成准确 resume readiness；
9. Gen15 后只读 generation terminal audit，create-once Reservation，顺序执行两次 final eval并形成 Result；
10. 独立 G5、资源清理、最终报告与文档闭环；
11. 最终提交/push/server sync。是否再开另一场完整实验不在本计划范围内。

若实现或服务器门发现重大问题，停止在 GPU/provider 前；若运行中发现需要改变冻结语义，保留当前
session 原始证据，不跨提交 resume，不以新增 segment 或预算补齐 15 代。

## 12. 完成定义

P6-D 只有同时满足以下条件才完成：

- 15 条连续 generation ledger 在同一 Genesis/contract/epoch 下闭合；
- 五段正常生命周期和实际 recovery chain 合法，累计预算从未重置；
- accepted winner 的机制运行门均通过，head transition 可重算；
- final Plan 在 Genesis 前、Reservation 在 Gen15 后、Result single-use，且两次 eval runner-owned；
- final diagnostic 没有 acceptance/head/prompt 权限；
- 所有 provider/train/eval/control/GPU/disk cap 通过；
- local、origin、server、formal report identity 一致；
- 全回归、legacy replay、三视角审计与独立 G5 通过；
- 终态无目标 GPU compute process、session/GPU lease或 owned process residual。

少于 15 代、超预算、链不可验证或 finalization 不完整都只能报告为 P6-D incomplete，不能用“已训练很多”
替代正式完成。

## 13. 2026-08-24 独立复核后的实施修订

本修订不改变 P6-D 的科学协议、预算、接受权威或 final confirmation，只收缩工程启动面并补足已经由
P6-C 真实结果证明有用的 Stair authoring 信息：

- 受支持的 readiness CLI/launcher 只生产和消费 formal v3 receipt；无入口、无覆盖的 v2 分支已删除，
  v1 只保留被 v3 synthetic-subject 校验复用的 mapping validator。历史 P6-B/P6-C readiness 证据在其
  冻结提交上离线审计，不要求当前主线继续承载旧 loader/producer。
- P6-C 的 tracked runtime monitor 不进入 P6-D 当前运行链。启动前 compute-idle、runner-owned
  process-group cleanup、generation resource closure 与终态 G5 已覆盖当前需要的资源权威；不再为外部
  占卡事件维护独立的不可变采样账本。
- Stair prompt 增加最近一代结构完整的两条 public-accept-vs-parent 比较，显式呈现 objective/guard
  对冲；它是 task-owned public projection，不进入通用 memory 权限，也不暴露 sealed/private evidence。
- selector 在不足两条 guard-eligible trial 时把 paired statistics 标成 `not_applicable`；冻结 v3 字节和
  replay 保持严格，当前 schema 不再把过滤结果伪装成配对统计。
- 不引入 lane fallback、短训练筛选、非对称 lane 预算或新的恢复/监控框架。它们会改变科学合同或继续
  扩大当前已经过重的启动脚手架。
- 低层 generation CLI 已移除 `--validation-session`，不能绕过正式 launcher 继续创建 P6-C 开发
  session；共享 validation core 只作为历史/测试实现保留，不再有受支持生产入口。
- Stair prompt 比较只从 scrub 后仍存在、且与 task/eval/acceptance protocol 绑定的 public score stats
  构造；完整 `CurrentBestEvalStats` loader 仍要求 sealed 字段，prompt 不获得 sealed/private 权限。
- formal report 使用 v5，并从 ledger 单独汇总有效科学代、accepted 数和 head transition；15-generation
  production-shaped 回归必须至少接受一条 candidate，使终局 finalization 覆盖移动后的 head。

因此，`16175c1` 的已停止预启动只保留为失败/中断证据。用户重新审计通过后，必须从届时准确的新提交、
新 output、新 readiness receipt 和新 Genesis 开始；在此之前不得继续 reference train、provider 或
generation session。

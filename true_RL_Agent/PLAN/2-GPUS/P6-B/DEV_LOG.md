# P6-B 开发日志

## 2026-08-22：启动门根修与正式 attempt 1（资源污染 No-Go）

### 启动门根修

- 静态审计确认，P6-B 既有计划要求的 production-shaped Linux attempt-chain fault probe
  没有真实入口；既有 reward CUDA fault probe 不能替代 logical-job attempt/retry/cleanup 证明。
- 新增 `scripts/run_p6b_attempt_chain_fault_probe.py`，直接走现有
  `SubprocessRunner -> JobAttemptJournal -> run_owned_process` 路径，固定产生
  `summary_missing -> retry -> terminal` 两次 create-once result、唯一 retry 前序 hash、
  terminal chain head，并证明两次 child/grandchild PGID 均无残留。
- 新增 `scripts/run_stair_train_teardown_probe.py` 和最小真实 Stair smoke 入口，以正式
  Isaac Python、完整 GPU UUID、正式 `run_owned_process` 和 shared lease root 运行固定
  `16 env x 8 steps x 1 iteration = 128 env-steps`，验证正常退出后的 PGID、compute PID
  和 active lease holder 全部清零。
- `--preflight-only` 的真实服务器阻断来自 compute-idle stage 白名单漏掉
  `preflight-only`；修复后同参数 10 项私有持久化往返全部通过，正式 output、lease、
  attempt、Genesis、cost reservation 和 provider 调用均未创建。
- 提交 `a572e5ad3f05e4919357af8be7cf5ba5487445d1` 后，local/origin/server HEAD 一致、
  server tracked clean。本地 unittest 为 `1362 / OK / skipped=26`，pytest 为
  `1336 passed / 26 skipped / 975 subtests`；服务器 Linux-only process-group tests 与
  两个 production-shaped probe 均通过。

启动门独立证据：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_prelaunch_probes/a572e5a_20260821T203556Z`

- attempt-chain report：`attempt_chain/report.json`，status `pass`；
- 真实 Stair teardown：`stair_teardown/result.json`，status `pass`；
- resource readiness：`resource_readiness.json`，两张目标 GPU idle、shared lease
  acquire/release/reacquire、swap growth 0；
- preflight 原始 stdout 由既有 append-only Codex transcript 追溯归档，明确标记为
  retrospective operational archive，不冒充服务器原生 receipt 或 acceptance authority。

### 正式 attempt 1

正式 fresh output：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_a572e5a_20260821T203556Z`

launcher attempt：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_a572e5a_20260821T203556Z.launcher-attempts/attempt-20260821T205102.306010Z-pid2791996-48b232b2`

运行事实：

- reference train 完成全部 `196,608,000` steps；attempt result 为 `completed`，
  `formal_training_executed=true`，预算、environment preflight、backend manifest、私有三字段
  GPU binding 和 PID=PGID 均通过；
- reference checkpoint checksum 为
  `sha256:239cc5d2ac9ecc623f5f3811277680b36d6ff6621f7a816bdd9dfe5d719e2c5e`，
  后续 eval argv 使用同一 checksum；训练 PGID 退出后无成员；
- reference eval 已闭合 public-dev `3/3` 和 public-accept `5/5`，进入 sealed-validation
  `1/7` 时，独立资源监控 sample `seq=419 @ 2026-08-21T22:00:58Z` 捕获目标 GPU 上的
  外部 compute context：GPU0 同时存在正式 eval PID `3131022` 与
  `VLLM::Worker` PID `3167708`，GPU1 同时存在 `VLLM::Worker` PID `3167709`；
  两个 VLLM worker 各占约 810 MiB。相邻 `seq=418/420` 均无 VLLM，污染持续上界约
  20.28 秒，但冻结计划把 unknown target process 明确列为自动暂停与 Stop/No-Go，
  不允许按持续时间或显存大小事后放宽。

停止与清理：

- 先向精确 eval PGID `3131022` 发送 TERM；Isaac 在 70 秒 grace 后仍从 public-accept
  继续进入 sealed-validation，因此再向同一已核验 PGID 发送 KILL；未触碰 VLLM 或其他
  用户进程；
- eval attempt create-once 闭合为 `infrastructure_failure / summary_missing /
  retryable=false`；该 `summary_missing` 是受控停止后的派生分类，不是 7200 秒 runner timeout；
- launcher 独占 `exit.json` 为 `status=failed / returncode=1`，launcher/stdout/stderr log
  checksum 完整，原始 traceback 可恢复；
- launcher、generation session、eval PGID 和所有相关 compute PID 均已退出；两张 GPU
  连续多个样本 idle，shared lease 无 active holder；
- resource monitor 正常 SIGTERM，stop receipt 为 `status=stopped`，共 448 个连续样本
  `seq=0..447`，JSONL checksum 为
  `sha256:8a55f3dbb60efaeccc5cbe85b3815af996cea2b0c638a2cacb08f0af135db486`；
  全程 swap growth 为 0。

成本链没有删除失败消耗。`generation_actual_cost/0001-genesis-failed-0001.json` 记录：

- train jobs / steps：`1 / 196,608,000`；
- eval seed-units：`15`（失败 attempt 按完整预留保守结算）；
- simulated control steps：`199,590,000`；
- allocated GPU capacity-hours：`1.2261282268902778`；
- observed output：`333,763,303 bytes`；
- provider attempts：`0`；
- 所有当前 physical cap checks 仍为 true。

### 结论与下一门

- attempt 1 是不可变资源污染 No-Go，不生成 Genesis、generation ledger、科学卡或 acceptance
  结论，也不重投影成成功。
- reference eval 的 pre-Genesis scope 无 retry，且 Genesis 尚未形成；该 output 不能合法 resume。
  下一次 P6-B 必须使用新的 fresh output/launcher-attempt root。
- 重跑前必须确认两张目标 GPU 的独占窗口；仅凭某一时刻 `nvidia-smi` idle 不足以排除外部
  VLLM 再次进入。继续保留独立运行期监控，unknown PID 再现时仍按同一冻结门停止。
- P6-B 尚未 GO，不进入 P6-C。P6-C 的多 segment/operator-pause 与 nested job PGID 根修
  只做了静态设计审计，不回灌或热改本次 P6-B 语义。

## 2026-08-22：正式 attempt 2（provider 402 闭合）与 GPU 前账户门

### attempt 2 不可变终态

提交 `8f71d6b9a3165c10414a90f7d71bbab117ba97ae` 的 fresh output：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_8f71d6b_20260822T002232Z`

launcher attempt：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_8f71d6b_20260822T002232Z.launcher-attempts/attempt-20260822T003724.391998Z-pid3963946-eeb35882`

同 commit 的启动门证据：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_prelaunch_probes/8f71d6b_20260822T002232Z`

- 10 分钟 1 秒采样为 606 个连续 idle 样本；resource readiness 9/9、attempt-chain 两次
  create-once result/唯一 retry head/零残留、离线 preflight 10/10、真实 128-step Stair
  teardown 全部通过；
- width1 性能基线在任何 reference result/Genesis/generation 之前冻结为 2026-08-13 正式
  15 轮的中位轮 R07，`4625.151178823s`，1.35 gate 为 `6243.954091411s`；同 session
  reference train 完成后只追加观察，不反改基线选择；
- reference train、reference full eval、parent public-dev 三个 job 均完成，checkpoint checksum
  为 `sha256:239cc5d2ac9ecc623f5f3811277680b36d6ff6621f7a816bdd9dfe5d719e2c5e`；
- lane-a 第一次 authoring request 返回 HTTP 402，持久字段为
  `runtime_error/http_error/http_status=402/retryable=false`。所以一次即终止符合冻结 retry
  语义；lane-b 未启动，无 candidate train、selection、public-accept、winner sealed 或 final
  confirmation；generation 正常闭合为 `paired_authoring_incomplete`，current best 仍是 reference；
- ledger 恰一行，SHA256
  `f75b1fa6990be45004037e2cc8b5c540c5defdc8bbe6304599bfa302ac0b9133`；execution receipt
  内嵌 record 与 ledger 逐对象相等。report SHA256
  `3cfe46ff4c17710c01283e154f98a5b286e41bc5624ef7bffed918a7ba6135c5`，segment exit SHA256
  `af2daa67544db81f71224f83b4a88f5bc493f9b24102cc9a601bfbc2406ffe72`；
- 三个 reservation/result 全闭合且 scope/canonical hash 一致。lease wall 精确求和为
  `4,875,071,675,133ns = 1.3541865764258334 GPU-h`；cost chain、segment、terminal binding
  和 report 指向同一终点；
- 独立资源 monitor 为 504 个连续样本 `seq=0..503`，仅出现三个 owned job PID、unknown PID
  为 0、swap growth 为 0；monitor JSONL SHA256
  `fb341a48f2000442c1692812badd276971579816ee921e53b98536eace8f2278`。闭合后 launcher、
  session、Isaac、monitor、compute PID 和 active OS lease 全部为 0；`.lock` sidecar 是设计上
  持久的元数据，不是活跃 file lock。

`rc=0` 和 segment `completed/normal` 只证明 lifecycle 完整关闭，不能覆盖
`paired_authoring_incomplete` 的科学结果。该 output 已终局，不得充值后原地复活；P6-B 仍未完成。

### 402 根因与启动门根修

DeepSeek 官方错误表把 HTTP 402 定义为余额不足，官方 `GET /user/balance` 的
`is_available` 表示账户当前是否有余额可供 API 调用：

- https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/
- https://api-docs.deepseek.com/zh-cn/api/get-user-balance/

服务器用同一 secret 边界做只读余额查询得到 `is_available=false`；只输出可用性/币种，未输出
credential 或余额数值。因此这是外部账户阻断，不是 candidate schema、模型响应通道或 runner
故障。既有 `--preflight-only` 有意零 API，只验证 credential 格式，故无法提前发现该状态。

提交 `4e4af51bc4d48c60c00daabeeade4299c61eff1b` 新增 task-owned DeepSeek balance gate：

- fresh 或 ledger 为空的 incomplete resume 在 source checkout 校验后、首个 formal write、
  foundation、reservation、inventory/lease、Genesis 和 runner 之前强制执行一次官方 balance GET；
- 通用 `UrllibJsonTransport` 只增加 concrete `get_json`，复用既有 bounded HTTPS、credential、
  TLS、no-redirect 和 HTTP 分类；`JsonTransport`/`CandidateProvider` 插件协议及旧 POST 请求
  method/body/headers 不变；
- DeepSeek endpoint、`is_available`、CNY/USD 与余额字段 schema 解释只在 Stair 叶模块；结果
  只投影可用性和币种，不保存余额字段、原始响应或 secret，不计 scientific provider attempt；
- `--preflight-only` 继续明确禁止 GET/chat；新增 `--provider-readiness-only` 输出带 source
  commit、session、UTC、endpoint、GET/非生成请求和隐私声明的完整脱敏证据；
- 已有 final report 或合法非空 ledger 的 settle/replay-only resume 跳过网络门，保持闭合产物
  的离线可读性；chat 的原有 402 fail-closed 继续覆盖检查后的账户状态变化。

本地最终门禁：unittest `1370 / OK / skipped=26`；pytest
`1344 passed / 26 skipped / 981 subtests passed`；compileall、触及文件 F 检查和
`git diff --check` 通过。服务器 Python 3.10 首轮定向测试暴露测试注解误用 3.11-only
`typing.Self`，产品路径未执行；提交 `67a30a7659fcd63a1aa61310342ba41a67e88b93` 根修后，
服务器定向 `89 / OK`。local/origin/server 最终 HEAD 一致且 server tracked clean。

### 真实失败门证据与当前 No-Go

在 `67a30a7` 上由 tracked launcher、正式 Isaac Python、相同 secret/UUID/成本参数运行
`--provider-readiness-only`：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_67a30a7_provider_readiness_gate.launcher-attempts/attempt-20260822T024638.557193Z-pid358149-8402054b`

- `2026-08-22T02:46:38.557418Z` 启动，`02:46:51.419045Z` 结束，12.86 秒；
- `StairProviderReadinessError: provider readiness reports insufficient balance`，launcher
  `status=failed/returncode=1`；exit SHA256
  `8749fb4799ab80a4e58e8f69370b32e37ff5bddea4c270aba01596e9c453d917`，stderr SHA256
  `250289b0d48a60d76583e4879c2819c872385111d2d0d624ee62fd58837e7126`；
- 正式 output 前后均不存在，owned launcher/child 均退出，没有 active GPU lease；失败从原来的
  `1.354 GPU-h` 前移到 13 秒且零本任务 GPU job。

同一终态检查发现两张目标 GPU 已再次被外部用户的 `VLLM::Worker_TP0/TP1` 占用，各约
29,012 MiB；未触碰这些非本任务进程。所以下一轮 fresh formal 同时等待两个独立启动门：

1. 同一 DeepSeek 账户充值后 `--provider-readiness-only` 必须得到 `is_available=true`；
2. 两张目标 GPU 恢复计划要求的独占 idle 窗口，并重新通过完整 resource/preflight/probe 门。

在两门同时满足前不得再烧 reference，也不得进入 P6-C。

## 2026-08-22：最终单代闭合、独立重算与启动证据根修

### 正式 P6-B 终态

最终科学运行使用提交 `5f2d7557ca51b73f1f16911e117a9401a103c171`，权威输出为：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_generation_5f2d755_20260822T025737Z`

session ID 为 `p6b-5f2d755-20260822T025737Z`。该 session 完成完整 reference、parent
public-dev、双 lane authoring/train/eval、selection、public-accept、winner sealed、ledger append、
cost settlement 和 terminal report；没有运行 final confirmation。独立重算得到：

- 10 个 logical job 均由 `attempt_00` 完成，零 retry、零 open reservation；3 个 train job，
  `589,824,000` train env-steps，41 eval seed-units，`597,974,800` simulated control steps；
- allocated-active GPU-hours 为 `3.725287525868333`；provider 为 2 attempts / 69,516 units；
  accounted output 约 1.004 GB；全部 create-once physical cap checks 通过；
- lane A/B 的 P7 运行期激活均可从 train summary 重算：两者 call count 均为 48,001，
  `max_abs_weighted_contribution` 分别为 `0.30000001 / 1.0`，不是“机制未触发”的空实验；
- selector 冻结 lane A；winner sealed 7/7，mean `0.46122449`；
- objective candidate `0.44928571`，parent `0.53857143`，delta `-0.08928571`，未达到
  required `+0.09228571`；guard 改善并通过，但 objective gate 正确拒绝。因此 scientific
  结论是 rejected，current best/head 保持不变；该拒绝不影响 P6-B 的工程 GO；
- report SHA256
  `cdc4c5ca931735eb114b6b631c1a7ae960703d9688f81066aa3da0db584c139e`，ledger SHA256
  `845f1108cb72190f0bb0776ddf90252f95af8758c2eb9f13da32c8aa0cc88847`，Genesis SHA256
  `3766c728bd1b016dde092d0f3e233ae6da02a899b92c4c6974517ffde7a63ead`，segment exit
  SHA256 `430bc15965c96896c69ea1c88a7e5ef83e807ab90c6c7474ce8fd82b064c2877`；
- `--resume-existing` 在约 13 秒内完成纯离线复验；上述文件字节和 hash 均不变，零新增
  provider/GPU work；
- 运行结束后全部 owned PID/PGID 消失、两卡为 0% / 6 MiB、shared lease 无 active holder。
  独立 monitor 共 957 个连续样本 `seq=0..956`，unknown/wrong UUID 为 0，含 14 个
  post-launch idle 样本；JSONL SHA256
  `40820cea11f4e1644ac07db0fb4ba87f8bb5638aa56637b69b1c5b300a9f08b4`，不能由
  aggregate report 代替。

### 明示例外：不追写不存在的 prelaunch 证据

本次 `5f2d755` 科学运行启动前没有由 formal launcher 强制消费的 600 秒 launch-readiness
receipt。它使用了同次保存的 preflight、attempt-chain、短 Stair teardown 和独立运行期 monitor，
但没有把 605 秒静态 idle 窗口做成可由 launcher 绑定的 create-once 凭据。该项按用户明确批准
作为一次性 readiness waiver 记录，始终不标记为“已通过”，也不把事后证据回填到既有 output。
由于科学工作、sealed、ledger、成本和资源闭合均已完成，且该缺口只影响启动证据，不改变科学
输入或结果，决定不再为补证据重烧约 3.7 GPU-hours。

### 源头修复与事后真机验证

提交 `8c2c6b87ecc6f6badc48c010ff87dd5dd4462c4a` 先将 producer 与 formal launcher
绑定；随后静态审计发现非终态 resume、位于 formal output 内的 attempt 目录，以及直接调用
generation child 仍可绕过这条边界。提交
`ba71c5d308f5b498196d53164411f6d7fc0cd542` 在同一源头补齐三处入口，而没有改变科学协议：

- 新增 Stair-owned launch-readiness producer；在 formal output 外 create-once 保存双卡
  inventory、同时 lease acquire/release、600 秒 1 秒 cadence、双卡 reacquire/release、
  RAM/swap/disk/compute-idle、source/host/boot/argv 和关键源码字节证据；
- receipt 严格绑定 source root/commit、session、output、machine、hostname/boot、两张 UUID、
  lease root、output cap 和 `nvidia-smi`；超过 300 秒、采样回退/缺口、源码漂移、foreign
  checkout 或字段/hash 漂移均 fail closed；
- formal launcher 在创建 attempt 目录和 child 之前必须加载同一份新鲜 receipt；helper、
  preflight 和 provider-readiness 不冒充 fresh formal launch；所有 resume 在 P6-C 的版本化
  resume-readiness 合同落地前 fail closed；
- attempt root 必须是绝对路径且位于 formal output 外；无效路径在 mkdir 前拒绝；
- launcher attempt 新增 create-once `start.json`，保存完整 launcher/child argv、cwd、PID/PGID/
  SID、source/host identity、关键文件 checksum 和 readiness ref；exit v3 再绑定原始 start
  bytes。日志仍按 attempt 独立，旧 traceback 不覆盖；
- formal child 必须消费 start path/hash 内部上下文，严格核对 canonical bytes、schema、parent
  PID、完整 argv/hash、cwd、source/host identity 和同一份 fresh readiness；成功后删除内部环境
  变量，防止传给 Isaac/runner 子孙。直接调用 child 在任何 output/GPU/provider 副作用前拒绝；
- 公共 generation 科学协议、selector/eval、Push 和 legacy width=1 均未改变。

最终 `ba71c5d` 本地全量 unittest 为 `1383 / OK / skipped=26`，服务器为
`1383 / OK / skipped=12`；两端 launch/readiness/session 定向 45 项均通过，compileall、Ruff
format、Ruff `E4/E7/E9/F` 和 `git diff --check` 均通过。独立对抗审计确认 producer 源码漂移、
monotonic 回退、超过 2 秒 cadence gap、301 秒 stale、foreign source root、三类正式入口绕过都
被拒绝。

代码提交、origin 和服务器 HEAD 对齐后，服务器真实运行完整 producer：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_readiness_postclose/8c2c6b8_20260822T093645Z/launch_resource_readiness.json`

- receipt hash：`dd3cddb864fddd432a8a29cdccb599fa38cb8824659b7d63f2a1380fb140f378`；
  文件 SHA256：`7af17de12ab0d266e2b05f38e7a18316ce692f5d58d84b1a8577d1216aa4ade0`；
  size：214,926 bytes；
- 601 个 sampling-window 样本覆盖 `600,169,110,418 ns`，全 receipt 共 606 个样本；
  最大间隔独立重算与记录一致，为 `1,168,574,813 ns`，低于 2 秒硬门；
- 最低 `MemAvailable=289,450,688 KiB`，swap growth 为 0，最低可用磁盘
  `155,663,847,424 bytes`；全程两卡无 compute app；
- initial 和 reacquire 两次均同时持有 cuda:0/cuda:1 对应物理 UUID，随后释放；
- formal output 前后均不存在，provider requests / GPU workloads / ledger writes 均为 0；
- producer 返回后，正式 launcher 的真实 `_validated_launch_readiness_ref` 成功加载同一文件并
  返回完全相同的 path/hash/size/ref；两卡仍为 0% / 6 MiB、无 compute app、server tracked
  source clean。

该 post-close 证据只证明 `8c2c6b8` 及以后启动门已经可执行，不能倒推成 `5f2d755` 启动前
已有该 receipt。

在最终根修提交 `ba71c5d` 上重新生成了不可复用的新证据：

`/data/yuyangliu/true_rl_agent_phase_g/p6b_readiness_postclose/ba71c5d_20260822T102136Z/launch_resource_readiness.json`

- receipt hash：`f8b7015e2166bc514f09a8dbb9277e20adf243cd20e1ce1ecdac484a7d2baff2`；
  文件 SHA256：`f693942483bee6cb3bd8fbd8e55388e0c4f60dd44271d373c3503594abe1c8c3`；
  size：214,926 bytes；
- 606 个连续样本，其中 sampling window 601 个；独立重算最大相邻间隔
  `1,168,023,961 ns`，低于 2 秒硬门；
- 最低 `MemAvailable=290,104,468 KiB`，swap growth 为 0，最低可用磁盘
  `155,662,594,048 bytes`，全部样本 compute app count 为 0；
- receipt canonical bytes、所有 checks、双 lease 首尾 acquire/release 均通过；正式 launcher
  的真实 `_validated_launch_readiness_ref` 在新鲜窗口内返回完全相同的 path/hash/size/ref，
  formal output 前后均不存在；
- 服务器真实 direct-fresh child 绕过探针以 `rc=1` 在任何正式副作用前拒绝；日志 SHA256
  `d6917a70349960943828a059aff66fd3f3035f5ab052a2293d22e92d679a5c81`、574 bytes；
  随后两卡仍为 0% / 6 MiB、无 compute app，server tracked source clean。

这份新证据绑定 `ba71c5d`，取代 `8c2c6b8` 作为未来启动治理的最终验证基线；两份 receipt
都保留，且都不能倒推修补 `5f2d755` 已明确豁免的历史 prelaunch 缺口。

### P6-B 结论

P6-B 以“工程与证据 GO、科学 rejected”闭合。单代生产 binding、真实 fixed-width generation、
sealed/ledger/cost/resource/replay 均已完成；一次性 readiness waiver 已如实保留，未来 formal
launch 的同类旁路由 `ba71c5d` 在 launcher 与 child 两端共同禁止。用户于 2026-08-22 同意
无重大问题可进入后续 Phase；因此
P6-C 可在全新 development Genesis/output 上按冻结计划实施，不继承 P6-B ledger，也不把本次
科学拒绝改写为接受。

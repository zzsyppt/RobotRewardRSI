# P6-C 开发日志

## 2026-08-22：三代实现、本地审计与服务器启动门准备

### 冻结边界

- 实现基线为 `99302fb918d09f1e0bde9f8e4d88109be0b88989`；P6-B 科学运行
  `5f2d7557ca51b73f1f16911e117a9401a103c171` 已闭合且不会重跑。
- P6-C 仍是 Stair-owned、固定三代、每代 width=2 的 development validation；没有新增
  通用 DAG、终局 confirmation、运行时控制通道或候选权限。
- 成功路径、retry、provider、资源与四段 continuation 上限均保持冻结计划数值。实现和本地
  审计期间没有调用 generation provider，也没有启动 GPU workload。
- 每个物理 launch segment 的 session `--timeout-s` 冻结为 `43200`，沿用同 profile P6-B
  成功正式 attempt 的预注册值；fresh、resume 与 recovery 均不得事后放宽或借此增加成本额度。

### 实现

- 三代 driver 以 generation-scoped authoring/candidate/attempt/result/closure 目录串行闭合
  generation，保留每代双 lane 并行 train/eval、单次 selector 和单次 ledger append。
- ledger/cursor/guard、session-wide provider/retry/actual-cost admission、generation closure、
  clean pause、resume、硬崩恢复与 terminal report v3 均从 create-once 文件重放；正常计划固定为
  gen1 pause、gen2 crash/recovery 后 pause、gen3 terminal，最多四个 invocation segment。
- launcher 使用 attempt 独立日志、持久异常类型/脱敏消息和 tracked 只读 runtime monitor；monitor
  在 spawn 前、owned child 活跃期、process-group cleanup 后依次留样，不拥有 kill、lease、成本或
  科学权限。
- readiness v2 绑定 fresh/resume subject、准确 commit/argv/monitor/probe evidence、provider 只读
  readiness、双卡资源窗口和正式 output 不变性；fresh、gen1 后 resume、计划 crash 后 recovery 与
  gen2 后 resume 各自生成 launch-attempt-scoped receipt，不复用较早 argv/hash/subject；terminal
  audit 只做离线复验。
- provider 请求按 generation/operation 独立 attempt scope；中断请求在 P6-C 保守计满后使用新
  attempt，P6-B 旧行为保持 fail closed。公共 runner 只增加默认关闭、任务中立的 admission、
  timeout leaf 与 process-start callback；Stair 语义留在任务层。

### 静态审计发现并已在源头修复

- 私有 job spec 不再复用 public GPU-binding 投影；private reference/train/eval 持久化均保留
  `host_execution_device`、`process_execution_device`、`resource_binding_hash` 三字段并有往返测试。
- `preflight-only` 成为真实 compute-idle stage；至少一条测试不再 mock idle validator。preflight
  对正式 output、lease、attempt、Genesis、cost reservation 与 provider 保持零副作用。
- actual-cost 与 lifecycle recovery 作为一条 Stair-owned combined chain 重放；readiness 与 session
  共用同一纯函数。active segment 的 lifecycle head 可合法落后 actual tail，partial recovery receipt
  可在下次 session lease 内补齐；fork、orphan、cycle 继续 fail closed。
- resume readiness 不再访问 cost contract 不存在的 `session_id`，而是验证 cost contract hash 等于
  generation-session contract 已冻结的 `session_cost` hash；未修改公共 schema。
- generation closure 固定绑定对应 `generation-NNN-settlement`；pause/terminal report 绑定当前
  combined head。修复了 missing terminal report 在 later failure receipt 后错误复用 current head 的
  离线恢复问题，并补真实 journal/create-once 回归。
- 已存在 continuation policy 的复验改为 bounded strict read，不再先写临时 sibling；首次创建与
  create-once 竞争语义不变。
- 历史 P6-B v2 terminal loader 只在 `resume_existing + pilot + canonical terminal report` 下允许
  stored contract 的 `authoring_policy` 与当前环境身份不同；session/task/backend/width/counts及其余
  bound hashes必须一致，report 与 Genesis 继续绑定同一 stored contract ref。非终态、P6-C 和其他
  contract 漂移仍拒绝。
- runtime monitor 逐样重放会重新验证 UTC、GPU raw/structured 对应、UUID、PID ancestry、launcher/
  session observation、phase 与权威文件路径范围；重算 sample/stop hash 的自洽篡改仍被拒绝。
- active/unclean segment 的 readiness 不再把 segment-start lifecycle 高度误作当前 ledger 高度；只在
  `start_height <= actual_ledger_height <= max_generations` 时允许进入既有恢复，closed/recovered 边界
  仍要求严格相等，combined cost replay 未放松。真实落盘回归覆盖 start=0/ledger=1 与闭合反例。
- 首次服务器 Linux 全量门暴露 tracked attempt-chain probe 漏传 runner 新增的
  `attempt_timeout_s`；probe 现把 attempt journal 给出的同一有效值原样传入 `_run_job`。新增非
  Linux-gated 回归真实执行两次 attempt/journal 链并锁定两次 timeout 传递，避免 Windows 绿色套件
  再次漏过 Linux-only probe 的接口漂移；公共 runner 签名和默认语义未改。
- Linux 测试 fixture 改为完整 canonical candidate package；process-group 测试使用真实
  `run_owned_process` child，生产 runner 的 PID=PGID=SID 严格检查未放宽。

### 本地证据

- 最终全量 `python -m unittest discover -s tests`：`Ran 1479 tests in 257.159s`，
  `OK (skipped=29)`；是在 active-ledger readiness 与 attempt-timeout 根修后重新执行。
- readiness/session/cost/lifecycle 聚焦矩阵：`95 passed, 1 skipped, 18 subtests passed`；唯一 skip
  是 Windows 上的 Linux procfs test。
- 真实落盘 resume fixture：3/3，通过 active-head lag、partial recovery transaction、只读 subject、
  后续 start reconciliation 以及 fork/orphan 反例。
- P6-B terminal contract、legacy resume 与 continuation-policy 聚焦矩阵：`15 passed,
  1 skipped`；skip 是 Windows 上的 Linux-only procfs test。
- 44 个本轮 Python 文件通过 Ruff `E4/E7/E9/F`、`py_compile` 与 `git diff --check`。
- production-shaped 三代 integration 使用跨进程 setup/pause/crash/resume/finalize 和真实
  JSON/hash/create-once/private binding/attempt/lifecycle/cost/ledger 路径；本机无 Torch，因此最终
  有效运行保留为服务器 CPU+Torch 硬门，Windows skip 不计通过。

历史 P6-B loader 的本地 provisional 稀疏镜像预演直接调用当前真实 `_create_foundation`、Genesis loader
和 `_load_and_validate_existing_report`，249 files / 1,003,470,865 bytes 的镜像树前后 hash 均为
`063b8d4adcfdc8007a96c2ba6aa4ee8345e072337fbcad067c1c306ba0e5dd08`；write/network/
unexpected-subprocess 尝试均为 0。证据：

`outputs/p6b_audit_20260822/historical_loader_replay_provisional/report.json`

文件 SHA-256：`65848586fa29a365e675084acc0fdd086c3f9576d48e27b814a2a96a8de7e525`。
该预演的 non-PT 与 reference checkpoint 是真实 bytes，其余中间 `.pt` 为等尺寸稀疏占位，因此不
冒充最终门；提交同步后仍须直接重放服务器原始 formal bundle。

### 尚未通过的真实启动门

以下各项仍为 No-Go，全部闭合前禁止 P6-C provider/GPU：

1. 仅提交本轮代码/测试/计划，push 后证明 local/origin/server commit 与 tracked source 一致。
2. 服务器 Linux 全量测试；CPU+Torch 三代 integration、fixed-width、真实 process-group runtime
   monitor、launcher-monitor 顺序和真实 resume-recovery fixture必须执行且零 skip。
3. 在服务器原始 P6-B formal bundle 上运行 tracked historical-loader driver，要求完整原树前后
   path/size/SHA hash 一致。
4. production-shaped attempt-chain fault probe 保存两个 create-once result、唯一 retry head和零残留
   process-group 证据。
5. 短真实 Stair train teardown 保存 Isaac/Kit 正常退出、process group 安静、GPU compute PID 与
   active lease holder 清零证据。
6. 对准确 P6-C commit/output/argv 运行 preflight、tracked runtime-monitor Linux probe和 600 秒
   fresh launch readiness；fresh formal output 前后不存在，provider readiness 只读。正式运行中
   gen1 后 resume、计划 crash 后 recovery、gen2 后 resume 也必须各自重新生成并绑定当次 argv/hash
   与既有 subject heads 的 600 秒 v2 readiness，且均使用冻结的 `--timeout-s 43200`。

全部真实门为绿且没有新的重大问题时，2026-08-22 的条件授权才允许启动正式 P6-C。

> 服务器首次全量尝试曾以 `CUDA_VISIBLE_DEVICES=""` 启动，触发父进程 GPU-remap 守卫并产生
> 一组预期的假失败；该次不计验收证据。Linux 全量与 CPU/Torch hard gate 都必须用
> `env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES`，即父环境无 remap，由被测 runner
> 自己绑定 child GPU 可见性。

## 2026-08-23：首次 formal session 在 gen2 GPU 前失败，根修 sealed authoring anchor

### 旧 session 的权威状态

- 源提交：`df5a301e28acaf3fbb3fbb342d65451513f2ede0`。
- formal output：
  `/data/yuyangliu/true_rl_agent_phase_g/p6c_generation_df5a301_20260822T155725Z`；
  session ID：`p6c-df5a301-20260822T155725Z`。
- fresh launcher attempt：
  `attempt-20260822T163933.854803Z-pid2718199-b483c7fe`。gen1 正常闭合并在 generation
  boundary 写入 `paused/operator_pause`；ledger 恰好 1 条，SHA-256 为
  `776a2e1424bbeaa004b96425fbc9b9ea5be95de983780933d489790f612554ae`。
- gen1 lane A 为 `qualified_not_selected`；lane B 通过 public-dev、public-accept、sealed 7/7
  并 accepted。`goal_progress_reward` 实际调用 48,001 次，最大绝对加权贡献为 0.5；该代不是
  空机制胜出。
- gen1 累计消费为 3.8038030956908333 GPU-hours、3 个 train、589,824,000 train steps、
  41 eval seed-units；计入 eval 后的总 simulator steps 为 597,974,800。provider attempts 在
  gen1 后为 2。
- gen2 resume launcher attempt：
  `attempt-20260822T194331.140972Z-pid3465265-960fb423`。authoring 消耗使累计 provider
  attempts 从 2 增至 6，但 train/eval/steps/GPU-hours 均未增加。
- gen2 lane B 的第 3 个 authoring attempt 已通过 verifier、component smoke、anti-degenerate
  与 sibling-distinct；随后 fixed-width generation 外层抛出
  `ValueError: verified lane B has unavailable sibling comparison`。失败发生在任何 gen2
  train/eval reservation 之前，ledger 仍为同一条 gen1 记录，没有 gen2 public-accept 或 sealed
  科学结果。
- segment exit 已持久化 `failed/exception` 且无 open reservation。launcher、session、owned
  children、GPU compute PID、active lease 与 runtime monitor 均已清理。旧 output 必须原样保留，
  不删除、不覆盖、不把 gen1 拼接进另一 session。

### 根因与源头修复

- `CandidateAuthoringLoop` 正确地在第一次 provider call 前密封 round authoring anchor，并在每次
  repair 中继续用该首 prompt 做 anti-degenerate/sibling 比较。最终 `static_repair` prompt 按设计
  只保留 `attempt/latest_failed/training_code/components.py`，不再内联 current-best。
- `execute_fixed_width_generation` 却在 authoring 成功后，用最终 selected repair prompt 重做 sibling
  comparison。lane A repair 会被静默标成 `sibling_exclusion_unavailable`，lane B repair 会抛上述
  ValueError。原 fixed-width integration 把 `max_attempts` 固定为 1，因此从未组合覆盖该边界。
- 同源审计发现 round 的 `mechanism_operation_delta` 投影和历史 evaluated-candidate 重复机制守卫也
  使用最终 selected prompt；前者会把合法增量静默记为 `None`，后者会在 repair 后漏掉跨轮重复。
- 根修在 task-neutral `prompt_artifacts` 中收敛唯一入口：v4 evidence 从已校验的
  `round_authoring_anchor` 读取首 prompt；旧 v2-v3 evidence 继续使用 selected-prompt fallback。
  generation lane A/B、round operation delta 与历史 evaluated-candidate projection 统一消费该入口。
  最终 selected prompt 仍负责 selected-attempt、lane intent 与 sibling public projection 绑定。
- 没有修改 compact repair policy、provider/selector/阈值/预算、Stair 科学语义、candidate 权限或
  width=1 runner；没有把 current-best 重新塞入 repair prompt，也没有新增 schema。

### 回归与独立审计

- fixed-width 完整链路现在真实覆盖 lane B `static failure -> repair -> verified`，旧实现会在与服务器
  相同的 lane-B replay 点抛错；修复后完成 train/eval/selector/ledger commit，且两 lane 的
  `mechanism_operation_delta` 均非空。
- 对称覆盖 lane A repair 后仍能建立 sibling exclusion 并运行 lane B；另有独立测试断言 generation
  anchor loader 在 selected=第二次 compact repair 时仍返回 create-once 第一次 prompt。
- authoring/prompt/mechanism 聚焦测试 96 项通过；fixed-width/trial/legacy loop 29 项通过；
  exploration 22 项通过。
- 最终本地全量：`Ran 1480 tests in 269.817s`，`OK (skipped=29)`；`git diff --check` 通过。
- 三路独立只读审计确认：当前生产路径中没有剩余把 v4 selected/static-repair prompt 冒充冻结首轮
  anchor 的消费者；旧 evidence fallback 与 width=1/旧任务兼容边界未放宽。

### 为什么必须新建 P6-C session

- 旧 output 的 `generation_source_checkout.json` create-once 绑定 `df5a301`。resume readiness 与
  session 都严格复验 source commit/checkout identity；修复提交会被 commit drift 守卫拒绝，继续用
  `df5a301` 又必然复现缺陷。不得为本次恢复新增跨提交迁移豁免。
- 旧 continuation policy 冻结 `max_segments=4`，现已消耗 gen1 pause 与本次 exception 两段；剩余
  两段无法同时容纳“计划 active TERM、gen2 recovery clean pause、gen3 terminal”。本次 GPU 前的
  正常 exception exit 不是计划要求的 active TERM/no-exit，不能冒充该证据。
- 因此旧 session 永久保持 P6-C incomplete。修复提交完成 local/origin/server 同步及全部服务器门后，
  只能使用新 output、新 session ID、新 Genesis 按原四段计划从 gen1 重跑。P6-B 已闭合，不重跑；
  只需用其原始 formal bundle 做严格历史 loader/replay 回归。
- 这次属于重大问题，2026-08-22 的“无重大问题且门全绿”条件授权不再成立。新 P6-C GPU 启动前必须
  向用户报告并取得明确重跑授权；P6-D 仍未获准。

## 2026-08-23：第二个 formal session 在构建 gen3 prompt 时失败

### session 结果与保留证据

- 源提交：`ac8e291f9e685bb0a9d7fa7564ae8fab92ba6acf`；formal output：
  `/data/yuyangliu/true_rl_agent_phase_g/p6c_generation_ac8e291_20260822T205212Z`；
  session ID：`p6c-ac8e291-20260822T205212Z`。
- gen1 正常闭合为 accepted：winner `g1_stair_torso_clearance_v1` 的 public-accept objective 为
  `0.7714285714`，sealed objective 为 `0.7719387755`，7 个 sealed seeds 全通过，runtime
  activation 非零。
- gen2 的计划 active TERM 精确发生在两条 train 和两条 public-dev result 均 create-once 之后；
  recovery 只复用这四个原 job 并运行两条 public-accept，没有重复 train/eval reservation。selector
  合法闭合为 `no_selection`：两条 trial 均为 `qualified_not_selected`，`selected_trial_id=null`、
  `formal=null`，因此按冻结合同不运行 sealed，current-best 保持 gen1 winner。
- 两条 ledger 记录均有效保留，ledger SHA-256 为
  `50b0a3f75e2030c1739bb3eddb9d2f4527dbdb3fbfd980332556940fc26d6192`。gen3 的 manifest、
  authoring、provider 请求与任何 GPU job 均未创建。
- 截止失败的正式成本为 `6.194288285862778 / 12.5 GPU-h`、57/94 eval seed-units、
  994,371,600/1,591,551,200 simulator control steps、5/8 train jobs、4/30 provider attempts 和
  1,673,355,019/3,758,096,384 output bytes。失败段本身没有 provider/job/GPU 消耗，仅增加
  19,853 bytes 的 lifecycle 与异常证据。

### 失败状态与不可续跑结论

- 第四个 launcher attempt `attempt-20260823T020012.665357Z-pid827949-78968323` 在读取 gen2
  public history、构建 gen3 prompt 时抛出
  `ValueError: current provider record has invalid mechanism attribution`。segment 3 exit 已持久化
  `failed/exception`、`generations_at_end=2`、`continuation_eligible=false`；stderr SHA-256 为
  `bee643869dfa50b649cee91c1ae36d17b21a67c58d73f730eba92f28077d59d5`。
- 冻结 continuation policy 的 `max_segments=4` 已由 index 0–3 全部消耗。journal start 与 readiness
  会同时拒绝 `continuation_eligible=false` 和 segment 数量超限；修改 policy、删除 receipt、重写
  exit 或追加第五段都会破坏冻结合同/create-once 证据。因此本 session 永久保持 incomplete，禁止
  再执行 `--resume-existing`，也不得把前两代拼接到新 session。
- launcher/session/readiness PID 均已退出，两张 GPU compute-app 为空、利用率 0%，无 session/GPU
  lease 内核锁持有；磁盘上的 lease payload 只是已释放锁的持久记录。

### 根因与源头修复

- generation public history 有两种合法规范形状：有 formal winner 时顶层是唯一 round-v7 candidate；
  无 formal winner 时顶层是 generation-v1 envelope，两个 sibling 只存在于 `generation.trials`，不可能
  有唯一的顶层 `mechanism_attribution`。
- 通用 experiment memory 把 generation descriptor 的 `mechanism_delta_bound=true` 错解为“顶层必有
  一个 candidate attribution”，所以在合法的 gen2 `no_selection/formal=null` 后无法构建下一代。
- 修复只在已识别的 generation public projection 且 `generation.formal is None` 时返回空的 singular
  candidate component 列表。它不选择任一 lane、不合成 attribution；两个 sibling 的机制与结果继续
  由 generation-owned exploration projection 逐 trial 承载。有 formal winner 的 accepted/
  formal-rejected、普通 round-v7、legacy width=1 和 raw/unprojected generation 的 fail-closed 路径不变。
- 该分支同时要求规范 nested public-history schema、显式 `formal` 字段和值为 null；缺字段或伪造 schema
  仍按 provider attribution 规则 fail closed。共享 projection schema identity 移至轻量 generation
  contract；generation record 保留原名 re-export，旧导入路径兼容且 memory/context 不形成导入环。
- 回归把真实 `no_selection` 与零训练 `paired_authoring_incomplete` generation receipt 回灌
  `PublicContextBuilder.build_prompt`，验证下一代 context 可构建、generation outcome 被保留且没有伪造
  `candidate_id`/`candidate_components`。相关 memory/exploration/source-catalog/acceptance/session/
  fixed-width 矩阵共 163 项通过，1 项 Windows 平台跳过。最终本地全量为
  `Ran 1481 tests in 263.564s`、`OK (skipped=29)`；修改文件同时通过 Ruff `E4/E7/E9/F`、
  `py_compile` 与 `git diff --check`。服务器 Linux 全量门仍须在提交同步后记录。

该失败再次触发重大问题审批边界：修复完成 local/origin/server 精确同步与全部 Linux 门后，也只能在
获得新的明确授权后用新 commit、新 output、新 session ID 和新 Genesis 从 gen1 重启 P6-C。P6-B 不
重跑；P6-D 仍以完整 P6-C G4 为前置条件。

## 2026-08-23：最终 formal session 闭合，独立 G4 为 GO_FOR_P6D

- 最终源提交为 `415bfad6eb6e85c6cf0cf94851c6b39d1067116b`；local、origin、服务器 checkout
  与 formal report identity 精确一致，tracked checkout 干净。
- 新 output 为
  `/data/yuyangliu/true_rl_agent_phase_g/p6c_generation_415bfad_20260823T042422Z`，session 为
  `p6c-415bfad-20260823T042422Z`。它没有复用或拼接前两个失败 session。
- 生命周期按冻结四段闭合：Gen1 clean pause；Gen2 preselection 四个 job create-once 后 active TERM
  且无 segment exit；唯一 recovery 原样复用四个 job并在 Gen2 后 clean pause；Gen3 正常 terminal。
- Gen1 为 `paired_authoring_incomplete`，Gen2 lane B accepted 并成为 current-best，Gen3
  `formal_rejected` 且保持 Gen2 head。具有完整双公开证据的两代没有 selector material reversal。
- 17/17 reservation/result 全部 completed，0 open、0 retry。实际成本为 5 train jobs、
  983,040,000 train steps、64 eval seed-units、995,763,200 simulated control steps、10 provider
  attempts、6.281334262245833 allocated GPU capacity-hours、1,674,207,212 accounted output bytes；
  所有累计 cap 均通过。
- accepted winner 的运行激活门通过；一个未入选、贡献恒为 0 的 lane 被运行门拒绝，未改变科学 head。
  negative verifier-equivalent trained repeat 与 selector reversal 均为 0；frontier priority/high 本次无
  到达项，报告明确按 vacuous pass 处理。
- 终态两张 GPU compute-app、formal/GPU lease、owned process group 与同 incarnation 进程均为空。
- official report SHA-256 为
  `5b10f1a8e4d851ae542e2e7fd23d44f7f1e56718b3ee5fd58cd9721946c5afcc`。独立 G4 审计文件
  `p6c_g4_audit_v3.json` SHA-256 为
  `ad691299c0ca5b08383f984a68f514d987c1831cdd3b24d367a9ca34d7266a93`；审计前后 formal output
  metadata hash 不变，审计结论为 `GO_FOR_P6D`。

完整摘要见 `P6C_ACCEPTANCE_REPORT_ZH.md`。P6-C 至此完成且不需要重跑；后续 P6-D 使用新 Genesis、
空 ledger 和独立 output。

## 2026-08-24：P6-B/P6-C 独立复核后的主线减法

- 独立复核确认 P6-C 的 hash、ledger、四段 lifecycle、17/17 job/reservation/result、0 retry、成本 cap
  与终态资源闭合均真实成立；同时把科学口径收紧为“三条 generation 记录、两代有科学结果”。Gen1
  authoring incomplete、零训练零评测，不再被描述为完成的第三代科学实验。
- Stair smoke fixture 的 `num_envs=2` 与双足物理轴撞维，已改为不与当前任何 reward trailing dimension
  相同的 5，并由 shape-collision 回归锁定。这是任务夹具修复，不修改 reward、runner 或公共组件协议。
- 当前主线不再保留 P6-A2 的一次性 paired-ready correction/probe，也不再运行 P6-C 的 tracked runtime
  monitor。后者的不可变采样证据仍由冻结提交 `415bfad` 和原 formal bundle 解释；P6-D 当前主线依赖的
  权威边界收敛为启动前 compute-idle、runner-owned process-group cleanup、generation resource closure
  与终态独立审计。
- 新的受支持 launch/readiness CLI 与 launcher 只接受 formal readiness v3。后续静态审计确认 v2 reader/
  validator 同样无当前入口和覆盖，因此已整体删除；v1 仅保留被 v3 复用的底层 mapping validator。历史
  P6-B/P6-C readiness receipt 在对应冻结提交上审计，不由当前主线伪装兼容。
- 验收报告引用的是服务器 formal output 内 create-once 的 G4 制品；仓库根目录未跟踪的一次性审计
  driver 不属于验收权威，已删除，避免让当前主线隐式依赖无版本控制脚本。
- Gen2/Gen3 暴露的 Stair authoring 信息缺口在任务层修复：下一代 prompt 可看到最近一代结构完整的
  两条 public-accept-vs-parent 比较，即使更新的一代 authoring incomplete；内容只含公开 objective、
  guard、materiality 与 reason code，不引入 sealed/private/checkpoint 权限。
- 组合回归发现初版比较投影在 adapter 前把已脱敏的 parent 当成完整 `CurrentBestEvalStats` 重载；严格
  loader 正确拒绝缺少 `sealed_validation_evidence` 的 21 字段映射。根修改为先执行 Stair prompt scrub，
  再从保留下来的 protocol-bound public score stats 构造只读比较视图；不放宽完整 current-best loader，
  也不把 sealed evidence 放回 prompt。只有同时覆盖 lane A/B 的完整 pair 才能替换上一代比较；零公开
  结果或单 lane public-accept 的 incomplete generation 都继续向前查找。真实 15-generation
  pause/resume/finalization 链已覆盖脱敏边界。
- paired selector 的当前 schema 在少于两条 guard-eligible trial 时显式记录
  `paired_statistics.status=not_applicable` 且不输出伪数值；冻结 v3 contract/result 继续严格重放。
- 没有增加 lane A 失败后强行启动 lane B、短训练预筛或非对称预算。它们会改变冻结的固定宽度科学协议
  并引入多保真选择权威；若未来要研究，必须进入新合同，而不是作为本轮工程补丁。

上述改动不重写已闭合 P6-C formal bytes，也不授权重跑 P6-C。旧开发 session 在新 selector/prompt
contract hash 下不可 resume；历史证据只读重放，新实验必须使用新提交、新 output 和新 Genesis。

后续 P6-D 启动前独立复核用当前主线重新闭合：formal integration 模块以外的 Windows 全仓测试为
`Ran 1471 tests in 265.574s`、`OK (skipped=26)`；Torch-gated 三代用例明确 `OK (skipped=1)`；完整
formal 15-generation production-shaped 用例已改为接受 candidate 并移动 head，单独复跑为
`Ran 1 test in 576.545s`、`OK`。Linux/真实 Isaac/GPU 门没有在本轮执行，不能由这些本地证据替代。

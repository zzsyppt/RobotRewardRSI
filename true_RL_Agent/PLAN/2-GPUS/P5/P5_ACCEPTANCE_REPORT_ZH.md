# P5 验收报告：同 session Continuation、探索索引与有界重访

> 开发基线：`96e03abb792ac6525cf4f562ec2e2a8b27a60773`
>
> 分支：`true-rl-agent-refactor`
>
> 验收日期：2026-08-18
>
> 结论：**本地实现与 CPU/fake 验收通过；真实 Linux process-group/incarnation 门仍须在 P6-D 前实机通过。未启动 P6，等待用户检查。**

## 1. 交付结论

P5 已完成四条相互隔离、但能够组合的能力链：

1. generation session 可由多段物理进程续行，segment start/exit/unclean recovery、exclusive lease、ledger 高度和累计成本形成 create-once 线性证据链；
2. logical job hard crash 后只能在生命周期审计结算旧 reservation 后消费唯一 retry；job 已完成而上层 receipt 尚未发布时，只读重验既有 summary/checkpoint，不重复执行；
3. generation ledger 可确定性投影为 `ExplorationEvidenceCard → RevisitFrontier → lane intent/prompt memory`，删除派生索引后可重建同 hash；
4. 高价值 loser 可占下一代最多一条 revisit lane，另一条始终保留 novel；accepted、已重访、等价已吸收、重复 eligible 和完整负向项都有明确、可重算的去向。

科学权威没有改变：ledger/replay 仍是唯一 current-best 与 acceptance 真相；lifecycle、frontier、memory 和 prompt 投影都没有 acceptance authority。

## 2. 实际修改

### 2.1 同 session 生命周期与故障收口

- 新增 task-neutral generation continuation policy、session identity、segment journal、audit、cursor 和 terminal binding。
- start/exit/recovery receipt 使用独立 generation schema，不把 generation 数塞入旧 width=1 的 round lifecycle 字段。
- controlled pause/failed exit 与 hard crash 分开表示；hard crash 只能在取得同一 session lease、确认旧进程实例/专属进程组死亡并核对 ledger/cost head 后发布唯一 recovery receipt。
- start receipt 绑定 process-incarnation token，而非只记录可复用 PID：Linux 使用 boot ID + procfs starttime，Windows 使用进程创建时间。相同 PID 但不同 token 被识别为 PID 复用；旧 leader 已消失但旧进程组仍活时仍拒绝恢复。
- recovery receipt 逐条引用未闭合 attempt reservation；旧 attempt 以 `abandoned_unclean` 保守结算，失败成本不会因新 segment 清零。
- cursor 只给出下一 generation/segment 与当前 head，不保存 candidate、metric 或 decision。
- terminal binding 要求15条合法 generation、完整 continuation chain、无未闭合 attempt 和一致 replay head；它只是终局 Reservation 的输入，不复制 Stair final-confirmation 的 cohort/statistics 语义。

### 2.2 Runner exactly-once 恢复

- `JobAttemptJournal` 支持从 reservation 路径重建唯一 journal authority，并区分普通 result 与 recovery-bound result。
- 未闭合 reservation 仍必须先经过 lifecycle inspector，runner 不会自行猜测旧进程是否死亡。
- 新增已完成输出复用：存在 `completed/scientific_failure + output_hash` 时，runner 不申请 GPU lease、不启动子进程、不清理文件；它读取原 summary/checkpoint，重做 job identity、protocol、environment、device、checkpoint 和 typed result 校验，并核对 attempt output hash。
- summary/checkpoint 缺失或被修改时 fail closed，且不会以“再跑一次”掩盖问题。
- retry 次数、输入、environment、device、resource binding 和预算仍由原 attempt chain 单源约束。

### 2.3 探索索引与重访

- 新增 task-neutral `ExplorationEvidenceCard`、`ExplorationIndex`、`RevisitFrontier`、`GenerationExplorationPlan` 和 lane intent。
- 卡片分开记录证据完成度、公开观测方向、价值标签、失败范围、选择状态、谱系和重访处置，不压成一个伪精确总分。
- Stair-owned projector 只读取 public-dev/public-accept 与 parent 的公开 objective/guard；sealed 只保留既有 generic formal 状态，不进入价值分。
- provider/static/runtime/infra 未形成完整公开科学证据时保持 unknown/not-evaluable，不会给机制负分。
- frontier 的容量、backlog、逐卡 deferred 和 ineligible reason 显式可见；`verified_not_run` 不会伪造 public evidence，但会消耗一次服务机会。accepted-equivalent 与 duplicate-equivalent loser 不会重复占队列。
- `GenerationExplorationPlan`、lane intent 与 prompt 在消费前都会将 frontier 从同一 index 重新规范投影并逐字比较；调用方不能靠保留 `index_hash`、删卡或换序绕过 priority 服务。
- 自动重访只服务 priority/high；medium/eligible 继续作为提示但不自动占 GPU lane。priority 卡按 verifier-equivalence class 服务次数、head transition 距离、Stair-owned 公开增益排序键和新近度排序；同一等价类最多2次服务，任一等价卡取得 public evidence 即消费整类，超过1次 head transition 后过期。
- verifier 全量等价键仍是唯一阻断集合。模型最多看到2张 recent verifier-equivalent avoidance，以及最多2条按“组件种类 + 公开输入字段”聚合的非阻断负向变体 advisory；advisory 不建立机制家族，也不进入 acceptance。
- 完整 EvidenceCard、prompt 与 lane intent 不再共用错误的容量假设：12 KiB卡片保留在可重建索引；6 KiB prompt由唯一有限收敛器生成，每步必减少信息，无法表达的卡整张省略并记账；2 KiB lane复用 generation contract 的唯一字节门，只携带有界机制preview，放不下的task summary保留hash并显式标记省略。合法大卡不会再让prompt忙等，也不会在冻结lane时击穿容器。
- experiment memory 仍使用既有 14 KiB 总预算；最多投影有界正向/负向卡，完整索引不因 prompt 截断丢失。

### 2.4 单一 verifier 等价身份

- 新增的 verifier equivalence key 直接由既有 single-component operation delta 派生：`operation + component kind + parameter-normalized structure + public input fields`。
- key 有意不包含 parent、candidate/source hash、组件名或函数名，因此能跨 parent 拦截改名、改权重和普通数值幅值变化。
- tensor 下标、slice、`dim/axis/keepdim` 等结构配置仍保留在结构 hash 中，`[:,2]` 与 `[:,0]`、`sum(dim=1)` 与 `sum(dim=2)` 不会被误合并。
- P4 sibling exclusion、P5 负向历史 gate 和 authoring repair 共用同一 primitive；没有创建第二套 AST 分析器。
- 完整受信任排除集进入 authoring gate；prompt 只显示有界摘要。省略、截断或伪造完整集合 hash 会在 authoring 开始前拒绝。
- trial-specific `exact_implementation_key` 只用于审计实现身份；科学负向处置明确命名为 `suppress_equivalent`，不再用误导性的 `suppress_exact`。

### 2.5 Generation 与 memory 接线

- generation public history projection 升级后显式携带 parent current-best、lane intent 与 guard reference 的公开绑定；私有路径/receipt 内容仍不进入投影。
- generation-session contract 版本化增加 task-neutral exploration projector binding；Stair composition 只提供 Stair 自己的 projector contract hash。
- context/memory builder 的 exploration 参数是可选能力；不提供时 width=1 与旧插件仍走原路径、原 schema。

## 3. 严格验收证据

| 验收面 | 结果 |
|---|---|
| lifecycle/attempt hard-crash、gap/fork、identity、ledger height、budget chain | 状态机与本地注入通过；真实 Linux procfs/killpg 门待 P6-D 前实机验收 |
| completed runner output 复用与缺失/tamper 拒绝 | 通过；真实子进程第二次进入 launch count 保持1 |
| exploration index 重建、head 对齐、canonical frontier/lane、防伪造删卡、卡片分级、等价类服务上限/过期、ineligible/deferred/backlog | 通过 |
| 跨 parent/改名/改参数等价负向阻断，axis/index 结构差异放行 | 通过 |
| accepted incorporated、优秀 loser、mixed、verified-not-run、无科学证据 | 通过 |
| 合法大 EvidenceCard 的 prompt 有限降级、omission记账与2 KiB lane preview | 通过；无忙等且完整索引证据保留 |
| 15 generations / 30 trials experiment-memory 压力与 public safety | 通过 |
| P4 generation、P3 phase/selector、P2 width=1、Push/legacy 相关组合 | 通过 |
| 定向组合 | `Ran 130 tests; OK (skipped=1)` |
| 全量回归 | `Ran 1256 tests; OK (skipped=21)` |
| Python 编译 | `python -m compileall -q src tests` 通过 |
| 补丁静态检查 | `git diff --check` 通过 |

Linux-only 真实非 GPU 进程组用例在 Windows 本机按平台条件跳过；这不是通过证据，必须在 P6-D 前于 Linux 服务器或 CI 留下真实执行日志。其余 skip 为既有环境条件跳过，P5 没有用 skip 规避逻辑失败。

## 4. 真实15轮历史复核

我用当前生产 Stair projector 重新读取本地保存的15轮公开证据，而不是只依靠测试 fixture：

- 源ledger：`outputs/author_only_paired_ready_probe_20260817/source_snapshot/ledger.jsonl`，SHA-256 `1d49e22d0530b40332ced06e1f562d57855ba9ded927d258c32850b5634a95e2`；
- 当前生产projector输出：`outputs/p5_loser_heuristic_audit_20260818/REAL_ROUND_PROJECTION.json`，SHA-256 `a2bce73875e79c81b2801d14bb061250e287d78a6dacdf44377dc644213c35b3`；
- 等价碰撞明细：`outputs/p5_loser_heuristic_audit_20260818/EQUIVALENCE_COLLISIONS.json`，SHA-256 `2920c2bbbe15e4ece3c1593c3f5f5bd00ade7cabf61f943cd1e88ce671be5769`。

- 修复 `guard_reference=None` 后，分布为 `high=1 / unknown=1 / low=13`，而不是旧探针的 `high=2 / low=13`。旧 R2 缺少本代 parent public-dev guard reference，只能是 unknown；R3 是唯一 strong-positive/high。accepted trial 会在真实 generation 记录中标为 incorporated，不会成为待重访卡。
- 因此这15轮真实历史中的待重访 priority loser 实际为0；不能把唯一high当loser，也不能据此声称已经观察到重访收益。重访保持为机会型能力，P6即使没有priority loser到达也不算系统失败，且不得为提高触发率放宽high门。
- 15条候选形成14个 verifier equivalence class，只有 R1/R8 是已证实等价重复；该阻断保留。
- “base/foot 位置 + progress”公开输入签名出现6次、其中多数负向，但结构不同，不能升级为阻断或机制家族。当前实现只在相同 current-head context 中形成非阻断 advisory。
- 因此，当前证据支持的预期收益顺序是：continuation/副作用与成本保护；R1/R8等价负向避错（反事实重放表明，若两条出现在同一generation session，R8会在训练前被挡，可避免1次训练和评测；尚无P6真实节省样本）；同输入负向饱和advisory（已覆盖R5/R6/R7/R14/R15一类变体，但是否减少后续浪费仍待pilot）；最后才是尚无真实收益样本的high-loser重访。旧 width=1 历史不能给出 width=2 loser 的精确概率，也不能用 `0.133²` 当正式容量依据。P6只报告真实到达率和收益，不在同一 session 内调阈值。

## 5. 关键文件与验收时 SHA-256

| 文件 | SHA-256 |
|---|---|
| `agent_core/generation_lifecycle.py` | `5de69dac41610f330bbcfb8df19e37e65e8f85f165a7018b8b17493770f1b591` |
| `agent_core/exploration.py` | `7cc94cd8ff348abe9599a0a0554fa52d84423d54f0788eb25cac1167d9b02fe1` |
| `agent_core/exploration_projection.py` | `2b9ba6b318c8d5aed5ff15db6b3b483ea63a0c0d67fbd08b2cde6892ae608654` |
| `tasks/stair_climbing/exploration.py` | `1c1fe2ce875514ae4d58c430303c14eb6c9412fa768299d9449802f70cf6b7e9` |
| `runner/job_attempts.py` | `b74bdb144dd29ea36634b49a5859afec1a21c56af7552e6623e824dabf455a61` |
| `runner/subprocess_runner.py` | `79f455d59d33f81f89a932a97ddae365f86a70fe9fb5b78ea2752159408d7fb0` |
| `candidates/authoring_policy.py` | `88fd1b03b36f604ac68f820ec6ef8aad3a44afe7c7ca2eac3c40230a12158018` |
| `candidates/sibling_exclusion.py` | `4e831614c0e0f4e13881b730e6f862f72a3c87818b4839810bf7c719f32b0ac4` |
| `agent_core/authoring_loop.py` | `69c375acb33db2398d59597cf572ae1e6722636118095500546aca6af1c93afa` |
| `agent_core/generation_record.py` | `b4a9467bdd1f155554483d4165131ee86c775f10ab077388e26dbe07a553d473` |
| `agent_core/generation_contract.py` | `ada8a3ec3b56bcd02a8e38b9f5b8b184ee39cb8ca66f967bb357f6fd7d72b7c5` |

这些 hash 属于以 `96e03abb` 为 base 的未提交 P5 集成工作树，不能冒充 base commit 内容。
全部P5生产源码与测试文件的冻结清单见 [`P5_SOURCE_TEST_SHA256.txt`](P5_SOURCE_TEST_SHA256.txt)；报告/日志本身因引用该清单而不纳入，避免自引用hash。

## 6. 未实测与 P6 边界

- 未使用真实 GPU、Isaac、provider、服务器或正式 generation output；双卡并发吞吐、CPU/RAM/IO 争用、真实磁盘增长仍必须由 P6 分级 pilot 测量。
- 未在 Linux 上真实创建独立 process group 并执行 procfs incarnation + killpg 存活/回收门；P6 正式入口还必须保证 coordinator 是专属组 leader，Windows fallback 或 mock liveness 不能替代该验收。
- P5 提供 generation continuation authority 和 cursor，但正式 CLI/runbook、generation-aware final-confirmation 执行接线属于 P6；不得用本地 schema 测试声称正式15代已经可运行。
- provider request/token/金额、retry reserve、GPU-hours 和正式资源阈值仍需在 P6 Genesis 前由用户批准并冻结。
- 本阶段没有设计跨不同完整15-generation session 的自动重试或证据拼接。

因此，P5 可以提交用户验收，但**不能自动进入 P6**。

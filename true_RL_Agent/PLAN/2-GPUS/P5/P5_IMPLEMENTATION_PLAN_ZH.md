# P5 实际实施计划：同 session 续行、探索索引与有界重访

> 基线：`96e03abb792ac6525cf4f562ec2e2a8b27a60773`，分支 `true-rl-agent-refactor`。
>
> 范围：只实现 P5 所需的本地合同、投影、编排接口和 CPU/fake 验收；不调用 provider，不启动 GPU，不连接服务器，不开始 P6 pilot 或正式 15-generation 实验。

## 1. 目标与不变量

P5 解决两件不同的事，不能把它们做成同一个“图状态机”：

1. **运行续行**：同一个已经批准的 generation session 可以跨物理进程 segment 继续；所有退出、未闭合 job 和已消耗重试预算都留下不可改写证据，恢复只补未完成节点。
2. **探索复用**：从已验证 generation ledger 确定性生成探索卡和有界重访队列，让公开证据较好的 loser 能在新 parent 上成为一条明确的探索意图。

以下不变量贯穿实现：

- generation ledger 仍是唯一科学事实源；lifecycle receipt 只有运行与成本含义，不能产生 acceptance、score 或 current-best。
- 探索索引是可删除、可重建的只读投影；不能由模型写入，也不能成为第二个 ledger/experience 数据库。
- common 层只拥有 session、segment、receipt、hash、evidence tier、队列和字节边界；Stair 的 objective、guard、公开 gate 和价值分级只在 Stair adapter 中。
- 继续复用 P1 `JobAttemptJournal`、P2 TrialReceipt、P3 phase receipt、P4 generation record；不复制 runner、selector、acceptance 或 verifier 逻辑。
- 当前 width=1 round 的 single-invocation 与 Push 行为逐字保持；P5 新能力只属于 generation-capable session。
- 只讨论同一套 output/Genesis/session contract/final plan/ledger epoch 内的 continuation，不设计跨不同完整 15-generation 实验的重试或拼接。

## 2. 当前边界

### 已有且复用

- P1 已有 create-once logical-job reservation/result、一次 typed infra retry、GPU lease 和精确进程清理。
- P2 已有无 ledger 副作用的 trial executor、节点 receipt 和 exactly-once width=1 commit。
- P3 已有 public-dev/public-accept/sealed phase receipt、Stair selector 和 typed assembly。
- P4 已有严格 A→B authoring、固定 width=2、两 lane 并行执行、一次 selection、winner-only sealed、一次 generation commit 和 generation-v1 replay。
- `execute_fixed_width_generation()` 已优先复用闭合 authoring/train/phase/execution receipt，因此续行层不需要重新实现节点调度。

### 真实缺口

- generation 还没有独立 lifecycle segment schema、exclusive session lease、unclean recovery receipt 或终局 continuation audit。
- P1 attempt reservation 在 hard crash 后可能只有 reservation 没有 result；另一个更窄但同样重要的窗口是 runner 已写完 result，而上层 Trial/Phase receipt 尚未发布。P5 必须分别处理“未闭合 attempt 的保守结算与唯一 retry”和“已闭合输出的只读重验复用”，两者都不能重跑已经完成的科学 job。
- generation ledger 中的两条 trial 还没有被投影为稳定 EvidenceCard/frontier；现有 experiment memory 主要面向 width=1 round。
- P4 的 lane intent 已被 manifest/hash 绑定，但还没有一个由 frontier 生成、可验证 `inspired_by_trial_id` 的规范化 schema。

## 3. 实施包

### P5-A：generation lifecycle 与 hard-crash 收口

新增 task-neutral generation lifecycle 模块：

- create-once continuation policy，冻结 segment 上限、允许退出/恢复原因、一次 logical-job infra retry和 exclusive lease 规则；其 hash 必须等于 generation session contract 已绑定的 `continuation_policy` hash。
- generation 专用 segment `start/exit/unclean-recovery` receipt；不用旧 round 的 `rounds_requested` 等字段。
- start 绑定 session/Genesis/session-contract/final-plan/ledger-epoch/output-root、continuation index、前驱 receipt hash、ledger generation 高度、cost head和 lease identity。
- controlled exit 绑定最后 ledger 高度、未闭合 logical job ref、cost head和是否允许继续；旧 receipt 永不覆盖。
- hard-crash recovery 必须先取得同一个 session lease并证明旧进程实例已死亡，再为旧 start 发布唯一 recovery receipt；不能只凭可复用的 PID 判断，也不能伪造旧 exit。generation start 记录 task-neutral process-incarnation token：Linux 使用 boot ID 与 `/proc/<pid>/stat` starttime，Windows 使用进程创建时间。
- recovery inspector 只把无 result 的 reservation 结算为 `infrastructure_failure / abandoned_unclean`，并让 result 绑定 recovery receipt hash；已闭合、非 infra 或 retry 已耗尽的 attempt 不会获得新机会。
- 已有 `completed/scientific_failure` attempt result 时，runner 不再申请资源或启动进程；它只从同一 job 的既有 summary/checkpoint 重建 typed result，重走身份、协议、checkpoint 和输出 hash 校验。文件缺失、漂移或 hash 不同都 fail closed，绝不把它降级成新 attempt。

验收重点是链 gap/fork、并发 owner、identity/environment 漂移、ledger 高度回退、attempt 删除或预算重置全部 fail closed；不尝试防御有权限删除整个 output 的攻击者。

### P5-B：同 session continuation cursor 与终局资格

实现只读 audit/cursor：

- 从 generation ledger replay 得到唯一 current head和下一 generation index。
- 从 lifecycle receipts 验证唯一线性 segment chain、前后 ledger 高度单调、受控 exit 或合法 unclean recovery、同一 session identity和连续 cost head。
- cursor 只指出“下一步从哪里继续”，实际节点仍由 P4 coordinator 根据 immutable receipts复用；cursor 不保存 candidate/result。
- generation terminal eligibility 要求：从空 generation ledger 开始、恰好15条合法 generation、完整 segment chain、没有未结算 reservation、无 active旧进程、预算链连续。旧 width=1 final gate不变。

本阶段提供 generation final-confirmation 的资格绑定输入，不复制 Stair 已有 paired cohort/statistics/`acceptance_authority=false` 语义；真正 terminal Plan/Reservation/Result 继续复用并版本化现有 Stair final-confirmation 模块。

### P5-C：确定性 ExplorationEvidenceCard

新增 task-neutral card/index schema：

- card identity 绑定 session、generation/trial、parent/current-head context、candidate/source、verifier-owned single-component delta/equivalence、train/eval/selector contract和证据 ref hash。
- 正交字段：`trial_status`、`evidence_tier`、`observed_effect`、`novelty`、`failure_scope`、`selection_status`、`lineage_status`、`revisit_disposition`。
- common projector只负责遍历已验证 generation records、形状与 hash；task adapter负责生成有界 `public_evidence_summary` 和 observed effect。
- Stair adapter只使用 public-dev/public-accept、冻结 parent objective/guard和既有 acceptance threshold/gate定义：好 loser可为 high/medium；dev/accept冲突为 mixed；infra/author/static/runtime不形成负机制结论；sealed只允许 generic formal status，不进入价值分。
- exact identity/equivalence只引用 P2 verifier产物；不按组件名、rationale、embedding或输入字段自动聚合“机制家族”。

### P5-D：RevisitFrontier、lane intent 与 memory

- frontier 是完整索引上的有界、确定性队列。自动重访只服务 `priority/high` 卡；`eligible/medium` 只作为 prompt advisory，避免证据很弱的旧卡占用稀缺训练 lane。
- priority 卡先按 verifier-equivalence class 已服务次数、相对 current head 的接受推进距离、task-owned 公开增益排序键和新近度排序，而不是机械地“最老优先”。同一等价类最多获得2次服务机会，任一等价卡取得 public evidence 即消费整类；超过1次 head transition 后过期。一次没有 public evidence 的重访仍计为一次服务，防止同类卡轮换着永久卡住队首。
- 每代最多一条 revisit；至少一条 novel。没有可用 priority 卡时两条都是 novel。
- revisit lane intent绑定 `inspired_by_trial_id/card_hash`，明确这是“在当前 head 上重新实现”的假设；不复用旧 checkpoint、不复制旧源码。
- accepted card保留正式证据但标记 incorporated，不留在待重访队列；完整公开负向证据只抑制 verifier 能证明等价的单组件语义（`suppress_equivalent`）。该等价身份忽略 parent、source、组件/函数改名和参数幅值，但保留 tensor 下标、slice、axis/dim 等结构语义；一次 candidate/infra 失败不会污染机制价值。
- trial-specific `exact_implementation_key` 只用于审计“这一份实现是否原样重复”，不再作为机制排重真相。完整受信任排除集可覆盖全 session；prompt 显示完整阻断数量、最多2张最近 verifier-equivalent avoidance，以及最多2条按“组件种类 + 公开输入字段”聚合的非阻断负向变体 advisory。advisory 只提醒模型别继续磨同一输入区域，不生成机制家族身份，也不能缩小或扩大 verifier gate。
- frontier 容量外的每张卡都保留 `card_hash + queue_position + deferred reason`；不能只留下一个不可追溯的 backlog 数字。
- frontier、lane intent 与 prompt 都必须是同一 `ExplorationIndex` 的 canonical 重算结果；仅携带相同 `index_hash` 的删卡、重排或未知 card hash 投影一律拒绝。
- 完整 EvidenceCard 与下游载荷分层：12 KiB 卡片只存在于可重建索引，prompt 使用独立6 KiB allowlist投影，lane intent 使用 generation contract 的2 KiB权威门。prompt降级只能调用一套有限、每步严格改变状态的收敛器；单卡无法表达或最小卡仍超限时，整张prompt预览被确定性省略并增加omitted计数，不能删除索引证据或忙等。lane只携带等价key、操作、组件种类、有限输入字段preview及总数/省略数；task-owned summary放不下时保留hash并显式标记正文省略。
- experiment memory按 generation 投影：优先保留当前 head、最新 generation和最多4张 frontier/avoidance卡；仍在现有14 KiB总上限内，超限时确定性降级卡片而不删除current-head必要证据。
- provider prompt只接收公开卡片摘要和lane intent；raw sealed、seed、checkpoint path、provider transcript、私有receipt path均不可见。

### P5-E：故障注入、兼容和文档

- 使用 fake coordinator/runner 在 A后、B后、train后、两screen后、confirmation后、append前后模拟 pause和hard crash；继续后只补缺节点且commit exactly once。
- 覆盖 loser high/medium、both-good/no-selection、accepted incorporated、mixed、verifier-equivalent negative、无卡的infra/partial、队列backlog和15代/30 trial memory压力。
- frontier 必须显式区分“已经有公开证据的重访”“已被 accepted lineage 吸收的等价项”“服务次数耗尽”“相对当前 head 已过期”和“多个 eligible card 属同一等价项”；`verified_not_run` 不能消费科学证据，但必须消耗一次调度服务机会。
- 使用合法大卡覆盖两类预算倒挂：超过prompt字段显示合同的卡必须保留索引并记omitted；full/minimal prompt都超过6 KiB时必须有限步删去预览而非自旋。合法4 KiB内task summary不得击穿2 KiB lane intent，lane应保留等价身份并显式记录summary/字段省略。
- 使用真实15轮公开证据回放 Stair projector：缺少 parent public-dev guard reference 时不得产生正向/高价值判断，不能因调用方漏字段而异常；若 public-accept guard 已独立证明失败，仍可保留 negative。输入签名聚合只能形成非阻断 advisory。
- 证明关闭 exploration projector 后，给定相同candidate receipt，selection、formal decision、sealed gate和current-best replay完全一致。
- 跑 lifecycle/exploration/generation/acceptance/memory/Push聚焦测试，再跑全量测试和`git diff --check`。
- 产出 P5 acceptance report与DEV_LOG，明确实现、验证、未实测和留给P6的边界。

## 4. 有意不做

- 不引入 LangGraph、Temporal、Ray、Neo4j、SQLite event store、MLflow/W&B作为运行真相。
- 不做动态DAG、任意width、工作窃取、跨机器调度、自动扩缩容或通用Agent runtime。
- 不做UCB、bandit、MAP-Elites网格、learned surrogate或全局0–100机制分数。
- 不让frontier改变selector、acceptance、sealed查询规则或current-best。
- 不为一般程序等价增加新AST分析器；继续复用现有single-component verifier。
- 不用大规模安全测试覆盖不可达组合；只覆盖能够污染证据、重置成本、重复副作用或泄漏private信息的边界。

## 5. 完成门

P5只有在以下条件全部满足时才算完成：

1. lifecycle与attempt recovery是create-once、线性、可重算且同session绑定；hard crash不能获得第二份免费attempt。
2. continuation cursor与terminal audit能独立发现gap/fork/stale parent/未结算reservation；width=1 gate保持不变。
3. exploration index从ledger删除缓存后重建同hash，且replay head一致。
4. 所有符合Stair预注册公开条件的loser生成对应卡；公开 negative 只抑制 verifier-equivalent 单组件语义，结构不同的同输入变体仍可探索，unknown不污染价值。
5. frontier最多占一条lane，只有 priority 卡可自动占用；弱卡不能饿死强卡，同一 verifier-equivalence class 最多服务2次，任一等价卡取得 public evidence 即消费整类，超过1次 head transition 后过期，容量不足时逐卡报告 deferred；inspired-by不改变旧card。
6. 15代/30 trial prompt-memory压力仍不超过14 KiB，private/sealed信息为零。
7. P4 generation、P3 selector/eval、P2 width=1、Push及全量测试均通过。
8. 不启动GPU/provider/服务器实验，等待用户验收后才进入P6。

## 6. 实际落地结果

实现按上述边界完成，并在审计中做了七项源头收敛：

1. lifecycle 与科学历史彻底分层：segment/attempt 只证明“谁运行过、成本是否连续、是否允许续行”，只有 generation ledger/replay 能生成 current head 和探索卡。
2. 跨代避错不再使用 source hash、parent 或组件名。唯一 verifier 等价 key 由现有 single-component operation delta 的参数归一化结构与公开输入依赖派生；同一 primitive 同时服务历史负向 gate、P4 sibling exclusion 和 P5 frontier。
3. completed job 的恢复不再停留在“receipt 已存在就报错”：runner 可从闭合 attempt result 只读重验既有 output 并继续封存上层节点，避免 coordinator 在极窄崩溃窗口里重复训练/评测。
4. 重访调度不再机械最老优先：仅 priority 自动重访，task-owned 公开增益排序、服务次数、head 距离和新近度共同决定顺序；服务次数与过期边界防止单卡垄断。
5. 全量 verifier-equivalent 负向集合继续是唯一阻断真相；prompt 另提供少量 verifier-equivalent avoidance 与同输入负向变体 advisory，后者明确非阻断、不构造机制家族。
6. lifecycle 不再把 PID 当进程身份；start/recovery receipt 绑定 process-incarnation token。真实 Linux procfs/killpg 分支必须在 P6-D 前于 Linux 环境验收，本地 Windows/mock 结果不能替代该门。
7. 完整EvidenceCard、prompt与lane intent按用途分层；prompt共用唯一有限收敛器并显式记省略，lane复用generation contract的唯一2 KiB门且只带有界preview。合法大卡不再造成忙等或冻结lane失败，完整索引证据不受提示预算影响。

本地验收结果：

- P5 lifecycle/exploration/authoring/runner 定向组合：`Ran 130 tests; OK (skipped=1)`；
- 全量：`Ran 1256 tests; OK (skipped=21)`；
- `python -m compileall -q src tests`：通过；
- `git diff --check`：通过；
- 没有调用 provider、GPU、服务器，也没有启动 P6 或正式 generation session。

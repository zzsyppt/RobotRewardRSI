# P4 实施计划：固定 Width=2 Generation、一次提交与兼容回放

> 实施状态：2026-08-17 已完成本地代码、fake/CPU 验收，等待用户验收。未运行 provider、GPU 或服务器实验，未进入 P5。

## 1. 本阶段目标

P4 只把已经完成的 P2 单候选执行能力和 P3 分阶段评测能力编排成固定的双候选 generation。它不重新定义 Stair 的评分、selector、sealed、acceptance，也不提前实现 P5 的跨 segment continuation、探索价值与重访队列。

完成后的固定拓扑为：

1. 冻结同一 parent、base public context、训练初始化、评测绑定和 A/B lane 资源；
2. 严格串行 author A，再将 verifier 生成的公开安全单组件语义排除提供给 B；
3. A/B 都 verified 后才开放训练屏障，并行训练与 public-dev；
4. 调用 task-owned phased adapter 封存一次 selection；
5. 两个 sibling 都运行 public-accept，只有冻结 winner 运行 sealed；
6. 复用正常 decision/validation，整个 generation 原子提交一条 generation-v1 记录，head 最多推进一次。

## 2. 不可越过的边界

- common core 只管理固定拓扑、身份、receipt、并发、一次提交和回放投影；Stair 的 guard、objective、seed 数和评测摘要只存在于 Stair adapter。
- coordinator 不自行拼 runner argv、不解析 Stair summary、不比较分数、不读取 sealed 原始证据。
- sibling 等价性只调用 `compare_single_component_deltas`；任何 unavailable 都是 B lane 内可修复的 authoring invalid，不能视为 distinct。
- selection receipt 必须早于任何 public-accept receipt；public-accept 不得触发重选。
- 旧 round-v7、Session Genesis v2/v3、Push/width=1 writer 和 replay 字节语义保持不变。
- generation ledger 是唯一科学真相；运行 journal/receipt 不能推进 current-best。
- P4 保存 P5 重建 EvidenceCard 所需的公开事实与引用，但不计算 value band、frontier 或跨代去重结论。

## 3. 实现分解

### 3.1 合同与身份

- 新增 create-once `generation_session_contract.v1` 和每代 `generation_manifest.v1`。
- 以 Session Genesis 新 capability schema 显式绑定 generation contract 的逻辑引用和 checksum；v2/v3 保持只读兼容。
- session contract 冻结 width=2、A-first/B-second、generation writer kind、selector/cost/protocol hash、最大 generation/trial/train 数和两条 lane 的资源绑定；manifest 只引用合同并冻结本代 parent、base context、guard reference 和 lane assignment。
- generation ID、trial ID、lane ID 使用稳定 closed contract；完成顺序不进入 canonical 排序。

### 3.2 Genesis parent public-dev

- 将 Stair phased cost contract 升级，显式拆分一次性的 session bootstrap public-dev（3 seed-units）和每代 23 seed-units。
- 从已验证 Genesis 的 reference eval plan 生成 public-dev-only phase spec，使用现有 runner 和 TrialReceiptJournal 发布 create-once receipt，再调用唯一的 Stair screen evidence producer。
- 第一代 manifest 绑定该 parent evidence；后续代绑定已晋升 winner 的 public-dev receipt。不得从旧 monolithic ledger 另写 screen projector。

### 3.3 串行 authoring 与语义排除

- 定义 task-neutral、公开安全的 `single_component_semantic_exclusion.v1`，由现有 operation delta 投影生成；provider 可见部分仅含 operation、组件种类、canonical delta、公开输入依赖和结构 hash。
- A/B 使用独立 provider、目录、journal 和预算；A 完成前 B 不启动。
- B 的每次候选在 normal verifier/smoke/anti-degenerate 之后，再调用同一单组件比较原语；identical/parameter-only 与 unavailable 都进入 B 自己的 bounded repair。
- 任一 lane authoring 不完整时 generation 零训练闭合；A verified/B failed 时使用 `verified_not_run`，不降级 width=1。

### 3.4 分阶段 Trial Executor 与 Coordinator

- 从现有 width=1 executor 中抽取候选准备、训练、eval-plan binding 和正式结果收口原语；旧 `execute_candidate_trial()` 继续通过这些原语执行单体评测，避免两份 candidate seal/materialize/decision 逻辑。
- coordinator 只在 both-candidates-ready 后并发调用两条训练 lane；每条 lane 继续由 runner/GPU lease/attempt chain 控制资源。
- public-dev 完成后构造 typed screen，task adapter 产生并封存 selection；selection 是否有 winner 都不能再被后续证据改写。
- 两个有效 sibling 随后都并发运行 public-accept；无 winner 时不运行 sealed、head 保持不变，有 winner 时仅 winner 运行 sealed。adapter assembly 与 runner projection 生成正常 `PublicEvalResult`，再复用现有 decision/record validation。

### 3.5 Generation record、一次提交与回放

- 新增独立 `generation_ledger_record.v1`；round-v7 与 generation-v1 都是各自 record kind 的 current writer，不切换全局 round schema。
- record 内按 lane ID 稳定排序，保存 authoring/operation delta、train receipt、public phase receipt、selection、正式 winner record/ref 和公开安全事实；私有路径只在顶层 private artifacts，不进入 nested public projection。
- head transition 由唯一 selected lane 的正式 record确定性投影，generation record 不再复制第二份顶层 candidate/result/decision。
- commit 先在锁外对不可变 ledger 快照完成 receipt-backed replay、parent、generation index 与 session contract 校验；锁内只重新加载 canonical ledger、执行 snapshot compare-and-swap、commit-slot 唯一性和一次 append。重复完全相同提交是 no-op，冲突或并发快照漂移 fail closed，慢速 receipt I/O 不占用写锁。
- acceptance replay、public history 与现有消费者先通过统一 generation head-transition projector；无 head transition 的 generation只贡献公开历史，不改变 current-best。

## 4. 验收矩阵

### 4.1 正确拓扑

- A→B 顺序、B 唯一动态可见信息、ready barrier、两条训练并发、selection-before-accept、双 public-accept、winner-only sealed。
- exact/rename/parameter/alpha-equivalent 均阻断；结构不同放行；comparison unavailable 消耗 B attempt。
- A 失败不启动 B；B 耗尽不启动训练；不存在第三 trial、第二 selection、第二 sealed 或 loser fallback。

### 4.2 身份与防篡改

- parent/base context/Genesis/session contract/ledger epoch/eval binding/training initialization/lane resource 全部交叉绑定。
- guard reference 必须等于 manifest 冻结 parent evidence；换成旧 parent、sibling、自造 receipt 或跨 plan receipt 均拒绝。
- 调换异步完成顺序得到同一 canonical generation record/hash。
- nested path/private evidence 不进入公开 history/prompt。

### 4.3 一次提交与兼容性

- append 前后故障、重复 commit、同 ID 不同内容、stale parent、round/generation 混写均有确定结果。
- v2/v4/v5/v6/v7 round replay保持；v3仍 fail closed；Push/width=1 全部回归。
- generation replay 的 current-best 与独立投影一致，零 transition generation 不更新 head。

### 4.4 成本与任务边界

- 每代科学上限严格为 2 train、2 dev、2 accept、1 sealed、23 seed-units；session 另有一次 Genesis parent dev 3 seed-units。
- common core 的测试不导入 Stair score/guard 字段；selector 只存在于 Stair adapter。
- 本阶段只跑 fake/CPU 测试，不调用真实 provider、不启动 GPU、不触碰服务器正式实验。

## 5. 交付门

1. 小范围合同与单元测试通过；
2. fake generation 端到端与故障注入通过；
3. round-v7/Push/Session Genesis/P2/P3 聚焦回归通过；
4. 全量测试不低于 P4 开始时 `1209 passed / 21 skipped` 的行为覆盖，新增测试全部通过；
5. `git diff --check` 干净，P4 验收报告明确区分“已实现/本地验证”和“尚未真机 pilot”；
6. 未经用户下一道审批，不启动任何 GPU/API/正式 generation。

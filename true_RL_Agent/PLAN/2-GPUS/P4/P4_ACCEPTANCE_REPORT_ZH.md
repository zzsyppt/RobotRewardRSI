# P4 验收报告：固定 Width=2 Generation

## 1. 验收结论

P4 的本地实现与 CPU/fake 验收通过，可提交用户审查。当前交付已经把 P2 的无副作用 trial 原语和 P3 的分阶段评测原语编排为一个固定 width=2 的 scientific generation；没有复制 Stair 评分、selector、sealed 或 acceptance 逻辑，也没有让 worker 直接写 ledger。

本结论不等于可以启动正式双卡实验。审计后仅追加了 5 次无 GPU 的生产同形状 provider 复查；32K 成稿能力仍未闭合。双 GPU 资源竞争、generation live lifecycle、session-wide 金额治理、P5 frontier 和 generation-aware 终局确认均尚未运行或尚未实现；未经用户验收不得进入下一阶段。

## 2. 基线与范围

- 分支：`true-rl-agent-refactor`
- P4 开发基线：`3af99dd6999f5119c5458fd6218efe22e34febda`
- 初始 P4 实现只修改代码、测试和 `PLAN/2-GPUS/P4` 文档；用户审计后又修正 provider 默认、width=1 failure commit 幂等性和 P4 文档。
- 初始验收未调用 provider。随后按用户要求对 5 份冻结 prompt 做生产 8 字段 exclusion 复查，调用了 provider/服务器网络，但未启动 GPU、train/eval、scientific ledger、shadow 或正式 generation。
- 工作树中原有的历史输出、压缩包和其他未跟踪文件不属于 P4，也未被修改或清理。

## 3. 已实现结构

### 3.1 单一合同链

- 新增 create-once `generation_session_contract.v1` 和每代 `generation_manifest.v1`。
- Session Genesis 以 v4 capability 显式绑定 generation contract ref；v2/v3 仍是 width=1 历史/当前兼容路径，未原地放宽。
- manifest 只冻结本代 parent、base context、guard reference、训练初始化、lane intent 与资源 assignment；selector、cost、provider、retry 等会话级合同只通过 session contract hash 绑定。
- Stair phased cost v2 将“一次性 parent public-dev bootstrap 3 seed-units”和“每代 23 seed-units”分开表达；删除了 v2 内重复的旧别名字段。

### 3.2 串行 A→B authoring

- A、B 使用独立 authoring callback、provider 状态、输出目录和 journal；A 未闭合时 B 不启动。
- A verified 后，由现有 verifier-owned 单组件比较原语投影公开安全的 `single_component_semantic_exclusion.v1`；B 只能看到该净化投影。
- lane-specific intent 和 B exclusion 不仅写入 manifest：coordinator 会读取最终 provider prompt 验证实际注入，generation replay 还会从 sealed prompt/evidence 再验证一次。
- B 的相同、改名、参数等价或 comparison-unavailable 候选走既有 bounded authoring repair；coordinator 不另写第二套等价判断。
- A verified、B 失败时记录 A=`verified_not_run`、B=`candidate_failed`，整个 generation 零训练；不存在 effective-width=1 降级。

### 3.3 固定执行图

已实现且顺序固定为：

1. A→B 串行 author/verify；
2. 两份有效 candidate evidence 齐备后才开放 ready barrier；
3. A/B 并行训练；
4. A/B 并行 public-dev；
5. Stair task-owned adapter 生成 typed screen 并 create-once 封存 selection；
6. selection 封存后，A/B 才并行 public-accept；
7. 仅冻结 winner 运行 sealed 并复用 P2 正式 decision/validation；
8. 整代只由 coordinator 原子提交一次。

no-selection 仍保留两条完整公开评测证据，但不查询 sealed、不推进 head。winner 正式失败后不回捞 loser。

### 3.4 Receipt、ledger 与 replay

- 复用 P2 `TrialReceiptJournal`，补充 train/eval phase receipt 的独立权威加载器。
- 新增 manifest、selection、execution 三类 create-once generation receipt；execution receipt 只是幂等恢复依据，不拥有 acceptance 权。
- 新增独立 current record kind `generation_ledger_record.v1`。一次 generation 一行，A/B 固定排序，私有路径只在顶层 private artifacts。
- commit 先在写锁外对 canonical ledger 快照完成 receipt-backed replay，并校验 frozen parent、generation index 与 session contract；写锁内只做 canonical snapshot compare-and-swap、commit-slot 唯一性和 append。相同重提是 no-op，冲突或并发快照漂移拒绝，慢速 receipt I/O 不占写锁。
- round-v7 与 generation-v1 不能在同一 epoch 混写；旧 v2/v4/v5/v6/v7 round replay 保持，v3 继续 fail closed。
- acceptance replay、public history、source catalog 和 experiment record 通过 generation 的唯一 validated projection 消费结果；只有 selected formal trial 能产生 head transition。
- record replay 会重新加载并核对 authoring、train、public-dev、public-accept、sealed、selection 和 guard reference receipts，不能只信 nested projection。

### 3.5 Stair-owned parent guard reference

- 第一代从已验证 Genesis reference eval plan 额外执行一次 public-dev-only phase并发布正式 receipt。
- 后续 generation 只从上一代已接受 winner 的真实 public-dev phase receipt 恢复 parent guard reference。
- 两条路径都复用唯一的 `StairPhasedEvaluationAdapter.project_public_dev_screen`，没有第二个从旧 monolithic ledger 猜测 screen 的生产者。

## 4. 严格验收结果

### 4.1 行为与故障注入

已覆盖：

- A→B author 顺序与动态信息最小化；
- B 缺少实际 exclusion、exclusion hash 被篡改、authoring evidence 被篡改时 fail closed；
- A author failure、B author failure、B 等价机制耗尽均零训练；
- 训练/phase 并发完成顺序不改变 canonical record；
- selection-before-public-accept、双 public-accept、winner-only sealed、no-selection 零 sealed；
- guard reference 必须等于 manifest 冻结 parent；
- A 已闭合而 B 中断后，只复用 A 并继续 B，不重 author A；
- append 前后恢复、重复 commit、stale parent、session-contract 漂移与 round/generation 混写；
- generation public projection 不泄漏 receipt 路径或 private artifacts；
- 已接受 winner 的 source 与 public-dev receipt 可作为下一代唯一 parent 输入。

### 4.2 测试

- 初始 P4 核心 generation/contract/receipt/authoring：44 项通过。
- 初始 P4 的 P2/P3、replay、memory、source catalog、round-v7、Push、backend 联合回归：219 项通过。
- Stair cost 单一真相清理后的相关回归：35 项通过。
- 审计修正后的 generation/provider/live-loop 定向回归：84 项通过。
- 最终全量测试数字以本报告末尾的“审计后最终验收”记录为准。
- 新增模块编译检查通过；`git diff --check` 通过。
- 当前虚拟环境未安装 Ruff，因此没有临时安装新工具或把 lint 结果冒充验收证据。

### 4.3 用户审计后的收敛

- **生产 exclusion 证据纠偏：** 旧 `86.7%` 探针使用了信息更充分且违反最终信息流的 payload，现已明确降为上界。补做 5 次生产 8 字段同形状复查；只有 R01 形成完整候选且人工确认 distinct，另外 4 次均没有正式候选，不能用来估计碰撞率。完整记录见 `PRODUCTION_EXCLUSION_RECHECK_REPORT_ZH.md`。
- **Provider 合同：** 后续默认切到 `deepseek-v4-flash`、`reasoning_effort=max`；thinking 请求不再发送 temperature。16K 样本中 3 次推理耗尽输出上限，因此默认 `O` 最小幅度提升到 32K；32K readiness 仍为 OPEN，不授权 GPU。
- **测试可诊断性：** 原 1026 行端到端方法保留一套共享 fixture，避免复制复杂 setup，但 13 个验收场景已用具名 `subTest` 隔离；selected-state 和锁外 replay 另有独立单元测试。单个断言失败不再遮蔽所有后续场景。
- **锁内复杂度：** receipt-backed 全历史 replay 已移出 ledger 写锁；锁内只解析 ledger、比较预验证快照并 append。15 代仍需线性读取 canonical ledger，但不再把每代所有 receipt I/O 放进临界区。
- **分层债务：** `acceptance_replay` 对 generation public projector 的函数内反向 import 是真实低风险债务；本轮没有为消除一条 lazy import 搬动稳定模块或引入 registry。P5 在建立 ledger-derived exploration projection 时应把 generation head/public projection 下沉到 task-neutral ledger projection 层。
- **Width=1 幂等性：** provider/static authoring failure 两条旧写入路径改为 `append_ledger_record_once`，与主 round 路径一致。
- **错误诊断：** selector 指向未知或重复 trial 时现在抛出明确 `ValueError`，不再泄出 `StopIteration`。
- **组件名边界：** exclusion 中的 canonical 组件名是 provider-visible locator；它不是重复判定真相，blocking comparison 继续忽略改名并使用 normalized semantics。代码合同和主计划均已写明。

### 4.4 审计后最终验收

- `python -m compileall -q src scripts tests`：通过。
- `git diff --check`：通过。
- generation/provider/live-loop 定向集合：84 项通过。
- `python -m unittest discover -s tests`：`Ran 1221 tests`，`OK (skipped=21)`。
- 集成大场景仍复用一套真实 fixture，避免复制 600 余行 setup；13 个行为/故障场景已分别命名并隔离，另有 7 个 generation contract/receipt 独立测试。报告不再把相邻既有用例数量写成“新增测试数”。
- 服务器临时目录 `/tmp/production_exclusion_recheck_20260817` 已删除并确认不存在；没有启动或遗留 GPU 作业。

## 5. 明确未交付与下一阶段门槛

以下内容不属于本次“本地 P4 scientific generation”完成声明：

- 尚无正式 generation live CLI/15-generation session 入口；当前正式 width=1 live loop 未被改写。
- generation lifecycle segment、跨进程 continuation、generation live summary/audit/final-confirmation gate由 P5 接续；旧 width=1 single-invocation 合同未放宽。
- session-wide provider input/output/金额 reservation 与累计 cost governor 尚未接到生产 generation 入口。P4 只冻结 contract hash 和固定“最多两条 lane、各自最多五次既有 authoring attempt”的拓扑；当前默认 `O=32,768` 也只是代码默认，不是已获批 session 成本合同。数值化金额与 retry reserve 必须在 P5/P6 启动前获批并接线。
- 未验证两张真实 GPU 的并行利用率、CPU/RAM/PCIe 竞争、墙钟收益和失败后的进程清理；这些属于 P6-A/P6-B。
- 未实现 ExplorationEvidenceCard、RevisitFrontier、跨代重复索引或 prompt 投影；generation record 只保存 P5 可重建的公开事实和 refs。
- generation-aware session-end final confirmation 尚未实现；现有 width=1 Stair final confirmation 保持原样。

因此，下一步只能在用户验收后进入 P5 的会话层和探索索引实现，不能直接启动双卡实验。

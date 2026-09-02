# P0 实施计划与验收清单

> 状态：已完成，等待用户验收
>
> 执行日期：2026-08-15；根因纠偏验收：2026-08-16
>
> 基础 commit：`true_RL_Agent@e8b1e9e80f4b2a12af993b691cb8a76692cc3dfd`
>
> 集成候选：基础 commit + 用户已有 final-confirmation 修复 + 本次 P0 verifier 根修；不是 `e8b1e9e` 的干净工作树

## 1. P0 的目标

P0 不实现双卡 coordinator，也不改变训练、评测或接受语义。它完成四件事：

1. 用当前代码和真实正式产物建立可复核的单卡基线；
2. 冻结 P1 以后必须遵守的协议边界、成本口径和任务归属；
3. 把缺失信息明确标为缺口，避免后续用口头估计或新增第二真相源补齐。
4. 对 P0 复核中已被真实源码和生产函数证明会击穿 width=2 的判重根因，复用现有 verifier 分析流水线做一次源头纠偏，并冻结后续 sibling exclusion 必须复用的同一原语。

P0 的成功不是“代码越多越好”，而是后续阶段不再争论基线、权限和计量口径。

## 2. 明确不做

- 不实现 P1–P6；
- 不启动 GPU、API、shadow pilot 或下一轮正式实验；
- 不同步服务器、不提交、不推送；
- 不修改现有 eval-v6、20-step stable success、acceptance、sealed 或 round-v7 语义；
- 不把 Stair 的价值分类、cell、失败类型放入公共层；
- 不建立 Graph 数据库、第二份 ledger、第二套重复机制分析器或新经验库；
- 不改写、删除或补写已闭合正式 ledger；
- 不触碰工作树中用户已有的 final-confirmation 修复；
- 不运行 author-only API 探针；该探针只能在 P4 生产 exclusion 路径完成后、P6-B 前经用户批准执行。

## 3. 分步实施

### 步骤 1：冻结代码与工作树边界

工作：

- 核对本地 HEAD、origin、服务器 checkout 和分支；
- 记录已有 tracked/untracked 用户修改；
- 明确区分干净 commit、P0 入场工作树和 P0 集成候选；
- 允许修改的生产代码仅限 verifier 单组件语义比较、历史 authoring-anchor 投影及其直接测试；其余仍只写 `PLAN/2-GPUS/P0` 和更新主计划。

验收：

- 本地/origin/服务器基础 commit 身份一致；
- 用户已有修改保持原样；
- 当前集成候选的变更文件与 SHA256 单独记录，不冒充 commit；
- `git diff --check` 通过。

### 步骤 2：闭合真实单卡基线

工作：

- 只读核对正式 session 的 process/tmux/GPU/lifecycle；
- 核对 15 条 round-v7 ledger、15 个 train summary、15 个 eval summary；
- 核对唯一 invocation、live summary、current-best、checkpoint/hash 和 final-confirmation 状态；
- 汇总 authoring、训练、三种 eval split、provider usage 和磁盘成本。

验收：

- closed ledger 数、live summary 数和 current-best 一致；
- partial/运行中产物不冒充正式记录；
- 运行错误与科学拒绝分开计数；
- 无法从持久产物证明的 GPU/RAM 峰值明确标为“未记录”。

### 步骤 3：冻结 provider 与成本口径草案

工作：

- 记录当前 provider endpoint/model、temperature、reasoning、transport、`O=16,384` 和 512 KiB response envelope；
- 使用 provider 官方价格快照重算本次真实 authoring 成本；
- 给 P1 提供简单、可实现的 request 上限 `I` 与 generation/session 金额上限草案。

验收：

- reasoning units 不与 response units重复计费；
- input cache hit/miss 分开计价；
- `I` 明确标为 P1 待实现的发送前硬门，不冒充当前已有能力；
- `I=300,000` 和金额上限只作为宽松 reserve 草案；P1 tokenizer/preflight 完成后必须按真实分布与最大 contract-valid prompt 重新校准；
- 未经用户批准，P6 不得使用金额草案启动请求。

### 步骤 4：冻结探索价值和重复处置边界

工作：

- 只用公开 runner evidence 定义 Stair-owned `strong_positive / partial_or_mixed / negative / unknown`；
- 明确 value 是 context-scoped trial evidence，不是全局机制分；
- 用真实 R1/R8 和 R4/R11 源码复现旧门的 anchor 漂移与改名漏口，纠正“只因 supplied source 不完整”的旧根因；
- 复用现有 AST/组件分析流水线，将比较对象收敛为“候选与历史候选分别相对各自 anchor 的唯一变化组件”：忽略组件/函数名，比较 kind、存在性、输入依赖、callable/clip 结构及参数归一化签名；归一化声明参数与 callable 中的普通算术、缩放和阈值幅值，保留 tensor 下标/slice 及 reduction/shape 调用的维度/轴配置；
- identical/parameter-only 历史重复由 verifier blocking；同输入但结构不同的 rejected 变体跨 parent 保持 advisory；frontier 只消费结果，不重做 AST、名称或 embedding 判定；
- 为 P4 暴露同一 task-neutral `compare_single_component_deltas` 原语；未来 A→B sibling exclusion 必须 blocking，不能另写一套协调器判重；单组件语义不可用时抛专用异常并由 B fail closed/repair，不能把 unavailable 当 distinct；
- 无论历史比较是否命中，在 evidence 中记录有/无 authoring anchor 的历史 source 数量与 candidate anchor 可用性；legacy full-source fallback 只保留兼容权，不得成为 sibling exclusion 权威；
- 冻结 A-first/B-second、B 唯一可见 exclusion、temperature/reasoning 规则。

验收：

- sealed raw、provider rationale、人工标签不进入 value；
- infra/static/runtime incomplete 不被写成科学 negative；
- 一次失败最多抑制 exact implementation/context，不封杀机制家族；
- R1↔R8 必须归为 `parameter_only_delta` 并阻断；R4↔R11 必须保持结构不同，不得误杀；
- `projected_gravity[:,2]` 与 `[:,0]`、`sum(dim=1)` 与 `sum(dim=2)` 必须保持 structurally distinct，而 `x*0.5` 与 `x*0.25` 仍为 parameter-only；
- 缺历史 anchor 时必须可从 evidence 直接看见 coverage 降级；compound/unavailable sibling 必须修复或失败，不得训练；
- B 看不到 A 的源码、rationale、response、路径、训练或评测结果。

### 步骤 5：冻结兼容与测试基线

工作：

- 记录历史 v5/v6 和当前 v7 ledger golden 的路径、记录数、字节数与 SHA256；
- 运行 final-confirmation/lifecycle/replay/provider/mechanism/prompt 定向测试；
- 运行全量单元与集成测试、compileall 和 diff check。

验收：

- 干净 `e8b1e9e` 固化为 `1150 passed, 20 skipped`；
- P0 入场工作树（另含用户 final-confirmation 修复）固化为 `1153 passed, 20 skipped`；
- P0 集成候选的定向与全量结果在完成根修后重新执行并记录，不预填数字；
- 没有新增 skip；
- compileall 与 `git diff --check` 通过。

### 步骤 6：形成 P0 handoff

工作：

- 写出权威基线与协议冻结报告；
- 记录已解决、仍存在、留到 P1/P6 的事项；
- 更新主计划状态，但不把 P1 标为已启动。

验收：

- 主计划、P0 报告和开发日志互相一致；
- 报告具有 SHA256，可在后续阶段按哈希引用；
- 明确写出“等待用户批准 P1”。

## 4. P0 出口门

P0 判定为完成需要同时满足：

- 代码身份和正式 session 身份闭合；
- 15/15 科学 live 已闭合，正式 head 可由 ledger 重放解释；
- final-confirmation 的“live 闭合”和“终局结果已产生”没有混淆；
- 单卡成本分布、provider usage 和磁盘量已量化；
- GPU/RAM 峰值缺口没有被伪造；
- `64 GiB` 只作为 P6-A 待实测的暂定值；
- Stair value/repeat/authoring 规则已由文档与 verifier 根修共同冻结，并明确未来实现归属；
- paired-ready 只冻结未来探针方法，不在 P0 消耗 API；
- 本地回归通过；
- 没有服务器写入或新实验。

## 5. P0 之后

P1 只有在用户检查并批准 P0 后才能开始。P1 的首要内容仍是：runner process ownership、typed logical-job attempt/retry、双 GPU lease，以及 contract-valid prompt 的确定性 minimum-core；不能直接跳到两个现有 `run_candidate_round()` 并行。

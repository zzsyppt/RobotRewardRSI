# Phase 1 验收：证据进入作者前的可信边界

日期：2026-09-23。**PASS，完成当前主线的完整 Phase 1 范围。** 本次是生产源码修复与 CPU 链路验收，0 次真实 provider 调用、0 次训练、0 次新模拟器评估。它不证明诊断质量或搜索收益；高层姿态、时序档案、视觉和预测记忆仍由后续阶段交付。

实施依据：[实施卡](./IMPLEMENTATION_PLAN_ZH.md)。起点为 `fe96455ddd1a8c6ba30ee16e6a3a015422afe8d3`，改动保留在工作区。最初验收的[检查记录](./acceptance_checks.json)与三份原日志保留历史身份；审计后的修订及复测见[逐条处置](./AUDIT_RESPONSE_ZH.md)。P0 历史测试不计入本阶段。

## 最终架构与所有权

```text
绑定的训练 / phase receipt
  → 共用的 live / replay phase 投影
  → task adapter 定义 public-dev 事实及任务含义
  → 隔离的作者视图 + 固定身份 / 决策检查
  → 有界历史与逐 lane 记忆
  → 最终 provider 请求 JSON
```

- `generation_record.build_trial_phase_projection` 是 generation 的 phase 表示构建点，live 和 replay 共用。重放从已绑定回执重建摘要，ledger 的 `public_summary` 不再能独立改写后成为作者事实。摘要与回执不一致时拒绝该消费，不静默换来源。任务统计仍由 `PhasedEvaluationAdapter` 解释。
- `RuntimeActivationEvidence.measurement_summary` 统一测量状态：缺证据为 `not_available`；部分字段未知为 `null`；真实计数 0 和真实贡献 `false` 保留。两个 runtime producer 补自身已知的步数和声明 term 数，不从空采样列表推断“没有 term”。加载包只证明加载，不能冒充执行。
- `PublicContextBuilder` 给任务 projector 独立副本，已验证历史继续服务身份校验和源码解析。任务可以删减反馈、改为 public-dev 评分，不能通过原地修改污染 runner 的身份或决定。
- 普通轮次记忆保留短 `runtime_observation`；generation 仅在 `trial_outcomes` 第 10 列保存每 lane 的奖励执行状态，删除顶层重复项。详细计数仍在原 runtime 反馈和回执中。旧 9 列补 `not_available`，旧测量字典压为同义状态；仅有记录而无法证明执行时为 `unknown`，不从 runtime effect 或单独的 false 补造执行。既有 16 轮 / 14 KiB 边界和遗漏计数继续生效。

生产实现均在 `src`，回归在 `tests`，没有新增运行服务、数据库、模型调用或任务公式公共层。P1 的生产改动不读 PLAN；已有 reference 的 PLAN 字符串是历史来源元数据，实际执行源码来自 `src`，没有因此改写其历史身份。

**信任基底的适用范围：**本次新增的逐 phase 回执重建适用于当前 generation 主线。仍可调用的旧 width-one round 在提交时核对完整 execution receipt，重放则以已提交 ledger 的公开结果为可信基底、重新验证决定，不从外部回执重建任意诊断字段。本次没有为旧格式补一套回执引用协议，因此不宣称可检测其提交后任意 metrics 修改。第三任务测试也使用无 schema 的通用记录 fixture，只验证公开投影/记忆/请求边界，不冒充 round-v7 回执真实性测试。

## 验收对应

| 要求 | 实际证据与结论 |
|---|---|
| 回执、身份及 live/replay 一致 | 完整 generation 测试使用真实文件形式的 CPU fixture receipt；正常 live phase 与重建 phase 相等，重复公开历史/最终 payload 相等。仅改 dev/accept 摘要或交换 sibling 摘要均拒绝。 |
| accepted / rejected / 未训练 / 未选中 / 首次 reference / mixed sibling | `test_fixed_width_generation` 覆盖所有这些分支与恢复；reference 使用 journal 真正发布的 fixture 回执，替换旧不存在的假路径。mixed 无 formal 时，成功 sibling 的 phase、父 public-dev 和父源码身份仍闭合。 |
| 缺测、无 term、零执行、实测零和容量遗漏 | 新 runtime 测试直接执行 Push runtime producer；另检查共享归一化、两任务反馈、public history、3 次记忆重压缩。generation 第 10 列逐 lane 保持缺测/零值；容量遗漏沿用 `omitted_for_capacity`，不变成 0。Stair producer 合同通过现有回归。 |
| 记忆与预算 | 20 代合成 sibling 历史产生明确遗漏计数，保留行的身份和测量不变；原 generation 压力回归保留未选中 lane 索引，重压缩幂等，大小不超过 14 KiB。 |
| public-dev 与私有 / accept / sealed 分界 | 复用任务投影、公开安全及容量测试；第三任务特意在输入放 canary，验证共享私有过滤与任务 public-dev 白名单后，最终请求中均无 canary。接受结果与作者反馈评分分别保留其用途。 |
| 任务无关性与最终请求 | 最小 `fixture_orbit_tracking` adapter 通过真实 acceptance replay、任务记忆、context builder、provider 请求构建到 JSON；4/6 分母、证据编号、身份、`null` / `0.0` 保持。没有新增生产玩具任务。 |
| 投影纯净与幂等 | 合法的原地反馈删减不改变原件；candidate / accepted 原地篡改被拒绝，已验证历史不变。临时取消隔离时回归确实失败，证明测试能检出原漏洞。 |
| 原有运行路径 | 40 个 round / live-loop 集成回归通过，保留真实训练之前失败、训练失败、后续重试和拒绝的既有语义。未改变 success、接受门或奖励计算。 |

“真实文件形式的 fixture 回执”指本地测试经生产 journal 生成并验证的文件，不是机器人实测；最终 provider 检查止于真实请求构造和序列化，没有请求模型。历史 P0 原件重算用于基线闭合，不冒充本阶段新增机器人证据。

## 首次验收的历史运行结果

| 批次 | 数量 | 结果 / 原始日志 |
|---|---:|---|
| 18 个相关单元模块 | 223 | PASS，28.326 s；[日志](./verification_unittest.log) |
| fixed-width generation 完整集成模块 | 2（含多个生命周期子例） | PASS，63.373 s；[日志](./verification_generation.log) |
| fake round、round public boundaries、live loop | 40 | PASS，47.631 s；[日志](./verification_round.log) |
| P0 原件 / 库存 / 请求 / 计量检查 | 28 项检查 | PASS；[P0 记录](../P0/acceptance_checks.json) |

首次验收共 **265 个测试通过，无跳过**；这些日志不是后续审计修订版本的测试结果。P0 的 28 项检查单列，不与测试数相加。补充的 `generation_session + trial_executor` 长批次在旧内存版本下启动，未完成后主动停止，不计入 PASS；本次不把额外 15 代会话回放扩成必要验收。可复跑命令见 [开发记录](./DEVLOG_ZH.md)。

## 审计修订后的验收

本轮源头修复及未采纳建议的理由见[逐条处置](./AUDIT_RESPONSE_ZH.md)。完整计量不再逐代复制进有限记忆；短状态保留缺测、部分未知和实测零的区别，没有改变生产者计数、奖励计算或接受规则。

| 批次 | 数量 | 结果 / 本轮原始日志 |
|---|---:|---|
| 原18个相关单元模块，包含新增部分记录反例 | 224 | PASS，19.631 s；[日志](./verification_audit_unittest.log) |
| fixed-width generation完整集成模块 | 2 | PASS，52.629 s；[日志](./verification_audit_generation.log) |
| fake round、round public boundaries、live loop | 40 | PASS，37.229 s；[日志](./verification_audit_round.log) |
| 三个真实历史公开记忆的冻结fixture | 1（3个检查点） | PASS，0.023 s；[日志](./verification_audit_retention.log) |

修订版本共 **267个测试通过，无跳过**。没有将重复执行相加，也没有把独立审计对先前源码运行的1735项全量测试算作本轮结果。正常测试fixture保留41个原件hash，字段与原公开payload核对一致；测试运行不读取PLAN。固定预算下，fixture的Stair第13/16代前保留12/13代，Push第6代前保留5代，分别为13,814/14,111/6,525字节；这是压缩回归，完整历史回执重放另列。

**真实历史回放：PASS。** 沿服务器原回执路径，原基线与当前源码各完整回放Stair 15代、Push 5代，原驱动未修改且使用full模式。修订后Stair第13/16代前保留12/13代、Push第6代前保留5代，与fixture结果一致，满足事前11/13/5门槛；字节均不超过14,336。Stair长检查点比原基线14代少1代，属于已登记容量取舍。

- 33/33与15/15份phase摘要与ledger语义相等，三个完整payload重复构建相等，两个任务current-best与基线一致。
- 仅改public_summary时，当前源码均拒绝；原基线仍接受，保留P1修复有效的对照证据。
- 已生成payload共60行前9列与基线相同，最新及最近接受代均保留，省略计数闭合，第10列均为短状态且无generation顶层重复。
- 10个同步源码hash与最终本地一致，历史两会话的8个身份文件前后hash一致。此检查只覆盖这8份身份文件，不称为全历史目录hash审计；没有改写旧receipt或启动新机器人实验。

完整命令、来源与比较见[真实回放记录](./retention_regression/run_summary.json)，机器汇总与最终指纹见[本轮检查记录](./audit_fix_checks.json)。原审计及首次验收记录保持不变。

## 已处理的 P0 七项与下一阶段前置

七项均真实存在，其中人工评注硬编码属于维护性问题，未改变历史结论。已删除旧估计器及派生字段、孤儿 JSON 和完成使命的一次性脚本；修正 3 个链接；采集器直接保存原件；302 个库存来源采用仓库相对路径；人工评注成为有来源绑定的数据 fixture；parent 的远端权重和训练原件已补齐。原请求、receipt、候选及历史 source baseline 保留，详见 [P0 补修记录](../P0/AUDIT_RESPONSE_ZH.md)。

P2 有一个明确前置：fca parent 的训练 manifest 是 v1（`54cd180c…`），Gen8-A 是 v2（`193fa903…`）。训练合同相同，不能据此自动推定旧 parent 已获当前 v3 runner 准入；现有下游仅接纳当前或直接前驱 manifest。启动 P2 时应处理可验证的祖先兼容路径，不能改写历史 manifest 或简单放宽 hash gate。完整核查在 [P2 范围卡](../P0/RESOURCE_SCOPE_AND_MIN_LOOP_CARD_ZH.md)。本次没有越界修改该准入规则。

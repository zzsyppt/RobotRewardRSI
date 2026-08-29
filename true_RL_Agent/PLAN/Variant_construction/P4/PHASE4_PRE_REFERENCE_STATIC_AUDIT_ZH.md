# Phase 4 预 reference 多轮静态审计

> 审计日期：2026-08-29
> 起始基线：`b22ef28a2f379505d0ef4069290303e2fc65b5c7`；最终修复提交以本文件所在 Git commit 为准
> 状态：独立静态审计反馈已闭合；当前静态门通过，下一步按 P4-B 运行 reference

## 1. 审计目标与边界

本轮只验证 Phase 4 启动前的系统边界，不评价策略表现：

- runner 继续独占训练、评测、接受、sealed 与 current-best；
- LLM 只接收 `public_dev` 反馈，只能提交完整 reward program；
- `public_accept` 只参与系统内部接受判定，sealed 不进入下一轮 prompt、memory 或 exploration；
- Stair 的 suite、family mask、ID/OOD 与 C2 语义留在 Stair TaskPlugin；共享层只提供任务可插拔的
  provider-view 投影边界；
- 不新增 terrain cell、LOOP、runner 或通用安全框架。

本轮没有启动 reference 训练、candidate 训练、LLM/API 请求或 GPU 实验。修复已同步到服务器线性
checkout，并在同 SHA 临时隔离测试检出中完成 Torch 全量回归；真实 provider request 扫描和行为验收仍
属于 P4-B/P4-C。

## 2. 多轮审计结论

### Round 1：权威 replay 与任务投影顺序

确认数据流为：原始 ledger → runner-owned acceptance replay → task-owned provider projection → 通用
scrub/prompt。任务投影不能反向影响 replay 或 current-best。

发现共享层原先只保护少数身份字段，任务投影理论上可改写 job、binding 等非反馈事实。修复后：

- `public_result/decision/generation` 之外的顶层事实整体冻结；
- generation index/outcome、trial status/candidate identity、formal selection 与 runner decision 继续冻结；
- 篡改 train job 的 `program_source_checksum` 会在 prompt 构造前 fail-closed。

### Round 2：`public_dev` 与已存 feedback 二源

发现 unified Stair provider view 原先直接信任 ledger 中已存的 `prompt_feedback`。虽然正常运行时它由
`public_dev` 生成，但 prompt 时没有证明两者仍一致。

修复后，首次投影必须用冻结 suite、evaluation semantics、Genesis family mask 和原始 `public_dev`
重算 feedback，并与已存值逐字段相等；漂移立即拒绝。二次投影只处理已经删去原始内部 split 的
provider view，因此保持幂等。

同时缩小 generation 暴露面：

- trial 只保留 `public_dev`、候选身份、状态和机制摘要；
- 删除 `public_accept`、内部 status reason、guard reference 原文；
- formal 只保留选择/结果语义，删除 sealed receipt checksum；
- 测试同时注入隐藏 marker、内部 accept/sealed 分数并证明最终投影均不含这些值。

### Round 3：四条 provider 消费链与 legacy 接口

逐条核对 public history、task sections、source lineage、memory/exploration：

- accepted/current-best/best-observed 的 provider 分数均来自统一的
  `prompt_feedback.generalization_summary.public_dev_objective_score`；
- exploration 仅比较 candidate `public_dev` 与 parent `public_dev`，不读取 acceptance threshold；
- memory 的 trend/best/latest 均读取投影后的 `public_dev`；
- 内部 `public_accept` 和 sealed 仍完整保留给 decision/replay，不因 provider 投影而削弱。

`TaskPromptAdapter` 的投影方法已由 Neutral、Stair 和 Push 三个内建 adapter 显式实现；Push 使用恒等
投影，旧任务语义不变。

### Round 4：初始 anchor 的接受分数回退旁路

发现一条未触发但真实存在的恢复旁路：任务声明自定义 provider score source，却漏传相同语义的
初始 anchor score 时，共享层会回退到 `CurrentBestEvalStats` 的内部接受分数。

现已在共享边界收敛为单一不变量：非 bootstrap anchor 一旦切换 provider score semantics，就必须
显式提供对应 anchor score；否则 fail-closed，禁止静默回退。正常 Phase 4 Genesis 已提供该值。

### Round 5：P3B 静态门的跨 Phase 收敛

P3B 验收器原先把评分聚合源码也当成永久不变的物理闭包；Phase 4 在 reference 之前修正 family mask
规则后，suite/protocol/manifest 身份必然改变，但 C2 geometry、训练分布和历史 GPU 回执并未改变。

处理方式不是改写历史 GPU 证据：

- 历史 C2 runtime/training receipts 保持原 hash、原内容；
- verifier 只允许列明的 Phase 4 评分源码与协议发生漂移，并逐项核对当前 C2 geometry digest、训练
  profile 和 cell id 仍与历史物理回执一致；
- 当前 snapshot 明示 historical/current contract hashes 及
  `rollout_geometry_or_training_changed=false`，不把旧行为证据冒充为新评分结果。

### Round 6：三代终态 stale summary 消费者

独立复现确认 unified `public_evidence_summary` 已使用 `guard_floor/named_guards/objective` 新形状，而
validation 终态仍读取旧 `guard/materiality_threshold/materiality_met`。这会在三代全部关账后崩溃；同源的
`public_dev_accept_consistency` 也会永久得到空分母。

源头修复为：

- unified 终态不再从 provider/public-dev 摘要猜测 public-accept 结果；
- runner 根据 ledger 中绑定的两个 sibling `public_accept` phase receipts，使用冻结 mask、acceptance
  threshold 和 named-guard APIs 独立复算 reversal 与 consistency；
- legacy summary 消费路径保留，旧任务行为不变；该私有人工报告指标不进入 prompt、memory 或 selector。

### Round 7：32 KiB 反馈容量与目标可解释性

真实形状 preflight 证明旧压缩顺序会在首代降到 `minimum_core` 后删除 stratum/C2 diagnostics，而约
74% 预算被静态 ABI 和说明占用。根因不是预算本身，而是重复字段和错误的压缩优先级。

修复后：

- reward ABI 字段改为自描述表格，删除重复 prose，不删除字段身份、dtype、shape 或 unit；
- current-best 与 latest candidate 都保留 family scores、八层 stratum table、每层主要失败和全部非零
  diagnostic flags；重复的 per-cell/factor membership 数组退出 minimum core；
- `public_eval_contract.objective_structure` 直接由 suite `score_contract` 派生，公开
  `J_ID/J_OOD/J_primary`、family→strata 和冻结 mask 规则，不在 prompt 复制第二份手写 family 表。

### Round 8：ceiling family 无保护与空 mask 异常

双边 `[0.10, 0.90]` 规则把 reference 已掌握的 ceiling family 从目标中剔除，候选可无成本破坏该能力；
而所有 family 同时落在 floor 下时，评分投影会先于 qualification 抛异常，使计划中的“停在诊断”成为
死路径。

收敛为唯一规则：

- mask 在 family 聚合层只排除 `score < 0.10` 的 reference-floor family；ceiling family 继续进入
  `J_OOD`；
- strata 成员由 suite 预注册，不按 reference 逐层挑题；
- 空 mask 只允许在 reference 诊断链存在，仍产出全 family measurements、具名 readiness failure 和
  mask hash；candidate/selection/replay 继续要求非空 mask。

### Round 9：重复 parent public-dev 与 operation topology

unified Genesis 已拥有三相 reference 的正式 `public_dev` receipt，generation bootstrap 却再次运行同
checkpoint、同 suite、同 seed 的 216 episodes，并靠改写 argv 中 job-id/summary-path 避免碰撞。

现已让 unified generation 直接严格重载 Genesis 的绑定 phase receipt，删除 argv 手术、额外 job 和
parent operation slot；legacy Stair 仍保留旧 bootstrap。`reference_eval_phased` 布尔参数不再穿透八层
函数，cost contract 根据已冻结的 ordered-suite topology 派生是否复用 receipt，reference phase slots
由 `EvaluationPhase` 枚举派生并拥有独立命名空间。

### Round 10：同源与边界复审

再次扫描确认：

- active unified Python 不再硬编码 `F_scalar/F_c1/F_c2` 聚合清单，family 来自 suite
  `score_contract`；
- pre-Genesis phase role 集合与 reservation kind 均由 `EvaluationPhase` 派生；
- active source/tests 中不存在 `reference_eval_phased`，产品路径不存在 argv job-id 隔离 helper；本文只在
  历史问题说明中保留该名称；
- Stair roughness、family、stratum、mask 和 failure 语义仍只在 Stair TaskPlugin；共享 acceptance/runner
  没有新增 Stair 术语；
- 没有启动真实 reference、API/provider、Isaac、GPU、pilot 或 validation。

### Round 11：服务器回归、calibration 与 minimum-core 单一真相

服务器 Torch 环境解除本地 skip 后，三代集成用例暴露一条旧测试期望：重复 parent public-dev 已在
Round 9 删除，成本边界的 `eval_seed_units` 正确为 `0`，但同一测试仍保留 `==3`。该断言已改为
`0`；服务器真实用例完整穿过三代 pause/resume/crash recovery、终态报告和 private
public-accept selector 指标。

同轮复审还收敛了四个窄缺口：

- calibration cohort 只含部分 OOD family 时，聚合仅消费当前 cohort 实际拥有的 family，不再把
  declared family 当作必然齐全而触发 `KeyError`；无任何 OOD family 时仍沿用原 fail-closed；
- minimum-core 不再维护第二份手写 selection guidance，直接保留 task adapter 生成的 reference、
  cold-start 或 legacy 模式化 guidance；
- minimum-core 恢复 `public/public_accept/sealed_validation = 3/5/7` seed 数及
  `seed_values_exposed=false`，LLM 可正确判断反馈统计量级；
- unified 终态 selector 报告要求 Genesis family mask 已冻结，缺失时立即拒绝，不再静默退回
  pre-reference 聚合。

相似项复扫未发现第二处硬编码 family 聚合、漏改 parent 工作量断言或不受约束的 unified 终态 mask
消费者。`post-parent-bootstrap` 仍保留为既有成本账本边界名，但 unified 路径工作量为零；不为改名
引入账本迁移。

### Round 12：stratum 失败率截断真值与服务器最终静态门

独立审计使用真实 `StairEpisodeOutcome -> unified summary -> prompt feedback -> compact entry` 链路复现：
每层只保留 top-2 非零 failure，但全局列名取各层并集；旧表格对本层未保留的列统一执行
`get(..., 0.0)`，因而会把“确实非零但被 top-k 截断”的失败类型伪装成“真实零”。这会直接误导
LLM 对 C2 等层的失败归因，属于进入真实 API 前必须关闭的反馈真值缺陷。

源头修复保持既有容量边界，不扩张反馈：

- 每层在压缩时同时记录 top-2 投影和已知被截断的非零失败名；
- 表格 schema 升为 `unified_stratum_diagnostics_table.v2`：已投影值保持数值，已知截断的非零值写
  `null`，从未发生的真实零仍写 `0.0`；
- schema 常量只有一份，初次压缩和 latest-feedback 幂等路径共同消费，不再维护两份字面量；
- 回归测试从真实 outcome 构造三种等频 C2 failure，证明第三种非零 failure 为 `null`，同时证明真实
  timeout 零仍为 `0.0`，并覆盖二次压缩幂等；
- 相似压缩复扫确认另一处 `omitted_rates_are_zero` 对应完整计数网格，没有 top-k 丢弃，省略项确为
  真零，不做无关改造。

该修复只位于 Stair prompt adapter；runner、接受协议、训练/评测分布、suite、family mask 和旧任务插件
均未改变。尚未发起 reference、provider/API、candidate 或 validation。

## 3. 验收结果

- 权威服务器完整回归：commit `66e9c3bb97267663406e91b63f826acd97cc9e12`，Torch
  `2.7.0+cu128`，`1549 tests` 全部通过，`3 skipped`，耗时 `1008.815s`。P4 回归门只认该服务器
  数字；
- 本地快速诊断：unit `1462 tests` 全部通过、`28 skipped`；integration `86 tests` 全部通过、
  `2 skipped`。这些 skip 包含服务器实际执行的 Torch 路径，因此不再把本地结果写成静态 GO；
- P3B verifier：PASS；其自测 `4/4`。当前 snapshot hash 为
  `92b94b749e490e45b8ad9daf70079299174d0827a544b3c930d980b22474b861`，formal cost 仍为
  `1080 episodes`，历史 runtime/training receipt hashes 未变；
- unified 最坏合法 prompt：`30798 / 32768 bytes`，headroom `1970 bytes`，content hash
  `a5465d32e9ce7a7fa403687730fb664086818e070b8c00ca8982d1e3334fcefc`；两条 feedback entry 均保留
  3 个 family、8 个 strata、causal failure/diagnostic table 和 45 个 reward ABI 字段，同时保留 12 条
  原始模式化 guidance 与 `3/5/7` seed 数；v2 failure table 不再把已知截断的非零率伪装成零；
- unified validation scientific cost：reference eval `15` seed-units，每代 `23`，成功路径合计 `84`；
  `parent_bootstrap_seed_units=0`。operation plan 无 `parent_public_dev`，legacy 兼容路径仍有独立测试；
- `compileall`、`git diff --check`：PASS；静态扫描确认 active source/tests 无 `reference_eval_phased`、
  argv job-id 隔离 helper、硬编码 `F_scalar/F_c1/F_c2` 聚合或 Stair task 语义越出任务目录；
- 新增 helper/接口均有产品调用者；未保留重复评测套件、开发偏 launcher 或额外安全框架。

## 4. 尚未产生的证据

本静态 GO 不代表 Phase 4 完成，也不代表 reference 可学、C2 family 达到支持地板或 LLM 能提升策略。以下
证据仍未产生：

- official reference checkpoint、资格化结果与 `objective_family_mask_hash`；
- 1-generation pilot 的真实 provider/train/eval/decision/ledger 闭环；
- fresh 3-generation validation、真实 provider request 泄漏扫描与 GPU resource closure；
- `loop_verified` 和 `improvement_observed`。

独立静态审计提出的最后一项阻断反馈真值缺陷已在 Round 12 关闭，服务器静态门也已重新通过。下一状态
固定为：按 P4-B 运行 official reference，先产出资格化和逐 family 统计证据；若 readiness 不满足则停在
诊断，不得跳过 reference 直接发起 pilot，也不得为追求结果改 generation 数、suite、seed 或阈值。

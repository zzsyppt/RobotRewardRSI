# Eval 改造计划：从轻量 public eval 升级为全量分层验收

## 这次审计后的修正结论

这份计划已经按当前代码重新收敛。几个原计划里的关键表述需要改掉：

- `trusted_seed_count` 不能只存在于 `EvalJobSpec` 和 seal 里，必须进入 task-owned invariant 和 backend invariant 校验，否则它不是 runner-owned 固定协议。
- “trusted split”不能写成一个普通 prompt-facing 第三 split。更合理的定义是 sealed validation lane：它可以包含 runner-owned holdout rollout，但它首先是非 prompt 验证通道，负责 checkpoint 安全加载、候选代码隔离、协议 hash 和最终保守门控。
- sealed validation 不能只写进 summary。它必须进入 accept gate，并被 decision 真实消费；否则只是审计字段，不是评测能力。
- full eval 完整形态下，sealed validation 必须包含 runner-owned holdout rollout。checkpoint safety、候选代码隔离和 protocol hash 只能证明“评测没被污染”，不能证明“策略没有对 public-accept 偶然过拟合”。
- eval CLI 不能继续让 sealed validation 依赖 `--candidate` 路径。public eval 也应逐步去掉 candidate path，只保留 candidate id / artifact ref 作为记账信息。
- seal 不能只基于 `CheckpointRef` 的结构检查。真正的硬证据必须来自 eval 进程内的 byte-level safe-load 报告。
- current best 不能继续只是 scalar score。全量 eval 后的决策需要 current-best eval stats、protocol hash 和 gate 状态。
- prompt/public history 不能直接出现 `trusted_eval`、`trusted_seed` 这类当前 guardrail 禁词。对 LLM 公开时统一投影成 `sealed_validation_gate` / `non_prompt_validation` 这类安全命名。
- raw artifact 和 prompt-facing ledger 必须分层，而且边界不只包括 eval raw summary。`jobs.eval.checkpoint.metadata`、`train_result.checkpoint.metadata`、runner summary 里的 local path / host path / cwd / config / per-episode detail 都必须重新审计；prompt 只能看到 public projection。

下面的计划按这些修正后的边界展开。

## 背景判断

当前新系统的 eval 链路比旧 Phase8E 更干净：backend 物化 train/eval job，runner 执行，task plugin 投影 public result，shared decision 做接受/拒绝。这个方向应保留。

当前薄弱点不是“没有 eval”，而是验收层还不足：

- `public_accept_seed_count` 默认只有 1，无法稳定估计方差。
- `trusted_seed_count` 当前没有进入 task invariant，配置可能在 backend/CLI 层漂移。
- eval CLI 当前只跑 public-dev/public-accept，未执行 sealed validation lane。
- eval CLI 当前仍要求 `--candidate` 路径，和“验证路径不读取候选代码”的目标冲突。
- `TrustedEvalSeal.safe_load_status` 当前不是 byte-level checkpoint load 证据。
- `run_live_loop()` 和 `decide_candidate()` 仍以 current-best scalar 为主，无法严格比较 full eval stats。
- PushRecovery prompt adapter 当前仍把 eval 描述为 fixed public eval plus non-prompt checks；full eval 落地时必须同步更新 prompt-facing eval contract，否则模型看不到新的证据结构。

旧 Phase8E 的价值在于更保守的 trusted probe、安全加载、候选代码隔离、public-only 投影和预算/运动质量门控。旧系统的问题是 linked pipeline 耦合重，不应原样搬回。新系统应该吸收旧系统的验收强度，但按新系统的分层接口重做。

## 总目标

把每个 trainable round 的 eval 升级为完整、可复查、不可被候选控制的分层验收：

1. 每个 trainable round 都执行 full eval，不用 component smoke 作为是否进入 eval 的先验门。
2. full eval 包含两个 prompt-facing rollout split 和一个 non-prompt sealed validation lane：
   - `public_dev`
   - `public_accept`
   - `sealed_validation_gate`
3. `public_dev` 用于诊断和下一轮反馈，不单独决定 accept。
4. `public_accept` 用于公开的 candidate-vs-current-best 主比较。
5. `sealed_validation_gate` 用于非 prompt 的保守验收：checkpoint byte-level safe-load、候选代码隔离、协议 hash、runner-owned sealed holdout rollout。
6. accepted candidate 必须同时通过统计进步、行为安全、预算一致性、sealed validation 和 current-best 可比性检查。
7. 所有 raw eval artifact 可复查，但只有 public-safe projection 进入 ledger/prompt/history。

## 概念定义

### public-dev

public-dev 是公开诊断 split。它可以进入 prompt history 的压缩摘要，用于告诉模型最近失败主要来自 fall、drift、timeout、motion gate、budget gate 还是 eval completion。

public-dev 不允许单独触发 accept。

### public-accept

public-accept 是公开晋级比较 split。它的 success-rate stats、failure breakdown 和 motion gate 进入 public projection，用于和 current best 比较。

public-accept 必须有足够 count 来估计 standard error；count 不足时只能 reject 或标记 evidence insufficient。

### sealed-validation

sealed-validation 不是普通 prompt-facing split，也不应在 prompt payload 中暴露为 `trusted_eval` 或 `trusted_seed`。

sealed-validation 的职责是：

- 在不加载 candidate package 的进程/路径里验证 checkpoint。
- 记录 byte-level safe-load 结果，例如 `torch.load(..., weights_only=True)` 是否通过。
- 证明 eval config、checkpoint、task invariant、split counts 和 protocol version 未漂移。
- 执行 runner-owned sealed holdout rollout，其聚合结果只参与 gate，不把 raw detail 暴露给 prompt。

对 prompt 公开时，只允许出现安全投影，例如：

- `sealed_validation_gate.status`
- `sealed_validation_gate.reason`
- `sealed_validation_gate.checkpoint_safety_status`
- `sealed_validation_gate.candidate_isolation_status`
- `sealed_validation_gate.protocol_match_status`

不要在 prompt-facing payload 中出现 `trusted_eval`、`trusted_seed`、raw seed、hidden profile 或内部采样规则。

## 分层边界

### 公共层

公共层只定义通用 eval 协议和决策所需结构：

- `EvaluationProtocol`
- `EvalSplitStats`
- `GateResult`
- `SealedValidationSummary`
- `CurrentBestEvalStats`
- `EvaluationComparison`

公共层不理解 push recovery 的 fall、drift、base height、roll/pitch、push window 等任务语义。

### 任务层

push recovery task eval adapter 负责：

- 将 raw rollout summaries 投影成 public-dev/public-accept。
- 生成 failure breakdown、dominant failure、motion quality gate。
- 生成 prompt-safe diagnostic summary。
- 定义哪些行为指标是 required gate。

### backend / runner 层

backend / runner 负责：

- 固定 eval protocol。
- 物化 eval job。
- 派生 split-specific seeds。
- 执行 public rollout 和 sealed validation。
- 生成 raw artifact 与 public projection。

LLM/provider 不参与这些决定。

## Phase 0：协议冻结与 invariant 补齐

目标：先把 eval 协议变成不可漂移的 runner-owned/task-owned contract。

需要交付：

- 在 task config 中显式包含 sealed validation count，而不是只在 backend config 中保存。
- 将 `public_seed_count`、`public_accept_seed_count`、`sealed_validation_seed_count`、`eval_num_envs`、`eval_protocol_version` 纳入 `TaskInvariantSpec`。
- `TaskSpec.fixed_eval_protocol` 中必须包含同一组 eval protocol 字段。
- backend 的 invariant 校验必须覆盖 sealed validation count。
- `EvalJobSpec`、backend config、task invariant、materialize job 输出之间的字段必须一致。
- 字段命名一次性收敛：新协议统一使用 `sealed_validation_seed_count`。旧的 `trusted_seed_count` 只能作为短期兼容输入在边界层读取并立即转换为 canonical 字段；protocol hash、invariant 比较、prompt/public projection、ledger 统一只使用 canonical 字段。

严格验收：

- 任一 eval count 在 backend/CLI/task invariant 间不一致，materialize 阶段直接 fail。
- sealed validation count 缺失时，candidate 不允许进入 train/eval。
- eval protocol hash 必须由固定字段生成，不能由 candidate manifest 或 provider JSON 影响。
- 同一 artifact 中不得长期并存 `trusted_seed_count` 和 `sealed_validation_seed_count` 两套语义；如果为了迁移短期并存，必须由 audit 明确标记 legacy alias，并禁止它进入 protocol hash。

## Phase 1：重接 eval CLI 与 sealed validation contract

目标：eval CLI 不再依赖 candidate path 来完成验证，sealed validation path 明确拒绝候选代码输入。

需要交付：

- eval CLI 移除或降级 `--candidate` 路径参数；如果需要记录候选身份，只传 candidate id，不传 candidate package path。
- sealed validation 子路径必须不接受 candidate path；如果收到 candidate path，应 fail-closed。
- backend eval argv 不再把 candidate package path传给 sealed validation。
- eval summary 中显式记录：
  - `candidate_package_loaded_in_eval = false`
  - `candidate_code_loaded_in_eval = false`
  - `sealed_validation_gate.candidate_isolation_status`

严格验收：

- sealed validation 进程中检测到候选模块加载，直接 reject。
- sealed validation 参数中出现 candidate path，直接 reject。
- eval CLI 不能通过 candidate manifest、operation_diff、diagnostics 或 components.py 改变 eval 配置。

## Phase 2：byte-level checkpoint safety 进入 seal

目标：seal 代表真实 eval 进程里的 checkpoint 安全证据，而不是只代表 `CheckpointRef` 结构看起来合法。

需要交付：

- eval 进程生成 `CheckpointSafetyReport`。
- report 至少包含：
  - checkpoint hash。
  - file exists。
  - byte-level safe-load status。
  - weights-only load status。
  - loader error class 的 public-safe 摘要。
- `TrustedEvalSeal` 或后续命名的 `SealedEvalSeal` 必须从 eval 进程 safety report 构建。
- seal hash 覆盖 eval protocol、task invariant、checkpoint hash、split counts、safe-load status、candidate isolation status。

严格验收：

- byte-level safe-load 未通过，直接 reject。
- seal 缺失 checkpoint safety report，直接 reject。
- seal 只来自 Python 侧 `CheckpointRef` 结构检查时，不允许作为 accept 证据。

## Phase 3：full eval 执行与 split 投影

目标：每个 trainable round 都跑完整 eval，并将 raw artifact 与 public projection 分开。

需要交付：

- 每个 trainable round 至少产生：
  - raw public-dev rollout summaries。
  - raw public-accept rollout summaries。
  - raw sealed validation summary。
  - public projected eval summary。
  - sealed validation public gate。
  - decision comparison summary。
- raw artifact 只作为 private artifact ref 进入 ledger。
- ledger/prompt 只保存 public projection。
- public projection 必须通过 public safety assert。
- 重新审计现有 ledger 写入链路，不只保护 eval raw summary：
  - `jobs.eval.checkpoint.metadata` 不得把 local checkpoint path 带入 prompt-facing ledger。
  - `train_result.checkpoint.metadata` 不得把 runner local path 带入 prompt-facing ledger。
  - train/eval summary 中的 host、cwd、config path、per-episode detail 只能出现在 private artifact。
  - `CheckpointRef.to_public_dict()` 的 prompt-facing使用必须有 scrubbed projection，不能默认原样暴露 metadata。

严格验收：

- train 成功但 eval artifact 缺失，round 记为 eval incomplete，不允许 accept。
- public projection 中出现 raw seed、真实路径、hidden profile、内部 checkpoint path，直接 fail-closed。
- raw eval summary 可以保留 host/cwd/config/per-episode 供审计，但不得直接进入 prompt-facing history。
- accepted ledger record 中如果包含 unsanitized checkpoint metadata、host path 或 cwd，audit 必须失败。

## Phase 4：public-accept 统计加厚

目标：降低单 seed 偶然性，避免小样本噪声导致错误 accept。

需要交付：

- `public_accept_seed_count` 提升到能估计方差的数量。
- public-accept stats 至少包含 count、episode_count、mean、std、standard_error、safe values projection。
- 当 count 不足或 standard error 不可信时，输出 `insufficient_accept_evidence` gate。
- candidate 和 current best 必须使用同一 protocol hash 下的 stats 比较。

严格验收：

- public-accept count 不足，不能 accept。
- candidate mean 必须超过 current-best mean，并超过 `max(min_delta, pooled_standard_error_margin)`。
- public-dev 好但 public-accept 不足，只能作为 feedback，不允许 accept。
- public-accept variance 过大时，不能因为 mean 偶然过线而 accept。
- sealed holdout rollout 是最终 accept 的 required evidence；public-accept 过线但 sealed holdout 未执行或未通过时，不能 accept。

## Phase 5：current-best 从 scalar 升级为 eval stats

目标：accept 决策比较 full eval stats，而不是只比较 scalar score。

需要交付：

- 定义 `CurrentBestEvalStats`，至少包含：
  - public_accept stats。
  - sealed validation gate status。
  - motion / behavior gate status。
  - eval protocol hash。
  - accepted round id。
  - accepted candidate id。
- `RoundInputs` 不再只传 `current_best_score: float`，而应传 current-best eval stats。
- `live_loop` 从 ledger/baseline record 中恢复 current-best eval stats。
- `decide_candidate()` 使用 candidate stats 和 current-best stats 生成 comparison detail。
- `DecisionRecord` 或 decision detail 必须保存 comparison summary，供 audit 和 prompt-safe history 使用。
- `PublicEvalResult.metrics.acceptance.required_pass_gates` 必须包含 `sealed_validation_gate`，并且 `decide_candidate()` 必须真实检查该 gate。只把 sealed validation 写到 summary 或 ledger，不算完成。
- decision detail 必须记录 consumed gate names、gate statuses 和 comparison stats，供 run audit 判断“decision 实际消费了 gate”。

严格验收：

- protocol hash 不一致时，不允许直接比较；必须重评 best 或进入 bootstrap 逻辑。
- current-best stats 缺失时，只有 bootstrap mode 可用，且 bootstrap accept 仍必须通过 full eval、absolute floor、sealed validation、motion/budget gates。
- `best_score_stats` 不能只是空 hook，必须由 current-best eval stats 填充。
- 如果 `sealed_validation_gate` 缺失、未通过或未被 decision 消费，candidate 不能 accept。

## Phase 6：behavior gates 升级为 required gates

目标：避免只优化 success rate，导致动作爆炸、姿态恶化、漂移过大或恢复质量下降。

需要交付：

- push recovery task eval adapter 输出行为 gate：
  - fall / timeout / drift / unrecovered velocity breakdown。
  - action magnitude。
  - action rate。
  - min base height。
  - max roll / pitch。
  - final xy drift。
  - recovery window success 或 recovery latency。
- 公共 decision 只消费 gate status，不解释任务字段。
- prompt history 只保留 dominant failure 和 failed gates 的压缩摘要。

严格验收：

- motion gate fail，直接 reject。
- budget gate fail，直接 reject。
- eval metric missing / nonfinite，直接 reject。
- failure breakdown 无法解析，不能 accept。

## Phase 7：prompt/public 投影规则

目标：让 LLM 得到足够反馈，但不泄露 eval 私有信息，也不触发 prompt safety 冲突。

需要交付：

- prompt-facing eval feedback 使用安全命名：
  - `sealed_validation_gate`
  - `non_prompt_validation`
  - `checkpoint_safety_status`
  - `candidate_isolation_status`
- PushRecoveryPromptAdapter 的 eval contract 必须从 `fixed_public_eval_plus_non_prompt_checks` 升级为描述 full eval 证据结构：
  - public-dev 是 prompt-facing diagnostic split。
  - public-accept 是 prompt-facing comparison split。
  - sealed validation 是 non-prompt gate，只公开 gate 摘要。
  - LLM 不能请求或修改 eval seed、holdout、profile、checkpoint policy 或 threshold。
- 禁止在 prompt-facing payload 中使用：
  - `trusted_eval`
  - `trusted_seed`
  - `hidden_eval`
  - `raw_seed`
  - raw checkpoint path
  - host-local artifact path
- prompt history 保留：
  - round id。
  - decision outcome。
  - public score。
  - comparison summary。
  - failed gates。
  - dominant failure。
  - sealed validation gate summary。
  - changed components。
- 多轮历史做聚合趋势，不追加完整 per-seed 或 per-episode detail。

严格验收：

- prompt payload 必须通过 public safety assert。
- prompt payload 必须解释 accept/reject 原因。
- history 压缩不能丢失 current best、最近失败主因、gate regression、sealed validation failure。
- prompt-facing contract 不能继续暗示只有 public eval 加少量 non-prompt checks；必须让模型知道 sealed validation gate 是 accept 的硬条件，但不能暴露其 raw rollout 或 seed 细节。

## Phase 8：run-level audit 与半升级防护

目标：防止出现“字段加了但没进决策”“sealed validation 跑了但拒绝原因没进 public history”这种半升级状态。

需要交付：

- run audit 检查每个 trainable round：
  - train 是否执行。
  - full eval 是否执行。
  - public-dev 是否完整。
  - public-accept 是否完整。
  - sealed validation 是否完整。
  - public projection 是否通过 safety assert。
  - decision 是否消费了 required gates。
- run audit 必须检查 accepted record 中 `decision.consumed_gates` 或等价 detail，确认 `sealed_validation_gate`、budget gate、motion/behavior gate 和 comparison threshold 都被 decision 使用，而不是只存在于 artifact。
- live summary 区分：
  - authored candidate count。
  - trainable candidate count。
  - trained candidate count。
  - full-eval completed count。
  - sealed-validation completed count。
  - accepted count。
  - eval-incomplete count。
- audit 结果能横向总结 15 轮中失败集中在 authoring、train、public eval、sealed validation、motion gate、budget gate 还是 decision threshold。

严格验收：

- 任一 trainable round 没有 full eval artifact，summary 必须明确归因。
- decision reason 与 eval gates 不一致时，audit fail。
- accepted round 缺少 sealed validation 证据，audit fail。
- accepted round 的 sealed validation 证据存在但没有进入 required_pass_gates 或 consumed gates，audit fail。

## 推荐交付拆分

### 交付 1：EvaluationProtocol 与 invariant 收敛

目标：先堵住配置漂移。

主要触点：

- task config。
- task plugin 的 `TaskSpec` / `fixed_eval_protocol` / `TaskInvariantSpec`。
- backend invariant 校验。
- materialize job 输出。
- unit tests。

验收：

- 修改 backend eval count 会被 invariant 测试拦住。
- `sealed_validation_seed_count` 缺失会失败。
- protocol hash 对 split counts 和 eval env count 敏感。
- `trusted_seed_count` legacy alias 只允许在迁移边界出现，并被转换为 `sealed_validation_seed_count`；核心协议、hash、prompt projection 不出现 legacy 名称。

### 交付 2：eval CLI 去 candidate path + sealed validation lane

目标：重建 eval 执行接口。

主要触点：

- eval script parser。
- backend eval argv。
- subprocess runner summary validation。
- sealed validation summary schema。
- negative tests：传 candidate path 给 sealed validation 必须失败。

验收：

- eval 可以仅凭 checkpoint、task config、protocol counts 执行。
- sealed validation 不读取 candidate package。
- candidate isolation status 被写入 public-safe gate。

### 交付 3：checkpoint safety report 与 seal

目标：让 seal 代表真实 checkpoint bytes 的安全证据。

主要触点：

- eval script safe-load report。
- safety seal builder。
- runner summary parsing。
- tests for unsafe checkpoint / missing report / seal hash drift。

验收：

- byte-level safe-load 失败时不能 accept。
- seal 缺失或 hash 不一致时不能 accept。

### 交付 4：CurrentBestEvalStats 与 decision 迁移

目标：从 scalar compare 迁移到 stats compare。

主要触点：

- live loop current-best state。
- RoundInputs。
- decision API。
- ledger accepted record。
- public history projection。
- dry-run / fake runner tests。

验收：

- `best_score_stats` 由 current-best stats 填充。
- protocol mismatch 时不能 accept。
- scalar-only current best 不再作为正常路径。
- `sealed_validation_gate` 是 required pass gate，并被 decision detail 标记为已消费。

### 交付 5：behavior gates 和 prompt-safe feedback

目标：让 eval 对下一轮有足够诊断，但不泄露，并让 prompt adapter 同步表达 full eval 的新证据结构。

主要触点：

- push recovery metrics projection。
- prompt adapter eval feedback。
- PushRecoveryPromptAdapter 的 public eval contract。
- public safety tests。
- history compression tests。

验收：

- motion/budget/failure breakdown gates 被 decision 消费。
- prompt payload 不含 forbidden key/value。
- prompt history 能解释拒绝原因。
- prompt 中能看到 sealed validation gate 的安全摘要，但看不到 raw holdout seed、raw rollout、hidden profile 或真实路径。

### 交付 6：run audit

目标：确保 15 轮 run 的 eval 链路完整可查。

主要触点：

- live summary。
- audit script。
- integration tests with fake runner。

验收：

- trainable round 全部有 full eval accounting。
- authoring failure 不被误算为 eval 缺失。
- accepted candidate 必须有 public-accept stats、sealed validation gate 和 comparison detail。
- audit 能证明 decision 消费了 sealed validation gate，而不是只把它写入 summary。

## 最终 accept 条件

一个 candidate 只有同时满足以下条件才允许 accept：

1. verifier 通过，且不是 parameter-only / diagnostic-only。
2. train job 完成，训练预算和 task invariants 与 runner-owned config 一致。
3. full eval 完成：public-dev、public-accept、sealed validation 全部有记录。
4. public projection 不泄露 raw seed、hidden profile、真实路径或可用于 eval 过拟合的细节。
5. public-accept 统计足量。
6. candidate 相对 current best 满足 noise-aware improvement。
7. current-best protocol hash 与 candidate protocol hash 可比较。
8. sealed validation 通过：checkpoint safe-load、candidate isolation、protocol seal 均通过。
9. sealed holdout rollout 通过，且其聚合结果进入 sealed validation gate。
10. budget gate 通过。
11. behavior / motion gates 通过。
12. decision detail 能说明 accept/reject 原因，并与 eval artifact 一致。
13. decision detail 证明 sealed validation gate、budget gate、behavior gate 和 comparison threshold 已被实际消费。
14. prompt-facing feedback 足够指导下一轮，但只使用 public-safe projection。

## 不做的事情

- 不把旧 Phase8E linked pipeline 原样搬回。
- 不把 sealed validation 当成普通 prompt-facing 第三 split。
- 不让 LLM 选择 eval seed、eval profile、eval budget、checkpoint selector 或 accept 阈值。
- 不保留 `trusted_seed_count` 和 `sealed_validation_seed_count` 两套长期并行协议。
- 不把 push recovery 特殊指标写入公共 eval 层。
- 不把 raw per-env trajectory、raw seed、hidden profile 或真实路径放进 prompt。
- 不用 smoke 代替 full eval。
- 不因为 sealed validation 暂时未接完就降级为 public-only accept。

## 推荐执行顺序

1. 先补 EvaluationProtocol、fixed_eval_protocol、task invariant 和 backend invariant。
2. 再重接 eval CLI，去掉 sealed validation 对 candidate path 的依赖。
3. 接入 byte-level checkpoint safety report 和 seal。
4. 扩展 eval summary/public projection，区分 raw artifact 与 prompt-safe projection。
5. 把 current-best 从 scalar 升级为 eval stats。
6. 加厚 public-accept 统计和 behavior required gates。
7. 接 prompt-safe feedback 和 history compression。
8. 最后加 run-level audit，防止半升级状态。

这个顺序先解决协议漂移和安全边界，再解决评测强度，最后解决反馈质量。它保留新系统的分层架构，同时吸收旧 Phase8E eval 的严格验收思想。

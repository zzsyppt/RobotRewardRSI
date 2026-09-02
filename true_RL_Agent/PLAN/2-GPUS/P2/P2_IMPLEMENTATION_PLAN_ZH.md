# P2 实际实施计划：无副作用 Trial Executor 与 width=1 等价

> 日期：2026-08-16  
> 基线：`81565dd970a6726fda6a219704bb9a18651e6bb8` 加当前未提交 P0/P1 工作树  
> 本阶段不运行 GPU/API/服务器实验，不实施双 lane、selector、分阶段 eval 或 generation ledger。

## 1. 本阶段目标

P2 只解决一个底层结构问题：当前 `run_candidate_round()` 同时拥有候选封存、验证、job 构造、训练、评测、决策、record 构造和 ledger append。若未来直接并行调用两次，两个 sibling 会分别写 ledger，并让 acceptance replay 顺序推进 head，破坏共同 parent 语义。

本阶段把它收敛为：

1. `execute_candidate_trial()` 完成当前单候选的全部科学计算，但不得 append ledger 或推进 head；
2. train、eval 和最终 trial 结果使用 create-once、hash-bound 的私有 operational receipt；
3. `run_candidate_round()` 只作为 width=1 兼容 wrapper，调用 executor 后恰好提交一次现有 `round-v7` record；
4. 当前 record 的 public 内容、decision、gates、parent、authoring anchor 和 replay head 与重构前保持等价；
5. verifier 现有单一 AST 分析器额外投影 operation-aware delta v2，P2 receipt 封存该事实，但不修改 authoring evidence v4、round-v7 或历史 ledger。

这里的“无副作用”专指**无科学 ledger/head 副作用**。候选封存、runner job、日志、checkpoint 和 operational receipt 本来就是 trial 执行必需产物，不伪装成纯函数。

## 2. 权威边界

### 2.1 继续保持唯一真相的对象

- 科学结果：append-only ledger；
- 接受结果：现有 decision/acceptance replay；
- 候选机制识别：`candidates/component_source_semantics.py` 的 active-builder/AST 分析链；
- authoring 历史：现有 authoring evidence v4 与 `RoundAuthoringJournal`；
- train/eval 进程重试：P1 的 `JobAttemptJournal`；
- 任务语义：`TaskPlugin` 及各 task-owned adapter。

Trial receipt 是 ledger 提交前的私有 operational 证据。它只能保存“将要原样提交的 record”和 runner phase 结果，不能自行接受候选、改变 current-best，或形成第二条科学历史。

### 2.2 明确不做

- 不实现 width=2、GenerationCoordinator、generation schema 或 sibling barrier；
- 不拆 public-dev/public-accept/sealed；
- 不引入 Graph、Ray、Temporal 或通用 workflow engine；
- 不改变 eval-v6、Stair success、acceptance、sealed 或 final-confirmation；
- 不把 Stair 字段、selector、frontier 或 prompt memory 导入 executor；
- 不修改 authoring evidence v4 和 round-v7 的现有字节；
- 不承诺在旧 width=1 正式 Stair session 中跨 invocation 恢复；其 single-invocation 合同保持不变。

## 3. 实际结构

```text
RoundInputs
   │
   ▼
prepare candidate / authoring binding / jobs
   │  生成稳定 trial contract hash
   ▼
load reusable phase receipts ──有──► validate and reuse
   │ 无
   ├─ run_train ─► create-once train receipt
   └─ run_eval  ─► create-once eval receipt
   │
   ▼
project public result / decide / build exact round-v7 record
   │
   ▼
validate decision + prompt artifacts + public safety
   │
   ▼
create-once final TrialExecutionReceipt
   │
   ▼
width=1 wrapper single idempotent ledger commit
```

借鉴 Codex 的部分仅限：执行状态使用稳定 ID，运行结果与上层状态提交分离，commit 由唯一 owner 完成。不会复制 Codex 的通用 agent、tool registry、动态图或会话协议。

## 4. 分步工作

### 4.1 Authoring typed-size 合同

建立一处 task-neutral artifact size registry：

- budget artifact：固定小 cap；
- attempt result：由最大合法 public attempt、continuation/public-context、source-request payload 上限和固定 envelope 推导；
- authoring evidence：由 `max_provider_attempts × 单 attempt 上限 + 固定 envelope` 推导，并保持当前正式上限 5 attempts；
- producer 写前、Ref、bounded reader、resume loader 和 verifier 使用同一函数/常量；
- cap+1 直接在解析前拒绝，不用静默截断。

不把真实 15-round 最大文件大小直接写成合同；真实值只用于确认公式有合理余量。

### 4.2 Verifier-owned mechanism delta v2

复用 `_changed_component_mechanisms()`、active component records 和相同 signature：

- operation 仅允许 `add/remove/modify/rename`；
- 保存 before/after canonical identity、结构 hash、parameter-normalized 结构 hash、输入依赖和 candidate/anchor source checksum；
- 名字、operation 和 hash 全部从 verifier-owned source projection 生成；
- history、trial executor 和后续 graph/frontier 不得二次猜 operation；
- compound/unavailable 返回明确不可投影，不降级成虚构 delta。

P2 只把 v2 放进 TrialExecutionReceipt。round-v7 继续投影既有 candidate mechanism delta v1，以保证 legacy 字节不变。

### 4.3 Trial receipt 与恢复

一个 trial journal 使用稳定 contract hash 定位，包含：

- train phase receipt：绑定 trial contract、train spec hash、完整 runner-owned TrainJobResult；
- eval phase receipt：绑定 trial contract、checkpoint-bound eval spec hash、原始 PublicEvalResult；
- final receipt：绑定上述 phase hash、exact round-v7 record hash、decision 和 operation delta v2。

所有 receipt 均 canonical JSON、create-once、bounded read、严格字段集。若文件已存在，字节必须完全一致。加载时必须重算 trial contract、job spec、checkpoint/source/training/eval binding；半文件、缺前驱、hash 漂移或跨 session/ledger epoch/round 搬运均拒绝。

训练完成而 eval 前中断时允许复用已验证 train receipt；eval 完成而 ledger append 前中断时允许复用 final receipt。P1 job-attempt receipt 继续负责一次同参数 infra retry，P2 不再创建第二套 retry 分类。

### 4.4 Width=1 单提交 wrapper

- `execute_candidate_trial()` 不导入或调用 ledger append；
- `commit_candidate_trial()` 只能提交 executor 已验证的 exact record；
- ledger 锁内按 `{schema, session_genesis_hash, ledger_epoch_hash, round_id}` 做一次性 commit：完全相同视为已提交，不同内容视为冲突；
- `run_candidate_round()` 只组合这两步并返回既有 `RoundResult`；
- live loop、fake loop、Push 保持调用 `run_candidate_round()`，不产生第二条执行路径。

## 5. 严格验收

### 5.1 等价性

- 固定 fake candidate 下，直接 executor+commit 与 width=1 wrapper 的 canonical public record 完全相同；
- decision、gate statuses、parent、authoring anchor、current-best replay 逐字段一致；
- verifier fail、task invariant fail、candidate runtime fail、eval incomplete 和 accepted 路径全部覆盖；
- 每轮最多一个 ledger row；append 后重入不新增 row，冲突 record 拒绝。

### 5.2 Receipt

- train 后中断：恢复不再训练，只补 eval/record/commit；
- eval 后中断：恢复不再 train/eval，只 commit；
- tampered contract/spec/checkpoint/result/hash、缺失前驱和跨 session/epoch/round 复用 fail closed；
- partial receipt 不冒充 completed；
- candidate/scientific failure 不被 P2 当成 infra retry。

### 5.3 Mechanism delta

- add/remove/modify/rename 正确；
- name-only no-op 不形成新机制；
- 数值参数变化与 tensor index/axis 结构差异保持 P0 已冻结语义；
- compound/unavailable 明确拒绝投影；
- delta v2 tamper 不能通过 receipt loader；
- round-v7 与 authoring evidence v4 字节不增加 v2 字段。

### 5.4 兼容与工程质量

- 当前 v7 及历史 v2/v4/v5/v6 replay 不变，v3 仍拒绝；
- Push/Neutral/Stair width=1 全部通过；
- executor 公用层不得新增 Stair/Push 指标语义；
- `git diff --check`、compileall、Python 3.10 grammar、Windows 全量与相关 WSL 定向测试通过；
- 不增加 production dependency，不运行 GPU/API/服务器实验。

## 6. 退出条件

P2 只有在上述验收全部闭合、阶段报告和开发日志写入本目录后才完成。若无法证明 width=1 等价，立即保留现有 wrapper 行为并停止，不进入 P3/P4。

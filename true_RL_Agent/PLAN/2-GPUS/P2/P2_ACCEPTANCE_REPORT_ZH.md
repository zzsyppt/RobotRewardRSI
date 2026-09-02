# P2 验收报告：无副作用 Trial Executor 与 width=1 等价

> 验收日期：2026-08-16  
> Git 基点：`81565dd970a6726fda6a219704bb9a18651e6bb8`  
> 工作树口径：上述基点加同一工作树中已验收、尚未提交的 P0/P1 修改，再叠加本次 P2。本文不把这组未提交修改错误归属为 Git 基点内容。  
> 运行限制：未调用 provider API，未运行 GPU 训练/评测，未访问或修改服务器，未 commit/push。

## 1. 验收结论

P2 通过本地工程验收，可以作为 P3/P4 的底座，但**尚未授权进入 P3、P4 或正式实验**。

本阶段完成了四项收敛：

1. 把单候选科学执行与 ledger/head mutation 分开；
2. 用私有、create-once、hash-bound receipt 保存 train、eval 和最终 trial 执行结果；
3. 让旧 `run_candidate_round()` 成为 `execute_candidate_trial() + commit_candidate_trial()` 的 width=1 兼容包装；
4. 从既有 verifier AST 分析链投影 operation-aware mechanism delta v2，未修改 authoring evidence v4 或 round-v7。

这里的“无副作用”只表示 executor 不写科学 ledger、不推进 current-best。候选封存、runner 产物、checkpoint 和 operational receipt 仍是 trial 执行本身的必要副作用。

## 2. 实际落地结构

```text
RoundInputs
   │
   ▼
execute_candidate_trial()
   ├─ seal / verify candidate
   ├─ build trusted train/eval specs
   ├─ load or create train receipt
   ├─ load or create eval receipt
   ├─ project result / decide / build exact round-v7 record
   └─ create final TrialExecutionReceipt
   │
   │  无 ledger/head 写入
   ▼
commit_candidate_trial()
   ├─ reload and validate final receipt
   ├─ bind current session / epoch / round / parent / contracts
   └─ exactly-once append one round-v7 row
```

旧 live loop、fake loop 和 Push 路径仍只调用 `run_candidate_round()`；没有生成第二套 width=1 执行逻辑。

## 3. 修改范围

### 3.1 Trial 执行与唯一提交者

- `agent_core/round.py`
  - 新增 `TrialExecution`；
  - 新增无 ledger/head 写入的 `execute_candidate_trial()`；
  - 新增唯一提交入口 `commit_candidate_trial()`；
  - `run_candidate_round()` 收敛为 width=1 兼容 wrapper；
  - 提交前重新绑定 session、ledger epoch、round、parent、evaluation binding、task、backend、prompt/public context 和 decision contract。
- `agent_core/ledger.py`
  - 新增锁内 `append_ledger_record_once()`；
  - 完全相同的当前 writer record 重入为 no-op；
  - 同一当前身份但内容不同会拒绝；
  - 历史 v2/v4/v5/v6 record 不会被误当作当前 round-v7 identity 冲突。

### 3.2 Operational receipt

- 新增 `agent_core/trial_receipts.py`；
- contract、train、eval、final receipt 均为 canonical JSON、create-once、bounded read；
- receipt contract 绑定 session、epoch、round、parent、candidate、验证结果、训练/评测 spec、authoring artifact ref、current-best、evaluation/acceptance/backend contract；
- train receipt 保存 runner-owned `TrainJobResult` 和 checkpoint 身份；
- eval receipt 绑定同一 checkpoint 的 eval spec 与原始 `PublicEvalResult`；
- final receipt 绑定 phase receipt hash、exact round-v7 record hash、round stage、decision 和 mechanism operation delta v2；
- loader 重算前驱、hash、spec/result 关系，receipt 不能自行接受候选或推进 head。

P1 的 `JobAttemptJournal` 继续是 logical-job infra retry 的唯一 owner。P2 receipt 只负责复用已经完整闭合的 phase，不新增 retry 分类或循环。

### 3.3 Authoring artifact 大小单源

- 新增 `authoring_artifact_contract.py`；
- budget、attempt result、authoring evidence 使用各自独立的公式化 cap；
- producer 写前、Ref、bounded reader、resume/verifier 共享同一合同；
- `max_attempts` 与正式上限 5 由同一 task-neutral validator 约束；
- cap+1 在解析前拒绝，不截断、不从真实单次文件大小反推合同。

### 3.4 Mechanism operation delta v2

- 在 `candidates/authoring_policy.py` 定义严格 schema；
- 在 `candidates/component_source_semantics.py` 复用现有 active-builder/AST/signature 分析器投影 `add/remove/modify/rename`；
- before/after identity、source checksum、结构 hash、parameter-normalized hash 和输入依赖由 verifier-owned source projection 产生；
- name-only no-op 与 compound/unavailable 情况不伪造 delta；
- v2 只进入私有 TrialExecutionReceipt，未增加 round-v7 或 authoring evidence v4 的字段。

## 4. 严格验收结果

### 4.1 Executor / commit

| 场景 | 结果 |
|---|---|
| 执行完成但未 commit | ledger 不存在 |
| 首次 commit | 精确写入 executor 生成的唯一 round-v7 record |
| 相同 execution 再次 commit | ledger 仍只有一行 |
| train 后 eval 中断 | 恢复时 train 调用 0 次，只补 eval |
| eval 完成、final receipt 发布前中断 | 恢复时 train/eval 调用均为 0 |
| final receipt 已存在 | 重执行时 train/eval 调用均为 0 |
| final receipt 字节篡改 | commit fail closed，ledger 不创建 |
| receipt 搬到另一 session / epoch / round | commit fail closed，ledger 不创建 |

### 4.2 兼容性

- fake width=1 round、live loop、Push/Neutral/Stair 相关测试均通过；
- 当前 round-v7 writer 与历史 v2/v4/v5/v6 replay 保持兼容，v3 仍按既有合同拒绝；
- verifier fail、task invariant fail、candidate runtime fail、eval incomplete 与正常完整评测仍走原有状态/decision 语义；
- 新公用模块未引入 Stair/Push 指标、selector、frontier、public-dev 或 sealed 任务语义；
- 未增加 production dependency。

### 4.3 自动测试

P2 开始前，同一 P0/P1 工作树的全量基线：

```text
Ran 1183 tests
OK (skipped=21)
```

P2 最终 Windows 全量回归：

```text
Ran 1197 tests
OK (skipped=21)
```

相对 P1 基线新增 14 项测试：Trial Executor 8 项、typed artifact cap 1 项、mechanism operation delta 1 项、prompt/section 合同 3 项、ledger 冲突诊断 1 项。

Windows P2/邻接聚焦回归：

```text
Ran 101 tests
OK
```

prompt、memory、live loop、Push、ledger/replay 邻接回归：

```text
Ran 221 tests
OK
```

WSL 非 GPU 定向回归：

```text
Ran 101 tests
OK
```

另外完成：

- P2 及本次复审涉及的生产文件 Python 3.10 AST grammar：10/10 通过；
- `compileall`：通过；
- `git diff --check`：通过。

## 5. 验收中发现并源头修正的问题

实现 exactly-once append 时发现：若对混合历史 ledger 的所有记录直接套用当前 writer identity，旧 schema 记录可能被错误视为 round-v7 冲突。最终实现只对当前 writer schema 计算 exactly-once identity，历史 schema 继续交给既有 replay/compatibility validator。这是 writer 能力边界修正，没有修改历史记录。

新增跨身份测试首次运行时，测试使用的“篡改哈希”与 fake session 固定哈希相同，导致测试实际没有制造差异。修正 fixture 值后，session、epoch、round 三种搬运均按预期拒绝；产品逻辑无需放宽。

用户复审后又完成一次针对性纠偏：

- Stair 压缩入口将缺失、`None`、错误类型或非 mapping 的 `feedback_entries` 统一归一为空列表；minimum core 不再因可选列表缺失产生 `TypeError`；同一路径的 `detail_omissions` 采用相同规则；
- provider 出厂 section 有序清单成为总 task section 集合的来源，`task_capability_metadata` 是唯一显式 task-only 例外，不再维护两份可漂移的完整清单；
- Trial receipt 在发布和加载时验证 contract 与 record 的全部直接共享字段，commit 又显式验证 contract 的 session、ledger epoch、round 与当前 inputs；contract 不再只是自洽但未被决定步骤消费；
- exactly-once ledger identity 继续有意排除 candidate/parent，但冲突现在报告首个顶层差异字段和双方 canonical record hash，便于区分 continuation 漂移与 artifact 篡改；
- `DEFAULT_MAX_PROVIDER_ATTEMPTS=5` 与 artifact 上限 5 已由共享 validator 强制耦合，不另做无收益重构；四类 Trial receipt 与四个独立可续跑节点一一对应，保持现状。

## 6. 明确未做与后续入口

P2 没有实现：

- width=2 或 GenerationCoordinator；
- sibling authoring、barrier、selector 或 generation ledger；
- public-dev/public-accept/sealed 分阶段评测；
- Graph、frontier 或探索价值；
- 旧正式 Stair single-invocation 合同的跨进程放宽；
- 真机双 GPU、provider API 或服务器验证。

P3/P4 只能消费本阶段的 typed executor/receipt 接口，不得复制 `round.py` 的执行逻辑，也不得让 trial worker 直接写 ledger。进入下一阶段前仍需用户检查并明确授权。

## 7. 关键生产文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `agent_core/round.py` | `aa9197a4975fb790af2d0cf2963553ac5c07cd3af46378902ea56809d23cb6e0` |
| `agent_core/trial_receipts.py` | `569d5f880b009be4f6b0f1b0224f02797e6c4891139201fe701a9fa56e1e55b3` |
| `agent_core/ledger.py` | `be963ddf9d7ef40a76d5b475364fceceb310c6b57d0c2af25079633af73862ef` |
| `authoring_artifact_contract.py` | `ac9532320bd4593cb1cdcf8afe615a1e06cb66624ee9b05d3da6f915c9501d3b` |
| `prompt_artifacts.py` | `96ebb0cf8188108a528dda2e1548a36fb3f901b0c29d09b89d819cbc0aee3302` |
| `candidates/authoring_policy.py` | `91a5457325f909cd340feb84847498943070841983dfd4f123962eddcdaabd45` |
| `candidates/component_source_semantics.py` | `6da78e8772dc724b2c8dc2a11ae70c5eb554f447bfeead7c544a78638a866b9b` |
| `tasks/stair_climbing/prompt_adapter.py` | `ba649838803cf0a2d50b35e1ad8ad0e7b7868580068e616d74c172a7a4eccf46` |
| `prompting/task_sections.py` | `ec461bd51ba11b12683d95158622230fe51494f8ee720178475630ac5d82c140` |

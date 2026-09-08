# Phase 1 独立审计（第三方复核）

> - 审计日期：2026-08-26
> - 审计对象：`PLAN/Variant_construction/P1/` 全部交付物 + Phase 1 全部产品代码改动
> - 基线：`4392aa2`（Phase 0 冻结提交）；跟踪树有 15 个修改文件 + 12 个新文件
> - 方式：逐层读源码、实际导出两份 resolved manifest、独立设计并运行 3 组新测试（含用 Phase 0 回执真实 Go2 事实模拟 Phase 5 接入）、跑回归
> - 本文不修改任何代码

---

## 0. 结论

**机器人语义抽离本身是正确的，而且抽得比我预期的干净。** 分层、provide/require 词汇表、
与 Phase 0 回执的逐字段绑定、legacy 字节兼容、子进程重载——都是从源头做的，不是补丁。
复杂度也没有失控：`robots/` 只有 566 行，没有新服务、数据库或调度器。

但有 **两个已实测的前向阻塞**，都会在下一两个 Phase 撞上，而且都属于"现在看起来是好卫生、
实际上把物理事实写错了"的类型：

| 编号 | 问题 | 严重度 | 何时爆炸 | 是否实测 |
|---|---|---|---|---|
| **B1** | `ambiguous_role_target` 规则会挡住 **Stair 自己的 Phase 2 角色集** | **高** | Phase 2 | ✅ 已实测 |
| **B2** | Push 声明了它**根本不消费**的 `support_effector(2,2)` | **中高** | Extension A (Go2×Push) | ✅ 已实测 |
| B3 | Stair 的 `(2,2)` 是第四处"2"，债务记录未点名 requirement 声明 | 中 | Phase 2/5 | — |
| B4 | EvaluationBinding 未获得 TrainingBinding 那样的 legacy/VNext 分流 | 中 | Phase 3+ | — |
| B5 | `legacy_training_binding_v1` 无结构性约束，未来 cell 可自行退回 V1 | 低-中 | 任意新 cell | — |
| B6 | G1-29 的 `articulation_source` 指向**任务环境模块**而非关节体资产 | 低 | Phase 5/Extension | — |
| B7 | Push 占位标签仍在执行路径上（已记录，但未指派责任 Phase） | 低 | Extension A | — |

---

## 0.1 第二轮复核（2026-08-26 修复后）

**七条全部从源头解决，无新 BUG，无二源，复杂度仍受控。** 逐条实测：

| 编号 | 修法 | 我的验证 |
|---|---|---|
| **B1** | 引入 `_MUTUALLY_EXCLUSIVE_ROLE_PAIRS`，只有 `support_effector × forbidden_contact` 结构性互斥；全局唯一规则删除 | ✅ 实测：`base+forbidden` 同为 `torso_link` → **supported**；`support+forbidden` 同为双踝 → 拒绝且原因指名两角色（`conflicting_role_target:forbidden_contact:support_effector:<target>`） |
| **B2** | Push requirements 与 binding 双双只保留 `base`，同时移除 `support_contact` 能力 | ✅ 实测：真实 Go2（4 足）接 Push → **supported**，Extension A 解除阻塞。且 binding 同步清理，未留下 `unrequested_binding_role` |
| B3 | 债务表点名 `StairClimbingTaskPlugin.robot_requirements()` 为"这四处属于同一 Phase 2 迁移" | ✅ 验收报告第 95 行 |
| B4 | 做出显式决定：legacy 保留 base hash（否则改写历史 comparison identity）；**责任归 Phase 3A**；主计划 §3.3 同步改写 | ✅ 计划、实施计划、验收报告三处一致 |
| B5 | 未采用我建议的白名单（会把任务 ID 写进任务中立编译器、违反其自有边界测试），改为**内容绑定 grant hash**，覆盖 profile/adapter/binding/environment/requirements/base-protocol/contract hashes | ✅ 实测：把合法 grant 原样复制到环境被改过的 cell → `unsupported: legacy_training_binding_v1_grant_mismatch`。**比我的建议更好** |
| B6 | `articulation_source` 改为 `unitree_rl_lab.assets.robots.unitree:UNITREE_G1_29DOF_CFG` | ✅ 两个 profile 现在同为关节体配置 |
| B7 | 责任明确归 Extension A，并加门："未完成前不得运行" | ✅ dev log §4.3 + 验收报告第 99 行 |

**未引入新问题（逐项查过）**：

- **二源**：`legacy_training_binding_v1` 布尔字段**已从数据模型删除**，只剩派生属性
  `uses_legacy_training_binding_v1 = (grant_hash is not None)`；grant 的两个计算点共用同一个
  `_legacy_training_binding_v1_grant_hash_from_values`，无重复实现。
- **测试是演进不是删除**：旧的 `ambiguous_role_target:torso_link` 断言被替换为语义正确的
  `conflicting_role_target:...`；合法的 `base+forbidden` 重叠由 happy path 覆盖；新增
  `test_legacy_training_binding_v1_grant_is_content_bound`（12 → 13 条）。
- **回归**：全量 **1496 tests OK / 29 skipped**（823 s），与 P1 回执声明的 1496 逐数一致。
- **复杂度**：`robots/` 589 行（+23），`cell_manifest` 864 行（+156，其中 grant 机制约 85 行）。
  仍是零新服务、零数据库、零调度器。grant 的三层函数是重算所需，不是冗余。
- **回执诚实性**：我独立复算了 `P1_MATRIX_CONTRACT_SNAPSHOT.json` —— 26 个被钉文件哈希全部匹配，
  `product_tree_hash = canonical_json_hash(file_hashes)` 与
  `receipt_hash = canonical_json_hash(快照去掉自身 hash)` 均复现成功。

**残留三条小项（均不阻塞，供后续处理）**：

1. **P1 回执没有自动验收器。** Phase 0 有 `verify_phase0_snapshot.py` + 自测；Phase 1 的回执数据
   诚实且可复算（我已复算），但没有脚本或测试自动复核。建议后续 Phase 统一模式，
   否则到 Phase 5 会分不清哪些回执还有效。
2. **`build_default_robot_registry()` 每次解析被调用两次**（grant 一次、resolve 一次），
   重复加载两个 profile JSON（其中一个 50 KB）。只是效率，不影响正确性。
3. **Phase 5 前向提醒**：Go2 上游 rough 奖励含 `undesired_contacts`（thigh/calf）。
   Go2 profile 必须把 thigh/calf **也列入 `forbidden_contact` 的可选目标**，
   否则 Stair binding 会撞 `role_target_outside_profile`。

---

## 0.2 第三轮复核（行尾/回执可复算问题修复后）

上一轮唯一的阻塞项是：`push_shared_loop_evidence` 的 24 条字节绑定证据，
其中 11 个是 CRLF 而 `.gitattributes` 强制 `eol=lf`，`ledger.jsonl` 反之，
**提交后任何一次 checkout 会让 12 条哈希失效**。

**已从源头解决，且修法比我给的三个选项都更贴切。**

采用的是"冻结字节"而不是"规范化行尾"：

```gitattributes
# true_RL_Agent/.gitattributes（纯增量）
PLAN/Variant_construction/P1/PUSH_SHARED_LOOP_DRY_RUN/** -text -eol
```

对**字节绑定的证据**而言，`-text` 是 git 里唯一正确的机制——它让 git 在任何平台
存取时都不做转换，因此既不用重新生成产物，也不用改写已有哈希。

### 我的验证

| 检查 | 方法 | 结果 |
|---|---|---|
| 根 `.gitattributes` 是否被误伤 | diff | **未改动**，`*.sh/*.py/*.md/*.yaml` 全局规则完好；本次改动纯增量 |
| git 实际会存什么 | `git add` 后逐条 `git show :<path>` 与工作区字节对比 | **24/24 blob 与工作区逐字节相同且哈希匹配**，`git add` 零行尾警告 |
| 属性是否生效 | `git check-attr text eol` | 证据目录全部 `unset`（含 `.md`，说明更具体的规则正确覆盖了根规则） |

这比我上一轮的"模拟检出"更强——是直接检查 git 存储层。

### 顺带解决了残留项 1：Phase 1 现在有验收器了

新增 `verify_phase1_snapshot.py`（397 行）+ `test_verify_phase1_snapshot.py`（3 条）。
它**调用产品自身的 compiler 重算**（`build_default_robot_registry` / `compile_cell_manifest`），
而不是重实现一遍——所以比 P0 的 636 行更短却覆盖更多。

我实测了它的漂移检测能力：

| 注入的漂移 | 结果 |
|---|---|
| 改 `robots/base.py` | ✅ 拦截 `product tree mismatch` |
| 证据字节漂移（`audit_summary.json` CRLF→LF，136 字节） | ✅ 拦截 `evidence entries mismatch` |
| 证据内容漂移（`ledger.jsonl` +1 行） | ✅ 拦截 |
| **删掉 `.gitattributes` 里的 `-text` 保护行** | ✅ 拦截（`.gitattributes` 本身在被钉的产品树里） |

最后一条是关键：**这个修复是自守护的**——保护被移除会立刻被验收器发现。
每次注入后我都还原并复跑，均回到 `pass`。

另外它还有一条专门的 `_verify_git_preserves_evidence_bytes`：直接问 git
每个证据文件的 `text`/`eol` 属性是否为 `unset`，失败时明确报错而不是静默跳过。

### 其余状态

- **回归**：全量 **1496 tests OK / 29 skipped**（750 s），无回归；P1 验收器自测 3/3 通过。
- **产品代码本轮零改动**（`robots/` 589 行、`cell_manifest` 864 行与上轮相同），
  改的只有 `.gitattributes` 一行区块 + 新增验收器。**没有越改越乱。**
- **残留项 3** 已记录：Go2 `forbidden_contact` 候选需覆盖 P0 确认的 4 thigh + 4 calf，
  写在 dev log 与主计划 Phase 5 验收条款里。
- **残留项 2** 已在最终收口处理：Push/Stair 各次 cell resolution 只构建一次 registry，并在 grant/compile
  间复用；没有引入缓存或全局生命周期。

**结论：无阻塞项，已推送（`43cfac2`）并同步服务器。**

### 0.2.1 推送后在服务器上发现的一处文档缺口（非功能缺陷）

同步后我在 Linux 上复跑两个 Phase 的验收器：

- **P1 验收器：`pass`**，自测 3/3 通过 → 行尾修复跨平台成立 ✅
- **P0 验收器：`FAIL`**（连同它的 2 条自测）

失败点**只有一个字段**（其余协议/评测/能力/reward ABI/35 cells/schema hash 全部一致）：

| 任务 | P0 记录 | 当前重算 |
|---|---|---|
| Stair | `genesis_snapshot_hash = 264f213d…` | `4b26b227…` |
| Push | `genesis_snapshot_hash = b0924edd…` | `02fb3ca5…` |

**根因**：Phase 1 有意修改了 `ResolvedTaskComposition.genesis_identity_snapshot()` ——
有 manifest 时 schema 升到 `runtime_composition_snapshot.v2` 并加入 `resolved_cell` 块。
这正是"cell 身份进入 Genesis"的实现方式，**是预期行为**，也符合 P0 自己写下的策略
（"开始有意修改受保护产品源码或契约后，它按设计失败"）。

**历史兼容边界需要区分两条路径**：`verify_session_genesis_for_runtime()`
（`agent_core/session_genesis.py:1297`）核对 session/task/backend/anchor/evaluation binding/
执行环境/文件未改，但不重算 `runtime_snapshot_hash`；历史 artifact reader 与 ledger 的只读回放仍可用。
不过 `prepare_session_genesis()` 重开既有会话时会经 `_load_verified_session_genesis()`（第 1849 行）
把该 hash 与当前 composition identity 对照，因此旧 V1 Genesis 不能被当前 V2 composition 续跑，
会按设计报 `session Genesis identity drift`；历史会话应在其冻结源码身份下审计。

**问题在于：P1 的四份文档没有任何一处记录这件事。** 后果是，任何人按 P0 开发日志的指示
"Phase 1 开始时应先运行 `verify_phase0_snapshot.py`"，都会得到一个无法区分
"预期失效"与"真实回归"的失败。

**建议补一段话**（在 P1 验收报告里，约三句）：

1. 两个 legacy cell 的 Genesis composition identity **有意改变**，因为 cell 身份需要进入 Genesis；
2. 因此 `verify_phase0_snapshot.py` 自 `43cfac2` 起**按设计失败**，Phase 0 基线不再是当前入口门；
3. 历史 artifact/ledger 只读回放仍受支持；旧 V1 Genesis 不能用当前 V2 composition 续跑，后者会被 identity-drift 门拒绝。

### 0.2.2 一个供 Phase 3A 参考的观察（非本轮引入）

Genesis composition hash 是**会话创建/重开边界**的执行点，但不是每次 runtime 调用都会重算的持续执行点。
Phase 1 的其余强制来自子进程 manifest 重载、checkpoint lineage 校验和非 legacy cell 的
TrainingBindingV2；Phase 3A 把 resolved hash 提升进 TaskSpec/EvaluationBinding 时，不能把“写进 Genesis”
单独当成运行证据，必须让这些链路与 EvaluationBinding comparison domain 一起闭合。

---

## 1. 机器人语义抽离是否合理 —— 核心判断

### 1.1 分层结果

| 层 | 实际承载 | 判断 |
|---|---|---|
| `RobotProfile` | profile/family/display id、`runtime_facts`（关节/刚体/支撑体/默认姿态/形态尺度/actuator/资产闭包，逐字段等于 P0 回执）、`semantic_roles`、`provided_capabilities` | ✅ 纯物理事实，无任务语义 |
| `RobotRuntimeAdapter` | backend_id、runtime_provider、upstream_revision、articulation_source、capabilities | ✅ 薄，见 B6 |
| `RobotTaskBinding` | `role_bindings`（本任务从 profile 提供的角色里**选**哪些）、task-side capabilities、`task_projection`（P0 binding facts：actions / policy_observation / sensors / control） | ✅ 任务侧投影正确落位 |
| `TaskPlugin` | `robot_requirements()`（角色基数 + 能力）、其余任务语义原样保留 | ⚠️ 见 B2/B3 |

**几处抽得特别对的地方**（值得保留为后续 Phase 的样板）：

1. **`height_scan` 是 capability，scanner 的挂载点与 pattern 在 binding 的 `task_projection` 里，
   不在 profile。** 这是最容易搞错的一处——传感器挂载是任务决策，不是机器人属性。抽对了。
2. **profile 提供 `support_body_names`（物理上哪些体能承载）与 `semantic_roles.support_effector`
   （可作支撑的候选），binding 再从中"选"。** provide/select 两步分离，语义精确。
3. **`_validate_role_targets` 强制角色目标 ⊆ runtime_facts 的真实体名。**
   结构上不可能声明一个不存在的 link——这从类型层消灭了"按裸名猜"的整类 bug。
4. **`forbidden_contact` 角色词汇已定义，但两个 profile 都没有声明它。**
   正确：禁碰是任务判断，不该由机器人自报。（但见 B1——规则却假设了它不能和 base 重叠。）

### 1.2 该保留的任务语义确实保留了

`isaaclab_reward_runtime.py` / `isaaclab_runtime_env.py` / `transition_decoder.py` /
`episode_evidence.py` / `terrain_evidence.py` **全部未被修改**——H2/H3 债务原样留给 Phase 2。
验收报告第 92 行如实写明"这属于 Phase 2，不在 Phase 1 伪装成已解耦"。这是正确的分寸。

### 1.3 是否为后续 Phase 做好拓展性

结构上是的：新增机器人 = 一份 profile JSON + 一个 adapter 条目；新增 cell = 一份 binding + catalog 注册。
compiler 完全任务中立（测试断言源码里不含 `stair`/`push_recovery`/`roughness` 标记）。

**但 B1/B2 是两颗定时炸弹**：它们让"加一行机器人"这件事在下一个 Phase 就会先撞墙，
而墙的位置不在机器人轴，在**任务声明的要求**里。

---

## 2. B1【高·已实测】歧义规则挡住 Stair 自己的 Phase 2 角色集

### 现象

`compile_cell_manifest` 有一条规则：一个 body 只能承担一个角色。

```python
for target in selected:
    previous_role = selected_by_target.get(target)
    if previous_role is not None and previous_role != requirement.role_id:
        reasons.append(f"ambiguous_role_target:{target}")
```

而 Stair 的**冻结语义**里，base 和 forbidden contact 是**同一个 link**：

```
configs/tasks/stair_evaluation_semantics_v6.json
  torso_contact.source = "contact_forces.torso_link"
robots profile
  base -> ["torso_link"]
```

### 实测

我构造了 Phase 2 会用的真实角色集（`base=torso_link`、`support_effector=两踝`、
`forbidden_contact=torso_link`），跑编译器：

```
结果: unsupported
reasons: ['ambiguous_role_target:torso_link']
```

### 为什么这是源头问题而不是小 bug

- **物理上，一个刚体完全可以既是 base、又是禁碰体。** 躯干就是最典型的例子。
  把"一体一角色"当成通用卫生规则，是把一条不成立的物理假设写进了共享编译器。
- **而且它被测试钉死了**（`test_compiler_returns_deterministic_unsupported_reasons`
  第 419 行断言 `ambiguous_role_target:torso_link` 必须出现）。
  Phase 2 要落地 `forbidden_contact`，第一件事就是**推翻 Phase 1 的规则和测试**——
  这正是"表面补丁"的典型形态，只不过补丁提前写在了下一层。

### 建议方向（不代实现）

歧义检查应该只作用于**语义上互斥的角色对**，而不是全局唯一。
例如：`support_effector` 与 `forbidden_contact` 重叠是真错误（脚既要承重又禁碰，自相矛盾）；
`base` 与 `forbidden_contact` 重叠是正常配置。
把互斥关系做成一张显式的小表，比"全局唯一"更贴合物理，也更容易在 Phase 5 复用。

---

## 3. B2【中高·已实测】Push 声明了它不消费的 support_effector

### 现象

```python
# tasks/push_recovery/plugin.py
SemanticRoleRequirement(SUPPORT_EFFECTOR_ROLE, "body", 2, 2)
```

但 **Push 的产品代码里零处引用足端**：

```
grep -rln "foot|feet|ankle" src/true_rl_agent/tasks/push_recovery/  →  (空)
```

Push 的成功判据是 base 高度 + roll/pitch + 平面漂移（`isaac_config.py: success_thresholds`），
完全不解码逐足支撑。

### 实测

我用 **Phase 0 回执里真实的 Go2 事实**（12 joints / 19 bodies / `FL_foot FR_foot RL_foot RR_foot`）
构造 profile，模拟 Phase 5 接入：

```
Stair: unsupported  reasons=['role_cardinality:support_effector']
Push : unsupported  reasons=['role_cardinality:support_effector']
```

Stair 被拒是**诚实的**（decoder 确实只会两足，Phase 2 债务）。
**Push 被拒是没有理由的**——它根本不用足端，却因为一条自己不消费的声明而拒绝四足机器人。

### 为什么值得单独提

计划把 **Extension A（Go2 × Push）** 定为"跨任务复用机器人行"的**强证明**。
现在这条路被 Push 自己的一行声明堵死了，而堵它的原因是 requirement 被从 Stair 复制过来，
没有问"Push 真的需要这个角色吗"。

如果不改，届时最自然的动作是"把 Push 的 max 从 2 放宽到 4"——那就是在补丁上加补丁。
正确的问题是：**Push 应不应该声明 `support_effector`？** 我的判断是不应该（它只需要 `base`），
或者最多声明 `minimum_count=0`（表示"有就用，没有也行"）。

---

## 4. B3 【中】`(2,2)` 是第四处"2"，债务记录没点到它

Phase 0 的债务清单 H3 列出了双足硬编码的三处：
`transition_decoder.py` 的 `range(2)`、`episode_evidence.py` 的 `len != 2`、
`terrain_evidence.py` 的 `must contain two feet`。

Phase 1 新增了**第四处**：`plugin.py` 的 `SemanticRoleRequirement(..., 2, 2)`。

P1 验收报告第 92 行的债务表述是：

> 当前 Stair **runtime/reward/evidence** 内部仍有 G1-37 具体 body 名和双 support 假设

——**没有点名 `robot_requirements()`**。

后果：Phase 2 如果按这句话去改 runtime/reward/evidence，改完 decoder 支持 2/4/6 了，
Go2 仍然会被拒，而错误信息是 `role_cardinality:support_effector`，
看起来像"机器人能力不匹配"而不是"我们忘了放宽声明"。

建议：债务清单显式加上 `tasks/*/plugin.py: robot_requirements()`。

---

## 5. B4 【中】EvaluationBinding 没有拿到 TrainingBinding 那样的分流

### 计划怎么说

> §3.3：`base_task_protocol_hash + manifest_hash` 派生最终 `resolved_cell_task_protocol_hash`，
> 写入 TaskSpec、Genesis、**EvaluationBinding** 和 TrainingBindingV2

### 实现怎么做

- **TrainingBinding**：做了完整分流。legacy cell → V1（无字段，hash 与冻结值逐字节相同）；
  非 legacy cell → V2 + `resolved_cell_task_protocol_hash`。设计很干净：单个类，
  `schema_version` 由字段是否存在派生，V1 的 hash payload 完全不变。
- **EvaluationBindingV1**：**没有任何 manifest 感知**，两个任务都仍传 base hash：
  ```python
  stair:  task_protocol_hash=task_config.task_protocol_hash
  push:   task_protocol_hash=task_plugin.task_spec.resolved_task_protocol_hash()
  ```

### 影响评估（不夸大）

`current_best` 的比较域同时记录 `task_protocol_hash`（来自 eval binding）和
`training_binding_hash`。非 legacy cell 的 training binding 已经带 resolved hash，
所以**训练驱动的 current-best 推进路径是有保护的**。

残留暴露只在：**同一 base 协议 + 同一评测套件、但环境不同**的两个 cell 之间，
走纯评测路径（baseline eval、eval-only replay）时无法区分。
Phase 3 各变体大概率会有不同的评测套件，所以实际暴露窄。

**但这是与计划文本的明确偏离。** 建议二选一并写下来：
要么修订 §3.3 的措辞（承认身份只由 TrainingBindingV2 承载），
要么在 Phase 2/3 补 eval 侧的同款分流。**不要让代码和计划各说各话。**

---

## 6. B5–B7 【低-中】三处应当指派责任 Phase 的项

**B5 `legacy_training_binding_v1` 无结构性约束。**
它只是 `CellVariantDefinition` 上的一个 bool，任何未来 cell 都能设 `True` 从而退回 V1。
计划说"新矩阵 session 不允许降级生成 V1 training binding"，但没有编译期约束。
建议：冻结一份 legacy 白名单（就两个 id），编译器对不在名单里的 cell 拒绝 `True`。

**B6 G1-29 的 `articulation_source` 指向任务环境模块。**

| profile | articulation_source | 是什么 |
|---|---|---|
| G1-37 | `isaaclab_assets:G1_MINIMAL_CFG` | ✅ 关节体配置 |
| G1-29 | `unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg` | ⚠️ **任务 env 配置** |

同一字段两种含义，而且其中一个把机器人身份绑到了**任务环境**——正是这根轴要消除的耦合。
P0 回执单独哈希过 `unitree_rl_lab/assets/robots/unitree.py`，说明存在真正的关节体来源可指。
今天只是一个字符串不影响运行，但会在 Phase 5 误导"新机器人该填什么"。

**B7 Push 占位标签仍在执行路径上。**
`push_recovery_obs_v1` / `unitree_actions_v1` 仍是 backend config 实际使用的值（`composition.py:288-289`），
而真正内容绑定的 `policy_io_schema_hash`（`b9808edd…`）只存在于 manifest 里。
dev log §4.3 给了充分理由（直接替换会破坏 legacy job/baseline 身份），验收报告 P1-G4 也写明了——
**这是一个记录在案的决定，不是隐藏的补丁**。但它没有责任 Phase。
建议指派给 Extension A（届时 Push 本来就要动）。

---

## 7. 我独立复核**通过**的部分

| 复核项 | 我做了什么 | 结果 |
|---|---|---|
| `robots/` 是否任务中立 | 逐文件看 import | 只 import `evaluation.binding`（hash 工具）、`json_contract` 和自身；**零 task 引用** |
| profile 是否真的等于 P0 事实 | 读测试 + 实际加载 | `assertEqual(profile.runtime_facts, receipt[key])` **逐字段比对**，加生产加载器内部 hash 重算 |
| 角色目标是否可能指向不存在的体 | 读 `_validate_role_targets` | 结构上不可能——目标必须 ⊆ runtime_facts 的真实名 |
| 两个 profile 是否真的不同 | 实际导出 | `8dc40b0e…` vs `fea00204…`；37 vs 29 关节；资产闭包不同 |
| 能力词汇是否名副其实 | 实际导出 | Stair 有 `height_scan`，Push 有 `external_body_force`——差异真实反映任务 |
| 身份派生是否自指 | 读代码 | `resolved = H(schema, base_hash, manifest_hash)`，**无自引用** |
| legacy 字节兼容 | 跑冻结协议测试 | `formal_training_binding_hash = 2a469d70…` 仍被 config 重算校验；40 个协议/入口/binding/replay 测试通过 |
| 子进程重载是仪式还是真检查 | 读 runner + entrypoint | **真检查**：entrypoint 在 App 创建前重载并校验 task_id/base protocol/hash；runner 用 manifest 拒绝 cell 身份不符的 checkpoint |
| compiler 是否任务中立 | 读测试 + 源码 | 源码不含 `stair`/`push_recovery`/`roughness`/`phase0_binding_facts` |
| 二源检查 | 全面扫 | `runtime_facts` 是 P0 回执的第二份拷贝，**由测试逐字段守住**（不是生产加载器守住，可接受但要知道）；Push 双身份见 B7；未发现其它 |
| 新测试 | 跑 `test_robot_matrix_contract` | 12 tests OK，覆盖全部 7 条验收 |
| 回归（子集） | 跑协议/入口/binding/replay 子集 | 40 tests OK |
| 回归（全量） | `unittest discover -s tests`（835 s） | **1495 tests OK / 29 skipped / 0 失败**；Phase 0 时为 1483，增量恰为新增的 12 条矩阵测试，**无回归** |

---

## 8. 过度复杂化？没有

| 指标 | 数值 | 判断 |
|---|---|---|
| `robots/` 总行数 | 566（base 396 / registry 67 / defaults 76 / init 27） | 精简 |
| compiler | 708 行，一个 `compile_cell_manifest` 函数 + 5 个数据类 | 与职责相称 |
| 新增运行时组件 | **0**（无服务、无数据库、无调度器、无第二 registry） | ✅ |
| 修改产品文件 | 15 个，共 +527/-15 行 | 外科手术式 |
| 死代码 | 无 | ✅ |

唯一"重"的地方是 `unitree_g1_29dof_*.json` 有 50 KB，其中 36 KB 是 209 条 asset closure 条目。
但那是**与 P0 回执逐字段绑定的必然代价**——砍掉就破坏了交叉校验。属于正确的重。

一处可以更省的：asset closure 的 `entries` 只在 profile 里被哈希、从不被逐条读取。
将来六足/Go2 的 description tree 更大时，可以只存 `closure_hash + file_count + total_bytes`，
把 entries 留在回执里。**现在不必改**，记一笔即可。

---

## 9. 建议的处理顺序

| 优先级 | 项 | 理由 |
|---|---|---|
| 1 | **B1** 歧义规则改为互斥角色对 | Phase 2 第一件事就会撞上；现在改只动 compiler + 1 个测试，Phase 2 再改要连带推翻已冻结的行为 |
| 2 | **B2** 重新判定 Push 是否需要 `support_effector` | 决定 Extension A 能不能走通；且现在改只动一行声明 |
| 3 | B3 债务清单点名 `robot_requirements()` | 一句话，防止 Phase 2 漏改 |
| 4 | B4 决定 eval 侧要不要同款分流（或修订计划文本） | 让代码与计划一致 |
| 5 | B5 legacy 白名单约束 | 十行以内 |
| 6 | B6/B7 指派责任 Phase | 文档 |

前两项都是**现在改成本极低、拖到下个 Phase 改成本翻倍**的类型。

---

## 10. 一句话

> 机器人轴抽对了：分层干净、与 Phase 0 事实逐字段绑定、legacy 一字节没动、复杂度没有失控。
>
> 但共享编译器里有两条**把物理事实写错了的规则**：
> 一条认为一个刚体只能有一个角色（Stair 的躯干同时是 base 和禁碰体），
> 一条让 Push 要求它根本不用的两只脚（会挡住 Go2 × Push 这个跨任务复用的强证明）。
>
> 两条都已实测复现，都在现在改最便宜。

---

## 附：本次审计运行的命令

```bash
# 实际导出两份 resolved manifest 并逐字段检查
<scratchpad>/… resolve_stair_cell_manifest / resolve_push_cell_manifest

# 用 P0 回执真实 Go2 事实模拟 Phase 5 接入
<scratchpad>/p1_probe.py
# → Stair: unsupported role_cardinality:support_effector
# → Push : unsupported role_cardinality:support_effector

# 模拟 Phase 2 真实角色集（base 与 forbidden_contact 同为 torso_link）
<scratchpad>/p1_phase2_probe.py
# → unsupported ambiguous_role_target:torso_link

# Push 是否真的消费足端
grep -rln "foot|feet|ankle" src/true_rl_agent/tasks/push_recovery/   # 空

# 回归
python -m unittest -q tests.unit.test_robot_matrix_contract          # 12 OK
python -m unittest -q tests.unit.test_stair_protocol_v2 \
    tests.unit.test_stair_entrypoint_materialization \
    tests.unit.test_evaluation_binding tests.unit.test_acceptance_replay   # 40 OK
python -m unittest discover -s tests -q                              # 全量（见正文）
```

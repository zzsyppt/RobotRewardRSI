# Stair Climbing Phase 1B 详细实施计划

- 计划日期：2026-07-27
- 实施基线：`81ea436`（Phase 1A 已完成）
- 对应总计划：`PLAN/stair/stair_climbing_task_plan.md` 的阶段 1B
- 阶段目标：让 runner-owned baseline eval 成为真实、不可变、可恢复、可重放的初始 current-best
- 阶段边界：不注册生产 Stair `TaskPlugin`，不实现楼梯环境、reward、失败分类或训练/eval entrypoint
- 实施状态：已完成（2026-07-27）

## 1. 当前事实与设计判断

Phase 1A 已经提供以下可信输入：

- `ResolvedTaskComposition` 只解析一次 task/backend/runner/policy initialization；
- `BaselineManifest` 将 runtime baseline 单向绑定到任务自有来源、checkpoint 和兼容性；
- `BackendAdapter.materialize_baseline_eval()` 与 `Runner.run_baseline_eval()` 能执行无候选、无训练的 baseline eval；
- `EvaluationBindingV1` 已固定 task/eval/acceptance/seed cohort 四个比较域；Phase 1B 负责 session 级 create/load/validate seam，具体 seed/bin manifest 仍由任务包拥有；
- checkpoint 在 subprocess 启动前完成真实字节哈希核对，进程内 `weights_only=True` 为二次防线；
- baseline/candidate eval 已共享 subject、checkpoint safety、sealed rollout 和 isolation 证据合同。

当前缺口不是 Stair 语义，而是 session 级可信起点仍不存在：

- live/replay/context/audit 仍分别接收 scalar `initial_best_score`；
- replay 固定从 bootstrap 开始，不能恢复 baseline 的完整统计、checkpoint 和 EvaluationBinding；
- ledger 没有绑定创建它的 session genesis；
- live output 有 ledger 时直接拒绝，不能安全恢复；
- prompt 只有“accepted candidate source”与“best observed source”，无法表达“有 baseline current-best，但没有候选源码”。

参考 `D:/humanoid_agent/codex` 的 thread/rollout 持久化边界，本阶段采用两条原则：

1. create 与 resume 是不同路径；resume 读取并验证已有 head metadata，不通过重建来修复漂移。
2. 不可变 session head 与 append-only history 分离；Genesis 固定比较域，ledger 只追加每轮派生事实。

## 2. 目标架构与硬边界

### 2.1 单一 session 信任根

新增 task-neutral `SessionGenesisArtifact`，canonical payload 只包含：

- schema、genesis kind、session/task/backend/policy initialization identity；
- 完整 `EvaluationBindingV1`；
- bootstrap anchor，或 baseline anchor；
- canonical `session_genesis_hash`。

baseline anchor 包含：

- baseline ID、manifest hash、checkpoint hash；
- runner 返回的完整 public result 及其 canonical hash；
- `TrustedEvalSeal`；
- private normalized baseline-eval artifact ref/checksum。

不在 Genesis 中保存调用方自报的 `initial_current_best`。校验器必须从 anchor 证据重新推导
`CurrentBestEvalStats`，成功后才返回不可直接构造的 `VerifiedSessionGenesis`。

### 2.2 两种明确起点

- `baseline`：用于带兼容 baseline manifest 的任务；初始 current-best 为
  `mode="accepted"`、`anchor_kind="baseline"`。训练初始化可独立声明，但 Stair V1 明确使用同一 manifest 的 warm-start。
- `bootstrap`：仅当 `TaskSpec.requires_baseline_genesis=false` 且 policy initialization 为
  `random` 时允许；保留 Push 当前 absolute-floor 首次接受行为。

调用方不能把 warm-start task 临时降级为 bootstrap，也不能同时传 Genesis 和另一份 scalar/current-best。

### 2.3 共享层与任务层

共享层只识别以下通用概念：

- baseline/candidate/bootstrap anchor；
- EvaluationBinding、score stats、checkpoint safety、sealed validation；
- task-declared gate 名称和状态。

共享层禁止出现 stair height bin、top hold、toe collision、base contact、terrain 或 Stair protocol 字段。
任务行为门继续由未来 Stair plugin/eval projection 产生，Genesis 只验证任务声明的门是否完整通过。

baseline anchor 不要求 candidate verifier、训练预算或 runtime activation。`budget_accounting` 仍是 candidate-only
公共门；baseline 仍必须通过 accept evidence、checkpoint safety、sealed validation 和所有 task-owned eval gates。

## 3. 分阶段实施

### 3.1 纯合同：Genesis 与 backend binding

**改动**

- 在 `BackendAdapter` 增加 `evaluation_binding_for(task_spec)` 以及 task-neutral 的 `resolve_session_evaluation_binding(..., allow_create)` / `validate_session_evaluation_binding(...)`：
  - 默认实现严格返回既有静态 binding，Push 行为不变；
  - `allow_create=false` 的 override 必须只读，禁止重新抽样或覆盖；
  - Phase 2 的 Stair backend 在任务包内覆盖该 seam，负责私有 seed/bin manifest，公共层只消费 `EvaluationBindingV1`。
- candidate/baseline job materialization 显式接收已验证 session binding；job spec、Isaac argv、公开结果和 seal 不得回退到 backend config 中另一份 binding。
- 新增 `agent_core/session_genesis.py`：
  - `SessionGenesisArtifact` 的严格 schema、canonical serialization/hash；
  - baseline/bootstrap anchor 的互斥字段合同；
  - `VerifiedSessionGenesis` 与 contextual verifier；
  - public baseline anchor 投影，不公开 private URI、checkpoint 本地路径或 seed 明细。
- 将 `TrustedEvalSeal` 的现有 builder 作为 baseline/candidate 共用 seal 权威，不另造第二种 seal。
- 为 acceptance evidence 增加显式 subject kind；baseline 忽略 candidate-only budget gate，其余共享门和 task-owned 门不弱化。

**严格验收**

- malformed hash、未知字段、baseline/bootstrap 混合字段、错误 subject 或非 canonical binding 均拒绝。
- backend 返回的 binding 必须与 TaskSpec、eval protocol、acceptance protocol 和 seed cohort 一致。
- baseline 缺 accept/sealed/task gate、checkpoint safety 或 seal 任一项均不能产生 Verified Genesis。
- baseline 不因不存在训练预算证据而失败；伪造 candidate budget pass 也不能替代 baseline 必需证据。

### 3.2 create-once 创建与恢复

**改动**

- 实现 `prepare_session_genesis()`：
  - 在 baseline eval 前先 create/load task-owned session binding；恢复已有 Genesis 时强制 `allow_create=false`；
  - 新 baseline session：materialize baseline eval -> runner eval -> trusted seal -> normalized private artifact -> Genesis；
  - 新 bootstrap session：直接从已验证 TaskSpec/backend binding 创建；
  - 已有 session：只加载并重新验证 `session_genesis.json`，不调用 baseline runner。
- `session_genesis.json` 和 normalized baseline-eval artifact 使用同目录临时文件、fsync 和 create-only 原子发布；禁止覆盖。
- 若 ledger/partial artifact 已存在但 Genesis 缺失，fail closed，不静默重跑 baseline。
- 恢复时重新核对 session/task/backend/policy initialization、manifest、checkpoint、完整 binding、private artifact bytes 和 trusted seal。

**严格验收**

- 同一 session 第二次 prepare 的 baseline runner 调用计数保持不变，Genesis bytes/hash 不变。
- CLI/task/backend/baseline/checkpoint/protocol/seed cohort 任一漂移在 provider 前失败。
- 篡改 Genesis、private artifact 或 checksum 后恢复失败；失败路径不覆盖任何原文件。
- 新建时并发/重复发布只能有一个成功文件，不出现 last-writer-wins。

### 3.3 Full current-best、decision 与 replay

**改动**

- `CurrentBestEvalStats` 增加：
  - `anchor_kind`、`anchor_id`；
  - `checkpoint_hash`、`baseline_manifest_hash`；
  - task/eval/acceptance/seed cohort hash 与 `evaluation_binding_hash`。
- baseline current-best 由 baseline public result、seal 和 binding 推导；candidate 接受后从该轮 eval checkpoint/public result 推导并切换为 candidate anchor。
- `replay_acceptance_ledger()`、`project_public_history()`、`build_eval_run_audit()` 接收完整 initial current-best 和 expected Genesis hash，不再接收 scalar。
- 每条 ledger record 顶层写入同一个 `session_genesis_hash` 和完整 `evaluation_binding`；provider/static failure 记录也不例外。
- decision comparison 写入完整 prior-anchor identity；candidate job、public result、top-level record 与 current-best 的 binding 必须相同。

**严格验收**

- 首个 candidate 高于 absolute floor、但不超过 baseline noise-aware threshold 时拒绝。
- 接受 candidate 后 anchor/checkpoint 切换，训练初始化仍保持固定 baseline，不随 current-best 改变。
- replay 恢复的 score stats、anchor、checkpoint、manifest/binding/protocol hash 与在线状态逐字段一致。
- 篡改任一 ledger Genesis hash、binding、comparison prior-best、eval checkpoint 或 public subject 均使 continuation/replay 失败。
- Push bootstrap 首次接受仍使用 absolute floor，后续轮次仍使用 noise-aware delta。

### 3.4 Live loop、恢复与旧入口迁移

**改动**

- `run_live_loop()` 只接收 `VerifiedSessionGenesis`；进入 provider 前重新核对文件、runtime identity、ledger sequence 和 replay。
- `rounds` 定义为 session 的目标总轮数；恢复从已有 ledger 长度继续生成下一连续 round ID。
- `scripts/run_loop.py` 先 resolve composition、prepare/load Genesis，再构造 provider；baseline task 不允许 `--initial-best-score` 覆盖。
- replay-only `AgentLoop` 和 Push dry-run audit 同样使用显式 Genesis；删除剩余 scalar replay 起点。
- summary 增加 genesis kind/hash、resume 起点和最终完整 current-best，既有计数语义保持不变。

**严格验收**

- raw/missing/incomplete Genesis 无法进入 provider；用 marker provider 证明零调用。
- 有效恢复从下一 round 继续，不重复已有轮次；目标轮数等于已有轮数时不调用 provider。
- session ID/round sequence 漂移在 append 前失败。
- Push live fixture、Push dry-run audit、`run_round.py` 与 `materialize_job.py` 既有行为不退化。

### 3.5 Prompt/source catalog baseline anchor

**改动**

- `ProjectedPublicHistory` 保留 verified initial current-best，不再只保存 initial score。
- 尚无 accepted candidate 且 initial anchor 为 baseline 时：
  - current-best catalog item：`role="baseline_policy_anchor"`、`status="baseline_anchor"`、`requestable=false`、`max_bytes=0`；
  - lineage 公开 baseline ID、public score 与 protocol/binding 状态；
  - 不生成 current-best source payload，不暴露 checkpoint 路径、private artifact 或 seed 明细。
- best-observed nonaccepted source 继续独立工作；它只有 authoring 参考权，不覆盖 baseline policy-best 权威。
- 首个 accepted candidate 后恢复既有 accepted-source catalog/payload 行为。

**严格验收**

- baseline anchor 的 current-best logical ID 存在但不能 request，payload 列表中没有对应内容。
- baseline + best rejected 可以同时表达，二者 authority 不混淆。
- 首个 accepted candidate 后 current-best source 精确切换为该 candidate；旧 baseline 不能伪装成源码。
- prompt public-safety 检查确认无本地路径、private URI、checkpoint bytes 或 hidden seed。

### 3.6 回归、故障注入与收尾审计

**测试层级**

- 单元：Genesis schema/hash、anchor 推导、baseline gate、binding、current-best、source catalog。
- 集成：baseline eval -> Genesis -> live first decision；resume；ledger replay；Push bootstrap。
- 故障注入：逐字段篡改 manifest/checkpoint/task/eval/acceptance/seed/seal/private artifact/ledger hash。
- 全量：现有 unit/integration/script tests、Ruff、compileall、diff check。

**完成判定**

- 所有第三节硬验收均有行为测试，不使用源码字符串顺序测试代替运行行为。
- Phase 0 报告仍只负责 baseline admission，不被 Genesis 当作 acceptance evidence。
- Stair 仍未注册到生产 registry；Phase 2 只需提供 task-owned plugin/protocol/eval projection 和 session seed-manifest resolver，不再修改 Genesis/replay/作业绑定基础设施。
- 计划、实现与当前 hash/命令同步；不记录未实际执行的验收结果。

## 4. 验收命令

实施完成后至少执行：

```powershell
uv run --no-project --with pytest python -m pytest -q tests/unit/test_session_genesis.py
uv run --no-project --with pytest python -m pytest -q tests/integration/test_session_genesis_live_loop.py
uv run --no-project --with pytest python -m pytest -q tests/unit/test_current_best_eval_stats.py tests/unit/test_acceptance_replay.py tests/unit/test_prompt_source_catalog.py
uv run --no-project --with pytest python -m pytest -q tests/integration/test_live_loop.py tests/integration/test_dry_run_audit_loop.py tests/integration/test_baseline_eval_only.py
uv run --no-project --with pytest python -m pytest -q
$files = @((git diff --name-only -- '*.py')) + @((git ls-files --others --exclude-standard -- '*.py'))
uv run --no-project --with ruff ruff check $files --select E9,F,I
uv run --no-project python -m compileall -q true_RL_Agent/src true_RL_Agent/scripts true_RL_Agent/tests
git diff --check
```

Ruff 从 `D:/humanoid_agent` 执行，验收本阶段改动文件；不把 Phase 1A 已明确保留的历史全仓 import-order 债务误报为本阶段完成项。

## 5. Phase 1B 退出条件

只有同时满足以下条件才标记完成：

1. warm-start fixture 的 baseline eval 可以生成 Verified Genesis 和 full accepted current-best；
2. live/provider、ledger、decision、replay、eval audit、prompt 全部消费同一 Genesis 派生状态；
3. resume 不重跑 baseline、不重建 Genesis，所有身份漂移 fail closed；
4. baseline anchor 没有伪造 candidate/source，accepted candidate 切换正确；
5. Push bootstrap 与所有旧入口通过全量回归；
6. 共享代码中没有 Stair 特殊语义，生产 Stair task 仍保持禁用。

## 6. 实施与验收结果

2026-07-27 按本计划完成，实际交付如下：

- 新增 create-once `session_genesis.json`、normalized private baseline-eval artifact、严格 contextual verifier 和不可直接构造/不可变的 `VerifiedSessionGenesis`；baseline current-best 只从已验证评估证据推导。
- baseline 与 candidate 共用 EvaluationBinding、checkpoint safety、sealed rollout 和 trusted seal；baseline 明确不要求 candidate-only `budget_accounting`，但不会弱化 task-owned gate。
- backend 已具备 session binding create/load/validate seam，candidate/baseline job spec 与 Isaac eval argv 使用同一份已验证 binding；默认 Push 后端保持原静态 binding 行为。
- live loop 已支持目标总轮数语义和安全 resume；Genesis、round sequence、task/backend identity、ledger binding 和 replay 在 provider 前验证。replay-only loop 与 Push dry-run audit 已迁移到显式 Genesis。
- full current-best 已绑定 anchor/checkpoint/manifest/task/eval/acceptance/seed/binding 和完整统计；内部映射递归只读，不能在拿到 Verified Genesis 后原地改分数。
- prompt 已支持无伪造源码的 baseline anchor、独立 best-observed rejected 证据，以及 accepted candidate 接管后的 source 切换。

最终验收结果：

1. 全量测试：`574 passed, 4 skipped, 414 subtests passed in 28.27s`。4 项为本地未安装 Torch 时动态 Phase 0 probe 行为测试的既有条件跳过，不是失败。
2. warm-start 行为链覆盖 baseline `0.25`、候选 `0.28` 拒绝、`0.31` 接受、固定 baseline 训练起点、resume 零重复 baseline eval、目标轮数 no-op 和完整 replay 等价。
3. 故障注入覆盖 malformed/rehashed Genesis、manifest、trusted seal、private artifact 内容与路径、bootstrap score、runtime snapshot、session binding、ledger Genesis/binding/task identity，以及并发 create-only 发布。
4. 本阶段全部改动 Python 文件 Ruff `E9/F/I` 通过；`compileall` 与 `git diff --check` 通过。
5. 共享 `agent_core/evaluation/backend base/runner` 任务语义扫描无 stair height/bin/contact/terrain/top-hold 字段；生产 registry 中仍无 `unitree_stair_climbing` 或旧 `future_stair_climbing`。

阶段 2 的剩余边界明确：在 Stair 任务包内实现 plugin/protocol/eval projection 和私有 seed/bin manifest resolver；不得把楼梯字段加入本阶段共享合同，也不得把 Phase 0 admission report 当作 baseline acceptance evidence。

# 上楼梯任务 Phase 1A 详细实施计划

- 制定日期：2026-07-24
- 代码基线：`810b760`（`true-rl-agent-refactor`）
- 阶段目标：完成多任务组合入口与 task-neutral baseline eval-only 基础设施
- 阶段边界：不注册生产楼梯 `TaskPlugin`，不创建 Session Genesis，不改变 current-best、acceptance replay 或候选训练起点
- 实施状态：已完成并通过第五节验收（2026-07-25）

## 1. 当前缺口

当前核心抽象已经有 `TaskPlugin`、`TaskRegistry`、`BackendAdapter`、`BackendRegistry`、
`PolicyInitialization` 和 baseline manifest，但生产组合仍有四个未闭合点：

1. `run_loop.py` 和 `materialize_job.py` 直接构造 push config/plugin/backend，CLI 默认值也由脚本复制。
2. `run_round.py` 名称通用，但实际只运行 push audit。
3. `EvalJobSpec`、eval summary 和 Runner 都把 `candidate_id` 当作唯一评估身份，无法表达 baseline。
4. backend 与 Runner 只有 candidate train+eval 路径，没有不经过 candidate/verifier/train 的 baseline eval-only 路径。

Phase 0 已证明官方 G1 Rough checkpoint 可以作为 warm-start 候选，但没有证明它是楼梯
current-best。Phase 1A 只建立“如何安全评估 baseline”的能力；将评估结果变成可重放
current-best 属于 Phase 1B。

## 2. 设计原则

### 2.1 组合发生在应用根，不进入 AgentCore

- 共享层只定义 composition provider、registry、resolved snapshot 和两阶段解析协议。
- task-owned composition provider 负责 task defaults、plugin、memory adapter、backend config、
  Runner 配置和 initialization。
- 默认应用 composition root 只做显式注册，不按 task ID 写条件分支。
- `TaskPlugin` 继续只拥有任务语义，不返回训练命令，不执行 checkpoint 或 eval。
- `BackendAdapter` 继续只物化 job，不拥有 provider、accept/reject 或 ledger 权限。

该设计参考 `D:\humanoid_agent\codex` 已收敛的做法：配置按明确优先级解析为一个快照，
扩展按稳定 ID 注册和排序，运行期消费已解析快照，不在共享执行路径中重新猜测扩展默认值。

### 2.2 评估对象统一，训练对象保持 candidate-only

- 新增 `EvaluationSubject(kind="candidate"|"baseline", subject_id=...)`。
- candidate `EvalJobSpec` 与 baseline `BaselineEvalJobSpec` 使用同一评估身份合同。
- `TrainJobSpec`、candidate verifier 和 candidate source 仍保持 candidate-only。
- baseline 不得出现伪造的 `candidate_id`、candidate package、verifier、train result 或 code-level surface。
- job spec 可以保留只读兼容属性，但原始 eval summary 只允许
  `subject_kind/subject_id` 这一套扁平身份；顶层或 metrics 中的 `evaluation_subject`、
  `subject`、`candidate_id`、`baseline_id` 等第二身份一律拒绝。公共 result 中的结构化
  `evaluation_subject` 只能由 Runner 根据 job spec 生成。

### 2.3 EvaluationBindingV1 是完整比较域

新增 task-neutral `EvaluationBindingV1`，严格绑定：

- `eval_protocol_hash`
- `acceptance_protocol_hash`
- `task_protocol_hash`
- `seed_cohort_hash`
- 由以上字段计算的 `evaluation_binding_hash`

共享层只校验结构、hash 和字段关系，不理解楼梯高度、分桶、成功阈值或失败分类。
push 与未来 stair 分别在任务包内生成自己的 `task_protocol_hash` 和 `seed_cohort_hash`。

Phase 1A 要求 binding 进入：

- candidate/baseline eval job spec；
- eval argv 与 summary；
- sealed rollout evidence 和 sealed validation gate；
- Runner 返回的 public result；
- trusted eval seal。

CurrentBest、decision comparison 和 ledger replay 对 binding 的强制检查留在 Phase 1B。

### 2.4 baseline manifest 与 checkpoint 分层校验

- task-owned source loader 负责具体 manifest schema、任务语义和准入证据；共享层不解析
  Phase 0 字段。
- 共享 `BaselineManifest`（schema v2）提供 runtime canonical manifest hash、通用兼容字段和可选
  `source_binding`。`source_binding` 只表达通用的 source schema/hash、信任锚 hash 和
  admission evidence hash，不包含楼梯高度、地形或成功语义。
- Stair 使用任务包内的确定性 adapter：先重新验证 admitted Phase 0 bundle 和当前 checkpoint
  字节，再从冻结 source manifest 派生 runtime manifest；provenance 和 compatibility 不由
  CLI/调用者手工补写。
- Phase 0 source manifest hash 与 runtime manifest hash 属于不同 schema，数值不要求相等；
  runtime manifest hash 必须承诺完整 `source_binding`，因此二者不再是可独立漂移的身份。
- backend materialization 前检查 task/backend/robot/observation/action/policy architecture、
  checkpoint role、seal 和完整 SHA-256 格式。
- `SubprocessRunner` 在启动 eval 或 warm-start train 子进程前读取本地文件字节并校验精确
  SHA-256；文件缺失、symlink、不可读或 hash 不符时不得启动子进程。
- eval/train 子进程继续通过共享 checkpoint safety 入口复核相同 hash，并以
  `weights_only=True` 加载，作为纵深防御而不是首次字节校验。
- composition 同时存在 initialization 与 baseline anchor 时比较完整 runtime manifest hash，
  不能只比较 baseline ID、task ID 和 checkpoint。
- metadata 中的本地 checkpoint 路径只用于私有 argv，不进入公共 manifest/job 投影。

## 3. 实施步骤

### 1A-1：统一评估身份与协议 binding

**修改范围**

- `evaluation/subject.py`
- `evaluation/binding.py`
- `runner/job_identity.py`
- `runner/specs.py`
- `tasks/base.py`
- `evaluation/sealed_validation.py`
- `safety/eval_seal.py`

**完成目标**

- 增加严格、不可变、可 canonical hash 的 subject/binding 合同。
- `EvalJobSpec` 强制携带 candidate subject 和完整 binding。
- 新增只接受 baseline subject、warm-start checkpoint 和 manifest hash 的
  `BaselineEvalJobSpec`。
- `TaskSpec` 暴露 task protocol hash 和 `requires_baseline_genesis`，未显式提供 hash 的
  现有测试任务使用共享 canonical task contract hash；正式 stair 后续必须显式使用
  `StairProtocolV1` hash。
- sealed evidence、gate 和 trusted seal 同时绑定 subject/checkpoint/evaluation binding。

**硬验收**

- subject kind、ID、任一 hash 缺失、格式错误或自哈希不一致均拒绝。
- candidate checkpoint 与 baseline checkpoint 角色不能互换。
- 篡改 seed cohort、task protocol 或 acceptance protocol 后，即使 eval protocol 未变，
  seal/evidence 仍失效。

### 1A-2：baseline eval-only 后端与 Runner 路径

**修改范围**

- `backends/base.py`
- `backends/specs.py`
- `backends/isaac_lab_rsl_rl.py`
- `runner/base.py`
- `runner/subprocess_runner.py`
- `runner/checkpoint_preflight.py`
- `runner/dry_run.py`
- `policy.py`
- `policy_baselines.py`
- `tasks/stair_climbing/baseline_adapter.py`

**完成目标**

- `BackendAdapter.materialize_baseline_eval(...) -> BaselineEvalJobSpec`。
- `Runner.run_baseline_eval(...) -> PublicEvalResult`。
- `IsaacLabRslRlBackendAdapter` 复用同一个私有 eval argv/materialization 实现，不复制一套
  baseline subprocess 逻辑。
- `SubprocessRunner` 复用同一个 summary、checkpoint、evidence 和隔离校验实现，但按
  `EvaluationSubject` 验证身份。
- backend capability 显式声明是否支持 baseline eval-only。

**硬验收**

- baseline eval argv 不含 candidate ID、candidate path 或 candidate source。
- baseline job 不生成 `TrainJobSpec`，也不经过 candidate verifier。
- manifest hash、checkpoint hash、task/backend/schema/architecture 任一不匹配时，
  在 subprocess/provider 前失败。
- checkpoint 文件实际字节与声明 hash 不符时，baseline/candidate eval 和 warm-start train
  均在 subprocess 启动前失败；故障注入的子进程 marker 必须不存在。
- summary 将 baseline 错报为 candidate 或错报 baseline ID 时，Runner fail closed。
- raw summary 或 metrics 携带第二套嵌套/别名身份时，Runner fail closed。
- Stair runtime manifest 必须由 admitted Phase 0 bundle 单向派生，并绑定 source manifest、
  descriptor trust anchor 和 Phase 0 report；错误 checkpoint 字节不得完成转换。

### 1A-3：task composition resolver 与两阶段 CLI

**修改范围**

- `composition/base.py`
- `composition/registry.py`
- `composition/cli.py`
- `composition/defaults.py`
- `tasks/push_recovery/composition.py`

**完成目标**

- provider 按 task ID 注册，返回一个不可变 `ResolvedTaskComposition`。
- snapshot 至少包含 plugin、memory adapter、backend adapter/config、Runner、
  task defaults、initialization 和可选 baseline manifest。
- 第一阶段只解析 `--task-id`，选择 provider 后再应用该任务 defaults，最后完成完整解析。
- 显式 CLI 参数覆盖 task defaults；没有显式参数时绝不继承另一个任务的值。
- production registry 当前只注册 push；stair production provider 留到 Phase 2。

**硬验收**

- 两个具有不同预算、profile、schema 和 initialization 的测试 provider 不发生默认值串线。
- 未注册 task、重复 provider、provider 返回错误 task ID 或不兼容 backend 时均在执行前失败。
- resolver 共享代码不包含 push/stair task ID、楼梯高度、push profile 或任务包导入。
- warm-start initialization 与 baseline anchor 的 runtime manifest hash 不同，即使
  baseline ID、task ID 和 checkpoint 相同，也必须拒绝 composition。

### 1A-4：迁移三个组合入口并保持 push 行为

**修改范围**

- `scripts/run_loop.py`
- `scripts/materialize_job.py`
- `scripts/run_round.py`
- `scripts/isaaclab_rsl_rl_eval.py`
- `tasks/push_recovery/loop_audit.py`

**完成目标**

- 三个入口都通过同一 registry/provider 选择任务和 defaults。
- 每个入口每次运行只 resolve 一次 composition；不保留会各自重复 resolve 的兼容 helper。
- push config/backend 构造逻辑收敛到 push-owned composition provider。
- `run_round.py` 通过 provider 的 audit capability 执行；不再直接导入 push audit。
- `materialize_job.py` 不再包含 push config/plugin/backend 分支。
- push eval entrypoint 改为读取 subject/binding，并在任务包内验证实际 seed cohort hash。

**允许的兼容变化**

- eval identity flags 从 `--candidate-id` 迁移为
  `--subject-kind candidate --subject-id <id>`。
- eval argv 新增四个 binding constituent/hash 参数。

其余 push 默认预算、profile、runtime config、train argv、checkpoint 路径、seed 数、
eval env 数和候选隔离行为必须保持一致。

### 1A-5：测试、故障注入与阶段复审

**新增或扩展测试**

- composition registry、两阶段 defaults 与显式覆盖。
- baseline manifest canonical hash 和兼容性故障注入。
- candidate/baseline subject 混淆、summary identity 篡改。
- raw summary/metrics 第二身份注入。
- eval 与 warm-start train 的错误 checkpoint 字节、零子进程启动故障注入。
- Stair source manifest 到 runtime manifest 的确定性派生、source binding 和错误字节拒绝。
- baseline eval-only backend argv 与 Runner 动态 summary 测试。
- binding constituent、sealed evidence/gate/seal 篡改；实际字段与全部 `expected_*`
  哈希均覆盖“格式错误但彼此相等”的故障注入。
- materialize CLI 的 push 兼容快照。
- `run_loop.py --mode dry-run-audit` 与 `run_round.py` 的显式 task runtime 参数必须
  出现在实际 ledger job 中，不得回落到 push 默认值。
- project boundary：共享 composition/evaluation/runner 层不得导入 stair/push 包或出现任务语义。

**全量验收命令**

1. Phase 1A focused unit/integration tests。
2. `python -m unittest discover -s tests -p "test_*.py"`。
3. Ruff `F`/`I` 检查。
4. `python -m compileall src scripts tests`。
5. `git diff --check`。
6. 对共享层执行 push/stair 语义字符串与 import 扫描。

## 4. 阶段完成定义

Phase 1A 只有在以下条件全部满足后才能标记完成：

- 所有 eval job 都有唯一 `EvaluationSubject` 和完整 `EvaluationBindingV1`。
- baseline eval-only 能独立物化和执行，不伪造 candidate 或训练证据。
- `run_loop.py`、`materialize_job.py`、`run_round.py` 不再直接构造 push 组件。
- task defaults 由被选择的 provider 提供，显式覆盖顺序确定且有行为测试。
- push 既有 dry-run、live composition、train/eval materialization 和隔离合同无回归。
- 共享层没有楼梯或 push 的任务特殊语义。
- stair 仍保持 production plugin/training disabled，Phase 0 report 仍不是 baseline eval evidence。
- Phase 1B 所需的输入已经形成稳定接口，但本阶段没有创建 Genesis、baseline current-best
  或修改 replay。

## 5. 实施结果与严格验收

Phase 1A 已按本计划完成，实际交付如下：

- 新增 task-neutral 的 `EvaluationSubject`、`EvaluationBindingV1`、
  `BaselineEvalJobSpec`、baseline eval backend capability 与 Runner 执行路径。
- baseline manifest、checkpoint、subject、完整 evaluation binding、sealed evidence、
  validation gate、trusted seal、summary 和 public result 已形成同一条绑定链。
- sealed checkpoint 的字节校验与进程输入绑定已收敛到 Runner 的同一个启动前入口，同时覆盖
  baseline/candidate eval 与 warm-start train。Runner 只复制一次结构化 `cli_argv`，要求 checkpoint
  path/hash 参数各恰好出现一次并与 spec 的 `execution_checkpoint()` 完全一致，然后启动被校验的
  同一份 argv；random-init train 也会拒绝隐藏的 warm-start 参数。子进程保留 hash 与
  `weights_only=True` 二次复核。
- raw eval summary 只接受扁平 `subject_kind/subject_id`；顶层与 metrics 的第二身份表示均拒绝，
  公共结构化 subject 只由 Runner 投影。
- Stair Phase 0 source manifest 保持任务自有；任务包 adapter 从 admitted bundle 和当前
  checkpoint 字节确定性派生 runtime `BaselineManifest`（schema v2）。runtime hash 绑定 source manifest、
  descriptor 和 Phase 0 report，调用者不再手工拼接第二套 provenance。
- 新增统一 composition provider/registry/resolver；push 的具体 defaults、plugin、memory、
  backend 和 Runner 构造已收敛到 push-owned provider。
- `run_loop.py`、`materialize_job.py`、`run_round.py` 已移除 push 分支；production registry
  目前只显式注册 push，stair 仍保持禁用。
- `run_loop.py` 已删除三个重复 resolve composition 的兼容 helper，生产与测试统一消费一次
  resolved snapshot。
- dry-run 与 `run_round.py` 现在也先解析同一个 immutable snapshot；push-owned audit 只替换
  replay provider、dry-run Runner 和 verifier，task plugin、memory adapter、backend、
  policy initialization 与 acceptance protocol 均来自该 snapshot，不再重建默认 push config。
- push provider 的 CLI/programmatic 输入先收敛为一个 task-owned、不可变的
  `PushRecoveryCompositionConfig`。`TaskPlugin`、backend、Runner 和 `task_defaults` snapshot 全部只从
  该对象派生，不再对原始 mapping 使用 `int()`/`str()` 容错；snapshot 与实际 runtime 逐字段一致。
  `PushRecoveryTaskConfig`、`PushRecoveryTaskPlugin` 与 `SubprocessRunner` 构造入口也分别严格拒绝
  boolean、字符串化整数、浮点整数和错误 config 类型。
- 训练预算四元组现在在 task-owned config/composition 形成时复用共享
  `build_budget_accounting()`，严格绑定
  `budget_steps == train_num_envs * train_steps_per_iteration * train_max_iterations`。完整 backend
  配置和 dry-run train summary 使用同一核算源；`run_loop.py` 的 live-only 重复公式及 dry-run eval
  的手写 pass 已删除。`run_loop` live、dry-run、`run_round.py` 与 `materialize_job.py` 均在 job/ledger
  生成前拒绝不一致预算。
- 相邻审计将 push 实际消费的 YAML runtime 控制量收敛为
  `public_runtime_dynamics.v2`：terrain、standing command、push evidence、success/behavior 阈值、
  metrics、policy action scale 和 runner controls 全部严格校验并参与 runtime hash。候选不应看到的
  curriculum control 只进入私有 binding，不进入 prompt 投影；seed policy、accept-count 和 candidate
  runtime adapter 同步删除隐式整数转换。
- sealed rollout 公共 validator 现在区分裸 64 位 lowercase digest 与
  `sha256:<64 位 lowercase digest>` checkpoint checksum；实际
  `eval_protocol_hash`、`evaluation_binding_hash`、`checkpoint_hash` 及对应
  `expected_*` 参数任一格式错误都会令 evidence incomplete、gate fail，不能通过自哈希或
  自我一致绕过。dry-run 合成 checkpoint 同步改为确定性的 canonical checksum。
- checkpoint 计划输出与可信字节引用已在类型层分开：`CheckpointOutputPlan` 只表达目标 URI，
  不携带伪摘要；`CheckpointRef` 只接受 canonical `sha256:<64 位 lowercase digest>`。
  candidate eval 只能在训练完成后，由 round orchestration 将 Runner 返回的 sealed
  `CheckpointRef` 绑定到执行 spec；未绑定计划、未 sealed checkpoint 或格式错误摘要都会在
  eval 子进程启动前失败。真实 subprocess Runner 与 dry-run audit Runner 都在自身入口重新
  调用 `execution_checkpoint()`，不只依赖上层调用顺序。
- checkpoint safety report、trusted eval seal、`EvidenceArtifact` 与私有 artifact reference
  统一复用 canonical checksum 契约；trusted seal 还会重新验证 eval checkpoint 的 role 和
  sealed 状态，不依赖调用方事先检查。
- 删除了只检查 metadata 却返回 `checkpoint_safe_to_load` 的旧 `safe_load_checkpoint()`；
  `build_checkpoint_safety_report()` 是唯一会读取实际字节并执行 `weights_only=True` 检查的安全加载
  证据入口，metadata/reference 校验不再伪装成字节安全结论。
- baseline manifest 可作为独立 eval anchor，不再被错误地等同于每轮候选训练 initialization；
  warm-start initialization 与 manifest 同时存在时必须连完整 runtime manifest hash 一致。
- `materialize_job.py --job-kind baseline-eval` 只物化 eval，不生成 train job，不构造 candidate，
  不经过 candidate verifier。
- 最终相邻审计补充关闭四条通用绕过：artifact identity 只接受有界 ASCII ID；
  eval summary 禁止同时携带 subject 与旧 candidate/baseline identity；所有 eval 数量字段严格
  拒绝 boolean 或非整数值；训练预算和 backend 的 legacy `trusted_seed_count` 别名复用同一个
  strict positive-int validator，不再通过 `int(...)` 接受字符串、浮点数或 boolean。

2026-07-27 最终复审后的验收结果：

1. checkpoint argv/字节绑定、strict integer、真实 subprocess 与 dry-run audit 专项为
   `80 passed, 1 skipped`，另有 `71 subtests passed`；重复或错误 checkpoint 的 candidate eval、
   baseline eval、warm-start train 以及 random-init train 隐藏参数都在 marker 子进程启动前拒绝。
2. 项目全量 `unittest`：共 561 项，`OK (skipped=4)`；补充全量 pytest 为
   `557 passed, 4 skipped`，另有 `411 subtests passed`，无 warning。旧记录中的测试工厂曾因
   `test_` 命名被 9 个模块误收集，故原始数量不可直接比较；当前结果不包含误收集项。
3. `pyproject.toml` 已固定 Ruff 的 `src`/`tests` 项目根；从项目目录和父仓目录对全部 72 个当前
   改动 Python 文件执行 `E9`/`F`/`I` 检查均通过，不再依赖调用工作目录；不声明已清理与
   Phase 1A 无关的历史全仓 import-order 债务。
4. `python -m compileall -q src scripts tests`：通过。
5. `git diff --check`：通过。
6. 三个通用入口的 `--help` smoke：全部通过。
7. task-neutral composition/evaluation/runner/backend/task base 扫描：没有 push/stair
   任务 ID、任务包导入、楼梯高度、失败原因或 task profile 语义。
8. 三个通用脚本扫描：没有 push/stair 分支或任务包导入；具体任务注册只存在于应用组合根，
   具体默认值与构造只存在于 task-owned provider。
9. 本轮从共享 checkpoint 安全源修复根因，因此旧 Phase 0 报告按 provenance 契约正确失效。
   已用当前工作树快照在 Isaac Lab 服务器重新执行 56-env GPU probe 并生成新报告：
   56/56 到达第一阶、56/56 `base_contact`、warm-start admission 为 true、current-best
   仍为 false；服务器 Phase 0 合同、adapter 与动态 recorder 专项 24 项通过；本地对应专项为
   `21 passed, 3 skipped`，另有 `21 subtests passed`，3 项跳过是因为本地无 Isaac/Torch runtime。
10. Push 旧任务、seed policy、runtime adapter、backend、composition、dry-run 与 materialize
    独立回归：`97 passed`，另有 `86 subtests passed`。

本阶段不执行 `StairProtocolV1` 的正式 baseline eval-only rollout，也不产生真实 stair baseline
score。本轮 GPU 执行仅用于共享冻结源码变化后的 Phase 0 兼容性证据重建。Phase 1A 的边界只包含
共享 eval-only 能力；生产 stair provider、`StairProtocolV1` 的运行时物化和 Session Genesis
分别属于 Phase 2/4 与 Phase 1B。Phase 0 报告仍不是 acceptance evidence，当前代码也不会把官方
checkpoint 声明为 current-best。

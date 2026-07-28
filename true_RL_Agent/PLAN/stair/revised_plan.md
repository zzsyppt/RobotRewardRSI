# 楼梯任务奖励自进化循环修订实施计划

## 0. 文档目的与结论

本文修订 `stair_climbing_task_plan.md` 在 Phase1B 之后的实施方向。目标循环严格定义为：

```text
官方 G1 楼梯完整奖励源码 R0
    -> 固定协议下策略随机初始化，从零训练
    -> 固定协议评估，获得七桶表现与闭合失败反馈
    -> LLM 读取完整 R0 与公开反馈，生成完整奖励源码 R1
    -> 使用相同初始化合同重新随机初始化，从零训练
    -> 在相同评估合同下比较 R1 与 R0
    -> 仅当 R1 被接受，下一轮才继承 R1 源码并生成 R2
```

关键结论：

1. Phase1B 已建立的 composition、Genesis、current-best、ledger replay、runner 预检、trusted seal 和 baseline 能力继续保留，不做大范围重写。
2. 不把 R0 伪装成 candidate，不创建 sequence-0 acceptance record，也不改变候选 ledger 的轮次含义。
3. 新增窄边界 `ReferenceProgramGenesisAnchor`：R0 完成从零训练和评估后，原子创建 session Genesis，并派生初始 current-best。
4. 在正式 R0 之前，必须先完成生产级 Stair 任务链和 auto-reset 前的完整奖励替换运行时。当前 Phase0 证据不能代替这两项。
5. R0 不使用候选的 delta/noise acceptance 规则，但必须通过任务专属、事先冻结的 `reference_capability_gate`，证明从 R0 源码训练出的策略确实具备最低上楼能力。
6. 首版只开放“完整奖励源码替换”。课程、动作/观测、终止、命令等能力以后通过版本化 capability profile 逐项开放，未实现的权限不得进入 prompt。

---

## 1. 当前系统事实与修订原因

### 1.1 已完成且必须复用的基础设施

Phase1A/1B 已经形成以下可复用边界：

- task composition 单次解析并形成不可变 snapshot；
- task-owned config 和共享 backend/runner 的严格类型边界；
- checkpoint planned reference、已验证摘要和字节预检分离；
- structured argv 与 job spec 在 subprocess 启动前绑定；
- baseline eval-only、trusted seal、private evidence 和 public result；
- Session Genesis、current-best、candidate ledger replay 和 resume；
- candidate source materialization、静态 verifier、训练预算和评估绑定；
- Push Recovery 旧任务的 additive reward 运行路径。

本计划只扩展这些边界无法表达的 `reference program` 和 Stair 完整奖励语义，不另建平行系统。

### 1.2 当前缺失的生产 Stair 链

生产 registry 当前只有 Push Recovery。Stair 目录中的 Phase0 terrain、probe、report 和 baseline adapter 主要用于可行性证明，尚不等价于以下生产组件：

- `StairClimbingTaskConfig` 与严格 task defaults；
- `StairClimbingTaskPlugin`；
- `StairClimbingCompositionProvider`；
- 正式 train/eval profile 与 entrypoint；
- 七高度桶采样、逐桶统计和宏平均；
- auto-reset 前 episode recorder；
- primary terminal reason 闭合和允许重叠的 diagnostic flags；
- private seed/bin resolver；
- Stair prompt、memory、smoke adapter；
- Stair candidate capability surface。

因此不能从当前状态直接执行 R0 正式训练。上述生产链必须位于 pilot 和正式 R0 之前。

### 1.3 当前 Push reward wrapper 不能作为 Stair 完整奖励权威

Push wrapper 在底层 `env.step()` 返回后替换返回 reward。该模式适合现有 additive 扩展，但不适合完整奖励替换：

- 终止环境可能已经 auto-reset；
- Isaac Lab 内部 episode reward accumulator 和日志已消费旧 reward；
- curriculum 或 termination 统计可能已读取旧值；
- PPO 返回值与内部证据可能形成两个事实源。

Stair 必须在 auto-reset 前、RewardManager 的权威计算位置产生最终 reward，且 PPO、episode accumulator、训练日志和 runtime evidence 只能消费这一份结果。Push 运行时不迁移、不退化。

### 1.4 当前 candidate ABI 不能直接用于完整奖励

现有环境观测辅助函数可能直接返回环境 tensor，静态 verifier 也未禁止 `zero_()` 等原地修改。完整奖励候选若得到这些引用，可以污染训练状态。

Stair 完整奖励 ABI 必须提供 detached clone，并同时建立静态禁止、运行时输入完整性和 RNG 隔离三层约束。

### 1.5 当前作业身份无法表达 reference

当前 subject 只允许 baseline/candidate，train job 仍以 candidate identity 为中心。R0 既不是外部 checkpoint baseline，也不是 LLM candidate，不能只写入 metadata。

需要一次窄的共享身份合同升级，使 train/eval/summary/seal/checkpoint lineage 能结构化表达 `reference`；candidate ledger 和 acceptance 语义保持不变。

---

## 2. 最终架构

### 2.1 唯一源码布局与完整替换语义

所有 R0/R1/R2 奖励程序使用同一 canonical 布局：

```text
training_code/components.py
entrypoint: training_code.components:build_training_components
```

不得同时支持 `reward_program.py` 或第二个 entrypoint。

约束如下：

- R0 是任务维护者提交的完整源码；
- R1+ 是 LLM 每轮输出的完整 `components.py`；
- 新源码不 import、wrap 或隐式调用上一代 reward；
- Stair 环境的其他基础 reward terms 必须禁用；
- runtime 只执行当前 source head 构造出的完整 reward program；
- accepted 只改变下一轮 prompt 的源码 head，不把 accepted checkpoint 作为下一轮 warm start。

源码来源与 session 角色拆成两个正交概念：

- `authorship_origin`：`official_upstream | llm_generated | human_authored`，写入源码 manifest，表示不可变来源；
- `session_role`：`reference | candidate`，由该 session 的 Genesis/job identity 赋予，不回写源码来源。

`current_head` 不是第三种静态角色，而是由 Genesis 加 candidate ledger 重放得到的状态投影。这样，上一 session 的 accepted LLM 源码可在新协议下作为新 reference，`authorship_origin` 仍保持 `llm_generated`，不会被错误标成官方源码。

### 2.2 ReferenceProgramManifest

新增 task-neutral 的不可变 `ReferenceProgramManifest`，只描述可信源码，不包含 Stair 指标语义。至少绑定：

- schema/version；
- task ID；
- canonical source path 和 entrypoint；
- source SHA-256 与 manifest SHA-256；
- `authorship_origin`；
- upstream revision/provenance；
- candidate ABI/profile hash；
- UTF-8 source byte count。

Stair 的具体 reward 输入、七桶和能力门仍由 task-owned protocol/profile 解释。

现有 `CandidatePackageRef` 和 candidate materializer 继续用于 R1+。两者共享相同 canonical 源码布局和 source hash 算法，但不强行合并成新的通用 artifact map。

### 2.3 版本化 capability profile

任务组合结果必须复用现有 capability registry/profile 机制，绑定一个 task-owned、不可变的 capability profile hash，不另建第二套权限系统。

Stair V2 首版只公开：

- 读取冻结的 `StairRewardInputsV2`；
- 返回每环境 scalar reward；
- 完整替换 reward program。

七桶进度、成功、失败原因和公开 diagnostics 全部由 task-owned recorder/evaluator 从环境 transition 独立计算。候选即使提供辅助分解项，也只能作为不可信训练调试量，不能进入 acceptance、seal 或公开失败反馈。

首版不公开：

- curriculum；
- observation/action space；
- termination；
- command generator；
- terrain generation；
- runner/backend/evaluation protocol。

以后开放课程等权限时，必须先完成 compiler、verifier、materializer、runtime evidence 和 replay 绑定，再发布新的 capability profile 与 task protocol；旧 session 不原地升级。

### 2.4 TrainingBinding：公平比较的唯一训练合同

R0 和所有候选都绑定同一种 `TrainingBinding`：

- `initialization_mode = random`；
- 相同的 policy architecture hash；
- 相同的 initialization seed schedule；
- 相同 optimizer/training profile hash；
- 相同的实际 policy/PPO combined agent configuration hash；
- 相同训练预算；
- 不携带 warm-start checkpoint；
- 相同的私有训练 seed resolver 版本。

每个作业运行前记录 `initial_policy_state_hash`。R0 与候选必须满足预先定义的 initialization pairing；不一致时在训练开始前拒绝。

其中 policy/PPO hash 不能由 `rsl_rl_ppo_mlp`、`strong_push` 等标签代替。task-owned canonical 配置必须由 structured argv 传入，Runner 在 subprocess 启动前绑定，训练进程应用到实际 `agent_cfg` 后再次回显 canonical hash；seed 参数必须显式提供且与 seed schedule 绑定。

训练随机性与 reward program 的执行环境隔离。候选不得读写 Python、NumPy、Torch 全局 RNG。

### 2.5 最小 Genesis 扩展

新增 `ReferenceProgramGenesisAnchor`，与现有 bootstrap/baseline anchor 并列。它包含：

- `ReferenceProgramManifest`；
- `TrainingBinding`；
- R0 train job/result identity；
- 训练预算核算和 runtime evidence hash；
- R0 checkpoint 的已验证摘要与 lineage；
- R0 public evaluation result；
- trusted eval seal 与 private artifact reference；
- task protocol、evaluation binding 和 capability profile hash；
- `reference_capability_gate` 结果。

创建流程：

1. 在 durable session state 之外的专用 Genesis staging 区执行 R0 train+eval；
2. 完成 checkpoint、summary、seal、协议和能力门验证；
3. 将验证通过的私有产物固化为不可变、内容寻址 artifact；
4. 以 create-once 原子写入 `session_genesis.json` 作为 session 唯一提交点；
5. 从 anchor 派生 `VerifiedSessionGenesis.initial_current_best`；
6. Genesis 成功后才允许 provider 生成第一个 candidate。

若 R0 训练、评估或能力门失败：

- 不产生有效 Genesis；
- 不产生 candidate ledger 记录；
- 不允许进入 provider；
- 该失败 staging 不作为 durable state 或可恢复 session，必须清理/隔离；修改源码/协议后使用新 session。

候选 ledger 仍从第一轮 candidate 开始。现有 round 计数、resume、replay 和 acceptance record 基本不变；不引入 sequence 0。

### 2.6 唯一结构化 subject 身份

将当前 subject 合同升级为唯一的 `ExperimentSubjectV2`：

```text
kind = baseline | candidate | reference
id   = canonical subject id
```

共享层只理解 kind/id，不解释 Stair。迁移范围严格限定为：

- `TrainJobSpec` 和 `EvalJobSpec`；
- backend job materialization 与 structured argv；
- subprocess summary identity validator；
- `TrainJobResult`、checkpoint output plan 和 checkpoint lineage；
- public result、trusted eval seal 和 evidence manifest；
- Genesis/current-best 的初始构造。

现有 `BaselineEvalJobSpec` 保持 eval-only 专用合同，但也使用同一个 structured subject；不得在 baseline 专用字段中再保存另一份身份。

禁止同时保留一个可独立变化的 `candidate_id` 和一个 subject。兼容读取如确有必要，只能是从 subject 派生的只读 property，不能序列化第二份身份。

规则：

- candidate 才进入 candidate verifier、mechanism budget 和 acceptance ledger；
- reference 进入可信源码 manifest、固定预算、runtime integrity 和 reference capability gate；
- baseline 保持现有 checkpoint eval-only 语义；
- evaluation 始终只加载已训练 checkpoint，不加载 R0/R1 源码；eval profile 禁用训练 reward program，评分只来自 task-owned metric recorder。

### 2.7 Stair task-owned pre-reset reward authority

Stair 环境新增 task-owned RewardManager integration，在 auto-reset 前完成一次权威 reward 计算：

1. 从当前 transition 的 pre-reset 状态构造 `StairRewardInputsV2`；
2. 调用当前完整 reward program；
3. 校验 shape、dtype、device、finite、per-step bound 和输入完整性；
4. 将同一 reward tensor 提交给 PPO、episode accumulator、训练日志和 runtime evidence；
5. termination recorder 使用同一 transition snapshot 完成进度、成功和失败原因记录；
6. 此后才允许 reset。

禁止：

- `env.step()` 返回后再覆盖 reward；
- 内部统计继续累计官方 base reward；
- 为日志重新调用一次 candidate reward；
- 从 reset 后观测重建终止 transition；
- Stair 特殊逻辑进入共享 runner/evaluator。

公开统计来自 task-owned evaluator 的闭合投影；private evidence 可保留逐 term mask 和逐 episode/bin 细节。

### 2.8 完整奖励 ABI 的不污染合同

`StairRewardInputsV2` 的每个 tensor 都必须是 detached clone，不得暴露环境、action manager 或 recorder 的可变引用。

静态 verifier 至少拒绝：

- tensor 原地方法及名称以 `_` 结尾的可疑变异调用；
- 对输入属性、subscript 或容器的赋值/删除；
- `__setitem__` 等显式变异入口；
- Python/NumPy/Torch 随机 API；
- 可变 module global、跨调用 counter/cache；
- 文件、网络、进程、动态 import/exec 等现有禁区。

runtime 每次调用还必须：

- 调用前后核对输入 tensor 指纹；
- 核对 Python/NumPy/Torch RNG state 未变化；
- 对相同 cloned input 重复调用得到相同输出；
- 拒绝非有限值、错误 shape/device/dtype 和超出协议边界的结果。

clone 开销须做基准测试，但性能压力不得成为回退到共享可变引用的理由。

### 2.9 评估、反馈、接受与源码继承

R0 评估通过后形成初始 current-best：

- `CurrentBestEvalStats.mode` 新增明确的 `reference` 状态，不能伪装成 `accepted`；
- current-best 绑定 R0 source/manifest hash、checkpoint hash、training binding、task/eval protocol hash；
- 这不是 candidate acceptance，也不写 acceptance ledger；
- R0 不要求相对自身的 `min_delta`，但必须通过 reference capability gate。

每轮 candidate 流程：

1. prompt 读取 current-head 的完整 `components.py`；
2. prompt 读取上一轮公开七桶表现、primary failure breakdown、diagnostics 和明确限制；
3. LLM 输出完整新 `components.py`；
4. verifier/materializer 固定源码身份；
5. 按同一 `TrainingBinding` 从零训练；
6. 在相同评估协议下与 current-best 比较；
7. accepted 时同时更新 current-best checkpoint 身份和 current-head source hash；
8. denied 时保留原 current-head，下一轮 feedback 记录拒绝原因。

accepted checkpoint 只作为评估证据和 current-best 统计锚，不作为后续训练初始化。

ledger replay 继续从 Genesis initial current-best 开始，顺序应用 candidate decision records；replay 必须同时恢复 current-best 统计和 accepted source head。

### 2.10 Prompt/source 大小硬边界

正式实施前必须对真实 R0 源码做机器可复现的 preflight：

- 需要完整内联的源码 UTF-8 大小不超过 24 KiB；
- 完整 authoring prompt 不超过 256 KiB；
- provider 的 8192 output-token 上限能够容纳完整 `components.py`；
- source request/read 总量符合现有 256 KiB 边界。

验收必须使用真实 R0 和实际 prompt builder 测量，不允许用估算代替。若不满足，先版本化调整 source packaging 或限额；不得静默截断、摘要替代完整 current source，或让 provider 输出残缺文件。

### 2.11 共享层与任务层边界

允许进入共享层的仅是：

- `ExperimentSubjectV2`；
- `ReferenceProgramManifest` 和 `ReferenceProgramGenesisAnchor` 的通用身份/完整性结构；
- `TrainingBinding`；
- 通用 checkpoint、job、seal、ledger 和 source hash 绑定；
- capability profile 的 opaque ID/hash。

必须留在 `tasks/stair_climbing/` 的包括：

- 楼梯 terrain、spawn、goal 和七高度桶；
- success、fall、base contact、slip、clearance 等定义；
- `StairRewardInputsV2`；
- pre-reset reward/termination recorder integration；
- reference capability gate；
- task prompt/memory/smoke adapter；
- Stair train/eval profile 和 protocol loader。

Push Recovery 保留 additive reward 和现有任务语义，不迁移到 Stair 运行时。

---

## 3. StairProtocolV2 冻结策略

Phase0/V1 descriptor、protocol、report 和官方 checkpoint 证据保持不可变，只作为可行性与 schema 来源，不重新解释为 R0 scratch-training 结果。

V2 采用“两次冻结、禁止事后放宽”的流程：

### 3.1 Pilot 前冻结 draft V2

在任何 R0 convergence pilot 前，draft V2 必须固定：

- 七个高度桶和采样映射；
- success、primary terminal reason、diagnostic flags 与优先级；
- episode horizon/timeout；
- 统计单位：先按 seed 计算七桶等权宏平均，再跨 seed 形成 mean/std/SE；
- reference capability predicate 和精确数值阈值；
- 允许测试的训练预算档位及最大 pilot 上限；
- `StairRewardInputsV2` 和 capability profile；
- formal evaluation 与 private seed resolver 的版本。

reference capability gate 必须体现“确实有些上楼能力”，例如预先指定低阶桶非零成功或明确的多阶进度门。最终采用哪一个谓词和数值必须写入 draft V2，pilot 后不得为了让 R0 通过而降低。

### 3.2 Pilot 后冻结 final V2

pilot 只用于从预先允许的预算档位中选择最小可行训练预算，并验证 runtime 性能/收敛。pilot seed、结果和 checkpoint：

- 与 formal R0 的 private seed/binding 分离；
- 不进入 Genesis；
- 不成为 current-best；
- 不进入 prompt 或 acceptance；
- 不允许改变 reference capability floor。

选择预算后冻结 final V2。正式 R0、所有 candidate 和 replay 只接受 final V2 hash。

若最大允许 pilot 预算仍不能通过已冻结能力门，结论是 R0/runtime/任务设计尚不可用；停止进入正式 session，不下调门槛掩盖问题。

---

## 4. 分阶段实施计划

以下工作量是 Phase0/1A/1B 已通过后的条件估算，不含长时间 GPU 排队。

### 阶段 A：窄化共享合同与 draft V2

**实施状态：已完成（2026-07-27）。详细设计、实际变更与验收记录见 `phase_a_implementation_plan.md`。**

**目标**

让系统能够表达 reference program 和完整奖励权限，同时不改变 candidate ledger/acceptance 主干。

**工作**

- 引入 `ExperimentSubjectV2`，消除可独立变化的 candidate identity 二源；
- 引入 `ReferenceProgramManifest`、`TrainingBinding`、`ReferenceProgramGenesisAnchor`；
- 将 `TaskSpec.requires_baseline_genesis` 收敛为单一 `genesis_anchor_kind = bootstrap | policy_baseline | reference_program`，迁移现有 Push/测试；
- composition 对不同 anchor kind 做互斥验证；
- 扩展 current-best/source projection，使 `mode=reference`、program source hash 和 training binding hash 可经 Genesis/ledger 唯一恢复；
- task-owned capability profile 进入 composition snapshot 和 job/eval binding；
- 编写 draft `StairProtocolV2`，冻结能力门和 pilot 预算集合；
- 为现有 Phase1B Genesis/ledger 文件提供明确的 schema migration 或 fail-closed 版本拒绝，不做隐式猜测。

**严格验收**

- reference 合同可贯穿已有可信 eval artifact、Genesis、current-best、source projection 和 replay；任一 identity/hash 不一致都在相应信任边界拒绝；
- 本阶段不验收 reference train/eval argv 执行链；该链在阶段 E 实现 materializer 后统一验收，禁止用手工 spec 代替；
- serialized job 中不存在第二份可变化的 `candidate_id`；
- Push bootstrap、现有 baseline eval-only、Genesis replay 测试全部不退化；
- candidate ledger 首记录仍是第一轮 candidate，round/replay 语义未改变；
- R0 初始 best 的 mode 为 `reference`；第一条 accepted candidate 才把 mode 推进为 `accepted`；
- Stair 特殊字段没有进入共享 schema；共享层只保存 opaque task/profile/protocol hash；
- draft V2 在 pilot 前已有精确 reference capability threshold，且 hash 可重放。

**预计工作量：2–3 人日。**

### 阶段 B：生产 Stair 任务链

**目标**

补齐当前缺失的生产任务插件，使 Stair 可通过正式 composition、backend 和 runner 物化 train/eval job。

**工作**

- 实现严格 `StairClimbingTaskConfig`；
- 实现 `StairClimbingTaskPlugin` 和 `StairClimbingCompositionProvider`；
- 实现正式 train/eval profile、entrypoint 与 Isaac Lab env builder；
- 实现 private seed/bin resolver、七桶均衡映射和公开宏平均；
- 实现 auto-reset 前 termination transition recorder；
- 实现互斥 primary terminal reason、重叠 diagnostic flags 和闭合投影；
- 实现 Stair prompt/memory/smoke adapter；
- 让同一个 provider 实例可通过显式 staging/integration registry 解析，但暂不加入生产默认 registry；禁止复制 task defaults 或另建 staging provider。

**严格验收**

- Stair composition 只 resolve 一次，snapshot、runtime config、argv 和 protocol hash 完全一致；
- train/eval budget、seed/env/bin 数关系在物化前严格验证；
- 每个 episode 恰好归入 success 或一个 primary failure，计数总和等于 episode 总数；
- timeout 与 contact 并发、auto-reset 终态、done/mask 不一致均有动态行为测试；
- 七桶在每个规定 seed 下覆盖完整，统计单位符合 V2；
- private seed/bin 映射不进入 prompt/public result；
- public failure/diagnostic evidence 只能由 task recorder 产生，candidate 返回的任何辅助量都不能满足 pass gate；
- Push 的物化、dry-run、run-round 和 live-loop 行为保持一致；
- 阶段 D–F 的真实作业使用同一个 production provider/config，仅由显式 staging registry 选择；生产默认 registry 仍只启用已验收任务，Stair 此阶段不可被误启动。

**实施状态（2026-07-28）**

- 已完成 task-owned config/plugin/provider、显式 staging registry、专用 train/eval entrypoint、private seed/bin resolver、七桶宏平均、pre-reset recorder、episode 闭合分类以及 prompt/memory/public projection；
- 服务器 Isaac Lab 2.3.0 真实跑通 3/5/7 split seed，共闭合 168/280/392 个 episode，classification warning 为 0；
- 官方 Phase 0 checkpoint 的成功率仍为 0，sealed gate 正确 fail；该运行只验收 Phase B 执行链，不进入 R0、Genesis 或 acceptance；
- complete-reward runtime 仍按阶段边界 fail closed，留给阶段 C；默认生产 registry 仍只有 Push。

**预计工作量：3–5 人日。**

### 阶段 C：pre-reset 完整奖励运行时、ABI 与 R0 等价性

**目标**

先建立真实完整替换执行面，再谈 R0 收敛。

**工作**

- 在 Stair RewardManager 权威位置接入完整 reward dispatcher；
- 禁用 Stair 其他 base reward terms；
- 确保 reward、episode accumulator、日志和 runtime evidence 共享同一次计算结果；
- 实现 `StairRewardInputsV2` detached-clone builder；
- 扩展 candidate verifier 的变异、随机和可变全局状态禁区；
- 实现输入/RNG 指纹、重复调用确定性和输出合同检查；
- 将官方 R0 转写为 canonical `training_code/components.py`；
- 对同一 transition 批次比较官方 reward 与 R0 program，完成逐项/总值数值等价性测试；
- 对 clone、dispatcher 和 recorder 做 1024-env 性能基准。

**严格验收**

- done transition 使用 auto-reset 前状态，跨阶进度和终止原因不丢失；
- PPO reward、episode reward sum、训练日志和 runtime evidence 对同一 episode 完全一致；
- 注入 sentinel base reward 后，上述所有出口均不受其影响；
- 每个 step 只调用一次当前 reward program；
- `inp.base_lin_vel.zero_()`、输入 subscript 写入、全局 counter、Python/NumPy/Torch RNG 故障注入全部在训练前或首次调用时拒绝；
- 候选调用前后环境 tensor、action buffer 和 RNG state 不变；
- 相同输入重复调用输出相同；
- R0 与官方 reward 在冻结容差内等价，差异报告为零或有被批准的显式迁移说明；
- 性能回归有测量值和门限，训练配置不启用 Phase0 probe recorder；
- Push additive wrapper 未被改变。

**预计工作量：4–6 人日。**

### 阶段 D：R0 scratch convergence pilot 与 final V2

**目标**

在真实运行时上证明 R0 从随机初始化可以在合理预算内达到预先冻结的最低能力，并选择正式固定预算。

**工作**

- 运行最小 smoke：少量 env/steps 验证数值、吞吐、终止与日志；
- 按 draft V2 允许的预算档位运行 R0-only pilot；
- 检查训练稳定性、七桶进度、低阶能力、失败分布和 seed 方差；
- 从通过能力门的档位中选最小合理固定预算；
- 冻结 final `StairProtocolV2`、training profile、evaluation binding 和 capability profile hash；
- 使用真实 R0/prompt builder 做 24 KiB、256 KiB、8192 token preflight。

**严格验收**

- pilot 只在阶段 C runtime 上执行，不存在 mock/post-step reward 路径；
- 至少一个允许预算档位通过预先冻结的 reference capability gate；
- 不因 pilot 结果降低能力门或改变成功/失败定义；
- final V2 的预算等式、七桶映射、seed 数、episode 数、timeout 和 acceptance 参数完整绑定到统一 hash；
- pilot 与 formal seed/binding 可机器判定为不同命名空间；
- 真实完整 R0 和 prompt/output 均满足大小边界；
- 若任一项失败，阶段停止，不进入正式 Genesis。

**预计工作量：2–3 人日，加 GPU pilot 时间。**

### 阶段 E：正式 R0 与 ReferenceProgramGenesisAnchor

**目标**

在 final V2 下完成唯一一次正式 R0 从零训练/评估，并原子建立可重放 session 起点。

**工作**

- 通过 runner 物化 reference train job，subject.kind=`reference`；
- 使用正式 private training binding 从随机初始化训练；
- 校验预算、initial policy hash、checkpoint 字节和 runtime evidence；
- 通过正式 eval job 完成七桶 public/trusted evaluation；
- 运行 reference capability gate；
- 原子发布 `ReferenceProgramGenesisAnchor` 和初始 current-best；
- prompt source catalog 将 R0 注册为 current-head/reference anchor。

**严格验收**

- R0 train/eval 经过与 candidate 相同的 runner、checkpoint 和 evaluation 信任边界；
- reference identity 不藏在 metadata，所有 summary/seal/lineage 一致；
- eval 子进程只接收 checkpoint，不接收 reward source；
- eval profile 不实例化训练 reward program，七桶 score/failure/diagnostics 仅来自 task-owned metric recorder；
- R0 通过完整性门和 reference capability gate；
- Genesis 绑定 source、training、checkpoint、task/eval protocol、seal 和 private artifact hash；
- 私有 artifact 可先内容寻址固化，但 create-once Genesis 是唯一 session 提交点；中途失败没有半有效或可恢复 session；
- candidate ledger 仍为空，initial current-best 可从 Genesis 单独恢复；
- 重新加载/replay 得到完全相同的 R0 current-best/source head；
- R0 失败时 provider 不被调用。

**预计工作量：2–3 人日，加正式训练评估时间。**

### 阶段 F：R1/R2 完整奖励闭环

**目标**

完成图片所示的真实两轮最小闭环，证明只继承 accepted 源码。

**工作**

- prompt 同时提供完整 current-head 源码和上一轮公开失败反馈；
- provider 输出完整 R1；
- 通过 verifier/materializer 和同一 TrainingBinding 从零训练 R1；
- 使用 final V2 比较 R1 与 R0；
- 分别覆盖 accepted 和 denied 路径；
- accepted 后以 R1 完整源码生成 R2，denied 后仍以 R0/current-head 生成下一轮；
- replay candidate ledger 并恢复 source head/current-best。

**严格验收**

- R0、R1 的 policy architecture、initialization schedule、预算和评估 binding 完全一致；
- 训练 argv 不含 warm-start checkpoint；
- R1 源码不 import/call R0，runtime 不含 base reward；
- acceptance 使用冻结的 noise-aware 参数和 required gates；
- accepted 记录同时绑定 candidate source hash 与 checkpoint/eval 证据；
- denied 记录不能改变 current-best 或 current-head；
- 下一轮 prompt 的源码 hash 与 replay 恢复的 source head 相同；
- source lineage 无环、父节点存在且只由 accepted 决策推进；
- prompt 只暴露已实现的 complete-reward capability，不出现 curriculum 等未来权限。

**预计工作量：3–4 人日，加候选训练评估时间。**

### 阶段 G：全量回归与生产启用

**目标**

确认新任务没有侵入共享语义或造成旧任务退化，再显式开启 Stair。

**工作**

- 运行共享、Push、Stair、Genesis/replay、runner safety 全量测试；
- 运行 malformed identity/hash/argv、source mutation、RNG、auto-reset、统计二源故障注入；
- 审计 task-specific 名称是否泄漏到共享包；
- 更新实施记录和机器生成的 protocol/hash 证据；
- 最后一步才将 Stair provider 注册到生产 composition registry。

**严格验收**

- 现有 Push unit/integration/dry-run/materialize/live-loop 测试全部通过；
- baseline eval-only 与 warm-start 安全入口行为不变；
- Stair 只有在 reference Genesis 完整时才可进入 provider/live loop；
- 修改任何 source/training/protocol/evaluation/checkpoint/seal hash，replay 或启动边界均 fail closed；
- shared 层不包含 stair height、success/failure、reward input 名称；
- Ruff、compileall、pytest、git diff --check 全部干净，无 warning/误收集测试；
- 本地最小真实 smoke 能从 R0 Genesis 进入 candidate round；
- 默认 task 仍为 Push，显式选择 Stair 才进入新任务。

**预计工作量：2–3 人日。**

---

## 5. 总体验收矩阵

| 目标 | 必须证明的证据 |
|---|---|
| 循环语义正确 | R0 从零训练；R1 读取完整 R0+反馈后完整重写；R1 再从零训练；只有 accepted 源码成为下一轮 head |
| R0 有实际能力 | formal R0 通过预先冻结、独立于 candidate acceptance 的 reference capability gate |
| 比较公平 | R0/R1 共享 TrainingBinding，initialization pairing、预算、架构和 eval protocol 一致 |
| reward 唯一 | pre-reset 单次计算；PPO、accumulator、日志、evidence 同源；base sentinel 无影响 |
| 状态不污染 | detached clone 输入；变异/RNG/全局状态被拒绝；调用前后环境和 RNG 不变 |
| 身份唯一 | reference/candidate/baseline 通过单一 structured subject 贯穿 job、summary、lineage、seal/result |
| Genesis 可重放 | R0 成功后原子创建 anchor；无 sequence 0；ledger 从第一 candidate 开始；source head 可恢复 |
| 评估可信 | 七桶、失败闭合、seed/bin、统计单位和 seal 统一绑定 final V2 hash |
| 任务边界清晰 | Stair 语义只在 task-owned 包；共享层只承载通用身份、hash 和执行合同 |
| 旧任务不退化 | Push additive runtime、baseline/warm-start、dry-run/materialize/live-loop 全回归通过 |
| 未来可扩展 | capability profile 版本化；课程等权限未实现前不出现在 prompt；升级使用新协议/session |

---

## 6. 明确不做的事情

- 不把 R0 写成伪 candidate acceptance record；
- 不引入 ledger sequence 0；
- 不因 reference 而重写现有 candidate round/replay 主干；
- 不让 accepted checkpoint 成为下一轮训练 warm start；
- 不在 Stair 使用 Push 的 post-step reward wrapper；
- 不把 Stair 七桶、失败原因或 reward inputs 放入共享层；
- 不提前开放 curriculum 等尚无完整编译/验证/证据链的权限；
- 不将 Phase0 checkpoint 的零成功报告当作 R0 scratch-training 能力证据；
- 不用 metadata、文档 hash 或重复字段代替结构化身份与运行时绑定；
- 不通过放宽 capability floor、截断源码或省略反馈来让流程表面通过。

---

## 7. 工作量与开工条件

预计总工作量：**18–27 人日，加 GPU pilot 与正式训练评估时间**。

该估算成立的前提：

- 官方 G1/H1 Rough Locomotion 环境和 R0 reward 可在目标 Isaac Lab 版本运行；
- Phase0 资产、terrain 和机器人 schema 继续有效；
- Phase1A/1B 的共享合同回归保持通过；
- 目标服务器具备满足预算 pilot 的 GPU 时间。

若阶段 C 的 R0 数值等价性或阶段 D 的预冻结能力门无法通过，应回到 reward/runtime/任务定义定位原因，不继续堆叠 Genesis、ledger 或 acceptance 补丁。

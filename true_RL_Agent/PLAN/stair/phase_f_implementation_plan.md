# Stair Phase F 实施计划：R1/R2 完整奖励闭环

## 0. 文档状态

- 编写日期：2026-07-31。
- 规划起始基座：`42d495c107023e14ff9697b6188e74b4b40f1968`；实施收口代码提交：`4e0af4b8c6bd9d59c62590a2b3a8352c31622157`。
- 2026-08-03 后续审计更正：正式三轮 ledger 为 v2，只保存 path-only prompt 引用，没有不可变 prompt checksum/size 和逐 attempt authoring evidence。两次 2000-iteration 训练、评估、checkpoint、decision 与 R0 current-best 结果仍是有效科学执行证据，但不能证明 provider 当时看到的精确 prompt，也不能满足本计划第 8 节第 6 项的完整 authoring lineage。因此 Phase F 的“完整闭环完成”状态撤回；R0 Genesis 无需重建，正式 v2 ledger 不迁移、不回填，后续须在新 ledger v5 下重新累计至少两条完整候选评估。v5 同时逐记录绑定独立 ledger epoch；历史 Push v4 + 完整 v2 evidence 只读兼容，不作为新 Stair 重跑格式。
- Phase D 冻结基座：`51911dc542416757b03995934b1ea37787261240` 之后由操作者完成并推送的新 Phase D。
- 固定训练预算：`4096 envs × 24 steps/env × 2000 iterations = 196,608,000 env-steps`。
- Phase F 只在新 Phase D 和已完成的 Phase E 基础设施上继续开发，不重新设计或调参 Phase D。
- 本文是 Phase F 的详细实施合同。实现必须先通过第 2 节的 Phase E 开工门。

---

## 1. 阶段目标

实现并验证以下真实闭环：

```text
正式 R0 Genesis 中的完整奖励源码与 current-best
  -> 构造一个有界、无二源的本轮 prompt snapshot
  -> LLM 输出完整奖励 R1
  -> 可变 draft 上的有界修复预检查
  -> runner-owned seal
  -> 对同一份 sealed artifact 执行权威 static verifier + task-owned qualification
  -> 与 R0 相同的 TrainingBinding、随机初始化种子和初始策略状态训练 2000 iterations
  -> Final StairProtocolV2 固定评估
  -> accepted 才推进 current-best/current-head
  -> denied 只进入公开反馈，不推进源码头
  -> replay 唯一恢复 current-best/current-head
  -> 用恢复出的 current-head 生成 R2
```

本阶段必须证明的不是“LLM 能改一个奖励权重”，而是：

1. 候选提供的是完整 `training_code/components.py`，运行时完整替换 R0 奖励。
2. R0、R1、R2 每次都从随机初始化重新训练，不继承策略 checkpoint。
3. 公平比较由同一 TrainingBinding、同一初始化计划、同一初始策略状态和同一 Final V2 评估域共同保证。
4. 只有 accepted 的完整源码成为下一轮父源码；denied 源码不能污染 current-head。
5. prompt 只开放当前已经真实生效的 `complete_reward_program`，课程等未来权限继续隐藏。

---

## 2. 开工硬门：Phase E 必须先原子完成

Phase F 代码可以先完成静态设计，但不得启动真实候选训练，直到正式 R0 满足以下全部条件：

1. 正式 R0 从随机初始化完成恰好 2000 iterations，训练作业退出码为 0。
2. 正式 checkpoint、train summary、15-seed eval summary、trusted eval seal 和 reference qualification 均通过字节与身份复核。
3. `session_genesis.json` 已通过 create-once 原子提交，candidate ledger 为空。
4. Genesis 中的 `ReferenceProgramGenesisAnchor` 完整绑定：
   - R0 源码及 checksum；
   - reference manifest/artifact hash；
   - TrainingBinding hash；
   - initial policy state hash；
   - checkpoint hash；
   - Final V2 task/eval binding；
   - trusted seal。
5. 独立 load-only/replay 不启动 train/eval/provider，恢复出的 current-best、current-head 与 Genesis 完全一致。
6. 正式 R0 失败时停止，不使用 Phase D pilot 或历史诊断 checkpoint 代替 Genesis。

Phase E 尚未通过时，本计划保持“已设计、未开工”状态。

---

## 3. 冻结边界与明确禁区

### 3.1 不修改的冻结事实

- `max_iterations=2000`，不搜索 500/1000/其他预算。
- `num_envs=4096`、`steps_per_env=24`、随机初始化和最终 iteration checkpoint 选择。
- Final `StairProtocolV2`、七桶定义、seed 数、成功/失败定义、acceptance 参数和 reference capability floor。
- Phase D 的完整奖励 ABI、pre-reset reward authority、termination capture、terrain/runtime dynamics 和 R0 数值实现。
- Phase D pilot artifact、manifest、trust anchor 和历史诊断结论。

若发现会让候选实际执行错误、证据失真、失败反馈不可行动、信任链可绕过或边界不合理的问题，应回到对应权威源修复并重跑受影响验收；不得因训练曲线、候选效果或实现便利修改上述事实。

### 3.2 共享层允许承载的内容

共享层只允许理解以下 task-neutral 合同：

- capability ID、profile hash 和 readiness 状态；
- subject identity、program checksum、artifact hash；
- TrainingBinding、expected/actual initial policy hash；
- checkpoint/evaluation binding；
- accepted/denied 决策和 candidate-owned runtime failure 分类；
- replay 恢复出的 current-best program checksum。

共享层禁止出现楼梯高度、七桶、脚尖碰撞、净空、top hold、楼梯 reward term 或任何 Stair 阈值。

### 3.3 Task-owned 内容

以下内容继续只存在于 `tasks/stair_climbing` 或 Stair 入口：

- complete reward 的真实 runtime readiness 证据；
- Stair `RewardInputs`、完整奖励 smoke fixture 和 pre-reset 注入；
- candidate reward qualification 的任务语义；
- Final V2 的评估投影、失败闭合和诊断；
- Stair staging composition 和正式 Genesis 定位。

### 3.4 Push 不退化

- 默认 production registry 仍只暴露 Push，Phase G 前不注册 Stair。
- Push 不获得 `complete_reward_program` 权限。
- Push 的 additive wrapper、bootstrap/baseline Genesis、CLI 默认值和现有 dry-run/live 行为不变。
- 新增的可选共享字段对 Push 必须为 `None`/空 readiness，并且不改变其 argv。

---

## 4. 当前代码差距

### 4.1 完整奖励权限没有进入真实 authoring 链

- Stair Final V2 capability profile 已选择 `complete_reward_program`。
- prompt capability registry 仍把该能力全局标记为隐藏/未生效，且该条目没有提供完整奖励所需的 `RewardTerm`/`TrainingComponents` authoring ABI。
- `CandidateAuthoringLoop` 已能接收 `runner_capability_readiness`，但 `run_live_loop()` 没有传入 task-owned readiness。
- Stair `StairClimbingComponentSmokeAdapter.build_component_smoke_fixture()` 当前明确返回 `None`；因此即使直接打开 prompt 开关，authoring 仍会跳过完整奖励 smoke。
- 当前共享 control-effectiveness 文本只描述 additive reward；若原样复用，会错误告诉 provider 完整奖励仍叠加 base reward。
- provider 的 changed-component 合同只认识 `reward_callable` 等执行类别；当前计划还没有明确 `complete_reward_program` capability ID 与 `reward_callable` 执行类别之间的唯一映射。
- Stair prompt metadata 中的 `runtime_available` 不能参与激活判断，否则会与 profile/readiness 形成第二个开关。

结果是协议声明、运行时能力和 prompt 暴露尚未形成一条链。

### 4.2 Stair 尚无 load-only live 入口

- Stair 仍由 staging registry 组合，production registry 只有 Push，这是正确的 Phase G 前边界。
- 通用 `run_loop.py` 的 live 路径会调用 `prepare_session_genesis()`；Phase F 必须明确要求加载既有正式 Genesis，不能在候选循环中意外重训 R0。
- 不能复制一套 Stair live loop。

### 4.3 初始策略状态 hash 没有传给 candidate train job

- Phase E Genesis 已保存 R0 的 `initial_policy_state_hash`。
- Stair training runtime 已支持在 PPO 开始前比较 `expected_initial_policy_state_hash`。
- candidate 的 `RoundInputs -> backend -> TrainJobSpec -> argv -> Stair train entrypoint` 当前没有传递该期望值。
- train result、checkpoint lineage、decision record 和 replay 当前也没有绑定 expected/actual 配对事实。

因此“相同 TrainingBinding”尚不能证明 R1 与 R0 的实际初始网络字节相同。

### 4.4 current-head 源码仍可能从可变路径重新读取

- replay 已能恢复唯一 `CurrentBestEvalStats.program_source_checksum`。
- source catalog 仍独立扫描最新 accepted record，并从 private artifact URI 下直接读取 `training_code/components.py`。
- 读取时没有把 sealed artifact identity、manifest、artifact hash 和 replay current-best checksum 作为一个联合条件重新验证。

这是 Phase F 必须消除的源码权威二源。

### 4.5 sealed artifact 尚未承载完整的候选准入证明

- authoring loop 当前先在 provider 可变目录上执行 verifier/smoke，随后 round 才 seal 并重新执行 static verifier。
- 可变目录上的检查适合生成 LLM repair feedback，但不能作为可信执行证据。
- Stair 训练入口虽会再次执行 task-owned qualification，当前 `VerifiedCandidateArtifactRef` 的证据并未证明 task-owned qualification 针对同一份 sealed bytes 通过。
- `run_loop.py` 还在 composition 外单独创建 `StaticCandidateVerifier()`；因此 composition 声明的 readiness 与实际执行 verifier 可能不是同一对象。
- `materialize_job.py` 也独立硬编码 `StaticCandidateVerifier()`；若只修 live loop，官方物化入口仍可绕过 Stair sealed qualification。

Phase F 必须让 composition 持有唯一 candidate admission policy，并使 backend/materializer 只接受由该 policy 对 sealed artifact 产生的执行能力。不得把“多跑一次 smoke”当作修复。

### 4.6 candidate 失败所有权尚未闭合

- verifier 和 authoring smoke 能过滤大多数错误。
- Stair 训练入口会在 simulator 启动前执行 candidate qualification，完整奖励也可能在 PPO 运行中失败。
- 当前 Stair execution 把异常收敛为通用 failed summary，Runner 又把任意 failed train summary 映射成 `CandidateRuntimeError`。

候选自身错误应形成 denied 反馈并允许下一轮继续，但 reference、身份、binding、artifact、checkpoint 或基础设施错误仍必须 fail closed。

---

## 5. 设计决策

### 5.1 复用现有 candidate ledger，不新增 source ledger

current-head 的唯一权威是：

1. 无 accepted candidate 时：Genesis 的 reference source checksum。
2. 有 accepted candidate 时：`replay_acceptance_ledger()` 恢复的 current-best program checksum。

源码 resolver 只负责根据这个权威 checksum 找到并验证不可变字节，不再自行决定“谁是最新 best”。不引入 sequence-0 ledger、source ledger 或第二套 current-best 状态机。

### 5.2 每轮 prompt 是一个不可变快照

参考 `D:\humanoid_agent\codex` 的 session/turn 组织原则，每轮 provider 输入从一次已验证的 session snapshot 派生：

- Genesis identity；
- replay current-best/current-head；
- 有界公开历史和失败反馈；
- task capability profile 与当前 readiness；
- 本轮 source payload。

同一次 provider attempt 的 prompt payload 落盘后不再被后续文件变化重解释。这里只借鉴“单快照构造一次 turn”的设计逻辑，不移植 Codex 的通用会话实现。

### 5.3 readiness 与 capability profile 取交集

将现有命名过窄的 `runner_capability_readiness` 收敛为 task-neutral 的 `authoring_capability_readiness`，不保留第二套兼容 alias。它属于一次解析出的 `ResolvedTaskComposition` 不可变快照，而不是由 `TaskPlugin` 单独声称 backend 状态。

readiness 不保存一组可被手工伪造的平行布尔开关。composition 从实际注入的 task profile、candidate admission policy、materializer/backend、runtime profile 和 task-owned qualification/smoke 所声明的 capability ID 集合计算一次交集，并冻结 `profile_hash + effective_capability_ids + admission_policy_id`。诊断信息可由这些实际对象派生，但不参与第二次激活。Push composition 的有效集合默认为空。

prompt 中能力必须同时满足：

1. capability ID 在共享 registry 中已知；
2. capability ID 被 task-owned profile 选择；
3. capability ID 位于 composition 已冻结的 effective intersection；
4. candidate admission policy、materializer、runtime 和 task-owned qualification/smoke 四个执行面均声明同一能力有效；
5. readiness 绑定的 capability profile hash 与当前 task spec 相同。

共享 registry 为 `complete_reward_program` 定义 task-neutral 的完整替换 authoring ABI 和 activation mode；它与 runner-owned curriculum schema preview 使用不同的 activation mode，不能把完整奖励误投影成 `candidate_writable=false` 的 preview。唯一执行映射冻结为：`complete_reward_program` 是 provider-visible capability ID，candidate manifest/组件派生的 changed component 仍为既有 `reward_callable`。不新增 `complete_reward_enabled` 等布尔值，也不让 task prompt metadata 的 `runtime_available` 参与授权。Stair composition 只激活该能力；curriculum 等未实现能力不进入 effective intersection。共享层不硬编码 Stair。

`complete_reward_program` 生效后的公开合同必须明确：

- provider 输出完整 `training_code/components.py`；
- `build_training_components()` 返回完整 `TrainingComponents`；
- 允许构造完整 `RewardTerm` 列表；
- 所有 weighted terms 的和就是唯一训练 reward；
- 不存在 `training_reward + base_reward` 或调用 R0 的路径；
- verifier 派生的 change class 仍为 `reward_callable`。

### 5.4 initial policy hash 是独立比较锚，不进入 TrainingBinding

`initial_policy_state_hash` 是依据实际初始化后网络状态计算的运行时事实，不放入静态 TrainingBinding，避免循环依赖。

增加可选 task-neutral `expected_initial_policy_state_hash`，其唯一来源是已验证 Genesis reference anchor，并沿以下链传递：

```text
VerifiedSessionGenesis
  -> run_live_loop / RoundInputs
  -> BackendAdapter.materialize_jobs
  -> TrainJobSpec
  -> runner-owned unique argv binding
  -> Stair train entrypoint
  -> execute_stair_training pre-PPO comparison
  -> train summary + TrainJobResult + checkpoint lineage
  -> trusted eval evidence + decision record
  -> ledger replay against Genesis expected hash
```

Push 没有该锚时保持 `None`，argv 中必须完全不存在对应参数。

replay 新增显式 `expected_initial_policy_state_hash` 输入，唯一来自已验证 Genesis。它只要求已经进入训练的记录携带 expected/actual/match；provider/verifier 等 pre-train denial 不伪造 actual hash。任何 evaluated/accepted 记录缺失或不匹配都使 continuation fail closed。

### 5.5 sealed artifact 的唯一准入生命周期

候选生命周期冻结为两类检查：

1. provider 可变 draft 上的 static/smoke 只是有界 authoring repair 预检查，不产生执行能力，也不进入可信 lineage。
2. round 将最终 draft 原子 seal 后，由 composition 持有的唯一 candidate admission policy 对同一份 sealed bytes 执行 static verifier 和 task-owned qualification；验证前后重算 artifact；通过后生成唯一 `VerifiedCandidateArtifactRef`，并把 admission policy ID/hash 纳入其证据。

`ResolvedTaskComposition` 必须持有实际使用的 candidate admission policy；应用入口不再额外构造 verifier。backend/materializer 校验该 artifact 的 admission policy 与 composition/backend 冻结值一致。Stair qualification 只理解 Stair reward ABI 和 fixture；共享 seal/验证管线只理解 task-neutral pass/fail、policy ID/hash 和 artifact identity。

Stair admission policy 必须复用 Phase D 已有的 `qualify_stair_candidate_reward_program()`：该函数已经只接受 `SealedCandidateArtifactRef`，并在读取前后重验同一 artifact。Phase F 只在其外层增加 task-owned synthetic smoke 与 admission evidence 绑定，不重写该函数、`QualifiedStairRewardProgram`、pre-reset reward authority 或 R0 数值实现。组合 evidence 只包含确定性的 schema/policy/fixture/term/identity hash 和 pass 状态，不包含耗时、绝对路径或进程环境，保证父子进程可重算同一 hash。

不得为此改写既有 Phase E Genesis 或提高其 runtime snapshot schema。第一个通过 sealed admission 并进入训练的 candidate record 建立本 session 的 `admission_policy_id/hash`；之后每个训练记录都必须相同，replay 在 continuation 前强制复核。这样既保持正式 Genesis 精确字节不变，也防止多轮中途切换准入策略。

对 Stair candidate，`VerifiedCandidateArtifactRef.verification_policy/hash` 还要结构化进入 TrainJobSpec 和唯一 argv。训练子进程在 AppLauncher 前对同一 artifact 重跑同一 admission policy，并要求 ID/hash 与 spec 完全一致；成功 summary、checkpoint lineage 和 decision record 回报该配对。Push 没有 complete-reward admission 要求时不新增对应 argv，保持现有执行字节合同。

### 5.6 candidate-owned failure 与可信基础设施失败严格分离

只允许以下失败转成 denied：

- subject.kind 为 `candidate`；
- 已通过 sealed artifact/identity/binding 校验；
- task-owned 候选代码调用边界抛出明确的 candidate-owned typed failure；
- 子进程写出完整、canonical、identity-bound 的失败 summary；
- summary 的 `failure_owner=candidate`、`failure_stage` 和 task-owned `reason_code` 均通过闭合合同。

以下情况继续中断 session：

- reference 资格失败；
- summary 缺失、格式错误或身份不匹配；
- artifact/source checksum 改变；
- TrainingBinding、initial hash、预算、argv 或 checkpoint 不一致；
- simulator/CUDA/进程/文件系统等基础设施失败。

Stair task runtime 只在“加载/资格检查/调用候选完整奖励”这几个窄边界把已知错误标记为 candidate-owned；外层未分类异常默认是 infrastructure。共享 Runner 只验证和消费通用 `failure_owner/failure_stage`，不解释楼梯 reason code；subject 为 reference 时即使同类 reward 错误也必须中断 session。

### 5.7 Phase F 只提供 staging 入口

共享应用驱动允许显式注入 composition registry 和 Genesis 策略。新增薄 Stair-owned 入口：

- 注入 Stair staging registry；
- 强制 `task_id=unitree_stair_climbing`；
- 强制 existing-Genesis/load-only；
- 复用同一个 `run_live_loop()`、provider、Runner 和 replay；
- 不复制参数解析、作业物化、接受逻辑或 ledger。

生产 registry 启用留给 Phase G。

---

## 6. 分步实施

### F0：完成 Phase E gate 并冻结输入

**工作**

- 等待正式 R0 训练、15-seed eval、qualification 和 Genesis create 完成。
- 执行独立 load-only/replay，记录 Genesis hash、R0 source checksum、TrainingBinding hash、initial policy hash 和 current-best。
- 将这些值作为 Phase F 动态测试输入，不复制进 Python 常量。

**严格验收**

- 第 2 节所有条件全部通过。
- load-only 前后 Genesis 精确字节、mtime、ledger 状态不变。
- load-only 没有 train/eval/provider 子进程。

### F1：接通 capability readiness 与 Stair load-only 入口

**工作**

- 在 `ResolvedTaskComposition` 增加唯一 `candidate_admission_policy`，以及由实际执行面求交后冻结的 task-neutral `authoring_capability_readiness`；Push 使用现有 static admission policy 且有效能力集合为空。
- 将 shared prompt/authoring 链中的 `runner_capability_readiness` 一次性迁移为 `authoring_capability_readiness`，不保留平行参数。
- Stair composition 从实际注入的 admission policy、materializer、完整替换 runtime 和 task-owned qualification/smoke 合同求交，构造唯一 `complete_reward_program` readiness；应用入口不得再单独创建 `StaticCandidateVerifier()`。
- `run_loop.py`、`materialize_job.py` 和其他官方 candidate 物化入口统一消费 composition 的同一个 admission policy，不保留脚本级 verifier 构造。
- `run_live_loop()` 将 composition readiness 传给现有 `CandidateAuthoringLoop`。
- 扩展 `complete_reward_program` registry 条目，使其在 ready 时公开完整 `RewardTerm`/`TrainingComponents` ABI、完整替换 control-effectiveness，以及唯一 `changed_component=reward_callable` 映射；不得复用 curriculum 的 schema-preview 投影，也不修改 provider changed-component 枚举。
- 实现 Stair task-owned authoring smoke fixture；shape/dtype 从 canonical `complete_reward_runtime.input_schema` 派生，不复制第二份字段表；在服务器依赖环境中用非退化的小批量 tensor 覆盖 R0 使用的三维索引、`where/exp/min/max/norm` 等真实操作。fixture 构建保持 lazy，不让 Push 或无 Isaac/PyTorch 的普通 import 路径新增硬依赖。
- 现有共享 smoke 输出摘要只认识 `SmokeTensor`/Python 数值；为其增加 task-neutral、lazy-import 的真实 tensor shape/finite 投影，不复制 Stair 字段、shape 或阈值，也不把 PyTorch 变成普通 import 路径的硬依赖。
- 同一 fixture/policy 同时支持可变 draft 的 repair 预检查和 sealed artifact 的权威 qualification；后者复用 `qualify_stair_candidate_reward_program()`，只有 sealed static verification + qualification + smoke 的组合证据进入 `VerifiedCandidateArtifactRef.verification_hash` 和执行 lineage。
- Stair backend 将该 policy ID/hash 结构化绑定进 candidate TrainJobSpec/argv；train preflight 重算并比较。删除、重复、替换参数或伪造 summary 均 fail closed，且发生在 AppLauncher 前。
- capability prompt 构建改为 registry/profile/readiness 三方交集，删除 `complete_reward_program` 的全局硬禁用二源和 additive reward 文本；删除或降级 Stair prompt metadata 中 `runtime_available` 这类授权型字段，使其不能参与激活。
- 参数化通用应用入口以接受显式 registry 和 Genesis mode；新增薄 Stair staging live 入口并强制 load-only。

**严格验收**

- Stair 第一轮 prompt 只出现一个可写且 effective 的 `complete_reward_program`。
- provider-visible component/control contract 明确是完整替换，包含构造 R0 和候选所需的 API，不出现 additive base reward 公式。
- prompt 不出现 `reward_weight_scale`、curriculum、environment mutation 或其他未实现权限。
- Push prompt 与当前快照字节等价，仍不暴露 complete reward。
- 官方 R0 源码在 mutable draft 和 sealed artifact 两条检查上均通过；错误 shape、非有限输出、原地修改输入、随机调用或超时的完整奖励在训练物化前失败，mutable 预检查可进入有界 repair feedback，但只有 sealed 检查产生可信拒绝证据。
- readiness 缺项、未知 capability、profile hash 不匹配或 verifier/materializer/runtime/smoke 任一执行面未就绪时 fail closed，且 provider 不被调用。
- 用故障注入证明更换 application verifier、伪造 readiness 布尔值、绕过 sealed qualification 或将错误 admission policy artifact 交给 backend 均在物化前失败。
- 用故障注入证明父进程通过后再替换 policy/hash，或让子进程重算结果不一致，均在 AppLauncher 前失败且不产生 checkpoint。
- Stair `materialize_job.py` 对仅通过 generic static verifier、但未通过 Stair qualification/smoke 的候选必须失败；Push 仍使用既有 static policy 并保持输出合同。
- 正式 Phase E Genesis 在 Phase F 代码下仍可精确 load-only；首个候选记录后篡改 admission policy ID/hash 会被 replay 拒绝，且 Genesis 不被迁移或重写。
- Stair live 入口在 Genesis 缺失时启动前失败；存在时不调用 reference materializer/train/eval。

### F2：绑定 R0/R1/R2 的实际初始策略状态

**工作**

- 为 `RoundInputs`、backend materialization 和 `TrainJobSpec` 增加可选 expected initial policy hash。
- 从 Verified Genesis 的 reference anchor 提供唯一值；调用方不能从 CLI 另行覆盖。
- backend 结构化生成唯一 `--expected-initial-policy-state-hash`；Runner 在 subprocess 启动前校验恰好一次且与 spec 相同。
- Stair train parser 和 preflight 接收该值，在 PPO 前比较实际 hash，并在 train summary 中记录 expected/actual/match。
- Runner 验证成功 summary 的三个字段，并将配对证据写入 `TrainJobResult`、sealed checkpoint metadata 和候选 trusted eval lineage；不匹配视为可信执行合同失败，不转成普通 candidate denial。
- decision record 对已经进入训练的记录保存 expected/actual/match；`replay_acceptance_ledger()` 接收 Genesis expected hash 并逐条复核。pre-train denial 明确没有 actual hash，不能伪造为已训练证据。

**严格验收**

- R0 Genesis、R1 和 R2 的 actual initial policy hash 完全相同。
- R1/R2 argv 均为 `--init random`，不存在 warm-start checkpoint 参数。
- 删除参数、重复参数、替换 hash、伪造 summary/checkpoint metadata/decision record 或实际初始化不一致均在训练开始前、summary 信任边界或 replay continuation 前拒绝。
- evaluated/accepted ledger 若缺 expected/actual/match，或其值不等于 Genesis anchor，replay 必须拒绝恢复；provider/verifier denial 则不得被要求携带不存在的 actual hash。
- Push spec 为 `None` 时 argv 无新增参数，现有 Push train materialization 精确不变。

### F3：让 replay current-best 成为唯一 source-head 权威

**工作**

- `run_live_loop()` 把本轮 `replay.current_best` 显式传给 context/source resolver；`ProjectedPublicHistory` 和 source catalog 不再自行扫描记录决定 best。
- reference head 只从已验证 Genesis 内嵌源码和 hash 构造。
- accepted candidate head 必须重新验证 runner-owned sealed artifact，并联合匹配：
  - accepted candidate subject ID；
  - subject manifest hash；
  - candidate artifact hash；
  - admission/verification policy ID 与 evidence hash；
  - program source checksum；
  - canonical `training_code/components.py` 路径和字节。
- current-head 与 latest-failed 分开解析；latest-failed 只用于反馈，不能成为父源码。
- 跨轮恢复时，latest-failed 完整源码只允许来自可重新校验的 runner-owned sealed artifact；只有公开失败反馈但没有不可变源码 artifact 时，保留反馈并省略源码，不回读可变 provider 目录。
- 明确区分两种摘要：执行/lineage 身份使用原始 canonical bytes 的 `program_source_checksum`；prompt payload 使用独立的 `content_sha256`。命名分别固定为 `current_head_program_source_checksum` 与 `current_head_prompt_content_sha256`，两者不得互相替代或直接比较。
- 本轮 prompt 保存上述两种摘要；runner-owned `parent_program_source_checksum`、decision/lineage 只能记录本轮开始时的 replay head 原始源码 checksum。该 parent 关系由 prompt snapshot/Runner 写入，不新增 provider 可声明或可覆盖的 manifest 字段。
- 不新增独立 head 文件或可写 alias。

**严格验收**

- 无 ledger 时 current-head 精确等于 R0 Genesis source。
- accepted R1 后 replay head 精确等于 R1；下一轮 R2 prompt 的父 hash 等于 R1。
- denied R1 后 replay head 仍为 R0；下一轮 prompt 可包含 R1 失败反馈，但父源码仍为 R0。
- 篡改 artifact 文件、URI、manifest、artifact hash、source hash 或伪造 accepted 顺序均在 prompt 构建前失败。
- 把 prompt `content_sha256` 冒充 program checksum，或把 raw checksum 冒充 sanitized projection hash，均被 schema/关系校验拒绝。
- accepted lineage 无环，且每个非 Genesis 节点的父节点在此前已是 current-head。
- inline source 继续满足 24 KiB，source read 不超过 256 KiB，完整 prompt/output 继续满足既有上限。

### F4：收敛 candidate 失败所有权语义

**工作**

- 在 task-owned reward load/qualification/call 边界引入窄的 typed candidate failure；Stair train 入口和 PPO runtime 只捕获该类型并写出 canonical failure summary。其他未分类异常默认属于 infrastructure。
- failure summary 绑定完整 job identity、artifact/source hash、TrainingBinding、expected initial hash、execution device、预算，以及闭合的 `failure_owner/failure_stage/reason_code`；不包含私有路径或异常堆栈。
- SubprocessRunner 先验证完整 summary 与 process completion，再仅在 `failure_owner=candidate`、subject 为 candidate 且 stage/reason 合法时映射为 `CandidateRuntimeError`；`failure_owner=infrastructure` 或未知值一律 `RunnerExecutionError`。
- 复用 `run_candidate_round()` 现有 candidate runtime denial 路径写 ledger 和公开反馈。
- reference/infrastructure/binding 错误继续抛出 `RunnerExecutionError`，不伪装为可继续候选失败。

**严格验收**

- 动态 fake entrypoint 证明 candidate 资格失败时 AppLauncher 未实例化、无 checkpoint、产生一条 denied ledger，下一轮可继续；另覆盖 PPO 中候选 reward 调用失败的同类路径。
- 同一故障放在 reference subject 上必须中断且不写 candidate denial。
- CUDA/OOM、checkpoint、initial hash、文件系统和 simulator 故障不得被归为 candidate；缺 summary、重复键、非法 hash、身份不匹配、未知 owner/stage/reason code 或伪造 candidate failure 均 fail closed。
- 公共 prompt 只看到有界 reason code/修复反馈，不看到服务器路径、stderr、源码外秘密或原始异常。

### F5：闭合 R1/R2 authoring、训练、评估与接受

**工作**

- 用正式 Genesis 启动 Stair staging live loop。
- 第一轮 prompt 提供完整 R0 current-head、Final V2 公开反馈结构、完整奖励 ABI 和唯一有效 capability。
- provider 输出完整 R1；mutable draft 上的 verifier/smoke 只提供 repair feedback，最终选中 draft 只 seal 一次，并在 sealed bytes 上执行权威 static verifier + Stair qualification，生成唯一 verified artifact。
- R1 按固定 2000 iterations 从零训练并在 Final V2 下评估。
- acceptance 只消费冻结的 noise-aware comparison、required gates 和 trusted evidence。
- 第二轮从 replay 结果构造 prompt：accepted 时父源码为 R1，denied 时仍为 R0/current-head；provider 输出完整 R2。
- R2 使用相同训练/评估合同执行。

**严格验收**

- provider 输出为完整 `training_code/components.py`，不 import/call R0，不存在 additive base reward 路径。
- candidate 只声明并实际激活 `reward_callable`/complete reward 机制；runtime evidence 证明 PPO、episode accumulator 和训练日志消费同一 pre-reset reward。
- R0 与所有完成训练评估的候选，其 TrainingBinding、initial policy hash、预算、Final V2 binding 和执行设备一致。
- evaluated accepted/denied 决策绑定源码、artifact/admission policy、initial policy pairing、checkpoint、public eval、trusted seal 和 Genesis；pre-eval denial 只绑定其实际到达阶段的权威证据，不伪造 checkpoint/eval。
- accepted 才推进 current-best/current-head；denied 不推进。
- provider 可以按冻结的 authoring 上限执行有记录的静态 repair attempts；每轮至多一个 selected verified artifact 进入训练，且该 artifact 的 train/eval 各只有一次权威执行，不存在隐藏训练重跑或换源码重试。

### F6：确定性双路径测试与真实服务器门

**确定性测试**

- 使用 fake provider/runner 构造 R1 accepted -> R2 parent=R1。
- 使用 fake provider/runner 构造 R1 denied -> R2 parent=R0。
- 重启进程后 replay 两条 ledger，恢复结果与未重启时一致。
- 覆盖 verifier failure、candidate qualification failure、train infrastructure failure、eval incomplete 和 sealed gate failure 的不同权限语义。

**真实服务器测试**

- 先执行不启动 PPO 的 materialization/preflight，检查 prompt、artifact、argv 和 Genesis load-only。
- 再执行最小真实闭环，直到累计至少两个候选完成 train、public eval 和 sealed validation；每个成功训练候选仍使用固定 2000 iterations，不允许用缩短预算的结果作为 Phase F 验收。
- provider/static/candidate-runtime 的受控拒绝形成独立闭合轮并允许继续，但不计入上述两次完整评估，也不能触发同一源码的隐藏训练重跑。
- 持续记录每轮 prompt 字节/token、source payload 重复量、失败反馈大小、GPU 进程、checkpoint 和 eval 证据。
- 真实两轮不强制人为制造“一次 accept、一次 deny”；两条控制流由确定性测试完整覆盖，真实运行只验证系统行为与证据链。

**严格验收**

- 每个成功训练候选恰好 2000 iterations；受控 candidate-runtime rejection 必须记录真实停止阶段且不得伪装成训练完成。每个子进程结束后无残留 simulator/Python 进程，GPU 显存回收。
- prompt 信息足量但无整段历史重复；current-head 只有一份完整源码，旧失败源码按预算投影。
- provider 输出、训练 summary、eval summary、decision、ledger 和下一轮 prompt 可以由 hash 串联。
- 任一硬合同失败立即停止真实运行，不以手工改 ledger/Genesis/checkpoint 继续。

### F7：回归、审计与发布

**静态与自动化验收**

- 全量 unit/integration tests。
- 标准 unittest discover 与 pytest 入口均无误收集或 warning。
- Ruff `E9,F,I`、compileall、`git diff --check`。
- job materialization、Runner prelaunch、strict JSON、artifact immutability 和 replay 故障注入。

**架构审计**

- 检查是否出现第二套 current-head、TrainingBinding、initial hash、argv 或 capability truth。
- 检查共享层是否出现 Stair 特殊字段。
- 检查 Phase D/Final V2/2000 预算是否有任何非必要改动。
- 检查 Push production registry、默认 CLI、materialize、dry-run 和 live loop 是否无退化。
- 检查偏离主线且无调用者的兼容 helper/临时入口，确认无用后删除，不保留平行流程。

**发布条件**

- 所有严格验收通过后才提交并推送 Phase F。
- 本地 HEAD、origin 分支和服务器正式仓库 HEAD 一致。
- 服务器只同步 clean committed source，不同步本地临时包或历史诊断文件。
- Phase G 前仍不把 Stair 加入默认 production registry。

---

## 7. 预期修改面

以下是当前预计的最小修改面，实施时以实际调用关系为准：

### 7.1 共享 task-neutral 层

- `src/true_rl_agent/composition/base.py`：composition-owned candidate admission policy，以及从实际执行面求交后冻结的 authoring capability readiness snapshot。
- `src/true_rl_agent/candidates/artifact.py`、`candidates/verifier.py`、`candidates/authoring_checks.py`：区分 mutable repair 预检查与 sealed artifact 权威准入，绑定 admission policy ID/hash，并以 lazy、task-neutral 方式投影真实 tensor smoke 输出。
- `src/true_rl_agent/agent_core/live_loop.py`、`agent_core/round.py`：传 readiness、Genesis initial hash、replay current-best 和唯一 admission policy。
- `src/true_rl_agent/runner/specs.py`、`runner/base.py`：可选 expected/actual initial policy pairing 与 checkpoint lineage。
- `src/true_rl_agent/backends/base.py`、`backends/isaac_lab_rsl_rl.py`：结构化传递该 hash。
- `src/true_rl_agent/runner/job_argv.py`、`runner/subprocess_runner.py`：prelaunch/summary 绑定及可信 candidate failure 分类。
- `src/true_rl_agent/prompting/capabilities.py`、`prompting/source_catalog.py`：完整替换 ABI、profile/readiness 交集和 replay-owned source resolver。
- `src/true_rl_agent/memory/context_builder.py`、`agent_core/authoring_loop.py`：统一迁移 task-neutral readiness 命名和快照传递。
- `src/true_rl_agent/evaluation/decision_record_contract.py`、`evaluation/acceptance_replay.py`、可信 eval seal 相关合同：initial policy pairing 的记录与 continuation 校验。
- `scripts/run_loop.py`、`scripts/materialize_job.py`：统一消费 composition admission policy；`run_loop.py` 另窄化为可注入 registry/Genesis mode 的单一应用驱动。

### 7.2 Stair task-owned 层

- `src/true_rl_agent/tasks/stair_climbing/component_smoke.py` 及必要的 task-owned admission policy：真实 fixture、复用既有 `qualify_stair_candidate_reward_program()` 的 sealed qualification 和 capability 声明；不改 Phase D reward authority。
- `src/true_rl_agent/tasks/stair_climbing/composition.py`、`staging.py`：实际 admission policy/readiness 绑定和 load-only staging composition。
- `src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py`：删除授权型二源，只保留由共享 capability projection 决定的公开 ABI。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py`、完整奖励 runtime：窄 candidate failure 边界与 infrastructure 默认分类。
- `scripts/isaaclab_stair_rsl_rl_train.py`：initial hash 参数与 identity-bound failure summary。
- 新增一个薄 Stair live 入口；不得复制 `run_live_loop()`。

### 7.3 测试与计划

- 新增 Phase F 合同、source-head、initial hash、failure semantics 和两轮 replay 测试。
- 扩展 Push 回归和 SubprocessRunner 故障注入。
- 完成后在本文追加实际变更、命令、动态证据、审计结论和提交 hash。

---

## 8. 阶段完成定义

只有同时满足以下条件，Phase F 才可标记完成：

1. 正式 R0 Genesis 已原子建立并通过独立 load-only/replay。
2. Stair prompt 只开放真实有效的完整奖励权限。
3. 至少两个候选从与 R0 相同的实际初始策略状态、固定 2000 预算完成训练和完整评估，且 decision/replay 可复核该配对；此前的受控拒绝不计入该数量。
4. 完整奖励从 sealed admission 到 PPO、日志和 checkpoint lineage 只有一条源码/运行时权威链；mutable draft 检查不被冒充为可信证据。
5. accepted/denied 两条路径均由动态行为测试证明，replay 后 current-head 不漂移。
6. 至少完成一次真实服务器连续执行，累计两条完整评估记录，并能按 hash 串联所有闭合拒绝轮及 prompt、源码、训练、评估、决策和下一轮。
7. Phase D 冻结事实未被修改，Push 和默认 production registry 无退化。
8. 全量检查通过，完成最终架构/二源/故障注入审计，未遗留并行补丁路径。

Phase F 完成后再进入 Phase G 的生产注册与全量启用；15-round 正式自进化实验在所有后续 Phase 完成后执行。

---

## 9. 实施与验收记录（2026-08-02）

### 9.1 正式会话与公平比较锚

- 正式会话：`stair-r0-formal-aacaee2`，artifact root 为服务器 `/data/yuyangliu/true_rl_agent_phase_e/formal_r0_aacaee2`。
- Genesis 文件的精确字节摘要和 mtime 在三轮前后保持不变；权威值只保存在会话 artifact 中，不复制进计划。R0 current-best 为 `0.5464285714`。
- R0 与两个完成训练的候选，其 TrainingBinding 和 initial policy state hash 均逐项相等；三者都使用 `cuda:1` 和 `4096 × 24 × 2000 = 196,608,000` env-steps。精确绑定值由 Genesis、train summary 和 ledger 交叉校验。
- 所有候选均从随机初始化训练；candidate eval 只加载最终 checkpoint，不加载候选源码或 candidate package。

### 9.2 三条真实闭合轮

1. `round-001`：候选 `stair_climbing_v1` 在训练期触发完整奖励输出合同错误，以 `candidate_runtime_failed` 闭合；没有伪造 checkpoint/eval，也不计入完整评估数量。
2. `round-002`：候选 `stair_reward_v2` 完成 2000 iterations，源码与 `model_1999` checkpoint 的精确摘要在 train/eval/ledger 中一致；`public_accept=0.4678571429`，sealed gate 通过，但相对 R0 的 noise-aware delta 不足，因此 denied。
3. `round-003`：首个 provider draft 因 `historical_mechanism_repeat/parameter_and_rationale_only_delta` 在训练前拒绝；修复 draft `stair_reward_v4` 完成 2000 iterations，源码与 `model_1999` checkpoint 的精确摘要同样闭合；`public_accept=0.125`，sealed mean `0.0867346939`，holdout floor 未通过，因此 required-gate denied。

两条完整候选评估均覆盖七个高度桶、public dev/public accept/sealed validation，并与 Genesis 保持同一 evaluation binding。三轮均未推进 current-best/current-head，replay 后仍恢复 R0。

### 9.3 真实运行暴露并修复的问题

- `73d13de` 修复已有 ledger 的 Genesis load-only replay；`ac4e09d` 让合法 `candidate_runtime_failed` 轮在 eval audit 中成为闭合受控记录，同时保持“训练成功但缺 eval”失败。
- R3 暴露出历史 experiment memory 只保留源码顺序前八项、Stair prompt 不投影已验证 reward-term runtime evidence 的问题。`4e0af4b` 改为按公开贡献幅值选择有界分量，并只从 `runtime_activation_gate=pass`、完整训练且公开评估完成的记录投影 Stair runtime feedback。
- R3 源码错误假设扁平 `height_scan` 前 30 项对应前方射线。根因是 Stair prompt ABI 没有明确说明索引映射未公开；修复已进入 task-owned `reward_inputs_contract`，从第一轮起禁止给固定切片赋予前后左右语义。该修复只重建 Phase D prompt preflight report；Final V2、训练/评估 binding、Genesis 与固定预算均未改变。

### 9.4 Replay、回归与资源验收

- 更新后执行 `--rounds 3` load-only：七项 live audit 全部为 true，执行轮数为 0；Genesis 与 ledger 的 hash/mtime 均未改变，精确值由会话 artifact 持有。
- 本地 `pytest`：`790 passed, 16 skipped, 584 subtests passed`；标准 `unittest`：`755 passed, 16 skipped`。Ruff `E9,F,I`、compileall 和 `git diff --check` 通过。
- 服务器定向 40 项回归通过；完整服务器回归为 `806 tests, OK, skipped=2`。
- 训练、评估和 replay 后无 Stair Python/Isaac 子进程；`cuda:1` 显存已回收，仅保留与本任务无关的既有进程。

**审计后结论：** Phase F 已证明两次候选训练/评估、受控拒绝、固定初始化公平比较和 decision 数值结果；但 v2 记录没有闭合精确 prompt 与逐 attempt authoring provenance，旧 load-only 的“七项全通过”不能继续作为完整 continuation 证据。当前实现必须对 v2 返回 `authoring_evidence_legacy_unverified`，不得以兼容豁免继续 provider loop。Phase F 只有在 ledger v5 下重新累计至少两条完整候选评估，并验证 epoch、prompt、attempt journal、candidate、训练、评估、decision 和下一轮 source head 后才重新完成。候选效果没有超过 R0 仍只是实验结果，不构成修改 Final V2 或 2000 iterations 预算的理由。

### 9.5 v5 authoring provenance 补充验收（2026-08-03）

- 正式 v5 artifact 位于服务器 `/data/yuyangliu/true_rl_agent_phase_f/formal_v5_0dbb214_20260803T084840Z`，复用原正式 R0 Genesis，并以独立 ledger epoch 保存新记录；旧 v2 ledger 字节未修改、未迁移、未回填。
- 新 epoch 累计两条完整候选训练/评估记录。第一轮经历一次 smoke 失败后由第二 attempt 修复并通过全部 authoring gate；第二轮一次 attempt 通过。每条 ledger 都绑定精确 provider prompt、全部 attempt、最终 selected prompt、candidate artifact、训练、checkpoint、public/sealed eval 与 decision。
- 两个候选都从与 R0 相同的初始策略状态随机初始化，使用相同 TrainingBinding 和 `4096 × 24 × 2000` 固定预算，没有 warm-start。两者 public accept 均为零且 sealed holdout floor 失败，因此均 denied；replay 后 R0 继续是 current-best/current-head。
- 后续同提交 load-only 执行零轮，Genesis、旧 ledger、epoch manifest 与新 ledger 的字节摘要保持不变；provider 未调用，source parent 仍为 R0。

**补充结论：** v2 authoring provenance 缺口已由新的 v5 正式重跑闭合，Phase F 重新满足阶段完成定义。旧 v2 科学结果继续只读保留，不被冒充为 v5 authoring evidence；候选效果不佳不改变 Final V2 或固定 2000 iterations 预算。

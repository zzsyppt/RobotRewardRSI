# 上楼梯任务接入计划（审计修订版）

- 修订日期：2026-07-27
- Phase 1B 实施基线：`81ea436`（`true-rl-agent-refactor`）
- 目标任务 ID：`unitree_stair_climbing`
- 当前状态：Phase 0、Phase 1A 与 Phase 1B 已完成；已有冻结协议、官方 G1 Rough baseline manifest、任务自有单向楼梯 probe overlay、task-neutral 组合与 baseline eval-only，以及不可变 Session Genesis、完整 baseline current-best、ledger replay/resume 和 prompt baseline anchor 闭环。仍没有生产 stair TaskPlugin、stair 训练命令或楼梯七桶正式 eval runtime；下一硬门为阶段 2。
- Phase 0 结论：官方 `Isaac-Velocity-Rough-G1-v0` checkpoint 通过共享安全加载、精确 runtime schema、terrain binding 和目标方向行为门，可作为定向训练的 warm-start 候选；它在 56 个首 episode 中成功数为 0，不能称为已会爬楼梯的 baseline，更不是 accepted current-best。官方金字塔楼梯不能直接表达“从底部向上爬”，因此目标环境使用楼梯任务包自有的 bottom-origin 单向楼梯 overlay。

## 一、审计结论

现有架构方向保持不变：楼梯语义应进入任务自有包，`AgentCore`、Runner、DecisionEngine 和 ledger 的候选轮次合同继续保持任务中立。

但原计划遗漏了两个必须先补的共享基础设施：

1. baseline 不能伪装成一个“已接受候选”。当前 accepted record 必须具备 candidate、verifier、train result 和 code-level surface，因此需要独立的 baseline eval-only 与 session genesis 合同。
2. 当前 `eval_protocol_hash` 只绑定版本、seed 数和 `eval_num_envs`，不足以阻止不同楼梯分桶、阈值或统计定义之间互相比较。

其余审计问题也确认存在：

- 成功定义包含跌倒，但失败分类没有闭合的互斥终止原因。
- 每轮 warm-start 起点没有明确冻结。
- `materialize_job.py` 与其他通用命名入口仍硬编码推倒恢复。
- `future_stair_climbing` 已在 Phase 0 资产中迁移为正式 `unitree_stair_climbing`；旧 ID 后续必须明确拒绝。
- `reward weight-only` 与新增 `RewardTerm` 的分类边界需要精确定义。

修订后的关键路径是：

`环境/基线预检 -> 完整 StairProtocolV1 -> 多任务组合与 baseline eval-only -> session genesis/current-best/replay -> 楼梯插件 -> 运行时 ABI -> Isaac train/eval -> 联调与启用`

## 二、先补齐的共享数据模型

### 2.1 BaselineEvalJobSpec

新增 task-neutral 的 baseline eval-only 作业能力：

- `BackendAdapter.materialize_baseline_eval(...) -> BaselineEvalJobSpec`
- `Runner.run_baseline_eval(...) -> PublicEvalResult`
- eval 公共身份统一为 `EvaluationSubject(kind="baseline"|"candidate", id=...)`；`TrainJobSpec` 仍保持 candidate-only。
- baseline identity 使用 `baseline_id`，不得填充伪造的 `candidate_id`。
- raw eval summary 只允许扁平 `subject_kind/subject_id`，不得再携带嵌套
  `evaluation_subject`、`subject` 或其他 candidate/baseline identity；公共结构化 subject
  只能由 Runner 从 job spec 投影。
- baseline checkpoint 保持 `warm_start` 角色；baseline eval 合同明确允许评估该角色，不把它伪装成候选 eval checkpoint。
- SubprocessRunner 可复用内部 eval 执行与安全校验，但 summary identity 必须区分 `subject_kind=baseline` 和 `subject_id=baseline_id`。
- SubprocessRunner 必须在 eval 或 warm-start train 子进程启动前读取 sealed checkpoint
  实际字节并核对 SHA-256；子进程内的 `weights_only=True` 安全加载只作为二次防线。
- Stair Phase 0 source manifest 由任务包内 adapter 派生为共享 runtime baseline manifest；
  runtime hash 必须绑定 source manifest、descriptor trust anchor 和 admitted report，二者
  是不同 schema 的单向身份链，不是两套可独立维护的 manifest。
- trusted eval seal builder 同时支持 baseline/candidate subject，并绑定 subject、task、checkpoint 与完整 EvaluationBinding。
- baseline eval 与候选 eval 都不得接收候选目录或加载候选代码。

### 2.2 SessionGenesisArtifact

采用独立的 `SessionGenesisArtifact`，不向候选 ledger 插入伪造的 acceptance record。

Genesis 必须由 runner-owned baseline eval 结果生成，至少绑定：

- session、task、backend 和 baseline ID；
- baseline manifest hash、checkpoint hash 和安全加载证据；
- `task_protocol_hash`、`eval_protocol_hash`、`acceptance_protocol_hash`；
- session 的 `evaluation_binding_hash`；
- baseline 的 public result hash、trusted eval seal hash、score stats、任务评估门和 sealed validation；
- 私有 baseline eval artifact ref/checksum；其 URI、原始 seed 和 rollout 不进入 prompt-public 投影；
- canonical `session_genesis_hash`。

规则：

- Genesis 中的 `initial_current_best` 必须由校验器从 baseline eval 证据推导，不能信任文件中自报的 current-best。
- Genesis 固定落盘为 session 根目录下的 `session_genesis.json`，采用原子、create-once 写入；创建后不得覆盖或就地更新。
- 恢复 session 时只能加载并重新验证已有 Genesis，不能静默重跑 baseline 后替换它；CLI、baseline 或协议不一致时必须新建 session。
- 校验成功后只向 live loop 暴露 `VerifiedSessionGenesis`，避免 Genesis 与另一个调用方提供的 current-best 成为双重真相。
- 每条候选 ledger record 都必须携带同一个 `session_genesis_hash`。
- 缺 Genesis、Genesis hash 不一致、baseline eval 不完整或协议不匹配时，live loop 和 replay 都 fail closed。
- baseline anchor 校验只要求可比较的评估证据、任务行为门、checkpoint 安全和 sealed validation；candidate verifier、训练预算门和运行激活门仍只属于候选接受路径。

### 2.3 CurrentBest 与 replay

将 replay 起点从 scalar 改为完整对象：

- `run_live_loop()` 接收单一 `VerifiedSessionGenesis`，并从中取得 `initial_current_best`、Genesis hash 和协议绑定。
- `replay_acceptance_ledger()` 接收由已验证 Genesis 派生的完整 `CurrentBestEvalStats` 和预期 Genesis hash；warm-start 楼梯不得回退到 scalar。
- 随机初始化任务使用显式 bootstrap Genesis 变体；是否允许该变体由 `TaskSpec` 声明，不能由调用方临时选择。
- `CurrentBestEvalStats` 增加 `anchor_kind`、`anchor_id`、`checkpoint_hash`、`task_protocol_hash` 和 `evaluation_binding_hash`。
- baseline 初始状态为 `mode="accepted"`、`anchor_kind="baseline"`；候选接受后切换为 `anchor_kind="candidate"`。
- replay 必须同时验证 Genesis、current-best、比较记录和每轮 ledger 的协议绑定。

### 2.4 Prompt/source catalog 的 baseline anchor

baseline 没有候选 `components.py`，因此 prompt 不得暗示存在可继承源码：

- 在尚无 accepted candidate 时，current-best source catalog item 使用 `role="baseline_policy_anchor"`、`status="baseline_anchor"`、`requestable=false`、无 source payload；catalog validator 必须显式接受该无源码状态。
- candidate lineage 公开 baseline ID、公开分数和协议绑定状态，不公开 checkpoint 本地路径或隐藏 seed。
- 现有 best rejected/bootstrap observed source 行为继续保留；它与 baseline current-best 是两个不同概念。
- 首个候选被接受后，current-best source 才切换为 accepted candidate source。

## 三、完整协议绑定

### 3.1 StairProtocolV1

阶段 0 必须产出 `configs/tasks/stair_protocol_v1.json`。该文件是 canonical JSON，生成 `task_protocol_hash`，且不允许出现 `null`、`TODO` 或依赖脚本默认值的字段。

必须冻结：

- 训练：terrain family、`0.05-0.20 m` 分布、`num_envs`、`steps_per_env`、iterations、总 budget、episode timeout、runner timeout、checkpoint selector。
- 评估：七个固定高度、各 split seed 数、每个 seed/每个高度的 episode 数、`eval_num_envs`、env 到 bin 的映射规则、seed policy 版本。
- 指标：成功条件、所有数值阈值、主分数公式、score stats 统计单位。
- 分类：互斥 primary reason、重叠 diagnostic flags 及优先级。
- 接受：`min_score_count`、`min_delta`、`absolute_accept_floor`、required gates。
- 候选边界：有效 `RewardInputs`、parameter-only 项和禁止项。

`StairProtocolV1` 是上述值的唯一来源：共享 `EvaluationProtocol` 和 `AcceptanceProtocol` 必须由其中对应 section 构造并交叉校验 hash，CLI、TaskSpec 与 backend config 不得再独立覆盖同名值。`public_accept_seed_count` 至少为 3，确保 noise-aware comparison 有真实的跨 seed 样本。

楼梯任务要求 baseline Genesis，因此缺 Genesis 时不得进入 bootstrap-floor 接受路径。

### 3.2 统计定义

首版固定采用以下统计方式：

1. 每个 seed 必须完整运行七个高度。
2. 每个高度先根据固定 episode 数计算成功率。
3. 一个 seed 的 score 是七个高度成功率的等权宏平均。
4. `score_stats.values` 是多个 seed 的宏平均分数。
5. `mean/std/standard_error` 在 seed 维度计算，`count` 等于该 split 的 seed 数。
6. public score 只绑定 `public_accept.score_stats.mean`。
7. `min_score_count` 必须等于 `public_accept` seed 数。
8. sealed validation 也必须完整覆盖七个高度；缺任一 seed、bin 或 episode 都不能 pass。

`public_dev` 可公开分桶级聚合诊断；`public_accept` 只公开 canonical score stats 和门状态；sealed validation 只公开门状态和绑定 hash。

### 3.3 EvaluationBindingV1

除静态 `task_protocol_hash` 外，每个 session 还要生成 runner-owned `EvaluationBindingV1`：

- `eval_protocol_hash`：共享 EvaluationProtocol 的 hash；
- `acceptance_protocol_hash`：从 StairProtocolV1 acceptance section 构造的共享 AcceptanceProtocol hash；
- `task_protocol_hash`：完整 StairProtocolV1 的 hash；
- `seed_cohort_hash`：各 split 隐藏 seed manifest、runner-owned 高熵 nonce 与 bin/env 分配的 canonical hash，原始 seed/nonce 不公开；
- `evaluation_binding_hash`：以上四者的组合 hash。

私有 seed manifest 必须在 baseline eval 前 create-once 落盘并由 Genesis 绑定；恢复 session 时复用同一份 manifest，不能重新抽样。公开 hash 只用于等价性检查，不作为 seed 内容的公开编码。

绑定位置：

- `TaskSpec` 暴露 task protocol ref；
- `TrainJobSpec`、train summary 和输出 checkpoint lineage 绑定 `task_protocol_hash`；
- `EvalJobSpec`/`BaselineEvalJobSpec` 绑定完整 EvaluationBinding；
- trusted eval seal、public result、CurrentBest、decision comparison 和 ledger 都保存相同 binding；
- replay 同时检查 candidate、current-best 和 Genesis 的 binding。

这样旧阈值、旧分桶、不同 seed cohort 或不同训练预算下的结果不能直接比较。

## 四、成功、失败与 warm-start 语义

### 4.1 成功与失败闭环

每个 episode 必须有且只有一个 `primary_terminal_reason`：

- `success`
- `base_contact`
- `fall`
- `base_pitch_instability`
- `slip`
- `insufficient_clearance`
- `timeout`

成功 episode 的 primary reason 固定为 `success`。失败 episode 按以下优先级选择第一个已触发原因：

`base_contact > fall > base_pitch_instability > slip > insufficient_clearance > timeout`

`toe_collision` 不进入 V1 主分类或候选 ABI。Phase 0 真实 body 探针只确认了左右 `ankle_roll_link` 接触 body，没有独立 toe body 或 contact-point 信号；在新协议版本证明可靠信号前，它保持 `reserved_not_admitted`。

同时保留允许重叠的 diagnostic flags。严格计数合同为：

`success_count + sum(primary_failure_counts) == total_episode_count`

该等式必须对每个 split、seed 和 bin 分别成立。无法分类的 episode 不能被丢弃，计数上回退为 `timeout` 并记录 classification warning；只要 warning 数大于 0，该次评估就不能成为 pass 或 acceptance evidence。每个 reason/flag 的精确传感器、body 集合、阈值和持续时间必须由 StairProtocolV1 冻结。

### 4.2 每轮固定从同一 baseline 开始

首版明确采用固定起点：

- 每个候选训练都从 Session Genesis 中同一个 sealed baseline checkpoint 开始。
- 每轮 `TrainJobSpec.training_start_checkpoint.checksum` 必须等于 Genesis 的 baseline checkpoint hash。
- accepted checkpoint 只用于 current-best 比较、公开 lineage 和最终产物，不作为下一轮训练起点。
- “从最新 accepted checkpoint 继续训练”属于后续独立协议版本，不能在 V1 中混入。

这样 reward 机制变化与初始化策略变化不会混在一起。

## 五、候选边界

- code-level：新增或修改真实 `RewardTerm` callable。
- diagnostic-only：`DiagnosticTerm`，只记录，不具备策略接受资格。
- parameter-only：
  - `ScheduleTerm(key="reward_weight_scale")`；
  - 只修改已有 term 的 scalar weight、楼梯范围或配置参数；
  - schedule-only 提案。
- 任意新增 `RewardTerm` 都由 verifier 推导为 `reward_callable`，不能因为它包含一个 weight 就称为 reward weight-only。
- 首版候选不得修改地形生成、分桶、seed、预算、PPO、checkpoint、成功定义、协议 hash、观测/动作空间或接受规则。
- 候选只接收低维、真实有效且有 provenance 的楼梯 `RewardInputs`；不暴露原始高度图和隐藏地形/评估信号。

## 六、分阶段实施

### 阶段 0：目标环境、baseline 与 StairProtocolV1 冻结（已完成）

**完成日期：2026-07-24**

**目标**

确认外部事实，在任何生产楼梯插件实现前冻结完整协议，并建立不污染共享层的任务自有 probe overlay。

**工作内容**

- 确认官方 source task、地形接口、脚部 body、接触传感器、进度、终点和终止信号。
- 七个高度逐一做 headless reset/step 探针。
- 校验 baseline checkpoint、SHA-256、来源、robot、观测/动作 schema 和 policy architecture。
- 通过短 benchmark 确定训练预算、eval 并行度、episode 数和 timeout。
- 提交最小 extension descriptor、无空字段的 `stair_protocol_v1.json`、baseline manifest、严格加载器、单向楼梯 overlay、任务自有 runtime 探针和绑定报告。

**严格验收标准**

- 七个高度均有恰好 8 个 env；每个 env 只统计首 episode，最多允许完整 `600 control steps = 12 s`。进度、跌倒和 top hold 必须使用 auto-reset 前的终止 transition；提前终止必须记录逐 manager-term mask 和按冻结优先级得到的主原因，不能用 reset 后状态或二分推断替代。
- Phase 0 的 task-specific prompt-effective 字段集合为空；计划字段均写明来源、shape、单位、设备、reset 语义和 Phase 3 激活门，不可靠字段不得提前进入有效 ABI。
- checkpoint 必须先通过共享 `build_checkpoint_safety_report()` 的 `weights_only=True` 安全入口；随后完成前向，文件身份、完整 body/joint 顺序、真实 observation/action manager schema 全部精确匹配。
- terrain grid origin、全部 env origin、bin 映射、spawn、上升方向和目标位置必须与冻结协议精确一致。
- `stair_extension_schema.json` 的 descriptor hash 是唯一不可变 V1 信任锚；protocol 与 manifest 校验自身结构、自哈希和安全关系，再由 bundle 校验 descriptor 中的 ref/id/hash 绑定。修改 protocol 或 manifest 后即使重算自哈希，也不能进入冻结 bundle；required gates、目标、spawn、timeout、generation mode 等安全关系仍须独立校验。
- 报告自身 hash、descriptor、task protocol、manifest、checkpoint、版本、runtime robot/observation/action/termination schema、几何和逐 env 首 episode 可由纯合同重新验证；派生摘要必须由逐 episode 记录重算。
- 训练分布内 `0.10/0.15/0.20 m` 三档每档至少 50% 首 episode 到达第一阶；成功不是 Phase 0 准入条件，但必须如实记录。
- 准入结果必须拆分为 checkpoint 结构、runtime schema、terrain binding、首 episode 完整性和 task viability，不能再压成一个自报的 compatibility 布尔值。

**完成证据**

- 冻结入口：`configs/tasks/stair_extension_schema.json`；其 descriptor 自哈希由任务包唯一常量校验，protocol、baseline manifest 和 report 的当前 hash 均从该入口及对应 canonical JSON 自动验证，不在计划中复制。
- checkpoint 文件身份由 `configs/baselines/isaaclab_g1_rough_stair_v1.json` 与共享 checkpoint 安全报告共同校验，不在计划中复制。
- 运行版本：Isaac Lab `2.3.0`、Isaac Lab Tasks `0.11.6`、Isaac Lab RL `0.4.4`、RSL-RL `3.0.1`、Torch `2.7.0+cu128`
- 实测 observation/action：`56 x 310` / `56 x 37`
- runtime 导出的 44 个 body、37 个 joint、observation/action/termination manager term、callable、参数与顺序、action resolved joint 顺序、terrain grid/env origin 均与协议精确匹配
- reset velocity、命令重采样/heading/standing 和外力事件均由 `runtime_controls` 显式冻结；terrain config 不再携带可用的协议默认几何值
- 七桶各 8 个首 episode 全部闭合记录；56/56 的终止 transition 均在 reset 前计入，逐 term mask 均为 `terminal_terms=["base_contact"]`；56/56 到达第一阶并触发 fall 诊断，七桶 success 均为 0
- 训练分布内三档的第一阶到达率均为 100%，因此通过“适合作为定向 fine-tune 起点”的行为门；该门不声称具备爬楼梯成功能力
- 报告：`PLAN/stair/phase0_report.json`
- 报告同时绑定薄脚本、任务 runtime probe、terrain、env overlay、Phase 0 contract 和共享 checkpoint 安全实现的源码 SHA-256；生产 train/eval 配置不挂 Phase 0 recorder，只有 probe 专用构建入口显式启用
- 报告结论拆分为 `checkpoint_structurally_compatible`、`runtime_schema_compatible`、`terrain_binding_compatible`、`first_episode_probe_complete`、`task_viability_gate_passed` 和 `warm_start_candidate_admitted`；同时保持 `accepted_current_best=false`、`report_is_acceptance_evidence=false`
- 已知故障注入及相邻内部一致性注入均被拒绝；本地项目全量测试 `505 passed, 4 skipped`（其中 3 项为本地无 Torch 的动态 probe 用例），Isaac 服务器上的 22 项 stair 合同/动态测试全部通过

### 阶段 1A：多任务组合与 baseline eval-only

**工作量：2-3 人日**

**状态：已完成（2026-07-25）**

**详细实施计划**

- `PLAN/stair/phase1a_implementation_plan.md`

**目标**

建立共享多任务启动入口和不伪造 candidate 的 baseline 评估路径。

**工作内容**

- 建立统一 task composition resolver，返回 plugin、memory adapter、backend config、task defaults 和 initialization。
- CLI 两阶段解析：先解析 task ID，再应用 task-owned defaults，最后解析完整参数。
- `run_loop.py`、`materialize_job.py` 和通用命名的 `run_round.py` 使用同一 resolver；保留的 push-only 工具必须显式命名或拒绝 stair。
- 新增 BaselineEvalJobSpec、baseline identity、backend materialization 和 Runner 执行能力。
- 新增通用 EvaluationSubject/EvaluationBinding 合同，并让 candidate eval 兼容迁移到该合同。
- composition 可携带独立 baseline manifest，并在 baseline eval 物化前完成 manifest/hash/compatibility 检查；本阶段不把评估结果提升为 Genesis/current-best。

**严格验收标准**

- stair 不会继承 push 的预算、profile、entrypoint 或 schema hash。
- resolver 与 `materialize_job.py` 不再包含 push 默认分支；用现有 push 和测试用 warm-start TaskPlugin/Backend fixture 验证两阶段解析与 baseline eval-only。
- baseline summary 使用 baseline identity；不存在伪造 candidate、verifier、train result 或 code-level surface。
- 错 manifest、checkpoint hash、task/backend/schema/architecture 时，在 baseline eval subprocess 启动前失败。
- push 现有启动和物化行为保持通过。

### 阶段 1B：Session Genesis、current-best、prompt 与 replay 闭环

**工作量：2.5-3.5 人日**

**状态：已完成（2026-07-27）**

**详细实施计划**

- `PLAN/stair/phase1b_implementation_plan.md`

**目标**

让 baseline 成为真实、可验证、可重放的 current-best。

**工作内容**

- 实现 SessionGenesisArtifact 的 canonical serialization、hash 和独立验证器。
- 从已验证 baseline eval 构造完整 CurrentBestEvalStats。
- live loop、context builder、acceptance replay、decision comparison 和 eval audit 接收 Genesis/full current-best。
- 对声明 `requires_baseline_genesis=true` 的任务，在 provider 或候选训练前强制完成 baseline eval 并验证不可变 Genesis。
- 每条 ledger record 绑定 Genesis 与 EvaluationBinding。
- backend 提供 task-neutral 的 session binding create/load/validate seam；默认 Push 保持静态 binding，阶段 2 的 Stair 实现在任务包内拥有隐藏 seed manifest。
- source catalog/candidate lineage 增加 baseline anchor 状态，并保留现有 best rejected source 行为。

**严格验收标准**

- 测试用 warm-start task 缺 Genesis 或 Genesis 不完整时无法进入 provider；阶段 2 注册 stair 后复用同一合同。
- baseline current-best 必须 `mode="accepted"`、`anchor_kind="baseline"` 且 `has_full_eval_stats()=true`；live loop 不接受独立覆盖它的第二个参数。
- 首个候选即使高于 absolute floor，只要未超过 baseline 的 noise-aware threshold 仍必须拒绝。
- ledger 结合 Genesis 重放后得到完全相同的 anchor、score stats、checkpoint hash 和全部协议 hash。
- 篡改 manifest hash、checkpoint hash、task/eval/seed binding、seal 或任一 ledger Genesis hash 均 fail closed。
- 同一 session 恢复时不会重建或覆盖 Genesis，也不会重新抽取 seed cohort；启动参数漂移会在 provider 前失败。
- baseline anchor 不产生伪造 source payload；首个 accepted candidate 后 current-best source 正确切换。

**完成证据**

- warm-start fixture 已覆盖 baseline eval -> Genesis -> 首轮拒绝 -> 候选接受 -> resume -> replay 全链；baseline 为 `0.25` 时，`0.28` 未超过 noise-aware threshold 而拒绝，`0.31` 才接管 current-best。
- 三轮候选训练均从同一 sealed baseline checkpoint 开始；accepted checkpoint 只改变比较权威，不改变后续训练起点。
- create-once 并发发布、Genesis/manifest/seal/private artifact/bootstrap score/ledger identity/binding 篡改均有行为故障注入；恢复不会覆盖文件或重跑 baseline eval。
- baseline 与 best rejected 可同时进入 prompt 且权威分离；首个 accepted candidate 后 source catalog 切换到该候选源码。
- 共享层未出现楼梯高度、分桶、碰撞、地形或 top-hold 语义；生产 stair registry 仍未启用。

### 阶段 2：楼梯 TaskPlugin、协议绑定与公开契约

**工作量：2-3 人日**

**目标**

在离线可测状态下实现楼梯语义，并贯通 task protocol binding。

**工作内容**

- 新建 task config、TaskPlugin、candidate surface、prompt/smoke/memory adapter、指标投影和失败分类器。
- 在 stair 任务包内实现 session evaluation binding resolver：create-once 私有 seed/bin manifest，resume 只读复用并校验 Genesis 中的 `seed_cohort_hash`；共享层不解释 manifest 内容。
- 新建 `StairClimbingRewardInputs(RewardInputs)` 与字段有效性合同。
- TaskSpec 引用 StairProtocolV1；train/eval spec、结果、seal、CurrentBest 和 replay 使用 EvaluationBinding。
- 使用 Phase 0 已迁移的 `unitree_stair_climbing` 正式 ID，运行开关继续保持 false。

**严格验收标准**

- task 只支持 `warm_start`，且声明 `requires_baseline_genesis=true`。
- resolver 可选择 `unitree_stair_climbing` 并应用其离线 defaults；旧 `future_stair_climbing` 明确拒绝而不是静默别名。
- 七桶统计、primary reason、diagnostic flags 和计数守恒均有单元/属性测试。
- 缺桶、重复桶、样本不足、非有限值、classification warning、协议或 seed cohort 不匹配均不能产生 pass。
- candidate surface 严格符合第五节，任意 RewardTerm 仍被推导为 code-level。
- prompt 不暴露 hidden seed、sealed rollout、checkpoint 路径或地形内部参数。
- 楼梯包不导入 push 包，也不依赖父项目运行时代码。

### 阶段 3：共享候选运行时与楼梯输入投影

**工作量：2-3 人日**

**目标**

复用候选执行安全逻辑，同时保持任务专属输入构造。

**工作内容**

- 从 push runtime adapter 提取任务中立的加载、VecEnv wrapper、shape/finite guard、诊断采样和激活统计。
- push 与 stair 分别提供 task-owned RewardInputs builder。
- 楼梯复用通用 `progress_fraction`，计划新增 `foot_clearance_m`、`foot_contact`、`terrain_progress_delta`；这些字段在真实 shape/finite/nonzero/reset 探针通过前不得成为 prompt-effective。
- 记录 observed/runner-derived provenance 和 reset 后历史状态。

**严格验收标准**

- push runtime 全部既有测试通过，摘要和激活门无回归。
- stair 输入在 mock 与目标探针上 shape、设备、单位和有限性正确。
- prompt-effective 字段不得来自 zero fallback；不可用字段必须从 ABI 与提示词同时移除或标记 reserved。
- 非零 RewardTerm 产生 `reward_term_calls > 0` 且 `nonzero_reward_contribution=true`。
- 原始高度图、隐藏地形参数和 eval-only 信号不进入候选对象。

### 阶段 4：楼梯 Isaac train/eval 与完整作业绑定

**工作量：3-5 人日**

**目标**

打通 baseline eval、固定起点训练和隔离候选评估。

**工作内容**

- 增加楼梯 config/overlay 与薄 train/eval entrypoint，共享安全和 summary 逻辑。
- 后端显式物化 task config、profile、schema、StairProtocolV1 和 EvaluationBinding。
- 训练固定使用同一 baseline 和协议预算。
- 评估按 protocol 中的 seed/bin/env 映射执行七桶，并生成闭合失败统计。

**严格验收标准**

- argv 显式包含 task config 和协议引用，不依赖 push 默认值。
- `num_envs * steps_per_env * iterations == budget_steps`，且 train summary/输出 checkpoint lineage 绑定 task protocol。
- runner 必须核对实际 argv、落盘 config 与 summary 中的协议字段；backend 默认值不得与 StairProtocolV1 形成第二套配置。
- 每轮 training start checkpoint hash 等于 Genesis baseline hash。
- eval 不含候选路径且不加载候选代码；baseline 与 candidate subject identity 均正确。
- public result、seal、CurrentBest 所见的 EvaluationBinding 完全一致，baseline/candidate seal 均绑定正确 EvaluationSubject。
- 每个 split/seed/bin 的 episode 数与 primary reason 计数守恒。
- 任一身份、协议、seed cohort、checkpoint 或隔离错误都明确失败，不降级为可接受结果。

### 阶段 5：本地端到端与服务器短联调

**工作量：1.5-2.5 人日**

**目标**

验证共享基础设施、两任务共存和失败恢复，再进入昂贵试运行。

**工作内容**

- 运行全量测试并分别物化 push/stair 作业。
- 服务器执行 baseline eval smoke、至少两个 RSL-RL iteration 的 warm-start train smoke 和七桶 eval smoke。
- 注入有效候选、运行失败候选、旧 task protocol 结果和错误 seed cohort。

**严格验收标准**

- 全量测试通过，push 的 plugin、prompt、runtime、train/eval 和 replay 无回归。
- baseline Genesis 与有效候选完成完整链路，在线状态和 replay 一致。
- 中断后恢复同一 session，验证 Genesis create-once、ledger replay 与下一轮编号/anchor 连续。
- 旧协议/不同 cohort 的候选被拒绝，不能与 current-best 比较。
- 运行失败候选记录为 `candidate_runtime_failed`，不伪造 eval，下一轮可继续。
- prompt/ledger 无私有路径、原始 seed 或 sealed rollout 泄漏。
- 命令结束后父子进程与 GPU 进程全部退出。

### 阶段 6：真实试运行与正式启用

**工作量：0.5-1 人日工程工作，GPU/provider 时间另计**

**目标**

用生产协议确认可审计闭环，最后开启楼梯能力。

**工作内容**

- 创建正式 Session Genesis，运行 1-3 个真实候选轮次。
- 核对 acceptance、current-best、checkpoint lineage、协议绑定和 replay。
- 更新 guide/schema、运行命令与运维说明。

**严格验收标准**

- baseline current-best 有完整评估统计与统一 EvaluationBinding。
- 至少一个候选完成真实训练、完整评估和确定性决策；不要求必须优于 baseline。
- 若接受，必须同时满足 baseline-relative noise-aware improvement、任务门、预算门、运行激活门和 sealed validation。
- Genesis + ledger 重放与在线 current-best 完全一致。
- 只有上述条件全部通过后才将 `runtime_plugin_enabled` 和 `training_command_enabled` 设为 true；`llm_stair_height_progression_enabled` 保持 false。
- 文档提供可复现命令、manifest/Genesis/产物位置、停止清理和回退方法。

## 七、工作量与完成定义

Phase 0 与 Phase 1A 已完成。Phase 1B 至 Phase 6 剩余预计 **11.5-17 人日**；完整 PPO 训练、provider 等待和 GPU 排队不计入。

完成必须同时满足：

- 单一 task-aware 组合入口可选择 stair，且所有 task-owned defaults 正确。
- baseline eval-only、Session Genesis 和完整 current-best replay 可执行且 fail closed。
- StairProtocolV1 与 EvaluationBinding 贯穿训练、评估、seal、比较和 ledger。
- 所有候选轮次从同一 baseline checkpoint 开始。
- 成功/失败 episode 完整计数且统计单位固定。
- 候选只通过公开、有效的 RewardInputs 影响训练。
- 评估只加载受信 checkpoint，不加载候选代码。
- push 无回归，楼梯语义没有进入共享核心分支。

## 八、修订后复审结论

第二轮对照当前 Runner、seal、prompt、session 和 backend 合同后，新增发现并已纳入计划：

- Genesis 必须是 session 根目录的不可变 artifact，恢复时不能重新评估并覆盖 baseline。
- live loop 只能接收验证后的 Genesis，不能同时接受另一份调用方自报 current-best。
- baseline/candidate eval 需要统一的 subject 合同，trusted eval seal 也必须识别并绑定该 subject。
- StairProtocolV1 必须是 EvaluationProtocol、AcceptanceProtocol 和 backend task defaults 的唯一来源。
- 原阶段 1A 在 stair plugin 尚不存在时要求物化 stair job，存在顺序矛盾；现改为共享 fixture 验收，真实 stair 物化移到阶段 2/4。
- classification warning 不能只记日志后继续 pass，必须阻断接受。

Phase 0 复审额外修正了两处边界：

- 官方 pyramid stairs 的 origin 在顶部/中心，不能通过改指标修补为目标任务；已在任务包底层实现 bottom-origin 单向楼梯生成器。
- 尚未实现的楼梯 RewardInputs 不能写成 effective；协议现将其标记为计划字段，task-specific prompt-effective 集合为空，等待 Phase 3 真实激活证据。
- 首 episode 证据现在由任务自有 Isaac recorder 在 auto-reset 前抓取终止 transition，探针不再读取 reset 后 robot state。
- termination manager 的 term/callable/time-out/参数顺序及 term-to-primary-reason 映射已由 StairProtocolV1 单源冻结；新增或修改 term 会在 rollout 前 fail closed，逐 episode 不再由 `time_outs` 二分猜测原因。

两个外部事实已经关闭：环境 overlay 可运行，官方 G1 checkpoint 结构兼容。Phase 1A 已关闭共享组合与 baseline eval-only 缺口；下一硬门是阶段 1B 的 Session Genesis、完整 current-best 和 replay 闭环。在它完成前，Phase 0 报告不得替代 baseline 评估或 acceptance evidence。

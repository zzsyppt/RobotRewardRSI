# True RL Agent 独立项目架构与实施计划

## 1. 背景与总目标

接下来的目标不是继续在父项目 `D:\humanoid_agent\src\humanoid_agent` 里做局部重构，而是在 `D:\humanoid_agent\true_RL_Agent` 下建立一个独立项目。父项目中已有的 Phase8E、push recovery、candidate verifier、training adapter、safe checkpoint、public projection 等实现可以作为参考，也可以复制必要代码和设计，但 `true_RL_Agent` 运行时不能依赖父项目内的 Python 包、配置或脚本。

新项目要做的是一个强化学习运控专用的 self-evolution loop agent。它不追求通用 Agent 能力，而是面向 humanoid RL policy improvement：

- 第一版必须能从 random policy 跑通 push recovery。
- 系统契约保留 warm-start baseline policy 能力，但第一版不要求接入 stair baseline 或跑 stair 训练。
- 必须保持真 code-level 改进，不允许退化成只调 reward weight、terrain range、PPO 超参或 eval threshold。
- LLM 负责提出研究假设和 candidate proposal；第一阶段不把 curriculum schedule 作为优化变量。
- Runner 负责固定预算训练、checkpoint、eval、accept/reject 和 artifact。
- TaskPlugin 负责任务语义、metric、failure classifier、固定训练/评估协议，以及后续 curriculum 扩展能力声明。
- BackendAdapter 负责把任务计划映射到具体训练系统，例如 Isaac Lab + RSL-RL。

第一版可交付目标是 push recovery 在独立项目里跑通：

```text
Task: Unitree G1/H1 push recovery
Initialization: random policy
Goal: 在独立 true_RL_Agent 项目中跑通 candidate -> verify -> train -> eval -> decision -> ledger
```

stair climbing 暂时不实现，只作为后续扩展目标。计划需要保证：以后新增 stair 任务时，主要新增 TaskPlugin、task config、baseline manifest 和必要 BackendAdapter 能力声明，而不需要大改 AgentCore、Runner、DecisionEngine 或 ledger。

第一阶段不要求 agent 调 curriculum。push 训练分布、eval protocol 和 random initialization 先由 TaskPlugin/Runner 固定，LLM 主要改 reward callable、diagnostic、failure classifier 等代码级组件。CurriculumPlan 和 CapabilityRegistry 先作为 schema、lineage 和未来扩展接口存在，避免一开始把系统复杂度压到课程调度上，也避免“只改 difficulty range 就声称进步”的退化。

## 2. 独立项目边界

### 2.1 项目根目录

所有新开发都落在：

```text
D:\humanoid_agent\true_RL_Agent
```

建议项目结构：

```text
true_RL_Agent/
  PLAN/
    true_rl_agent_refactor_plan.md

  README.md
  pyproject.toml

  configs/
    tasks/
      push_recovery.yaml
    backends/
      isaac_lab_rsl_rl.yaml
    agent/
      loop.yaml
      safety.yaml

  src/
    true_rl_agent/
      agent_core/
        loop.py
        round.py
        session.py
        provider.py
        context_builder.py
        decision.py
        ledger.py

      tasks/
        base.py
        registry.py
        push_recovery/
          plugin.py
          curriculum.py
          metrics.py
          failure_classifier.py

      backends/
        base.py
        registry.py
        isaac_lab_rsl_rl.py

      runner/
        train_job.py
        eval_job.py
        budget.py
        checkpoint.py
        isolation.py
        artifacts.py

      candidates/
        schema.py
        verifier.py
        loader.py
        prompt_projection.py
        surface_policy.py

      curriculum/
        schema.py
        compiler.py
        lineage.py

      safety/
        invariants.py
        permissions.py
        eval_seal.py
        task_bounds.py

      evaluation/
        public_result.py
        gates.py
        curves.py

      memory/
        records.py
        public_context.py
        task_memory.py

  docs/
    task_extension_guides/
      stair_climbing.md

  scripts/
    run_loop.py
    run_round.py
    materialize_job.py

  tests/
    unit/
    integration/
    fakes/
```

### 2.2 依赖边界

允许依赖：

- Python 标准库。
- 项目自己的 `true_rl_agent` 包。
- 真实外部运行依赖，例如 Isaac Lab、RSL-RL、PyTorch、Gymnasium、Unitree task packages。
- 通过配置显式指定的外部训练环境、baseline checkpoint 和数据路径。

不允许依赖：

- 不允许从父项目 `D:\humanoid_agent\src\humanoid_agent` import。
- 不允许调用父项目 `D:\humanoid_agent\scripts` 下的脚本作为运行时入口。
- 不允许读取父项目 `D:\humanoid_agent\configs` 作为新项目运行配置。
- 不允许把父项目中的 hidden eval、accept rule、budget 逻辑当作黑盒直接复用。

可以做的事情：

- 复制父项目中已经验证过的 schema、safe checkpoint、candidate verifier、public projection 等必要代码。
- 复制时必须改成 `true_rl_agent` 命名空间。
- 复制后由新项目自己的测试覆盖。
- 复制来源和设计取舍应在文档中说明，方便后续维护。

## 3. 当前父项目事实与可复用资产

父项目当前真实情况对新项目设计有参考价值：

- Phase8E 已有 candidate -> verifier -> curriculum -> train -> checkpoint -> eval -> decision -> ledger 的闭环。
- 当前 push recovery 是 random policy cold-start 任务，不是 warm-start baseline improvement 任务。
- 当前 train smoke 脚本支持 `--candidate-dir`，但没有明确的训练起点 checkpoint/resume 参数。
- 当前 eval 脚本支持 checkpoint policy，用于评估训练后 checkpoint。
- 当前 trusted eval 边界正确：eval 只加载 checkpoint，不导入 candidate code。
- 当前 curriculum contract 和 compiler 明显 push-specific，第一版应先服务 push 跑通；后续 stair 只能复用抽象边界，不能直接复用 push-specific curriculum。
- 当前 prompt-facing projection 已经有“只暴露真实接线能力”的正确方向。
- 当前父项目已经出现过“只改 reward 参数 / reward_weight_scale / pure knob”的风险信号，新项目必须把 anti-tuning gate 做成一等公民。

因此新项目应复用设计，不复用运行时依赖：

- 复用 trust boundary 思路。
- 复用 candidate package 和 verifier 思路。
- 复用 safe checkpoint 思路。
- 复用 public projection 思路。
- 复用 append-only ledger 思路。
- 不继承 push-specific curriculum registry。
- 不继承 random-only policy initialization。
- 不继承“只调参也可能被当成有效改进”的模糊边界。

## 4. 目标架构

### 4.1 AgentCore

AgentCore 是任务无关的 self-evolution loop。

它负责：

- 管理一次实验 session。
- 管理单轮 proposal -> verify -> train -> eval -> decision。
- 管理多轮 loop。
- 构造 LLM public context。
- 调用 LLM provider 或 replay provider。
- 调用 candidate verifier。
- 调用 TaskPlugin 获取任务语义。
- 调用 BackendAdapter materialize train/eval job。
- 调用 Runner 执行 job。
- 写入 ledger 和 round artifacts。

它不负责：

- 不理解 stair、push、redirection、reach 的具体任务细节。
- 不直接拼 Isaac Lab 或 RSL-RL 命令。
- 不直接加载 checkpoint。
- 不决定 hidden eval seed。
- 不把 budget、checkpoint selector、accept rule 暴露给 LLM。

### 4.2 TaskPlugin

TaskPlugin 只描述任务语义。

它负责：

- 任务目标。
- baseline 要求。
- policy initialization 支持情况。
- curriculum capability。
- public metric。
- eval protocol。
- failure taxonomy。
- public result projection。
- 下一轮 LLM 可见 failure summary。

它不负责：

- 不启动训练。
- 不加载 checkpoint。
- 不拼训练命令。
- 不直接控制预算。
- 不直接决定 accept/reject。

### 4.3 BackendAdapter

BackendAdapter 是训练后端适配层。

第一版只实现：

```text
IsaacLabRslRlBackendAdapter
```

它负责：

- 检查 TaskSpec 与 backend 能力是否匹配。
- 支持 random policy initialization。
- 支持 warm-start policy initialization。
- 把 TaskSpec、CurriculumPlan(v1 fixed lineage)、CandidatePackage、Budget、PolicyInitialization 映射为 TrainJobSpec。
- 把 EvalProtocol 和 CheckpointRef 映射为 EvalJobSpec。
- 描述 obs/action/policy/checkpoint interface。
- 生成 CLI job spec，而不是 import 训练脚本模块。

暂不实现 legged_gym、MuJoCo、SONIC 等空后端。后续有真实需求时再增加具体 adapter。

### 4.4 Runner

Runner 是执行权威。

它负责：

- subprocess 执行。
- rootfs/isolation 或等价隔离策略。
- 固定预算训练。
- checkpoint sealing。
- checkpoint safe-load。
- public eval 和 trusted eval。
- artifact 目录管理。
- budget enforcement。
- summary 收集。

Runner 不理解任务语义，只执行 BackendAdapter 生成的 TrainJobSpec 和 EvalJobSpec。

### 4.5 PolicyInitialization

这是新计划必须补上的核心接口。

系统必须支持两类初始化：

```text
random:
  每轮从随机策略开始训练。

warm_start:
  从一个已有 baseline checkpoint 初始化训练。
```

PolicyInitialization 需要表达：

- initialization_type: `random` 或 `warm_start`
- baseline_id
- checkpoint_path 或 checkpoint_ref
- checkpoint_format
- producing_backend
- compatible_backend
- robot
- obs/action compatibility
- provenance
- safe_load_status

warm-start checkpoint 只能作为训练起点，不代表候选自动通过 eval。每轮仍必须经过固定预算训练、checkpoint sealing、public eval、trusted eval 和 decision。

### 4.6 Candidate Surface Policy

第一版以当前系统已经验证的 candidate 入口为主：`candidate_manifest.yaml` + `training_code/components.py`，由候选返回受控 `TrainingComponents`。后续可以把文件拆成 `reward_terms.py`、`diagnostics.py`、`failure_classifier.py` 等更细入口，但 v1 不为了拆文件而破坏已验证的 loader/verifier 路线。

候选代码必须是真实但受控的代码入口：

- Candidate 可以写可执行 reward/diagnostic 逻辑。
- Candidate 不能任意改 repo、runner、eval、checkpoint、budget、seed、success metric。
- Observation/feature projection 这类会改变 obs/action contract 的代码，必须走 runner-owned approved extension，不能作为普通 candidate 自由改。
- Failure classifier、metric projection、backend compatibility 可以是 code-level engineering work，但不能单独证明 policy 变强。

Candidate surface 分类如下：

| Candidate 类型 | 是否影响训练策略 | DecisionEngine 结果上限 |
| --- | --- | --- |
| `RewardTerm` callable | 是 | 可以进入 `policy_improvement_accept`，但必须通过 public/trusted eval、runtime activation 和 mechanism attribution |
| Observation / feature projection | 是 | 可以进入 `policy_improvement_accept`，但必须 runner-owned 接线，普通 candidate 不能直接改 obs/action space |
| `DiagnosticTerm` callable | 间接影响下一轮 | 只能进入 `diagnostic_improvement_record`，不能单独 accept |
| Failure classifier | 间接影响下一轮 | 只能进入 `diagnostic_improvement_record`，不能单独 accept |
| Metric projection | 只影响 public summary | 只能进入 `diagnostic_improvement_record` 或 `engineering_improvement_record`，不能单独 accept |
| Backend compatibility logic | 工程能力 | 只能进入 `engineering_improvement_record`，不能作为策略提升 accept |
| PPO / reward weight / schedule 数值 | 是，但 parameter-only | 只能作为 diagnostic/tuning baseline，不能进入 policy improvement accept |
| `LossTerm` / optimizer-adjacent hook | 取决于 runner 是否接线 | 第一版不作为有效 policy surface；未接线时不得展示为 prompt-facing effective capability |

DecisionEngine 必须区分四类结果：

```text
policy_improvement_accept
diagnostic_improvement_record
engineering_improvement_record
reject
```

这样可以保留诊断和工程改进的价值，但不会把“public summary 更漂亮”误记成 policy 更强。

### 4.7 Code-Level Anti-Tuning Gate

新项目必须显式区分 code-level change 和 parameter-only tuning。当前父项目的做法提供了有用参考：它要求 curriculum 使用固定 capability registry、拒绝 reward_weight_scale-only/no-op/single-stage 设计，并在 history projection 里记录 pure knob streak。但这个机制只是第一层防线，效果有限，因为候选仍可能“贴上注册 capability 标签”，实际主要靠数值 schedule 或 reward weight 调整。

因此新项目采用更强的 `MechanismEvidenceGate`。第一阶段不要求 candidate 使用 curriculum capability；注册 capability 只是后续开放 curriculum tuning 时的准入接口。一个 candidate 要被计为 code-level improvement，必须同时有结构证据和运行证据；接近 accepted 的候选还需要机制归因证据。

不算 code-level 的改动：

- 只改 reward weight。
- 只改 `reward_weight_scale`。
- 只改 terrain height range。
- 只改 curriculum stage 数值边界。
- 只改 PPO 超参、学习率、batch size、iteration 数。
- 只改 eval threshold 或 accept threshold。
- 只改 seed、budget、checkpoint selector。

算 code-level 的改动：

- 新增或修改可执行 RewardTerm callable。
- 新增或修改可执行 DiagnosticTerm callable，并参与下一轮公开诊断。
- 新增任务相关 observation/feature projection 代码，并由 BackendAdapter/Runner 显式接线。
- 后续阶段新增 typed curriculum capability 的 runner-side 实现，而不是只在 YAML 里调数值。
- 新增 failure classifier 或 metric projection 代码，用于 public result 和下一轮 context。
- 新增 backend adapter 的 policy/obs/action/checkpoint 兼容逻辑。

注意：code-level 不等于 policy improvement。DiagnosticTerm、failure classifier、metric projection、backend compatibility 可以是有效代码改进，但默认只进入 diagnostic 或 engineering record；只有直接影响训练策略且通过 eval/activation/attribution 的机制，才可能进入 `policy_improvement_accept`。

候选提交必须包含 `CodeChangeManifest`，说明本轮至少一个非参数化代码级改动。Verifier 必须检查 manifest、代码 diff、operation_diff 和 candidate package 是否一致。DecisionEngine 不能因为 parameter-only candidate 的短期 public score 变好就接受为 code-level improvement；这类结果只能记录为 diagnostic 或 tuning baseline。

`MechanismEvidenceGate` 分四层：

1. Static code-change classification  
   检查 candidate 是否真的修改了可执行代码路径，例如 reward callable、diagnostic callable、feature projection、failure classifier、backend compatibility 逻辑。只改 YAML、manifest、operation_diff 数值或 schedule 表，归类为 parameter-only。

2. Runtime activation evidence  
   Runner 在训练时记录候选代码是否实际被调用、输出是否非零/有限、是否依赖 prompt-visible inputs、是否对 reward/diagnostic/public result 产生可观测影响。该证据必须由 runner-owned wrapper 统计，不能信任 candidate metadata 或 child process 自报。没有运行时激活证据的代码改动不能作为 accepted improvement。

3. Mechanism attribution / ablation evidence  
   对接近 accepted 的候选，Runner 运行轻量 shadow check：第一阶段保留相同 random initialization、固定 curriculum、budget class 和 eval protocol，对比“完整 candidate”和“去掉代码级机制但保留允许参数的 shadow”。shadow 只在 candidate 达到 public-dev promising threshold 或准备进入 accept gate 时触发，预算使用同档位短跑或同 seed replay，避免每轮都付出完整训练成本。只有完整 candidate 相比 shadow 有明确优势，才允许把提升归因到 code-level mechanism。后续开放 curriculum tuning 后，再把 schedule-only shadow 纳入同一接口。

4. Ledger attribution  
   每轮 ledger 必须记录 `code_change_classification`、`runtime_activation_summary`、`ablation_summary`、`mechanism_attribution` 和 `anti_tuning_decision`。连续出现 knob-only 或 attribution unclear 时，下一轮 prompt 必须显式反馈这个退化趋势。

### 4.8 Safety Invariants

除了 verifier、safe checkpoint、trusted eval、anti-tuning gate，还需要显式 `TaskInvariantSpec`。RL 运控里最常见的 hack 往往不是 Python import 越权，而是语义越权，例如把任务变简单或改评估定义。

第一版 push recovery 至少要固定这些 invariants：

- fall threshold / termination predicate。
- episode length / max steps。
- push magnitude、push timing、push direction distribution。
- success predicate。
- eval seed visibility 和 hidden eval seed。
- budget、checkpoint selector、accept threshold。
- public/private eval split。
- task config 中禁止 candidate 修改的字段。

后续 stair 任务需要同类 invariants，例如 stair height eval bins、terrain slope、toe collision logging、success predicate、episode length、eval seed 等。

`safety/` 模块职责：

- `invariants.py`：定义 `TaskInvariantSpec` 和 invariant diff/check result。
- `permissions.py`：定义 candidate 可见/可写/可请求的 logical ids。
- `eval_seal.py`：定义 trusted eval 输入输出 seal，不导入 candidate code。
- `task_bounds.py`：定义任务固定边界，例如 push magnitude、episode length、success predicate。

Task invariants 是 runner-owned gate。普通 code diff classifier 只能判断改了什么文件，不能完全判断“是否把任务偷偷变简单”；因此 TrainJobSpec materialization 前后都必须做 invariant check。

### 4.9 Memory/Context

完整 memory system 后置。第一版只做最小接口：

- ExperimentRecord
- PublicContextBuilder
- TaskMemoryAdapter

目标是保证每轮实验可追溯，且下一轮 LLM context 只来自 public projection。

## 5. 实施计划

### 阶段 0：独立项目初始化与边界确认

**总目标：** 把 `D:\humanoid_agent\true_RL_Agent` 建成独立 Python 项目，先确立“不依赖父项目运行时”的工程边界。

**要达成的功能：**

- 建立 `src/true_rl_agent` 包。
- 建立 `configs`、`scripts`、`tests`、`PLAN` 基础目录。
- 建立 `pyproject.toml` 和 README。
- 明确禁止 import 父项目 `humanoid_agent`。
- 建立一个轻量 import boundary test。

**验收标准：**

- 在 `true_RL_Agent` 根目录下可以 import `true_rl_agent`。
- 测试中禁止出现 `from humanoid_agent` 或 `import humanoid_agent`。
- 新项目不调用父项目 scripts 作为运行时入口。
- README 明确说明父项目只作为参考来源。

### 阶段 1：参考资产盘点与复制策略

**总目标：** 明确哪些父项目能力需要复制，哪些只保留设计思想，避免盲目搬运。

**要达成的功能：**

- 盘点 Phase8E 当前闭环中可借鉴的模块。
- 形成 extraction map：candidate schema、verifier、safe checkpoint、ledger、public projection、training adapter。
- 标注 push-specific 代码，不直接复制为通用实现。
- 标注 random-only policy initialization 的限制。
- 为后续复制代码建立最小目标列表。

**验收标准：**

- 有一份新项目内的 reference inventory 文档。
- 每个计划复制的组件都有用途说明。
- push-specific curriculum registry 不进入通用层。
- 父项目代码不会成为运行时依赖。

### 阶段 2：定义核心数据契约

**总目标：** 先定义系统内部稳定接口，为后续 AgentCore、TaskPlugin、BackendAdapter、Runner 打基础。

**要达成的功能：**

- 定义 TaskSpec。
- 定义 BackendSpec。
- 定义 PolicyInitialization。
- 定义 BaselinePolicyRef。
- 定义 CandidatePackageRef。
- 定义 CurriculumPlan(v1 frozen/runner_owned/task_default)。
- 定义 TrainJobSpec。
- 定义 EvalJobSpec。
- 定义 CheckpointRef。
- 定义 PublicResult。
- 定义 ExperimentRecord。
- 定义 CodeChangeManifest。
- 定义 CandidateChangeSet。
- 定义 CandidateSurfacePolicy。
- 定义 DecisionOutcome：`policy_improvement_accept`、`diagnostic_improvement_record`、`engineering_improvement_record`、`reject`。
- 定义 TaskInvariantSpec。

**验收标准：**

- random 和 warm_start 都能被 PolicyInitialization 表达。
- warm_start checkpoint 与 eval checkpoint 概念分开。
- TrainJobSpec 能表达训练起点 checkpoint。
- EvalJobSpec 能表达待评估 checkpoint。
- 所有契约都不包含 hidden eval seed 或 accept internals。
- CodeChangeManifest 能区分 code-level、parameter-only、mixed 三类 candidate。
- parameter-only candidate 默认不能进入 accept 路径，只能作为 tuning diagnostic 记录。
- DiagnosticTerm、failure classifier、metric projection、backend compatibility 不能单独产生 policy improvement accept。
- TaskInvariantSpec 能表达 push recovery 的 fixed eval/task bounds。

### 阶段 3：实现 TaskPlugin 与 BackendAdapter 注册机制

**总目标：** 建立任务层和后端层的插件边界。

**要达成的功能：**

- 定义 TaskPlugin 接口。
- 定义 TaskRegistry。
- 定义 BackendAdapter 接口。
- 定义 BackendRegistry。
- 实现 compatibility check 流程。
- 明确 TaskPlugin 不返回训练命令。
- 明确 BackendAdapter 不做 accept/reject。

**验收标准：**

- 未注册 task_id 会在训练前失败。
- 未注册 backend_id 会在训练前失败。
- TaskPlugin 只能返回任务语义和计划。
- BackendAdapter 只能 materialize job。
- AgentCore 可以通过 registry 获取 task 和 backend。

### 阶段 4：FakeTask + FakeBackend + FakeRunner 最小纵向闭环

**总目标：** 在接入 Isaac Lab 和 push 之前，先用 fake task/backend/runner 跑通一次最小单轮闭环，避免前期全在写抽象接口。

**要达成的功能：**

- 在 `tests/fakes/` 中定义 FakeTaskPlugin。
- 在 `tests/fakes/` 中定义 FakeBackendAdapter。
- 在 `tests/fakes/` 中定义 FakeRunner。
- FakeRunner 不启动训练，只生成 deterministic train/eval summary。
- 跑通 TaskSpec -> Public Context -> replay Candidate -> verifier -> fake train -> fake eval -> Decision -> Ledger。
- 在 fake round 中验证 parameter-only reject 和 diagnostic-only record。

**验收标准：**

- 不依赖 Isaac Lab 也能跑通一轮。
- ledger 能记录 candidate、job spec、eval result、decision outcome。
- `policy_improvement_accept`、`diagnostic_improvement_record`、`engineering_improvement_record`、`reject` 至少各有一个 replay fixture。
- fake loop 通过后再接真实 push backend。

### 阶段 5：实现 Runner Job 抽象

**总目标：** 让执行层只接受统一 job spec，不关心具体任务。

**要达成的功能：**

- train_job 执行 TrainJobSpec。
- eval_job 执行 EvalJobSpec。
- budget 模块检查固定预算。
- checkpoint 模块处理 checkpoint sealing 和 safe-load。
- artifacts 模块管理 session、round、train、eval 输出路径。
- isolation 模块预留 rootfs/subprocess 隔离策略。
- safety 模块在 materialize 和执行前后检查 TaskInvariantSpec。
- eval_seal 模块保证 trusted eval 只加载 sealed checkpoint，不导入 candidate code。

**验收标准：**

- Runner 可以 dry-run 打印 train/eval command。
- Runner 能区分 random train job 和 warm-start train job。
- checkpoint sealing 是 runner-owned。
- safe-load 结果写入 summary。
- task invariant violation 会在训练前失败，不能进入 eval/accept。
- trusted eval seal 记录 eval config hash、checkpoint hash、task invariant hash。
- Runner 不读取 TaskPlugin 内部字段。

### 阶段 6：实现 PolicyInitialization 与 Baseline 管理

**总目标：** 正式支持 random policy 训练起点，并把 warm-start baseline policy 做成稳定契约，为后续 stair/reach 等 baseline-improvement 任务预留入口。

**要达成的功能：**

- 定义 baseline manifest。
- 支持 baseline_id。
- 支持 checkpoint path/ref。
- 支持 checkpoint provenance。
- 支持 backend compatibility 检查。
- 支持 robot、obs/action、policy architecture compatibility 检查。
- 支持 safe-load 检查。
- TrainJobSpec 中明确携带 initialization 信息。
- 第一版 push 运行路径只要求 `random` initialization 可执行；`warm_start` 至少完成 manifest、compatibility、safe-load、dry-run job materialization。

**验收标准：**

- random initialization 不需要 checkpoint。
- warm_start initialization 必须有 BaselinePolicyRef。
- warm_start checkpoint 必须通过安全检查。
- backend 不兼容时训练前失败。
- warm-start train job 必须在 spec 里明确 `init_checkpoint` 与 eval checkpoint 的区别。
- 真正接入第一个 warm-start 任务前，BackendAdapter 必须补 `runner.load(init_checkpoint)` 级别的训练加载 smoke；这个检查不阻塞第一版 push random 交付。

### 阶段 7：实现 IsaacLabRslRlBackendAdapter

**总目标：** 支持第一版真实后端，把 push task、fixed CurriculumPlan、candidate package 和 policy initialization 映射为 Isaac Lab + RSL-RL 的 CLI job。

**要达成的功能：**

- 定义 backend config。
- materialize random train command。
- materialize warm-start train command spec，但第一版只要求 dry-run/compatibility 验证。
- materialize checkpoint eval command。
- 表达 obs/action/policy/checkpoint compatibility。
- 表达 backend 支持哪些 task capabilities。
- 对 Isaac/Unitree/RSL-RL 依赖做显式外部依赖声明。

**验收标准：**

- BackendAdapter 不 import 父项目脚本。
- BackendAdapter 生成 CLI job spec。
- random 和 warm_start 生成不同训练起点参数。
- eval job 明确 checkpoint policy。
- 后端能力不足时返回结构化 compatibility failure。
- push random train/eval command 能被 Runner dry-run 和真实执行。

### 阶段 8：实现 Candidate 与 Prompt Projection 基础能力

**总目标：** 建立 LLM proposal 的安全边界，并让 prompt 只展示真实可用能力。

**要达成的功能：**

- 定义 candidate package schema。
- 第一版沿用已验证入口形态：`candidate_manifest.yaml` + `training_code/components.py`，entrypoint 返回受控 `TrainingComponents`。
- 定义 candidate verifier。
- 定义 candidate loader。
- 定义 controlled code hooks：RewardTerm、DiagnosticTerm、ScheduleTerm whitelist；LossTerm/observation transforms 未接线时不得作为 effective capability。
- 定义 prompt-facing RewardInputs projection。
- 定义 candidate_allowed_controls projection。
- 定义 CandidateSurfacePolicy。
- 定义 closed source-pack policy。
- 定义 public-only prompt payload 检查。
- 定义 CodeChangeManifest。
- 定义 MechanismEvidenceGate。
- 定义 static code-change classifier。
- 定义 runtime activation evidence schema。
- 定义 knob-only shadow candidate 生成规则。
- 在 prompt 中明确：只调 reward weight、terrain range、threshold、seed、budget 不算有效 candidate。
- 明确 observation/feature projection 属于 runner-owned approved extension；第一版普通 candidate 不能任意改 obs/action space。
- 明确 DiagnosticTerm/failure classifier/metric projection/backend compatibility 只能形成 diagnostic 或 engineering record，不能单独 policy accept。

**验收标准：**

- Candidate 不能拿 env、runner、optimizer、checkpoint selector、eval config、success metric。
- Prompt 不展示未接线 capability。
- Source request 只能访问 allowlist logical ids。
- Candidate 越权修改会被 verifier 拒绝。
- Candidate 不能提交任意仓库 patch；只能提交受控 candidate package 文件。
- Candidate 如果只包含参数调整，会被标记为 parameter_only，不允许进入 accept 路径。
- Candidate 必须至少包含一个 runner 可加载或 verifier 可审计的代码级组件。
- 接近 accepted 的 candidate 必须生成 mechanism attribution，不能只用 public score 说明“代码有效”。
- runtime activation evidence 必须来自 runner-owned instrumentation，candidate 自报字段只能作为非权威说明。
- DecisionEngine 对每轮明确输出四类 outcome 之一，且 policy accept 只允许来自会影响训练策略的受控 surface。

### 阶段 9：预留 CurriculumPlan 与 CapabilityRegistry

**总目标：** 第一阶段不做 curriculum tuning，只把课程相关信息做成任务可扩展、可追踪、后续可启用的接口，避免把旧项目 push-specific curriculum 编译器搬进新架构。

**要达成的功能：**

- 定义通用 `CurriculumPlan` schema，但第一版只支持 `frozen` / `runner_owned` / `task_default` 三类状态。
- 定义 `CapabilityRegistry` 接口，但第一版只做注册、展示和有效性标注，不把 LLM curriculum proposal 接入训练。
- TaskPlugin 提供固定训练协议和固定 eval protocol；第一版先服务 push recovery，后续 stair 可用同一 schema 声明 baseline train distribution 与 eval bins。
- BackendAdapter 标注后端是否支持某类 future curriculum capability，但默认 `effective_now=false`。
- Curriculum compiler 第一版只负责 validate、lineage、explain，不负责优化或改写 schedule。
- 记录“任务默认课程 -> 实际训练配置”的 lineage，确保每轮可复现。
- 明确 curriculum 数值调度本身不是 code-level 改进；第一阶段任何 schedule-only proposal 都只能作为 diagnostic 记录，不能进入训练或 accept 路径。

**验收标准：**

- 第一版系统可以完全不让 LLM 调课程也能跑通 self-evolution loop。
- capability 必须来自 registry；未接线 capability 不显示为 `effective_now`。
- 编译结果能解释 `frozen`、`runner_owned`、`task_default`、`disabled_by_phase`。
- schema-level stair 扩展说明可以表达固定 baseline train distribution、固定 eval bins、height-map/terrain metadata，但第一版不实现 stair plugin，也不开放 stair_height progression 给 LLM 调。
- reward_weight_scale-only、height-range-only、single-stage numeric schedule 不能被标记为 code-level improvement，也不能因为 public score 变好进入 accepted improvement。

### 阶段 10：实现 AgentCore 最小闭环

**总目标：** 跑通任务无关的单轮和多轮 self-evolution skeleton。

**要达成的功能：**

- session 管理实验目录。
- round 管理单轮流程。
- loop 管理多轮流程。
- provider 支持 replay/dry-run。
- decision 支持 `policy_improvement_accept`、`diagnostic_improvement_record`、`engineering_improvement_record`、`reject`。
- ledger 支持 append-only record。
- context_builder 支持从 public record 构造下一轮 prompt context。
- Runner 收集 runtime activation evidence：candidate callable 调用次数、输出统计、非零贡献、diagnostic 输出、feature/capability 激活记录。
- DecisionEngine 支持 MechanismEvidenceGate：parameter-only candidate 即使 public score 上升，也只能作为 diagnostic 记录，不能成为 accepted code-level improvement。
- DecisionEngine 对 DiagnosticTerm/failure classifier/metric projection/backend compatibility 这类非策略面给出 diagnostic 或 engineering record，不允许单独 policy accept。
- 对 potential accept candidate 触发 knob-only shadow check 或等价轻量消融。

**验收标准：**

- 可以不启动真实训练，dry-run 一轮完整流程。
- 可以 replay 一个 candidate response。
- ledger 中能看到 candidate、curriculum、train job、eval job、decision。
- private 字段不进入 public context。
- ledger 中能看到 code_change_classification、runtime_activation_summary、ablation_summary、mechanism_attribution、anti_tuning_decision 和 pure_knob_streak 类反馈。
- ledger 中的 outcome 能区分 policy、diagnostic、engineering 和 reject。

### 阶段 11：实现 Push Recovery 参考任务

**总目标：** 把 push recovery 作为第一版唯一必须跑通的真实任务，用于验证新项目架构、训练闭环和 anti-tuning gate。

**要达成的功能：**

- 新增 push_recovery TaskPlugin。
- 复制或重写必要 push config schema。
- 定义 push metric。
- 定义 push failure classifier。
- 定义 push TaskInvariantSpec：fall threshold、episode length、push magnitude/timing、success predicate、eval seed visibility、budget、checkpoint selector。
- 定义 push 固定训练协议和固定 eval protocol。
- 预留 push curriculum capability 声明，但第一版不让 LLM 调课程。
- 支持 random initialization。
- 通过 IsaacLabRslRlBackendAdapter 生成 train/eval job。
- 跑通 candidate -> verify -> train -> checkpoint -> eval -> decision -> ledger。

**验收标准：**

- push recovery 不依赖父项目 package。
- push plugin 不直接启动训练。
- push plugin 不包含后端命令拼接。
- public result 能表达 success rate、failure breakdown、motion quality。
- push random policy train/eval 能真实执行并产出 sealed checkpoint、public result、ledger record。
- 任何放宽 push 难度、缩短 episode、改 success predicate、改 eval seed、改 accept threshold 的候选都会被 task invariant gate 拒绝。
- schedule-only 或 reward_weight_scale-only candidate 不能成为 accepted code-level improvement。

### 阶段 12：Stair Climbing 扩展性验收

**总目标：** 不实现 stair 任务，只验证当前架构未来能以较小改动接入 stair climbing。

**要达成的功能：**

- 编写 `docs/task_extension_guides/stair_climbing.md`，说明未来 stair plugin 需要提供哪些字段。
- 用 schema-level 示例说明未来 baseline：Isaac/Unitree G1 rough locomotion policy trained on stair_height 0.05-0.20 m。
- 用 schema-level 示例说明未来 eval bins：0.10、0.15、0.20、0.25、0.30、0.35、0.40 m。
- 说明 future stair fixed train protocol：baseline train distribution、terrain metadata、height map 可用性、foot clearance 观测/诊断入口、terrain progress 诊断入口。
- 说明 future stair failure taxonomy：toe collision、insufficient clearance、base pitch instability、slip、timeout。
- 说明 future stair code-level extension surfaces 中哪些属于普通 RewardTerm/DiagnosticTerm，哪些必须是 runner-owned approved observation/feature projection。

**验收标准：**

- 第一版不创建 `stair_climbing` runtime plugin，不创建 stair 训练命令，不运行 stair 实验。
- 文档能明确未来新增 stair 时应只新增 TaskPlugin、task config、baseline manifest、failure classifier、metrics，并按需扩展 BackendAdapter capability。
- 架构检查能说明 AgentCore、Runner、DecisionEngine、ledger 不需要因 stair 任务语义而修改。
- stair 的 height-map/terrain feature projection 被标为 runner-owned extension，不允许普通 candidate 任意改 obs/action space。
- 只扩大 stair_height range、只调 reward weight 或只改 schedule 的 future candidate 不允许被计为 code-level accepted improvement。

### 阶段 13：最小 Memory/Context 接口

**总目标：** 为长期记忆系统预留稳定接口，但不提前做复杂 memory。

**要达成的功能：**

- ExperimentRecord 记录每轮实验。
- PublicContextBuilder 构造下一轮 prompt。
- TaskMemoryAdapter 压缩任务经验。
- 区分 public record 和 private artifact。
- 保存 prompt payload。

**验收标准：**

- 下一轮 LLM context 来自 public projection。
- hidden eval、private seed、checkpoint selector 不进入 prompt。
- push 能提供任务特定 public summary；stair 扩展文档说明未来如何接入 task-specific public summary。
- 后续可以扩展长期 memory，而不改 AgentCore 主流程。

### 阶段 14：文档、测试与任务扩展规范

**总目标：** 让新项目具备可持续扩展任务和后端的工程规范。

**要达成的功能：**

- 文档化如何新增 TaskPlugin。
- 文档化如何新增 BackendAdapter。
- 文档化如何接入 baseline checkpoint。
- 文档化 random vs warm_start 的选择。
- 文档化 prompt-facing capability 准入原则。
- 文档化 trusted eval 和 runner-owned boundary。

**验收标准：**

- 新增 redirection/reach 有明确路径。
- 新增 backend 有明确路径。
- Baseline checkpoint 接入有明确 checklist。
- 测试覆盖注册、job materialization、policy initialization、public projection、安全边界。

## 6. 测试计划

### 6.1 单元测试

需要覆盖：

- 禁止依赖父项目 package。
- TaskRegistry 注册和未注册 task。
- BackendRegistry 注册和未注册 backend。
- PolicyInitialization random。
- PolicyInitialization warm_start。
- BaselinePolicyRef compatibility check。
- CandidateSurfacePolicy 分类。
- DecisionOutcome 分类：policy / diagnostic / engineering / reject。
- TaskInvariantSpec diff/check。
- CodeChangeManifest classification。
- MechanismEvidenceGate static classification。
- Runner-owned runtime activation evidence projection。
- Knob-only shadow candidate generation。
- Anti-tuning decision：parameter-only 或 attribution-unclear candidate reject/not-accept。
- CurriculumPlan v1 frozen/runner_owned/task_default 状态解释。
- LLM curriculum schedule proposal 在第一阶段被标记为 disabled_by_phase，不能进入训练配置。
- BackendAdapter job materialization。
- Candidate verifier 越权拒绝。
- Candidate package 只能使用受控入口，不能提交任意 repo patch。
- Candidate 自报 runtime activation 不作为权威证据。
- task invariant gate 拒绝语义越权：放宽 fall threshold、缩短 episode length、降低 push magnitude、改 success predicate、改 eval seed。
- PublicContextBuilder 不泄露 private 字段。

### 6.2 集成测试

需要覆盖：

- dry-run 单轮 AgentCore。
- FakeTask + FakeBackend + FakeRunner 最小纵向闭环。
- replay candidate 单轮 AgentCore。
- push recovery random job materialization。
- push recovery random train/eval smoke。
- warm-start job spec dry-run 与 compatibility failure case。
- replay 一个 parameter-only candidate，验证它不能进入 accept 路径。
- replay 一个 code-level reward callable candidate，验证它通过 code-level gate。
- replay 一个 DiagnosticTerm-only candidate，验证它只能进入 diagnostic_improvement_record。
- replay 一个 failure-classifier-only candidate，验证它不能 policy accept。
- replay 一个 backend-compatibility-only candidate，验证它只能进入 engineering_improvement_record。
- replay 一个“贴了 capability 但只有数值 schedule 变化”的 candidate，验证它被归类为 knob-only。
- replay 一个 public score 变好但 shadow check 同样变好的 candidate，验证它不能被归因为 code-level improvement。
- replay 一个 curriculum schedule-only proposal，验证第一阶段记录为 diagnostic/disabled_by_phase，不进入 TrainJobSpec。
- stair extension guide/schema fixture 不生成 runtime plugin、不生成 train job，只验证 TaskPlugin/BackendAdapter 边界可表达未来需求。
- checkpoint safe-load summary。
- public eval projection。

### 6.3 实验验收

Push recovery：

- 可以作为 random policy 任务跑通。
- public result 能提供下一轮可用反馈。

Stair climbing 扩展性：

- 本版不运行 stair。
- extension guide 能说明未来 baseline manifest 如何指向 0.05-0.20 m rough locomotion policy。
- extension guide 能说明未来 eval 如何输出 height-bin success curve 与 max passable stair height。
- extension guide 能说明未来目标可表达为向 0.40 m 台阶推进。
- extension guide 能说明 accepted improvement 仍必须关联到 code-level candidate、runner-owned runtime activation evidence 和 mechanism attribution，不接受纯调参 attribution。

## 7. 建议实施顺序

推荐顺序如下：

0. 独立项目初始化、import boundary、subprocess boundary。
1. 核心 schema 最小版：TaskSpec、TrainJobSpec、EvalJobSpec、CandidatePackageRef、DecisionOutcome、TaskInvariantSpec。
2. FakeTask + FakeBackend + FakeRunner 跑一轮，先验证 ledger 和 decision shape。
3. Candidate schema + CodeChangeManifest + CandidateSurfacePolicy + 最小 anti-tuning gate。
4. Runner artifact/checkpoint/ledger：artifact layout、checkpoint seal、safe-load、trusted eval seal。
5. Push TaskPlugin + IsaacLabRslRlBackendAdapter dry-run。
6. Push train/eval smoke。
7. Runner-owned runtime activation evidence。
8. Shadow ablation，只对 promising candidate 或 potential accept candidate 触发。
9. 多轮 loop 和 public context 迭代。
10. Stair disabled skeleton / extension proof：只写扩展文档和 schema fixture，不实现 runtime plugin。

这个顺序与旧计划相比有六个关键变化：

- 先做 FakeTask/FakeBackend/FakeRunner 最小纵向闭环，避免前期只写抽象层。
- warm-start 不再放到 stair plugin 里临时处理，而是作为系统级 PolicyInitialization 契约提前定义；第一版只要求 dry-run/compatibility，不要求真实 stair 训练。
- CurriculumPlan 和 CapabilityRegistry 在任务接入前只完成接口、lineage 和禁用态解释，避免把 push-specific curriculum 搬进新架构，也避免第一阶段变成调课程。
- CodeChangeManifest 和 MechanismEvidenceGate 提前完成第一层：static classification + runtime activation + parameter-only reject；shadow ablation 后置到 promising candidate。
- CandidateSurfacePolicy 和 DecisionOutcome 明确区分 policy improvement、diagnostic improvement、engineering improvement 和 reject。
- `true_RL_Agent` 是独立项目，不再走 `cold_start_lab` compatibility wrapper 路线。

## 8. 默认假设

- 第一版项目根目录是 `D:\humanoid_agent\true_RL_Agent`。
- 第一版 package 名是 `true_rl_agent`。
- 第一版不依赖父项目 `humanoid_agent` package。
- 第一版可以复制父项目中必要代码，但复制后必须归入新项目命名空间。
- 第一版后端固定为 Isaac Lab + RSL-RL。
- 第一版机器人以 Unitree G1/H1 为主。
- 第一版必须支持 push recovery random policy 真实运行。
- 第一版保留 warm_start 契约和 dry-run materialization，但不要求接入 baseline policy。
- 第一版不实现 stair climbing runtime plugin、不运行 stair 训练、不生成 stair train job。
- 第一版不让 LLM 调 curriculum；任务训练分布和 eval bins 由 TaskPlugin/Runner 固定。
- 第一版 CurriculumPlan 只做 frozen/runner_owned/task_default lineage 和后续扩展接口。
- 第一版不做通用 Agent。
- 第一版不开放任意工具调用。
- 第一版 candidate 入口以已验证的 `candidate_manifest.yaml` + `training_code/components.py` 为主，不开放任意仓库 patch。
- 第一版不实现复杂长期记忆系统。
- 第一版不接受 parameter-only candidate 作为 code-level improvement。
- 第一版允许记录 parameter-only run 作为 diagnostic/tuning baseline，但不能进入 accepted improvement。
- 第一版区分 policy improvement、diagnostic improvement、engineering improvement 和 reject。
- LLM 不获得 budget、checkpoint selector、hidden eval、accept rule 的控制权。
- LLM 不获得 task invariant 的控制权，例如 fall threshold、episode length、push magnitude、success predicate、eval seed。
- TaskPlugin 不直接启动训练，不直接加载 checkpoint。
- BackendAdapter 先做组合后端，不拆成独立 SimAdapter 和 AlgoAdapter。

## 9. 预期最终效果

完成后，新项目应支持：

- 独立运行，不依赖父项目内部代码。
- 同一套 AgentCore 能跑通 push recovery random-policy self-evolution loop，并保留 warm-start contract。
- 换任务主要新增 TaskPlugin。
- 换后端主要新增 BackendAdapter。
- stair climbing 后续可通过新增 TaskPlugin、task config、baseline manifest 和必要 backend capability 接入，不需要重写 AgentCore/Runner/DecisionEngine。
- accepted improvement 必须通过 code-level anti-tuning gate。
- diagnostic/engineering code change 可以被记录和复用，但不能冒充 policy improvement。
- LLM 只能看到真实可用的 prompt-facing 能力。
- Runner 始终保持训练、checkpoint、eval 和 decision 权威。
- 每轮实验可追溯，为后续 memory system 打基础。

最终闭环：

```text
TaskSpec + PolicyInitialization
  -> Public Context
  -> LLM Research Proposal
  -> Candidate Package
  -> Verifier
  -> Candidate Surface Policy
  -> Task Invariant Gate
  -> Code-Level Anti-Tuning Gate
  -> Fixed Task Curriculum / CurriculumPlan(v1 frozen)
  -> BackendAdapter TrainJobSpec
  -> Runner Fixed-Budget Training
  -> Sealed Checkpoint
  -> EvalJobSpec
  -> Trusted Eval
  -> Task PublicResult
  -> Decision + Ledger
  -> Next Round Context
```

这就是 `true_RL_Agent` 的目标：一个独立、可控、可复现、面向强化学习运控策略改进的自进化 Agent 项目。第一版以 push recovery 跑通闭环，stair climbing 作为后续任务扩展目标。

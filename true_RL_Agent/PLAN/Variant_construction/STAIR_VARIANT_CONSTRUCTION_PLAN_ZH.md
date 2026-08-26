# 机器人 × 任务矩阵与复杂 Stair 自进化建设计划

> - 文档状态：Phase 0、Phase 1、Phase 2 已实施并完成严格验收；下一阶段为 Phase 3A（本文件替代此前的 Stair variant family 草案）
> - 更新日期：2026-08-27
> - 适用仓库：`D:\humanoid_agent\true_RL_Agent`
> - 当前分支：`true-rl-agent-refactor`
> - Phase 0 捕获 provenance：2026-08-26 核验时，本地、`origin/true-rl-agent-refactor` 与服务器
>   `/data/yuyangliu/AGENT` 均为 `f3593567b3901453f7dbf80dd6c450554b957fb2`；该 SHA 只记录捕获来源，不作为后续入口验收门。
>   后续文档提交允许改变 HEAD；产品基线由 26 个相关产品文件、resolved contracts、runner source contract、reward ABI
>   与 normalized reference argv 的内容身份共同验证。其中产品代码修复提交为
>   `ab84aa3350700a2f7d46231afefaec4425ffa2ef`，`f359356` 只追加 P6-D 审计文档。
>   冻结的正式 15-generation 实验仍绑定 `c582eca04dbac7d5ee5e2df0f56c646500ead55f`，不得与当前产品字节混称。
>   同步前已有的未跟踪实验与审计产物保持原样；本文件与 `P0/` 是本次指定写入范围。
> - 当前执行环境边界：项目的真实 Stair 证据绑定 Isaac Lab 2.3.0；上游较新文档仅作架构调研，
>   本计划不顺带升级 Isaac Lab / Isaac Sim。

## 0. 一句话目标

> **在现有自进化系统内部建立“机器人 × 任务”的二维组合能力：机器人与任务是两根独立轴，
> 每个有效单元格只需一层很薄的绑定即可接入同一套 LLM LOOP；每个单元格在自己冻结的环境变体、
> 策略谱系和评测协议下尝试持续改进。Core V1 先让 G1-37DoF / Go2 在复杂 Stair 上完成真实闭环，
> 保留现有 G1-29DoF × Push Recovery，用共享的 RobotProfile schema、role 词汇与 binding compiler 证明两根轴可独立建设；首个
> “同一 concrete profile 跨两个任务复用”的强证明留给后续 Go2 × Push Recovery。**

这不是“给 Stair 复制几个机器人版本”，也不是另建一套 variant 系统。共享的仍然是现有
`AgentCore → LLM authoring → verifier → runner train/eval → decision → ledger/current-best`；
不同单元格共享软件结构和改进协议，但不共享策略权重、checkpoint、current-best 或实验历史。

## 1. 目标矩阵、Core V1 与完成定义

### 1.1 目标结构

```text
                              任务轴（TaskPlugin）
                       Stair        Push Recovery       Future Task
执行机器人轴  G1-37    已有→复杂化      不声明支持            薄 binding
(exact       G1-29    不声明支持        保留现有              薄 binding
profile)     Go2      Core V1 新增     后续强复用证明          薄 binding
             六足     后续扩展         后续扩展               薄 binding

每个有效单元格 = (RobotProfile + RuntimeAdapter) × TaskPlugin × RobotTaskBinding
单元格变体     = 有效单元格 × Task-owned EnvironmentProfile
              → ResolvedCellManifest → 现有 LLM LOOP
```

矩阵的扩展成本应是“新增一份行能力或列能力，再为真正支持的单元格补少量映射”，而不是复制
`机器人数量 × 任务数量` 份完整训练系统：

- 新增机器人行：实现一次任务无关的 `RobotProfile + backend-specific RuntimeAdapter`，再为要支持的任务增加声明式或很薄的
  `RobotTaskBinding`；
- 新增任务列：实现一次 `TaskPlugin`，复用已有机器人行，并为兼容机器人增加薄 binding；
- 新增单元格环境难度：由对应 `TaskPlugin` 增加 task-owned environment variant；Stair 是楼梯几何与
  surface，Push Recovery 则可以是推力、方向、接触面等，不能强行抽象成一个通用 TerrainPlugin；
- 启动一个自进化实验：冻结一个单元格及其环境变体，创建独立 Genesis / ledger / current-best，
  然后调用现有 LOOP。

为避免把 roughness 误认为矩阵的第三根轴，身份分成两级：

- `robot_task_cell_id = task_id + robot_profile_id`，表示矩阵里的一个单元格；
- `cell_variant_id = robot_task_cell_id + environment_profile + protocol_version`，表示该单元格的一种冻结实验条件。

`robot_family_id`（如 `unitree_g1`）只用于报告分组，不能作为执行身份或 artifact 兼容依据。执行行必须精确到
asset、关节/执行器集合、控制周期和上游包；因此当前 Stair 的 Isaac Lab G1-37DoF 与 Push 的
`unitree_rl_lab` G1-29DoF 是两个不同 `robot_profile_id`，不得伪装成同一行。

Core V1 有 3 个 execution-level Robot × Task 单元格：`G1-37DoF × Stair`、`G1-29DoF × Push Recovery`、
`Go2 × Stair`；clean/irregular/rough 使它们解析为下面 7 个 cell variants。LOOP session、Genesis 和策略谱系冻结到
`cell_variant_id`，matrix report 同时报告 cell 覆盖和 variant 证据等级。

### 1.2 Core V1 的具体范围

Core V1 只建立足以证明二维契约与稀疏矩阵落地的最小集合；同一 concrete profile 跨任务复用的强证明属于
Extension B：

| 单元格变体 | Core V1 责任 | 真实 LOOP 要求 |
|---|---|---|
| `stair.g1_37dof.clean_legacy_v1` | 保留当前 35-cell、reward ABI 与历史 replay | 回归，不重复做研究结论 |
| `stair.g1_37dof.irregular_v1` | 不等高/不等深、平踏面的首个复杂变体 | 必须完成真实多轮 LLM reward LOOP；是否改进不设 GO 门 |
| `stair.g1_37dof.rough_v1` | 在 irregular 上增加可复现踏面微起伏 | runtime / decoder / reference 资格化，不重复要求 3 代 |
| `stair.go2.clean_v1` | Go2 资产、控制、观测、奖励输入与 Stair 成功语义资格化 | 固定预算资格训练与评测 |
| `stair.go2.irregular_v1` | Go2 自己的可行不规则平踏面分布 | 必须完成真实多轮 LLM reward LOOP；是否改进不设 GO 门 |
| `stair.go2.rough_v1` | Go2 自己的有界 rough Stair 分布 | runtime / decoder / reference 资格化，不重复要求 3 代 |
| `push.g1_29dof.legacy_v1` | 保留现有厂商 G1-29DoF Push 路径 | parity / LOOP contract 证明，不新增 GPU 性能研究 |

这 7 个变体是 Core V1 的数量上限；其中新增的研究型多轮实验只有两个 irregular Stair 变体，rough 与 Push
变体不重复做 3-generation 性能研究。六足和新 LLM 权限仍在 Core V1 完成后扩展。
因此 Core V1 可以声明“LLM LOOP 已在 C1 irregular 复杂楼梯上闭合、C2 rough 已满足接入同一 LOOP 的前提”，不能
声明“LLM 已在 rough Stair 上改善策略”；后者必须由 C2 独立 session 的 `loop_verified + improvement_observed`
证据支持。

### 1.3 Core V1 的工程完成定义

只有同时满足以下条件，才能称为完成：

1. 仍只有一套 AgentCore、LLM provider、candidate authoring/verification 协议、runner、decision、ledger 协议和 replay；
   TaskPlugin 可以继续选择自己的 verifier/policy，但不得复制 LOOP；
2. G1-37DoF / G1-29DoF 以两个 execution profile 登记，但共享同一套 semantic-role 词汇与 binding compiler；
   TaskPlugin 源码不出现具体关节名、link 名、USD 路径或机器人上游包 import；
3. G1-37DoF 与 Go2 都通过真实 Isaac runtime 的 clean / irregular / rough Stair 资格验证；
4. `G1-37DoF × irregular Stair` 与 `Go2 × irregular Stair` 各自至少形成一条完整的多轮
   `LLM → reward program → train → eval → authoritative accept/reject/no-selection → memory → next generation` 证据链；
5. 两个 Stair 机器人各自使用独立 Genesis、reference reward、ledger、checkpoint 和 current-best，任何跨单元格
   复用都被现有 identity / replay 边界拒绝；
6. 新增的 G1-37/Go2 Stair cell 当前只开放 `complete_reward_program`；Push legacy 保留自己现有的
   `diagnostic_term / reward_term / reward_weight_scale_schedule` capability，只作兼容迁移。训练算法、预算、
   种子、checkpoint、任务成功/失败语义、评测集合、acceptance 与选择规则继续由 TaskPlugin/composition 定义、
   由 runner 强制执行；
7. 两个 Stair cell 达到 `loop_verified` 即满足工程闭环；`improvement_observed` 必须如实报告，但不作为 Core GO；
8. 新增一行或一列不需要修改 AgentCore。Core V1 只声明“共享矩阵契约已经落地”；只有 Extension B 的
   Go2 profile 在 Stair/Push 两列复用后，才声明“同一 concrete robot row 已跨任务复用”。若 Go2 接入仍要求复制
   TaskPlugin、训练入口或 LOOP，则矩阵抽象验收失败。

### 1.4 明确不做

- 不训练一份跨 G1 / Go2 / 六足的 universal policy；
- 不在同一 session 或 generation 的两个 lane 中切换机器人、任务或环境变体；
- 不新建 variant scheduler、variant ledger、全局 current-best、数据库或常驻服务；
- 不引入第三套 RobotPlugin 系统；RobotProfile/RuntimeAdapter 不包含最终 policy view、agent cfg、成功判据、
  奖励、地形或评测；
- 不让 LLM 修改 PPO、训练预算、训练种子、评测、接受门、机器人资产或 checkpoint；
- 不在 Core V1 建设通用 Curriculum Engine、通用 TerrainPlugin 或完整 `Go2 × Push Recovery`；
- 不因研究了上游新版本就升级当前已冻结的 Isaac Lab 2.3.0 执行环境。

## 2. 最新系统基线与真实阻塞点

### 2.1 可以原样复用的主干

当前代码已经具备矩阵所需的大部分系统主干：

- [TaskPlugin](../../src/true_rl_agent/tasks/base.py) 已拥有任务规范、固定 train/eval、acceptance、
  candidate surface、task-owned prompt、公开结果投影、component smoke 与 phased evaluation；
- [ResolvedTaskComposition](../../src/true_rl_agent/composition/base.py) 已统一装配 TaskPlugin、memory、backend、
  runner、verifier 和 policy initialization，并把 `task_protocol_hash + task_defaults` 纳入 Genesis identity；
- [composition registry](../../src/true_rl_agent/composition/defaults.py) 已按 `task_id` 解析任务；
- [run_live_loop](../../src/true_rl_agent/agent_core/live_loop.py) 与
  [round execution](../../src/true_rl_agent/agent_core/round.py) 不解释具体任务或机器人，只消费解析后的 composition；
- [Stair prompt adapter](../../src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py) 与
  [Stair memory adapter](../../src/true_rl_agent/tasks/stair_climbing/memory.py) 已提供 task-owned 的公开反馈入口；
- `EvaluationBindingV1` 已包含 `task_protocol_hash`，Genesis 也已绑定 composition snapshot，因此无需仅为矩阵
  再造第二套 evaluation binding 或 Genesis schema。

因此矩阵必须在 composition 解析阶段完成，之后继续进入同一套 LOOP。AgentCore 不应知道 `robot_profile_id`、
`foot_count` 或 roughness。

### 2.2 当前尚未具备的能力

同步后的 `f359356` 仍有以下硬阻塞：

- [Stair composition](../../src/true_rl_agent/tasks/stair_climbing/composition.py) 是唯一 Stair provider，
  但 robot、action/observation schema、agent cfg 和入口仍来自 G1-37 专用配置；
- [Stair EnvCfg](../../src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py) 直接继承 G1-37 rough 环境；
- [runtime observation](../../src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py)、
  [reward runtime](../../src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py)、
  [episode evidence](../../src/true_rl_agent/tasks/stair_climbing/episode_evidence.py) 与
  [transition decoder](../../src/true_rl_agent/tasks/stair_climbing/transition_decoder.py) 固定两只脚、G1-37 ankle / torso
  及 37 维动作语义；仅替换 USD asset 不可能支持四足或六足；
- [Stair terrain](../../src/true_rl_agent/tasks/stair_climbing/isaaclab_terrain.py) 仍是平整 box 台阶，difficulty
  只插值阶高；没有生产 roughness，现有解析式 surface evidence 也不能解释局部起伏；
- [Push runtime config](../../src/true_rl_agent/tasks/push_recovery/isaac_config.py) 选择厂商
  `Unitree-G1-29dof-Velocity`，而 [Stair frozen protocol](../../configs/tasks/stair_protocol_v1.json) 绑定 Isaac Lab
  G1 minimal 的 37 维动作/关节；共同的 `unitree_g1` 标签不是同一 profile 证据；
- train/eval 入口当前只接受唯一 canonical config；matrix manifest 尚不能被 runner 编译后传入并在 runtime 重载；
- `TrainingBindingV1` 没有 task/environment identity。不同单元格若架构和预算相同，训练绑定不足以阻止
  checkpoint 误用；
- Stair memory 目前主要按 `task_id` 过滤，尚不能显式阻止同任务不同机器人/环境的历史混入。

当前 G1-37 clean 的 `7 个阶高 × 5 个踏面深度 = 35 cells` 是必须保留的 legacy regression anchor。
当前生产训练分布已有 row-based 阶高 curriculum；Core V1 先复用这一简单机制，不引入自动 curriculum teacher。

### 2.3 最新 P6-C / P6-D 变化的影响

当前主线已经有 width=2 bounded generation、生命周期续行、成本与资源 receipt、final confirmation 等最新能力，
但证据边界必须准确：

- 内部 engine/config/test 覆盖 1 / 3 / 15 generation；当前 production CLI 只开放冻结的 P6-D formal
  15-generation，不能声称现在已经有可自由调用的 1 / 3-generation 生产入口；
- P6-C 在提交 `415bfad` 上有 3 条 generation lifecycle/ledger 记录，但只有 2 代具备完整科学结果，另 1 代为
  `paired_authoring_incomplete`；
- `c582eca` 上已经完成一次真实 formal 15-generation 实验；按修正后的 report v6 口径，公共分类为
  `accepted=4 / formal_rejected=7 / no_selection=4`，共 15 个有效科学 generation，冻结报告 SHA-256 为
  `6b7e324c09b58c62a1e450308080be5b9cef533b5737a74933ef389f2ebe0897`；
- `ab84aa3` 修复了后续审计确认的实现/证据缺口，`f359356` 只增加审计记录。相同产品字节已通过 Windows 与 Linux
  全量测试，但尚未在 `ab84aa3/f359356` 上重新执行 provider / Isaac / GPU formal 15-generation；因此可以引用
  `c582eca` 的历史正式实证，不能称为“当前精确字节已复跑”；
- [历史 R1 证据](../../outputs/formal_15round_b445125_evalv5_followups.md) 中 candidate generalization 为
  `0.2464286`、Genesis 为 `0.5900000`，说明 bounded LOOP 可以工程闭合但没有任何候选必然被接受；
- 现有 P6-D 成本上限来自 G1-37 当前 terrain 的预冻结假设，不能直接宣称适用于 Go2 或 complex Stair。
- 当前以 reference program 为 Genesis anchor 的 Stair `live` composition 会强制 15 rounds 和 final-confirmation
  plan；该历史路径只做兼容 smoke。Core V1 的非正式性能闭环统一走最新 generation foundation 的固定
  1-generation pilot → 3-generation validation，不再另加一套 3/6-round live 权威路径。

本计划会把 cell manifest 接入现有 generation foundation，但不复制 generation engine，也不把十五代正式实验
设为矩阵基础设施的前置条件。Phase 4 会把现有 runner、launcher、launch-context、readiness 与 Genesis
创建链一并扩展为固定枚举的 non-formal 1 / 3-generation profile；任何进入 generation campaign 的 variant 必须先用 1 代实测
成功路径成本并独立冻结 budget profile，才可进入 3 代。15 代 formal 仍保留原有独立授权语义。

证据边界可对照
[P6-C 验收报告](../2-GPUS/P6-C/P6C_ACCEPTANCE_REPORT_ZH.md)与
[P6-D 实现审计交接](../2-GPUS/P6-D/P6D_IMPLEMENTATION_AUDIT_HANDOFF_ZH.md)。

## 3. 最小矩阵架构

### 3.1 组合关系

```text
现有 TaskCompositionRegistry
          │ task_id + cell_variant_id
          ▼
┌──────────────────────┐       ┌────────────────────────────┐
│ TaskPlugin（任务列） │       │ RobotProfile + RuntimeAdapter│
│ 目标/反馈/评测/权限  │       │ 物理能力/asset/raw state    │
└──────────┬───────────┘       └────────────┬───────────────┘
           └──────────┬───────────────────┘
                      ▼
           RobotTaskBinding（薄单元格映射）
                      │
             Task-owned EnvironmentVariant
                      ▼
              ResolvedCellManifestV1
                      ▼
             现有 ResolvedTaskComposition
                      ▼
       现有 LLM LOOP / runner / train / eval / ledger
```

现有 `StairClimbingCompositionProvider` 继续是唯一 Stair provider；它按 `cell_variant_id` 解析 G1-37 / Go2 和
Stair environment profile，再返回现有 `ResolvedTaskComposition`。不注册 `stair_g1`、`stair_go2` 两个任务，
也不创建 `stair_climbing_variants` 第二 provider。Core V1 的 Push Recovery provider 只保留 G1-29DoF execution
profile；Extension B 获批后才新增消费同一 Go2 RobotProfile/RuntimeAdapter 的薄 binding。两个任务始终各自
保留 TaskPlugin 和任务语义。

矩阵层的通用保证是每个受支持 cell 都能进入现有 `run_live_loop` contract；当前 Stair P6-D generation 是复用
同一 authoring/round/runner/ledger 主干的 task-owned campaign wrapper。Core Stair 性能验证统一使用该最新
generation wrapper，Push parity 只验证通用 loop contract。未来 TaskPlugin 不必先复制 Stair generation 才能
成为有效任务列；若需要 width=2 generation，再单独证明其 wrapper 复用相同核心语义。

### 3.2 各层权责

| 层 | 拥有什么 | 明确不能拥有什么 |
|---|---|---|
| `TaskPlugin` | 任务目标、成功/失败/进度、task-logical observation/reward/termination、环境与 curriculum 语义、prompt/memory、评测、acceptance、required capabilities 和可开放的 LLM capability | 机器人 asset/raw link 名、训练进程、checkpoint 写入 |
| `RobotProfile` | 精确 asset/上游包/关节与执行器 identity、物理尺度、控制能力、默认姿态、对共享 semantic roles 的映射、可用传感器与挂载能力 | 最终 policy observation/action view、agent cfg、任务奖励/成功、地形、评测 |
| `RobotRuntimeAdapter` | backend-specific ArticulationCfg/asset、raw body/joint/sensor 解析与 raw-state 读取 | Task EnvCfg、reward ABI、评测和 decision |
| `RobotTaskBinding` | 用显式 semantic roles 满足 TaskPlugin requirements，并冻结 task-specific sensor mount、action mapping/scale、observation projection 和少量形态阈值 | 改写任务目标、评测、训练预算或 LOOP |
| composition / manifest | 合成最终 Task EnvCfg、policy observation/action schema、policy architecture、agent cfg、reward ABI，并在 Genesis 前冻结 | 动态改变已启动 session 的 robot/task/environment |
| runner / backend | 强制执行已解析的算法/预算/seed、train/eval 顺序、checkpoint、运行证据与资源预算 | 自行发明 task semantics，或接受 LLM 自报结果 |
| LLM candidate | Stair Core cells 只生成经 task verifier 接受的完整 reward program；依据公开反馈提出下一代候选 | policy 权重、robot、terrain、curriculum、训练、评测、seed、acceptance、current-best |

这里不创建 `RobotPlugin`。每个精确 execution embodiment 由一份 task-neutral `RobotProfile` 和一个 backend-specific
`RobotRuntimeAdapter` 组成；最终 EnvCfg、policy observation/action、agent cfg 和 reference reward 都是
`TaskPlugin + RobotTaskBinding + composition` 的单元格结果。相同厂商/型号但 asset、DoF 或上游包不同，必须是
不同 profile；不同 profile 只共享 semantic-role vocabulary，不共享 hash 或 artifact。

TaskPlugin 必须声明 required capabilities（例如 base state、support effectors、contact sensing、height scan、
action mode），RobotProfile/RuntimeAdapter 声明 provided capabilities。binding compiler 只引用显式 semantic
roles，不根据裸 link/joint 名猜测；缺失、重复或互斥的 role/sensor/action 映射在 composition/Genesis 前产生
`unsupported + reason`，不能拖到 Isaac runtime 才失败。同一 body 可以同时承担非互斥角色，例如 Stair 的
`base` 与 `forbidden_contact` 都指向躯干；当前共享词汇中只有 `support_effector × forbidden_contact` 是结构性互斥，
不能用“一个 target 只能有一个 role”的全局唯一规则代替语义判断。

共享词汇保持很小：`base`、`support_effector[*]`、`forbidden_contact[*]`、`joint_group[*]` 与 sensor capability。
TaskPlugin 只能引用这些 task-logical roles；具体 `left_ankle_roll_link`、`FL_foot`、关节名、`.usd` 路径和上游
package import 只能出现在 RobotProfile/RuntimeAdapter 或 cell binding。用一条源码边界检查约束所有任务，
不为每个机器人重复手写黑名单。

### 3.3 唯一跨边界编译产物

新增的唯一权威跨边界对象为 `ResolvedCellManifestV1`。为避免自引用 hash，解析顺序固定为：

1. TaskPlugin 先给出 effective/resolved `base_task_protocol_hash`，不能误用裸 `stair_protocol_v2.protocol_hash`；
2. manifest 输入由 base task protocol、RobotProfile、RuntimeAdapter protocol、binding、environment/eval、
   resolved obs/action/agent/reward ABI 和 capability profile 组成；
3. canonical serialization 产生 `manifest_hash`；
4. `base_task_protocol_hash + manifest_hash` 派生最终 `resolved_cell_task_protocol_hash`，写入 manifest 与 Genesis 的
   `resolved_cell` identity；非 legacy cell 同时写入 TrainingBindingV2；
5. 两个 exact legacy cells 为保持历史 identity，TaskSpec / EvaluationBinding 继续使用 base task protocol。首个
   non-legacy Stair cell 必须在 Phase 3A 把 resolved hash 提升为该 session 的 effective TaskSpec / EvaluationBinding
   comparison domain，不能带着 base hash 进入真实 LOOP。

manifest 最少冻结：

- `cell_variant_id`、`task_id`、TaskPlugin protocol/version；
- `robot_family_id`（仅报告）、`robot_profile_id`、RobotProfile/runtime adapter hash；
- RobotTaskBinding hash；
- task-owned calibration/train environment、curriculum、public_dev/public_accept/sealed_acceptance/unseen_final_holdout evaluation
  profile hash；
- `active_acceptance_protocol_hash`、`reference_capability_gate_contract_hash`、reference-gate cell-set hash、
  `public_accept` cohort hash 与 evaluation-semantics hash；它们引用 TaskPlugin 生成的同一份规范，不复制第二套
  acceptance/gate 配置；
- observation/action/agent/reward-input schema hash；
- CandidateCapabilityProfile hash；
- canonical manifest hash 与最终 resolved cell task protocol hash。

系统继续使用一条 TaskSpec / Genesis / EvaluationBinding / TrainingBinding 执行链，不建立平行的“variant
binding”。Phase 1 的 legacy 单元格由 Genesis `resolved_cell`、child manifest 与 checkpoint lineage 绑定 cell，
Phase 3A 起的 non-legacy 单元格还必须让 EvaluationBinding 的 `task_protocol_hash` 等于 resolved hash。manifest
不能只作为父进程标签：train/eval child 必须接收
`--resolved-cell-manifest <absolute path> + --resolved-cell-manifest-hash`（最终命名可调整），通过同一 strict
parser 重载并重算；权威 summary 必须回传 cell id/hash 以及 runtime-derived robot、obs/action 与 terrain
identity。path/hash/tamper/cross-cell 任一不一致必须在 AppLauncher 或训练开始前拒绝。

两个 legacy cells 由 task-owned config 构造这些 hash，并在 Phase 1 验收中逐字段对照；首个 non-legacy cell
必须在 Phase 3A 的 composition 门断言 `manifest.active_acceptance_protocol_hash == active
AcceptanceProtocol.protocol_hash == EvaluationBinding.acceptance_protocol_hash`。reference qualification evidence
回传的 gate-contract、cell-set 与 `public_accept` cohort hash 也必须等于 manifest。TrainingBindingV2 只通过
resolved task hash 传递该绑定，不再重复增加 acceptance 字段。这里增加的是显式审计与一致性断言，不是新的
acceptance 系统。

ledger 仍使用一套协议/实现，但每个 session 有自己的 ledger 文件；不增加全局或 variant ledger。

唯一必要的通用 binding 升级是：`TrainingBindingV2` 增加 `task_protocol_hash`。旧 G1-37 session 继续由
V1 reader replay。只有 `stair.g1_37dof.clean_legacy_v1` 与 `push.g1_29dof.legacy_v1` 可以通过绑定完整
profile/adapter/binding/environment/requirements/base-protocol 的 grant hash 生成 V1；grant 被复制到任何变更后的
cell 都必须拒绝，其他 cell 默认且只能生成 V2。

### 3.4 奖励输入 ABI 与可变形态

Stair 的 reward/evidence runtime 不能继续假设恰好两只脚。VNext ABI 采用：

- 所有机器人共有的 task-logical core：base pose/velocity、progress、terrain-local height、command、
  support contact/position/velocity/force、joint/action state、termination signals；
- `support_effectors` 轴长度由冻结 RobotProfile 决定，当前 Stair G1-37 为 2、Go2 为 4、未来六足为 6；
- morphology-specific 特征只在该单元格真实存在时公开，并计入 input schema hash；不存在的 arm、finger、
  ankle 或 torso joint 不用零向量伪装；
- G1-37 legacy reward ABI 与 reference program 保持可 replay，VNext 单元格使用新的确定性 ABI；
- 同一个 LLM LOOP 可以消费不同 ABI，因为 Stair TaskPromptAdapter 会把该单元格的公开 reward contract
  明确投影给模型；不要求一份 reward program 跨机器人复用。

任务语义仍属于 Stair。例如“稳定登顶”由 Stair 定义，RobotTaskBinding 只提供支撑末端数量和 body 映射；
最终需要几只支撑足、保持多久、哪些 contact 非法，必须在该单元格的 Stair protocol 中冻结，而不能由机器人
Profile/RuntimeAdapter 自行改变。

新机器人不能直接复制 G1-37 reference reward。Go2 reference 的首选路径是按冻结的 Isaac Lab 2.3.0 源码，解析
Go2 rough config 的父类 reward 与 Go2 overrides，移植为 task-owned 的完整 reward program，并像
`official_g1_rough_v1` 一样做 source/effective-term 等价验证。若官方源码缺失/无法映射冻结 ABI，或等价候选通过
却未通过 Stair reference capability gate，只允许启动一次预注册、有固定预算的 fallback；fallback 在人工
task-logical 基础 reward 与现有 reward-cold-start 中预先选择一条，不能两条连续搜索。
这里复用的是 reward/agent 构造方法，不复制官方 `random_rough` 的 terrain 幅值、curriculum、训练预算或 acceptance；
这些仍由 Stair cell protocol 冻结。
该 artifact 必须绑定 cell task protocol、reward-input schema 和 capability profile；通过资格门后才能成为
reference-anchored generation 的 Genesis。Go2 verifier 必须在训练前拒绝 G1-37 ankle/arm/37-DoF 等非法字段。

旧 G1-37-clean 的 `absolute_accept_floor=0.2` 与现有 `reference_capability_gate` 不能被新增 VNext cell 自动继承；
`stair.g1_37dof.clean_legacy_v1` 必须保持历史 active protocol/gate 与 replay identity。每个新增
irregular/rough/Go2-clean cell 在 LLM LOOP 前，以 calibration split 和预先声明的 cell-local 规则冻结自己的
acceptance protocol 与 reference qualification gate；它们的 hash 同时进入 manifest、Genesis、train/eval
binding 和 summary。标定失败应在 Phase 3A/3B 或 Phase 5 收口为 terrain/reference NO-GO，不能拖到 3-generation
实验后再降低地板。若新 cell 仍保持相同 `[0,1]` macro-success 语义且 calibration 证明 `0.2` 仍有明确能力含义，
可以重新采用，但必须是新 active protocol 的显式结论；reference 还要在不重叠的 calibration/public_accept/sealed_acceptance
cohort 上通过该 floor 和 gate，不能用 reference 自身结果反推一个保证过门的公式。最小数据流固定为
`calibration 冻结规则 → 独立 public_accept 执行 reference gate → sealed_acceptance 验证`，不新增单独 qualification seed split。

### 3.5 LLM 权限的现在与未来

现有 Stair V2 已要求 capability profile 恰好是完整 reward program，这与新增 Stair cells 的 Core V1 目标
一致。矩阵不新增一套权限系统，而是继续使用各 TaskPlugin 的
`CandidateCapabilityProfile + candidate surface + verifier`。下表只约束新增 Stair cells；
`push.g1_29dof.legacy_v1` 保留当前 legacy capabilities 与 StaticCandidateVerifier，不纳入本轮性能研究：

| 能力 | Core V1 | 未来扩展方式 |
|---|---|---|
| 完整 reward program | 开放；每轮重新训练 policy | 保持 task-owned ABI 与 verifier |
| 课程参数/课程组件 | 关闭 | 由 TaskPlugin 显式新增一种 capability；候选只提交有界方案，runner 编译为训练配置 |
| 观测选择、网络结构、PPO 超参 | 关闭 | 只有独立研究 Phase 获批后才可逐项增加，不预留通用自由编辑口 |
| 训练预算、seed、评测、acceptance、checkpoint | 永久不由 LLM 控制 | TaskPlugin/composition 定义并哈希，runner 强制执行 |

未来开放 curriculum 时，评测套件、训练总预算和 curriculum 可修改范围仍由我们冻结。LLM 只能在公开的候选
schema 中提案，不能返回可执行 runner 配置，也不能在一轮中改变 robot 或 evaluation distribution。

### 3.6 Prompt、memory 与谱系

Stair TaskPromptAdapter 在现有允许的 task-owned sections 中加入紧凑的公开单元格摘要：robot morphology、
support count、reward ABI、训练地形范围、当前公开 evaluation cells 和失败分解。不要增加新的通用顶层 prompt
section，也不要把 sealed_acceptance、unseen_final_holdout cells 或私有 seed 暴露给 LLM。

Stair TaskMemoryAdapter 除 `task_id` 外还必须按 effective task protocol / cell manifest 过滤。每个
`task × robot × environment profile × obs/action/reward schema` 谱系拥有独立：

- output root 与 session ID；
- Genesis 与 reference reward program；
- ledger epoch、checkpoint 与 current-best；
- prompt history 与公开 experiment memory；
- training/evaluation binding 与 final report。

跨机器人报告可以并排展示结果，但不能创建跨机器人 winner，也不能把 Go2 分数与 G1-37 分数送入同一个
current-best 选择器。warm-start 只允许同机器人、同 schema、同 task protocol 的显式兼容谱系。

## 4. Stair 复杂度与评测设计

### 4.1 Task-owned 难度阶梯

第一版只实现一套可解释、可复现的 Stair 生成器，不建设通用 terrain framework：

| 等级 | 几何 / surface | 作用 |
|---|---|---|
| C0 clean | 保持当前等高、等深、平踏面 | G1-37 35-cell legacy anchor；Go2 clean qualification |
| C1 irregular | 在冻结约束内生成不等高/不等深序列；每块踏面仍为平面，起点、riser、边缘和顶端目标区明确 | 先增加宏观落脚规划难度，保留精确解析 decoder |
| C2 rough | 在 C1 的踏面内部叠加有界 height-field roughness；边缘、riser 与目标区不被噪声重定义 | Core V1 的 complex Stair |
| sensitivity | 少量摩擦档位，和几何 roughness 分开记录 | 只作覆盖/稳健性评估，不展开 dynamics randomization |

roughness 必须通过真实 mesh / height samples 和 runtime surface query 生效，不能以配置字段或 prompt 文本存在
作为验收。空间尺度选择为 height scanner 可分辨的单一 regime；Core V1 不同时研究“完全不可感知 roughness”。

Core V1 还冻结一个简单范围约束：C2 相对名义踏面的 `max_abs_height_offset` 不超过
`0.5 × min(insufficient_clearance threshold, maximum_projected_surface_penetration)`；按当前两项均为
`0.03 m`，首版上限为 `0.015 m`。这只是控制首版范围，根治分类污染仍依赖真实 surface：生产 query 相对独立
mesh oracle 的 `ε_surface` 必须不超过 `min(7.5 mm, 1/4 × 最小相关判据阈值)`。在 peak/valley、riser 邻域和
支撑/摆动转换的固定 evidence 上，oracle 查询 binding 冻结的 support-effector collision/contact footprint，而不是
只查 link 中心点；逐帧比较 support level/local height、root/foot clearance、diagnostic flags、failure labels 与
stable-arrival/success。判据带外零分歧，带内只作为离线 calibration ambiguity 统计，不新增 runtime outcome 或
LLM 字段。C2 classifier receipt 通过前，failure grids、primary-failure breakdown、diagnostic rates、dominant
failure 等全部 classification-derived feedback 都不进入 prompt/memory。另报告固定 policy、matched seeds 下
C1→C2 的 failure distribution 变化，但不把真实行为变难误判成 decoder 漂移，也不新增一套 failure classifier。
首版还冻结最大局部坡度，使现有 world-XY slip proxy 与 local-tangent oracle 的误差低于预设容差；否则 C2 NO-GO，
不在本轮顺手建设新 slip estimator。

G1-37 与 Go2 的合格范围可以不同。RobotProfile 提供 standing/base height 等尺度事实用于生成初始范围，最终范围
必须经过固定规则资格化并写入 manifest。Phase 3A/3B 先冻结 G1-37 irregular/rough environment profiles 和通用
qualification/coverage 方法；
Go2 profile 与两机器人 common anchors 等 Phase 5 获得真实 Go2 runtime 后再最终冻结。计划同时保留：

- `common physical anchors`：两个机器人都运行的少量相同绝对几何，只作描述性对照；
- `robot-qualified suite`：按该机器人可行范围冻结，作为它自己的 LOOP objective / guard；
- 二者不合并成一个跨机器人总分。

### 4.2 训练分布与 curriculum

Core V1 复用现有 row curriculum 机制，不增加自动 teacher。Stair TaskPlugin/task config 定义并哈希 curriculum
及训练分布，composition 在 Genesis 前冻结，runner 只负责不可绕过地编译和执行：

- 每个 `cell_variant_id` 只在自己的 environment profile 内升难度，session 运行中不得从 C1 切换到 C2；
- C1 从 low irregular 进入 target irregular；C2 独立从 low roughness 进入 target roughness，二者各用自己的预冻结进度/登顶门；
- height/depth/roughness/realization seed 的映射由 Stair config 编译到 runtime，LLM 不可修改；
- 每个新增 VNext environment profile 的 acceptance protocol、reference capability gate 与分数尺度在 calibration
  split 上单独冻结；不得把 G1-37-clean 的 `0.2` absolute floor 或低台阶 gate 数值直接复制到 irregular/rough/Go2；
  G1-37 clean legacy 继续使用历史 identity；
- calibration、train、public_dev、public_accept、sealed_acceptance 与 unseen_final_holdout seed namespace 完全分离；
- 若某机器人没有任何难度档落入预先声明的“非平凡且可学习”资格区间，则该 cell 为 NO-GO，回到 terrain
  profile 重新定义，不能事后降低评测门伪造成功。

difficulty qualification 必须在任何 LLM LOOP 前冻结 easy/target/hard 阈值和 common-anchor 候选，且只使用
calibration seeds。生成器的几何指标必须随 difficulty 单调；固定 baseline/reference 应在 easy 可完成、target
落入预设非平凡区间、hard 显著更难。看到 LOOP 结果后不得重新挑选 target cells。

### 4.3 紧凑评测，不做笛卡尔积

Core V1 不枚举 `robot × height × depth × irregularity × roughness × friction × seed` 全组合：

- G1-37 的 clean 35-cell 只作为兼容性与阶段回归 anchor；
- 先冻结 factors、levels 与非法组合约束，再由 covering-array generator 给出最小 cells，并证明所有可行二元
  组合达到 100% coverage；每个 operational complex suite 总上限为 18 unique cells（含 hard corners 与
  sealed_acceptance）。若 18 个仍不足，必须减少 factors/levels 或重新审批预算，不能把固定数量冒充完整 pairwise；
- common physical anchors 从上述 cells 中选最多 6 个，不额外扩大评测；
- 一次 full phased evaluation 的总 `eval_num_envs` 不超过 512（当前 G1-37 基线为 280），并在真实服务器上记录
  steps/s、墙钟与输出量；
- coverage report 明确每个组合被覆盖、排除或留作 sealed_acceptance 的原因；
- public_dev 可以给 LLM 失败分解；sealed_acceptance 延续现有 guard，用于每代选择但不进入 prompt/memory；
- 另冻结一个不参与任何候选选择的 `unseen_final_holdout` seed/cell namespace。只有 final policy 冻结后运行一次，
  运行后不得继续 authoring 或换策略；它最多增加 4 个 holdout cells / 128 eval envs，正式未见泛化结论只能
  引用这一份结果。

## 5. 变体管理方式

### 5.1 源配置与运行产物

建议保持以下最小拓扑；目录名是责任边界，不要求一次完成所有占位文件：

```text
src/true_rl_agent/
├─ robots/                         # 机器人轴；不得 import 具体 task
│  ├─ profile + registry
│  ├─ runtime_adapters/            # backend-specific raw runtime
│  ├─ g1_37dof_isaaclab/
│  ├─ g1_29dof_unitree_rl_lab/
│  └─ go2/
├─ tasks/
│  ├─ stair_climbing/
│  │  ├─ plugin + composition      # 任务列
│  │  ├─ robot_bindings/           # G1-37 / Go2 的薄单元格映射
│  │  └─ environment_variants/     # clean / irregular / rough，完全 task-owned
│  └─ push_recovery/
│     ├─ plugin + composition
│     └─ robot_bindings/           # Core V1 只迁移 G1-29 parity
└─ composition/                    # 继续输出 ResolvedTaskComposition

outputs/<session>/
├─ resolved_cell_manifest.json
├─ session_genesis.json
├─ ledger.jsonl
├─ runner/<round>/train|eval/summary.json
└─ matrix_cell_report.json
```

操作者只选择一个受版本控制的 `cell_variant_id`，不能在命令行自由拼装 robot、terrain、eval 和 budget。
composition 将 catalog entry 一次解析为 immutable manifest；train/eval child 必须严格重载、重算 hash 并在
summary 回传 runtime-derived identity，不能只相信父进程传来的标签。

### 5.2 矩阵状态与报告

由 resolved catalogs 自动生成 `matrix_coverage_report`。每个 `cell_variant_id` 只使用下面四级证据状态；
`robot_task_cell_id` 只汇总其各 variant 的覆盖与证据，不另造一个会掩盖粒度差异的 cell 状态：

- `declared`：静态配置存在，尚无真实 runtime；
- `runtime_qualified`：真实资产、schema、train/eval 已通过；
- `loop_verified`：fresh 多轮生命周期/ledger 闭合，且至少一个 verified candidate 实际完成 train + public eval、
  reward runtime activation 和权威 accept/reject/no-selection 记录；不要求 accepted；
- `formal_evidence`：完成预注册的多 session / unseen_final_holdout 研究协议。

`cell_loop_ready` 不是第五种状态，而是 VNext variant 的派生资格布尔量：只有
`runtime_qualified=true`、active acceptance/reference-gate/evaluation identity 已冻结，并且绑定同一 manifest 的
reference 在独立 `public_accept` 与 `sealed_acceptance` 资格流程通过时才为真。它由 manifest 与权威 qualification
receipt 自动计算，不人工维护；失败时必须分别报告 terrain difficulty、reference capability 或协议绑定原因。

不支持的单元格记录 `unsupported + reason`，不创建空实现。报告只索引各 session 的权威 artifact，不复制或
重写 ledger。另由 ledger 的 accepted/head transition 派生布尔字段 `improvement_observed`，表示是否真的出现
current-best 更新，避免把工程连通与性能改善混成一个状态；它不是人工维护的第二份状态。这样能一眼看到矩阵
覆盖，又不会增加数据库和第二套状态机。

### 5.3 接入工作量预算

矩阵抽象通过后，正常扩展应满足：

- 新机器人行：一份 RobotProfile/runtime adapter；
- 新单元格：一份薄 binding，以及由 TaskPlugin/binding 解析出的 cell-specific action/observation/agent cfg
  和经资格化的 reference reward；
- 新任务列：一份 TaskPlugin + task-owned environment/eval/prompt/memory + 每个受支持机器人的薄 binding；
- 新环境变体：只改对应 TaskPlugin 的 variant catalog / runtime compiler / evaluation suite；
- 三种扩展都不得修改 AgentCore、共享 LLM authoring loop、ledger、decision 或另建训练入口。

普通新单元格不得增加 task/robot-specific AgentCore 分支或复制 TaskPlugin/runner 脚本；确有必要的共享 seam
必须保持 task-neutral，并至少由两个任务或两个单元格共同验证，否则先判定抽象失败，不继续堆兼容分支。

## 6. Phase 计划与严格验收

各 Phase 只定义目标和验收，不在计划中预写大量实现步骤。每个 Phase 的验收证据按需要来自四类：确定性解析、
真实 Isaac runtime、现有 LOOP 连通、固定预算科学比较。静态测试不能替代要求为“真实 runtime/LOOP”的验收。
标注的“人日”只含工程工作，不含 GPU 排队和实验墙钟；Phase 3A 冻结 C1/Phase 4 预算，Phase 3B 再冻结 C2
预算。Phase 4 可在 3A 通过后启动，不等待 3B。

### Phase 0：最新基线与二维边界冻结（2 人日）

**目标**：把当前 G1-37DoF Stair、G1-29DoF Push、现有 LOOP/generation 和执行环境固化为迁移基线，形成不再反复讨论的
ownership map。

**交付**：baseline inventory、当前 composition/task protocol/reward ABI/35-cell/runner argv/hash 快照、
三种执行体与三个 legacy/upstream task binding 的统一运行时回执、
Core V1 三个 cells / 七个 cell variants 定义，以及 AgentCore / TaskPlugin / RobotProfile/RuntimeAdapter / runner
权责表。

**严格验收**：

1. 捕获时 HEAD、origin 与服务器 provenance 一致；提交 Phase 0 文档后允许 live HEAD 改变，入口验收必须检查产品内容身份而非当前 Git SHA；已有未跟踪产物未被改动；
2. 当前 Stair、Push composition 和关键测试在未改语义下通过；服务器用统一字段集记录 G1-37、G1-29、Go2 的
   joint/body、默认姿态、物理尺度、actuator/PD、控制、policy I/O、传感器与完整资产闭包；G1-37 事实必须再与
   当前 Stair runtime protocol 交叉校验，不能只校验自有 hash；runtime 数值按 `1e-6` 量化并归一化负零，同一
   产品/环境连续两次探针的 profile/binding/runtime hash 必须完全相同；
3. 明确列出所有 G1-37/双足 hardcode，以及 legacy 35-cell 矩形网格对非规则评测套卷的结构性 hardcode；任何“当前产品已支持 roughness / Go2”或“`ab84aa3/f359356` 精确字节已重新完成
   15-generation formal”的错误表述为失败；`c582eca` 上的历史正式实验必须单独归属；
4. Core V1 固定为 3 个 cells / 7 个 cell variants，未获批不得在实施中扩成全矩阵；
5. 在服务器冻结的 Isaac Lab 2.3.0 环境中只读确认 Go2 asset、actuator/config 和可用 agent 基础；若缺失，
   Phase 5 在开始前先给出明确的资产补齐或 NO-GO 决策，不能暗中依赖 3.0 beta runtime。

**实施回执（2026-08-26）**：Phase 0 已通过，Phase 1 可以按冻结边界启动。权威产物为
[实际实施计划](P0/PHASE0_IMPLEMENTATION_PLAN_ZH.md)、
[机器基线快照](P0/PHASE0_BASELINE_SNAPSHOT.json)、
[运行时事实回执](P0/PHASE0_RUNTIME_PROBE_RECEIPT.json)、
[范围/权责/硬编码清单](P0/OWNERSHIP_SCOPE_AND_HARDCODES_ZH.md)、
[验收报告](P0/PHASE0_ACCEPTANCE_REPORT_ZH.md)与
[开发日志](P0/DEVELOPMENT_LOG_ZH.md)。该回执只证明迁移基线与二维边界已冻结，不表示 Phase 1 契约、Go2 Stair、
irregular/rough 地形或新策略性能已经实现。

### Phase 1：矩阵组合契约、共享角色词汇与两个 legacy G1 profiles（7–10 人日）

**目标**：建立最小 RobotProfile / RuntimeAdapter registry / RobotTaskBinding / ResolvedCellManifest 契约，
让现有 Stair 与 Push Recovery 以两个真实 G1 execution profiles 接入同一角色/能力词汇、compiler 和 LOOP，
不再用 `unitree_g1` 标签冒充物理身份相同。

**交付**：G1-37 Isaac Lab 与 G1-29 Unitree RL Lab 两份 RobotProfile/RuntimeAdapter、两份 legacy task binding、
required/provided capability compiler、单一
manifest compiler、`cell_variant_id` 解析、
`TrainingBindingV2(task_protocol_hash)` 与 V1 replay compatibility。

**严格验收**：

1. 同一 catalog 输入在两个全新进程中得到 byte-identical manifest 与相同 hash；任一 robot/task/binding/
   environment/schema 变化都改变 effective task protocol；
2. 两个 G1 profile 共享 schema、role vocabulary 和 compiler，并从 Phase 0 回执消费完整 asset closure、joint/body、
   默认姿态、形态尺度与 actuator/PD 事实，但因 asset/DoF/上游包不同而产生不同 RobotProfile/RuntimeAdapter hash；
   task-specific scanner mount、观测 projection/noise 留在 binding/TaskPlugin，不塞进 RobotProfile；互换 profile 必须得到
   `unsupported`，不能靠 family alias 通过；
3. 旧 `stair.g1_37dof.clean_legacy_v1` 与 `push.g1_29dof.legacy_v1` 默认调用、job materialization 和 replay 兼容；
   Push 还须通过一次现有 `run_live_loop` contract/dry-run 或权威 replay，证明它仍接共享 LOOP，但不要求 GPU
   性能改进，也不改变 legacy capabilities/verifier；
4. AgentCore、CandidateAuthoringLoop、runner、decision 和 ledger 的抽象接口没有因 robot 轴复制或分叉；
5. `robots/` 不 import Stair / Push 包；一条通用源码边界检查证明 TaskPlugin 不包含具体 joint/link 名、USD 路径
   或机器人上游包 import，只能引用 semantic roles；cell binding 不拥有训练预算、seed 或 acceptance；
6. 合法 schema 中缺失、越界或互斥的 semantic role/sensor/action capability 在 composition 阶段产生确定的
   `unsupported + reason`；`base + forbidden_contact` 同 target 必须合法，`support_effector + forbidden_contact`
   同 target 必须拒绝；重复声明等非法 schema 在 compiler 前由严格 loader 拒绝；不得依赖 raw-name 猜测或等到 Isaac 才报错；
7. train/eval child 严格重载 manifest；只保留三类代表性否定测试：错误 profile/manifest 在 composition 或启动前
   失败，错误 checkpoint/training binding 在 child 前失败，错误 session/ledger 在 replay 前失败；不枚举所有排列。

**实施回执（2026-08-26）**：Phase 1 已 GO。入口本地/远端 HEAD 均为 `4392aa2a6df2a45c7dd09456df35a8788ef41f6d`，仅作 provenance 记录；产品验收由 29 个文件的内容 hash、两个 legacy manifest、契约测试与永久 snapshot verifier 负责。权威产物为 [实际实施计划](P1/PHASE1_IMPLEMENTATION_PLAN_ZH.md)、[验收报告](P1/PHASE1_ACCEPTANCE_REPORT_ZH.md)、[开发日志](P1/DEVELOPMENT_LOG_ZH.md)、[机器回执](P1/P1_MATRIX_CONTRACT_SNAPSHOT.json) 与 [Push 共享 LOOP dry-run 原始回执](P1/PUSH_SHARED_LOOP_DRY_RUN/audit_summary.json)。B1–B7 及回执跨平台字节稳定性根因修复后，全仓 1496 tests 全部通过（29 skipped）；产品树 hash 为 `ef3b2c3316e041e0380b987e180f6cfe40f20051ef73aa1230dcced069a1a711`，机器回执 hash 为 `59eb454bdcd6c504d887dda38a71dca3643e3e4d1d532861d95395afa55732d3`。该回执只证明机器人轴、两格 legacy G1 组合契约、child/lineage 绑定与共享 LOOP 兼容已完成，不表示 Phase 2 形态无关 Stair runtime、Go2 或 C1/C2 复杂地形已经实现。

### Phase 2：Stair 任务语义与机器人形态解耦（5–7 人日）

**目标**：让 Stair 的 reward/evidence/evaluation 使用 binding 显式选择的 kinematic base 与可变数量 support effectors，先用 G1-37
证明抽象迁移没有改变已有行为。

**交付**：morphology-neutral Stair runtime projection、显式 kinematic-base binding、形态中立 evaluation semantics v7、VNext reward ABI、变长 support evidence、真实 surface
query 接口、G1-37 compatibility adapter 与 legacy ABI reader；同步把
`StairClimbingTaskPlugin.robot_requirements()` 的 `support_effector=(2,2)` 迁移为与已解析 profile 数量一致的
可变 cardinality contract。

**严格验收**：

1. 通用 Stair runtime、decoder、episode evidence 与 `StairClimbingTaskPlugin.robot_requirements()` 不再直接写
   G1-37 link/joint 名、`range(2)` 或最大数量 `2`；2 / 4 / 6 support-effectors contract test 全部通过，且 plugin
   不会在 runtime 已泛化后继续把 Go2/六足误报为 capability 不足；
2. G1-37 的实际 body/joint/sensor 解析全部来自对应 RobotProfile/RuntimeAdapter 与显式 binding，runtime summary
   回写解析结果和 schema hash；
3. 在预先冻结的 G1-37 clean cells 上，旧/新路径必须选择同一运动学 frame；位置/姿态/线角速度/projected gravity、observation/action shape、reference reward term、termination、逐 episode outcome 在声明的 exact/tolerance 规则内 parity；
4. G1-37 35-cell regression、历史 Genesis/ledger replay 与 reference reward program 均保持有效；
5. VNext 不为 Go2 伪造 arm/finger/torso 输入；不同 ABI 具有不同 input schema 与 capability profile identity。

**实施回执（2026-08-27）**：Phase 2 已 GO。入口 HEAD `5513076d5cbfe279987310c12c89938ff1a7691e` 只作 provenance；当前产品门由 29 个文件的 normalized-LF 内容身份、matrix/cardinality 契约、真实 runtime 回执与永久 snapshot verifier 组成。权威产物为 [实际实施计划](P2/PHASE2_IMPLEMENTATION_PLAN_ZH.md)、[验收报告](P2/PHASE2_ACCEPTANCE_REPORT_ZH.md)、[开发日志](P2/DEVELOPMENT_LOG_ZH.md)、[机器快照](P2/P2_MORPHOLOGY_DECOUPLING_SNAPSHOT.json) 与 [真实运行回执](P2/PHASE2_RUNTIME_PROBE_RECEIPT.json)。A–F 审计根因收敛后，全仓 1505 tests 全部通过（29 skipped，987.665 s）；产品树 hash 为 `148984daa4fa917f104bcb8a65bce348fc2c7760d472f24b11df1cb356bc4a11`，snapshot receipt hash 为 `077a3dc3a6c9113492ca90b92a8379f6cc65a2c29b45010ae45484b36dab8654`，真实 runtime receipt hash 为 `bea2c4678e82c10e2a220ab12077f3238979b2d5f4b5527834a4f2d34c963e26`。服务器实测只移除 legacy adapter 标识后，G1 的 7 组 kinematic-base 量最大绝对差全部为 0；同一有效 projection 有/无 manifest 的 hash 相同。Phase 1 verifier 因组合契约按计划迁移而被 P2 verifier 取代；历史 v6/V1 artifact 仍按其冻结源码/manifest 审计。该回执证明 Stair 核心已能按 profile/binding 消费可变 support 形态并保持 G1-37 物理/ABI parity，不表示 Go2、C1/C2 或 VNext candidate qualification 已完成。

### Phase 3A：C1 irregular、cell-local 门与首个复杂闭环前置（5–7 人日 + GPU）

**目标**：先建立 task-owned 的显式评测 cell-suite seam，再落地不等高/不等深但踏面仍平的 C1，让现有解析几何与 failure thresholds 保持精确；完成
G1-37 irregular 资格化后立即放行 Phase 4，不等待 roughness。

**交付**：兼容 legacy V5 的显式 ordered-cell suite contract、C1 generator、`stair.g1_37dof.irregular_v1` manifest、
resolved effective TaskSpec/EvaluationBinding、calibration/train/public_dev/public_accept/sealed_acceptance seed split、
cell-local acceptance/reference-gate hashes、C1 coverage report 与 Phase 4 job/GPU/重跑预算。

**严格验收**：

1. C0 的 G1-37 35-cell mesh、surface semantics、evaluation 与历史 replay 不变；V5 仍按原 7×5 读取，不被强制迁移；
2. VNext evaluation suite 以显式有序 cells 与 factor metadata 为权威，不从 height×tread 推导数量或用固定列宽切片；
   至少一份非笛卡尔、≤18-cell 的 C1 套卷完整通过 config → terrain assignment → episode evidence → public projection →
   prompt/feedback，所有模块报告相同 cell IDs/coverage，且任务特殊语义留在 Stair TaskPlugin；VNext TaskSpec/EvaluationBinding 不得携带或读取历史 `StairProtocolV1.policy_compatibility.robot_schema`，该字段只允许 legacy adapter/reader 消费；
3. `stair.g1_37dof.irregular_v1` 的 effective TaskSpec、EvaluationBinding 与 TrainingBindingV2 均使用同一个
   `resolved_cell_task_protocol_hash`；base hash 只作为 manifest ancestry。active acceptance/eval/reference-gate/cohort
   hashes 与 manifest 逐项一致后才允许建立 Genesis。Genesis composition hash 在创建/重开会话时由
   `prepare_session_genesis()` 强制，但 cheap runtime guard 不重算它；因此不得把“写入 Genesis”单独当成运行证据，
   child manifest 重载、checkpoint lineage、TrainingBindingV2 与 EvaluationBinding comparison domain 仍须同时闭合；
4. 同 manifest + seed 在独立进程产生相同 C1 geometry digest；runtime 独立重算逐级累计高度/深度边界、总高度
   与总长度，且每块踏面仍为平面，起点、riser/edge 与目标区不变；
5. C0 的 legacy/VNext decoder 保持 parity；C1 没有可比较的 legacy 路径，VNext 的逐级累计 surface 与独立
   geometry oracle 对同一 evidence 产生相同 support level/height 和 primary outcome/count，现有 `0.03 m`
   clearance/penetration 语义无需修改；
6. 在任何 LLM LOOP 前，只用 calibration 冻结 cell-local acceptance protocol 与 reference capability gate；不得
   继承 G1-37-clean 数值。随后用独立 `public_accept` 执行 reference gate、sealed_acceptance 做验证；通过则标记
   `cell_loop_ready=true`。失败则保持 false 并按 gate component 归因：只有 calibration 无法给出非平凡且可学习的
   合格范围时才判 terrain profile NO-GO；reference 单独失败则返回 reference qualification，不得降低地形或 acceptance 掩盖；
7. calibration/train/public_dev/public_accept/sealed_acceptance cells 与 seed 无交集；C1 operational suite、full phased eval 与未见 holdout
   分别受 §4.3 的 18 cells / 512 envs / 4 cells 上限约束；
8. 真实服务器 smoke 给出 steps/s、预计墙钟与输出量，Phase 4 的预算和停止条件在启动前冻结。

### Phase 3B：C2 roughness 与 failure-feedback 校准（5–7 人日 + GPU，可与 Phase 4 并行）

**目标**：在 C1 踏面内部加入有界微起伏，并证明 roughness 没有污染喂给 LLM 的 failure classification；
C2 失败不得推翻已经可用的 C1 复杂变体。

**交付**：C2 generator、`stair.g1_37dof.rough_v1` manifest、runtime local-surface evidence、decoder drift report、
C2 qualification/coverage report 与 C2 独立预算。

**严格验收**：

1. `max_abs_height_offset ≤ 0.015 m`，roughness 不改变起点、riser/edge 拓扑、名义 tread 或顶端目标区；
2. 同 manifest + seed 的 mesh/height digest 可复现，不同 realization seed 的 samples 确实不同；runtime samples
   可独立重算 roughness 幅值、空间尺度和坡度，height scanner samples 相对 mesh oracle 的误差在冻结容差内；
3. production local-surface query 对冻结 support footprint 满足 `ε_surface ≤ 7.5 mm`；固定 evidence 在判据带外的
   support/clearance/diagnostic/failure/success signals 对独立 oracle 零分歧，带内 ambiguity 只作离线统计；
4. classifier receipt 通过前，全部 classification-derived public feedback 在 behavior/prompt projection 前被阻断；
   最大局部坡度下 world-XY slip proxy 与 local-tangent oracle 的误差在冻结容差内。通过后再报告
   fixed-policy/matched-seed 的 C1→C2 failure distribution；
5. C2 factors/levels/非法组合冻结后覆盖所有可行二元组合，operational suite 与 eval 数仍在 §4.3 上限内；
6. C2 单独冻结 acceptance/reference gate hashes 并完成 G1-37 reference 资格化；任一项不满足则
   `cell_loop_ready=false` 并冻结原因：geometry/classifier 失败归 C2 environment/decoder，reference-only 失败归
   reference qualification；不能放宽 failure threshold、降低 acceptance 或改写 C1 结果；
7. train/eval child 回传 runtime-derived terrain/material/decoder identity；摩擦若保留，只通过 material receipt
   单独验证，不和 geometry roughness 混为一个难度数；
8. C2 真实服务器成本在独立批准预算内；Phase 4 固定在 Phase 3A 的 exact source commit/checkout 启动，Phase 3B
   可在独立 worktree 开发但不得同步进正在运行的 checkout，也不修改该
   session 的 C1 manifest/reward ABI/Genesis/ledger，任何共享 decoder 改动先保持 C1 parity。无论 C2 结果如何，
   不中断已经启动的 Phase 4 C1 session。

### Phase 4：G1-37 × irregular Stair 接入最新 generation LOOP（8–12 人日 + GPU）

**目标**：把 resolved cell manifest 接入当前 P6-D generation foundation，并以唯一的 non-formal
1-generation pilot → fresh 3-generation validation 路径证明系统能产生可审计的接受、拒绝或无选择闭环；legacy
`run_live_loop` 只做兼容 smoke，不成为第二种性能验收权威。

**交付**：现有 generation runner/launcher/launch-context/readiness/Genesis 链的固定枚举
`pilot_1 / validation_3 / formal_15`、cell-bound receipts、`stair.g1_37dof.irregular_v1` 1-generation pilot、
独立 3-generation validation、variant-owned budget profile、ledger/replay report。

**严格验收**：

1. 1/3/15 profile 复用同一 generation foundation、composition、TaskPlugin、CandidateAuthoringLoop、runner、
   selector 与 ledger 协议；不得新增 scheduler、第二 LOOP 或 G1 专用分叉；
2. production runner 与 launcher 只接受固定枚举，拒绝任意 `max_generations`；3-generation 被 launch-context
   识别为可追踪/续行的 multi-generation，旧 `--formal-session` 15 代入口与历史 replay 保持兼容；
3. `cell_variant_id + profile + manifest path/hash` 同时进入 Genesis creator、launcher command/readiness receipt
   与 child train/eval；任何 mismatch 在 provider 调用或 AppLauncher 前拒绝；
4. 先完成 1-generation 工程闭合路径（不要求 accepted），按实测 walltime/output/control steps 形成候选 budget
   profile；经独立评审冻结
   后，使用 fresh session / Genesis 启动 3-generation，不能在看到结果后扩大同一 session 预算；该 profile
   必须真正替代当前 G1-37-only cost 约束，失败 reserve 沿用现有保守公式推导，不要求故意制造故障；
5. 每代两条 lane 使用完全相同的 cell manifest、预算和评测协议；每个候选都有权威 phased-eval 结果或明确
   skip reason，只有通过前置门的候选进入下一阶段，被接受者必须完成全部规定阶段；
6. 3-generation 必须恰好产生 3 条 authoritative generation/ledger records；至少一个 verified candidate 完成
   train + public eval 且 reward program 有 runtime 调用证据。current-best replay 与 ledger 末端一致；若有 accepted，
   checkpoint 来自同一 record 的本地路径与 checksum；若没有，head 合法保留 Genesis reference；
7. `loop_verified` 只表示工程闭环。若至少一次完整 acceptance 更新 current-best，则另记
   `improvement_observed=true`；若没有，保留真实失败结果，不把 LLM 调用或 reward curve 写成改进有效；
8. unseen_final_holdout 此时不得运行，避免被后续选择间接使用。

### Phase 5：Go2 机器人行、官方 reference 与 Go2 × Stair 资格化（8–11 人日 + GPU）

**目标**：以官方资产基础较完整的 Go2 证明新增机器人行不需要复制 TaskPlugin、入口或 LOOP；优先移植官方
Go2 rough locomotion reward，完成 clean → irregular，并在 Phase 3B 通过后完成 rough 的固定预算资格验证。

**交付**：Go2 RobotProfile/Isaac RuntimeAdapter、Go2 × Stair binding、cell-owned action/observation/agent cfg、
由 Isaac Lab 2.3.0 effective Go2 reward 移植并资格化的完整 reference program，以及
`stair.go2.clean_v1` / `stair.go2.irregular_v1` / `stair.go2.rough_v1` manifests。

**严格验收**：

1. Go2 接入的共享代码改动不触及 AgentCore、ledger、decision 或新建 train/eval 脚本；新增内容限于机器人行、
   单元格 binding 和必要的 task-neutral runtime seam；
2. 真实 Isaac smoke 解析 asset、4 个 support feet、forbidden contacts、sensor mount、action/observation shape、
   agent cfg 与 reward ABI，所有 identity 与 manifest 一致；Go2 RobotProfile 的 `forbidden_contact` 候选目标全集
   必须至少覆盖 P0 已确认存在的 `FL/FR/RL/RR_thigh` 与 `FL/FR/RL/RR_calf` 8 个 body（不是同名 joint），cell binding 再按 Stair/官方 reference 语义选择
   实际禁碰子集，不能把 task-specific 禁碰判据写死在共享 profile，也不能因 profile 漏列目标而产生
   `role_target_outside_profile`；
3. argv、summary、checkpoint metadata 中不得残留 G1-37 gym id、37-DoF、左右 ankle/torso 等错误身份；
4. reference candidate 冻结 Isaac Lab `v2.3.0` commit
   `3c6e67bb5c7ada942a6d1884ab69338f57596f77`、相关 source digests，并解析父类 reward 与 Go2 overrides 的完整
   effective terms；source、weight、参数、调用和逐 term/total 数值满足预先冻结的 absolute/relative tolerance
   （默认沿用现有 G1-37 等价工作的 `1e-6`）。若 Stair gate 不通过，只允许 §3.4 的一次有界 fallback；
5. clean qualification 先通过，再运行 irregular；rough 只在 Phase 3B 通过后资格化。每个变体均完成真实训练、
   本地 checkpoint+checksum、phased eval 和 runtime activation evidence，但 rough 不重复要求 3-generation；
6. child runtime 重载 manifest 并回传 runtime-derived Go2/terrain/schema identity；错误 profile 或 support roles
   使用 Phase 1 的同一 contract suite，在训练前失败；
7. Go2 qualified suite 用独立 calibration seeds 按 Phase 3A/3B 的 easy/target/hard 规则冻结；获得真实 Go2 结果后，
   common physical anchors 在任何 Go2 LOOP 前最终冻结并单独报告，不用 G1-37 绝对难度强迫 Go2；
8. clean/irregular/rough 分别冻结 acceptance protocol 与 reference gate；不得继承 G1-37-clean 的 `0.2` 或
   G1-37 low-step bins。Go2 reference artifact 绑定本 cell 的 task protocol/reward ABI/capability，verifier 在训练前拒绝 G1-37
   ankle/arm/37-DoF 字段；
9. runtime/asset/control 通过但 reference 或 terrain 不可学习时，分别报告 `robot_runtime_qualified` 与
   `cell_loop_ready=false`；不能复制 G1-37 结果或降低评测语义冒充支持。

### Phase 6：Go2 × irregular Stair generation LOOP 与矩阵级收口（6–9 人日 + GPU）

**目标**：让 Go2 在自己的 C1 irregular Stair 上运行同一 LLM LOOP，评估是否产生 reward/policy 改进，并完成
两个 Stair 单元格的矩阵级工程验收。

**交付**：Go2 独立 1-generation pilot、冻结 budget 后的 fresh 3-generation validation、两机器人 cell reports、
matrix coverage report，以及 Phase 1 identity contract suite 在两个 Stair cells 上的一次实例化结果。

**严格验收**：

1. Go2 满足 Phase 4 相同的 1-generation 成本门、fresh 3-generation、authoring、phased gate、runtime
   activation、ledger replay 条件；
2. G1-37/Go2 只通过不同 `cell_variant_id` 进入同一 generation 入口，不存在机器人专用 loop/launcher 脚本；
3. Phase 1 的三类代表性 identity tests 在 G1-37/Go2 cells 上通过，不再枚举 Genesis/reference/history/checkpoint/
   summary/ledger 的全排列；
4. 两条谱系各自比较 candidate 与本机器人 current-best；common anchors 只作描述性表格，不生成跨机器人 winner；
5. matrix report 中 G1-37/Go2 irregular cells 达到 `loop_verified`，并分别报告 `improvement_observed`；G1-29 Push
   达到 contract/replay parity，不要求 GPU 性能结论；
6. 若任一 cell 没有 accepted improvement，工程闭环仍可通过并记录 `improvement_observed=false`；不得把
   “LLM 被调用”写成“LLM 改进有效”，也不得结果驱动增加 generation 数；
7. 每个 cell 的 child summary 均回传对应 runtime robot/terrain identity，current-best 与 ledger replay 闭合。

### Phase 7：矩阵报告、条件式未见 holdout 与 formal readiness（3–5 人日 + 条件式 GPU）

**目标**：在 G1-37/Go2 3-generation session 结束后完成矩阵、谱系、成本和研究证据分级；只有准备提出
性能/泛化结论且确有 frozen improved head 的 cell 才运行一次未见 holdout，本 Phase 不再 authoring 新候选。

**交付**：两条 final-head freeze receipts、必要时的一次性 unseen_final_holdout results（否则明确 `not_run`）、
matrix coverage/cell reports、ledger replay closure、onboarding guide，以及 formal 15-generation / 多 session
研究的预注册模板。

**严格验收**：

1. 每个机器人冻结唯一 final head；若 `improvement_observed=false`，Core 工程 GO 不要求额外 holdout GPU，报告
   `not_run_no_performance_claim`；
2. 若运行 holdout，先冻结 policy/program/checkpoint checksum，且 namespace 此前未被训练、prompt、memory、
   acceptance 或 current-best 选择使用；运行后不得继续该 session authoring 或换 checkpoint；
3. matrix report 同时区分 3 个 cells、7 个 variants、`runtime_qualified`、`cell_loop_ready`、`loop_verified`、
   `improvement_observed` 和 `formal_evidence`，并只索引权威 Genesis/runner/ledger artifacts；
4. G1-37/Go2 的 current-best、checkpoint、memory 和 ledger replay 各自闭合，不在本 Phase 重跑 Phase 1/6
   已通过的 identity suite；
5. 单次 3-generation 与可选 holdout 只称 cell-level 工程/初步性能证据。即使完成 3 个独立 session seeds，也只
   称重复的初步证据；formal scientific claim 必须在 pilot 后按目标区间精度预注册样本量、paired effect、
   interval 与通过标准，且不得查看正式结果后继续加 seed 直到显著；
6. formal 15-generation 只允许通过 3-generation、预算和 final-readiness 审查的少量代表 cell 使用 fresh
   session/Genesis；它不是 Core V1 完成条件，不复用当前 G1-37-only 成本常量，也不能拼接旧 ledger。

## 7. Core V1 后的受控扩展

### Extension A：C2 rough Stair 的独立 LLM campaign（每个机器人 4–6 人日 + 固定 GPU 预算）

**目标**：在 Phase 3B classifier receipt 与 Core V1 工程 GO 后，补齐“LLM 在 rough Stair 上运行并评估改进”的
研究证据；不把该昂贵 campaign 伪装成矩阵基础设施前置条件。

**验收**：先对 G1-37 rough 运行 1-generation pilot → fresh 3-generation；成本通过后才单独授权 Go2 rough。
两者复用 Core 的 manifest/LOOP/runner，不修改 failure、acceptance 或 terrain protocol，各用新 Genesis/ledger；
达到 `loop_verified` 即完成工程目标，分别报告 `improvement_observed`，不得因无 accepted 而追加 generation。

### Extension B：Go2 × Push Recovery（5–8 人日 + GPU，独立授权）

**目标**：新增一个真正的第二机器人 × 第二任务单元格，验证新增 cell 只需薄 binding 与 task-owned push
environment profile。

**验收**：Go2 Stair 与 Go2 Push 的 `robot_profile_hash` 相同且 `runtime_adapter_protocol_hash` 相同，真实 runtime
回传相同 asset/joint/actuator/capability digest；task id/TaskPlugin protocol、RobotTaskBinding、task-owned
environment/eval/reward semantics 与最终 resolved task protocol 必须按任务不同。policy observation/action/agent
只要求各自冻结并与 binding 一致，不强制为相同或不同。Go2 × Push 在 materialize 前必须以 manifest 中的
observation/action content hash 作为 backend 执行身份，禁止沿用 `push_recovery_obs_v1` / `unitree_actions_v1`
占位标签；这两个标签只允许留在 G1-29 legacy reader/path。不复制 Go2 机器人行或 Push TaskPlugin，不改 AgentCore；固定 train/eval
资格化后完成 TaskPlugin 支持的现有多轮 LOOP，并分别报告 `loop_verified` 与 `improvement_observed`；与 Go2 Stair、
G1-29 Push 的策略谱系完全隔离。若必须为 Push 修改 Go2 profile 本体或复制入口，强复用证明失败。

### Extension C：六足机器人行（8–12 人日 + GPU，先做资产 feasibility）

**目标**：验证 `support_count=6`，而不是追求一份跨形态 policy。

**验收**：只新增 Hexapod RobotProfile/RuntimeAdapter、Stair binding，以及 cell-owned agent/reference/variants；
不复制 Stair/LOOP；真实 asset/control/contact/observation 资格通过后，依次完成 clean → irregular → rough 和预注册多轮
LOOP。若没有可靠资产、控制模型或可资格化 reference，则以 NO-GO 收口，不创建占位机器人。

### Extension D：LLM curriculum capability（4–6 人日 + 独立实验）

**目标**：在 Stair reward-only cells 稳定后，验证“候选权限可以逐项扩展，但训练/评测语义仍由我们掌握”。

**验收**：默认保持关闭；只有声明支持的 TaskPlugin/cell 能公开一个有界 curriculum candidate schema；verifier
拒绝越界字段，runner 将候选编译为固定总预算训练配置；eval/seed/acceptance 完全不变；reward-only 对照与
reward+curriculum 实验使用相同预算。不得借此开放自由 Python runner 配置或通用 curriculum engine。

## 8. 研究与工程依据

本计划只采用能直接降低当前工程风险的一手资料结论：

| 资料 | 可采纳结论 | 本计划中的落点 |
|---|---|---|
| [Isaac Lab：Manager-Based RL Environment](https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/03_envs/create_manager_rl_env.html) | 官方建议通过 `ManagerBasedRLEnvCfg` 配置任务，而不是修改环境基类；任务规格可与环境实现分离 | TaskPlugin 保持任务列；RobotProfile/RuntimeAdapter 只供物理与 raw runtime，composition 做组合 |
| [Isaac Lab task package structure](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/source/isaaclab_tasks/docs/README.md) | 官方任务命名/目录显式区分 Task 与 Robot，并把 agent config 放在 task 下的 robot config 中 | 机器人行与任务列分开；最终 policy/agent cfg 属于 cell，不复制完整 task |
| [Isaac Lab 2.3.0 rough-locomotion 父配置](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)、[Go2 rough overrides](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py)、[Go2 RSL-RL config](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py) | Go2 继承共享 rough-locomotion task，并覆盖 asset、sensor、terrain/action scale 与有效 reward weights/contact terms；同版本有固定 agent 基础 | RobotTaskBinding 保持薄；Phase 5 解析父类+overrides，优先移植完整官方 effective reward，并冻结 cell-owned agent cfg |
| [Isaac Lab：Adding a New Robot](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html) | 新机器人首先应形成稳定 ArticulationCfg / actuator / spawn 配置 | RobotProfile/RuntimeAdapter 先独立资格化，再接 TaskPlugin |
| [Isaac Lab TerrainGeneratorCfg](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.terrains.html) | terrain seed、difficulty、curriculum 与缓存可配置；确定性 seed 是复现基础 | Task-owned complex generator 与 manifest；另用 runtime mesh evidence 证明真实激活 |
| [Isaac Lab terrain curriculum](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py) | 简单的按表现升降 terrain level 已是成熟工程模式 | TaskPlugin 定义、composition 冻结、runner 编译执行，不引入自动 teacher |
| [Rudin et al., 2022](https://proceedings.mlr.press/v164/rudin22a.html)；[Lee et al., 2020](https://arxiv.org/abs/2010.11251) | 大规模并行 + 简单/自适应 terrain curriculum 可提升复杂地形学习效率；一开始均匀采样过难地形会浪费训练 | clean→irregular→rough 的资格门和有界 curriculum |
| [Procgen, 2020](https://proceedings.mlr.press/v119/cobbe20a.html) | 程序化多样性必须有未参与选择的 test levels 才能讨论泛化 | sealed_acceptance 与一次性 unseen_final_holdout 分开 |
| [NIST IR 7878](https://csrc.nist.gov/pubs/ir/7878/final) | covering arrays 可压缩多因素组合，同时量化 t-way coverage | 由冻结因素模型生成并证明 100% 可行二元覆盖，不预设“12 个即完整” |
| [Agarwal et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | 少量 RL runs 的点估计不可靠，应报告逐 run 与不确定性 | 3 seeds 仅为初步重复证据；formal 样本量与区间精度须预注册，禁止可选停止 |
| [MetaMorph](https://openreview.net/forum?id=Opmqtk_GvYL)；[GenLoco](https://proceedings.mlr.press/v205/feng23a.html) | morphology-conditioned universal policy 是需要统一表示与新训练范式的独立研究问题 | 明确延后；矩阵共享系统，不共享 policy 权重 |

表中 2.3.0 源码可用于 exact-source 迁移；develop / 3.0 beta 链接只验证设计方向。实际代码、测试与服务器运行
必须服从本项目冻结的 Isaac Lab 2.3.0 `execution environment manifest`。若未来升级版本，应另立兼容性 Phase，
不与本计划混做。

## 9. 总体验收与停止规则

### 9.1 Core V1 工程 GO 与策略结果

Core V1 的工程 GO 只需要：

- Phase 0–6 全部通过：G1-37/Go2 irregular 各自完成 1-generation pilot 和 fresh 3-generation validation 并达到
  `loop_verified`；Phase 7 的矩阵、ledger/replay 与证据分级报告闭合；
- G1-37/Go2 rough variants 达到 `cell_loop_ready=true`（其定义已包含 `runtime_qualified`），C2 decoder/feedback 校准通过，但不重复要求
  3-generation；G1-29 Push 保持 legacy contract/replay parity；
- 两个 G1 execution profiles 明确不同，只共享 RobotProfile schema、role vocabulary 与 compiler；Core V1 不虚报同一 concrete G1 row
  已跨任务复用，强证明留给 Extension B 的 Go2 × Push；
- resolved manifest → task protocol → Genesis → training/evaluation summary → decision → ledger/replay lineage 闭合；
- Core V1 仍只有 3 个 cells / 7 个 cell variants；共享同一 AgentCore/authoring/runner/ledger 协议，每个 session
  只有自己的 ledger 文件，不存在全局 winner 或第二套矩阵状态机；
- 文档明确区分 runtime qualified、loop verified、improvement observed 与 formal scientific evidence。

策略结果不再作为工程 GO 门：两个 Stair cells 分别从 ledger 派生 `improvement_observed`。两者为真，只能称
“在两格观察到初步策略改进”；任一为假，仍可通过 Core 工程 GO，但必须报告真实结果，不能声称 LLM 已改善
策略。`formal_evidence` 仍需另行预注册，不能由一次 accepted 或一次 holdout 替代。

### 9.2 返回上一 Phase

以下情况直接返回最近的责任层，不增加旁路：

- G1-37 parity 失败：返回对应 RobotProfile/RuntimeAdapter / Stair projection，不继续 Go2；
- roughness 未在 runtime mesh/evidence 中激活：返回 Stair environment generator；
- Go2 资产、控制或 reference 不稳定：返回 Go2 RobotProfile/RuntimeAdapter 或 cell qualification，不修改 Stair
  success/eval 掩盖问题；
- prompt/reward ABI 错误：返回 TaskPromptAdapter / binding，不修改 LLM provider；
- checkpoint/history 可跨 cell 混用：返回 manifest/task protocol/training binding，不增加第二 ledger；
- generation 成本超预算：缩小获批实验或停止，不弱化训练/评测协议。

### 9.3 结果型停止

- 3-generation 生命周期闭合、且满足 §5.2 全部 `loop_verified` 条件，但无 accepted improvement：关闭并保留该 session，记录
  `loop_verified=true, improvement_observed=false`；不返回架构 Phase，也不追加同一 session 的 generation；
- 只有发现可归因的 ABI、runtime、terrain 或评测实现缺陷，才按 §9.2 返回责任层；
- 若形成新的研究假设，使用新 manifest/protocol/session 和单独批准的固定预算，不能以“再试几代”替代设计；
- 有 accepted improvement 时也只冻结 head 并按需运行一次未见 holdout，不自动升级为 formal claim。

### 9.4 成功后的新增规范

完成后，任何新机器人、新任务或新单元格 PR 都必须给出：

1. 新增的是行、列还是单元格 binding；
2. 它复用了哪些已有 RobotProfile/RuntimeAdapter / TaskPlugin / LOOP；
3. resolved manifest 与 capability profile；
4. runtime qualification 和必要的 LOOP 证据；
5. 未支持矩阵单元格及原因。

只要这五项能由 registry、manifest 和权威运行产物回答，系统就可以有序扩展，而无需为每个机器人或任务重建
一套自进化工程。

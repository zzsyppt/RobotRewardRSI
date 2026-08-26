# Phase 1 实际实施计划：矩阵组合契约、共享角色词汇与两个 legacy G1 profiles

> 实施日期：2026-08-26  
> 基线入口：Phase 0 产品内容验收通过；当前主线 `4392aa2`  
> 本阶段范围：只抽出机器人执行轴并迁移两个 legacy cells，不实现 Go2 Stair、形态无关 Stair runtime 或复杂地形

## 1. 目标与完成定义

本阶段把当前系统从“Task composition 内隐含机器人”收敛为：

```text
RobotProfile + RobotRuntimeAdapter
              × TaskPlugin
              × RobotTaskBinding
              × task-owned environment descriptor
              -> ResolvedCellManifestV1
              -> 现有 ResolvedTaskComposition / LOOP / runner / ledger
```

完成时必须同时满足：

- G1-37 Isaac Lab 与 G1-29 Unitree RL Lab 是两个精确、内容绑定的 execution profiles；
- 两者共享同一套小型 semantic-role / capability 词汇、registry 和 compiler，但不共享 profile hash；
- Stair 与 Push 仍各自拥有任务目标、reward/eval/acceptance、环境和 policy projection；
- 两个 legacy cell 通过原 TaskPlugin、原 backend 与原 LOOP，不复制入口、runner 或 ledger；
- 后续新增机器人只增加 profile/runtime adapter，新 cell 只增加 task-owned binding；AgentCore 不感知机器人。

## 2. 结合当前系统后的实施决策

### 2.1 机器人层只保存可复用的执行事实

`RobotProfile` 消费 Phase 0 真实回执中的完整 joint/body、默认姿态、形态尺度、actuator/PD 和资产闭包，并增加：

- 精确 `robot_profile_id` 与仅供报告的 `robot_family_id`；
- 最小共享角色：`base`、`support_effector`、`forbidden_contact`、`joint_group.*`；
- task-neutral provided capabilities。

它不保存最终 policy observation/action、agent cfg、height scanner、奖励、地形、成功判据或评测。

`RobotRuntimeAdapter` 是内容哈希的 backend descriptor，声明 profile、runtime provider、上游 revision 和 articulation source；本阶段不建立第三套 RobotPlugin 生命周期，也不把现有 Task EnvCfg 移入机器人层。

### 2.2 task requirements 与 concrete mapping 在组合边界相遇

`TaskPlugin.robot_requirements()` 只声明共享角色和 opaque capability ID，不出现关节/link/USD/上游包名。具体 body/joint、sensor mount、action scale、observation/noise 与 policy-I/O 事实全部留在对应 `RobotTaskBinding`。

公共 binding/compiler 不识别 `phase0_binding_facts` 等迁移字段，只把 `task_projection` 作为 task-owned、内容绑定的不透明对象。它与 Phase 0 外部回执的逐字段一致性由任务配置验收负责，避免把一次迁移过程固化成公共语义。

compiler 不按名字猜测：profile 不匹配、role 缺失/重复/互斥、sensor/action capability 缺失均返回稳定的
`unsupported + reasons`，不会拖到 Isaac 启动后才失败。target 不做全局唯一；`base + forbidden_contact`
可以映射同一躯干，只有 `support_effector + forbidden_contact` 同 target 被公共词汇判为冲突。

### 2.3 legacy 身份兼容与 V2 边界

主计划要求 `TrainingBindingV2(task_protocol_hash)`，同时要求 G1 legacy 默认调用、reference/Genesis 和 replay 不退化。当前 Stair V2 协议已冻结 V1 training-binding hash，并在 session-start/reference 链中交叉校验；直接强改为 V2 会重写历史身份，不是机器人轴抽取。

因此本阶段采用一个显式且收敛的兼容规则：

- reader 同时严格解析 `training_binding.v1` 与 `.v2`；V2 将 `task_protocol_hash` 纳入 binding hash；
- `clean_legacy_v1` / `legacy_v1` 两格通过绑定 exact profile/adapter/binding/environment/requirements/base protocol 的
  `legacy_training_binding_v1_grant_hash` 保持原 job binding 与历史 replay；布尔降级开关不存在，变更后的 cell 复制旧 grant 必须失败；
- 所有未来非 legacy cell 的 compiler 必须要求 V2，缺少 `task_protocol_hash` 直接失败；不得再增加第三种模式。

新生成的 legacy composition 仍把 manifest 绑定进 Genesis snapshot、train/eval argv 和 child summary；历史 artifact reader 继续识别没有 manifest 的旧记录。manifest 中同时区分冻结的 `base_task_protocol_hash` 与由 cell 输入派生的 `resolved_cell_task_protocol_hash`，不伪装二者相同。
两个 legacy TaskSpec/EvaluationBinding 为保持历史 comparison identity 继续使用 base hash；resolved hash 在 Genesis
`resolved_cell` 与 manifest/checkpoint lineage 中生效。首个 non-legacy cell 将 resolved hash 提升到 effective
TaskSpec/EvaluationBinding 的责任明确归 Phase 3A，不在 Phase 1 添加无运行消费者的分支。

### 2.4 只采用 Codex 中适用的三点

参考 `D:\humanoid_agent\codex` 当前提交 `5ce04789`：

1. manifest 先按严格 schema 解析，再把相对/外部实现解析成唯一 resolved descriptor；
2. registry 对稳定 ID 做唯一登记，重复 ID 在组合根失败；
3. discovery/descriptor 与 execution 分离，运行层只消费已解析、内容绑定的对象。

不采用 Codex 的插件安装、权限沙箱、远程同步、服务状态或并发管理系统。

## 3. 工作包

### P1-A：机器人轴与统一 registry

实现不可变 RobotProfile、RobotRuntimeAdapter、role/capability requirements 和单一 RobotRegistry；从版本控制下的两份 JSON profile 加载 Phase 0 全量事实，并逐份复算 P0 runtime-facts hash、profile hash 与 adapter hash。

### P1-B：cell catalog、binding compiler 与 manifest

实现 versioned `cell_variant_id` 的闭集 catalog、两份 task-owned legacy bindings、确定性的 capability compiler 与 `ResolvedCellManifestV1`。manifest 使用 canonical JSON，无自引用 hash；两个全新进程必须生成完全相同的字节。

### P1-C：接入两个现有 composition

Stair 与 Push provider 各自只解析自己的 legacy catalog entry，调用同一 compiler，并把 resolved manifest 附着到现有 `ResolvedTaskComposition`。现有 TaskPlugin、candidate capability、backend、训练预算、seed、acceptance 与环境语义保持原所有者和原值。

### P1-D：runner/child 身份传递与 TrainingBindingV2

在现有 IsaacLab/RSL-RL backend 的共享 job materializer 中传递绝对 manifest path/hash；四个 legacy train/eval child 在 AppLauncher 前严格重载并回传 cell identity，runner 对回传 hash 做一致性校验。实现 V2 reader/writer 与 V1 replay；不新建机器人专用入口。

### P1-E：严格验收、报告与清理

执行 deterministic/tamper/unsupported tests、两个 legacy composition/job parity、Push 共享 LOOP dry-run、代表性跨 cell lineage 否定测试、目标回归与源码边界审计。提供可重复的 Phase 1 snapshot verifier；对内部按原始源码字节绑定的 LOOP 回执，以明确 Git 属性保证跨平台检出不改写证据，再同时校验外层闭包与 sealed candidate 内部身份。只保留能证明本阶段边界的测试。

## 4. 严格验收门

| 门 | 通过条件 |
|---|---|
| P1-G1 确定性 | 两个全新 Python 进程对两格分别输出 byte-identical manifest；任一 profile/adapter/binding/environment/schema 输入变化均改变 manifest 与 resolved cell protocol hash |
| P1-G2 真实 profile | 两份 profile 完整复算 Phase 0 runtime facts；37/29 DoF、asset closure、上游 revision、profile/adapter hash 均不同；family alias 不参与兼容判定 |
| P1-G3 compiler | 正确组合为 `supported`；有效但不兼容的 profile、缺失/越界/互斥 role、缺少 capability 均得到确定的 `unsupported + reason`；`base + forbidden_contact` 重叠合法，`support_effector + forbidden_contact` 重叠拒绝；重复声明等非法 schema 在 compiler 前由严格 loader 拒绝，且全程不 import Isaac |
| P1-G4 legacy parity | Stair clean 与 Push legacy 的 TaskSpec、candidate capability、预算、seed、acceptance、policy-I/O hash 和原 entrypoint 保持一致；历史 TrainingBindingV1/Genesis/ledger reader 继续通过 |
| P1-G5 LOOP 复用 | Push 至少完成一次现有共享 LOOP contract dry-run/replay；原始证据在 Windows/Linux Git 检出后逐字节不变，自动 verifier 能复算闭包与 sealed candidate；AgentCore、authoring、decision、ledger 没有 robot 分支或复制实现 |
| P1-G6 child binding | 新 materialized train/eval argv 均携带 manifest 绝对路径/hash；child 在 AppLauncher 前拒绝 path/hash/task/profile tamper；runner 拒绝不回传或回传错误 manifest identity 的 summary |
| P1-G7 lineage | V2 training binding 拒绝错误 task protocol；legacy V1 只接受 exact content-bound grant；代表性错误 checkpoint/cell、错误 Genesis/session ledger 在 child/replay 前失败，不做全排列 |
| P1-G8 所有权 | `robots/` 零 task import；TaskPlugin 源码零具体 joint/link/USD/上游包；binding payload 零训练预算、seed、acceptance；Stair/Push 任务测试无退化 |

## 5. 停止与回退规则

- legacy parity 失败：回到 profile/binding/composition 根因，不修改旧任务阈值或 runner 语义掩盖；
- 为接入 profile 需要在 AgentCore、decision、ledger 增加机器人判断：判定抽象位置错误并停止；
- profile 事实与 Phase 0 回执不一致：不手改 hash，返回 profile 数据源；
- child manifest 无法在 AppLauncher 前验证：P1 不通过，不把父进程标签当运行证据；
- 不在本阶段顺手实现 Go2、C1/C2、变长 support runtime、通用 TerrainPlugin 或新的安全框架。

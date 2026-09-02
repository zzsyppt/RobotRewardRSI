# Phase 0 二维范围、权责与迁移债务清单

## 1. Core V1 范围冻结

Phase 0 冻结的是稀疏矩阵，不是完整笛卡尔积。

| execution-level cell | Phase 0 事实 | Core V1 变体 |
|---|---|---|
| `stair.g1_37dof` | 当前真实 Stair 执行路径；G1-37、310-D policy obs、37-D action、35-cell eval | `clean_legacy_v1`、`irregular_v1`、`rough_v1` |
| `push.g1_29dof` | 当前真实 Push 路径；厂商 G1-29 URDF 与 29-D all-joint action | `legacy_v1` |
| `stair.go2` | 冻结 runtime 已有 Go2 asset/config/agent，但项目 binding 与 Stair 语义尚未实现 | `clean_v1`、`irregular_v1`、`rough_v1` |

总数严格为 3 cells / 7 cell variants。`Go2 × Push` 是后续跨任务复用强证明，六足是未来新行；二者都不属于本轮实施。

## 2. 权责冻结

| 层 | 拥有 | 禁止拥有 |
|---|---|---|
| AgentCore / generation LOOP | generation 生命周期、LLM 调用、公开反馈流、通用 candidate 流程、opaque identity 传递 | 机器人、关节、脚、楼梯、roughness、Push、任务成功/失败语义 |
| TaskPlugin | 任务成功/失败、任务环境变体、评测集合、acceptance、prompt/memory 投影、候选权限、task protocol | 具体机器人 USD/URDF 路径、具体关节/link 名、上游机器人包 import |
| RobotProfile | 精确资产身份、关节/刚体/执行器 schema、控制时序、可提供的最小 semantic roles/capabilities | Stair/Push、地形、奖励、评测、成功判据、PPO agent 配置 |
| RuntimeAdapter | 在指定 backend 加载资产并解析真实 schema，把 profile 编译成 backend 配置 | 任务结果语义、LLM 权限、acceptance |
| RobotTaskBinding | 校验 task requirements 与 robot capabilities；为一个 cell 选择/映射 action、observation、agent cfg 和 reward-input projection | 复制 TaskPlugin、复制 runner、复制 LOOP；定义新的任务成功语义 |
| task-owned EnvironmentProfile | 对应任务的环境难度；Stair 拥有阶高/阶深/roughness，Push 拥有推力/方向/接触面 | 伪装成跨任务通用 TerrainPlugin |
| ResolvedCellManifest | 一次解析后的不可变完整身份，供 Genesis、runner、replay 与报告绑定 | 运行时服务、可变 registry、全局 current-best |
| runner | 强制执行已解析的 argv、预算、种子、checkpoint、训练/评测与 receipt | 自行猜测 robot family、修改任务协议、允许 LLM 改 PPO/评测/接受门 |
| LLM | 当前 Stair cell 内完整 reward program；Push legacy 保持原 capability | 机器人资产、PPO、预算、种子、checkpoint、地形、课程、评测、acceptance |

组合边界固定为：

```text
(RobotProfile + RuntimeAdapter)
  × TaskPlugin
  × RobotTaskBinding
  × task-owned EnvironmentProfile
  -> ResolvedCellManifest
  -> existing LLM LOOP
```

其中 Stair 的“上楼方向、楼梯几何、surface decoder、roughness、成功/失败、分类阈值、35-cell/未来复杂 cell、acceptance”必须全部留在 Stair TaskPlugin；Push 的推力、恢复与直立判据必须留在 Push TaskPlugin。共享 role 只能描述物理能力，不能携带这些任务含义。

## 3. 当前 G1-37 / 双足硬编码：迁移阻塞项

以下按语义 seam 穷举当前产品执行路径。历史报告中的普通文字、数值中偶然出现的 `37`、以及算法里无机器人含义的 left/right 比较不计入硬编码。

### H1：资产、环境与 agent 直接绑定 G1

- `configs/tasks/stair_protocol_v1.json`
  - `source_task_id = Isaac-Velocity-Rough-G1-v0`；
  - `robot_asset_id = isaaclab_assets.G1_MINIMAL_CFG`；
  - 冻结 37 joints / 44 bodies、G1 joint 顺序、37-D action 和 310-D policy observation。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py`
  - 直接 import/继承 `G1RoughEnvCfg`，类名为 `StairG1RoughEnvCfg`。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py`
  - train/eval 两处直接 import 并实例化 `G1RoughPPORunnerCfg`。
- `src/true_rl_agent/tasks/stair_climbing/config.py`
  - 全局 `STAIR_G1_ROUGH_AGENT_CONFIGURATION`，`training_agent_configuration()` 固定返回它。
- `src/true_rl_agent/tasks/stair_climbing/runtime_profile.py`
  - gym ID 与 env builder version 包含 G1，当前只能指向一个 G1 runtime。
- `src/true_rl_agent/tasks/stair_climbing/training_distribution.py`
  - 上游源码承诺直接绑定 G1 rough env/agent 文件与符号。
- `src/true_rl_agent/tasks/stair_climbing/phase0_contract.py` 与 `isaaclab_phase0_probe.py`
  - 冻结 G1 gym/asset、37 joints、`torso_link`、左右 ankle roll link，并按 `_ankle_roll_link` 找两只脚。

根治位置：Phase 1 的 RobotProfile/RuntimeAdapter/Binding 解析；不能在上述文件继续增加 `if go2`。

### H2：reward ABI 固定 37-DoF 人形与两只脚

- `configs/tasks/stair_protocol_v2.json` 及两个 draft：
  - `joint_pos/joint_vel/joint_pos_error/last_action/prev_action/action_delta = [37]`；
  - ankle `[4]`、locomotion torque `[12]`、hip/knee acceleration `[8]`、arm `[10]`、finger `[14]`、torso `[1]`；
  - foot time/contact/velocity/position 固定 `[2]`，force history 固定 `[3,2,3]`。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py`
  - `_FOOT_BODY_NAMES = (left_ankle_roll_link, right_ankle_roll_link)`；
  - `_BASE_BODY_NAMES = (torso_link,)`；
  - 通过 G1 joint-name token 切出 ankle、hip/knee/ankle、hip yaw/roll、shoulder/elbow、finger、torso 分组。
- `src/true_rl_agent/tasks/stair_climbing/reward_inputs.py`、`reward_source_semantics.py`、`component_smoke.py`
  - 将上述 G1/双足 shape 与轴语义作为当前完整 reward contract 使用。
- `src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_v1/`
  - reference reward 明确等价 Isaac Lab 2.3.0 G1 rough 的 15 个 active terms；不能直接充当 Go2 reference。

根治位置：每个 robot × task cell 必须资格化自己的 reward-input projection 和 reference reward。共享层只认带 hash 的 capability contract，不认识 ankle/finger。

### H3：runtime recorder、decoder 与结果语义固定双足

- `src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py`
  - 直接解析左右 ankle roll link 和 `torso_link`，输出两个 foot state。
- `src/true_rl_agent/tasks/stair_climbing/transition_decoder.py`
  - `foot_pos/vel/force` shape 固定 `(num_envs, 2, 3)`；
  - `for foot_index in range(2)`；
  - 输出值按 index 0/1 构造 pair。
- `src/true_rl_agent/tasks/stair_climbing/episode_evidence.py`
  - 所有 foot fields 为二元 tuple，验证器要求 pair；成功/稳定证据按两只脚聚合。
- `src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py`
  - 明确拒绝非两脚输入，报错为 `must contain two feet`。
- `configs/tasks/stair_evaluation_semantics_v6.json` 与 `evaluation_semantics.py`
  - torso contact 来源固定 `contact_forces.torso_link`；clearance/slip 来源描述为 ankle roll link；top support 语义建立在当前 foot pair 上。

根治位置：Phase 2 先由 binding 提供 role-resolved contacts/base，Stair decoder 再消费可变数量的 task roles。任务阈值和 success state machine 仍归 Stair，不能搬进 RobotProfile。

### H4：任务呈现与历史资格材料包含 G1 身份

- `src/true_rl_agent/tasks/stair_climbing/plugin.py`、`prompt_adapter.py`：任务名称直接为 Unitree G1 Stair；后续应从 resolved cell identity 投影展示名称。
- `src/true_rl_agent/tasks/stair_climbing/phase0_contract.py`、旧 protocol draft/pilot manifest、Phase D/P6 历史脚本与审计工具：它们是 G1 冻结历史证据，不应改写成通用协议，也不能当成 Go2 runtime。
- `scripts/isaaclab_stair_checkpoint_video.py` 的 G1 URDF/关节 alias 属于可视化适配，不是策略执行兼容证据；未来机器人可以有独立 renderer mapping，但不能影响 acceptance。

处理原则：历史 evidence 保持不可变；新矩阵协议通过 replay compatibility 读取它，而不是批量重命名或伪装成通用证据。

### H5：legacy 评测考卷把 cell 集合写成矩形网格

这不是机器人轴问题，却会比 Go2 更早阻塞 Phase 3A。当前 V5 的 35 cells 是
`7 个阶高 × 5 个踏深` 的笛卡尔积；多条链路把“有序 cell 集合”误当成“可按固定宽度切片的矩阵”：

- `evaluation_profile.py:222` 用 `len(eval_bins) * len(treads) * episodes` 推导环境数；
- `evaluation_profile.py:139` 用 `self._cells[::len(anchor_cells)]` 投影固定列；构造逻辑还嵌套遍历 tread 与 height bin；
- 构造/选择侧受影响：`config.py`、`evaluation_profile.py`、`generalization_contract.py`、
  `evaluation_contract.py`；
- evidence/公开投影侧受影响：`episode_evidence.py`、`transition_decoder.py`、`public_projection.py`；
- LOOP 反馈侧受影响：`prompt_adapter.py`、`plugin.py`；协议兼容侧还包括 `protocol_v2.py`、
  `phase0_contract.py`。

根治位置是 Phase 3A 的 Stair task-owned evaluation-suite seam：VNext 接受显式、有序的 cell 列表及 factor metadata，
所有 coverage/evidence/public/prompt 投影从该列表派生，不再要求矩形形状；legacy V5 35-cell reader 与 replay 保持原样。
Phase 0 只冻结这项债务，不提前重写上述 11 个模块，也不把评测形状逻辑搬进共享层。

## 4. Push G1-29 的独立物理身份

Push 当前通过 `src/true_rl_agent/tasks/push_recovery/isaac_config.py` 选择：

- gym ID `Unitree-G1-29dof-Velocity`；
- vendor env `unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg`；
- G1-29 URDF，29 joints / 35 bodies，29-D all-joint action；
- Push body `torso_link`、推力与恢复阈值均为 Push task-owned 语义。

它与 Stair G1-37 只有 family label 相同。经同一份运行时探针采集并以统一字段集校验后，两者的
`runtime_facts_hash` 分别为：

- G1-37 Stair：`93adedbe9022a62d19fda14057775bd39e15705757d18287e494ce18855c16c2`；
- G1-29 Push：`9038bdfdb8e5061adaa7bd2902fa94b65e124f73f76701f0085a155c2c678fbc`。

这些 hash 覆盖 joint/body、默认姿态、形态尺度、完整 actuator/PD 参数与资产闭包；浮点事实按 `1e-6` 量化并归一化负零，验收器还把 G1-37
逐字段交叉比对当前 Stair runtime protocol，因此不再是快照自证。Push 后端现有
`push_recovery_obs_v1` / `unitree_actions_v1` 仅按 `legacy_label` 记录；本次运行时回执另给出内容绑定的
policy-I/O hash，Phase 1 不得把旧标签伪装成内容哈希。

因此不能跨任务要求二者 profile hash 相同，也不能把 G1-29 checkpoint 接到 G1-37 Stair。

## 5. 事实否定表

| 表述 | Phase 0 判定 | 依据 |
|---|---|---|
| 当前 Stair 已有 roughness | 错 | `one_way_stairs_terrain` 只生成平 box，生产配置无 roughness/noise 参数 |
| 上游类名 `G1RoughEnvCfg` 证明楼梯 rough | 错 | Stair overlay 替换为自己的 box staircase；`rough` 仅是上游 locomotion 基类/agent 来源 |
| 当前项目已支持 Go2 | 错 | `f359356` 捕获时没有 Go2 profile/binding/registry 产品接入；只有冻结上游包具备 Go2 基础 |
| Go2 资产缺失，需要拉取 | 错 | USD `Result.OK`，config、actuator 与 RSL-RL agent 已实际核验 |
| 当前精确产品字节已跑完正式 15-generation | 错 | 正式实验绑定 `c582eca`；当前 `ab84aa3/f359356` 未复跑 |
| 系统从未有真实 15-generation 实证 | 错 | `c582eca` 有 15 个有效科学 generation；必须带提交身份引用 |
| G1-37 Stair 与 G1-29 Push 是同一 execution profile | 错 | asset closure、DoF、body/action schema、actuator、上游包和 runtime facts hash 均不同 |

## 6. 后续 Phase 的唯一允许方向

Phase 1 应把 H1 的物理事实下沉为精确 profile/runtime adapter，把 H2/H3 的 task-specific projection 留在 thin binding + Stair TaskPlugin，并让两个 legacy cells 先得到与当前行为 byte/contract 等价的 resolved manifest。Phase 3A 再根治 H5，以显式 cell suite 支持非规则考卷，同时保留 V5 35-cell replay。若实现需要在 AgentCore、共享 runner 或 TaskPlugin 中继续增加具体 joint/USD/G1/Go2 分支，或把 Stair 评测语义搬到公共层，说明抽象位置错误，应回到对应 composition/TaskPlugin 根修，而不是继续打补丁。

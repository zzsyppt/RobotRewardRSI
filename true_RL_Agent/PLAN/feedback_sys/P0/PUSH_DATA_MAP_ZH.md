# Phase 0：Push 协议、数据与作者证据地图

日期：2026-09-23。范围：当前工作树的 Push 数据链、保存样本与既有 CPU 契约回归。补充的十条作者 rationale、实际输入核对及后续 A/T 设计见 [作者A基线](./AUTHOR_EVIDENCE_BASELINE_ZH.md)。

证据边界统一说明：源码给出实现和配置；保存产物给出指定运行的事实；CPU测试覆盖mock/tensor契约。P0未新增模拟器采集或训练。旧velocity专项时序与当前force标准结果分开使用，聚合结果与机制解释分开。后续时序采集和A/T收益按对应阶段验收。

## 1. 已完成与结论

1. CodeGraph 优先定位后，核对了 runtime → episode record → public phase → task prompt → 实际保存的作者 payload。当前生产默认是 `recovery_v3` cell：**训练仍是速度增量，评估是有限外力脉冲序列**。旧 `rollout.v2` 与当前 `rollout.v3` 不能混为同一恢复协议。
2. 五个稳定子条件在当前控制步已计算，但标准评估只保存终局/极值/事件与恢复结果；**没有持久化逐步子条件、dwell 历程或打断原因**。首问不需要脚部传感器，确实需要这些时序证据。
3. `not_recovered` 不能直接解释为“dwell 从未完成”：完成过 dwell、但最终稳定条件不成立也会得到该标签。新诊断必须保留这两个分支。
4. 实际作者样本已正确接收同一 trial 的 public-dev 计数、分层成功率与训练激活摘要；逐回合时间、五条件轨迹和关节轨迹未投影。
5. 本次 **70 项既有 CPU unittest 全部通过，0 skip**。另对3份真实 public-dev 文件做当前投影核对、对8份历史诊断 NPZ 做hash/字段/连续性检查。

取证时 HEAD 为 `5cef484dfa5319130faa02d7e9bbcdd9e9b4a84e`；含未提交变更，以[数据探针回执](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_data_probe.json)的源码/ABI hash为基线。回执同时记录真实文件身份、当前默认cell解析结果与检查结果；历史运行使用各自冻结manifest。

## 2. 证据分层与本次实物范围

| 证据 | 本次动作 | 可得信息 |
|---|---|---|
| 当前源码与确定性配置 | 解析四个cell的协议/ABI/量化时序 | 当前实现、默认配置与权限 |
| 标准public-dev summary | 重做task公共投影 | 已保存字段及其与实际作者输入的一致性 |
| 历史diagnostic timing NPZ | 核hash、shape、有效掩码和步连续性 | 指定诊断运行的root时序 |
| 既有CPU unit tests | 禁用CUDA后新执行 | mock/tensor契约与修复回归 |

实物样本：

- [G1 P6 正式 public-dev](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/g1_qualified_01/formal/runner/p6-g1-stationary-v3-reference-20260912/eval/public_dev/summary.json)：`rollout.v2`，3×64=192 回合；96 success、51 not_recovered、24 fall_forward、13 fall_side、5 height_collapse、3 out_of_bounds。
- [Go2 P6 native public-dev](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/go2_pose_01/native/eval/public_dev/summary.json)：`rollout.v2`，192/192 success。这个保存样本的饱和不等于所有 Go2 Push 域均饱和。
- [Go2 当前 force 样本](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session/runner/go2-push-5rounds-20260923-a08-generation-004-lane-a/eval/public_dev/summary.json)：`rollout.v3`，192 回合；161 success、9 not_recovered、7 fall_side、14 height_collapse、1 fall_forward。9 个 not_recovered 均未记录到 sequence dwell 完成；这里只说明该样本，不排除其他样本的“已恢复后终局失稳”。
- 与上一项对应的 [G5 lane-b 作者 payload](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session/authoring/go2-push-5rounds-20260923-a08-generation-005/lane-b/authoring_attempts/attempt_000001/prompt_payload.json) 和 [G4 lane-a 训练回执](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session/runner/go2-push-5rounds-20260923-a08-generation-004-lane-a/train/summary.json)。

三份raw文件均通过 `project_push_split`；当前 `_v2_dev_summary` 重建的G4 public-dev与实际G5 payload对应对象完全相等。

## 3. 真实协议、形态与 ABI

### 3.1 四个已声明 cell 的关系

当前 `PUSH_ACTIVE_CELLS` 为 G1/Go2 的 `recovery_v3`；`recovery_v2` 仍可解析并支撑历史回放。更旧 `push.g1_29dof.legacy_v1/reference_v1` 使用 legacy overlay、分类与 ABI，不适用本文五条件恢复定义，不能被改名为 v2/v3。[config.py:26](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/config.py:26)、[config.py:79](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/config.py:79)。

| 当前 v3 cell | robot binding 的 base | 名义高度 L | policy observation / action | joint 数 | control / physics dt | sensor ABI |
|---|---|---:|---:|---:|---|---|
| `push.go2_12dof.recovery_v3` | `base` | 0.40 m | 48 / 12 | 12 | 0.02 / 0.005 s，decimation=4 | `{}` |
| `push.g1_37dof.recovery_v3` | `torso_link` | 0.74 m | 123 / 37 | 37 | 0.02 / 0.005 s，decimation=4 | `{}` |

两者policy输入由base线/角速度、projected gravity、零速度命令、关节相对位置/速度、last action构成；动作都是关节位置目标，Go2 scale=0.25，G1=0.5。维度由各形态ABI提供；reward接口另有22个字段。

绑定只要求 `base` 角色；不承诺支撑脚角色。physical overlay 明确关闭 `contact_forces` 与 `height_scanner`，并将地面设为平面、速度命令设为零。[ABI catalog](D:/humanoid_agent/true_RL_Agent/configs/tasks/push_recovery_robot_policy_abi_v2.json)、[robot_binding.py:170](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/robot_binding.py:170)、[runtime.py:54](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:54)。这里“从未采集脚接触”指这条 Push 诊断路径；不是断言仿真资产没有脚或关节状态。

### 3.2 训练速度增量与评估外力分开

| 范围 | 当前实现 | 注入/完成证据 | 恢复计时 |
|---|---|---|---|
| v2 评估；v3 训练 | world XY 根速度增量 `world_planar_velocity_increment`；保留 z 线速度及角速度 | 写前/写后 6D 速度、请求/实测 Δv、误差≤1e-5 m/s、下一个 physics 后速度 | 每个已注入 event 后独立窗口 |
| v3 评估 | `world_planar_external_force`；作用于绑定 base link，世界水平力、无直接扭矩 | 每个 physics substep 检查实际 apply；记录脉冲起止、actual/expected substeps、force、积分 impulse、误差、verified；检查非计划 wrench | 全部 planned pulses 完成且验证后，从最后 pulse end 后开始一段新窗口 |

有限力不是瞬时速度重写。力幅为 `实际 reset 后总质量 × impulse_velocity / 量化脉冲时长`；`impulse_velocity` 是冲量按质量归一化的协议参数，不是实际观测到的机器人 Δv。repeated 为两个脉冲，后者方向偏移 π/2，强度为 high 的0.8倍，脉冲间 gap=0.22 s。force runtime 只允许 evaluation；训练分布没有随 force 评估升级而改成外力。[force_protocol.py:70](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_protocol.py:70)、[force_runtime.py:34](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_runtime.py:34)、[force_runtime.py:96](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_runtime.py:96)。

当前 v3 默认训练 Δv 的 low/medium/high：Go2≈0.5943/1.1885/1.9809 m/s，G1≈0.6062/1.2124/2.0207 m/s；当前评估 impulse_velocity 则分别≈1.1885/2.3771/3.9618 与1.6166/3.2332/5.3887 m/s。旧 v2 默认与已冻结历史 manifest 可采用其他幅值；只按 cell 后缀或“strong_push”不能判断比较域相同。完整值与 hash 见探针回执。

## 4. 五条件、dwell、失败与时间窗口

### 4.1 五条件是合取，不是因果分类

每个 active 控制步用 articulation root 的世界位置/速度/姿态计算：

| 子条件 | 判定（含等号） | Go2 | G1 |
|---|---|---:|---:|
| 平面速度 | `norm(root_lin_vel_w.xy) ≤ 0.08 sqrt(gL)` | 0.158473 m/s | 0.215546 m/s |
| 三维角速度 | `norm(root_ang_vel_w.xyz) ≤ 0.5/sqrt(L/g)` | 2.476136 rad/s | 1.820491 rad/s |
| roll | `abs(roll) ≤ 0.20` | 0.20 rad | 0.20 rad |
| pitch | `abs(pitch) ≤ 0.20` | 0.20 rad | 0.20 rad |
| 高度 | `root_z - env_origin_z ≥ 0.70L` | 0.280 m | 0.518 m |

全部成立才 `stable=True`。跌倒条件是高度 `<0.45L` 或 roll/pitch 绝对值 `>0.80 rad`；v2 另有位移 `>3L` 的失败条件，v3 force 的 `max_displacement_m=None`，位移仅诊断。root 状态用于恢复，绑定的 `base` link 用于施力；不能仅因名字相近就将不同角色的位姿互换。[runtime.py:287](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:287)。

### 4.2 实际控制步规则

`window=ceil(8 sqrt(L/g)/dt)`，`required_dwell=ceil(1.5 sqrt(L/g)/dt)`。Go2 是81步窗口/16连续步；G1是110步窗口/21连续步。名义时间分别约1.61542/0.30289 s、2.19721/0.41198 s；量化计数对应1.62/0.32 s、2.20/0.42 s，必须报告实际 step 与 dt，不能混用名义值。

| 时序 | Go2 | G1 |
|---|---:|---:|
| v2 计划 event step | 51、172 | 69、234 |
| v3 首脉冲 start | 51 | 69 |
| v3 pulse steps / gap steps | 11 / 11 | 14 / 11 |
| v3 repeated 第二脉冲 start | 73 | 94 |
| v3 单脉冲恢复零点＝end | 62 | 83 |
| v3 repeated 恢复零点＝末 end | 84 | 108 |

只在 `0 < elapsed ≤ window`、episode 未关闭且尚未记录恢复的步骤累计 dwell；stable 则加1，否则清0。达到 required dwell 后记录 recovery step，后续不再继续更新该事件/sequence 的 dwell 计数。v3 还要求全部脉冲完成、verified 且没有非计划 wrench。因此“某个子条件在窗口外变假”“尚未完成注入”“有效窗口内已有计数被清零”是三种不同事实。[runtime.py:314](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:314)、[force_runtime.py:162](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_runtime.py:162)。

v2 success 要每个计划 event 已注入、已验证、已恢复且最终 stable，并且没有任务失败；v3 success 要整段 sequence 已恢复且最终 stable，并无 fall/invalid。`not_recovered` 包括“未达到 dwell”与“达到过 dwell 但最终 unstable”。其他主要标签为 fall_forward/backward/side、height_collapse、invalid_push_injection、no_push_control；out_of_bounds 仅 v2。标签的优先顺序不提供唯一物理根因，早跌倒仍保留在正式分母中。[failure_classifier.py:113](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/failure_classifier.py:113)、[force_runtime.py:199](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_runtime.py:199)。

## 5. 逐字段数据状态

本表“已给作者”指证据值进入当前 task prompt/核对的实际样本；reward 字段名称和使用说明进入合同，不等于给作者逐步数值。`已采未存`指标准 runtime/rollout 路径；第7节的专用诊断档案是另行注明的例外。

### 5.1 原始状态与恢复内部状态

| 字段或量 | 当前读取/计算 | 标准保存内容 | 作者得到什么 / 状态 |
|---|---|---|---|
| `root_pos_w`、`env_origins`、`initial_xy` | 每步；高度与相对 drift | min height、max drift | 逐步值已采未存；极值已存但当前 v2/v3 task prompt 未投影 |
| `root_quat_w` → signed roll/pitch | 每步，显式 quaternion 转换 | max/min pitch、max abs pitch/roll | 逐步 signed pose 已采未存；roll 方向不能由 abs 极值恢复 |
| `root_lin_vel_w` | 每步 | final planar speed；v2 event 前/后及下个 physics 速度 | 一般时序未存；保存的 event/terminal 数值未进作者 |
| `root_ang_vel_w` | 每步用于 angular norm | v2 event 6D velocity 中含角速度；v3标准记录无最终角速度值 | 五条件所需一般时序未存 |
| `speed_ok/angular_ok/roll_ok/pitch_ok/height_ok` | 在合取表达式即时计算，未作为独立命名日志 | 无每步数组/false集合 | 已计算未存；不是“传感器从未采过” |
| `stable` | 每步五条件合取 | `final_recovery_condition` | 最终布尔已存未直接给作者；不能反推途中每次失守 |
| `dwell` / `sequence_dwell` | mutable 当前计数，失稳清零 | 只保存恢复首次达成 step；不保存最终计数或重置轨迹 | 已采未存；无打断次数/原因/时间 |
| `steps/episode_index/closed` | 首回合 active 管理 | env_id、episode_index、episode_steps | 身份及长度在原件；作者聚合只保总体计数，不直接展开这些回合 |
| 五条件原值/阈值/失败集合及 dwell_before/after | 原值阈值可获得；逐条证据对象尚未构造 | 无 | 尚未构造/持久化，须新增确定性记录；不是已有诊断卡 |
| 逐脚接触、support position/velocity、足尖/着地点 | 当前 Push recovery recorder 不读取；contact sensor 关闭，binding只base | 无 | 本诊断路径从未采；补采需明确角色、ABI、开销与等价门 |
| 逐关节 `joint_pos/joint_vel` | reward adapter 可读，每步输入 | 无标准关节时序；可有候选自定义诊断摘要 | 不是全链不可用；仍不足以判 ankle/hip 策略 |

来源：[runtime.py:168](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:168)、[runtime.py:287](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:287)、[runtime_adapter.py:487](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:487)。

### 5.2 回合与事件记录

| 字段 | v2/v3保存情况 | 公开 phase 与作者投影 |
|---|---|---|
| `env_id/episode_index/episode_steps/stratum/control_only/planned_push_count` | 两者逐回合保存 | 作者仅见 stratum 分母/成功数、control/planned 总数 |
| `min_base_height_m/max_pitch_rad/min_pitch_rad/max_abs_roll_rad/max_abs_pitch_rad/max_xy_drift_m/final_planar_speed_mps/final_recovery_condition` | 两者保存 | phase 可有部分 secondary 均值；作者 `_v2_dev_summary` 不保留该细节 |
| `failure_label/success/control_success/control_failure_label` | 两者保存 | 作者见 failure counts、success stats、主失败归类；control 不混为恢复成功 |
| `push_events.event_index/control_step/time_s` | v2 保存 | 事件总数给作者；具体时间已存未投影 |
| `requested_delta_v_world_mps/observed_delta_v_world_mps/velocity_before_world_6d/velocity_after_write_world_6d/velocity_after_next_physics_step_world_6d/injection_verified` | v2 保存；next physics 可能为 null | 作者只见注入及无效计数，不见向量/时间 |
| `recovered/recovery_control_step/recovery_latency_s/recovery_censored` | v2 每个 event 保存；未恢复 latency 用量化 window 上限且 censored=true | 作者只见 recovered event 总数；该上限不能当实测恢复时间 |
| `force_events.event_index/start_control_step/end_control_step/actual_substeps/expected_substeps/completed/injection_verified` | v3 保存 | 作者见 pulse completion/verification 总数；不见起止时序 |
| `force_world_n/delivered_impulse_world_ns/max_force_error_n/actual_total_mass_kg` | v3 保存 | 已存未给作者；实际 impulse 不代表唯一接触/受力原因 |
| `sequence_recovered/sequence_recovery_control_step` | v3 保存 | 作者见 sequence_recovered_count；没有按每 pulse 伪造 recovery latency |
| `push_event_count/completed_push_event_count/verified_push_event_count/invalid_injection_count` | v3 保存 | 作者已见这些聚合计数及 planned/pushed/eligible 分母 |
| `mean_action_l2/mean_action_rate_l2` | rollout 逐回合均值及 secondary 均值 | 已存，当前 task 作者摘要未保留；不代表完整动作曲线 |

来源：[runtime.py:341](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:341)、[force_runtime.py:178](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/force_runtime.py:178)、[rollout.py:120](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/rollout.py:120)。

### 5.3 奖励授权 ABI：22 个字段，不提供评估特权

| 字段 | 当前奖励时刻语义 | 时序/作者证据状态 |
|---|---|---|
| `base_height` | root z 减 env origin z，m | 当前步输入；仅极值留评估原件 |
| `roll`、`pitch` | root 姿态带符号角，rad | 当前步输入；逐步未存 |
| `base_pos_xy` | runner提供的平面位置 | 当前步输入；不是脚落点 |
| `base_lin_vel`、`base_ang_vel` | 当前 Isaac 有效 raw 向量为 robot base frame，SI单位 | 不与 world-scaled 向量混用；fallback provenance记录可用性 |
| `projected_gravity` | 姿态派生重力方向 | 当前步输入；标准轨迹未存 |
| `commanded_base_vel` | 当前命令；此站立协议固定零命令 | 非未来扰动命令 |
| `joint_pos`、`joint_vel`、`joint_pos_error` | 当前形态的关节状态/默认位姿误差，J=12或37 | 当前步可读；未统一持久化关节窗口 |
| `last_action`、`prev_action`、`action_delta` | 当前动作、带reset处理的前一步动作及差 | 首步历史可能缺失；input_provenance需随步更新 |
| `progress_fraction` | 训练运行进度 | runner-owned；不是回合扰动相位 |
| `push_active`、`push_vec_public` | 已执行的当前训练 velocity increment；单位分别bool-like、world XY m/s | 事件步才有；不跨恢复期latched、不按机器人自主运动推断、不表示评估force |
| `nominal_base_height`、`base_height_relative` | 绑定 L 与高度/L | L不是成功阈值 |
| `base_lin_vel_scaled`、`base_ang_vel_scaled` | 世界线速度/√gL、世界角速度×√L/g | 与 raw base-frame向量区别明确 |
| `terminated_non_timeout` | 可信 pre-reset 非timeout终止mask | 不是私有accept或未来结果 |

字段完整顺序及合同快照见 `push_data_probe.json.reward_contract`。来源：[reward_inputs.py:11](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/reward_inputs.py:11)、[runtime_adapter.py:528](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:528)、[prompt_adapter.py:473](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:473)。

当前物理 Push 作者可写完整 reward program，附带 diagnostic callable 和受限 `reward_weight_scale` schedule；已有 anchor 时仍按当前单组件机制修改规则执行。RewardManager 对加权 reward rate 乘 control dt 一次。作者不能改变任务控制器、扰动分布/时序、success/dwell、训练预算、PPO、robot/action ABI、eval 或接受规则。`source_request` 只读既有封闭源码目录；本轮不存在已实现的开放 evidence request。[prompt_adapter.py:446](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:446)、[prompt_adapter.py:312](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:312)。

### 5.4 训练诊断记录

训练每步执行候选奖励；默认每24步采一次 activation/DiagnosticTerm，保存 observation_count、nonzero_count、max_abs、last_mean 等；另保存同一步/同env的累计 reward 绝对峰值 frame。`input_provenance` 是最新输入可用性，不是全训练覆盖。没有原生早/中/晚分段表，也没有完整轨迹，不能从最后值推断全程或因果。[runtime_adapter.py:261](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:261)、[runtime_adapter.py:341](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:341)、[runtime_adapter.py:382](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:382)。

核查的真实训练文件保存 `input_provenance` 和 reward peak frame；对应作者 payload 没有这两个结构键。作者收到 reward calls=72000、diagnostic calls=0、6个term的有界采样摘要；diagnostic outputs为空且计数为0，是这次实际程序没有诊断输出，不能推广成所有程序都不支持诊断。

## 6. 当前到作者的实际路径与回归

`episode_records → rollout.per_episode/aggregate/secondary_metrics → phase_result.task_evidence → project_public_phase_summary → project_push_split → task project_prompt_public_history/_v2_dev_summary → task sections → shared最终payload`。

第一道公共 phase 投影保留每个 run 的 score、failure breakdown、部分 secondary、recovery counts 与分层计数，不展开 per_episode。第二道 task 作者投影进一步只保留 score stats、整数recovery counts、strata、failure counts。训练反馈按选中的同一 trial 取 runtime，缺失时返回 not_available；minimum-core 明确 runtime detail 为 omitted_for_capacity。[metrics.py:396](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/metrics.py:396)、[phased_adapter.py:230](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/phased_adapter.py:230)、[prompt_adapter.py:537](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:537)、[prompt_adapter.py:1010](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:1010)、[prompt_adapter.py:704](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:704)。

真实 G5 payload 的主要反馈为 G4 lane-a（latest_trained_nonaccepted），其当前 parent 另列 current_best G4 lane-b；两者失败记录没有合并。主反馈与 runtime 的 source_round_id、candidate_id、program_source_checksum 一致。当前重新投影相同 raw 得到161/192，真实 payload 中 current_best 为175/192；它们是不同 trial，不是互相冲突的分数。

该 payload 结构中不存在 `per_episode/force_events/push_events/sequence_recovery_control_step/min_base_height_m/dwell_break_reason`，也没有 `public_accept/sealed_validation` 原始对象。此检查不能代替全局 canary 证明；相关正常/minimum-core反例由本次 CPU suite 同时覆盖。Push 的 descriptive candidate-parent public-dev delta 与 component causal effect 仍分开，不给正负分差直接赋因果。

## 7. 已有逐步诊断档案能复用到哪里

只读检查了 `P6/independent_timing_evidence_01/{go2,g1}/{strong_push,no_push}` 与 `P6/stationary_v3_evidence_01/{go2,g1}/timing/{strong_push,no_push}` 共8个目录：

- 每个有 `raw.npz/capture.json/independent.json`；NPZ含 `valid/steps/position/quaternion_xyzw/linear/angular/done`。Go2数组时长303步、G1 412步，64 env；本次逐env有效step均连续，8/8 raw hash与capture一致。
- capture明确 `diagnostic_only=true`、`enters_qualification=false`，记录在 physics 与计数递增后、TerminationManager.compute 前；有效掩码排除reset后的新回合。旧 independent 报告记录 timing与task一致；本次没有把读取旧PASS改称重新运行独立恢复重算。
- 全部属于旧速度增量机制。旧 independent G1 strong_push 是64个 not_recovered；stationary G1 strong_push 是64个 success。不同 checkpoint/来源不能充当同一策略内的随机成功失败配对，更不能自动当成当前force结果。
- 它们足以支持后续**带来源与协议资格的开发重算**；只有核对身份、协议与允许时间后才能进入新公共证据目录。原脚本已用独立NumPy公式计算五条件及卷积dwell，可作为重算参照，不能从目录名“stationary_v3”误判为 force `rollout.v3`。

来源：[audit_push_episode_timing.py:41](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/audit_push_episode_timing.py:41)、[capture边界](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/audit_push_episode_timing.py:184)。全部具体路径/hash/shape/标签记录在本次数据探针JSON；本次不重复建立训练谱系清单。

## 8. 首个 Push 问题的可行范围与最小缺口

登记问题：“not_recovered 回合里 dwell 被哪些子条件、在扰动后多久打断，success_recovered 组是否同样出现”。**算法上可行，无需脚部传感器；当前标准 force 保存结果不足以回答完整时序问题。**

前置限定：

1. 先按协议区分 v2 event 与 v3 sequence；force 对齐零点为最后一个完成的 planned pulse end，v2为该 event 的真实注入step。成功组使用相同stratum/完整扰动条件和相同相对窗口，不能按各自恢复时刻对齐后再比较“扰动后多久”。
2. not_recovered 先分“窗口内未达到有效dwell”与“曾达到但final unstable”。后一组另描述终局失稳，不能虚构一次未完成的dwell。
3. `dwell_before>0 && stable=False` 才称打断；`dwell_before=0 && stable=False`是尚未启动/持续不满足。五条件同时失守保留完整集合，不能固定挑一个“主要原因”。
4. early fall、未完成/无效注入、no_push、缺采样、窗口截断分别处理，保留总体及可比较子集分母。条件不满足的次数/时间只说明关联，不证明需要修改某个关节或奖励。

下一阶段最小记录建议（待对应验收卡冻结，不在P0改运行语义）：

- 身份：原session/task/robot/protocol/source/checkpoint/episode、event/sequence ID、证据首次可用位置；不把 raw seed/path公开。
- 每个有效control step：step/dt、root高度、signed roll/pitch、planar speed、angular norm及阈值/规则版本；五条件bitset、stable、active/window资格、dwell_before/after、首次恢复达成、terminal/pre-reset标记。
- 时间锚：实际event step或force全部planned pulses完成/验证及末end；失效/未采/截断状态。
- 只用必要低维全窗口或可验证充分记录覆盖到终局；稀疏几个姿态窗口不能重算连续dwell。计数、首次step、组别与现有success保持一致，不能新增奖励字段或改变接受gate。

可先用旧velocity诊断档案实现重算fixture。当前force真实时序与capture on/off等价由Phase 2补充；上述样本已有161个success与9个not_recovered作为类别入口，标准文件尚缺五条件/dwell轨迹。

## 9. 新执行的CPU回归与复现

执行器：项目现有 `.venv/Scripts/python.exe`，Python 3.12.14、Torch 2.11.0+cpu，`CUDA_VISIBLE_DEVICES=-1`，`torch.cuda.is_available()==False`，Torch单线程。最终70 tests，0 failure/error/skip，约39.1秒。

| 既有模块 | 本次覆盖的关键边界 |
|---|---|
| `test_push_physical_protocol` | 尺度/manifest、注入保留其他速度分量、连续dwell、首回合freeze、早跌倒分母、坐标转换、control区分 |
| `test_push_force_protocol` | 形态量化、固定gap、无空间失败gate、eval变化不改变train、旧cell不可静默改名 |
| `test_push_force_runtime` | 实际质量/世界力变换、substeps/impulse、pulse后fresh dwell、窗口含末步、early fall、不合法wrench与缺callback |
| `test_push_force_consumers` | sequence与pulse计数分离、raw证据闭合、旧/混合schema拒绝、force prompt不改变train信号单位、pre-reset媒体hook |
| `test_push_feedback_projection` | measured parent-dev、幂等、无formal时sibling记忆、初始reference、current-best独立失败证据、排除accept scores |
| `test_push_recovery_runtime_adapter` | 同env reward peak、candidate/infra异常类型、sealed source identity、shape/finite、public push信号、action history provenance |
| `test_push_recovery_v2_prompt_smoke` | ABI一致、双形态维度、wrong dimension拒绝、canonical整数计数、私有canary/正常与minimum-core、force tensor广播 |
| `test_push_phased_evaluation` | phase receipt、实际work计量、dev选择、reference解析/qualification、shared phase projection消费者 |

可复现命令（仅CPU）：

```powershell
& 'D:\humanoid_agent\true_RL_Agent\.venv\Scripts\python.exe' 'D:\humanoid_agent\true_RL_Agent\PLAN\feedback_sys\P0\push_cpu_regression.py'
& 'D:\humanoid_agent\true_RL_Agent\.venv\Scripts\python.exe' 'D:\humanoid_agent\true_RL_Agent\PLAN\feedback_sys\P0\push_data_probe.py'
```

[回归脚本](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_cpu_regression.py)、[最终日志](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_cpu_regression.log)、[结构化回执](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_cpu_regression.json)、[数据探针](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_data_probe.py)。

首轮执行器遗漏repo根sys.path，修正P0脚本后重跑；原始[日志](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_cpu_regression_bootstrap_failure.log)和[回执](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/push_cpu_regression_bootstrap_failure.json)保留。

## 10. 本子任务的验收状态

| 项目 | 状态与界限 |
|---|---|
| 当前Push协议、形态、ABI、contact设置登记 | 完成，具体源码/配置hash已留存 |
| 字段“已采/已存/已给作者”区分 | 完成；标准路径与历史专项记录分列 |
| 首问可行性与时序缺口 | 完成；未声称已有force时序可回答 |
| 既有修复CPU回归 | PASS，70/70，无skip；不是模拟器资格 |
| 当前force真实时序/新capture等价 | NOT_STARTED，需后续授权范围与验收卡 |
| provider/视觉/训练/收益验证 | 未执行，不由此文判通过 |

本图登记运行链路与缺采字段。补充审计发现的作者解释/候选符号矛盾另见[作者A基线](./AUTHOR_EVIDENCE_BASELINE_ZH.md)，历史策略与运行源码保持原样。

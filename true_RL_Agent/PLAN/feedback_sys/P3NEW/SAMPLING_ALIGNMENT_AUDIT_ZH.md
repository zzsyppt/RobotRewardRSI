# 同轨媒体与数值采样时序审计

## 后续源头修正（2026-09-29）

Stair 已由 wrapper 的 env.step 后取图，改为 `StairEvalTransitionRecorder.record_post_step` 中数值 clone/发布之后、reset/interval 之前的任务私有 hook。录制层禁用 wrapper 重复采样，退出恢复 hook。Stair 的 tracker 可在 Gym done 前因成功 latch 或任务 horizon 关闭，故按真实 outcome T 裁首回合，不能只靠环境 done 猜末帧。新原始采集使用逐控制步双视角流式 PNG；固定步历史模式仍按真实 samples 标记是否含 T，不冒称全程。Push 的终止图标记同步按实际 samples 修正。相关整合 CPU 103 项通过；真实渲染与 NPZ 对齐仍待独立采集核验。

真实 1080p 试拍另发现当前安装版 Isaac Lab 的 RGB 输出使用 `cfg.video_recorder.window_width/height`，不自动继承 `cfg.viewer.resolution`。已在创建环境前同步两处配置；禁止 resize 冒充更高采集分辨率。r2 首帧格式失败被保留，没有合格 PNG。下文为修正前的审计依据，不代表上述问题仍未修复。

2026-09-29。**源码接线审计与后续最小修复，不是实际双视角验证或 N.5 放行。** 初次审计只读检查本地源码、既有准备卡/开发脚本、旧开发 composition，以及远端已安装 Isaac Lab 源码；随后仅收敛 Push 原生媒体接线并执行 CPU 单测。没有实际仿真、GPU、模型 API 或新确认资料读取。修复前证据与修复后状态分别如下。

## Push 路径究竟是否 force

**N.5 尚未冻结 Push cell/config，因此不能认定其将采用 force。** `N5_PREPARATION_ZH.md` 明确为 `DRAFT / NOT_RELEASED`，只列合格 G1 checkpoint 和草案 base seed `1500000000`，要求真实命令、cohort、源码/checkpoint hash 和运行目录另入新卡。这个 seed 数字不能决定扰动机制。

现有可执行开发链则确定为 **velocity/v2，`force_sequence=False`**，证据不只是 cell 名字：

- `../P3R/record_push_probe.py` 和 `push_close_camera_probe.py` 均显式选择 `push.g1_37dof.recovery_v2`，沿用旧 base seed `1000000000`。
- 已只读远端旧开发文件 `/data/yuyangliu/AGENT/feedback_p3r_20260929/video_medium19_r2/composition/resolved_cell_manifest.json`：其 cell 为上述 v2；task projection **没有** `push_evaluation_protocol` 和 `push_resolved_evaluation_protocol`。
- 冻结旧 `source_r3/.../push_recovery/rollout.py:77` 与当前源码一致，实际以 `"push_evaluation_protocol" in config` 决定 `force_sequence`。
- 当前 `push_recovery/config.py:92-106,151-156,294-299` 仅对 active v3 配置生成 force 协议，拒绝 cell 与协议不符；`runtime.py:57,99-100` 再按该字段选择 force 或 interval velocity 注入。`scripts/push_numeric_replay.py:57-71` 从祖先取训练协议，按显式目标 cell 重新构造配置，祖先 checkpoint 名中的 v3 不能代替当前评估配置。

没有把 N.5 自行切到 force，也没有复用旧开发卡的放行状态。

## 修复前已核实的采样顺序

远端实际使用 `/data/yuyangliu/envs/isaaclab6_validation` 的 editable `isaaclab-6.1.14`，其源码位于 `/data/yuyangliu/isolated/isaac_sim_6_validation/IsaacLab-v3.0.0-beta2.patch1/source/`。`isaaclab/isaaclab/envs/manager_based_rl_env.py:231-273` 顺序为：物理/scene 更新 → episode 计数加一 → termination compute → reward → recorder post-step → reset → command/interval events → observation → 返回。本结论不外推到其他安装版本。

| 修复前路径 | NPZ 来源时点 | 媒体时点 | 首回合终止步 T |
|---|---|---|---|
| Stair | `StairEvalTransitionRecorder.record_post_step()` 在 reset 前克隆；随后离线分块保存 | 默认 native wrapper 在 `env.step()` 返回后采集 | NPZ 包含 T；最终媒体仅到 T−1 |
| Push force/v3 | termination 回调中 `_observe_pose()`，继而 `numeric.record()`，最后更新 `closed` | 同一 termination 回调先执行媒体 hook，再执行 `observe()` | 两者都包含 T |
| Push velocity/v2 | 同样在 termination 回调采数值 | 默认 wrapper 在 step 返回后采媒体 | 媒体仅到 T−1；另存在 interval 注入时点差异 |

出处：`stair_climbing/isaaclab_runtime_env.py:99-179`、`evaluation_runtime.py:334-351`、`isaaclab_execution.py:1305-1309`；`push_recovery/runtime.py:149-153,289-345`、`force_runtime.py:162-178`、`rollout.py:77-94,138-144`；以上任务文件均在 `src/true_rl_agent/tasks/` 下。

**v2 的真实配对风险：** 它在 termination compute 中保存当步速度，之后在 interval event 中执行 `write_root_velocity_to_sim`，最终 wrapper 才采 snapshot。同一步无 reset 也可能出现“snapshot 速度已注入、NPZ 速度尚未注入”。这是已证实的执行位置差异；不能把未实际比较的某张图直接判成错图，也不能把两种测量时点当成严格同状态。

Push 的 hook 执行时 `state.steps` 尚未由 `_observe_pose()` 更新；当前图步号应取 recorder 首回合序号或 `env.episode_length_buf`，不能取这个旧的 `state.steps`。渲染 warmup 不计控制步；后续统一 hook 后，这一要求适用于两种机制。

## 配对的最小准则与字段限制

**Stair：** 对实际媒体 sample 的 `control_step=k`，找同 run/同 env 的 NPZ chunk，行号为 `k - chunk.step_start`，核对 `valid=true`、`episode_length=k`。当前 G1/Go2 unified binding 均为 `kinematic_base.source=articulation_root`；故 `root_pos_w` 对 `base_pos_w`，`root_quat_w` 从 xyzw 重排为 wxyz 后对 `base_quat_w`，全关节 `joint_pos` 按 `resolved.joint_ids` 重排后对 NPZ 的 `joint_pos`。绑定和关节顺序必须读取实际 manifest/运行解析结果，不能推广为任意机器人都同序。Stair NPZ 未存 root 速度或 `joint_vel`，支撑体速度不是替代字段。

当前 Stair 配置关闭外力和 push events（`isaaclab_runtime_env.py:516-520`）；非终止保留步的数值克隆与后续截图之间不再推进物理。因此已有源码支持在这些姿态/位置/关节字段上建立严格同 step 比较；**仍须实采核验，本文未证明逐值通过。** 终止步 T 之后的 reset 画面不能配到 T 的数值。Native 新 snapshot 实现会在最终保存时按 `frame_count=T-1` 过滤；构建器只认最终 `samples`，不能把请求的 `selected_control_steps` 当成实际已采集合。

**Push：** 找 `valid=true`、`control_step=k` 且 `episode_index` 为该首回合的行；chunk 内通常为 `k-1-frame_start`，但应以数组字段复核。可核对：root 四元数推导的 roll/pitch 对 `pose[1:3]`；`root_lin_vel_w[:2]` 对 `planar_motion[:2]`，其范数对 `pose[3]`；`root_ang_vel_w` 范数对 `pose[4]`。高度 `pose[0]` 和 reset-relative 位移/漂移还需要同 run 的 `env_origin.z`、reset 时 `initial_xy`，不能把 step=1 位置冒充 reset 基准。现有 Push NPZ 没有完整 root pose 或关节位置，因此只能验证重叠派生量，不能宣称全身状态逐项一致。跨 Torch/NumPy 派生运算使用明确的原生 dtype 容差；位置/关节直接复制字段不应借宽容差掩盖错行。

**root 字段定义：** 本次实读安装版 `isaaclab_physx/assets/articulation/articulation_data.py:1815-1840` 和 `isaaclab_physx/assets/kernels.py:59-75`。旧 `root_state_w` 是 root-link pose 加 root-COM velocity，13 列为 `[position(3), quaternion_xyzw(4), angular_velocity(3), linear_velocity(3)]`。初步消息中沿用旧布局的速度切片建议已纠正。并行 recorder 实现已改用独立命名的 `root_pos_w/root_quat_w/root_lin_vel_w/root_ang_vel_w`，避免依赖此版本相关拼接布局；四元数和速度参考点语义仍须保留。

## 后续已实施的最小源头修复

`push_recovery/rollout.py` 现已让 velocity/v2 与 force/v3 的可选 native 录制共同使用 `capture_push_transition()` 中既有 `_push_pre_reset_capture` 入口：`start(task_records_steps=True)` 禁止 wrapper 再次录帧；hook 在 `observe()`、reset、interval 注入之前采图；`finally` 恢复原 hook；两条路径均按首回合长度保存 step 1 至终止步 T，且记 `includes_terminal_frame=true`。采集点明确为 `after_physics_before_task_observe_reset_and_interval_events`。未给 snapshot 新造分支，未改变物理、数值采样、判定、种子或所选协议。

`scripts/isaaclab_rsl_rl_eval.py` 的 Push 媒体 sidecar 版本升为 `true_rl_agent.push_recovery.native_video.v3`，写明上述采集点及 `first_episode_steps_1_through_terminal_inclusive` 范围。它是媒体合同版本，不表示 cell 被迁移为 force/v3。历史录像/回执未改写，仍须按其原采集时点和末帧规则解释。N.5 的 cell/config/实际运行协议仍未冻结。

Stair 接线未在这次修复中改动；T−1 以内的字段对齐仍按上文约束核验。若未来要求 Stair 终止图，再由任务层增加 pre-reset 媒体钩子并单独冻结合同。

CPU 验证命令：`.venv\Scripts\python.exe -m unittest tests.unit.test_push_force_consumers tests.unit.test_push_numeric_capture -v`，**18 项通过**。新增的一组测试使用两种机制的实际任务 runtime 与 NPZ 写入器，以 CPU 环境替身模拟 termination → reset → interval 顺序；验证选定环境提前终止时每步只录一次、包含终止图且不取 reset 后图、录制时的高度/平面速度与 NPZ 当步值一致、已有 hook 恢复，并确认录制开关不改变数值数组和任务统计。velocity 分支实际执行 interval 注入并确认速度改变，因此能区分修复前的采样时点。视频编码被替身替代，未启动 Isaac Lab 或实测双视角；这些测试不能作为实际同 run PNG+NPZ 配对或 N.5 放行证据。

修复前审计 Push `rollout.py` SHA-256：`ff4e5eb670c5e15debdf93115c3935c66c158b6f73b7a30da1e883cd33c663b2`；本次修复后：`62f468d3b461c6a457a0951a546a4b9dc9e077b7ae98b6b9a64fd8f0204d6007`。审计时 Stair `isaaclab_execution.py` SHA-256：`f71743436d8089cc95350971f10d21a1f7f8f950889fcf1c438dd0135a97ebd2`。实际采集仍须另封存命令、源码与最终样本证据，不能把源码判断或 CPU 单测写成实采 PASS。

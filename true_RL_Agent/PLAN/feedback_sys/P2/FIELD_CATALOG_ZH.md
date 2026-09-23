# P2 数值证据字段目录

2026-09-24，按 P2 审计修订后的实现整理。本文描述字段与证据边界；真实验证以当次验收报告为准，不将短 smoke、离线复算或高成功率视为正式训练收益。

## 共享容器：保存与校验，不解释任务

`runner/numeric_evidence.py` 写入独立数值 `chunk_*.npz`，最后发布 `manifest.json`。数组保留原 dtype/shape，只允许布尔、整数、浮点数；读取禁用 pickle。默认单归档上限 2 GiB，单 chunk 原始数组上限 64 MiB。各任务另有自己的缓冲和覆盖定义。

| 层级 | 实际字段与含义 |
|---|---|
| 引用 | `schema_version / manifest_path / manifest_sha256`；SHA 绑定 manifest 的实际字节 |
| manifest | `identity / metadata / chunks / stored_chunk_bytes / raw_bytes / complete` |
| chunk | 相对 `file`、文件 `sha256`、`stored_bytes / raw_bytes`、task metadata，以及每数组 `shape / dtype` |
| 读取 | 校验 manifest SHA、完成标记、chunk 顺序/文件 SHA/大小、数组集合/shape/dtype，再交给任务解释 |

共享 `complete` 表示归档已封存；不代表 episode 完整、每个训练窗口都已到达、媒体存在或候选合格。任务身份、坐标、终止、恢复和 public-dev 边界由任务及其 phase receipt 校验，不由 NPZ 容器授权。

## Stair public-dev：首个 episode 的逐控制步证据

当前 profile 为 `stair.unified_first_episode.v2`，仅用于 unified Stair 的 public-dev。旧 v1 仍按原字段重算，历史接触明确为未记录。归档按 seed/run 分块，数组动态轴为 `[control_step, env, ...]`；static 字段每 seed 的首块保存。环境、关节、support 和 forbidden body 的列顺序来自随档案保存的 `resolved_cell_manifest / runtime_projection`，G1 与 Go2 不能共用硬编码列号。

| 数组 | 单位、坐标与边界 |
|---|---|
| `base_pos_w` | 根部世界位置，m；不是自动减去 terrain origin 的局部坐标 |
| `base_quat_w` | 根部世界姿态四元数，`wxyz`，无量纲 |
| `joint_pos` | 当前 G1/Go2 转动关节位置，rad；保留运行时关节顺序。当前 decoder 校验其形状，**未新增关节限位超限统计** |
| `support_pos_w / support_lin_vel_w` | 绑定 support 刚体参考点的世界位置 m、世界线速度 m/s；不是接触斑中心或压力中心 |
| `support_contact_forces_w / forbidden_contact_forces_w` | 相应绑定刚体的世界接触力向量，N；不等于完整逐 physics substep 接触历史 |
| `support_contact_forces_w_history / forbidden_contact_forces_w_history` | v2 新增，现有传感器缓冲的三帧世界系净法向接触力向量，N，额外轴为 `[history, bound_body, xyz]`；第 0 槽最新。这是终止/reward 接触判据的历史操作数，不能定位接触对象、碰撞面法向或推断完整子步轨迹 |
| `terrain_env_origins_w / terrain_types` | 每 env 的世界地形原点 m、冻结地形类型整数索引；按 seed 保存 static 值 |
| `episode_length` | episode 内控制步整数；该 seed 的 `step_dt` 给出控制步到秒的换算 |
| `done_mask / wrapper_done_mask / termination_<index>` | 任务侧、wrapper 侧及每个已命名 termination term 的布尔记录；term 名称/顺序存在 chunk metadata |
| `valid` | 该 env 的计分首个 episode 在此帧是否仍有效；**包含 reset 前终止帧** |

首个 episode 关闭后，同一向量批次的后续值仅是无效 padding。Stair 对这些位置置零，并为 decoder 补可解析的 episode length/quaternion；必须先看 `valid`，不能将 padding 当作真实零姿态。归档保存 `seeds[].observed_control_steps / episode_steps / outcomes`：前者为该向量 rollout 已记录的控制步，episode 的有效步数为各 env 的实际 `episode_steps`，两者不是同一个分母。

`recompute_stair_numeric_evidence` 用已保存输入、冻结的 evaluation suite/semantics 和既有 decoder/tracker 重建每回合 `primary_reason / success / diagnostic_flags / max_progress_m / progress_fraction` 及既有整体/分层汇总，并要求与原值一致。主失败原因是唯一分类；diagnostic flags 可共现，不能把 diagnostic slip 次数当成 primary slip 失败数。这里没有另增姿态分位数、动作/力矩序列、关节速度或新的行为评价标准；已存数组可以支持后续有明确规则的分析，但不能提前称为已实现统计。

## Push public-dev：真实恢复判据及事件窗口

当前 schema 为 `true_rl_agent.push_recovery.numeric.v2`，支持 physical G1/Go2 recovery v2/v3 的 public-dev，每 seed 一个归档。旧 numeric v1 继续重算，方向标为 `not_collected_in_v1`。采集点为 **physics 与 task observe 后、termination/reset 前**；动态轴为 `[frame, env, ...]`。`num_envs`、seed/run、episode indices、strata 和 planned push counts 均记录在档案，覆盖分母不能用当前常见的 64 env × 3 seed 替代实际配置。

| 数组 | 列含义、单位与边界 |
|---|---|
| `pose[...,6]` | 依次为 `height_m, roll_rad, pitch_rad, planar_speed_mps, angular_speed_radps, xy_drift_m`；是原 dtype 的实际任务判据操作数，不是新估计器 |
| `planar_motion[...,4]` | v2 新增，与 pose 同帧的 `root_lin_vel_w.xy`（m/s）与 `root_pos_w.xy-reset_root_pos_w.xy`（m），保留世界系正负方向；离线检查范数一致性，保留原 GPU 范数作为阈值判据操作数 |
| height / drift | `root_pos_w.z - env_origin_w.z`（m）；`norm(root_pos_w.xy-reset_root_pos_w.xy)`（m） |
| roll / pitch | root quaternion 按 wxyz 转换的有符号 Euler 角，rad；与使用绝对倾角的阈值判断区分 |
| linear / angular speed | 世界 `root_lin_vel_w.xy` 的范数（m/s）、世界 `root_ang_vel_w.xyz` 的范数（rad/s），不是带方向的速度向量 |
| `conditions[...,5]` | 依次为 planar speed、angular speed、roll、pitch、height 的布尔通过条件；全通过才 stable |
| `control_step / episode_index / valid / failure / timeout` | episode 相对整数时钟及身份、active 首回合 mask、已有失败/超时判断；秒数为 `control_step × step_dt_s` |
| `event_steps / event_verified` | 两个 event slot 的注入时刻和既有注入验证；不是“请求过注入”就算实际验证通过 |
| `observed_dwell / observed_recovery_step` | 运行时已累计 dwell 与首次恢复时刻，供离线独立计算后比对；v2 两事件槽，v3 一个 sequence 槽 |
| v3 额外数组 | `event_ends / actual_substeps / unplanned_wrench`，用于核对脉冲结束、实际 physics-hook 子步计数和非计划 wrench |

metadata 保存 `step_dt_s / physics_dt_s`、量化后的事件/窗口/dwell 步数、协议阈值、坐标定义；force 额外记录作用 body 名称/索引、总质量 kg、pulse steps、expected substeps、decimation。**没有归档完整逐子步力向量曲线或接触序列**；注入 delivery 验证与事件 impulse 属于既有 hook 的测量输入，离线 reducer 不声称重新测量物理注入。

- **v2 velocity recovery**：每次速度扰动以该 event 的注入步为 anchor，各自需要连续 stable dwell；最终还要满足终局 stable。
- **v3 force recovery**：完整序列已完成且 verified 后，以最后一个 pulse 的结束为 anchor，要求一次新 sequence dwell；不是把两个独立 event 的恢复率简单平均。最终仍需 stable，恢复后再失稳不能算成功。
- 两者窗口均为 `0 < control_step-anchor <= recovery_window_steps`；首次达标恢复时刻锁定。阈值按原 pose dtype 比较，避免离线换精度在边界上改变标签。

`data_complete=true` 只指归档完成；`episode_coverage=complete/truncated` 与每 env 的 `closed / valid_steps` 单独报告。无效 padding 不计入统计。复算输出 `observed_window_steps / stable_window_steps / max_dwell_steps / failed_condition_steps`、dwell 中断的控制步/相对秒数/失败条件集合，以及 `final_unstable_after_recovery`。窗口覆盖明确区分 `no_push_control / recovered / anchor_not_observed / invalid_injection / full_window_without_dwell / terminal_before_window_end / rollout_truncated`；未到 anchor 或未观完整窗口不能当作零恢复时间或完整窗口失败。截断 episode 仅有 snapshot，`terminal_label=null`。

## 训练 reward 采样：实际值，不重新运行候选

`runner/training_numeric_capture.py` 只负责索引/分母/存储；Stair dispatcher 与 Push adapter 在原 reward 调用后读取已有值。没有拆分 `learn`，也没有为采样再次调用 reward。reference 与 candidate 使用同一采样位置。

| 字段 | 含义 |
|---|---|
| `raw_term_<index>` | 已经过输出 guard 和 clip、尚未乘权重的实际 term 值；**不是函数返回前或 clip 前原值** |
| `weighted_term_<index>` | Stair：raw × 声明 weight；Push：raw × 声明 weight × 当步 `reward_weight_scale` |
| Stair `total_rate` | dispatcher 实际加总 rate，随后由权重为 1 的 RewardManager 再乘一次 `control_dt_s` |
| Push `input_reward / candidate_total / returned_reward` | 实际输入奖励、实际候选加总、二者相加后的返回值；physical complete-reward 分支仍是 rate，外层乘一次 dt；legacy additive 分支已是 per-step reward |
| semantics | term `index/name/weight/clip_min/clip_max`、rate/per-step 含义、`control_dt_s`；训练代码自己拥有公式与权重来源 |
| 每采样步 metadata | `source_control_step / training_control_step / local_rollout_iteration / learning_iteration_from_control_steps / step_in_rollout / runner_reported_learning_iteration`；最后一项只是 runner 当时报告值 |

开始 `learn` 前冻结 `source_step_offset`，排除 activation preflight/构造阶段的 reward 调用。`planned_control_steps=iterations×steps_per_env`；当前训练采样 v2 在计划的 early/middle/late 三个互不重叠三等分内各等距分散取最多 8 点，early 首点改为第 1 步启动锚点，每步固定前 `min(num_envs,16)` 个 env。每段 `planned_steps` 与实际 `sampled_steps` 列在私有档案，作者仍只接收有界覆盖簿记。48000 控制步时为 1,3000,…15000 / 17000,…31000 / 33000,…47000。最大仍仅 384 个 env-transitions；不表示全训练均值、精确首次激活时间或随机代表性。旧 v1 连续窗口按原 schema 解读。

coverage 保存计划/已观察控制步、全体 env 数、采样 env IDs、每窗范围、计划/实际采样控制步和 env-transitions。窗口状态为 `sampled / partial / not_reached / not_sampled / not_scheduled`；采集整体为 `complete / not_enabled / not_started / incomplete`。失败前的部分档案可以 complete，但仍带 `training_status=fail`。Stair 128-step smoke 的三窗为 2/3/3 控制步，属于同一次 rollout；不能充当多次 PPO 更新后的早中晚训练证据。

保存的 raw/weighted/total 可离线复算加权关系及顺序加总；未另行存储 PPO 实际接收的 post-manager reward。不得把 task 声明的 dt 关系说成新采集了一条 PPO 消费链。

## 来源绑定、移动与作者可见边界

eval 档案绑定 checkpoint、job、evaluation plan/binding、实际 cell、seed/run/split 和执行环境；Stair 额外绑定已验证 source preflight 与冻结 suite/semantics。训练档案绑定 job/attempt、程序 checksum/qualification、training binding、seed/init/initial policy hash、run dir、cell 和执行环境。Stair 的 training binding 可合法保留祖先协议；数值档案的 `task_protocol_hash` 则属于当前实际 resolved cell，两者分别校验，不能强求相等。**训练程序来源标识不表示 eval 中又执行了一遍候选 reward**。public-dev 数值引用在各任务的 phase/task evidence 中，不扩张 accept/sealed 的作者可见范围。

manifest 内 chunk 文件名是相对路径，数值不依赖原 GPU 内存，可在有对应代码及 CPU 依赖的机器离线复算。引用的 `manifest_path` 仍是文件定位器；搬迁档案时应显式给离线读取器新 locator，保留 manifest 原字节/SHA与其中历史来源，不重写旧 phase receipt 来假装原路径未变。这个能力是数值解码与归约回放，不是重跑 simulator，也不豁免真实策略 replay 的 checkpoint/ABI/合法祖先准入检查。

作者 LLM 继续接收原任务公开结果、分层汇总、失败/诊断计数与受预算约束的 runtime feedback。新增训练采样公开投影只传覆盖状态与内容引用，不把 NPZ、逐步姿态、完整 semantics、路径或私有 seed 明细塞进 prompt。公开字段以最终 `runtime_feedback_summary.latest_runtime_activation.training_sampling` 为入口；精确字段在下节列出。

P2 没有交付视频/图片采集、视觉模型诊断、任意档案读取工具、按预算主动补查、跨模态记忆策略或最终 RL 收益验证。Push 新生成的恢复中断时序仍是离线分析结果，不能称为当前作者已经看见。P3+ 应在这些实测数值之上定义诊断、可见范围和验收，不能从“归档存在”直接推导解释正确或改进有效。

## 作者可见的训练覆盖投影

来源是训练 receipt 的 `training.numeric_evidence`。共享 allowlist 同时用于 `TrainJobResult` 的公开训练摘要和 `runtime_activation_summary.training_sampling`，任务再将后者放入 `runtime_feedback_summary.latest_runtime_activation.training_sampling`。这是覆盖事实，不是新诊断；不新增 14 KiB 长期实验记忆表的列。

| 公开部分 | 字段 |
|---|---|
| 状态与范围 | `status / scope / training_status / capture_schema_version`；scope 明确固定窗口及环境子集，不代表全训练分布 |
| 分母与时钟 | `planned_control_steps / observed_control_steps / num_envs / sampled_env_count / steps_per_iteration / source_step_offset_excluded` |
| 三窗 `windows` | 固定 early/middle/late，各有 `name / status / start_step / end_step / planned_sample_steps / sampled_control_steps / sampled_env_transitions / planned_sampled_env_transitions` |
| 内容引用 | `term_semantics_hash` 为原 task-owned semantics 整体 hash；`archive_manifest_sha256` 指实际附件 manifest 字节 |

非 complete 仅公开 `status`，保留 `not_enabled / not_started / incomplete`，旧数据缺采集为 `not_available`。窗状态保留前述五种，缺窗口为 `not_available`。`complete` 不覆盖 `training_status=fail` 或窗口 partial/not_reached；没有原 term 数组、精确采样 step/env ID 列表、完整 semantics 或绝对 manifest 路径。

有合法、已绑定训练失败 receipt 的 `CandidateRuntimeError` 分支也经同一 allowlist，将可选 `details.training_sampling` 传给 round/generation 的 runtime 摘要；不伪造 `TrainJobResult` 或训练完成。没有可读取 receipt 的强制终止/基础设施失败不能据此推定已有 partial 数值覆盖。此桥接是数据传递合同；失败链的 CPU fixture 不等于真实 GPU 候选崩溃验收。

## 代码真值入口

共享：`runner/numeric_evidence.py`、`runner/training_numeric_capture.py`。Stair：`tasks/stair_climbing/numeric_capture.py`、`evaluation_runtime.py`、`transition_decoder.py`、`episode_evidence.py`、`phased_evaluation.py`。Push：`tasks/push_recovery/numeric_capture.py`、`runtime.py`、`force_runtime.py`、`phased_evaluation.py`。训练值：Stair `reward_program.py / isaaclab_execution.py`，Push `runtime_adapter.py` 与 `scripts/isaaclab_rsl_rl_train.py`。这些相对路径均以本仓库 `src/true_rl_agent` 为前缀（`scripts` 除外）。

## 冻结源码一致性核查

[source_equivalence_audit.json](source_equivalence_audit.json) 逐文件比较当前 `src/configs/scripts/requirements/pyproject.toml/README.md` 与 r2/r5/r6 的 385 项冻结 SHA；不包含 PLAN、tests 或 Python cache。三组均无新增/删除：r2、r5 各 375 项相同、10 项不同；r6 为 378 项相同、7 项不同。JSON 保存每项当前/历史 SHA、差异用途及明确列出的 runtime 文件组，发布前再次读取全部当前文件以排除生成期间修改。

Stair/Push eval runtime 与 r2 完全相同；训练入口、reward/runtime、数值 I/O 与 sampler 与 r5/r6 完全相同，当前 smoke driver 与 r6 完全相同。r6 后 7 项变化均为公开投影/失败回执消费者。r2→当前还包含 smoke **venv 启动路径**修复，不能一概称为只有 checker 改动；r5→当前的 driver 差异是已述协议身份 checker 修复。整个生产树并非逐字节相同，新增消费者的 CPU 验收与真实 simulator 验收仍分别记账。

本轮新增 9 个 Python 模块/脚本无 PLAN 字面路径或 import。已追完旧 Push `upstream_source_path`：PLAN 字符串参与 manifest 来源身份，但真实读取的是 `src/.../standing_recovery_stationary_v3/training_code/components.py`，随后物化并执行已校验的 artifact bytes，不打开该 PLAN 路径。范围不应扩大为“仓库完全不依赖 PLAN”：旧 `stair_extension_schema.json` 的 `phase0_report.ref` 仍可由历史 `StairPhase0Bundle.load_report` 读取，属于三快照均未改变的旧接口。

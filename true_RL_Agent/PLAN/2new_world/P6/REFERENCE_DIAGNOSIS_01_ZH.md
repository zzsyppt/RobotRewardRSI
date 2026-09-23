# P6 首轮 reference 诊断与有界 dev 后续

2026-09-12。本文只读分析已下载的 [训练收据](completed_training_evidence_01/) 与 [评测收据](formal_evaluation_evidence_01/)，不启动 SSH、GPU 或新训练，不改原程序、V2 协议和已消费的 holdout。目标仍是得到可恢复且留有改进空间的 reference；本轮阴性资格结果是后续开发的起点，不是缩减 P6 目标。

## 实测事实

两格均完成 random-init、500 iterations、4096 × 24 × 500 = 49,152,000 env steps；训练源相同，为 `push_standing_recovery_reference_v2`，源码 SHA256 为 `f5b0faee114283ab96056cacbd37b346f797d53984daf6c18f52e13cbbd2506e`。当前 `runtime.py`、`runtime_adapter.py` 和 reference 源字节与原训练执行环境收据中的哈希一致。

| 原始 public-dev，3 seeds × 64 episodes | G1-37 | Go2 |
|---|---:|---:|
| 完整恢复 episode | 0/192 | 192/192 |
| 实际有效 / 计划推扰事件 | 240/240 | 240/240 |
| 恢复事件 | 0/240 | 240/240 |
| 第一次推前角速度范数超恢复门 | 179/192 | 0/192 |
| 第一次推前平移速度超恢复门 | 162/192 | 0/192 |
| 第一次推前角速度范数中位数，rad/s | 5.62548 | 0.043811 |
| 第一次推前平移速度中位数，m/s | 0.436361 | 0.002091 |
| 第一次推前按 reference 公式重算 stationary 中位数 | 0.278761 | 0.999959 |

G1 dev 中 191 个 `not_recovered`、1 个 `out_of_bounds`，没有跌倒分类；所有 episode 的最大绝对 roll ≤ 0.17785 rad、pitch ≤ 0.14528 rad、最低 base height ≥ 0.60234 m。末端只有 1/192 个 episode 满足瞬时完整稳定条件，且没有事件取得窗口内连续驻留成功。首次推前已普遍不静止，不能只归因为推扰太强。

G1 第一个 dev episode 在首推前 yaw = −11.71659565 rad/s，约 −1.865 转/秒。其 L = 0.74 m、T = sqrt(L/g) = 0.2746511 s；根据该时刻的完整六维速度，`exp(-2 * ||v_xy/sqrt(gL)||² - 0.5 * ||omega*T||²) = 0.00430606`。这是高速区奖励接近零的实例；整批首次推前 stationary 中位数约 0.279、均值约 0.335，不能写成“全体已饱和为零”。

Go2 的 dev 四档全部恢复。low/medium/high/repeated 的事件恢复耗时中位数分别为 0.36/0.40/0.71/0.40 s；high 最大 1.12 s，仍低于约 1.62 s 的恢复窗口。这是当前样本对该 checkpoint 缺少区分力，不能解释为任意扰动均能恢复。

已消费的 public-accept / sealed 汇总分别为 G1 0/192、0/448，Go2 192/192、448/448；仅保留其首轮资格事实，以下设计依据使用 dev 和训练收据，不用这两卷挑系数或重复验收。

## 源头与统计核查

- `runtime_adapter.py` 使用 `root_lin_vel_w / sqrt(gL)` 和 `root_ang_vel_w * sqrt(L/g)`；恢复门在 `runtime.py` 使用同一 world 速度，角速度门为 `0.5 / sqrt(L/g)`。角速度范数不因 body/world 旋转而改变，没有发现角度/弧度或缩放方向错误。
- 两格 dev 的全部 240 次事件在写入前后，z 线速度和三个角速度严格不变；xy 请求/观测最大误差分别为 G1 `1.19e-7`、Go2 `1.49e-8` m/s，均有下一物理步收据。G1 的推前自旋不是 push 写入添加的。
- 原 policy I/O 收据包含有效 `base_ang_vel` 观测，37/12 动作维数、关节顺序、实际 action scale、控制步长检查通过。记录来自 task pre-reset state；完整恢复依赖实际事件、窗口内连续驻留和末端稳定，未从 reward 或 reset 后状态合成。
- 两格每个 seed 的成功率分别恒为 0 和 1，取均值不会改变结论；分母和事件整数闭合，无无效注入。未发现本次 0/1 由均值算法或漏计事件造成。

## 可解释与不可证明的部分

reference 的正项是 `3 * upright_height + stationary`，再减 joint pose、action rate 和终止惩罚。G1 训练最后一个采样步的加权均值分别为 `2.645173 + 0.135368 - 0.014232 - 0.106556 - 0.019531`。保持直立却持续运动仍可获得主要正回报；指数 stationary 在高速区对速度改善的回报差别很小。这支持“奖励形状使静止恢复难以继续优化”的假设。

这不是 PPO 局部最优或收敛的证明：只有一次训练 seed，没有反事实奖励训练，也没有整段优化轨迹和逐条件驻留时序。PPO 不直接对物理速度求该指数的导数；不能把指数导数小等同于已证明策略梯度消失。`runtime_activation_summary` 是最后一次采样及调用汇总，不是整轮平均学习曲线。G1 也不是全部样本都处于指数饱和区。

长度归一化只统一这套无量纲物理尺度，不保证两种支撑结构、惯量、接触和执行器具有同等恢复难度。现有证据不能把 Go2 的满分归罪于计分 bug，也不能据此宣称 G1 策略能力不足。

## 最小接线：独立 dev reference，不经过资格 CLI

现有权限足够。Push V2 binding 已声明完整 reward program 替换、非空 reference、RewardManager 对 rate 总和只积分一次 dt；不需要给共享宿主新增机器人或程序特判。新实验必须有独立 `reference_id`、source checksum、subject manifest hash 和输出目录，不能继续声称执行 `push_standing_recovery_reference_v2` 原字节。原 task/robot 物理成功门、V2 默认 reference 和原 holdout 不改；新程序尚不是合格默认 anchor。

1. 将新程序和实验驱动放在 P6 独立 development 目录，事前冻结实际源字节与预算。两个机器人使用同一公式，只改变 stationary 形状为高速区仍有区分力的速度惩罚，其他四项和任务成功条件保持原样。系数应先经静态回报排序检查，防止主动跌倒或逃离 episode 更划算；本文件不选择或写入新系数。
2. 复用 `resolve_reference_program_artifact` 读取独立源，显式提供 `task_id=unitree_push_recovery`、新 reference 身份和现有 `push_complete_recovery_v2` capability hash。backend 的 `materialize_reference_jobs` 验证 task/capability 后按实际 manifest/source 建立新作业身份，不硬编码默认 reference ID。
3. 复用 `PushRecoveryCompositionProvider` 的两格物理配置与 `PolicyInitialization.random()`，显式固定 seed、4096 env、24 steps/iteration、500 iterations；通过 `materialize_reference_jobs` → `SubprocessRunner.run_train` 执行一次全预算新训练。新的 dev 实验有单独成本上限及 0 自动重试，不能视为旧资格训练的追加或恢复。
4. 用 `bind_eval_spec_to_checkpoint` 将 eval plan 绑定本次 runner 封存的真实 checkpoint，再 `materialize_eval_phase(...PUBLIC_DEV)` → `run_eval_phase`。只执行预声明的 dev cohort，保存真实 phase result；不调用 `run_eval`、三相 assembly、`qualify_reference` 或 Genesis。P6 已有 `push_pipeline_probe.py` 证明这条生产链；复用其结构，不修改旧 probe 的 2-iteration 历史合同。
5. no-push 是单独原始诊断，不能直接塞进正式 `run_eval_phase`：当前 `build_push_phase_summary` 调用 `project_push_split`，明确拒绝 control-only 分母。复用现有 `physical_runtime_config_from_manifest(..., "no_push")`、原生 checkpoint loader 和 `run_physical_push_rollout` 的 diagnostic 路径，保存 task raw summary；不放宽正式 phase 校验。另记录角/线速度各条件超限时间和最长连续稳定时间；已有 `control_success` 只表示未倒、未越界，不能代替静止恢复。

开发结果的判据是 G1 静止与恢复是否有实质改善、两格行为与 reward 分解是否一致；只改 stationary 本身未必解决 Go2 满分和任务难度问题，不能预先承诺下一轮资格通过。若 dev 支持进一步采用该程序，再单独冻结新的正式 reference/实验身份、预算和未消费的 accept/sealed。当前外部 VLLM 占用资源，先完成无 GPU 的源、身份和作业预检；本文件不宣称已启动新实验。

## 本地接口核实

在临时目录中，用原 reference 源字节配一个仅用于探针的新 reference ID，分别实际调用两格 composition、reference resolver、`materialize_reference_jobs` 和 train/eval argv validator：两格均通过；subject manifest 与默认 reference 不同，预算仍为49,152,000、500 iterations、random init。没有调用 runner、环境采样或 GPU，没有写入新的奖励实现。此检查证明接口接受独立 artifact，不证明尚未编写的新 stationary 程序可学习。

默认 task 的 `fixed_train_protocol.reference_program_id` 仍为 `push_standing_recovery_reference_v2`，表示原配置的默认 anchor；独立 dev 作业的 subject/reference manifest/source checksum 另行绑定。开发回执必须显式保存实际 artifact，不能仅凭默认 composition snapshot 把新程序称作原 reference，也不能把此独立作业直接当作旧配置的资格 Genesis。

另用两份已下载的真实 `target_wiring_evidence_01/{g1,go2}_02/no_push.json` 调用 `project_push_split`，均明确返回 `recovery_formal_cohort_must_have_planned_pushes_only` 和缺少正式 success rate。由此确认 no-push 必须走独立 diagnostic 路径；无需修改正式 phase/计分合同。文档 whitespace 检查通过。

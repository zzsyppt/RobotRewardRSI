# Stair 奖励源码冷启动 15 轮实验审计

## 1. 运行身份

- 状态：按用户要求暂停；正式账本保留 9 轮，第 10 轮训练未完成且不计入正式结果。
- 代码提交：`ddc249e`。
- 启动模式：`reward_cold_start`，Genesis 不含 R0、reference artifact 或 baseline checkpoint。
- 会话：`stair-cold-formal-ddc249e-20260808T183401Z`。
- 服务器代码：`/home/yuyangliu/AGENT_COLD_DDC249E/true_RL_Agent` 独立 detached worktree。
- 输出：`/home/yuyangliu/true_rl_agent_cold_experiments/formal_15round_reward_cold_start_ddc249e_20260808T183401Z`。
- 训练设备：进程环境固定 `CUDA_VISIBLE_DEVICES=0`，作业设备为 `cuda:0`。
- 对照实验：GPU1 上的 R0 会话使用服务器正式仓库 `e24777b` 完成 15 轮；两边未共享代码工作树、Genesis、ledger 或输出目录。

## 2. 启动前资源与安全检查

- 清理 `/data/yuyangliu/cache/pip` 的可再下载缓存以及四个已确认无效、无进程引用且被后续正式证据取代的实验目录后，`/data` 可用空间从 19 GiB 增至 33 GiB。
- 环境、代码、完整正式运行、当前 R0、Phase E completion manifest 指向的证据和仍有诊断价值的运行均保留。
- GPU0 连续采样约有 31.5 GiB 空闲；主机约有 269 GiB 可用内存，系统负载低。
- 服务器隔离 worktree 上 34 个冷启动、组合、验收和 replay 测试以及 31 个子测试通过；入口帮助与 `compileall` 通过。
- 冷实验启动后，GPU1 没有出现新的冷实验 CUDA 进程。R0 在冷实验启动前后的平均 iteration 时间分别为 1.7894 s 和 1.7885 s，未观察到吞吐回退。

## 3. 审计原则

以下情况属于硬错误，需要停止、修复、重新验证并使用新会话重跑：

1. 冷启动 prompt、Genesis、lineage 或作业中出现 R0/reference 源码身份；
2. 训练不是固定随机初始化，或实际 initial-policy hash 与 Genesis 锚不一致；
3. 候选完整奖励没有成为 PPO、episode 累计和训练证据的唯一奖励来源；
4. public/sealed 七桶证据不闭合、跨轮协议漂移或 replay 不能恢复相同 current-best；
5. 冷实验明显影响 GPU1 R0、其他用户进程或主机稳定性；
6. prompt、公开记忆或失败反馈无界增长，或者把私有/sealed 信息暴露给 provider。

候选奖励质量差、早期训练失败、某轮得分为零或 provider 的解释不准确，不自动构成系统硬错误。只要安全门、固定预算和评估闭环仍有效，就由正式评估决定接受或拒绝。

## 4. 第 1 轮

### 4.1 Prompt 与记忆

- 首次 authoring 仅使用一个 provider attempt，prompt 为 46,378 字节。
- `candidate_lineage.status=none`，current-best、parent checksum 和 best-observed experiment 均为空。
- `source_payloads` 为 0，recent outcomes 为 0，omitted rounds 为 0。
- prompt 不含官方 R0 ID、R0 源码 checksum、`reference_anchor` 或隐藏 reference 数据。
- 冷启动约束各出现一次：无 reference reward、无 hidden base reward、必须返回 complete replacement reward program。
- 80 字符以上的完全重复字符串只有一条组件签名，出现两次；没有发现大段精确重复。
- 最大固定区段为 `reward_inputs_contract` 约 9.4 KiB 和 `required_candidate_package` 约 8.6 KiB，均提供候选可执行所需的 ABI/安全证据，不属于跨轮历史膨胀。
- 与同协议 R0 首轮 prompt 做逐区段 canonical-byte 对比后，`task`、`loop_contract`、`required_candidate_package`、`candidate_capabilities`、`reward_inputs_contract`、`public_eval_contract`、`control_effectiveness_contract`、`runtime_feedback_summary` 和空历史完全相同。
- 冷 prompt 少约 11 KiB，差异主要来自移除 R0 `source_payloads`、reference lineage 和 reference public feedback，以及加入少量冷启动探索指令。任务知识和安全合同没有因无 R0 而被弱化。

### 4.2 候选与训练早期状态

- 候选：`stair_baseline_v1`，源码约 3.0 KiB，包含 8 个完整奖励项。
- 机制覆盖 goal progress、torso contact、roll/pitch、action delta、joint acceleration、joint torque、default pose 和 foot support。
- 候选通过静态验证、封存和 Stair task-owned smoke 后才进入训练。
- `goal_progress_fraction` 仅有下界、没有上界，可能产生大贡献；该风险应由 runtime contribution evidence 和固定评估裁决，当前不人工改候选或协议。
- iteration 75 时 mean episode length 为 32.5，base-contact termination 为 1.0，terrain level 为 0；iteration 118 时 mean reward 从 -2.57 改善到 -1.23，但仍是 1.0 base contact。数值、loss 和 policy noise 均有限。
- 该表现符合真正从零奖励探索的困难，不足以证明最终失败，也不构成运行时错误。

### 4.3 第 1 轮正式结果

- 训练完整执行 `4096 x 24 x 2000 = 196,608,000` env-steps，最终 checkpoint 为 iteration 1999；训练状态、预算核算和 checkpoint 安全检查均通过。
- 实际初始策略状态 hash 与 Genesis 固定随机锚一致；TrainingBinding 与 R0 正式实验相同，差异仅为独立 GPU 设备和无 R0 的任务自有启动模式。
- 完整奖励运行时处于 `active_pre_reset_complete_reward_v1`，8 个奖励项均被调用并产生非零贡献；训练 recorder 保持关闭。
- `public_dev`、`public_accept` 和 sealed validation 均覆盖七个高度。`public_accept` 的 5 个 seed、280 个 episode 全部为零成功，所有 primary outcome 均闭合为 `torso_contact`，无分类警告。
- `public_dev` 平均最大前进约 0.644 m，`public_accept` 约 0.656 m；这只能证明存在有限前向移动，不能替代完整登顶并保持的成功状态机。
- public score 为 0，低于首次接受绝对门槛 0.20；sealed validation 因 holdout floor 未满足而失败，因此第 1 轮以 `required_gate_failed` 拒绝。判定没有把训练奖励改善或 episode 变长误判成上楼能力。

### 4.4 第 2 轮提示词与静态修复

- 正常第 2 轮 prompt 为 60,362 字节。固定 ABI/安全区段保持不变，新增内容来自第 1 轮源码、七桶公开失败反馈、运行时奖励贡献和一条受限历史记录。
- 第 1 轮源码只以内联 payload 交付一次；best-observed 条目明确标记为“authoring anchor only”，`current_best` 和 parent checksum 仍为空，也没有出现 R0/reference 身份。
- 公开反馈包含七档零成功、`torso_contact=1.0`、slip/stall diagnostic rate、各档进度和 completion gap；运行时反馈包含全部 8 个奖励项的 bounded contribution summary。
- 长字符串精确重复仅有一个 158 字节源码摘要出现 3 次、一个 161 字节组件签名出现 2 次；没有发现大段源码、历史或协议重复。
- provider 首次生成的 `slip_penalty` 对环境维度执行全局 `mean()`，返回标量。静态 verifier 通过，但 Stair task-owned smoke fixture 以 `reward_output_wrong_shape` 拒绝，未启动训练。
- repair frame 为 67,486 字节，保留稳定合同、R1 最佳未接受源码、本次失败源码和精确 shape 反馈；省略可从账本恢复的 recent-outcome 明细。第二次 provider attempt 修复为逐环境输出，9 个奖励项全部通过 smoke 后才进入训练。
- 修复候选 `stair_slip_v2` 仍从同一固定随机策略开始、使用相同 TrainingBinding 和完整 2000 iteration 预算。它移除静态 foot-support 奖励、限制异常大的关节加速度贡献、加重 torso-contact，并新增逐环境 slip/height-scan 项，机制与第 1 轮公开失败证据一致。

### 4.5 第 2 轮结果与第一次会话中止

- 第 2 轮完整执行相同的 196,608,000 env-steps，initial-policy hash 再次与 Genesis 一致；训练、运行时激活、checkpoint 和评估身份链均通过。
- `public_accept` 仍为 5 个 seed、280 个 episode 全部零成功和 `torso_contact`，sealed holdout 未达到 floor，因此合理拒绝。
- 第 2 轮 `public_dev` 平均最大前进仅约 0.126 m，`public_accept` 约 0.147 m，明显低于第 1 轮约 0.644/0.656 m；训练 episode length 也长期停留在约 17.5 步。更重的终止惩罚没有改善接触，反而形成更快终止的退化机制。
- R3 prompt 正确保留两轮公开反馈，但 `candidate_lineage.best_observed_experiment` 错误指向第 2 轮。根因是 shared source lineage 与 Stair feedback 各自使用 `(public_score, latest_index)`：两轮 score 同为 0 时，最新但进度明显更差的源码覆盖了较优 bootstrap authoring anchor。
- 这不会污染 accepted current-best，因为两轮都未接受；但会给 LLM 错误的“best observed”身份，因此属于继续实验前必须修复的提示词/记忆语义问题。
- 发现时 R3 训练刚启动。先终止主循环，再发现训练子进程已孤立，随后单独终止并确认 GPU0 回落到空闲；账本保持 2 条完整记录，R3 未完成产物不进入正式账本。

### 4.6 同分排序修复边界

- shared prompt source catalog 只新增一个可选的 task-owned、public-only、有限数值 tuple 回调。正式 public score 始终为第一排序键，任务键只能在 score 相等时选择 authoring source，最后才按 recency 打平。
- Stair 使用公开 `public_dev.prompt_feedback.progress_summary.mean_max_progress_m` 作为同分次序；缺失或非法进度得到固定低优先级。该语义完全留在 Stair prompt adapter。
- Stair `public_feedback_summary` 与 shared source lineage 复用同一个 Stair 排序 helper，避免再次形成两套 best-observed 真相。
- Push 和 neutral adapter 返回空 tuple，因此继续使用原 `(public_score, recency)` 行为；accepted current-best、正式 acceptance、sealed gate、ledger replay 与 policy lineage 均未改动。
- 回归覆盖同分高进度胜出、任务键不能覆盖更高 public score、非法键 fail closed、较优/最新失败源码同时交付，以及 Push/Stair/live-loop/Genesis/replay 全链。

### 4.7 修复后的正式冷启动会话

- 修复提交为 `ddc249e`；会话为 `stair-cold-formal-ddc249e-20260808T183401Z`，使用独立 worktree、独立输出目录和 GPU0。GPU1 上的 R0 会话未共享代码目录、Genesis、ledger 或训练设备。
- 新 Genesis 再次确认 `reward_cold_start`、固定随机初始策略和空 reference/current-best/best-observed lineage；首轮 draft prompt 为 46,378 字节，未出现官方 R0 标识或 reference 源码。
- 第一次候选 `stair_progress_v1` 在任务 smoke 中失败：`base_contact` 将具有环境批次维度的 tensor 强制转换为 Python 标量。失败证据包含异常类型、奖励项名称、任务 fixture 身份、失败源码逻辑 ID 和仅修改候选源码的修复约束，训练未启动。
- 第二次候选 `stair_progress_v2` 修复了批量输出，但 `angular_vel` 对二维 fixture 使用固定第三维索引而失败。第二个 repair prompt 保留上一候选源码和精确 `IndexError`，没有放宽 verifier、训练预算或协议。
- 第三次候选 `stair_progress_v3` 将向量惩罚统一改为末维归约；7 个奖励项均在 Stair task-owned smoke 上返回有限、逐环境输出，随后才完成封存并进入固定 2000 iteration 训练。
- 两次 repair prompt 分别约 53.9 KiB 和 54.3 KiB，增长来自失败源码和结构化 smoke 证据；修复目标连续收敛，没有重复同一错误，也没有把任务 shape 语义推入共享层。
- 训练启动后的采样显示 GPU0 单独承载冷启动训练，GPU1 的 R0 进程仍保持活动；未发现跨设备进程或输出目录混用。
- 第 500 iteration 检查点中，`Train/mean_reward` 从初始约 -47.3 改善到 -1.42，但 mean episode length 约 19、`base_contact=1.0`、timeout 为 0，terrain level 从初始约 1.68 降到 0。这是“更好地最小化候选奖励、但未形成上楼能力”的早期退化证据；它不触发人工接受或中止，仍由完整 public/sealed 评估裁决。
- 第 1000 iteration 时该模式未恢复：mean reward 约 -1.18、episode length 约 19、`base_contact=1.0`、terrain level 0；value loss、surrogate loss、policy noise 和 FPS 均有限稳定。因此当前是奖励机制诱导出的稳定错误策略，而不是数值发散或训练基础设施失效。
- 第 1 轮完成全部 2000 iteration 后，预算核算为 `196,608,000` env-steps，实际 initial-policy hash 与 Genesis 锚一致，源码、TrainingBinding、runtime activation 和 checkpoint 字节摘要链均通过。
- 第 1 轮 public-dev 的 168 个 episode 与 public-accept 的 280 个 episode 均完整覆盖七档高度，成功数均为 0，所有 primary outcome 都闭合为 `torso_contact`；public-dev/public-accept 平均最大进度分别约 0.111 m/0.128 m，stall rate 为 1.0。public score 为 0，sealed holdout floor 未满足，因而以 `required_gate_failed` 合理拒绝并原子写入一条 ledger 记录。
- 第 2 轮 draft prompt 为 60,962 字节，`current_best` 仍为空；第 1 轮仅作为未接受的 best-observed authoring anchor。反馈同时包含七桶失败、接触/停滞/打滑诊断以及逐奖励项贡献，其中 hip/knee acceleration 的最大绝对加权贡献约为 9203，明显高于其他项。
- 第 2 轮候选 `stair_clip_hip_knee_acc` 一次通过验证和 task-owned smoke。它只为上一轮的 hip/knee acceleration 项增加 `clip_max=10.0`，其余完整奖励源码保持一致，并从相同初始策略、相同 TrainingBinding 和相同 2000 iteration 预算重新训练。这是一个可归因的单机制实验；裁剪是否过强由正式评估决定。
- 第 2 轮到 500 iteration 时，mean reward 从初始约 -1.57 改善到 -0.08，value loss 接近 0，但 episode length 约 17、`base_contact=1.0`、terrain level 0，policy noise 从约 1.00 降至 0.41。该裁剪修复了奖励尺度和优化难度，却更快固化为低代价接触终止；是否有任何公开进度改善仍等待完整评估。
- 第 2 轮完整训练和 checkpoint 身份链通过，但 public-dev/public-accept 的 168/280 个 episode 仍为七档 0 成功、全部 `torso_contact`，mean max progress 仅约 0.043 m/0.056 m，低于第 1 轮约 0.111 m/0.128 m；sealed holdout floor 再次失败，因而合理拒绝。
- 真实第 3 轮 prompt 验证了同分排序修复：两轮 public score 都为 0 时，`best_observed_experiment` 保持指向进度更高的第 1 轮 `stair_progress_v3`，而第 2 轮 `stair_clip_hip_knee_acc` 作为独立 `latest_failed` 源码交付。两份源码各内联一次，`current_best` 与 parent checksum 仍为空，未接受候选没有获得策略继承权。
- 第 3 轮首稿只返回无效文本，被 verifier 以源码首行语法错误拒绝；第二稿直接重复第 1 轮源码，虽通过 verifier/smoke，但被 anti-degenerate 门以 `historical_mechanism_repeat` 阻断。第三稿 `stair_v4_cmd_vel_air_time` 新增命令速度跟踪与 foot-air-time 机制，通过全部 authoring 门后才进入训练。
- 第 3 轮第二次 repair 中，latest-failed 与 best-observed 恰好是相同字节时，prompt 仍以两个逻辑 ID 重复内联约 3.2 KiB 源码。角色身份需要保留，但内容可按 hash 去重并由 alias 引用；这是非阻断的提示词冗余，不能在当前会话中途改变协议。
- 第 3 轮到 500 iteration 时，早期较长 episode 已回落到约 19，`base_contact=1.0`、terrain level 0；新增速度/腾空机制没有维持住上楼行为，数值与吞吐仍稳定。
- 第 3 轮正式评估仍为七档 0 成功、全部 `torso_contact`，public-dev/public-accept mean max progress 约 0.108 m/0.125 m；它显著优于第 2 轮但仍略低于第 1 轮的 public-dev 约 0.111 m，因 sealed floor 失败而合理拒绝。
- 第 4 轮 prompt 约 79.4 KiB，包含 3 条公开历史且无省略；同分排序继续保留第 1 轮为 best-observed，并把第 3 轮作为 latest-failed。未接受源码仍未成为 current-best 或训练 warm start。
- 第 4 轮首稿只提高 orientation 权重，被 anti-degenerate 门识别为纯常数调整并阻断；repair 后的 `stair_add_slip_penalty_v1` 新增接触足线速度平方惩罚，直接对应第 3 轮 `slip_detected=1.0`，通过 verifier、task smoke 和 anti-degenerate 后进入训练。
- 第 4 轮到 500 iteration 时，mean episode length 约 35.5，明显高于第 1/3 轮约 19 和第 2 轮约 17；policy noise 约 1.21，数值稳定。但 `base_contact=1.0`、terrain level 仍为 0，因此只能确认稳定性机制改变了训练行为，不能在正式七桶评估前声称上楼能力提升。
- 第 4 轮正式评估显示 public-dev/public-accept mean max progress 约 0.570 m/0.582 m，相比此前最佳第 1 轮约 0.111 m/0.128 m 提升约 5 倍；但七档仍全部 0 成功、全部 `torso_contact`，且 orientation-instability/slip/stall diagnostics 接近或等于 1.0。因此 score 保持 0、sealed floor 失败并合理拒绝，不能把前进距离替代完整登顶保持能力。
- 第 5 轮 prompt 将第 4 轮选为新的 best-observed authoring anchor，并仅内联一次同为 latest-failed 的源码；同分 task-owned progress tie-break 成功保留了真实行为改进，同时没有改变 current-best 或 acceptance。
- 第 5 轮 `stair_bound_hip_knee_acc_v2` 在第 4 轮 slip 机制上把 hip/knee acceleration 从平方和改为绝对值和，并设 `clip_max=10.0`；这是对第 2 轮失败裁剪与第 4 轮有效稳定机制的组合复验。
- 第 5 轮到约 514 iteration 已退化为 episode length 约 17、`base_contact=1.0`、terrain level 0、policy noise 约 0.42，第 4 轮约 35-38 步的稳定性优势完全丢失。该证据表明当前 hip/knee 裁剪会压制有效动态行为，而不是与 slip 机制互补；仍需完整评估后进入正式记忆。
- 第 5 轮正式 public-dev/public-accept mean max progress 仅约 0.053 m/0.067 m，远低于第 4 轮约 0.570 m/0.582 m；七档仍 0 成功、全部 `torso_contact`，因此正式拒绝。第 6 轮继续保留第 4 轮为 best-observed、把第 5 轮作为 latest-failed。
- 第 6 轮 prompt 约 84.9 KiB；recent outcome 开始按固定上限省略 1 条旧记录，但 best-observed 源码、latest-failed 源码和 bounded experiment memory 仍保留关键机制/结果证据，说明系统开始丢弃可恢复的旧明细而非无限增长。
- 第 6 轮 `stair_clip_hip_knee_acc_high_v3` 从第 4 轮 best anchor 出发，仅给原平方加速度项增加很高的 `clip_max=10000.0`，意图只裁掉极端样本。约 510 iteration 时仍退化为 episode length 约 17、`base_contact=1.0`、terrain level 0、policy noise 约 0.45；说明相对实际输入分布该上限仍是强裁剪，未能保留第 4 轮有效动态。后续 authoring 不应继续围绕该 clip 搜索。
- 第 6 轮完整训练、固定初始策略、预算和 runtime activation 证据均通过。正式 public-dev/public-accept mean max progress 分别约 0.053 m/0.066 m，168/280 个 episode 仍全部以 `torso_contact` 终止，七档成功率为 0、stall rate 为 1.0；它与第 5 轮同量级并远低于第 4 轮，因此以 `required_gate_failed` 拒绝是正确的。
- 第 7 轮 prompt 为 86,745 字节，best-observed 正确保留第 4 轮 `stair_add_slip_penalty_v1`，latest-failed 指向第 6 轮；两份不同 checksum 的源码各内联一次，current-best 与 parent checksum 仍为空。历史在固定上限内增长，没有把未接受源码提升为策略继承来源。
- 第 7 轮候选 `stair_mean_hip_knee_acc_v1` 不再猜测固定裁剪阈值，而是在第 4 轮防滑机制上按髋/膝关节数量归一化加速度惩罚。这是针对“平方和随关节维度放大”的机制修正，比第 5/6 轮的阈值搜索更有归因价值。
- 第 7 轮正式 public-dev/public-accept mean max progress 分别约 0.222 m/0.240 m，较第 5/6 轮有明确恢复，但仍低于第 4 轮约 0.570 m/0.582 m。两套公开评估的 168/280 个 episode 仍全部为 `torso_contact`，七档成功率为 0、stall rate 为 1.0；因此这是部分行为改善而非上楼能力，public score 保持 0 且以 `required_gate_failed` 拒绝，判定没有过强或过弱。
- 第 8 轮首个 provider 响应不是合法 JSON object，在候选物化前以 `json_parse_error` 拒绝。repair feedback 给出明确失败类别，并只允许修复响应或完整奖励源码；第二次响应通过静态验证、Stair smoke 和 anti-degenerate 门，未重复消耗训练预算。
- 第 8 轮候选 `stair_add_base_contact_force_v2` 从第 4 轮 best-observed 源码出发，保留已有 progress、orientation、slip 和动力学约束，新增裁剪到 1000 N 的 base-contact-force 范数惩罚，最大加权贡献为 -1.0。该机制直接针对连续多轮的 `torso_contact` 主失败，而不是继续修改评估或加速度裁剪阈值。
- 第 8 轮正式 public-dev/public-accept mean max progress 分别约 0.586 m/0.598 m，略高于此前 best-observed 第 4 轮约 0.570 m/0.582 m，因而可合理成为新的零分 authoring anchor。但 168/280 个 episode 仍全部为 `torso_contact`，七档成功率为 0，slip/stall diagnostic rate 仍为 1.0；前进改善尚未转化为上楼能力，public score 为 0 且以 `required_gate_failed` 拒绝是正确的。
- 第 9 轮正常 prompt 将同为 best-observed 与 latest-failed 的第 8 轮源码通过一个逻辑 payload 交付，没有重复内联；repair prompt 才同时交付第 8 轮 anchor 和本轮失败源码。该轮 prompt 约 85 KiB，源码角色、内容去重和失败修复边界均符合预期。
- 第 9 轮首稿 `stair_add_height_scan_obstacle_v1` 因在候选入口内定义嵌套函数被静态 verifier 以 `forbidden_nested_function` 拒绝，未启动训练。repair feedback 明确要求只修 verifier 问题、只使用公开 RewardInputs ABI，并禁止修改 runner、评估、种子、预算、checkpoint、课程或 acceptance；第二稿把函数移到模块级后通过 verifier 与 Stair smoke。
- 第 9 轮候选 `stair_add_height_scan_obstacle_v2` 在第 8 轮 anchor 上新增 height-scan 最大值惩罚，试图避免躯干接近障碍。正式 public-dev/public-accept mean max progress 分别约 0.569 m/0.582 m，低于第 8 轮约 0.586 m/0.598 m；168/280 个 episode 仍全部为 `torso_contact`，七档成功率为 0，slip/stall rate 为 1.0。因此该机制没有改善主失败，拒绝合理，第 8 轮应继续作为 best-observed。
- 用户在第 10 轮 `stair_add_ankle_limit_penalty_v1` 训练早期要求暂停。停止前该轮只产出 `model_0.pt` 与 `model_50.pt`，没有 train/eval summary，也没有写入 ledger；因此它不是可比较的正式轮次，后续不得把其局部 checkpoint 或中期曲线用于接受、best-observed 或 replay。
- 暂停时先停止本地只读监视，再只向冷启动主进程及其后代发送中断信号。冷启动主进程、训练进程和 resource tracker 均已退出，按会话 ID 与输出路径复查为 0 个残留进程；两张 GPU 均约 6 MiB、利用率 0。R0 会话未被本次停止操作触碰，并已在此前自然完成 15 条 ledger 记录。
- R0 会话日志末尾出现 `task prompt sections exceed 32768 bytes after sanitization: 33397`，随后仍输出全项通过的 loop audit。该异常不影响已经原子落盘的 15 条 R0 ledger，但说明“完成 15 轮”与“完成后的 prompt/audit 投影”边界可能存在硬错误；恢复任何实验前应从代码与产物确认触发位置并修复，不能通过简单提高上限掩盖历史压缩问题。

### 4.8 与当前开发基线并线审计

- cold-start 会话运行在共享仓库的 detached worktree，HEAD 固定为 `ddc249e`，工作区干净且不存在独立分支或未提交代码；当前开发基线 `c13f5d4` 是 `ddc249e` 的线性后继。因此两者不是两条代码线，也不需要 cherry-pick 或额外 merge commit。
- `ddc249e..c13f5d4` 的源码差异仅涉及 authoring prompt 顺序与版本、完整 experiment-memory 字节边界、provider usage 证据、候选选择指导和 Stair `projected_gravity` 提示语义。训练/评测入口、完整奖励运行时、成功状态机、固定预算、seed、checkpoint、Genesis、acceptance 与 ledger schema 均未改变。
- 运行中的会话必须继续使用冻结的 `ddc249e`，不能在第 9 轮中途切换 prompt/memory ABI；这保证 15 轮内部协议一致。实验结束后的新 cold-start 会话统一从 `c13f5d4` 或其后继主线启动，不延续 detached worktree 作为开发分支。
- 使用 `c13f5d4` 对前 8 条 live ledger 做了只读 acceptance replay，current-best 正确保持 bootstrap、score 为 0，task protocol hash 与 acceptance protocol hash 均与每条记录一致。随后从同一 Genesis 和私有候选源码生成 v3 下一轮 prompt，结果为 86,469 字节；任务段 28,644 字节，完整 memory 为 12,622/14,336 字节，8 条记录均保留且没有隐式 reference/current-best。
- c13 重建将第 8 轮 `stair_add_base_contact_force_v2` 选为零分中的 best-observed，并因它同时是 latest-failed 而只内联一份源码。这与 ddc 会话的 task-owned progress tie-break 一致，同时消除了旧 prompt 在角色别名相同内容时的重复交付。该重建不调用 provider、不训练、不改 ledger，因此只证明向前兼容，不改变冻结实验结论。

## 5. 待持续核对

- 每轮 prompt 总字节、稳定区段 hash、历史条数、实际重复字节和 repair 次数；
- 反馈是否包含七桶能力、互斥失败原因、进度、runtime contribution 和正确的候选身份；
- accepted-only 源码继承、bootstrap 首次接受与后续 paired acceptance/replay；
- 每轮相同 initial-policy hash、固定训练预算、训练/评估设备隔离与 sealed gate；
- 冷启动相对 R0 的探索速度、首个有效能力出现轮次和最终能力边界。

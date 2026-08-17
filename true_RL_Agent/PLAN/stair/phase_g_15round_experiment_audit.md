# Stair Phase G 后 15-round 实验审计

## 1. 目的与硬边界

本文件记录生产启用后的真实 15-round 运行，不把候选数量、训练曲线或 provider 成功调用冒充闭环有效性。最终验收必须同时证明：固定 2000 iterations 和随机初始化公平性不变；每个完成训练的候选均有 public/sealed 七桶证据；prompt 反馈足量但有界；accepted 决策才能推进 current-head；Push 默认路径不退化。

Final V2、评估 seeds/bins、acceptance 参数、TrainingBinding 和完整奖励 ABI 不是效果调参面。真实运行发现实现或证据错误时回到权威源修复；单纯候选效果不足不能放宽门槛。

## 2. 诊断运行 1：`f6c2806`

- 服务器目录：`/data/yuyangliu/true_rl_agent_phase_g/formal_15round_f6c2806_20260803T141333Z`
- session：`stair-r0-formal-aacaee2`
- ledger epoch：`phase-g-15round-f6c2806-20260803`
- 计划轮数：15
- 实际闭合轮数：3；第三条 ledger v5 原子落盘后主动停止
- current-best：官方 R0，public accept mean `0.5464285714`，standard error `0.0174963553`
- 接受阈值：候选至少超过 current-best `0.03`，并通过全部 required gates

### 2.1 逐轮结果

| 轮次 | 源码 checksum 前缀 | public accept | sealed mean | 结论 | 主要观察 |
|---|---|---:|---:|---|---|
| R1 | `58ff2c63` | 0.164286 | 0.094388 | denied | base pitch instability 与 stalled progress 明显 |
| R2 | `3e9011d2` | 0.000000 | 0.000000 | denied | 无界 forward-progress 奖励导致 base contact/stall 崩塌 |
| R3 | `2d21c0c2` | 0.517857 | 0.515306 | denied | 有界进度机制恢复低阶能力，但未超过 R0 + min_delta |

R3 固定预算完整结束：`196,608,000` env-steps、最新 checkpoint iteration `1999`、initial policy state hash 精确匹配。训练末尾 mean reward `24.9182`、mean episode length `600`、base-contact termination `0.00865`。public-accept 七桶均值依次为 `1.0, 1.0, 1.0, 0.6, 0.025, 0.0, 0.0`；主要不足已从 R1 的姿态崩溃转为 0.30 m 以上楼梯能力缺失。

### 2.2 为什么主动停止

R3 训练期间审计发现：

1. shared history 把 eval isolation 的 `candidate_package_loaded=false` 映射成训练未加载，和真实训练 runtime activation 证据冲突；
2. Stair authoring prompt 没有同时提供候选身份、七桶成功率、完整 primary failure breakdown 和 diagnostics，导致 best-observed 源码与 latest feedback 错配。

继续 R4-R15 会让 LLM 在错误公开记忆上决策，因此本次运行只保留为可信的 R1-R3 科学证据，不作为最终 15-round 结果。

## 3. 源头修正与验收门

- shared `public_history.v2` 使用 `candidate_runtime_activated_in_training`，只消费训练 summary、runner activation gate 和训练 runtime evidence；eval isolation 继续独立记录为 `candidate_code_loaded_in_eval`。
- Stair `prompt_feedback.v2` 从 validated `public_dev` 单源导出七桶均值、完整 primary failure breakdown、dominant failure、diagnostics 和 progress。
- Stair `public_feedback_summary.v2` 以 candidate ID/source checksum 绑定反馈，去重保留 reference anchor、最近三条、best nonaccepted 和 accepted current-best，总条数不超过 5。
- 真实 R0-R3 ledger 重建得到 4 条配对反馈，task-owned sections 为 `16958` bytes；R3 history 正确显示训练已激活、eval 未加载候选代码。
- 本地全量 unittest/pytest、Ruff `E9,F,I`、compileall、`git diff --check`、服务器同提交 replay 和 Push 回归全部通过后，才允许启动诊断运行 2。

## 4. 正式重跑要求

诊断运行 2 必须使用：

- 修正后的 clean commit；
- 原不可变 R0 Genesis；
- 新 output root；
- 新 ledger epoch ID；
- 固定 `4096 × 24 × 2000` 训练预算和 `cuda:1`；
- 独立的逐轮 prompt/authoring/训练/eval/decision 审计。

每轮至少检查 prompt 总字节、source payload 去重、反馈条数与身份配对、runtime activation、七桶成功率、primary failure 闭合、sealed gate、decision reason、current-best/source-head replay。任一实现或证据硬错误触发停止、修正、全量重验和新 epoch 重跑。

## 5. 正式重跑：`eb4ede9`

- 服务器目录：`/data/yuyangliu/true_rl_agent_phase_g/formal_15round_eb4ede9_20260804T063045Z`
- session：`stair-r0-formal-aacaee2`
- ledger epoch：`phase-g-15round-eb4ede9-20260804`
- 代码提交：`eb4ede94d01b48f8033e85d7ada257c8d6e6b4fb`
- 训练设备：`cuda:1`
- 计划/完成轮数：`15 / 15`
- 训练完成、完整评估、sealed validation：均为 `15 / 15`
- 接受/拒绝：`2 / 13`
- eval incomplete：`0`

全部候选使用 Genesis/Final V2 单源派生的同一 TrainingBinding、同一随机初始化策略状态、无 warm-start checkpoint，以及 `4096 x 24 x 2000 = 196,608,000` env-steps。完整摘要只保存在机器证据中，不在计划复制；全部训练摘要记录最终 checkpoint iteration `1999`。

评估统一使用 Final V2/Genesis 绑定的同一个 evaluation binding；完整摘要只保存在机器证据中。每轮 public accept 均为 5 seeds x 7 bins x 8 episodes，共 280 episodes；primary outcome 计数逐轮闭合。sealed validation 使用固定 7-seed cohort。评估进程只加载 sealed checkpoint，15 轮均满足 `candidate_code_loaded_in_eval=false`、`candidate_package_loaded_in_eval=false` 和 `candidate_path_argument_present=false`。

### 5.1 逐轮结果

七桶列顺序固定为 `0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 m`。

| 轮次 | candidate | public | sealed | 决策 | 主要失败 | 七桶 public 成功率 |
|---|---|---:|---:|---|---|---|
| R1 | `stair_repair_v1` | 0.253571 | 0.275510 | reject | base_contact 0.696 | .050,.200,.650,.750,.100,.025,.000 |
| R2 | `stair_base_force_v1` | 0.546429 | 0.528061 | reject | base_contact 0.407 | 1,1,1,.775,.050,0,0 |
| R3 | `stair_orientation_progress_fix_v1` | 0.278571 | 0.283163 | reject | base_contact 0.596 | 1,.925,0,.025,0,0,0 |
| R4 | `stair_height_progress_v1` | 0.296429 | 0.339286 | reject | base_contact 0.550 | .600,.700,.575,.200,0,0,0 |
| R5 | `stair_progress_v1` | 0.160714 | 0.150510 | gate reject | base_contact 0.718 | 1,.025,0,.100,0,0,0 |
| R6 | `stair_contact_stability_v1` | 0.000000 | 0.000000 | gate reject | fall 0.807 | 0,0,0,0,0,0,0 |
| R7 | `stair_stability_tune_v1` | 0.346429 | 0.329082 | reject | base_contact 0.464 | .275,.325,.750,.900,.150,.025,0 |
| R8 | `stair_progress_contact_v1` | 0.000000 | 0.000000 | gate reject | base_contact 1.000 | 0,0,0,0,0,0,0 |
| R9 | `stair_stable_progress_v2` | 0.589286 | 0.576531 | **accept** | base_contact 0.332 | 1,1,.975,.900,.250,0,0 |
| R10 | `stair_climb_height_contact_v1` | 0.300000 | 0.326531 | reject | base_contact 0.546 | .725,0,0,.875,.500,0,0 |
| R11 | `stair_safe_clearance_v1` | 0.435714 | 0.471939 | reject | base_contact 0.529 | .675,1,.875,.500,0,0,0 |
| R12 | `stair_clearance_progress_v1` | 0.000000 | 0.002551 | gate reject | base_contact 0.843 | 0,0,0,0,0,0,0 |
| R13 | `stair_stable_clearance_v2` | 0.000000 | 0.000000 | gate reject | base_contact 1.000 | 0,0,0,0,0,0,0 |
| R14 | `stair_stable_progress_contact_v3` | 0.621429 | 0.602041 | **accept** | base_contact 0.250 | .775,.900,1,.975,.675,.025,0 |
| R15 | `stair_balanced_footing_v1` | 0.396429 | 0.387755 | reject | base_contact 0.507 | .450,.300,.625,1,.400,0,0 |

### 5.2 accepted 源码推进

源码头只在 accepted 后推进，逐轮 parent 已按字节哈希验证：

1. R1-R9 的 parent 为官方 R0 `sha256:a1255b...c03a3`；
2. R9 以 `public=0.589286`、相对 R0 `delta=0.042857 > 0.03` 且全部 required gates 通过而接受；
3. R10-R14 的 parent 为 R9 `sha256:bdda86...2fade`；
4. R14 以 `public=0.621429`、相对 R9 `delta=0.032143 > 0.03` 且全部 required gates 通过而接受；
5. R15 的 parent 为 R14 `sha256:5d00ed...33682`，R15 拒绝后最终 head 不变。

最终 current-best 为 `stair_stable_progress_contact_v3`：public `0.621429 +/- 0.011845 SE`，sealed `0.602041 +/- 0.019372 SE`。相对 R0，public 宏平均绝对提升 `0.075`，base-contact 失败率从 `0.378571` 降到 `0.25`，0.30 m 桶从 `0.175` 提升到 `0.675`，0.35 m 桶首次达到 `0.025`。0.40 m 仍为 0，且 0.10/0.15 m 桶有一定回退，因此不能宣称楼梯任务已经完全解决。

### 5.3 authoring 与反馈闭环

- 15 个 round 共形成 24 份不可变 authoring-attempt prompt；最大 `101,114` bytes，低于 `256 KiB` 总上限。
- 24 份 prompt 全部通过公共性 validator；current-best 内联源码哈希、lineage parent 和 ledger replay current-head 全部一致。
- R9 authoring 中，重复 R2 接触力机制被 `historical_mechanism_repeat` 拒绝，smoke 失败稿也未进入训练，第三稿才训练并接受。
- R13 错把正接触力配置为 `clip_min=-1`，导致最大加权贡献 `21,444.84`、100% base contact 和零分。R14 prompt 消费该失败后改为 `clip_max=2.0`、weight `-0.2`，runtime 最大贡献被限制为 `0.40000001`，并获得第二次接受。
- R15 前两稿分别被 smoke 和历史机制重复门拒绝，第三稿才进入训练；其新增双脚平衡项实际最大贡献仅 `0.027753`，最终未达到接受阈值。

这证明循环不是“每轮都训练 provider 第一份输出”，也不是只做权重搜索：静态验证、task-owned smoke、历史机制门、真实训练贡献、公开失败反馈和 accepted-only lineage 都实际参与了决策。

## 6. 严格复放与逐字节验收

主循环自然退出后，使用同一生产入口、同一 Genesis/epoch 和 `--rounds 15` 进行 load-only replay。结果为 `resumed_from_round_count=15`、`rounds_executed_this_invocation=0`，10 项 live summary 检查和 12 项 eval-run audit 检查全部通过；未调用 provider，未启动训练或评估进程。

额外逐字节审计的 19 项检查全部通过：

- 15 个 sealed candidate artifact 重新加载并匹配 artifact/source/manifest/verification identity；
- 15 个 checkpoint 重新计算字节 SHA-256，并与 train result、eval job 和 safety report 一致；
- candidate、verifier、train/eval spec、train result、public result 和 sealed evidence 的源码/subject 身份唯一；
- 所有训练保持同一 binding、随机初始化、预算和 iteration；
- 所有 runtime input fingerprint 均为 detached、storage-unique；
- 24 份 prompt 的父代与内联 current-best 源码逐字节匹配；
- 15 轮七桶 episode 计数和 primary outcome 闭合；
- accepted 仅为 R9/R14，最终 replay head 精确恢复为 R14。

最终代码/文档回归：本地标准 unittest `850 tests, OK, skipped=16`；pytest `834 passed, 16 skipped, 595 subtests passed`；服务器正式仓库 unittest `850 tests, OK, skipped=2`。Ruff `E9,F,I`、compileall 和 `git diff --check` 全部通过。

运行结束后，Stair 主进程、train/eval 子进程和 GPU1 任务均已退出；只保留服务器原有 PID `748733`，未终止或修改。失效的外部 `live.pid` 标记已在确认 PID 不存在后删除，ledger、summary、日志、candidate、checkpoint 和 lease 证据全部保留。

## 7. 结论

正式 15-round 实验闭合且产生了可信的两级优化：`R0 0.546429 -> R9 0.589286 -> R14 0.621429`。本轮没有发现需要修改 Phase F/G 共享合同、Final V2、固定 2000 iterations 预算或完整奖励 ABI 的硬错误；观察到的问题均为候选机制质量，并被现有 smoke、历史门、固定评估和 accepted-only lineage 正确隔离。

当前系统已经证明真实奖励源码自进化闭环可运行并可重放，但任务仍有明确能力边界：0.40 m 未突破，0.35 m 仅极低成功率，最终奖励在低阶桶存在权衡。后续应继续在新 epoch 中从 R14 源码探索新的完整奖励机制，不能放宽评估门或把 R14 checkpoint 用作下一轮 warm start。

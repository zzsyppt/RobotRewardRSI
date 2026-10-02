# N4 四例文本作者开发读法检查

日期：2026-09-29。**非盲、开发评阅、不是 N5 正式评委，不评分，不作 A/T 或物理验收。** 本评阅者参与了准备与执行。四例全部按同一冻结五束问题检查；只读实际末轮 prompt 与生成源码，并做 AST/标量代数核对。没有新增 API、训练或 GPU 调用，没有修改候选、冻结输入或生产代码。

结论：四例确有使用数值反馈；两例 Push 正确认出了速度违规出现率不具区分力。最明确的矛盾是 stair-01 将 L3 cell 进度写为整个 joint_ood 的均值，以及 push-02 docstring 把旧斜率保持范围写得过宽。其余问题主要是对照/范围未带全、由全局累计量推区分力、假设或未来测量口径不够明确。没有发现代码读取提示白名单外字段，也没有发现声称已测得新训练收益。

冻结问题是评阅清单，不是附加作者答案格式；未引用某句不自动计错。四例只接收文本，缺视觉不归咎作者；真实 S4 父子差分、冲突句与视觉读法均未在本批验证。假设未验证，不等于已经声称因果成立。

证据定位：prompt_payload.json 均是实际保存的单行 JSON，下文用 **第 1 行 + JSON pointer + sentence id**，不制造排版后的行号。完整精简结构记录见 [reading_review.json](reading_review.json)。源码行号均指候选 training_code/components.py。

## stair-01

源码：[stair-01 components.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/provider_candidates/p3new-text-smoke-20260929-generation-001/lane-a/001_go2-stair-goal-progress-shaping-v1/training_code/components.py)；实际提示：[prompt_payload.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/authoring/p3new-text-smoke-20260929-generation-001/lane-a/authoring_attempts/attempt_000001/prompt_payload.json)。

| 冻结问题 | 检查结果 |
|---|---|
| Q1 | 部分使用旧数值表，区分 score 与 failure rate；但把 joint_ood L3 的进度写成整个 joint_ood 的均值。未复述 episode 分母。 |
| Q2 | 未引用新报告过程频数或其区分力；弱 OOD 表现不能直接证明命令速度误差。 |
| Q3 | 未使用 compatible-success 或同 cell/run 成功对照；这是未使用，不自动计错。 |
| Q4 | 未定位 body/step/run 过程句；停在高台阶前的叙事属于机制假设。 |
| Q5 | 明确预测未实测，正确说明进度即时/可下降；未讨论数值采样范围。纯文本无视觉不计错误。 |

- **S01-scope · contradicted**（源码 94–97）：`joint_ood 0.7142 with mean progress 0.672`。0.672 对应 joint_ood L3 cell，整个 joint_ood 的 mean_progress_fraction 是 0.7895609935（约 0.790）。
  提示第 1 行：`/public_feedback_summary/feedback_entries/0/feedback/generalization_summary/stratum_diagnostics/rows/5`；`/public_feedback_summary/feedback_entries/0/feedback/progress_summary/mean_progress_fraction_by_cell/public_dev.joint_ood_level_3`。
- **S01-mechanism · unsupported_as_guarantee**（源码 90–103）：`so mid-climb quiescence is no longer a competitive outcome while the top of the staircase is.`。提示没有实测 commanded-versus-realized velocity error；现有观察不能确认障碍接触或旧回报让停滞胜出。该句置于 hypothesis 段，不能据此算已声称训练因果事实；但静态代码也不能保证消除停滞竞争力。
  提示第 1 行：`/public_feedback_summary/behavior/profile/report/process/1`；`/public_feedback_summary/behavior/profile/report/visual/0`；`/reward_inputs_contract/field_semantics_constraints/goal_progress_fraction`。
  代数核对：p=0.5 且停滞时，新项仍给 2p=1 的逐步 reward rate；这不能保证停滞不再有竞争力。
- **S01-bound · ambiguous_requires_narrow_wording**（源码 105–110）：`strictly below the 3.5 ceiling of the unchanged tracking mechanism, so the parent's dominant weighting is preserved`。上界 2<3.5 与保留原权重都正确；若 dominant weighting 仅指最大系数，则不构成硬错误。它不能推出逐状态贡献或行为支配关系，建议明确只保留上界陈述。
  提示第 1 行：`/source_payloads/0/content_sha256`。
  反例仅针对“逐状态支配”的宽解释：速度误差 1 m/s 时 tracking=3.5 exp(-4)=0.0641；p=0.5 时新项=1。共同 control_dt 不改变此比较。最大权重系数仍是 3.5 这一窄陈述没有错误。

## stair-03

源码：[stair-03 components.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/provider_candidates/p3new-text-smoke-20260929-generation-001/lane-b/001_stair_goal_progress_potential_v1/training_code/components.py)；实际提示：[prompt_payload.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/authoring/p3new-text-smoke-20260929-generation-001/lane-b/authoring_attempts/attempt_000001/prompt_payload.json)。

| 冻结问题 | 检查结果 |
|---|---|
| Q1 | 正确引用 score/success/progress 分解及四个 cell 预测基线；未复述 episode 分母。 |
| Q2 | 使用 contrast-0 的 9/9 与 contrast-2 的 5/6；省略 run 与成功对照，不能把 9/9 当区分失败的证据。 |
| Q3 | 未引用同 cell/run 的成功 2/2、3/4，或具体 compatible-success。 |
| Q4 | 引用 root 组频数，未引用逐步过程；高台阶前停滞明确写为 hypothesis，未作为已观测接触事件。 |
| Q5 | 明确未训练、即时非 latch；未使用数值采样/残差缺口。 |

- **S03-process · supported_counts_incomplete_scope**（源码 85–92）：`public process notes report low root forward progress in 9/9 c1_conditional_frontier L5 forbidden_contact failures and 5/6 joint_ood L3 failures.`。两组频数准确。9/9 限 run 1，成功也是 2/2，报告明确不具较高失败出现率；5/6 限 run 2，成功 3/4。遗漏不能自动变成错误因果结论，但证据选择不足以支持“失败专有的停滞”。
  提示第 1 行：`/public_feedback_summary/behavior/profile/report/process/2`；`/public_feedback_summary/behavior/profile/report/process/4`；`/public_feedback_summary/behavior/profile/report/counterexample/0`。
- **S03-code · supported**（源码 93–106）：`the eight reference terms keep their exact functions, weights and clips`。AST 核对八个旧函数和八项 RewardTerm 定义未变；只新增 weight=1、clip=[0,1] 的即时 goal_progress_fraction。命名 potential 不等于声称 potential-based shaping 保策略不变；不据名称计错。
  提示第 1 行：`/reward_inputs_contract/field_semantics_constraints/goal_progress_fraction`。
  新项 p=0.5 时给 0.5 reward rate，上界 1 正确；没有从 potential 这个名称推定作者声称了策略不变性。

## push-02

源码：[push-02 components.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/provider_candidates/p3new-text-smoke-20260929-generation-002/lane-a/001_push_recovery_admissible_velocity_band_v1/training_code/components.py)；实际提示：[prompt_payload.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/authoring/p3new-text-smoke-20260929-generation-002/lane-a/authoring_attempts/attempt_000001/prompt_payload.json)。

| 冻结问题 | 检查结果 |
|---|---|
| Q1 | medium 30/48 与总 fall_forward=24、fall_side=13 引用正确；193 是中断事件计数，240 是 recovery units，不是同一单位的 incidence 分子分母。 |
| Q2 | 正确识别 planar/angular 出现率无区分力；但全局 193/54 不能证明 sustained excess 区分失败与成功。 |
| Q3 | 引用 high/repeated 成功组计数；未引用成功样例回稳后仍有 97 个不稳定步。 |
| Q4 | 采用整段 recovery-window 组频数，未逐步引用 anchor/latch；没有把终帧写成 strict-before。 |
| Q5 | 区分假设、预测和任务判定；笼统称 per-event timing offsets 不可见过宽，部分样例时序已在输入。 |

- **P02-counts · supported**（源码 60）：`the medium stratum reaches 30/48 success_recovered`。medium 30/48；high/repeated 的失败与成功组 roll 对照也准确。planar/angular 出现率无区分力的读法正确。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/report/results/4`；`/public_feedback_summary/public_dev/behavior/report/process/1`；`/public_feedback_summary/public_dev/behavior/report/process/2`。
- **P02-discrimination · unsupported**（源码 60）：`Sustained excess does: 193/240 dwell interruptions, 54/240 recovery units with no stable step at all`。193 为全局中断次数，54 为零稳定步的 recovery units，分母 240 为全部 recovery units；没有按失败/成功给出持续超限分布。roll 出现率对照是真实对照，但不是持续时间对照。不要将 193/240 解释成 193 个独立坏窗口或每 episode 阈值。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/comparison_basis/counts`；`/public_feedback_summary/public_dev/behavior/report/process/1`；`/public_feedback_summary/public_dev/behavior/report/process/2`。
- **P02-algebra · contradicted**（源码 8–10）：`the tail beyond the bound keeps its original slope.`。新惩罚 H(max(z-0.5,0)) 仅在 z>1.5 恢复旧线性斜率，docstring 的 beyond the bound（z>1）范围过宽。rationale 第 62 行明确约 1.5 倍则正确。整个稳定区间 z<=1 也并非全免费，只是 z<=0.5 免费。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/report/results/1`。
  令旧成本 H(z)=z²/2（z≤1），否则 z−1/2；新成本 H(max(z−0.5,0))。z=1.25 时旧/新成本导数分别 1 与 0.75；z=1 时新成本 0.125（单轴 reward=0.96875），并非整个容许区间免罚。
- **P02-missingness · ambiguous_overbroad**（源码 61–64）：`Limits: per-event timing offsets, per-stratum denominators beyond the published aggregates`。全部事件的 timing offsets 确实没有完整展开，但示例已有 anchor 69/window 70-179/response 70，成功示例有 anchor 69/234 与 latch 128/270。应限定为未显示事件的明细缺失。yaw-rate 是假设措辞；所用角速是三轴范数，不能把当前组差异归到 yaw，报告明确 yaw 观察不可用。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/report/process/0`；`/public_feedback_summary/public_dev/behavior/report/counterexample/0`；`/public_feedback_summary/public_dev/behavior/report/gaps/5`。

## push-04

源码：[push-04 components.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/provider_candidates/p3new-text-smoke-20260929-generation-002/lane-b/001_push_recovery_stability_envelope_v4/training_code/components.py)；实际提示：[prompt_payload.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1/authoring/p3new-text-smoke-20260929-generation-002/lane-b/authoring_attempts/attempt_000001/prompt_payload.json)。

| 冻结问题 | 检查结果 |
|---|---|
| Q1 | 正确使用重复组 13/48、全局 zero_stable_units 54/240 与 193 次中断；未把 score 当成功计数。 |
| Q2 | 正确引用 high/repeated roll 对照与 planar/angular 非区分性；没有把同现直接写为已验证因果。 |
| Q3 | 保留对应成功组分母 0/20、1/13 与 20/20、13/13；未用具体成功样例。 |
| Q4 | 引用的是 whole observed recovery windows；未来 refutation 却写 post-recovery，需明确新测量窗口及基线。 |
| Q5 | 正确声明 reward 没有 dwell age/planned timing/latch，瞬时项不直接奖励连续时长；明确未实测收益。 |

- **P04-process · supported**（源码 77）：`absolute-roll violations appear in failed high and repeated recovery windows (4/13 and 4/14) but in only 0/20 and 1/13 of comparable successful windows`。对应 high 1 executed event 与 repeated 2 executed events 的 whole observed recovery windows；数字与非区分性描述准确，未发明视觉/足接触观察。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/report/process/1`；`/public_feedback_summary/public_dev/behavior/report/process/2`；`/public_feedback_summary/public_dev/behavior/comparison_basis/counts`。
- **P04-anchor · ambiguous_self_qualified**（源码 75–77）：`The anchor penalizes only two of the five published recovery bounds`。严格按整个 anchor 理解不准确：旧 upright_height 已包含 roll、pitch、height 惩罚。后文第 77 行明确承认这三轴已有饱和指数信号，因此把主语改为 old stationarity callable 即可；不据此声称代码偷偷改了保留项。
  提示第 1 行：`/source_payloads/0/content_sha256`。
- **P04-window · underspecified_not_observed_misreport**（源码 81）：`refuted if repeated-stratum success stays at 13/48 while post-recovery absolute roll and pitch violation occurrences are unchanged.`。当前组频数是 whole recovery windows，不是 post-recovery。未来可以新测 post-recovery，但须定义 latch 后哪一段、提供同窗基线；不能直接用 4/14 与 1/13 作为它的现成基线。此处没有声称已观测的 post-recovery 数值，故不记作既成时序误报。
  提示第 1 行：`/public_feedback_summary/public_dev/behavior/report/process/1`；`/public_feedback_summary/public_dev/behavior/report/process/2`；`/public_feedback_summary/public_dev/behavior/report/counterexample/0`。

## 代码一致性与边界

Stair 两例：八个父函数与八项 RewardTerm 定义保持一致，分别仅新增 weight=2 与 weight=1 的即时空间进度项。Push-02：保留四个其他父函数，只改 stationary_v3 函数；Push-04：保留四个其他父函数，以五轴 envelope 替换 stationarity 项。上述是静态源码事实，不是训练效果、连续驻留行为或物理因果验证。

Q1=分母与结果；Q2=常见/不常见及区分力；Q3=兼容成功对照；Q4=过程的身体、窗口与时间关系；Q5=观测/采样缺口和假设边界。没有为未使用的 Q 项补零分。所有新措辞仅是建议，本轮未修补作者输出或生产提示。

完整性：file_manifest.json 中 62 个原生工件逐文件 SHA-256 与长度均一致（另有 manifest 本身）；四份实际 prompt 与候选源码 SHA-256 逐例记录在 JSON。新增的两个评阅文件不回写原生执行 manifest。

最小建议：修正 stair-01 的 cell/stratum 标签；让 stair-03 的局部频数带上 run 与成功对照；让 push-02 分清累计量、组间区分、半阈值免罚和 1.5 倍阈值后的斜率；让 push-04 限定主语为旧 stationarity callable，并定义未来 post-recovery 测量窗口。这些建议不构成新实验准入或参数调整。

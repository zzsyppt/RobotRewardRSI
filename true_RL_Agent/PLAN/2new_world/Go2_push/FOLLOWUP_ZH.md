## F1：Push 多轮记忆未建立 trial 的公共结果投影（已修复）

问题：共享多轮记忆从 trial 的 `prompt_outcome` 读取结果，但 Push 任务此前既未从已测公共证据构造该字段，也未保留已投影结果，导致 trial 表三列均为 `null`。其中候选相对父策略的 public-dev 差值有同卷实测依据；Push 没有独立公开 guard 分数，也没有组件级因果效果判定，后二者不能为了补满表格而从 accept 借值或推断。

证据：修复前 `tasks/push_recovery/prompt_adapter.py` 的 `_v2_prompt_record()` 没有构造 `prompt_outcome`；`agent_core/experiment_memory.py::_trial_outcome()` 读取该字段。先前合成探针证明已有字段也会被丢弃；进一步沿真实 generation 链确认，Push 本就没有该投影生产者。真实公共历史已携带 task-owned `guard_reference.score_stats` 或 `parent_public_dev.score_stats`，可与候选的 `public_dev.public_summary.score_stats` 比较，且不需要访问 `parent_current_best.public_score`。

修复证据：`tasks/push_recovery/exploration.py` 单源产生同卷 public-dev 描述性差值，任务提示与探索证据复用；`prompt_adapter.py` 在重复投影中保留该来源并构造 trial 结果，guard 为不适用，runtime effect 明确为 `not_assessed`，对应语义写入任务评测合同。`test_push_feedback_projection.py::test_trial_memory_uses_measured_parent_dev_and_is_idempotent` 经完整记忆构造核对差值 `-0.25`、禁止 accept 替代及幂等性；相关 114 项通过，见 `push_feedback_fix_test_receipt.json`。

## F2：Push 任务记忆摘要读不出投影后的失败分类（已修复）

问题：Push 的任务记忆摘要接收到聚合后的 public-dev 记录，却仍使用要求 `seed_results` 的旧读取函数，使有明确失败记录的历史被摘要为 `latest_failure_mode="unknown"`，`latest_failure_breakdown` 与 `aggregate_failure_breakdown` 均为空。近期任务反馈的聚合计数通过另一条路径保留。

证据：`src/true_rl_agent/memory/context_builder.py:282–305,414–416` 先做任务历史投影，再将投影结果交给任务记忆；`src/true_rl_agent/tasks/push_recovery/prompt_adapter.py:533–567,583–585` 产出不含 `seed_results` 的聚合 public-dev；`src/true_rl_agent/tasks/push_recovery/memory.py:75–80` 调用失败分类提取函数，而 `src/true_rl_agent/tasks/push_recovery/metrics.py:973–977` 在缺少 `seed_results` 时返回空。2026-09-22 生产函数探针使用 4 个 episode（成功 2、侧倒 2）：原始失败分类为 `{"fall_side":0.5}`，投影中仍有 `{"success_recovered":2,"fall_side":2}`；`PushTaskMemoryAdapter.build_public_summary()` 输出 `latest_failure_mode="unknown"`、两份失败分类均为 `{}`。

修复证据：任务 `metrics.py` 的失败分类读取函数直接支持其自身聚合格式，提示投影保留 episode 总数，继续使用同一个失败率解析器，不新增记忆侧算术。`test_task_memory_retains_failure_rates_after_aggregate_projection` 核对投影前后失败率和分类相等；A01 原始证据回放的主失败为 `height_collapse_dominated`，高度坍塌率为 `26/192`，见 `push_feedback_replay_audit.json`。

补充修复证据：`latest_failure_mode` 原来误用包含成功类别的 `dominant_outcome`，多数成功时会写成 `success_recovered`。现在候选与 reference 反馈均复用同一失败分类器；`test_failure_mode_reports_failure_even_when_successes_are_the_majority` 核对 2 个成功、1 个侧倒时主结果仍是成功，主要失败明确为摔倒，两类字段不再混同。最终受影响的三个模块共 74 项通过，见 `push_feedback_fix_test_receipt.json`。

## F3：reference 最终资格失败时未保存详细资格判定（已修复）

问题：reference 完成三阶段评测后，如果最终任务资格不通过，共享 Genesis 入口只抛出通用的 capability gate 错误；任务已经计算出的具体 gate 状态和资格证据没有持久化。各阶段原始回执仍保留，但要解释这次拒绝，需要另外重建资格判定，不能直接读取失败资格回执。

证据：`src/true_rl_agent/tasks/push_recovery/reference_program.py:147–175,192–196` 将 `gate_statuses`、accept/dev 分数等放入返回的 `private_evidence`；`src/true_rl_agent/agent_core/session_genesis.py:2290–2302` 在取得资格对象后遇到失败立即抛出 `reference program failed the task-owned capability gate`，包含该对象的私有评测文件直到 `2323–2337` 才写入。`src/true_rl_agent/agent_core/generation_application.py:213–222` 的 Genesis 异常路径只结算成本并重新抛出异常。此项为 2026-09-22 静态路径审计，尚非本轮已发生的运行失败。

修复证据：共享 Genesis 在资格失败时保存任务返回的 `qualification_gate`、`qualification_evidence` 及对应 session、manifest、checkpoint、evaluation binding 至私有 `reference_qualification_failure.json`，随后仍拒绝创建 Genesis。成功路径继续使用原有 eval artifact。`tests/unit/test_reference_genesis.py::ReferenceGenesisContractTests::test_failed_capability_gate_creates_no_genesis_or_ledger` 核对详细失败、证据 hash、协议与不创建 Genesis/ledger；本模块 24 项通过，见 `reference_qualification_failure_tests.log`。

## F4：初代 Push 提示词未提供已测 reference 的失败与分层证据（已修复）

问题：初代提示词要求依据 reference 源码和公开失败证据提出改进，但只提供源码和 dev 总均分，没有提供已存在的 reference 失败分类与分层成功计数。候选无法从初代上下文得知 reference 的主要失败集中在哪一类扰动及结果。

证据：`attempt_01/session/authoring/go2-push-5rounds-20260922-a01-generation-001/lane-a/authoring_attempts/attempt_000001/prompt_payload.json` 的 `public_history.best_public_score` 为 `0.734375`，reference 源码以内联形式提供；`public_feedback_summary.status` 为 `no_prior_public_feedback`，`runtime_feedback_summary.status` 为 `no_prior_runtime_feedback`，全文没有 `stratum_results`、`height_collapse`、`fall_side` 或 `fall_backward`。同目录会话的 `reference_eval_artifact.json` 中 `public_result.metrics.public_dev` 已保存 192 个 episode 的公开结果：低强度 `48/48`、中强度 `47/48`、高强度 `19/48`、重复推搡 `27/48`；失败为高度坍塌 26、侧倒 13、后倒 11、未恢复 1。lane-b 的初始提示词同样缺少上述 reference 反馈。

修复证据：Push adapter 使用共享层已有的 `initial_anchor_public_evidence`，在没有已测候选反馈时仅投影 reference 的 public-dev；既不修改共享接口，也不投影 accept/sealed 或私有身份。`test_initial_reference_feedback_uses_only_dev_for_both_robots` 覆盖 Go2/G1、完整/裁剪反馈和隐藏值隔离；`test_evaluated_candidate_feedback_takes_precedence_over_reference` 核对候选反馈优先。A01 原始分层和失败计数回放已完整呈现，见 `push_feedback_replay_audit.json`。

## F5：单机制重写的修复帧裁掉比较基准，触发额外源码请求（已修复）

问题：候选因同时修改多个机制而被拒绝时，下一次修复提示要求保留基准的其余机制，却裁掉基准源码，只保留失败候选源码。实际 provider 为完成这项比较重新请求同一份基准，增加一次请求往返并消耗源码请求配额。

证据：`attempt_01/session/authoring/go2-push-5rounds-20260922-a01-generation-001/lane-b/authoring_attempts/attempt_000003/prompt_payload.json` 的反馈为 `authoring_policy_violation`、`changed_mechanism_count=5`；帧回执以 `omitted_nonfailed_sources_for_static_repair` 裁掉 `current_best/training_code/components.py`，仅内联失败候选。该次 `attempt_result.json` 记录 provider 请求上述 logical ID，其 rationale 明确指出需要 exact reference source 才能遵守单机制修改和保留其他机制的要求；源码服务返回 `2473` 字节的封装载荷，`source_request_count=1`，本轮上限为 `2`。后续 `attempt_000005/attempt_result.json` 才完成 `changed_mechanism_count=1` 的通过判定。


修复证据：`prompting/authoring_frame.py` 对单机制策略重写（含未解决的嵌套修复）使用现有 `select_authoring_anchor_source`，保留 verifier 比较所需的同一基准源码；普通语法修复仍只保留失败源码，历史等无关内容仍裁掉。真实 A01 第 3 次提示词投影重放保留失败源码与 1,846 B reference，见 `repair_frame_replay_audit.json`。prompt projection、authoring loop、Genesis 三模块共 98 项通过，见 `repair_frame_and_failure_tests.log`。

## F6：输入来源回执混合首步与当前步状态（已修复）

问题：Push runtime 只在首步保存整份 `input_provenance`，后续仅更新 push 两字段。上一动作在首步用零初始化、随后使用真实历史，但 A01 训练 12,000 步后的 summary 仍标记 `prev_action="zero_fallback"`；同时 `action_delta="observed"` 没表达首步上一动作的零初始化来源。该问题影响回执解释，未改变奖励数值。

证据：A01 `session/runner/go2-push-5rounds-20260922-a01/train/summary.json` 的 candidate runtime provenance 与 `src/true_rl_agent/tasks/push_recovery/runtime_adapter.py` 的首次保存逻辑不一致。修复后按每次输入构造整体刷新来源，`action_delta` 同时考虑当前与上一动作是否可观测；明确该字段描述最新输入而非全程覆盖。`tests/unit/test_push_recovery_runtime_adapter.py` 新增两步真实 adapter 检查，第一步两字段为零初始化、第二步均为 observed，并核对数值算术；该模块 16 项通过。

## F7：训练出现奖励极值与价值损失尖峰，但缺少同刻分项证据（新运行的定位能力已修复；旧峰值不可追溯）

问题：reference 与部分候选训练出现十亿及以上量级的单步奖励率绝对值和价值损失尖峰。修复前回执只保存全程奖励率最大值与每 24 步抽样的分项最大值，没有峰值所在步、环境及同刻分项，无法确定该全局峰值来自哪次动作/状态及哪一项奖励。这些旧证据不能将峰值归因于 lane-b 新机制，也不能据此认定独立评测无效。

证据：A02 `session/runner/` 下 reference、`go2-push-5rounds-20260922-a02-generation-001-lane-a`、`go2-push-5rounds-20260922-a02-generation-001-lane-b` 的 `train/summary.json` 分别记录 `runtime_activation_summary.max_abs_candidate_reward=1049201664`、`6775.25927734375`、`1044671936`。A01 reference 也为 `1049201664`，不是 A02 的 lane-b 独有现象。三者均完成 12,000 个 control steps；末步奖励率均值分别为 `3.6579599380493164`、`3.635664939880371`、`3.706913948059082`。

证据：修复前 `src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:366–370` 每步记录所有环境加权奖励率之和的绝对最大值；该值尚未经过 reward manager 的 `dt=0.02` 积分，不是任务成功率或 episode 总分。`:337–342` 的分项记录每 24 步才采样；reference 与 lane-b 的抽样 `action_rate.max_abs_weighted_contribution` 分别为 `4897676`、`2425.3955078125`，其他抽样分项都小于等于 `20`。lane-b 冻结 `candidate_artifacts/.../push_recovery_posture_gated_damping_v1/.../training_code/components.py` 中新增替换的 `stable_damping` 权重为 `1` 且裁剪到 `[0,1]`，不能直接贡献十亿量级；未改动的 `action_rate` 为 `action_delta.square().mean(dim=1)`、权重 `-0.01`。抽样分项极值与全程总极值不在相同采样范围，不能直接相减完成因果定位。

证据：完整训练日志 `attempt_02/session/job_attempts/0920bb4d522c8ac71604d06c/attempt_00.logs/stdout.log` 的 reference `Mean value loss` 在 iteration `358` 达到 `39093918208`，iteration `499` 为 `0.0060`；`a484cf3bf8e45099df9c97ca/attempt_00.logs/stdout.log` 的 lane-b 在 iteration `187` 达到 `84979772416`，末次为 `0.0112`；`dfbb079628143135325df2a6/attempt_00.logs/stdout.log` 的 lane-a 最大为 `2.8636`、末次为 `0.0043`。日志均覆盖 iteration `0–499`，未发现 NaN/overflow/FloatingPointError 报告；这些日志仍没有与全程奖励率峰值绑定的逐环境输入。

证据：`reward_output_stats.finite_fraction=1` 来自 `runtime_adapter.py:1148–1152` 的末步分项摘要，不是奖励幅度或整段优化稳定性的度量；每次分项输出另在 `guard_vector` 检查形状和有限性。A02 两候选的 `eval/public_dev/summary.json` 记录 lane-a `156/192`、lane-b `142/192`，`invalid_injection_count` 均为 `0`，各 seed 的 `observation_finite=true`、checkpoint safety 为 pass、`candidate_code_loaded_in_eval=false`。`tasks/push_recovery/runtime.py:158–159` 在评测模式返回零训练奖励，成功判定使用独立任务状态。

证据：A03 第 1 代 lane-b 的 `attempt_03/session/runner/go2-push-5rounds-20260922-a03-generation-001-lane-b/train/summary.json` 记录 `max_abs_candidate_reward=72720842752`、`step_calls=12000`、`last_candidate_reward_mean=3.337674617767334`。同文件每 24 步采样的 `bounded_damping.max_abs_weighted_contribution=0.9996244311332703`，`action_rate.max_abs_weighted_contribution=281555488`。这仍是全步总奖励率峰值与周期采样分项峰值，二者没有同刻绑定，不能由此确定总峰值的来源；字段复核见 `attempt_03/prompt_audit_gen02.json` 的 `F7_measured_training_extremes`。

修复证据：Push `CandidateRuntimeAdapter` 在既有全程绝对最大值严格增大时，保存唯一的 `max_abs_candidate_reward_frame`：control step（从 1 开始）、环境下标（从 0 开始）、有符号总值，以及同一步同一环境各项在裁剪与有效权重之后的有符号贡献。该字段明确当前物理程序的单位为 control-dt 积分前奖励率，分项按运行 dtype 与原顺序求和对应总值；它随原训练 summary 落盘，没有新增 prompt 投影。没有修改奖励函数、裁剪、求和、原最大值统计或训练/评测协议，也没有保存逐步轨迹。只在新峰值时额外提取同环境标量，较小及相等峰值不覆盖原快照。

修复证据：`test_reward_peak_frame_uses_same_env_and_only_replaces_on_larger_peak` 使用列表和真实 Torch Tensor 两条生产 adapter 路径，核对第 1 步环境 1 的 `4 + (-15) = -11`、较小/相等峰值不覆盖、第 4 步环境 0 的裁剪后 `6 + (-21) = -15` 替换，并核对原奖励返回值。四步均早于原 24 步分项采样点；结果可 JSON 序列化。对应测试模块共 17 项通过（2.156 秒，exit 0），见 `reward_peak_frame_tests.log`；Ruff `F,E9` 与 diff 检查通过。此证据验证新运行的定位能力；A01–A03 未记录的旧峰值不能补算为已定位，也尚未证明策略为何产生巨大动作或状态。

修复实测证据：A06 第 1 代两次完整训练已实际记录同刻峰值，见 `attempt_06/gen01_reward_peak_audit.json`。A 在 control step 11877、环境 1926 的总奖励率为 `-147.9560546875`，其中 action_rate 为 `-125.9666976928711`、fall 为 `-20`；B 在 step 5140、环境 1196 为 `-25.158161163330078`。两份记录按 runtime float32 原顺序累加分项均精确等于总值，绝对值等于全程最大值。本次未复现旧批次十亿级峰值；这不补足旧轨迹，也不证明未来训练不会出现尖峰。

实测证据：A06 第 5 代 B 的峰值帧总奖励率为 `-2304276.5`，同刻 `action_rate=-2304250`，峰值主要来自该项，见 `attempt_06/trained_candidate_mechanism_audit.json`。新快照已解决“无法定位同刻分项”的记录缺口；这些数据仍不足以解释巨大动作差的成因。第 1 代未出现大峰值不代表整个 A06 未出现。

## F8：已测但未接受的 selected 候选被 lane 顺序挤出源码与详细公开反馈（已修复，完整真实链重放通过）

问题：A02 第 1 代 lane-a 的公开 dev 成绩高于 lane-b，之后在 accept 比较中正常拒绝。下一轮重建提示词保留了 A 的机制与 dev 差值，但源码检索仅覆盖 reference 和最后一个未接受候选 B；A 的源码、分层结果和失败分类没有进入该提示词，也不能通过现有闭合源码目录请求。详细评测与运行反馈均指向 B，且与 B 的内联源码一致。

证据：`attempt_02/repaired_public_history.json` 是同一份实际 A02 ledger 经修复后的完整 `PublicContextBuilder.build_prompt` 只读重放，未调用 provider。其 `provider_payload.public_history.experiment_memory.experiments[0].trial_outcomes` 保留 A 的 `push_standing_recovery_height_margin_v1`、`add`、`RewardTerm:height_collapse`、`formal_rejected`、`public_dev_delta=0.078125` 和 `not_run_due_to_early_reject`；B 的 dev 差值为 `0.005208`。`records[0].generation.trials` 中原始公开 dev 均分分别为 A `0.8125`、B `0.7395833333333334`，父 reference 为 `0.734375`。

证据：同文件 `provider_payload.source_catalog.items` 仅 `current_best/training_code/components.py` 与 `memory/latest_nonaccepted/training_code/components.py` 可请求，后者源码 checksum 对应 B；`memory/best_observed/training_code/components.py` 为 `pending`、`requestable=false`，理由为 `no_eligible_nonaccepted_bootstrap_anchor`。`candidate_lineage.best_observed_experiment=null`，探索卡为空。A 的上述记忆行仍可见，但 A 源码既未内联也不在可请求集合，且没有 A 的绝对成绩、分层或失败明细。当前父源码仍为 reference，没有将 A 当作已接受策略。量化与边界复核见 `attempt_02/no_formal_real_replay_audit.json`。

证据：源头为 `agent_core/generation_contract.py::latest_nonaccepted_trial` 在最近代按 `reversed(trials)` 取最后一个已训练且未接受的候选；两条 lane 的固定存储顺序令 B 优先。修复后同一 helper 优先该代已有 `selection.selected_trial_id` 指向的已训练、未接受候选，其余情况沿用原有倒序回退。`prompting/source_catalog.py`、Push 与 Stair 的详细反馈仍共用此选择；未新增源码角色或任务成绩比较，已接受候选仍不能进入非接受源码槽，父策略和验收结论不变。

证据：`attempt_02/source_feedback_selection_tests.log` 为相关 128 项本地测试全通过；新增用例覆盖 selected A 被拒绝时取 A、A 被接受时退回 B、无 selection 时沿用 B。首轮 128 项中唯一失败是 Stair 旧测试期待 B；已为 A 补齐该 trial 自己的运行观察并将来源断言更新为 A，原 accept/sealed 不泄露断言全部保留，首轮失败及更新依据见 `attempt_02/source_feedback_selection_fix_audit.json`。该回执还记录实际 A02 历史在共享选择器和 Push task prompt 上的只读重放：反馈与运行来源均变为 A，公开 dev 为 `0.8125`，源码 checksum 为 `sha256:31c5a5ba883d08440f6ff6d36d7ba2c89314456098c9df79833200ba25711ea4`；此阶段回执只覆盖选择器与任务反馈，完整源码目录重建证据见下条。

证据：目标环境的最终完整重放 `attempt_02/repaired_public_history_final.json` 经 `PublicContextBuilder.build_prompt` 通过；`attempt_02/no_formal_and_selection_final_replay_audit.json` 核对 A 的 3,126 B 源码与冻结候选文件逐字节一致，位于原有 `memory/latest_nonaccepted/training_code/components.py` 槽，且 `inline`、`requestable=true`。A 的公开反馈为 `156/192`，四层为 `48/48、45/48、26/48、37/48`；失败计数与原 dev summary 相等，运行反馈的 6 个奖励项、72,000 次调用和全部展示的激活统计均与 A 的 train summary 对应。B 的机制、状态和 dev 差值记忆完全保留，reference 父源码与 `0.734375` 的公开 dev 不变。提示词从 58,679 B 降至 58,586 B，记忆仍为 2,799 B；29 个私有路径、4 个 checkpoint hash 未进入提示词，public-only 与任务回执重算通过。此项为既有实际账本的离线重建，未调用 provider，也不是新的多轮实验完成证据。

证据：目标环境扩大回归发现 `tests/integration/test_fixed_width_generation.py` 的 formal reject 场景仍期待 `best_observed=A`、`latest_nonaccepted=B` 且 checksum 不同。此场景现由原有去重机制将 best observed 别名指向 latest nonaccepted，同一 A 源码仅内联一次。更新测试核对别名、源码与 lineage 的相同 hash，同时明确 `accepted=false`、`authoring_anchor_only_not_policy_best`，未改产品或私有边界；整个集成模块两项测试通过，日志为 `attempt_02/source_feedback_selection_integration_tests.log`（44.902 秒）。

## F9：候选对奖励机制的解释超出或违背源码证据（非阻断）

问题：A03 第 1 代 lane-a 候选将自己的姿态奖励描述为梯度永不消失、蹲得越深总会得分越低，但函数最终将输出裁剪到 `[-1,1]`。裁剪区间内存在高度变化而输出不变的输入，因此这些全称表述不能当作已验证的函数性质。它们属于候选提供的待检验机制主张，不是系统认可的行为结论或接受依据。

证据：`attempt_03/session/provider_candidates/go2-push-5rounds-20260922-a03-generation-001/lane-a/001_push_recovery_nonvanishing_stance_margin_v4/training_code/components.py` 的 `TrainingComponents.rationale` 写有 “the gradient toward a nominal stance never vanishes” 和 “a deeper crouch is always scored worse than a shallower one”。同文件 `recovery_posture_margin()` 最后一行为 `return margin.clamp(min=-1.0, max=1.0)`，对应 RewardTerm 再声明相同输出裁剪范围。

证据：按冻结函数算术代入 `roll=pitch=0.70`、`base_height_relative=0.50` 与 `0.48`，裁剪前 margin 分别为 `-1.1225` 与 `-1.1525`，最终均为 `-1`。这是输入边界的数学反例，不证明这些状态已在本轮训练或仿真轨迹中出现。该候选的 verifier、组件 smoke 和单机制检查均通过；独立 AST 审计确认只替换 `upright_height`，其余四个函数及 RewardTerm 配置未变，见 `attempt_03/prompt_audit_gen01.json`。

证据：A06 第 2 代 B 的 `session/provider_candidates/go2-push-5rounds-20260923-a06-generation-002/lane-b/001_push_standing_recovery_stance_barriers_v2/training_code/components.py` 同时声明 `clip_min=-3.0`、原始 tilt barrier 为 “non-saturating”，并称超过倾角带会提供 “persistent linear-plus-quadratic gradient”。原始 barrier 子表达式确实没有饱和；最终 RewardTerm 仍受裁剪，因此不能把该梯度主张解释为最终奖励的全域性质。候选已经明确写出裁剪边界，此处不是隐藏的数值行为或 verifier 误接受。随后完成的 `attempt_06/session/runner/go2-push-5rounds-20260923-a06-generation-002-lane-b/train/summary.json` 在 control step 565 的峰值帧记录 `upright_stance_barriers=-9`（权重 3），实际达到声明的输出下界；未保存该时刻裁剪前值，不能进一步重建当时的局部梯度。

证据：A07 第 2 代 A 的最终 `push_recovery` 候选将 `stationary_v3` 乘以姿态/高度相关 gate，但 rationale 声称 quietness 有界、改动最大影响仅 1 个奖励率单位。`attempt_07/gen02_candidate_source_audit.json` 按真实函数、生产 `guard_vector` 和权重 1 核对有限输入：relative height=0.8、roll=pitch=0、scaled planar velocity=(1.6,0)、scaled angular velocity=0，原 quiet=-3.875，gate≈0.61878339，新值≈-2.39778564，改变量≈+1.47721436；无输出 clip 或 schedule。因此该全称保证不成立。新状态×速度耦合仍是合法机制；反例不证明真实轨迹达到该输入，也不证明此项实际主导训练或造成有害行为，未据此新增准入闸门。

证据：A07 第 2 代 B 新增 `-angular_xy_rate_squared/(1+8*tilt_squared)`，却将“项非正”推为不会与直立目标冲突。`attempt_07/gen02_candidate_source_audit.json` 使用全部真实奖励函数/权重核对有限输入：relative height=1、roll 从0增加至0.1、pitch=0、scaled angular velocity=(sqrt(2),0,0)、其余运动/动作误差为0且未终止；新增项由-2变为约-1.851852，父总奖励率由3.417893降至3.280262，而候选总奖励率由1.417893升至1.428410。该反例只否定“永不冲突”的全称说法，不证明实际训练出现此状态、投机或性能受损。机制本身独立且符合单项修改合同。

证据：A07 第3代A最终v8将作者阶段被拒的 `tilt_brake_v7` 写成拥有dev delta=-0.1875；原测量身份实际上是Gen2B的 `tilt_rate_brake_v6`。二者函数在改名后确实完全相同，但v7本轮没有训练/评测，不能将等价机制的历史成绩记为一次新实测。原prompt与历史门来源正确，见 `attempt_07/gen03_candidate_source_audit.json` 和 `prompt_audit_gen03.json`。同回执还核实其“只在nominal upright为零”等措辞过强（upright且relative height>=1均为零）；这些仅为候选解释问题，没有更改评价或接受依据。

证据：A08 Gen1A 的 `push_recovery_nonsaturating_stance_v1` 声称奖励 never flattens、近目标与旧信号相同，但 term 声明 clip_min=-4、weight=3。`attempt_08/gen01_candidate_source_audit.json` 按生产裁剪核对 relative height=0与0.05时加权输出均为-12；近目标height=0.96、roll=0.08时新值为3、旧值约2.90552，故两项全称保证不成立。作者将height_collapse汇总直接解释为shallow optimum也缺乏因果证据。这些是数学/证据表述问题，不证明轨迹达到反例，不改变合法的新机制资格或任务评测。

证据：A08 Gen2A 声称整个0.1–0.9倾角段的梯度都更陡。`attempt_08/gen02_candidate_source_audit.json` 按真实函数、weight与clip，在height=1、roll=0.8、pitch=0（未裁剪）核对新/旧加权奖励率对roll的局部导数约-14.399996/-21.696，旧函数此处反而更陡。新非线性形状仍是合法机制；这一有限输入反例不证明实际策略访问该状态或任何性能原因。

证据：A08 Gen3A 的 sink=max(0.05-v_z_scaled,0)，在名义站立、线/角速度均0时，新项按实际guard与权重仍贡献-0.0525，故“near-goal为零、只惩罚超过阈值的下落”等解释不成立。Gen3B 的 recovery_damping 返回非正值，weight=-0.5、clip范围[-2,0]，实际新增贡献非负；height=0.8、roll=0.5、pitch与角速度为0时，scaled平面速度由0增至0.2，新项由0增至+0.0609375，与其“阻尼惩罚”描述方向相反。两者证据见`attempt_08/gen03_candidate_source_audit.json`，只证明相应项的数学性质，不推断总奖励或真实轨迹。原smoke只证明有限值，没有把方向解释校验为真；实际评价与接受规则未被候选修改。

真实运行证据：A08 Gen3B训练峰值快照step2092/env1558中，recovery_damping的signed weighted contribution=+1.0、effective weight=-0.5；最后采样raw mean=-0.020189、weighted mean=+0.010094。`attempt_08/training_evidence_through_gen03.json` 与 `prompt_audit_gen04.json` 对照原train和第四代实际prompt，确认同一Gen3B source checksum的公开反馈原样传回-0.5权重、+0.010094均值及1.0最大绝对贡献，6项全部展示。该问题是候选解释/设计错误，执行和反馈没有将正项伪装为负惩罚；采样均值也不等于全部训练平均。

证据：A08 Gen4两候选新增项仍为非正函数乘负权重，与“恢复惩罚”解释相反。`attempt_08/gen04_candidate_source_audit.json` 使用实际函数、guard与权重核对：A在height=1、roll由0增至0.5时新增贡献由0增至+0.117551；B在roll=0、scaled侧向速度由0增至0.2时新增贡献由0增至+0.1，也否定其“roll死区内恒零”声明。两者虽同名lateral_brace，实际输入/函数不同，满足机制区分；上述反例只描述新增项，不推断总奖励或实际轨迹因果。

证据：A08 Gen5的新增项仍存在同类双负号问题。`attempt_08/gen05_candidate_source_audit.json` 核对A的roll死区平方项在roll=0.1/0.3/0.5时，实际加权贡献为0/+0.04/+0.16；B在roll=0.2、scaled angular-x=0.5时实际新增贡献+0.15，方向与解释中的惩罚相反。两条均保留父策略六项、仅新增一个真实不同机制，未访问额外私有输入；该记录不推断最终策略行为或性能原因。

## F10：候选将非接受兄弟的失败分布用于推断父策略改善机制（提示证据缺口已修复）

问题：A06 第 2 代 B 根据上一代未选中 A 的失败明细与已接受 B 的较高总分，声称父策略的 height barrier 已改善高度塌陷，并推测剩余失败主要由倾倒主导。这些是候选作出的未验证归因，不能由两条候选的总分差推出。实际提示词对来源标注正确，但只包含非接受 A 的详细失败表，父 B 的失败摘要已在内部上下文构造，却未序列化给作者。

证据：`attempt_06/session/authoring/go2-push-5rounds-20260923-a06-generation-002/lane-b/authoring_attempts/attempt_000001/prompt_payload.json` 的父策略为 `push_standing_recovery_height_barrier_v1`、dev `0.78125`；唯一详细失败表明确标注 `latest_trained_nonaccepted`、`push_recovery_height_hinge_v4`、dev `0.703125`，含 height 29、backward 14、side 9。对应 `attempt_result.json` 的 `continuation_state.selected_round_public_context.task_public_summary` 另有父 B 的失败率，但该对象没有进入实际 prompt。候选源码 rationale 使用 “Its one-sided, downward-modulated height barrier addressed…” 及 “residual failure mass is therefore expected…”；这超出了作者收到的证据。

证据：审计读取上一代 B 的真实 `eval/public_dev/summary.json` 为 height 16、backward 9、side 13、not_recovered 4，共 150/192 成功。该结果不证明因果机制，审计也未将其回填作者输入。源码、公开反馈、记忆和父策略身份分别保持正确；本项没有改变独立评测或接受判定，见 `attempt_06/prompt_audit_gen02.json`。

修复证据：`tasks/push_recovery/prompt_adapter.py` 从已接受公共记录定位当前父策略，复用 `_v2_dev_summary()`，在非接受候选的明细旁提供独立标明 trial、candidate、源码 checksum 的 `current_best` dev 简表；当前详细来源已经是父策略时不再重复。没有直接转发记忆的 latest/aggregate 汇总，也未引入 accept/sealed 数值、checkpoint 或训练私有证据。`attempt_07/prompt_repairs_audit.json` 用服务器原生账本投影及 A06 实际第 2 代提示词离线重放，新增父 B 的 150/192、四层计数与失败表均逐项等于原始 dev rollout，原非接受 A 的反馈完全不变；只有公共反馈及帧预算两个 section 改变，增加 1,281 B。原始 A06 提示词和回执均未修改，未调用 provider；未声称新增信息能自动证明或阻止候选的因果归因。两个机器人、后续拒绝轮、同源去重及 minimum-core 的覆盖见 `attempt_07/prompt_repairs_local_tests.json`，相关 52 项通过。

真实运行证据：A07 第2代实际提示词已把父Gen1A的dev 164/192（height18/back3/side5/not_recovered2）与非接受Gen1B的0/192（height192）分开；第3代仍保留父A独立摘要，并把最新非接受来源正确切换为Gen2 selected A，而非lane顺序末尾B。两源各只内联一次，见 `attempt_07/prompt_audit_gen02.json` 与 `prompt_audit_gen03.json`。

## F11：历史程序重复的修复帧裁掉比较源码，触发一次补取（已修复）

问题：F5 已覆盖单机制策略违规的修复，但 `historical_exact_program_repeat` 修复帧仍只保留失败候选，裁掉用于比较的当前父策略和已测非接受候选源码。作者下一次请求重新取回这两份公开源码，增加一次往返并消耗 5 次作者尝试中的一次。该候选后来仅通过了当时的作者准入；第 3 代账本因 B 作者失败以 `paired_authoring_incomplete` 闭合，A 为 `verified_not_run`，没有训练或评测。终审又发现 A 的纯数值族准入漏判，因此不能把当时的通过写成准入正确性或实验有效性的证明。

证据：`attempt_06/session/authoring/go2-push-5rounds-20260923-a06-generation-003/lane-a/authoring_attempts/` 的第 2 次回执因历史精确重复继续修复；第 3 次 `prompt_payload.json` 为 49,576 B，`source_payloads` 仅内联 `attempt/latest_failed/training_code/components.py`，没有父策略及 `memory/latest_nonaccepted`。第 3 次实际响应请求 `current_best/training_code/components.py` 和 `memory/latest_nonaccepted/training_code/components.py`，理由明确为比较已评测机制；源码服务返回 8,860 B（含封装），`source_request_count=1`。第 4 次提示词重新内联两份源码，合计正文 7,461 B；第 5 次为当时的准入通过回执。此前第 1、2 次提示词已经内联同一对公开源码，没有新私有证据被释放。

证据：A06 第 5 代 A 的第 3 次实际提示词同样由 `historical_exact_exclusion` 进入仅含失败源的修复帧，其第 1、2 次均包含父源和 latest-nonaccepted 源。该缺口不代表 verifier 错用父源：作者循环仍用轮首冻结比较源和既定 parent checksum。

修复证据：`prompting/authoring_frame.py` 沿用 F5 的 `select_authoring_anchor_source()`，为历史精确比较修复保留比较父源及已经公开的 latest-nonaccepted 源；普通语法修复仍只保留失败源。`attempt_07/prompt_repairs_audit.json` 对第 3、5 代实际修复输入运行生产投影、源码最终化和帧预算构造：父源、非接受源、失败源均保留，修复反馈未改，未放回公共历史/行为反馈，最终为 58,893 B、61,138 B，三个源码正文 checksum 均与原产物相等。该回执证明必要源码现在可直接读取，未声称已经实测减少新一轮的 provider 请求次数；相关 52 项测试见 `attempt_07/prompt_repairs_local_tests.json`。

真实运行证据：A07 第3代A第2次提交虽通过静态/单机制检查，但因与已评Gen2B相同而在历史精确排除门被拒；第3次实际repair prompt保留current_best 3709B、latest_failed 5164B、latest_nonaccepted 5749B，三份来源/hash一致且各一份，最终prompt 61689B，三次source_request_count均0。最终新机制通过，未将中途静态PASS冒充整体准入。见 `attempt_07/prompt_audit_gen03.json`。

## F12：首个高思考请求反复耗尽输出预算而无最终候选（非阻断）

问题：部分 lane 的首个请求将全部32768输出单位用于 reasoning，最终正文为0字节；这些请求有真实成本，也消耗作者尝试次数。系统已有关闭thinking的自动重试且能继续，因此不是服务中断或账本失真。

证据：A07 第1代A、第2代B、第3代A的 `session/authoring/go2-push-5rounds-20260923-a07-generation-NNN/lane-L/authoring_attempts/attempt_000001/attempt_result.json` 均记录 `reasoning_output_exhausted`、`finish_reason=length`、`reasoning_units=32768`、`response_text_utf8_bytes=0`，合计98304 reasoning单位。各自后续回执记录按已冻结fallback关闭thinking并完成候选；原请求用量保留，没有伪造候选或抹去成本。逐代原始响应回执与提示词核对见 `attempt_07/prompt_audit_gen01.json`、`prompt_audit_gen02.json`、`prompt_audit_gen03.json`。

A08 实跑证据（五代作者阶段完成，快照 `go2_push_a08_live_17.zip`）：第2代B、第3代B、第4代A/B、第5代A的首个请求均记录 `reasoning_output_exhausted`、`finish_reason=length`、正文0字节；reasoning实际分别32768、32767、32768、32768、32768单位，5次合计163839 reasoning单位、253423总单位。原始回执位于 `attempt_08/session/authoring/go2-push-5rounds-20260923-a08-generation-NNN/lane-L/authoring_attempts/attempt_000001/attempt_result.json`，逐条路径与用量汇总见 `attempt_08/prompt_audit_gen05.json`。5条的第2次回执均为 `requested_thinking_mode=disabled` 并最终 `candidate_verified`；第5代A实际最终回执记录1582输出单位、5778字节正文。重试保留原父源、latest_nonaccepted源、公开反馈、完整历史集合身份与当时的跨轮记忆，仅新增本次响应失败反馈；未虚构failed candidate或抹去首次用量。提示及实际准入核对见 `attempt_08/prompt_audit_gen02.json` 至 `prompt_audit_gen05.json`。完整五代共15次实际请求、10个verified候选、5次上述失败及5次成功fallback；第5代B首个请求直接通过，无额外失败。

## F13：完整已评历史未接入数值变体排除门（已修复，A08 非空历史实跑已对账）

问题：A07 Gen3B 的全部奖励函数与已评 Gen2B 相同，只将 tilt_rate_brake 权重1改为0.5，违反实际提示的历史纯数值变体禁令却开始训练。已评历史只进入 exact 实现键比较，现有 normalized verifier 门只有负样本集合；不是比较算法无法识别，也不是任务评价失误。A07 已因本阻断停止，不能算作有效5轮。

证据：`attempt_07/gen03_candidate_source_audit.json`、`prompt_audit_gen03.json`、`blocking_stop_07.json`；两程序 normalized key 相同，原 anti-degenerate 历史源码数为0，不能作为已对全部历史比较的证明。

修复证据：共享 exploration 从完整同 Genesis、同父已评卡片生成既有 verifier key，generation_lanes 传入同一个历史门；活跃 exact-only 分支已删除，负样本原身份约束不变，回执记录实际合并集合及各自来源。`attempt_07/historical_admission_fix_audit.json` 的真实源码重放对原样/改名/纯权重变体均拒绝，首次机制及新机制仍允许；`attempt_08/source_boundary_review.json` 独立复核集合身份和直接/嵌套修复帧，不增加任务分支或全历史源码。上述证明为修复验收，不是新5轮完成证明。

真实运行证据：`attempt_08/prompt_audit_gen03.json` 从完整账本重建Gen2两条同父已评family，实际Gen3两lane prompt的count=2/hash=8e1978a9…，实际历史比较的evaluated及union hash也逐项一致；未内联的Gen2A仍参与，原negative集合为空且身份独立。Gen3A/B独立源码投影均不命中该集合，回执为distinct/pass，与源码审计一致。此结论证明新门在非空历史下实际接线，不宣称本次自然发生过历史重复修复或5轮已完成。

# 5 rounds 启动与阻断修复记录

2026-09-22。

## 正式入口轮数

正式 CLI 原先把 composition 与 final plan 的代数固定为 15，不能实现本次要求的完整 5 rounds。共享成本、会话合同、循环及终态本身已支持冻结代数。

修复仅在 `scripts/run_generation_session.py` 增加 `--max-generations`，同一值进入 composition、final plan 与 launch；`GenerationLaunchConfig` 补正整数检查，暂停必须位于最终代之前。默认仍为 15，无 Push 或 Go2 分支，未改任务规则。runbook 同步说明默认值和显式代数。

目标环境的 CLI、成本及 5 代完整闭合/只读重放专项 8 项全部通过，exit 0，106.631 秒，见 `target_round_count_tests.json/log`。本地测试初轮暴露既有坏 manifest fixture 的异常包装与文案预期过窄；修正测试后入口 3 项通过，损坏文件不得重写、缺失文件不得补造的核心检查保留。成本与 Push 完整会话用例通过，本地无法执行的 Stair 集成明确记为 skip；完整范围见 `round_count_local_test_receipt.json`，不将不同运行拼成一次全量通过。

## A01 执行冻结

- 正式身份：`go2-push-5rounds-20260922-a01`，5 generations、width 2。
- 当前工作源码形成独立执行提交 `f60324a904a0addd933c9850c812607a3264d43c`；373 个执行文件内容一致，原工作区的分支与暂存区未改。
- 服务器执行目录：`/data/yuyangliu/AGENT/true_RL_Agent/.tmp/go2_push_source_01`；结果目录：`PLAN/2new_world/Go2_push/attempt_01/session`。完整提交保存在 `execution_source_01.bundle`，来源与文件摘要见 `source_snapshot_01.json`。
- `prepare-only` 通过，使用同一参数加 `--resume-existing` 执行；原生宿主负责新 reference 的训练及 dev/accept/sealed 资格，随后才进入候选循环。旧 D checkpoint 仅作为成本来源，不被冒充当前完整资格。
- `launch_01.json`、`provider_policy.json`、`cost_contract.json`、`cost_sources.json` 保存本轮输入及操作上限；成本中的外推和预留均有标注。未改训练预算、reference 奖励、成功判定或扰动强度。
- 启动前两张 GPU 无计算进程。原 D/G1/c6 的 37/49/1271 个保护文件大小与 mtime 均匹配既有清单，见 `protected_before_01.json`；本项不是一次新的全量内容重哈希。

本记录说明启动与实现，不签发实验完成或策略收益结论。非阻断问题单独写在 `FOLLOWUP_ZH.md`，该文件只包含问题与证据。

## A01 中断：smoke 执行器误拒合法候选

A01 的 reference 完整资格通过（dev 141/192、accept 142/192、sealed 343/448），进入第 1 代后，lane-a 原始候选通过静态验证，却被 `SmokeTensor` 的 `unsqueeze` 缺失拒绝。相同 fixture 数值及未改字节的候选在真实 CPU Torch 上，五个奖励均输出有限 `(4,)`；该模拟实现还缺少正常广播。此错误改变了合法候选的搜索空间与修复方向，按阻断处理，不能把自动改写成功当作系统正确。证据见 `attempt_01/first_candidate_real_tensor_audit.json` 与原始 `attempt_result.json`。

仅对经进程参数与组长核对的本会话宿主及两路训练组发送 SIGINT，三组均退出、GPU 无计算进程、持久锁文件保留，实际 GPU lease 已释放（额外非阻塞锁核验见 `attempt_01/resource_release_01.json`）。停止回执为 `attempt_01/blocking_stop_01.json`，原始失败批次保留，不并入下一批 5 代。

用户随后要求在重启前一并审查并修复 followup 中必要问题；修复状态和证据写回各条目，所有修改验收并冻结后再创建 A02，运行中的旧快照不修改。

## A02 重启前的源头修复

- B1：Push fixture 使用真实 CPU float32 Torch；删除共享 `SmokeTensor` 的模拟运算与生产识别分支。Go2/G1 均用原候选源码通过完整生产 spawn smoke，真实错误 shape/NaN 仍拒绝，既有启动/调用隔离保留。
- F1/F2/F4：任务自身负责 dev 反馈、父策略同卷描述差值及聚合失败分类，复用已有投影和分类函数；初代 reference dev 的分层/失败反馈不再缺失。无公开 guard 时不编造 guard delta，没有组件因果证据时明确 not_assessed。
- F3：宿主持久化任务返回的失败资格证据；失败会话仍无 Genesis/ledger，也不隐式重建。
- F5：单机制策略重写保留 verifier 使用的同一比较源码；普通静态修复仍裁剪无关上下文，不新增源码请求渠道。
- F6：任务 runtime 的输入来源回执同步描述最新构造输入，奖励数值不变。

同时审计了探索卡：旧 Push projector 含 accept 分数，但当前 unknown/cooldown 消费链不将其送往 provider，未认定本批发生泄漏。此处在任务 owner 收敛为 dev-only，并升级其投影合同版本；不改变正式 accept 判据，也不把描述性 delta 当作机制因果收益。

本地分别完成 smoke 122 项、反馈相关初轮 114 项与最终标签修复 74 项、修复帧/authoring/Genesis 合并 98 项、runtime 37 项专项；这些运行有重叠，不能相加声称全量唯一测试数。回执与真实 A01 数据投影回放均在本目录。服务器将对合并源码运行独立验收，通过后再启动新会话。

## A02 正式启动

服务器合并验收 275 项全部通过（194.438 秒、exit 0），包含真实张量 smoke、authoring、Push 反馈/运行/评测、reference Genesis 以及完整 5 代闭合重放，见 `target_followup_tests_02.json/log`。新源码提交 `46d5d9e5fff072e7154840bbe8b6fd474cb236be`，会话 `go2-push-5rounds-20260922-a02`，宿主 PID 1186350。`prepare-only` 通过后正式续接同一准备合同；这是新会话的正常启动，不复用 A01 的候选或 ledger。

17 个源/测试文件在确认服务器主目录仍匹配 A01 或最终内容后同步；373 个执行文件与本地、A02 独立快照逐文件 hash 一致，见 `source_sync_02.json`。既有 D/G1/c6 的 1357 个保护文件大小/mtime 无变化，见 `protected_before_02.json`。执行提交完整保存在 `execution_source_02.bundle`。

运行仍进行原生 reference 训练与 dev/accept/sealed 完整资格，随后才进入 5 代候选闭环；目前不签发完成结论。

## A02 reference 与第 1 代作者验收

独立 reference 训练完成 49,152,000 env steps，dev 141/192、accept 142/192、sealed 343/448，通过任务资格；三阶段逐 episode 结果与 A01 相同，仅运行元数据变化。F6 的最新输入来源回执为 observed，不能据此声称首步没有零初始化。具体身份、checkpoint 与回执核对见 `attempt_02/reference_reproduction_audit.json`。

第 1 代两条 lane 均首次作者提交通过 static verifier 与真实 Tensor smoke，0 源码请求、0 修复循环。实际输入分别 12,381 / 12,580 tokens，payload 48,301 / 48,990 B；每个候选仅改一个奖励机制，reference 的 dev 分层和主要失败反馈完整。lane-b 仅接收兄弟机制排除信息，不接收 lane-a 源码，见 `attempt_02/prompt_audit_gen01.json`。候选随后进入两卡原生训练；本节不声明收益或五轮完成。

保存的 reference 832 个 episode / 1,025 次外力事件还经独立算术复核：分类、计数、质量尺度外力、实际冲量和整段推搡后的恢复时间边界一致。40 个因先摔倒而未完成全部推力的 episode 仍计为失败，没有从分母删除。该核对只使用已保存 summary，不是对连续帧姿态与 dwell 的独立重建，见 `audit_physical_evidence.py` 和 `attempt_02/physical_evidence_audit.json`。

## A02 中断：无正式候选的代记录丢失结构标记

A02 第 1 代完成两次原生训练与 dev：lane-a 156/192、lane-b 142/192，选择 A 正确。A 的 accept 为 150/192，对 reference 的 142/192 提升 8/192=0.0416667，小于冻结 min_delta=0.05，正常提前拒绝，sealed 标记为 not_run_due_to_early_reject。该代 formal_rejected 已落入 ledger，head 不变。随后准备下一代 context 时抛出 current provider record has invalid mechanism attribution，进程 exit 1；这属于系统阻断，A02 不能作为完整五轮。

根因是 Push 提示投影丢掉已校验 generation 中的 formal:null，破坏共享层用于表达“本代没有单一正式候选”的结构合同。共享记忆于是错误进入要求单一候选 attribution 的分支；并非实际候选的机制证据造假。Stair 已保留这一结构事实。修复应只在 Push 投影保留显式空 formal，不能放松共享 attribution 验证，也不能把两个 lane 的归因拼给一条代记录。

宿主自然退出后，两自有 PID 均已消失，两 GPU 无计算进程，两 lease 文件经非阻塞 flock 确认未占用，见 attempt_02/resource_release_02.json（服务器回执）。原始 ledger、phase receipt、完整日志与候选源码保留；没有删除本批或历史 checkpoint。修复与实际历史重投影验收完成后再冻结 A03，重新开始完整五轮。

## A03 重启前的 followup 复核

F1–F6 的修复保留；B2 只保留 generation.formal 的显式空结构标记，同一 A02 receipt-bound 历史在旧代码复现失败，在新投影的完整 PublicContextBuilder.build_prompt 中通过。7 个任务投影单元用例与 1 个真实回执集成场景通过，共 8 项；有正式候选却缺失归因的情况仍拒绝。证据见 attempt_02/no_formal_projection_fix_tests.json、no_formal_real_replay_audit.json。

F7 已核实：训练存在大幅奖励率和价值损失尖峰，但 reference 同样存在，新增 bounded reward 无法直接贡献该极值，且独立 checkpoint 评测没有失效。其余量不足是定位证据，不是另改 reward 或新增硬门的依据；作为非阻断未解决项保留，不在本次修订科学条件。

F8 需要本次修：最近一代的源码/反馈选择倒序偏向 lane-b，导致已通过 dev 选择但最终未接受的 A 缺少可检索源码与详细反馈。修复复用同一共享选择 helper，优先最近一代已训练且未接受的 selected_trial_id；否则沿旧回退。任务 selector 仍是 dev 选择真相，共享层不比较机器人分数，不增加源码角色，也不晋升未接受策略。

首次 A03 准备快照的目标回归运行 80 项、1 error：验收包未带入测试直接读取的 docs/task_extension_guides/stair_climbing.md。此失败如实保留于 target_projection_tests_03.json/log；补齐文件并合并 F8 后须重新验收，尚未开始 A03 正式会话。

## A03 正式启动

最终执行提交为 `f9305ed993f6277c65ae0f94f4e281bb6bcb41f0`。本次相对 A02 仅两处产品修改：Push 投影保留 formal:null；共享已训练非接受来源优先任务 selector 已选中的候选。均未改任务科学定义、训练预算、reference 奖励、public/holdout 划分或接受门槛。

目标环境合并回归 201 项先通过 200 项，唯一失败为完整 generation 集成对旧 A/B 双源行为的断言；产品 hash 不变，更新为同一 A 的 alias/hash/非接受权限后，整个集成模块 2 项复跑通过（37.774 秒、exit 0）。分阶段原始证据为 target_restart_tests_03.json/log 与 target_selection_integration_tests_03.json/log；不把这些记录拼成一次不存在的全量运行。修复后的完整 A02 账本重建同时通过；A 源码、分层成绩、运行反馈同源，B trial 记忆与 reference 父策略保留，见 attempt_02/no_formal_and_selection_final_replay_audit.json。

F1–F6、F8 为已修复；F7 的训练极值定位证据缺口保留为非阻断。完成这一逐项判断后才启动 A03。服务器主目录的 6 个源/测试文件在校验旧 hash 后同步，全部 373 个执行文件与本地及最终快照相同，见 source_sync_03.json；执行 bundle 含准备快照历史和最终提交。原 D/G1/c6 共 1357 个保护文件大小与 mtime 无变化，见 protected_before_03.json。

prepare-only 通过，合同 hash 为 `672fa64ec00c8cfb940de287224fb7a5a37752b3eecaf3ebb5664ac51cef2f23`。新会话 `go2-push-5rounds-20260922-a03`，宿主 PID 1208663，supervisor PID 1208662。启动前 GPU 空闲、lease 可获得；启动仍执行新 reference 原生训练及三相资格，再进入新的完整 5 generations。A01/A02 继续保留，不拼接轮次、不人工替换候选。

## A03 第 1 代闭合与第 2 代自动接续

A03 新 reference 完整训练与三相资格通过，dev 141/192、accept 142/192、sealed 343/448；科学结果与 A02 相同，身份与原始回执核对见 `attempt_03/reference_audit.json`。第 1 代两候选各完成 49,152,000 env steps，dev 为 A 159/192、B 161/192，任务 selector 选 B。B accept 154/192，相对 reference 提升 0.0625；sealed 348/448，accept-to-sealed 差 0.0252976。正式决定为 `policy_improvement_accept`，current-best 变为 B。

`attempt_03/gen01_decision_audit.json` 独立重建任务 sealed 证据及正式决定，核对 execution、ledger、closure 与 head 的 hash；本代两个训练、16 个 eval seed 的成本闭合，open reservation 为 0。第 1 代的接受不代表五轮完成，也不证明跨种子训练收益稳定。

宿主自动构建第 2 代。当前最优源码为 B，已测未接受来源回退到 A，A 的源码与详细 dev/runtime 反馈一致；两条 trial 的同卷 dev 差值均进入有界记忆。lane-a 首次 provider 响应 `finish_reason=length`，被记录为 `response_truncated` 并自动重试，第二次提交通过 verifier 与真实 Tensor smoke；未调整输出预算、未人工补写候选。后续真实多轮提示词审计在 `attempt_03/prompt_audit_gen02.json` 持续记录。

## A03 中断：纯系数候选被误判为机制变更（B3）

第 2 代 B `push_recovery_lean_guarded_damping_v5` 相对已接受父程序，将 `1 - 0.5 * lean` 改为 `(1 - lean).clamp(min=0)`，而 lean 自身已有上界 1，因此新 clamp 数学上冗余，有效改动只是系数 0.5→1。原始源码和逐项证明见 `attempt_03/prompt_audit_gen02.json` 的 `lane_b_parent_mechanism_concern`。两条兄弟候选数学上并不等价，问题是 B 相对父程序的资格，不是兄弟重复或评测信息泄漏。

冻结提示明确禁止对已评测源码只做数值变体，但共享比较器将此项记为 `mechanism_delta_present`；直接纯参数分支本身还保留 warning、不阻断的旧行为。源码静态比较能力有限不能证明一般数学等价，但已确认的合同违反不能冒充机制改进。故本批按资格错误放行中断，不把已完成的第 1 代拼入下一批，也不否定其原始物理结果。

确认 PID、PGID 和 A03 会话参数后，仅向宿主 1208663 及两训练进程 1224590/1224591 发送 SIGINT。supervisor 自行落盘 returncode=-2；四个自有进程均退出、GPU 无计算进程、双租约 flock 可重获，见 `attempt_03/stop_for_mechanism_gate_03.json`、`resource_release_03.json`。受保护的 D/G1/c6 共 1357 文件大小/mtime 均未变化，见 `protected_after_03.json`。保留全部原始记录及 checkpoint。

重启前只修可证明的准入比较问题，未知表达式维持结构比较，不构造通用代数证明系统，不改变原始源码/机制身份 hash 的算法。同时复核 followup：F1–F6/F8 保留既有修复；F7 补最小同刻峰值分项记录，奖励和训练条件不变；F9 是候选假说的过度表述，保留证据，不新增 LLM 审查门。所有新增修改经真实反例与相关链路验收后，才冻结并启动新的完整五轮。

## A04 准备与 reference 复用边界

B3 修复归共享 `component_source_semantics.py`：只在准入比较的源码副本中约简已证明冗余的 clamp，并将显式/隐式标量系数归入同一参数族；未知表达式和复杂控制流不额外推断。原始源码与机制 delta hash 的算法保持不变。Stair 原有重复奖励项、阈值严格性家族规则同时迁入共享 owner，删除任务层重复 helper，任务输入/物理奖励/资格约束保留。真实 A03 的 B 反例拒绝、A 结构变更放行，原两份 delta hash 的断言保留。

F7 已补 Push runtime 的单个同刻峰值快照：只在全程绝对峰值刷新时存 step、env、有符号总值与各项加权贡献，保持计算顺序、奖励输出、原最大值统计和提示投影不变。该模块 17 项通过，包括真实 Torch；旧峰值没有同刻数据，不能追认成因。

A04 首个验收快照 `c912602f430058e221d960f579618b67e0707e01` 在目标环境运行 271 项，268 项通过、3 项错误，原始记录已保留为 `target_restart_tests_04_initial.json/log`。三个错误来自作者循环共用的旧测试夹具：新增 `support=base_height*0.5` 只是现有项的别名参数变体，被新门合理提前拒绝，未到达原历史排除/修复帧断言。夹具改为 `base_height*roll.square()` 的真实新结构，原断言保留；本地作者循环完整 45 项通过（27.376 秒）。这不是已经完成的新会话，也不把分阶段测试拼成一次全通过。

用户追加要求优先复用 reference。A04 因此不再默认新训：保留 A03 原 reference 的训练 job、执行环境、summary 与 checkpoint 身份，只在科学条件相同且原训练证据经正式 Runner 校验后复用；本次代码下重新执行三相资格评测。五代候选仍使用全新会话和空账本，不拼接 A03 首代。现有完整 Genesis import 不接受 generation-bound 来源，复用将在已有 completed-reference-training 接口上接入正式入口，不伪造 standalone qualification 回执。reference 的保守成本上限可以保留，实际训练必须记为 0 次。


## A04 最终冻结与启动

最终执行提交 `23b2bd26765f33fba64caf90b85f5be60573fd89`；目标环境合并回归 **351 项通过，270.414 秒，exit 0**，见 `target_restart_tests_04.json/log`。独立 B3 审计见 `attempt_03/mechanism_admission_fix_audit.json`；reference 复用审计见 `attempt_04/reference_reuse_code_audit.json`，补充验证了生产成本消费者不重复计旧训练。

正式入口新增已完成 reference 训练来源参数，保留 A03 原训练 spec/result/checkpoint/执行身份。新 Genesis 的执行信息绑定来源 Genesis 的路径、hash 和 checksum；科学身份不因执行位置改变。新三相评测使用本次执行身份，成本只计本次评测。恢复仍要求同一来源，不导入旧候选账本，也不为旧峰值诊断重新训练 reference。该能力归共享 Genesis/runner/成本 owner，没有增加 Push 或 Go2 分支。

prepare-only 通过，合同 `01c7793f51ac1f96c29a00ed6e501b99982dd134016a0723c1cab34b74e481861`。会话 `go2-push-5rounds-20260922-a04` 的 supervisor PID 为 1240178。启动前 GPU 无计算进程；原 D/G1/c6 共 1357 个保护文件大小与 mtime 无变化，A03 来源 Genesis、私有证据、训练 summary 与 checkpoint 的实际 bytes hash/mtime 已记录。服务器主目录与执行快照全部 373 个执行文件相同，见 `source_sync_04.json`。此时仅完成启动，尚未宣称 reference 实跑验收或五轮完成。


## A04 在首个作业前退出：已绑定训练回执的只读恢复缺陷

A04 宿主 PID 1240179 启动后，在 `SubprocessRunner.recover_completed_train()` 校验 A03 原 summary 时退出（exit 1）：`unbound summary cannot claim a GPU process binding`。恢复调用没有传入执行 context，旧 summary 校验器因此按 unbound 解释；实际旧 TrainJobSpec 已保存 GPU 绑定。该真实组合此前被 mock recovery 的 Genesis 测试漏掉，351 项回归通过不代表它已验收。

失败发生在新作业、Genesis 和候选生成之前；无 reference 重训、无仿真评测作业。保留 A04 冻结源码和失败回执，在 runner 底层修正只读恢复对原执行身份的校验，原回执和 checkpoint 不修改。修复后另起 A05，不拼接中断批次，继续复用 A03 reference。实际来源字节/mtime 和 GPU 释放检查见 `attempt_04/startup_failure_audit.json`。


## A05 真实恢复通过并进入新资格评测

修复仅将只读恢复的 GPU 绑定校验来源收敛到原 TrainJobSpec；正常启动仍使用实际 launch context，不创建假 lease/context，也不放松未绑定或错绑定拒绝。独立复核用真实恢复器复现去掉该传参时的 A04 原错误，修复后通过。目标环境本次受影响链回归 **115 项全通过，69.109 秒**，见 `target_restart_tests_05.json/log`。

执行快照为 `e4966704e6af631cc296b210daeef5f543186114`。在启动前，正式 `SubprocessRunner` 对服务器 A03 原训练 spec/summary/checkpoint 做只读恢复成功；checkpoint 仍为 `sha256:1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff`，原训练 job `go2-push-5rounds-20260922-a03-train`。探针禁止调用 run_train，证据见 `attempt_05/real_reference_recovery_audit.json/log`，不是 mock recovery。

A05 supervisor 1243084、宿主 1243085 已启动。首次作业为新的 `eval_phase`（子进程 1243095），无 reference train attempt，stderr 为空；正式三相资格正在进行。原来源四文件 bytes/hash/mtime 仍完整，D/G1/c6 的 1357 个保护文件元数据仍未变化。五代完成与最终验收仍待实际运行，不将启动视为闭环通过。


## A05 reference 验收通过；余额不足终止候选实验

`attempt_05/reference_audit.json` 的 63 项核对通过：复用 A03 原训练，A05 三相新测为 dev 141/192、accept 142/192、sealed 343/448，完整任务证据仅运行标识和时间不同；新阶段的 PID 与执行环境独立。reference 阶段实际 **0 次训练、13 个 eval seed、252096 control steps、0.0546513 GPU 小时**。首张实际提示词 48301 B，39 个私有来源值均未出现，reference 公开分层/失败反馈与 A05 新测逐项一致。

第 1 代 lane-a 先发生 reasoning_output_exhausted，按既有策略关闭 thinking 后得到 json_parse_error，第 3 次为 HTTP 402。用户随后确认 API 余额不足并已充值。旧宿主将服务失败闭合为 `paired_authoring_incomplete`，但在第 2 代仅构造提示词后，被原有 `prior provider request has no authoritative usage receipt` 预算门阻止，自动 exit 1。实际只有 1 个未生成候选的闭合代，**没有第 2 代 API 调用、候选训练或有效候选评测**，不能算作五轮真实验证。检查时进程已退出，未向任何 PID 发出停止信号，GPU 无计算进程。

原始失败、provider 分类和完整 stderr 保留于 `attempt_05/provider_402_audit.json`、`.tmp/go2_push_a05_stopped.zip`。402 缺少权威 token usage，保留未知，不推定免费或放松成本门。必要源修复仅将明确的 billing/auth HTTP 401/402/403 提升为会话级服务阻断：持久化实际失败后上抛，不封成候选代；JSON/机制等候选失败仍沿原合同处理。修复与验收完成后另开 A06 完整五代，仍复用 A03 reference 训练，不拼接 A05 无候选代。


## A06 服务阻断修复与准备

产品仅修改共享 provider 错误分类和 authoring loop：明确的非重试 HTTP 401/402/403 在实际 attempt 持久化后抛 `ProviderServiceUnavailableError`，由原 session 异常路径保留成本/退出证据，不封闭候选代、不启动兄弟 lane。候选 JSON/机制失败和一般重试规则保留；未知 usage 仍使用既有保守核算及预算守卫，不推定 402 为零用量。反馈明确等待服务可用后重发，不要求改候选源码。

本地定向 3 项通过（包含 401/402/403 子例、失败持久化、纯 authoring 的恢复、一般非重试 400 与瞬时传输重试）；实际 shared generation 集成新增 402 不写 ledger、不启动 lane-b/训练的断言。执行快照 `7c58ec32f6dcdd4da238fe52321c58c9abf582c8`，新会话 `go2-push-5rounds-20260923-a06`；目标合并回归仍在完成，不提前宣称通过。

用户已确认充值。新持久 SSH 连接检查发现两张卡各有 VLLM worker（1383501/1383502，约 29 GB/卡）；用户选择自行释放，因此未停止这些进程。prepare-only 与对 A03 实际 reference 的只读恢复再次通过。新会话尚未启动，等待目标回归和资源释放；仍将复用原训练、重做三相资格、从空候选账本开始完整五代。


## A06 正式启动

目标合并回归 **112 项通过，84.730 秒，exit 0**（作者循环、provider、成本与完整 generation 集成），见 `target_restart_tests_06.json/log`。独立 service-stop 快审通过，使用 A05 真实 402 回执核对分类与层级传播；没有修改未知 usage 规则，见 `attempt_05/provider_service_stop_audit.json`。

用户释放资源后重新检查两卡无计算进程，再启动 supervisor **1410178**。A06 完整五代会话已开始，实际来源仍为 A03 已训练 reference；三相资格会新执行。原 1357 个保护文件元数据未变，A03 来源四文件 bytes/hash/mtime 未变，服务器主目录和执行快照全部 373 个执行文件一致。启动资源回执见 `attempt_06/launch_resources.json`。A05 的 API 原因已明确为用户确认的余额不足，不计其不完整代进入 A06。

## A06 reference 资格与第一代启动

`attempt_06/reference_audit.json` 完成 **64 项检查、28 份证据**：实际复用 A03 reference 训练，新执行 dev **141/192**、accept **142/192**、sealed **343/448**。本批 reference 实际训练为 **0**，新评测为 **13 个 seed、252096 control steps、0.0555400 GPU 小时**。旧 train result 的执行身份保留，新评测绑定 A06 执行环境；资格前后 A03 四个来源文件的 size、mtime_ns 和 SHA 全部一致，见 `reference_source_before.json` 与 `reference_source_after_qualification.json`。

第一代 A/B 均首次作者提交通过 verifier 和 smoke，随后由正式宿主并行训练。A 替换姿态/高度奖励为非对称高度 hinge，B 在原姿态奖励中加入与向下速度相乘的低高度 barrier；这是候选机制描述，不是效果结论。未改训练、评测、接受门或冻结源码，也未人工修改候选。五代、最终确认与终态审计仍待完成。

## A06 第 1 代闭合

两条 lane 各完成 500 iterations、49,152,000 steps，初始策略配对及运行激活通过。公开 dev 为 A **135/192**、B **150/192**，正式选择 B；只有 B 执行 accept **160/192**、sealed **346/448**。与 reference 的 accept 差为 **0.09375**，按本批冻结合同接受 B。`attempt_06/gen01_decision_audit.json` 独立重算选择、决策、身份、ledger、成本及资源闭合，44 份证据通过。本代实际新增 2 次训练、16 个 eval seed、2 次 provider 请求；没有开放中的作业回执。

语义核对：当前 candidate acceptance 不声明 public-to-sealed score-drop 数值门；`0.1` 位于 reference qualification。B 的 accept 与 sealed 差 `0.0610119` 只是描述，不能拿 reference 准入门代判候选，也没有在运行中改门。首代提示词审计见 `prompt_audit_gen01.json`：48–49 KB、记忆 996 B、单份 reference 源码，未发现私有评测或兄弟源码泄漏。`physical_evidence_through_gen01.json` 对 7 份 summary、1,856 条 episode 的记录算术通过，不宣称独立逐帧物理验证。

## A06 五代终态与新准入阻断

A06 自动完成五条闭合及最终确认，终态只读重放四项 PASS；8 次候选训练、26 次 provider 请求、70 个 eval seed，最终 head 为第 1 代 B。第 2、4、5 代正常未过提升门，第 3 代 B 作者耗尽，整代未训练。具体成绩、成本、保护范围与证明见 `attempt_06/ACCEPTANCE_ZH.md`。

独立源码审计发现第 3 代 A 的新增 hinge 其实是父 Huber 尾部重复加权，纯系数修改通过语法改写逃过了准入。把同一函数写成显式尾部系数 3 时，冻结门会拒绝；新增 `clamp` 写法却通过，见 `attempt_06/mechanism_review_gen03a.json`。候选未进入训练或 head，但错误准入影响作者/兄弟约束，故 **A06 整体验收 NO-GO**，不以冻结程序终态 PASS 覆盖该问题。

源修复限定为共享 admission-only 的已证实 clamp 尾部恒等式及原有基底系数比较，不更改源码或 mechanism delta 身份哈希。同步复核 F9–F11：候选夸大解释继续保留非阻断证据，不新增解释验证系统；父策略 dev 细节缺失与历史重复修复帧缺比较源在各自投影层修复。完成必要回归及真实产物重放后，另开 A07 完整五代，仍复用 A03 reference 训练。

## A07 源修复验收与冻结

仅三个产品文件变化：候选准入的 clamp 尾部等价比较、Push 当前父 dev 投影、历史重复修复帧的公开比较源码保留。相对 A06 的九个已通过候选重新运行准入，只有已确认纯系数等价的第 3 代 A 改为拒绝，九份 mechanism delta 身份哈希不变；八个真正训练候选仍通过。新阈值/输入/索引/非线性/状态相关权重的用例保留结构资格。独立复核见 `attempt_07/source_boundary_review.json`。

F10/F11 真实历史与修复帧重放通过，见 `attempt_07/prompt_repairs_audit.json`，原始 11 份证据不变。父 B 自身的 dev 成功 150/192 与失败分类进入任务自有摘要，增加 1281 B；A 的详细反馈原样。第 3/5 代历史重复修复帧分别 58893/61138 B，保留必要公开源码，普通语法修复不扩张。FOLLOWUP 已标记已修复。

目标 Python 在冻结快照上执行七模块相关回归 **155 项、124.335 秒、exit 0**，无跳过，见 `target_restart_tests_07.json/.log`。冻结执行提交 `5f13b475335b871a8aa458d667ece16763dd78f6`，全部 373 个执行文件与服务器主目录一致；A03 四份 reference 源文件大小/mtime/SHA 均不变，历史 1357 个保护文件大小/mtime 不变。新会话为 `go2-push-5rounds-20260923-a07`，新的空账本、原科学合同和成本上限，复用 A03 训练后重新执行 reference 三相资格。

A07 首次启动复用了旧运维 helper，它无条件追加 `--resume-existing`，被入口正确拒绝（新 session 目录尚不存在）。当时无 Genesis、provider 或仿真作业；原日志另存 `attempt_07/prelaunch_resume_flag_failure/`。已仅改本次运维 helper，使用冻结 launch 原 argv 创建新会话，未改产品或科学条件，亦未复用任何轮次。

## A07 reference 资格通过、第一代进入训练

`attempt_07/reference_audit.json` 的 63 项检查通过：新的三相资格为 dev 141/192、accept 142/192、sealed 343/448，原始任务证据除运行 ID/时间外复现 A03。本批 reference 训练实际为 0、eval seed 13、control steps 252096、GPU 容量时间约 0.0533005 小时。第一代 A 经推理耗尽、复合机制拒绝、JSON 格式失败三次修复，第 4 次新增单个 tilt_damping 通过；B 首次通过。宿主已自动进入双卡各 500 iterations 的训练，当前无新代闭合。候选源码与修复帧分别由独立审计核对，来源和证据限制见 `gen01_candidate_source_audit.json` 与 `prompt_audit_gen01.json`。

## A07 第 1 代闭合

独立重算选择、接受和成本/ledger 链通过，见 `attempt_07/gen01_decision_audit.json`（44 份证据）。A dev 164/192、accept 156/192、sealed 371/448；B dev 0/192，A 被接受并成为父策略。accept 差值 0.0729167 大于冻结 noise-aware threshold 0.0551198，未改验收门。两条训练均完成 500 iterations；首代训练的最终数值/峰值证据另核，不以训练奖励冒充任务成功率。

B 的全部192个episode在第11步（0.22秒）高度跌至0.16366–0.16650米，低于0.18米失败阈值；第一次外力计划第51步，因而push=0符合先发生失败的时序，原192个分母全部保留。计数与分类在任务层的解释成立，仍不能从汇总记录推断优化原因。正式五代结束及资源释放后，追加一次原条件原生短回放核对视觉，不让诊断作业干扰配对训练或成为接受依据。

## A07 第 2 代闭合

A/B 均完成训练，dev 分别136/192、128/192；选中A的accept为132/192，低于父156/192，按冻结noise-aware门提前拒绝，sealed未运行。`attempt_07/gen02_decision_audit.json` 对46份证据重算通过，父head仍是第1代A，系统自动进入第3代。完整作者输入审计证实F10真实接线：父A的164/192及其失败分布与未接受B的0/192分别投影；两源各一份、memory3551B、未泄露私有数值。候选的两条数学全称保证反例归并现有F9，仅记录问题与证据，不以解释错误新建准入门。

## A07 第 3 代准入阻断与停止

Gen3B 与已经评测的 Gen2B 奖励函数及其他行为相同，唯一有效变化为 tilt_rate_brake 的权重由1变为0.5。实际提示明确禁止已评源码的纯数值变体；生产的 normalized verifier key 也已经能识别二者等价，但正式链路只将完整已评历史接入 exact key 比较，normalized gate 收到的却只是负样本抑制集合。因此问题在历史集合接线，不再扩展算术恒等式解析器。Gen3A 的第一次重复被正确拒绝、随后新机制通过，并不能抵消 Gen3B 的误准入。

发现后核对会话主进程命令、所有后代及进程启动时间，仅停止 A07 所有者进程树；回执 `attempt_07/blocking_stop_07.json` 与 supervisor 的 returncode=-15 保留。两张卡随后无计算进程。A07 仅有两条完整闭合代，第三代两条训练均未完成；最新已结算成本只覆盖第二代，不能冒充包含第三代 API 与部分训练的最终总额。原始 job reservation 未重写为成功，也不拼接本批到下一批。原生视觉短回放在停止后单独运行，只核对 Gen1B 先于外力的高度失败，无接受或账本权限。

修复范围限定为共享 exploration 从完整同父、同 Genesis 已评记录提取既有 verifier key，由 generation_lanes 接入作者循环已有历史比较门，删除重复的活跃 exact 比较路径；负样本集合原有身份绑定独立保留，比较回执绑定实际合并集合。修复帧识别现有 historical_verifier_equivalent_repeat 分类并保留已公开的比较源码。科学协议、机制身份、任务投影及机器人逻辑均不改。

重启前复核 FOLLOWUP：F9 是有实证反例的候选解释问题，没有进入成功或接受判定，不增加解释证明系统；F10/F11 既有修复保留并检查新历史门反馈组合；F12 已有冻结的关闭 thinking 自动回退且实际用量保留，当前不调整模型或预算以改变结果。完成真实 A07 正反案例重放、必要回归和独立复核后，另开 A08 完整五代，继续复用 A03 reference 原训练。

## A08 准入修复验收、冻结与启动

`attempt_07/historical_admission_fix_audit.json` 用真实 A07 源码进入生产作者循环：Gen2B 第一次允许，已评后原样/改名/权重变体均拒绝，Gen3A 新机制仍允许；身份算法三文件与 A07 完全相同。`attempt_08/source_boundary_review.json` 独立核对完整历史范围、负样本原 lane 身份、实际合并集合回执及直接/嵌套修复提示；没有新增问题。F13 已记录修复，尚不把离线通过当作新会话完成。

执行提交 `f13ceb6660e01e5367f3a0961fca9a269737bf2c`，相对 A07 仅4产品文件、3测试文件及4个真实候选小夹具变化。目标 Python 在冻结快照执行 **119 项相关单元/集成测试，83.102秒，0 failure/error/skip，exit0**，见 `target_restart_tests_08.json/log`。准备合同 `3cb7d137681fd8272726a7449f27f3a66d111be157517d6214c7e3cec2d5f3811`；启动前服务器373执行文件与快照一致，A03四来源的size/mtime/SHA不变，历史1357保护文件size/mtime不变，两GPU无计算进程。

新会话 `go2-push-5rounds-20260923-a08` supervisor 为1489543，原任务/机器人/科学协议/seed/provider/预算/接受门不变。新的三相reference资格及五代候选依次自动运行；不导入A07候选或其head，仍只复用A03完成的reference训练。训练等待时同步检查实际提示、源码、阶段决定与任务/本体归属。


A08 首次调用在首个 Genesis/作业前被入口拒绝：prepare-only 已创建目录，运维启动却未带 `--resume-existing`。这是本次启动调用错误，不是产品缺陷；失败的 argv/process/stdout/stderr 保留在 `attempt_08/prelaunch_prepared_directory_failure/`。确认没有 Genesis、ledger 或作业 attempt 后，实际 launch 加上准备目录恢复参数，从同一已冻结 foundation 开始首次运行；没有导入旧实验或拼接代数。辅助状态脚本中遗漏的旧 A07 文件名也已修正，之前输出的两代状态属于 A07，不作 A08 进度。

## A08 reference 资格通过

实际运行 supervisor1490349、宿主1490350。`attempt_08/reference_audit.json` 的63项检查通过：复用A03原训练，本次新的dev141/192、accept142/192、sealed343/448与原任务证据一致（仅运行标识与时间不同）。reference实际0次训练、13eval seed、252096control steps、0.0547427458 GPU容量小时；A03四来源bytes/hash/mtime仍不变。A08新的Genesis为`0d1d2442cc53d661a9b7aae07ed0091c6450f7529b8ccdd1d3406220af21d4bb`。宿主自动进入第一代作者阶段。

A07 Gen1B视觉诊断独立完成，见`attempt_07/diagnostics/gen01_lane_b_native/native_replay_audit.json`：192个episode除时间字段外与原数据逐字段相等；3段视频实际解码均11帧，查看的第一段第0/5/10帧与PNG像素hash相同，观察到站姿下降、终帧腿折叠机腹低，未见侧翻。视觉只覆盖该代表样本，不证明全部episode画面或训练因果。诊断30.475687秒与2112env-control steps独立记录，没有修改正式账本或决定；A07仍NO-GO。

## A08 第一代闭合并接受 A

两条训练各完成500 iterations、49,152,000 env-steps，初始化配对、源码、激活与同刻峰值算术均通过。A dev169/192、B dev163/192，系统选A；A accept159/192、sealed394/448，accept相对reference142/192的delta=0.0885417超过冻结阈值0.05，正式接受。`attempt_08/gen01_decision_audit.json` 对44份证据独立重算通过；训练/物理证据见两份`through_gen01`记录，未发现注入异常或失败分母丢失。峰值A=-36.6024、B=-388.1665，不据此推导任务效果原因。

两lane均首次完成作者阶段，实际提示48301/48994 B、记忆各996 B，各一份1846 B reference源码，source request为0；B仅看到受限兄弟机制签名，未见A源码/推理/未来成绩。共享框架审计确认完整历史索引独立于LLM记忆裁剪，本次4文件无任务/本体语义，见`prompt_audit_gen01.json`、`gen01_candidate_source_audit.json`及`harness_boundary_audit.json`。F9新增A声明过强的有限数学反例，不改变实验合同。系统自动进入第二代；当前仅1代完成，不宣称5轮闭环或矩阵泛化已通过。

## A08 第二代闭合并拒绝 B

A/B均完成500 iterations、49,152,000 env-steps，dev为104/192、106/192；选中B的accept为99/192，相对父A的159/192下降60/192，按冻结noise-aware门提前拒绝，sealed未运行。`attempt_08/gen02_decision_audit.json` 对46份证据独立重算通过，head仍为Gen1A，系统自动进入第三代。训练输入来源、初始化、奖励激活和分项峰值均通过；失败/外力算术由`through_gen02`物理报告记录，不以候选机制解释推断性能下降的因果。

本代A一次通过；B首次32768输出单位全部用于reasoning而无正文，自动关闭thinking后第二次通过，真实成本保留，补充F12证据。三张实际prompt63418/64123/65751 B，memory各3465 B，父A169/192与未接受Gen1B163/192及失败明细保持独立；B重试保留必要两源、反馈和记忆，未额外索取源码。当前父已从reference改为Gen1A，所以本代同父已评集合为空符合范围，不等于丢弃首代历史；下一代将有Gen2两条同父已评候选可供实际集合接线核对。

## A08 第三代闭合并拒绝 B

A/B均完整训练，dev为0/192、147/192；选中B的accept为152/192，仍低于父Gen1A的159/192，提前拒绝且不跑sealed。`attempt_08/gen03_decision_audit.json` 对51份证据重算通过，head保持，自动进入第四代。A的192条episode均第11步（0.22秒）高度不足，最低0.16527696–0.16913377米、阈值0.18米；第一次推搡计划第51步（1.02秒），所以没有实际force event不构成漏注入，完整192分母保留。此时序与A07原生视觉已核实的早败类型一致，不推断相同训练原因，也不把旧视觉冒充A08新视频。

Gen3两lane的实际历史比较都使用Gen2两条同父已评family，evaluated与union hash与完整账本重建一致，未内联的Gen2A同样参与；negative空集合原身份独立。提示/源码双路审计通过，F13已补真实非空历史接线证据。Gen3B先输出预算耗尽、后自动回退通过，原费用保持。

F9补充候选解释/设计错误：A声称名义静止处零贡献，实际新项为-0.0525；B所谓damping负输出乘负权重成为正项。B训练同刻峰值记录该项+1.0，最后采样weighted mean=+0.010094，第四代实际提示原样提供其权重、贡献和来源，未被名称改写为惩罚。`training_evidence_through_gen03.json` 与 `prompt_audit_gen04.json` 已对账；候选错误由固定评测裁决，没有人工修候选或加符号启发式闸门。

## A08 第四代闭合并接受 B

两条训练各完成500 iterations，dev为A161/192、B175/192。选中B的accept174/192，相对父Gen1A159/192的delta=0.078125超过冻结阈值0.05；sealed409/448、全部证据门通过，正式接受B。`attempt_08/gen04_decision_audit.json` 对60份证据独立重算通过，新head为`push_recovery_lateral_brace_v1`、checkpoint `sha256:5714a1f0f73ffee68bc980fe28439fc48eb70f31320c058a4a75f3a9b868118f`。系统自动进入第五代，没有人工代写候选或更换评测规则。

两lane均在首个thinking请求耗尽后自动关闭thinking、第二次完成。`prompt_audit_gen04.json` 与 `gen04_candidate_source_audit.json` 对完整四条同父已评机制、实际合并集合、兄弟排除、来源及修复帧逐项对账；原三代六trial记忆与父/最新反馈保留。两条虽同名lateral_brace，但实际机制不同。候选新增项符号/死区解释错误仍归F9；不以rationale决定真实策略效果，也不改准入或接受门。第四代闭合时累计8次候选训练、63 eval seed、12次provider尝试、430296总单位和1.856287536 GPU容量小时，均在冻结上限内。

## A08 第五代闭合、作者审计完成

第5代A/B均完成500 iterations，dev172/192、169/192；系统选A，accept159/192低于父Gen4B174/192，按冻结noise-aware门提前拒绝且不运行sealed。`attempt_08/gen05_decision_audit.json` 对62份证据重算通过；五代共2接受、3正式拒绝，最终head保持Gen4B。系统自动开始两项最终确认，当前阶段不把五代闭合提前写成终局验收完成。

`gen05_candidate_source_audit.json` 完成全部10个最终候选的源码与准入审计：单组件修改、保留项、兄弟机制和公共输入范围通过；同父已评family依代为0/0/2/4/0，换父时清空的是比较域而不是记忆。`prompt_audit_gen05.json` 汇总15个实际请求、10个verified、5次reasoning耗尽后自动disabled fallback成功、0次源码补取。15prompt为48301–72684 B，memory逐代996/3465/3948/4409/5624 B，task section16096–20907 B；40个不变原始文件及所有已观察私有标识边界复核通过，源码未重复内联。真实provider总用量551792（request258805+response292987；reasoning277999含在输出内），五次失败的253423总单位保留计账，不抹除。

A08实际历史门在Gen3/4共四次使用完整非空集合且distinct通过；本次没有自然历史重复拒绝或历史比较修复帧。F13的重复拒绝、F11的直接/嵌套历史修复仍以已有实际历史重放及离线边界测试为对应证据，不能冒充本轮自然触发。F9中的候选符号/数学解释错误和F12费用问题继续单独记录，没有为制造更好结果而改候选、provider条件或验收门。

## A08 终局验收 GO

主进程完成全部五代及自动最终确认，exit0；另从冻结生产入口执行只读terminal audit，账本、成本、final回执和资源闭合四项PASS。`attempt_08/final_session_audit.json` 汇总29份审计/终态证据：2接受、3正式拒绝、10次完整候选训练、86 eval seed、32已完成作业、15次作者调用，无开放作业或未知费用。实际训练491520000 env-steps，总模拟control steps493131648，GPU容量2.339795370小时；观察产物590606116 B，原冻结上限均通过。

最终同sealed对照为reference343/448、Gen4B409/448，差66/448=14.7321429个百分点。`physical_evidence_final.json` 共22份summary通过记录算术与身份核验；新增两组各448条episode与各自原sealed完全一致，仅task evidence的运行ID与创建时间不同。十条训练原证据SHA未变，未追加训练；同样本重放不当作新的泛化数据。中文完整结论与限制见 `attempt_08/ACCEPTANCE_ZH.md`，交叉复核已收紧换父范围的表述，没有由本格外推2×2矩阵或因果结论。

收尾工具在生产terminal replay和10个候选checkpoint实际SHA核对通过之后，因原始source_snapshot_08.json只在本地而未在服务器落盘中断。同步相同原始清单后只补做保护核验，没有重跑训练/评测或再次执行terminal replay。最终373执行文件SHA不变，A03四来源size/mtime/SHA不变，历史1357文件size/mtime不变（未重新哈希历史内容），GPU无计算进程。没有删除checkpoint，旧A06/A07的NO-GO保持。

F9候选解释错误与F12无正文推理开销保持非阻断记录，实际反馈符号、失败费用和证据来源完整；当前不引入证明解释正确的额外框架，也不为好看结果改验收条件。

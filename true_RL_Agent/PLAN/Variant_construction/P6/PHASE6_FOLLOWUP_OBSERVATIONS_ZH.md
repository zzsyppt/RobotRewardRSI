# Phase 6 运行观察清单与历史处置日志

> 本文在 Phase 6 启动前建立，并在 pilot/validation 期间追加运行证据及已关闭 blocker 的历史处置；它不是
> 终态可提升项清单。终态非阻断问题只收录于 `PHASE6_FOLLOWUP_IMPROVEMENTS_ZH.md`。这些观察不是新的
> 入口验收门；若运行证据未显示实际问题，不追加实现。

## O1：真实 prompt 容量与降级层级

**当前证据**

- G1-V2 / Go2-V2 的高信息量 minimum-core stress 形状分别为 `36,590 B` / `35,944 B`，超过任务段目标 `32,256 B`。
- 有界降级为单条 latest-evaluated feedback 后分别为 `28,983 B` / `28,337 B`；因此这两个余量是降级后的余量，不能解释为完整双锚点信息仍有余量。
- Phase 4 第三代真实 payload 的任务段为 `31,433 B`，但它只含旧版 6 字段 trial comparison，不能严格代表当前 Phase 6 形状。

**Phase 6 观测**

- Go2 pilot 第一代两条 lane 的真实 user prompt 分别为 `69,571 B` / `70,467 B`；任务区均为
  `31,875/32,768 B`，`headroom=893 B`。实际投影是 `feedback core`，不是 `minimum_core`；一条
  current-best feedback、8 个 strata、3 个 guards、family score 和逐层失败/诊断信息均保留。
- 对这份真实任务区做纯内存 minimum-core 投影后为 `23,197 B`，39 个 reward ABI fields、
  8 个 strata、3 个 guards、current-best 和诊断表仍在。这证明当前降级策略不会立即变成
  “只剩全局标量”，但不能代替三代 validation 的真实证据。
- 当前 capacity preflight 的 synthetic comparison 使用了真实 ledger 不会产生的交叉字段：
  `selected=true/selector_status=selected` 与 `trial_status=qualified_not_selected` 同时出现，且比真实
  unified entry 多一个 `guard_score`。它只用于字节 stress，额外字段是保守高估，不会低估容量；
  但 synthetic/real 的交叉字段还没有共用同一 validator。
- 在每行第一代实际 provider 请求前后记录 `provider_task_sections.serialized_bytes`、`headroom_bytes`、`public_feedback_summary.projection_level`、保留/省略的 feedback entry 数。
- 若真实请求进入 `minimum_core_single_entry`，确认 latest-evaluated 的 family/stratum delta、selector 判据和父代标量差仍在；只把信息损失记入实验解释，不临时放宽 32 KiB 边界。
- 只有真实 payload 仍频繁逼近硬上限或关键反馈被删时，才另行评估去重；不得凭 stress 数提前扩大预算。

## O2：稳定前缀是否产生实际缓存收益

**当前证据**

- `exploration_directive` 已移到 lane 专属 `required_candidate_package` 之前，稳定段不再因 lane 包提前分叉。
- 静态字节相同只能证明可缓存前提，不能证明 provider 实际命中缓存。

**Phase 6 观测**

- Go2 pilot 第一代 lane-a/lane-b 的 19 个顶层段中有 17 个完全相同，原始公共前缀为
  `62,950 B`（lane-a `90.5%`，lane-b `89.3%`）；lane-a cache hit 为 `0`，lane-b 仅
  `10,496/18,424` request units 命中。这证明前缀已稳定，但 provider 收益仍需跨代观察。
- 对同代 lane-a/lane-b 以及同一 lane 的 retry 记录 provider 返回的 cache-hit / cache-miss units 与请求字节。
- 若稳定前缀相同但持续无缓存收益，只记录 provider 行为；不为缓存再造第二套 prompt。

## O3：advisory retry 与 incomplete generation 的实际发生率

**当前证据**

- advisory 最多消费一次已有 provider attempt；理论上会缩小后续 authoring 余量，但目前没有 Phase 6 真实发生率。
- 合法的 `paired_authoring_incomplete`、`candidate_execution_incomplete` 和缺失 selection 已修为“本代无 selector 反馈”，不会再使下一代 prompt 构建崩溃。

**Phase 6 观测**

- Go2 pilot 第一代两条 lane 目前均只消耗 1 次 provider attempt，且 provider/verifier 均 pass；
  尚未出现 advisory retry 或 incomplete，因此还不能用它判断三代发生率。
- 静态回放发现，`paired_authoring_incomplete`、`candidate_execution_incomplete` 与 `no_selection`
  在通用 `public_history.recent_outcomes` 中都投影成同一个
  `generation_head_retained/runner_did_not_accept_candidate`。`no_selection` 还可从 selector comparison 取回具体原因；
  `selection=None` 的 incomplete 则只在已有可评价候选时保留机制 card。它不影响恢复，但可能使下一代不知道
  上一代是创作未完成还是执行未完成。
- 记录每代每 lane 的 provider attempt 数、advisory retry 数、最终 generation outcome 和剩余 attempt。
- 只有 incomplete 明确由 advisory 重试耗尽造成且反复出现时，才重新评估 attempt 分配；不得先扩预算。

## O4：顶层支撑力 `10 N` 证据门是否产生形态偏差

**当前证据**

- `minimum_support_effector_upward_force_n=10.0` 是“支撑端确实压在顶层”的证据阈值，不是机器人承重目标；当前 G1-V2 与 Go2-V2 reference 均通过资格化，尚无误判证据。
- 当前 RobotProfile / Phase 0 运行事实没有权威质量字段，立即按质量缩放会引入新的二源真相并改变冻结评测身份。

**Phase 6 观测**

- 对首次合法到顶附近的 episode 记录每个 support effector 的向上接触力与 top-support 判定，核对是否出现明显着顶却因 `10 N` 被拒、或轻触即被误认的案例。
- 只有出现可复现误判，并先补齐机器人质量或等价的权威归一化事实后，才设计缩放规则；本轮不修改阈值。

## O5：Go2 诊断 flag 的判别力

**当前证据**

- Go2 public_accept 360 局：`slip_detected` `342/360=0.950`，逐 stratum `0.84~1.00`；`insufficient_clearance_detected` `0`；`off_course_detected` `2`；`first_valid_top_arrival_achieved` `310` 恰等于成功数，`goal_not_reached_detected` `50` 恰为其补集。
- 同一套语义下 G1 为 `slip 117/360=0.325`（逐 stratum `0.16~0.56`）、`insufficient_clearance 28`，具备判别力。
- 因此 Go2 的六列诊断中，两列饱和、一列恒零、两列与 success rate 冗余，只有 `base_orientation_instability_detected`（`25/360`，集中在 `joint_ood` / `tread_ood`）携带增量信息。

**Phase 6 观测**

- 每代记录候选的六列逐 stratum 比率；确认 `slip` 是否持续全表 ≥0.9、`insufficient_clearance` 是否持续为零。
- 这些列同时占用 minimum-core 字节，与 O1 一起评估。

**边界**：本轮不改冻结的 signal 阈值与 flag 集合；确认无判别力后只在新协议版本调整投影，不在 Phase 6 中途改。

## O6：只缩放了长度，力/速度/时间按 G1 绝对值继承

**当前证据**

- 已按 `0.589087` 缩放：riser 包络、C1 riser offset、C2 amplitude、`insufficient_clearance` 阈值 `0.03→0.018`、`support_effector_surface_clearance_range_m` `[0,0.12]→[0,0.071]`。
- 未缩放且与 G1 逐字节相同：`forbidden_contact.force_threshold_n=1.0`、`minimum_support_effector_upward_force_n=10.0`、评测指令 `0.8 m/s`、`episode_horizon_s=12.0`、`base_orientation_instability` 阈值 `0.65 rad`、`slip` 参数。
- 角度无量纲，不按长度缩放是对的。Froude 相似只给出速度尺度 `sqrt(gL)` 与时间尺度 `sqrt(L/g)`；力阈值
  还取决于质量、重力和接触定义，不能由长度比例单独推出。Go2 的 Froude 数 `0.393` 比 G1 的 `0.302`
  高约 `30%`，只证明相同 `0.8 m/s` 并非严格动力学等价，不证明某个唯一缩放值。

**Phase 6 观测**

- 记录超时率、首次到顶附近的实际接触力分布、失稳触发时的真实 roll/pitch，判断这些绝对量对 Go2 是过松还是过紧。

**边界**：任何一项的改动都会改变冻结评测身份；本轮只观测，与 O4 合并处理。

## O7：`forbidden_contact` 只绑 `base` 是否适合四足

**当前证据**

- Go2 RobotProfile 提供 15 个候选（`base` / 头 2 / hip 4 / thigh 4 / calf 4），binding 只选 `base`；G1 profile 本身只有 `torso_link` 一个候选，所以两行“只绑一个”并非同一种设计选择。
- Go2 public_accept 的 44 次失败全部由这一条产生，其余 20 格零触发；它是 Go2 目前唯一的活失败通道。

**Phase 6 观测**

- 记录触发时实际接触的 body 与力值；同时观察 hip/thigh/calf 在正常上楼中是否频繁接触（若是，扩大禁碰集会误杀正常步态）。

**边界**：这是行级 binding 的设计抉择而非缺陷；改动需要新 binding 版本与重新资格化。

## O8：课程初始等级与升降阈值直接沿用 G1

**当前证据**

- `initial_max_level` `c0_regular=5` / `c1_irregular_flat=1`、`move_up=0.8` / `move_down=0.4` 与 G1 完全相同。
- 但 Go2 在同一绝对几何上的能力远高于 G1：public_dev 中两行题面逐字节相同的 `c0_id_01` / `c1_id_01`，Go2 均为 `1.000`，G1 为 `0.508` / `0.241`。因此同一初始等级对两行含义不同。

**Phase 6 观测**

- 记录 Go2 训练中 c0/c1 的 level 直方图随迭代的演化；若 c0 很早压满上限，课程对 Go2 已不再提供难度阶梯，只是恒定地形。

**边界**：训练分布与课程是冻结项，本轮只观测，不调初始等级。

## O9：LLM 拿不到 Go2 的四足几何事实

**当前证据**

- reward-input contract 只暴露形态中立的 support effector 向量（Go2 为 4 项），不含前后髋间距 `0.3868 m`，也不含前后腿配对关系。
- 两个代表题格的差异与“踏深和髋间距的相位关系”假设一致：`tread_ood_03` 的 riser 只有训练上限的
  `0.915`（分布内），`髋距/踏深=1.42` 时得分 `0.269`；`height_ood_02` 的 riser 为训练上限的
  `1.130`（分布外），该比值约 `0.97` 时仍为 `1.000`。题格还同时改变其他几何量，这组相关性不能单独
  证明相位是唯一原因。

**Phase 6 观测**

- 观察 LLM 是否提出与机身跨级、俯仰调节或步态相位相关的机制；即使始终未提出，也只能说明“输入不足”与
  “有限搜索未命中”都仍可能，缺少控制消融时不作单因归因。

**边界**：补输入必须走 VNext ABI 新版本且保持形态中立（不得写入 Go2 专有字段名）；本轮不做。

## O10：Go2 验收阈值量级与可达性

**当前证据**

- Go2 `J_primary=0.933915`，阈值 `0.2×(1−J)=0.013217`；五 seed 的 `J_primary` 标准误为 `0.001565`，只有阈值的 1/8，因此瓶颈是改进幅度而非噪声。
- 24 格中 20 格满分；四个活格单格修满的收益为 `+0.0111`（`height_ood_03`）/ `+0.0153`（`joint_ood_01`）/ `+0.0181`（`tread_ood_03`）/ `+0.0216`（`joint_ood_03`），后三者任一做满即过线。
- 阈值恒为剩余余量的 `20%`，故换更弱 reference、剔除饱和 family 或整体缩放几何都不改变通过条件，只改变数字表观。

**Phase 6 观测**

- 记录三代内是否出现 `≥0.013217` 的候选，以及改进是否集中在那四格。
- 若三代都低于阈值且失败机制相同，才把“目标缺乏分辨力”当作已证事实，再评估 OOD 阶梯是否需要重心校准。

**边界**：不改套卷、reference、冻结 guard 或 `headroom_reduction_fraction`。

## O11：导入 reference 后仍保留历史 reference 的保守预算槽

**当前证据**

- P6 fresh session 只读导入已资格化的 P5 reference，目标 session 的 `post-genesis`
  `train_jobs/train_steps/eval_seed_units` 都应为零；pilot 已证明实际没有 reference job。
- 首次三代 validation `p6-go2-validation3-ea089cc` 暴露了真实阻断：旧 V2 operation table 虽不执行
  reference 工作，却仍把 P6B 历史 lane-train reserve 当成准入 timeout。lane-a/lane-b 因此分别在
  `4,593 s / 4,653 s` 被终止，只运行到约 `1399/1372` iteration，generation 尚未形成。
- 这证明 phantom topology 与历史耗时不只是保守记账，而会阻止合法 candidate train，满足本观察项原先的
  升级条件。

**Phase 6 观测**

- 新增 imported-reference validation cost V4，仅在 operator 明确导入既有 reference Genesis 时启用；
  非导入 validation 与历史 V1/V2/V3 的 loader/replay 语义不变。
- V4 删除 `reference_train`、三相 `reference_qualification`、`parent_public_dev` 和
  `reference_train_retry` 槽；成功工作冻结为 6 次 candidate train、69 eval seed-units，retry 只保留
  7 seed-units 的 sealed retry。
- 六次 candidate train 均按 `7,200 s + 16 s cleanup` 进入 operation plan；成功/重试/总 GPU cap
  分别为 `13.6 / 0.2 / 13.8 h`。训练准入返回 `None`，让 SubprocessRunner 的协议 timeout 成为唯一
  进程时限；评测仍按 operation reserve 限时。
- 失败 session 保持 immutable，两个 `model_1350.pt` 只作为基础设施诊断，不进入 GO 证据；修复后启动
  fresh successor validation，不续训也不伪造旧 generation closure。

**状态**：已由观察项升级为 blocker，并按原定边界通过新 cost-contract 版本收敛。它只改变 runner-owned
执行成本/准入语义，不改变训练、地形、PPO、评测、acceptance 或 reward 权限。

## O12：候选 reward 的行内形态假设是否造成反向行为

**当前证据**

- lane-a `stair_swing_lift_v1` 通过的是 Go2 cell-local reward 候选，其 `support_contact.sum() < 3.5`
  显式依赖四支撑端数量。它没有进入共享层，因此不违反本轮“每格 reward 可独立改进”
  的权限边界，也不是当前 pilot 停机理由。
- 该项用 `max(support_z)-mean(support_z)` 作摆腿代理，没有 airborne mask 或局部地形高度；
  `moving` 只取决于命令速度。因此存在长时间抬高单腿也持续得分的可能，但目前只是机制推断。
- lane-b 的 pitch/roll deadband 是形态中立标量机制，但对 nose-up/nose-down 对称，且 clip
  将严重倾倒的额外惩罚限制为 `-2.5`；它是可验证假设，不是已证结论。
- pilot 的 public_dev 实测为：lane-a `J_primary=0.917064`，lane-b `0.942763`，reference
  `0.934438`；selector 按冻结规则选择 lane-b。lane-a 的单腿摆高代理没有取得选择优势，因此本轮没有
  证据支持为它增加通用防御规则。
- lane-b 正式 candidate score 为 `0.943657`，reference 为 `0.933915`，提升 `0.009742`；
  三条 named guards 全部通过，但未达到冻结 materiality threshold `0.013217`，因此 formal decision
  合法为 `reject/named_guards_paired_headroom_delta_not_met`。这只能说明该候选有小幅正信号但未达到接受标准，
  不能反推 deadband 是因果最优机制。

**Phase 6 观测**

- 只用权威 eval/guards 和必要的可视化判断是否出现单腿高举、极端俯仰或通过牺牲 ID
  换 OOD 的反向行为；训练 reward/progress 曲线不能代替该判断。
- 若候选未被选中或评测已被守门拒绝，不为这些假设增加特制规则。
- 最终 train summary 只能证明候选 delta term 的 `max_abs_weighted_contribution` 是否非零。当前统计会包含
  第一次 preflight transition，且不记 nonzero 频率；因此本轮最多报告“runtime gate observed”，
  不将单一 max 写成 PPO 期间的因果改进证据。
- pilot 中两条 candidate 均有 `dispatcher_calls=48,001`、`runtime_status=active_pre_reset_complete_reward_v1`
  且 runtime-effect gate pass；lane-a `swing_foot_lift` 的末次 weighted mean 为 `0.209809`、最大绝对贡献
  `0.5`，lane-b `climb_orientation_l2` 分别为 `-0.011774` / `2.5`。这些数据证明接线和运行时出现过贡献，
  仍不提供贡献频率或因果归因。
- 若后续需要将“PPO 分布内持续生效”升为 promotion 证据，优先在 preflight 后清零统计，并增加
  per-term `nonzero_sample_count/total_sample_count/fraction`；不用这个观察项追溯改写当前冻结 pilot。

**边界**：reward program 是 cell-local 可进化物；共享 harness 不要因单个候选的形态假设增加 Go2 分支。

## O13：真实训练时长与 runner timeout 余量

**当前证据**

- Go2 pilot 两条完整训练实际耗时约 `6,456 s / 6,630 s`，均在协议 `7,200 s` 内闭合。
- 首次 validation 的两个超时并非训练超过 `7,200 s`：V2 admission hook 将历史 P6B 成本 reserve
  转成了 `4,593 s / 4,653 s` 的隐藏第二 timeout。两条训练在终止前均无 Traceback/OOM/NaN，迭代连续。

**Phase 6 观测**

- 已从源头分离“成本准入”和“进程 timeout”：训练成本按 `7,216 s` lease ceiling 预留，真正的进程
  timeout 仍为协议 `7,200 s`；不调训练预算、不调 iteration、不扩大 retry。
- fresh successor 逐 lane 核对 operation reservation 中的 `7,216,000,000,000 ns` 与子进程实际
  timeout `7,200 s`。若此后训练确实超过协议时限，再形成新的运行证据；不得把本次隐藏 timeout
  误记成 Go2 算力或训练算法失败。

**状态**：隐藏第二 timeout blocker 已修复，协议 timeout 未改变。fresh successor 的六次 candidate train
均完成 `2,000` iterations，固定三代闭合为 `accepted / no_selection / no_selection`，因此该运行阻断已由
服务器实跑关闭。

## O14：历史资源与 provider usage 的事后可复算边界

**当前证据**

- pilot 的 provider receipt 可逐项求和为 request/reasoning/response/total units
  `36,596 / 29,446 / 31,770 / 68,366`，但这些计数仍由 provider response 自报；不重新发 API、也没有
  独立账单时，验收器只能验证 receipt 与成本链一致，不能独立证明 provider 计量真实性。
- 7 个 attempt result 的 lease duration 可求和为 `14,179,867,235,778 ns = 3.938852 h`，但底层时长仍来自
  runner monotonic timer。startup/final compute-idle、PID→GPU UUID 和 simultaneous lease reacquire 同样是运行时
  时间点观测；终态 `nvidia-smi`/PID/lock 清洁只能佐证最后状态，不能重建启动瞬间。
- settlement 记录的 `observed_output_bytes/files=589,284,493/182` 是结算时快照。终态后只新增 settlement、
  segment exit 与 terminal report 三个 create-once 文件，共 `15,754 B / 3 files`；按产品相同的普通文件口径，
  当前树 `589,300,247 B / 185 files` 减去三者可精确还原历史快照。`du -sb` 会计入目录元数据，不能用于该门。
  `accounted_output_bytes=590,333,069` 仍等于 historical observed + `1 MiB` reserve。

**Phase 6 处理**

- `verify_phase6.py` 将 provider usage与历史 GPU/PID/lease observation 明确列为
  `attested_not_recomputed`；它们仍需 typed schema、receipt hash chain 和内部一致性通过，但不声称独立重测。
- checkpoint 实体、train/eval summaries、attempt closure、ledger/decision/current-best、成本算术、历史 output snapshot
  的 create-once 尾文件回退，以及当前资源清洁继续作为可独立复算硬门。

**边界**：不为三代实验新增录像式 GPU 审计、provider 账单抓取或周期性资源快照系统。只有这些 attestation
后来出现真实争议，才单独设计 create-once observation receipt。

## O15：同轮 component-smoke 修复反馈可只投影失败项

**当前证据**

- validation generation-001 lane-b 第二次候选通过静态校验，但 component smoke 发现
  `forbidden_contact_force_l2` 返回 `[5,3]`，而 reward term ABI 需要 `[5]`；第三次请求收到精确失败项与失败源码后通过。
- 第三次请求的 repair section 为 `4,681 B`，其中完整 `component_smoke_result` 为 `3,421 B`；
  8 条已通过 reward rows 占 `2,322 B`，唯一失败 row 约 `230 B`。
- 按真实对象只保留失败 row、`passed_reward_term_count=8`、fixture/check 汇总后，repair section 可由
  `4,681 B` 降至约 `2,044 B`，节省 `2,637 B`（`56.3%`），且不删除本轮可操作的失败证据。

**后续处理**

- 本轮第三次修复已经成功，prompt 总量 `66,822 B`、任务段 `22,746/32,768 B`，因此不是 Phase 6 阻断项，
  不在冻结实验中改投影协议。
- 后续若同轮 smoke repair 频繁发生或任务段再次逼近上限，可把通过项压成计数、仅展开失败 rows；
  仍由 task-owned smoke 结果提供事实，共享层只做有界投影，不另造第二套失败判定。

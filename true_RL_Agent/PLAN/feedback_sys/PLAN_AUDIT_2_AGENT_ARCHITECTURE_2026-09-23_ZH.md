# 反馈系统升级计划第二次审计：粒度、行为描述、多模态与面向 RL 的 agent 架构

日期：2026-09-23。对象：[FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md](FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md) v1.1 第 3 章目标架构与 Phase 2/3/6/7/8。
前一份审计 [PLAN_AUDIT_2026-09-22_ZH.md](PLAN_AUDIT_2026-09-22_ZH.md) 已处理顺序、统计功效、provider 事实与上下文预算，本文不重复，只回答四个问题：

1. 当前反馈粒度到底粗在哪，计划补到了哪一层。
2. "高层动作描述"应该由谁产生、长什么样、计划里有没有。
3. 多模态输入应该先喂什么、原生视频的真实价值在哪。
4. 整个系统是否在向"面向 RL 的 agent 架构"发展，计划缺哪一块。

依据：当前工作树源码、15 代正式结果、G13 提示词重建、本轮联网核实的研究与工程（第 7 节）。本文是审查意见，没有改代码、没有启动实验。

## 0. 结论

1. **粒度判断成立，粗在"折叠"而不在"没采"。** 作者 LLM 现在看到的是 cell 级计数、终局标签、部分条目的诊断标志率、进度统计、奖励项 last/max。正式循环的 public-dev/accept summary 其实已经持久化了逐回合事实（当前 v3 协议绑定的套卷就是校准套卷，`records_episode_facts=True`），逐步数据也在内存里（Stair 每步约 36 个标量，含逐脚接触、所在台阶层、净空、平面速度、支撑力），只是 tracker 用 streak latch 把逐步折叠成了 6 个布尔，投影又把逐回合折叠成了比率。Push 更粗：运行时只观测 base 位姿，逐脚接触根本没采，"迈步恢复"这类描述在 Push 上目前无数据可算。
2. **计划补的是 L0（原始时序）和 L1（规则事件），L2（行为描述）只有半句话。** 架构图里的节点 D"任务侧行为与阶段描述"和 Phase 3 的"阶段—部位—动作—方向—时长—频率"是对的方向，但没有词表、没有逐回合 transcript 对象、没有 candidate 对 parent 的行为差分对象、没有"这条描述能用哪个授权输入去改"的标注。MotionSegment 同时出现在 Phase 3（确定性）和 Phase 7（VLM 输出），来源类型有区分，但谁是主谁是补没有写死。
3. **高层动作描述应当确定性优先、VLM 补充，而不是反过来。** 现有字段足以在 Stair 上算出逐脚摆动/触地/落点相对台阶棱、绊擦、滑、停滞、base 撞第几级、前后倾方向、步态相位与不对称；在 Push 上补上逐脚接触和逐步稳定子条件后，能算出 ankle/hip/stepping 策略类别、恢复步数、dwell 被哪个条件打断。这一层同时是上下文的压缩器：一个 cell 的行为画像几十行，比 216 回合任何形式的原始数据都小。
4. **多模态第一步不该是原生帧。** 由测量数据渲染的诊断图（侧视足迹轨迹叠台阶剖面、四脚接触时序、姿态曲线并标事件）与事件同源、每回合都有、每张约 1k token、VLM 读图可靠；SDS 的闭环反馈正是"关键点网格 + 接触时序图 + 分项奖励统计 + 分数"这一组合。原生帧的价值在词表之外的现象和人工复查，应作为第二种媒体。计划并没有禁止派生图：Phase 6 的不通过项是"用合成示意图冒充原生观测"，Phase 7 允许分开标记的示意图。缺的是把由测量数据渲染的诊断图列为 Phase 6 的交付物、给它预算与选择规则、并作为分析器的默认输入；现在它只在 Phase 7 以"若使用"的姿态出现。
5. **系统在向 agent 架构走，但只走了感知这条腿。** 按 CoALA 的四维（观察、动作空间、记忆、决策过程）对照：观察在升级；动作空间只有 `complete_reward_program`，13 个 runner-owned 组件全部冻结；记忆是 session 内按 trial 的结果卡，revisit lane 有结构化的机制重访指令，但作者自己的假说只是源码里的一个字符串字面量，没有事前预测与事后核对；决策过程是每代一次生成加最多一次补查。Eureka→DrEureka→Eurekaverse→HARBOR 的演进恰好是动作空间从 reward 扩到 DR、环境课程、超参数。本次升级冻结动作空间是对的，但第 3 章必须把四个合同写出来，并在 L2 描述上标注"可由哪个组件解决"，否则反馈系统会绑死在 reward-only，下一次扩动作空间时还要重做一遍反馈。
6. **建议把计划第 3 章改写为 agent 架构合同，Phase 不动，交付重映射**：Phase 3 增加词表 v1、EpisodeTranscript、BehaviorProfile、BehaviorDiff、affordance 标注；Phase 5/8 把预测写成词表项并自动裁决；Phase 6 增加派生诊断图媒体类型；Phase 9 增加"预测命中率"和"假说可执行性"两项。前一份审计的薄纵切仍然成立，第一条 transcript 就做 Stair 的 forbidden_contact 回合。

## 1. 按 agent 四维给当前系统画像

| 维度 | 当前实现（源码事实） | 计划 v1.1 的改动 | 缺口 |
|---|---|---|---|
| 观察 | 提示词里：`public_feedback_summary` 的 cell 级 success / primary_reason 计数、部分条目 diagnostic_flag_rates、progress 统计；`runtime_feedback_summary` 的奖励项 last_mean / max_abs / nonzero 与同 transition 的 candidate−parent 差分；trial 比较的 objective_delta / guard_delta / runtime_effect | Phase 2 存时序，Phase 3 出规则事件与阶段，Phase 4 可查询，Phase 6/7 视觉 | 没有 L2 行为描述与行为差分；没有"事件→可用输入"的标注 |
| 动作 | `candidate_authority.mutable_components = [complete_reward_program]`；13 个 runner-owned 组件不可改；输出形状 `candidate_id / declared_changed_components / components_py`，假说是 `TrainingComponents.rationale` 字符串字面量；另有 `source_request` | 增加 `evidence_request`（只读） | 动作空间不变（本次合理），但架构未声明动作空间与扩展接口 |
| 记忆 | `ExplorationEvidenceCard`：mechanism_operation_delta、observed_effect（strong_positive/positive/mixed/neutral/negative/unknown）、value_label、novelty、failure_scope、revisit_disposition；revisit lane 另有 ≤2 KiB 的结构化 hypothesis 信封（等价键、操作、效果、组件类型、输入字段预览），它是索引给作者的指令，不是作者的事前预测；`experiment_memory` 行：objective/guard delta、failed_gates、component 激活、trial_outcomes；以 `session_genesis_hash` 为界；rationale 文本只随源码可见（契约要求它写明最接近的先前候选、代码级差异和如何应对观察到的失败，但这段文字不被解析、不进入记忆行） | Phase 5 HypothesisRecord、双时间、supersedes | 效果只有分数级，没有行为级；预测无法自动裁决 |
| 决策 | 每代两条 lane：至多一条 revisit、必保一条 novel（`freeze_exploration_lane_intents`）；`exploit_refine` 指令；一次生成、静态修复、源码补查；无跨代规划 | Phase 8 一次证据补查 | 没有"假说→预测→实验→裁决"的循环对象 |

对比先例（第 7 节核实）：

| 系统 | 观察 | 动作 | 记忆 / 搜索 |
|---|---|---|---|
| Eureka（ICLR 2024） | 中间 checkpoint 的分项奖励与任务适应度数值序列（reward reflection） | 奖励代码 | 每轮 16 样本、保留最好者，5 轮 |
| DrEureka（RSS 2024） | 同上 + 仿真物理可行域 | 奖励 + 域随机化参数 | 同上 |
| Eurekaverse（CoRL 2024） | 策略在各地形上的表现 | 环境/课程代码 | 课程逐代进化 |
| SDS（CoRL 2025） | 训练录像关键点网格 + 四脚接触时序图 + 分项奖励统计的自然语言摘要 + GPT-4o 三项评分 | 奖励代码 | 每轮 8 个候选，5 轮 |
| Video2Reward（2024） | 学得轨迹与参考视频关键点的 FastDTW 相似度分数表 | 奖励代码 | 迭代精炼 |
| RDA（2026） | 多 rollout 视频 + 子任务日志 → VLM 失败汇总 | 奖励 + 子任务定义 | 迭代 |
| HARBOR（2026） | 训练日志与评测产物；持久 artifact、可执行 gate | 奖励 + 算法超参 + 环境参数 | 并行 trial、跨 run 经验 |
| 本项目 | cell 级统计 + 奖励项 last/max | 仅完整奖励程序 | session 内结果卡；配对 seed、sealed、五道 gate |

本项目在验证严谨度上超过所有先例，在观察粒度和动作空间上最窄。先补观察是对的顺序，但架构合同要把另外三维写出来。

## 2. 粒度：原始 → 已保存 → 已展示

### 2.1 Stair

| 层 | 内容 | 状态 |
|---|---|---|
| 采集（`isaaclab_runtime_env.py` recorder） | 每控制步：base 位姿、joint_pos、逐支撑体位置/速度/接触力、逐禁触体力、终止掩码 | 在 GPU 上逐步发布 |
| 解码（`StairEnvTransition`，20 个字段） | progress_m、lateral_m、base_height_m、base_clearance_m、abs_roll/abs_pitch、forbidden_contact（布尔）、逐脚 contacts / progress / surface_levels / lateral / planar_speeds / surface_clearances / upward_forces、done、termination_terms | 全段 host_history 只在内存；joint_pos 与逐禁触体力仍留在 host_history 的原始步里，但 `StairEnvTransition` 不带、也不落盘；roll/pitch 去符号；forbidden_contact 只剩布尔；base 速度未存（可由 progress 差分近似） |
| 跟踪（`StairEpisodeTracker`） | 6 个 streak latch（slip 5 步、clearance 3 步、orientation 10 步、fall、off_course、forbidden）、首次到顶 | 丢失 onset、持续、发生在哪只脚、所在台阶层、与终局的时间距离 |
| 汇总（`build_ordered_stair_summary`） | 按 cell 的 primary 计数、diagnostic 计数；当前套卷即校准套卷，public-dev/accept summary 已含逐回合 episode_facts（primary、steps、progress、flags） | 逐回合事实已落盘但没有时序；投影只取比率 |
| 展示（`prompt_adapter.py`） | 上表的率与计数，经 56 KiB 裁剪链 | 作者只见比率 |

15 代结果里最有力的例证不变：主失败是 base 撞台阶（Go2 的禁触角色是 base），`insufficient_clearance` 诊断 618 次而 primary 仅 1 次。G13 提示词里的实验记忆显示，G1-a 和 G4-a 直接惩罚禁触力，G3、G6、G7 到 G12 则连续改 swing-foot 机制；两条路线都被拒或只接受一次，而没有任何证据说明脚净空事件是否先于 base 接触、发生在前脚还是后脚、成功回合是否同样出现。这不是"信息没采"，是"采了但在 tracker 处折叠"。补充一点：既然逐回合 flags 已在 summary 里，"forbidden_contact 终局的回合中有多少同时带 clearance/slip 标志，成功回合又有多少"这张回合级 2×2 表今天就能从现有产物算出，不需要重跑；时序关系才需要逐步数据。

### 2.2 Push

`PushRuntimeState._observe_pose` 只读 root 位姿与速度：stable 由 speed、angular speed、|roll|、|pitch|、height 五个阈值合取；failure 由 height、tilt、可选 drift 决定。逐回合记录（`episode_records`）保存扰动事件的控制步、请求/实测 Δv 或冲量、恢复步、censored 延迟、极值、终态标签。没有任何逐脚数据，也不保存逐步的五个稳定子条件。

后果：Push 的"高层动作描述"目前最多能说"扰动后 0.8 s 内俯仰达到 0.4 rad、1.6 s 后进入恢复带"，说不了"用了迈步还是髋策略、迈了几步、哪只脚先动、dwell 是被角速度还是倾角反复打断"。Phase 2 对 Push 的验收（"记录本协议真实扰动起止、实际执行、恢复各条件、dwell 与截止"）要明确加上逐脚接触与逐步子条件布尔量，否则 Phase 3 的"支撑调整"阶段无从计算。

### 2.3 训练过程

`runtime_summary` 每 `evidence_sample_interval_steps=24` 采样一次 last/max/nonzero，没有早/中/晚分段，没有分项贡献分布。Eureka 的 reward reflection 是分项在若干中间 checkpoint 上的序列，其消融显示去掉这一项会降低表现；SDS 把分项统计转成自然语言摘要。计划 Phase 3 已要求"训练早、中、晚阶段的采样覆盖、激活、贡献分布与饱和摘要"，这项与本文其他建议无冲突，保留。

## 3. 高层动作描述层（L2）应该是什么

### 3.1 定位

计划的三层是"原始记录 / 确定性事件 / 模型笔记"。审计意见是明确加一层：

| 层 | 产生者 | 对象 | 作者默认可见 |
|---|---|---|---|
| L0 原始时序 | recorder | NPZ + manifest | 否（可查） |
| L1 规则事件 | task projector | 事件表：类型、onset、持续、部位、台阶层、参考值 | 目录 |
| **L2 行为描述** | task projector（确定性模板） | EpisodeTranscript、BehaviorProfile（按 cell）、BehaviorDiff（candidate 对 parent、配对 seed） | **是** |
| L3 模型笔记 | Visual Analyst / 作者 | visual_observation、hypothesis、proposed_rule | 经 Claim Validator |

L2 的三个对象是缺失的关键。EpisodeTranscript 是一回合的时间线（"第 212 步左前脚离地；第 231 步左前脚净空 0.012 m 低于阈值持续 4 步，脚尖位于第 3 级棱前 0.03 m；第 236 步 base 前倾 0.41 rad；第 242 步 base 与第 3 级立面接触，终止 forbidden_contact"）。BehaviorProfile 是一个 cell 内成功组与各失败组的事件发生率、时序中位数、步态参数。BehaviorDiff 是同 seed 同 cell 下 candidate 相对 parent 的画像差分，这是 agent 判断"我的修改是否做到了我预期的事"的唯一直接证据，也是 HypothesisRecord 能被自动裁决的前提。

L2 同时解决上下文问题：BehaviorProfile 每 cell 约 30 行、BehaviorDiff 每 cell 约 15 行，24 个 cell 的画像与差分在 20 KB 内，比现在 `public_feedback_summary` 的 19.8 KB 信息密度高一个量级；transcript 按需查询，L0 只给绘图与重算。

### 3.2 词表 v1（建议交付物，不是最终定义）

Stair，全部可由现有 `StairEnvTransition` 字段加台阶几何（`FlatStairSurfaceQuery` 的 step_height、tread_depth）计算：

| 项 | 判据来源 | 备注 |
|---|---|---|
| swing / touchdown（逐脚） | support_contacts 的下降/上升沿 | 给出摆动时长、触地时所在 surface_level |
| touchdown_offset | 触地时 support_progress_m 相对该级棱的距离 | 落在棱上、棱前、踏面深处三类 |
| toe_catch | 净空低于阈值且脚水平速度骤降、progress 增长停滞 | 与现有 insufficient_clearance 区分：前者是撞棱的运动学证据 |
| scuff / slip | 接触且平面速度 > 阈值（现有 slip 判据）；scuff 为短于 5 步的同类事件 | 现有 slip 只在 5 步以上 latch |
| base_strike | forbidden_contact 首次为真时 base 所在台阶层与前后脚状态 | 逐禁触体力已在 host_history 原始步，需要进入 transition 或落盘 |
| stall | progress_m 在 N 步内净增 < ε | 与 timeout 的关系 |
| pitch_dive / pitch_back | 带符号 pitch 越过阈值的方向与时刻 | 需要恢复符号（一行改动） |
| lateral_drift | lateral_m 单调偏移 | 与 off_course 的时间关系 |
| gait_pattern | 四脚 duty factor 与相位差（Hildebrand 步态图变量） | 报告 trot / pace / 混合、左右与前后不对称 |
| top_arrival | 现有 first_valid_top_arrival | 不变 |

Push，需要 Phase 2 补采逐脚接触与逐步子条件：

| 项 | 判据来源 |
|---|---|
| response_strategy | 扰动后首个恢复窗内：有脚离地并落点前移 → stepping；无脚离地 → 非迈步。ankle 与 hip 的进一步区分需要关节角，Push 运行时目前不记录，Phase 2 需补采；概念来自文献中的 ankle / hip / stepping 三策略与 change-in-support |
| step_count / first_step_latency | 逐脚接触边沿 |
| dwell_break_reason | 逐步五个子条件中最先失守的那一个及次数 |
| overshoot | pitch 反向越过零点的幅度 |
| fall_direction | 现有 fall_forward / backward / side / height_collapse |

两张表都应先做 fixture 上的 oracle 一致性（计划 Phase 3 已有此门），再做 12–24 个真实回合的人工核对；词表不超过 15 项，扩展只通过 3.4 的机制。

### 3.3 描述的可执行性标注

每条 L2 描述附一个 `affordance` 字段：能否用 `StairRewardInputsVNext` 现有输入表达对应的塑形（例如 toe_catch ↔ support_surface_clearances / support_lin_vel / support_pos_relative_to_terrain_origin；base_strike ↔ forbidden_contact_force / base_height；gait 不对称 ↔ support_air_time / support_contact_time），或者标注"需要 runner-owned 组件（curriculum、terrain、DR）"。前者告诉作者往哪改，后者积累的是将来扩动作空间的证据。这个标注是"面向 RL 的 agent"与"更好的日志"之间的分水岭。

### 3.4 VLM 在 L2 上的角色

不是产生 transcript，而是三件事：报告词表外的现象（"后腿在踏面上交叉"）；对派生图与原生帧做一致性核对；提出候选新规则（Code-as-Monitor 的思路：模型给出可编译的判据，任务侧编译、在保存的 L0 上重算、与人工核对后才进入词表）。这样词表增长是有证据的，而不是靠模型描述堆出来的。

## 4. 多模态输入

### 4.1 先喂派生图，再喂原生帧

| 媒体 | 与事件同源 | 覆盖 | 单张成本 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|---|
| 派生诊断图（足迹侧视叠台阶剖面、四脚接触时序、姿态与净空曲线并标事件） | 完全同源（由 L0 渲染） | 每回合都可画 | 约 1k token/张（1280×720 按主流 provider 折算） | 时序关系、相对几何、多脚并行 | 视觉外观、未采集量 |
| 原生帧 / 网格合成帧 | 需 control-step 映射与回放一致性 | 每 seed 一个 env（`env_index`），其余 env 被隐藏 | 同上；网格 4×4 一张可载 16 帧 | 词表外现象、姿态是否"像样"、人工复查 | 距离、接触力、被遮挡关节 |

SDS 的做法值得直接借：训练录像先抽关键点，再排成时间网格，另附二值接触序列图，再加分项奖励统计的文字摘要，一起给 GPT-4o；Video2Reward 干脆不用 VLM 看视频，只把关键点轨迹的 DTW 相似度分数表给 LLM。两者都说明"结构化的运动表示"比"原始像素"更好用。计划 Phase 6 的"用合成示意图冒充原生观测"不通过项本身没有错，Phase 7 也已允许分开标记的示意图；要补的是在 Phase 6 交付物里正式加入派生图媒体类型：标 `derived_from` 与渲染版本、不得用于证明接触或距离、有自己的预算与选择规则，并作为 L0 的可视化成为分析器的默认输入。

### 4.2 原生录制的三个硬问题（补充前一份审计）

- 相机：`NativeVideoOptions.eye=(4.5,−4.5,3.0)`、`lookat=(0.8,0,0)` 相对机器人根，约 7 m 外的 3/4 俯视。1.8 cm 的净空阈值和脚尖与台阶棱的关系在 1280×720 下只有几个像素；判断脚级事件需要侧视近景或 ROI，计划 Phase 6 提到 ROI，但相机池要在 Phase 0 就定。
- 覆盖：一个 seed 只录一个 env，其他 env 在画面里隐藏。主失败回合大概率不在录制池；可行路径仍是"数值档案选回合 → 同 seed 诊断回放 → 与档案逐步核对一致再标同源"。
- 内存与 provider：`_frames` 无上限；默认作者模型是纯文本。Visual Analyst 作为独立 VLM 角色是对的，选型和每图计费必须在 Phase 0 实测冻结。

### 4.3 关于给 VLM 的数值

近期工作表明把机器人本体轨迹与图像一起给 VLM 能改善动作描述与子任务分割，计划 Phase 7 的"分析输入只含必要任务/身体映射、窗口、对照和数值"与此一致；具体应把 L1 事件表和 L2 transcript 一并给分析器，让它做"核对与补充"，不做"从零描述"。

## 5. 面向 RL 的 agent 架构：计划走到了哪一步

### 5.1 四个合同

建议把计划第 3 章改成显式合同，Phase 编号不变：

1. **观察合同**：L0–L3 四层，每层的产生者、对象、可见性、查询接口（Phase 2/3/4/6/7）。
2. **动作合同**：当前 `complete_reward_program` + `source_request` + `evidence_request`；列出 13 个 runner-owned 组件为"冻结的候选动作"，L2 的 affordance 标注是它们将来解冻的证据来源。本次不解冻，但接口写出来。
3. **记忆合同**：结果卡（已有）+ HypothesisRecord（Phase 5）+ BehaviorDiff 裁决（新）；session 内为主，跨 session 需 protocol/robot 相容门。
4. **决策合同**：每代"读画像与差分 → 检索相关假说 → 写假说（词表项 + 预测方向 + 授权机制） → 生成候选 → 训练评测 → 自动裁决预测 → 更新记忆"。作者一次生成、一次补查的预算不变。

### 5.2 假说必须能被自动裁决

Phase 8 的 sidecar 记"预测指标方向"，Phase 5 记"supported / refuted"，但计划没有说由谁、根据什么判定。有了 BehaviorDiff，裁决可以是确定性的：预测"height_ood 层 toe_catch 率下降"，实验后读同 cell 配对 seed 的 BehaviorDiff，给出 supported / refuted / insufficient（样本或覆盖不足）。这一步不需要模型，应写进 Phase 5 的记忆更新规则，并作为 Phase 9 的一项指标（预测命中率），以及 Phase 10/11 的次要终点。

### 5.3 与先例的差异要写清楚

HARBOR、Agent² 这类"agent 造 RL"框架把动作空间开得很大，验证很弱；本项目相反。计划 7.3 的"不做"清单是对的，但第 3 章应说明这是分期，不是终局：反馈系统的设计目标是"任何将来被授权的组件，其效果都能在同一 L2 画像上被观察和裁决"。否则将来解冻 curriculum 或 DR 时，会发现 L2 只描述了奖励能改的东西。

## 6. 对计划的具体修改（按章节）

| 位置 | 修改 |
|---|---|
| §3 架构图与 3.1 | 改为四个合同；组件表增加 Behavior Describer（task-owned，产出 L2 三对象）与 Prediction Adjudicator（共享层，只读 BehaviorDiff 与 HypothesisRecord） |
| §3.2 对象 | 增加 EpisodeTranscript、BehaviorProfile、BehaviorDiff；MotionSegment 标明 `producer ∈ {rule, visual}`，rule 为主、visual 为补；HypothesisRecord 的预测字段限定为词表项 + 方向 + 适用 cell |
| Phase 2 | Stair 补：带符号 roll/pitch、禁触体身份与所在台阶层、终止步、joint_pos 有界窗口、base 速度；Push 补：逐脚接触、逐步五个稳定子条件布尔量 |
| Phase 3 | 交付词表 v1（Stair ≤ 15 项、Push ≤ 8 项）、三对象、affordance 标注；验收增加"每项可由 Phase 2 字段重算"和"每条描述有 affordance 或 unsupported" |
| Phase 5 | 记忆更新规则中加入自动裁决；预测命中率入 ContextReceipt 之外的单独账 |
| Phase 6 | 媒体类型增加 `derived_plot`；相机池与 ROI 在 Phase 0 定；网格合成帧作为预算手段 |
| Phase 7 | 分析器输入含 L1/L2；输出增加 `proposed_rule`（可编译判据），进入词表前须任务侧重算与人工核对 |
| Phase 8 | 假说 sidecar 用词表项表达预测；`evidence_request` 与 `max_provider_attempts` 的关系按前一份审计 2.7 写明 |
| Phase 9 | 盲评标签集包含 L2 词表项；指标增加"假说可执行性（有 affordance 的机制占比）"与"预测命中率" |
| 首批落点 | `episode_evidence.py`（事件 onset/持续/部位/台阶层）、`ordered_summary.py`（BehaviorProfile 与配对 BehaviorDiff）、`tasks/stair_climbing/prompt_adapter.py`（画像与差分进入 feedback entry）、`push_recovery/runtime.py` / `force_runtime.py`（逐脚接触与子条件采集） |

薄纵切顺序不变：先做 Stair transcript 与 forbidden_contact 回合的 BehaviorProfile，用保留 checkpoint 离线做 A/T 盲评，再决定 Evidence Service、Broker、视觉的投入。

## 7. 本轮核实的研究与工程

| 来源 | 核实方式 | 采用点 | 边界 |
|---|---|---|---|
| CoALA，arXiv 2309.02427 | 摘要 | 记忆 / 动作空间 / 决策过程三维（本文加"观察"为四维） | 综述框架，不是实验证据 |
| Language to Rewards，arXiv 2306.08647 | 摘要与提示词描述 | "运动描述符"作为语言与奖励之间的中间表示；本文的 L2 是其逆向（观测→描述） | 用 MJPC 而非 RL |
| Eureka，arXiv 2310.12931 | 摘要；reward reflection 细节、消融以及每轮样本数与轮数均沿用论文正文记忆，本轮未重新打开正文 | 分项奖励中间序列是最低限度的训练过程反馈 | 不证明物理因果 |
| DrEureka，arXiv 2406.01967 | 摘要 | 动作空间扩到域随机化 | 四足与灵巧手 |
| Eurekaverse，arXiv 2411.01775 | 摘要 | 动作空间扩到环境课程代码 | 四足 parkour |
| SDS，arXiv 2410.11571（CoRL 2025） | HTML 正文 | 关键点时间网格（4×4 / 6×6 帧一张图）、二值接触序列图、分项奖励统计的自然语言摘要、0–10 分数向量；每轮 8 候选、5 轮 | 周期步态、单视频模仿 |
| Video2Reward，arXiv 2412.05515 | HTML 正文 | 不用 VLM 看视频，用关键点轨迹 FastDTW 相似度分数表反馈 LLM；对比 Eureka | 需要参考视频 |
| RDA，arXiv 2606.01672 | 前一份审计已核 | 多 rollout 视频 + 子任务日志 → VLM 失败汇总 | 2026 预印本 |
| HARBOR，arXiv 2606.08610 | 摘要 | 标准化命令、持久 artifact、可执行 gate、跨 run 经验；动作含奖励、超参、环境参数；6 基准 16 任务 | 未看源码 |
| Agent²，arXiv 2509.13368 | 摘要 | MDP 建模 + 算法优化两阶段、训练反馈分析 | 非机器人基准 |
| GVL，arXiv 2411.04549 | 摘要 | VLM 对乱序帧估计任务进度；可作为"视觉进度"与 progress_fraction 的一致性探针 | 操作任务为主 |
| Proprioception Enhances VLM Captioning，arXiv 2512.20876 | 摘要 | 图像 + 本体轨迹一起给 VLM 改善动作描述与分段 | 操作任务 |
| StairMaster，arXiv 2606.25765；Blind Stair Climbing，arXiv 2402.06143 | 摘要与检索摘要 | 台阶失败用语：leg trapping、torso collision、front leg trapping、hind leg tripping、棱碰撞引起侧漂 | 感知与实机为主 |
| Hildebrand 步态图 | 综述检索 | duty factor 与相位差作为步态描述变量 | 生物力学定义 |
| Horak & Nashner 1986、Maki & McIlroy、Pratt 捕获点 | 综述检索 | ankle / hip / stepping 三策略与 change-in-support；捕获点作为迈步落点参考 | 人体与人形 |

未核实：HARBOR 与 Agent² 的源码与许可；Eureka 消融的具体数值；StairMaster 正文是否给出失败分类统计。

## 8. 一句话

计划把"看得见"做对了一半（L0、L1），要成为面向 RL 的 agent，还差"说得清"（L2 行为描述与差分）、"能裁决"（预测自动核对）和"留接口"（动作空间合同）。这三样都不贵，而且都能在薄纵切里先做出来。

## 9. 逐条复核（2026-09-23）

用户要求把本文每条判断重新对照源码与结果核一遍，并独立判断是否真的需要改计划。结果如下；标"已改"的条目正文已同步修订。

| 条目 | 复核结果 | 计划是否需要改 | 本次修订 |
|---|---|---|---|
| 0.1 粒度粗在折叠 | 成立：`StairEnvTransition` 20 字段、tracker 6 个 latch、Push `_observe_pose` 只读 root；补充正式循环已持久化逐回合 facts（当前套卷即校准套卷） | 是 | 已改 |
| 0.2 缺 L2 层 | 成立：计划 3.2 对象表没有 transcript / profile / diff，Phase 3 无词表；MotionSegment 同时是 Phase 3 与 Phase 7 的交付 | 是 | 无 |
| 0.3 确定性优先 | 成立：Stair 各项可由现有字段与 `FlatStairSurfaceQuery` 几何算出；Push 的 stepping 可由逐脚接触判定，ankle/hip 需关节角 | 是 | 已改 Push 词表 |
| 0.4 派生图 | 原文"计划列为不通过项，需要改"不准：计划未禁止派生图，Phase 7 允许分开标记的示意图；真正缺的是把测量数据渲染图列为 Phase 6 交付物并作分析器默认输入 | 是，范围缩小 | 已改 |
| 1 四维画像 | 动作空间（仅 `complete_reward_program`，13 个 runner-owned 组件）、输出形状、记忆字段、lane 规则均核实；补充契约对 rationale 叙述的内容要求 | 是，第 3 章改写 | 已补 |
| 2.1 解码丢失项 | "joint_pos 丢弃、禁触体身份丢失"不完整：二者仍在 host_history 原始步，只是 transition 不带、不落盘 | 不改计划，改审计 | 已改 |
| 2.1 15 代例证 | 成立；补充 G1-a/G4-a 直接惩罚禁触力的路线，回合级 2×2 表可从现有 summary 直接算 | 无 | 已改 |
| 2.2 Push 无逐脚数据 | 成立：`rollout.py` 与 `runtime.py` 都不读脚部状态 | 是 | 无 |
| 2.3 训练过程反馈 | 成立：`_evidence_interval_steps` 来自 `evidence_sample_interval_steps=24`，只有 last/max/nonzero；Eureka 消融沿用记忆 | 已在计划 Phase 3 | 无 |
| 3.3 affordance 标注 | 建议性条目；`StairRewardInputsVNext` 的 39 个输入名已核实，可支撑标注 | 是 | 无 |
| 4.2 相机 / 覆盖 / 内存 | 成立：`eye=(4.5,−4.5,3.0)`、单 `env_index`、其他 env `MakeInvisible`、`_frames` 无上限 | 是 | 无 |
| 5.2 假说无人裁决 | 成立：HypothesisRecord 无裁决者；`_compact_entry` 不含 rationale | 是 | 无 |
| 7 先例表 | HARBOR、Agent² 仅摘要；Eureka 搜索参数沿用记忆，已标注 | 无 | 已标 |

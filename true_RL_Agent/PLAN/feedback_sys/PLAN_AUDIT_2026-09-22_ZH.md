# 反馈系统升级计划审计

日期：2026-09-22。对象：[FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md](FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md) v1.1 与 [RESEARCH_AND_ENGINEERING_REVIEW_ZH.md](RESEARCH_AND_ENGINEERING_REVIEW_ZH.md)。
依据：当前工作树源码（HEAD `5cef484d` 加未提交变更）、一次真实 G13 提示词重建、15 代正式结果、P5/P6 审计产物，以及本轮独立联网复核。本文只是审查意见，没有改运行代码、启动实验或把任何 Phase 标为通过。

## 0. 结论

计划的架构方向成立，值得做；但按现在的顺序和统计设计，最贵的两个阶段大概率得不到结论。具体判断：

1. **分层架构合理。** "原始测量 / 确定性事件 / 模型笔记"三层、只读 public-dev 证据域、复用 journal、recorder、ledger 派生索引，与代码里"任务拥有语义、共享层拥有生命周期"的边界一致，没有引入第二份真值。
2. **价值验证放得太晚。** 第一个"新反馈是否改变诊断"的信号要到 Phase 9，而 Phase 9 CORE 路径前置依赖 Phase 3、4、5、8。其中 4、5、8 都是中等工程量，却不是回答"时序证据有没有用"所必需的。应先做一条薄纵切。
3. **Phase 11 按历史基率几乎必然 `INCONCLUSIVE`。** 15 代正式结果是 11 个完整代里 1 次接受，接受候选相对 reference 的 ΔJ_primary 在 public-dev 口径是 +0.0138（public-accept +0.0267，sealed 同卷 +0.0120，最终确认新 seed +0.0158）。Phase 11 的主终点正是 public-dev 口径的 current-best 差值；按 n=8、δ=0.01 粗算，置信下界不可能超过 δ，即使升级把接受率翻倍也不行。主终点需要改成 session 级连续量，或明确 Phase 11 为探索性。
4. **现状描述有三处不够精确**，会误导 Phase 1/2/5 的工作量：字节裁剪已有确定性优先级和回执；Push 逐 episode 事件时刻已采集只是没投影；Stair 正式循环用的套卷就是校准套卷，public-dev/accept summary 已持久化逐回合 facts，缺的只是逐步时序。
5. **视觉分支的成本和 provider 依赖没有落到 Phase 0。** 当前默认模型是 `deepseek-v4-flash`；DeepSeek 视觉能力和每图计费在 2026-08/09 之间变动过，必须实测冻结。24 图/轮按官方上限 1024 tokens/图就是 ≈24.6k tokens，与整份文本提示词同量级。
6. **上下文管理缺三件便宜的事**：bytes→token 校准（代码按 4 B/token 估，实测 3.1–3.7）、前缀缓存命中率基线、静态合同占比（约 38%）的处理。它们比 Context Broker 更早见效。

下面分节给证据和修订建议。

## 1. 现状核对：计划 2.1 的判断对不对

| 计划判断 | 代码事实 | 结论 |
|---|---|---|
| 逐步运动状态没有持久化 | Stair recorder 每步发布 base 位姿、joint_pos、逐支撑体位置/速度/接触力、逐禁触体力、终止掩码（`isaaclab_runtime_env.py:97-170`）；`evaluation_runtime.py` 把整段 host_history 只留在内存，tracker 按 latch 输出 `StairEpisodeOutcome`；当前 v3 协议绑定的 `unified_evaluation_suite.v2` 即校准套卷（`records_episode_facts=True`），public-dev/accept summary 已持久化 `episode_facts_by_cell`（primary_reason、episode_steps、progress_fraction、diagnostic_flags） | 逐步部分正确，逐回合部分已经解决。补充：`abs_roll/abs_pitch` 在 decoder 去符号；joint_pos 与逐禁触体力仍在 host_history 原始步里，只是 `StairEnvTransition` 不带、也不落盘；transition 已有"哪只脚"的支撑级信息，只是 tracker 没记 onset/持续 |
| Push 只剩极值 | `runtime.py` / `force_runtime.py` 逐 episode 记录事件 control step、恢复步、censored latency、冲量与力误差、极值；`project_push_split` 与 `_v2_dev_summary` 只投影分数、计数、strata、标签 | 半对。属于"已保存未展示"，成本远低于新采集。真正缺的是逐步稳定条件分量（哪个条件反复打断 dwell） |
| 训练反馈只有 last/max | Stair `runtime_summary` 每 `_evidence_interval_steps` 采样 last/max/nonzero；另有同 transition 的 candidate−parent 差分 `operation_effect`；Push 同构 | 正确。早/中/晚分段确实没有 |
| 图片路径不构成视觉输入 | provider 只发 system + user 文本，`response_format=json_object`；无 image block | 正确 |
| 字节限额"尚不能表达哪些证据应优先保留" | Stair 裁剪链是确定性的：static-core → 压缩条目 → 历史降级 → 丢历史 → minimum-core，每步有 capacity receipt；条目选择按 recent / latest_nonaccepted / best_nonaccepted / accepted / reference anchor | 不精确。真实缺口是：优先级按记录角色而非诊断价值；各段独立上限（task sections 56 KiB、memory 14 KiB、history 8 KiB）没有全局分配；口径是字节不是 token |
| journal 有恢复与预算 | `RoundAuthoringJournal` 有 reserved attempt、continuation、budget contract、逐 attempt provider usage | 正确 |
| recorder 已支持任务选时点 | `NativeVideoRecorder.start(task_records_steps=…)`、`no_pre_reset_frame` 状态 | 正确。但 `_frames` 把整段留在内存：1280×720×3 B ≈ 2.6 MiB/帧，Stair 锚定 horizon 600 步 ≈ 1.6 GB/段 |
| 记忆已有机制卡与等价排除 | `ExplorationEvidenceCard`、revisit frontier（≤4）、blocked verifier keys、advisories；experiment_memory 16 轮/14 KiB 含 trial_outcomes | 正确。没有"假说→预测→结果"记录；索引以 `session_genesis_hash` 为界，跨 session 不存在 |

### 1.1 一次真实提示词的构成（G13 重建，102,380 B）

| 段 | 字节 | 占比 | 性质 |
|---|---|---|---|
| reward_inputs_contract 13.6K、required_candidate_package 11.3K、public_eval_contract 4.6K、task 3.1K、candidate_capabilities 2.4K、loop_contract 2.0K、selection_guidance 1.7K | ≈39K | 38% | 每轮重复的静态合同 |
| public_history 23.3K（experiment_memory 15.0K、recent_outcomes 7.9K）、public_feedback_summary 19.8K、runtime_feedback 1.6K | ≈45K | 44% | 证据与历史 |
| source_payloads 12.1K、source_catalog 4.2K、candidate_lineage 1.7K | ≈18K | 17% | 源码 |

容量探针实测：初代 58,959 B → 16,669 tokens（3.54 B/token）；G13 114,943 B → 37,367 tokens（3.08）；修复帧 49,248 B → 13,391（3.68）；source followup 102,010 B → 33,645（3.03）。代码里 `authoring_frame._BYTES_PER_TOKEN_ESTIMATE = 4` 对晚代提示词低估约 25%。

### 1.2 本地最有力的"行为证据不足"案例

15 代正式结果（`PHASE6_15_ROUND_RESULTS_ZH.md`）：主失败是 `forbidden_contact`，而 Go2 的禁触角色绑定的是 base，不是脚；"全程 clearance diagnostic 618 次而 primary 仅 1 次"。G13 提示词的实验记忆显示两条路线都试过：G1-a 与 G4-a 直接惩罚禁触力，均被拒；G3、G6、G7 到 G12 连续提出 swing-foot 系列机制（clearance_penalty → low_forward_penalty → level_sweep，G8 接受 → vertical_lift → banded_vertical_gate → forward_nudge → lift_velocity / rise_clearance），此后再无接受。没有任何证据说明"脚净空不足"是否先于"base 撞台阶"发生、发生在哪只脚、成功回合里是否同样发生。G12-lane-b 的 `slip_detected=0.343` 与 `slip=0` 也在这份提示词里。这就是 Phase 3 应该回答的第一个问题。对照包的最小形态可以很小：回合级的"终局类别 × 是否出现 clearance/slip 标志"2×2 表今天就能从现有 public-dev summary 的 episode_facts 算出，不需要重跑；"终局前 N 步内"的时序版本才需要逐步数据。

## 2. 架构逐项审查

### 2.1 组件与所有权：成立，但首版应更薄

- Evidence Service 应作为 `source_request` 的同构扩展：同一 journal、同一 attempt 记账、同一 public-safety 断言，不另起一套服务。
- Claim Validator 首版只做四件事：引用的 evidence ID 存在、时间窗在 episode 内、身份（task/robot/protocol/checkpoint）匹配、被引用值与档案一致。"物理因果不能自动证明"已在计划里，无需再造规则引擎。
- Context Broker 首版应先是"预算表 + 遗漏回执"，而不是排序算法：现有 capacity receipt 已有原件 hash、投影 hash、omitted 列表，可直接扩到 payload 级。
- Evidence Inspector 用现有 HTML/JSON 导出即可；不建议在 Phase 3 就绑定 Rerun。

### 2.2 顺序：先做薄纵切，再决定服务、Broker 与视觉

建议在 Phase 0 之后插入一条不新增组件的验证链，并把 Phase 9 CORE 比较的前置依赖从"3、4、5、8"降为"2、3"：

1. **Phase 2-lite**：`evaluation_runtime._execute_stair_seed_rollout` 已把全部原始步和 `StairEnvTransition` 物化到 host；按 seed 写一份 NPZ + JSON manifest（identity、control step、单位、缺失状态）。只存 transition 约 72 envs × 600 步 × ~36 个标量 ≈ 6 MB/seed（float32）；连原始步的 joint_pos、逐支撑体位置/速度/接触力和逐禁触体力一起存约 80 个标量、≈14 MB/seed。Push 加逐步五个稳定子条件布尔量与逐脚接触。都不改物理、动作和终点。
2. **Phase 3-lite**：只做 Stair 事件表。从现有 transition 字段可直接算：每个支撑体的接触 onset/offset、surface level 变化、clearance 低于 0.018 m 的 onset 与持续、slip（接触且平面速度 > 0.5 m/s）onset 与持续、进度停滞、forbidden_contact 步、首次到顶步、带符号的 roll/pitch（恢复符号是一行改动）。输出"时间 × 部位"事件表与成功组对照发生率。
3. **投影**：把事件表加进现有 `public_feedback_summary`（56 KiB 之内，走现有裁剪链），不建 Evidence Service。
4. **Phase 9-lite**：用保留的 checkpoint 重跑 public-dev。Gen9 的隔离诊断回放给出了真实时长：一个候选的 216 个 public-dev 回合 runner wall 124–150 s（`PHASE6_OBSERVATIONS_ZH.md`），24 个案例约 1 GPU 小时。冻结 12–24 个案例，做 A（现有反馈）对 T（加事件表）的离线盲评。评分维度加两项：LLM 假说是否引用了支持它的事件证据；提出的机制是否针对实测发生部位与时序。
5. 只有 T 显示可核查的收益，再建 evidence_request、机制记忆和视觉。

案例来源可行：c6 正式运行的输出目录保留 71 个 checkpoint（P6 姿态审计记录，1271 个文件），含被拒候选与 reference，足以满足"≥3 个训练谱系"；另一场运行的保留情况本轮未核实。

### 2.3 Phase 9 的指标

- 确定性投影下，"主要失败阶段/部位定位准确率 ≥90%"对 T 系配置近乎由构造保证，评的其实是 LLM 的阅读与引用，可以保留，但不能当作模块收益证明。
- 建议增加"假说可检验性"评分：作者输出是否给出授权机制 + 预测指标方向 + 支持/反证引用；这正是 Phase 8 sidecar 的内容，应在离线阶段就开始评。
- V 的价值在规则词表之外的行为。标注集必须允许标注者写出词表外事实，否则 recall 指标对 V 不公平。计划已提到"均匀覆盖帧"和"罕见成功动作"，但标注协议要明确写出。

### 2.4 Phase 11 的统计设计：按历史基率不可能 PASS

历史事实：11 个完整代中 1 次接受；接受候选相对 reference 的 ΔJ_primary 在 public-dev 口径为 +0.0138（reference 0.934438 → Gen8-A 0.948267），最终确认新 seed 上的配对差为 +0.0158（SE 0.0029）。Phase 11 主终点用 public-dev 口径。

粗算（只为判断量级）：每代接受概率 p ≈ 1/11；每 session 3 代，P(session 有接受) ≈ 1 − (10/11)³ ≈ 0.25；接受时 Δ ≈ 0.014，否则 0。基线臂 session 均值 ≈ 0.0035。若升级把 p 翻倍，P ≈ 0.45，升级臂均值 ≈ 0.006，两臂差 ≈ 0.003。配对差的标准差约 0.009–0.010，n=8 时 SE ≈ 0.0035，t(7) 半宽 ≈ 0.008，下界 ≈ −0.005。要下界 > δ=0.01，需要两臂均值差 ≥ 0.018，即升级臂几乎每个 session 都接受一个约 +0.02 的候选。

结论：以"current-best 差值"为主终点、n=8、3 代/session 的设计，几乎必然 `INCONCLUSIVE`。计划的 Phase 0 敏感性仿真会算出同样结果，但现在就应改：

- 主终点改为 session 级连续量，例如 session 内全部已训练候选的 public-dev ΔJ（相对该代 parent）的均值或中位数；统计单位仍是 session，不违反"不按候选数量扩大样本量"。current-best 差值降为次要终点。
- 或选择余量更大的比较域：本任务 ID、F_c1、F_c2 已在天花板，余量集中在 F_scalar（joint-OOD 层 final 为 17/63、19/63）。
- 或按计划已有条款，把 Phase 11 明确记为探索性，预算转投 Phase 9/10。

### 2.5 视觉分支（Phase 6/7）

- **provider 事实**：`provider_limits.py` 默认 `deepseek-v4-flash`。DeepSeek 官方 vision 指南（2026-09 读取）写 `deepseek-flash` 接受图像、每图上限 1024 tokens、单请求最多 600 图、旧名 `deepseek-v4-flash-vision-exp` 已退役；而 2026-08 的第三方报道称 vision 为独立实验端点、每图 384 tokens、V4-Flash 生产端点仍为纯文本。两个来源冲突。2026-09-23 复核官方定价页：`deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 两个旧名都由 DeepSeek-V4.1-Flash 服务，`deepseek-flash` 支持图像输入而 `deepseek-v4-pro` 不支持；冲突属于版本演进，不是矛盾。本地 provider 仍只发文本，Phase 0 仍须发真实 image block、读回 usage 后冻结每图计费。
- **成本**：24 图 × 1024 ≈ 24.6k tokens/轮，约等于当前一份晚代文本提示词；Claude 约 (w×h)/750 ≈ 1.2k/张（1280×720），OpenAI high detail 约 85+170×6 ≈ 1.1k/张。分辨率（例如 640×360）和 ROI 策略应在 Phase 6 第一项决定，不能等 Phase 9 再算账。
- **覆盖**：recorder 每 seed 只录 `env_index`（默认 0）一段，216 个 public-dev 回合只有 3 段视频，主失败大概率不在录制池。计划的 capture selector 正确指出了这一点；实际可行路径是"数值档案选出失败回合 → 同 seed 诊断回放 → 与档案逐步核对一致再标为同源"。Go2 posture audit 已观察到一次 192 回合逐字段一致的回放，可作 Phase 0 的允许误差登记依据。
- **内存**：`_frames` 无上限，600 步 ≈ 1.6 GB/段；应作为 Phase 6 第一条硬门，而不是验收清单中间的一条。
- **价值预期**：主失败是 base 撞台阶、集中在 joint/height-OOD 层；数值层能定位哪只脚、何时、是否先出现 clearance/slip。视觉的增量在"未命名行为"。VISION 保持可选、不阻塞 CORE 是对的。

### 2.6 记忆与上下文：先做六件便宜的事

1. **token 校准**：用现有 provider 回执（`input_tokens`、prompt bytes）按模型和段类型拟合 bytes→token；数字/哈希密集段更贵。Broker 之前先把 4 B/token 换成校准值，或直接调用 DeepSeek 离线 tokenizer。
2. **前缀缓存**：DeepSeek 自动前缀缓存以 64 token 为单位、必须从第 0 token 起完全一致。`payload.py` 的段顺序是 task、loop_contract、candidate_capabilities、reward_inputs_contract、public_eval_contract、selection_guidance 在前（≈27 KB 静态），之后是 source_catalog、source_payloads、feedback、history 等可变段，而 `required_candidate_package`（11.3 KB，静态）排在 exploration_directive 之后。把它前移到静态区，可缓存前缀从约 27 KB 增到约 39 KB（≈13k tokens）。探针里 G13 只命中 512 是因为一次性请求没有前序同前缀，修复帧命中 5,376、source followup 命中 6,656 说明机制有效。Phase 0 应记录"可缓存前缀 token 数 / 命中率"，保证 system prompt 与静态合同字节级稳定。这不缩短上下文，但每轮成本可降约三分之一。
3. **静态合同瘦身**：`reward_inputs_contract` 的 fields 6.8 KB 与 field_semantics_constraints 4.8 KB 每轮重复；可转为 requestable source（现有 `source_request` 通道）或依赖缓存前缀。
4. **两级预算**：把各段独立上限改为"固定核心 + 可变证据池"：固定核心保底身份、授权、当前 parent、最近待解释 trial、事件目录；可变池按诊断相关性分配，遗漏写回执。现有 capacity receipt 可直接复用。
5. **记忆范围**：明确 session 内还是跨 session。Phase 11 的"共同只读初始历史快照"暗含跨 session；当前索引以 `session_genesis_hash` 为界，跨 session 需要 protocol/robot 相容性门。
6. **位置效应**：证据段位于 payload 40 KB 之后；不建议为此加实验臂，Phase 9 的短/长上下文探针已覆盖，Phase 0 基线记录"证据段起始 token 位置"即可。

### 2.7 Phase 8 的一个实现细节

`evidence_request` 会消耗 provider attempt。现有循环里每次 provider 调用占一个 attempt，`source_request` 落在最后一次尝试即判 `provider_failed`（`authoring_loop.py` 的 `source_request_denied` 分支）。15 代里 3 代因成对 authoring 不完整未训练（G6、G13、G14，其中 G13/G14 是 B lane 耗尽 authoring 预算），另有 1 代候选运行失败、1 代无合格选择；authoring 失败本身就是损失来源。补查应有独立的 attempt 槽（例如仅供 evidence_request continuation 使用的 +1），否则会抬高 authoring 失败率。计划写了"分别限额"，但没写清与 `max_provider_attempts` 的关系。

### 2.8 验收条款密度

按条目统计，Phase 2 有 10 条验收、Phase 8 有 10 条、Phase 9 有 16 条、Phase 11 有 12 条，且大量数值门标注为"Phase 0 可修订"。建议每个 Phase 分三层：硬门（身份、权限、计数、缺失语义）、效果门（数值阈值）、延后项（MCAP 式索引、原子发布、Rerun）。可合并：Phase 1 并入 Phase 0/2（P5 已完成的一致性修复只需回归矩阵）；Phase 4+5；Phase 6+7。Phase 10 pilot 保留。

## 3. 研究与开源核查（本轮独立复核）

| 来源 | 复核结果 | 对计划的影响 |
|---|---|---|
| ProTracer arXiv 2609.21369（2026-09-18） | 存在；本体信号→语言，失败起点定位，操作任务 | 借"三级取证 + 起点定位"合理；仍是审计前 4 天才出现的预印本 |
| KITE arXiv 2604.07034，ICRA 2026 | 存在；关键帧 + BEV + 检测置信度，RoboFAC | 借紧凑证据包合理；BEV 不证明距离的边界写对了 |
| RAPT arXiv 2602.01515 | 存在；humanoid sim-to-real OOD 监视，temporal saliency + LLM | 首版不训练检测器是对的 |
| LongMemEval-V2 arXiv 2605.12493 | 存在；451 问；AgentRunbook-R 三池（观察/事件/策略笔记） | 三层记忆有先例 |
| RDA arXiv 2606.01672 | 存在；ManiSkill 12 + HumanoidBench 4；VLM 评轨迹、总结失败、改 reward | 与本项目最接近的先例，但改子任务定义不照搬是对的 |
| Q-Frame ICCV 2025 | 存在；Gumbel-Max 随机 | 计划要求确定性选帧正确 |
| AHA（NVlabs）LICENSE | NVIDIA 许可，非商业（research/evaluation） | 只借方法 |
| Video2Reward 仓库根目录 | 仅 README.md 与 fig2.png | 无可接入实现，计划记录准确 |
| Mem0 `memory/main.py@a39a802` | infer 路径只产生 ADD 事件 | "ADD-only"准确 |
| ACE `playbook_utils.py@82709de` | 仅实现 ADD；UPDATE/MERGE/DELETE 为 TODO | 准确 |
| DeepSeek 视觉能力 | 官方文档与 8 月报道冲突（见 2.5） | Phase 0 实测 |

计划可补充的参考：STRIDE（arXiv 2502.04692，人形 locomotion 奖励自动化，摘要未给反馈细节）；CARD（arXiv 2410.14660，用轨迹偏好评估减少每轮重训）；Code-as-Monitor（CVPR 2025，VLM 生成约束监视代码，可作为"由模型提议、任务侧编译校验的规则事件"的思路，但不进入 gate）；RoboReward（arXiv 2601.00675，VLM 奖励模型也会犯简单错误，支持"视觉观察 ≠ 测量"）；Curie（2025）与 AI Scientist-v2 的实验严谨性与假说记录，可作 Phase 5 HypothesisRecord 的设计参照。

未复核项：ProTracer、KITE、RAPT 的源码文件级声明（SHA、许可）本轮未逐个打开，沿用研究记录的说法。

## 4. 建议修订（按优先级）

1. 在 Phase 0 后插入薄纵切（2-lite / 3-lite / 投影 / 9-lite），Phase 9 CORE 前置依赖降为 2、3；首个问题固定为"forbidden_contact 终局前是否先出现 clearance/slip，与成功组对照"。
2. 重写 Phase 11 主终点与功效预估：session 级连续量为主，current-best 差值为次；或改比较域；或明确探索性。
3. Phase 0 增补五项基线：provider 视觉能力与每图计费实测、bytes→token 校准表、前缀缓存命中率、`_frames` 内存上限、诊断回放一致性允许误差。
4. 修正计划 2.1 的三处现状描述（裁剪优先级已存在；Push 逐 episode 事件已采集；Stair 当前套卷即校准套卷，逐回合 facts 已持久化）。
5. Phase 8 写明 evidence_request 与 `max_provider_attempts` 的关系；Phase 5 写明记忆的 session/跨 session 范围。
6. 验收分层（硬门/效果门/延后项）；合并 1→0/2、4+5、6+7。
7. 首批源码落点（对应 2-lite/3-lite）：`evaluation_runtime.py`（按 seed 写 NPZ + manifest）、`episode_evidence.py`（onset/持续/支撑索引/带符号倾角）、`ordered_summary.py`（事件表与对照发生率汇总）、`tasks/stair_climbing/prompt_adapter.py`（扩展 feedback entry，受 56 KiB 与裁剪链约束）。

## 5. 逐条复核（2026-09-23）

用户要求把上述每条判断重新对照源码与结果核一遍，并独立判断是否真的需要改计划。结果如下；标"已改"的条目正文已同步修订。

| 条目 | 复核结果 | 计划是否需要改 | 本次修订 |
|---|---|---|---|
| 0.1 分层架构合理 | 成立 | 否 | 无 |
| 0.2 验证太晚 | 成立：计划 Phase 表原文写 Phase 9 CORE 依赖 3–5、8 | 是 | 无 |
| 0.3 Phase 11 功效 | 成立，但原文引用了 sealed 口径的 +0.016；主终点是 public-dev 口径，实际 +0.0138 | 是 | 已改数字，结论不变 |
| 0.4a 裁剪已有优先级与回执 | 成立：`_compact_public_feedback_to_fit_task_sections`、`build_prompt_capacity_receipt`（original/projected hash、omitted），选择依据 recent_evaluated / latest_evaluated / latest_nonaccepted / best_observed_nonaccepted / accepted_current_best / reference_anchor | 是，改计划 2.1 措辞 | 无 |
| 0.4b Push 事件已采集未投影 | 成立：`episode_records` 含事件 control step、恢复步、censored 延迟；`project_push_split` 只出 score_stats、recovery_counts、strata | 是 | 无 |
| 0.4c Stair episode facts | 原文说"只有校准套卷"不准：v3 协议绑定的 `unified_evaluation_suite.v2` 就是校准套卷，正式循环的 public-dev/accept 已持久化逐回合 facts；Phase 2 的逐回合工作量比原文估计更小，逐步时序仍缺 | 是 | 已改 |
| 0.5 视觉 provider / 成本 / 覆盖 / 内存 | provider 默认 `deepseek-v4-flash` 与纯文本消息成立；`_frames` 无上限成立；每 seed 单 env、其他 env 隐藏成立；DeepSeek 文档冲突为联网结论，本轮未重复 | 是 | 无 |
| 0.6 token 校准 / 前缀缓存 / 静态占比 | 成立；补充 `required_candidate_package` 位于可变段之后，前移可扩大可缓存前缀 | 是 | 已补 |
| 1.x joint_pos "校验后丢弃" | 不完整：joint_pos 与逐禁触体力仍在 host_history 原始步，只是 transition 不带、不落盘 | 不改计划，改审计 | 已改 |
| 1.2 15 代例证 | 成立；补充 G1-a/G4-a 曾直接惩罚禁触力并被拒；回合级 2×2 表可从现有 summary 直接算 | 是 | 已改 |
| 2.2 薄纵切 | 可行；原文时长引用的是 Push 3×64 回放，已换成 Stair Gen9 诊断回放的 216 回合 124–150 s；71 个 checkpoint 属 c6 一场 | 是 | 已改 |
| 2.3 Phase 9 指标 | 意见性条目，未发现事实错误 | 是 | 无 |
| 2.5 回放一致性依据 | P6 记录 192 回合逐字段一致，成立 | 是 | 无 |
| 2.6 记忆范围 | 成立：索引以 `session_genesis_hash` 为界，Phase 10/11 的共同初始快照跨 session | 是 | 无 |
| 2.7 attempt 槽 | 成立：`authoring_loop.py` 的 `source_request_denied` 分支；原文对 4 个 incomplete 代的描述混入了 no_selection，已改 | 是 | 已改 |
| 2.8 条款数 | 原文计数错误：Phase 2 为 10 条、Phase 9 为 16 条 | 分层建议保留，仅为可读性 | 已改 |
| 3 研究核查 | 联网结论，本轮未重复打开 | 无 | 无 |

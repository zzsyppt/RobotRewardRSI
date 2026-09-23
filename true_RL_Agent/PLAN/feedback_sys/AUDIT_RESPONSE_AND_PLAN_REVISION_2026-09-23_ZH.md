# 两份反馈计划审计的逐条处理与v1.2修订说明

日期：2026-09-23。处理对象为用户提供的两份审计：

- A：[PLAN_AUDIT_2026-09-22_ZH.md](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/PLAN_AUDIT_2026-09-22_ZH.md)。
- B：[PLAN_AUDIT_2_AGENT_ARCHITECTURE_2026-09-23_ZH.md](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/PLAN_AUDIT_2_AGENT_ARCHITECTURE_2026-09-23_ZH.md)。

被审的[v1.1计划](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/archive/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.v1.1.md)已原样归档；本次修改落在[v1.2计划](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md)。A、B原文均未改动。本文把重复意见合并核验，但保留原章节对应关系；“采纳”只指修改计划，不代表实现或验收完成。

## 1. 总判断

两份审计抓住了三项真正需要改的事：价值验证太晚；高层行为交付与预测裁决不够具体；上下文、视觉和补查成本应与现有实现对齐。这些已纳入v1.2。

同时有几类建议不能原样照搬：把不可测的物理解释写成确定性词条；把文件数量当独立谱系；用一条自适应历史推导未来功效；用候选均分替换最终搜索收益；把新增传感器、规则编译和额外调用视为免费小改。相应条目作了收窄或拒绝。

本轮核查使用当前源码与本地产物；代码定位先用CodeGraph，再窄读实际实现。联网只补查影响判断的官方provider文档与相关论文正文。没有修改运行代码，没有启动训练、回放、provider试验或统计仿真。

## 2. 审计A逐项处理

| 原条目 | 是否存在/需要解决 | 处理及v1.2落点 |
|---|---|---|
| A §0.1、§2.1：总体分层与所有权合理 | 成立，保留 | 继续复用ledger、journal、recorder、task projection；组件是职责，不要求独立进程。§3–4 |
| A §0.2、§2.2、§4.1：价值验证到Phase 9太晚 | 成立，需要解决 | 新增0→1→2-min→3-min早期A/T价值门。完整Phase 9仍验收集成，不降低其依赖来假装服务/记忆已通过。Phase 3 |
| A §0.3、§2.4：n=8、δ=.01风险很高 | 风险成立；“几乎必然”不是已证事实 | 默认Phase 11为有界探索性；确认性须事前证明设计可行。不以历史接受率当独立概率。Phase 11 |
| A §0.4a、§1裁剪行、§2.6：已存在优先级和receipt | 成立，原计划措辞过宽 | 明确现有确定性优先级、minimum-core及回执；升级全局分配与token计量，不重造裁剪器。§2、Phase 0/5 |
| A §0.4b、§1 Push行：事件已采集未投影 | 成立，但限episode记录 | 先公开投影已有事件/恢复事实，再补逐步稳定条件；逐脚观测另过能力门。Phase 2 |
| A §0.4c、§1 Stair行：正式套卷已保存episode facts | 成立 | 共现表先用已有public-dev产物，时序问题才新增逐步档案。不把已有facts重复列为新采集。Phase 2 |
| A §1原始字段、§5 joint_pos：丢在折叠而非从未采集 | 成立 | 明确joint_pos/禁触合力仍在host history，但transition/持久证据较粗；先保存所需字段。新增signed pose须schema兼容，不按“一行改动”估全部工作量。Phase 2 |
| A §1训练反馈：last/max缺早中晚 | 成立，v1.1已有要求 | 保留分阶段贡献/激活与覆盖，不另建完整训练轨迹库。Phase 2 |
| A §1 journal/视频recorder/机制卡已有基础 | 成立 | 保留既有生命周期、预算和机制索引。媒体缓冲问题单列，已有record时点能力不重建。Phase 1/5/6/8 |
| A §1.1：102,380 B重建、38%静态合同 | 文件大小核实；占比属于指定历史重建口径 | 不作为当前wire固定占比或普适压缩收益。正式记录按当前模型、payload、版本重新分段计量。Phase 0 |
| A §1.1、§2.6.1：4 B/token低估晚代输入 | 成立，需要解决 | 历史四帧实测3.537/3.076/3.678/3.032 B/token；G13基础帧与source followup分别低估约23%/24%，repair约8%。采用真实tokenizer或有误差的保守上界。Phase 0/5 |
| A §1.2：clearance/slip与禁触缺时序关系 | 成立 | 作为第一条Stair问题；共现不叫先因，先后也不自动叫因果。Phase 2/3 |
| A §2.1：Evidence Service/Validator/Broker/Inspector应薄 | 采纳，但不删必要硬门 | 查询与journal共用；校验只证明身份/引用/数值等，不能证明模型解释；Broker先预算表，Inspector可只读导出。§3、Phase 4/5/7 |
| A §2.2：按seed NPZ、36/80标量与6/14 MB估算 | 原型思路可行，数量是条件估算 | 文件格式不写死；数据量依robot/schema/dtype/覆盖而变，整段host NPZ不自动满足有界内存和完成发布门。Phase 2 |
| A §2.2：24案例约1 GPUh | 单位混用，不能作承诺 | 历史时长是完整216-episode评估；24次整评约1小时量级，不等于24个episode案例。费用按checkpoint/整评/episode/窗口分别计。Phase 0 |
| A §2.2：71 checkpoints足够≥3独立谱系 | 不成立，必须纠正 | 71文件只有47个不同hash，含副本及同次失败训练的25个中间checkpoint；按真实训练与祖先关系划分。Phase 0/9 |
| A §2.3：阶段/部位阅读正确不能单独证明收益 | 成立 | 离线另评证据引用、授权合规、机制关联和可证伪预测；T输出正确标签不是完整价值证明。Phase 3/9 |
| A §2.3：视觉应覆盖词表外现象 | 成立 | 原始池独立标注规则外事实；漏选帧仍计系统漏检。Phase 7/9 |
| A §2.4建议1：以已训练候选均值/中位数替代主终点 | 不采纳为主终点 | 会改研究问题且有幸存者偏差；只作保留全部机会分母的次要指标。current-best固定资源收益保持主效用。Phase 11 |
| A §2.4建议2：改F_scalar/joint-OOD比较域 | 有余量可作为设计线索，不能事后改目标 | 公开基线用于事前资格检查；看过的sealed/final子层不能用来挑反馈系统。新比较域须重新登记与资格验证。Phase 0/11 |
| A §2.4建议3：探索性或延期 | 采纳 | 当前预算默认探索性，不能事后见到显著才改确认性。Phase 11 |
| A §0.5、§2.5：默认模型与视觉能力冲突 | 本地发纯文本成立；远端模型判断已过时 | 官方现已将旧Flash别名路由到V4.1 Flash。模型能力、代码传输、实际usage三者分开；启用VISION前有限实测。Phase 0/6 |
| A §2.5：24图≈24.6k token、其他provider公式 | 当前DeepSeek每图1024为官方上界；不能普遍套公式 | 24×1024是特定配置上界，不是每轮实耗。预算按真实尺寸/模型/端点/布局冻结，本文不据旧跨厂商公式定价。Phase 6 |
| A §2.5覆盖/回放：单env与选中失败不一致 | 单env限制成立，未录主失败概率未测 | 记录真实录制池；重放有独立run身份和对齐质量，不能凭同seed改名原episode媒体。Phase 6 |
| A §2.5、§5：192回合逐字段一致可证明回放同源 | 回合记录一致成立，完整逐步同轨不成立 | P6比较的是episode/事件字段，缺完整逐步姿态；只能支持结果匹配的回放，不直接通过新媒体时序验收。Phase 6 |
| A §2.5内存：600帧约1.6 GB | `_frames`无界成立，数值是未压缩条件量级 | 媒体profile第一道资源门；实际长度/FPS/编码开销实测，不作为每段必然消耗。Phase 6 |
| A §2.6.2：前缀前置、缓存基线 | 方向采纳，固定64-token规则与省1/3不采纳 | 当前官方缓存规则已变化且best-effort；核心合同完整，按真实wire与多次同前缀探针判断费用。缓存不减少上下文占用。Phase 0/5 |
| A §2.6.3：静态合同直接转source_request | 只采纳压缩冗余/可选说明 | 现有源码目录封闭，新增合同片段需typed契约；权限、单位、参考系、时间语义及执行ABI必须留核心。Phase 5 |
| A §2.6.4：固定核心＋可变证据池 | 采纳 | 复用容量receipt，先计量/去重/遗漏再增加排序复杂度。Phase 5 |
| A §2.6.5：session与跨session范围 | 成立，需明确 | 默认session内；同hash只读初始包是显式导入，不是实时跨session检索。§3.3、Phase 5/10 |
| A §2.6.6：证据位置与干扰 | 采纳测量，不新增庞大实验臂 | Phase 0记位置，Phase 9短/长探针核对，不从位置本身推断性能下降。 |
| A §2.7：补查消耗attempt，应加+1 | 问题成立；直接+1不是中性修复 | 优先统一额度内预留最终答复；加槽必须更新round/generation总额、费用和公平对照。§3.4、Phase 8 |
| A §2.8、§4.6：验收太密，合并Phase | 采纳可读性与逐阶段定方案；不机械合并 | 保留12阶段以控制单段工作量，改为目标/交付、严格验收、可延后；具体实现由阶段验收卡确定。§5–6 |
| A §3：研究条目及新增先例 | 原研究记录中已核来源继续保留；新摘要线索不足以决定依赖 | 不因更多论文名增加自动规则编译/新动作空间。ProTracer“10天前”日期不准确，2026-09-18到两审计日期为4/5天，不影响其新预印本定位。 |
| A §4.7：首批源码落点 | 总方向采纳 | 路径是所有权地图，schema/存储/actor字段以开发时源码定；不按文件列表机械施工。§8 |

A §5的复核表是以上同一批问题的再次表述，分别对应此表裁剪、Push/Stair facts、字段、实验、回放、记忆与attempt等行；其“已自行核实”不是本次直接采纳的依据。

## 3. 审计B逐项处理

| 原条目 | 是否存在/需要解决 | 处理及v1.2落点 |
|---|---|---|
| B §0.1、§2.1：粒度粗在折叠 | 基本成立 | 保留原始/已保存/已展示三种状态，先复用Stair facts；Push另区分从未采集的脚部观测。§2、Phase 2 |
| B §0.2、§3.1、§9：缺L2、Phase 3无词表 | “完全没有”不成立；交付不够明确成立 | v1.1已有有限行为词表、MotionSegment、对照和身体×时间表。v1.2具体化三个派生视图，不再建真值库。§3.1、Phase 3 |
| B §0.3、§3.2：确定性优先 | 采纳方法，不采纳所有拟议物理词条 | 先用可重算操作名称，明确unknown；详细词条裁决见下表。Phase 3 |
| B §0.4、§4.1：派生图先于原生图且默认最好 | 派生图值得增加；默认优越、读图可靠/固定成本未证 | 正式登记derived_plot，和文本/原生图做有限开发比较；图是源数值的表达，不是额外独立证据。Phase 6/7/9 |
| B §0.5、§1：四维架构与动作空间 | 四个合同作为本项目设计采纳 | 不宣称CoALA原文直接规定这四合同；不以“更像agent”为理由扩大权限。runner-owned组件不是作者待解锁动作。§3 |
| B §1：假说只是rationale字符串 | 过强，需收窄 | 已有结构化revisit hypothesis，但缺事前行为预测及其后续核对；改为在已有结构上扩展。§3.3、Phase 5 |
| B §1：observed_effect枚举 | 审计枚举不准确 | 当前为strong_positive/positive/mixed/neutral/negative/unknown，不能按improved/regressed直接设计迁移。Push当前保守unknown行为保留，除非新证据规则明确支持。Phase 5 |
| B §1“本项目验证严谨度超过所有先例” | 无充分统一比较证据，不采纳 | 不把本地gate数量当跨论文严谨性排名；只说明本项目自己的验证边界。 |
| B §2.1：joint_pos/禁触身份彻底丢失 | 审计后文已收窄，采纳其修正版 | 原步仍有字段，transition和持久化不足；不能把manager布尔和当步最大合力直接当碰撞对象。Phase 2/3 |
| B §2.2、§9：Push无脚部状态 | 评估路径成立，补采需真实工程 | contact sensor当前被关闭，任务只要求base角色；逐脚profile需映射、观测和capture等价门。Phase 2 |
| B §2.2、§3.2：Push没有关节角 | 只对评估Pose记录成立，非全链事实 | reward runtime已读joint_pos/joint_vel；需按来源决定诊断采样。即使有角度也不直接证明ankle/hip策略。Phase 2/3 |
| B §2.3：训练反馈早中晚 | 成立，已在原计划 | 保留有界采样、覆盖和贡献；不把某论文记忆中的消融数值写成本项目验收阈值。Phase 2 |
| B §3.1：transcript/profile/diff | 采纳为三个task-owned视图 | 逐条回指原件；全体分母、配对条件、不确定性齐全。轨迹不同不能简单同step相减。§3.1 |
| B §3.1：24 cells画像差分必在20 KB、信息密度高一个量级 | 未测估算，不作目标承诺 | 实测序列化token与遗漏；不按行数承诺模型信息利用率。Phase 0/5 |
| B §3.3：每条描述有affordance、固定39输入 | 有用，但不能强求可干预或固定字段数 | 绑定实际schema/hash及时间/参考系；允许diagnostic_only、越权和unknown。不用“可映射比例”代替建议质量。§3.2 |
| B §3.4、§6 Phase 7：VLM提议可编译新规则 | 首版不采纳执行/编译链 | 可保存自然语言rule_proposal；正式词表变更另经任务开发和回归，不在线执行模型代码。Phase 7 |
| B §4.1：derived_plot不得证明距离/接触 | 原限制过宽，需要按来源区分 | 可度量图能展示源测量中的距离/接触变量；不能证明未测量物理事实。非度量示意图另标。Phase 6 |
| B §4.1：网格16帧一张可省预算 | 可试，但不是免费压缩保证 | 单张图输入与源帧量分别计；可读性、总像素、token、重复发送都计量，禁止拼图绕上限。Phase 6 |
| B §4.2：相机7m/厘米级只有几像素 | 配置值成立，具体可辨率未测 | 验收实际投影尺寸、遮挡和时间采样；看不清脚级事件时降级为全身姿态观察。Phase 6 |
| B §4.2：纯文本默认模型 | 本地代码发文本成立，模型能力陈述已过时 | 按最新官方文档与实际模型返回身份核对，见§5；不凭别名选型。Phase 0/6 |
| B §4.3：给VLM必要L1/L2与数值 | 采纳 | 做核对与补充；提供过的规则标签不能单独算视觉发现，按原始池另标词表外事实。Phase 7/9 |
| B §5.1：观察/动作/记忆/决策合同 | 采纳 | 写进§3，冻结当前权限，保留未来组件用同一证据接口的设计空间，本次不实现新控制面。 |
| B §5.2：自动裁决归共享层 | 裁决缺口成立，所有权必须调整 | task定义比较资格、指标、分母、阈值及不确定性；shared执行契约和更新。不从正负差值直接宣判机制因果。§3.3、Phase 5 |
| B §5.2、§6 Phase 9：离线新假说预测命中率 | 不成立，缺后续实验 | 离线只评可检验性及核对器fixtures；真实命中率到Phase 10/11。严格历史隐藏结果回放可以另做，不能给新提案找“相似旧候选”代替结果。 |
| B §5.3：面向将来扩动作空间 | 采纳接口兼容，不做权限预授予 | 诊断不局限于reward能改的量，但当前作者不能选择冻结组件。§3.2 |
| B §6改动表 | 按本表逐项采纳/收窄 | Phase 2观测、3派生视图、5核对、6媒体、7规则提议、8答复预留、9离线指标均已明确，未保留不可执行的组合承诺。 |
| B §7研究表及§9自复核 | 研究线索与事实证据分开 | 已读SDS/Video2Reward正文支持多种表达方式，不能证明图总比数值好。HARBOR/Agent²源码许可未核、Eureka记忆数值和部分生物力学摘要不成为当前硬验收。 |

### B的词表建议逐项裁决

| 拟议词条 | 可保留的定义与限制 |
|---|---|
| swing / touchdown | 接触边沿可测；命名摆动/落脚还需位移、迟滞与持续条件。掉落/抖动不能误作主动摆腿 |
| touchdown_offset | 报绑定支撑刚体参考点相对实际地形边界的投影；没有toe keypoint不叫脚尖，粗糙/不规则地形不能统一套FlatStairSurfaceQuery |
| toe_catch | 不作确定测量词条；低净空、减速、停滞共现可作规则事件与候选解释，需要额外接触证据才提升物理含义 |
| scuff / slip | 保留明确协议的接触时高速区间与旧slip；短于5步不自动叫scuff。旧全局any→streak与新增逐脚streak分开 |
| base_strike | 保留禁触角色触发及同时状态；需核对manager实际判据及历史窗口，不能假定等于当步最大力，更不能凭投影层识别碰撞面 |
| stall | 可用窗口内净进度定义；阈值、窗口、分辨率、删失和正常停顿必须登记 |
| pitch_dive / pitch_back | 可由带符号姿态定义越阈事件；加字段维护schema，不从abs恢复符号，不自动解释关节根因 |
| lateral_drift | 报相对参考系的偏移及持续，不把少量单调样本当整体失稳 |
| gait_pattern | 优先报告接触占空与相位；只有周期长度和body mapping有效时才分类，否则mixed/unknown，不能强贴登阶过渡动作 |
| top_arrival | 复用原first_valid_top_arrival规则，禁止改成功口径 |
| response_strategy | 有接触＋位置/着地点才能建立迈步响应操作定义；无离地不证明踝/髋策略，迈步不只向前 |
| step_count / first_step_latency | 在消抖、完整接触/位移和窗口可用条件下计算；接触翻转、腾空、跌落不能都算有效迈步 |
| dwell_break_reason | 保存全部同时失守条件及原值/阈值/时间；同一步多条件失败不强选唯一原因，旧event dwell与force sequence dwell分协议 |
| overshoot | 明确相对哪个姿态/速度目标及扰动方向；pitch过零不是所有任务通用超调定义 |
| fall_direction | 复用真实协议已有分类及方向参考系；不把终局方向当导致倒地的原因 |

## 4. 关键反例与证据锚点

### 4.1 采集与可观测性

- Stair保存的是支撑刚体参考位置和合力：[isaaclab_runtime_env.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py:123)。clearance来自参考点与surface query，禁触来自manager：[transition_decoder.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/transition_decoder.py:551)。这些不足以证明脚尖/接触patch/被撞面。
- 旧全局streak可由不同脚交替贡献：[episode_evidence.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:429)。新增逐脚词条需要独立规则版本，不能改变旧统计。
- 诊断contact看当前力，reward contact看历史窗口最大力：[isaaclab_reward_runtime.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:231)。字段同名不代表时间语义相同，affordance必须携带定义。
- Push关闭contact sensor：[runtime.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:63)；当前角色要求仅base：[plugin.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/plugin.py:36)。其五条件合取见[runtime.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime.py:287)，reward路径读关节状态见[runtime_adapter.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:506)。

### 4.2 上下文、记忆与预算

- 当前已有裁剪及minimum-core：[prompt_adapter.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py:1790)；4 B/token估算位于[prompting/authoring_frame.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/prompting/authoring_frame.py:36)。[容量探针](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P5/audit_capacity/result.json)直接支持上述历史比值，不能代替当前wire重新计量。
- 当前源码请求是封闭目录：[source_request_limits.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/source_request_limits.py:24)、[source_request_service.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/source_request_service.py:62)。移走合同不是只改prompt顺序，需要相应类型与权限支持。
- 已有revisit hypothesis：[exploration.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/exploration.py:1253)；任务效果判定的所有权见[exploration.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/exploration.py:116)。新行为预测与效果枚举不机械混为一类。
- 末次source request失败分支：[authoring_loop.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/authoring_loop.py:598)；generation预算随attempt额度派生：[generation_provider_budget.py](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/generation_provider_budget.py:210)。独立+1会影响资源合同，不能只改局部循环。

### 4.3 实验量级和统计单位

[正式结果](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md:7)记录1 accepted、9 rejected、1 no_selection、4 incomplete。应分别说明1/15全部代次机会、1/11完整科学判决、1/10正式接受判决的条件分母。Gen8-A public-dev差为0.013829；新seed两个固定checkpoint的差0.015778及SE 0.002853不是“搜索session效果”的标准误。

[保护清单](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/shared_history_c6_01/protected_before.json)重算为71个.pt条目、47个不同hash；这只证明文件与内容数量，仍不证明独立训练来源。[旧观察记录](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_OBSERVATIONS_ZH.md:149)的123.966/149.752秒是旧会话完整216回合runner时间，外层为162.21/191.61秒；24次整评约为runner 0.83–1.00小时、外层1.08–1.28小时，还未包括本次新增开销。

[P6 comparison.json](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/go2_pose_01/native/comparison.json)核对的是192个回合记录及其事件字段；[recordings.json](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P6/go2_pose_01/native/video/recordings.json)将3段302帧视频绑定到回放中的env32，其episode为303步、`start_control_step=1`、`starts_at_reset=false`。这些材料能证明回放身份和回合结果匹配，不能证明原评估每一步姿态同轨，也不能弥补缺失的首帧或时序字段。

审计A的低功效计算在其假设下有参考价值：独立代接受概率取1/11、升级翻倍、session只发生一次固定提升时，期望差约0.0028，难以跨过0.01门槛。但概率稳定、最大一次收益、方差和两臂相关性均不是现有数据已识别的事实。因此采纳“高风险，先判设计可行性”，不采纳“任何升级几乎都不可能通过”或“均值至少0.018”的普遍命题。本轮只核算条件算术，没有做功效仿真。

## 5. 联网补查改变了哪些判断

截至2026-09-23，[DeepSeek官方模型说明](https://api-docs.deepseek.com/quick_start/pricing/)写明旧`deepseek-v4-flash`与`deepseek-v4-flash-vision-exp`别名均由V4.1 Flash服务；[当前视觉指南](https://api-docs.deepseek.com/guides/vision/)支持图像输入、按尺寸计量并给出每图1024 token上界。它与[2026-08-21实验版公告](https://api-docs.deepseek.com/news/news260821/)的384上界反映版本演进，不能拿旧报道认定当前生产端点纯文本。也不能仅凭文档保证本地provider已发出图片；本轮未发任何测试请求。

[当前缓存指南](https://api-docs.deepseek.com/guides/kv_cache/)描述持久化prefix unit、共同前缀检测和best-effort命中，不能把旧64-token分块当不变合同。计划只要求版本化实测、保留使用量和稳定核心，不承诺固定节省比例。

[SDS正文](https://arxiv.org/html/2410.11571)确实结合时序网格、接触模式和训练统计；[Video2Reward正文](https://arxiv.org/html/2412.05515)用关键点轨迹相似度列表反馈LLM。它们支持尝试结构化表达及多模态组合，不证明任何场景“先喂派生图一定最好”，也不支持把一般VLM读图当可靠物理测量。

## 6. 为什么保留12个Phase但改了推进方式

用户需要目标和验收明确，同时能随开发边界调整。v1.2因此保留小至中等的职责分段，不预先规定数据库、NPZ布局、类名、固定词表数字或完整API工作流。

稳定的是四个合同和全局硬门；每阶段启动前才锁定对应实现范围、数据、预算、效果门槛与停止规则。薄纵切先获得早期信息价值，Phase 9再选择集成方案，Phase 10验证真实闭环，Phase 11单独判断搜索收益。早期pilot不自动构成CORE完成，探索性完成也不自动构成收益PASS。

新增早期A/T输出及媒体开发探针必须进总账，不能悄悄塞进v1.1的312/216报告数。原600/300等配额是旧矩阵假设，v1.2要求阶段前明确分配有限总预算；默认16/96候选机会及18/104实际启动上限保留，任何调整都必须发生在相应研究开始前。

## 7. 文档复核记录

已完成两轮分方向复审及主审复核：

- 数据/行为/媒体：收窄不可观测物理词条，保留全局旧指标与新逐脚规则的版本区别，明确2-min不因训练日志缺失而重训。
- 记忆/上下文/调用：恢复可信token计量、当前parent与最近trial必留、只读回放不发调用及冲突后不可静默换parent；校正历史token低估的对应帧名称。
- 实验/统计：复核文件去重、机会分母、整评与episode单位、回放记录粒度，以及探索性完成与收益PASS的区别。
- 文档：Phase 0–11连续，12个Phase均有目标/交付、严格验收与可延后项；当前三份工作文档32处本地链接均有效，围栏配对、表格行完整，无冲突标记或替换乱码。
- 保留：v1.1归档hash与原文件一致；两份用户审计的文件大小、修改时间均与读取时相同，未编辑原审计。运行代码未改动。

以上范围内未发现仍需修改的实质性矛盾。此结论仅是文档、源码与保存证据的审查；实际计量、采集、视觉、闭环和收益仍须按各Phase验收，不能视为已经通过。

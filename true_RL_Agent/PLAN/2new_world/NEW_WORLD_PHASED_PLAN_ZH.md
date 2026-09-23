# 新一轮建设计划：统一后端、有效评测与 Go2 × Push Recovery 闭环

编写日期：2026-09-05。核对基线：本地及服务器 `0773a9a815e5a0ccd151f351fca0e369f157fa05`。

本文是开发与实验的建设计划；实际执行状态以各 Phase 验收为准。清理、身份/恢复和统一后端分别见 [P0](P0/ACCEPTANCE_ZH.md)、[P1](P1/ACCEPTANCE_ZH.md)、[P2](P2/ACCEPTANCE_ZH.md)。P2 工程 GO，旧/新后端数值等价未成立。2026-09-09 的 [P3 当前验收](P3/ACCEPTANCE_ZH.md)在固定 C2 标定基底、重新冻结和测量后，记录 G1/Go2 均通过原 public-accept reference 资格门；目标全量 1640 项零失败、零跳过，部署及资源收尾完成。具体修订见 [卷间构造审计](P3/COHORT_ALIGNMENT_ZH.md)，此前工程修复见 [回归复审](P3/REGRESSION_FOLLOWUP_ZH.md)。当前资格通过不等于各卷区分力相同或全部泛化目标完成，不能根据已审阅的 accept/sealed/final 继续改题求通过。2026-09-10 的 [P4 工程验收](P4/ACCEPTANCE_ZH.md)及[复审](P4/AUDIT_FOLLOWUP_ZH.md)已闭合共享编排、成本、恢复和终态；复审最终代码本地与目标各跑完整 1599 项，目标零失败、零跳过，已同步并复核原实验保护；共享正式 CLI 要求显式 task/cell，Stair composition/reference 默认仍为 G1 V3。新物理研究需冻结实测成本合同。2026-09-11 的 [P5 当前验收](P5/ACCEPTANCE_ZH.md)及[历史回放复审](P5/REPLAY_CONTEXT_FOLLOWUP_ZH.md)已闭合终态恢复、同源反馈、跨代记忆、机制准入和冻结正式报告回放；只读沿用历史 authoring 身份，实际续跑保持严格。未选择 trial 的已测 dev 与失败原因已恢复，详细反馈限制为当前最优及最新已测项。两端各一次完整 1611 项零失败/错误；目标仅两项显式 opt-in 物理检查跳过，前一批专项证据保留其原范围。本轮 515 文件源码一致性、原实验保护和资源收尾均通过。工程 GO 不代表策略收益。P6/P7 尚未完成。

## 1. 对目标的确认

项目的核心是一个**受控的 RL 代码研究与验证系统**：LLM 在冻结条件下提出 reward program 的机制改动，可信系统负责验证、随机初始化训练、评测、选择、接受和账本。机器人与任务构成二维矩阵；新增机器人应主要增加形态事实和薄绑定，新增任务应主要增加任务语义，二者都应复用已有的实验生命周期。

本轮建设包含五项明确交付：

1. 重做已经缺乏区分力的评测指标及题池资格化，使目标仍能测到改进，并建立适用于不同机器人、不同任务的评测方法。
2. 对正式实验记录的未解决项逐项处置，修复当前仍成立的问题，清除无用兼容面；不将观察性假说擅自变成新接受门。
3. 将训练、评测和原生录像统一到已验证录像能力的运行时路线，最终只有一套活动生产后端。
4. 以 **Extension A：Go2 × Push Recovery** 作为真实第二任务，完成必要的编排解耦、reference 资格化和多代闭环。
5. 实际删除无价值的临时文件、旧执行逻辑和保护范围外的未引用 checkpoint，使入口、文档和证据更容易理解。完整 15 rounds 实验涉及的 checkpoint 保留；清理贯穿开发，最后集中验收。

本计划不扩展六足机器人、LLM curriculum 权限、通用工作流引擎或新的仿真物理引擎。未来机器人通过边界与尺度设计获得接入空间，本轮用已有 G1/Go2 验证这些边界。

## 2. 已核实的起点

### 2.1 本轮实验成立的结论与局限

正式会话是 `formal-go2-stair-15round-20260901-c6ab8ba`，活动格子为 `stair.go2_12dof.unified_v2`。15 代结果为：1 accepted、9 formal rejected、1 no selection、4 incomplete；唯一接受的是 Gen8-A，11 代形成完整科学判决。

| 证据 | 已核实事实 | 对下一轮的影响 |
|---|---|---|
| Reference 三套卷 | `J_ID=F_c1=F_c2=1`；public-dev 的 `J_primary=0.934438`，成功率为 `187/216` | 高分不等于整个任务接近解决，已有大块目标权重没有上升空间 |
| 当前评分契约 | cell 分数为 `0.5×success+0.5×progress`；三个 OOD family 等权；`J_primary=(2×J_OOD+J_ID)/3` | 本次全部 family 合格时，已饱和的 ID、F_c1、F_c2 合计占主目标 `7/9` 权重，只有 scalar family 的 `2/9` 权重仍有明显余量；此比例由现行公式推导 |
| 最终配对确认 | Reference `425/504`，Gen8-A `441/504`；配对 J 增量 `0.015778`、SE `0.002853` | 支持本次新评测 seed 上的平均改善 |
| 最终分层 | height `47→60/63`，tread `44→49/63`，joint `19→17/63`；其余五层均 `63/63` | 需要保护困难层，不能用平均分掩盖局部退化 |
| 泛化范围 | 最终确认换了 seed，仍使用原有 24 道 sealed 几何；未做多训练 seed 复现 | 未证明新几何、未见机器人或训练稳定性的普遍泛化 |

本次已检查两份 CSV 的 48 行评测指标与 384 行分层记录：成功/失败计数及分层加总全部闭合。服务器报告、最终 Result 和两份 final 原始 summary 的 SHA256 与结果文档一致，报告记录的 11 项检查均为真。本次没有重新执行耗时的完整终态 replay，也没有重新跑 GPU 实验；完整 replay 通过属于本轮保存的历史验收证据。

### 2.2 当前工程边界

- `agent_core/generation.py` 已有双 lane 执行、公开筛选、私有确认和一次原子提交。解耦应复用这一条链。
- Stair 的 `generation_session.py` 当前 9,449 行，成本模块 2,964 行，混合了大量会话、恢复、成本和报告职责。行数是定位线索，验收以职责和实际依赖为准。
- Push 的 `robot_binding.py` 当前只注册 G1-29 legacy cell，并声明没有 reference-program Genesis；Go2 Push、non-legacy policy-I/O 内容身份及对应 generation 装配尚未闭合。
- 正式运行环境为 Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3 系列、RSL-RL 3.0.1、Torch 2.7。录像环境的实测为 Python 3.12、Isaac Sim 6.0.1、Isaac Lab `v3.0.0-beta2.patch1` 源码路线、RSL-RL 5.0.1、Torch 2.11。Lab 发布标签与包内 distribution version 不是同一种版本字段，必须分别记录。
- 现有 Sim 6 legacy 录像已有 manifest。今天新增的 Lab 3 对齐录像脚本仍位于父仓库 `.tmp`，明确使用展示用更宽地形、选定 cell 和重复环境，并保留与 formal evaluation 的差异声明；它是迁移取证材料，尚非统一生产实现。
- Formal reference 仍依赖输出树外的 P5 `best.pt`。本次确认文件存在且 SHA256 为 `fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`，不能按“旧 P5 目录”直接删除。
- `README.md` 仍描述 Phase 0，部分 runbook 指向旧单 lane 入口。文档与废弃执行路径需要一起收敛。

未解决清单共 **55 项**。其证据涉及多个历史会话，修复时必须按完整 session/trial/source 身份区分；作废或未完成会话只提供诊断证据，不并入本次 15 代科学结论。

## 3. 全程约束与统一验收规则

### 3.1 保持一份权威事实

| 所有者 | 本轮应承担的职责 |
|---|---|
| RobotProfile / RuntimeAdapter | 资产、关节/刚体事实、角色候选、标称尺度、状态读取、动作执行及机器人配置 |
| TaskPlugin 及 task-owned 组件 | 任务成功/失败、扰动或地形分布、评测题池、指标、诊断、字段语义、任务反馈 |
| RobotTaskBinding / ResolvedCellManifest | 角色选择、必要尺度映射及机器人与任务组合后的不可变执行身份 |
| AgentCore / runner | 有界 authoring、生命周期、资源、成本、seed 隔离、固定作业、通用判决执行、checkpoint 和 ledger |
| LLM candidate | 当前协议授予的完整 reward program；训练分布、PPO、预算、seed、评测和接受权继续由系统持有 |

共享层可以理解阶段、具名分数、配对样本、成本能力和结构化状态，不能解释 Stair 净空、Go2 足端或 Push 恢复条件。接口从 Stair 和 Push 的真实共同需求中提取，优先使用现有 composition、phased adapter、receipt 和 registry；不再建一套 runner、ledger、解析器或配置真相。

### 3.2 版本与实验隔离

- 旧 15 代协议、ledger、原始 summary、接受结果和 source/checkpoint 身份不改写。
- 根据后续确认，新系统完整 15 rounds 实验涉及的全部 checkpoint 保留，包括中间模型、被拒绝候选和相关 reference；不能仅因当前没有引用就删除。本约束优先于 P0/P8 的一般清理及 retention 规则。
- 运行时、奖励 ABI、任务语义、题池、指标、训练支持、PPO、预算或 seed 协议的实质改变，必须形成新版本与新的比较域。新评测不能给旧 accepted 记录补一套“更好看”的分数。
- 旧 checkpoint 可作迁移对照和 calibration anchor；新协议 reference 应按新冻结条件重新资格化。需要重新训练时，从随机初始化开始，不把旧 checkpoint 偷换为新 reference 或 candidate warm start。
- 先验公开的任务规则可用于 authoring；public-accept、sealed、final 的私有测量与原始证据不回流给 provider。已在本轮人工分析过的题目不再声称为下一轮“从未见过”的 holdout。

### 3.3 每个 Phase 的退出门

每个 Phase 必须留下一个简洁验收记录，注明：实际变更、问题编号处置、source/protocol/runtime hash、测试命令及结果、真实实验 receipt、剩余限制和 GO/NO-GO。证据引用原产物，不复制成多套报告。本文及后续验收记录集中于本目录；产品操作说明只维护一个当前入口。

统一采用以下退出规则：

1. 对应行为有正反例验证；失败、缺失、被跳过的数据不得填零或伪装成完整评测。
2. 身份、权限、计数、public/private 边界及旧证据读取回归全部通过。关键测试因缺环境而 skip，不算该 Phase 已验收。
3. 涉及仿真、训练、录像和真实 provider 的改动，必须取得相应真实运行证据；fake runner 只证明编排。
4. 实验前在协议中冻结数值阈值、种子数量、样本量、停止规则、GPU/API/磁盘/墙钟预算；带待定参数的协议不得进入正式运行。
5. 每次正式启动检查作业归属、GPU/lease、源码和依赖；结束核对全部子进程、资源及成本。当前其他任务或 `.tmp` 中仍在使用的实验不纳入本次清理。
6. 工程 GO 与 `improvement_observed` 分开。有效执行后没有提升可以是科学阴性结果；无法执行、证据不闭合或 reference 不可学必须 NO-GO。

## 4. Phase 顺序与工作量边界

| Phase | 主要交付 | 工作量边界 | 前置条件 |
|---|---|---|---|
| P0 | 基线与依赖清单，首批确定无用内容清理 | 小；盘点及明确可删除项 | 本计划进入执行 |
| P1 | 身份、回执与恢复闭合 | 中；不改变评分和搜索策略 | P0 GO |
| P2 | 单一训练/评测/录像后端 | 较大；分 ABI 与真实执行两次验收 | P1 GO |
| P3 | 有区分力的指标和跨机器人评测协议 | 较大；分设计与校准冻结两次验收 | P2 GO；设计可提前，冻结测量必须用目标后端 |
| P4 | 由两任务驱动的最小编排解耦 | 中；生命周期与任务装配边界 | P3 合同形状稳定 |
| P5 | Authoring、prompt、memory 与机制证据修复 | 较大；分信息一致性和候选准入两批 | P1/P3/P4 GO |
| P6 | Go2 Push 任务语义与 reference 资格化 | 中；真实任务及基线，不跑长周期搜索 | P2/P3/P4/P5 GO |
| P7 | 真实多代 LOOP、复现与泛化验收 | 中；主要成本是 GPU/API | P6 GO，各正式协议冻结 |
| P8 | 最终删除、入口文档与发布验收 | 中；清理和完整回归 | P7 合法收尾 |

先统一后端，再冻结新指标的测量，避免将动力学迁移收益误认为 reward 改进。先给出两任务合同，再提取编排，避免仅把 Stair 大文件搬进共享层。清理随替换发生，P8 不承担全部历史债务。

## P0：建立可开发的基线与删除边界

**工作：**

- 固定当前代码、三个相关机器人任务格子的活动状态、本次 15 代证据及录像迁移材料，区分正式产物、失败诊断、历史 replay 和在用临时材料。
- 建立一份轻量清单：文件/目录、用途、调用或引用来源、是否在用、处置与验收依据。后续授权已将范围扩展到本地父目录旧 MVP、下载缓存及 `/data/yuyangliu` 下确认闲置且可重建的副本；保持现有 Git 根和当前项目路径，保留必要工具、独立项目与在用环境。实际状态统一见 [扩围清理记录](P0/OUTER_CLEANUP_ZH.md)。
- 优先删除确认未引用且无诊断价值的临时输出、重复副本和失效占位物。源码导出、旧 checkpoint、冻结 predecessor 和外部 anchor 先完成引用闭包核对。
- 给后续各类验证设定分开的预算池：迁移对照、calibration、reference、候选搜索、复现、final；禁止从 final 预算临时补搜索。

**验收：**

- 能从一个入口定位 active cell、真实 launcher、reference、accepted head、ledger 和复算依据。
- 未解决清单 55 项全部有初始处置，已关闭项不重复开发；本次之外的未提交修改有独立记录。
- 首批删除有清单及前后文件数/字节数，仍被 receipt、manifest、文档或恢复流程引用的实物没有悬空。
- 跑出本地完整测试基线和目标运行环境可执行性清单；现有失败必须明确归属，不能通过后续删测试掩盖。

## P1：先修证据身份、执行位置与恢复

**工作：**

- 将 scientific identity 与 execution placement 分离，并保存可诊断的版本化 snapshot 实体。协议、资产/源码、物理/控制设置、reward/observation/action、PPO、seed 与预算仍属于科学身份；路径、PID、boot、GPU ordinal 等是执行位置证据。影响数值行为的软硬件差异仍需兼容性验证，不能简单从 hash 中删除。
- 新 session 建立时把唯一 reference anchor 复制为内容哈希校验的会话自有只读实物；旧会话维持原链，迁移必须保留原引用可解析证据。
- 修复 incomplete trial 恢复的 train receipt 丢失、final metadata 身份副本、planned URI 冒充实物、reference-head 终态覆盖。所有恢复都从既有 journal 重建，保证一次提交、一次计费。
- 跨 boot 恢复重新验证当前主机/资源，将新 readiness 与旧 attempt/teardown 证据以明确关系续接；child admission 验证父会话冻结的 source identity，禁止混用磁盘新代码与父进程旧模块。
- 对浮点派生统计定义版本化 canonicalization；计数尽量由整数事实推导，原始身份仍精确验证，统计比较的容差不能替代 receipt/hash 验证。

**验收：**

- 在正式支持的 Python 版本间复算同一原始证据，取得按新契约一致的规范身份；旧 reader 按旧版本读历史记录。
- 注入 train 后退出、eval timeout、trial 已封而 generation 未封、重复 resume、重启换 boot、源码漂移、路径迁移等故障：无重复 train/eval/收费，既有训练投影不丢，非法漂移在作业启动前拒绝。
- final 的 candidate-head 和 reference-head 两条链均覆盖；reference-head 双 job 身份不同、checkpoint 相同，配对差近零且 head 不动。
- 私有实物列表全部指向确实存在且身份有效的产物；未产生的计划输出明确标记，不要求伪造空文件。

## P2：将训练、评测和录像统一到同一后端

**目标路线：** 以当前已具备原生录像证据的 Isaac Sim 6.0.1 / Isaac Lab 3 / PhysX 路线为迁移目标，在受控运行环境中固定准确源码、依赖和资产版本。RSL-RL 5 的训练兼容性必须实测。最终生产入口只能选择这一套已验收实现，历史 reader 不等于保留第二条生产后端。

**P2-A：迁移与 ABI 对齐。** 逐项核对 observation 顺序与归一化、joint/action 顺序和缩放、actor/critic、checkpoint 容器、reset/RNG、四元数约定、控制 dt/decimation、接触传感器、reward rate 积分及 PPO timeout/bootstrap。训练、eval、video 复用同一 resolved cell 和环境构造；RobotProfile 不再维护重复 PPO 表，机器人执行来源与 task reference 来源按原所有权分开。

**P2-B：真实训练和原生录像。** 从随机初始化完成短训练、checkpoint 保存/加载/续跑、评测和录像，再完成至少一个完整预算的代表性训练任务。录像只是该环境的呈现选项；允许训练关闭相机，允许显式 train/eval profile 差异，但所有差异要有同源配置和身份。展示用加宽地形、多机器人排布不能混入科学录像/评测模式。

**验收：**

- 同一观测输入下，迁移 actor 的动作与旧 checkpoint 解释器在预设容差内一致；训练状态、normalizer 和 optimizer 的可迁移性分别报告。旧 checkpoint 无法等价转换时只能作诊断 anchor，新 reference 重训。
- 从相同初态与动作做一步及短程状态、奖励、终止比较；在预注册 public calibration cells/seeds 上比较旧/新运行时，报告分层偏移和容差。GPU 长程逐位相同不是默认要求，但 unexplained 语义偏移必须解决后才能 GO。
- 在目标运行时上，录像开/关共享资产、题面 digest、控制 dt、状态机和 checkpoint。固定小样本比较计数；多 seed 比较若存在数值分叉须满足事前容差。不得由 video exporter 重新分类结果。
- 真实验收包括：Go2 Stair C0/C1/C2、G1 Stair 代表样本、双 lane 并发、完整训练保存、失败清理和连续录制/结束。视频必须可完整解码，帧数、仿真时间、播放时长关系正确，原生进程退出且无遗留 GPU 子进程。
- 日志按 job/lane 唯一路径输出。新 Gym ID 不误标机器人；新后端身份不可伪装成旧 epoch。旧入口、独立 actor/地形复制实现的退役清单交 P8 清除。
- 完成 G1-29 Push 的新运行时最小执行检查；若其 vendor 依赖尚不兼容，必须在本 Phase 解决兼容边界，不能以继续保留旧生产后端结束迁移。

## P3：重建指标区分力与跨机器人评测方法

### P3-A：指标和难度设计

本 Phase 同时处理**分数设计、题池难度与机器人尺度**。单纯改权重无法修复满分题；只增加更难题也可能制造 reference 全零和不可学习目标。

2026-09-22用户后续修订reference的总体目标：具备可观察的基础能力且有改进空间，不仅凭固定成功率区间判断。当前代码修订仅落实于Push新外力reference准入，默认 `observed_outcomes_v1` 要求完整有效dev证据中存在成功与失败；自动门只排除全失败/全成功，近饱和仍看分层结果，不另立百分比。Stair既有P3的20–80%参与冻结题池learnable-cell/mask及评分，保持原样；后续新题池可依新目标另行声明，不据此重写旧评分域。原资格结果和机器回执保留，具体见[准入修订](P6/REFERENCE_ADMISSION_REVISION_ZH.md)。

- 保留明确的任务成功率及原始计数，progress 作为未完成行为的连续信息。重点比较当前 success/progress 混合分与“成功优先、质量补充”的小规模方案；质量只选择任务需要且能可靠测量的少量指标。
- Stair 优先考察合法完成的效率、困难层成功/进度及最差层表现；Push 优先考察施扰后恢复成功率、恢复时间及恢复窗口内速度/姿态偏差。完成时间需要处理未完成 episode，禁止只在成功样本上报均值而奖励筛掉困难样本。
- 对原有饱和层保留能力守卫，同时建立仍有可测余量的难度梯度。ID 不永久降为仅 guard：不同机器人的 ID 可能仍有显著改进空间。后续新题池的目标权重与基础能力、改进空间证据由任务定义，逐机器人采用同一方法核实；不以固定成功率区间替代这些事实，也不追溯修改已冻结的Stair题池与mask评分。
- 为 joint-OOD 等关键困难层定义独立的报告与非劣保护，防止 height/tread 的增益完全遮蔽交互退化；不为每个 diagnostic 增加接受门。
- 尺度从 RobotProfile 的已有事实派生。Stair 可使用高度/有效支撑尺度、踏深/身体水平尺度，Push 可使用质量归一化冲量或归一化速度扰动与恢复时间。具体分母及适用范围需要任务论证和实测，不能把 G1 的米/牛顿阈值平移到 Go2，也不能把一个 `task_scale` 同时解释成地形与 reward 缩放。
- 每项同时保存物理单位值与规范化值；报告 cell 内配对改进和分层结果。跨机器人分数可比性需要单独资格化，不能仅因规范化就将所有 episode 混池。

### P3-B：校准、冻结与泛化边界

在公开 calibration 池上，使用固定 anchor 和预算检查 reference 可学性、指标饱和/地板比例、方差和区分能力；G1 与 Go2 至少各有真实测量。先冻结候选方案与校准退出条件，再按规则形成正式题池，禁止用 Gen8 或新 candidate 的胜负挑指标。

需要明确区分：同分布新 seed、新几何实例、连续强度外推、组合/类别留出、机制变化，以及新机器人接入。只验证到哪一层，就声明到哪一层。Stair 的 C2 若继续作为训练外机制分布，必须验证真实训练暴露为零。

**验收：**

- 签发新的 task/cell protocol、完整评分公式、具名 guards、配对不确定性规则、reference qualification、预算及 suite/hash。所有目标、聚合顺序、分母、无效/缺失数据处置和数量阈值明确，无执行阶段的待定项。
- 每个被声明为优化对象的 family，至少有通过预注册 learnability 与余量标准的有效题层；饱和能力保留守卫。全零/全满且没有区分力时 NO-GO，不能进入 LLM 长跑碰运气。
- 用成功、失败、到达但违规、短时异常后成功、提前终止等反例验证指标；诊断与成功可共存，不能自动当失败。原始 outcome 整数计数可独立重建所有聚合分数。
- G1/Go2 的尺度只需改 profile/binding 和任务校准数据，TaskPlugin 算法没有新增机器人名称分支；确认 support-effector 候选属于实际可支撑 body 集合。
- Dev、accept、sealed 的目标题面与 seed 按预声明规则隔离；难度可比性须用实测检验，并明确尚未达到的部分，不能把不同几何自动视为同难度。C2 保持标定基底、留出表面相位；其配对平面控制允许复用且不计入主目标，见 [卷间构造修订](P3/COHORT_ALIGNMENT_ZH.md)。新增 final holdout 在搜索前完成生成规则和隐藏身份预承诺，包含未消费的几何/扰动与 seed；协议冻结后一次性揭盲，不能依据搜索结果换题。已经审阅过的题面即使换了 hash，也不能重新宣称未见。
- final 身份绑定独立 study/cohort namespace；同会话 resume 复用原 cohort，不同新研究不意外复用。主动重复某 cohort 时明确标记 replication，不再声称“从未使用”。

## P4：用 Stair 与 Push 提取最小共享编排

**工作：**

- 基于现有 `ResolvedTaskComposition`、`execute_fixed_width_generation`、生命周期与 receipt 接口，让正式 session 的基础准备、预算预约、lane 运行、恢复、终态确认和报告外壳成为一套任务中立执行路径。
- Stair 保留题池、reference 规则、指标与分层反馈；Push 给出真实 phased-evaluation/cost/reference 合同形状。先完成 Push 的最小合同装配和 fake-runner 通路，P6 再用真实环境完成任务资格化。
- 将 cost/slot 分派改为所需能力和显式合同，消除用 Stair/V3/V4/V5 继承身份猜测执行拓扑。新 final plan 字段使用真实含义；旧 V1/V2 字段只由版本 reader 解释。
- 将只读历史 loader、一次性验收工具和 test-only session 从活动任务模块移出；不将 9,000 行文件原样搬到 shared 包。
- 在现有 acceptance contract 上实现确定性早拒与成本预约：public-accept 已使接受不可能时可跳过 sealed。同一 verified snapshot 内复用报告所需验证结果，消除重复扫描和重复完整投影，并显示终态阶段耗时。

**验收：**

- 同一个正式入口通过显式 cell 选择 Stair 或 Push；生产编排无任务名/机器人名分支，无两套会话状态机。任务含义只经任务拥有的接口传入。
- 两任务共同通过 accepted、rejected、no-selection、authoring incomplete、runtime failure、基础设施失败、resume、reference-head final 等编排测试。
- 早拒与原完整判决在相同可用证据上严格一致；缺失 sealed 明记 `not_run_due_to_early_reject`，不伪造完整评测或成功门，且私有拒绝细节不回传 prompt。
- 真实执行拓扑、保守 reserve、实际计费和释放分别可复算；取消不用的 operation reserve 不改写历史成本。
- 历史报告重建值不变；缓存仅限内容身份不变的当次验证视图，新的独立 replay 仍重新验证输入。记录优化前后 wall time/扫描次数，不能通过省略权威验证换取性能。
- 后续接入相同能力形状的新机器人不需修改共享编排；实际改动清单证明哪些文件被抽取、删除或留在任务侧。

## P5：修复 authoring、反馈选择与机制准入

**第一批：信息一致性与有界记忆。**

- Source、code delta、详细 public-dev feedback 和 runtime summary 共用同一个 trial identity。真实本轮 latest-failed 与历史未接受源码使用不同身份；裁剪后的 catalog、lineage 和 recoverability 从最终 payload 派生。
- 分开保存尚未关闭的 candidate repair 与最新 provider transport failure；跨代保留有界 lane status、验证成功但未训练的 signature、失败阶段/原因及所有已评测 lane 的机制结果索引。
- No-selection 和 sibling 也提供有界 family/stratum 变化摘要，不复制完整历史。不完整候选不作为性能负卡，基础设施失败不变成机制失效。
- 删除重复逐 cell success 别名及 deprecated payload `public_context`；补齐 task-owned 成功条件、诊断语义、坐标系、opaque 轴、接触 onset、标称尺度及 `step_reward=control_dt×sum(weight×clipped_raw)`。保留权威计数与有价值的 progress/分层证据。
- Provider 的实际 context/tokenizer 能力和 requested output 联合可达性进入 policy/receipt；复用同一 payload 对象做 compact JSON。分别处理输入超窗、empty-final 和 reasoning 耗尽输出，修复现有有限重试策略，不能靠无条件加 attempts。

**第二批：候选质量和真实作用证据。**

- 在已有 canonical delta 索引上执行同 parent/seed/comparison domain 下的历史 exact 程序去重；名称不能构成新机制。参数化负向 suppression 与精确重复去重分开。
- 按现行 reward-only 研究权限识别纯改名、numeric-only 和真实结构变化。仅数值修改进入 repair；若未来允许参数搜索，必须独立版本化授权，不能以改名绕过当前规则。
- 同一 task smoke 增加真实标称姿态、接触过渡、直立/倒置等有限组合，返回有依据的零作用/clip 饱和修复反馈。Fixture 未覆盖只能报未观测，不能宣称数学恒零；不引入符号求解器。
- 同一 runtime recorder 记录采样非零频率与覆盖分母；modify 使用同 transition 上 candidate-parent 差值证明改动有作用。非零与频率不等于因果收益，也不能由 checkpoint 分叉代替。
- 在训练 receipt 闭合后复用同一 runtime-effect evaluator，对确定不合格项在 public-dev 前止损。对采样未覆盖的稀疏信号明确证据局限，不建立第二套阈值。
- 成对 authoring 的失败隔离在下一 generation contract 中预注册：默认两 lane 都获得各自既定尝试机会，成功候选不被匿名丢失；本轮保留双候选筛选前提。是否允许单 lane salvage 必须另有明确选择与计费语义，本计划不隐式启用。

**验收：**

- 对真实历史失败构造有限回归：shape failure→transport failure→repair、no-selection、sibling 错位、半成功 authoring、opaque 轴误读、错误重力符号、标称姿态饱和、rename exact repeat、numeric-only rename、modify 差分恒零。
- 每条可编辑源码都有同源反馈；不存在“inline 但实际未发送”或 provider 无法访问的恢复来源。
- 55 项中属于本 Phase 的问题逐项关闭。初代、晚代和修复帧容量测试通过，公开安全扫描及私有字段 canary 测试通过；实际 provider probe 证明策略能发出有效完整候选。
- 作者假说、静态事实、runtime 作用证据和性能结论有明确层级。运行期非零统计不直接产生性能正卡或接受权。
- 没有新增无限历史、通用张量类型系统、自然语言证明器、重复源码分析器或新的接受真相。

## P6：闭合 Go2 × Push 的任务与 reference

**工作：**

- 复用 Go2 RobotProfile、资产、RuntimeAdapter 和机器人 PPO 配置；新增 Push-owned capability、薄 binding、non-legacy manifest 和实际 observation/action 内容 hash，承接 P1-B7。Push 不消费的逐足/台阶语义不进入其协议。
- 先核实实际施扰方式：当前 overlay 配置 velocity-range 事件，capability 文案又含 external force。新协议选择并准确声明速度重置、速度增量或力/冲量中的实际机制，记录施扰前后状态与单位，不能把这些方式混称为同一个 push。
- 冻结站立/指令、施扰方向/强度/时机、恢复窗口、稳定驻留、跌倒/姿态/越界与多次施扰计数。机器人尺度由绑定派生，任务定义不写死 G1 身高或 Go2 足数。
- 定义 Push 的公开训练/评测信号边界。`push_active/push_vec_public` 若仍为 zero-fallback 则保持不可用声明；若启用，只公开协议允许、已经发生的事件信息并验证非零接线，不泄漏未来扰动计划或私有题目。
- 按 Push 的恢复目标预声明接受门和独立验证域。现有 reference_v1 sealed 仅有绝对分数下限，没有 public→sealed 下降门；P4 工程闭合不证明候选对新扰动或新 seed 泛化。P6 应先明确需要约束的失效模式，再冻结对应判据，不能按已观察的 accept/sealed 成绩反复改门，也不机械照搬 Stair 阈值。
- 补齐 random-init reference-program Genesis、固定预算训练、public-dev/accept/sealed、失败分类、task prompt/memory、科学录像和 phase receipt，走 P4 的共享 session。新 Push reward 权限明确为本协议所授予的 reward program，禁止由 Stair 私有类代建。

**验收：**

- 真实 Go2 上验证无扰动、已知方向与强度扰动、成功恢复、未恢复、跌倒、连续扰动等情形；测量注入确实发生，分类整数计数闭合。全程没有被推的 episode 不能计作“抗扰恢复成功”。
- Reference 从随机初始化完成冻结训练预算，具有真实基础能力和改进空间；当前Push新准入不要求成功率进入固定区间，完整 seed/manifest/checkpoint/source/receipt 链仍须可复算。dev自动门要求完整有效证据中有成功与失败，accept/sealed须观察到成功；相对改进、统计判据及accept→sealed最大0.1下降门保持。近饱和由分层结果说明，不把自动门通过当作余量充分证明。低表现须区分任务接线、不可学分布与策略性能，不能仅凭低于20%判断能力缺失，也不能用无限重训刷过门。历史manifest继续按其原冻结规则解释。
- 同一 reference checkpoint 的目标后端 eval 与原生视频一致，至少覆盖一个恢复成功和一个失败案例；视频只辅助解释。
- G1-29 Push 的历史读取/回放不退化，并在统一后端取得当前执行回归；Go2 Stair 的共享编排与 ABI 回归通过。发现必须复制 TaskPlugin 或修改共享 session 才能跑 Go2 时，返回 P4，不能标记扩展完成。

## P7：真实 LOOP、训练复现与最终泛化验收

**工作：**

- Go2 Push 先完成最短的真实双 lane 端到端闭环，证明 provider→verifier→TRAIN→public-dev→selector→formal→ledger→下一代 feedback 全部有实物。
- 随后按冻结协议运行 Go2 Push 的 15 代、width 2 正式会话；预算采用 Push 资格化后自己的完整训练成本，不照搬 Stair 数字。早拒/incomplete/resume 仍必须合法关账。
- 对新指标/新后端下的 Go2 Stair 做独立短会话回归和固定方案评测，证明旧主任务仍能经共享入口运行；不默认再付一轮 Stair 15 代成本。
- 对最终冻结的 reference 与 final program 做至少 **3 个独立训练 seed 的配对复现**，每个 checkpoint 使用至少 **5 个配对评测 seed**；预算与样本计划必须在正式搜索前预留。新 Stair reference/选定程序采用同一复现口径，重复的合格训练证据可按预注册身份规则复用。
- 在候选、协议和停止决定冻结后，一次性消费预承诺 final holdout。对使用过的 final 证据不再回头调候选或重新挑选 best。

**验收：**

- Go2 Push 15 代全部有合法 closure，接受/拒绝/无选择/不完整分开统计，current-best 仅由接受记录更新。若全程没有有效候选评测，只能记 execution completed，不能称为任务 LOOP 已验证。
- 原始 per-seed/per-stratum 成功 n/N、主目标、关键 guard、失败类型、成本、选择的不确定性与未测部分齐备；缺失不是零。
- 报告分别说明：工程闭环是否通过、单次搜索是否观察到提升、多训练 seed 是否支持稳定改进、final 新题是否支持泛化。以训练 seed 为独立复现单位，不把同 checkpoint 的大量 episode 当独立训练复现。
- 冻结候选在多训练 seed 配对结果中的改进置信区间和困难层 guard 满足预注册规则，才允许声明“稳定改进”；不满足则如实给出不确定或阴性结论。新机器人泛化只声称到已实测的矩阵格子。
- Final 不拥有 acceptance、feedback 或重新选择权；报告、成本、ledger/head、checkpoint 及资源闭合通过独立只读 replay，现场进程/GPU/lease 验收完成。
- 预算耗尽或无提升时按预注册规则停止。增加代数、改题、改阈值、扩大训练分布需要新研究协议，不作为本 Phase 的自动补救。

## P8：删除无用内容并完成可读性验收

**删除原则：有必要的最小证据保留，无价值内容直接删除。** 不把整个旧目录搬到一个巨大 archive 来代替清理；Git 已能保存的旧实现通常直接退役，只有旧证据解码确实需要的最小 reader 保留。

**工作：**

- 删除无生产或历史读取消费者的导出、test-only 产品模块、旧单 lane wrapper、Phase-C/D 生成/探针脚手架、一次性 P6-B 审计入口、deprecated prompt 别名，以及已被统一实现替代的录像/训练复制逻辑；对应无价值测试一同退役，保护合同的测试迁移到新入口。
- 先把冻结 manifest 需要的最小 validator 与旧生成栈分开，再删除生成侧脚手架。仍被 transition decoder 调用的 legacy normalizer、V2 所引用 V1 的协议/binding/语义证据，必须在其真实依赖仍存在时保持可解析。
- 每次新增活动证据格式，在该 Phase 记录旧版本的具体消费者、旧 writer 退出条件与 reader 退役条件；已有支持的冻结审计入口满足需求后，删除重复实现，不累积无人使用的兼容层。P1 的首份清单见 [版本边界](P1/AUDIT_FOLLOWUP_ZH.md)。
- 对本项目拥有且已结束的输出做引用扫描：完整 15 rounds 实验涉及的全部 checkpoint 单独保护；其余输出保留必要 reference/accepted/final checkpoint、receipt 实际引用文件、复现样本与不可替代失败诊断，只删除保护范围外且已确认无引用、无诊断价值的模型、重复拷贝和失败半成品。训练收尾的 retention 必须遵守同一保护清单。
- 用已有 lease 所有者清理逻辑处理确认死亡且身份不匹配的锁，不清空共享 lease 根目录。退役旧运行环境前验证没有活动任务及必要依赖；生产文档中不保留旧后端作为备用路线。
- 重写 README、系统概览、任务扩展指南和 runbook，只展示当前单一启动、评测、录像、恢复和审计入口；修正 Extension A 标签及旧语义链接。历史报告保留原数值与版本声明。

**验收：**

- 实际提交删除清单，逐类给出源码/脚本/文档/产物文件数、代码行数与磁盘字节的前后变化；不能只加 ignore 或只写删除建议。
- 正式证据引用零悬空，保留的历史 lineage 可读取，reference 不依赖已删除路径；清理前后的 ledger、权威 receipt 和 accepted checkpoint 身份不变。
- 新同事仅依据当前文档即可定位机器人/任务/绑定/编排所有权，并用唯一入口完成 list/resolve、短验证、eval/video 和 terminal audit。
- 全量相关测试、独立历史 replay、新后端最小真实回归及文档命令检查通过；不以行数减少作为唯一通过标准，不删除仍必要的审计证据制造“整洁”。
- 55 项全部具有最终状态和证据：`fixed`、`retired`、`already_fixed` 或 `retained_with_required_consumer`。最后一种必须写出仍需要的具体消费者及理由；没有无编号、无理由的“以后再说”。

## 5. 遗留问题如何进入开发

问题详情统一引用 [PHASE6_UNRESOLVED_FOLLOWUP_ZH.md](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)，本计划不再复制逐项清单。上面的 Phase 按根因承接：身份/恢复归 P1，后端与机器人配置归 P2，指标/尺度/holdout 归 P3，编排/成本/报告归 P4，作者与反馈归 P5，Push 特有缺口归 P6，删除和文档归 P8。

P0 复核全部问题的当前适用性；各 Phase 的验收记录只列原编号、处理结论及验证引用，问题背景继续读原文件。一个根因可合并关闭多项，跨 Phase 项必须等全部相关门通过后关闭。原文中“当前不要修改”针对当时冻结运行，本次实验已结束，后续按新版本边界推进；若当前代码已经解决，则以证据关闭，不重复开发。

需要特别保留的判断是：无行为消费者的 `task_scales` 要么进入唯一可测的尺度链，要么从新活动身份删除；机器人行重复的任务不变量应收回任务侧；纯诊断相关性不能直接升级为接受目标。清理项也不能一律标记“历史保留”，必须有具体消费者或复算价值。

## 6. 本轮最终完成定义

只有同时达到以下条件，才能宣布本轮建设完成：

- 活动训练、评测和科学录像使用同一已验收运行时、环境构造与身份链。
- 新评测对 reference 有经测量的区分力，任务成功、质量、诊断和困难层保护各自清楚；跨机器人只共享方法和事实来源，不强行共享同一物理阈值。
- Go2 Push 通过真实任务、reference、双 lane、多代、失败恢复和终态审计；Stair 及 G1 Push 的必要回归保持有效。
- 共享生命周期没有新的 Stair/Go2 特例，也没有复制一套 Push session。
- 修复项有证据、阴性结果诚实报告、未测泛化不作声明；新结果不污染旧协议或私有反馈边界。
- 无价值内容已经删除，当前文档只指向当前入口；最小可复算证据仍完整。

## 7. 阅读与验收依据

项目内相对路径均以 `D:\humanoid_agent\true_RL_Agent` 为根；执行记录应保存可迁移的逻辑引用，下面的本地链接用于当前工作区阅读。

- [本轮逐代结果与最终确认](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md)
- [55 项未解决工程问题及动态补充](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)
- [观察项及不同会话的证据边界](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_OBSERVATIONS_ZH.md)
- [矩阵主计划与 Extension A](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md)
- [共享双 lane 执行](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/generation.py)、[组合合同](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/composition/base.py)、[共享会话宿主](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/agent_core/generation_application.py)
- [现有 Push binding](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/robot_binding.py)、[Push 配置](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/config.py)、[施扰 overlay](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/isaac_overlay.py)
- [当前 Go2 Stair score contract](D:/humanoid_agent/true_RL_Agent/configs/tasks/stair_go2_12dof_unified_evaluation_suite_v2.json)、[原 15 代冻结 score contract](D:/humanoid_agent/true_RL_Agent/configs/tasks/stair_go2_12dof_unified_evaluation_suite_v1.json)

制定计划时只读核对的服务器证据根为 `/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba`。终态报告 SHA256 为 `4ad1e53cc72afa0a9fe20ab69a8b5bc41e4ed9249efda4a7958477eb0fc150ca`；final Result SHA256 为 `046582e2665e9c301cc88e6cc336d40f8e7bc85637d49f4a801006bc06e99b42`。录像环境实测来自 `/data/yuyangliu/envs/isaaclab6_validation/bin/python` 的包元数据及已有录像 manifest；临时对齐脚本只作为迁移参考，不作为产品依赖。

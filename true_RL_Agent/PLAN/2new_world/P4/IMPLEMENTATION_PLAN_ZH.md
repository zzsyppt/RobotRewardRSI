# P4 实施计划：Stair 与 Push 的最小共享编排

2026-09-10。进场提交 `cd48221384c7b658eac18c275298a393e1d7b381`，工作区干净。本文是实施与验收清单；尚未完成的项目不能据此视为已经实现。

## 目标与已有边界

正式入口根据显式任务 × 机器人 cell 装配，使用同一套基础准备、预算预约、两 lane 执行、恢复、终态确认和报告流程。新增具有相同执行能力的机器人只扩展任务绑定和机器人事实，不修改共享状态机。Stair 的几何、分层评分、reference 资格 mask，以及 Push 的扰动、恢复判据仍属于任务层。

P3 已复核：当前 478 个源码/配置/测试文件与验收快照一致，两本体 reference 资格通过；C1 卷间差异及地形形态限制仍成立。P4 不调整训练、地形、reference、资格门或候选接受阈值，不重新消费 holdout 来寻找通过条件。原完整 15 rounds 的协议、ledger、checkpoint 和历史报告保持可复算。

进场时 `agent_core/generation.py` 已拥有单代两 lane 状态机。真实拓扑是两 lane train/dev、仅 dev 选中的候选 public-accept/sealed；未选中者不执行确认。旧计划或报告中“两候选都 accept”的描述不能作为新成本合同的依据。当时正式会话集中在 9479 行 Stair 模块中，成本模块 2964 行且包含多代版本继承；Push 尚未实现完整 phased-evaluation 能力。这些是本次需要收敛的源头。

参考本地 Codex `codex-rs/core/src/tasks/mod.rs` 的有限思想：执行宿主拥有统一开始/完成/取消时序，具体任务实现任务行为；生命周期记录先持久化再宣布完成。只借用职责分离，不引入通用工作流语言、插件框架或新的调度系统。

复核原始 F14/F15/F16/F17/F25/F27 后进一步限定：F27 的旧 `23` seed-unit 是保守科学上界，实际 operation reserve 已是一个 selected public-accept；本次需要在新合同区分上界与实际拓扑，不能把旧账解释成多跑了一次。F25 的成对 authoring 失败策略保持当前合同，P4 负责让两任务行为一致，探索策略变更归 P5。

## 1. 固定合同与行为基线

- 列出实际调用链、任务依赖和各版本活消费者，区分正式运行、历史只读与测试工具。
- 记录当前同一套固定 runner 场景的 wall time、报告输入验证/扫描次数；选用已有完整历史报告复算基线。
- 明确共享 phase/selection/guard/reference 接口：只交换有身份的 named score、配对样本、phase receipt 和任务提供的投影，不让共享层识别 Stair/Push/机器人名字。
- 固定实际执行 topology、operation reserve 与最终确认形状。新写入合同以字段含义驱动；旧字段的解释留给版本 reader。

验收：基线用例实际运行，既有统计与身份有可比对的输出；后续优化不能通过省略权威验证获得性能。

## 2. 用两个任务完成真实 phased-evaluation 装配

- 将单代 engine 使用的私有结构协议整理为最小公共能力合同，复用已有 phase job、receipt 和 acceptance API。
- Stair 迁移到该合同；Push 实现自己的 dev screen、selector、guard、三阶段组装、reference 和成本输入，使用真实 Push 配置与指标定义。
- 补齐同一正式入口的 cell 解析和任务装配。现有兼容入口只转发到这一条生产路径，不能保留另一套活动状态机。

验收：Stair 与 Push 通过同一 engine 和固定 runner 走通 accepted、rejected、no-selection、authoring incomplete、runtime failure 与 infrastructure failure。Push fake-runner 结果只证明编排合同，实际 Go2 Push 资格归 P6，不冒充完成。

## 3. 收敛会话、成本、恢复与终态

- 从 Stair 会话中提取真正共享的生命周期、operation 预约/实际计费、恢复和报告外壳；保留任务拥有的准备及投影实现。按职责拆分，不把原 9479 行文件改名后整体搬入 shared。
- 取消按 `StairGenerationSessionCostContractV3/V4/V5` 的继承身份猜测拓扑，改为显式合同数据及所需能力。历史 reader 保持原 hash 和算法。
- 统一正常结束、暂停恢复、崩溃恢复、reference 仍为 current head 时的 final 路径。新 final 字段明确区分 phase、seed 和实际工作量，不延续旧字段的歧义。
- 将一次性 preflight 验收、test-only 模式和只读历史 loader 从活动任务模块分离；仅删除确无活消费者的支路。

验收：两个任务的 resume 与 reference-head final 均经过共享生产路径；operation 预约上限、实际消耗、取消释放能独立复算。崩溃恢复不重复训练、不重复 ledger commit，也不改写既有成本。

## 4. 确定性早拒与报告复用

- 在 public-accept 已经足以证明拒绝时，复用现有 acceptance 判据决定不运行 sealed。不得把缺失 sealed 填成通过，也不得把不完整评测伪装成完整评测。
- 回执显式记录 `not_run_due_to_early_reject`，取消未用的 operation reserve；拒绝的私有细节不进入 provider 上下文。
- 在当次内容身份固定的 verified view 内复用已验证输入，减少重复全量扫描/投影。独立 replay 新建验证视图，不能缓存跨变更的结论。
- 记录终态验证、final 执行、报告投影各阶段时间。

验收：早拒与原完整判决在相同已有证据上给出一致拒绝；会因 sealed 改变结论的情形不能早拒。覆盖余量、配对误差、guard、运行失败及边界相等；沿用有意义的现有测试，不添加大量镜像实现的防御用例。报告值与基线一致，验证/扫描次数和 wall time 有实测前后对比。

## 5. 最终验收与删除审计

- 本地相关测试通过后执行一次完整回归；目标 Python 执行必需的真实生产装配与 Linux 恢复场景。新增变化、失败或证据缺口才扩大/重复测试。
- 静态审计共享编排没有任务/机器人名分派；使用第三种 cell 绑定验证相同能力的新本体无需改共享代码。
- 旧正式报告与 ledger 独立复算不变，核验受保护的 15 rounds 产物与 checkpoint。
- 写明抽取、删除、任务侧保留的实际文件及理由；清除已替换支路，不制造永久兼容脚手架。
- `ACCEPTANCE_ZH.md` 逐项对照以上要求列出真实证据、测试环境、跳过与剩余限制。任何缺项均保留 P4 未完成状态。

默认验收使用固定 provider/runner，无外部模型请求、无新科学训练。需要目标环境检查时先核对进程/GPU占用，使用独立输出目录和命令级环境设置；不停止其他人的进程。若实现证据要求额外物理作业，启动前在此记录具体预算和停止条件。

## 实施记录

### 共享 phase 边界与第二任务形状（进行中）

- 从 `agent_core/generation.py` 提取已有 screen/selection/phased adapter 结构协议到 `evaluation/phased_adapter.py`，补齐实际 reference 组装签名；没有新建调度器。
- `tasks/stair_climbing/generation_bootstrap.py` 已退役，父策略 phase receipt 绑定、复用与 accepted-head guard replay 移至 `agent_core/generation_guard.py`。共享代码根据显式 `parent_public_dev_source` 能力工作，不读取 Stair config 或识别机器人。
- phase plan 身份、receipt 绑定与计费合并移到 `evaluation/phase_assembly.py`，Stair 与新 Push adapter 共用。评分和 selector 仍由任务实现。
- Push 新增真实指标投影、dev 选择、三 phase/reference 组装。新测试通过 `TrialReceiptJournal` 持久化 phase，并调用生产 `SubprocessRunner.project_assembled_eval_summary` 核验；不是替换公共投影的 mock。实际 Push 进程入口、完整 session 和 reference 资格仍待接入。
- Push 指标投影中的即时日期字段被删除：执行 summary 已拥有时间；同一份固定 phase 证据的投影及重建不再因调用时刻变化。旧回执文件未修改。
- 已运行 38 项相关接口回归，以及修改后 150 项 Push/Stair 指标、phase、会话单元回归，均通过。此为阶段验证，不能替代最终全量、恢复和性能验收。
- 性能基线保留在独立 detached worktree `D:/humanoid_agent/.tmp/p4_baseline_cd48221`，提交与进场相同；尚未采集报告扫描次数/前后性能，不能声称已经优化完成。

### 执行、成本与多代循环收敛（进行中）

- Push 增加独立的 reference-program cell `push.g1_29dof.reference_v1`。它保留现有 G1-29 locomotion reward 与 Push overlay，以现有 motion/behavior/绝对分数下限形成最小 qualification 合同；旧 Push 默认 cell 的协议 hash 保持原值。这不是 Go2 Push 资格化，也不证明每个成功 episode 都实际受到扰动，相关科学条件仍归 P6。
- Push 训练入口接收真实 reference artifact 与初始策略身份，评测入口接收 phase/plan hash，并且只运行所要求的 seed cohort。单阶段摘要记录实际 vector step、episode step 与耗时，不生成缺失阶段的成功门。
- GPU 参数、进程身份和外部 PID→UUID 观察迁入 `runner/gpu_process_binding.py`，Stair 与 Push 入口共用。新写入使用 runner schema；历史 Stair schema 只读兼容，测试确认解码不改原对象 hash，也保留旧 observation 的版本与内容。
- 单代 lane 输入、scope 恢复、任务 phase 组装已共享；多代循环进一步迁入 `agent_core/generation_session.py`，原 Stair bounded driver 只转发装配和边界投影。公共循环使用 adapter 的 `parent_public_dev_source`，不读取 Stair config。正式入口与 final/report 装配仍需继续收敛，不能据此宣称完整 session 已跨任务通过。
- Provider reserve 从冻结的 provider budget 读取；实际工作量、job receipt 核验、operation allocation、资源回执匹配和 cost journal 已从 Stair 模块抽出。Cost journal 根据显式 `uses_operation_reserves` 采用计费方式，不按 Stair V3/V4/V5 的继承身份分派。旧合同继续给出原 wire schema，历史数字与哈希没有被重新解释。
- 以上变更最近一次本地集中回归：102 项通过，覆盖原会话、cost recovery、成本合同、provider 中断恢复、GPU public/private 投影与 Push phase/reference 装配。模块抽取后同步更正 mock 的所属模块；原判据与断言保留。尚未执行 P4 最终全量与双任务完整生产场景。
- 目标机 baseline 提交仍为 `cd48221384c7b658eac18c275298a393e1d7b381`。已启动固定 provider/runner 的 reference-head 15 代基线，输出为 `/data/yuyangliu/AGENT/p4_baseline_cd48221`；不运行真实训练或外部模型调用。性能回执必须区分完整运行与独立 audit，并记录测量开销，不能把 generator resume 次数称为实际目录扫描次数。

### 报告复用与独立重放（阶段验收通过）

- 同一次报告只创建一个完整 public-history 验证快照；每代查询使用已经重放出的真实 prefix head，未使用 ledger 自报 head。快照拥有输入副本，新的独立审计仍重新读取与验证。
- 同一次报告只加载一次成本链快照，报告内部按 event 查询；未缓存跨调用结果。成本链发现重复 event 时仍拒绝。
- 同一旧版 15 代 fixture、相同 v2 测量器：独立审计 wall time 从 327.911 秒降为 81.651 秒，`project_public_history` 从 18 次降为 1 次，`load_chain` 从 21 次降为 3 次，`read_bytes` 从 1385 次降为 684 次，`glob` 从 91 次降为 73 次，`rglob` 保持 22 次。含 profiler 开销，不能与未加 profiler 的完整运行时间混比。
- 新代码重建的旧报告值完全一致；251 个 fixture 文件的 size、mtime_ns 和 SHA256 均未变化。见 `baseline_report_audit_measurement.json`、`shared_report_audit_stage2_measurement.json`、`runtime_stage2_readonly_audit.json`。目标 Python 的 118 项专项检查全部通过、无跳过，见 `runtime_stage2_unit_results.json`。

### 新正式合同与宿主接入（进行中，尚未整体验收）

- 新 `FrozenGenerationCostContract` 从 adapter 的真实 workload 与显式 operation source 构造。区分两个 dev、一个 selected accept、一个 sealed 的真实工作量，以及训练/评测 retry、final 和会话文件的保守预约。旧版本数字仍由旧 reader 解释。
- 新 final plan 使用 `phase`、`seed_count_per_job`、`total_eval_seed_units`，不沿用旧 `seed_count` 的混合含义。Stair 继续预承诺独立 study holdout；P4 Push 只复用现有 sealed cohort，明确不声明新 seed 或新地形泛化。新增 Push 科学协议仍归 P6。
- 新共享入口 `scripts/run_generation_session.py` 接收显式 task/cell、冻结成本合同与 provider policy。`generation_application.py` 负责准备、开始、恢复、终态执行和报告；任务仅绑定 Genesis 资格事实、final cohort 与分数投影。
- 三个真实 cell 的 composition/foundation 检查通过，接入 final plan 不改变其 TaskSpec 与 manifest 身份。Genesis 前只读取静态探索契约；Stair 实际投影仍要求 Genesis 提供 qualification mask。
- 首次完整固定运行发现并修正新成本合同的两个缺口：会话文件须有独立空间预约；reference 三 phase 必须分配已经冻结的整数总量，不能各自再次取整突破总预约。新容量闸门直接使用整数 ns；旧合同保持原执行限额。

### 公共入口、提前拒绝与恢复验证（进行中）

- 正式 CLI 要求显式 `--task-id` 与 `--cell-variant-id`。三格真实装配和准备后恢复已通过；成本输入必须来自已绑定 cell 的 workload，不能用绑定前的 Task hash 代替。恢复未改写冻结文件。
- `PublicAcceptScores` 只携带可比较的分数；公共 `compare_public_accept_scores` 与完整判决共用原有余量、配对噪声和 guard 算术。新会话冻结 generation ledger v2，确定性拒绝写独立私有证明和公开 `not_run_due_to_early_reject`；不生成缺失 sealed 的完整结果，不赋予 acceptance authority。旧 v1 writer/reader 解释不变。已关闭代的未使用 sealed slot 由现有 operation reserve 计算自然释放，没有增加取消调度器。
- 阈值相等、提升不足、guard 退化与配对噪声的算术对照通过。Push 已区分提前拒绝、sealed 依赖的拒绝、dev 无选择、候选运行失败和基础设施超时。正式路径发现并修复 Push reference gate 的任务协议身份，以及 public score statistics 缺少 evaluation protocol hash；这些身份分别归任务 qualification 和任务分数投影。
- 两任务 reference-head 的 15 代与只读重放通过：Push 214 文件、Stair 215 文件的 size/mtime/SHA256 未变化。见 `shared_push_reference_stage3_results.json`、`shared_stair_reference_stage4_results.json`；这是固定 runner/provider 的工程证据，没有物理性能声明。
- Push 已完成第 1 代接受、第 3 代暂停、恢复至 15 代、final 和独立只读 audit，283.160 秒，见 `push_accepted_resume_test.log`。完整 final 成本重放改用已有 final-aware replayer，避免把 final settlement 当作普通代成本。
- Stair stage5 候选测试失败于 fixture 给两个 sibling 相同 dev 分数，选择器正确返回 exact tie；没有改 selector。修正 dev 排序后，stage6 的拒绝分支通过，但运行失败 fixture 用错任务 stage 名，且 hard-exit 暴露入口继承父进程组的问题。前者按任务声明修 fixture，后者在 Linux CLI 启动边界取得独立进程组；没有放松恢复时的存活进程检查。stage7 全部通过，详见下文。
- `generation_history.py` 已提取旧报告实际需要的读取依赖，复算同一冻结 15 代报告一致，251 文件 size/mtime/SHA256 不变，见 `p4_history_reader_audit.json`。原执行模块的删除与测试迁移尚在进行；固定 provider/runner 已移出旧 integration driver，不需要复制旧宿主才能测试新宿主。

### 入口退役与最终验收（2026-09-10，进行中）

- stage7 两任务真实进程退出 91 后恢复、第 3 代暂停、继续至 15 代及 final/audit 均通过。每个会话只训练 3 次（reference + 两候选），ledger 前缀不变；Push 290 文件、Stair 291 文件的 size/mtime/hash 在独立审计前后不变。见 `stage7_push_recovery_results.json`、`stage7_stair_recovery_results.json`。Stair 的提前拒绝、sealed 拒绝、无选择、候选运行错误和基础设施超时均通过，见 `stage7_stair_branch_results.json`。
- 原 cell manifest 写入会覆盖不同内容，恢复也可能补造丢失文件。现改为首次创建、之后只核验；恢复/审计要求 manifest、cost、continuation、session contract 和 final plan 已存在。三格真实 CLI 验证不一致与缺失文件保留现场。provider 配置在准备和执行时复用同一原有配置校验，不到第一代才发现必需字段遗漏。
- 删除旧 `tasks/stair_climbing/generation_session.py`、`launch_readiness.py`、`launch_context.py` 与 readiness 生成脚本。两个 Stair 命令文件只转发到 `run_generation_session.py`，不再维护旧参数或第二条执行路径。原始版本可从进场 Git commit 获取，没有把整套旧宿主复制到 tests/archive。
- 历史读取由 `generation_history.py` 与 `launch_history.py` 保留：前者复算旧报告，后者只核验旧 source/lifecycle 证据。历史成本从冻结文件读取，匹配任务及合同身份；不按今天的运行默认值创建旧预算。删除后同一旧 15 代报告重建一致，251 文件不变，20.992 秒，见 `retired_host_history_replay.json`。
- 有效成本、provider、资源、只读历史断言迁到实际所有者；旧启动/一次性 writer 测试随对应入口退役。final checkpoint/GPU rematerialization 断言扩展到三格矩阵，并保留自洽但外来的 workload 被拒绝的测试。逐项迁移和删除说明见 `retirement_test_inventory.json`。59 项相关回归通过，新增两项迁移检查也通过；它们不能替代全量。
- 原完整 15 代保留全部 71 个 checkpoint：23 个权威 checkpoint 连同 884 份受保护产物核对 SHA256，48 个中间 checkpoint 核对原 size/mtime；P0 所选 932 文件共 517212889 字节，无变化。目录另有 339 个早于 P4 的日志/锁文件，P0 本来未将它们列入保护清单，不能误把“所选文件数”当作目录总数。未删除这些文件。见 `protected_history_check.json`。
- stage8 使用独立 Git worktree，99 文件源码/文档/测试快照。本地首次全量 1590 项中有 1 个 ERROR：原测试仍访问已改名的私有 `memory._exploration_projector`。将断言迁到现有公开方法 `exploration_projector()`，没有增加兼容别名或改变判据。修正并补齐 final 绑定断言后的完整单元回归，本地 1499 项通过（15 项目标环境限定），目标 Python 1499 项通过且零跳过，含 Isaac 配置、recorder、tensor/video 和 Linux 检查。原始失败结果保留在 `local_full_results.json`，后续分别见 `local_unit_after_migration.json`、`target_unit_after_migration.json`。
- 删除旧入口后的复审补回首次正式 CLI 的本地凭据检查，在 reference 工作前发现缺失密钥；使用已有 provider 解析与检查，不做网络探测。prepare-only、只读审计和不再需要 provider 的恢复不被该检查阻断。本地及目标 Python 的 62 项入口、final 绑定和 provider 回归均通过，见 `local_cli_provider_followup.json`、`target_cli_provider_followup.json`；stage10 相对 stage9 只改 CLI、provider 本地检查和对应测试。
- 目标完整集成回归仍在执行。此轮所有新增会话使用固定 runner/provider，未启动新科学训练或外部模型请求。

### 最终闭合（2026-09-10）

- 目标首轮全量实际为 1590 项、2 个 ERROR、零跳过。除旧测试方法名外，还暴露 Stair final nonce 恰好生成已有 calibration 几何的问题。固定反例已复现，修复归任务 holdout 创建器：首次提交前按原有隔离规则排除几何重合，不读取任何策略表现；已提交实例与 replication 禁止重抽。两本体的固定 nonce 完整题池映射与进场源码一致。
- 修复后的目标 27 项复核通过，包含原先失败的 Stair runtime/infrastructure 场景；本地相关 26 项通过。最后保留初版 P4 两项 final 合同测试并与迁移后的两项绑定/workload 测试一起复核，本地及目标各 4 项通过。没有用替换断言减少测试责任。
- 当前 1594 个测试（1503 unit、91 integration）逐项对应目标成功记录，缺项为 0；首轮 FAIL 和各后续 PASS 原样保留，见 `verification_coverage.json`。目标新动态检查和本地 tensor/video 检查实际运行。
- 共享所有权、删除消费者和当前文档链接审计通过；最终 98 个源码/测试文件已同步服务器主项目并核对 hash，11 个明确旧文件已删除，三个 CLI 的实际 help 通过。同步后再核验原 71 个 checkpoint 的 SHA256 均未变化；验收进程已退出。

**P4 工程 GO**。最终要求、证据与剩余科学边界统一见 [ACCEPTANCE_ZH.md](ACCEPTANCE_ZH.md)。

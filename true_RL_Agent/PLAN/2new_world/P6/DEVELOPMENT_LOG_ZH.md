# P6 开发与接线记录

## 2026-09-11：实物前边界与首轮检查

P5 修复的真实历史验收见 AUDIT_DISPOSITION_ZH.md 和 shared_history_c6_01/result.json；新 P6 证据不得覆盖 c6 的原 prompt、ledger 或 checkpoint。

新 Push 使用 G1-37/Go2 原有 RobotProfile、RuntimeAdapter 和 PPO；任务只拥有站立与施扰协议。身份冻结包含动作/观测实际配置、解析后的有序关节与维数、任务尺度及完整 reward 语义。首次物理资格前修正了 XYZW 输入转换，以及 reference CLI 错用 generation 专属参数的装配问题。

目标环境专项 target_development_tests_01.log：35 项、0 failure/error。首次实际 GPU 探针 outputs/p6_wiring_20260911/{go2,g1}_01 均在 PPO 启动前因 sensors 声明不匹配停止；未产生训练或资格结论。新 Push 的全部观测、奖励、终止及公开输入均不消费接触传感器，因此在任务 overlay 移除该无用组件，绑定声明为空并严格检查实际 scene.sensors 为空。机器人配置与 Stair 不改动。这是删除无消费者依赖，不是放松检查来换通过。

独立检查目标 IsaacLab 的实际调用顺序确认：termination → reward → reset → interval events → observation；RewardManager 唯一执行一次 weight × control_dt 积分。新 Push 不再包旧 additive reward wrapper，终态在重置前保存。完整评测的两个统计调用补齐同一 manifest 接受协议，防止使用 legacy 默认域。

本地首次全量（local_regression_01.log）：1635 项，19 跳过，2 failure、7 error，952.532 s。失败集中在旧 Push bootstrap/warm-start 测试未显式指定历史格子及旧审计入口的默认引用；保留原断言并明确历史边界后再回归，不用新增 reference 的权限迁就旧测试。

所有以上运行均为开发检查；尚无 P6 完整 reference 训练、留出成绩或科学视频验收。正式测量前须另行写入不可变实验声明。

## 2026-09-12 00:13：等待连接冷却时的恢复点

四格共享 reference CLI 真实物化预检通过，见 matrix_preflight_01.json；本地采用短输出路径，避开当前 Windows 的260字符文件读取限制，未修改系统设置。旧 bootstrap 审计仍有 CLI 消费者，保留并明确拒绝 reference 格子；两组历史 fixture 显式 legacy 后，原始 warm-start、身份漂移和 argv 断言保持不变。第二轮完整 discovery 进行中，结果写 local_regression_02.{log,json}。

服务器 SSH 在完成首轮探针和第二份开发包上传后于握手阶段关闭；用户提示可能是频繁连接触发流控。后续改用持续连接，不再由多代理分别连接。development_source_02 对应源码修复包已上传到 /data/yuyangliu/AGENT/.tmp/p6_development_source_02.zip（SHA256 71d4db439e8bb02c0253f9343e275901b22fa171f67f8061766d06f5934b64a4），但解包命令因连接失败未执行。后续旧audit入口修复尚在本地；正式训练前需再次核对整个 source snapshot。

下一步：连接恢复后安装已上传开发包，重跑 P6/push_wiring_probe.py（64 env、72控制步，训练/resume + dev无扰动/施扰）、P6/push_pipeline_probe.py（3072 env-steps 的生产 CLI train→真实checkpoint→仅public_dev）。这些都是diagnostic，不能消费accept/sealed或签资格。通过后冻结正式协议/资源声明，再进行两本体完整reference资格、native视频和终态验收。


## 2026-09-12 00:32：接线通过与正式声明签发

前述 SSH 流控是00:13恢复点的历史状态。主任务随后通过持续连接完成部署和目标机检查，当前不再以连接阻塞作为进度状态。

目标原始收据已下载至 `target_wiring_evidence_01/` 并核对：[G1 wiring](target_wiring_evidence_01/g1_02/result.json)、[Go2 wiring](target_wiring_evidence_01/go2_02/result.json) 及生产短 train→真实 checkpoint→public-dev 流水线（[G1](target_wiring_evidence_01/g1_pipeline_01/result.json)、[Go2](target_wiring_evidence_01/go2_pipeline_01/result.json)）均 PASS，对应 owned process 返回码均为0。生产探针每格仅3072 env-steps，只执行public-dev，不能替代完整 reference 资格。G1 原时序零动作探针早跌，实际事件覆盖0/80；Go2 为80/80。随后 [G1 独立提前时序 diagnostic](target_wiring_evidence_01/g1_event_01/result.json) 验证实际注入 **80/80**、16个重复事件 episode 及下一物理步证据，全部接线检查通过；80/80 是事件覆盖数，不是恢复成功率，不改正式协议时序。

完整本地回归 [local_regression_02.log](local_regression_02.log)：**1652项，OK，19 skipped，931.347秒**；这些 skipped 不替代目标机实测。协议默认切换 active 后，主任务另完成定向30项 PASS。先前失败的旧 fixture 已显式选择历史格子，原断言保留。

[formal_declaration_01.json](formal_declaration_01.json) 已在首次 holdout 消费前于北京时间00:32:15签发；[声明说明](FORMAL_DECLARATION_ZH.md)固定了每格一次完整训练、资格门、4小时总墙钟上限、单作业36000秒及0次重试。默认 active 只表示协议冻结，reference 资格仍待验证。本记录不表示正式训练已启动或 P6/P7 已获 GO；后续必须用完整训练、三阶段资格、独立重载及同 checkpoint 原生视频的真实证据验收。

## 2026-09-12 00:56：完整训练已完成，Genesis完成标志阻断

两格原定 random-init 4096×24×500 均已完成；原始 summary status=pass、latest_checkpoint_iteration=499、budget_accounting=pass，最终 checkpoint 字节重新核对通过（Go2 4,584,363 B；G1 5,545,131 B）。原始路径为 outputs/p6_reference_formal_20260912/{go2,g1}/runner/p6-reference-{go2,g1}-20260912/train/summary.json。

两次资格进程均在任何评测之前被 session_genesis._validate_reference_train_result 的 formal_training_executed 标志检查阻止，退出1。此标志仅Stair生产writer给出，Push生产缺失；这不是实际预算未训练。仍未消费本轮 dev/accept/sealed。不得改写原summary、删除started marker或重新执行训练。正在将已验证完成结果的规范化收回Runner，并核实旧训练执行来源与修复后评测来源的显式续接；源码修复前的原始声明保留。

同阶段开始复用原有checkpoint执行Stair两格当前V3 public-dev回归，0训练/accept/sealed/final；当前物理结果待回执。

## 2026-09-12 01:25：完成标志修复、只读导入与显式续接

共享 Runner 依据实际绑定预算、最终 checkpoint iteration、初始策略身份及原摘要/字节证据规范化缺失的训练完成标志，不改原文件；共享 Genesis 可接收已完成 TrainJobSpec / TrainJobResult，保留旧训练来源，只在新的执行目录评测。新 CLI 的 resume-evaluation 要求原科学身份不变；原 started marker 和失败收据保留。普通资格今后在训练前保存完整执行环境与实际 TrainJobSpec。详见 [续接声明](CONTINUATION_DECLARATION_ZH.md)，没有改变任何物理条件或资格阈值。

目标旧冻结源码全量回归 [target_regression_01.log](target_regression_01.log)：1652项、0 failure/error、2 skipped，2732.801 s。结束后才部署修复，避免混合源码。目标修复专项 [target_continuation_tests_01.log](target_continuation_tests_01.log)：21项全部通过，16.615 s；两份真实原始摘要经 [只读导入预检](target_continuation_preflight_01.json) 通过，无训练/评测调用。新执行源码 hash 为 eeea38f845775a77d94ecb24a19bf8faa3f8e9c8aa8265dec09e8952924223d4；[当前评测环境](evaluation_execution_environment_01.json) hash 为 f0b44b6a00d6652c03a5dc38286c57331006ce0e7b311a96c474dcad72f07709。旧完整环境 hash 仍为 68dea27a9b939b8432c2ef7960482f87e21cacce4f816be52795c2509d00cf3c，二者分别归属训练与评测。

两格 continuation 已通过外层 wrapper + CLI 的同一原 process receipt 扣减原耗时，0新训练，只执行原定3相。实际结果仍待三相与后续独立验收，不提前声明 GO。

[Stair当前dev回归](stair_dev_summary_01.json) 两格各33项核验通过：G1 618/864，J_primary=0.7436874183847737；Go2 689/864，J_primary=0.7466970595216049。保存的全部 episode facts 与 P3 同一 dev 完全相等；0训练、0accept/sealed/final/provider。本次生产观测组、constructed learner 与 checkpoint 推理检查真实执行，但未额外展开新的关节/动作/传感器 ABI 报告，不能把下载件未包含的检查宣称为新增实测。


## 2026-09-12：首轮资格阴性结果与独立诊断

两格完整三相已完成：Go2 832/832、G1 0/832，每格1040次注入均有效。Go2超dev余量上界，G1低于基本可学性/恢复下限；两格均NO-GO、均无Genesis。每格一次native pass，全部384个保存episode字段与正式dev严格一致；六段影片完整解码，仍缺每checkpoint同时具备成功/失败的案例。详见 [当前验收](ACCEPTANCE_ZH.md)，不把工程闭合替代reference资格。

最终本地1666项通过、19跳过。修复后目标全量仍在执行；旧源码目标1652项与修复专项21项不是最终全量的替代。执行收尾已确认c6全1271文件/71checkpoint和旧训练证据不变，原物理进程退出、lease已释放。

针对用户提出“训练是否缺push”和“评价是否有问题”，分开核对训练施扰与奖励项：原训练代码确接入interval速度增量，五项reward均有实际调用；摘要未存训练事件累计，不能用wiring后的评测事件数冒充。完整训练日志、TensorBoard和目标框架源码已下载至training_framework_evidence_01，13份原件逐hash核对。

G1首次施扰之前已普遍移动/自旋，故先独立复算时序与无扰动站立，再判断奖励。stationary_v3仅为未训练的独立CPU草案，未替换默认。后续边界和停止规则见 [独立开发计划](DEV_REFINEMENT_PLAN_ZH.md)。本次服务器仍有VLLM占显存，用户表示由其释放；未停止外部进程。


### 同日：独立原状态对照通过

按independent_timing_declaration_01执行256个episode，两个进程返回0、四profile均PASS。CPU从真实pre-reset位姿/速度及实际写入独立复算stable、dwell、终止和恢复，与产品结果零分歧；strong_push首dev与原正式记录全字段相等。无扰动G1完整稳定驻留0/64、Go2 64/64，G1自旋并不依赖施扰。原件28文件完整下载并校验，详情见independent_timing_summary_01.json。未修改旧协议/奖励/checkpoint，无holdout、无训练。

为新训练补最小任务证据：PushRuntimeState保留跨reset的设备端两个累计数，Push训练入口在learn结束后一次读回到training.push_exposure。injection_attempt_count是速度写入尝试；verified_event_count是finite且写后读回满足容差的次数，不是下一物理步或恢复成功数。未在每步增加CPU .item()、未增加共享Runner特判。旧原始训练不回填。12项定向测试通过，正在做最终全量；一次本地stdin测试启动因Windows multiprocessing无法导入<stdin>而取消，记录于local_regression_04.json，修正为模块启动后运行local_regression_05，不将启动器错误当产品失败。


### 同日：stationary_v3独立开发已启动

目标18项物理协议/已完成训练续接/完成标志测试通过；新计数不改奖励、物理、PPO或默认reference。停止并保留已被新源码替代的target_regression_02中途日志，统一用新源码运行最终target_regression_03（未标通过），避免把混合源码回归当证据。local_regression_05同步进行。

两格目标实际preflight均PREPARED，新reference subject/source/manifest贯穿物化Train/Eval。stationary_v3_declaration_01在新训练前冻结：每格一次4096×24×500 seed42，3×64 public-dev，0资格/holdout/provider/重试，train+dev每格总3600秒；后续原checkpoint诊断每格最多900秒。新实验只检验stationary假设，不回调旧门；G1基础可学性或无扰站立不成立则停止本假设，Go2饱和留给独立任务强度标定。实际driver PID：G1 2720728、Go2 2720729；目标最终回归driver PID2720730。新训练结果尚未取得，不宣称修复成功。


### B开发完成与有限难度标定准备

两格各一次500轮训练完成，dev均190/192；actual training exposure全部写后验证，初始policy hash与原轮相同。两格原checkpoint的strong/no_push独立跟进各64条通过；G1无扰动完整dwell从0/64到64/64，角运动明显下降。原始77文件批量取回并逐件校验，报告见stationary_v3_summary_01和stationary_v3_independent_audit_01；不将开发通过当资格GO。

补齐Push canonical seed/reference/protocol输入，没有把路径写进科学身份；保留v3原source/provenance。早门归task adapter，共享只先持久化dev receipt、执行status，失败保留holdout，Stair原行为不变。共享ReferencePhasedEvaluationAdapter仅一份定义。60项gate/回归专项及41项输入专项通过；local full06启动，server full03继续读原冻结源码。

留出核查纠正：曾提出base20260912，仅与P6的42不交但会和已执行P3的20260908 sealed重叠三项，已撤回、未冻结或执行。另记录P2旧probe直接使用731 RNG流，并无三相派生。后续候选base1000000000与这些已核实集合不交，仍需最终冻结，不能声称跨协议的旧probe就是当前同一道题。


### 同日：有限强度标定停止规则执行与G1正式启动

C声明在任何新增测量前冻结。G1 ×1.5=167/192，首点入带即停止；Go2 ×1.5=190/192、×2=176/192，网格耗尽NO-GO，未追加点/seed/训练或改变门。55份下载物与576条分类/715事件独立复核通过，新增物理成本0.03474 GPUh。

现存Stair日志补齐关节顺序/维度/时步ABI，无额外GPU；G1-29使用旧P2 checkpoint在当前后端做一次eval-only短诊断，原件与decoder可读性核验通过；全208条均失败，保留旧budget/motion/sealed失败，不将执行成功混为资格。

G1所选协议落入configs/tasks，task-owned新建默认使用v3、base1e9；Go2/legacy原默认不变。native种子缺省从42改为None，避免覆盖新装配；显式旧G1仍读出原manifest。41项专项在本地与目标均通过，最终本地全量07启动。

正式声明g1_fresh_qualification_declaration_01在started marker前冻结，源码b73c055c54ecbe445c59e632f928712a43a163e6494d4cd0ab0406b79dc450f2，环境c451dd809d4a26db2d61fb2c9ea08de83ef39a8b8f809882a9c40e7430bbb2c8。G1新训练由wrapper2746912启动，原B checkpoint仅作为开发锚，不导入训练；1.5协议同时进入训练/评测，固定500轮，fresh dev失败则停止，原accept/sealed不复用，P7未启动。


### 同日：G1正式资格、同checkpoint原生回放与执行收尾完成

G1新训练完成500轮/49,152,000 env-steps，实际施扰attempt/verified为166,301/166,301。新dev96/192，通过事前早门后才执行accept106/192与sealed245/448；下降0.005208小于冻结0.1，真实Genesis已创建。独立预检核对声明提前启动标记34.48秒，最终离线审计从832条episode重算分层、1,016次事件、速度写入和保存的时序，PASS。没有该次资格全部逐步mask，不夸大dwell独立证据；训练seed仅42。

一次原生dev回放固定env48、三个既定seed，依次侧跌/成功恢复/前跌，每段两次真实扰动；192条完整记录及sidecar与正式dev逐字段相同。三段298/411/263帧、1280×720、50fps完整解码，查看了终止前关键帧。资格目录49文件/77,977,952B在只读重载和录像前后hash/size/mtime不变。新资格1102.780秒，native145.055秒；原训练导入源、B锚、c6的1271文件和71 checkpoint均保护通过，任务进程/GPU/lease释放。原件分别存g1_qualified_01、g1_native_01，最终证据见execution_closure_02.json。

最终本地全量07为1677项/19 skipped，发现2条probe测试仍硬编码旧seed42。产品和实际argv已正确采用G1新默认1e9；只修这两条测试、保留Go2默认42/G1显式旧42/lineage拒绝断言，本地与目标各2项专项PASS，未修改冻结产品源码。保留原全量07失败回执，不伪写全量exit0，不为纯测试期望变化重复全部长测。Ruff F检查通过。

活动README、概览、Push/Stair指南与当前验收已同步：首轮阴性、公开开发标定、G1新正式资格分开呈现。Go2有限C标定依旧NO-GO，无新Go2资格、未启动P7；不将G1完成缩小为P6整体完成。

## D：Go2匹配训练开发对照结束（2026-09-12）

执行前独立冻结GO2_MATCHED_TRAIN_DEV_ZH.md及go2_matched_dev_declaration_01.json，C网格不延长；同一已有×2协议、原dev种子，随机初始化500轮。既有开发入口只增加显式protocol/session/v3输入，走canonical composition，保留B旧入口字节，产品373源码不变。本地物化及目标实际preflight通过后启动。

实际训练49,152,000步，209,350次施扰全部验证；dev三个seed均64/64，四层均48/48。工程执行与离线审计PASS、开发NO-GO，未进入正式资格或holdout。已有C176/192结论保留；D192/192说明本次匹配训练仍饱和。0重试/provider，679.348秒（0.18871GPU分配小时）。42下载文件SHA核验、实际checkpoint重算、B/G1/c6保护和资源收尾通过。原生录像未为这个未合格的新reference启动。具体见GO2_MATCHED_TRAIN_RESULT_ZH.md。

离线脚本两处原schema假设修正后复算：训练协议位于physical_protocol内；同runtime的episode_index随三次reset累计为1/2/3。原产物/声明均未改，审计不调用产品统计器、不声称从保存的时刻独立证明全部dwell。第二位审阅者只读复核C/D原计数、两处修正及边界通过。

## D 后姿态诊断与任务范围复核（2026-09-12）

在 D 完成后另冻一次原 checkpoint/原 dev 的 native 姿态诊断，未修改此前 D 声明。105.740 秒完成，192 条原记录完全一致；三个 env32/high 成功案例各302帧，906帧完整解码并查看施扰前、受扰后与恢复图像。28原件一次下载SHA核验。D37文件/12ckpt、G1新资格49文件/13ckpt、c61271文件/71ckpt与373源码均保持不变，任务进程/GPU释放。见 GO2_POSTURE_RESULT_ZH.md。

再次独立审查原目标：连续扰动不必然指恢复窗口重叠，当前重复施扰不是由此可判定的实现遗漏。重叠时“各事件期限恢复”与“整段序列结束后恢复”并不等同，不能靠删除一个校验偷换语义。当前未发现新的物理/评测阻断项；Go2依旧受原余量门限制。没有新建时序协议、追加强度点、修改门或消耗holdout，待明确任务研究方向。

## E：用户授权连续双推对照（2026-09-21）

用户明确要求“改成连续双推试试，看看reference成功率如何变化”。按独立计划与目标预声明，只使用D现有checkpoint进行eval-only；新序列定义为最后一推后重新累计完整恢复窗口/驻留，不伪报两个独立event恢复，不改产品默认。PLAN诊断入口复用产品环境、注入、rollout、ABI加载与owned process；新timing参数单独冻结，源manifest只作原训练/ABI来源。

本地和目标4个针对性CPU测试通过，独立只读代码审阅无阻断。目标preflight与D环境一致，原源码373文件不变。一次批次原时序/连续双推各3×64，第二推从172步提前至61步，第一推51步、每推medium强度及+π/2方向不变；实际每arm240次施扰全部验证。双推均48/48，总分均192/192；每arm144个控制episode与原D逐字段相等。连续组第二推前40/48瞬时未稳定仍全部恢复。原始位置/四元数/速度与实际注入记录保留在sequence_01，41文件SHA校验通过。

执行53.938575秒，0训练、holdout、provider和自动重试；D37文件/12ckpt、G1新资格49文件/13ckpt、c61271文件/71ckpt在前后hash/size/mtime一致，GPU和任务进程释放。完整结果见CONTINUOUS_PUSH_RESULT_ZH.md。试验结束，不追加档位/间隔、不把开发诊断当新资格；P6余量问题仍未解决。

独立离线审计sequence_audit_01.json为PASS、0 issues：不导入产品runtime或新评分器，从原始数组重算96个双推episode，核对192次双推事件及144个单推配对控制记录。第二推前瞬时稳定数原组48/48、连续组8/48；恢复结果和延迟均与执行记录一致，不据此宣称显著改善或新资格。

## F：现有Go2/G1 checkpoint真实外力推搡（2026-09-22）

用户授权不重训练、先Go2后G1。先核对目标beta2.patch1的composer/PhysX写入/实际质量API，使用binding base角色的link COM、世界水平矩形脉冲、F=M_total×原Δv/实际τ；τ按T向上取整control步。关闭旧速度事件，每physics子步提交真实力并保存PhysX入参；恢复从脉冲结束下一control样本开始，早跌/未发生后推保留分母。实现仅位于PLAN诊断入口，训练/共享宿主/生产协议不改。

两端各6项评分测试及目标CPU身份预检通过，定义和依赖在启动前冻结。一次Go2作业192/192、一次G1作业87/192（27/32/28）；Go2实际240次完整推搡，G1完整223次，17次因先前失败未发生。原状态和实际wrench返回force_01，67文件SHA核验。Go2 41.934839秒、G1 72.201098秒，0训练/holdout/provider/运行重试，进程/GPU释放；D/G1/c6及373源码保护通过。详见EXTERNAL_FORCE_RESULT_ZH.md，旧资格与C/D/E结论不改，不追加网格或签发新资格。

持久化独立审计force_audit_01.json为PASS、0 issues，另一个独立审阅者用旋转矩阵及连续窗口复核，384条分类和恢复步数均一致。脉冲外、直接力矩、其他link受力均为零；G1的17次后推缺失全部由先前真实失败解释。Go2施力期间并非一直稳定，无没施力/提前计时导致的全满假象。有效mask仍来自closed标记，不声称独立证明整个实时终止采集闭环。

## G：有限外力强度扫描与原生视觉复核（2026-09-22）

按用户授权，事前固定1.5×/2×，保留F的1×作为已有基线；同D checkpoint、三公开dev seed、0.22秒/方向/施力点/原成功标准不变，不训练、不消费holdout、不追加第三档。两次统计首次成功：185/192、152/192，高强度层41/48、14/48。独立NumPy重算含基线576条及实际PhysX力/冲量/清零，目标与本地PASS；成功集末尾实际连续稳定最短分别0.76/1.42/1.24秒，没有末帧偶然稳定的假成功。

按事前规则选8个录像案例。第一批7段因请求960×720而实际输出1280×720，完整解码显示303图像被编码404帧，拒绝为视觉证据；修正PLAN入口画幅、真实shape及解码count检查后重新录制。第二批4个seed1案例精确相同，另4个seed2/3跳过前序仿真导致轨迹不同，拒绝冒充原回放；最小修复为保留原前序seed运行，只录指定seed。最终4个v2+4个v3组成有效集合：1741帧完整解码，8×64个对应episode分类和首episode有效状态/物理数组逐位一致。统计结果从未重跑或覆写，所有技术问题原件保留且有拒绝/准入说明。

人工审阅原生关键帧及时序帧表：同env35在1×恢复站立，在1.5×/2×明确后倾越过原失败门；2×临界成功样本在大幅倾斜后重新直立，未见重置站起或躯干拖行冒充成功。该观察不声明足端零滑动或未采样子步完美。跨倍率第一seed施力前全状态相同，后续seed有亚毫米级位置历史差，完整顺序复现能够逐位重现原实际轨迹；诚实保留此执行边界。

本轮产品373文件未变，只扩展PLAN诊断入口。21个作业退出、GPU释放，D/G1/完整15rounds c6的96个checkpoint及关联文件hash/大小/mtime不变。两端6项评分测试、编译和Ruff致命错误检查通过，两个下载包357+91文件SHA核验。详见FORCE_STRENGTH_RESULT_ZH.md。仅完成固定策略公开研究，不签发新资格，不改变P6 NO-GO。

## H / I：短间隔60%与80%外力（2026-09-22）

用户先授权在2×外力下缩短重复推搡间隔，后在H已测42/48时另行授权每推从high的60%改为80%。分别在SHORT_FORCE_PLAN_ZH.md、SHORT_FORCE_80_PLAN_ZH.md预声明；I不是在H之前预注册的点。两次均不训练、不消费holdout、不追加其他比例。short_force_01和short_force_80_01保留各自冻结源码、参数、命令和结果。

实现只扩展PLAN入口：scheduled51/73、每推0.22秒、空档0.22秒；有限外力整段评分抽共用驻留函数，全部计划力完成后才从最后结束重新累计81步窗口/16步驻留，末态必须稳定。v3冻结新时序/序列定义；v4额外冻结repeated_high_fraction=0.8，只覆盖repeated的实际力计算，不改medium源参数或单推层。旧F/G独立恢复含义和冻结副本保留。

三组同序列口径为：G长60% 152/192、repeated45/48；H短60%149/192、42/48；I短80%121/192、14/48。H第二推前44/48瞬时不稳定且48/48未完成首推驻留，I有第二推前采样的47例均不稳定；严格保留提前失败分母。I每推比H加1/3，实际203.282–252.129N。29例成功转失败、1例失败转成功；其34次失败由18越界、8侧倾、7塌落、1后倾组成，未把全部失败说成倒地。

目标和本地独立NumPy审计均PASS，覆盖长/短60/短80共576条原始统计轨迹及实际力/清零/首episode终止闭合；新统计384。四例原生视频共756帧完整解码，4×64个选定seed的episode与有效物理数组逐位一致；前序seed重放不计新样本。数值、关键帧及时序帧表支持两个失败与两个真实恢复案例，未见reset后站立冒充成功。后两seed跨条件有小幅初始历史差、后期轨迹可放大；不把单推标签不变说成全轨迹逐位一致。

本地与目标各9项评分测试通过，本地编译及Ruff致命错误检查通过；未重跑无变化的产品全量。H三个作业207.267秒，I三个175.178秒，共382.445秒分配墙钟，无仿真重试；373产品源码及既有96个checkpoint/完整15rounds关联文件保持hash/大小/mtime，GPU与进程释放。结果和视觉链接见SHORT_FORCE_RESULT_ZH.md；本轮完成固定旧策略研究，不签发新外力reference资格或启动P7。

## J：按平衡恢复目标取消位移失败（2026-09-22）

用户明确要求取消位移限制，同时固定high80%和0.22秒空档。先声明NO_DISPLACEMENT_PLAN_ZH.md，再以同一Go2 D checkpoint完成一次192例公开dev评测，0训练/holdout/provider。产品仅Push运行状态和分类器新增显式null阈值含义，同步PLAN原始状态评分；有限阈值仍保持原义，位移诊断保留。不复制运行时、不使用巨大假阈值、无共享层或机器人特例。执行来源分离旧训练环境与新执行环境，源码差异限定为上述两个文件。

J总体142/192（44/52/46），四层48/45/22/27；重复三个seed为11/9/7。与I相比原27个越界例中21个成功、6个真实姿态/高度失败；其中重复18个对应13成功、5失败。旧边界前11/27条轨迹逐位相同，其他存在执行历史差，保留配对研究的边界。新J所有50个失败均为姿态/高度失败，没有位移失败或无效注入。完整结果见NO_DISPLACEMENT_RESULT_ZH.md。

两端各14项Push测试及9项评分测试通过；本地编译/Ruff致命检查和原始状态分类的有限/取消/跌倒路径检查通过。最初目标shell缺少未限定的python及PYTHONPATH，源文件尚未落地、仿真尚未启动时即失败；修正启动环境后才进行冻结和执行，没有物理重试。独立代码终审核对373文件，只含声明的两个产品文件变化，还原新增条件可精确得到旧SHA。

目标NumPy原始审计PASS；按预声明规则选两个越过旧边界的重复案例，原生回放共396帧完整解码、128个选定seed回合及有效物理数组逐位一致；两次都重放前两seed，共384例回放不计新统计样本。三个作业共197.435秒分配墙钟，exit0、进程及GPU释放；D/G1/c6全部关联文件和96个checkpoint hash/size/mtime不变。

主任务及独立视觉复核均支持：e55移位1.49米后逐渐撑起站稳；e57在0.172米高度触发门槛，但并未明显躺平。保留该解释边界，不为更高成功率再动高度规则。下载因网络速度很低曾切换传输方式；用户随后取消完整视频下载，已停止所有大文件传输。原视频与数组留在服务器，仅20项回执和13项视觉材料下载并核对SHA；没有伪称本地全量原始审计。已有三组本地视频保持原样。

用户之后再次要求本地视频：使用普通scp只传J两段原生MP4，33.47秒完成2,912,620字节；不调整网络设置、不重跑仿真。两份SHA与原回执一致，396帧在本地完整解码；新增第四组视频入口及local_video_delivery.json。此前慢速只是一段传输的观测，没有证据归因于Tailscale出口。NPZ仍留在服务器。

## K：G1 / Go2 外力评价统一落地（2026-09-22）

按用户要求将无位移门、high80%重复力、0.22秒空档收敛为两机器人共用的生产评价。先写FORCE_UNIFICATION_IMPLEMENTATION_ZH.md；新增任务拥有的ForceEvaluationProtocol与有限外力状态，沿已有RobotProfile/runtime/binding解析尺度、base角色、实际质量和ABI。训练push_protocol与评价push_evaluation_protocol独立冻结，旧checkpoint及原训练身份未重标。v3统计/公开投影/资格dev门均消费真实序列结果；公共层仅增加通用录像逐帧接入，以保存复位前失败末帧。

新seed基准1500000000、每格3×64。Go2为141/192（48/47/19/27），G1为33/192（29/4/0/0）；两格0无效注入，早跌的部分脉冲和未发生的第二推仍保留分母。独立重计384条原记录、校验力幅值/方向/累计冲量/时序，并调用生产dev门：Go2通过，G1低于20%下界。no-push各64例未触发倾角/高度失败且0恢复分母，末态稳定Go2为64/64、G1为44/64；审计纠正了把control_success等同末态稳定的文档表述，未改原数据或判据。zero-action各0/64。未消费holdout、训练或provider，不为改善G1分数改门或改统一协议。

四个成功/失败原生案例完整901帧解码，256条回放记录与原seed逐字段相同；查看36张关键帧，支持成功样本大幅移位后稳定及失败样本越过姿态门。G1失败末帧不是明显躺平，保留判据含义；本轮未另存逐步轨迹，不把回执一致性称作独立重建全部驻留。10个物理/控制/录像作业全部exit0、500.331秒分配墙钟，无物理重试。启动前相对路径预检修正与03准备版本不算成已执行的物理测量。

本地1696项回归通过、19skip；目标全量1696项/2779.269秒/0failure或error/2skip，最终OK且进程退出，原启动器未单独保存exit code。两项Isaac动态skip显式启用后另跑2项exit0，文档专项4项通过。过时评价YAML和10个顶层诊断/评分/标定脚手架已在两端按精确路径及SHA退役，必要冻结源码移入对应证据目录；旧G1训练JSON归档。活跃文档不再把G1-29历史行、Push旧V2评价或P4-only旧规则写作当前默认。目标全量结束后再次确认373产品源码与两格实测快照逐文件一致，D/G1/c6共1357文件/96checkpoint的内容、大小、mtime和文件集合保持不变。结果、边界与完整验收回执见FORCE_UNIFICATION_ACCEPTANCE_ZH.md；本轮工程PASS，P6整体仍NO-GO，不启动P7。

## 用户修订reference基础能力要求（2026-09-22，K完成后）

用户主动明确：reference无需进入20%等固定成功率区间，只需具有基础能力且不接近饱和。当前Go2 141/192、G1 33/192的原始成绩及分层均不变；G1已有真实恢复成功正例和仍需改进的困难层，不再仅因低于20%就判定其缺乏基础能力。先同步修正当前文档，随后按授权仅修改Push新外力reference准入，未改施力协议、episode成功条件或原始机器回执，没有新增训练或物理评测。

上节实施当时按原[0.2,0.9]生产门得到的Go2通过/G1未通过，以及更早各批的PASS/NO-GO，均保留为历史事实。范围明确限定为Push当前force新reference；Stair的20–80%参与冻结P3题池、learnable-cell/mask与评分，本轮维持，不取消其评分门。

Push新默认准入 `observed_outcomes_v1` 已接入代码：dev要求完整有效证据中至少一次成功和一次失败，public_accept/sealed要求观察到成功；旧相对改进、统计判据及accept→sealed最大0.1下降门不变。旧manifest缺少新策略声明时沿用旧规则，准入变化另绑定新身份。现有G1 33/192、Go2 141/192按新dev门复核均PASS，见[复核记录](reference_admission_revision_01.json)与[规则说明](REFERENCE_ADMISSION_REVISION_ZH.md)。这是收到新要求后的规则修订与既有证据复核，不是预注册重测或补发完整三相资格；自动门只排除全失败/全成功，近饱和仍看分层，不将1/192或191/192通过门写成余量充分。

该准入修订两端同组9模块各100项通过、0失败/错误/跳过、exit0。本地42.832秒，目标50.572秒（进程墙时51.601秒）；Ruff F/E9与差异空白检查通过。6产品文件、4测试文件及复核JSON共11文件SHA两端一致。原测试日志、回执和同步清单见[修订说明](REFERENCE_ADMISSION_REVISION_ZH.md)，不沿用K实施前的1696项测试数字冒充新代码验证。

# P3-NEW 开发进展

2026-10-01。**IN_PROGRESS；整阶段尚未验收。** 主计划 v1.4.28。开发阶段固定复用同步的旧原量与原图；诊断修正不再立即触发新采集。模型工作集、上下文与记忆继续按固定边界管理，不恢复作者 A/T 价值实验。

**最新接口收尾：**一次已登记的旧 focus-07 DeepSeek 开发调用正常返回，30个证据ID均有效，但字符串/对象形状不一致使原校验失败；原答和失败保留，无重试。当前统一入口直接解析证据ID，历史对象和错误路径仍严格处理，8项定向检查及明确标注的后验解析回放通过。详见[记录](./results/n5_confirmation_execution_r4/anchor_only_focus07_dev_r1/RESULT_ZH.md)。该次计量US$0.073839，当前账本累计US$1.120625436；不是独立N.5分数，未新增采集、训练或作者调用。

**最新评阅准备：**当前结构审计、全80事实工程屏障、本人原答/catalog绑定及三轴report入口已通过8项离线fixture；另用旧R4原件完成40/40真实材料接线，4个媒体包原11条观察全保留，真实seal/report仍为0。见[实际接线回执](./results/n5_confirmation_execution_r4/report_preparation_engineering_r1/development_materials/README.md)。Stair中立算术索引已通过4项定向检查、6旧包原字段保留和独立区间复算，见[索引证据](./results/n5_confirmation_execution_r4/focus_arithmetic_navigation_r1/README.md)。唯一旧focus-07干净SOL检查已返回：五问均有实质过程分析，29条原事实、95处引用通过结构检查，但全任务两类计数确有错误（原答127/37，原始成员为125/39），原答保留，不重试、不判语义PASS。实际输入156825/output14895 tokens，无已观察压缩或工具调用；详见[终态读查](./results/n5_confirmation_execution_r4/focus07_navigation_sol_dev_r1/development_readcheck.md)。

**R5运行准备：**500文件源码已冻结，58项相关工程检查及两任务CPU preflight通过；新cohort的实际seed、原checkpoint与模块来源已核，见[源码回执](./results/n5_confirmation_source_r5/source_freeze_receipt.json)。采集卡仍为候选，尚未采集、观察或正式评阅；按用户最新要求，不再等待过时全量图像归档，保留实际评阅/结论引用的关键图像及原始结果后，直接清理无用中间帧。2026-10-01 UTC 01:15 前后的资源检查发现两卡均被其他用户的 VLLM 占用；用户表示自行释放，当前不动该服务。实际磁盘/GPU门仍按原标准核对。当前作者8槽不增加，DeepSeek只留固定少量可行性请求。以上均不关闭N.5或证明训练收益。

**收尾范围收敛：**用户指出版本过多、留存与脚手架拖慢收尾，并进一步明确验收通过仍是目标。正式启动前先对采集后处理、评阅输入、派发和汇总做端到端接口审计，利用已有真实资料排除明显错误，不用新正式轮调试接口。若有失败或证据不足，保持 FAIL/INCONCLUSIVE 原记录，查明问题与修复依据后再决定必要重验，不无限换版本、不回写旧结果或降低验收阈值。仅完成必要清理、当前轮执行和验收，不新增作者有效性实验或外围平台。

**前序开发（R3）：**R3 采集与 CPU 核验已完成；因 Stair 部位身份反转，整轮保留开发用途和 INCONCLUSIVE，正式 facts/report 为 0。任务侧 calf 定位修正已在相同图片上完成两次新观察及 root 非盲模块检查，五句均有支持；这是开发结果，不回改旧六句。修正后的[离线投递](./results/stair_body_identity_dev_r1/offline_delivery/README_ZH.md)保留短报告 7,181 B、作者 user payload 71,154 B；仅一条视觉句进入短报告，其余在外部档案，非视觉内容不变。没有追加作者请求。

**前序载体检查：**固定旧 R2 media-04 的一次 GPT-6.1-SOL/high CLI 运行正常结束；[独立终态补核](./results/cli_clean_media_dev_r1/final_access_reaudit.json)确认 12 张原图按序实际进入工具返回、129 处引用可解析、原答结构完整、无已观察压缩。保留旧自动提取器的 TRANSPORT_NOT_VERIFIED 原回执，以单独补核说明 code-mode 看图包装；不重跑、不计正式语义票。最大单次请求输入 95,816 tokens；累计 1,049,554 输入包含 12 次工具往返，不能当作同时上下文占用。生产 DeepSeek 的[作者及观察者容量](./results/production_author_capacity.md)另行计量。

**当前工程交付：**两任务非盲审读与必要源头修复见[本轮工程收尾](./results/n5_confirmation_execution_r4/ENGINEERING_CLOSEOUT_ZH.md)：Stair增加回撤同段的roll/pitch、成功相容性及残差定义；Push增加曾恢复/末态/整回合成功的路径区别，并收敛解释措辞以容纳父代比较。默认报告7757/8192 B、11643/12288 B，登记压力分别最高7809/8192 B、12250/12288 B；只允许明确省略可选视觉，必需过程/反例/残差/权限不丢。使用已有R4资料，无新采集/模型/训练调用；最新模板的四类HTTP投递是离线回归，不是作者实收。新模板使用R4开发，不能继承R4独立资格。原正式批次与限额修订均已停止，固定80 facts为11完整、1引用失败、68未发，0 seal/0 report；原答和失败保留。独立N.5仍未完成，不记任何任务profile PASS。N.4作者8槽已用完，3-live/P4未启动。

以下按原时点保留历史记录；其中“当前”“尚未”等不替代本页以上状态和实际回执。

**v1.4.23 当前状态：**新版40包/80请求离线工程核验1456项通过；28包独立算术1274组、273983结构/标量零差异，来源补链28/28通过。两例开发检查已结束：Stair02一次DeepSeek（82有效引用），Push13一次干净GPT-6.1-SOL（121有效引用）；详见[原答、非盲复核和载体证据](./results/semantic_input_development_r1/probes/RESULT_ZH.md)。Stair仍有过程覆盖遗漏，输入历史槽顺序的歧义已从生成处修正；Push主要过程关系受独立开发复核支持。未重试，不发布N.5成绩。本次API计量US$0.0853377，总账US$0.723248796；桌面费用未知、实际运行token计数留档。混合双上下文接线已实现（62项相关检查通过），新正式R3的[源码冻结与CPU预检](./results/n5_confirmation_source_r3/README_ZH.md)已完成：485文件、84项测试、两任务实际静态预检通过。新GPT-6.1-SOL载体[一次性校准](./review_calibration_r3_gpt61/README_ZH.md)达到18/20事实判定一致率门，分歧不回写。用户授权确认后清理；已[逐哈希合并15238对重复PNG并清理临时包/缓存](./results/cleanup_20260930/README_ZH.md)，原图和全部引用路径保留，远端释放约26.35GB，清理后剩余46.17GB。新机器卡已冻结并部署；Stair于UTC 2026-09-30 14:23:06启动固定checkpoint采集（485源文件与权重再次核验），Push和正式facts/report尚未启动；[角色名单](./N5_REVIEW_ASSIGNMENT_R3.json)仅冻结评阅槽位，不是采集/API放行。P3-NEW保持IN_PROGRESS，P4/3-live未启动，作者8槽不扩充。

**v1.4.22 历史准备状态：**[v2证据块40包核验](./results/reference_blocks_development_r1/README_ZH.md)已完成，4413项检查通过，80请求最大文字209157代理token，加既定73728预留为282885；图片和SOL实际token不在此计量内。新 `related_v2` 已实现：恢复Stair足速/参考点净空完整直接列，六项省略量说明重建关系；两任务增加现有原量导出器的中立算术浏览索引；原始/related/媒体facts统一任务终局投影，去掉Stair额外诊断旗标与Push一致性核验bool。独立审查发现并修复Push focus缺测分支绕过投影的问题；5项相关输入测试、4项终局边界测试、9项既有材料测试和42项引用测试通过。当前正在 `results/semantic_input_development_r1/` 用实际旧资料重建与核验新输入；旧40包容量不能代替新版容量。另已[事前登记两包开发检查](./results/semantic_input_development_r1/PROBE_REGISTRATION_ZH.md)，须上述离线工程/容量核验通过后才可放行，目前尚未调用；不会重试旧episode-01或发布N.5分数。

**用户再次确认的架构边界：**独立诊断模块与模型观察者处理原档案/选窗画面，主作者接收短报告和必要的原始结果/测量/来源。全量N.5评阅包属于验收上下文，不能流入主作者；当前没有新增整段自由叙述的代理平台，生产数值骨架与模型形态描述分工保持。最新作者离线装配的诊断报告为Stair6930B、Push11084B；此处不是新增作者接收回执。

以下为上述修订前的过程记录；当前状态以上段及实际回执为准。

**本次进度核对与继续工作：**当前停留在 N.5 的证据输入修正，未进入 P4。`reference_blocks.v2` 已将结构块编号、原始地址和完整值相邻呈现；共享解析重建原材料，沿用 v1 目录身份，旧合同不回写。42项引用/编码/runner定向测试通过，真实40包 v2 无损核验正在执行，尚未做 v2 模型调用。[过程输入审查](./results/reference_anchor_probe_r1/PROCESS_INPUT_AUDIT_ZH.md)进一步确认：本例八个省略直接字段均可从已供应操作数重建，原答仍把部分量当成不可判断；直接姿态列也未充分形成时序关系。facts 的原始终局容器还携带生产 `diagnostic_flags`，需在材料投递源头剔除诊断标签，原始归档不改。后续须明确真正缺测、仅省略可重建表示与未分析完成，并评估恢复必要中立直接量；这些语义输入修正尚未完成，不能以定位测试或程序一致性代替整阶段验收。当前无新增训练、采集或模型调用。

**当前源头工作：**[引用契约修正](./REFERENCE_CONTRACT_REVISION_ZH.md)已接入现有runner/coverage/材料生成，23项定位单测、4项集成及9项既有材料测试通过；真实40包离线保真和文字容量核验完成：旧HTTP逐字一致、48原图保持、3910个锚点可解析，最大完整文字198055代理token。R2的[来源与焦点补证](./results/n5_confirmation_execution_r2/SOURCE_AND_FOCUS_SUPPLEMENT_ZH.md)核实144条Push原始字段链、1343项全池/覆盖/残差/代表与对照比较均一致；focus其余供应形式对象也已完成下述独立补证。已[登记一次旧episode-01开发检查](./results/reference_anchor_probe_r1/REGISTRATION_ZH.md)，只核新引用实际可执行性，预留US$0.15，不重试、不算正式资格；已执行且结果见下段。以下R2正式验收结果仍然有效，不能把定位工程或独立算术补证改写成整阶段语义通过。

**上述开发检查的后续真实结果：**唯一调用已完成但未通过。全部anchor ID和目录SHA有效，84处引用解析，4处仍错（元数据块误作原值块、局部路径漏父层）；原答保留，没有重试或facts seal。新增计量US$0.0869157，累计US$0.637911096，详见[原答与处置](./results/reference_anchor_probe_r1/RESULT_ZH.md)。正在审查编号与实际证据相邻呈现的源头方案，不立即启动下一正式采集。12focus剩余供应操作数的[独立补证](./results/n5_confirmation_execution_r2/focus_operand_audit_r1/README_ZH.md)已完成：Push19670、Stair38974比较对象均0差异；规范几何一致性、物理真值和语义资格保持分开。

**最新验收结果：**第二轮首个DeepSeek facts正常返回，但19个引用路径无效；冻结校验正确拒绝，故本轮 **INCONCLUSIVE**。实际调用1次、合格facts 0/40，SOL/report均0，后续批次已停止且无重试。原答未改；新增用量费US$0.0825996，累计US$0.550995396（登记费率，非发票）。当前工作转为证据锚点的源头设计和独立核验覆盖补证，不能把下方准备阶段状态或root模块票当作整阶段通过。详见[首包诊断](./results/n5_confirmation_execution_r2/README_ZH.md)。

**第二轮：**[执行记录](./results/n5_confirmation_execution_r2/README_ZH.md)为当前入口。477 文件统一源、18 项定向测试、两任务实际 preflight 通过，任务拥有的新诊断 cohort 已贯通。Stair 于 UTC 2026-09-30 08:20:49 完成采集，5,122 张原图；Push 于 08:57:08 完成采集，4,360 张原图。四次 fresh SOL 观察原答已收齐；root 分别查看全部24张实际输入原图后，Stair 3/3、Push 6/6 原句获支持，模块票于08:46:05和09:31:13封存，仅非盲模块准入。24回合独立算术范围内无差异；36数值+4媒体完整facts材料已生成，两项实验控制接线修正与旧失败明确留档。正式 facts/report 均未启动，容量及离线装配继续核对。[第二轮合同](./N5_RUN_CARD_R2_ZH.md)固定两任务 fresh SOL/high，双评规则不变。同一地形域的新 reset 轨迹不称新地形泛化；原轮仍 INCONCLUSIVE。

**第一轮历史：**[2026-09-30 第一轮执行记录](./results/n5_confirmation_execution_r1/README_ZH.md)。GPU 已按用户授权继续运行并完成 Stair 864 回合/4900 张 PNG、Push 192 回合/4696 张 PNG。Push 选择器误排除成功回合的问题使原冻结 N.5 为 **INCONCLUSIVE**；修复后同批仅做工程诊断，未发正式双评。原始视觉文字的 root 模块复核为 Stair 2/6、Push 5/6 supported；数值来源与任务定义的打包连接、任务拥有的视觉回合优先级均已修正并验收。原观察/review/数值内容不变，实际离线报告为 Stair 6956/8192 B、Push 11063/12288 B。Stair 的一次性独立载体开发对照另行登记，全部原答保留；不将模型或过滤后的表现写成独立资格。N.4 八槽已用完，后续仅离线序列化，未发新作者请求。

以下为采集前及各轮开发的历史状态，保留失败与边界；其中“当前”“未启动”等只指对应记录时点，最新事实以以上执行入口及原回执为准。

**当前入口：**[N.5执行合同](./N5_RUN_CARD_ZH.md)及 `results/n5_final_freeze_r1/capture_card.json`。r4的400文件与两任务实际服务器静态预检通过，唯一生产差分为此前Push的36字节缺测说明。采集准备与模型完整执行分门：旧长包失败不回判，修复材料源头后不再追加同例调用；新资料按冻结规则生成，完整语义未评完仍不能通过。以下旧段中的r3及“下一步”保留历史时点，不覆盖此入口。新采集仍因既有vLLM占用GPU未启动；无新增作者/观察者/评委调用。

当前待解决的是评阅任务能否实际完成及新资料上的语义资格；两任务数值骨架不重复改造。r3源快照的两项真实静态预检、检查点字节、资产缓存及来源绑定已核对，正式采集未启动。512k独立CLI第一次旧报告返回0 claims；剥离整轮管理指令后的一次执行返回47 claims、37/41项目录已评，仍有四组共享定义未核及匹配接口错误，均不算通过，不继续追加同例模型尝试。已从源头明确匹配键名，并补齐与原档案 schema hash 绑定的两任务奖励输入合同，见[合同修复](./results/review_task_contract_development_r2/RESULT_ZH.md)。汇总与原记录见[当前收敛说明](./EXECUTION_CONVERGENCE_ZH.md)。下文保留各历史轮次的开发结果，其“下一步”不构成重复运行指令。

**最新收尾：**N.4累计8次实际调用已经收齐，[最后两槽](./results/native_author_current_r1/RESULT_ZH.md)内容送达均通过；Stair候选静态通过，Push一次改7个机制被拒绝，不重试、不训练。本次约US$0.04248，8次作者共US$0.24309；桌面消耗不包含在内。后续仅在Push任务层澄清了数值缺测的模态范围，默认11344/12288字节、38项检查通过，见[措辞修正](./results/push_observation_scope_r1/RESULT_ZH.md)。该后续文字未发送作者，当前源码已不同于r3预检快照，最终N.5冻结须更新。两GPU仍各占29024/32607 MiB，未停止既有vLLM；新N.5采集与独立语义验收未完成。

## 当前源头修订

已完成[部位几何与相机定位接线](./BODY_GROUNDING_NEXT_ZH.md)、稀疏r1和完整逐步r2工程实采。r2的192回合、12个有效数值字段及全部终局保持，935时刻/1870双视角原图完整；主代理实际查看本次输入全部12张原图。实际r11为42个S0、5条测量事实、72点、12原图、29.39MB，同源输入独立重构通过。一次真实回复因JSON末尾缺括号机械失败；原文主要复述像素差，且混淆脚坐标缺失与图像可见性，**诊断输出不通过**。原答不修不重发；同源失败投递10966/12288字节，0视觉S3、S5明确解析失败，数值和非视觉段保持。见[r2工程记录](./results/body_grounding_dense_r2/README_ZH.md)和[r11完整复核](./results/body_grounding_observer_r11/R11_REVIEW_ZH.md)。

从任务源头修复默认Push只详述第一未恢复窗、漏掉后续恢复与再次失稳：全部恢复单位先简述，最多一窗详述，条件组合n/N与实例时间序列分开。34项针对检查通过；默认10937字节，附父差分及最大4字节Unicode视觉压力后11893/12288字节，必需数值/骨架/反例/残差均保留。见[真实旧开发新投影](./results/push_sequence_projection_r1/README_ZH.md)。

[事前登记r12](./R12_REGISTRATION_ZH.md)已完成：全新SOL/high、同一盲文字和12原图、一次原答，机械通过；主代理结合画面支持全部三条端点形态。实际生产投影11308/12288B，数值与非视觉段不变，见[完整复核](./results/model_carrier_probe_r12/R12_REVIEW_ZH.md)。不同载体的服务端fingerprint/usage不可得，费用未知；不作普遍模型排名或N5结论。暂不增加图像标记或新平台。

最长旧Stair实际facts接入后为173243代理token，静态预算可容纳；一次真实report执行却发生上下文压缩，返回complete=false及1.59MB原答。这个失败已保留，见[执行记录](./results/capacity_report_execution_r1/RESULT_ZH.md)。按[职责修订](./REVIEW_RESPONSIBILITY_REVISION_ZH.md)实施两任务独立全量复算，让模型核全部解释/关系与必要过程，程序票不进入模型分子。原始完整曲线和原对象不删，新材料/覆盖/容量仍须核验；确认资料尚未采集。

根据r7–r9反复出现的姿态方向、复合命题和蹲起误述，重新核查PoseScript/PoseFix、MotionScript、REFLECT、逐事实核对等研究及作者代码，形成[语义与证据分工修订](./SEMANTIC_GROUNDING_REVISION_ZH.md)。已实现任务拥有的root姿态短事实、无损去重的S0输入，以及绑定原件的逐观察语义review附件。没有明确支持评阅的自由文字现在只记入S5缺口；不能凭机械PASS或单个数值谓词一致进入S3。数值过程、成功反例和残差保持。112项相关单测与9个实验入口离线场景通过，后者见results/semantic_grounding_revision_r1_sender_cpu.json。

固定两个旧开发案例已先登记[一次性r10检查](./R10_REGISTRATION_ZH.md)，冻结395个文件。24张原始图片及已有S0与r9完全一致；Push新增roll/pitch来自原档案。去重后请求文本Stair由12297降至8325字节，Push由10345降至9511字节，没有减少精确数值。见[准备回执](./results/diagnostic_module_probe_r10/control/preparation_receipt.json)。

r10已各完成一次、无重试，两份机械PASS。主代理重新查看24原图后，Stair原始3条均支持，Push只支持左肘1条，左膝类别跃迁和手的身份/起点2条矛盾；另有一条not_visible错误质疑同状态多视角。**原始语义质量仍不通过**，不以过滤后的结果改判。实际review投递回放通过：未评阅时不送视觉S3，评阅后只送支持的原句，全部数值材料与非视觉报告段不变；最终Stair7772/8192B、Push10970/12288B。详见[r10全体复核](./results/diagnostic_module_probe_r10/R10_REVIEW_ZH.md)。下一源头缺口是部位身份与局部几何，root事实不能替膝角或左右手定位；不能靠叠提示或重复请求刷通过。

## 当前已经核实的结果

v1.4.14的[独立算术入口](./results/n5_arithmetic_split_development_r1/README_ZH.md)已在两任务各12个旧回合完成CPU核验：Stair4367事件，Push16恢复单元；已声明检查一致，8种实质错误变异均被发现。底层测量生成、身份及语义未核范围明确保留。report-only `arithmetic_audit`实际构建/阶段边界通过。保留完整原始轨迹和完整派生的[单次新职责执行检查](./results/semantic_execution_scope_r2/RESULT_ZH.md)已返回45KB原答、未观察到压缩；但root发现4个claim/7个fact-match定位格式错误及共享条款覆盖追踪缺口，不凭complete=true放行。Stair原始力/历史范数与表面导出操作数也已完成12回合、52710标量补核，见[补核回执](./results/n5_arithmetic_operand_link_development_r1/stair/README_ZH.md)。

| 工作 | 结果与证据边界 |
|---|---|
| 任务数值/语义主线 | Stair/Push 各自拥有词条、时间关系、成功反例、残差和选窗；公用层只管理合同、引用、容量和投递。开发数值重算与现状见 [N1/N2 记录](./N1_N2_COMPLETION_ZH.md)及 results/*_development_current.json。 |
| 原生逐步双视角 r3 | Stair 3 个首回合共 1,902 张原图；Push 共 1,870 张。真实 1920×1080、两机位、逐控制步、各回合按自身 T 裁尾，包含真实末步。来源见 [采集卡](./results/dual_view_development_r3/capture_card.json)。 |
| 有限双环境录制 r4 | 两任务各 2 env × 3 seed 同 rollout；Stair 864 回合/300105 有效步、Push 192 回合/67331 有效步的全部原始字段 dtype/字节及终局，与 r3 完全相同。122 张稀疏原图绑定与裁尾正确；此次固定点未含任何真实末步，终止帧标记全部正确为 false。见 [工程核验](./results/multi_env_engineering_r4/README_ZH.md)。 |
| 媒体/数值同轨 | Stair root pose 和具名关节在已核样本精确相同；Push 已定义重叠量误差 ≤1.19e-7。缺少共同基准的量不核；元数据对齐不是像素或全部物理字段的真值证明。 |
| 像素检查 | r3 实际准备的 Stair/Push 原图已逐张查看；r4 两任务 run1、两 env、step1/80、双视角共16图全身范围与环境参照可辨，见 [抽样原图检查](./results/multi_env_engineering_r4/pixel_review/pixel_review.json)。保留白色 Stair 的低对比、自然遮挡及稀疏时间缺口；不据此认证所有原图或正式语义。 |
| 实际作者文本烟测 | 4 个旧开发案例各1次，均形成终态并通过候选静态校验；见 [执行摘要](./results/text_author_smoke_r1/execution_summary.json)。非盲读法检查记录了总体/子组混用、次数/窗分母混用和省略成功反例，原产物不改写。 |
| 评阅开发校准 | r1 原始一致率70%；按事前全体规则修订后 r2 为18/20=90%，剩余两项分歧保留。见 [校准结果](./review_calibration_r2/RESULT_ZH.md)。正式评者必须是全新无开发历史 GPT6-SOL 与 DeepSeek，开发子代理不能改任正式评者。 |


v1.4.15 新进展：[Stair进度过程源头修复](./results/stair_progress_process_development_r1/README_ZH.md)。864回合重算的全部旧测量/终局相等；新增2925个回撤与重返/截尾片段，独立12回合49段通过。数值报告6515B；最长四字节视觉正文/不确定性与合成父差分8177/8192B，过程、成功反例、残差均保留。76项相关单测通过。组内残差和对照合并，旧字段分母不删，正式词义/奖励/终止不扩张。新确认仍未启动。

## 真实观察请求暴露的问题与修正

旧录像两次开发观察因 not_visible 超限机械失败，来源和画质问题保留于 [旧视觉复核](./DEVELOPMENT_VISUAL_REVIEW_ZH.md)。旧拼图曾有均衡和强制缩放、机器人过小，不继承其画质资格；当前使用原始全分辨率 PNG。

原生 Stair v2 又进行了2次实际调用：r3 为16图，r5 为8图，累计24图。两次均收到真实回复，但机械检查未通过；r3 有多余顶层字段等错误，r5 有空间可见性与时间稀疏混用、文本超限及没有极性的数值标签。全部原答、回执与无效项保留，见 [索引](./results/observer_contract_r5/artifact_index.json)。Push v2 仅准备了16图，未发送，见 [处置记录](./results/observer_contract_r5/push_v2_disposition.json)。

v3 从源头简化职责：模型给身体、逐图引用、形态文字、空间可见性、状态/变化关系及不确定性；代码由引用推导时间范围、引用步与间隔。可选量化命题改为 `{id, asserted}`，任务比较真实谓词与声明极性，缺测不可核；这不验证整句自然语言。原始输出不补字段、不纠正、不删除无效项。67项针对性 CPU 检查通过，见 results/observer_contract_v3_r6/cpu_verification.json；旧结果不能用新合同回判为通过。

v3 r7 两任务真实请求已返回且机械 PASS，但开发检查发现 Stair 的 nose-down 与资产绑定的前向轴方向相反；Push 比较缺少基线引用且有跨稀疏间隔的持续性措辞。全部原答和机械结果保留，不能把机械通过写成语义通过。详见 results/observer_contract_v3_r7 和 [前向轴核验](./ROOT_FRAME_DIRECTION_AUDIT_ZH.md)。

已完成有资产依据的外置 root-frame 定义（不改变原 profile hash）、任务前向轴 S0、完整比较引用提示与过程优先投影。主计划先登记一次新的 r8 开发检查：两任务各一次固定原图、high thinking/32768、新预留 US$0.10，历史图片和费用累计；不是正式语义确认。r8 70项相关CPU检查通过；最长中文容量压力仍为Stair8035/8192、Push12264/12288，反例/残差和数值不变；24个生产序列化离线帧、464项句子检查通过，见 results/visual_projection_observer_v3_r8 与 results/delivery_observer_v3_r8。r8两任务均收到真实原答：Stair机械PASS，文字由nose-down改为head-end-up，与已核资产前向轴一致；Push三条原句有一条正文超限，整体机械非PASS，保留该无效项，另一条有完整基线引用的过程句进入默认报告。不能把此开发检查写成独立语义通过。r8新增2次调用，实际usage按峰时价上界US$0.024548172，见 results/observer_contract_v3_r8/usage.json。

真实最大媒体形状容量检查已补齐：每条观察6步×2视角、12条独立原图引用，同时保留200字正文、120字不确定性、反例/残差/S4。Stair8037/8192B、Push12266/12288B，父出处与12个逐图出处均可解析。引用未进入公共id/text投影，因此未为无收益的压缩修改生产逻辑，见 results/visual_projection_native12_r1/result.json。

任务事件媒体选择已在生产任务模块实现并经37项相关检查及旧r3六回合72张原图核验：最多6个时刻，先保留首/中/末，余3点覆盖任务事件前中后；偏移按实际dt计算0.5秒，不补造帧。Stair优先已确认姿态/运动过程，Push绑定真实执行扰动与实际响应/恢复窗口。事件优先级只是取景顺序，不是原因或严重程度判断。两回合×六时刻×双视角的24图用于N5待冻结卡；本步无新采集或API，见 results/task_media_selection_r1/result.json。

## 接下来的顺序

v1.4.14当前顺序以[验收职责修订](./REVIEW_RESPONSIBILITY_REVISION_ZH.md)末节为准：独立复算覆盖、完整语义材料与实际容量、最终冻结、新N.5。下列r8/r9等是保留的历史诊断与后续已完成事项，不将其“下一步”误作当前重复调用指令。

[诊断模块开发验收](./results/diagnostic_module_review_r1/REVIEW_ZH.md)及[固定三例r9复核](./results/diagnostic_module_review_r1/R9_REVIEW_ZH.md)已完成，**结论不通过**。原r8全部5条观察和实际投递句已结合24原图核对，另看三回合新selector36图，共52张不同原图。r9的3次唯一真实回复中，1次机械PASS、1次无效JSON、1次顶层结构失败；机械PASS也不代表语义通过。Stair抓住末段翻倒但把80步前部抬高写成平直；Push仍将不同步局部变化写成整体蹲起。原答/原机械结果保留，不重试刷通过。下一步在任务/机器人层提供少量定义明确的同轨部位变化事实，模型补环境关系及未知形态，逐句核对后再投递。模块与主上下文边界见[设计](./DIAGNOSTIC_MODULE_BOUNDARY_ZH.md)。

本轮同时修复两个实际工程边界：合法长Unicode视觉句及加入父差分可能超报告字节上限，现复用整句容量规则，省略只针对可选视觉原话，数值/反例/缺口与冲突计数保留，见[容量回执](./results/visual_projection_capacity_fix_r1/RECEIPT_ZH.md)；无效JSON/顶层结构失败现在明确产生视觉未交付S5，不中断数值报告、不冒称零错误，见[真实同源离线回放](./results/native_observer_failure_boundary_actual_r2/receipt.json)。95项相关单测和9个实验入口CPU场景通过；真实r9原答回放仅验证解析/投影边界，数值主线内容保持，不作为新语义通过。旧冻结源与结果未覆盖。

1. r8一次性开发观察已收齐；原答、机械失败和用量均保留。新确认资料未使用。
2. 两任务各1次原生视觉作者烟测均已收到唯一真实回复；加已完成4次文本，共6个 episode，无重试、无训练。Stair原终态FAILED保留：旧material源码污染静态检查子进程的导入路径；修复实验入口后离线admission/delta通过，并按真实回执补齐journal，恢复记录为static_failed/verifier_error，不把历史FAILED改为候选成功。Push使用单独的恢复启动卡，只执行原lane-b一次，candidate_verified/PASS/not_run。两包均使用自身数值、完整派生和真实观察输出。准备时发现并修正Stair证据身份把resolved-cell绑定哈希误放进base任务语义哈希的问题：严格生产闸门不放宽；旧回执保留，独立复建相同HTTP字节与原审计，仅派生身份更正另留记录。S4父子域只做本阶段既有离线覆盖，不伪称两份单策略包形成父子配对。详见results/native_author_stair_recovery_r1与results/native_author_push_receipt_r1。
3. N.2仍为开/中/末±25步数值检查窗；独立的“基础时间覆盖+任务事件过程”媒体选帧已实现，正式选择卡待冻结；原始密采与模型输入工作集分别记覆盖。完成 N5 原始事实资料与判分规则冻结。24个确认回合一次性双评，双方先封存事实，再看报告；全包转开发规则和阈值保持。见 [准备清单](./N5_PREPARATION_ZH.md)及 [评阅协议草案](./N5_FACT_REVIEW_PROTOCOL_ZH.md)。
4. N5 新采集、正式双评、3-live、P4 均未开始。工程通过不等于独立语义通过。

## 费用与历史记录

截至r10的历史快照，实验 API 共22次：合成图读回1、旧机器人观察2、评阅校准2、文本作者烟测4、原生观察11、原生视觉作者2。r9累计US$0.359492028；r10新增2次、US$0.025706604，按实际usage与登记峰时费率计算累计上界 **US$0.385198632**，不是账单，见[r10用量](./results/diagnostic_module_probe_r10/control/usage.json)。共6次作者调用的独立共享账本合计US$0.200609736，已经包含在上述总数中，不能再次相加。桌面Codex/子代理费用未知，不能记零或混入实验provider账本；旧失败和错误输出保留在累计数中。部位定位r1持GPU锁101.042196685秒，完整r2为887.111022876秒，均无训练。r11实际新增1次API、US$0.0177084登记费率计量上界，当前累计23次/US$0.402907032，见[本次用量](./results/body_grounding_observer_r11/control/usage.json)。r12及容量facts为另记的桌面子代理，不混入DeepSeek调用计数，费用未知；N5尚无实际调用。

双视角 r3 Stair 持锁2387.186286秒、Push853.836686秒；r4 Stair386.299754秒、Push116.553185秒。按每次保守占两GPU容量分别计1.326214603、0.474353714、0.214610975、0.064751770 GPUh；这些是资源记录，不是停止条件。之前失败采集和机位开发仍计费/计时并保留：r1超时、r2实际RGB尺寸配置错误等不从历史抹去。未启动新训练。

更早的详细开发快照见 [v3修正前记录](./results/observer_contract_r5/development_status_before_v3.md)；其中“待采集”等时态只代表当时状态，以本页和各自封存回执为准。

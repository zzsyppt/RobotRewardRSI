# N.5 第二轮执行记录

2026-09-30。**本轮 INCONCLUSIVE：首个正式 facts 请求的引用校验失败，后续调用已停止；整阶段仍 IN_PROGRESS。** 两任务采集、模块复核及离线装配完成。DeepSeek 实际请求1次、合格facts 0/40，SOL正式调用0次，report调用0次。执行定义见[第二轮合同](../../N5_RUN_CARD_R2_ZH.md)。第一轮仍为 [INCONCLUSIVE](../n5_confirmation_execution_r1/README_ZH.md)，旧观察、评分分歧与工程检查不转成新轮成绩。下文的“尚未调用”等为对应准备时点；当前结论以本段和末尾首包记录为准。


启动回执：[stair.started.json](./stair.started.json)。UTC 2026-09-30 07:01:32 启动，driver PID 3606193、入口 PID 3606196；GPU 仿真进程已出现。477 个来源文件及 checkpoint 字节在启动前通过核验，磁盘空 37,896,851,456 B，高于 Stair 登记门。实际 [capture_card.json](./capture_card.json) SHA 为 `6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794`，统一源 SHA 为 `497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76`。18 项定向测试和两条实际 preflight 通过；没有新模型请求。

Stair 于 UTC 08:20:49 正常退出，约 79 分 17 秒；wrapper `PASS`、numeric `complete=true`，六段首回合共 5,122 张原 PNG。采集身份核对的第一条临时断言误把 wrapper 的 `sha256:` 前缀当作差异；按实际格式逐字检查后，权重、评估绑定、种子和 full 原件均一致，未修改任何结果。Push 于 UTC 08:23:28 启动，driver PID 3637722、入口 PID 3637775，启动前空余 29,868,654,592 B，高于登记门。采集通过尚不代表语义验收通过。

本轮密集捕获成本偏高：两个 env、两个视角，每步各视角先丢一帧再保存一帧，共八次渲染、四张 PNG；另有 CPU 原图压缩与保存。GPU 利用率的单次读数不能证明具体耗时比例，尚未做 profiler 分解。六段实际首回合长度为 306、302、308、557、451、637 步。后续性能优化应减少冗余捕获与渲染，保持必要时窗、真实相机和原始图像质量；本轮已冻结的采样/描述/验收规则不作事后变更。

Stair 两个 fresh SOL/high 观察调用已返回，各自原答、真实访问回执及调用身份见 [observers/stair](./observers/stair/desktop_invocations.json)。冻结机械核对通过，共三条原始观察；root 实际查看全部 24 张原图及定义后，按正文和 uncertainty 全命题口径判三条 supported，于 UTC **08:46:05.242044** 封存[模块审查](./observers/stair/root_module_review.json)与[回执](./observers/stair/root_module_review_receipt.json)。这是非盲模块准入判断，**不是独立 N.5 准确率或资格通过**；三条原句均保留在后续正式视觉分母，不改写、不重试。画面能支撑粗粒度机身—台阶关系，不能据此补出受力、真实足底接触或未显示的连续运动。正式 facts/report 调用仍为零；桌面调用费用不可得，未计作零。

Push 于 UTC **08:57:08.255676** 正常退出，约 33 分 40 秒。六段首回合为 412、412、412、412、120、412 步，共 **4,360 张**原图；三个数值档案 complete，权重、当前评估绑定、run/env 与 full 原件检查一致，见 [采集核对](./push.capture_check.json)与[退出回执](./push.exit.json)。两任务采集已结束；Push 四步媒体后处理与两任务数值八步 CPU 后处理已启动，实际成功仍以逐步回执为准。

数值八步已于 UTC 09:01:36 全部 exit 0，24/24 回合在既定独立算术范围内无差异；72 份数值 facts 完整文本的最大 DeepSeek tokenizer 计数为 174,126，加既定预留为 247,854。Push 独立原档案到物理构造的缺口仍在，静态容量也不代替实际模型评完。

Push 原 export 成功，但 selector 在选例前读取旧 `cohort.derived_public_dev_runner_seeds` 路径报错；R2 卡的派生列表已在顶层，源头应使用已有任务种子函数。当前主线 selector 改为从 `cohort.eval_base_seed` 调用同一 `derive_runner_seed`，不再依赖重复列表的位置；R1/R2 均与登记值一致，16 项既有种子/选帧测试通过，选例函数 AST 未变。原 477 文件冻结树、卡、采集与错误日志全部保留；实际后处理使用单独封存的[修正控制来源](./postcapture_selector_fix/registration.json)，不能宣称与旧 selector 字节相同。此修正仅处理字段读取，不改结局分类、哈希排序、时刻选择、提示或评分；正式调用尚未启动。

Push 两次 fresh SOL/high 原答的机械核对通过；root 看完 24 张原图，并按正文与 uncertainty 的全部命题核对 6 条原句，于 UTC **09:31:13.024858** 封存[模块审查](./observers/push/root_module_review.json)及[回执](./observers/push/root_module_review_receipt.json)，6 条 supported。描述限于可见手臂位置、脚的投影间距及端点变化，没有由截图推出接触、连续恢复或因果。与 Stair 一样，这是非盲模块准入，不代替独立 N.5；所有原句保留在正式分母。

4 媒体单位、8 份 facts 已在 UTC 09:27:01 完成。首次生成器错误造出 v1 正式选择，和既有 native 的 v2 合同不一致；已在源头改用原完整 v2 选择及真实 alias，原失败和冻结源码保留。新控制来源与实际产物核对见[媒体接线修正](./postcapture_media_fix/README_ZH.md)。48 原图、完整窗口、alias、数值来源及请求重建均一致；不更改选帧、描述或判分规则。媒体最大文本 184,862 token；图像开销须另计，静态计量不等于实际评完。

## 本轮已明确的边界

- Stair 同一 checkpoint、任务与地形 suite，显式诊断 cohort `n5_r2_20260930`；新 namespace 改变实际 reset 的 runner seeds。旧 `v1` 原义保留，不增加实验专用生产目的或任务公用层语义。
- Push 同一合格 checkpoint，base seed `1600000000`，通过原入口重新生成实际比较绑定。历史未用性核对仅覆盖本任务既有运行承诺，不声称穷尽整台服务器的私下执行。
- 统一来源纳入已验收的终局分组、数值来源/任务定义绑定、任务拥有的视觉优先级，以及本轮诊断 cohort 入口；代码清单和服务器预检必须在启动前封存。
- 两任务各两个入选媒体回合，每回合一次 fresh SOL/high；正式评者仍为 DeepSeek 与另一独立 SOL。root 模块复核在正式 facts 前封存；过滤前原句分母不变。
- N.4 八个作者槽已用完，本轮只生成离线作者请求，不发新作者请求、不训练、不做价值 A/T。

## 存储准备

[存储回执](./storage_preflight.json)记录第一轮原图的只读统计、最大帧数依据及真实 PNG 小样本兼容检查。新 export 使用 `--png-storage hardlink`，只对完成采集的 PNG 同盘建立硬链接；路径和内容核验保持，元数据及提取 NPZ 独立写出。硬链接共享 inode，不是额外备份，原件和 export 均不得修改。旧 capture/export 未改。

按每任务实测最大 PNG 字节×最坏帧数×1.5，再加 2 GiB 元数据/数值余量和 8 GiB 操作余量：Stair 启动门 31,162,953,808 B，Push 启动门 28,276,851,520 B。这是有 50% 余量的压缩估计，不是理论上界；串行每任务前重查。第一轮实际复制的约 16 GB export 不再作为新轮额外图像开销。

## 尚未通过的范围

采集等待期间已整理 [N.1–N.4 证据索引](./N1_N4_EVIDENCE_INDEX_ZH.md)与[后处理命令清单](./postcapture_commands.md)。前者区分已有工程事实、历史样本结论和未核范围；Stair 的四项 CPU 导出/选择/准备已执行通过，其余以后续回执为准。独立算术只读取控制侧完整派生，不把报告提前放进 facts 输入；四份任务定义通过 pack 的已实现来源绑定参数接入。旧源码和本轮冻结卡均未因这两份索引改变。

[数值核验入口](./verify_numeric_arithmetic.py)与[完整请求文字计量入口](./count_prepared_text.py)均已实际执行，结果见[数值核验回执](./numeric_readonly_review.json)及[媒体修正记录](./postcapture_media_fix/README_ZH.md)。前者调用冻结的既有核验函数；后者复用已缓存的冻结 tokenizer，不联网下载。实际 SOL token 和实际上下文峰值仍未测；静态文字计量不承担语义放行。

完整长包仍须实际完成双模型语义评阅；容量估算和独立复算不代替模型理解。Push 独立派生检查没有独立认证全部 NPZ 来源操作数及姿态物理含义，按原责任清单保留未核范围。两任务采集及原图模块查看已结束，必要过程召回及正式句子支持率仍无本轮独立验收结论。

历史 DeepSeek usage 按登记单价计量 US$0.468395796，桌面费用未知单列；本轮原预留正式 DeepSeek 最多 US$12，失败即停止，不把预留当实际支出。

## 首个正式 facts 的真实失败与后续边界

`episode-01` 的唯一 DeepSeek 请求于 UTC 10:16:40 发出，10:18:14 返回，`finish_reason=stop`；不是网络失败或输出截断。原答案自报 `complete=true`，冻结机械检查发现 19 条 `unresolvable_reference`，因此真实回执 `complete=false`。控制器停止，未重试、未生成 facts seal、未进入 report；其余39个 DeepSeek 包及全部 SOL 包未调用。

原答案、覆盖检查、停止记录、账本及七个文件的远端/本地 hash 核对见 [actual_first_call](./deepseek_facts_control/actual_first_call/recovery_manifest.json)。本次 prompt 174,172、completion 25,290 token，按登记费率 US$0.0825996；累计 US$0.550995396，非服务商发票。

[逐项定位诊断](./deepseek_facts_control/actual_first_call/reference_failure_diagnosis.json)确认：对应字段实际都存在；错误是 `geometry`、`contact_history` 漏 `definitions` 父层，`success`、`failure_priority` 漏 `episode_outcomes` 父层，以及力范数位于 `direct_columns` 却被写到 `raw_discrete`。19项为五类路径错误，不等于19个已证明的语义错误。校验器正确拒绝这些地址；诊断所列真实位置仅解释失败，不能替换原答、补封存或补PASS。语义与必要性仍未完成独立判定。

SOL 输入40包的正文/原图等价检查完成，但桌面更新使登记旧CLI路径消失，当前为0.159.2；此环境变化和0调用状态见[SOL控制预检](./SOL_FACTS_CONTROL_PREFLIGHT_ZH.md)。任何后续执行须如实登记实际载体，不冒称旧二进制。

离线生产装配见[assembly_receipt.json](./delivery_offline_r1/assembly_receipt.json)：Stair报告6,930/8,192 B，Push11,084/12,288 B，`offline_serialized_not_sent`，数值/非视觉内容不变，新增作者调用0。报告预算不等于整个user payload大小。[工程审查](./DELIVERY_ENGINEERING_AUDIT_ZH.md)区分了已核范围、Push来源链补证和12焦点组级确定性检查仍需闭合的范围。

本轮不能继续冒充一次成功确认。后续引用材料/输出合同的修正只在新版本与开发材料上实施；原冻结树、采集、原答与本次失败保持。先解决证据定位的生成职责及独立核验覆盖，再按未使用cohort纪律重新确认，不立即重采或重发同包刷通过。

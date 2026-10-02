# R4 独立确认执行记录

2026-10-01。**原R4及限额修订均INCONCLUSIVE，整个阶段尚未验收。** 原批次因32,768输出截断停止；修订批在focus-07发生真实引用路径错误后停止，所有在飞已自然终态，未开启report。原图、原量、生产诊断及被评报告保持，不为评委引用错误重采。

## 当前状态

**最新工程修订（不改变原确认结论）：**[非盲审读与收尾](./ENGINEERING_CLOSEOUT_ZH.md)已在任务源头补齐姿态轴、成功相容性、恢复路径和规则含义，并修复Push加入父代后容量超限。当前候选为Stair r1（7757 B）与Push r2（11643 B）；两任务登记压力分别7809/8192 B、12250/12288 B。旧冻结源、原始图/量、评者答案及r1失败保留；新开发输出另存，没有新模型/采集/训练调用。以下R4的“原生产诊断保持”指冻结原件，不指现在的开发源码；新模板已使用R4开发，独立N.5不因此通过。

UTC **22:27:31**，修订SOL在飞工作全部自然终态；DeepSeek已于22:17:06停止。固定80事实槽累计11完整、1引用失败、68未派发，0 seal/0 report。focus-07只有NF-07的一条块内路径多了`episode_outcomes`层：原输入完整、其余24引用可解析；finish_reason=stop，completion=20,943，未耗尽65,536。这是评阅答案定位错误，不等于生产诊断语义错误；原答不修、不重试。监督原件保存在[修订终态目录](./control_cpu/facts_runtime_supervision_amendment_r1/)。当前进行[Stair](./engineering_review_stair/scope.json)与[Push](./engineering_review_push/scope.json)非盲工程审读，不生成正式report或N.5分数。下段为原时点记录。

UTC **22:04:17**，修订登记中的唯一恢复请求完整返回；[原件与根核查](./amendment_r1_recovery/root_gate.json)记录12条必要事实、引用/结构错误0，原请求除`max_tokens`之外完全相同。实际输出25,116 tokens低于旧上限，因此只证明该次新请求完成，不能因果归于提高上限；新增计量US$0.030555072，阶段累计US$0.902730636。没有语义评分或facts seal。随后根启动剩余68个SOL事实槽（最多2并行，原输入与argv保持），见`sol_facts_remaining_r1/start.json`；DS其余6槽仍待新控制核对，report保持0。

UTC **21:45:55** 前两批已收尾：SOL四份、DeepSeek一份原facts完整且通过机械/结构检查；另一个DeepSeek原答因输出限额不完整，共6次实际facts调用，0 seal/0 report。原轮[不可覆盖终态](./original_round_closeout/result.json)保存INCONCLUSIVE；新DeepSeek计量费US$0.148926768，P3-NEW累计US$0.872175564（登记费率，非发票；桌面费用未知）。全部原件保留，不将不完整计零分或诊断错误。

源头见[输出限额定位](./output_limit_failure.md)。[运行修订登记](./RUNTIME_LIMIT_AMENDMENT_R1_ZH.md)只提高DeepSeek评委输出cap至65,536，保留五份完整原答，只允许当前截断槽一次新独立请求；其余未发槽统一新额度，不改消息、图像、评分或生产代码。先完成这个恢复槽的原请求差异/完整性核对再续新批。后续结果只能称同一冻结cohort、执行参数后登记的工程语义验收，不能改写原预注册轮。

以下采集、装配、首派发均为实际历史步骤；其“尚未启动”等只描述当时状态。

- Stair 于 UTC **2026-09-30 19:59:50** 完成，采集 exit 0、结果 PASS、public-dev binding 与冻结卡一致。原始终态及结果保存于 `downloaded_capture_control/`。任务级导出、固定选例及观察输入准备全部完成；两个盲包原图逐字节移交后，已各发出一次 fresh GPT-6.1-SOL/high 观察，登记见 `stair_observer_inputs/desktop_invocations.json`。尚无独立语义成绩。
- Push 于 UTC **20:39:35** 完成，driver PID `3877432`、child `3877435` exit 0，结果 PASS，public-dev binding 与冻结卡一致。任务级导出、选例和观察输入准备已通过；两个固定盲包分别取得一次 fresh GPT-6.1-SOL/high 原答，访问与逐句模块审查均已完成（结果见下段）。原回执见 `push_observer_inputs/`。未安排重采或模型重试。[旧 PNG 冷归档和清理](../cleanup_20261001/README_ZH.md)已完成：3,740 张逐成员校验，root 重读 tar 哈希一致，随后仅移除其 7,480 个 source/export 名字，净释放约 6.9 GB；新采集与非 PNG 未动。
- UTC **20:47:05–20:50:45**，数值打包、媒体打包、路由验证与九项独立核验全部 exit 0；[原证据及摘要](./control_cpu/completed_sequence/cpu_summary.json)已逐文件下载核 SHA。覆盖 24 个完整回合、12 个焦点、4 个媒体单位；24 回合派生、28 个输入算术包、Push 源链与焦点操作数等在各自明示范围内无差异。单纯快照不作为复算通过，CPU 一致性不计模型语义票。
- UTC **21:27:55**，完成原字节离线准备和容量核对后，正式 SOL facts 固定批次启动，最多并行 2 个独立 CLI 上下文；批控原始状态见 `sol_facts_batch_r1/start.json` 和各槽 `return.json`。DeepSeek 固定 8 槽此时尚未启动，report 为 **0**。保留 40 单位双评和全部 80 份 facts 先于任一 report 的顺序；开发模块检查不算独立票。没有新增作者或训练调用。

UTC **21:18:10**，Push 两份观察完成 [root 非盲模块审查](./push_observer_inputs/root_module_review.json)：原始六句为 5 supported、1 insufficient。root 查看全部 24 张原始 1920×1080 PNG；未支持句的主姿态方向可见，但它对两个端点中躯干与骨盆遮挡的笼统说法证据不足，整句不进入投递，原答不改写。访问元数据的绝对路径及序号键与冻结核验器格式不符，首次机械失败原件保留；[薄格式适配](./audit_push_access_transport.py)只转换已核实的路径和序号后调用原核验器，未改模型材料、原答或语义规则。

UTC **21:19:41**，实际封存的两任务模块审查完成[离线装配](./offline_delivery/assembly_receipt.json)：Stair **7,042/8,192 B**，Push **11,188/12,288 B**，各显示一条获支持的视觉句，非视觉内容保持。状态为 `offline_serialized_not_sent`；N.4 八槽已用完，没有把离线序列化写成新的作者接收证明。Push 六条原观察仍全部进入正式视觉分母。72 个 SOL 原请求已逐字节移交并生成独立载体；完整 stdin 最大 233,285 DeepSeek 文字代理 tokens，加既定 73,728 预留为 307,013；这不是 SOL 实际 token 或图像容量证明。原请求、48 张 PNG、顺序与哈希保持，见 [离线准备](./facts_offline_preparation/README_ZH.md)。

UTC 20:24:09，Stair 两份观察的原始五句已完成 [root 非盲模块审查](./stair_observer_inputs/root_module_review.json)：5 supported、0 contradicted、0 insufficient。root 实际查看全部 24 张原始 1920×1080 PNG；确认范围是引用时刻的部位与轮廓，不延伸为接触、连续运动或失败原因。机械审计无违规。[可观察访问审计](./stair_observer_inputs/desktop_access_audit.json)另核两包各 12 次原图返回的字节、尺寸、顺序，以及一次原答写入；实际模型为 GPT-6.1-SOL/high，未观察到压缩。工具访问证明与语义模块票分开；五句获支持不能证明必要过程覆盖或整阶段通过。原答、审查及访问轨迹已逐字节镜像服务器；没有重试或改写回答。

当前生产容量另做了[整请求 CPU 核对](./production_capacity/production_capacity.md)：R4 作者文本输入 Stair **20,351**、Push **15,850** DeepSeek V4 tokens，均另有65,536输出上限；当前所有字节门通过。四份观察者形态请求各12张1920×1080原图，文字5,424–8,182 tokens；48张图片的实际SHA、顺序和尺寸均与观察者盲包吻合。没有新增生产API调用，也不把独立评委的大材料计入主作者。当前memory为空，满载16轮及服务端图像/封装用量不冒称已测。

以下为采集中的历史观测，按原时点保留：

UTC 19:01:23 实查同一 Stair driver 仍存活，尚无 exit；已写 1,824 张 PNG，最新文件距查询不足一秒。该数量只表示进度，不作覆盖率或通过结论。UTC 19:00:20 已将 12 个采集后 CPU 控制部署至本轮服务器控制目录；逐字复核仅为已登记身份替换及 assignment 绑定，规则不改。见 [部署回执](./postcapture_deployment.json)与[远端文件逐哈希回执](./postcapture_deployment.stdout.json)。仅完成部署，未执行派生、观察或评阅。

UTC 19:14:17 同一 driver 存活，已进入 seed 2；seed 1 的 env 2/63 分别保留 612/680 张，seed 2 两个 env 各写到 330 张。运行中 PNG 总数曾回落，已从冻结源码核实是预期裁尾：任务 `_finish_stair_seed_video` 将各 env 首回合 `episode_steps` 传给 recorder，通用 `_finish_snapshots` 仅删除 `step > T`，保留终点。向量 rollout 等待全组 Gym done 期间可能生成这些额外尾帧，故临时文件数并不单调。这不是重采或已证实的丢图；最终 `1..T`、SHA 和绑定仍由既定采集后检查验收。本次未读新轨迹数值、终局或像素，未改源码。

## 固定来源与执行边界

最终 [492 文件冻结回执](../n5_confirmation_source_r4/source_freeze_receipt.json)及两任务 CPU preflight 已通过。累计唯一相关检查为 46；打包缺少既有 fixture、随后错误提示文案不一致的两次失败原件均保留，后续只续检未完成部分。生产诊断逻辑没有因这些运行准备问题修改。

- 源清单 SHA：`1b7cb40e5b1d1780011e1748a9e6adae88517dd880b74c8b0b5f23df66a1a645`。
- 本轮卡 SHA：`811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041`。
- driver SHA：`a08d69b1f20adfc174d535afacb12bae05a54fccb2aab7bb2afcf290d86c68f2`，仅替换 R3 driver 中的卡哈希，单任务检查及运行逻辑不变。
- Stair cohort：`n5_r4_20261001`；Push eval base seed：`1800000000`。实际预检确认 seed 与 R1–R3 不重叠，历史检查范围见冻结回执，不声称穷尽服务器历史。

Stair 启动前再次核对 492 文件与 checkpoint 原字节，空闲 GPU 和原磁盘阈值通过。实际启动前磁盘空闲 `31,544,016,896 B`。本次只运行已有 checkpoint 的诊断评估，未训练新策略。

最初的远端部署文本因换行转义产生 Python 语法错误，解析阶段即拒绝，没有创建远端目录或启动进程；错误输出保留在 `stair_dispatch.stderr.txt`。修正部署文本并在本地编译后，远端再次确认目标目录不存在，才首次实际派发。没有重跑仿真或模型请求。

## 证据入口

- [机器卡](./capture_card.json)、[root 制卡审查](./root_prelaunch_review.json)。
- [实际派发](./stair.driver.dispatch.json)、[实际启动](./stair.started.json)、[首次进程/GPU核对](./stair_first_live_observation.json)。
- [旧图文运行的终态补核](../cli_clean_media_dev_r1/final_access_reaudit.json)与[修正 Stair 的开发离线投递](../stair_body_identity_dev_r1/offline_delivery/README_ZH.md)。

当前不能写 P3-NEW PASS，不能将 CPU 预检、进程启动或旧资料开发结果代替本轮独立验收。3-live、P4 均未启动。

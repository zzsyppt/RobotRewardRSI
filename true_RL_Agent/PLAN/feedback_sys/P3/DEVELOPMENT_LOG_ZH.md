# P3 开发与修正记录

日期：2026-09-24。验收结论以最终 `ACCEPTANCE_ZH.md` 为准；此文件记录实现中暴露的问题，不替代验收。

## 已发现并处理的设计问题

- **参考资格不能由目录名推断。** G1 的 `stationary_v3` 是参考奖励程序版本，已有资格实际属于 Push velocity v2。作者对照使用它的原 cohort；force v3 仅参与行为语义验证。Stair 使用已有真实资格的 `7b5826…`，不以历史 fca→Gen8 的描述比较替代新的作者准入。
- **压缩不能只满足字节上限。** Push r1 在部分真实数据上把代表性时间线全部挤掉，已改为紧凑画像并保护成功/失败代表。Stair r2 在新参考策略的 24 行画像上只留下一个 timeout 例子，需进一步保护成功反证；不在实验脚本中手选片段掩盖默认选择器的问题。
- **初轮与后续轮都必须接入正式入口。** Stair 的空历史合格参考策略原先缺少行为反馈入口；同时完整评估装配与公开白名单也需保留已封存的 public-dev 行为。修正位置在任务层，不构造虚假历史候选。
- **真实 evaluator 接线需要运行验证。** Push r2 新接线错误地从 `eval_summary_binding` 读取该 helper 不提供的程序 checksum，实际首错为 `KeyError('program_source_checksum')`，外层才表现为 phase schema mismatch。应使用已经由 CLI 验证的 `args.program_source_checksum`，并补执行真实函数体的 CPU 回归，不能放宽 runner 回执检查。
- **实验的公平性与来源闭合。** 模型信息必须从实际 `provider_response_receipt` 读取；冻结绑定生产源码和实验脚本；A/T 只允许任务 behavior 子树及其派生容量计量不同。A 从同一原始 phase 去掉新字段后走原投影，避免把已裁剪的 T 当成 A。失败作者 episode 也保留，不从分母删除。

## 已取得的中间证据

- r2 离线派生覆盖 11 组、5,472 个原始回合，既有成功/失败计数全部复现。Stair 独立核对逐回合事件、分组计数、根部立面锚点及保存的接触历史；Push 独立核对 7 个预选过程实例。
- P1 Push 真实历史重放保留 5/5 轮、10 个 lane 行；15 个旧 phase 摘要重建一致，14 KiB 记忆上限内，未回写历史回执。
- 新 Stair r2 真实 public-dev 重放通过：864 个回合，实际 rollout/summary aggregation 284.24428 s，GPU lease 总持续 675.79384 s。两种时间口径不同，不能用前者声称整个诊断免费。
- Push r2 失败运行仍计入资源：56.96118 GPU capacity s。失败原件保留，后续修复重跑另有身份与目录。

人工按用户要求待补；上述机器结果不构成人工事实支持率、作者收益或 RL 改进证据。

## 最终验收补查

- **可操作性按实际 ABI 修正。** 瞬时 roll/pitch、支撑力和速度可写入纯奖励，但无法直接实现诊断侧连续若干 control steps 的历史事件。现有 `forbidden_contact` 当前 mask 和瞬时禁触力确实暴露，不能因审计建议而错误宣称不可用；未暴露的是禁触力历史缓冲。保持当前权限，不为诊断增状态。
- **独立验证范围要写准。** 原 Stair 全档 oracle 只覆盖旧三类事件、终局和锚点。新 low_progress 与逐身体滑移/升降另增加固定真实 raw 数组抽核，不将旧项通过冒充全部新谓词独立验收。身体重排和镜像测试验证数值轴、名字及符号，不声称完成物理镜像仿真。
- **Push 终局与恢复时点分开展示。** 时间轴条带仅覆盖恢复观察窗，补注“最后条件失守”和恢复后的首次再失稳点。空白不代表稳定，恢复计时一度达标也不代表最终成功。图仍是测量派生图，不是视频。
- **实际工作词表收窄。** 实现提供 low_progress 窗口和有符号范围，没有独立“回退”事件；实施卡删除这个未经实现的名称。

- **A/T 来源 hash 不是可随意抹掉的差异。** 首包预检发现唯一额外差异在 Stair 的 `feedback_entries[0].source_public_result_hash`。A 的合法投影不含 behavior，T 含 behavior，因此 canonical 公开结果 hash 本来就不同。实验卡登记并重算这一个精确派生差异，其它来源/分数/权限仍严格相等；不伪造同 hash，也不忽略所有来源字段。

## 作者实验运行检查点（2026-09-24T07:55:13.593490+00:00）

最终生产源码 r5，closure `79afb56d72409d0829f88ba0142361661c4307fcebd4b91522126636c1d90db1`，384 文件核验一致。105 项 Stair 检查通过；完整新重放、导出与 sealed profile 相等验证通过；累计 GPU capacity 0.621113 h。

真实 48-episode 作者实验已启动，远端 Python PID `1903212`，wrapper `1903210`，运行根 `/data/yuyangliu/AGENT/feedback_p3_20260924/author_run_r5`，日志同级 `author_run_r5.log`、终态 `author_run_r5.exit`、时间 `author_run_r5.time`。必须以实际进程与终态文件确认状态，不能因 SSH 观察超时或临时 journal 将在途条目标为 interrupted 而重发。

首个确认终态是 Stair-06 / T / repeat 1，candidate_verified；两次请求均有真实 usage，第一输出达到 32768 token cap，第二通过。实际响应模型 `deepseek-flash`，指纹 `aeb56401ca74e127821c4f9126dcb669`，两次峰价用量合计 USD 0.068648172，无协议违规。静态/CPU 通过不等于语义正确或 RL 改进。后续结果必须按冻结的 48 个终态完整收集，不能缩样、清账本或加免费重试。

尚待真实 48 终态与全部匿名材料（含约 10% 复阅项）、用量汇总、最终交付核对。人工评阅明确待补，价值门始终 pending；不以本检查点代替最终验收。

## HTTP 402 中断与源头恢复修订

原作者进程在运行 31:59.81 后 exit 1，9 个终态、16 个已占 attempt；第 5 个配对槽 lane-a 的第 2 次请求是 completed HTTP 402，没有 usage。其余 15 次实际 usage 峰价折算 USD 0.500479548，加完整未知预留 USD 0.0513216，总 USD 0.551801148。原传输回执只保留状态码，没有余额不足响应正文；[DeepSeek 官方错误码](https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/) 将 402 定义为余额不足。用户已明确充值完成。

发现的源头问题：authoring loop 已支持将服务拒绝保存后停止，既有费用汇总也会对缺 usage 完整预留，但正式预算守卫只支持未完成请求的特定恢复入口，对 completed 的 402 仍一律拒绝。这会导致充值后继续被本地守卫阻断。

r6 将服务恢复结构判断统一在 `agent_core/provider_errors.py`，预算守卫复用它，仅对明确的 completed HTTP 401/402/403 服务拒绝完整预留原输入/输出额度。保留原件、缺失 usage 与尝试次数；普通未知失败保持原阻断。没有写入假零、清除账本、增加免费 retry 或引入任务语义。

修改既有 billing/auth 测试，执行真实 journal、正式 guard、停止与恢复的组合；同时核验原件不变、保守费用、恢复只调用一次和次数耗尽不能扩容。邻接作者流程、provider transport 与既有预算/中断恢复共 111 tests 通过（36.370 s）。限定独立复核无必须修问题。生产仅改两份公共模块，行为与 5 份冻结实验脚本保持不变；r6 另建源码和冻结修订，并要求 24 首包与 r5 相等后才续跑。原采集仍以实际 Stair r5 / Push r3 来源登记，不再消耗 GPU 重做相同测量。

根代理已核 384 生产文件、两处精确差异、5 份未变实验脚本及原 stage/case/schedule；本地 canonical 首包 hash 和远端 24 个实际文件字节均与 r5 相同。r6 source closure 为 `0ff923abb96f1fa2858bb453d745bab0e7b7a133045b799fe2131e1306cedd55`，preflight SHA 为 `b030e33eece324c1f5ab50135eb18f379a7eafb9cacbd3cef5a929b8b39f261d`，见 `results/author_recovery_root_review_r6.json`。原 147 文件失败现场已逐 hash 保存。

恢复段在确认无旧进程和原 9 个结果完整后单次启动，Python PID `1922596`（wrapper `1922594`，time `1922595`）。仍使用 `author_run_r5` 数据目录，新运行段单独使用 `author_run_r6.log/.time/.exit`。实际模型响应与最终 48 终态仍须完成核对，不能用启动成功代替完成。

首个真实恢复结果已确认：`stair-01/A/rep1` 原 sequence 3 返回已知 usage、finish=length；sequence 4 返回已知 usage、finish=stop 并通过静态/CPU 检查，两次仍是 `deepseek-flash`。第 4 次通过，没有重发旧 1/2，也未放大原 5 次上限。该检查点累计 10/48 终态、18 attempts，实际峰价折算或未知预留合计 USD 0.605810064；HTTP 402 的未知用量及 USD 0.0513216 预留仍保留。PID `1922596` 当时存活，继续其余预登记实验。


## 供应商输出偏差与 r7 预算语义修订

r6 在 45/48 作者终态时 exit 1。`push-04/T/rep1` 第一次回执为 input 16,262 / output 32,769，原请求 max_tokens 32,768，finish=length、最终正文 0 B；第二个 prompt 已保存，但 guard 在 delegate 前拒绝。87 个账本槽包含 85 个已知用量、旧 402 与本次未发出的保守预留，合计峰价实际或预留 USD 2.640883296。848 个现场文件已下载并逐 hash 核验，不删除偏差或改写未知 usage。

源头修正区分“发给供应商的请求限制”和“供应商返回的真实消耗”。通用 guard 删除单次输出及其派生总数的永久拒绝，继续完整累计输入/输出/total，并保留输入上限、原全局单位预算和 attempt 上限。实验端把输出偏差独立登记，实际费用不截断，不能标作原合同完全合规；不设置 +1 特例、不增豁免注册表、不增加预算。既有保守中断恢复用于消耗原第二次名额与全额预留，该 episode 只剩原 5 次中的 3 次。

5 个定向测试（2.762 s）通过：正常余量、超过请求限制但在全局余量内、输入超界、无全局余量时输出仅多 1 仍阻断、明显超额阻断；并覆盖旧服务恢复与普通未知调用的边界。日志见 `verification_provider_accounting_r7.log`。下一步只核验 r7 来源与 24 首包等价后恢复，不重跑已完成作者结果或仿真。

根代理核验 r7 的 384 生产文件、两份完整 harness diff、原 stage/cases/schedule、24 首包内容与 hash 均通过。唯一首包元数据差异是预检保存目录由 r6 变为 r7；原 45 个终态 hash 保持。见 `results/author_recovery_root_review_r7.json`。r7 closure `5686469111532083501de144d7a6a5ac4965a8175d6bccacc0b0de47eb27a2aa`；review freeze `aad0bb34fd97d1d17890e65b63f6419e09563a1a2dcb98b7c69e2f8864bd4306`。确认旧进程不存在后仅启动一次续跑，wrapper PID `1980687`；日志、时间与退出文件使用 `author_run_r7` 前缀，原 `author_run_r5` 数据目录与所有历史回执保留。


## r7 第二次服务拒绝（47/48）

r7 补齐 `push-04/T/rep1` 和 `push-04/T/rep2` 后，在最后 `push-04/A/rep2`（generation-024-lane-a）的 sequence 1 收到 HTTP 402，原进程 exit 1。该 episode 已占 1/5，还剩 4 次。累计 91 个账本槽，88 个真实 usage，2 个 402 与旧 cap 拦截各按完整未知预留，USD 2.751448656。原输出 32,769 偏差继续保留；48项汇总及53项匿名包尚不具备完整性。

已向用户说明需要恢复账户额度，未重启或重复 API。新停止现场 936 文件封存，ZIP SHA `81c8636215eb13ef7255fbd1ee97e486fc07a711f2df96fc5ef1dc01306e5bf9`；下载核验与剩余已取得材料的机械检查独立进行。只有用户明确恢复后才原账本继续，不返还名额、不换模型、不缩减分母。


## 停止期间的材料完整性核对

远端再次确认 r7 exit 1、原 wrapper 不存在；未把用户未回复当成额度已恢复。主计划和 5 份当前 P3 说明的 30 个本地链接均可解析。原工程索引 49 个证据文件均在位；变动只涉及已登记的主计划、两份预算汇总 harness 和两份当前状态说明。

47项部分 QA 的缺失逐项查明：三个末次 repair 包没有父源码，但实际首包与冻结首包一致且匹配结果的 parent checksum；新增独立 sidecar 补3项符号与2项Stair已保存样本算术，未修改冻结导出器或原QA。Push23项没有持久化逐样本raw，保留N/A，不重执行候选。根代理核对补充记录引用的结果、候选、实际末次输入及首包原件hash。更新验收只说明可证明范围，最后一项仍等服务恢复。


## 最终执行、材料与根复核

用户再次明确“已恢复”后，只读确认原47终态/91槽、r7源闭包/5份harness/stage/case/freeze不变且无旧进程；仅启动 `author_run_r7_recovery2`（wrapper2004628），使用原数据目录和新的日志段。最后 `push-04/A/rep2` 的原 sequence2通过，exit0；无需新增源版本、重复首包预检或新GPU任务。

最终48/48终态均为 candidate_verified；92个槽含89个权威usage、两次402未知预留和旧发送前拦截预留。费用原值重算 USD2.618139576 + 0.1539648 = 2.772104376。原32769/32768输出偏差保持，原47结果hash未变。无候选训练，不将静态准入当语义或RL收益。

冻结summary/blind_export导出48项+固定5项复阅。独立离线后处理只添加4张测量图、匿名父本补充与材料链接，不改冻结输入/候选/评分；处理源码与前后hash记录在审计区。完整1080文件下载并逐hash/size核验；仅评阅ZIP不含臂名映射。根代理独立核48唯一三元组、92槽原始回执和费用、48份原预算、5项复阅、53实际输入/候选等价、空评分表、无显式身份标记及4图完整性，见 `results/final_root_delivery_review.json`。

最终48项可核声明的父子奖励符号，24项Stair已有样本算术PASS；24项Push逐样本raw未持久化，保留N/A。人工事实支持/建议质量未评分，价值门保持pending，Phase4及后续未启动。本轮工程、实验与材料交付完成，人工评阅是用户明确延后的剩余阶段工作。

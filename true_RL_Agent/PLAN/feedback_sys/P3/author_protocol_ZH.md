# P3 作者对照：执行口径

原计划 **48/48 个作者 episode 已完成，53 项匿名材料已交付**。24 个真实首包预检通过；运行记录始终保留在 `author_run_r5`，四个日志段采用 r5、r6、r7 冻结，最后一段沿用原 r7。所有中断、未知用量预留和输出计量偏差保留。48 项均通过静态准入及 CPU 检查，这不代表语义正确或训练改进。按用户要求，人工评分为空，价值门保持待评。

首段完成 9 个终态后因 HTTP 402 停止。用户确认充值后，修正了确定服务拒绝但无 usage 的预算恢复缺口；原次数和全额预留保留，实验不重置。24 首包逐字节相同、两文件预算修订及新冻结经根代理核验后，已从原账本继续。当前恢复修订见 [r6 修订记录](AUTHOR_RECOVERY_AMENDMENT_R6_ZH.md)，第二段日志为 `author_run_r6.log/.time/.exit`。

恢复段在 `push-04 / T / repetition 1` 的第 1 次响应遇到服务商计量输出 32,769 units，超过请求及冻结上限 32,768；finish_reason 为 length，只有 reasoning，没有最终候选文本。下一次 prompt 已保存，但进入 provider delegate 前即被预算守卫阻断，第二段退出码为 1。没有修改上限、重发或把这次预留写成零。第二段停止时缺 3 个终态：该中止项、`push-04 / T / repetition 2` 和 `push-04 / A / repetition 2`。完整现场及原件 hash 见 [停止报告](results/author_cap_stop_r6.json) 与 [848 文件封存清单](results/author_r6_cap_stopped_snapshot.json)。当时未生成要求完整 48 项的 summary 或 53 项盲包。

[r7 计量修订](AUTHOR_OUTPUT_ACCOUNTING_AMENDMENT_R7_ZH.md) 将请求输出控制和服务商实际计量分开：请求参数不变，完整实际 usage 仍进入全局预算，输出超请求量明确登记为协议偏差；输入、总预算和次数硬限保留。新冻结及与 r5/r6 的 24 个首包字节一致性已由根代理复核，并按 `author_freeze_api_r7.json` 释放，从原 45 个终态、87 个账本槽继续，第三段日志为 `author_run_r7.log/.time/.exit`。旧中断槽不退款或清除。此为启动后的修订，不能宣称原合同完全合规。

第三段完成两个 Push T 终态后，在最后的 `push-04 / A / repetition 2`、`generation-024-lane-a` attempt 1 再遇 HTTP 402，退出码 1。当时共 91 槽：88 次已知 usage、2 次 HTTP 402 未知 usage、1 次原预算阻断后的中断预留；总 peak/reserved USD 2.751448656，硬 violations 为空，原 output 32769 协议偏差继续保留。最后一项已用 1/5 槽，只剩 4 次，服务恢复并再次放行前保持停止。见 [r7 服务停止报告](results/author_service_stop_r7.json) 和 [936 文件封存清单](results/author_r7_service_stopped_snapshot.json)。当时仍未生成完整 summary 或盲包。

[47 项临时机械核查](results/author_partial_qa_r7.json) 明确标为 PARTIAL：实际末次输入的原件引用均通过校验，源码/输入/检查记录的显式 arm/lane 标记扫描为 0 条警告；数值与标识只做引用位置定位。有限 CPU fixture 的奖励算术有 22 项可核且通过、25 项无此证据，不把后者当成功。全部语义评分与价值门仍待人工，临时核查不替代完整 48/53 项交付。

用户确认服务恢复后，第四段 `author_run_r7_recovery2` 沿用同一 r7 源码、5 份冻结脚本和 API freeze，未重新预检、未重跑已完成项。最后的 Push A 在原 attempt 2 完成，第四段耗时 1:35.38、退出 0。最终共 **92 个账本槽**：91 份已完成 provider 记录，其中 89 次有权威 usage、2 次 HTTP 402 无 usage；另 1 个旧中断槽按全额预留保留。已知用量峰价 USD 2.618139576，三份未知预留 USD 0.1539648，合计 **USD 2.772104376**。输出 32769 偏差不消失，协议状态仍为 `HAS_RECORDED_DEVIATIONS`；最终硬 violations 为空。

最终交付入口：[作者描述性汇总](results/author_summary_final.json)、[机器交付核查](results/author_final_delivery_checks.json)、[完整 1080 文件清单](results/author_final_bundle.json)、[本地逐 hash 验证](results/author_final_bundle_local_verification.json)。全部原始结果、journal、实际输入、回执和私有映射位于 `results/author_final_bundle`。交给盲评者使用 [独立匿名 ZIP](results/author_final_bundle/author_blind_review_reviewer.zip) 或其中的 [中文说明](results/author_final_bundle/author_blind_review/reviewer/README_ZH.md)，不要把同级的私有映射和含臂名的原始材料一起交给评阅者。

53 项由全部 48 个终态加 5 个确定重复项组成，评分字段保持空白，显式身份标记扫描为 0 条警告。冻结导出后只附加 4 张数值时间轴及来源 sidecar、独立父程序核查附件。三个末次修复输入缺少 current_best 的案例，使用已绑定的首包父本补充审阅比较，原末次输入不变；含臂名/私有路径的原 sidecar 保留在 withhold 区，匿名附件按映射对应为 4 项（含一个重复）。后处理来源及文件 hash 另行登记，不改冻结脚本。

最终 48 项均可投影声明的奖励权重/clip 符号；24 个 Stair 的已存有限样本算术通过，24 个 Push 未持久化逐样本带符号 raw/weighted 数值，继续标 N/A。只有绝对值极值的 Push 回执不能反推符号。引用位置和静态符号不是事实支持率或干预有效性的自动评分。

采用 12 个预先声明的焦点问题：6 个 Go2 Stair v3 stratum，6 个 G1 Push velocity v2 问题（4 个 stratum、全套、逻辑 run 01）。它们只有 **2 份底层观察包、2 个策略簇**。每问题 A/T 各 2 次新生成，共 48 个作者 episode；重复焦点与同策略相关性单独报告，不把 12 当独立样本。问题选择见 `case_design.json`，不包含正确诊断。

Stair 的作者父本是已有真实资格的 `7b5826…11c4c`，源码 `10b1a3…272cb`。原资格脚本未晋升 Genesis；读取原 assembly 后复用 runner 公开投影和任务资格 reader，原 gate 必须完全复现。空历史 builder 的 namespace 是资格束计算出的新诊断上下文指纹，不是历史 SessionGenesisArtifact，也不授予正式搜索准入。fca→Gen8 的合法历史差分另作机器验收，此作者实验 Diff 明示没有已测子策略。

Push 的 `p6_g1_stationary_v3_qualification_20260912` 中 v3 指参考程序版本，真实资格 cell 为 `recovery_v2`，不是 force v3。本实验因此使用该真实 velocity 资格与同 checkpoint `91c00…1749` 的新 public-dev 采集。cohort 使用原声明 `eval_base_seed=1000000000` 并逐项匹配 evaluation binding；其它 cohort 的新采集只作为工程验证。Force v3 仍在独立的数值语义验收内，不扩资格评测凑作者案例。

`author_experiment.py` 调用真实 PublicContextBuilder、任务权限/源目录、CandidateAuthoringLoop、静态准入、CPU component smoke 与持久 author journal。实验 adapter 只替换冻结的 `public_feedback_summary`，两臂追加相同中性焦点。A 必须从同 phase 去掉新 behavior 字段后经过当前旧反馈路径生成，保留原 aggregate；T 使用生产默认有界 selector。默认工作集没有某焦点的代表时保留 omitted，不人工补故事。冻结时比较实际 prompt：Stair 仅允许 feedback.behavior、Push 仅允许 feedback.public_dev.behavior 有差异；frame 仅允许派生预算计量变化，硬限/源码/权限保持相同。

调用上限沿原生产 CostGuardedProvider：24 个配对槽 × 两臂 × 每臂最多 5 attempts，最大 240 请求；这些目录名中的 generation 只用于复用预算守卫，**没有执行 RL generation**。源码请求及修复占同一 attempt 预算，2 次 source request、每次最多 2 IDs、累计 48 KiB。32768 输出、40000 输入预留，两臂一致。没有额外 retry 池；原 journal 中未知调用仍占位并阻止盲重发。

按 2026-09-24 已核 DeepSeek 峰价，40k 输入全 miss 加 32768 输出每请求 USD 0.0513216，240 请求计划 USD 12.317184，阶段上限 USD 15。这是预留单位内的预算，不是精确输入 tokenizer 证明；不沿用旧 P0 拟合作硬限制。实际 usage、cache hit/miss、响应模型和 fingerprint 从原 provider_response_receipt 汇总。原 r5/r6 将任何单次超预留视为停止条件；r7 修订后已知输入超预留、全局用量/成本或模型漂移仍停止，输出超请求量登记偏差并完整计费。首次 prompt 冻结，修复/源码跟进由冻结的生产代码与真实回执派生。启动/恢复复核源 snapshot 与实验脚本 hash，不仅核首次 prompt。

CPU 检查涵盖实际已资格 Stair 参考→真实 builder→假 transport→真实静态/CPU smoke→author journal、预算读取，另核错误 source hash、A/T 越界分数或源码权限变更拒绝。这仅证明接线；假 transport 不是模型效用证据。真实模型输出产生候选但不训练，静态通过也不是 policy 改进。

`case_freeze.py` 要求 12 完整引用、24 首包 preflight、原件 hash、源码指纹与固定配对顺序。根代理审读后才以 `--release-api` 冻结启动许可。`blind_export.py` 隐藏臂名、打乱输出，并插入约 10% 同人复阅项；映射单独保管。先交空评分表和同一 public-dev 真值材料，不自动填正确性。一个评阅者只能报告自身一致性，不能声称评阅者间一致性。

机器可确定原件、域、预算、源码准入与合同，并定位明确的 episode / 字段 / 来源标识引用；数值含义、事实支持、因果克制、动作相关性和可检验性仍逐项盲评。没有后续训练时不定义“预测改进命中率”，只评价可证伪预测是否清楚。人工未完成时不宣称 P3 价值 PASS，更不宣称 RL 收益。

`case_prepare.py` 将独立导出的完整行为真值与已封存 phase 做生产投影逐值比较，并记录新的诊断可用时间；完整真值只提供给盲评，不进入任何作者臂。`build_case` 每次重读已绑定的 phase、原资格和真实 task adapter，验证冻结反馈、parent、plan、archive 与可用时间。两臂共有原资格身份和分数；新的 public-dev 观察不冒充原资格时点。

本次预先选择质量收益门，成本只描述，不在输出后切换成本优势路线。匿名导出保留全部 48 个作者终态（含无候选失败），附中文阅读说明、共同真实父源码和完整 public-dev 真值；如源码直白包含 lane/arm 标记，仅记录待审，不修改建议掩盖泄露。数值引用定位/格式通过不等于机器判定事实支持率达到 95%。

`machine_reference_locations.json` 单独列出实际末次作者 feedback 与完整真值中的引用位置；此前失败候选自己的文字不当成测量证据。它只匹配明确标识，不理解自然语言改写、不判数字正确性，也不填写人工得分。匿名身份扫描覆盖建议源码、末次输入及评分条目，未解决的明确臂名泄露需在交付前说明。

Stair 首包另有一个必要派生差异：`public_feedback_summary.feedback_entries[0].source_public_result_hash`。A 删除 behavior 后的公开对象与 T 的对象内容不同，必须各自计算正确 hash。预检比较 hash 前的对象，确认只差 behavior，再核验该精确路径的两值；值登记在案例卡和预检报告，不泛化忽略任何其它来源 hash。篡改这一字段仍拒绝。

机器附件还列出 rationale 中数值字面量的精确匹配位置，不推断百分比、舍入、单位或分母。奖励检查直接复用生产 canonical 活跃组件分析器，给出修改前后权重/clip 符号，并核对本轮已有 CPU fixture 的 raw→clip→weight 算术；只证明有限 fixture 内的数值方向，不能自动判定修改有益或与解释吻合。

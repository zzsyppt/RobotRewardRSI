# A07 验收：历史准入漏接，NO-GO，中止于第 3 代

会话 `go2-push-5rounds-20260923-a07`，冻结执行源码 `5f13b475335b871a8aa458d667ece16763dd78f6`。本结论依据已下载的 live11 停止快照、原始会话记录及逐代审计；**只有 2 个闭合代，第 3 代不完整，未完成五代验证，不存在整轮终态 PASS**。修复后须重新开始完整五代，不能拼接 A07 前两代或其他批次代数。

## 阻断事实与归属

第 3 代 B `push_recovery_tilt_rate_brake_v7` 与已实测的第 2 代 B `push_recovery_tilt_rate_brake_v6` 共用同一 accepted parent。七个函数的 AST、RewardTerm 顺序、绑定、clip 和其他组件一致，唯一有效修改是 `tilt_rate_brake` 权重 **1.0 → 0.5**。这违反实际 prompt 中“不得提交已评测源码的纯数值、字符串或格式变体，历史源码门会在训练前拒绝”的冻结要求。

现有比较器把这对源码判为 `parameter_only_delta`，两者的 verifier family key 同为 `53747b3c96045bcdf264f26e8690b491fe024485fe20324d00beec3e70bdd871`。但运行中的历史准入只比较 exact implementation key，权重变化使 exact key 不同；旧的逐历史源码路径实际记录 `evaluated_history_source_count=0`。候选遂被误判 `candidate_verified`，两条训练均已启动。根因是**共享准入层未覆盖全部已评测历史家族**，不是 Push 物理判据问题，也无需新建任务专属规则或另一套数学比较器。

证据见 [第 3 代源码审计](gen03_candidate_source_audit.json) 与 [第 3 代实际 prompt/修复审计](prompt_audit_gen03.json)。关键源码 SHA256：

| 源码 | SHA256 |
|---|---|
| 第 2 代 B v6，已训练与 dev 评测 | `127a3d5bbf1ce80ce69533186dd42c9e01dc3a2198a937a264572a92e98dabdc` |
| 第 3 代 B v7，误准入 | `5353c685f0241b15de7dd07ab6876abda0c21ce42e595f65a59f88d35d388047` |

这已破坏本轮“准入后再训练”的实验流程，足以阻断；没有证据表明第 3 代候选完成了训练、评测、科学接受或污染 accepted head。保留原 prompt、候选、日志、reservation 和前两代结果，不改旧回执制造闭合。

## 已完成的实测及有效范围

reference 复用 A03 训练 checkpoint，本批重新执行三阶段资格评测，未重训 reference；见 [reference_audit.json](reference_audit.json)。

| 对象/代 | A dev | B dev | 被选候选 accept | sealed | 原始流程结果 |
|---|---:|---:|---:|---:|---|
| reference | 141/192 | — | 142/192 | 343/448 | 本批资格评测完成 |
| 第 1 代 | 164/192 | 0/192 | A：156/192 | A：371/448 | 接受 A，闭合 |
| 第 2 代 | 136/192 | 128/192 | A：132/192 | 未运行 | 提升门未过，提前拒绝并闭合 |
| 第 3 代 | 未评测 | 未评测 | 未运行 | 未运行 | 两条训练途中中止，未闭合 |

[第 1 代决策审计](gen01_decision_audit.json) 和 [第 2 代决策审计](gen02_decision_audit.json) 已分别重算选择、分数、接受/提前拒绝、身份与当代成本闭合。第 1 代 accept 提升为 `0.0729167`，超过当次冻结噪声门 `0.0551198`；第 2 代提升门失败后未浪费 sealed 评测，accepted head 仍为第 1 代 A。这些是局部有效结果，不能替代整轮验收，也不能据一次固定 cohort 测量宣称稳定泛化收益。

[训练证据](training_evidence_through_gen02.json) 覆盖前两代 4 条完整训练：各有连续 `0..499` learner 日志、49,152,000 env-steps、12,000 control steps，初始策略/训练条件/来源身份一致，最新输入 provenance 与修复后基线一致。峰值同刻分项按 float32 顺序求和吻合；第 1 代 A 的极大负峰主要来自 `action_rate`，不是第 1 代 B 的早败证据，也不单独构成“训练崩溃”的因果结论。

[物理导出算术审计](physical_evidence_through_gen02.json) 覆盖 10 份 summary，核对无位移失败门、重复推为 high 的 80%/间隔 0.22 s、外力时序与冲量、提前跌倒后的不完整脉冲及成功/失败计数。PASS 仅指保存的导出记录一致；不是从原始轨迹独立重建 PhysX、连续稳定窗口或逐帧视觉的证明。

第 1 代 B 的 192 个 dev episode 全在第 11 control step（0.22 s）关闭，首次计划外力是第 51 step（1.02 s）。最低高度 0.1637–0.1665 m 低于 0.18 m，姿态未先越过倾倒阈值，按冻结优先级判 `height_collapse` 合理；零实际注入是推前早败的结果，全部 episode 仍计入分母。见 [早败审计](gen01_lane_b_early_failure_audit.json)。

原生诊断已在正式会话停止后独立完成，见 [native_replay_audit.json](diagnostics/gen01_lane_b_native/native_replay_audit.json)。它保留原 B checkpoint、64 个环境和三个 dev run；checkpoint 的服务器前后 SHA 相同，192 个 episode 与原评测逐字段一致，整份 task evidence 只有三个生成时间不同。三段 env0 视频均绑定各自 summary 记录；服务器实际逐帧解码到 EOF，各为 11 帧、1280×720、50 fps，本地文件 SHA 与录像及解码回执相符，不以清单中的帧数代替独立解码。

主执行方实际查看了第一段视频 env0 的第 0、5、10 帧：机体由较高站姿持续下降，终帧腿部明显折叠、机腹接近地面，未见侧翻，与 `height_collapse` 一致。本地另将三张 PNG 解压后的 RGB SHA 与对应解码帧绑定。这只是 **run 1/env0 的代表样本视觉核对**，不是 192 个 episode 的全量视觉验证；具体米高来自遥测，不从图像估尺，也不据此推断训练优化原因。诊断耗时 30.475687 s、2,112 env-control steps，单独留存，不进入正式会话 ledger、接受或成本闭合，亦不改变 A07 的 NO-GO。

## 提示、记忆与边界检查

[Gen1](prompt_audit_gen01.json)、[Gen2](prompt_audit_gen02.json)、[Gen3](prompt_audit_gen03.json) 的实存 prompt 均有来源审计。第 3 代保留两代、四个 trial 的 4,017 B 记忆；当前父策略 dev 164/192 与最近未接受候选 dev 136/192 分开呈现。历史重复修复时，实际 prompt 保留已公开的当前父源码、最近未接受源码和本次失败源码，无须额外 source-request。已扫描的私有路径与 checkpoint hash 无泄漏命中；这里不把有限扫描扩张为所有可能泄漏的证明。

这些反馈和记忆改进真实生效，但不能覆盖历史准入误放行。第 3 代源码比较说明现有 task-neutral 比较器已能识别问题，缺口位于共享历史接线；不应把 Push/Go2 成功率、外力规则或本体条件引入共享门。候选 rationale 对未训练草稿与历史实测对象的归因错误仍按证据记录在上级 followup，不当作新的测量结果。

## 停止、未结算成本与未完成终态

[blocking_stop_07.json](blocking_stop_07.json) 记录只对核实属于本 A07 会话的进程树发送停止信号，记录的 `sigkill_remaining_pids` 为空。停止快照中 ledger 仅两行、generation closure 仅第 1/2 代；`final_confirmation` 仅有计划，没有完成回执。该停止记录不等于生产入口终态验收或第三代成本结算。

最后已结算成本是 [0004-generation-002-settlement.json](session/generation_actual_cost/0004-generation-002-settlement.json)，**累计到第 2 代，包含本批 reference 资格评测，不包含第 3 代 API 与部分训练**：

| 已结算项目 | 累计值 |
|---|---:|
| 完整候选训练 | 4 次 / 196,608,000 train steps |
| eval seed units | 38 |
| 已记账 job attempts | 14 |
| provider attempts / 总 token 单位 | 8 / 246,599 |
| 已记账 GPU 容量小时 | 0.9244331183 |
| accounted output bytes | 240,912,364 |

这不是本批被中止后的最终总成本，不能据此宣称成本全部闭合或推算账单金额。第 3 代两份原 reservation 保持不变，均没有对应 result：`job_attempts/a3559429464c1bbb244db527/attempt_00.reservation.json`（A）与 `job_attempts/b8140b83477d4de816eb6bf9/attempt_00.reservation.json`（B）。live11 的训练 stdout 最后分别记录 iteration `182/500`、`167/500`，无完成 summary；这些只证明部分训练进展，不替代完整物理用量结算。第 3 代额外 API 已发生，旧结算记录尚未包含它；不补写伪造的 completed/result/closure 来消除开放记录。

**A07 最终判定：NO-GO，因已确认的准入接线缺陷主动中止。** 前两代保留为有效局部证据；第 3 代保留为阻断复现。后续修复验收及新五代实验使用新身份，本文件不预先宣称其成功。

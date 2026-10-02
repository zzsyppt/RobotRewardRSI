# R0-b 全体过程预测类别审查方法（2026-09-29）

这是**事后类别审查，不是独立新实验**。只审冻结 AR2 中“并给出可检验的过程预测”这一必要部分；不赋总 AR、机制对应或门结论，不改原分数。

材料限定为冻结规则与指定 reviewer 包的 32 份提案、32 份父源码、32 份 seen、32 份 initial。没有读取 withheld mapping、旧评阅分数/aggregate、Audit/rescore 文档；没有调用模型 API、仿真或执行提案。128 文件均作结构读取，32 个 rationale 均作语义复核。初次/最终输入只有 010、030 不同。源码经 AST 只读解析，记录原文、行号、SHA256、机制 callable 与 RewardTerm；不声称完成全部事实/代码硬错误核对。

统一口径：过程量可以是速度、姿态、进度、持续/先后、条件违反步数、驻留中断或条件下的轨迹变化。成功率、综合分数、primary 终止率单独只是结果预测；已有过程观察、奖励公式性质和泛化先验也不能自动转为未来过程预测。不限于 Expectation 标题，不额外要求数值效应阈值、显著性检验或每项必须写时间顺序。

mean progress 允许作为过程量：003 给出四个具体 cell、父值与上升方向，直接连到 progress 奖励；022 在联合反驳句中提到 mean progress 不变，保留为较弱的过程检验及范围歧义。030/032 只在观察中给进度，不能补造进度预测。014 在 Hypothesis 段明确预期摆腿轨迹抬高与净空增加，应记录为定性过程预测。

运行项贡献、设计意图、latch 命名与 could 风险句的边界统一保留 ambiguous 类别，不为得到整齐二分而投票。009/013 为 runtime proxy；008 为 asks 的意图/行为歧义；026 未区分 primary termination 与 detected；030 为风险可能性。过程预测存在与预测基线正确、过程支持正确、代码兑现和全部 AR2 是不同问题。

特别可复核项：001 将57总中断数写成 per pushed episode 阈值；实际为 repeated/not_recovered 的14回合、28计划窗口（57/14=4.0714，57/28=2.0357）。025 的稳定占比 >0.52 低于真实父值1609/3080=0.522403，因此阈值本身不保证严格提高。019/029 的子组/聚合范围也需保留歧义。完整数值和JSON路径见同目录 process_prediction_all32.json。

下面仅列过程预测类别，不能相加推导任何变体有效、AR2数量或固定出口。

| 提案 | 类别 |
|---|---|
| proposal-001 | present_scope_error |
| proposal-002 | present |
| proposal-003 | present |
| proposal-004 | present |
| proposal-005 | present |
| proposal-006 | absent |
| proposal-007 | absent |
| proposal-008 | ambiguous_intent_only |
| proposal-009 | ambiguous_runtime_proxy |
| proposal-010 | present |
| proposal-011 | absent |
| proposal-012 | absent |
| proposal-013 | ambiguous_runtime_proxy |
| proposal-014 | present_qualitative |
| proposal-015 | absent |
| proposal-016 | absent |
| proposal-017 | absent |
| proposal-018 | absent |
| proposal-019 | present_scope_ambiguity |
| proposal-020 | present |
| proposal-021 | present |
| proposal-022 | present_weak_joint_falsifier |
| proposal-023 | absent |
| proposal-024 | absent |
| proposal-025 | present |
| proposal-026 | ambiguous_latch_semantics |
| proposal-027 | present |
| proposal-028 | absent |
| proposal-029 | present_scope_ambiguity |
| proposal-030 | ambiguous_risk_only |
| proposal-031 | absent |
| proposal-032 | absent |

类别中 present 表示找到可检验过程命题；附加后缀说明范围/量化限制。absent 表示只找到结果预期、代码作用或一般意图。ambiguous 表示原句本身存在指定语义边界，不能靠增加新规则消除。所有判断保留提案完整 rationale 与精确摘录，供复核。

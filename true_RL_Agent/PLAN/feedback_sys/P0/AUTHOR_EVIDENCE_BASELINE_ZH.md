# 作者 A 已有证据基线

2026-09-23，回应审计 §2.3 与建议 2、3、4、6。结论：**作者已经会引用聚合结果、分层和负历史；“10 条全部引用所有字段”不成立。现有证据的正确使用，也应列入 A/T 对照。**

原始 rationale、对应输入与回执、源码行号、JSON pointer 及逐条核对结果见 [author_evidence_baseline.json](./author_evidence_baseline.json)。[提取脚本](./author_evidence_baseline.py) 只用 AST 读取字符串，不导入或执行候选。

## 1. 十条 Push rationale 实际引用了什么

10 条候选对应 attempt_08 的 15 次真实 authoring 输入；5 次失败后续写另有输入，按实际生成候选的 attempt 配对。10/10 源码 checksum、同次 payload checksum 与回执一致。表中“分层”只统计写出 `n/N`，“负历史”包括前轮非正结果，不将 `qualified_not_selected` 一律改称正式拒绝。

| 候选 | parent 均分 | 失败计数 | 分层 n/N | 负历史机制 | 负候选分数 | 未获过程观测支持的主要解释 |
|---|---|---|---|---|---|---|
| G1-A nonsaturating_stance | — | — | — | — | — | 高度塌陷“正是”低姿态局部最优，拉升斜率不足 |
| G1-B gated_damping | — | — | — | — | — | 阻尼恰在需要纠正运动时最强，因此抑制恢复 |
| G2-A linear_tilt_ramp | — | 15侧倒、1前倒 | — | — | — | 小角度侧倾先失控，之后只剩跌倒惩罚 |
| G2-B tilt_aware_stance | 0.8802 | 15侧倒、1前倒 | — | — | — | 残余侧倒发生于同时下沉和roll的状态 |
| G3-A rate_damped | 0.8802 | 完整5类计数 | 四层 | 有 | — | base下降/旋转比腿更换支撑快 |
| G3-B vel_damped_stance | 0.8802 | 15侧倒、1前倒 | — | 有 | 0.5521 | 未受罚漂移导致侧倒，gate绝不阻挡恢复 |
| G4-A side_braced_stance | 0.8802 | 15侧倒、1前倒 | — | 有 | 0.5521、0.7656 | 前轮失败是因为干扰恢复瞬态或近目标信号 |
| G4-B lateral_brace | 0.8802 | 15侧倒、1前倒、3塌陷 | — | 有 | — | 侧倾与横向漂移积累；另有下述符号矛盾 |
| G5-A roll_braced_stance | 0.9115 | 10侧倒、7未恢复 | high/low/medium | 有 | 0.8385 | 强推下大roll积累是线性侧倾项过弱所致 |
| G5-B lateral_tilt_decay | 0.9115 | 175成功、10侧倒、7未恢复 | high/repeated | 有 | 0.8385 | 新项缩短仍在侧滚的base被再次推倒的时间窗 |

统计为 **7/10 均分、8/10 失败计数、3/10 分层 n/N、6/10 负历史，其中4/10写出负候选分数**；四类数值引用同时具备的只有 G5-A/B。第一代没有既往 sibling 结果，但两条均定性讨论了可见的失败类型和分层。G4-A 的 0.5521 可由当时 parent 0.880208 与 G2-B public-dev delta −0.328125 重算，其余列出的负候选分数可直接定位。

这些关键计数与分数通过了脚本核对。每条 rationale 的完整文字仍保留：奖励公式/梯度是源码可核的主张，真实运动发生了什么是过程主张，提出新项会改善什么是预测。上述三者分开记录，标签或成绩差异本身不裁定失败原因。实际输入中没有逐回合记录、五条件/dwell时序、足部轨迹或图像序列。

## 2. 一个现有证据就能发现的反例

accepted G4-B 的 [lateral_brace 源码](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session/provider_candidates/go2-push-5rounds-20260923-a08-generation-004/lane-b/002_push_recovery_lateral_brace_v1/training_code/components.py:28) 返回非正的 `-(roll_excess + 0.5*lateral_speed)`，weight 为 −1、clip 为 [−2, 0]。[运行时](D:/humanoid_agent/true_RL_Agent/src/true_rl_agent/tasks/push_recovery/runtime_adapter.py:326) 将裁剪后的值乘权重，因此该项加权 reward-rate 在 **[0, 2]**，与 rationale 所称“侧倾惩罚/恢复斜率”的方向相反；之后乘正的 dt 不改变符号。G5-B 的真实输入已带该项 `weight=-1, last_mean=+0.068479`。

这里登记的是作者解释与实现的矛盾，保持历史策略原样。该候选的分数提升不能据此归因于其文字描述的机制。T 的价值应同时考察：是否增加可引用的过程事实，是否减少无依据的过程解释，以及是否更准确地使用已有计数、源码和激活摘要。

## 3. formal 的36份材料能做什么

已只读取回 [36份源码](./author_evidence_raw/formal_c6/provider_candidates/)，共186,679字节；完整 literal rationale 均进入 JSON，并与本地53份实际输入配对。30份直接匹配同次回执的源码 checksum；6份静态失败回执没有该字段，使用同次候选身份及随后 repair 输入中的原文 checksum 交叉核对。

按 **相同 round、candidate source** 关联历史 runner：23份有完成训练和 public-dev；12份没有对应完成训练，另1份 G3-B 训练失败、无完成 public-dev。36条适合做裁决规则 fixture 库，其中23条能关联真实评估结局；静态/训练失败只检验相应失败处理。重复源码与修复稿保持来源，不能当独立策略数。

formal 同样混合了观测和解释。例如 G12-B 引用 joint_ood success 0.222、forbidden-contact 0.667，同时断言“脚未先升高，所以胫部撞立面”；后者没有足部/碰撞时序支持。它还把 `last_mean=6.9e-07` 解释为此前项几乎没有学习信号；单次末采样均值不能给出整个训练期的激活结论。JSON保留36条原文与假说尾句；本次完整数值引用核对集中于10条Push，formal不声称已逐句验证所有算式与数值。

## 4. 对 Phase 3/5 的最小修订

1. **A/T 都生成回答。** 历史 payload 可作为 A 输入和回归样例；历史回答可作为固定参考。新对照应对同一 case、同一可用时点、同一模型配置和预算分别生成 A/T 输出，T 只改变待测反馈。历史 A 回答与新 T 回答混配会混入模型版本、采样和修复次数差异，不能据此免除 A 的生成预算。
2. **先统一评估域与时点。** formal 历史输入属于 v2；后来得到的 v3 重放结果属于新观测。若做 v3 对照，用同批重放的 parent 证据构建 A 与 T，记录新截止时点，再留出候选结局作裁决；保留历史 v2 输入作为另一组回归 fixture。把未来 v3 结局只加到历史 A 的 T 版本，会造成信息不对称。
3. **历史假说只做事后规则测试。** 先标注主张类型与可裁决条件，再用对应结果测试 `supported / contradicted / insufficient_evidence` 等规则。聚合分数只能裁定预先明确的结果预测；腿部运动或接触机制需要对应过程证据。这里不计算预测命中率。
4. **一名人工 + 机械引用核对。** 机械部分检查来源/输入hash、case与时点、JSON pointer、数值/单位/分母、同协议及缺测标记；人工判断引用是否真正支持主张、行动建议是否合理、是否把相关性写成因果。随机化臂名与展示顺序，固定评分表，保存原始评分及争议条目。单人复评可检查自身一致性，不能宣称双评者一致性或独立盲评。

边界：本项建立历史作者基线和离线fixture目录；没有生成新A/T回答、运行候选、修改奖励、调用provider或启动GPU。机械匹配证明引用的存在与数值一致，不自动证明完整语义、时序先后或因果关系。

复现：`.venv/Scripts/python.exe PLAN/feedback_sys/P0/author_evidence_baseline.py`。脚本从已保存原件重建 JSON，缺原件或身份/关键引用不一致时直接失败。

人工选取的10条Push标注、数值引用来源和源码符号反例现存 [author_review_annotations.json](./author_review_annotations.json)。分析脚本只解释这些显式数据并核原件hash、JSON pointer与AST条件；无按特定代数或候选写死的分支。这仍是人工回顾审读，并非自动语义grader。

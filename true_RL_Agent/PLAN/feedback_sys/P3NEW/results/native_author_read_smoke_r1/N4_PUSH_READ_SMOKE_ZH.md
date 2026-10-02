# N4 Push 原生视觉输入后的作者读法检查

本检查沿用冻结 `review_contract.json` 的引用与事实分类方式，身份为**参与开发的非盲 Codex 子代理**。不是 N5 独立评分、A/T 比较或新的价值门；没有再调用作者、训练策略或改写候选/原评分。Push 的疑点由本次读实际产物形成，不预设 Stair 的结论也适用于它。

## 结论

这份作者成稿**确实使用了新增数值过程报告**：引用了稳定 11/110 步、约推后 100 步才出现最长稳定段、21 步连续驻留及 110 步恢复窗，并用这些现象解释为何尝试让速度更深入稳定范围。没有找到视觉句的引文或可唯一识别的释义；未采用视觉不算错误。

有一处可直接确立的反馈概括错误：把组内“出现过超限”的 episode 统计扩写成双方 **every observed recovery window** 都有平面/角速度超限。low 成功组实际只有 **32/33** 回合出现角速度超限，完整派生中能找到反例。

代码基本实现了它所说的“膝点内移至原阈值 0.75 倍”，其余四个项保留。另有一项与事实读法分开的作者契约问题：这个修改数学上只是原函数阈值的重参数化，等价于把 0.08/0.5 改成 0.06/0.375，而实际提示明确禁止 numeric-only 变体。原机器终态仍是 `candidate_verified / PASS / train not_run`；本检查不改写它，也不把机器通过等同于满足了数学意义上的结构创新。

## 1. 实际输入与原输出核对

`request.http_body.json` 的 `/messages/1/content` 与实际 `user_payload.json` 原文相同。`/public_feedback_summary/public_dev/behavior` 与 `public_report.json` 完全相同。原 HTTP response 与解码 JSON 一致，解码回答里的 `components_py` 与候选源文件一致；不是用准备稿代替真实传输输入。

| 文件 | SHA-256 |
|---|---|
| HTTP request | `2d43d94f31e8d541d4aacc854328bb3c4bdb7cada8eff3aff9f5ed55f17a97d7` |
| user payload | `a9f82056aaa0ac9497d9e73e64a45c606f8a973a689a9c1fe5bb608b7534fa5a` |
| HTTP response | `ee7c31aae6b56c5da7fd2829588112012fbf710716cde0d454e8b6a0b6c78932` |
| public report | `76e7dbd8a09742e45df3787dbe52762e4ab358a62f9c81ca0fa76557489423fe` |
| components.py | `18a170b7da366f6c469f72ac4d54837a7a204bcff3348e4b866ef013f2f863d0` |
| 原执行终态 | `c408f3b919991a5da21b24dba9e8e4c2d7220e36abaf3aff2242a1a71104cf21` |

原字节在 `push_originals/`；完整派生件位于服务器 `materials/push/behavior_full.json`，SHA-256 为 `92eadd3366b911dfca39177f89e589b9941f26b5c4ee40ca75bb455cd1a15de4`。源引用、实际函数 CPU 算例在 `push_source_and_math.json`；反例的派生记录在 `push_evidence_excerpt.json`。这是一个已返回候选的读法记录，未生成新的实验臂或正式分数。

## 2. 可定位的新增过程证据使用

以下均见 rationale 第 1 行及 `components.py` 第 64–70 行。实际输入 JSON 前缀为 `/public_feedback_summary/public_dev/behavior/report`。

| 作者表述 | 对应实际输入 | 判断 |
|---|---|---|
| “verified dwell of 21 consecutive stable steps … anchor+1 to anchor+110” | `/results/1`，ID `push.results.rule` | 支持。21 是连续驻留要求，不是累计稳定步数。 |
| “process example settles for only 11 of 110 steps” | `/process/0`，ID `push.process.example` | 支持，按整个 110 步窗内共 11 个 all-five stable 步理解；没有说 11 步连续驻留。 |
| “longest stable run about 100 steps after the push” | 同上：anchor=69，最长稳定段 170–179 | 支持。最长段起点在 anchor 后 101 步，“about 100”合理。 |
| 失败与兼容成功回合都出现速度超限，所以单纯“是否超限”不足以区分 | `/process/1` 至 `/process/4` | 基本观察有支持，但作者随后使用 every 的范围过强，见下一节。 |

这里不是靠出现 “recovery” 等宽泛单词判断引用，而是用 11/110、21/110 和可计算的 170−69 锁定了新增报告。没有显式句 ID，也没有逐字引用；属于**可唯一定位的释义**。

visual 中确有同回合稀疏姿态描述：r1.e19 的前倾、屈膝变化和采样限制。作者最终理由没有使用该视觉观察；也未把它当作精确角度/接触测量。缺少视觉引用不是本次错误项。

## 3. 已确立的反馈概括错误

原句（`components.py` 第 64–67 行）：

> group comparisons report planar-speed and angular-speed violations in every observed recovery window of both failing episodes and compatible successful episodes

冻结类别使用 `invented_unobserved_fact`，补充范围问题为 `subgroup_fraction_as_cohort_fraction`；只按**这一组过度概括**登记一项，不按三个措辞重复计数。

直接反证在实际 `/public_feedback_summary/public_dev/behavior/report/process/3`，ID `push.process.group.low`：angular speed **13/13 vs 32/33**，并非成功侧全有。对应完整证据引用为 `profile:matched:low:not_recovered`。

从完整派生的全部 low 成功回合独立重数：33 个成功回合中，32 个有角速度超限；余下的 **r3.e0** 在 `/transcripts/128`，event_0 anchor=69，`angular_speed` 超限区间为空，稳定 97/110 步，于 94 latch，最终成功。这个真实派生反例比语言上的争议更直接。

另两处范围必须保留：组比较对象是 `not_recovered`，不能未经检查扩展到全部 fall 等失败类别；其分子/分母单位是 episode，repeated 下不能由每个 episode 至少发生一次，推出每个 recovery window 都发生。正确的受限描述可以是：“所列匹配组中，平面速度超限在失败与成功回合均普遍出现；角速度亦普遍出现，low 成功为 32/33。因此仅以是否出现过超限不能充分区分结果。”

## 4. 驻留、代码与预测的边界

“what separates them is sustained settling … because success requires … dwell” 中，“成功要求驻留”是正确的必要条件。如果理解成“有没有完成窗口内驻留就是所有成功/失败的唯一分界”，则不成立：实际输入 `/comparison_basis/counts/all_recovered_then_final_unstable` 已给出 **34** 个全部恢复完成后最终仍不稳定的回合。完整派生 `/transcripts/3` 的 r1.e3 就于 107 latch，却最终因 planar_speed 不稳定而记为 not_recovered。

作者没有明确写“唯一充分条件”，所以本检查将这一句记为**需要限定**，不另加一个硬错；也不把它遗漏后续不稳定过程算成事实错误。合理改写应同时提到窗口驻留和末态稳定。

原 `stationary_v3` 与新 `recovery_margin` 的核心函数都是：

`h(x) = 0.5*x²`（0≤x≤1），否则 `h(x)=x−0.5`；运动项为 `1−0.25*(h(planar)+h(angular))`。

新代码对两速度再除以 0.75，确实把膝点向内移；原两个归一化阈值与任务展示阈值一致。四个未改函数 `upright_height`、`joint_pose`、`action_rate`、`fall` 的 AST 与 RewardTerm 均保持原样。没有加入 dwell 的历史状态或修改任务成功门槛。

CPU 直接执行原、新函数的部分结果：

| scaled planar / angular | 原运动项 | 新运动项 |
|---|---:|---:|
| 0 / 0 | 1 | 1 |
| 0.06 / 0.375 | 0.859375 | 0.75 |
| 0.08 / 0.5（原两个膝点） | 0.75 | 0.583333 |
| 0.16 / 1.0 | 0.25 | −0.083333 |

这些是函数值，不是训练或录像结果。“grazing the boundary cost almost nothing / almost full credit” 是不精确的程度词：两个速度都在原阈值时，原运动项确实从 1 降到 0.75；若其余正项达到最大，总正向值从 4 降到 3.75。缺少固定的“almost”判准，本轮给出确值，不硬判为零惩罚，也不认定它已证明速度边界是实际失败原因。

未来成功数上升、跌倒数不增是明确的可证伪预期，不能提前计为收益，也不因尚未训练而判错。“若跌倒增多，就是 over-stiffening”应维持机制假说身份；单次计数变差不能独立证明这个因果解释。

## 5. 单列的 numeric-only 契约问题

实际输入 `/loop_contract/evidence_discipline/4` 禁止 numeric-only、string-only、formatting-only 的已评估源码变体。新函数保持同一分段函数和算子组合的数学形状，且：

- `(scaled_planar / 0.08) / 0.75 = scaled_planar / 0.06`
- `(scaled_angular / 0.5) / 0.75 = scaled_angular / 0.375`

六组 CPU 计算也与“原函数仅替换这两个常量”的结果一致，浮点尾差在 2.3e−16 内。因此这是阈值内移的数值重参数化，不能仅因新增中间变量和函数改名就认作数学上的新结构机制。

这是**作者指令与机械准入之间的实际差距**，不是视觉/数值过程描述的错误，也不强塞进现有事实错误类别。保留原 candidate_verified；若主线要求禁止这类改法，需要另行澄清结构变化的准入口径。不能为了这一个候选新增一套过重的通用符号证明器。

## 6. 本次能够支持的后续方向

Push 已显示新增过程能够进入作者的机制理由，但仍需要限制“普遍/每个/所有”的扩写，并保留统计单位、匹配对象和 necessary/sufficient 的区别。过程引用能说明建议的来源，**不能单独保证建议正确、结构新颖或训练有效**。

本次不建议强迫作者引用视觉。当前机制针对速度收敛，稀疏屈膝描述没有提供足以进一步支撑它的速度/驻留证据，允许不使用这条观察是合理的。完整机器可读记录见 `N4_PUSH_READ_SMOKE.json`。

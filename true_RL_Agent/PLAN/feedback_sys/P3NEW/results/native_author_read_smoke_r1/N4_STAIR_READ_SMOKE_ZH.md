# N4 Stair 原生视觉输入后的作者读法检查

这是 **非盲、开发用途的单次读法检查**，不是 N5 独立评分、A/T 对比、价值门或训练效果验收。评阅者是参与开发的 Codex 子代理；已收到主代理的待查疑点，因此不宣称评阅独立盲化。没有再次调用作者、修改候选或修改原回执。本次只检查 Stair，未检查 Push。

## 结论

实际请求完整带入了新的过程和视觉报告，但这份作者成稿中，没有找到可唯一定位的新过程链、成功反例、残差/采样限制或视觉句的引用。作者主要使用旧的按 cell 成功数、分数、失败比例和参考代码。**没有采用某条反馈不是事实错误，也不能据此证明模型没有读过它。** 结果汇总中有与旧统计重复的内容，不能把这样的匹配认作使用了新增过程证据。

可以直接确立两处代码—说明不一致：完整程序的正向贡献上界 4.75 不成立；“完全腾空时该项严格为零”也超出了 `support_contact` 的历史接触语义。新增项自身的加权上界 0.25 是正确的，八个原有项的函数和系数也确实保留。

## 1. 以实际传输内容为准

本次已存的唯一 transport 请求有两条消息。`request.http_body.json` 的 `/messages/1/content` 与 `user_payload.json` 原文逐字相同；后者 `/public_feedback_summary/behavior/profile` 与 `public_report.json` 完全相同。实际输入中有 19 条报告句，其中 `process` 7 条、成功反例 1 条、gaps 7 条、parent_diff 1 条、results 1 条、visual 2 条。

原 HTTP response 与 `raw_response.json` 解析后相同；响应中 `/choices/0/message/content` 解码后的 `/components_py` 与交付的候选代码逐字相同。request/response 两个 HTTP 文件的哈希与传输回执相同。这里检查的是最终返回内容与候选文件，不使用作者的内部推理文本给候选补充理由。

| 文件 | SHA-256 |
|---|---|
| 实际 HTTP request | `774af035cc1ade5dabd166d15668917f837b26be3a4d824612505ef9f1d518a7` |
| 实际 user payload | `790f7ecf7dc6729b56ab0547d082eb481348a6bc32d938b3679f66cc9cff1156` |
| 原始 HTTP response | `8ab486c27cc384a41cd54c06137ac01a1e9179fe3404edf9894dc0b32fab459c` |
| 可见过程报告 | `b19a016b84bd5c6c2bb63587b462e2958f0fb1f0ebd90a9e000fcfa29436041a` |
| rationale.md | `5c2ee242de8bff0ffc48c9009066adee31e909b501a07848fd76e197715a88b1` |
| 候选 components.py | `393f6b1fb27f2a1459bf30857d86e52df03ba896c2356645f4d4032635924d4b` |

这些文件的原字节副本在 `originals/`。完整派生件未重复下载：服务器 `materials/stair/behavior_full.json` 的 SHA-256 为 `d481b4b5f549431cf61c1172822e992837bc118fd3267de9454244266c2ddb35`；具体句 ID、完整证据引用和 CPU 算例保存在 `code_math_and_full_refs.json`。

原运行结果仍为 `delivery_status=FAILED`、`stop_reason=execution_failed_or_interrupted`、`exception_class=IsolatedWorkerProtocolError`；它的 `actual_final_seen_input` 是 null。本检查独立沿实际 HTTP 与唯一保存的 attempt-1 payload 核对输入，不能把这份补充审查回写为原运行 PASS。主代理另行完成的离线 admission/delta PASS 也不能替代原运行状态。spawn/导入故障及后续记账故障不计为作者事实错误。

## 2. 作者实际用了什么

`rationale.md` 第 3 行引用了 F_c1=0.6255、前沿 L3/L4/L5 成功数 36/36、17/36、2/36，height_ood 的 19/36、9/36，joint_ood 的 19/36、16/36，forbidden_contact≈0.1308、clearance flag≈0.0752、slip flag≈0.8275、goal_not_reached flag≈0.2002。这些均能在实际输入 `/public_feedback_summary/feedback_entries/0/feedback` 下定位，分数与成功计数没有被混为一谈。

第 1、5 行关于原程序缺少显式摆动脚抬升项、保留八个原项及系数的描述，可由 `/source_payloads/0/content` 和候选源码核对。八个旧函数 AST 一致；新增函数只有 `swing_foot_clearance`，原八个 RewardTerm 的 weight、fn、clip 设置均未变。

新增报告已提供但没有可辨认引用的内容包括：

- `/public_feedback_summary/behavior/profile/report/process/0`：r1.e63 的 root tilt 232–262 与 low progress 255–294 **有重叠**，终止步 338。
- 同路径 `/process/1`：整回合姿态范围及 213→283 的 0.261 m 最大后退。
- `/process/2` 与 `/counterexample/0`：同 cell/run 成功回合也有 low progress；不能用高出现率直接区分失败。
- `/gaps`：规则未覆盖与比较缺失、数值采样范围及未采样部分。
- `/visual/0`：r1.e282 在 v0 的 step 1 平、99/196 头端上仰、343 再平；仅为稀疏相机视角观察，不是接触或精确角度测量。
- `/visual/1`：2 个原始陈述，1/864 回合，最终展示 1 句。

按冻结口径，本轮新增报告的显式 ID、引文、唯一可识别释义均为 **未识别到**。作者关于“脚抬高”的新机制，与报告的“躯干上仰”不等同，不能仅凭共同出现“抬升/姿态”就算成视觉证据使用。其“超出 prompt 的 timing unknown”没有否认 prompt 内已有时序，不算硬错。

## 3. 两处直接成立的问题

### 3.1 完整正向贡献的 4.75 上界不成立

原句（rationale 第 7 行；源码第 116–117 行）：

> Unweighted positive bound of the complete program before control_dt: 3.5 plus 0.75 plus 0.25 plus 0.25 equals 4.75.

对应冻结错误类别：`reward_bound_as_pointwise_dominance`，具体是把系数当成该项的逐状态贡献上界。虽然原文写了 “Unweighted”，列出的却是 weighted coefficients；无论怎样理解这个词，4.75 都不是其所写完整程序的正确正向贡献上界。

实际输入 `/source_payloads/0/content` 已给出 `feet_air_time`：`sum((last_air_time - 0.5) * first_contact) * moving`。候选第 40–46 行保留了这个函数，第 70 行系数仍为 0.25，**没有 clip**；`last_air_time` 的可见契约单位是秒，不是 [0,1] 的归一化分数。

直接从原候选 AST 提取函数、在 CPU float64 上执行：四个支撑体在本步 first_contact=True，last_air_time 各为 1.1 s，commanded_xy=(0.5,0)。原函数返回 `4*(1.1-0.5)=2.4`，加权后为 **0.6**；四体 contact=True 时新增 clearance 项为 0。两 tracking 项在跟踪误差为零时共 4.25，所以正向项合计 **4.85 > 4.75**。

这是合法 ABI 数值上的函数反例，用来反驳“程序上界”；**不声称本次评估真实经过该状态**。既没有训练新策略，也没有用这个例子推断策略会跳跃。

### 3.2 历史接触掩码不能保证当前完全腾空时严格为零

原句（rationale 第 5 行；源码第 105–106 行）：

> multiplies by a whole-axis has-stance mask so the term is exactly zero when the robot is fully airborne.

对应冻结错误类别：`invented_unobserved_fact`，具体是把历史接触指示当成当前物理支撑事实，并由此宣称一个代码保证。与此相关的 “highest contacting support body” 也必须限定为**掩码选出的支撑体**，不是已证明仍在当前接触的支撑体；不重复计成第二个问题。

实际输入 `/reward_inputs_contract/field_semantics_constraints/support_contact` 明确写了：最大力取自 3 个历史采样；`recent-history indicator, not current-sample-only contact`。因此刚离地时，即使当前四体都没有物理接触，掩码仍可能为 `[True,False,False,False]`。

同样执行原 `swing_foot_clearance` 函数：四体 z 为 `[0.05,0.20,0.20,0.20]`，上述历史掩码使 `has_stance=1`、参考高度 0.05，返回 **0.45**，通过声明 clip 后仍为 **0.25**。所以代码保证的是“历史掩码全 False 时为零”，不是“当前完全腾空时必为零”。CPU 算例并不声称该瞬间在本次录像中出现。

## 4. 不应误判为硬错的部分

- 新增项的 weight=1、clip=[0,0.25] 确实给出逐状态加权上界 0.25。
- “at most 7.1 percent of the 3.5-weight velocity anchor term” 明确带有 **3.5-weight**。可按两个项的最大尺度之比理解，不额外判成“每个状态必低于实际 tracking 贡献 7.1%”。其表述可以更清楚，但本轮不重复计错。
- `support_pos[...,2]` 选的是已公开的 z 坐标；支撑成员轴采用整轴 reduction，无证据表明越过 opaque-axis 权限。没有改多机制、加新输入或修改门槛。
- 未来 F_c1、J_OOD、success 改善写成 falsifiable expectation；尚未运行，不当作已实现收益，也不因没有实现而记错。
- “ordinary flat-trot stepping is not opposed” 的严格可确认部分只是新增项非负、低于参考高度时不直接扣分；对学习后步态的影响未验证，不把一句设计动机扩张为它已宣称实测不退化。
- “Every weak cell demands a larger step-up than the regular bins” 的范围不清楚：实际输入省略了 cell→具体几何的成员映射，且 regular bins 未定义。本检查标为**需要限定的未充分支撑解释**，不作为已确立的第三处硬错，更不能把它当作新增时序/视觉报告证明了“脚撞立面”的事实。
- `amax(heights * contact)` 若所有被选高度都小于零，未选成员的零会改变最大值；当前可见 ABI 没有给出这类状态的真实可达性。本轮仅记录这个条件边界，不推断本次策略已经触发，也不新增硬错计数。

## 5. 对主线的有限启示

这次烟测说明真实过程与视觉短句已经到达作者请求，但单次成稿仍主要围绕旧统计和参考 reward 构造假说。可以把“提供了什么”与“最终理由明确用到了什么”继续分别记录；不能用这一份开发样本证明新增反馈无用，也不据此重开 A/T 价值实验。

两个明确问题都发生在作者对 reward 代码及输入字段语义的解释上。它们不证明过程报告或视觉观察错误。后续若调整提示，优先让作者明确区分 ABI 的历史量与当前量、weight 与完整函数上界，并允许它说明某条过程证据不足以支撑当前机制；不要强制为每项改动编造视觉引用。

机器可读记录见 `N4_STAIR_READ_SMOKE.json`；本报告没有给出任何正式门判定。

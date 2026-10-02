# N.5 focus 材料：窄范围复核建议

2026-09-30。**DRAFT / NOT_FROZEN**；只核旧开发资料的字段、覆盖和字节，未评分、未采集、未调用 API。结论：**现有 focus 包足以开始建立五束事实，不应把“未提供所有回合完整曲线”列为整体阻碍。** 缺口影响特定时序主张；24个选中回合已有全程数组，其完整派生评阅无需再扩关节数据。

| 五束 | 已有原始/直接摘要字段 | 足以做什么；具体边界 |
|---|---|---|
| ① 分母与结局 | `complete_task_pool_members`、`scope_denominator/outcomes/successes`、每行 `original_success/outcome/terminal_record` | 全成员计数、任务/域/run 范围、主导原始失败类；不拿选中12例代替全池。 |
| ② 常见与不常见过程 | Stair：`threshold_intervals_by_component` 的低进展、倾斜、同脚 loaded motion/unloaded low speed、带符号 vz、clearance、历史接触；`curve_extrema`。Push：每事件 `condition_false_intervals/stable_intervals/first_dwell_end_step_by_arithmetic/window_extrema` 与原始事件记录 | 按相同谓词/极性/持续定义逐成员求 n/N；Stair 各现有区间可直接求交、先后和持续，**无需再发全量脚/关节数组**。Push 摘要区间仅覆盖各恢复窗口，不能推广到整回合。 |
| ③ 相容成功反例 | 上述每个成员同样有成功/失败、`cell/run_group/scope`；另有 `selected_episodes.json` 的 matched_to/match_level | 两侧同域频次及反例定位，缺匹配如实记；不是只有挑出的成功例才进入分母。 |
| ④ 主要过程关系/关键量 | 所有成员的区间、极值/步号、原始终止标记/事件锚点；选中24回合完整逐步值 | 已能回答“同脚低运动与低进展是否重叠”“某违反段在 latch 前/后且位于窗口内”“极值/终止谁先”等，也满足“至少一个有定位的主要关系/关键量”。不能从极值摘要恢复未列出的整段变化，缺项见下。 |
| ⑤ 缺观测与采样范围 | `missing_fields`、definitions 的 source/axes/clock/units/contact_history/sample_limit、原始终止/删失与窗口实际步范围 | 可区别真实未采、历史槽时间未知、控制步离散采样及未发送媒体。**NPZ已有但摘要未送的量应记 not_supplied，不叫 not_recorded。** |

具体尚不能从 focus 摘要确定的关系和最小补充：

- **Stair：** `root_x_drawdown_m` 的最大值与发生步已给，但回退**起点**未给；全程 roll/pitch 极值及合并的 `abs_roll_or_pitch` 区间不能恢复各自正/负方向的全部时段；up_z 最小值/步不能给出首次 `<0`、持续长度，或在最小值恰位于 T 时判断 T 前是否已经 `<0`。优先给 focus 增补可复算的算术结果：最大回退起止的原始 x 值/步，以及按既有协议阈值的 signed roll/pitch、up_z<0 区间；明确它们是直接算术，不是行为真值标签。所需底层原始操作数仅 `episode_length + base_pos_w.x + base_quat_w[wxyz]`，不需要 joints/全身体状态。只有要求模型逐值重算这些全池区间时，才额外投这5个数值列的全程流；不要默认扩大全部数组。
- **Push：** 摘要已覆盖恢复窗口内的条件顺序/重叠、dwell/latch；但不能确定窗口结束后至下一注入/终局的再次违反，也缺锚点/首个响应/窗口末的完整 pose 和 planar_motion，故不能计算这些点间净位移或联合姿态变化。最小原始补充是全程五列 `conditions`（可无损游程编码）及固定 `anchor, anchor+1, observed_window_end, terminal_T` 的 `pose[6] + planar_motion[4]` 原值。保留原始事件与现有 window_extrema，无须复制整段所有 pose，也不新增关节/接触字段。编码必须能还原每个有效步的原布尔值，不能把若干短段合成连续段。

若某问题/报告实际主张上述缺项，就在事实阶段统一补相应 scope 的操作数/直接算术，不能只补报告阳性成员；若五束用现有关系已可回答，不为追求无限过程发现而扩材料。未定义的新组合仍可由24回合完整轨迹提出，但不能以这些例子声称全池普遍存在。是否采用补充、其列和算法须采集前冻结，不能看到新答案后决定。

**旧开发包实测（UTF-8 文件字节，不是 token）：** 217份可读JSON合计31,055,857 B，其中 pool/focus/episode 有重复用途，不应整包发送。按24 episode＋12 focus拆为36个 logical units：

| 包 | 每包原文件合计 | 当前 runner 同型 text-block 数组最大值 |
|---|---:|---:|
| Stair episode（metadata＋全部连续parts） | 382,185–1,365,728 B | 1,456,514 B |
| Push episode | 67,742–222,635 B | 242,757 B |
| Stair focus（自含全scope rows/全池成员/定义） | 1,111,707–1,334,967 B | 1,596,063 B |
| Push focus | 253,157–979,629 B | 1,148,418 B |

不要再向 focus 包重复添加同一份 `pool/*.json`/definitions。若使用上述 Push 最小补充，新增每focus 174,104–767,201 B；旧池336,655个原始条件布尔值已做编码/解码完全一致检查，未生成评阅答案。作为容量备选，Stair 全程 root-only 原值新增每focus 3,739,406–5,085,901 B；因此优先算术区间/起止见证，完整 root 流只用于确需逐值重算的范围。

完整窗口原图另计，不能被这些数值字节替代。以上 text-block 数组尚不含协议/问题/system prompt/原图及输出预留，不是最终HTTP body或上下文合格证。36个单位也不是放行的调用数。逐包先做离线容量，再用现有 runner；不需要新平台。

逐包文件/hash/字节与补充投影测量见 [packet_size_draft.json](./packet_size_draft.json)。来源：`export_confirmation_raw.py:148–243`（直接摘要），`:250–277`（完整selected回合），`:395–427`（全池与focus）；对应原始 manifest/NPZ SHA 已核。旧31MB文件及原始NPZ均未修改。

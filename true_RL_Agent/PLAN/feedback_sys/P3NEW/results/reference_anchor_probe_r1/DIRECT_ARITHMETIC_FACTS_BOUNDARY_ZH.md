# facts 中立算术浏览辅助：最小边界与字段清单

2026-09-30。接续 [过程输入审查](./PROCESS_INPUT_AUDIT_ZH.md)。只读当前协议/生成路径，并查看既有 Push 开发包的字段结构；只新增本文，未修改源代码、冻结材料、原答或评分，未调用 API/GPU。本文供后续 `related_v2` 冻结使用，不追认旧轮。

**可采用现有独立算术对象作为 episode 的浏览辅助。** 让模型重新对所有标量计数、找区间，既不是必要独立性，也与已采纳的职责划分不符。独立性在于：必要事实不由生产报告、规则阳性或预写答案决定；模型仍看完整原量并独立选择和解释关系。算术浏览对象不具有“必要事实清单”或“生产结果已核通过”的权威。

## 1. 已有依据与当前路径

[N5_FACT_REVIEW_PROTOCOL_ZH.md](../../N5_FACT_REVIEW_PROTOCOL_ZH.md) 的以下条款应合读：第10行禁止自动过程报告/生产阳阴性标签；第20行禁止把程序全量核验票塞入 facts；第53行明确“导出器的阈值查询……不自动成为必要事实”；第67行允许原始 conditions、事件/终止标记作为记录的测量/判据结果；第124/146行已允许直接算术区间、极值、分布与匹配操作数。[职责修订](../../REVIEW_RESPONSIBILITY_REVISION_ZH.md) 第17–23、31行要求全量确定性由程序负责，同时保留原量和未知新关系。

当前 [export_confirmation_raw.py](../../export_confirmation_raw.py) 的 `stair_measurements`（148–224行）从原量计算完整阈值区间、曲线极值、回撤见证；不导入生产 behavior/reducer。`push_measurements`（268–298行）从 pose 按原dtype阈值比较、验证原始conditions，并算每个原始注入窗的区间/驻留。`episode.json` 已将这些存入 `episode.direct_measurements`（306–325行）。

[prepare_confirmation_materials.py](../../prepare_confirmation_materials.py) 的 original profile 直接投递这些原件（443–446行）；related focus 从它们构造全成员操作数（144–255行），而 related episode（94–128行）把整个 `direct_measurements` 剔除。因此恢复有限算术浏览辅助是统一当前已有职责，**不是新建行为摘要模型或降低必要过程要求**。

但现有对象不可盲目全传：Push `direct_measurements.arithmetic_latch_equals_original_record`（导出器289/298行）是与生产终局记录的一致性结果，属于核验回执，必须留在 report/audit，不能作为 facts 算术浏览内容。

## 2. `direct_arithmetic` 的最小白名单

它作为既有 `raw_numeric` 材料内部一个清楚标为 **derived arithmetic from supplied raw measurements** 的子对象即可，无须新工具、平台或请求阶段。原材料名/容器角色不能把派生值伪装成传感器原始输出。全程原量及时间点照常送达。

| 任务/字段 | 精确定义与用途 | 边界 |
|---|---|---|
| Stair `curve_extrema` | 既有每列 `component_shape/c_order_components` 极值与原始步号；有限值参与，最早并列步依既有定义 | 是单列数值索引；不能把不同列极值拼成同一步姿态或因果链。无需再复制全部曲线。 |
| Stair `threshold_intervals_by_component` | 既有全部谓词、全部身体的完整闭区间；例如 signed roll/pitch、abs tilt、up_z、loaded planar motion、unloaded clearance、low-progress等；保留空列表和缺量区别 | 只解释为精确定义成立的采样步，不使用 `slip_detected/倒地/碰撞/恢复` 等诊断名称；不只给满足驻留的长段，不按生产阳性筛选。不同脚的区间不能合并成同脚连续。 |
| Stair `raw_termination_intervals` | 从原始 termination/done 离散列无损取出的闭区间，term顺序来自原合同 | 是原始终止判据记录；不是导致终止的完整物理机制。 |
| Stair `root_x_max_drawdown_witness` | 最大先前world-x到后来world-x的差；带首尾原值与步号，最早终点/前缀峰值规则 | 不表示该整段单调回退，不表示典型或必要过程。 |
| Stair `root_first_last_xyz_terrain_m` | 原始首末同一行的三个terrain-relative分量 | 只证明端点，不能直接命名中间动作。 |
| Stair `focus_operand_missing_fields` | 对这组辅助对象缺操作数的字段列表 | 不能代替完整包的可用性映射；空列表只证明本组操作数齐全。 |
| Push `curve_extrema` | 全回合 pose/planar_motion 各列极值与步号，仍按原字段顺序 | 不把全部回合极值缩成注入窗内，也不拼接不同时间分量。 |
| Push `events[].event_index/anchor_step/window_observed_steps` | 实际事件身份、原始anchor；观测窗口 `0 < step-anchor <= recovery_window_steps`，完整记录实际有效步 | anchor样本在velocity write之前；首个响应是anchor+1。窗口实际长度不足必须保留，不能补齐。 |
| Push `events[].condition_false_intervals` | 五个明确阈值条件逐个为false的全部闭区间；阈值按pose原dtype比较 | “某条件违反”与“某条件成立”极性不能混；范围仅本事件观测窗口。 |
| Push `events[].stable_intervals` | 同一时间步五条件AND为true的全部闭区间 | 名字中的stable仅表示五判据同时成立，不是独立证明物理稳定/恢复机制；须在对象定义中明确。 |
| Push `events[].first_dwell_end_step_by_arithmetic` | 已验证注入窗中首个AND真区间长度≥required_dwell_steps时的 `start+dwell-1` | 不是此后一直稳定；null应由注入验证、覆盖范围与区间联合解释，不笼统写“没有恢复”。保持当前验证门，不将未验证事件当已验证。 |
| Push `events[].pre_latch_stable_intervals/window_extrema` | 首次驻留确认点前截取的AND区间；窗口内逐列极值 | 同时保留完整window AND区间，避免把首次latch后的再违反藏掉；不能把pre-latch最大驻留写成全窗最大。 |
| Push `focus_raw_operands` | 已有全程conditions无损RLE，固定anchor/anchor+1/窗口末/T及最早速度极值步的完整同一行pose-motion、缺步/缺字段列表 | 对episode与完整原量重复时可只保留所需原始landmarks引用，不再复制一套conditions RLE；不得引入手工选择的关键时刻。 |

**明确排除** `arithmetic_latch_equals_original_record`、`checks/matches_runtime/PASS/expected_vs_actual`、生产 `events/segments` 或摘要文本、残差阳阴性、代表性排序、必要性排名/necessary_reason、事实ID/答案，以及观察者答案。不能为规避限制仅改这些对象的名字。

区间浏览足以降低算术负担，暂不需要再增加几百条逐实例解释文案或另一模型摘要。Stair 的OR及驻留若复用已有定义，可提供纯算术视图；但最小实现先提供全部逐部位区间与明确any-OR定义，不增加新行为谓词。若提供首次确认步/计数，只能是同一冻结公式的派生索引，完整未达驻留的区间不能删除。

## 3. 独立性与未知如何保留

生成器只接受原始manifest/chunks、固定测量定义、原始终局/事件时钟；不接受被评生产报告、生产语义事件集合、模型事实或评分。既有导出器复用 P3R 的 `spans/extrema` 通用算术helper与task-owned几何定义，这只能证明实现来源不同于生产行为reducer，不能由同一helper给自己宣布全量正确。独立CPU核验继续绑定原量、定义、生成器hash，核全部区间/极值/空值/身份；report才发送其一致性结果。facts只发送算术对象本身与来源，不发送“已通过”的票。

对象定义用一段明确说明即可：这些是预定义数值查询的浏览索引，不是必要事实或诊断结论；所有相关原量1..T仍可查。模型需审读主要阶段、重复/重叠、先后和反例，不必对每个标量重复算术判词；可提出索引未命名的关系。索引未覆盖的量保持可用性/未分析范围，不能用无索引推断没有过程。对于未知原因、接触面、未录媒体等仍保持unknown；对于已有操作数而没有实际审读的范围，记录未评，不伪装成not_recorded。

必要清单仍由两位评委分别封存、取并集并保留争议；辅助字段数或区间数不成为事实数量，也不进入支持率分子。新模型包增加此对象是新版本表示变更，应冻结规则/源码/材料/容量；旧答案与成绩保持。

## 4. Push 终局字段不能按名字一刀切

核对既有开发包 `results/n5_numeric_packing_related_v1/materials/related/episode-13.json`、`.../materials/raw/episodes/push-0001/episode.json` 的实际键，并对照 [runtime.py](../../../../../src/true_rl_agent/tasks/push_recovery/runtime.py) 355–385行、[failure_classifier.py](../../../../../src/true_rl_agent/tasks/push_recovery/failure_classifier.py) 135–197行及 [numeric_capture.py](../../../../../src/true_rl_agent/tasks/push_recovery/numeric_capture.py) 274–315、331–388行。这里是历史开发包字段观察，不声称它等同R2最终包。

| 字段组 | 实际含义 | facts处理建议 |
|---|---|---|
| `success/failure_label`；control-only时的`control_success/control_failure_label` | 任务runtime的原始结果分类；`failure_label`也被source_rows映射为`original_outcome` | 既有协议允许原始终局，可保留一个明确的结果表示；不能用fall_forward等终局名证明全程身体方向或机制。control_failure_label若作为控制组原始终局则允许；在非control回合只是附带分类，不应当过程真值。 |
| `episode_steps/stratum/control_only/planned_push_count/push_event_count` | 原始长度、分组/计划与实际事件数量 | 保留，用于时域/分母与实际注入范围。 |
| event的`event_index/control_step/time_s/requested_delta_v/observed_delta_v/velocity_before/after_write/after_next_physics` | 原始执行回执，before/after-write与下一physics观测时刻不同 | 保留原样和时钟定义；不能把delta-v叫持续外力、冲量传感器或接触真值。 |
| `injection_verified`、原始`event_verified` | 原始执行验证判据记录，不是行为诊断标签 | 保留，后续定义需要它决定有效窗口；不能删掉后默认事件全有效。 |
| `conditions/failure/timeout/observed_dwell/observed_recovery_step`；event的`recovered/recovery_control_step/recovery_censored`；`final_recovery_condition` | 采集时原始阈值/状态机判据与latch记录。`recovered`仅表示latch存在，`recovery_censored`表示未观测到latch | 按协议67行可保留，标为recorded criterion而非独立物理真值；同时给原量和定义。不能把所有布尔列误删为“生产阳阴性标签”。是否额外保留终局重复字段可按固定白名单去重。 |
| `recovery_latency_s` | 已latch时为 `(latch-anchor)*dt`，否则runtime填满计划窗口长度 | 若保留必须连同censored，未latch时的值不是实际恢复用时；不能当0或已恢复事件。 |
| `recovered_event_count/invalid_injection_count` | 从原始事件判据/计划计算的数量 | 可作为有定义的原始结果统计，不是物理机制；为减少重复，可在facts终局投影不再复制，用原事件或中立算术计数。别把invalid归零或删去实际缺注入证据。 |
| height/pitch/roll/drift极值、`final_planar_speed_mps`、action统计 | runtime数值摘要；action原量本包不一定供应 | 不是诊断标签。若保留须标原记录摘要，明确哪些能从已给原量复算；不能把未供应actions写成已独立核验。 |
| `final_unstable_after_recovery`、`recoveries[].coverage`中的解释性类别、`failure_label=..._dominated`、`dominant_failure`、过程报告词条/残差标签 | numeric/behavior/report层额外归纳、分类或解释；不是注入原量 | 不作为facts的行为答案输入；保留其原操作数/完整覆盖。`coverage`若是纯raw采样状态如complete/truncated则仍需保留，不能连数据完整性一起删。 |

Stair 的终局 `diagnostic_flags` 是额外过程诊断列表，应从facts投递视图排除；Push的主原始终局与原始条件状态不能因形式类似就等同该列表。最终白名单应在两个任务各自定义，公共层负责执行投影/传输，不硬编码fall/slip等行为词汇。

## 5. 最小可实行更改顺序

1. 用一个任务分支明确的投递helper选择上述有限 `direct_arithmetic` 字段；三条投递路径（original、related episode、media）都经过同一边界，focus保持全成员操作数。特别清除original Push路径里已有的一致性bool，不能只修related。
2. related_v2恢复Stair planar speed/clearance完整列；其余六项以已供操作数/公式标为可重建。所有旧原量、步号、缺失语义及身份保留。算术辅助与数值块编号相邻，模型不必从元数据推断值。
3. 新协议工作版说明recorded criterion、neutral arithmetic、production diagnostic、audit verdict四类边界；核心是对象来源、声明范围与用途，不能只按字段名称判断。写入模型实际会收到的JUDGE条款。
4. 离线核验只有必要三件：字段白名单/禁止项检查；新增辅助与原量/既有独立核验范围一致；最终真实包容量。包括空区间/null/未验证注入/截窗/首次latch后再违反及same-body对any-body。它们都不能转成模型语义PASS；随后仍按新冻结的事实与报告流程验收，不重试旧例求通过。

此方案不要求新的平台、评分框架、多轮代理、删时间点或生产依赖PLAN；也不承诺仅凭输入改进就能解决长包语义阅读。若模型继续只复述终局与缺口，那是实际过程覆盖未完成，应保留该结果。

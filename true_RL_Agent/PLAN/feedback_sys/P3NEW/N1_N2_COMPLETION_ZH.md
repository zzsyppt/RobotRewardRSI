# N.1 / N.2 缺项修复与历史回溯

2026-09-29。范围限词条元数据、历史回溯、候选登记与规程。**开发修复通过，不是 P3-NEW 整体通过，也不是新的视觉资格或独立语义验收。** 新增 API 0、GPU 0，没有重跑原始大数组或修改历史封存结果。

## 词条单位修复

`tasks/push_recovery/process_report.py` 的9张词条卡原来没有显式量纲：驻留中断、条件违反区间、全条件稳定区间、latch后失稳、首次响应、窗内速度峰值、窗末响应、删失、未执行事件。

现在按实际 `numeric_capture.POSE_FIELDS`、保存的中断事件、`behavior._sample` 和 `_spans` 补齐量纲：控制步/秒/次数/布尔与类别分别说明；首次、峰值、末次响应是复合快照，分别列出高度m、roll/pitch rad、线速度m/s、角速度rad/s及可选world-xy字段。没有将整个快照错误标成单一速度单位。

阈值、词条定义、派生数值、短报告文字均不变。用旧已核验的v2派生和原phase中的seed来源重建新full，逐字段检查所有非lexicon字段相等、9张卡只有units新增，公开对象只有 `full_derivation_sha256` 改变；完整元数据变化必须有新的来源摘要，不能伪称整个请求字节不变。

- [新的Push完整派生](./results/n1n2_completion_r1/push_full_units_current.json)：canonical SHA256 `9299a937d63251679b2a2c00b38de18e7dd7b237b6c9cc09af0e668ef415de12`。
- [可单独阅读的词条卡](./results/n1n2_completion_r1/push_term_cards.json)和[新公开报告](./results/n1n2_completion_r1/push_public_units_current.json)。
- [输入来源和比较结果](./results/n1n2_completion_r1/verification.json)；原 `delivery_offline_r1` 未覆盖，后续新作者phase需要绑定新full/public再冻结。

相关CPU命令：`.venv/Scripts/python.exe -m unittest tests.unit.test_push_process_report`，**10项通过，7.890秒，进程退出0**。在原投影测试中补了实际保存的响应/区间字段均有量纲的检查。另由下述开发脚本核验真实192回合工件的非词条字段与全部报告文字不变。

## 三种历史形态的实际回溯

数值 v1 与当前full共享同一个数值archive `22fb7496…`，checkpoint、cohort、episode身份等域字段相等；因此能比较该数值档案上的词表变化。旧录像是**独立重放**：checkpoint、cell、终局和回合步数对应，但未证明逐步同轨，不把图像附成当前数值回合的同步观察。

“v1测量谓词命中”与“当前N.2区分状态”不是同一口径。v1没有N.2的三态残差分类器，下面不补造一个旧分类器。三个强视觉名称在两版都没有正式可测定义，均未晋升；已命中低层测量不等于已识别完整形态。

| 原形态与档案回合 | v1 实际测量 | 当前实际测量及 N.2 状态 | 本次有限视觉描述与边界 |
|---|---|---|---|
| 后翻，r1.e64，426步禁触 | 终止前 tilt、low_progress、slip_any 命中；末段tilt 385–426，394确认 | 新增up_z<0首次418步；当前签名含tilt、low_progress、up_z<0。但同cell/run成功仅2例，必须为 **comparison_missing**，不能硬判阳性/阴性 | 重放显示418–425的采样格中躯干倒置、腿朝上；不从它推精确旋转轴、接触面或失败原因 |
| 腹部下沉，r1.e156，232步禁触 | tilt、low_progress均未命中；slip_any命中仅说明受载参考点运动，不识别腹部形态 | 当前四项签名为空，同域6个成功中4个也为空，实际为 **rule_negative** | 低躯干姿态伴腿形变化可见；图像不能量腹部下沉幅度或确认腹部撞台沿 |
| 冻结跨立，r1.e55，588步超时 | low_progress 106–588命中；tilt、slip_any未命中 | 新增RR_foot同体无载低运动113–588，137确认；同域6成功中0个匹配该组合，实际为 **rule_positive** | 180、250、330、410、500、587的离散格形态近似重复；不能说未显示间隔内连续冻结，也不能把单脚低运动扩大为全身/控制器冻结 |

当前三个残差状态、对照数和匹配数另与已封存的独立L0 oracle相等。没有为了“旧形态难描述”而改写正式残差：e64保留对照缺失，e55保留规则阳性。三种形态仍有**命名与可观测范围的缺口**，这类缺口进入候选台账，与正式规则阴性分开。

全部数值片段、媒体回执引用/图像hash、显示步、原描述和收窄记录见[逐案回溯](./results/n1n2_completion_r1/stair_retrospective.json)。本开发代理重新看过三张旧拼图；没有完整播放视频或新建采集。旧均衡化/缩放质量问题仍成立，所以这些例子只用于历史规程演练。

## 候选台账与归并规程

[实际台账](./results/n1n2_completion_r1/candidate_lexicon.json)包含3条Stair候选，每条保留原名称、范围收窄后的名称、图像与回执hash、显示步、不能断言的内容、归并操作与不晋升理由。同一录像多种名称计1次出现；不把翻译与重复查看当新案例。Push按同一字段合同保留空表及原因，不能从空表推“Push没有未知语义”。

[两任务登记、归并与晋升规程](./CANDIDATE_TERM_PROTOCOL_ZH.md)要求任务拥有测量定义、开发池同域统计、困难反例、独立确认及显式版本化。当前 **自动晋升0、手工晋升0**；不执行模型生成的规则，不新建公用任务语义框架。

开发案例和观察陈述位于[数据文件](./stair_retrospective_cases.json)，核查脚本[complete_n1n2_development.py](./complete_n1n2_development.py)只读取已指明的历史输入、提取数值和记录身份检查。运行命令：`.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/complete_n1n2_development.py`。输出 `units_fixed=9, historical_cases=3, promoted=0, public_text_and_metrics_unchanged=true`。

## 仍然缺少的证据

三个视觉名称均没有可直接晋升的数值定义与独立确认；e64还缺足量同域成功对照。旧媒体与数值未同步、拼图尺度和后处理限制未消失。以上均已进入台账，而不是用“未可判”冒充阴性或用开发回溯冒充正式通过。此修复没有扩大N.3/N.5范围；相关真实采集与正式验收仍由主代理按当前计划继续。

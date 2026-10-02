# N5 R2 最终离线投递工程核验

2026-09-30。结论限于产物与入口的工程事实：**两任务最终短报告均在既定容量内，完整进入保存的生产序列化请求；未发现本次报告到请求之间的内容丢失。** 当前仍是 `offline_serialized_not_sent`，不是作者实际收到、独立语义通过、必要事实零遗漏或 Phase 3-NEW PASS。本核验未发模型、未启动训练/仿真、未重跑测试，也未评阅正式 facts/report 模型原答或评分。

后续经授权完成只读 CPU 补证，结果见 [R2 来源与焦点补证](./SOURCE_AND_FOCUS_SUPPLEMENT_ZH.md)：Push 12 包 144 条记录字段来源链一致；真实 Stair 864 / Push 192 全池的焦点分母、残差、数值覆盖及固定代表/对照选择，共 1,343 项比较无差异。下文涉及旧算术票的缺口描述保留为历史票据范围；新增证据闭合的子集以补证文件为准，未改冻结原件或旧票。它仍不构成完整责任轴 100% 或正式语义分数。

依据为主计划 Phase 3-NEW（正文行 392–445）、[N.1–N.4 索引](./N1_N4_EVIDENCE_INDEX_ZH.md)、[R2 合同](../../N5_RUN_CARD_R2_ZH.md)、冻结协议与源清单、[装配回执](./delivery_offline_r1/assembly_receipt.json)、两任务 `public_report.json` / `user_payload.json` / `request.http_body.json`、[数值只读回执](./numeric_readonly_review.json)。代码先查询 CodeGraph；查询未返回所需新符号后，才读取指定文件。所读的 12 个生产入口/投影/视觉/容量文件，以及 Push 独立算术 helper、事实协议和装配入口，当前字节均与 R2 冻结清单相符。

## 最终交付边界

| 工程事实 | Stair | Push |
|---|---:|---:|
| 短报告字节 / 上限 | 6,930 / 8,192 | 11,084 / 12,288 |
| 完整 user payload 字节 | 70,674 | 60,286 |
| 保存的 HTTP body 字节 | 76,927 | 65,771 |
| 报告字符串叶子；未进入 payload | 53；0 | 91；0 |
| 报告句数；容量删除句 | 14；0 | 21；0 |
| S3 显示句数 | 1 | 1 |

六个文件的 SHA256 和长度均与装配回执一致；两份 HTTP body 都含 system/user 两条消息，解码 user 内容后与对应 `user_payload.json` 完全相等。完整报告对象分别位于 `/public_feedback_summary/behavior/profile` 和 `/public_feedback_summary/public_dev/behavior`，与独立报告文件完全相等。字符串叶子数只用于字节投递核验，不是必要事实分母或语义分数。8/12 KiB 限制针对行为报告，完整作者请求还包含任务合同、权限、来源源码等必要底座。

最终 payload 保留短报告及全量派生指纹，没有装入原始数组、完整回合派生或全部观察者原答。七节 `results/process/counterexample/visual/gaps/parent_diff/actionability` 均在报告内。`evaluation/process_report.py:161–240` 按整句投影，容量处理只删除已登记的可选句，保护必需句及各节最后一句；本轮 `omitted_statement_ids=[]`。这只能证明当前既有报告完整投递，必要过程本来是否充分仍由独立分母验收。

Stair 数值例为 `r2.e173`，保留根部 x 起终点、两段回撤、是否重返原 x、局部 signed pitch 与终点截尾；成功反例另标 `r2.e177`。视觉句另标 `r3.e63/base`，不能接成同一回合过程。Push 数值例为 `r1.e18`，保留 anchor、观察窗、dwell/latch、latch 后再次失稳、末段到观察截止；成功反例另标 `r1.e16`，视觉句另标 `r3.e59/left_palm_link`。两条视觉句均明确 `sparse cited steps; same_run`、不确定性和 `Visual only`；这里的 same_run 只说明该视觉记录与其自身绑定数值同源，不代表与报告的数值代表例相同。

未知范围实际进入请求：Stair 写明 16/864 数值抽样、残差/对照缺失 4/97 覆盖及 93 未抽样，且“数值抽样不等于视觉覆盖”；Push 写明 10/192 数值抽样、各域规则阴性/对照缺失和未检查数，并说明缺 foot/contact/yaw。两者分别保留 2/864、2/192 的视觉回合范围。无可比父策略时均显示 missing/unavailable；可操作性映射区分可表达、proxy、需新观测、诊断专用和权限外，未把观察写成已证实原因或奖励改动收益。

## 拒收、来源与生产边界

`evaluation/visual_review.py:36–80` 要求原观察与不确定性的完整票、来源绑定和评阅身份；没有 review 时为 `not_evaluated`。`evaluation/visual_observer.py:603–617` 仅允许机械有效、来源别名获准、且绑定 review 为 supported 的项进入 S3；数值谓词一致不自动支持自由文字。`visual_review_binding` 保留原答、身份、图像和测量绑定；任务后来增加的拒收标记不扩大可用集合。

两任务 `attach_*_visual_observations` 先核 checkpoint/protocol/run/episode/source，拒收来源不符项；冲突优先，随后由任务层按数值代表例、失败、其余回合排序，默认只显示一条。完整观察仍留在完整件内。装配入口 `assemble_delivery.py:148–173,230–241` 对原始 full、合并机械审计与 review 做绑定，核对视觉附加未改变非视觉完整件和报告的必需过程/反例/缺口/可操作性内容，并限制实际加载的生产包在冻结 `src` 内。

报告中的“3/6 semantically supported”来自 **root 非盲模块核对票**；本审查只确认其投递机制，不认可为正式正确率。过滤前全部 3/6 条观察、所有 uncertainty 命题和最终保留句仍须按正式协议独立评分；只显示一条不能缩小视觉分母。

对两任务 `phased_evaluation.py`、`behavior.py`、`process_report.py`、`visual_observation.py`，以及公用 `process_report.py`、`visual_observer.py`、`visual_review.py`、`prompting/task_sections.py` 共 12 文件实查，未见 PLAN import、PLAN 路径加载、importlib 或 sys.path 注入。finalizer 使用任务自有生产派生/投影，视觉投影使用正常 `src` 公用层。离线装配/评阅 harness 确实仍在 PLAN；不能由此宣称“PLAN 已只剩文档”，也没有证据将它当作上述生产模块的运行依赖。

## “责任清单 100%”与 Push 来源缺口

冻结采集卡用 `deterministic_declared_coverage=1.0`，事实协议行 20、33、98 明确责任清单内形式化对象全量核验，未核不能自动通过。R2 合同行 29 和采集卡 `known_limits` 又在采集前明确保留 Push NPZ 来源、pose 物理构造和 action norm 的未核范围。二者应同时报告，不能用 24/24 零差异消掉缺口，也不能事后新增无限的物理真值验收。

| 范围 | 现有证据与判断 |
|---|---|
| 已供操作数上的正式派生 | 当前 24 episode 结果为 `CHECKED_WITH_EXPLICIT_LIMITS`，差异 0。Push 12 包有 4,807 个递归检查路径，覆盖完整恢复集合、条件、区间/顺序、布尔/null、dwell/latch、再失稳、同一步响应与范围。它证明已供操作数到声明派生之间的一致性。 |
| 原件身份与操作数来源 | 旧票 Push 12 包均为 `NOT_AVAILABLE_IN_FROZEN_HELPER`；helper 行 329–330 排除 `source/env_id`，行 343–347 列 source、source_pointer、original_logical_id 等未核。新增 [来源回执](./source_link_audit_r1/receipt.json) 已对这 12 包核原 archive→导出 NPZ/JSON→related 12 字段字节及 full 身份，144 条链零差异。该部分现已补证，旧票保持原样；不能延伸成 pose 物理构造或全池全部字段认证。 |
| 原始 pose 物理构造与动作指标 | helper 明示缺四元数、完整角速度、absolute/reset root position、action vectors；仅复核 pose 阈值与范围。它们在 R2 已明确未核，不能计入现有算术 PASS，也不能由此断言测量错误。完整独立 L0/物理构造仍未证明；不应追加“独立仿真碰撞真值”等未冻结目标。 |
| 终局含义、共享定义、群体比较 | terminal_label 只核原记录复制，含义/优先级、词条限制、ABI、关系与 shared 不在旧算术票内，仍由相应独立来源/定义/语义责任处理。新增两份 focus 回执已从本轮原池核 12 focus 成员/结果计数、逐失败残差、固定代表/对照和数值采样覆盖；未核全部 focus packed 区间/极值、自然语言和必要事实。24 episode 旧结果及 N.1/N.2 旧池结果不能代替这些本轮对象。 |

所以可写“已声明算术检查零差异，未核项保留”，当前不能写“完整形式化责任轴 100% 已完成”。最终需要逐项对接实际已核路径、来源轴证据和模型负责范围；本审查不将机器检查计为第二家模型的 supported 票。

## 仍未完成的验收

1. 双家事实封存、报告后揭示、实际完整执行，以及完整派生/有界报告/最终请求的支持率、必要过程召回与定位；本审查不读取或预判正式结果。主线程通知正式调用已停，引用契约修正由主线程处理，不属于本次只读补证。
2. 过滤前全部观察句的独立画面支持、越权、相关误差与冲突结果；root 模块支持不是该成绩。
3. 补证后仍未覆盖的责任对象逐项闭合，包括全部 focus packed 区间/极值等形式对象；已核来源链、组级/实例子集不能扩大成全部责任完成，详见补证的最小剩余清单。
4. 当前源码的最终请求只有离线序列化证明。R2 合同明确 N.4 八槽已满，本轮不得为“补实际送达”追加作者调用；历史八槽接收只覆盖其当时源码。

本次只核已存测试的覆盖：R2 18 项 CPU 测试针对 Stair cohort/replay 和 native snapshot，另两任务静态 preflight；继承旧投递回归核对序列化、容量、修复帧与必送字符串。这些证据不验证自然语言真实性、全部必要事实或实际作者有效利用。未建议扩安全脚手架、补泛化测试、重训或改验收门；整阶段仍应维持 `IN_PROGRESS`，由正式独立结果和责任闭合决定出口。

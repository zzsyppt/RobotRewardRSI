# N.5 新资料实际执行记录

当前状态：**两任务采集完成；原冻结 N.5 为 INCONCLUSIVE，修复后同批仅作工程诊断**。阶段仍为 IN_PROGRESS。此页记录实际执行，不更改已冻结的采集卡、方法、旧结果或资格。

用户于本轮明确释放显卡并要求继续。启动前服务器两卡分别占用 7/6 MiB，磁盘余约 67.59 GiB；400 个生产文件及冻结卡逐项核对，使用全新输出目录。Stair 于 **2026-09-30 02:36:08 UTC** 真正启动，见 [原始启动回执](./stair.started.json)。无自动重试。

- 采集源：`/data/yuyangliu/AGENT/feedback_p3new_20260929/source_n5_preflight_r4`。
- 输出根：`/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20260930`。
- 运行回执及日志：`/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930`。
- 顺序为 Stair → Push；只有真实终态与结果核验后才推进。逐控制步、双视角、原始分辨率保持冻结值。
- 首个 seed 的四张原始首帧已实际查看，全身及地形参照可见，见 [有限画质检查](./early_media_quality.json)。这不是完整录像或语义验收。
- [独立后处理树](../n5_postprocess_source_r1/README_ZH.md)保留冻结源码字节；三个上下文依赖用采集前 N.4 的既有 SHA 补录，未替换运行中的采集源。

Stair 于 **03:58:21 UTC** 退出，code 0、执行结果 PASS：864 个回合、4,900 张原始 PNG。关键 evaluation binding、数值 manifest 和完整过程文件哈希核对一致，见 [Stair 退出回执](./remote_execution/stair.exit.json)及[启动 Push 前的核验](./remote_execution/stair.before_push_check.json)。Push 于 **04:00:36 UTC** 启动、**04:37:52 UTC** 退出，code 0、执行结果 PASS：192 个回合、4,696 张原始 PNG，见[退出回执](./remote_execution/push.exit.json)和[执行/绑定核验](./remote_execution/push.capture_check.json)。原量导出均完成；独立语义资格另计。

Stair 原始证据导出、冻结媒体选择和两包观察者准备均已完成；选择得到成功与失败各一回合、共 24 张原图。24 张均实际以原始分辨率查看，见[选定输入画质检查](./stair_selected_media_quality.json)；全身与相关台阶可见，局部低对比和自遮挡仍作为限制。图像、选择与提示未改。

Stair 两次 DeepSeek 首次观察均已收到，fingerprint 与冻结值一致，格式/数值审计通过，共 6 条原句。两次按回执用量与已登记峰时单价计量合计 **US$0.023006376**，不是发票实扣；全局工作账本保留历史条目，[当前副本](./remote_execution/budget_ledger.working.json)。两次发送均由[轻量执行适配](./run_stair_observer_once.py)在原 sender 外登记预算，没有重试或更改 sender。

root 作为非盲开发者对全部原图、原句及 uncertainty 做了一次模块准入审查：2 条 supported、4 条 insufficient，见[原始合并审计](./observer_prepared_snapshot/stair/combined_audit.json)、[完整模块审查](./observer_prepared_snapshot/stair/root_module_review.json)和[封存回执](./observer_prepared_snapshot/stair/root_module_review_receipt.json)。细节主要涉及把整体姿态扩写成四足共同高度、外张/收拢或所有腿的方向。这是模块过滤判断，**不是独立 N.5 评分**；6 条原句全部保留在正式视觉分母内，不改写后补过门。

Push 原[媒体选择卡](./push_original_selection/card.json)错误排除了四个真实成功回合：任务保存 `success=True, failure_label=success_recovered`，旧选择器却要求标签为 null。原观察者准备因此缺成功 e1 而失败；并非采集池确实没有成功。原源树、原选择卡和失败均保留，本轮不放行正式 facts/report。按冻结纪律，修复后的同批只作工程诊断，不回填原轮 PASS。

现行源码将已登记终局的分组交给 Push 任务层 `recovery_outcome_class`，实验层保留原 hash/缺类/顺序规则。成功空标签、失败配成功标签以及未知终局不再被误收；实际生产 v2/v3 终局输出未改变。测试 fixture 改由生产 classifier 生成，9 项选择测试及 25 项已有 physical protocol/numeric capture 测试通过。独立只读代码复核未发现必须修正项；[实际六成员重算](./push_selection_repair_check.json)选出 run3/env3 成功、run3/env59 失败，未改原数据。

截至上述记录无新的正式评委或作者调用；本线程及开发子代理的桌面模型消耗另计、费用未知。N.4 八次作者机会已经用完；新最终请求只做 `offline_serialized_not_sent` 核验。修复后的工程检查与任何后续新确认必须分开登记。采集进展不代表语义资格通过。

## 同批工程核查进展

Push 修复后两包已按事前登记交给两个全新、无历史的 GPT6-SOL/high 上下文。文字与 24 张 1920×1080 原图从既有请求无损提取，原答共 6 条，机械核查通过。root 查看全部原图后封存非盲模块审查：5 条 supported，1 条 insufficient；后者在可支持的手臂外展之外，又加入未限定参照系的“肩部高度”。原答未修写，原始分母不变，见 [完整审查](./push_engineering_repair/push/root_module_review.json)、[实际载体记录](./push_engineering_repair/push/desktop_invocations.json)与 [CPU 核查](./push_sol_audit_r1/sol_engineering_audit_receipt.json)。这不是独立正式评分。

两任务先通过一次真实生产链的[离线装配](./engineering_delivery_r1/README_ZH.md)：Stair 6979/8192 B、Push 11126/12288 B，数值和非视觉内容保持，最终作者请求已序列化但未发送。检查实际文本时发现唯一视觉位置依赖原答包顺序，均选了成功回合；失败回合已有支持的形态却未进入短报告。现已由任务层提供数值代表案例、真实失败、其余的优先级；共享排序仍先保留数值冲突，再按任务等级、过程比较和原始顺序选句，不理解任务终局含义。

该最小修正已通过服务器 **114 项相关 CPU 测试**，并用原观察、原审查、原数值资料再次执行既有生产装配：Stair 改选失败 `r3.e63/root` 的 1–320 步比较，报告 **6956/8192 B**；Push 改选失败 `r3.e59/torso_link` 的 209–288 步比较，报告 **11063/12288 B**。各 6 条原观察、完整 audit/review、2/864 和 2/192 的覆盖分母不变；`nonvisual_unchanged=true`，原 445 文件冻结树不变。见[真实装配回执](./projection_priority_r1/downloaded/delivery/assembly_receipt.json)及[核验和最终请求哈希](./projection_priority_r1/download_receipt.json)。这些视觉回合仍不是数值正文的同一代表回合，必须保留各自 ID，不能拼成一个过程。

数值评阅材料已从临时接线收回正式实验入口：导出 manifest 绑定真实 `_private/sources.json`；confirmation pack 在建目录前核验该绑定及每个 task/run/numeric manifest 对应的任务定义，旧 development 明记未绑定状态。9 项定向测试通过，真实 CPU 重导/打包核对 222 个 JSON 对象、24 个具名数组归档和 72 份请求；445 个旧冻结文件未变。跨进程 JSON 键及 NPZ 条目顺序不同的首次比较失败保留，随后核对解码对象和具名数组的 dtype/shape/bytes，而未改原导出结果。见[源码与验收回执](./numeric_materials_r1/pack_task_definition_engineering_r2/change_receipt.json)。

Stair 原始观察中仍有多部位细节越界，现有提示已经要求独立命题与粗粒度支持。本轮不继续堆叠提示，按[载体开发登记](./stair_sol_carrier_development_r1/registration.json)固定两个原输入包，各用全新 SOL 上下文观察一次。保持原文字、数值、图像和顺序，不提供旧答、复核结论或任务结局；不重试、不改写原 N.5。新原答共 3 句，root 查看全部 24 原图后为 3 supported，内容收敛在机身与台阶的粗空间关系；完整记录见[开发对照结果](./stair_sol_carrier_development_r1/README_ZH.md)。下一次确认固定两任务均使用 fresh SOL/high 观察，不继续同包试提示/载体。该有限开发结果不认证必要过程覆盖，也不改原 DeepSeek 六句分母；载体差异包括模型及工具式图像读取，不能解释成单纯模型能力因果结论。

截至本轮工程收尾，DeepSeek 工作账本为历史 US$0.44538942 加本轮 Stair 两次 US$0.023006376，按既有单价与回执计量合计 **US$0.468395796**；桌面观察、开发子代理及本线程费用未知，不能计为零或混入该合计。无新增作者、正式评委或训练调用。下一步先统一冻结全部已验收修正及实际未使用的采样身份，保持 Push 独立算术核验的已声明范围、长包尚未实际完整评阅的限制；原轮不会回填通过。

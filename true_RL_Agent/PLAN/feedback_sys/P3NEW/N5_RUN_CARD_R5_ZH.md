# N.5 下一轮准备卡

2026-10-01，**DRAFT / NOT_RELEASED**。本页登记准备范围，不是采集或模型运行回执。原 R4 和限额修订保持 INCONCLUSIVE；旧原答、失败和费用不改写。

## 验收对象

继续验收诊断模块的描述保真、必要过程覆盖、未知/冲突表达及实际投递。两任务分别判定，原阈值、原始视觉分母和三条投递轴保持；不恢复作者 A/T，不增加 N.4 作者槽，不启动训练或进入 3-live/P4。当前生产报告候选为 Stair r1 与 Push r2，身份和离线验收见[工程收尾](./results/n5_confirmation_execution_r4/ENGINEERING_CLOSEOUT_ZH.md)。

开发继续复用旧图和原量。正式来源冻结前的工程收敛状态如下：

1. 已统一证据ID字符串的解析及原catalog身份，保留历史严格回放；旧错误路径不自动修复，编号有效不等于事实支持。
2. Stair focus已增加中立算术索引，4项定向检查、6份真实旧包逐值保留及独立区间复算通过；它不预写必要事实答案，也不把未分析完冒充缺观测。root已放行唯一一次旧focus-07干净SOL新视图开发检查，正在运行，结果未知。[索引证据](./results/n5_confirmation_execution_r4/focus_arithmetic_navigation_r1/README.md)。
3. 当前结构审计与全80工程屏障、本人原答/catalog及report三轴入口已完成，8项离线fixture通过；实际旧R4材料40/40接线通过，4个媒体包原11条观察全保留。未生成真实seal/report，不证明必要事实充分或语义通过；实际本人facts加入后的最终容量仍须另核。[控制与真实接线](./results/n5_confirmation_execution_r4/report_preparation_engineering_r1/development_materials/README.md)。

这些修改先在旧资料上验证。不得把旧数据上的后验检查写成新独立资格，也不以旧包模型回答完美作为无限追加调用的前置条件。

## 拟沿用的固定实验边界

- 24 episode、12 focus、4 media；每单位两份独立事实原答。全部 80 facts 完整、按协议核验并分别封存后，才准备或揭示任何 report。报告阶段只提供该角色自己的原答和原 catalog，不互喂、不合并后回传。
- A 路 40 个全新 GPT-6.1-SOL/high 上下文；B 路每任务首个 episode/focus 加全部四个 media 固定为 DeepSeek，其余 32 个为另一全新 GPT 上下文。按实际打包复核任务路由，不能按答案换分配。
- facts/report 合计最多 144 个 SOL 请求、16 个 DeepSeek 请求；另两任务各两次独立 SOL 观察。无应用层自动重试、补抽或择优；底层不可观察的重试如实标明。
- DeepSeek 输出预留/上限为 65536；SOL 沿现有可观察配置和 32768 输出预留，不能冒称 CLI 强制硬 cap。各实际请求重新核容量，不能只改卡片而保留旧 32768 算法。作者窗口与评阅窗口分开。
- 引用问题、调用不完整和描述本身的错误分别记录。原始答案不可修成通过，程序算术不计模型语义票，root 非盲模块票不计独立票。

继承[主计划](../FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md)、[事实先行协议](./N5_FACT_REVIEW_PROTOCOL_ZH.md)及[职责分工](./REVIEW_RESPONSIBILITY_REVISION_ZH.md)；任何规则或来源差异须在新采集揭示前明确冻结。

## 来源、资源与费用

候选 Stair cohort 为 `n5_r5_20261001`，Push eval base seed 为 `1900000000`。已有窄范围历史查询未命中，正式冻结前重新核实历史使用及实际 runner seeds；不把相同 namespace 的重跑当新资料。同 checkpoint、同任务域的新 reset 轨迹不代表新地形泛化。

拟使用服务器 `feedback_p3new_20260929` 下全新的 `n5_confirmation_source_r5_final`、`n5_confirmation_20261001_r5`、`n5_execution_20261001_r5` 三个目录。当前无本轮来源冻结、采集卡或放行。只在完整来源/所需 CPU 检查、实际磁盘及 GPU 状态核实后顺序采集两任务；沿用双视角原图、各 env 自身终点和固定选窗规则。

旧 Stair 图像正在按固定清单迁往本地冷归档，只有逐成员验收后才清理远端对应名字。原答、数值、metadata 和当前 R4 不在清理范围；下载开始不能算空间已恢复。

继承既有 `p3new-20260929-total-api` US$15 账本和全部历史条目，不另开零账本。目前登记费率累计 US$1.120625436；实际放行前重新读取账本并足额预留。Codex 费用不可得，不记为零。

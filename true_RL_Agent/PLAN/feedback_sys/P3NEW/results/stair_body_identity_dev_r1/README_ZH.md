# Stair 四个 calf 定位点：开发输入验收

范围：`development_after_r3`。两包已完成各一次全新 GPT-6.1-SOL/high 观察、机械核对及 root 非盲画面检查。本目录用于同一 R3 样本上的输入修正开发检查，不能作为新的确认集或正式 PASS。根登记的 `probe_card.json` 未修改。

## 开发验收结果（2026-10-01）

- 两个上下文均 `fork_turns=none`，各实际读取 14 个白名单文件、按顺序查看 12 张原图。root 检查了两份完整工具调用输入：各 15 次调用，仅输入读取、12 次原始精度读图、一次 CreateNew 写答；没有继承旧答案、读源码、调用外部模型或重试。见 [访问检查](./desktop_access_audit.json)。
- e1 原答 3 条、e2 原答 2 条，全部保留；重建请求与准备字节相同，机械审计通过。见 [机械回执](./sol_audit_receipt.json)。
- root 非盲模块评阅为 **5 supported、0 contradicted、0 insufficient**，包括原文字及不确定性中的全部命题。重新打开对应关键原图，结合已验证的同帧定位点，未发现左右腿身份反转。描述限于采样时刻的可见轮廓和粗粒度变化，未扩写接触、受力、因果或未观测的连续动作。见 [逐句理由](./root_module_review.json) 与 [评阅范围](./root_module_review_receipt.json)。
- 源头修正的相关测试共 45 项通过（2 项 Stair 投影、26 项任务视觉、8 项姿态事实、4 项测量上下文、5 项共享投影；重复执行不重复计数）。独立源码检查未发现需补改的实质问题；生产共享观察者文件未改变。
- 两包复用了原 R3 样本，因此这五句不能替换原 R3 六句，也不能用 5/5 与旧 4/6 声称因果改善或正式准确率。必要过程召回和新 cohort 的独立资格仍待验证。没有新 DeepSeek、作者或训练调用；桌面模型成本没有 provider 账单，保持未知。

## 真实 CPU 结果

- 远端开发副本为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/stair_body_identity_dev_r1/source`，已有独立副本未再次复制。与冻结 R3 清单逐项核对 485 文件，仅 Stair visual_observation、Go2 body_frame 配置和 prepare_native_observer 三个已审变化不同；冻结 R3 源准备前后均保持原 SHA。
- 使用原 R3 Stair export 和每包原 selection，调用开发副本的 `native.build`。e1 选步为 1/64/89/114/155/310，e2 为 1/63/125/207/232/249；各步原 v0/v1 均保留。
- 完整新旧 request 比较通过：仅首个 public JSON 的 `body_image_points` 改变；其他字段、文本、system 提示、模型模板 envelope、24 张 PNG 字节及顺序完全同值。原 S0、definitions、measured_facts 与 measurement_claims 同值。
- 每张图固定 `FR_calf, FL_calf, RR_calf, RL_calf` 四个具名原点，共 96 点。逐点对比此前独立 `geometry_identity_check/geometry_receipt.json`，96/96 坐标和状态一致，最大像素坐标差为 0。
- 每包白名单恰为 14 文件：`materials.json`、`blind_input_text.json`、12 张 PNG。本地 PNG 从原盲材料复制并前后核 SHA，没有重新下载图片。
- 私有数据的变化仅为新源码路径、frame spec SHA/绑定新增名字、request SHA；精确路径清单见 `cpu_preparation_receipt.json` 的 `private_changed_paths`。

## 接续入口

新输入根：

- `e1/blind_inputs/materials.json`：SHA256 `5097adb4da440ed1a461e90f52ed10a476a50b767382d68ce1f7e1ed7cbaf57b`
- `e2/blind_inputs/materials.json`：SHA256 `8732f404b852a8b02fee81f4ea0040101e6e511c8289a666d09725dbd9a99324`

本轮 fresh 观察者只读取各自 14 文件白名单，以原始精度读图；未接触本 README、控制回执、独立几何、旧原答或另一包。已用完事前登记的每包一次 GPT-6.1-SOL/high，不再重试。本次 CPU 准备本身没有模型、API、GPU、训练或作者调用。

## 来源与范围处理

`prepare_cpu.py` 仅是本次固定两包的 CPU 操作记录，调用原 `sol_observer_control.prepare_packet` 生成材料，未修改该 helper。它在新 carrier 控制回执首次落盘前标记开发范围；没有改写输入提示。原 v2 selection schema 只允许 `n5_confirmation`，因此 `selection.json` 保留原字节，该字段在此仅表示历史来源；新卡和回执均明确整轮属于 `development_after_r3`，`confirmation_admitted=false`。

`request.json` 仅保留在远端新 e1/e2 目录；其 SHA 和完整等价检查在 CPU 回执中。下载只包含 19 个文本/控制文件，见 `control_materials.tar.gz`；`populate_local.py` 完成本地原图复用、白名单核验和独立几何比较。

主要证据：`source_receipt.json`、`development_source_manifest.json`、`cpu_preparation_receipt.json`、`local_preparation_receipt.json`。投影只提供具名原点的图像坐标，不证明可见、接触、形态结论或模型改进。

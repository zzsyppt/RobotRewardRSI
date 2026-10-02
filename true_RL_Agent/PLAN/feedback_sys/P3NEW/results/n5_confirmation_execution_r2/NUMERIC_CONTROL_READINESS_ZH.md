# R2 数值 CPU 后处理准备

状态更新：数值八步已由 root 实际运行完成，全部 exit 0；24 回合算术零差异，72 份全文计量已完成，范围与未核项见 `numeric_readonly_review.json`。准备时的语法/来源 hash 记录仍保留在 `prepare_numeric_control.preparation.json`，不是实际执行回执。4 个媒体单位后来发现并修正了生成器的 v1/v2 合同接线，真实完成记录见 `postcapture_media_fix/README_ZH.md`。没有模型调用或语义/实际容量放行。

## 可执行控制

`prepare_numeric_control.py` 是一次性顺序命令包装，复用已冻结的导出、定义、打包入口及已有算术/计量脚本；没有新增诊断规则。脚本 SHA-256 为 `dad01a248c4b799686b85f5147a514f6b7c3315cc949d417c04de9faed213c2f`。由 root 核实并上传同名字节至服务器 `n5_execution_20260930_r2/` 后，在两任务采集完成时执行：

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python -B \
  /data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r2/prepare_numeric_control.py
```

脚本先要求 Stair/Push 都有真实 exit 0、wrapper `PASS` 和本轮 phase binding；核 card、source manifest、四个入口和两个控制 helper 的实际 hash。随后按既有参数运行八步：原量导出；Stair 定义；三个 Push 定义；数值 pack；独立算术；完整文本计量。定义放 `C/numeric_task_definitions/`，不会预先创建 pack 目录。

`prepare_confirmation_materials.py` **自身已为 36 个数值单位生成两评委共 72 份 facts prepare**，无需再次 prepare。这里不调用报告容量 fixture、不产生 report 请求或 facts seal。`C/postcapture/numeric/` 记录实际 argv、UTC、退出码、stdout/stderr 和最终回执；任何步骤失败均停止下游并保留输出，不自动重跑覆盖。

输出沿原清单：`CAP/numeric_raw/`、`CAP/numeric_materials/`、`C/numeric_task_definitions/`、`C/control/numeric_arithmetic/` 和 `C/postcapture/numeric/prepared_text_counts.json`。其中 `CAP=/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20260930_r2`、`C` 为同父目录的 `n5_execution_20260930_r2`。

独立算术仅覆盖既定 24 个 episode；12 个 focus 没有新独立复算器。Push 档案操作数到物理构造的独立来源核验仍未完成。全文计量使用冻结 DeepSeek tokenizer，SOL 仍是代理，不包含图像 token、聊天模板/运行时开销，不是实际容量放行。正式 facts 发送仍须等待全部原始观察的 root 模块核对封存；双方全部 facts 封存后才允许 report。

## 四个媒体单位的明确边界

媒体包在两任务固定选择完成后独立生成。本轮原冻结生成器首次执行暴露 v1/v2 合同接线错误，后续使用 `postcapture_media_fix` 登记的唯一修正来源与新输出路径，实际命令和回执见该目录 README；不能再运行原失败入口。输入形态不变：`purpose=confirmation`，含真实 `capture_card`、两份 `selections` 的 `{file,sha256}`，及四个从选择文件读取真实 run/env 的 units。每项为 `{id,export,run_index,env_index,full:{file,sha256}}`，不得带历史 `observer_directory` 或 `duplicate_of`。该分支实际生成 4×2 份离线 facts，后续 `count_prepared_text.py` 已完成它们的全文计量；72 数值 + 8 媒体共 80 份已计量，图像 token 仍未测，静态计量不证明实际模型完成。

静态核对确认以下接口差异。root 已决定沿冻结边界分别安排 facts 测量定义和 report 独立任务合同；具体实际引用映射仍须在正式发送前封存，本文件不是已执行或已放行凭据：

- 媒体 `related_numeric.definitions` 原样继承原量导出定义。Stair 含 outcome signals、坐标、身体角色、时钟、单位、接触历史、几何、采样限制；Push 含 capture_point、time、pose/conditions/planar_motion 字段顺序、参数、坐标、event/dwell 时序和缺接触说明。`clock` 保持 1..T、stride=1；未供字段明确列出。配合 `media.json` 的 S0、相机和实际图像步，足以开展单回合原始结局、主要关系/关键量、视觉与数值采样范围的事实评阅。组级频率和跨回合反例不能从单媒体单位推断，这是其冻结 scope 的限制。
- 这些测量定义**不含候选修改权限及 reward ABI**。冻结媒体 facts 生成器没有 `task_definition` 参数，实际 materials 也未附该角色；不能把 `full_episode.json`、报告文本、数值一致或 hash 当成独立权限来源。
- 独立权限与输入字段操作数来自 `prepare_review_task_contract.py`：Stair 的匹配原始任务 protocol 中 `candidate_authority` 和绑定 reward schema；Push 的 task-owned candidate surface / runtime adapter 摘录和 resolved-cell 绑定 reward ABI。协议第 110、123 行要求正式包提供独立 `task_definition`。这里的媒体 facts 五束仅建立物理事实、原始结局和可回答范围，不要求它在未见报告前评候选修改权限。
- **root 采用的阶段安排**：媒体 facts 保留原入口输出；report 使用既有 `prepare_report_manifest` 追加该回合对应 task/run 的独立 `task_definition`，原 facts 全部材料保留，并另附本人真实封存 facts。不会向 facts 偷加自动报告，不从另一包/另一评委继承权限答案。媒体 facts 未核权限必须原样记未核；report 若独立操作数仍不足，记 unsupported/unanswerable，不补成 supported。

准确对应不按包序或 e1/e2 猜测：媒体生成后的 `<unit>/raw/source_numeric.json` 含其真实原始 numeric manifest 的 `file/sha256` 与 `identity`；以 `(receipt.units[].task, source_numeric.manifest.sha256)` 在四份定义的 `(contract.task, contract.numeric_manifest.sha256)` 中唯一匹配。Stair 三个 run 共享一份数值 manifest，因此都对应 `stair.json`；Push 必须同时核 `receipt.units[].source_run`、原 manifest 的 `identity.run_index` 与该定义的实际来源，分别对应真实 `push_run01/02/03`，不能仅看数字相近就套用。追加引用使用已核原件的绝对 `file`、真实 SHA、`role=task_definition` 和唯一 id；保留来源 JSON 内的原字段。源原件 hash 和 run/binding 是否一致仍由实际材料生成/核验回执证明。

上述结论来自 frozen source 中 `prepare_confirmation_media.py::numeric_episode/build_unit/prepare_confirmation`、`export_confirmation_raw.py::stair_definitions/push_definitions`、`prepare_confirmation_materials.py::related_episode/_prepare_task_definition_inputs/prepare_report_manifest` 和事实协议；不是读取模型评分后的调整。

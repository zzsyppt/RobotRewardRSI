# 评阅材料修正：区分诊断观测与奖励输入

2026-09-30。仅开发材料与 CPU 验证，无新模型请求、无新确认数据，不改历史回答或评分。

上一版任务合同携带采集字段、坐标和作者权限，却没有完整携带奖励输入 ABI。评阅者因此缺少判断“作者代码能否读取该量”的独立依据。Push 的源码摘录还在 V2 覆盖之前结束，可能把旧 V1 字段说明误作当前边界。这是材料缺口，不能直接当作被评诊断的错误。

修正在原生成入口 `prepare_review_task_contract.py`：Stair 从原数值档案内的 resolved-cell manifest 重建输入 projection，并逐项匹配原 runtime projection；Push 使用单独保存的原 resolved-cell manifest，先核对它与 numeric identity 的 manifest hash，再核对 V2 奖励字段 schema hash。Push 未提供绑定 manifest 时明确记 `not_supplied`，不根据任务名称猜测。源码摘录同时包含当前 V2 覆盖和输入字段定义。

本次 Stair 39 个奖励字段，schema hash `cf1d7295eba9decc1ad778d699153a3ea212bf762306212a91a9a9c5c9ebda93`；Push 22 个，schema hash `8e481be6ec32f253739427e08c113d6bba81f9aa6bda63ffc77af3fb602e675b`。诊断可观测字段与奖励可用字段分别列出，不能互相替代；字段可读也不等于作者有权修改终止或任务成功条件。

[verification.json](./verification.json)记录 6 项检查：两份绑定合同重算相同；两任务的伪造奖励字段均被拒绝；不同 resolved-cell manifest 被拒绝；Push 缺 manifest 明确留下缺口。生成脚本原字节保存在本目录，旧 r1 脚本和材料原样保留。

此结果只证明上述合同绑定与材料修复。当前源码 hash 不能证明历史运行执行了相同源码；正式新资料仍须绑定实际执行快照。没有补判旧语义支持率，也没有宣称 N.5 通过。

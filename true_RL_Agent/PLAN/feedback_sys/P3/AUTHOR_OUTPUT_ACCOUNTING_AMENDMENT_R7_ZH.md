# P3 作者实验：请求输出上限与实际计量的 r7 修订

这是实验启动后的透明修订，不能把它描述成原合同始终完全合规。原 48 项顺序、12 个焦点、A/T 输入、模型、请求参数及每项 5 次上限不变；已发生的失败、协议偏差、次数和费用全部保留。

## 已发生的事实

r6 完成 45 个作者终态后停止。在 `push-04 / T / repetition 1`，`generation-023-lane-b` 的 attempt 1 请求 `max_tokens=32768`，服务商回执报告 output/reasoning units 均为 32769，finish_reason 为 length，没有最终文本。输入 16262 units，完整实际 usage 费用为 USD 0.04194348。现有回执不能解释多出的 unit，不能自行归因为 EOS，也不能把它改成 32768。

随后 attempt 2 的 prompt 已持久化，预算守卫在进入 provider delegate 前因历史计量越界拒绝调用。原账本保留该槽，仍按输入 40000、输出 32768 全额预留，不改写为零、不退回次数。停止时共有 87 个账本槽：85 次权威 usage、此前 HTTP 402 的未知 usage、这次已有 prompt 的中断槽。当前已知费用 USD 2.538240096，两个保守预留 USD 0.1026432，总计 USD 2.640883296。这是预登记峰价下的计量或预留，并非服务商实付账单。

完整现场见 [停止报告](results/author_cap_stop_r6.json) 和 [848 文件封存清单](results/author_r6_cap_stopped_snapshot.json)。原 r5/r6 日志、源码、冻结和停止现场不覆盖。

## 修订边界

生产预算守卫区分请求控制与服务商事后计量：请求仍传 32768；不再因历史单次 output 或由它派生的 total 超过请求预留而永久拒绝后续请求。历史实际 usage 原值进入会话累计账本，输入 40000 的准入边界、全局 input/output/total units 和 attempt 上限继续生效。没有固定加 1、舍入、截断或服务商特例。

P3 harness 将已知输入超预留继续列为硬 violations；实际输出大于请求值则列为 `protocol_deviations`，逐条包含 round、sequence、requested/actual/excess、finish_reason 和响应模型。summary 同时保留整体及 task×arm 的偏差。完成所有作者项只表示执行终态齐备，不能消除这些偏差。

恢复启用已有的 `allow_conservative_interrupted_recovery`，按原预留计入无回执槽，不增加免费重试。受影响 Push T episode 已用 2/5 个槽，只剩 3 个；其它完成的 45 项跳过。原 87 个槽全部保留，旧 HTTP 402 也保留。后续 A/T 比较须显式说明 Stair A 的服务中断与 Push T 的输出计量偏差和中断预留，不能把这些成本差异全归因于反馈形式。

## 冻结与交付要求

r7 使用独立 `source_r7`、`tools_r7`、review/API freeze 和第三段日志；采集仍注明 Stair r5、Push r3，不重跑仿真。恢复前验证实际 24 首包与 r5/r6 逐字节相同，原 case、stage 和 48 项顺序相同，仅登记的生产预算文件及两份 harness 文件变化。根代理审读新冻结后才释放 API。

上述验证已完成：`results/author_recovery_equivalence_r7.json` 记录 24 首包、原 45 个结果 hash、stage/case/schedule 全部相同；source r6→r7 仅 `generation_provider_budget.py`，两份 harness diff 单独保留。生产闭包为 `5686469111532083501de144d7a6a5ac4965a8175d6bccacc0b0de47eb27a2aa`，review freeze SHA 为 `aad0bb34fd97d1d17890e65b63f6419e09563a1a2dcb98b7c69e2f8864bd4306`。根代理独立复核记录为 `results/author_recovery_root_review_r7.json`。随后生成独立 API freeze（SHA `3a396e7902add53235b0c2bea7b4aeabde3f7c3c5b99dfb3ca1be091fb0d443c`）并仅启动一次第三段；已有 45 个结果不重跑。

恢复后继续交付全部 48 个作者终态及 53 个匿名评阅项（含失败与 5 个重复复阅项）；人工评分保持空白，价值门保持待评。静态/CPU 通过不是事实支持、诊断有效或 RL 改进的证明。

## 第三段实际停止状态

第三段执行 4:29.17 后退出 1。受 output 计量偏差影响的 Push T repetition 1 在 attempt 3 完成候选检查，原中断 attempt 2 继续占用次数和全额预留；Push T repetition 2 也已完成。最后的 `push-04 / A / repetition 2` 在 attempt 1 得到 HTTP 402，无 usage，原服务恢复机制正确停止。没有修改模型、密钥、请求量或预算来绕过服务拒绝。

第三段停止时为 47/48 作者终态、91 账本槽，最后一项已用 1/5 次，剩 4 次。88 次已知 usage 按峰价计 USD 2.597483856，三份未知预留共 USD 0.1539648，总计 USD 2.751448656。新增 HTTP 402 同样保留在 A 臂，不删除或写零。旧 output 偏差仍在 `protocol_deviations`，没有硬 violations。服务恢复并经根代理重新放行前保持停止。

完整证据见 `results/author_service_stop_r7.json`、`results/author_r7_service_stopped_snapshot.zip` 及其清单/解压目录。ZIP SHA 为 `81c8636215eb13ef7255fbd1ee97e486fc07a711f2df96fc5ef1dc01306e5bf9`，936 个文件已在本地逐 hash 验证。原 r5、r6 停止现场均保留。

## 服务恢复后的完成状态

用户确认服务恢复后，核验原 r7 source/tools/API freeze 未变且无旧进程，单次启动新日志段 `author_run_r7_recovery2`，未新增源码版本、重跑首包或重跑已有 47 项。最后的 Push A 在原 attempt 2 完成静态及 CPU 检查，日志段耗时 1:35.38、退出 0；完整 48 项终态及 53 项匿名材料现已交付。

最终 92 槽含 89 次已知 usage、2 次 HTTP 402 和 1 次原中断预留；已知峰价费用 USD 2.618139576，未知预留 USD 0.1539648，总计 USD 2.772104376。输出计量偏差继续显式记录，不能声称原合同完全合规，也不能将三个服务/预算中断槽从臂比较中删除。原 47 份终态文件 hash 均未改变。

最终材料见 `results/author_final_bundle.json` 和 `results/author_final_bundle_local_verification.json`：1080 个文件本地逐 hash 通过，包含完整账本、53 项匿名包与分离私有映射。机器及人工边界见 [执行口径](author_protocol_ZH.md)；人工价值门仍待评，没有执行候选训练。

# P3 作者实验：HTTP 402 后的有限恢复修订

本记录保留原实验及全部不利结果。没有新增案例、替换作者输出或重新开始计费。

第一运行段 `author_run_r5` 已正常落下退出码 1，异常为 `ProviderServiceUnavailableError (HTTP 402)`。完成 9 个作者终态，共占用 16 次 attempt；15 次 usage 已知，另 1 次是明确 HTTP 402 服务拒绝但没有 usage。原回执没有保存“余额不足”响应正文，不能把没有 usage 写成零消耗。用户随后确认充值。

已知 usage 按预登记峰价折算 USD 0.500479548；HTTP 402 保守预留 40,000 输入 / 32,768 输出，即 USD 0.0513216。当前总账 USD 0.551801148。该数字不是服务商账单实付。

受影响位置是 `stair-01 / A / repetition 1`，目录 `p3-at-development-20260924-generation-005/lane-a`。其 attempt 1 为已知 usage 的 `reasoning_output_exhausted`，attempt 2 为上述 HTTP 402。恢复后仍是原 episode，**最多 5 次，已用 2 次，只剩 3 次**。服务噪声保留在该案例、次数和费用中，不能从 A/T 分母或费用中删除；解释这个配对的结果时必须说明该干扰，不能把差异全归因于反馈形式。

旧生产预算守卫对 `completed` 且缺 usage 的尝试一律停止。原恢复开关只处理 `interrupted`，不能解决这个已封存的服务拒绝。r6 的修正仅涉及 `provider_errors.py` 与 `generation_provider_budget.py`：复用一个服务恢复判据；对确定的 completed HTTP 401/402/403 服务拒绝，按原 caps 全额预留并保留原次数。普通缺 usage、未知超时仍停止。没有改写 journal、补写假 usage 或放开免费重试。

r5 与 r6 的生产源快照分别为 `79afb56d72409d0829f88ba0142361661c4307fcebd4b91522126636c1d90db1` 和 `0ff923abb96f1fa2858bb453d745bab0e7b7a133045b799fe2131e1306cedd55`。原 stage、12 个案例、48 个 episode 顺序、模型参数、修复规则、5 次上限、5 份实验脚本及公共证据原件保持不变。r6 已重新预检 24 个首包并与 r5 canonical prompt 逐一比较，全部逐字节相同，结果见 `results/author_recovery_equivalence_r6.json`；根代理独立复核后释放恢复，记录在 `results/author_recovery_root_review_r6.json`。受影响的 Stair A episode 随后在第 4 次 attempt 得到静态及 CPU 检查通过的候选，原 9 份结果 hash 未变，见 `results/author_recovery_observed_r6.json`。

恢复使用原 `author_run_r5` 账本和结果目录，跳过已完成的 9 个终态，从现有 continuation 继续。新运行段日志、耗时、退出码单独写 `author_run_r6.log/.time/.exit`，不覆盖第一段。Stair 数值采集仍来自真实 r5 producer，Push 来自原 r3 producer；这次纯预算恢复不重跑仿真，也不把旧证据改标为 r6 采集。

中止现场完整保存在 `results/author_r5_stopped_snapshot.zip` 及同名 manifest / 解压目录：147 个文件逐 hash 验证，ZIP SHA-256 为 `2096e026de554460cfee9a936badd3bf7125ec04b7b2d488b2a58cc7755d369f`。旧源快照、旧 freeze 和旧日志保留。最终依然交付完整 48 个作者终态及 53 个匿名评阅项，人工评分与价值门保持待评。

恢复段后来在 45 个终态处因另一独立事件停止：Push T 的服务商实际输出计量 32,769 超过冻结 32,768，原预算守卫拒绝后续调用，未套用 HTTP 服务拒绝恢复。当时完整 48 项尚未完成；停止时证据见 `results/author_cap_stop_r6.json`，第二次停止现场单独保存在 `results/author_r6_cap_stopped_snapshot.zip`，不覆盖 HTTP 402 现场。后续 r7 透明修订及最终 48/53 项交付另见 [r7 记录](AUTHOR_OUTPUT_ACCOUNTING_AMENDMENT_R7_ZH.md)，不回写或消除本记录中的中断。

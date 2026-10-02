# r2 开发校准原始结果

记录时间：2026-09-29。职责：封存与机械比较；本记录不承担第三位评阅者或裁决者角色。

## 结果及范围

两份完整原答均先保存并计算 SHA-256，再逐项比较固定的全部 20 个条目。**原始 verdict 一致率为 18/20 = 90%，达到冻结的最低 90% 门槛；kind 一致率为 19/20 = 95%。** 没有补评、重试、针对性改分或裁决后的替代分数。

这仅表示本次旧开发材料的评阅一致率门通过，不是新确认实验、科学资格认证或 20 项判断全部正确的证明。本次记录工作没有查看或揭示任何 N5 新确认资料，也没有执行训练、评估或新 API 请求。

## r1 与 r2 的关系

r1 原始 verdict 一致率 **14/20 = 70%**、kind 一致率 **16/20 = 80%** 保留于 `../review_calibration_r1/raw_comparison.json`，没有被覆盖。

r2 对相同的 20 个旧开发摘录使用一次预先冻结的通用规则修订，并由两位新上下文评阅者重新评阅全部条目。修订覆盖：合取事实与所断言范围、限制与错误的区分、未运行预测/机制的标签、策略输入与作者历史反馈的受众/时序区别，以及两个程序的同类公式算术补充。算术网格是合成状态，不是实测轨迹或预期判决。r2 原答未获提供 r1 分数或另一评阅者答案。

冻结 manifest 明确只允许本次 r2 修订；若仍低于 90%，应审计证据与语义分歧，不能持续重评直到通过。本次结果恰好达到 90%，剩余分歧仍全部保留。

## 原始分歧

| 条目 | DeepSeek 原始 kind / verdict | GPT 原始 kind / verdict | 两份原答理由的差别摘要 |
| --- | --- | --- | --- |
| sample-x-10 | fact / supported | fact / contradicted | DeepSeek 引用策略当前步 push 信号和因果归因限制；GPT 指出作者历史反馈包含事件/回合分母，并认为原句把受众与限制混在一起。 |
| sample-y-09 | fact / supported | fact / insufficient_evidence | DeepSeek 采用 roll=0 的公式特例；GPT 区分 pitch 项与含 roll 项的完整奖励率，并指出峰值 pitch 时同步 roll 未提供。 |
| sample-y-10 | prediction / not_a_factual_claim | hypothesis / not_a_factual_claim | verdict 相同；对未运行机制的证伪条件应归为 prediction 还是 hypothesis，kind 不同。 |

以上仅摘要两份原答已给出的理由，不是重新审理或判定哪位评阅者正确。verdict 分歧 ID 为 `sample-x-10`、`sample-y-09`；kind 分歧 ID 为 `sample-y-10`。全部 20 项原始标签与布尔相等结果见 `raw_comparison.json`。

## 封存、传输与费用

- DeepSeek 原始回执远端已完成。随后从 `message.content` 精确取 UTF-8 字节，未附加换行，生成原答与封存卡；三件文件复制到本目录后与远端 SHA-256 逐一核对。
- `transfer_verification.json` 记录传输 hash、release 对 manifest/request 的冻结 hash、全部 allowlist 材料 hash、请求中的材料对象与本地材料一致，以及回执 content 与原答字节一致。
- GPT 注册记录保留原样；其注册时的 `original_answers_sealed_before_comparison: false` 是历史状态。完成时封存证据为新增的 `gpt6_sol_seal.json`，比较时间晚于两份原答封存时间。
- DeepSeek：`deepseek-flash`，回执模型与指纹匹配，1 次调用、0 自动重试，finish reason 为 `stop`。prompt 31,175 token（cache miss 31,175；cache hit 0），completion 22,233 token。
- 依冻结单价和实际 usage 计算的 API 费用上限为 **$0.0360321**，不是账单实付；release 中的 $0.0693216 是预留上限。GPT 后台实际指纹、token 和费用不可得，费用不能记为零。

| 原始文件 | SHA-256 |
| --- | --- |
| `gpt6_sol_response.json` | `817ce9c65cb1e7b04d7bc3f0e925a98416515dda7f290a03c9f76dcb7be0ffbc` |
| `deepseek_response.txt` | `2cadb73aa85db7b1aceda70d6773ba57f922c3711ccf913c89158901f4538c85` |
| `deepseek_receipt.json` | `a94e27935245fbe6eeb08dab0f16975ad80f03315b942e2149e489fdb01b5d16` |

机械处理代码为本目录 `seal_and_compare.py`，仅使用标准库和本次 r2 已完成材料；写入采用排他创建，已有封存/比较文件不会被覆盖。脚本完成后再次验证原答、回执、封存卡、manifest 和 request 的 hash。生产代码、总计划及原始材料/答案均未修改。

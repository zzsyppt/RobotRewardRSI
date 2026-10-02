# Push SOL 双包 combined audit：只读装配审计

2026-09-30。只检查冻结源码、冻结 N5 卡与旧 r12；未读新 N5 数据，未调用模型/API/GPU/SSH，未修改生产源码、445 文件源码树或原封存回执。本记录不声称新 N5 材料已经生成或验证。

结论：可以沿原冻结 e1/e2 顺序，将两个独立 SOL 原答的机械审计装配为 `aggregate_report` 相同的核心结构，再由 root 对 combined audit 全量一次复核。需要的是一次操作层数据装配；不需更改研究规则或生产源码。没有现成的 SOL 双包 CLI 可直接运行。

## 可复用入口与边界

- `PLAN/feedback_sys/P3NEW/prepare_native_observer.py:112` 的 `build(export, selection, episode_alias=...)` 构造每包真实 `windows` / `numeric_check`。分别使用 e1、e2，保留每包原图、S0、定义、alignment、private sidecar 与封存选择绑定。该生成器仍将 HTTP 模板标成 DeepSeek，不能把这些模板字段当成 SOL 实际传输设置。
- `src/true_rl_agent/evaluation/visual_observer.py:430` 的 `audit_observer_output(original_json, windows, numeric_check=...)` 可直接审计原 SOL JSON，无须合成 `choices` 或 provider receipt。审计保留原观察及独立 numeric_check；不会改写原句。
- `PLAN/feedback_sys/P3NEW/send_native_observer.py:229` 展示双包装配规则，但不能直接调用它处理 SOL：`rebuild_report` 强制 DS model/fingerprint/release/预留，`audit_response` 强制 HTTP `response.choices[0].message.content`。不得伪造这些输入来使原函数通过。
- `src/true_rl_agent/evaluation/visual_review.py:21,66` 的 `visual_review_binding` / `attach_visual_review` 接受上述核心结构；必须每条 observation 有且仅有一票。随后走 `tasks/push_recovery/visual_observation.py:331` 的 `attach_push_visual_observations`，再 `project_push_process_behavior`。额外 calls/provenance 字段不会阻断这条链。
- 当前冻结 `visual_observer.py` 没有 desktop carrier receipt 专用构造/验证入口。实际桌面合同存在于 r12 的操作材料，而非该生产模块。

## 最小装配约束

1. 先按冻结 report/selection 核验唯一两包，顺序严格 `[e1,e2]`。每包原始回答字节另存及 SHA 封存；以原 JSON 做机械审计，并验证审计 raw_output 等于解析后的原答。不要将错误 alias 自动改为预期 alias；缺包、解析失败、身份/图像/输入不匹配都保留失败，不补答、不重试。
2. `calls[0]`、`calls[1]` 保留各自完整 audit、audit SHA、原答 SHA、真实 access receipt SHA、实际载体材料清单 SHA、registration/agent 身份。DS prepared request SHA 只标为输入模板来源；实际 SOL 文本/图片清单单独绑定。没有 provider raw_response 就用 null/缺失及解释，不能造一个 HTTP 回执。
3. `raw_output={"calls":[deepcopy(audit_e1.raw_output),deepcopy(audit_e2.raw_output)]}`；按 e1 然后 e2 顺序遍历全部 observations，只给复制出的记录增加 `source_call_index=i`，将 `reference` 前缀改成 `calls[i].`。除此之外 raw_observation、alias、image_evidence、media_identity、numeric_status、numeric_check、violations 等逐字段完全相同。不得只合并 supported/机械通过的句子。
4. 顶层保留 `schema_version=true_rl_agent.visual_observer.v3`、`scope=n5_confirmation`、冻结 report SHA、calls、raw_output、observations、violations、`semantic_support=not_evaluated`、`all_raw_observations_retained=true`。逐包容器/接收/来源错误以 `calls[i]:...` 加前缀；`passed` 仅当无这些错误且所有原 observation violations 为空。未知桌面 fingerprint/usage/cost 如实缺失；不套 DS `eligible_for_confirmation` 判据，不将 Push 记成 DS $0.10，也不把未知费用记为零。调用数/预留字段只能据真实登记和执行记录填入，并明确为桌面载体；核心结构同形不要求伪造 DS 传输字段。
5. root 查看两包实际收到的全部原图、S0/定义、原观察及 uncertainty、机械/数值审计，对 combined 所有 observation 逐项决定；不改原句、不改数值判断。以 combined 的 `visual_review_binding` 生成一次完整 review 后 `attach_visual_review`。该 binding 只绑定 schema/raw_output/observations（排除局部 violations），不绑定 calls 元数据，所以附件另登记整个 combined JSON 文件 SHA 及每包真实载体证据 SHA。所有模块复核附件按现冻结顺序在任何正式 facts 请求前封存。

## r12 现有事实

`results/model_carrier_probe_r12` 只有一个 e1 包、1 window、3 observations；不是现成的双包例子。

- `output/access_receipt.json`：schema `p3new.observer_access_receipt.v1`；SHA `d4521e2dc546b884181cc2e21d398a9196f18769cbb2adcfdd039e8c2ba13d0f`。绑定材料清单、文字 + 12 原图 SHA、逐图顺序/step/view/1920x1080/original 与完成声明。
- `registration.json`：请求 gpt-6-sol/high/fork none。实际服务端版本、fingerprint、usage 等仍 unknown；盲输入里的 deepseek-flash 仅是源模板声明。
- `control/offline_audit_receipt.json`：原答 SHA `2f8e1109135ed806e8c39c605922bac6224e9db2f35ea3b7899d5ffec174004d`；机械审计 SHA `232fa635f333d251b139ab4e4b6b6941964727fff6167c353ec19c49fb9fa0d2`。
- `control/verify_reviewed_projection.py:92-106,144` 已使用真实 SOL 原答 → 原机械审计 → supplied root review → 原生产投影，明确没有合成 provider request/response receipt。它证明统一消费者不依赖 DS HTTP 回执；不能整段直接重跑其硬编码 r12 路径/哈希。

此次只比较以下 5 个当前文件与独立 postprocess/source 快照，均字节同 SHA：

| 文件 | SHA-256 |
|---|---|
| send_native_observer.py | 1c9cb29e9f9e4eb94259ab4921ccb08563770d182f45d728ac8377800ac1a24b |
| prepare_native_observer.py | 26cab38d3759d5d11c1d58050011099b12d039cf2ccfba0664bc87bbea321098 |
| evaluation/visual_observer.py | b02ed4817e06efb6cf73786a9c18f511284239d7d0a4bc0d85b8b7d0dbb6e030 |
| evaluation/visual_review.py | 334aad85f623616d0c6566cff738298bc8513282ac210b2e4b15dfc70150dff3 |
| tasks/push_recovery/visual_observation.py | 09adffc8e4367543c12d0b2528ca92b5e22a3f57a3b367ed7bd2cfa74110f636 |

未决项仅是未来实际 e1/e2 原答及其真实访问/输入/agent 绑定、一次离线装配及 root 全量复核；本次未生成这些新证据。

# P5 复审实施与验收

2026-09-11，第一次复审记录。本文对应训练后拒绝终态恢复、历史反馈读取和上下文保留的上一批验收；它没有覆盖后续发现的冻结正式 CLI 回放问题。最新状态以 [ACCEPTANCE_ZH.md](ACCEPTANCE_ZH.md) 和 [历史回放复审](REPLAY_CONTEXT_FOLLOWUP_ZH.md) 为准。以下保留当时证据及其覆盖范围。

## 实施顺序与边界

1. 先用现有双 lane 生产夹具复现训练后拒绝、代回执发布前中断的恢复问题。由持久化 trial 判决恢复终态，避免重复训练、任务门和终结；同时保留已有评测阶段恢复。
2. 统一历史选择为最近有训练证据的未接受 trial；源码、运行反馈、详细公开反馈使用同一选择。派生的 prompt 反馈在读取时生成，旧科学证据保持原样。用两次真实实验及旧 P4 Genesis 验证。
3. 用紧凑机制结果替换冗长 lane 状态副本，所有当前 prompt 子预算按实际发送格式计量。上限不加大，保留必要跨代结果和明确省略计数，历史封存字节的读取另按原版本处理。
4. 合成 smoke 的观察恒零不再触发修复；训练后真实差分负责作用判断。完善参数归一化对比较严格性的处理，验证 F62 原始源码。删除无用修复分支与无消费者的 continuation 迁移；双就绪执行约束下恢复 authoring 失败即停止，推理耗尽后的关闭状态在本 lane 内保持。
5. 用真实晚代、修复和源码请求帧替换不充分的容量证据；只做必要的行为回归、真实历史核对和最终整体验收。复审额外发现按同一所有者修复，不扩展新框架、评分规则或物理门。

机器人事实仍属于 profile，任务物理与反馈含义属于 task，公共层只处理生命周期、通用结果与字节预算。禁止改地形、reference、接受阈值、正式训练预算；完整实验与 checkpoint 继续保护。

## 逐项处置

代码、直接证据与最终两端完整回归均已通过，验收范围及交付核对以 `ACCEPTANCE_ZH.md` 为准。

| 审计项 | 核实、源头修复与边界 | 直接证据 |
|---|---|---|
| H1 终态重复 finalize | 实际复现。恢复以最终 trial 回执为准；已训练但没有 public-dev 的终态直接恢复判决，不再运行 task gate。保留后续 eval 阶段原有恢复路径。 | `audit_h1_before.log` 复现原异常；`audit_h1_after.log`；fixed-width 生产夹具在发布前中断、恢复、重复执行时不增加 train/gate/eval 次数。 |
| H2 跨代记忆丢失 | 成立。冗长 lanes 改为每 trial 一行机制结果；代/lane、候选、操作、项名、public-dev 目标/守门 delta、状态及 runtime 作用保留。先压缩重复详情，再考虑省略代；14 KiB/16 代上限不增加。 | 两场真实历史重建，见下表；新增 15 代/30 trial 的保留与幂等回归。 |
| M1 合成观察触发无效修复 | 方向成立。取消观察全零/全截断引发的额外 provider 调用，仅保存观察；真正执行错误仍修复。raw 等于 clip 边界不算截断。真实训练后差分门不扩展。 | `audit_candidate_history.json`：37 个已训练候选仅 1 个在旧训练采样中恒零；当前夹具有 22 个全零/全截断，其中包括两个 accepted 候选。 |
| M2 字节双口径 | 成立。新 writer、provider、总量和所有子预算都按 compact UTF-8 计量。历史封存格式只在 reader 验证。 | 所有历史重建帧 memory 的 wire/metered 字节相等；`audit_receipt_formats.log` 74 项通过。 |
| M3 最新未接受选择不一致 | 成立。唯一 selector 跨过未训练代，在已接受代内仍可选未接受 sibling；源码、runtime、详细反馈共用选定 trial。选定后缺失某种证据时不借别人的数据。 | c6 G7→G5-B、G9→G8-B、G13/G15→G12-B；c8 G9→G8-B，三处身份相同。 |
| M4 旧派生 feedback 相等性 | 成立。active unified 新结果不再存 prompt_feedback；prompt 从 public-dev 和冻结 family mask 派生。原 Genesis/ledger 不重写。pre-unified 的历史接口仍有现存消费者，保留其原投影。 | 两次正式历史与 P4 G1 Genesis 均可直接构造当前 prompt；旧反馈被改变/移除时公开原始数据仍决定新视图。 |
| M5 F62 真例放行 | 成立。仅在准入参数族比较中合并同方向严格/非严格比较；不更改已封存结构/delta hash，方向反转仍是不同结构。 | 原 c6 G9-B 相对 G8-A 已返回 `numeric_only_reward_change_requires_structural_mechanism`；37 份源码在同一 Python 与进场分析器对照完全一致。 |
| M6 晚代容量证据不实 | 成立。退役初代+预检冒充晚代的构造脚本，改为直接测量真实历史重建帧。 | 当前第 13 代 114943 B / 37367 输入 token；见下方四帧实际 provider usage。 |
| exact-repeat 修复被覆盖 | 删除覆盖路径；已有专用阻断反馈优先保留。 | authoring loop 回归。 |
| A 失败后无收益的 B authoring | 删除独立 authoring 分支；A 失败时 B 为 not_started，保留双 ready 才训练。没有引入单 lane 科学选择。 | fixed-width v3 回归核对 B 未调用、没有执行和可重放提交。 |
| adaptive 思考反复开启 | 从已封存 attempts 派生本 lane 是否耗尽过推理；显式启用降级后，后续重试及恢复持续关闭。未新建持久化状态。 | `audit_reasoning_resume.log`：推理耗尽→verifier 失败→中断→恢复，后续请求持续 disabled；未授权此策略时仍按原配置。 |
| continuation v1、烟测噪声 | 删除无活动消费者的 v1→v2 恢复翻译；失败反馈只留出错项，不重复整个样例输出。历史封存 artifact 仍可读取。 | authoring、artifact、实际旧 prompt 验证。 |
| 未验证的供应商 context 错误码 | 删除猜测码表；HTTP 400 保留真实状态并终态处理。未制造百万 token 请求来验证小概率错误分支。 | provider HTTP 错误及显式容量合同回归。 |

“旧训练采样中恒零”是对已有采样记录的陈述，不等同于重新证明所有训练 transition 恒零。当前 22 个合成观察也不是原审计旧夹具 20 个的复算；两者配方不同，结论共同支持观察不能代替真实训练证据。

## 真实历史与容量

`c6` 为 `formal_go2_stair_15round_20260901_c6ab8ba`，`c8` 为 `formal_go2_stair_15round_20260901_83824cd`。原始 Genesis、ledger、source 均保持原样；前后结果分别存于 `audit_history_before_*` / `audit_history_after_*`。

| 上下文 | 复审前保留代 | 当前保留代 | 当前 trial 行数 | memory 实发/计量 B | 总 prompt B |
|---|---|---|---:|---:|---:|
| c6 G7 | 2–6 | 1–6 | 12 | 8696 / 8696 | 112455 |
| c6 G9 | 6–8 | 1–8 | 16 | 13615 / 13615 | 105707 |
| c6 G13 | 8、12 | 1–12 | 24 | 13990 / 13990 | 114943 |
| c6 G15 | 8、13、14 | 1–14 | 28 | 13597 / 13597 | 114583 |
| c8 G9 | 3、7、8 | 1–8 | 16 | 12812 / 12812 | 112363 |

前测直接构造 prompt 会先触发 M4 的旧 feedback 相等性错误；前测容量数据明确是在审计脚本中临时派生该视图后取得，以单独量化 H2/M2，不是旧系统正常执行成功。后测没有此绕过，使用生产 builder 完整读取历史。

`audit_capacity/result.json` 使用真实消息组装和 provider usage；每次关闭推理、只请求 1 个输出 token。输出截断是该测量的预期行为，不是候选生成成功。保留原 `provider_live_01` 的真实完整候选探针作为其当时范围的证据。

| 帧 | B | 实际输入 token | 加预留 32768 输出 |
|---|---:|---:|---:|
| c6 初代重建 | 58959 | 16669 | 49437 |
| c6 G13 真实晚代 | 114943 | 37367 | 70135 |
| G13 实际 verifier 失败的修复帧 | 49248 | 13391 | 46159 |
| G13 上执行 source-request 服务的受控分支 | 102010 | 33645 | 66413 |

总实测输入 101072、输出 4 token。四帧均低于已冻结的 1048576 token 窗口与 256 KiB prompt 总上限。两场原实验没有 source-followup 请求；这里明确为真实历史上的受控分支，没有伪造该事件曾在原实验发生。未因晚代信息更足而抬高任何上限，也未将容量探针写进科学 ledger。

## 同类问题自审

- 标称 support 位置从绑定 profile 的真实默认身体位置派生，与 base 高度、roll 和 projected gravity 对齐。原 F54 的 c6 G4-B 在标称直立时得到 raw≈0.49，超过 clip_max=0.4；其原始不合理常罚现在能在夹具中看见。观察只展示，不增加一个物理启发式拒绝门。
- 补齐容量回执的历史格式边界：当前 task receipt 标识 `canonical_prompt_json.compact.v1`，capacity receipt 写 v3；旧 task receipt 的缩进格式及 capacity v2 只用于历史验证。`audit_historical_receipts.json` 验证 P4 原始 15 份 prompt 及其真实回执；`audit_original_prompt_artifacts.json` 验证两场实验 ledger 绑定的 45 份旧 prompt。
- 完整回归发现三处旧断言：两个单 round 集成测试把未训练失败当作性能源码，并在下次接受后丢弃该源码；一个裁剪测试用缩进格式的固定字节常量。前者改为分别验证未训练失败仍留在错误历史、真正训练后拒绝的源码跨接受轮继续保留；后者以实际完整消息少 1 B 的上限验证优先裁掉最早返回项，拒绝原因仍完整。未删掉这些合同测试，也未为旧断言退回两套选择器或计量格式。
- Python 3.12 AST dump 与原 Python 3.11 的旧 hash 存在版本差异，进场源码已有此行为。`audit_candidate_history.json` 在同一 Python 对照进场分析器，37 份分析一致；旧 ledger 的 hash 没被迁写，也没有宣称跨后端机制 hash 相同。
- 原真实 provider 候选又以当前代码通过作者 delta 检查和封存执行授权，见 `audit_provider_readmission.json`；其源码 SHA256 未变。该检查不重新调用 provider，不启动训练，也不将旧版第三次恢复思考的行为当作新版证据。
- 不保留已退役的独立 B authoring、v1 continuation 翻译和猜测错误码。已有 legacy 评分/投影 reader 有真实历史消费者，未为清理它们扩大本次协议迁移。
- 本地 Codex 可借鉴的是 `context_manager/history.rs` 对持久化历史/模型窗口的区分，以及 `compact_token_budget.rs` 的确定性窗口压缩。这里复用现有回执和有界 memory，未引入 LLM 摘要器、事件总线或新存储系统。

## 最终验收状态

本轮真实历史、实际 token、37 个候选源码/旧训练观察、原 provider 候选当前准入及两项显式 Isaac 检查均已通过。旧断言修正后，本地和目标各一次完整 discovery 均为 1610 项、零失败/错误。本地 19 项平台/安装条件跳过由目标覆盖；目标仅两项 opt-in 跳过，已另行启用并通过。先前三处旧断言失败和中断片段单独保留为 `audit_*_before_test_alignment.*`，不合并为成功结果。

完整回归后，`protected_after.json` 再次核对完整实验 932 个文件、517212889 B、71 个 checkpoint，与 P0 保护清单完全一致。两端 516 个源文件、配置和测试无遗漏或差异；本次进程已退出，已有 GPU lease 均未上锁。具体交付及资源记录由 `ACCEPTANCE_ZH.md` 引用。此 GO 仅覆盖 P5 工程边界，不代表策略收益或 P6/P7 已完成。

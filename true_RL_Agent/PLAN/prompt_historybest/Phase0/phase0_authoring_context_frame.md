# Phase0 上下文帧与预算协议实施记录

## 目标

Phase0 先建立公共 authoring context frame 协议，让每次 provider-facing prompt 都能说明：当前是哪类 authoring turn、包含了哪些 section、省略了哪些内容、source payload/hash 状态是什么、上下文预算是多少。此阶段不实现 source request，不读取 current-best 源码，不开放课程或环境 surface。

## 当前实施计划

1. 在共享 prompt 层新增 `AuthoringContextFrame` 协议与 frame kind 合同。
2. V1 provider 仍保持 candidate-only，所以默认 live provider-facing prompt 使用 `candidate_draft` frame。
3. 当存在 `candidate_repair_feedback` 时，assembler 生成 `static_repair` frame；该 frame 保留修复所需合同和 repair feedback，但不重复展开完整 `public_history.recent_outcomes`。
4. 在 prompt artifact 顶层记录 `authoring_context_frame`，包括 provider-visible included sections、legacy alias sections、omitted sections、source payload/hash 占位、feedback items、bytes/context-unit 预算。
5. 对最终 prompt payload 设置硬 bytes 上限；超限直接拒绝构建，而不是把超长上下文交给 provider。
6. 保持任务特殊语义只来自 `TaskPromptAdapter` / `TaskMemoryAdapter`。公共 frame 只理解 section、source、feedback、budget 和 loop kind，不解释 push recovery 字段或失败语义。

## Codex 参考取舍

Codex 的相关设计重点是上下文窗口标识、压缩/省略事件、预算事实和可追踪的 model-visible context。新项目借鉴这些原则，但不借鉴 Codex 的 shell/tool 权限模型：本项目的 Phase0 只做受控 prompt frame 元数据，不给 LLM 任意文件读取、shell 或仓库遍历权限。

## 边界

- 不改变 `CandidateProvider.next_candidate() -> CandidatePackageRef`。
- 不引入 `source_request` provider union response。
- 不把 current-best source、candidate path、真实文件路径、raw seed、checkpoint 或 private artifact 放入 prompt。
- 不把 push recovery 的字段、失败标签、success 语义写入公共 frame。
- `source_followup` 只作为未来 V2 概念保留，当前不导出、不接受、不能构建 provider-facing payload。
- `static_repair` 只面向 provider/schema/verifier/static authoring feedback；真实 train/eval runtime feedback 边界仍由后续 Phase3 处理。

## 验收

- `candidate_draft` payload 顶层包含 `authoring_context_frame`，并通过 public-only assertion。
- `static_repair` payload 不展开完整 recent public history，省略项必须写入 `omitted_sections`，并声明只能从 prior ledger public records 恢复。
- 过渡期 legacy alias sections 必须在 frame 中显式列出并计入 section bytes；frame 不能只统计 canonical sections 却声称描述了 provider context。
- frame 记录 `source_payloads`、`source_payload_summaries`、`source_hashes`，Phase0 均为空或占位，不伪装成已读 source。
- frame、每个 provider-visible section 和最终 prompt payload 都有 bytes/context-unit 预算与硬上限。
- section 顺序保持任务、loop、反馈、candidate package、frame metadata 的阅读路径，不退化成字母排序。
- 非 push 任务不会因为 Phase0 新增公共 frame 继承 push 语义。

## 已完成

- 新增公共 `true_rl_agent.prompting.authoring_frame`。
- `build_prompt_payload()` 接入 frame kind 推断、repair history 省略、frame metadata 和最终 payload 预算检查。
- 新增 Phase0 回归测试覆盖 candidate draft frame、static repair frame、省略记录、单 section 超限和总 payload 超限拒绝。

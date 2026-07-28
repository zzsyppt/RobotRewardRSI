# Phase7: 受控 Source Request 与 Source Followup

## 目标

Phase7 将 provider 的 `source_request` 从“只能拒绝的保留协议”升级为可服务的受控回环。provider 可以请求 prompt `source_catalog` 中声明的 logical id；runner 只从本轮已经构造出的 prompt source sections 中返回允许公开的 source payload，然后用 `source_followup` frame 让 provider 继续生成 `candidate_draft`。

## 边界

- `source_request` 不是文件系统访问权限，provider 不能提供路径、不能请求 catalog 外 logical id。
- resolver 不读磁盘、不追 private artifact，只消费当前 prompt payload 中的 `source_catalog` 与 `source_payloads`。
- source logical id 是 closed allowlist；hash、bytes、delivery、requestable 必须和 catalog 同源校验。
- `source_followup` 可以携带 source payload 子集，但仍复用同一套 source section validator。
- followup frame 压缩 `public_history.recent_outcomes`，并避免重复大块 legacy alias；canonical contract 仍保留。
- 每轮 source request 有次数、id 数量、总返回 bytes 预算；超限时返回 public-safe feedback，而不是崩溃或泄露路径。
- resolver 入口必须重新验证 `source_catalog`、`candidate_lineage`、`source_payloads`、`exploration_directive` 的同源关系；不能只信调用方已经构造过合法 payload。
- resolver 的可服务源库是本轮非 `source_followup` prompt 构造出的完整 source sections；provider-facing `source_followup` 只展示累计已返回的 source payloads。两者不能共用同一个收缩后的 payload 子集，否则多次 source request 会误拒绝仍在 catalog 中声明的源。
- 如果 `source_request` 发生在同轮 static repair 之后，`source_followup` 重新组装时必须保留产生该 repair repository 的 `latest_failed_candidate` 状态；否则 `attempt/latest_failed` payload 会和新 catalog 断开同源关系。
- `source_request` 服务目前只接在 `CandidateAuthoringLoop` 主链路；旧 `AgentLoop` 与 dry-run audit 的 candidate-only/replay provider 路径不应使用真实 provider source-request 回环，后续若继续保留需统一迁到 authoring loop 或显式禁用 source_request contract。

## 调用流程

1. `CandidateAuthoringLoop` 构造普通 `candidate_draft` prompt。
2. provider 若返回 `source_request`，`next_authoring_response()` 先校验 logical ids 属于当前 prompt source catalog。
3. `resolve_source_request()` 从当前 prompt payload 的 inline source payload 中取可请求内容，生成 `source_request_feedback`。
4. 下一次 attempt 使用 `source_followup` frame，携带返回的 source payload 子集与 feedback。
5. provider 应在获得足量信息后返回 `candidate_draft`；若继续请求 source，runner 继续按预算服务或拒绝。

## 验收

- 非 allowlist logical id 在 provider response 解析/校验阶段被拒绝。
- 无 current-best 或 source 不可请求时，返回 bounded `source_request_denied` followup，而不是文件读或路径泄露。
- repeated source request 达到每轮上限后返回 `source_request_turn_limit_exceeded`。
- `source_followup` payload 不重复完整首轮上下文，`recent_outcomes` 被省略且可从 ledger 恢复。
- 多次 source request 可从同一轮完整 repository 中请求不同 logical id；后续 followup prompt 累计包含已返回 payloads，保证无状态 provider 调用仍有足量上下文。
- verifier/component smoke/anti-degenerate 等同轮 static repair 后，请求 `attempt/latest_failed/training_code/components.py` 能生成合法 `source_followup`，不会因重新构建 catalog 而丢失同轮 failed source。
- provider-facing feedback 和 attempt record 均通过 public-only guard。
- 篡改 `source_payload.content`、`bytes`、`content_sha256` 或 catalog hash 会在 resolver 返回内容前失败。
- 聚焦测试覆盖 authoring loop、prompt assembler、source catalog、OpenAI-compatible provider 协议。

# Phase6: AuthoringProviderResponse 协议迁移

## 目标

Phase6 的目标是把 provider 输出从隐式 candidate-only 返回值收敛成显式 authoring response envelope，为后续受控 source request 服务留出协议位置，同时不破坏现有 replay、audit、OpenAI-compatible provider 的 candidate-only 主链路。

## 实施计划

1. 引入共享 `AuthoringProviderResponse`。
   - `CandidateDraftResponse` 表示已 materialize 的候选草稿。
   - `SourceRequestResponse` 表示 provider 请求 prompt source logical id。
   - 旧 `CandidatePackageRef` 返回路径通过兼容桥自动包装成 `CandidateDraftResponse`。

2. 扩展 provider 协议但保持兼容。
   - 旧 provider 仍只需要实现 `next_candidate(prompt_payload=...) -> CandidatePackageRef`。
   - 新 provider 可选实现 `next_authoring_response(prompt_payload=...) -> AuthoringProviderResponse`。
   - `CandidateAuthoringLoop` 通过统一桥接函数调用 provider，不直接假设返回值就是 candidate。

3. 迁移 OpenAI-compatible provider。
   - provider 原始 JSON 先解析 `response_mode`，再决定进入 candidate materializer 或 source request parser。
   - 缺省 `response_mode` 仍视为 `candidate_draft`，保持旧 JSON 响应兼容。
   - 显式 `response_mode="candidate_draft"` 与旧 candidate-only 行为一致。
   - `response_mode="source_request"` 会被识别为独立协议模式，不再被 candidate materializer 当作 unsupported candidate key。

4. 暂不启用 source request 服务。
   - Phase6 只完成协议迁移，不返回源码，不启用 `source_followup` frame。
   - `source_request_service_enabled=false` 写入 provider contract。
   - 同轮出现 source request 时，authoring loop 将其转成 public-safe `source_request_denied` static repair feedback，要求 provider 返回 candidate draft。

5. 保持安全边界。
   - source request 只接受 prompt logical id，不接受真实路径、Windows drive path、反斜杠、绝对路径或 `..`。
   - source request 的 logical id 必须绑定同一次 prompt payload 中 `source_catalog.items[].logical_id` 形成的 closed allowlist。
   - authoring bridge 会重新用当前 prompt catalog 校验 provider 返回的 `SourceRequestResponse`，不信任 provider 自报的 catalog。
   - `SourceRequestResponse.to_public_dict()` 自身必须 public-safe；含 `api_key`、`raw_seed`、`trusted_seed`、`sk-...` 等 forbidden marker 的 rationale/logical id 会在构造或导出阶段被拒绝。
   - attempt record 只记录 public-safe response mode、candidate id、change set、failure category，不记录 candidate path。
   - 任务特殊语义仍只在 task-owned prompt adapter / task-owned smoke adapter 中，不进入共享 provider response 层。

## Codex 参考取舍

Codex 的可借鉴点是协议 envelope 和事件/响应分层：原始模型输出先归一化成有明确 kind/status 的内部协议对象，再由上层状态机处理。Phase6 采用这个组织原则，把 provider 原始 JSON 先归一化为 `candidate_draft` 或 `source_request`，但不引入 Codex 的工具调用、自由文件读取或长 transcript 模型。

## 边界

- 不开放 source resolver。
- 不开放 `source_followup` provider-facing frame。
- 不改变真实 train/eval 只对最终候选执行一次的边界。
- 不改变 accept/reject 决策逻辑。
- 不把 public eval threshold、private seed、checkpoint、runner internals 或真实文件系统路径暴露给 provider。
- 不把 push recovery 特殊字段写进共享 provider response 协议。

## 验收

- 旧 candidate-only provider 继续可用。
- OpenAI provider 支持旧 candidate JSON 和显式 `response_mode="candidate_draft"`。
- OpenAI provider 能解析 `source_request`，并且不会 materialize candidate package。
- OpenAI provider 会拒绝不在本轮 prompt source catalog closed allowlist 内的 logical id。
- 手工构造的 `SourceRequestResponse` 也不能导出非 public-safe dict。
- `next_candidate()` 调用方遇到 source request 时得到 public-safe `source_request_denied`，而不是 unsupported-key materializer 错误。
- `CandidateAuthoringLoop` 遇到 source request 时生成 static repair turn，并在下一次 attempt 继续 candidate draft 流程。
- provider-facing payload 合同明确 `source_request_service_enabled=false`。
- focused tests、full tests、compileall 和 diff-check 通过。

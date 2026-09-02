# Phase3: Runtime Failure 边界收敛

## 目标

修正 live authoring loop 中真实 runner failure 的边界：candidate 一旦通过静态 verifier 并进入真实 runner，若发生 `CandidateRuntimeError`，当前 round 必须结束。runtime failure 只作为 public-safe ledger record 和下一 round prompt evidence，不允许在同一 round 内触发 provider retry，也不允许同一 round 内再次跑真实 train/eval。

Phase3 只处理 runtime failure 边界，不抽出完整 `CandidateAuthoringLoop`，不新增 `source_request`，不改变 provider candidate-only 协议。

## 当前问题

当前 `live_loop.py` 在 `CandidateRuntimeError` 后会：

- 记录本 attempt 的 `runner_status=fail`。
- 构造 `provider_runtime_repair_feedback`。
- `continue` 进入下一次 provider attempt。
- 若同一 round 没有成功 result，收尾路径还可能再次调用 `run_candidate_round(... record_candidate_runtime_failures=True)`。

这会把真实 runner failure 当成同轮可修复静态 feedback，给 LLM 一次“看 public runtime failure 后马上再试”的机会，偏离“同一 round 只对最终 candidate 真实训练/评估一次”的目标。

## 实施计划

1. 保留同轮静态 repair。
   - provider JSON/schema/materialize failure 仍可进入下一 attempt。
   - verifier failure 仍可进入下一 attempt。
   - failed-attempt source 继续通过 Phase2 的 `source_catalog` / `source_payloads` 暴露给静态 repair。

2. 终止同轮 runtime repair。
   - `run_candidate_round()` 对 live loop 内部 runner 调用使用 `record_candidate_runtime_failures=True`。
   - `CandidateRuntimeError` 不再被 live loop 捕获后继续 provider attempt；round record 由 `run_candidate_round()` 写入 ledger。
   - runtime failure attempt record 只作为 summary/audit metadata，不生成同轮 `candidate_repair_feedback`。
   - runtime failure 后 `prompt_payload_paths` 记录触发真实 runner 的 prompt artifact。

3. 下一轮继承 runtime feedback。
   - runtime failure ledger record 进入 `build_public_context()`。
   - 下一 round prompt 的 `public_history` / task prompt adapter 可以看到 public-safe `runner_error` 与 decision reason。
   - 这些 feedback 不暴露真实路径、raw seed、checkpoint、private artifact。

4. 防止重复训练。
   - runtime failure 后不再走收尾 `result is None` 的二次 `run_candidate_round()`。
   - summary 中该 round 只出现一次 provider attempt、一次 candidate path、一个 ledger record。
   - train_result/public_result 为空是可接受的 reject 结果，不应被当作缺失 round。

## 边界

- 不改变 `CandidateProvider.next_candidate(prompt_payload=...) -> CandidatePackageRef`。
- 不实现 source request 或 provider union response。
- 不把 runtime failure 的 task-specific 语义写入公共层；任务层继续负责解释任务反馈。
- 不改变 accept/reject 决策规则，只改变 runtime failure 的同轮 authoring 控制流。
- 不让真实 train/eval feedback 进入同一 round provider prompt。

## 验收

- `CandidateRuntimeError` 后不会再次调用 provider。
- `CandidateRuntimeError` 后不会再次调用 runner。
- runtime failure round 写入 ledger，decision reason 为 `candidate_runtime_failed`。
- 下一 round prompt public history 包含 public-safe runtime failure summary。
- provider/schema/verifier failure 仍可在同一 round 内 repair。
- runtime failure 不产生同轮 `provider_or_runtime_repair_feedback` prompt。
- live summary 能区分 runtime-failed round 与 provider failure round。
- full unit/integration/compileall/diff-check 通过。

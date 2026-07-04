# Phase4: CandidateAuthoringLoop 静态 authoring 收敛

## 目标

把 live loop 中 provider attempts、prompt artifact 保存、provider/schema/materialize repair、verifier repair、latest failed source 接入这些静态 authoring 职责抽成 `CandidateAuthoringLoop`。`live_loop` 只负责逐 round 调用 authoring loop，并对最终选中的候选执行一次 `run_candidate_round()`。

Phase4 保持 V1 candidate-only provider 协议，不开放 `source_request`，不引入无限 transcript，不把 runtime train/eval failure 放回同一 round repair。

## 实施计划

1. 新增 `true_rl_agent.agent_core.authoring_loop`。
   - `CandidateAuthoringLoop.run_round()` 接收 round id、已公开 ledger records、context builder、provider、verifier、round prompt 目录和最大 attempts。
   - 每个 attempt 构建并保存完整 prompt payload，再把同一份 payload 传给 provider。
   - attempt metadata 只记录 public-safe authoring 状态和 prompt artifact path。

2. 保留现有静态 repair 行为。
   - provider JSON/schema/materialize failure 继续由 `ProviderCandidateError` 生成 `candidate_repair_feedback`。
   - verifier failure 继续生成 `verifier_error` repair feedback，并通过 `attempt/latest_failed/training_code/components.py` 暴露 latest failed source。
   - repair turn 使用 `static_repair` frame，省略不必要 recent history，不复制无限上下文。

3. 保留 Phase3 runtime 边界。
   - authoring loop 不调用 runner，不处理真实 train/eval runtime feedback。
   - runtime failure 只能由 `run_candidate_round(... record_candidate_runtime_failures=True)` 记录为本 round reject，并进入下一 round public history/source context。
   - 同一 round 内不因 runtime failure 重新调用 provider。

4. live loop 职责缩小。
   - 每 round 先调用 `CandidateAuthoringLoop`。
   - provider 全部失败时，写 public-safe provider failure ledger record。
   - 有候选时，只调用一次 `run_candidate_round()`，包括最终 verifier fail 的 reject 记录；最终 verifier fail 不触发 train/eval，summary 记录 `runner_status=not_run`。
   - summary 继续保留 `provider_attempt_records` 和 `provider_attempt_prompt_payload_paths`。

5. 为 Phase5 保留扩展位置但不提前伪实现。
   - component smoke 和 anti-degenerate 的任务输入、字段 shape、任务失败语义仍归后续 task-owned adapter/runtime fixture 处理。
   - Phase4 不在共享层硬编码 push recovery 字段、success predicate 或 reward 语义。

## Codex 参考取舍

Codex 的可借鉴点是 bounded turn context、明确 request 生命周期、context/frame metadata 和省略信息可追踪。新项目只借鉴这些组织原则，不借鉴 Codex 的工具调用、任意文件读取、source follow-up 或 shell 权限模型。

## 边界

- 不改变 `CandidateProvider.next_candidate(prompt_payload=...) -> CandidatePackageRef`。
- 不新增 provider union response 或 `source_request`。
- 不让 provider 访问文件系统；source 只能来自 runner 解析后的 `source_catalog` / `source_payloads`。
- 不把任务特殊语义写入共享 authoring loop。
- 不改变 accept/reject 决策规则。
- 不改变 trusted eval、seed、checkpoint、budget、curriculum 权限边界。

## 验收

- provider parse/schema/materialize failure 可以进入 static repair turn。
- verifier failure 可以进入 static repair turn，并携带 latest failed logical source slot。
- repair turn 使用 `static_repair` frame，而不是无限 transcript。
- provider 全部失败时写 public-safe failure record，不生成候选。
- verifier 全部失败时不执行 train/eval，ledger 记录 `candidate_verifier_failed`。
- runtime failure 不回到同一 round repair。
- existing OpenAI provider、ReplayProvider、dry-run audit/live loop 调用协议不需要迁移到 source request。
- full unit/integration/compileall/diff-check 通过。

## 已完成

- 新增 `src/true_rl_agent/agent_core/authoring_loop.py`。
- `live_loop.py` 改为调用 `CandidateAuthoringLoop`。
- verifier repair 文案改为引用 `reward_inputs_contract`，不在共享层举具体任务字段。
- 新增 `tests/unit/test_candidate_authoring_loop.py`。
- 扩展 `tests/integration/test_live_loop.py` 覆盖 max-attempt verifier failure 不执行 train/eval。

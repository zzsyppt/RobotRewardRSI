# Step 11 安全与边界回归

## Scope

本记录对应 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step11。

目标是把前面 Step0-Step10 已确立的 prompt 安全边界变成自动回归，而不是依赖人工检查：

- prompt 不暴露 private、hidden、trusted seed/eval、checkpoint selector、API key、secret、raw seed id。
- 未接线能力不作为 effective controls 暴露给模型。
- fake non-push task 不继承 push recovery 的 success predicate、failure taxonomy、push-specific RewardInputs。
- runner/eval/budget/seed/checkpoint/PPO 只能作为 runner-owned 禁止修改项出现，不能出现在 candidate-writable surface。
- Step10 的 provider request、saved prompt artifact、ledger prompt ref traceability 保持可测。

## Codex Design Reference

参考 `D:\humanoid_agent\codex\AGENTS.md` 的 model-visible context 原则：

- model-visible context 必须有硬边界，不允许无界或私有数据进入 prompt。
- request/payload 应使用结构化断言测试，而不是只靠字符串人工扫读。
- 共享层只维护通用安全/结构规则；任务特殊语义由 task-owned adapter 暴露。

## Implementation Plan Used

1. 审计 shared public safety、prompt assembler、task prompt adapter、provider request、ledger ref 测试覆盖。
2. 先写 raw seed 红灯测试：`assert_public_only_prompt_payload({"raw_seed_id": ...})` 必须失败。
3. 将 `raw_seed` 加入 shared `FORBIDDEN_PUBLIC_KEY_PARTS`，使 sanitizer 和 assert 共用同一规则。
4. 增加 `test_prompt_safety_boundaries.py`，集中覆盖 Step11 的安全边界和 fake non-push 隔离。
5. 扩展 memory context 测试，确认 raw mapping record 中的 `raw_seed_id` 会在 prompt builder 源头被 scrub/drop。
6. 复用 Step10 的 provider/saved/ledger traceability 测试作为本步骤的 traceability 回归。

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/public_safety.py`
- `true_RL_Agent/tests/unit/test_prompt_safety_boundaries.py`
- `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- `true_RL_Agent/tests/unit/test_memory_context_phase13.py`

## Red Test Observed

实现前新增断言失败：

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_prompt_projection_and_mechanism.PromptProjectionAndMechanismTests.test_public_only_prompt_payload_rejects_private_hidden_or_secret_keys
```

失败原因：

- `FORBIDDEN_PUBLIC_KEY_PARTS` 没有包含 `raw_seed`。
- `raw_seed_id` 可通过 public-only assert，违反 Step11 明确要求。

修复后，`raw_seed` 由 shared public safety 层统一处理。

## Boundary Decisions

- 不在 shared layer 写入 push recovery 的 success/failure 语义；fake non-push 回归只检查不会出现 push-specific 字符串。
- 不把 `budget`、`seed`、`eval protocol` 等变成 candidate-writable controls；它们只在 `candidate_must_not_change` / runner-owned 文案中出现。
- 未接线能力仍不 prompt-facing：`LossTerm`、`ObservationTransform`、`train_observation_transforms`、`not_instrumented` 不出现在 generic prompt。
- `raw_seed` 是通用安全 key，不是任务特殊语义，适合放在 shared public safety。

## Post-Review Hardening

复审时发现原始 Step11 对 key-level 泄露覆盖较强，但对 text value 中嵌入的 forbidden marker 覆盖不足：

- `trusted_eval`、`hidden_eval`、`trusted_seed`、`checkpoint_selector` 等如果出现在 public-history reason/error 文本中，旧断言不会拒绝。
- 修复后，shared public safety 会拒绝 text value 中的精确敏感 marker，例如 `hidden_eval`、`hidden eval`、`api_key`、`api key`、`checkpoint_selector`、`checkpoint selector`、`raw_seed`、`raw-seed`。
- public-history 现在区分 key-level denylist 和 free-text denylist：key 仍按 shared forbidden key parts 删除，free text 只按精确敏感短语 redacted，避免把 `invalid token at line 1` 这类非敏感错误误删。
- 合法边界说明仍保留，例如 `counts_only_no_raw_seed_ids`、`raw seed identifiers are omitted`、runner-owned `checkpoint_policy`，不会被误杀。

## Verification Targets

Step11 至少需要通过：

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_prompt_safety_boundaries tests.unit.test_prompt_projection_and_mechanism tests.unit.test_task_prompt_adapter tests.unit.test_memory_context_phase13 tests.unit.test_openai_compatible_provider tests.integration.test_fake_round.FakeRoundTests.test_round_ledger_records_prompt_payload_ref
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover
git diff --check
```

## Remaining Scope

- Step12 需要运行单轮 smoke，验证真实 prompt artifact、provider user payload 和 candidate pipeline。
- Step13 需要运行多轮 live/audit，检查 prompt 不会随轮次变薄，并继续监控安全边界。

# Step 12 单轮 Smoke 验收

## Scope

本记录对应 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step12。

目标是用最小真实 loop 流程验证升级后的 prompt 能生成、保存、被 provider 实际消费，并进入 candidate 验证、训练、公开评估和 ledger 记录链路。

本步骤不启动 Isaac、不联网、不调用真实 LLM。使用现有 `run_live_loop()`，配合 fake task/backend/runner/verifier 和一个捕获 provider user payload 的 smoke provider。

## Smoke Design

smoke provider 做两件事：

- 将实际收到的 `prompt_payload` 用 `serialize_prompt_payload()` 写入 `provider_capture/provider_user_payload_001.json`。
- 返回一个 fake `CandidatePackageRef`，让现有 fake backend/runner/verifier 完成单轮 candidate pipeline。

验证点：

- `prompt_payload.json` 包含 Step3/Step10 后要求的 canonical sections。
- provider 捕获到的 user payload 文本与保存的 `prompt_payload.json` 完全一致。
- ledger 的 `prompt_payload.path` 指向同一个 saved artifact。
- public-only assert 通过。
- candidate pipeline materialized：provider pass、verifier pass、runner pass、train result 和 public result 均出现。

## Local Smoke Artifact

本地 smoke 通过：

- output dir: `D:\humanoid_agent\true_RL_Agent\runs\step12_prompt_smoke_20260702T183040Z`
- prompt payload: `D:\humanoid_agent\true_RL_Agent\runs\step12_prompt_smoke_20260702T183040Z\rounds\step12-smoke-20260702T183040Z-round-001\prompt_payload.json`
- provider user payload: `D:\humanoid_agent\true_RL_Agent\runs\step12_prompt_smoke_20260702T183040Z\provider_capture\provider_user_payload_001.json`
- summary: `D:\humanoid_agent\true_RL_Agent\runs\step12_prompt_smoke_20260702T183040Z\step12_smoke_summary.json`

All checks were true:

```json
{
  "at_least_one_public_result": true,
  "at_least_one_train_result": true,
  "candidate_count_matches_rounds": true,
  "candidate_pipeline_materialized": true,
  "ledger_prompt_ref_matches_saved_artifact": true,
  "prompt_payload_count_matches_rounds": true,
  "prompt_public_only_assert_passed": true,
  "provider_payload_matches_saved_artifact": true,
  "required_sections_present": true,
  "round_count_matches_request": true
}
```

## Server Smoke Artifact

服务器 smoke 通过：

- output dir: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_prompt_smoke_20260702T183154Z`
- prompt payload: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_prompt_smoke_20260702T183154Z/rounds/step12-smoke-20260702T183154Z-round-001/prompt_payload.json`
- provider user payload: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_prompt_smoke_20260702T183154Z/provider_capture/provider_user_payload_001.json`
- summary: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_prompt_smoke_20260702T183154Z/step12_smoke_summary.json`

All checks were true:

```json
{
  "at_least_one_public_result": true,
  "at_least_one_train_result": true,
  "candidate_count_matches_rounds": true,
  "candidate_pipeline_materialized": true,
  "ledger_prompt_ref_matches_saved_artifact": true,
  "prompt_payload_count_matches_rounds": true,
  "prompt_public_only_assert_passed": true,
  "provider_payload_matches_saved_artifact": true,
  "required_sections_present": true,
  "round_count_matches_request": true
}
```

## Repair Smoke Artifact

为了覆盖 repair attempt 条件，额外运行 provider-fail-then-success 单轮 smoke。第一次 provider attempt 抛出 public-safe `ProviderCandidateError`，第二次 provider attempt 成功并进入 candidate pipeline。

本地 repair smoke 通过：

- output dir: `D:\humanoid_agent\true_RL_Agent\runs\step12_repair_smoke_20260702T183339Z`
- first prompt: `D:\humanoid_agent\true_RL_Agent\runs\step12_repair_smoke_20260702T183339Z\rounds\step12-repair-20260702T183339Z-round-001\prompt_payload.json`
- repair prompt: `D:\humanoid_agent\true_RL_Agent\runs\step12_repair_smoke_20260702T183339Z\rounds\step12-repair-20260702T183339Z-round-001\prompt_payload_attempt_002.json`
- summary: `D:\humanoid_agent\true_RL_Agent\runs\step12_repair_smoke_20260702T183339Z\step12_repair_smoke_summary.json`

服务器 repair smoke 通过：

- output dir: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_repair_smoke_20260702T183413Z`
- first prompt: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_repair_smoke_20260702T183413Z/rounds/step12-repair-20260702T183413Z-round-001/prompt_payload.json`
- repair prompt: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_repair_smoke_20260702T183413Z/rounds/step12-repair-20260702T183413Z-round-001/prompt_payload_attempt_002.json`
- summary: `/data/yuyangliu/AGENT/true_RL_Agent/runs/step12_repair_smoke_20260702T183413Z/step12_repair_smoke_summary.json`

Repair checks were true:

```json
{
  "candidate_pipeline_materialized": true,
  "first_provider_attempt_failed": true,
  "ledger_prompt_ref_matches_selected_attempt": true,
  "provider_payloads_match_saved_attempts": true,
  "repair_feedback_promoted": true,
  "repair_feedback_public_safe": true,
  "second_provider_attempt_passed": true,
  "selected_prompt_is_second_attempt": true,
  "two_provider_attempts_recorded": true
}
```

## Boundary Decisions

- Step12 smoke 不新增 task-specific shared logic。
- smoke 使用 fake task，因此不验证 push recovery 训练质量，只验证 prompt/candidate/ledger 管线。
- 真实 provider/API 与真实 Isaac 训练留到 Step13 live/audit 验收。
- smoke artifact 留在 ignored `runs/` 下，不提交到 Git。

## Verification Commands

Focused Step12 smoke 是通过 inline Python 调用 `run_live_loop()` 完成的。

提交前仍需通过：

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover
git diff --check
```

服务器同步后至少复核：

```bash
cd /data/yuyangliu/AGENT/true_RL_Agent
PYTHONPATH=src /usr/bin/python3 -m unittest tests.unit.test_prompt_safety_boundaries tests.unit.test_openai_compatible_provider tests.integration.test_live_loop.LiveLoopTests.test_live_loop_passes_saved_prompt_payload_to_provider
```

# Step 2 Provider Repair Feedback Record

## Goal

Implement Step 2 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: provider-side candidate failures must become public-safe repair feedback for the next provider attempt instead of terminating the live round with an uncaught exception.

## Current-State Decisions

- Step0 same-source prompt delivery and ledger prompt references are treated as the base contract.
- Step2 does not add richer task sections, RewardInputs semantics, public history projection, or task-owned prompt adapters.
- Step2 does not change shared accept/reject success semantics.
- Old Phase 8E is used only as a repair-feedback shape reference: expose categorized public issues, not raw provider text or private artifacts.

## Implementation Plan Used

1. Add a provider failure type with public-safe categories:
   - `json_parse_error`
   - `schema_error`
   - `candidate_materialize_error`
   - `verifier_error`
   - `runtime_error`
2. Make `OpenAICompatibleCandidateProvider` wrap candidate proposal parse, schema, declared-component, and materialization failures as `ProviderCandidateError`.
3. Reject invalid `declared_changed_components` instead of silently materializing a no-change candidate.
4. Make `run_live_loop()` catch `ProviderCandidateError`, record the failed attempt, and pass sanitized repair feedback into the next prompt attempt.
5. If all provider attempts fail before a candidate exists, write a ledger-safe provider failure record for auditability.
6. Add regression tests for malformed JSON, missing `components_py`, invalid `declared_changed_components`, materialization failure, verifier repair, runtime repair, provider repair, and max-attempt provider failure.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/provider_errors.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- `true_RL_Agent/tests/integration/test_live_loop.py`

## Implemented Behavior

- Provider candidate failures now expose public dictionaries with `schema_version`, `category`, `reason`, and sanitized `details`.
- Repair feedback includes `failure_category`, `provider_status`, concise reason, and instructions to fix only the candidate response or `training_code/components.py`.
- Repair feedback explicitly keeps runner, eval protocol, budget, seeds, checkpoints, PPO, and acceptance rules unchanged.
- Provider failure details redact `sk-*` values and Windows/Linux-like local paths before entering prompt or ledger records.
- The path scrubber covers Windows backslash paths such as `D:\...`, Windows forward-slash paths such as `D:/...`, and common Linux/server absolute paths.
- `run_live_loop()` records each attempt's prompt payload path and provider status.
- If provider repair succeeds on a later attempt, the round consumes only the successful candidate.
- If all attempts fail before candidate materialization, the ledger records `provider_failure` with `candidate: null` and decision reason `provider_candidate_generation_failed`.
- Existing verifier and runtime repair feedback now includes `failure_category` values `verifier_error` and `runtime_error`.

## Red Tests Observed

Before implementation, focused Step2 tests failed at import time because `true_rl_agent.agent_core.provider_errors` did not exist. This confirmed the new tests were exercising missing Step2 behavior.

## Verification

Focused commands run with bundled Python:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.unit.test_openai_compatible_provider
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest true_RL_Agent.tests.integration.test_live_loop
```

Focused results:

- `true_RL_Agent.tests.unit.test_openai_compatible_provider`: `Ran 9 tests`, `OK`
- `true_RL_Agent.tests.integration.test_live_loop`: `Ran 7 tests`, `OK`

Follow-up review regression:

- Added a focused regression for `D:/humanoid_agent/private/raw.txt` in provider repair feedback and provider failure ledger records.
- Verified `true_RL_Agent.tests.unit.test_openai_compatible_provider`: `Ran 10 tests`, `OK`
- Verified `true_RL_Agent.tests.integration.test_live_loop`: `Ran 7 tests`, `OK`

## Remaining Scope

- Step3 should define the full prompt information-sufficiency standard.
- Step4/Step5 should introduce task-owned prompt adapter and move RewardInputs contract out of shared generic projection.
- Step8 should make provider failure ledger records more visible in bounded public history.

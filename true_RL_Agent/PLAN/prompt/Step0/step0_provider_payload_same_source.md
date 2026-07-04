# Step 0 Provider Payload Same-Source Record

## Goal

Fix the provider interface so every candidate provider receives the complete prompt payload that was built for the round/attempt. This prevents future prompt sections from being saved as artifacts but silently omitted from the model request.

## Implementation Plan Used

- Change `CandidateProvider.next_candidate()` from `public_context` to `prompt_payload`.
- Update replay, audit replay, fake, live, and OpenAI-compatible providers to use the full payload contract.
- Make live loop and audit loop pass the exact payload object that is saved for the attempt.
- Make the simple `AgentLoop` construct a full prompt payload before calling its provider.
- Remove OpenAI provider-side prompt rebuilding from `public_context`.
- Stop sorting top-level prompt payload keys in the OpenAI provider user message.
- Add RED tests proving:
  - OpenAI provider uses the supplied payload and does not rebuild it.
  - live loop passes the saved prompt payload to provider.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/provider.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/round.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/live_loop.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/loop.py`
- `true_RL_Agent/src/true_rl_agent/audit/loop_audit.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`
- `true_RL_Agent/tests/integration/test_live_loop.py`

## Red Tests Observed

Before implementation:

- `OpenAICompatibleCandidateProvider.next_candidate(prompt_payload=...)` failed with unexpected keyword `prompt_payload`.
- `run_live_loop()` failed against a payload-only provider because it still called `next_candidate(public_context=...)`.

These failures confirmed the tests were exercising the Step0 bug.

## Implemented Behavior

- `CandidateProvider.next_candidate()` now accepts `prompt_payload`.
- `ReplayProvider` and `AuditReplayProvider` ignore but accept the complete prompt payload.
- `run_live_loop()` passes the same `prompt_payload` object that it saves to the round artifact.
- `AgentLoop.run()` builds a full prompt payload from public context before provider selection.
- `OpenAICompatibleCandidateProvider` consumes the supplied prompt payload directly.
- OpenAI provider user message uses `json.dumps(prompt_payload, ensure_ascii=False, indent=2)` without `sort_keys=True`.

## Traceability Addendum

- `RoundInputs` carries an optional `PromptPayloadRef`.
- `run_candidate_round()` writes `prompt_payload` into each ledger record when the caller saved a public prompt artifact.
- `run_live_loop()` passes the selected attempt's saved prompt payload ref into the ledger path, including repair attempts.
- `run_loop_audit()` passes each saved audit prompt payload ref into the ledger path.
- Regression tests verify live, repair, and audit ledger records point to the same prompt payload path recorded in the run summary.

## Verification

Commands run with bundled Python:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s true_RL_Agent/tests
```

Result:

- `Ran 125 tests`
- `OK`

Focused checks also passed:

- `true_RL_Agent.tests.unit.test_openai_compatible_provider`
- `true_RL_Agent.tests.integration.test_live_loop`
- `true_RL_Agent.tests.integration.test_agent_core_loop`
- `true_RL_Agent.tests.integration.test_dry_run_audit_loop`
- `true_RL_Agent.tests.unit.test_prompt_projection_and_mechanism`
- `true_RL_Agent.tests.unit.test_memory_context_phase13`
- `true_RL_Agent.tests.unit.test_push_recovery_task`

## Remaining Scope

Step0 does not make the prompt semantically richer. It only fixes same-source delivery. Step2 should now build on this by turning provider parse/schema/materialize failures into public-safe repair feedback.

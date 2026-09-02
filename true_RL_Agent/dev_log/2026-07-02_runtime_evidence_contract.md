# Runtime Evidence Contract Hardening

## Context

The server-side one-round runtime smoke reached train and trusted eval, then failed while parsing mechanism evidence:

`RuntimeActivationEvidence.from_mapping()` expects `diagnostic_outputs` to be mapping-shaped, but the push recovery runtime summary emitted a list of diagnostic summaries.

## Change

- Converted `CandidateRuntimeAdapter.runtime_summary()["diagnostic_outputs"]` into a diagnostic-name-keyed mapping.
- Added a regression assertion that the runtime summary can be consumed directly by `RuntimeActivationEvidence.from_mapping()`.
- Kept diagnostic content limited to public numeric summaries produced by the runtime adapter.
- Added explicit Isaac trusted-eval summary fields recording that candidate code/package was not loaded during eval.

## Verification

- `python -m unittest tests.unit.test_push_recovery_runtime_adapter.PushRecoveryRuntimeAdapterTests.test_candidate_reward_adapter_adds_guarded_rewards_and_records_activation -v`
- `python -m unittest tests.unit.test_push_recovery_runtime_adapter tests.unit.test_prompt_projection_and_mechanism tests.unit.test_push_recovery_task -v`
- `python -m compileall -q src scripts tests`
- `python -m unittest discover -s tests -v`
- Static scans for parent-project/runtime-boundary leaks returned no matches.

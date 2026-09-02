# Step 7 Candidate Package and Provider Output Contract

## Scope

This record completes Step 7 from `PLAN/prompt/prompt_upgrade_execution_plan.md`: make the provider-facing candidate package contract explicit in the user payload, while keeping the first implementation bounded to the existing OpenAI-compatible JSON materializer.

Step7 does not expand candidate authority. It does not allow runner patches, eval patches, budget changes, seed changes, checkpoint changes, PPO changes, success metric changes, or task-invariant changes.

## Implementation Plan Used

1. Add failing tests that `required_candidate_package` exposes an explicit provider JSON output contract.
2. Add failing tests that the prompt-visible candidate package contract describes materialized files and provider-owned artifacts.
3. Add failing tests that provider materialization writes both `operation_diff.json` and `diagnostics.json`.
4. Add failing tests that unsupported or prohibited provider output keys are rejected as public-safe schema errors.
5. Move provider JSON key policy into `agent_core.provider_contract`.
6. Reuse the provider contract from the prompt payload and OpenAI-compatible provider.
7. Re-run focused prompt/provider tests, live-loop/provider integration tests, full unittest discovery, and diff hygiene.

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/provider_contract.py`
- `true_RL_Agent/src/true_rl_agent/agent_core/openai_compatible_provider.py`
- `true_RL_Agent/src/true_rl_agent/prompting/payload.py`
- `true_RL_Agent/tests/unit/test_prompt_projection_and_mechanism.py`
- `true_RL_Agent/tests/unit/test_openai_compatible_provider.py`

## Provider JSON Output Contract

The prompt-visible `required_candidate_package.provider_json_output` now states:

- The response must be one JSON object.
- No prose is allowed outside JSON.
- Required and allowed keys are exactly:
  - `candidate_id`
  - `rationale`
  - `declared_changed_components`
  - `components_py`
- Additional keys are not allowed.
- Prompt-advertised `declared_changed_components` remains restricted to currently wired values:
  - `reward_callable`
  - `diagnostic`
  - `reward_weight`

The materializer may still parse legacy declared component labels for backward compatibility, but those labels are not advertised to the model.

## Candidate Package File Contract

The prompt-visible `required_candidate_package.candidate_files` now describes the package materialized by the provider:

- `candidate_manifest.yaml`: provider-owned fixed schema, candidate id, and controlled entrypoint.
- `training_code/components.py`: materialized from `components_py`.
- `rationale.md`: materialized from `rationale`.
- `operation_diff.json`: provider-owned public provenance for declared changed components; not a runner patch.
- `diagnostics.json`: provider-owned diagnostics placeholder for future package diagnostics.

The LLM is not asked to output raw files directly. It outputs the allowed JSON fields only; the provider materializes files.

## Boundary Decisions

- `operation_diff` and `diagnostics` are prohibited provider JSON keys in v1. They are provider-materialized artifacts, not LLM-controlled surfaces.
- `diagnostics.json` is not the same as task runtime `DiagnosticTerm`. Runtime diagnostics still belong in `components.py` through the candidate component contract.
- Unsupported or prohibited provider JSON keys raise `ProviderCandidateError(category="schema_error")` with public-safe key-only feedback.
- The prompt explicitly says the candidate must not change runner code, eval protocol, success metric, training budget, seed policy, checkpoints, PPO, or acceptance rules.
- The existing `candidate_manifest.yaml` schema remains the compact new-system schema; old Phase 8E manifest fields were not copied into the new project.

## Red Tests Observed

Before implementation:

- `test_required_candidate_package_defines_provider_output_and_file_contract` failed because `required_candidate_package` had no `candidate_files` contract.
- `test_provider_materializes_candidate_package_from_json_response` failed because `diagnostics.json` was not written.
- `test_provider_rejects_unsupported_or_prohibited_output_keys` failed because an LLM-supplied `runner_patch` key was ignored instead of rejected.

These failures matched the Step7 gap.

## Verification

Focused commands run with bundled Python:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_prompt_projection_and_mechanism tests.unit.test_openai_compatible_provider
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_prompt_projection_and_mechanism tests.unit.test_openai_compatible_provider tests.integration.test_live_loop tests.integration.test_dry_run_audit_loop
git -C D:\humanoid_agent diff --check
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests
```

Focused results:

- `test_prompt_projection_and_mechanism` + `test_openai_compatible_provider`: `Ran 26 tests`, `OK`
- Prompt/provider/live-loop focused group: `Ran 41 tests`, `OK`
- `git diff --check`: no whitespace or patch formatting issues

Full-suite result:

- `unittest discover -s tests`: `Ran 156 tests`, `OK`

Note: the full-suite output includes an argparse usage/error message from an existing test that intentionally validates training budget mismatch handling; the unittest process still exits with `OK`.

## Remaining Scope

- Step8 must improve bounded public history and runtime feedback projection.
- Step9/10 still own final provider role split and readability cleanup.
- Step12/13 must validate real prompt artifacts and provider user payload captures.

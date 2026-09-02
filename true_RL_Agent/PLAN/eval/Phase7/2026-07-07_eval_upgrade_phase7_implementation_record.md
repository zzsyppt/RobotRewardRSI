# Phase7 Implementation Record - Prompt-Safe Eval Feedback

## Implemented

- Added task-owned `prompt_feedback` to push-recovery public metrics.
  - Full multi-split eval now emits `prompt_feedback.dominant_failure` from public split aggregates.
  - Legacy single-eval projection uses the same bounded prompt-feedback shape so it cannot drift from the full path.
  - The dominant-failure summary contains only status, source split, label, and rate. It does not include seed identifiers, per-episode data, artifact paths, or sealed validation internals.

- Upgraded `PushRecoveryPromptAdapter` public eval contract to v2.
  - The contract now describes the full prompt-facing evidence structure: `public_dev`, `public_accept`, `sealed_validation_gate`, and `non_prompt_validation`.
  - The contract states that eval seeds, holdout, eval profile, checkpoint policy, accept threshold, public-accept count, sealed-validation count, and runner eval argv are not candidate-writable.
  - Prompt-facing seed visibility is normalized to `counts_only_seed_identifiers_omitted`; the runner-owned config/protocol values were not changed.

- Added bounded eval feedback to `public_feedback_summary`.
  - Includes latest split stats, current-best comparison summary, failed-gate trend, latest dominant failure, sealed validation summary, and non-prompt validation status.
  - Drops raw `values` arrays, checkpoint report paths, and other low-level evidence details.

- Extended shared public history projection generically.
  - Added allowlisted `comparison`, `dominant_failure`, and `sealed_validation_gate` summaries.
  - Shared code only copies task-owned `metrics.prompt_feedback.dominant_failure`; it does not classify push-recovery failures or inspect task-specific metric names.
  - Shared comparison summaries keep only bounded stats and public current-best identifiers, omitting eval protocol hashes and raw value arrays.

## Boundary Decisions

- Task-specific semantics remain in `src/true_rl_agent/tasks/push_recovery`.
- `src/true_rl_agent/agent_core/context.py` only projects generic public fields and gate summaries.
- `sealed_validation_gate` is prompt-visible only as status evidence. Candidate contamination booleans, checkpoint report paths, and sealed rollout details remain out of the prompt.
- Public prompt feedback is richer but still bounded by existing public-history count/byte limits.

## Verification

- `python -m unittest tests.unit.test_push_recovery_task tests.unit.test_memory_context_phase13`
- `python -m unittest tests.unit.test_prompt_safety_boundaries tests.unit.test_prompt_projection_and_mechanism tests.unit.test_eval_public_projection tests.unit.test_eval_isolation tests.unit.test_runner_jobs`
- `python -m unittest discover -s tests\unit -p "test_*.py"`

All commands passed in the local workspace with the bundled Codex Python runtime.

## Audit Notes

- Static scan of `src/true_rl_agent/agent_core/context.py` found no push-recovery metric names such as fall, push-specific signal names, height, pitch, roll, drift, recovery-window, or planar-speed terms.
- Prompt adapter output tests assert no `raw_seed`, `hidden_eval`, `trusted_seed`, `checkpoint_selector`, Windows paths, or `/data/` paths in the prompt-facing sections.
- `PLAN/eval/Phase7` now follows the two-file phase convention: one implementation plan and one implementation record.

## Post-Review Correction

- Review found that `prompt_feedback` could fall back from `public_dev` to `public_accept` failure labels when public-dev had no usable failure breakdown.
- The fix keeps Phase7's declared boundary: prompt-facing failure diagnosis now uses direct legacy public breakdowns or `public_dev` only. `public_accept` remains prompt-visible as aggregate stats only.
- `PushTaskMemoryAdapter` and `PushRecoveryPromptAdapter` now use the same prompt-specific failure extractor/classifier, so task memory summary and task prompt summary cannot diverge on this boundary.

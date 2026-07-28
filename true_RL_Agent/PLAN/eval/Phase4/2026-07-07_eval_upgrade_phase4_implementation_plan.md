# Phase4 Eval Upgrade Implementation Plan

## Goal

Complete Phase4 by thickening the public-accept evidence used by acceptance decisions without moving task-specific push-recovery semantics into shared decision code.

## Current Boundary

- Phase0 fixed the canonical eval protocol fields and protocol hash.
- Phase1 removed eval-side candidate package paths and added candidate isolation evidence.
- Phase2 made checkpoint safety report ingestion fail-closed.
- Phase3 separated raw artifacts from public projections and moved public-safety checks to round boundaries.
- Phase4 must not migrate `current_best_score` to full current-best eval stats; that is Phase5.

## Design

1. Increase the push-recovery default `public_accept_seed_count` from `1` to `3`.
   - This mirrors the old Phase8E full-round shape that used three public-accept rollouts.
   - It is the smallest useful budget that can estimate sample variance while keeping eval cost bounded.
2. Extend push-recovery public metrics so `public_accept.success_rate_stats` contains:
   - `count`
   - `episode_count`
   - `mean`
   - `std`
   - `standard_error`
   - rounded public `values`
3. Add a task-owned `accept_evidence_gate`.
   - `pass` only when public-accept count meets the configured minimum and score stats are finite/trusted.
   - `fail` with reason `insufficient_accept_evidence` when public-accept count is too small.
   - `fail` with reason `accept_score_stats_untrusted` when standard error or mean is missing/non-finite.
4. Add `accept_evidence_gate` to `acceptance.required_pass_gates` for full public split projections.
   - Shared `decide_candidate()` already rejects failed required gates.
   - Shared code remains task-neutral; it consumes gate names and score stats only.
5. Add shared, task-neutral decision checks for acceptance metadata:
   - If `acceptance.min_score_count` is present and `score_stats.count` is below it, reject with `insufficient_accept_evidence`.
   - If both candidate and best score stats include `eval_protocol_hash`, reject protocol mismatches with `eval_protocol_hash_mismatch`.
   - Keep scalar `current_best_score` as the comparison input until Phase5.
6. Keep the legacy single-result dry-run builder compatible.
   - It may continue to emit `single_eval.success_rate` with count 1 for dry-run audits.
   - Formal full eval paths must use `build_push_public_result_from_summaries()`.

## Files

- Modify `src/true_rl_agent/tasks/push_recovery/config.py`
  - Set `public_accept_seed_count = 3`.
- Modify `src/true_rl_agent/tasks/push_recovery/metrics.py`
  - Add public accept minimum count constant.
  - Add `min_public_accept_count` argument to `build_push_public_result_from_summaries()`.
  - Add `episode_count` to split stats.
  - Add `accept_evidence_gate`.
  - Add the gate to full-eval `required_pass_gates`.
- Modify `src/true_rl_agent/agent_core/decision.py`
  - Add generic `min_score_count` and protocol hash checks over acceptance metadata.
- Modify tests:
  - `tests/unit/test_push_recovery_task.py`
  - `tests/unit/test_core_schema.py`
  - any protocol/default expectation tests affected by the count increase.

## TDD Steps

1. Add a push-recovery test proving a single public-accept summary yields:
   - `accept_evidence_gate.status == "fail"`
   - reason `insufficient_accept_evidence`
   - `accept_evidence_gate` appears in required pass gates
   - decision rejects instead of accepting a high public score.
2. Add a push-recovery test proving three public-accept summaries yield:
   - count 3
   - episode_count 24 in existing fixtures
   - finite `std` and `standard_error`
   - pass accept evidence gate
3. Add a shared decision test proving `min_score_count` rejects low-count acceptance metadata even if no task gate exists.
4. Add a shared decision test proving protocol hash mismatch rejects when both candidate and best stats expose hashes.
5. Run focused tests and confirm they fail before implementation.
6. Implement the minimal production changes.
7. Run focused tests until green.
8. Run full unit and integration suites.
9. Write Phase4 implementation record.

## Non-Goals

- Do not add push-recovery fall/drift/motion semantics to `agent_core.decision`.
- Do not replace `RoundInputs.current_best_score` in Phase4.
- Do not require best eval stats in normal current production path until Phase5.
- Do not expose raw seed ids, per-episode detail, checkpoint paths, hidden profiles, or internal rollout artifacts.

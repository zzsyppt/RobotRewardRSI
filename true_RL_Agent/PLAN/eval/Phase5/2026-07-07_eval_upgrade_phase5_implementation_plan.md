# Phase5 Eval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move acceptance from scalar current-best comparison to current-best eval stats with auditable comparison details and consumed gate evidence.

**Architecture:** Keep `agent_core.decision` task-neutral by comparing generic score stats, protocol hashes, and required gate statuses only. Put current-best state in a shared evaluation object restored from the public ledger. Keep push-recovery gate names and behavior semantics in task projection; shared decision only requires the common `sealed_validation_gate` and consumes whatever required gates the task declares.

**Tech Stack:** Python dataclasses, existing `unittest` tests, existing `PublicEvalResult`, `RoundInputs`, `run_live_loop`, and push-recovery projection helpers.

---

## File Structure

- Create `src/true_rl_agent/evaluation/current_best.py`
  - Owns generic `CurrentBestEvalStats`.
  - Restores accepted current-best stats from ledger public records.
  - Attaches eval protocol hash to candidate score stats after eval spec binding.
- Modify `src/true_rl_agent/agent_core/decision.py`
  - Accepts `CurrentBestEvalStats` instead of scalar-only best score.
  - Rejects policy accepts when `sealed_validation_gate` is not a required consumed gate.
  - Compares candidate public-accept stats against current-best public-accept stats outside bootstrap mode.
  - Records `comparison`, `consumed_gates`, and `gate_statuses` in `DecisionRecord`.
- Modify `src/true_rl_agent/agent_core/round.py`
  - Replaces `RoundInputs.current_best_score` with `current_best_eval_stats`.
  - Attaches eval protocol hash from executed `EvalJobSpec` before public projection is validated.
  - Passes current-best stats into `decide_candidate()`.
- Modify `src/true_rl_agent/agent_core/live_loop.py`, `src/true_rl_agent/agent_core/loop.py`, and `src/true_rl_agent/audit/loop_audit.py`
  - Restore current-best eval stats from ledger records each round.
  - Preserve `final_best_score` for compatibility, but derive it from `CurrentBestEvalStats`.
  - Add `final_best_eval_stats` to live summaries.
- Modify `src/true_rl_agent/tasks/push_recovery/metrics.py`
  - Add `sealed_validation_gate` to `acceptance.required_pass_gates` when the runner-projected metrics contain it.
- Modify `src/true_rl_agent/runner/dry_run.py` and `tests/fakes/fake_components.py`
  - Emit full-eval-shaped synthetic public metrics for dry-run/fake acceptance tests only.
- Modify tests:
  - `tests/unit/test_core_schema.py`
  - `tests/unit/test_current_best_eval_stats.py`
  - `tests/unit/test_push_recovery_task.py`
  - `tests/integration/test_fake_round.py`
  - `tests/integration/test_live_loop.py`
  - Existing call sites that instantiate `RoundInputs`.

## Tasks

### Task 1: CurrentBestEvalStats Contract

- [x] Add tests proving latest accepted ledger record restores:
  - `accepted_round_id`
  - `accepted_candidate_id`
  - `public_accept_stats`
  - `eval_protocol_hash`
  - `sealed_validation_gate`
  - required gate statuses
- [x] Add tests proving no accepted ledger record returns bootstrap mode from `initial_best_score`.
- [x] Implement `CurrentBestEvalStats` and ledger restoration helpers.

### Task 2: Decision Uses Stats, Not Scalar

- [x] Add tests proving a high `public_score` cannot accept when `acceptance.score_stats.mean` fails the noise-aware current-best stats comparison.
- [x] Add tests proving accepted decisions include comparison detail and consumed gate statuses.
- [x] Add tests proving missing `sealed_validation_gate` in `required_pass_gates` rejects before accept.
- [x] Add tests proving bootstrap mode accepts only after full eval gates and absolute floor pass.
- [x] Implement decision migration and `DecisionRecord` detail fields.

### Task 3: Round and Loop Wiring

- [x] Add integration tests proving `RoundInputs` passes current-best eval stats and ledger decisions store comparison detail.
- [x] Add live-loop test proving round 2 restores current-best stats from round 1 ledger, not from an in-memory scalar.
- [x] Attach eval protocol hash from executed `EvalJobSpec` to candidate score stats before decision.
- [x] Update `run_live_loop`, replay `AgentLoop`, and audit loop to restore/update `CurrentBestEvalStats`.

### Task 4: Push-Recovery Required Gate Wiring

- [x] Add task test proving a runner-projected `sealed_validation_gate` becomes a required pass gate after push-recovery public projection.
- [x] Implement `attach_push_budget_gate()` gate merging without hardcoding push-specific semantics into shared decision.
- [x] Update dry-run/fake runners to produce full-eval-shaped synthetic metrics for tests and dry-run audit only.

### Task 5: Verification and Record

- [x] Run the focused RED tests and confirm they fail before production code.
- [x] Run focused GREEN tests after implementation.
- [x] Run full unit tests, full integration tests, compileall, diff check, and static boundary probes.
- [x] Write `PLAN/eval/Phase5/2026-07-07_eval_upgrade_phase5_implementation_record.md`.

## Non-Goals

- Do not add push, fall, drift, base-height, roll, pitch, or recovery-window semantics to shared decision code.
- Do not make sealed validation a prompt-facing raw rollout split.
- Do not accept candidate-provided or eval-summary-provided best stats as the source of truth for current-best comparison.
- Do not remove compatibility summary fields such as `final_best_score`; keep them derived from stats.
- Do not implement Phase6 behavior-gate expansion beyond consuming required gates already projected by the task.

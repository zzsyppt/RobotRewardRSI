# Phase6 Eval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade push-recovery behavior evidence into required eval gates without moving task-specific semantics into shared decision logic.

**Architecture:** Push-recovery owns behavior metric parsing, behavior limits, failure breakdown parsing, and prompt diagnostic labels. Shared decision remains task-neutral and consumes only required gate names/statuses. Shared public history may expose compressed failed gate names/statuses, but does not interpret push/fall/drift/base-height semantics.

**Tech Stack:** Python dataclasses/helpers, existing `PublicEvalResult`, task-owned push-recovery projection, shared public context builder, `unittest`.

---

## File Structure

- Modify `src/true_rl_agent/tasks/push_recovery/metrics.py`
  - Add `behavior_quality_gate` as a task-owned required gate.
  - Validate failure breakdown parseability for public-dev/public-accept summaries.
  - Require finite behavior metrics for action magnitude/rate, min base height, max roll/pitch, final xy drift, and final planar speed when present in the eval summary contract.
  - Keep `motion_quality_gate` for compatibility and as a separate required motion sanity gate.
- Modify `scripts/isaaclab_rsl_rl_eval.py`
  - Emit `mean_final_planar_speed_mps` and per-episode `final_planar_speed_mps` from actual eval state.
  - Do not change the success predicate in Phase6.
- Modify `src/true_rl_agent/agent_core/context.py`
  - Add generic compressed `failed_gates` public-history field derived from `decision.gate_statuses`.
  - Do not add push-specific labels or behavior metric interpretation to shared context.
- Modify `tests/unit/test_push_recovery_task.py`
  - Add RED tests for behavior gate requiredness, missing/nonfinite behavior metrics, malformed failure breakdown, and eval-script source output.
- Modify `tests/unit/test_memory_context_phase13.py`
  - Add RED test proving prompt public history retains failed gate names/statuses in compressed form.
- Modify `tests/fakes/fake_components.py` and `src/true_rl_agent/runner/dry_run.py`
  - Add synthetic `behavior_quality_gate` for fake/dry-run full-eval-shaped metrics.

## Tasks

### Task 1: Behavior Gate Projection

- [x] Add tests proving projected public eval metrics include `behavior_quality_gate`.
- [x] Add tests proving `behavior_quality_gate` is in `acceptance.required_pass_gates`.
- [x] Add tests proving missing or nonfinite required behavior metrics fail the behavior gate.
- [x] Implement task-owned behavior metric validation.

### Task 2: Failure Breakdown Parse Gate

- [x] Add tests proving malformed failure breakdown fails `behavior_quality_gate`.
- [x] Add tests proving parse failure does not leak raw seed ids or hidden source markers.
- [x] Implement explicit failure breakdown parseability checks in push-recovery projection.

### Task 3: Eval Source Metric Output

- [x] Add source-level test proving `scripts/isaaclab_rsl_rl_eval.py` emits `final_planar_speed_mps` and `mean_final_planar_speed_mps`.
- [x] Implement final planar speed tracking in the eval script.
- [x] Keep success predicate unchanged in Phase6.

### Task 4: Generic Public History Failed Gates

- [x] Add tests proving public history includes compressed failed gate statuses.
- [x] Add tests proving shared public history does not add task-specific behavior fields.
- [x] Implement generic failed gate projection from `decision.gate_statuses`.

### Task 5: Fake/Dry-Run Compatibility

- [x] Update fake runner metrics to include passing `behavior_quality_gate`.
- [x] Update dry-run audit metrics to include passing synthetic `behavior_quality_gate`.
- [x] Verify round/live tests still consume required gates.

### Task 6: Review, Verification, and Record

- [x] Run focused RED tests before implementation.
- [x] Run focused GREEN tests after implementation.
- [x] Audit shared layers for task-specific terms.
- [x] Run full unit/integration tests, compileall, materialize smoke, static boundary probes, and `git diff --check`.
- [x] Write `PLAN/eval/Phase6/2026-07-07_eval_upgrade_phase6_implementation_record.md`.

## Non-Goals

- Do not change the push-recovery success predicate in Phase6.
- Do not put fall, timeout, drift, base height, roll, pitch, recovery, or push semantics into `agent_core.decision`.
- Do not let candidates or prompt payloads choose behavior thresholds.
- Do not expose raw per-env/per-episode records, raw seeds, hidden profiles, local paths, or runner-private config in prompt history.
- Do not replace full eval with smoke or fake behavior checks.

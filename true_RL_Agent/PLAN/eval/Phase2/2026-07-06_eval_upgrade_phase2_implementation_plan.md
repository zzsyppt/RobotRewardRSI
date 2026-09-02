# Phase2 Checkpoint Safety Seal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sealed validation depend on eval-process checkpoint byte safety evidence, while adding a task-neutral evidence manifest that can later carry multimodal artifacts safely.

**Architecture:** Phase2 keeps task semantics out of shared code. The shared runner/safety layer owns checkpoint byte reports, seal hashing, evidence artifact manifests, and eval summary ingestion. Push recovery remains responsible only for rollout metrics and diagnostic content; prompt/history still receive only public-safe projections.

**Tech Stack:** Python dataclasses, `unittest`, runner-owned subprocess summaries, IsaacLab eval CLI, existing `EvaluationProtocol`, `EvalJobSpec`, `TaskInvariantSpec`, and `PublicEvalResult`.

---

## Context And Boundaries

- Phase0 froze canonical eval protocol fields.
- Phase1 removed eval-side candidate package paths and made candidate isolation fail-closed.
- Phase2 must not implement sealed holdout rollout, current-best stats, behavior gates, or prompt-facing multimodal feedback; those are Phase3+.
- Phase2 must not introduce push recovery-specific terms into shared runner/safety modules.
- Future multimodal support is represented only as a generic evidence artifact manifest:
  - artifact kind, mime type, hash, byte size, producer, visibility, projection ref.
  - no local path, raw seed, host path, hidden profile, raw video/image bytes, or task-specific semantic label in public projection.

## Files

- Create: `src/true_rl_agent/runner/evidence.py`
  - Task-neutral public-safe evidence artifact manifest helpers.
- Modify: `src/true_rl_agent/runner/checkpoint.py`
  - Add byte-level `CheckpointSafetyReport`, file hashing, non-raising report builder, and report validator.
- Modify: `src/true_rl_agent/safety/eval_seal.py`
  - Build trusted eval seal from `CheckpointSafetyReport` instead of structural `safe_load_checkpoint()`.
- Modify: `src/true_rl_agent/runner/subprocess_runner.py`
  - Require and validate `checkpoint_safety_report` before public metric projection.
- Modify: `src/true_rl_agent/runner/eval_isolation.py`
  - Keep candidate isolation logic task-neutral; only allow optional gate evidence fields to be checked by runner/safety code.
- Modify: `src/true_rl_agent/backends/isaac_lab_rsl_rl.py`
  - Pass expected checkpoint sha256 to eval CLI.
- Modify: `scripts/isaaclab_rsl_rl_eval.py`
  - Generate `checkpoint_safety_report` before `runner.load()` and attach safe evidence manifest.
- Modify: `tests/unit/test_runner_jobs.py`
  - Red/green coverage for checkpoint report and seal construction.
- Modify: `tests/unit/test_subprocess_runner.py`
  - Red/green coverage for missing/failing safety report rejection and public-safe projection.
- Modify: `tests/unit/test_isaac_lab_backend.py`
  - Assert eval argv passes expected checkpoint hash.
- Modify: `tests/unit/test_push_recovery_task.py`
  - Assert eval script uses report before `runner.load()` and records public-safe fields.
- Create: `tests/unit/test_evidence_artifacts.py`
  - Red/green coverage for future multimodal-safe manifest shape.
- Create: `PLAN/eval/Phase2/2026-07-06_eval_upgrade_phase2_implementation_record.md`
  - Implementation record after verification.

## Task 1: Evidence Manifest Extension Point

- [ ] Write `tests/unit/test_evidence_artifacts.py`.
  - `test_json_evidence_artifact_is_hash_only_and_public_safe`
  - `test_multimodal_artifact_manifest_shape_has_no_path_or_raw_payload`
  - `test_evidence_manifest_rejects_path_url_or_raw_payload_keys`
- [ ] Run the test and confirm RED because `true_rl_agent.runner.evidence` does not exist.
- [ ] Add `runner/evidence.py` with:
  - `EvidenceArtifact`
  - `evidence_artifact_from_json()`
  - `validate_evidence_artifact_manifest()`
  - public-safe fields only.
- [ ] Re-run evidence tests and confirm GREEN.

## Task 2: Checkpoint Byte Safety Report

- [ ] Add failing tests in `tests/unit/test_runner_jobs.py`:
  - report records sha256, file existence, byte read, weights-only status, and public-safe error class.
  - report fails when expected checksum mismatches.
  - report validator rejects missing or failing report.
- [ ] Run targeted tests and confirm RED.
- [ ] Extend `runner/checkpoint.py`:
  - compute file sha256 from bytes.
  - build non-raising `CheckpointSafetyReport`.
  - support dependency injection for unit tests without requiring torch.
  - default loader uses `torch.load(..., weights_only=True)` only; no unsafe fallback.
  - validate public-safe mapping shape.
- [ ] Re-run targeted tests and confirm GREEN.

## Task 3: Seal From Eval-Process Safety Report

- [ ] Update failing tests in `tests/unit/test_runner_jobs.py`:
  - `build_trusted_eval_seal()` requires `checkpoint_safety_report`.
  - seal hash changes when safety status/report changes.
  - unsafe or missing report cannot build seal.
- [ ] Run tests and confirm RED.
- [ ] Update `safety/eval_seal.py`:
  - remove internal `safe_load_checkpoint()` call from seal construction.
  - accept `checkpoint_safety_report`.
  - include `checkpoint_safety_status`, `checkpoint_safety_report_hash`, `candidate_isolation_status`, and `seal_hash`.
  - keep existing public fields where safe for compatibility.
- [ ] Re-run targeted tests and confirm GREEN.

## Task 4: Subprocess Eval Summary Ingestion

- [ ] Add failing tests in `tests/unit/test_subprocess_runner.py`:
  - missing `checkpoint_safety_report` rejects.
  - report status fail rejects.
  - gate checkpoint safety status fail rejects.
  - pass report projects only public-safe report and evidence manifest.
- [ ] Run tests and confirm RED.
- [ ] Update `SubprocessRunner._public_result_from_summary()`:
  - validate safety report before public projection.
  - require report status pass.
  - require `sealed_validation_gate.checkpoint_safety_status == "pass"`.
  - project `checkpoint_safety_status`, `checkpoint_safety_report`, and safe `evidence_manifest`.
- [ ] Re-run targeted tests and confirm GREEN.

## Task 5: Eval CLI Report Generation

- [ ] Add/update failing tests:
  - backend eval argv includes `--checkpoint-sha256`.
  - eval script source calls report builder before `runner.load()`.
  - eval summary records `checkpoint_safety_report`, `checkpoint_safety_status`, `sealed_validation_gate.checkpoint_safety_status`, and `evidence_manifest`.
- [ ] Run targeted tests and confirm RED.
- [ ] Update backend eval argv to pass `checkpoint.checksum`.
- [ ] Update `scripts/isaaclab_rsl_rl_eval.py`:
  - parser requires `--checkpoint-sha256`.
  - build report before environment/runner checkpoint load.
  - fail before `runner.load()` if report status is not pass.
  - attach report and safe evidence manifest to pass and fail summaries.
- [ ] Re-run targeted tests and confirm GREEN.

## Task 6: Record And Verification

- [ ] Write Phase2 implementation record with:
  - actual files changed.
  - red/green evidence.
  - boundary notes.
  - multimodal extension boundary.
- [ ] Run focused Phase2 suite:
  - `python -m unittest tests.unit.test_evidence_artifacts tests.unit.test_runner_jobs tests.unit.test_subprocess_runner tests.unit.test_isaac_lab_backend tests.unit.test_push_recovery_task -v`
- [ ] Run full suite:
  - `python -m unittest discover -s tests -v`
- [ ] Run compile check:
  - `python -m compileall -q src scripts tests`
- [ ] Run whitespace check:
  - `git diff --check`

## Self-Review

- Spec coverage:
  - checkpoint report enters summary and runner validation.
  - seal construction depends on eval-process report.
  - seal hash covers safety status/report and candidate isolation status.
  - artifact manifest gives future multimodal extension without exposing raw artifacts.
- Placeholder scan: no TBD/TODO placeholders.
- Boundary scan:
  - shared layer contains no push recovery failure labels.
  - prompt-facing projection gets no raw path, raw payload, hidden profile, seed, image, or video bytes.
  - task cannot downgrade eval isolation or checkpoint policy.

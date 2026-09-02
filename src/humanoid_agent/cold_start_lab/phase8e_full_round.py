"""Full Phase 8E code-level cold-start round orchestration.

This module is intentionally a thin runner-owned coordinator.  Candidate code
may define training components, but it never owns verification, budget,
checkpoint selection, rollout configuration, trusted eval, or ledger decisions.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.code_sandbox.isolation import sanitized_environment
from humanoid_agent.code_sandbox.verifier import verify_candidate_package
from humanoid_agent.immutable_runner.train_start_paths import TRAIN_START_CHECKPOINT_PREFIX

from .phase8e import Phase8EError, append_ledger_record, build_ledger_record
from .curriculum_compiler import (
    CurriculumCompileError,
    compile_v1_curriculum_for_training,
    enrich_curriculum_summary_with_immutable_context,
    project_curriculum_compile_summary,
)
from .phase8e_linked import Phase8ELinkedError, resolve_train_start_checkpoint_for_eval, run_phase8e_linked_round
from .phase8e_provider import run_phase8e_provider_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FULL_ROUND_SUMMARY_SCHEMA = "humanoid_agent.phase8e.full_round.summary.v1"


class Phase8EFullRoundError(RuntimeError):
    """Raised when the full Phase 8E round cannot safely continue."""

    def __init__(self, message: str, *, status: str = "fail", reason: str | None = None, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason or status
        self.exit_code = exit_code if exit_code is not None else _exit_code_for_status(status)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    timed_out: bool
    elapsed_seconds: float
    log_file: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "log_file": self.log_file,
        }


CommandRunner = Callable[[list[str], Path, dict[str, str], Path, int], CommandResult]
ProviderRunner = Callable[..., dict[str, Any]]
CheckpointResolver = Callable[..., dict[str, Any]]
LinkedRunner = Callable[..., tuple[int, dict[str, Any]]]


def run_phase8e_full_round(
    *,
    config_path: Path,
    run_dir: Path,
    lab_dir: Path | None = None,
    candidate_dir: Path | None = None,
    provider_name: str | None = None,
    replay_response: Path | None = None,
    allow_api_spend: bool = False,
    append_ledger: bool = False,
    inherit_if_accepted: bool = False,
    require_server_enforcement: bool = True,
    python_executable: str | None = None,
    command_runner: CommandRunner | None = None,
    provider_runner: ProviderRunner = run_phase8e_provider_candidate,
    checkpoint_resolver: CheckpointResolver = resolve_train_start_checkpoint_for_eval,
    linked_runner: LinkedRunner = run_phase8e_linked_round,
) -> tuple[int, dict[str, Any]]:
    """Run candidate -> verifier -> fixed-budget train -> rollout -> decision."""

    started = time.monotonic()
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_mapping(config_path)
    budget_config = _budget_config(config)
    runner_cfg = _mapping(budget_config.get("runner"), "budget.runner")
    full_round_cfg = _mapping(config.get("full_round", {}), "full_round")
    command_runner = command_runner or _run_subprocess
    python_executable = python_executable or sys.executable

    summary: dict[str, Any] = {
        "schema_version": FULL_ROUND_SUMMARY_SCHEMA,
        "status": "running",
        "created_at_utc": _now(),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "stages": {},
        "errors": [],
        "formal_training_executed": False,
        "formal_training": {
            "status": "not_started",
            "source": "full_round_fixed_budget_training_stage",
            "note": "Nested linked/decision summaries may keep formal_training_executed=false because those subprocesses do not train.",
        },
        "candidate_code_scope": {
            "controls_training_components_only": True,
            "runner_owns_budget_eval_checkpoint_and_decision": True,
        },
    }
    summary_path = run_dir / str(full_round_cfg.get("summary_filename", "phase8e_full_round_summary.json"))
    summary["summary_file"] = str(summary_path)

    try:
        provider_summary, resolved_candidate_dir = _prepare_candidate(
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            lab_dir=lab_dir,
            candidate_dir=candidate_dir,
            provider_name=provider_name,
            replay_response=replay_response,
            allow_api_spend=allow_api_spend,
            require_server_enforcement=require_server_enforcement,
            python_executable=python_executable,
            provider_runner=provider_runner,
        )
        summary["stages"]["candidate"] = _project_candidate_stage(provider_summary, resolved_candidate_dir)
        if provider_summary.get("status") != "pass" or provider_summary.get("training_ready") is not True:
            pretraining_reason = _provider_pretraining_reject_reason(provider_summary)
            feedback = _append_failure_feedback(
                config=config,
                run_dir=run_dir,
                lab_dir=lab_dir,
                provider_summary=provider_summary,
                candidate_dir=resolved_candidate_dir,
                append_ledger=append_ledger,
                status="reject",
                reason=pretraining_reason,
                detail={
                    "stage": "provider_or_phase8d_gate",
                    "training_started": False,
                    "verifier_status": provider_summary.get("verifier_status"),
                    "public_failure_feedback": _public_failure_feedback(
                        stage="provider_or_phase8d_gate",
                        verifier_status=provider_summary.get("verifier_status"),
                    ),
                },
            )
            if feedback is not None:
                summary["stages"]["pretraining_feedback"] = feedback
            raise Phase8EFullRoundError(
                f"candidate is not training-ready after provider/8D gate: status={provider_summary.get('status')}, "
                f"verifier={provider_summary.get('verifier_status')}",
                status="reject",
                reason=pretraining_reason,
            )

        curriculum_compile = _compile_curriculum_for_training(
            config=config,
            config_path=config_path,
            budget_config=budget_config,
            provider_summary=provider_summary,
            run_dir=run_dir,
            candidate_dir=resolved_candidate_dir,
        )
        summary["stages"]["curriculum_compile"] = project_curriculum_compile_summary(curriculum_compile)
        if curriculum_compile.get("status") != "pass":
            feedback = _append_failure_feedback(
                config=config,
                run_dir=run_dir,
                lab_dir=lab_dir,
                provider_summary=provider_summary,
                candidate_dir=resolved_candidate_dir,
                append_ledger=append_ledger,
                status="reject",
                reason="curriculum_compile_reject",
                detail={
                    "stage": "curriculum_compile",
                    "training_started": False,
                    "curriculum_compile_status": curriculum_compile.get("status"),
                },
            )
            if feedback is not None:
                summary["stages"]["curriculum_feedback"] = feedback
            raise Phase8EFullRoundError(
                f"runner curriculum compiler rejected candidate: {curriculum_compile.get('reject_reason')}",
                status="reject",
                reason="curriculum_compile_reject",
            )

        training = _run_fixed_budget_training(
            config=config,
            config_path=config_path,
            budget_config=budget_config,
            run_dir=run_dir,
            candidate_dir=resolved_candidate_dir,
            curriculum_compile=curriculum_compile,
            command_runner=command_runner,
            python_executable=python_executable,
        )
        summary["stages"]["training"] = training["public_summary"]
        if training["status"] != "pass":
            _update_formal_training_summary(summary, training)
            training_status = str(training["status"])
            public_reason = _public_training_failure_reason(training_status)
            detail = {
                "stage": "fixed_budget_training",
                "training_started": True,
                "training_status": training_status,
                "public_failure_feedback": training.get("public_failure_feedback"),
            }
            feedback = _append_failure_feedback(
                config=config,
                run_dir=run_dir,
                lab_dir=lab_dir,
                provider_summary=provider_summary,
                candidate_dir=resolved_candidate_dir,
                append_ledger=append_ledger,
                status=training_status,
                reason=public_reason,
                detail=detail,
            )
            if feedback is not None:
                summary["stages"]["training_feedback"] = feedback
            raise Phase8EFullRoundError(
                f"fixed-budget isolated training did not pass: {training['reason']}",
                status=training_status,
                reason=public_reason,
            )
        _update_formal_training_summary(summary, training)

        checkpoint_for_eval = checkpoint_resolver(
            training["child_summary"],
            runner_artifacts_root=Path(training["runner_artifacts_root"]),
            validate_safe_load=True,
        )
        summary["stages"]["checkpoint_for_eval"] = _project_checkpoint_for_eval(checkpoint_for_eval)

        rollout_paths = _run_public_rollouts(
            config=config,
            budget_runner_cfg=runner_cfg,
            full_round_cfg=full_round_cfg,
            run_dir=run_dir,
            checkpoint_path=Path(checkpoint_for_eval["host_path"]),
            command_runner=command_runner,
            python_executable=python_executable,
        )
        summary["stages"]["public_rollouts"] = {
            "status": "pass",
            "public_dev_summary_count": len(rollout_paths["public_dev"]),
            "public_accept_summary_count": len(rollout_paths["public_accept"]),
            "summary_paths": {
                "public_dev": [str(path) for path in rollout_paths["public_dev"]],
                "public_accept": [str(path) for path in rollout_paths["public_accept"]],
            },
        }

        linked_exit, linked_summary = linked_runner(
            config_path=config_path,
            run_dir=run_dir / "linked_round",
            lab_dir=lab_dir.resolve() if lab_dir is not None else None,
            candidate_dir=resolved_candidate_dir,
            train_summary_path=Path(training["child_summary_file"]),
            runner_artifacts_root=Path(training["runner_artifacts_root"]),
            public_dev_summary_paths=rollout_paths["public_dev"],
            public_accept_summary_paths=rollout_paths["public_accept"],
            curriculum_lineage_path=(
                Path(curriculum_compile["curriculum_lineage_file"])
                if isinstance(curriculum_compile.get("curriculum_lineage_file"), str)
                else None
            ),
            append_ledger=append_ledger,
            inherit_if_accepted=inherit_if_accepted,
            require_server_enforcement=require_server_enforcement,
        )
        summary["stages"]["linked_round"] = {
            "status": linked_summary.get("status"),
            "exit_code": linked_exit,
            "summary_file": str((run_dir / "linked_round" / "phase8e_linked_round_summary.json").resolve()),
        }
        summary["status"] = str(linked_summary.get("status", "unknown"))
        exit_code = int(linked_exit)
    except Phase8EFullRoundError as exc:
        summary["status"] = exc.status
        summary["failure_classification"] = {"status": exc.status, "reason": exc.reason}
        summary["errors"].append({"type": type(exc).__name__, "message": str(exc), "status": exc.status, "reason": exc.reason})
        exit_code = exc.exit_code
    except Phase8ELinkedError as exc:
        status = _classify_linked_error(exc)
        summary["status"] = status
        summary["failure_classification"] = {"status": status, "reason": _reason_for_linked_error(exc)}
        summary["errors"].append({"type": type(exc).__name__, "message": str(exc), "status": status})
        exit_code = _exit_code_for_status(status)
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator must persist a summary.
        summary["status"] = "fail"
        summary["failure_classification"] = {"status": "fail", "reason": "unexpected_orchestrator_exception"}
        summary["errors"].append({"type": type(exc).__name__, "message": str(exc), "status": "fail"})
        exit_code = 1
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 6)
        summary["completed_at_utc"] = _now()
        _write_json(summary_path, summary)

    return exit_code, summary


def _prepare_candidate(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    run_dir: Path,
    lab_dir: Path | None,
    candidate_dir: Path | None,
    provider_name: str | None,
    replay_response: Path | None,
    allow_api_spend: bool,
    require_server_enforcement: bool,
    python_executable: str | None,
    provider_runner: ProviderRunner,
) -> tuple[dict[str, Any], Path]:
    if candidate_dir is not None:
        resolved = candidate_dir.resolve()
        verifier_summary = verify_candidate_package(
            candidate_dir=resolved,
            config_path=_resolve_project_path(config["verifier"]["config_path"]),
            run_dir=run_dir / "phase8d_verifier",
            require_server_enforcement=require_server_enforcement,
            python_executable=python_executable,
        )
        summary = {
            "status": "pass" if verifier_summary.get("status") == "pass" else "fail",
            "training_ready": verifier_summary.get("status") == "pass",
            "candidate_dir": str(resolved),
            "verifier_status": verifier_summary.get("status"),
            "verifier": verifier_summary,
            "source": "preexisting_candidate_dir",
        }
    else:
        summary = provider_runner(
            config_path=config_path,
            run_dir=run_dir / "provider_candidate",
            lab_dir=lab_dir.resolve() if lab_dir is not None else None,
            provider_name=provider_name,
            replay_response=replay_response,
            allow_api_spend=allow_api_spend,
            verify=True,
            require_server_enforcement=require_server_enforcement,
            python_executable=python_executable,
        )
        resolved = Path(summary.get("candidate_dir", "")).resolve()

    if not resolved.is_dir() and (summary.get("status") != "pass" or summary.get("training_ready") is not True):
        return summary, resolved
    if not resolved.is_dir():
        raise Phase8EFullRoundError(f"candidate_dir missing after provider/8D gate: {resolved}", status="reject", reason="candidate_dir_missing")
    return summary, resolved


def _provider_pretraining_reject_reason(provider_summary: Mapping[str, Any]) -> str:
    inner_loop = provider_summary.get("inner_loop")
    if isinstance(inner_loop, Mapping):
        reason = inner_loop.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return "candidate_not_training_ready_after_verifier"


def _load_provider_prompt_payload(
    provider_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Path | None, Mapping[str, Any] | None]:
    path = _provider_prompt_payload_path(provider_summary, config)
    if path is None or not path.is_file():
        return path, None
    payload = load_mapping(path)
    return path, payload if isinstance(payload, Mapping) else None


def _provider_prompt_payload_path(provider_summary: Mapping[str, Any], config: Mapping[str, Any]) -> Path | None:
    prompt = provider_summary.get("prompt")
    if isinstance(prompt, Mapping) and isinstance(prompt.get("payload_file"), str):
        return Path(prompt["payload_file"]).resolve()
    run_dir = provider_summary.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        filename = "phase8e_prompt_payload.json"
        paths = config.get("paths")
        if isinstance(paths, Mapping):
            filename = str(paths.get("prompt_payload_filename", filename))
        return (Path(run_dir) / filename).resolve()
    return None


def _compile_curriculum_for_training(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    budget_config: Mapping[str, Any],
    provider_summary: Mapping[str, Any],
    run_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    runner_cfg = _mapping(budget_config.get("runner"), "budget.runner")
    budget_contract = _mapping(budget_config.get("budget_contract"), "budget.budget_contract")
    full_round_cfg = _mapping(config.get("full_round", {}), "full_round")
    train_cfg = _mapping(full_round_cfg.get("training"), "full_round.training")
    rollout_cfg = _mapping(full_round_cfg.get("public_rollout"), "full_round.public_rollout")
    phase1_config_path = _resolve_project_path(runner_cfg.get("phase1_config", "configs/phase1_push_recovery.yaml"))
    phase2_config_path = _resolve_project_path(runner_cfg.get("phase2_config", "configs/phase2_baseline.yaml"))
    try:
        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate_dir,
            phase1_config_path=phase1_config_path,
            phase2_config_path=phase2_config_path,
            output_dir=run_dir / "fixed_budget_training" / "curriculum_compiler",
            train_profile=str(runner_cfg.get("train_profile", "phase2_train")),
            max_iterations=int(runner_cfg["max_iterations"]),
            steps_per_iteration=int(budget_contract["steps_per_iteration"]),
        )
        prompt_payload_path, prompt_payload = _load_provider_prompt_payload(provider_summary, config)
        effective = summary.get("effective_curriculum")
        summary = enrich_curriculum_summary_with_immutable_context(
            summary,
            prompt_payload=prompt_payload,
            prompt_payload_path=prompt_payload_path,
            artifacts={
                "loop_config_path": config_path,
                "budget_config_path": _resolve_project_path(config["runner"]["budget_config_path"]),
                "original_phase1_config_path": phase1_config_path,
                "original_phase2_config_path": phase2_config_path,
                "compiled_phase1_config_path": summary.get("phase1_config_path"),
                "compiled_phase2_config_path": summary.get("phase2_config_path"),
                "training_script_path": PROJECT_ROOT / str(train_cfg.get("script", "scripts/phase8e_minimal_rootfs_probe.py")),
                "eval_script_path": PROJECT_ROOT / str(rollout_cfg.get("script", "scripts/phase1_push_recovery_eval.py")),
                "eval_config_path": phase1_config_path,
                "train_profile": runner_cfg.get("train_profile", "phase2_train"),
                "planned_policy_iterations": effective.get("planned_policy_iterations") if isinstance(effective, Mapping) else None,
                "planned_training_steps": effective.get("planned_training_steps") if isinstance(effective, Mapping) else None,
                "rollout_profile": rollout_cfg.get("profile", "phase2_public"),
                "rollout_num_envs": rollout_cfg.get("num_envs", 64),
                "rollout_max_steps": rollout_cfg.get("max_steps"),
                "rollout_seed_visibility": rollout_cfg.get("seed_visibility"),
                "public_dev_seeds": _int_sequence(rollout_cfg.get("dev_seeds", [1101, 1102, 1103])),
                "public_accept_seeds": _int_sequence(rollout_cfg.get("accept_seeds", [2101, 2102, 2103])),
            },
        )
        return summary
    except CurriculumCompileError as exc:
        return exc.summary or {
            "status": "reject",
            "schema_version": "humanoid_agent.phase8e.curriculum_compiler.summary.v1",
            "compiler_version": None,
            "summary_file": None,
            "reject_reason": str(exc),
        }


def _append_failure_feedback(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    lab_dir: Path | None,
    provider_summary: Mapping[str, Any],
    candidate_dir: Path,
    append_ledger: bool,
    status: str,
    reason: str,
    detail: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Append one public failure-feedback ledger record for the next round.

    Used by both failure paths (provider/8D gate before training, and the
    fixed-budget training stage).  It records only public verifier status/issue
    codes, the candidate's public package metadata, and a runner-owned ``detail``
    supplied by the caller; no eval seeds, hidden metrics, raw candidate error
    text, or runner-owned internals are exposed to the next prompt.  The whole
    record passes ``assert_public_only_payload`` inside ``build_ledger_record``.
    """

    if not append_ledger:
        return None
    if not candidate_dir.is_dir():
        return {"status": "skipped", "reason": "candidate_dir_missing"}

    candidate = provider_summary.get("candidate")
    if not isinstance(candidate, Mapping):
        candidate = {
            "candidate_dir": str(candidate_dir.resolve()),
            "manifest": None,
            "fingerprint": None,
            "operation_diff": None,
            "diagnostics": None,
        }
    verifier = provider_summary.get("verifier")
    if not isinstance(verifier, Mapping):
        verifier = {"status": provider_summary.get("verifier_status"), "issues": []}

    decision = {
        "schema_version": "humanoid_agent.phase8e.public_decision.v1",
        "created_at_utc": _now(),
        "decision": status,
        "accepted": False,
        "reason": reason,
        "candidate_dir": str(candidate_dir.resolve()),
        "verifier": provider_summary.get("verifier_public_projection"),
        "public_result": None,
        "detail": dict(detail),
        "feedback_safe": {
            "public_only": True,
            "private_eval_used": False,
            "eval_seed_config_included": False,
        },
    }
    target_lab_dir = lab_dir.resolve() if lab_dir is not None else run_dir
    ledger_path = target_lab_dir / str(_mapping(config.get("paths"), "paths").get("ledger_filename", "phase8e_ledger.jsonl"))
    redacted = False
    try:
        record = build_ledger_record(dict(config), dict(candidate), dict(verifier), decision)
    except Phase8EError:
        # A caller-supplied free-text detail (e.g. a candidate training-error
        # message) tripped the public-only guard.  Drop the free text and record
        # status-only feedback so the round is not lost over a coincidental token.
        safe_detail = {k: v for k, v in dict(detail).items() if k in ("stage", "training_started", "training_status", "verifier_status")}
        safe_detail["redacted"] = "detail_tripped_public_only_guard"
        safe_decision = dict(decision)
        safe_decision["detail"] = safe_detail
        record = build_ledger_record(dict(config), dict(candidate), dict(verifier), safe_decision)
        redacted = True
    append_ledger_record(ledger_path, record)
    return {
        "status": "appended",
        "ledger_path": str(ledger_path),
        "reason": reason,
        "public_only": True,
        "detail_redacted": redacted,
    }


# The structured fields (component_name, exception type, shape_mismatch, stack
# frame) carry the actionable signal; the free-text message is a bounded human-
# readable supplement.  512 leaves room for verbose runtime errors (e.g. torch
# broadcast messages) without letting a candidate stuff the prompt.
PUBLIC_FAILURE_MESSAGE_MAX_CHARS = 512


def _public_failure_feedback(
    *,
    stage: str,
    training_status: str | None = None,
    verifier_status: Any = None,
    training_error: Mapping[str, Any] | None = None,
    checkpoint_missing_due_to_training_failure: bool | None = None,
) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        "schema_version": "humanoid_agent.phase8e.public_failure_feedback.v1",
        "stage": stage,
        "evidence_granularity": "stage_level_public_summary",
    }
    if training_status is not None:
        feedback["training_status"] = str(training_status)
    if verifier_status is not None:
        feedback["verifier_status"] = verifier_status
    if checkpoint_missing_due_to_training_failure is not None:
        feedback["checkpoint_missing_due_to_training_failure"] = bool(checkpoint_missing_due_to_training_failure)

    if isinstance(training_error, Mapping):
        component_name = _first_text(training_error, ("component_name", "term_name", "name"))
        if component_name:
            feedback["component_name"] = component_name
            feedback["evidence_granularity"] = "component_level_public_training_error"
        exception_type = _first_text(training_error, ("type", "exception_type", "class"))
        message = _first_text(training_error, ("message", "public_message", "error"), max_chars=PUBLIC_FAILURE_MESSAGE_MAX_CHARS)
        exception: dict[str, str] = {}
        if exception_type:
            exception["type"] = exception_type
        if message:
            exception["message"] = message
        if exception:
            feedback["exception"] = exception
        frame = _public_candidate_stack_frame(training_error.get("candidate_stack_frame"))
        if frame is not None:
            feedback["candidate_stack_frame"] = frame
        shape_mismatch = _public_shape_mismatch(training_error.get("shape_mismatch"))
        if shape_mismatch is not None:
            feedback["shape_mismatch"] = shape_mismatch
    return feedback


def _first_text(mapping: Mapping[str, Any], keys: Sequence[str], *, max_chars: int = 160) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:max_chars]
    return None


def _public_candidate_stack_frame(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    path = value.get("path")
    if not isinstance(path, str) or not path.startswith("training_code/") or ".." in Path(path).parts:
        return None
    frame: dict[str, Any] = {"path": path}
    line = value.get("line")
    if isinstance(line, int) and line > 0:
        frame["line"] = line
    function = value.get("function")
    if isinstance(function, str) and function.strip():
        frame["function"] = function.strip()[:160]
    return frame


def _public_shape_mismatch(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in ("lhs_shape", "rhs_shape", "expected_shape", "actual_shape"):
        shape = value.get(key)
        if _is_public_shape(shape):
            out[key] = list(shape)
    return out or None


def _is_public_shape(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in value)


# Runner-owned training-failure classes are the only training reasons exposed to
# the next prompt.  Raw training reasons may embed candidate-influenced error
# text, so the feedback reason is derived solely from the runner classification.
_PUBLIC_TRAINING_FAILURE_STATUSES = frozenset({"blocked", "reject", "failed_safety_gate", "fail"})


def _public_training_failure_reason(status: str) -> str:
    safe = status if status in _PUBLIC_TRAINING_FAILURE_STATUSES else "fail"
    return f"fixed_budget_training_{safe}"


def _run_fixed_budget_training(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    budget_config: Mapping[str, Any],
    run_dir: Path,
    candidate_dir: Path,
    curriculum_compile: Mapping[str, Any],
    command_runner: CommandRunner,
    python_executable: str,
) -> dict[str, Any]:
    runner_cfg = _mapping(budget_config.get("runner"), "budget.runner")
    budget_contract = _mapping(budget_config.get("budget_contract"), "budget.budget_contract")
    full_round_cfg = _mapping(config.get("full_round", {}), "full_round")
    train_cfg = _mapping(full_round_cfg.get("training"), "full_round.training")
    minimal_run_dir = run_dir / "fixed_budget_training" / "minimal_rootfs"
    log_file = run_dir / "fixed_budget_training" / "runner_artifacts" / "phase8e_full_round_training.log"
    summary_filename = str(train_cfg.get("summary_filename", runner_cfg.get("train_summary_filename", "train_start_smoke_summary.json")))
    child_summary_file = minimal_run_dir / "candidate_output" / summary_filename
    runner_artifacts_root = minimal_run_dir / "runner_artifacts"

    max_iterations = int(runner_cfg["max_iterations"])
    num_envs = int(runner_cfg["num_envs"])
    steps_per_iteration = int(budget_contract["steps_per_iteration"])
    planned_env_steps = num_envs * max_iterations * steps_per_iteration
    expected_budget = int(budget_contract["env_step_budget"])
    if planned_env_steps != expected_budget:
        raise Phase8EFullRoundError(
            f"fixed-budget config mismatch: planned_env_steps={planned_env_steps}, env_step_budget={expected_budget}"
        )

    phase1_config = str(curriculum_compile.get("phase1_config_path") or runner_cfg.get("phase1_config", "configs/phase1_push_recovery.yaml"))
    phase2_config = str(curriculum_compile.get("phase2_config_path") or runner_cfg.get("phase2_config", "configs/phase2_baseline.yaml"))

    command = [
        "unshare",
        "-Urmnfp",
        python_executable,
        str(PROJECT_ROOT / str(train_cfg.get("script", "scripts/phase8e_minimal_rootfs_probe.py"))),
        "--candidate-dir",
        str(candidate_dir),
        "--run-dir",
        str(minimal_run_dir),
        "--summary-file",
        summary_filename,
        "--repo-root",
        str(PROJECT_ROOT),
        "--device",
        str(runner_cfg.get("device", "cuda:0")),
        "--expected-schema-version",
        str(budget_config.get("candidate_policy", {}).get("allowed_schema_version", "humanoid_agent.cold_start.candidate.v1")),
        "--train-start-smoke",
        "--train-script",
        str(runner_cfg.get("train_script", "scripts/phase1_push_recovery_train_smoke.py")),
        "--phase1-config",
        phase1_config,
        "--phase2-config",
        phase2_config,
        "--task",
        str(runner_cfg.get("task", "Unitree-G1-29dof-Velocity")),
        "--profile",
        str(runner_cfg.get("train_profile", "phase2_train")),
        "--num-envs",
        str(num_envs),
        "--max-iterations",
        str(max_iterations),
        "--seed",
        str(runner_cfg.get("seed", 801)),
        "--train-timeout-seconds",
        str(runner_cfg.get("timeout_seconds", 7200)),
    ]
    hidden_probe_file = train_cfg.get("hidden_probe_file")
    if hidden_probe_file:
        command.extend(["--hidden-probe-file", str(hidden_probe_file)])
    env = sanitized_environment(repo_root=PROJECT_ROOT, allowed_names=list(train_cfg.get("env_allowlist", [])) or None)
    result = command_runner(
        command,
        PROJECT_ROOT,
        env,
        log_file,
        int(runner_cfg.get("timeout_seconds", train_cfg.get("timeout_seconds", 7200))),
    )
    child_summary = _load_json_mapping(child_summary_file)
    train_start = _mapping(_mapping(child_summary.get("probes"), "probes").get("train_start"), "probes.train_start") if child_summary else {}
    trusted_latest = train_start.get("trusted_latest_checkpoint") or train_start.get("latest_checkpoint")
    expected_prefix = str(TRAIN_START_CHECKPOINT_PREFIX)
    status, reason = _classify_training_result(
        command_result=result,
        child_summary=child_summary,
        train_start=train_start,
        trusted_latest_checkpoint=trusted_latest,
        expected_prefix=expected_prefix,
    )

    training_error = train_start.get("training_error") if isinstance(train_start.get("training_error"), Mapping) else None
    checkpoint_present = bool(train_start.get("checkpoint_present"))
    public_failure_feedback = None
    if status != "pass":
        public_failure_feedback = _public_failure_feedback(
            stage="fixed_budget_training",
            training_status=status,
            training_error=training_error,
            checkpoint_missing_due_to_training_failure=not checkpoint_present,
        )
    return {
        "status": status,
        "reason": reason,
        "training_error": dict(training_error) if training_error else None,
        "public_failure_feedback": public_failure_feedback,
        "checkpoint_present": checkpoint_present,
        "child_summary_file": str(child_summary_file.resolve()),
        "runner_artifacts_root": str(runner_artifacts_root.resolve()),
        "child_summary": child_summary,
        "public_summary": {
            "status": status,
            "reason": reason,
            "public_failure_feedback": public_failure_feedback,
            "summary_file": str(child_summary_file.resolve()),
            "runner_artifacts_root": str(runner_artifacts_root.resolve()),
            "command_result": result.to_summary(),
            "budget": {
                "primary_unit": "env_steps",
                "num_envs": num_envs,
                "max_iterations": max_iterations,
                "steps_per_iteration": steps_per_iteration,
                "planned_env_steps": planned_env_steps,
            },
            "isolation": {
                "mode": "unshare_user_mount_net_pid_namespace_plus_minimal_rootfs",
                "candidate_code_runs_in_training_child": True,
            },
            "formal_training_executed": status == "pass",
            "curriculum_compile": project_curriculum_compile_summary(dict(curriculum_compile)),
        },
    }


def _run_public_rollouts(
    *,
    config: Mapping[str, Any],
    budget_runner_cfg: Mapping[str, Any],
    full_round_cfg: Mapping[str, Any],
    run_dir: Path,
    checkpoint_path: Path,
    command_runner: CommandRunner,
    python_executable: str,
) -> dict[str, list[Path]]:
    rollout_cfg = _mapping(full_round_cfg.get("public_rollout"), "full_round.public_rollout")
    output_dir = run_dir / "public_rollouts"
    dev_paths = _run_rollout_split(
        split="public_dev",
        seeds=_int_sequence(rollout_cfg.get("dev_seeds", [1101, 1102, 1103])),
        rollout_cfg=rollout_cfg,
        budget_runner_cfg=budget_runner_cfg,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        command_runner=command_runner,
        python_executable=python_executable,
    )
    accept_paths = _run_rollout_split(
        split="public_accept",
        seeds=_int_sequence(rollout_cfg.get("accept_seeds", [2101, 2102, 2103])),
        rollout_cfg=rollout_cfg,
        budget_runner_cfg=budget_runner_cfg,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        command_runner=command_runner,
        python_executable=python_executable,
    )
    if not dev_paths:
        raise Phase8EFullRoundError("at least one public-dev rollout summary is required")
    return {"public_dev": dev_paths, "public_accept": accept_paths}


def _run_rollout_split(
    *,
    split: str,
    seeds: Sequence[int],
    rollout_cfg: Mapping[str, Any],
    budget_runner_cfg: Mapping[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    command_runner: CommandRunner,
    python_executable: str,
) -> list[Path]:
    paths: list[Path] = []
    for index, seed in enumerate(seeds, start=1):
        summary_file = output_dir / split / f"rollout_{index:02d}.json"
        log_file = output_dir / split / f"rollout_{index:02d}.log"
        command = [
            python_executable,
            str(PROJECT_ROOT / str(rollout_cfg.get("script", "scripts/phase1_push_recovery_eval.py"))),
            "--headless",
            "--device",
            str(budget_runner_cfg.get("device", "cuda:0")),
            "--config",
            str(budget_runner_cfg.get("phase1_config", "configs/phase1_push_recovery.yaml")),
            "--profile",
            str(rollout_cfg.get("profile", "phase2_public")),
            "--task",
            str(budget_runner_cfg.get("task", "Unitree-G1-29dof-Velocity")),
            "--policy",
            "checkpoint",
            "--checkpoint",
            str(checkpoint_path),
            "--num-envs",
            str(int(rollout_cfg.get("num_envs", 64))),
            "--seed",
            str(seed),
            "--summary-file",
            str(summary_file),
        ]
        max_steps = rollout_cfg.get("max_steps")
        if max_steps is not None:
            command.extend(["--max-steps", str(int(max_steps))])
        env = sanitized_environment(repo_root=PROJECT_ROOT)
        result = command_runner(command, PROJECT_ROOT, env, log_file, int(rollout_cfg.get("timeout_seconds", 1800)))
        if not summary_file.is_file():
            raise Phase8EFullRoundError(f"{split} rollout did not create a summary: exit_code={result.exit_code}, log={log_file}")
        paths.append(summary_file.resolve())
    return paths


def _classify_training_result(
    *,
    command_result: CommandResult,
    child_summary: Mapping[str, Any],
    train_start: Mapping[str, Any],
    trusted_latest_checkpoint: Any,
    expected_prefix: str,
) -> tuple[str, str | None]:
    if not child_summary:
        return "blocked", "train_summary_missing_or_invalid"
    child_status = str(child_summary.get("status", "unknown"))
    train_status = str(train_start.get("status", "unknown"))
    reject_reason = train_start.get("reject_reason") or child_summary.get("error")
    if child_status == "blocked" or train_status == "blocked":
        return "blocked", str(reject_reason or "isolated_training_boundary_blocked")
    # A missing checkpoint means training crashed and produced no policy: that is a
    # candidate-side reject, not a safety-gate failure.  ``failed_safety_gate`` is
    # reserved for a checkpoint that IS present but tampered/unsafe.
    if _train_start_checkpoint_absent(train_start, trusted_latest_checkpoint):
        return "reject", str(reject_reason or "fixed_budget_training_no_checkpoint")
    if _train_start_checkpoint_unsafe(train_start):
        return "failed_safety_gate", "train_start_checkpoint_failed_safe_load_gate"
    if not str(trusted_latest_checkpoint).startswith(expected_prefix):
        return "failed_safety_gate", "train_start_checkpoint_not_runner_sealed"
    if command_result.exit_code == 0 and child_status == "pass" and train_status == "pass":
        return "pass", None
    if command_result.timed_out:
        return "blocked", "fixed_budget_training_timeout"
    if child_status == "reject" or train_status == "reject":
        return "reject", str(reject_reason or "fixed_budget_training_rejected")
    if command_result.exit_code != 0:
        return "blocked", str(reject_reason or "fixed_budget_training_process_failed")
    return "reject", str(reject_reason or "fixed_budget_training_not_pass")


def _train_start_checkpoint_absent(train_start: Mapping[str, Any], trusted_latest_checkpoint: Any) -> bool:
    if train_start.get("checkpoint_present") is False:
        return True
    return trusted_latest_checkpoint is None or not str(trusted_latest_checkpoint)


def _train_start_checkpoint_unsafe(train_start: Mapping[str, Any]) -> bool:
    if train_start.get("checkpoint_safe_load") is False:
        return True
    safety = train_start.get("checkpoint_safety")
    return isinstance(safety, Mapping) and safety.get("safe_load") is False


def _update_formal_training_summary(summary: dict[str, Any], training: Mapping[str, Any]) -> None:
    public = _mapping(training.get("public_summary"), "training.public_summary")
    command = _mapping(public.get("command_result"), "training.command_result")
    budget = _mapping(public.get("budget"), "training.budget")
    executed = training.get("status") == "pass"
    summary["formal_training_executed"] = bool(executed)
    summary["formal_training"] = {
        "status": "executed" if executed else str(training.get("status", "not_executed")),
        "source": "full_round_fixed_budget_training_stage",
        "summary_file": training.get("child_summary_file"),
        "runner_artifacts_root": training.get("runner_artifacts_root"),
        "elapsed_seconds": command.get("elapsed_seconds"),
        "budget": {
            "primary_unit": budget.get("primary_unit"),
            "planned_env_steps": budget.get("planned_env_steps"),
            "num_envs": budget.get("num_envs"),
            "max_iterations": budget.get("max_iterations"),
            "steps_per_iteration": budget.get("steps_per_iteration"),
        },
        "nested_summary_note": (
            "phase8e_linked_round_summary and phase8e_round_summary do not train themselves; "
            "the top-level full_round summary is authoritative for formal training execution."
        ),
    }


def _classify_linked_error(exc: Phase8ELinkedError) -> str:
    text = str(exc).lower()
    safety_markers = (
        "safe-load",
        "checkpoint",
        "candidate_dir_argument_forbidden",
        "candidate-dir-forbidden",
        "public_result_file_required",
        "public-result-file-required",
    )
    return "failed_safety_gate" if any(marker in text for marker in safety_markers) else "reject"


def _reason_for_linked_error(exc: Phase8ELinkedError) -> str:
    text = str(exc).lower()
    if "safe-load" in text:
        return "checkpoint_failed_safe_load_gate"
    if "checkpoint" in text:
        return "checkpoint_contract_failed"
    if "trusted eval" in text:
        return "trusted_eval_gate_failed"
    return "linked_round_rejected"


def _exit_code_for_status(status: str) -> int:
    return 0 if status in {"pass", "accept", "promising"} else 2 if status == "blocked" else 1


def _run_subprocess(command: list[str], cwd: Path, env: dict[str, str], log_file: Path, timeout_seconds: int) -> CommandResult:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            exit_code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
    return CommandResult(exit_code=int(exit_code), timed_out=timed_out, elapsed_seconds=time.monotonic() - start, log_file=str(log_file.resolve()))


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    process.terminate()


def _budget_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    runner = _mapping(config.get("runner"), "runner")
    return load_mapping(_resolve_project_path(runner["budget_config_path"]))


def _project_candidate_stage(provider_summary: Mapping[str, Any], candidate_dir: Path) -> dict[str, Any]:
    return {
        "status": provider_summary.get("status"),
        "training_ready": provider_summary.get("training_ready"),
        "verifier_status": provider_summary.get("verifier_status"),
        "candidate_dir": str(candidate_dir),
        "provider_summary_file": provider_summary.get("summary_file"),
    }


def _project_checkpoint_for_eval(checkpoint_for_eval: Mapping[str, Any]) -> dict[str, Any]:
    safety = _mapping(checkpoint_for_eval.get("checkpoint_safety"), "checkpoint_safety")
    return {
        "status": "pass" if safety.get("safe_load") is True else "fail",
        "isolated_path": checkpoint_for_eval.get("isolated_path"),
        "host_path": checkpoint_for_eval.get("host_path"),
        "safe_load": safety.get("safe_load"),
        "sha256": safety.get("sha256"),
    }


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _int_sequence(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Phase8EFullRoundError("seed list must be a sequence of integers")
    return [int(item) for item in value]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase8EFullRoundError(f"{label} must be a mapping")
    return value


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

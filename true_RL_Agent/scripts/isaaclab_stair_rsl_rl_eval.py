#!/usr/bin/env python3
"""Evaluate a runner-sealed checkpoint on the fixed Stair protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.evaluation.binding import require_sha256_digest  # noqa: E402
from true_rl_agent.evaluation.phases import (  # noqa: E402
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationPhase,
)
from true_rl_agent.runner.checkpoint import (  # noqa: E402
    build_checkpoint_safety_report,
)
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
    STAIR_EVAL_PURPOSE_FORMAL,
    StairEvalEntrypointContext,
    validate_stair_eval_invocation,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (  # noqa: E402
    StairGpuRuntimeBinding,
    add_stair_gpu_process_binding_arguments,
    attach_stair_gpu_runtime_evidence,
    capture_stair_gpu_compute_context,
    validate_stair_gpu_process_binding_invocation,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (  # noqa: E402
    STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION,
    build_stair_eval_phase_summary,
    build_stair_full_eval_summary,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    StairEvaluationRuntimeLease,
    VerifiedStairExecutionSourceManifest,
    execute_stair_evaluation,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.composition.cell_manifest import (  # noqa: E402
    add_resolved_cell_manifest_arguments,
    load_optional_resolved_cell_manifest_from_args,
    resolved_cell_summary_fields,
)


def _build_parser(app_launcher_class: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    required_actions = []
    for option in (
        "--job-id",
        "--round",
        "--task-id",
        "--subject-kind",
        "--subject-id",
        "--subject-manifest-hash",
        "--program-source-checksum",
        "--training-binding-hash",
        "--checkpoint",
        "--checkpoint-sha256",
        "--eval-protocol-version",
        "--task-protocol-hash",
        "--acceptance-protocol-hash",
        "--seed-cohort-hash",
        "--evaluation-binding-hash",
        "--capability-profile-hash",
        "--summary-path",
        "--profile",
        "--runtime-dynamics-hash",
        "--backend-execution-environment-manifest-hash",
        "--config",
    ):
        required_actions.append(parser.add_argument(option))
    parser.add_argument("--evaluation-phase")
    parser.add_argument("--evaluation-plan-hash")
    parser.add_argument("--base-task-protocol-hash", default=None)
    parser.add_argument(
        "--stair-eval-purpose",
        choices=(
            STAIR_EVAL_PURPOSE_FORMAL,
            STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
        ),
        default=STAIR_EVAL_PURPOSE_FORMAL,
    )
    add_resolved_cell_manifest_arguments(parser)
    for option in (
        "--public-seeds",
        "--public-accept-seeds",
        "--sealed-validation-seeds",
        "--num-envs",
        "--seed",
    ):
        required_actions.append(parser.add_argument(option, type=int))
    app_launcher_class.add_app_launcher_args(parser)
    add_stair_gpu_process_binding_arguments(parser)
    for action in required_actions:
        action.required = True
    parser.set_defaults(headless=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    app_launcher_class: Any | None = None,
    terminate_process: bool | None = None,
) -> int:
    if terminate_process is not None and not isinstance(terminate_process, bool):
        raise TypeError("terminate_process must be a boolean or None")
    owns_process = (
        app_launcher_class is None
        if terminate_process is None
        else terminate_process
    )
    if app_launcher_class is None:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        app_launcher_class = AppLauncher

    args = _build_parser(app_launcher_class).parse_args(argv)
    try:
        args.device = require_execution_device(
            args.device,
            "stair eval execution device",
        )
        gpu_runtime_binding = validate_stair_gpu_process_binding_invocation(
            args,
            environment=os.environ,
        )
        if (
            args.resolved_cell_manifest_path is not None
            or args.resolved_cell_manifest_hash is not None
        ) and args.base_task_protocol_hash is None:
            raise ValueError(
                "resolved cell manifest requires --base-task-protocol-hash"
            )
        args.resolved_cell_manifest = load_optional_resolved_cell_manifest_from_args(
            args,
            expected_task_id=args.task_id,
            expected_base_task_protocol_hash=(
                args.base_task_protocol_hash
                if args.base_task_protocol_hash is not None
                else args.task_protocol_hash
            ),
        )
        context = validate_stair_eval_invocation(
            args,
            project_root=PROJECT_ROOT,
            resolved_cell_manifest=args.resolved_cell_manifest,
        )
        phase = _optional_evaluation_phase(args)
        source_manifest_preflight = preflight_stair_execution_source_manifest(
            context.config
        )
        execution_environment_verification = (
            verify_current_isaac_lab_execution_environment(
                args.backend_execution_environment_manifest_hash
            )
        )
    except (TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    checkpoint_report = build_checkpoint_safety_report(
        context.checkpoint_path,
        expected_checkpoint_hash=context.checkpoint_hash,
    ).to_public_dict()
    if checkpoint_report["status"] != "pass":
        print(str(checkpoint_report["reason"]), file=sys.stderr)
        return 2

    app_launcher = (
        app_launcher_class(args, multi_gpu=False)
        if gpu_runtime_binding is not None
        else app_launcher_class(args)
    )
    simulation_app = app_launcher.app
    runtime_lease = StairEvaluationRuntimeLease()
    status_code = 1
    try:
        try:
            summary = _run_eval(
                args,
                context=context,
                source_manifest_preflight=source_manifest_preflight,
                execution_environment_verification=(
                    execution_environment_verification
                ),
                checkpoint_report=checkpoint_report,
                runtime_lease=runtime_lease,
                phase=phase,
            )
            status_code = 0
        except Exception as exc:  # noqa: BLE001
            summary = _failure_summary(
                context,
                source_manifest_preflight=source_manifest_preflight,
                execution_environment_verification=(
                    execution_environment_verification
                ),
                checkpoint_report=checkpoint_report,
                execution_device=str(args.device),
                error=exc,
                phase=phase,
                evaluation_plan_hash=args.evaluation_plan_hash,
                gpu_runtime_binding=gpu_runtime_binding,
            )
        compute_context_observation = (
            capture_stair_gpu_compute_context(gpu_runtime_binding)
            if gpu_runtime_binding is not None
            else None
        )
        summary = attach_stair_gpu_runtime_evidence(
            summary,
            gpu_runtime_binding,
            compute_context_observation=compute_context_observation,
        )
        summary.update(
            resolved_cell_summary_fields(args.resolved_cell_manifest)
        )

        summary_path = _local_output_path(args.summary_path, "summary_path")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"status": summary["status"], "summary_path": str(summary_path)},
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        status_code = 1
        print(f"stair evaluation evidence publication failed: {exc}", file=sys.stderr)
        raise
    finally:
        if owns_process:
            _flush_standard_streams()
            os._exit(status_code)
    return close_isaac_app_on_success(
        simulation_app,
        exit_code=status_code,
    )


def _run_eval(
    args: argparse.Namespace,
    *,
    context: StairEvalEntrypointContext,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    execution_environment_verification: Mapping[str, str],
    checkpoint_report: Mapping[str, Any],
    runtime_lease: StairEvaluationRuntimeLease,
    phase: EvaluationPhase | None = None,
) -> dict[str, Any]:
    if not isinstance(args.device, str) or not args.device:
        raise ValueError("stair eval device must be a non-empty string")
    splits = (
        (phase.value,)
        if phase is not None
        else ("public_dev", "public_accept", "sealed_validation")
    )
    execution = execute_stair_evaluation(
        config=context.config,
        source_manifest_preflight=source_manifest_preflight,
        backend_execution_environment_verification=(
            execution_environment_verification
        ),
        checkpoint_path=context.checkpoint_path,
        device=args.device,
        seed_namespace=context.seed_namespace,
        splits=splits,
        runtime_lease=runtime_lease,
        resolved_cell_manifest=getattr(args, "resolved_cell_manifest", None),
    )
    if phase is not None:
        assert args.evaluation_plan_hash is not None
        return build_stair_eval_phase_summary(
            config=context.config,
            identity=context.identity,
            evaluation_plan_hash=args.evaluation_plan_hash,
            phase=phase,
            execution=execution,
            evaluation_binding=context.evaluation_binding,
            checkpoint_safety_report=checkpoint_report,
            expected_seed_namespace=context.seed_namespace,
        )
    return build_stair_full_eval_summary(
        config=context.config,
        identity=context.identity,
        evaluation_binding=context.evaluation_binding,
        execution=execution,
        checkpoint_safety_report=checkpoint_report,
        subject_manifest_hash=context.subject_manifest_hash,
        program_source_checksum=context.program_source_checksum,
        training_binding_hash=context.training_binding_hash,
        capability_profile_hash=context.capability_profile_hash,
    )


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass


def _failure_summary(
    context: StairEvalEntrypointContext,
    *,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    execution_environment_verification: Mapping[str, str],
    checkpoint_report: Mapping[str, Any],
    execution_device: str,
    error: BaseException,
    phase: EvaluationPhase | None = None,
    evaluation_plan_hash: str | None = None,
    gpu_runtime_binding: StairGpuRuntimeBinding | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": (
            EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION
            if phase is not None
            else STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION
        ),
        **context.identity.to_public_dict(),
        "evaluation_protocol": context.config.evaluation_protocol().to_public_dict(),
        "evaluation_binding": context.evaluation_binding.to_public_dict(),
        "execution_device": execution_device,
        "backend_execution_environment_verification": dict(
            execution_environment_verification
        ),
        "status": "fail",
        "public_score": 0.0,
        "metrics": {
            "environment_source_preflight": (
                source_manifest_preflight.to_summary_dict()
            ),
            "stair_evaluation": {},
        },
        "checkpoint_safety_report": dict(checkpoint_report),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [{"type": type(error).__name__, "message": str(error)}],
    }
    if phase is not None:
        assert evaluation_plan_hash is not None
        result["evaluation_phase"] = phase.value
        result["evaluation_plan_hash"] = evaluation_plan_hash
    return result


def _optional_evaluation_phase(args: argparse.Namespace) -> EvaluationPhase | None:
    raw_phase = args.evaluation_phase
    raw_plan_hash = args.evaluation_plan_hash
    if (raw_phase is None) != (raw_plan_hash is None):
        raise ValueError(
            "evaluation phase and evaluation plan hash must be supplied together"
        )
    if raw_phase is None:
        return None
    require_sha256_digest(raw_plan_hash, "evaluation_plan_hash")
    return EvaluationPhase.parse(raw_phase)


def _local_output_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field_name} must be a local path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

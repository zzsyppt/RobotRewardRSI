#!/usr/bin/env python3
"""Train one complete Stair reward program with Isaac Lab and RSL-RL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.backends.isaac_lab_runtime import require_physx_success  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    bind_isaac_log_path,
    close_isaac_app_on_success,
)
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.stair_climbing.candidate_failure import (  # noqa: E402
    StairCandidateProgramFailure,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (  # noqa: E402
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    StairTrainEntrypointContext,
    validate_stair_train_invocation,
)
from true_rl_agent.runner.gpu_process_binding import (  # noqa: E402
    GpuRuntimeBinding,
    add_gpu_process_binding_arguments,
    attach_gpu_runtime_evidence,
    capture_gpu_compute_context,
    validate_gpu_process_binding_invocation,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    VerifiedStairExecutionSourceManifest,
    execute_stair_training,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.composition.cell_manifest import (  # noqa: E402
    add_resolved_cell_manifest_arguments,
    load_optional_resolved_cell_manifest_from_args,
    resolved_cell_summary_fields,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_artifact,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    QualifiedStairRewardProgram,
    StairRewardProgramError,
    qualify_stair_candidate_reward_program,
)


def _build_parser() -> argparse.ArgumentParser:
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
        "--init",
        "--training-binding-hash",
        "--policy-architecture-hash",
        "--optimizer-training-profile-hash",
        "--agent-configuration-hash",
        "--agent-configuration-json",
        "--initialization-seed-schedule-hash",
        "--capability-profile-hash",
        "--summary-path",
        "--checkpoint-output",
        "--profile",
        "--runtime-dynamics-hash",
        "--backend-execution-environment-manifest-hash",
        "--config",
    ):
        required_actions.append(parser.add_argument(option))
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument("--candidate")
    artifact_group.add_argument("--reference-artifact")
    parser.add_argument("--candidate-artifact-hash", default=None)
    parser.add_argument("--candidate-verification-hash", default=None)
    parser.add_argument(
        "--candidate-execution-authorization-json",
        default=None,
    )
    parser.add_argument(
        "--candidate-execution-authorization-hash",
        default=None,
    )
    parser.add_argument("--reference-artifact-hash", default=None)
    parser.add_argument(
        "--expected-initial-policy-state-hash",
        default=None,
    )
    parser.add_argument("--candidate-failure-contract-hash", default=None)
    parser.add_argument("--task-protocol-hash", default=None)
    parser.add_argument("--base-task-protocol-hash", default=None)
    add_resolved_cell_manifest_arguments(parser)
    for option in (
        "--budget-steps",
        "--num-envs",
        "--num-steps-per-env",
        "--max-iterations",
        "--seed",
    ):
        required_actions.append(parser.add_argument(option, type=int))
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-checkpoint-sha256", default=None)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", required=True)
    else:
        AppLauncher.add_app_launcher_args(parser)
    add_gpu_process_binding_arguments(parser)
    for action in required_actions:
        action.required = True
    artifact_group.required = True
    parser.set_defaults(headless=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        args.device = require_execution_device(
            args.device,
            "stair train execution device",
        )
        gpu_runtime_binding = validate_gpu_process_binding_invocation(
            args,
            environment=os.environ,
        )
        if (
            args.resolved_cell_manifest_path is not None
            or args.resolved_cell_manifest_hash is not None
        ):
            if args.base_task_protocol_hash is None:
                raise ValueError(
                    "resolved cell manifest requires --base-task-protocol-hash"
                )
            args.resolved_cell_manifest = (
                load_optional_resolved_cell_manifest_from_args(
                    args,
                    expected_task_id=args.task_id,
                    expected_base_task_protocol_hash=(args.base_task_protocol_hash),
                )
            )
        else:
            args.resolved_cell_manifest = None
        context = validate_stair_train_invocation(
            args,
            project_root=PROJECT_ROOT,
            resolved_cell_manifest=args.resolved_cell_manifest,
        )
        if args.headless is not True:
            raise ValueError("stair production train must be headless")
        summary_path = _local_output_path(args.summary_path, "summary_path")
        checkpoint_output = _local_output_path(
            args.checkpoint_output,
            "checkpoint_output",
        )
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

    try:
        reward_program = _load_and_qualify_program(context)
    except StairCandidateProgramFailure as exc:
        _write_candidate_failure_summary(
            args=args,
            context=context,
            source_manifest_preflight=source_manifest_preflight,
            execution_environment_verification=(
                execution_environment_verification
            ),
            failure=exc,
            summary_path=summary_path,
            gpu_runtime_binding=gpu_runtime_binding,
        )
        return 1

    from isaaclab.app import AppLauncher

    bind_isaac_log_path(args, summary_path)
    simulation_app = (
        AppLauncher(args, multi_gpu=False).app
        if gpu_runtime_binding is not None
        else AppLauncher(args).app
    )
    exit_code = _run_training(
        args=args,
        context=context,
        source_manifest_preflight=source_manifest_preflight,
        execution_environment_verification=execution_environment_verification,
        reward_program=reward_program,
        summary_path=summary_path,
        checkpoint_output=checkpoint_output,
        gpu_runtime_binding=gpu_runtime_binding,
    )
    return close_isaac_app_on_success(
        simulation_app,
        exit_code=exit_code,
    )


def _load_and_qualify_program(
    context: StairTrainEntrypointContext,
) -> QualifiedStairRewardProgram:
    contract = StairClimbingComponentSmokeAdapter(
        context.config,
        resolved_cell_manifest=context.resolved_cell_manifest,
    ).contract
    if context.identity.subject.kind == "candidate":
        authorization = context.candidate_execution_authorization
        if (
            context.candidate_artifact is None
            or context.candidate_verification_hash is None
            or authorization is None
            or context.reference_artifact is not None
        ):
            raise ValueError("stair candidate train context artifact mismatch")
        try:
            trusted_reference, _ = resolve_official_stair_reference_artifact(
                context.config,
                project_root=PROJECT_ROOT,
                resolved_cell_manifest=context.resolved_cell_manifest,
            )
            program = qualify_stair_candidate_reward_program(
                artifact=context.candidate_artifact,
                contract=contract,
                static_verifier_policy=authorization.static_verifier_policy,
                trusted_reference_artifact=trusted_reference,
            )
        except StairRewardProgramError as exc:
            raise StairCandidateProgramFailure.qualification() from exc
        return authorization.verify_program(program)
    if context.reference_artifact is None or context.candidate_artifact is not None:
        raise ValueError("stair reference train context artifact mismatch")
    from true_rl_agent.tasks.stair_climbing.reward_program import (
        qualify_stair_reference_reward_program,
    )

    return qualify_stair_reference_reward_program(
        artifact=context.reference_artifact.program,
        contract=contract,
    )


def _run_training(
    *,
    args: argparse.Namespace,
    context: StairTrainEntrypointContext,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    execution_environment_verification: Mapping[str, str],
    reward_program: QualifiedStairRewardProgram,
    summary_path: Path,
    checkpoint_output: Path,
    gpu_runtime_binding: GpuRuntimeBinding | None,
) -> int:
    attempt_id = uuid.uuid4().hex
    run_dir = summary_path.parent / "run" / f"attempt-{attempt_id}"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result = execute_stair_training(
        config=context.config,
        source_manifest_preflight=source_manifest_preflight,
        agent_configuration=context.agent_configuration,
        reward_program=reward_program,
        device=str(args.device),
        run_dir=run_dir,
        checkpoint_output=checkpoint_output,
        run_name=f"true_rl_agent_{args.round}",
        expected_initial_policy_state_hash=(
            context.expected_initial_policy_state_hash
        ),
        resolved_cell_manifest=getattr(args, "resolved_cell_manifest", None),
    )
    fields = result.to_summary_fields()
    compute_context_observation = (
        capture_gpu_compute_context(gpu_runtime_binding)
        if gpu_runtime_binding is not None
        else None
    )
    status = str(fields.pop("status"))
    formal_training_executed = bool(fields.pop("training_executed"))
    training = fields.get("training")
    execution_device = (
        str(training.get("device"))
        if isinstance(training, Mapping) and training.get("device") is not None
        else str(args.device)
    )
    if status == "pass":
        fields.pop("activation_preflight", None)
    summary = _status_summary(
        status,
        args=args,
        context=context,
        source_manifest_preflight=source_manifest_preflight,
        execution_environment_verification=execution_environment_verification,
        reward_program=reward_program,
        attempt_id=attempt_id,
        formal_training_executed=formal_training_executed,
        execution_device=execution_device,
        gpu_runtime_binding=gpu_runtime_binding,
        compute_context_observation=compute_context_observation,
        **fields,
    )

    if summary["status"] == "pass":
        require_physx_success(summary_path.with_suffix(".kit.log"))
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {"summary_path": str(summary_path), "status": summary["status"]},
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


def _status_summary(
    status: str,
    *,
    args: argparse.Namespace,
    context: StairTrainEntrypointContext,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    execution_environment_verification: Mapping[str, str],
    reward_program: QualifiedStairRewardProgram | None,
    attempt_id: str,
    formal_training_executed: bool,
    execution_device: str,
    gpu_runtime_binding: GpuRuntimeBinding | None = None,
    compute_context_observation: Mapping[str, Any] | None = None,
    **payload: Any,
) -> dict[str, Any]:
    binding = context.training_binding
    summary = {
        **payload,
        "schema_version": "true_rl_agent.task.stair_climbing.train_summary.v1",
        **context.identity.to_public_dict(),
        "training_binding_hash": binding.binding_hash,
        "policy_architecture_hash": binding.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            binding.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": binding.agent_configuration_hash,
        "initialization_seed_schedule_hash": (
            binding.initialization_seed_schedule_hash
        ),
        "capability_profile_hash": binding.capability_profile_hash,
        "program_source_checksum": context.program_source_checksum,
        "environment_source_preflight": source_manifest_preflight.to_summary_dict(),
        "backend_execution_environment_verification": dict(
            execution_environment_verification
        ),
        "execution_device": execution_device,
        "status": status,
        "formal_training_executed": formal_training_executed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "round_id": args.round,
            "train_attempt_id": attempt_id,
            "seed_policy": "runner_owned_not_public",
        },
        **resolved_cell_summary_fields(
            getattr(args, "resolved_cell_manifest", None)
        ),
    }
    if reward_program is not None:
        if reward_program.program_source_checksum != context.program_source_checksum:
            raise ValueError("Stair reward program source changed before summary")
        summary["reward_program_qualification_hash"] = (
            reward_program.qualification_hash
        )
    if context.candidate_failure_contract_hash is not None:
        summary["candidate_failure_contract_hash"] = (
            context.candidate_failure_contract_hash
        )
    authorization = context.candidate_execution_authorization
    if authorization is not None:
        generic = authorization.authorization
        summary.update(
            {
                "candidate_verification_hash": (
                    context.candidate_verification_hash
                ),
                "candidate_admission_policy_id": generic.admission_policy_id,
                "candidate_admission_policy_hash": (
                    generic.admission_policy_hash
                ),
                "candidate_execution_authorization_schema_version": (
                    generic.payload_schema_version
                ),
                "candidate_execution_authorization_hash": (
                    generic.authorization_hash
                ),
            }
        )
    return attach_gpu_runtime_evidence(
        summary,
        gpu_runtime_binding,
        compute_context_observation=compute_context_observation,
    )


def _write_candidate_failure_summary(
    *,
    args: argparse.Namespace,
    context: StairTrainEntrypointContext,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    execution_environment_verification: Mapping[str, str],
    failure: StairCandidateProgramFailure,
    summary_path: Path,
    gpu_runtime_binding: GpuRuntimeBinding | None,
) -> None:
    summary = _status_summary(
        "fail",
        args=args,
        context=context,
        source_manifest_preflight=source_manifest_preflight,
        execution_environment_verification=execution_environment_verification,
        reward_program=None,
        attempt_id=uuid.uuid4().hex,
        formal_training_executed=False,
        execution_device=str(args.device),
        gpu_runtime_binding=gpu_runtime_binding,
        candidate_failure=failure.evidence.to_public_dict(),
        training={
            "expected_initial_policy_state_hash": (
                context.expected_initial_policy_state_hash
            )
        },
        errors=[],
    )
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {"summary_path": str(summary_path), "status": summary["status"]},
            sort_keys=True,
        )
    )


def _local_output_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field_name} must be a local path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

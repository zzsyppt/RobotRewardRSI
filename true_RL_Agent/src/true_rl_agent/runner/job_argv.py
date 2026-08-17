"""Bind runner job specifications to the exact argv launched by a subprocess."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.candidates.artifact import verify_verified_candidate_artifact
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.reference_artifact import verify_reference_program_artifact_ref
from true_rl_agent.runner.argv_contract import (
    require_exact_option_value,
    require_option_absent,
    unique_option_value,
)
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    EvalJobSpec,
    EvalPhaseJobSpec,
    EvaluationJobSpec,
    TrainJobSpec,
)
from true_rl_agent.training_binding import validate_training_execution_argv


_DISTRIBUTED_EXECUTION_OPTIONS = frozenset(
    {
        "--distributed",
        "--multi-gpu",
        "--multi_gpu",
        "--local-rank",
        "--local_rank",
        "--world-size",
        "--world_size",
    }
)


def validate_train_job_argv(
    spec: TrainJobSpec,
    argv: Sequence[str],
) -> None:
    """Validate all task-neutral train identity and execution bindings."""

    _validate_common_identity(spec, argv)
    require_exact_option_value(
        argv,
        "--program-source-checksum",
        expected=spec.program_source_checksum,
    )
    require_exact_option_value(
        argv,
        "--budget-steps",
        expected=str(spec.budget_steps),
    )
    require_exact_option_value(
        argv,
        "--init",
        expected=spec.policy_initialization.kind,
    )
    require_exact_option_value(
        argv,
        "--device",
        expected=spec.execution_device,
    )
    _require_metadata_path(spec.metadata, argv, "summary_path", "--summary-path")
    _require_metadata_path(
        spec.metadata,
        argv,
        "checkpoint_output_path",
        "--checkpoint-output",
    )
    unique_option_value(argv, "--round")
    validate_training_execution_argv(argv, binding=spec.training_binding)
    if spec.expected_initial_policy_state_hash is None:
        require_option_absent(
            argv,
            "--expected-initial-policy-state-hash",
        )
    else:
        require_exact_option_value(
            argv,
            "--expected-initial-policy-state-hash",
            expected=spec.expected_initial_policy_state_hash,
        )

    if spec.candidate_failure_contract is None:
        require_option_absent(argv, "--candidate-failure-contract-hash")
    else:
        require_exact_option_value(
            argv,
            "--candidate-failure-contract-hash",
            expected=spec.candidate_failure_contract.contract_hash,
        )

    if spec.subject.kind == "candidate":
        artifact = spec.candidate_artifact
        if artifact is None:
            raise ValueError(
                "candidate train job requires a verified candidate artifact"
            )
        verified = verify_verified_candidate_artifact(artifact)
        if verified != artifact:
            raise ValueError("candidate artifact reference changed during verification")
        require_exact_option_value(argv, "--candidate", expected=artifact.path)
        require_exact_option_value(
            argv,
            "--candidate-artifact-hash",
            expected=artifact.artifact_hash,
        )
        authorization = artifact.execution_authorization
        if authorization is None:
            require_option_absent(argv, "--candidate-verification-hash")
            require_option_absent(
                argv,
                "--candidate-execution-authorization-json",
            )
            require_option_absent(
                argv,
                "--candidate-execution-authorization-hash",
            )
        else:
            require_exact_option_value(
                argv,
                "--candidate-verification-hash",
                expected=artifact.verification_hash,
            )
            require_exact_option_value(
                argv,
                "--candidate-execution-authorization-json",
                expected=authorization.canonical_json,
            )
            require_exact_option_value(
                argv,
                "--candidate-execution-authorization-hash",
                expected=authorization.authorization_hash,
            )
        require_option_absent(argv, "--reference-artifact")
        require_option_absent(argv, "--reference-artifact-hash")
        return

    artifact = spec.reference_artifact
    if artifact is None:
        raise ValueError(
            "reference train job requires a verified reference artifact"
        )
    verified = verify_reference_program_artifact_ref(artifact)
    if verified != artifact:
        raise ValueError("reference artifact changed during verification")
    require_exact_option_value(
        argv,
        "--reference-artifact",
        expected=artifact.path,
    )
    require_exact_option_value(
        argv,
        "--reference-artifact-hash",
        expected=artifact.artifact_hash,
    )
    require_option_absent(argv, "--candidate")
    require_option_absent(argv, "--candidate-artifact-hash")
    require_option_absent(argv, "--candidate-verification-hash")
    require_option_absent(argv, "--candidate-execution-authorization-json")
    require_option_absent(argv, "--candidate-execution-authorization-hash")


def validate_eval_job_argv(
    spec: EvaluationJobSpec,
    argv: Sequence[str],
) -> None:
    """Validate all task-neutral eval identity and protocol bindings."""

    _validate_common_identity(spec, argv)
    protocol = spec.evaluation_protocol()
    binding = spec.evaluation_binding
    exact_options = {
        "--public-seeds": str(protocol.public_seed_count),
        "--public-accept-seeds": str(protocol.public_accept_seed_count),
        "--sealed-validation-seeds": str(
            protocol.sealed_validation_seed_count
        ),
        "--eval-protocol-version": protocol.eval_protocol_version,
        "--task-protocol-hash": binding.task_protocol_hash,
        "--acceptance-protocol-hash": binding.acceptance_protocol_hash,
        "--seed-cohort-hash": binding.seed_cohort_hash,
        "--evaluation-binding-hash": binding.binding_hash,
        "--capability-profile-hash": spec.capability_profile_hash,
        "--num-envs": str(protocol.eval_num_envs),
        "--device": spec.execution_device,
    }
    for option, expected in exact_options.items():
        require_exact_option_value(argv, option, expected=expected)
    _require_metadata_path(spec.metadata, argv, "summary_path", "--summary-path")
    unique_option_value(argv, "--round")
    unique_option_value(argv, "--seed")

    if isinstance(spec, EvalPhaseJobSpec):
        require_exact_option_value(
            argv,
            "--program-source-checksum",
            expected=spec.program_source_checksum,
        )
        require_exact_option_value(
            argv,
            "--training-binding-hash",
            expected=spec.training_binding_hash,
        )
        require_exact_option_value(
            argv,
            "--evaluation-phase",
            expected=spec.phase.value,
        )
        require_exact_option_value(
            argv,
            "--evaluation-plan-hash",
            expected=spec.evaluation_plan_hash,
        )
    elif isinstance(spec, EvalJobSpec):
        require_exact_option_value(
            argv,
            "--program-source-checksum",
            expected=spec.program_source_checksum,
        )
        require_exact_option_value(
            argv,
            "--training-binding-hash",
            expected=spec.training_binding_hash,
        )
        require_option_absent(argv, "--evaluation-phase")
        require_option_absent(argv, "--evaluation-plan-hash")
    elif isinstance(spec, BaselineEvalJobSpec):
        require_option_absent(argv, "--program-source-checksum")
        require_option_absent(argv, "--training-binding-hash")
        require_option_absent(argv, "--evaluation-phase")
        require_option_absent(argv, "--evaluation-plan-hash")
    else:
        raise TypeError("unsupported evaluation job specification")


def _validate_common_identity(
    spec: TrainJobSpec | EvaluationJobSpec,
    argv: Sequence[str],
) -> None:
    validate_single_device_job_argv(argv)
    exact_options = {
        "--job-id": spec.job_id,
        "--task-id": spec.task_id,
        "--subject-kind": spec.subject.kind,
        "--subject-id": spec.subject.subject_id,
    }
    for option, expected in exact_options.items():
        require_exact_option_value(argv, option, expected=expected)
    execution_environment_hash = spec.metadata.get(
        "backend_execution_environment_manifest_hash"
    )
    if execution_environment_hash is None:
        require_option_absent(
            argv,
            "--backend-execution-environment-manifest-hash",
        )
    else:
        require_exact_option_value(
            argv,
            "--backend-execution-environment-manifest-hash",
            expected=require_sha256_digest(
                execution_environment_hash,
                "job backend execution environment manifest hash",
            ),
        )
    manifest_hash = spec.subject_manifest_hash
    if manifest_hash is None:
        require_option_absent(argv, "--subject-manifest-hash")
    else:
        require_exact_option_value(
            argv,
            "--subject-manifest-hash",
            expected=manifest_hash,
        )


def validate_single_device_job_argv(argv: Sequence[str]) -> None:
    """Reject distributed launch switches; one job owns exactly one device."""

    for part in argv:
        if not isinstance(part, str):
            raise TypeError("job argv entries must be strings")
        option = part.split("=", 1)[0]
        if option in _DISTRIBUTED_EXECUTION_OPTIONS:
            raise ValueError(
                f"runner job cannot enable distributed or multi-GPU execution: {option}"
            )


def _require_metadata_path(
    metadata: Mapping[str, Any],
    argv: Sequence[str],
    key: str,
    option: str,
) -> None:
    expected = metadata.get(key)
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            f"job metadata {key} is required and must be a non-empty string"
        )
    require_exact_option_value(argv, option, expected=expected)


__all__ = [
    "validate_eval_job_argv",
    "validate_single_device_job_argv",
    "validate_train_job_argv",
]

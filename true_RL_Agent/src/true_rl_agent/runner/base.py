"""Runner interface and runner-owned result objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.public_projection import project_public_train_summary
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.phases import EvaluationAssembly, EvaluationPhaseResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.candidate_failure import (
    validate_candidate_runtime_failure_details,
)
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
    require_execution_device,
)


class RunnerFailureKind(str, Enum):
    """Closed task-neutral reasons why runner-owned execution did not complete."""

    CONTRACT_INVALID = "contract_invalid"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    LAUNCH_TRANSIENT = "launch_transient"
    LAUNCH_FAILED = "launch_failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    SUMMARY_MISSING = "summary_missing"
    SUMMARY_INVALID = "summary_invalid"
    RESOURCE_UNAVAILABLE = "resource_unavailable"


_RETRYABLE_INFRASTRUCTURE_FAILURES = frozenset(
    {
        RunnerFailureKind.LAUNCH_TRANSIENT,
        RunnerFailureKind.TIMEOUT,
        RunnerFailureKind.SUMMARY_MISSING,
    }
)


class RunnerExecutionError(RuntimeError):
    """Raised when runner-owned execution cannot complete."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: RunnerFailureKind = RunnerFailureKind.CONTRACT_INVALID,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("runner execution error message must be non-empty")
        if not isinstance(failure_kind, RunnerFailureKind):
            raise TypeError("runner failure_kind must be a RunnerFailureKind")
        super().__init__(message)
        self.failure_kind = failure_kind

    @property
    def retryable_infrastructure_failure(self) -> bool:
        return self.failure_kind in _RETRYABLE_INFRASTRUCTURE_FAILURES


class CandidateRuntimeError(RunnerExecutionError):
    """Raised when a verified candidate fails inside runner-owned execution."""

    def __init__(self, message: str, *, public_details: Mapping[str, Any]) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("candidate runtime failure message must be non-empty")
        super().__init__(message)
        self.public_details = validate_candidate_runtime_failure_details(
            public_details
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "details": dict(self.public_details),
        }


@dataclass(frozen=True)
class TrainJobResult:
    """Result of a runner-owned training job."""

    job_id: str
    subject: ExperimentSubject
    subject_manifest_hash: str | None
    program_source_checksum: str
    training_binding_hash: str
    capability_profile_hash: str
    execution_device: str
    checkpoint: CheckpointRef
    gpu_process_binding: GpuProcessBinding | None = None
    initial_policy_state_pairing: InitialPolicyStatePairing | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ExperimentSubject):
            raise TypeError("train job result subject must be an ExperimentSubject")
        if self.subject.kind == "reference":
            require_sha256_digest(
                self.subject_manifest_hash,
                "train job result subject_manifest_hash",
            )
        require_sha256_digest(
            self.training_binding_hash,
            "train job result training_binding_hash",
        )
        require_sha256_checksum(
            self.program_source_checksum,
            "train job result program_source_checksum",
        )
        require_sha256_digest(
            self.capability_profile_hash,
            "train job result capability_profile_hash",
        )
        require_execution_device(
            self.execution_device,
            "train job result execution_device",
        )
        if self.gpu_process_binding is not None:
            if not isinstance(self.gpu_process_binding, GpuProcessBinding):
                raise TypeError(
                    "train job result gpu_process_binding must be a "
                    "GpuProcessBinding"
                )
            if (
                self.execution_device
                != self.gpu_process_binding.process_execution_device
            ):
                raise ValueError(
                    "train job result execution device does not match GPU "
                    "process binding"
                )
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError("train job result checkpoint must be a CheckpointRef")
        if self.checkpoint.role != "eval" or not self.checkpoint.sealed:
            raise ValueError(
                "train job result requires a runner-sealed checkpoint with "
                "role='eval'"
            )
        if not isinstance(self.summary, Mapping):
            raise TypeError("train job result summary must be an object")
        if self.initial_policy_state_pairing is not None:
            if not isinstance(
                self.initial_policy_state_pairing,
                InitialPolicyStatePairing,
            ):
                raise TypeError("train job result initial policy pairing is invalid")
            self.initial_policy_state_pairing.require_match()
        metadata = self.checkpoint.metadata
        if metadata.get("training_subject") != self.subject.to_public_dict():
            raise ValueError("train job result checkpoint subject lineage mismatch")
        if metadata.get("subject_manifest_hash") != self.subject_manifest_hash:
            raise ValueError("train job result checkpoint manifest lineage mismatch")
        if metadata.get("program_source_checksum") != self.program_source_checksum:
            raise ValueError("train job result checkpoint source lineage mismatch")
        if metadata.get("training_binding_hash") != self.training_binding_hash:
            raise ValueError("train job result checkpoint training lineage mismatch")
        if metadata.get("capability_profile_hash") != self.capability_profile_hash:
            raise ValueError("train job result checkpoint capability lineage mismatch")
        if metadata.get("execution_device") != self.execution_device:
            raise ValueError("train job result checkpoint execution device mismatch")
        if (
            self.gpu_process_binding is not None
            and metadata.get("gpu_process_binding")
            != self.gpu_process_binding.to_dict()
        ):
            raise ValueError(
                "train job result checkpoint GPU process binding mismatch"
            )
        declared_pairing = metadata.get("initial_policy_state_pairing")
        if self.initial_policy_state_pairing is None:
            if declared_pairing is not None:
                raise ValueError(
                    "train job result checkpoint has an unexpected initial policy pairing"
                )
        elif declared_pairing != self.initial_policy_state_pairing.to_public_dict():
            raise ValueError(
                "train job result checkpoint initial policy pairing mismatch"
            )

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "job_id": self.job_id,
            "subject": self.subject.to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "execution_device": self.execution_device,
            "checkpoint": self.checkpoint.to_public_dict(include_metadata=False),
            "summary": project_public_train_summary(self.summary),
        }
        if self.initial_policy_state_pairing is not None:
            result["initial_policy_state_pairing"] = (
                self.initial_policy_state_pairing.to_public_dict()
            )
        if self.gpu_process_binding is not None:
            result["gpu_process_binding"] = self.gpu_process_binding.to_dict()
        return result


class Runner(ABC):
    """Execution boundary for training and eval jobs."""

    @abstractmethod
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        raise NotImplementedError

    @abstractmethod
    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        raise NotImplementedError

    def run_eval_phase(
        self,
        eval_spec: EvalPhaseJobSpec,
    ) -> EvaluationPhaseResult:
        _ = eval_spec
        raise RunnerExecutionError("runner does not support phased eval")

    def project_assembled_eval_summary(
        self,
        eval_spec: EvalJobSpec,
        assembly: EvaluationAssembly,
    ) -> PublicEvalResult:
        _ = eval_spec, assembly
        raise RunnerExecutionError(
            "runner does not support assembled eval summaries"
        )

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        _ = eval_spec
        raise RunnerExecutionError("runner does not support baseline eval")

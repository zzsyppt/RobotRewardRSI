"""Runner interface and runner-owned result objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.public_projection import project_public_train_summary
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    TrainJobSpec,
)


class RunnerExecutionError(RuntimeError):
    """Raised when runner-owned execution cannot complete."""


class CandidateRuntimeError(RunnerExecutionError):
    """Raised when a verified candidate fails inside runner-owned execution."""

    def __init__(self, message: str, *, public_details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.public_details = dict(public_details or {})

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
    checkpoint: CheckpointRef
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
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError("train job result checkpoint must be a CheckpointRef")
        if self.checkpoint.role != "eval" or not self.checkpoint.sealed:
            raise ValueError(
                "train job result requires a runner-sealed checkpoint with "
                "role='eval'"
            )
        if not isinstance(self.summary, Mapping):
            raise TypeError("train job result summary must be an object")
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

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "subject": self.subject.to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "checkpoint": self.checkpoint.to_public_dict(include_metadata=False),
            "summary": project_public_train_summary(self.summary),
        }


class Runner(ABC):
    """Execution boundary for training and eval jobs."""

    @abstractmethod
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        raise NotImplementedError

    @abstractmethod
    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        raise NotImplementedError

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        _ = eval_spec
        raise RunnerExecutionError("runner does not support baseline eval")

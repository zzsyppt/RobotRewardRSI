"""Runner interface and runner-owned result objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec


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
    checkpoint: CheckpointRef
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "checkpoint": self.checkpoint.to_public_dict(),
            "summary": dict(self.summary),
        }


class Runner(ABC):
    """Execution boundary for training and eval jobs."""

    @abstractmethod
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        raise NotImplementedError

    @abstractmethod
    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        raise NotImplementedError

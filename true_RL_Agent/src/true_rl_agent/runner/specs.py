"""Runner-owned train/eval job specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.policy import PolicyInitialization


@dataclass(frozen=True)
class CheckpointRef:
    """Reference to a checkpoint with an explicit role."""

    uri: str
    checksum: str
    role: str
    sealed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"warm_start", "eval", "train_output"}:
            raise ValueError(f"unsupported checkpoint role: {self.role}")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "checksum": self.checksum,
            "role": self.role,
            "sealed": self.sealed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainJobSpec:
    """Materialized fixed-budget training job."""

    job_id: str
    task_id: str
    candidate_id: str
    policy_initialization: PolicyInitialization
    budget_steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget_steps <= 0:
            raise ValueError("budget_steps must be positive")

    def training_start_checkpoint(self) -> CheckpointRef | None:
        if self.policy_initialization.baseline is None:
            return None
        checkpoint = self.policy_initialization.baseline.checkpoint
        if not isinstance(checkpoint, CheckpointRef):
            raise TypeError("baseline checkpoint must be a CheckpointRef")
        return checkpoint

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "policy_initialization": self.policy_initialization.kind,
            "budget_steps": self.budget_steps,
            "training_start_checkpoint": (
                self.training_start_checkpoint().to_public_dict()
                if self.training_start_checkpoint() is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EvalJobSpec:
    """Materialized eval job. Its checkpoint is separate from train initialization."""

    job_id: str
    task_id: str
    candidate_id: str
    checkpoint: CheckpointRef
    public_seed_count: int
    trusted_seed_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.checkpoint.role != "eval":
            raise ValueError("EvalJobSpec requires a checkpoint with role='eval'")
        if self.public_seed_count <= 0:
            raise ValueError("public_seed_count must be positive")
        if self.trusted_seed_count <= 0:
            raise ValueError("trusted_seed_count must be positive")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "checkpoint": self.checkpoint.to_public_dict(),
            "public_seed_count": self.public_seed_count,
        }

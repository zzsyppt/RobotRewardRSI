"""Backend adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import EvalJobSpec, TrainJobSpec
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec


@dataclass(frozen=True)
class BackendJobBundle:
    """Backend-materialized job specs, with no decision authority."""

    train_spec: TrainJobSpec
    eval_spec: EvalJobSpec
    task_invariants: TaskInvariantSpec | None = None


class BackendAdapter(ABC):
    """Backend boundary for materializing jobs without owning decisions."""

    @property
    @abstractmethod
    def backend_spec(self) -> BackendSpec:
        raise NotImplementedError

    @property
    def backend_id(self) -> str:
        return self.backend_spec.backend_id

    @property
    def job_bundle_type(self) -> type[BackendJobBundle]:
        return BackendJobBundle

    @abstractmethod
    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
    ) -> BackendJobBundle:
        raise NotImplementedError

"""Backend adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.artifact import VerifiedCandidateArtifactRef
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.runner.specs import BaselineEvalJobSpec, EvalJobSpec, TrainJobSpec
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
    def evaluation_binding_for(self, task_spec: TaskSpec) -> EvaluationBindingV1:
        """Return the backend's validated comparison domain for one task."""

        raise NotImplementedError

    def resolve_session_evaluation_binding(
        self,
        *,
        session_id: str,
        session_dir: Path,
        task_spec: TaskSpec,
        allow_create: bool,
    ) -> EvaluationBindingV1:
        """Create/load one task-owned session binding without exposing its internals.

        Overrides must be read-only when ``allow_create`` is false and must use
        create-once publication when it is true.
        """

        _ = session_id, session_dir, allow_create
        return self.evaluation_binding_for(task_spec)

    def validate_session_evaluation_binding(
        self,
        *,
        task_spec: TaskSpec,
        evaluation_binding: EvaluationBindingV1,
    ) -> None:
        """Validate a resolved binding before materializing runner jobs."""

        if evaluation_binding != self.evaluation_binding_for(task_spec):
            raise ValueError("backend session evaluation binding mismatch")

    @abstractmethod
    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> BackendJobBundle:
        raise NotImplementedError

    def materialize_baseline_eval(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        baseline_manifest: BaselineManifest,
        evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> BaselineEvalJobSpec:
        """Materialize eval-only work without creating a candidate or train job."""

        raise NotImplementedError(
            f"backend {self.backend_id!r} does not support baseline eval"
        )

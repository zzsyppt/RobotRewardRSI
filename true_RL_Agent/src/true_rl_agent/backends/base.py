"""Backend adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.artifact import VerifiedCandidateArtifactRef
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    EvalJobSpec,
    EvalPhaseJobSpec,
    TrainJobSpec,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec
from true_rl_agent.training_programs import VerifiedReferenceProgramArtifact

BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION = (
    "true_rl_agent.backend_execution_environment_verification.v1"
)


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

    @property
    def candidate_execution_capability_ids(self) -> frozenset[str]:
        """Candidate capabilities wired through materialization and runtime."""

        return frozenset()

    @property
    def candidate_admission_policy_identity(self) -> tuple[str, str] | None:
        """Admission policy identity required by candidate materialization."""

        return None

    @property
    def candidate_execution_authorization_schema(self) -> str | None:
        """Opaque task authorization schema accepted by materialization."""

        return None

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

    def resolve_session_execution_environment(
        self,
        *,
        session_id: str,
        session_dir: Path,
        allow_create: bool,
    ) -> Mapping[str, Any] | None:
        """Resolve the backend-owned execution identity frozen by Genesis.

        Backends without a production simulator environment retain ``None``.
        A production backend override must recompute its identity rather than
        trusting a caller-supplied snapshot.
        """

        _ = session_id, session_dir, allow_create
        return None

    def validate_session_execution_environment(
        self,
        *,
        execution_environment: Mapping[str, Any] | None,
    ) -> None:
        """Validate a resolved backend environment before Genesis/runtime use."""

        if execution_environment is not None:
            raise ValueError(
                "backend does not support an execution environment manifest"
            )

    def validate_runtime_session_execution_environment(
        self,
        *,
        execution_environment: Mapping[str, Any] | None,
    ) -> None:
        """Validate the Genesis environment required for formal execution."""

        self.validate_session_execution_environment(
            execution_environment=execution_environment
        )

    @abstractmethod
    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
        expected_initial_policy_state_hash: str | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
        train_attempt_scope: JobAttemptScope | None = None,
        eval_attempt_scope: JobAttemptScope | None = None,
    ) -> BackendJobBundle:
        raise NotImplementedError

    def materialize_baseline_eval(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        baseline_manifest: BaselineManifest,
        evaluation_binding: EvaluationBindingV1 | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
    ) -> BaselineEvalJobSpec:
        """Materialize eval-only work without creating a candidate or train job."""

        raise NotImplementedError(
            f"backend {self.backend_id!r} does not support baseline eval"
        )

    def materialize_eval_phase(
        self,
        *,
        eval_plan: EvalJobSpec,
        phase: EvaluationPhase,
        attempt_scope: JobAttemptScope | None = None,
    ) -> EvalPhaseJobSpec:
        """Materialize one split from an already checkpoint-bound eval plan."""

        _ = eval_plan, phase, attempt_scope
        raise NotImplementedError(
            f"backend {self.backend_id!r} does not support phased eval"
        )

    def materialize_reference_jobs(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        reference: VerifiedReferenceProgramArtifact,
        artifact_root: Path,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
    ) -> BackendJobBundle:
        """Materialize formal reference train/eval work through the normal runner."""

        _ = (
            session_id,
            task_spec,
            reference,
            artifact_root,
            policy_initialization,
            evaluation_binding,
            backend_execution_environment_manifest_hash,
        )
        raise NotImplementedError(
            f"backend {self.backend_id!r} does not support reference training"
        )

"""Task data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.candidates.schema import CandidateSurfacePolicy
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.public_result import PublicEvalResult

if TYPE_CHECKING:
    from true_rl_agent.agent_core.generation_finalization import GenerationFinalPlan
    from true_rl_agent.evaluation.phased_adapter import PhasedEvaluationAdapter
    from true_rl_agent.robots.base import RobotRequirements
    from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter
    from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter
    from true_rl_agent.tasks.reference_qualification import (
        ReferenceQualificationResult,
    )


@dataclass(frozen=True)
class TaskInvariantSpec:
    """Runner-owned task semantics that candidates must not weaken."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def assert_compatible_with(self, other: "TaskInvariantSpec") -> None:
        if dict(self.values) != dict(other.values):
            raise ValueError("task invariant mismatch")


@dataclass(frozen=True)
class TaskSpec:
    """Public task identity and runner-owned invariant contract."""

    task_id: str
    name: str
    supported_policy_initializations: frozenset[str]
    genesis_anchor_policy: Mapping[str, str]
    candidate_capability_profile: CandidateCapabilityProfile
    invariants: TaskInvariantSpec
    acceptance_protocol: AcceptanceProtocol | None = None
    fixed_train_protocol: Mapping[str, Any] = field(default_factory=dict)
    fixed_eval_protocol: Mapping[str, Any] = field(default_factory=dict)
    default_curriculum: CurriculumPlan = field(default_factory=lambda: CurriculumPlan(mode="task_default"))
    task_protocol_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("task_id", "name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task {field_name} must be non-empty")
        if not isinstance(self.supported_policy_initializations, frozenset):
            raise TypeError("supported_policy_initializations must be a frozenset")
        if not isinstance(self.genesis_anchor_policy, Mapping):
            raise TypeError("genesis_anchor_policy must be an object")
        policy = dict(self.genesis_anchor_policy)
        if set(policy) != set(self.supported_policy_initializations):
            raise ValueError(
                "genesis_anchor_policy must define every supported policy initialization"
            )
        allowed_anchor_kinds = {
            "bootstrap",
            "policy_baseline",
            "reference_program",
        }
        if any(value not in allowed_anchor_kinds for value in policy.values()):
            raise ValueError("genesis_anchor_policy contains an invalid anchor kind")
        for initialization_kind, anchor_kind in policy.items():
            if anchor_kind == "bootstrap" and initialization_kind != "random":
                raise ValueError("bootstrap Genesis requires random initialization")
            if anchor_kind == "policy_baseline" and initialization_kind != "warm_start":
                raise ValueError("policy-baseline Genesis requires warm_start initialization")
            if anchor_kind == "reference_program" and initialization_kind != "random":
                raise ValueError("reference-program Genesis requires random initialization")
        object.__setattr__(self, "genesis_anchor_policy", MappingProxyType(policy))
        if not isinstance(
            self.candidate_capability_profile,
            CandidateCapabilityProfile,
        ):
            raise TypeError(
                "candidate_capability_profile must be a CandidateCapabilityProfile"
            )
        if not isinstance(self.invariants, TaskInvariantSpec):
            raise TypeError("invariants must be a TaskInvariantSpec")
        if not isinstance(self.fixed_train_protocol, Mapping):
            raise TypeError("fixed_train_protocol must be an object")
        if not isinstance(self.fixed_eval_protocol, Mapping):
            raise TypeError("fixed_eval_protocol must be an object")
        if self.task_protocol_hash is not None:
            require_sha256_digest(self.task_protocol_hash, "task_protocol_hash")

    def supports_policy_initialization(self, kind: str) -> bool:
        return kind in self.supported_policy_initializations

    def genesis_anchor_kind_for(self, policy_initialization_kind: str) -> str:
        try:
            return self.genesis_anchor_policy[policy_initialization_kind]
        except KeyError as exc:
            raise ValueError(
                "task does not define a Genesis anchor for policy initialization: "
                + str(policy_initialization_kind)
            ) from exc

    def resolved_task_protocol_hash(self) -> str:
        if self.task_protocol_hash is not None:
            return self.task_protocol_hash
        acceptance_hash = (
            self.acceptance_protocol.protocol_hash
            if self.acceptance_protocol is not None
            else None
        )
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.task.protocol.derived.v1",
                "task_id": self.task_id,
                "supported_policy_initializations": sorted(
                    self.supported_policy_initializations
                ),
                "genesis_anchor_policy": dict(self.genesis_anchor_policy),
                "candidate_capability_profile_hash": (
                    self.candidate_capability_profile.profile_hash
                ),
                "invariants": dict(self.invariants.values),
                "fixed_train_protocol": dict(self.fixed_train_protocol),
                "fixed_eval_protocol": dict(self.fixed_eval_protocol),
                "acceptance_protocol_hash": acceptance_hash,
            }
        )


class TaskPlugin(ABC):
    """Task semantic boundary for AgentCore."""

    @property
    @abstractmethod
    def task_spec(self) -> TaskSpec:
        raise NotImplementedError

    def project_public_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
    ) -> PublicEvalResult:
        _ = train_summary
        return public_result

    def project_candidate_public_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> PublicEvalResult:
        """Project a candidate result with verifier-owned attribution context."""

        _ = mechanism_operation_delta
        return self.project_public_result(
            public_result=public_result,
            train_summary=train_summary,
        )

    def qualify_reference_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
    ) -> "ReferenceQualificationResult":
        _ = public_result, train_summary
        raise NotImplementedError(
            f"task {self.task_spec.task_id!r} does not support reference qualification"
        )

    def candidate_surface_policy(self) -> CandidateSurfacePolicy:
        """Return the task-owned taxonomy for candidate change components."""

        return CandidateSurfacePolicy()

    def robot_requirements(self) -> "RobotRequirements":
        """Declare task-neutral embodiment requirements at composition time."""

        from true_rl_agent.robots.base import RobotRequirements

        return RobotRequirements()

    def prompt_adapter(self) -> "TaskPromptAdapter":
        """Return the task-owned prompt semantics provider for this task."""

        from true_rl_agent.tasks.prompt_adapter import NeutralTaskPromptAdapter

        return NeutralTaskPromptAdapter(self.task_spec)

    def component_smoke_adapter(self) -> "ComponentSmokeAdapter":
        """Return the task-owned component smoke fixture provider for this task."""

        from true_rl_agent.tasks.component_smoke import NeutralComponentSmokeAdapter

        return NeutralComponentSmokeAdapter(self.task_spec.task_id)

    def phased_evaluation_adapter(self) -> "PhasedEvaluationAdapter | None":
        """Return an optional task-owned phased-evaluation capability."""

        return None

    def bind_genesis_evidence(self, public_evidence: Mapping[str, Any]) -> None:
        """Bind task-owned qualification facts after verified Genesis replay."""

    def bind_final_eval_plan(self, eval_plan, *, plan, plan_path: Path):
        """Attach task-owned final cohort arguments to an already bound job."""
        raise NotImplementedError("task does not support final evaluation")

    def summarize_final_comparison(self, *, plan, plan_path, reference, final):
        """Interpret final task evidence without acceptance or feedback authority."""
        raise NotImplementedError("task does not support final comparison")

    def prepare_final_confirmation(
        self, *, output_dir: Path, session_id: str,
        required_closed_generations: int, resume_existing: bool,
        replication_source: Path | None = None,
    ) -> "GenerationFinalPlan":
        """Commit the task's final cohort before a shared generation session."""

        raise NotImplementedError(f"task {self.task_spec.task_id!r} has no final confirmation contract")

"""Production TaskPlugin for the staged Stair task."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.candidates.schema import CandidateSurfacePolicy
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.robots.base import (
    BASE_ROLE,
    FORBIDDEN_CONTACT_ROLE,
    SUPPORT_EFFECTOR_ROLE,
    RobotRequirements,
    SemanticRoleRequirement,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.stair_climbing.candidate_surface import (
    build_stair_candidate_surface_policy,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.session_start import (
    STAIR_DEFAULT_START_MODE,
    StairSessionStartProfile,
    resolve_stair_session_start_profile,
)

if TYPE_CHECKING:
    from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter
    from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter


class StairClimbingTaskPlugin(TaskPlugin):
    """Own Stair semantics while leaving execution and decisions to shared layers."""

    curriculum_capabilities: Mapping[str, Mapping[str, Any]] = {}

    def robot_requirements(self) -> RobotRequirements:
        return RobotRequirements(
            roles=(
                SemanticRoleRequirement(BASE_ROLE, "body", 1, 1),
                SemanticRoleRequirement(
                    SUPPORT_EFFECTOR_ROLE,
                    "body",
                    1,
                    None,
                    require_all_profile_targets=True,
                ),
                SemanticRoleRequirement(
                    FORBIDDEN_CONTACT_ROLE,
                    "body",
                    1,
                    None,
                ),
            ),
            capabilities=frozenset(
                {
                    "articulation_runtime",
                    "base_state",
                    "height_scan",
                    "joint_position_action",
                    "joint_state",
                    "policy_observation_projection",
                    "support_contact",
                }
            ),
        )

    def __init__(
        self,
        config: StairClimbingTaskConfig,
        *,
        session_start_profile: StairSessionStartProfile | None = None,
        resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
    ) -> None:
        if not isinstance(config, StairClimbingTaskConfig):
            raise TypeError("stair plugin config must be StairClimbingTaskConfig")
        self.config = config
        if config.is_unified_protocol and resolved_cell_manifest is not None:
            if (
                resolved_cell_manifest.cell_variant_id != config.cell_variant_id
                or resolved_cell_manifest.resolved_cell_task_protocol_hash
                != config.task_protocol_hash
            ):
                raise ValueError("stair plugin manifest/config identity mismatch")
        self.resolved_cell_manifest = resolved_cell_manifest
        self._objective_family_mask: dict[str, Any] | None = None
        training_binding = config.formal_training_binding()
        if session_start_profile is None:
            session_start_profile = resolve_stair_session_start_profile(
                STAIR_DEFAULT_START_MODE,
                task_protocol_hash=config.task_protocol_hash,
                training_binding_hash=str(training_binding.binding_hash),
            )
        if not isinstance(session_start_profile, StairSessionStartProfile):
            raise TypeError("stair session start profile is invalid")
        if session_start_profile.task_protocol_hash != config.task_protocol_hash:
            raise ValueError("stair start profile task protocol mismatch")
        if session_start_profile.training_binding_hash != training_binding.binding_hash:
            raise ValueError("stair start profile training binding mismatch")
        self.session_start_profile = session_start_profile

    def objective_family_mask(self) -> Mapping[str, Any] | None:
        return (
            copy.deepcopy(self._objective_family_mask)
            if self._objective_family_mask is not None
            else None
        )

    def bind_objective_family_mask(self, mask: Mapping[str, Any]) -> None:
        """Bind the reference-derived Stair OOD objective for this session."""

        if not self.config.is_unified_protocol:
            raise ValueError("legacy Stair does not use an objective family mask")
        from true_rl_agent.tasks.stair_climbing.unified_scoring import (
            validate_objective_family_mask,
        )

        canonical = validate_objective_family_mask(
            suite=self.config.ordered_evaluation_suite,
            mask=mask,
        )
        if (
            self._objective_family_mask is not None
            and self._objective_family_mask != canonical
        ):
            raise ValueError("Stair objective family mask is already frozen")
        self._objective_family_mask = copy.deepcopy(canonical)

    @property
    def task_spec(self) -> TaskSpec:
        config = self.config
        if config.is_unified_protocol:
            return self._unified_task_spec()
        base = config.protocol_v1.to_dict()
        protocol_v2 = config.protocol_v2.to_public_dict()
        evaluation_protocol = config.evaluation_protocol()
        acceptance_protocol = config.acceptance_protocol
        agent = config.training_agent_configuration()
        train_runtime = config.train_runtime_profile()
        eval_runtime = config.eval_runtime_profile()
        seed_resolver = config.seed_resolver()
        required_gates = tuple(base["acceptance"]["required_pass_gates"])
        bin_ids = tuple(str(item["bin_id"]) for item in config.protocol_v1.eval_bins)

        fixed_train_protocol = {
            "training_distribution": "task_owned_seeded_stair_height_distribution",
            "budget_steps": config.budget_steps,
            "train_num_envs": config.train_num_envs,
            "train_max_iterations": config.train_max_iterations,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "policy_initialization": "random_fixed_session_seed",
            "initialization_seed_schedule_hash": (
                config.initialization_seed_schedule_hash()
            ),
            "private_training_seed_resolver_hash": (
                config.private_training_seed_resolver_hash()
            ),
            "runtime_profile": config.train_profile,
            "runtime_dynamics_hash": train_runtime.runtime_dynamics_hash,
            "complete_reward_runtime": train_runtime.to_public_dict()[
                "complete_reward_runtime"
            ],
            "policy_architecture_hash": agent.policy_architecture_hash,
            "optimizer_training_profile_hash": (
                agent.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": agent.agent_configuration_hash,
        }
        fixed_eval_protocol = {
            "evaluation_protocol": "fixed_stair_anchor_and_tread_generalization",
            **evaluation_protocol.to_public_dict(),
            "acceptance_protocol": acceptance_protocol.to_public_dict(),
            "seed_visibility": "counts_and_commitment_only_no_raw_seed_ids",
            "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "seed_cohort_hash": seed_resolver.namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            "private_seed_resolver_version": seed_resolver.resolver_version,
            "score_statistics_unit": "one_equal_height_tread_cell_macro_score_per_seed",
            "runtime_profile": config.eval_profile,
            "runtime_dynamics_hash": eval_runtime.runtime_dynamics_hash,
            "required_pass_gates": list(required_gates),
        }
        fixed_eval_protocol.update(
            {
                "height_bin_ids": list(bin_ids),
                "evaluation_profile_hash": config.evaluation_profile.profile_hash,
                "tread_depths_m": list(config.evaluation_profile.tread_depths_m),
                "evaluation_cell_ids": [
                    cell.cell_id for cell in config.evaluation_profile.cells
                ],
                "episodes_per_seed_per_cell": (
                    config.evaluation_profile.episodes_per_seed_per_cell
                ),
                "anchor_score_role": "fixed_tread_nonregression_guard",
            }
        )
        invariants = {
            **fixed_train_protocol,
            **evaluation_protocol.to_public_dict(),
            "task_protocol_hash": config.task_protocol_hash,
            "base_task_protocol_hash": config.protocol_v1.protocol_hash,
            "evaluation_semantics_hash": (
                config.evaluation_semantics.protocol_hash
                if config.evaluation_semantics is not None
                else None
            ),
            "candidate_capability_profile_hash": (
                config.protocol_v2.capability_profile.profile_hash
            ),
            "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "seed_cohort_hash": seed_resolver.namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            "eval_runtime_profile": config.eval_profile,
            "eval_runtime_dynamics_hash": eval_runtime.runtime_dynamics_hash,
            "train_runtime_profile": config.train_profile,
            "train_runtime_dynamics_hash": train_runtime.runtime_dynamics_hash,
            "episode_outcomes": config.episode_outcomes(),
            "required_pass_gates": list(required_gates),
            "protocol_status": protocol_v2["status"],
        }
        invariants.update(
            {
                "height_bin_ids": list(bin_ids),
                "evaluation_profile_hash": config.evaluation_profile.profile_hash,
                "tread_depths_m": list(config.evaluation_profile.tread_depths_m),
                "episodes_per_seed_per_cell": (
                    config.evaluation_profile.episodes_per_seed_per_cell
                ),
                "anchor_episode_horizon_s": float(
                    base["evaluation"]["episode_length_s"]
                ),
                "episode_horizon_policy": (
                    config.evaluation_profile.episode_horizon_policy
                ),
                "episode_horizons_s_by_tread_m": (
                    config.evaluation_profile.episode_horizons_by_tread_m
                ),
            }
        )
        return TaskSpec(
            task_id=STAIR_TASK_ID,
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={
                "random": self.session_start_profile.genesis_anchor_kind
            },
            candidate_capability_profile=config.protocol_v2.capability_profile,
            invariants=TaskInvariantSpec(invariants),
            acceptance_protocol=acceptance_protocol,
            fixed_train_protocol=fixed_train_protocol,
            fixed_eval_protocol=fixed_eval_protocol,
            default_curriculum=CurriculumPlan(mode="task_default"),
            task_protocol_hash=config.task_protocol_hash,
        )

    def _unified_task_spec(self) -> TaskSpec:
        """Project the single-cell protocol without consulting legacy Stair bytes."""

        config = self.config
        protocol = config.unified_protocol
        suite = protocol.evaluation_suite
        evaluation_protocol = suite.evaluation_protocol
        acceptance_protocol = protocol.acceptance_protocol
        agent = config.training_agent_configuration()
        train_runtime = config.train_runtime_profile()
        eval_runtime = config.eval_runtime_profile()
        seed_resolver = config.seed_resolver()
        training = protocol.training
        fixed_train_protocol = {
            "training_distribution": "task_owned_unified_c0_c1_distribution",
            "training_profile_hash": protocol.training_profile.profile_hash,
            "terrain_regime_allocation": dict(protocol.training_profile.allocation),
            "budget_steps": config.budget_steps,
            "train_num_envs": config.train_num_envs,
            "train_max_iterations": config.train_max_iterations,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "policy_initialization": str(training["policy_initialization"]),
            "initialization_seed_schedule_hash": (
                config.initialization_seed_schedule_hash()
            ),
            "private_training_seed_resolver_hash": (
                config.private_training_seed_resolver_hash()
            ),
            "runtime_profile": config.train_profile,
            "runtime_dynamics_hash": train_runtime.runtime_dynamics_hash,
            "policy_architecture_hash": agent.policy_architecture_hash,
            "optimizer_training_profile_hash": (
                agent.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": agent.agent_configuration_hash,
            "candidate_authority": protocol.to_dict()["candidate_authority"],
            "vnext_reward_authority_hash": (
                protocol.reward_authority.authority_hash
            ),
        }
        suite_payload = suite.to_dict()
        fixed_eval_protocol = {
            "evaluation_protocol": "fixed_stratified_stair_id_ood_suite",
            **evaluation_protocol.to_public_dict(),
            "acceptance_protocol": acceptance_protocol.to_public_dict(),
            "ordered_evaluation_suite_hash": suite.suite_hash,
            "ordered_cell_set_hash": suite.cell_set_hash,
            "split_cell_ids": {
                split: [cell.cell_id for cell in suite.cells_for_cohort(split)]
                for split in ("public_dev", "public_accept", "sealed_validation")
            },
            "score_contract": suite_payload["score_contract"],
            "qualification_rule": suite_payload["qualification_rule"],
            "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
            "seed_visibility": "counts_and_commitment_only_no_raw_seed_ids",
            "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "seed_cohort_hash": seed_resolver.namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            "runtime_profile": config.eval_profile,
            "runtime_dynamics_hash": eval_runtime.runtime_dynamics_hash,
            "required_pass_gates": list(protocol.required_pass_gates),
        }
        invariants = {
            **fixed_train_protocol,
            **fixed_eval_protocol,
            "task_projection_hash": protocol.task_protocol_hash,
            "resolved_cell_task_protocol_hash": config.task_protocol_hash,
            "episode_outcomes": config.episode_outcomes(),
            "candidate_capability_profile_hash": (
                protocol.candidate_capability_profile.profile_hash
            ),
            "protocol_state": protocol.protocol_state,
        }
        return TaskSpec(
            task_id=STAIR_TASK_ID,
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={
                "random": self.session_start_profile.genesis_anchor_kind
            },
            candidate_capability_profile=protocol.candidate_capability_profile,
            invariants=TaskInvariantSpec(invariants),
            acceptance_protocol=acceptance_protocol,
            fixed_train_protocol=fixed_train_protocol,
            fixed_eval_protocol=fixed_eval_protocol,
            default_curriculum=CurriculumPlan(mode="task_default"),
            task_protocol_hash=config.task_protocol_hash,
        )

    def project_public_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
    ) -> PublicEvalResult:
        from true_rl_agent.tasks.stair_climbing.public_projection import (
            project_stair_public_result,
        )

        return project_stair_public_result(
            config=self.config,
            public_result=public_result,
            train_summary=train_summary,
            resolved_cell_manifest=self.resolved_cell_manifest,
            objective_family_mask=self.objective_family_mask(),
        )

    def project_candidate_public_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> PublicEvalResult:
        from true_rl_agent.tasks.stair_climbing.public_projection import (
            project_stair_public_result,
        )

        if self.config.is_unified_protocol and self._objective_family_mask is None:
            raise ValueError("unified Stair candidate projection lacks Genesis mask")
        return project_stair_public_result(
            config=self.config,
            public_result=public_result,
            train_summary=train_summary,
            mechanism_operation_delta=mechanism_operation_delta,
            resolved_cell_manifest=self.resolved_cell_manifest,
            objective_family_mask=self.objective_family_mask(),
        )

    def qualify_reference_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
    ):
        from true_rl_agent.tasks.stair_climbing.reference_qualification import (
            qualify_stair_reference_public_result,
        )

        projected = self.project_public_result(
            public_result=public_result,
            train_summary=train_summary,
        )
        qualification = qualify_stair_reference_public_result(
            config=self.config,
            public_result=projected,
        )
        if self.config.is_unified_protocol and qualification.passed:
            mask = qualification.public_result.metrics.get("objective_family_mask")
            if not isinstance(mask, Mapping):
                raise ValueError("unified Stair reference qualification lacks its mask")
            self.bind_objective_family_mask(mask)
        return qualification

    def candidate_surface_policy(self) -> CandidateSurfacePolicy:
        if self.config.is_unified_protocol:
            return CandidateSurfacePolicy(
                diagnostic_components=frozenset({"diagnostic"}),
                engineering_components=frozenset(
                    {"failure_classifier", "logging", "public_projection"}
                ),
                code_level_components=frozenset({"reward_callable"}),
            )
        assert self.config.protocol_v1 is not None
        return build_stair_candidate_surface_policy(self.config.protocol_v1)

    def prompt_adapter(self) -> "TaskPromptAdapter":
        from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
            StairClimbingPromptAdapter,
        )

        return StairClimbingPromptAdapter(
            self.config,
            self.session_start_profile,
            resolved_cell_manifest=self.resolved_cell_manifest,
            objective_family_mask_provider=self.objective_family_mask,
        )

    def component_smoke_adapter(self) -> "ComponentSmokeAdapter":
        from true_rl_agent.tasks.stair_climbing.component_smoke import (
            StairClimbingComponentSmokeAdapter,
        )

        return StairClimbingComponentSmokeAdapter(
            self.config,
            resolved_cell_manifest=self.resolved_cell_manifest,
        )

    def phased_evaluation_adapter(self) -> object:
        from true_rl_agent.tasks.stair_climbing.phased_adapter import (
            StairPhasedEvaluationAdapter,
        )

        return StairPhasedEvaluationAdapter(
            self.config,
            objective_family_mask_provider=self.objective_family_mask,
        )


__all__ = ["StairClimbingTaskPlugin"]

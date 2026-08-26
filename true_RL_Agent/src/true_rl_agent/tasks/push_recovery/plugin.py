"""Push recovery TaskPlugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.candidates.schema import CandidateSurfacePolicy
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.prompting.capabilities import build_candidate_capability_profile
from true_rl_agent.robots.base import (
    BASE_ROLE,
    RobotRequirements,
    SemanticRoleRequirement,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.push_recovery.candidate_surface import (
    PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import attach_push_budget_gate

if TYPE_CHECKING:
    from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter
    from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter


class PushRecoveryTaskPlugin(TaskPlugin):
    """Task-owned push recovery semantics without backend execution authority."""

    curriculum_capabilities = {
        "external_push_curriculum": {"supported": True, "effective_now": False},
        "train_only_dr_scale": {"supported": True, "effective_now": False},
    }

    def robot_requirements(self) -> RobotRequirements:
        return RobotRequirements(
            roles=(
                SemanticRoleRequirement(BASE_ROLE, "body", 1, 1),
            ),
            capabilities=frozenset(
                {
                    "articulation_runtime",
                    "base_state",
                    "external_body_force",
                    "joint_position_action",
                    "joint_state",
                    "policy_observation_projection",
                }
            ),
        )

    def __init__(self, config: PushRecoveryTaskConfig | None = None) -> None:
        if config is None:
            config = PushRecoveryTaskConfig()
        elif not isinstance(config, PushRecoveryTaskConfig):
            raise TypeError("push recovery plugin config must be PushRecoveryTaskConfig")
        self.config = config

    @property
    def task_spec(self) -> TaskSpec:
        config = self.config
        eval_protocol = config.evaluation_protocol().to_public_dict()
        acceptance_protocol = config.acceptance_protocol()
        train_runtime_dynamics = config.train_runtime_dynamics()
        eval_runtime_dynamics = config.eval_runtime_dynamics()
        agent_configuration = config.training_agent_configuration()
        seed_cohort_hash = config.seed_cohort_hash()
        fixed_train_protocol = {
            "baseline_train_distribution": "runner_owned_target_distribution",
            "push_distribution": "fixed_target_push",
            "budget_steps": config.budget_steps,
            "train_num_envs": config.train_num_envs,
            "train_max_iterations": config.train_max_iterations,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "curriculum_control": "task_default_no_llm_schedule",
            "runtime_profile": config.train_profile,
            "runtime_dynamics_hash": train_runtime_dynamics["runtime_dynamics_hash"],
            "train_seed": config.train_seed,
            "policy_architecture_hash": (
                agent_configuration.policy_architecture_hash
            ),
            "optimizer_training_profile_hash": (
                agent_configuration.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": (
                agent_configuration.agent_configuration_hash
            ),
        }
        fixed_eval_protocol = {
            "eval_protocol": "fixed_public_accept_and_sealed_validation",
            "success_predicate": config.success_predicate,
            "seed_visibility": config.eval_seed_visibility,
            **eval_protocol,
            "checkpoint_selector": config.checkpoint_selector,
            "acceptance_protocol": acceptance_protocol.to_public_dict(),
            "seed_cohort_hash": seed_cohort_hash,
            "runtime_profile": config.eval_profile,
            "runtime_dynamics_hash": eval_runtime_dynamics["runtime_dynamics_hash"],
        }
        return TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=build_candidate_capability_profile(
                profile_id="unitree_push_recovery_v1",
                capability_ids=(
                    "diagnostic_term",
                    "reward_term",
                    "reward_weight_scale_schedule",
                ),
            ),
            invariants=TaskInvariantSpec(
                {
                    "success_predicate": config.success_predicate,
                    "eval_seed_visibility": config.eval_seed_visibility,
                    **eval_protocol,
                    "budget_steps": config.budget_steps,
                    "train_num_envs": config.train_num_envs,
                    "train_max_iterations": config.train_max_iterations,
                    "train_steps_per_iteration": config.train_steps_per_iteration,
                    "train_seed": config.train_seed,
                    "policy_architecture_hash": (
                        agent_configuration.policy_architecture_hash
                    ),
                    "optimizer_training_profile_hash": (
                        agent_configuration.optimizer_training_profile_hash
                    ),
                    "agent_configuration_hash": (
                        agent_configuration.agent_configuration_hash
                    ),
                    "train_runtime_profile": config.train_profile,
                    "eval_runtime_profile": config.eval_profile,
                    "train_runtime_dynamics_hash": train_runtime_dynamics["runtime_dynamics_hash"],
                    "eval_runtime_dynamics_hash": eval_runtime_dynamics["runtime_dynamics_hash"],
                    "seed_cohort_hash": seed_cohort_hash,
                    "checkpoint_selector": config.checkpoint_selector,
                    "accept_threshold": config.accept_threshold,
                }
            ),
            acceptance_protocol=acceptance_protocol,
            fixed_train_protocol=fixed_train_protocol,
            fixed_eval_protocol=fixed_eval_protocol,
            default_curriculum=CurriculumPlan(mode="task_default"),
        )

    def project_public_result(
        self,
        *,
        public_result: PublicEvalResult,
        train_summary: Mapping[str, Any],
    ) -> PublicEvalResult:
        training = train_summary.get("training")
        budget_summary = training if isinstance(training, Mapping) else train_summary
        return attach_push_budget_gate(public_result, budget_summary=budget_summary)

    def candidate_surface_policy(self) -> CandidateSurfacePolicy:
        return PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY

    def prompt_adapter(self) -> "TaskPromptAdapter":
        from true_rl_agent.tasks.push_recovery.prompt_adapter import (
            PushRecoveryPromptAdapter,
        )

        return PushRecoveryPromptAdapter(self.config)

    def component_smoke_adapter(self) -> "ComponentSmokeAdapter":
        from true_rl_agent.tasks.push_recovery.component_smoke import (
            PushRecoveryComponentSmokeAdapter,
        )

        return PushRecoveryComponentSmokeAdapter(self.config)

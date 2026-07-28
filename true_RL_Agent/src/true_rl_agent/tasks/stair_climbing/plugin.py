"""Production TaskPlugin for the staged Stair task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.candidates.schema import CandidateSurfacePolicy
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.stair_climbing.candidate_surface import (
    build_stair_candidate_surface_policy,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)

if TYPE_CHECKING:
    from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter
    from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter


class StairClimbingTaskPlugin(TaskPlugin):
    """Own Stair semantics while leaving execution and decisions to shared layers."""

    curriculum_capabilities: Mapping[str, Mapping[str, Any]] = {}

    def __init__(self, config: StairClimbingTaskConfig) -> None:
        if not isinstance(config, StairClimbingTaskConfig):
            raise TypeError("stair plugin config must be StairClimbingTaskConfig")
        self.config = config

    @property
    def task_spec(self) -> TaskSpec:
        config = self.config
        base = config.protocol_v1.to_dict()
        draft = config.protocol_v2.to_public_dict()
        evaluation_protocol = config.evaluation_protocol()
        acceptance_protocol = config.protocol_v1.acceptance_protocol
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
            "evaluation_protocol": "fixed_stair_seven_bin_macro_seed",
            **evaluation_protocol.to_public_dict(),
            "acceptance_protocol": acceptance_protocol.to_public_dict(),
            "seed_visibility": "counts_and_commitment_only_no_raw_seed_ids",
            "seed_cohort_hash": seed_resolver.cohort_hash,
            "private_seed_resolver_version": seed_resolver.resolver_version,
            "height_bin_ids": list(bin_ids),
            "episodes_per_seed_per_bin": (
                seed_resolver.episodes_per_seed_per_bin
            ),
            "score_statistics_unit": "one_seven_bin_macro_score_per_seed",
            "runtime_profile": config.eval_profile,
            "runtime_dynamics_hash": eval_runtime.runtime_dynamics_hash,
            "required_pass_gates": list(required_gates),
        }
        invariants = {
            **fixed_train_protocol,
            **evaluation_protocol.to_public_dict(),
            "task_protocol_hash": config.protocol_v2.protocol_hash,
            "base_task_protocol_hash": config.protocol_v1.protocol_hash,
            "candidate_capability_profile_hash": (
                config.protocol_v2.capability_profile.profile_hash
            ),
            "seed_cohort_hash": seed_resolver.cohort_hash,
            "eval_runtime_profile": config.eval_profile,
            "eval_runtime_dynamics_hash": eval_runtime.runtime_dynamics_hash,
            "train_runtime_profile": config.train_profile,
            "train_runtime_dynamics_hash": train_runtime.runtime_dynamics_hash,
            "height_bin_ids": list(bin_ids),
            "episodes_per_seed_per_bin": seed_resolver.episodes_per_seed_per_bin,
            "episode_length_s": float(base["evaluation"]["episode_length_s"]),
            "episode_outcomes": base["episode_outcomes"],
            "required_pass_gates": list(required_gates),
            "draft_status": draft["status"],
        }
        return TaskSpec(
            task_id=STAIR_TASK_ID,
            name="Unitree G1 Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={"random": "reference_program"},
            candidate_capability_profile=config.protocol_v2.capability_profile,
            invariants=TaskInvariantSpec(invariants),
            acceptance_protocol=acceptance_protocol,
            fixed_train_protocol=fixed_train_protocol,
            fixed_eval_protocol=fixed_eval_protocol,
            default_curriculum=CurriculumPlan(mode="task_default"),
            task_protocol_hash=config.protocol_v2.protocol_hash,
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
        )

    def candidate_surface_policy(self) -> CandidateSurfacePolicy:
        return build_stair_candidate_surface_policy(self.config.protocol_v1)

    def prompt_adapter(self) -> "TaskPromptAdapter":
        from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
            StairClimbingPromptAdapter,
        )

        return StairClimbingPromptAdapter(self.config)

    def component_smoke_adapter(self) -> "ComponentSmokeAdapter":
        from true_rl_agent.tasks.stair_climbing.component_smoke import (
            StairClimbingComponentSmokeAdapter,
        )

        return StairClimbingComponentSmokeAdapter(self.config)


__all__ = ["StairClimbingTaskPlugin"]

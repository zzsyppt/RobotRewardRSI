"""Push recovery TaskPlugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import attach_push_budget_gate

if TYPE_CHECKING:
    from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter


class PushRecoveryTaskPlugin(TaskPlugin):
    """Task-owned push recovery semantics without backend execution authority."""

    curriculum_capabilities = {
        "external_push_curriculum": {"supported": True, "effective_now": False},
        "train_only_dr_scale": {"supported": True, "effective_now": False},
    }

    def __init__(self, config: PushRecoveryTaskConfig | None = None) -> None:
        self.config = config or PushRecoveryTaskConfig()

    @property
    def task_spec(self) -> TaskSpec:
        config = self.config
        fixed_train_protocol = {
            "baseline_train_distribution": "runner_owned_target_distribution",
            "push_distribution": "fixed_target_push",
            "budget_steps": config.budget_steps,
            "train_num_envs": config.train_num_envs,
            "train_max_iterations": config.train_max_iterations,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "curriculum_control": "task_default_no_llm_schedule",
        }
        fixed_eval_protocol = {
            "eval_protocol": "fixed_public_and_trusted",
            "success_predicate": config.success_predicate,
            "seed_visibility": config.eval_seed_visibility,
            "eval_num_envs": config.eval_num_envs,
            "public_seed_count": config.public_seed_count,
            "public_accept_seed_count": config.public_accept_seed_count,
            "checkpoint_selector": config.checkpoint_selector,
        }
        return TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec(
                {
                    "fall_threshold": config.fall_threshold,
                    "episode_length": config.episode_length,
                    "push_magnitude_xy": config.push_magnitude_xy,
                    "push_timing": config.push_interval_s,
                    "success_predicate": config.success_predicate,
                    "eval_seed_visibility": config.eval_seed_visibility,
                    "eval_num_envs": config.eval_num_envs,
                    "public_seed_count": config.public_seed_count,
                    "public_accept_seed_count": config.public_accept_seed_count,
                    "budget_steps": config.budget_steps,
                    "train_num_envs": config.train_num_envs,
                    "train_max_iterations": config.train_max_iterations,
                    "train_steps_per_iteration": config.train_steps_per_iteration,
                    "checkpoint_selector": config.checkpoint_selector,
                    "accept_threshold": config.accept_threshold,
                }
            ),
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

    def prompt_adapter(self) -> "TaskPromptAdapter":
        from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter

        return PushRecoveryPromptAdapter(self.config)

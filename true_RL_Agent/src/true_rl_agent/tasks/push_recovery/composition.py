"""Push-recovery-owned application composition."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.isaac_lab_rsl_rl import (
    IsaacLabRslRlBackendAdapter,
    IsaacLabRslRlBackendConfig,
)
from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionProvider,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.memory import PushTaskMemoryAdapter
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin

PUSH_RECOVERY_TASK_ID = "unitree_push_recovery"
PUSH_RECOVERY_BACKEND_ID = "isaac_lab_rsl_rl"
PUSH_RECOVERY_ROBOT = "unitree_g1"
PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH = "push_recovery_obs_v1"
PUSH_RECOVERY_ACTION_SCHEMA_HASH = "unitree_actions_v1"
PUSH_RECOVERY_POLICY_ARCHITECTURE = "rsl_rl_ppo_mlp"
DEFAULT_ISAAC_PYTHON = "/data/yuyangliu/envs/isaaclab/bin/python"
DEFAULT_TRAIN_ENTRYPOINT = "scripts/isaaclab_rsl_rl_train.py"
DEFAULT_EVAL_ENTRYPOINT = "scripts/isaaclab_rsl_rl_eval.py"


@dataclass(frozen=True)
class PushRecoveryCompositionConfig:
    """Canonical validated input for one push-recovery composition snapshot."""

    isaac_python: str
    train_entrypoint: str
    eval_entrypoint: str
    budget_steps: int
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_protocol_version: str
    train_seed: int
    runtime_config: str | None
    train_profile: str
    eval_profile: str
    train_num_envs: int
    train_max_iterations: int
    train_steps_per_iteration: int
    eval_num_envs: int
    runner_timeout_s: int

    def __post_init__(self) -> None:
        for field_name in (
            "isaac_python",
            "train_entrypoint",
            "eval_entrypoint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.runtime_config is not None and (
            not isinstance(self.runtime_config, str)
            or not self.runtime_config.strip()
        ):
            raise ValueError("runtime_config must be a non-empty string when provided")
        self.to_task_config()

    @classmethod
    def defaults(cls) -> "PushRecoveryCompositionConfig":
        task_config = PushRecoveryTaskConfig()
        return cls(
            isaac_python=DEFAULT_ISAAC_PYTHON,
            train_entrypoint=DEFAULT_TRAIN_ENTRYPOINT,
            eval_entrypoint=DEFAULT_EVAL_ENTRYPOINT,
            budget_steps=task_config.budget_steps,
            public_seed_count=task_config.public_seed_count,
            public_accept_seed_count=task_config.public_accept_seed_count,
            sealed_validation_seed_count=(
                task_config.sealed_validation_seed_count
            ),
            eval_protocol_version=task_config.eval_protocol_version,
            train_seed=task_config.train_seed,
            runtime_config=task_config.runtime_config_path,
            train_profile=task_config.train_profile,
            eval_profile=task_config.eval_profile,
            train_num_envs=task_config.train_num_envs,
            train_max_iterations=task_config.train_max_iterations,
            train_steps_per_iteration=task_config.train_steps_per_iteration,
            eval_num_envs=task_config.eval_num_envs,
            runner_timeout_s=task_config.runner_timeout_s,
        )

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
    ) -> "PushRecoveryCompositionConfig":
        field_names = tuple(field.name for field in fields(cls))
        missing = [name for name in field_names if name not in values]
        if missing:
            raise ValueError(
                "push composition values missing: " + ", ".join(missing)
            )
        return cls(**{name: values[name] for name in field_names})

    def to_task_config(self) -> PushRecoveryTaskConfig:
        return PushRecoveryTaskConfig(
            budget_steps=self.budget_steps,
            train_num_envs=self.train_num_envs,
            train_max_iterations=self.train_max_iterations,
            train_steps_per_iteration=self.train_steps_per_iteration,
            eval_num_envs=self.eval_num_envs,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_protocol_version=self.eval_protocol_version,
            train_seed=self.train_seed,
            runner_timeout_s=self.runner_timeout_s,
            runtime_config_path=self.runtime_config,
            train_profile=self.train_profile,
            eval_profile=self.eval_profile,
        )

    def to_task_defaults(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class PushRecoveryCompositionProvider(TaskCompositionProvider):
    @property
    def task_id(self) -> str:
        return PUSH_RECOVERY_TASK_ID

    def cli_defaults(self) -> Mapping[str, Any]:
        return PushRecoveryCompositionConfig.defaults().to_task_defaults()

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        composition_config = PushRecoveryCompositionConfig.from_values(values)
        task_config = composition_config.to_task_config()
        task_plugin = PushRecoveryTaskPlugin(task_config)
        backend_config = _build_push_backend_config(
            composition_config=composition_config,
            context=context,
            task_config=task_config,
            task_plugin=task_plugin,
        )
        return ResolvedTaskComposition(
            task_id=self.task_id,
            task_plugin=task_plugin,
            task_memory_adapter=PushTaskMemoryAdapter(),
            backend_adapter=IsaacLabRslRlBackendAdapter(backend_config),
            runner=SubprocessRunner(
                allowed_executables=(composition_config.isaac_python,),
                timeout_s=composition_config.runner_timeout_s,
                cwd=context.project_root,
            ),
            policy_initialization=PolicyInitialization.random(),
            genesis_anchor_kind="bootstrap",
            task_defaults=composition_config.to_task_defaults(),
        )

    def run_audit(
        self,
        *,
        composition: ResolvedTaskComposition,
        output_dir: Path,
        rounds: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        from true_rl_agent.tasks.push_recovery.loop_audit import (
            run_push_recovery_loop_audit,
        )

        return run_push_recovery_loop_audit(
            composition=composition,
            output_dir=output_dir,
            rounds=rounds,
            session_id=session_id,
        )


def _build_push_backend_config(
    *,
    composition_config: PushRecoveryCompositionConfig,
    context: TaskCompositionContext,
    task_config: PushRecoveryTaskConfig,
    task_plugin: PushRecoveryTaskPlugin,
) -> IsaacLabRslRlBackendConfig:
    if task_config != composition_config.to_task_config():
        raise ValueError(
            "push task config does not match canonical composition config"
        )
    if task_plugin.config != task_config:
        raise ValueError("push task plugin config does not match composition config")
    train_runtime_dynamics = task_config.train_runtime_dynamics()
    eval_runtime_dynamics = task_config.eval_runtime_dynamics()
    evaluation_protocol = task_config.evaluation_protocol()
    acceptance_protocol = task_config.acceptance_protocol()
    evaluation_binding = EvaluationBindingV1(
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        acceptance_protocol_hash=acceptance_protocol.protocol_hash,
        task_protocol_hash=task_plugin.task_spec.resolved_task_protocol_hash(),
        seed_cohort_hash=task_config.seed_cohort_hash(),
    )
    return IsaacLabRslRlBackendConfig(
        isaac_lab_python=composition_config.isaac_python,
        train_entrypoint=composition_config.train_entrypoint,
        eval_entrypoint=composition_config.eval_entrypoint,
        supported_tasks=frozenset({PUSH_RECOVERY_TASK_ID}),
        default_budget_steps=task_config.budget_steps,
        public_seed_count=task_config.public_seed_count,
        public_accept_seed_count=task_config.public_accept_seed_count,
        sealed_validation_seed_count=task_config.sealed_validation_seed_count,
        eval_num_envs=task_config.eval_num_envs,
        eval_protocol_version=task_config.eval_protocol_version,
        evaluation_binding=evaluation_binding,
        robot=PUSH_RECOVERY_ROBOT,
        observation_schema_hash=PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
        action_schema_hash=PUSH_RECOVERY_ACTION_SCHEMA_HASH,
        policy_architecture=PUSH_RECOVERY_POLICY_ARCHITECTURE,
        artifact_root=str(context.output_dir / "runner"),
        train_profile=task_config.train_profile,
        eval_profile=task_config.eval_profile,
        train_num_envs=task_config.train_num_envs,
        training_agent_configuration=(
            task_config.training_agent_configuration()
        ),
        train_max_iterations=task_config.train_max_iterations,
        train_steps_per_iteration=task_config.train_steps_per_iteration,
        eval_base_seed=task_config.eval_base_seed,
        train_seed=task_config.train_seed,
        runtime_config_path=task_config.runtime_config_path,
        train_runtime_dynamics_hash=train_runtime_dynamics[
            "runtime_dynamics_hash"
        ],
        eval_runtime_dynamics_hash=eval_runtime_dynamics[
            "runtime_dynamics_hash"
        ],
        work_dir=str(context.project_root),
        curriculum_capabilities=PushRecoveryTaskPlugin.curriculum_capabilities,
    )


__all__ = [
    "PUSH_RECOVERY_TASK_ID",
    "PushRecoveryCompositionConfig",
    "PushRecoveryCompositionProvider",
]

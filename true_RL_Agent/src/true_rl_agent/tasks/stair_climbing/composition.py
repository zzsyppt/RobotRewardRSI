"""Stair-owned application composition for explicit staging use."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
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
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.memory import StairTaskMemoryAdapter
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_PROFILE,
    STAIR_TRAIN_PROFILE,
)
from true_rl_agent.training_programs import ReferenceProgramManifest

DEFAULT_ISAAC_PYTHON = "/data/yuyangliu/envs/isaaclab/bin/python"
DEFAULT_STAIR_TRAIN_ENTRYPOINT = "scripts/isaaclab_stair_rsl_rl_train.py"
DEFAULT_STAIR_EVAL_ENTRYPOINT = "scripts/isaaclab_stair_rsl_rl_eval.py"
DEFAULT_STAIR_PROTOCOL_V2 = STAIR_PROTOCOL_V2_RELATIVE_PATH


@dataclass(frozen=True)
class StairClimbingCompositionConfig:
    isaac_python: str
    train_entrypoint: str
    eval_entrypoint: str
    budget_steps: int
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_protocol_version: str
    train_seed: int
    runtime_config: str
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
            "eval_protocol_version",
            "runtime_config",
            "train_profile",
            "eval_profile",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "budget_steps",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "runner_timeout_s",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.train_seed, bool)
            or not isinstance(self.train_seed, int)
            or self.train_seed < 0
        ):
            raise ValueError("train_seed must be a non-negative integer")
        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.train_num_envs,
            num_steps_per_env=self.train_steps_per_iteration,
            max_iterations=self.train_max_iterations,
        )
        if self.train_entrypoint != DEFAULT_STAIR_TRAIN_ENTRYPOINT:
            raise ValueError("stair train_entrypoint is not canonical")
        if self.eval_entrypoint != DEFAULT_STAIR_EVAL_ENTRYPOINT:
            raise ValueError("stair eval_entrypoint is not canonical")
        if self.runtime_config != DEFAULT_STAIR_PROTOCOL_V2:
            raise ValueError("stair runtime_config is not the canonical draft V2")
        runtime_path = PurePosixPath(self.runtime_config)
        if runtime_path.is_absolute() or ".." in runtime_path.parts:
            raise ValueError("stair runtime_config must be project-relative")

    @classmethod
    def defaults(cls, *, project_root: Path) -> "StairClimbingCompositionConfig":
        task = StairClimbingTaskConfig.defaults(
            project_root / DEFAULT_STAIR_PROTOCOL_V2
        )
        return cls(
            isaac_python=DEFAULT_ISAAC_PYTHON,
            train_entrypoint=DEFAULT_STAIR_TRAIN_ENTRYPOINT,
            eval_entrypoint=DEFAULT_STAIR_EVAL_ENTRYPOINT,
            budget_steps=task.budget_steps,
            public_seed_count=task.public_seed_count,
            public_accept_seed_count=task.public_accept_seed_count,
            sealed_validation_seed_count=task.sealed_validation_seed_count,
            eval_protocol_version=task.eval_protocol_version,
            train_seed=task.train_seed,
            runtime_config=DEFAULT_STAIR_PROTOCOL_V2,
            train_profile=STAIR_TRAIN_PROFILE,
            eval_profile=STAIR_EVAL_PROFILE,
            train_num_envs=task.train_num_envs,
            train_max_iterations=task.train_max_iterations,
            train_steps_per_iteration=task.train_steps_per_iteration,
            eval_num_envs=task.eval_num_envs,
            runner_timeout_s=task.runner_timeout_s,
        )

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> "StairClimbingCompositionConfig":
        names = tuple(item.name for item in fields(cls))
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError("stair composition values missing: " + ", ".join(missing))
        return cls(**{name: values[name] for name in names})

    def to_task_config(self, *, project_root: Path) -> StairClimbingTaskConfig:
        return StairClimbingTaskConfig(
            protocol_v2_path=project_root / self.runtime_config,
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
            train_profile=self.train_profile,
            eval_profile=self.eval_profile,
        )

    def to_task_defaults(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class StairClimbingCompositionProvider(TaskCompositionProvider):
    """The sole Stair provider; registration requires a real reference identity."""

    def __init__(self, *, reference_program_manifest: ReferenceProgramManifest) -> None:
        if not isinstance(reference_program_manifest, ReferenceProgramManifest):
            raise TypeError("stair composition requires a ReferenceProgramManifest")
        if reference_program_manifest.task_id != STAIR_TASK_ID:
            raise ValueError("stair reference program task_id mismatch")
        if reference_program_manifest.authorship_origin != "official_upstream":
            raise ValueError("stair reference program must be official_upstream")
        self.reference_program_manifest = reference_program_manifest
        self._project_root = Path(__file__).resolve().parents[4]

    @property
    def task_id(self) -> str:
        return STAIR_TASK_ID

    def cli_defaults(self) -> Mapping[str, Any]:
        return StairClimbingCompositionConfig.defaults(
            project_root=self._project_root
        ).to_task_defaults()

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        composition_config = StairClimbingCompositionConfig.from_values(values)
        task_config = composition_config.to_task_config(
            project_root=context.project_root
        )
        plugin = StairClimbingTaskPlugin(task_config)
        profile_hash = plugin.task_spec.candidate_capability_profile.profile_hash
        if self.reference_program_manifest.capability_profile_hash != profile_hash:
            raise ValueError("stair reference capability profile mismatch")
        backend = _build_stair_backend_config(
            composition_config=composition_config,
            context=context,
            task_config=task_config,
            plugin=plugin,
        )
        return ResolvedTaskComposition(
            task_id=self.task_id,
            task_plugin=plugin,
            task_memory_adapter=StairTaskMemoryAdapter(),
            backend_adapter=IsaacLabRslRlBackendAdapter(backend),
            runner=SubprocessRunner(
                allowed_executables=(composition_config.isaac_python,),
                timeout_s=composition_config.runner_timeout_s,
                cwd=context.project_root,
            ),
            policy_initialization=PolicyInitialization.random(),
            genesis_anchor_kind="reference_program",
            task_defaults=composition_config.to_task_defaults(),
            reference_program_manifest=self.reference_program_manifest,
        )


def _build_stair_backend_config(
    *,
    composition_config: StairClimbingCompositionConfig,
    context: TaskCompositionContext,
    task_config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
) -> IsaacLabRslRlBackendConfig:
    if plugin.config is not task_config:
        raise ValueError("stair plugin config does not match composition config")
    evaluation_protocol = task_config.evaluation_protocol()
    binding = EvaluationBindingV1(
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        acceptance_protocol_hash=task_config.protocol_v1.acceptance_protocol.protocol_hash,
        task_protocol_hash=task_config.protocol_v2.protocol_hash,
        seed_cohort_hash=task_config.seed_resolver().cohort_hash,
    )
    compatibility = task_config.protocol_v1.to_dict()["policy_compatibility"]
    return IsaacLabRslRlBackendConfig(
        isaac_lab_python=composition_config.isaac_python,
        train_entrypoint=composition_config.train_entrypoint,
        eval_entrypoint=composition_config.eval_entrypoint,
        supported_tasks=frozenset({STAIR_TASK_ID}),
        default_budget_steps=task_config.budget_steps,
        public_seed_count=task_config.public_seed_count,
        public_accept_seed_count=task_config.public_accept_seed_count,
        sealed_validation_seed_count=task_config.sealed_validation_seed_count,
        eval_num_envs=task_config.eval_num_envs,
        eval_protocol_version=task_config.eval_protocol_version,
        evaluation_binding=binding,
        robot=str(compatibility["robot"]),
        observation_schema_hash=str(
            compatibility["policy_observation"]["schema_hash"]
        ),
        action_schema_hash=str(compatibility["action"]["schema_hash"]),
        policy_architecture=str(compatibility["policy_architecture"]),
        artifact_root=str(context.output_dir / "runner"),
        train_profile=task_config.train_profile,
        eval_profile=task_config.eval_profile,
        train_num_envs=task_config.train_num_envs,
        training_agent_configuration=task_config.training_agent_configuration(),
        train_max_iterations=task_config.train_max_iterations,
        train_steps_per_iteration=task_config.train_steps_per_iteration,
        eval_base_seed=task_config.seed_resolver().base_seed,
        train_seed=task_config.train_seed,
        runtime_config_path=composition_config.runtime_config,
        train_runtime_dynamics_hash=(
            task_config.train_runtime_profile().runtime_dynamics_hash
        ),
        eval_runtime_dynamics_hash=(
            task_config.eval_runtime_profile().runtime_dynamics_hash
        ),
        work_dir=str(context.project_root),
        curriculum_capabilities=StairClimbingTaskPlugin.curriculum_capabilities,
    )


__all__ = [
    "DEFAULT_STAIR_EVAL_ENTRYPOINT",
    "DEFAULT_STAIR_PROTOCOL_V2",
    "DEFAULT_STAIR_TRAIN_ENTRYPOINT",
    "StairClimbingCompositionConfig",
    "StairClimbingCompositionProvider",
]

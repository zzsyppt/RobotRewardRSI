"""Push recovery task-owned fixed configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.robots.defaults import (
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    GO2_12DOF_ISAACLAB_ADAPTER_ID,
    build_default_robot_registry,
)
from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol
from true_rl_agent.tasks.push_recovery.force_protocol import ForceEvaluationProtocol

PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID = "push.g1_37dof.recovery_v2"
PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID = "push.go2_12dof.recovery_v2"
PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID = "push.g1_37dof.recovery_v3"
PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID = "push.go2_12dof.recovery_v3"
PUSH_ACTIVE_CELLS = (PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID, PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID)
PUSH_FORCE_EVAL_BASE_SEED = 1500000000
OBSERVED_REFERENCE_ADMISSION = "observed_outcomes_v1"
PUSH_RECOVERY_RUNTIME_ADAPTERS = {
    PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID: G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID: GO2_12DOF_ISAACLAB_ADAPTER_ID,
    PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID: G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID: GO2_12DOF_ISAACLAB_ADAPTER_ID,
}

PHASE8B_TRAIN_NUM_ENVS = 4096
PHASE8B_TRAIN_MAX_ITERATIONS = 500
PHASE8B_TRAIN_STEPS_PER_ITERATION = 24
PHASE8B_ENV_STEP_BUDGET = PHASE8B_TRAIN_NUM_ENVS * PHASE8B_TRAIN_MAX_ITERATIONS * PHASE8B_TRAIN_STEPS_PER_ITERATION
PHASE8B_RUNNER_TIMEOUT_S = 7200

@dataclass(frozen=True)
class PushRecoveryTaskConfig:
    success_predicate: str = "no_fall_and_recovers_velocity_after_push"
    eval_seed_visibility: str = "counts_only_no_raw_seed_ids"
    budget_steps: int = PHASE8B_ENV_STEP_BUDGET
    train_num_envs: int = PHASE8B_TRAIN_NUM_ENVS
    train_max_iterations: int = PHASE8B_TRAIN_MAX_ITERATIONS
    train_steps_per_iteration: int = PHASE8B_TRAIN_STEPS_PER_ITERATION
    eval_num_envs: int = 64
    public_seed_count: int = 3
    public_accept_seed_count: int = 3
    sealed_validation_seed_count: int = 7
    eval_protocol_version: str = DEFAULT_EVAL_PROTOCOL_VERSION
    train_seed: int = 42
    eval_base_seed: int = 42
    runner_timeout_s: int = PHASE8B_RUNNER_TIMEOUT_S
    checkpoint_selector: str = "runner_sealed_train_output_best_public_dev"
    acceptance_score_source: str = "public_accept.score_stats.mean"
    absolute_accept_floor: float = 0.05
    min_policy_delta: float = 0.05
    runtime_config_path: str | None = None
    train_profile: str = "strong_push"
    eval_profile: str = "strong_push"
    reference_program_genesis: bool = False
    cell_variant_id: str | None = None
    push_protocol: PushRecoveryProtocol | None = None
    push_evaluation_protocol: ForceEvaluationProtocol | None = None
    runtime_adapter_id: str | None = None
    push_robot_scale_m: float | None = None
    resolved_cell_task_protocol_hash: str | None = None
    reference_program_id: str | None = None
    reference_admission_policy: str | None = None

    @classmethod
    def for_cell(cls, cell_variant_id: str, **values: Any) -> "PushRecoveryTaskConfig":
        if cell_variant_id in {"push.g1_29dof.legacy_v1", "push.g1_29dof.reference_v1"}:
            values["reference_program_genesis"] = cell_variant_id.endswith("reference_v1")
            return cls(**values)
        if cell_variant_id not in PUSH_RECOVERY_RUNTIME_ADAPTERS:
            raise ValueError("unsupported Push recovery cell: " + cell_variant_id)
        registry = build_default_robot_registry()
        adapter_id = PUSH_RECOVERY_RUNTIME_ADAPTERS[cell_variant_id]
        profile = registry.profile(registry.runtime_adapter(adapter_id).robot_profile_id)
        values["reference_program_genesis"] = True
        values["success_predicate"] = "received_push_then_stable_recovery_dwell_without_failure"
        values["absolute_accept_floor"] = 0.2
        if cell_variant_id in PUSH_ACTIVE_CELLS:
            values.setdefault("reference_admission_policy", OBSERVED_REFERENCE_ADMISSION)
            if values["reference_admission_policy"] == OBSERVED_REFERENCE_ADMISSION:
                values["absolute_accept_floor"] = 0.0
            # Preserve the declared training distributions; only evaluation advances.
            from dataclasses import replace
            from true_rl_agent.tasks.push_recovery.reference_program import STATIONARY_REFERENCE_ID
            training_rows = {
                PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID: (0.225, 0.45, 0.75),
                PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID: (0.30, 0.60, 1.00),
            }
            low, medium, high = training_rows[cell_variant_id]
            values.setdefault("push_protocol", replace(PushRecoveryProtocol.defaults(),
                low_velocity_scale=low, medium_velocity_scale=medium, high_velocity_scale=high))
            values.setdefault("push_evaluation_protocol", ForceEvaluationProtocol())
            values.setdefault("eval_base_seed", PUSH_FORCE_EVAL_BASE_SEED)
            if values.get("reference_program_id") is None:
                values["reference_program_id"] = STATIONARY_REFERENCE_ID
            values["success_predicate"] = "complete_force_sequence_then_fresh_stable_dwell_and_final_stable_without_fall"
        values.setdefault("push_protocol", PushRecoveryProtocol.defaults())
        return cls(
            cell_variant_id=cell_variant_id,
            runtime_adapter_id=adapter_id,
            push_robot_scale_m=float(profile.runtime_facts["morphology"]["default_root_height_m"]),
            **values,
        )

    @classmethod
    def from_manifest(cls, manifest: Any, **values: Any) -> "PushRecoveryTaskConfig":
        projection = manifest.robot_task_binding["task_projection"]
        if manifest.cell_variant_id in PUSH_ACTIVE_CELLS:
            if "push_evaluation_protocol" not in projection or "push_resolved_evaluation_protocol" not in projection:
                raise ValueError("Push v3 manifest requires its frozen force evaluation")
            frozen_force = ForceEvaluationProtocol.from_mapping(projection["push_evaluation_protocol"])
            if frozen_force.resolve(projection["push_robot_scale_m"]) != projection["push_resolved_evaluation_protocol"]:
                raise ValueError("Push v3 resolved evaluation differs from its frozen source")
        if manifest.cell_variant_id in PUSH_RECOVERY_RUNTIME_ADAPTERS:
            # An absent field belongs to the previously frozen numeric-band policy.
            values["reference_admission_policy"] = projection.get("reference_admission_policy")
            values["push_protocol"] = PushRecoveryProtocol.from_mapping(projection["push_protocol"])
            values["resolved_cell_task_protocol_hash"] = manifest.resolved_cell_task_protocol_hash
            values["reference_program_id"] = projection.get("reference_program_id")
            if "push_evaluation_protocol" in projection:
                values["push_evaluation_protocol"] = ForceEvaluationProtocol.from_mapping(projection["push_evaluation_protocol"])
        config = cls.for_cell(manifest.cell_variant_id, **values)
        if config.push_protocol is not None and (
            projection["push_robot_scale_m"] != config.push_robot_scale_m
            or manifest.runtime_adapter["runtime_adapter_id"] != config.runtime_adapter_id
        ):
            raise ValueError("Push manifest robot scale or runtime adapter differs from its row")
        return config

    def __post_init__(self) -> None:
        if self.reference_admission_policy not in (None, OBSERVED_REFERENCE_ADMISSION):
            raise ValueError("unknown Push reference admission policy")
        if self.reference_admission_policy is not None and self.push_evaluation_protocol is None:
            raise ValueError("observed reference admission requires the force evaluation rows")
        if self.reference_admission_policy is not None and self.absolute_accept_floor != 0.0:
            raise ValueError("observed reference admission has no absolute success-rate floor")
        if (self.cell_variant_id in PUSH_ACTIVE_CELLS) != (self.push_evaluation_protocol is not None):
            raise ValueError("Push force evaluation belongs only to the active v3 rows")
        if self.push_evaluation_protocol is not None:
            if not isinstance(self.push_evaluation_protocol, ForceEvaluationProtocol):
                raise TypeError("push_evaluation_protocol must be ForceEvaluationProtocol")
            self.push_evaluation_protocol.resolve(self.push_robot_scale_m)
        if self.reference_program_id is not None:
            from true_rl_agent.tasks.push_recovery.reference_program import reference_program_id

            reference_program_id(self)
        if self.push_protocol is None:
            if any(value is not None for value in (
                self.cell_variant_id, self.runtime_adapter_id, self.push_robot_scale_m,
                self.resolved_cell_task_protocol_hash,
            )):
                raise ValueError("Push recovery row requires its physical protocol")
        else:
            if not isinstance(self.push_protocol, PushRecoveryProtocol):
                raise TypeError("push_protocol must be a PushRecoveryProtocol")
            if self.runtime_config_path is not None:
                raise ValueError("Push recovery v2 uses its manifest protocol; runtime_config_path is legacy-only")
            if (not self.reference_program_genesis
                or PUSH_RECOVERY_RUNTIME_ADAPTERS.get(self.cell_variant_id) != self.runtime_adapter_id):
                raise ValueError("Push recovery protocol requires a declared reference-program robot row")
            self.push_protocol.resolve(self.push_robot_scale_m)
            if self.resolved_cell_task_protocol_hash is not None:
                from true_rl_agent.evaluation.binding import require_sha256_digest

                require_sha256_digest(self.resolved_cell_task_protocol_hash, "Push resolved task protocol hash")
        if not isinstance(self.reference_program_genesis, bool):
            raise TypeError("reference_program_genesis must be boolean")
        for field_name in (
            "success_predicate",
            "eval_seed_visibility",
            "eval_protocol_version",
            "checkpoint_selector",
            "acceptance_score_source",
            "train_profile",
            "eval_profile",
        ):
            _require_non_empty_string(getattr(self, field_name), field_name)
        for field_name in (
            "budget_steps",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "runner_timeout_s",
        ):
            _require_positive_integer(getattr(self, field_name), field_name)
        for field_name in ("train_seed", "eval_base_seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("absolute_accept_floor", "min_policy_delta"):
            _require_finite_number(getattr(self, field_name), field_name)
        if self.runtime_config_path is not None:
            _require_non_empty_string(
                self.runtime_config_path,
                "runtime_config_path",
            )

        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.train_num_envs,
            num_steps_per_env=self.train_steps_per_iteration,
            max_iterations=self.train_max_iterations,
        )

        # Reuse the protocol constructors for task-owned relational validation.
        self.evaluation_protocol()
        self.acceptance_protocol()

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    def acceptance_protocol(self) -> AcceptanceProtocol:
        return AcceptanceProtocol(
            protocol_id=("unitree_push_recovery_acceptance_v3" if self.reference_admission_policy is not None
                         else "unitree_push_recovery_acceptance_v2" if self.push_protocol is not None
                         else "unitree_push_recovery_acceptance_v1"),
            score_source=self.acceptance_score_source,
            min_score_count=self.public_accept_seed_count,
            absolute_accept_floor=self.absolute_accept_floor,
            min_delta=self.min_policy_delta,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        )

    def seed_cohort_hash(self) -> str:
        from true_rl_agent.tasks.push_recovery.seed_policy import (
            build_seed_cohort_hash,
        )

        return build_seed_cohort_hash(
            base_seed=self.eval_base_seed,
            evaluation_protocol=self.evaluation_protocol(),
        )

    def train_runtime_dynamics(self) -> dict:
        if self.push_protocol is not None:
            return self._recovery_runtime_dynamics(self.train_profile)
        from true_rl_agent.tasks.push_recovery.isaac_config import (
            load_public_runtime_dynamics,
        )

        return load_public_runtime_dynamics(self.runtime_config_path, self.train_profile)

    def eval_runtime_dynamics(self) -> dict:
        if self.push_protocol is not None:
            return self._recovery_runtime_dynamics(self.eval_profile, evaluation=True)
        from true_rl_agent.tasks.push_recovery.isaac_config import (
            load_public_runtime_dynamics,
        )

        return load_public_runtime_dynamics(self.runtime_config_path, self.eval_profile)

    def _recovery_runtime_dynamics(self, profile: str, *, evaluation: bool = False) -> dict[str, Any]:
        from true_rl_agent.robots.isaaclab_locomotion import isaaclab_locomotion_runtime_spec
        from true_rl_agent.tasks.push_recovery.isaac_config import public_runtime_dynamics

        assert self.push_protocol is not None and self.runtime_adapter_id is not None
        runtime = isaaclab_locomotion_runtime_spec(self.runtime_adapter_id)
        config = {
            "push_protocol": self.push_protocol.to_dict(),
            "push_robot_scale_m": self.push_robot_scale_m,
            "runtime_adapter_id": self.runtime_adapter_id,
            "active_profile": profile,
            "task": {
                "runtime_adapter_id": self.runtime_adapter_id,
                "environment_cfg_entrypoint": runtime.environment_cfg_entrypoint,
                "runner_cfg_entrypoint": runtime.runner_cfg_entrypoint,
                "agent_configuration_hash": runtime.agent_configuration.agent_configuration_hash,
            },
        }
        if evaluation and self.push_evaluation_protocol is not None:
            registry = build_default_robot_registry()
            robot = registry.profile(registry.runtime_adapter(self.runtime_adapter_id).robot_profile_id)
            config["push_evaluation_protocol"] = self.push_evaluation_protocol.to_dict()
            config["force_body_name"] = robot.semantic_roles["base"].targets[0]
        return public_runtime_dynamics(config)

    @property
    def accept_threshold(self) -> float:
        """Compatibility projection for prompt text; protocol is authoritative."""

        return self.min_policy_delta


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(value: Any, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    return value

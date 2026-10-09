"""Push-owned, robot-independent physical protocol and dimensional resolution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.push_recovery.force_protocol import ForceEvaluationProtocol

PUSH_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.task.push_recovery.protocol.v2"
PUSH_RESOLVED_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.task.push_recovery.resolved_protocol.v2"
PUSH_STRATA = ("low", "medium", "high", "repeated")


@dataclass(frozen=True)
class PushRecoveryProtocol:
    schema_version: str = PUSH_PROTOCOL_SCHEMA_VERSION
    protocol_id: str = "push_recovery_planar_delta_velocity_v2"
    protocol_state: str = "active"
    gravity_mps2: float = 9.81
    episode_time_scale: float = 30.0
    first_push_time_scale: float = 5.0
    second_push_time_scale: float = 17.0
    recovery_window_time_scale: float = 8.0
    stable_dwell_time_scale: float = 1.5
    low_velocity_scale: float = 0.15
    medium_velocity_scale: float = 0.30
    high_velocity_scale: float = 0.50
    recovery_velocity_scale: float = 0.08
    recovery_angular_velocity_scale: float = 0.5
    recovery_tilt_rad: float = 0.20
    recovery_height_scale: float = 0.70
    fall_height_scale: float = 0.45
    fall_tilt_rad: float = 0.80
    max_displacement_scale: float = 3.0
    injection_tolerance_mps: float = 1e-5

    def __post_init__(self) -> None:
        if self.schema_version != PUSH_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported Push physical protocol schema")
        if self.protocol_state not in {"draft", "active", "archived"}:
            raise ValueError("invalid Push protocol state")
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ValueError("Push protocol id must be nonempty")
        for field in fields(self):
            if field.name in {"schema_version", "protocol_id", "protocol_state"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Push protocol {field.name} must be positive and finite")
        if not self.low_velocity_scale < self.medium_velocity_scale < self.high_velocity_scale:
            raise ValueError("Push severity levels must be strictly ordered")
        if self.stable_dwell_time_scale > self.recovery_window_time_scale:
            raise ValueError("Push dwell exceeds recovery window")
        if self.first_push_time_scale + self.recovery_window_time_scale >= self.second_push_time_scale:
            raise ValueError("Push recovery windows must not overlap")
        if self.second_push_time_scale + self.recovery_window_time_scale >= self.episode_time_scale:
            raise ValueError("Push episode does not contain complete recovery windows")
        if not self.fall_height_scale < self.recovery_height_scale <= 1.0:
            raise ValueError("Push height thresholds are inconsistent")
        if self.recovery_tilt_rad >= self.fall_tilt_rad:
            raise ValueError("Push recovery tilt must be below fall tilt")

    @classmethod
    def defaults(cls) -> "PushRecoveryProtocol":
        return cls()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PushRecoveryProtocol":
        if not isinstance(value, Mapping) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("Push protocol fields mismatch")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def protocol_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def resolve(self, robot_scale_m: float) -> dict[str, Any]:
        if isinstance(robot_scale_m, bool) or not isinstance(robot_scale_m, (int, float)) or not math.isfinite(robot_scale_m) or robot_scale_m <= 0:
            raise ValueError("Push robot_scale_m must be positive and finite")
        length = float(robot_scale_m)
        velocity = math.sqrt(self.gravity_mps2 * length)
        time = math.sqrt(length / self.gravity_mps2)
        result = {
            "schema_version": PUSH_RESOLVED_PROTOCOL_SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash,
            "robot_scale_m": length,
            "mechanism": "world_planar_velocity_increment",
            "velocity_scale_mps": velocity,
            "time_scale_s": time,
            "episode_length_s": self.episode_time_scale * time,
            "push_times_s": [self.first_push_time_scale * time, self.second_push_time_scale * time],
            "recovery_window_s": self.recovery_window_time_scale * time,
            "stable_dwell_s": self.stable_dwell_time_scale * time,
            "velocity_magnitudes_mps": {"low": self.low_velocity_scale * velocity,
                "medium": self.medium_velocity_scale * velocity, "high": self.high_velocity_scale * velocity},
            "recovery_velocity_mps": self.recovery_velocity_scale * velocity,
            "recovery_angular_velocity_radps": self.recovery_angular_velocity_scale / time,
            "recovery_tilt_rad": self.recovery_tilt_rad,
            "recovery_height_m": self.recovery_height_scale * length,
            "fall_height_m": self.fall_height_scale * length,
            "fall_tilt_rad": self.fall_tilt_rad,
            "max_displacement_m": self.max_displacement_scale * length,
            "injection_tolerance_mps": self.injection_tolerance_mps,
            "reset": {"root_pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [-math.pi, math.pi]},
                "root_velocity_range": {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
                "joint_position_scale": [1.0, 1.0], "joint_velocity_range": [0.0, 0.0]},
            "evaluation_cohort_counts": {name: 16 for name in PUSH_STRATA},
            "evaluation_direction_policy": {"count": 16, "phase": "python_random_runner_seed_uniform_0_2pi_v1",
                "angular_range_rad": [0.0, 2 * math.pi], "second_event_offset_rad": math.pi / 2},
            "training_sampling": {"stratum_probabilities": {name: 0.25 for name in PUSH_STRATA},
                "angular_range_rad": [0.0, 2 * math.pi], "event_directions": "independent_uniform"},
        }
        result["resolved_protocol_hash"] = canonical_json_hash(result)
        return result


def physical_protocol_from_config(config: Mapping[str, Any]) -> tuple[PushRecoveryProtocol | ForceEvaluationProtocol, dict[str, Any]] | None:
    if "push_protocol" not in config:
        if "push_robot_scale_m" in config:
            raise ValueError("Push robot scale requires a physical protocol")
        return None
    if "push_evaluation_protocol" in config:
        protocol = ForceEvaluationProtocol.from_mapping(config["push_evaluation_protocol"])
    else:
        protocol = PushRecoveryProtocol.from_mapping(config["push_protocol"])
    return protocol, protocol.resolve(config["push_robot_scale_m"])


def physical_public_runtime_dynamics(config: Mapping[str, Any]) -> dict[str, Any]:
    physical = physical_protocol_from_config(config)
    if physical is None:
        raise ValueError("physical Push runtime requires its protocol")
    protocol, resolved = physical
    adapter_id = config.get("runtime_adapter_id")
    task = config.get("task")
    if not isinstance(adapter_id, str) or not adapter_id or not isinstance(task, Mapping):
        raise ValueError("physical Push runtime requires robot execution metadata")
    if task.get("runtime_adapter_id") != adapter_id:
        raise ValueError("physical Push runtime adapter identity mismatch")
    expected = {"runtime_adapter_id", "environment_cfg_entrypoint", "runner_cfg_entrypoint", "agent_configuration_hash"}
    if set(task) != expected or not all(isinstance(value, str) and value for value in task.values()):
        raise ValueError("physical Push runtime task metadata mismatch")
    profile = config.get("active_profile")
    if profile not in {"recovery_v2", "strong_push", "no_push"}:
        raise ValueError("unsupported physical Push runtime profile")
    payload = {
        "schema_version": "true_rl_agent.task.push_recovery.public_runtime_dynamics.v3",
        "active_profile": profile, "runner_owned": True, "candidate_can_change": False,
        "runtime_adapter_id": adapter_id, "task": dict(task),
        "push_protocol": protocol.to_dict(), "push_robot_scale_m": float(config["push_robot_scale_m"]),
        "push_resolved_protocol": resolved,
        "environment": {"episode_length_s": resolved["episode_length_s"]},
        "standing_command": {"commanded_locomotion_is_zero": True},
        "reward_execution": {"mode": "complete_program_replacement", "output": "reward_rate",
            "integration": "reward_manager_once_by_control_dt", "state": "pre_reset_transition"},
        "observation_corruption": {"train": True, "eval": False},
    }
    if "push_evaluation_protocol" in config:
        payload["schema_version"] = "true_rl_agent.task.push_recovery.public_runtime_dynamics.v4"
        payload["push_evaluation_protocol"] = payload.pop("push_protocol")
        payload["push_resolved_evaluation_protocol"] = payload.pop("push_resolved_protocol")
        payload["force_body_name"] = config["force_body_name"]
    payload["runtime_dynamics_hash"] = "sha256:" + canonical_json_hash(payload)
    return payload

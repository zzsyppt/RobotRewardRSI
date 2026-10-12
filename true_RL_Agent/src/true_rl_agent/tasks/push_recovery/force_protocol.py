"""Task-owned finite-force evaluation; robot facts only supply length and mass.

Impulse velocity is external impulse / total robot mass, NOT a velocity write.
Training protocols and checkpoint training lineage remain separately frozen.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash

FORCE_EVALUATION_SCHEMA = "true_rl_agent.task.push_recovery.force_evaluation.v1"
FORCE_MECHANISM = "world_planar_external_force"


@dataclass(frozen=True)
class ForceEvaluationProtocol:
    schema_version: str = FORCE_EVALUATION_SCHEMA
    protocol_id: str = "push_force_sequence_no_displacement_v1"
    gravity_mps2: float = 9.81
    low_impulse_scale: float = 0.60
    medium_impulse_scale: float = 1.20
    high_impulse_scale: float = 2.00
    repeated_high_fraction: float = 0.80
    repeated_gap_s: float = 0.22
    pulse_duration_time_scale: float = 1.0
    first_push_time_scale: float = 5.0
    episode_time_scale: float = 30.0
    recovery_window_time_scale: float = 8.0
    stable_dwell_time_scale: float = 1.5
    recovery_velocity_scale: float = 0.08
    recovery_angular_velocity_scale: float = 0.5
    recovery_tilt_rad: float = 0.20
    recovery_height_scale: float = 0.70
    fall_height_scale: float = 0.45
    fall_tilt_rad: float = 0.80
    force_tolerance_n: float = 0.001

    def __post_init__(self) -> None:
        if self.schema_version != FORCE_EVALUATION_SCHEMA or not self.protocol_id:
            raise ValueError("unsupported Push force evaluation protocol")
        for field in fields(self):
            if field.name in {"schema_version", "protocol_id"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
        if not self.low_impulse_scale < self.medium_impulse_scale < self.high_impulse_scale:
            raise ValueError("Push force severity must increase")
        if self.repeated_high_fraction > 1 or self.stable_dwell_time_scale > self.recovery_window_time_scale:
            raise ValueError("invalid Push sequence fraction or recovery dwell")
        if self.fall_height_scale >= self.recovery_height_scale or self.recovery_tilt_rad >= self.fall_tilt_rad:
            raise ValueError("Push recovery must be inside the failure boundaries")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ForceEvaluationProtocol":
        if set(value) != {field.name for field in fields(cls)}:
            raise ValueError("Push force evaluation requires its complete frozen mapping")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def protocol_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def resolve(self, robot_scale_m: float) -> dict[str, Any]:
        if isinstance(robot_scale_m, bool) or not isinstance(robot_scale_m, (int, float)) or not math.isfinite(robot_scale_m) or robot_scale_m <= 0:
            raise ValueError("Push force scale must be a positive RobotProfile length")
        length = float(robot_scale_m)
        velocity, time = math.sqrt(self.gravity_mps2 * length), math.sqrt(length / self.gravity_mps2)
        result = {
            "schema_version": "true_rl_agent.task.push_recovery.resolved_force_evaluation.v1",
            "protocol_hash": self.protocol_hash, "mechanism": FORCE_MECHANISM,
            "robot_scale_m": length, "velocity_scale_mps": velocity, "time_scale_s": time,
            "scale_source": "RobotProfile.runtime_facts.morphology.default_root_height_m",
            "force_application": "world_horizontal_at_bound_base_link_com_zero_direct_torque",
            "mass_source": "sum_actual_link_masses_after_reset",
            "force_rule": "mass_kg * impulse_velocity_mps / quantized_pulse_duration_s",
            "control_step_quantization": "ceil_duration_over_control_dt",
            "recovery_scope": "all_planned_pulses_then_one_fresh_window_and_final_stable",
            "displacement_failure_rule": "disabled_displacement_is_diagnostic_only",
            "episode_length_s": self.episode_time_scale * time,
            "push_times_s": [self.first_push_time_scale * time],
            "pulse_duration_s": self.pulse_duration_time_scale * time,
            "repeated_gap_s": self.repeated_gap_s, "repeated_high_fraction": self.repeated_high_fraction,
            "impulse_velocity_mps": {name: getattr(self, name + "_impulse_scale") * velocity for name in ("low", "medium", "high")},
            "recovery_window_s": self.recovery_window_time_scale * time,
            "stable_dwell_s": self.stable_dwell_time_scale * time,
            "recovery_velocity_mps": self.recovery_velocity_scale * velocity,
            "recovery_angular_velocity_radps": self.recovery_angular_velocity_scale / time,
            "recovery_tilt_rad": self.recovery_tilt_rad, "recovery_height_m": self.recovery_height_scale * length,
            "fall_height_m": self.fall_height_scale * length, "fall_tilt_rad": self.fall_tilt_rad,
            "max_displacement_m": None, "force_tolerance_n": self.force_tolerance_n,
            "reset": {"root_pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [-math.pi, math.pi]},
                "root_velocity_range": {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
                "joint_position_scale": [1.0, 1.0], "joint_velocity_range": [0.0, 0.0]},
            "evaluation_cohort_counts": {name: 16 for name in ("low", "medium", "high", "repeated")},
            "evaluation_direction_policy": {"count": 16, "phase": "python_random_runner_seed_uniform_0_2pi_v1",
                "angular_range_rad": [0.0, 2 * math.pi], "second_event_offset_rad": math.pi / 2},
        }
        # The fixed gap is not scaled by morphology; check the dimensional schedule.
        if (result["push_times_s"][0] + 2 * result["pulse_duration_s"] + self.repeated_gap_s
                + result["recovery_window_s"] >= result["episode_length_s"]):
            raise ValueError("Push sequence recovery must fit inside its episode")
        result["resolved_protocol_hash"] = canonical_json_hash(result)
        return result

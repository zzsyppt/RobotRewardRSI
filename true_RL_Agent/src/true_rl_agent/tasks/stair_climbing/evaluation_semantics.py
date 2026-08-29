"""Versioned task-owned outcome semantics for Stair evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import StairProtocolV2

STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH = (
    "configs/tasks/stair_evaluation_semantics_v7.json"
)


@dataclass(frozen=True)
class StairEvaluationSemantics:
    """Immutable outcome contract layered over the historical V1 geometry."""

    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_protocol: StairProtocolV1,
        training_protocol: StairProtocolV2,
    ) -> "StairEvaluationSemantics":
        payload = load_strict_json_object(
            Path(path),
            context="Stair evaluation semantics",
        )
        _validate_semantics(
            payload,
            base_protocol=base_protocol,
            training_protocol=training_protocol,
        )
        return cls(_payload=_freeze_json(payload))

    @classmethod
    def from_task_outcomes(
        cls,
        *,
        task_protocol_hash: str,
        task_id: str,
        episode_outcomes: Mapping[str, Any],
    ) -> "StairEvaluationSemantics":
        """Bind morphology-neutral outcomes already validated by a task protocol."""

        payload = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.evaluation_semantics.unified.v1"
            ),
            "task_id": task_id,
            "task_protocol_hash": task_protocol_hash,
            "episode_outcomes": _thaw_json(_freeze_json(episode_outcomes)),
        }
        payload["protocol_hash"] = canonical_json_hash(payload)
        return cls(_payload=_freeze_json(payload))

    @property
    def protocol_hash(self) -> str:
        return str(self._payload["protocol_hash"])

    @property
    def outcomes(self) -> Mapping[str, Any]:
        return _mapping(self._payload["episode_outcomes"], "episode_outcomes")

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


def effective_stair_task_protocol_hash(
    *,
    training_protocol: StairProtocolV2,
    evaluation_semantics: StairEvaluationSemantics,
    acceptance_protocol: AcceptanceProtocol,
    evaluation_profile_hash: str,
) -> str:
    """Bind training, outcome, and acceptance authorities for a new session."""

    return canonical_json_hash(
        {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.effective_protocol.v3"
            ),
            "training_protocol_hash": training_protocol.protocol_hash,
            "evaluation_semantics_hash": evaluation_semantics.protocol_hash,
            "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
            "evaluation_profile_hash": evaluation_profile_hash,
        }
    )


def _validate_semantics(
    payload: Mapping[str, Any],
    *,
    base_protocol: StairProtocolV1,
    training_protocol: StairProtocolV2,
) -> None:
    expected_keys = {
        "base_protocol_hash",
        "episode_outcomes",
        "protocol_hash",
        "protocol_id",
        "schema_version",
        "status",
        "task_id",
        "training_protocol_hash",
    }
    if set(payload) != expected_keys:
        raise ValueError("Stair evaluation semantics fields mismatch")
    if payload["schema_version"] != (
        "true_rl_agent.task.stair_climbing.evaluation_semantics.v7"
    ):
        raise ValueError("Stair evaluation semantics schema mismatch")
    if payload["protocol_id"] != "unitree_stair_climbing_evaluation_semantics_v7":
        raise ValueError("Stair evaluation semantics protocol_id mismatch")
    if payload["status"] != "active_morphology_neutral_stable_top_completion":
        raise ValueError("Stair evaluation semantics status mismatch")
    if payload["task_id"] != base_protocol.task_id:
        raise ValueError("Stair evaluation semantics task_id mismatch")
    if payload["base_protocol_hash"] != base_protocol.protocol_hash:
        raise ValueError("Stair evaluation semantics base protocol mismatch")
    if payload["training_protocol_hash"] != training_protocol.protocol_hash:
        raise ValueError("Stair evaluation semantics training protocol mismatch")
    expected_hash = canonical_json_hash(
        {key: value for key, value in payload.items() if key != "protocol_hash"}
    )
    if payload["protocol_hash"] != expected_hash:
        raise ValueError("Stair evaluation semantics hash mismatch")

    outcomes = _mapping(payload["episode_outcomes"], "episode_outcomes")
    expected_outcome_keys = {
        "classification_warning_blocks_pass",
        "counting_invariant",
        "diagnostic_flags",
        "failure_priority",
        "primary_reasons",
        "signals",
        "success",
        "top_platform_stability_requirement",
    }
    if set(outcomes) != expected_outcome_keys:
        raise ValueError("Stair evaluation outcome fields mismatch")
    if outcomes["classification_warning_blocks_pass"] is not True:
        raise ValueError("Stair classification warnings must block evaluation")
    if outcomes["counting_invariant"] != (
        "success_count_plus_primary_failure_counts_equals_total_episode_count"
    ):
        raise ValueError("Stair evaluation counting invariant mismatch")
    primary_reasons = _unique_strings(outcomes["primary_reasons"], "primary_reasons")
    failure_priority = _unique_strings(outcomes["failure_priority"], "failure_priority")
    expected_failures = (
        "forbidden_contact",
        "fall",
        "off_course",
        "base_orientation_instability",
        "slip",
        "insufficient_clearance",
        "timeout",
    )
    if primary_reasons != ("success", *expected_failures):
        raise ValueError("Stair evaluation primary taxonomy mismatch")
    if failure_priority != expected_failures:
        raise ValueError("Stair evaluation failure taxonomy mismatch")
    if _unique_strings(outcomes["diagnostic_flags"], "diagnostic_flags") != (
        "base_orientation_instability_detected",
        "slip_detected",
        "insufficient_clearance_detected",
        "off_course_detected",
        "goal_not_reached_detected",
        "first_valid_top_arrival_achieved",
        "post_arrival_safety_failure_detected",
    ):
        raise ValueError("Stair evaluation diagnostic taxonomy mismatch")

    signals = _mapping(outcomes["signals"], "signals")
    required_signals = {
        "base_orientation_instability",
        "fall",
        "insufficient_clearance",
        "off_course",
        "slip",
        "timeout",
        "forbidden_contact",
    }
    if set(signals) != required_signals:
        raise ValueError("Stair evaluation signal taxonomy mismatch")
    signal_fields = {
        "base_orientation_instability": {
            "comparison",
            "consecutive_control_steps",
            "source",
            "threshold",
            "unit",
        },
        "fall": {
            "comparison",
            "consecutive_control_steps",
            "maximum_projected_surface_penetration_m",
            "no_contact_fallback",
            "source",
            "threshold",
            "unit",
        },
        "insufficient_clearance": {
            "comparison",
            "consecutive_control_steps",
            "source",
            "threshold",
            "unit",
        },
        "off_course": {
            "comparison",
            "consecutive_control_steps",
            "source",
            "unit",
        },
        "slip": {
            "consecutive_control_steps",
            "contact_force_threshold",
            "force_unit",
            "source",
            "speed_threshold",
            "speed_unit",
        },
        "timeout": {"deadline_authority", "source", "unit"},
        "forbidden_contact": {
            "manager_term",
            "other_body_contacts_policy",
            "source",
            "trigger",
        },
    }
    for signal_name, expected_fields in signal_fields.items():
        signal = _mapping(signals[signal_name], signal_name)
        if set(signal) != expected_fields:
            raise ValueError(f"Stair {signal_name} fields mismatch")
    expected_descriptors = {
        "base_orientation_instability": {
            "comparison": "greater_than",
            "source": "kinematic_base_roll_or_pitch_abs",
            "unit": "rad",
        },
        "fall": {
            "comparison": "less_than",
            "no_contact_fallback": (
                "lowest_vertically_feasible_projected_support_surface"
            ),
            "source": (
                "kinematic_base_height_above_lowest_vertically_feasible_"
                "contacting_support_surface"
            ),
            "unit": "m",
        },
        "insufficient_clearance": {
            "comparison": "less_than",
            "source": (
                "noncontact_support_effector_clearance_above_projected_"
                "local_stair_surface"
            ),
            "unit": "m",
        },
        "off_course": {
            "comparison": "outside",
            "source": "kinematic_base_xy_outside_own_terrain_patch",
            "unit": "m",
        },
        "slip": {
            "force_unit": "N",
            "source": "support_effector_planar_speed_while_in_contact",
            "speed_unit": "m/s",
        },
        "timeout": {
            "deadline_authority": "stair_evaluation_profile",
            "source": "evaluation_profile_cell_episode_horizon",
            "unit": "s",
        },
    }
    for signal_name, descriptors in expected_descriptors.items():
        signal = _mapping(signals[signal_name], signal_name)
        if any(signal[key] != value for key, value in descriptors.items()):
            raise ValueError(f"Stair {signal_name} descriptor mismatch")
    for signal_name in (
        "base_orientation_instability",
        "fall",
        "insufficient_clearance",
        "off_course",
        "slip",
    ):
        _positive_int(
            _mapping(signals[signal_name], signal_name)[
                "consecutive_control_steps"
            ],
            f"{signal_name} consecutive_control_steps",
        )
    if signals["off_course"]["consecutive_control_steps"] != 1:
        raise ValueError("Stair off-course ownership latching must be immediate")
    for signal_name, threshold_name in (
        ("base_orientation_instability", "threshold"),
        ("fall", "threshold"),
        ("fall", "maximum_projected_surface_penetration_m"),
        ("insufficient_clearance", "threshold"),
        ("slip", "contact_force_threshold"),
        ("slip", "speed_threshold"),
    ):
        threshold = _finite_number(
            _mapping(signals[signal_name], signal_name)[threshold_name],
            f"{signal_name} {threshold_name}",
        )
        if threshold <= 0.0:
            raise ValueError(f"Stair {signal_name} {threshold_name} must be positive")
    forbidden_contact = _mapping(signals["forbidden_contact"], "forbidden_contact")
    if (
        forbidden_contact["source"] != "resolved_forbidden_contact_role_force"
        or forbidden_contact["manager_term"] != "base_contact"
        or forbidden_contact["trigger"] != "manager_term_active"
        or forbidden_contact["other_body_contacts_policy"]
        != "allowed_unless_another_failure_signal_triggers"
    ):
        raise ValueError("Stair forbidden-contact policy mismatch")
    success = _mapping(outcomes["success"], "success")
    if set(success) != {
        "completion_mode",
        "minimum_progress_fraction",
        "minimum_support_effectors_on_top",
        "minimum_support_effector_upward_force_n",
        "post_arrival_failure_policy",
        "pre_arrival_blocking_latches",
        "requires_orientation_stable",
        "requires_kinematic_base_clearance_above_top",
        "requires_kinematic_base_within_top_platform_bounds",
        "requires_support_effectors_within_top_platform_bounds",
        "support_effector_surface_clearance_range_m",
    }:
        raise ValueError("Stair success fields mismatch")
    if success.get("completion_mode") != (
        "stable_top_traversal_after_first_valid_top_arrival"
    ):
        raise ValueError("Stair success completion mode mismatch")
    if success.get("post_arrival_failure_policy") != (
        "blocks_success_until_stability_requirement_met"
    ):
        raise ValueError("Stair post-arrival failure policy mismatch")
    if _unique_strings(
        success.get("pre_arrival_blocking_latches"),
        "success pre_arrival_blocking_latches",
    ) != ("forbidden_contact", "fall", "off_course"):
        raise ValueError("Stair pre-arrival blocking latches mismatch")
    if _finite_number(
        success.get("minimum_progress_fraction"),
        "success minimum_progress_fraction",
    ) != 1.0:
        raise ValueError("Stair success progress fraction mismatch")
    if _positive_int(
        success.get("minimum_support_effectors_on_top"),
        "success support-effector count",
    ) < 1:
        raise ValueError("Stair success support-effector count is invalid")
    support_clearance_range = success.get("support_effector_surface_clearance_range_m")
    if (
        not isinstance(support_clearance_range, list)
        or len(support_clearance_range) != 2
    ):
        raise ValueError("Stair success support clearance range is invalid")
    minimum_clearance = _finite_number(
        support_clearance_range[0],
        "success minimum support-effector surface clearance",
    )
    maximum_clearance = _finite_number(
        support_clearance_range[1],
        "success maximum support-effector surface clearance",
    )
    if minimum_clearance >= maximum_clearance:
        raise ValueError("Stair success support clearance range is empty")
    for field_name in ("minimum_support_effector_upward_force_n",):
        if _finite_number(success.get(field_name), f"success {field_name}") <= 0.0:
            raise ValueError(f"Stair success {field_name} must be positive")
    required_success_flags = {
        "requires_orientation_stable",
        "requires_kinematic_base_clearance_above_top",
        "requires_kinematic_base_within_top_platform_bounds",
        "requires_support_effectors_within_top_platform_bounds",
    }
    if any(success.get(key) is not True for key in required_success_flags):
        raise ValueError("Stair success safety constraints must be enabled")

    stability = _mapping(
        outcomes["top_platform_stability_requirement"],
        "top_platform_stability_requirement",
    )
    if set(stability) != {
        "acceptance_role",
        "command_policy",
        "consecutive_control_steps",
        "maximum_support_effector_planar_speed_mps",
        "requires_first_valid_top_arrival_latched",
        "source",
    }:
        raise ValueError("Stair top-platform stability fields mismatch")
    expected_stability = {
        "acceptance_role": "primary_success_required",
        "command_policy": "fixed_forward_command_continues",
        "requires_first_valid_top_arrival_latched": True,
        "source": "valid_top_arrival_plus_support_effector_speed_bound",
    }
    if any(stability.get(key) != value for key, value in expected_stability.items()):
        raise ValueError("Stair top-platform stability policy mismatch")
    _positive_int(
        stability.get("consecutive_control_steps"),
        "top-platform stability consecutive_control_steps",
    )
    if _finite_number(
        stability.get("maximum_support_effector_planar_speed_mps"),
        "top-platform stability maximum_support_effector_planar_speed_mps",
    ) <= 0.0:
        raise ValueError("Stair top-platform stability speed must be positive")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stair evaluation {field_name} must be an object")
    return value


def _unique_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"Stair evaluation {field_name} must be unique strings")
    return tuple(value)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Stair evaluation {field_name} must be a positive integer")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Stair evaluation {field_name} must be finite")
    return float(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH",
    "StairEvaluationSemantics",
    "effective_stair_task_protocol_hash",
]

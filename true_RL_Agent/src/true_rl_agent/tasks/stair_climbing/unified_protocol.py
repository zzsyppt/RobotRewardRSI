"""Single-cell task protocol assembled from direct Stair-owned semantics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
    StairUnifiedEvaluationSuite,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    StairUnifiedTrainingProfile,
)
from true_rl_agent.tasks.stair_climbing.vnext_reward_authority import (
    StairVNextRewardAuthority,
)

STAIR_UNIFIED_PROTOCOL_RELATIVE_PATH = "configs/tasks/stair_unified_protocol_v1.json"
STAIR_UNIFIED_ACCEPTANCE_PROTOCOL_RELATIVE_PATH = (
    "configs/tasks/stair_unified_acceptance_protocol_v6.json"
)
STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.unified_protocol.v1"
)
STAIR_UNIFIED_TASK_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.task_projection.v1"
)
STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID = "stair.g1_37dof.unified_v1"

_FIELDS = {
    "schema_version",
    "protocol_id",
    "task_id",
    "cell_variant_id",
    "protocol_state",
    "training",
    "training_execution",
    "runtime_controls",
    "episode_outcomes",
    "candidate_authority",
    "component_refs",
    "reference_program",
    "reference_readiness",
    "required_pass_gates",
    "component_hashes",
    "task_protocol_hash",
    "protocol_hash",
}


@dataclass(frozen=True)
class StairUnifiedProtocol:
    """The only new-session Stair protocol; C0/C1 are sample regimes."""

    _payload: Mapping[str, Any] = field(repr=False)
    training_profile: StairUnifiedTrainingProfile = field(repr=False)
    evaluation_suite: StairUnifiedEvaluationSuite = field(repr=False)
    acceptance_protocol: AcceptanceProtocol = field(repr=False)
    reward_authority: StairVNextRewardAuthority = field(repr=False)
    candidate_capability_profile: CandidateCapabilityProfile = field(repr=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_cell_variant_id: str | None = None,
    ) -> "StairUnifiedProtocol":
        resolved = Path(path)
        payload = load_strict_json_object(resolved, context="Stair unified protocol")
        if set(payload) != _FIELDS:
            raise ValueError("Stair unified protocol fields mismatch")
        if payload.get("schema_version") != STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("Stair unified protocol schema mismatch")
        if payload.get("task_id") != "unitree_stair_climbing":
            raise ValueError("Stair unified protocol task mismatch")
        cell_variant_id = payload.get("cell_variant_id")
        if not isinstance(cell_variant_id, str) or not cell_variant_id.strip():
            raise ValueError("Stair unified protocol cell identity is missing")
        if (
            expected_cell_variant_id is not None
            and cell_variant_id != expected_cell_variant_id
        ):
            raise ValueError("Stair unified protocol cell identity mismatch")
        if payload.get("protocol_state") not in {"draft", "active", "archived"}:
            raise ValueError("Stair unified protocol state is invalid")

        component_refs = _mapping(payload.get("component_refs"), "component_refs")
        if set(component_refs) != {
            "training_profile",
            "evaluation_suite",
            "acceptance_protocol",
            "vnext_reward_authority",
        }:
            raise ValueError("Stair unified component refs mismatch")
        training_profile = StairUnifiedTrainingProfile.load(
            _resolve_component_ref(
                resolved,
                component_refs["training_profile"],
                "training profile",
            )
        )
        evaluation_suite = StairUnifiedEvaluationSuite.load(
            _resolve_component_ref(
                resolved,
                component_refs["evaluation_suite"],
                "evaluation suite",
            ),
            training_profile=training_profile,
        )
        acceptance_protocol = AcceptanceProtocol.from_mapping(
            load_strict_json_object(
                _resolve_component_ref(
                    resolved,
                    component_refs["acceptance_protocol"],
                    "acceptance protocol",
                ),
                context="Stair unified acceptance protocol",
            )
        )
        reward_authority = StairVNextRewardAuthority.load(
            _resolve_component_ref(
                resolved,
                component_refs["vnext_reward_authority"],
                "VNext reward authority",
            )
        )
        capability_ids = _string_tuple(
            _mapping(payload["candidate_authority"], "candidate_authority").get(
                "capability_ids"
            ),
            "candidate capability_ids",
        )
        candidate_capability_profile = CandidateCapabilityProfile(
            profile_id="unitree_stair_complete_reward_unified_v1",
            capability_ids=tuple(sorted(capability_ids)),
        )
        instance = cls(
            _payload=_freeze(payload),
            training_profile=training_profile,
            evaluation_suite=evaluation_suite,
            acceptance_protocol=acceptance_protocol,
            reward_authority=reward_authority,
            candidate_capability_profile=candidate_capability_profile,
        )
        instance._validate()
        return instance

    def _validate(self) -> None:
        expected_components = {
            "training_profile_hash": self.training_profile.profile_hash,
            "evaluation_suite_hash": self.evaluation_suite.suite_hash,
            "acceptance_protocol_hash": self.acceptance_protocol.protocol_hash,
            "vnext_reward_authority_hash": self.reward_authority.authority_hash,
        }
        if (
            dict(_mapping(self._payload["component_hashes"], "component_hashes"))
            != expected_components
        ):
            raise ValueError("Stair unified protocol component hashes mismatch")
        suite_guard_names = tuple(
            str(name)
            for name in self.evaluation_suite.score_contract["named_guards"]
        )
        acceptance_guard_names = tuple(
            guard.name for guard in self.acceptance_protocol.named_guards
        )
        if suite_guard_names != acceptance_guard_names:
            raise ValueError(
                "Stair unified evaluation/acceptance named guards diverged"
            )
        training = _mapping(self._payload["training"], "training")
        if (
            training.get("budget_steps") != 196_608_000
            or training.get("num_envs") != 4096
            or training.get("max_iterations") != 2000
            or training.get("steps_per_env") != 24
            or training.get("checkpoint_selector") != "final_iteration"
            or training.get("policy_initialization") != "random"
        ):
            raise ValueError("Stair unified production training contract drifted")
        if int(training["num_envs"]) * int(training["max_iterations"]) * int(
            training["steps_per_env"]
        ) != int(training["budget_steps"]):
            raise ValueError("Stair unified training budget arithmetic mismatch")
        command = _mapping(training.get("command"), "training command")
        if set(command) != {
            "forward_mps_range",
            "lateral_command_mps",
            "yaw_command_radps",
            "standing_env_fraction",
            "heading_command_enabled",
        }:
            raise ValueError("Stair unified training command fields mismatch")
        forward_range = command["forward_mps_range"]
        if not isinstance(forward_range, list | tuple) or len(forward_range) != 2:
            raise ValueError("Stair unified forward command range is invalid")
        forward_low, forward_high = (
            _finite_number(value, "forward command") for value in forward_range
        )
        if forward_low < 0.0 or forward_low >= forward_high:
            raise ValueError("Stair unified forward command range is invalid")
        standing_fraction = _finite_number(
            command["standing_env_fraction"],
            "standing environment fraction",
        )
        if not 0.0 <= standing_fraction < 1.0:
            raise ValueError("Stair unified standing fraction is invalid")
        if command["heading_command_enabled"] is not False:
            raise ValueError("Stair unified heading command must remain disabled")
        runtime_controls = _mapping(
            self._payload["runtime_controls"], "runtime_controls"
        )
        if set(runtime_controls) != {
            "train_observation_corruption_enabled",
            "eval_observation_corruption_enabled",
            "spawn",
            "reset_velocity_ranges",
            "evaluation_command",
            "external_force_events_enabled",
            "push_events_enabled",
        }:
            raise ValueError("Stair unified runtime control fields mismatch")
        if (
            runtime_controls["train_observation_corruption_enabled"] is not True
            or runtime_controls["eval_observation_corruption_enabled"] is not False
            or runtime_controls["external_force_events_enabled"] is not False
            or runtime_controls["push_events_enabled"] is not False
        ):
            raise ValueError("Stair unified runtime controls drifted")
        evaluation_command = _mapping(
            runtime_controls["evaluation_command"],
            "evaluation command",
        )
        if set(evaluation_command) != {
            "forward_mps",
            "lateral_mps",
            "yaw_radps",
        }:
            raise ValueError("Stair unified evaluation command fields mismatch")
        if not (
            math.isclose(
                forward_high,
                _finite_number(
                    evaluation_command["forward_mps"],
                    "evaluation forward command",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                _finite_number(
                    command["lateral_command_mps"],
                    "training lateral command",
                ),
                _finite_number(
                    evaluation_command["lateral_mps"],
                    "evaluation lateral command",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                _finite_number(
                    command["yaw_command_radps"],
                    "training yaw command",
                ),
                _finite_number(
                    evaluation_command["yaw_radps"],
                    "evaluation yaw command",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("Stair unified train/eval commands diverged")
        outcomes = _mapping(self._payload["episode_outcomes"], "episode_outcomes")
        if set(outcomes.get("primary_reasons", ())) != {
            "success",
            "forbidden_contact",
            "fall",
            "off_course",
            "base_orientation_instability",
            "slip",
            "insufficient_clearance",
            "timeout",
        }:
            raise ValueError("Stair unified outcome taxonomy drifted")
        if set(_mapping(outcomes.get("signals"), "outcome signals")) != {
            "forbidden_contact",
            "fall",
            "off_course",
            "base_orientation_instability",
            "slip",
            "insufficient_clearance",
            "timeout",
        }:
            raise ValueError("Stair unified outcome signals drifted")
        if self._payload.get("task_protocol_hash") != self.task_protocol_hash:
            raise ValueError("Stair unified task protocol hash mismatch")
        if self._payload.get("protocol_hash") != self.protocol_hash:
            raise ValueError("Stair unified protocol hash mismatch")

    @property
    def task_id(self) -> str:
        return str(self._payload["task_id"])

    @property
    def cell_variant_id(self) -> str:
        return str(self._payload["cell_variant_id"])

    @property
    def protocol_state(self) -> str:
        return str(self._payload["protocol_state"])

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self._payload["training"], "training")

    @property
    def training_execution(self) -> Mapping[str, Any]:
        return _mapping(self._payload["training_execution"], "training_execution")

    @property
    def episode_outcomes(self) -> Mapping[str, Any]:
        return _mapping(self._payload["episode_outcomes"], "episode_outcomes")

    @property
    def runtime_controls(self) -> Mapping[str, Any]:
        return _mapping(self._payload["runtime_controls"], "runtime_controls")

    @property
    def reference_program(self) -> Mapping[str, Any]:
        return _mapping(self._payload["reference_program"], "reference_program")

    @property
    def reference_readiness(self) -> Mapping[str, Any]:
        return _mapping(self._payload["reference_readiness"], "reference_readiness")

    @property
    def required_pass_gates(self) -> tuple[str, ...]:
        return _string_tuple(
            self._payload["required_pass_gates"], "required_pass_gates"
        )

    def task_projection(self) -> dict[str, Any]:
        acceptance = self.acceptance_protocol.to_public_dict()
        acceptance.pop("protocol_id")
        acceptance.pop("acceptance_protocol_hash")
        return {
            "schema_version": STAIR_UNIFIED_TASK_PROJECTION_SCHEMA_VERSION,
            "task_id": self.task_id,
            "training": _thaw(self._payload["training"]),
            "training_execution": _thaw(self._payload["training_execution"]),
            "runtime_controls": _thaw(self._payload["runtime_controls"]),
            "training_distribution": self.training_profile.task_projection(),
            "evaluation": self.evaluation_suite.task_projection(),
            "episode_outcomes": _thaw(self._payload["episode_outcomes"]),
            "acceptance": acceptance,
            "reference_readiness": _thaw(self._payload["reference_readiness"]),
            "candidate_authority": _thaw(self._payload["candidate_authority"]),
            "reward_authority": self.reward_authority.task_projection(),
            "required_pass_gates": list(self.required_pass_gates),
        }

    def runtime_task_contract(self) -> dict[str, Any]:
        """Return only morphology-neutral quantities consumed inside Isaac."""

        from true_rl_agent.tasks.stair_climbing.unified_training import (
            goal_progress_m_by_terrain_type,
        )

        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.runtime_task_contract.v1"
            ),
            "terrain": {
                "goal": {
                    "progress_start_m_from_origin": 0.0,
                    "progress_goal_m_by_terrain_type": list(
                        goal_progress_m_by_terrain_type(self.training_profile)
                    ),
                }
            },
            "episode_outcomes": _thaw(self._payload["episode_outcomes"]),
        }

    @property
    def task_protocol_hash(self) -> str:
        return canonical_json_hash(self.task_projection())

    @property
    def protocol_hash(self) -> str:
        payload = {
            key: _thaw(value)
            for key, value in self._payload.items()
            if key != "protocol_hash"
        }
        return canonical_json_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._payload)


def _resolve_component_ref(
    protocol_path: Path,
    value: Any,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stair unified {field_name} ref is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Stair unified {field_name} ref must be protocol-relative")
    return protocol_path.parent.joinpath(*relative.parts)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stair unified {field_name} must be an object")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Stair unified {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Stair unified {field_name} must be finite")
    return result


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Stair unified {field_name} must contain strings")
    if len(value) != len(set(value)):
        raise ValueError(f"Stair unified {field_name} contains duplicates")
    return tuple(value)


def _freeze(value: Any) -> Any:
    copied = json.loads(json.dumps(value, allow_nan=False))
    return _freeze_copied(copied)


def _freeze_copied(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_copied(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_copied(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID",
    "STAIR_UNIFIED_ACCEPTANCE_PROTOCOL_RELATIVE_PATH",
    "STAIR_UNIFIED_PROTOCOL_RELATIVE_PATH",
    "STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION",
    "STAIR_UNIFIED_TASK_PROJECTION_SCHEMA_VERSION",
    "StairUnifiedProtocol",
]

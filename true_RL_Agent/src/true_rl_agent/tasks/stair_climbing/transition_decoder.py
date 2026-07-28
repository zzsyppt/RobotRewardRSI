"""Decode production recorder batches into strict Stair episode transitions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEnvTransition
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
    expected_stair_surface_height_m,
)

_REQUIRED_FIELDS = {
    "schema_version",
    "root_pos_w",
    "root_quat_w",
    "foot_pos_w",
    "foot_lin_vel_w",
    "foot_contact_forces_w",
    "base_contact_forces_w",
    "episode_length",
    "terrain_env_origins_w",
    "terrain_types",
    "termination_term_order",
    "termination_masks",
    "done_mask",
}


class StairTransitionDecodeError(RuntimeError):
    """Raised when raw recorder evidence is incomplete or inconsistent."""


def decode_stair_transition_batch(
    raw: Mapping[str, Any],
    *,
    wrapper_done_mask: Any,
    protocol: StairProtocolV1,
    expected_bin_assignment: Sequence[str],
) -> tuple[StairEnvTransition, ...]:
    """Validate one pre-reset batch and derive task-level transition signals."""

    if not isinstance(raw, Mapping) or set(raw) != _REQUIRED_FIELDS:
        raise StairTransitionDecodeError("Stair transition recorder fields mismatch")
    if raw.get("schema_version") != STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
        raise StairTransitionDecodeError("Stair transition recorder schema mismatch")
    if not isinstance(protocol, StairProtocolV1):
        raise TypeError("Stair transition decoder requires StairProtocolV1")

    bin_assignment = tuple(expected_bin_assignment)
    expected_num_envs = len(bin_assignment)
    if expected_num_envs <= 0:
        raise StairTransitionDecodeError("Stair bin assignment must be non-empty")
    frozen_bins = {
        str(item["bin_id"]): (index, float(item["height_m"]))
        for index, item in enumerate(protocol.eval_bins)
    }
    if set(bin_assignment) != set(frozen_bins):
        raise StairTransitionDecodeError("Stair bin assignment differs from protocol")

    root_pos = _numeric_tensor(raw["root_pos_w"], (expected_num_envs, 3), "root_pos_w")
    root_quat = _numeric_tensor(raw["root_quat_w"], (expected_num_envs, 4), "root_quat_w")
    foot_pos = _numeric_tensor(raw["foot_pos_w"], (expected_num_envs, 2, 3), "foot_pos_w")
    foot_vel = _numeric_tensor(
        raw["foot_lin_vel_w"],
        (expected_num_envs, 2, 3),
        "foot_lin_vel_w",
    )
    foot_forces = _numeric_tensor(
        raw["foot_contact_forces_w"],
        (expected_num_envs, 2, 3),
        "foot_contact_forces_w",
    )
    _numeric_tensor(
        raw["base_contact_forces_w"],
        (expected_num_envs, 1, 3),
        "base_contact_forces_w",
    )
    episode_length = _integer_tensor(
        raw["episode_length"],
        (expected_num_envs,),
        "episode_length",
        positive=True,
    )
    origins = _numeric_tensor(
        raw["terrain_env_origins_w"],
        (expected_num_envs, 3),
        "terrain_env_origins_w",
    )
    terrain_types = _integer_tensor(
        raw["terrain_types"],
        (expected_num_envs,),
        "terrain_types",
        positive=False,
    )
    done_mask = _boolean_tensor(raw["done_mask"], (expected_num_envs,), "done_mask")
    wrapper_done = _boolean_tensor(
        wrapper_done_mask,
        (expected_num_envs,),
        "wrapper_done_mask",
    )
    if not _tensor_equal(done_mask, wrapper_done):
        raise StairTransitionDecodeError(
            "Stair wrapper done mask differs from recorder done mask"
        )

    term_order = raw.get("termination_term_order")
    term_masks = raw.get("termination_masks")
    if (
        not isinstance(term_order, tuple)
        or not term_order
        or len(term_order) != len(set(term_order))
        or any(not isinstance(name, str) or not name for name in term_order)
        or not isinstance(term_masks, Mapping)
        or set(term_masks) != set(term_order)
    ):
        raise StairTransitionDecodeError("Stair termination mask identity is invalid")
    protocol_payload = protocol.to_dict()
    manager_to_reason = dict(
        protocol_payload["phase0_probe"]["terminal_reason_by_manager_term"]
    )
    if set(term_order) != set(manager_to_reason):
        raise StairTransitionDecodeError(
            "Stair termination manager terms differ from frozen task mapping"
        )
    decoded_term_masks = {
        name: _boolean_tensor(
            term_masks[name],
            (expected_num_envs,),
            f"termination_masks.{name}",
        )
        for name in term_order
    }
    term_union = decoded_term_masks[term_order[0]]
    for name in term_order[1:]:
        term_union = term_union | decoded_term_masks[name]
    if not _tensor_equal(term_union, done_mask):
        raise StairTransitionDecodeError(
            "Stair done mask differs from OR(termination masks)"
        )

    expected_type_indexes = [frozen_bins[bin_id][0] for bin_id in bin_assignment]
    if _to_list(terrain_types) != expected_type_indexes:
        raise StairTransitionDecodeError("Stair terrain type/bin assignment drifted")
    _assert_origins(
        origins,
        bin_assignment=bin_assignment,
        frozen_bins=frozen_bins,
        protocol_payload=protocol_payload,
    )

    root_pos_values = _to_list(root_pos)
    root_quat_values = _to_list(root_quat)
    foot_pos_values = _to_list(foot_pos)
    foot_vel_values = _to_list(foot_vel)
    foot_force_values = _to_list(foot_forces)
    episode_steps = _to_list(episode_length)
    origin_values = _to_list(origins)
    done_values = _to_list(done_mask)
    term_values = {name: _to_list(mask) for name, mask in decoded_term_masks.items()}
    contact_threshold = float(
        protocol_payload["episode_outcomes"]["signals"]["slip"][
            "contact_force_threshold"
        ]
    )
    base_contact_terms = tuple(
        name for name, reason in manager_to_reason.items() if reason == "base_contact"
    )
    if len(base_contact_terms) != 1:
        raise StairTransitionDecodeError(
            "Stair protocol must map exactly one base-contact manager term"
        )

    transitions: list[StairEnvTransition] = []
    for env_index, bin_id in enumerate(bin_assignment):
        origin = origin_values[env_index]
        root = root_pos_values[env_index]
        progress = float(root[0]) - float(origin[0])
        step_height = frozen_bins[bin_id][1]
        root_surface = expected_stair_surface_height_m(
            progress_m=progress,
            step_height_m=step_height,
            protocol=protocol_payload,
        )
        root_clearance = float(root[2]) - float(origin[2]) - root_surface
        foot_contacts: list[bool] = []
        foot_speeds: list[float] = []
        foot_clearances: list[float] = []
        for foot_index in range(2):
            force = foot_force_values[env_index][foot_index]
            velocity = foot_vel_values[env_index][foot_index]
            position = foot_pos_values[env_index][foot_index]
            force_norm = math.sqrt(sum(float(value) ** 2 for value in force))
            foot_contacts.append(force_norm > contact_threshold)
            foot_speeds.append(
                math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
            )
            foot_progress = float(position[0]) - float(origin[0])
            surface = expected_stair_surface_height_m(
                progress_m=foot_progress,
                step_height_m=step_height,
                protocol=protocol_payload,
            )
            foot_clearances.append(float(position[2]) - float(origin[2]) - surface)

        active_terms = tuple(
            name for name in term_order if bool(term_values[name][env_index])
        )
        transitions.append(
            StairEnvTransition(
                env_index=env_index,
                bin_id=bin_id,
                episode_step=int(episode_steps[env_index]),
                progress_m=progress,
                root_clearance_m=root_clearance,
                abs_pitch_rad=abs(_pitch_from_wxyz(root_quat_values[env_index])),
                base_contact=bool(
                    term_values[base_contact_terms[0]][env_index]
                ),
                foot_contacts=(foot_contacts[0], foot_contacts[1]),
                foot_planar_speeds_mps=(foot_speeds[0], foot_speeds[1]),
                swing_foot_clearances_m=(
                    foot_clearances[0],
                    foot_clearances[1],
                ),
                done=bool(done_values[env_index]),
                termination_terms=active_terms,
            )
        )
    return tuple(transitions)


def _assert_origins(
    origins: Any,
    *,
    bin_assignment: tuple[str, ...],
    frozen_bins: Mapping[str, tuple[int, float]],
    protocol_payload: Mapping[str, Any],
) -> None:
    expected_origins = protocol_payload["terrain"]["evaluation_grid_origins_m"]
    observed = _to_list(origins)
    for env_index, bin_id in enumerate(bin_assignment):
        expected = expected_origins[frozen_bins[bin_id][0]]
        if any(
            not math.isclose(
                float(observed[env_index][axis]),
                float(expected[axis]),
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
            for axis in range(3)
        ):
            raise StairTransitionDecodeError("Stair terrain origin/bin identity drifted")


def _pitch_from_wxyz(quaternion: Sequence[float]) -> float:
    if len(quaternion) != 4:
        raise StairTransitionDecodeError("Stair root quaternion shape drifted")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-8 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=5.0e-3):
        raise StairTransitionDecodeError("Stair root quaternion is not normalized")
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return math.asin(pitch_arg)


def _numeric_tensor(value: Any, shape: tuple[int, ...], field_name: str) -> Any:
    tensor = _tensor(value, shape, field_name)
    isfinite = getattr(tensor, "isfinite", None)
    if callable(isfinite):
        finite = isfinite()
    else:
        try:
            import torch

            finite = torch.isfinite(tensor)
        except (ImportError, TypeError) as exc:
            raise StairTransitionDecodeError(
                f"Stair {field_name} cannot be checked for finite values"
            ) from exc
    if not _all_true(finite):
        raise StairTransitionDecodeError(f"Stair {field_name} contains NaN/Inf")
    return tensor


def _integer_tensor(
    value: Any,
    shape: tuple[int, ...],
    field_name: str,
    *,
    positive: bool,
) -> Any:
    tensor = _tensor(value, shape, field_name)
    dtype = str(getattr(tensor, "dtype", ""))
    if not any(token in dtype for token in ("int", "long")) or "bool" in dtype:
        raise StairTransitionDecodeError(f"Stair {field_name} must be integer")
    values = _to_list(tensor)
    minimum = 1 if positive else 0
    if any(isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in values):
        raise StairTransitionDecodeError(f"Stair {field_name} values are invalid")
    return tensor


def _boolean_tensor(value: Any, shape: tuple[int, ...], field_name: str) -> Any:
    tensor = _tensor(value, shape, field_name)
    if str(getattr(tensor, "dtype", "")) not in {"bool", "torch.bool", "bool_"}:
        raise StairTransitionDecodeError(f"Stair {field_name} must be boolean")
    return tensor


def _tensor(value: Any, shape: tuple[int, ...], field_name: str) -> Any:
    try:
        actual_shape = tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StairTransitionDecodeError(f"Stair {field_name} is not tensor-like") from exc
    if actual_shape != shape:
        raise StairTransitionDecodeError(
            f"Stair {field_name} shape mismatch: expected={shape} actual={actual_shape}"
        )
    return value


def _to_list(value: Any) -> Any:
    current = value
    for operation in ("detach", "cpu", "tolist"):
        method = getattr(current, operation, None)
        if not callable(method):
            raise StairTransitionDecodeError(
                "Stair recorder tensor cannot be converted to immutable evidence"
            )
        current = method()
    return current


def _tensor_equal(left: Any, right: Any) -> bool:
    return _all_true(left == right)


def _all_true(value: Any) -> bool:
    all_method = getattr(value, "all", None)
    if not callable(all_method):
        raise StairTransitionDecodeError("Stair tensor does not support all()")
    result = all_method()
    item = getattr(result, "item", None)
    return bool(item() if callable(item) else result)


__all__ = [
    "StairTransitionDecodeError",
    "decode_stair_transition_batch",
]

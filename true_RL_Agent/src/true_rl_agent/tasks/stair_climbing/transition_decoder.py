"""Decode production recorder batches into strict Stair episode transitions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEnvTransition
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationCell,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
    expected_stair_surface_height_m,
    stair_surface_level,
    supporting_stair_surface_height_m,
)

_REQUIRED_FIELDS = {
    "schema_version",
    "root_pos_w",
    "root_quat_w",
    "joint_pos",
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
_TENSOR_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "joint_pos",
    "foot_pos_w",
    "foot_lin_vel_w",
    "foot_contact_forces_w",
    "base_contact_forces_w",
    "episode_length",
    "terrain_env_origins_w",
    "terrain_types",
    "done_mask",
)
_STATIC_TENSOR_FIELDS = {"terrain_env_origins_w", "terrain_types"}
STAIR_TRANSITION_HISTORY_CHUNK_STEPS = 64


class StairTransitionDecodeError(RuntimeError):
    """Raised when raw recorder evidence is incomplete or inconsistent."""


def materialize_stair_transition_history(
    raw_history: Sequence[Mapping[str, Any]],
    *,
    wrapper_done_masks: Sequence[Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Any, ...]]:
    """Batch-copy a rollout history to host memory before scalar decoding."""

    raw_steps = tuple(raw_history)
    wrapper_steps = tuple(wrapper_done_masks)
    if not raw_steps or len(raw_steps) != len(wrapper_steps):
        raise StairTransitionDecodeError(
            "Stair transition history must contain matching non-empty steps"
        )

    term_order: tuple[str, ...] | None = None
    field_values = {name: [] for name in _TENSOR_FIELDS}
    term_values: dict[str, list[Any]] = {}
    for raw in raw_steps:
        if not isinstance(raw, Mapping) or set(raw) != _REQUIRED_FIELDS:
            raise StairTransitionDecodeError(
                "Stair transition recorder fields mismatch"
            )
        if raw.get("schema_version") != STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
            raise StairTransitionDecodeError(
                "Stair transition recorder schema mismatch"
            )
        current_order = raw.get("termination_term_order")
        current_masks = raw.get("termination_masks")
        if (
            not isinstance(current_order, tuple)
            or not current_order
            or len(current_order) != len(set(current_order))
            or any(not isinstance(name, str) or not name for name in current_order)
            or not isinstance(current_masks, Mapping)
            or set(current_masks) != set(current_order)
        ):
            raise StairTransitionDecodeError(
                "Stair termination mask identity is invalid"
            )
        if term_order is None:
            term_order = current_order
            term_values = {name: [] for name in term_order}
        elif current_order != term_order:
            raise StairTransitionDecodeError(
                "Stair termination term order drifted within rollout"
            )
        for name in _TENSOR_FIELDS:
            field_values[name].append(raw[name])
        for name in term_order:
            term_values[name].append(current_masks[name])

    assert term_order is not None
    tensor_values = [
        value
        for values in field_values.values()
        for value in values
    ]
    tensor_values.extend(
        value
        for values in term_values.values()
        for value in values
    )
    tensor_values.extend(wrapper_steps)
    torch = _torch_for_history(tensor_values)
    if torch is None:
        copied = tuple(
            {
                **dict(raw),
                "termination_masks": dict(raw["termination_masks"]),
            }
            for raw in raw_steps
        )
        return copied, wrapper_steps

    host_fields = {
        name: _materialize_history_field(
            torch,
            values,
            field_name=name,
            allow_shared_static=name in _STATIC_TENSOR_FIELDS,
        )
        for name, values in field_values.items()
    }
    stacked_terms = {
        name: _stack_to_host(
            torch,
            values,
            field_name=f"termination_masks.{name}",
        )
        for name, values in term_values.items()
    }
    stacked_wrapper = _stack_to_host(
        torch,
        wrapper_steps,
        field_name="wrapper_done_mask",
    )
    host_steps: list[Mapping[str, Any]] = []
    for index in range(len(raw_steps)):
        host_steps.append(
            {
                "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
                **{
                    name: value if shared else value[index]
                    for name, (shared, value) in host_fields.items()
                },
                "termination_term_order": term_order,
                "termination_masks": {
                    name: values[index]
                    for name, values in stacked_terms.items()
                },
            }
        )
    return tuple(host_steps), tuple(
        stacked_wrapper[index] for index in range(len(raw_steps))
    )


def decode_stair_transition_batch(
    raw: Mapping[str, Any],
    *,
    wrapper_done_mask: Any,
    protocol: StairProtocolV1,
    semantics: StairEvaluationSemantics,
    expected_bin_assignment: Sequence[str],
    expected_cell_assignment: Sequence[StairEvaluationCell] | None = None,
) -> tuple[StairEnvTransition, ...]:
    """Validate one pre-reset batch and derive task-level transition signals."""

    if not isinstance(raw, Mapping) or set(raw) != _REQUIRED_FIELDS:
        raise StairTransitionDecodeError("Stair transition recorder fields mismatch")
    if raw.get("schema_version") != STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
        raise StairTransitionDecodeError("Stair transition recorder schema mismatch")
    if not isinstance(protocol, StairProtocolV1):
        raise TypeError("Stair transition decoder requires StairProtocolV1")
    if not isinstance(semantics, StairEvaluationSemantics):
        raise TypeError("Stair transition decoder requires evaluation semantics")

    bin_assignment = tuple(expected_bin_assignment)
    cell_assignment = (
        tuple(expected_cell_assignment)
        if expected_cell_assignment is not None
        else None
    )
    expected_num_envs = len(bin_assignment)
    if expected_num_envs <= 0:
        raise StairTransitionDecodeError("Stair bin assignment must be non-empty")
    frozen_bins = {
        str(item["bin_id"]): (index, float(item["height_m"]))
        for index, item in enumerate(protocol.eval_bins)
    }
    if set(bin_assignment) != set(frozen_bins):
        raise StairTransitionDecodeError("Stair bin assignment differs from protocol")
    if cell_assignment is not None:
        if len(cell_assignment) != expected_num_envs or any(
            not isinstance(cell, StairEvaluationCell)
            for cell in cell_assignment
        ):
            raise StairTransitionDecodeError(
                "Stair evaluation cell assignment is invalid"
            )
        if tuple(cell.bin_id for cell in cell_assignment) != bin_assignment:
            raise StairTransitionDecodeError(
                "Stair evaluation cell/bin assignment drifted"
            )
    root_pos = _numeric_tensor(raw["root_pos_w"], (expected_num_envs, 3), "root_pos_w")
    root_quat = _numeric_tensor(raw["root_quat_w"], (expected_num_envs, 4), "root_quat_w")
    joint_count = len(
        protocol.to_dict()["policy_compatibility"]["robot_schema"]["joint_names"]
    )
    _numeric_tensor(
        raw["joint_pos"],
        (expected_num_envs, joint_count),
        "joint_pos",
    )
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

    expected_type_indexes = (
        [cell.terrain_type_index for cell in cell_assignment]
        if cell_assignment is not None
        else [frozen_bins[bin_id][0] for bin_id in bin_assignment]
    )
    if _to_list(terrain_types) != expected_type_indexes:
        raise StairTransitionDecodeError("Stair terrain type/bin assignment drifted")
    _assert_origins(
        origins,
        bin_assignment=bin_assignment,
        frozen_bins=frozen_bins,
        protocol_payload=protocol_payload,
        cell_assignment=cell_assignment,
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
    semantics_payload = semantics.to_dict()
    contact_threshold = float(
        semantics_payload["episode_outcomes"]["signals"]["slip"][
            "contact_force_threshold"
        ]
    )
    maximum_projected_surface_penetration_m = float(
        semantics_payload["episode_outcomes"]["signals"]["fall"][
            "maximum_projected_surface_penetration_m"
        ]
    )
    torso_contact_manager_term = str(
        semantics_payload["episode_outcomes"]["signals"]["torso_contact"][
            "manager_term"
        ]
    )
    base_contact_terms = tuple(
        name for name, reason in manager_to_reason.items() if reason == "base_contact"
    )
    if (
        len(base_contact_terms) != 1
        or base_contact_terms[0] != torso_contact_manager_term
    ):
        raise StairTransitionDecodeError(
            "Stair protocol must map exactly one base-contact manager term"
        )

    transitions: list[StairEnvTransition] = []
    for env_index, bin_id in enumerate(bin_assignment):
        cell = cell_assignment[env_index] if cell_assignment is not None else None
        origin = origin_values[env_index]
        root = root_pos_values[env_index]
        progress = float(root[0]) - float(origin[0])
        lateral = float(root[1]) - float(origin[1])
        root_height = float(root[2]) - float(origin[2])
        step_height = cell.height_m if cell is not None else frozen_bins[bin_id][1]
        tread_depth = cell.tread_depth_m if cell is not None else None
        foot_contacts: list[bool] = []
        foot_progresses: list[float] = []
        foot_lateral: list[float] = []
        foot_surface_levels: list[int] = []
        foot_speeds: list[float] = []
        foot_clearances: list[float] = []
        foot_upward_forces: list[float] = []
        for foot_index in range(2):
            force = foot_force_values[env_index][foot_index]
            velocity = foot_vel_values[env_index][foot_index]
            position = foot_pos_values[env_index][foot_index]
            force_norm = math.sqrt(sum(float(value) ** 2 for value in force))
            foot_contacts.append(force_norm > contact_threshold)
            foot_upward_forces.append(float(force[2]))
            foot_speeds.append(
                math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
            )
            foot_progress = float(position[0]) - float(origin[0])
            foot_progresses.append(foot_progress)
            foot_lateral.append(float(position[1]) - float(origin[1]))
            foot_surface_levels.append(
                stair_surface_level(
                    progress_m=foot_progress,
                    protocol=protocol_payload,
                    tread_depth_m=tread_depth,
                )
            )
            surface = expected_stair_surface_height_m(
                progress_m=foot_progress,
                step_height_m=step_height,
                protocol=protocol_payload,
                tread_depth_m=tread_depth,
            )
            foot_clearances.append(float(position[2]) - float(origin[2]) - surface)

        support_surface = supporting_stair_surface_height_m(
            foot_progress_m=(foot_progresses[0], foot_progresses[1]),
            foot_contacts=(foot_contacts[0], foot_contacts[1]),
            foot_surface_clearance_m=(foot_clearances[0], foot_clearances[1]),
            step_height_m=step_height,
            maximum_projected_surface_penetration_m=(
                maximum_projected_surface_penetration_m
            ),
            protocol=protocol_payload,
            tread_depth_m=tread_depth,
        )
        root_clearance = root_height - support_surface
        roll, pitch = _roll_pitch_from_wxyz(root_quat_values[env_index])

        active_terms = tuple(
            name for name in term_order if bool(term_values[name][env_index])
        )
        transitions.append(
            StairEnvTransition(
                env_index=env_index,
                bin_id=bin_id,
                episode_step=int(episode_steps[env_index]),
                progress_m=progress,
                lateral_m=lateral,
                root_height_m=root_height,
                root_clearance_m=root_clearance,
                abs_roll_rad=abs(roll),
                abs_pitch_rad=abs(pitch),
                torso_contact=bool(
                    term_values[base_contact_terms[0]][env_index]
                ),
                foot_contacts=(foot_contacts[0], foot_contacts[1]),
                foot_progress_m=(foot_progresses[0], foot_progresses[1]),
                foot_surface_levels=(
                    foot_surface_levels[0],
                    foot_surface_levels[1],
                ),
                foot_lateral_m=(foot_lateral[0], foot_lateral[1]),
                foot_planar_speeds_mps=(foot_speeds[0], foot_speeds[1]),
                foot_surface_clearances_m=(
                    foot_clearances[0],
                    foot_clearances[1],
                ),
                foot_upward_forces_n=(
                    foot_upward_forces[0],
                    foot_upward_forces[1],
                ),
                done=bool(done_values[env_index]),
                termination_terms=active_terms,
                evaluation_cell_id=(cell.cell_id if cell is not None else None),
            )
        )
    return tuple(transitions)


def _assert_origins(
    origins: Any,
    *,
    bin_assignment: tuple[str, ...],
    frozen_bins: Mapping[str, tuple[int, float]],
    protocol_payload: Mapping[str, Any],
    cell_assignment: tuple[StairEvaluationCell, ...] | None,
) -> None:
    expected_origins = protocol_payload["terrain"]["evaluation_grid_origins_m"]
    observed = _to_list(origins)
    for env_index, bin_id in enumerate(bin_assignment):
        expected = (
            cell_assignment[env_index].expected_origin_m
            if cell_assignment is not None
            else expected_origins[frozen_bins[bin_id][0]]
        )
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


def _roll_pitch_from_wxyz(quaternion: Sequence[float]) -> tuple[float, float]:
    if len(quaternion) != 4:
        raise StairTransitionDecodeError("Stair root quaternion shape drifted")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-8 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=5.0e-3):
        raise StairTransitionDecodeError("Stair root quaternion is not normalized")
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_arg)


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


def _torch_for_history(values: Sequence[Any]) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    tensor_flags = tuple(isinstance(value, torch.Tensor) for value in values)
    if any(tensor_flags) and not all(tensor_flags):
        raise StairTransitionDecodeError(
            "Stair transition history mixes torch and non-torch tensors"
        )
    return torch if all(tensor_flags) else None


def _stack_to_host(
    torch: Any,
    values: Sequence[Any],
    *,
    field_name: str,
) -> Any:
    try:
        host_chunks = []
        for start in range(0, len(values), STAIR_TRANSITION_HISTORY_CHUNK_STEPS):
            stop = start + STAIR_TRANSITION_HISTORY_CHUNK_STEPS
            host_chunks.append(
                torch.stack(
                    tuple(value.detach() for value in values[start:stop]),
                    dim=0,
                ).cpu()
            )
        if len(host_chunks) == 1:
            return host_chunks[0]
        return torch.cat(tuple(host_chunks), dim=0)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise StairTransitionDecodeError(
            f"Stair transition history {field_name} cannot be batch-copied"
        ) from exc


def _materialize_history_field(
    torch: Any,
    values: Sequence[Any],
    *,
    field_name: str,
    allow_shared_static: bool,
) -> tuple[bool, Any]:
    if allow_shared_static and all(value is values[0] for value in values):
        try:
            return True, values[0].detach().cpu()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise StairTransitionDecodeError(
                f"Stair transition history {field_name} cannot be copied"
            ) from exc
    return False, _stack_to_host(torch, values, field_name=field_name)


__all__ = [
    "STAIR_TRANSITION_HISTORY_CHUNK_STEPS",
    "StairTransitionDecodeError",
    "decode_stair_transition_batch",
    "materialize_stair_transition_history",
]

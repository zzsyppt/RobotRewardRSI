"""Decode production recorder batches into strict Stair episode transitions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEnvTransition
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationCell,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    OrderedStairEvaluationCell,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import (
    STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION,
    normalize_stair_transition_payload,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
    StairRuntimeProjection,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
    FlatStairSurfaceQuery,
    StairSurfaceQuery,
    stair_surface_query_for_ordered_cell,
    supporting_stair_surface_height_m,
)

_REQUIRED_FIELDS = {
    "schema_version",
    "source_schema_version",
    "base_pos_w",
    "base_quat_w",
    "joint_pos",
    "support_pos_w",
    "support_lin_vel_w",
    "support_contact_forces_w",
    "forbidden_contact_forces_w",
    "episode_length",
    "terrain_env_origins_w",
    "terrain_types",
    "termination_term_order",
    "termination_masks",
    "done_mask",
    "runtime_projection_hash",
    "reward_input_schema_hash_vnext",
    "support_count",
    "forbidden_contact_count",
}
_TENSOR_FIELDS = (
    "base_pos_w",
    "base_quat_w",
    "joint_pos",
    "support_pos_w",
    "support_lin_vel_w",
    "support_contact_forces_w",
    "forbidden_contact_forces_w",
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
    normalized_steps: list[Mapping[str, Any]] = []
    for source_raw in raw_steps:
        try:
            raw = normalize_stair_transition_payload(source_raw)
        except (TypeError, ValueError) as exc:
            raise StairTransitionDecodeError(str(exc)) from exc
        if set(raw) != _REQUIRED_FIELDS:
            raise StairTransitionDecodeError(
                "Stair normalized transition recorder fields mismatch"
            )
        if raw.get("schema_version") != STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION:
            raise StairTransitionDecodeError(
                "Stair normalized transition recorder schema mismatch"
            )
        normalized_steps.append(raw)
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
    tensor_values = [value for values in field_values.values() for value in values]
    tensor_values.extend(value for values in term_values.values() for value in values)
    tensor_values.extend(wrapper_steps)
    torch = _torch_for_history(tensor_values)
    if torch is None:
        copied = tuple(
            {
                **dict(raw),
                "termination_masks": dict(raw["termination_masks"]),
            }
            for raw in normalized_steps
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
    for index in range(len(normalized_steps)):
        host_steps.append(
            {
                "schema_version": STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION,
                "source_schema_version": normalized_steps[index][
                    "source_schema_version"
                ],
                "runtime_projection_hash": normalized_steps[index][
                    "runtime_projection_hash"
                ],
                "reward_input_schema_hash_vnext": normalized_steps[index][
                    "reward_input_schema_hash_vnext"
                ],
                "support_count": normalized_steps[index]["support_count"],
                "forbidden_contact_count": normalized_steps[index][
                    "forbidden_contact_count"
                ],
                **{
                    name: value if shared else value[index]
                    for name, (shared, value) in host_fields.items()
                },
                "termination_term_order": term_order,
                "termination_masks": {
                    name: values[index] for name, values in stacked_terms.items()
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
    protocol: StairProtocolV1 | None,
    semantics: StairEvaluationSemantics,
    expected_bin_assignment: Sequence[str],
    expected_cell_assignment: Sequence[
        StairEvaluationCell | OrderedStairEvaluationCell
    ]
    | None = None,
    runtime_projection: StairRuntimeProjection | None = None,
    surface_queries: Sequence[StairSurfaceQuery] | None = None,
) -> tuple[StairEnvTransition, ...]:
    """Validate one pre-reset batch and derive task-level transition signals."""

    try:
        raw = normalize_stair_transition_payload(raw)
    except (TypeError, ValueError) as exc:
        raise StairTransitionDecodeError(str(exc)) from exc
    if set(raw) != _REQUIRED_FIELDS:
        raise StairTransitionDecodeError(
            "Stair normalized transition recorder fields mismatch"
        )
    if raw.get("schema_version") != STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION:
        raise StairTransitionDecodeError(
            "Stair normalized transition recorder schema mismatch"
        )
    if protocol is not None and not isinstance(protocol, StairProtocolV1):
        raise TypeError("Stair transition decoder protocol is invalid")
    if not isinstance(semantics, StairEvaluationSemantics):
        raise TypeError("Stair transition decoder requires evaluation semantics")
    if runtime_projection is not None and not isinstance(
        runtime_projection,
        StairRuntimeProjection,
    ):
        raise TypeError("Stair transition runtime projection is invalid")

    bin_assignment = tuple(expected_bin_assignment)
    cell_assignment = (
        tuple(expected_cell_assignment)
        if expected_cell_assignment is not None
        else None
    )
    expected_num_envs = len(bin_assignment)
    if expected_num_envs <= 0:
        raise StairTransitionDecodeError("Stair bin assignment must be non-empty")
    frozen_bins = (
        {
            str(item["bin_id"]): (index, float(item["height_m"]))
            for index, item in enumerate(protocol.eval_bins)
        }
        if protocol is not None
        else {}
    )
    ordered_assignment = bool(cell_assignment) and all(
        isinstance(cell, OrderedStairEvaluationCell) for cell in cell_assignment
    )
    legacy_cell_assignment = bool(cell_assignment) and all(
        isinstance(cell, StairEvaluationCell) for cell in cell_assignment
    )
    if cell_assignment is not None and not (
        ordered_assignment or legacy_cell_assignment
    ):
        raise StairTransitionDecodeError(
            "Stair evaluation cell assignment mixes incompatible course semantics"
        )
    if ordered_assignment:
        assert cell_assignment is not None
        if tuple(cell.cell_id for cell in cell_assignment) != bin_assignment:
            raise StairTransitionDecodeError(
                "Stair ordered cell assignment differs from explicit identities"
            )
        if protocol is not None:
            raise StairTransitionDecodeError(
                "ordered VNext Stair decoding must not consult a legacy protocol"
            )
    elif set(bin_assignment) != set(frozen_bins):
        raise StairTransitionDecodeError("Stair bin assignment differs from protocol")
    if cell_assignment is not None:
        if len(cell_assignment) != expected_num_envs or any(
            not isinstance(cell, StairEvaluationCell | OrderedStairEvaluationCell)
            for cell in cell_assignment
        ):
            raise StairTransitionDecodeError(
                "Stair evaluation cell assignment is invalid"
            )
        if tuple(cell.bin_id for cell in cell_assignment) != bin_assignment:
            raise StairTransitionDecodeError(
                "Stair evaluation cell/bin assignment drifted"
            )
    support_count = raw.get("support_count")
    if (
        isinstance(support_count, bool)
        or not isinstance(support_count, int)
        or support_count <= 0
    ):
        raise StairTransitionDecodeError("Stair support count is invalid")
    source_schema = raw.get("source_schema_version")
    if source_schema == STAIR_TRANSITION_VNEXT_SCHEMA_VERSION:
        if runtime_projection is None:
            raise StairTransitionDecodeError(
                "Stair VNext transition requires its runtime projection"
            )
        if (
            raw.get("runtime_projection_hash") != runtime_projection.projection_hash
            or raw.get("reward_input_schema_hash_vnext")
            != runtime_projection.reward_input_schema_hash_vnext
        ):
            raise StairTransitionDecodeError(
                "Stair VNext transition projection identity mismatch"
            )
    if runtime_projection is not None and (
        support_count != runtime_projection.support_count
    ):
        raise StairTransitionDecodeError(
            "Stair transition support count differs from runtime projection"
        )
    forbidden_count = raw.get("forbidden_contact_count")
    if (
        isinstance(forbidden_count, bool)
        or not isinstance(forbidden_count, int)
        or forbidden_count <= 0
    ):
        raise StairTransitionDecodeError("Stair forbidden-contact count is invalid")
    if runtime_projection is not None and (
        forbidden_count != runtime_projection.forbidden_contact_count
    ):
        raise StairTransitionDecodeError(
            "Stair transition forbidden-contact count differs from runtime projection"
        )
    if surface_queries is None:
        resolved_surface_queries: tuple[StairSurfaceQuery, ...] | None = None
    else:
        resolved_surface_queries = tuple(surface_queries)
        if len(resolved_surface_queries) != expected_num_envs or any(
            not isinstance(item, StairSurfaceQuery) for item in resolved_surface_queries
        ):
            raise StairTransitionDecodeError(
                "Stair surface query assignment is invalid"
            )
    base_pos = _numeric_tensor(
        raw["base_pos_w"],
        (expected_num_envs, 3),
        "base_pos_w",
    )
    base_quat = _numeric_tensor(
        raw["base_quat_w"],
        (expected_num_envs, 4),
        "base_quat_w",
    )
    joint_pos, joint_count = _numeric_matrix(
        raw["joint_pos"], expected_num_envs, "joint_pos"
    )
    if runtime_projection is not None and joint_count != runtime_projection.joint_count:
        raise StairTransitionDecodeError(
            "Stair joint_pos width differs from runtime projection"
        )
    support_pos = _numeric_tensor(
        raw["support_pos_w"],
        (expected_num_envs, support_count, 3),
        "support_pos_w",
    )
    support_vel = _numeric_tensor(
        raw["support_lin_vel_w"],
        (expected_num_envs, support_count, 3),
        "support_lin_vel_w",
    )
    support_forces = _numeric_tensor(
        raw["support_contact_forces_w"],
        (expected_num_envs, support_count, 3),
        "support_contact_forces_w",
    )
    _numeric_tensor(
        raw["forbidden_contact_forces_w"],
        (expected_num_envs, forbidden_count, 3),
        "forbidden_contact_forces_w",
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
    protocol_payload = protocol.to_dict() if protocol is not None else None
    semantics_payload = semantics.to_dict()
    forbidden_contact_manager_term = str(
        semantics_payload["episode_outcomes"]["signals"]["forbidden_contact"][
            "manager_term"
        ]
    )
    manager_to_reason = (
        dict(protocol_payload["phase0_probe"]["terminal_reason_by_manager_term"])
        if protocol_payload is not None
        else {
            "time_out": "timeout",
            forbidden_contact_manager_term: "forbidden_contact",
        }
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

    base_pos_values = _to_list(base_pos)
    base_quat_values = _to_list(base_quat)
    support_pos_values = _to_list(support_pos)
    support_vel_values = _to_list(support_vel)
    support_force_values = _to_list(support_forces)
    episode_steps = _to_list(episode_length)
    origin_values = _to_list(origins)
    done_values = _to_list(done_mask)
    term_values = {name: _to_list(mask) for name, mask in decoded_term_masks.items()}
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
    base_contact_terms = tuple(
        name
        for name, reason in manager_to_reason.items()
        if reason in {"base_contact", "forbidden_contact"}
    )
    if (
        len(base_contact_terms) != 1
        or base_contact_terms[0] != forbidden_contact_manager_term
    ):
        raise StairTransitionDecodeError(
            "Stair protocol must map exactly one base-contact manager term"
        )

    transitions: list[StairEnvTransition] = []
    for env_index, bin_id in enumerate(bin_assignment):
        cell = cell_assignment[env_index] if cell_assignment is not None else None
        origin = origin_values[env_index]
        base = base_pos_values[env_index]
        progress = float(base[0]) - float(origin[0])
        lateral = float(base[1]) - float(origin[1])
        base_height = float(base[2]) - float(origin[2])
        surface_query = (
            resolved_surface_queries[env_index]
            if resolved_surface_queries is not None
            else (
                stair_surface_query_for_ordered_cell(cell.spec)
                if isinstance(cell, OrderedStairEvaluationCell)
                else FlatStairSurfaceQuery(
                    protocol=protocol_payload,
                    step_height_m=(
                        cell.height_m
                        if isinstance(cell, StairEvaluationCell)
                        else frozen_bins[bin_id][1]
                    ),
                    tread_depth_m=(
                        cell.tread_depth_m
                        if isinstance(cell, StairEvaluationCell)
                        else None
                    ),
                )
            )
        )
        support_contacts: list[bool] = []
        support_progresses: list[float] = []
        support_lateral: list[float] = []
        support_surface_levels: list[int] = []
        support_speeds: list[float] = []
        support_clearances: list[float] = []
        support_upward_forces: list[float] = []
        for support_index in range(support_count):
            force = support_force_values[env_index][support_index]
            velocity = support_vel_values[env_index][support_index]
            position = support_pos_values[env_index][support_index]
            force_norm = math.sqrt(sum(float(value) ** 2 for value in force))
            support_contacts.append(force_norm > contact_threshold)
            support_upward_forces.append(float(force[2]))
            support_speeds.append(
                math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
            )
            support_progress = float(position[0]) - float(origin[0])
            support_progresses.append(support_progress)
            support_lateral_position = float(position[1]) - float(origin[1])
            support_lateral.append(support_lateral_position)
            surface = surface_query.sample(
                progress_m=support_progress,
                lateral_m=support_lateral_position,
            )
            support_surface_levels.append(surface.level)
            support_clearances.append(
                float(position[2]) - float(origin[2]) - surface.height_m
            )

        support_surface = supporting_stair_surface_height_m(
            support_progress_m=tuple(support_progresses),
            support_lateral_m=tuple(support_lateral),
            support_contacts=tuple(support_contacts),
            support_surface_clearance_m=tuple(support_clearances),
            maximum_projected_surface_penetration_m=(
                maximum_projected_surface_penetration_m
            ),
            surface_query=surface_query,
        )
        base_clearance = base_height - support_surface
        roll, pitch = _roll_pitch_from_wxyz(base_quat_values[env_index])

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
                base_height_m=base_height,
                base_clearance_m=base_clearance,
                abs_roll_rad=abs(roll),
                abs_pitch_rad=abs(pitch),
                forbidden_contact=bool(term_values[base_contact_terms[0]][env_index]),
                support_contacts=tuple(support_contacts),
                support_progress_m=tuple(support_progresses),
                support_surface_levels=tuple(support_surface_levels),
                support_lateral_m=tuple(support_lateral),
                support_planar_speeds_mps=tuple(support_speeds),
                support_surface_clearances_m=tuple(support_clearances),
                support_upward_forces_n=tuple(support_upward_forces),
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
    cell_assignment: tuple[
        StairEvaluationCell | OrderedStairEvaluationCell, ...
    ]
    | None,
) -> None:
    expected_origins = (
        protocol_payload["terrain"]["evaluation_grid_origins_m"]
        if cell_assignment is None
        else None
    )
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
            raise StairTransitionDecodeError(
                "Stair terrain origin/bin identity drifted"
            )


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


def _numeric_matrix(
    value: Any,
    expected_rows: int,
    field_name: str,
) -> tuple[Any, int]:
    """Validate a runtime-owned matrix while deriving its non-empty width."""

    try:
        actual_shape = tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StairTransitionDecodeError(
            f"Stair {field_name} does not expose a tensor shape"
        ) from exc
    if (
        len(actual_shape) != 2
        or actual_shape[0] != expected_rows
        or actual_shape[1] <= 0
    ):
        raise StairTransitionDecodeError(
            f"Stair {field_name} matrix shape is invalid: {actual_shape}"
        )
    tensor = _numeric_tensor(value, actual_shape, field_name)
    return tensor, actual_shape[1]


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
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum
        for item in values
    ):
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
        raise StairTransitionDecodeError(
            f"Stair {field_name} is not tensor-like"
        ) from exc
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

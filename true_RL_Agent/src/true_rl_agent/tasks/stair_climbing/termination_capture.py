"""Strict current-step termination capture for every Stair runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any

STAIR_TERMINATION_CAPTURE_SCHEME_VERSION = (
    "stair_term_callable_current_step_capture_v3"
)
_CAPTURE_STORE_ATTRIBUTE = "_true_rl_agent_stair_termination_capture_v3"
_CAPTURE_CALLABLE_MARKER = "_true_rl_agent_stair_termination_capture_v3"


class StairTerminationCaptureError(RuntimeError):
    """Raised when current-step termination evidence is incomplete or invalid."""


@dataclass(frozen=True)
class StairTerminationMaskSnapshot:
    term_order: tuple[str, ...]
    timeout_term_names: tuple[str, ...]
    term_masks: Mapping[str, Any]
    done_mask: Any
    time_out_mask: Any
    terminated_non_timeout_mask: Any

    def __post_init__(self) -> None:
        if (
            not isinstance(self.term_order, tuple)
            or not self.term_order
            or len(self.term_order) != len(set(self.term_order))
            or any(not isinstance(name, str) or not name for name in self.term_order)
        ):
            raise StairTerminationCaptureError(
                "termination snapshot term order must be unique and non-empty"
            )
        if set(self.term_masks) != set(self.term_order):
            raise StairTerminationCaptureError(
                "termination snapshot masks do not match the active term set"
            )
        if (
            not isinstance(self.timeout_term_names, tuple)
            or not self.timeout_term_names
            or len(self.timeout_term_names) != len(set(self.timeout_term_names))
            or not set(self.timeout_term_names) < set(self.term_order)
        ):
            raise StairTerminationCaptureError(
                "termination snapshot timeout classification is invalid"
            )
        object.__setattr__(self, "term_masks", MappingProxyType(dict(self.term_masks)))


@dataclass(frozen=True)
class _CapturedTerminationTerm:
    step_count: int
    mask: Any


def install_stair_termination_capture(terminations_cfg: Any) -> tuple[str, ...]:
    """Wrap every configured Stair termination term at its execution source."""

    try:
        config_items = tuple(vars(terminations_cfg).items())
    except TypeError as exc:
        raise StairTerminationCaptureError(
            "Stair termination config must expose task-owned term attributes"
        ) from exc

    installed: list[str] = []
    for term_name, term_cfg in config_items:
        if term_cfg is None or not hasattr(term_cfg, "func"):
            continue
        if not isinstance(term_name, str) or not term_name:
            raise StairTerminationCaptureError(
                "Stair termination config contains an invalid term name"
            )
        source_func = getattr(term_cfg, "func", None)
        if getattr(source_func, _CAPTURE_CALLABLE_MARKER, False) is True:
            raise StairTerminationCaptureError(
                f"Stair termination term {term_name!r} is already instrumented"
            )
        if not callable(source_func):
            raise StairTerminationCaptureError(
                f"Stair termination term {term_name!r} has no callable source"
            )
        raw_params = getattr(term_cfg, "params", None)
        if raw_params is None:
            source_params: dict[str, Any] = {}
        elif isinstance(raw_params, Mapping):
            source_params = dict(raw_params)
        else:
            raise StairTerminationCaptureError(
                f"Stair termination term {term_name!r} params must be a mapping"
            )
        term_cfg.func = _build_captured_termination_callable(
            term_name=term_name,
            source_func=source_func,
        )
        # Isaac Lab resolves SceneEntityCfg values only at the top level of
        # term_cfg.params. Keep the original parameter shape intact.
        term_cfg.params = source_params
        installed.append(term_name)

    if not installed:
        raise StairTerminationCaptureError(
            "Stair runtime requires at least one configured termination term"
        )
    if len(installed) != len(set(installed)):
        raise StairTerminationCaptureError(
            "Stair termination config contains duplicate term names"
        )
    return tuple(installed)


def is_stair_termination_capture_callable(value: Any) -> bool:
    """Return whether a termination callable has task-owned capture installed."""

    return callable(value) and getattr(value, _CAPTURE_CALLABLE_MARKER, False) is True


def stair_termination_manager_resolution_contract(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the expected resolved V1 base-contact shape from the protocol."""

    try:
        termination_schema = protocol["phase0_probe"]["termination_schema"]
        terms = termination_schema["terms"]
    except (KeyError, TypeError) as exc:
        raise StairTerminationCaptureError(
            "Stair protocol termination contract is incomplete"
        ) from exc
    if not isinstance(termination_schema, Mapping) or not isinstance(terms, list):
        raise StairTerminationCaptureError("Stair protocol termination schema is invalid")
    active_terms = [term.get("manager_term") for term in terms]
    if active_terms != ["time_out", "base_contact"]:
        raise StairTerminationCaptureError(
            "Stair V1 termination terms must be time_out and base_contact"
        )
    base_term = terms[1]
    base_params = base_term.get("parameters")
    if not isinstance(base_params, Mapping) or set(base_params) != {
        "sensor_cfg",
        "threshold",
    }:
        raise StairTerminationCaptureError(
            "Stair V1 base_contact parameters are invalid"
        )
    sensor = base_params["sensor_cfg"]
    if not isinstance(sensor, Mapping) or set(sensor) != {"name", "body_names"}:
        raise StairTerminationCaptureError(
            "Stair V1 base_contact sensor declaration is invalid"
        )
    body_names = _canonical_string_sequence(
        sensor.get("body_names"),
        "Stair V1 base_contact body_names",
    )
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing."
            "termination_manager_resolution.v1"
        ),
        "status": "pass",
        "termination_schema_hash": _canonical_text(
            termination_schema.get("schema_hash"),
            "termination schema_hash",
        ),
        "active_terms": active_terms,
        "base_contact": {
            "scene_entity_name": _canonical_text(
                sensor.get("name"),
                "base_contact sensor name",
            ),
            "body_names": list(body_names),
            "manager_scene_body_resolution": "match",
        },
    }


def verify_stair_termination_manager_resolution(
    env: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify Isaac resolved every task termination parameter as declared."""

    expected = stair_termination_manager_resolution_contract(protocol)
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        raise StairTerminationCaptureError("Stair environment has no termination manager")
    if list(_active_term_order(manager)) != expected["active_terms"]:
        raise StairTerminationCaptureError(
            "Stair termination manager active term order drifted"
        )
    get_term_cfg = getattr(manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise StairTerminationCaptureError(
            "Stair termination manager does not expose term configs"
        )
    scene = getattr(env, "scene", None)
    if scene is None:
        raise StairTerminationCaptureError("Stair environment has no scene")
    declared_terms = protocol["phase0_probe"]["termination_schema"]["terms"]
    for declared in declared_terms:
        term_name = declared["manager_term"]
        term_cfg = get_term_cfg(term_name)
        if not is_stair_termination_capture_callable(getattr(term_cfg, "func", None)):
            raise StairTerminationCaptureError(
                f"Stair termination term {term_name!r} is not capture-instrumented"
            )
        if getattr(term_cfg, "time_out", None) is not declared["time_out"]:
            raise StairTerminationCaptureError(
                "Stair termination manager timeout classification drifted"
            )
        runtime_params = getattr(term_cfg, "params", None)
        if not isinstance(runtime_params, Mapping):
            raise StairTerminationCaptureError(
                "Stair termination manager params are not a top-level mapping"
            )
        declared_params = declared["parameters"]
        if set(runtime_params) != set(declared_params):
            raise StairTerminationCaptureError(
                "Stair termination manager top-level parameter shape drifted"
            )
        if term_name == "time_out" and runtime_params:
            raise StairTerminationCaptureError("Stair time_out parameters drifted")
    base_cfg = get_term_cfg("base_contact")
    if base_cfg.params["threshold"] != declared_terms[1]["parameters"]["threshold"]:
        raise StairTerminationCaptureError("Stair base_contact threshold drifted")
    sensor_cfg = base_cfg.params["sensor_cfg"]
    resolved = expected["base_contact"]
    runtime_name = getattr(sensor_cfg, "name", None)
    runtime_body_names = _canonical_string_sequence(
        getattr(sensor_cfg, "body_names", None),
        "runtime base_contact body_names",
    )
    runtime_body_ids = _canonical_int_sequence(
        getattr(sensor_cfg, "body_ids", None),
        "runtime base_contact body_ids",
    )
    if (
        runtime_name != resolved["scene_entity_name"]
        or list(runtime_body_names) != resolved["body_names"]
    ):
        raise StairTerminationCaptureError(
            "Stair termination SceneEntityCfg resolution drifted"
        )
    try:
        scene_entity = scene[runtime_name]
    except (KeyError, TypeError) as exc:
        raise StairTerminationCaptureError(
            "Stair termination SceneEntityCfg scene entity is missing"
        ) from exc
    resolved_ids, resolved_names = scene_entity.find_bodies(
        list(runtime_body_names),
        preserve_order=True,
    )
    if (
        list(_canonical_string_sequence(resolved_names, "resolved body names"))
        != resolved["body_names"]
        or _canonical_int_sequence(resolved_ids, "resolved body ids")
        != runtime_body_ids
    ):
        raise StairTerminationCaptureError(
            "Stair termination scene body resolution drifted"
        )
    return expected


def verify_stair_vnext_termination_manager_resolution(
    env: Any,
    *,
    runtime_projection: Any,
    force_threshold_n: float,
) -> dict[str, Any]:
    """Verify VNext term ownership against resolved roles, not robot names."""

    from true_rl_agent.tasks.stair_climbing.runtime_projection import (
        StairRuntimeProjection,
    )

    if not isinstance(runtime_projection, StairRuntimeProjection):
        raise TypeError("Stair VNext termination projection is invalid")
    manager = getattr(env, "termination_manager", None)
    if manager is None or list(_active_term_order(manager)) != [
        "time_out",
        "base_contact",
    ]:
        raise StairTerminationCaptureError(
            "Stair VNext termination manager term set drifted"
        )
    for term_name in ("time_out", "base_contact"):
        term_cfg = manager.get_term_cfg(term_name)
        if not is_stair_termination_capture_callable(term_cfg.func):
            raise StairTerminationCaptureError(
                f"Stair VNext term {term_name!r} is not capture-instrumented"
            )
    base_cfg = manager.get_term_cfg("base_contact")
    if float(base_cfg.params.get("threshold")) != float(force_threshold_n):
        raise StairTerminationCaptureError(
            "Stair VNext forbidden-contact threshold drifted"
        )
    sensor_cfg = base_cfg.params.get("sensor_cfg")
    expected_names = tuple(runtime_projection.forbidden_contact_body_names)
    runtime_names = _canonical_string_sequence(
        getattr(sensor_cfg, "body_names", None),
        "Stair VNext forbidden contact body_names",
    )
    if (
        getattr(sensor_cfg, "name", None) != runtime_projection.contact_sensor_name
        or runtime_names != expected_names
    ):
        raise StairTerminationCaptureError(
            "Stair VNext forbidden-contact role resolution drifted"
        )
    scene_entity = env.scene[runtime_projection.contact_sensor_name]
    body_ids, body_names = scene_entity.find_bodies(
        list(expected_names), preserve_order=True
    )
    if (
        tuple(body_names) != expected_names
        or tuple(body_ids)
        != _canonical_int_sequence(
            getattr(sensor_cfg, "body_ids", None),
            "Stair VNext forbidden contact body_ids",
        )
    ):
        raise StairTerminationCaptureError(
            "Stair VNext forbidden-contact body IDs drifted"
        )
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing."
            "termination_manager_resolution.vnext.v1"
        ),
        "status": "pass",
        "active_terms": ["time_out", "base_contact"],
        "forbidden_contact": {
            "scene_entity_name": runtime_projection.contact_sensor_name,
            "role_id": "forbidden_contact",
            "body_count": len(expected_names),
            "force_threshold_n": float(force_threshold_n),
        },
    }


def _build_captured_termination_callable(
    *,
    term_name: str,
    source_func: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(source_func)
    def captured(env: Any, **source_params: Any) -> Any:
        return capture_stair_termination_term(
            env,
            term_name=term_name,
            source_func=source_func,
            source_params=source_params,
        )

    setattr(captured, _CAPTURE_CALLABLE_MARKER, True)
    return captured


def capture_stair_termination_term(
    env: Any,
    *,
    term_name: str,
    source_func: Callable[..., Any],
    source_params: Mapping[str, Any],
) -> Any:
    """Run one original term and retain its exact current-step return mask."""

    if not isinstance(term_name, str) or not term_name:
        raise StairTerminationCaptureError("captured termination term name is invalid")
    if not callable(source_func):
        raise StairTerminationCaptureError(
            f"captured termination term {term_name!r} source is not callable"
        )
    if not isinstance(source_params, Mapping):
        raise StairTerminationCaptureError(
            f"captured termination term {term_name!r} params are invalid"
        )

    current_mask = source_func(env, **dict(source_params))
    captured_mask = _clone_boolean_vector(
        current_mask,
        field_name=f"termination term {term_name}",
    )
    step_count = _current_step_count(env)
    store = getattr(env, _CAPTURE_STORE_ATTRIBUTE, None)
    if store is None:
        store = {}
        setattr(env, _CAPTURE_STORE_ATTRIBUTE, store)
    if not isinstance(store, dict):
        raise StairTerminationCaptureError(
            "Stair termination capture store has invalid runtime state"
        )
    previous = store.get(term_name)
    if isinstance(previous, _CapturedTerminationTerm) and previous.step_count == step_count:
        raise StairTerminationCaptureError(
            f"termination term {term_name!r} executed more than once in one step"
        )
    store[term_name] = _CapturedTerminationTerm(
        step_count=step_count,
        mask=captured_mask,
    )
    return current_mask


def capture_current_termination_masks(env: Any) -> StairTerminationMaskSnapshot:
    """Read only term returns captured during the environment's current step."""

    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        raise StairTerminationCaptureError("environment has no termination manager")
    term_order = _active_term_order(termination_manager)
    get_term_cfg = getattr(termination_manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise StairTerminationCaptureError(
            "termination manager does not expose term classifications"
        )
    timeout_term_names: list[str] = []
    for term_name in term_order:
        term_cfg = get_term_cfg(term_name)
        is_time_out = getattr(term_cfg, "time_out", None)
        if type(is_time_out) is not bool:
            raise StairTerminationCaptureError(
                f"termination term {term_name!r} has invalid timeout classification"
            )
        if is_time_out:
            timeout_term_names.append(term_name)
    store = getattr(env, _CAPTURE_STORE_ATTRIBUTE, None)
    if not isinstance(store, dict) or set(store) != set(term_order):
        raise StairTerminationCaptureError(
            "current-step termination capture does not match the active term set"
        )
    step_count = _current_step_count(env)
    term_masks: dict[str, Any] = {}
    for term_name in term_order:
        captured = store.get(term_name)
        if not isinstance(captured, _CapturedTerminationTerm):
            raise StairTerminationCaptureError(
                f"termination term {term_name!r} has invalid capture state"
            )
        if captured.step_count != step_count:
            raise StairTerminationCaptureError(
                f"termination term {term_name!r} was not captured in the current step"
            )
        term_masks[term_name] = captured.mask

    return build_termination_mask_snapshot(
        term_order=term_order,
        timeout_term_names=tuple(timeout_term_names),
        term_masks=term_masks,
        done_mask=getattr(termination_manager, "dones", None),
    )


def build_termination_mask_snapshot(
    *,
    term_order: tuple[str, ...],
    timeout_term_names: tuple[str, ...],
    term_masks: Mapping[str, Any],
    done_mask: Any,
) -> StairTerminationMaskSnapshot:
    """Validate and clone one complete current-step termination snapshot."""

    if (
        not isinstance(term_order, tuple)
        or not term_order
        or len(term_order) != len(set(term_order))
        or any(not isinstance(name, str) or not name for name in term_order)
    ):
        raise StairTerminationCaptureError(
            "termination manager active term set is invalid"
        )
    if not isinstance(term_masks, Mapping) or set(term_masks) != set(term_order):
        raise StairTerminationCaptureError(
            "termination snapshot masks do not match the active term set"
        )
    if (
        not isinstance(timeout_term_names, tuple)
        or not timeout_term_names
        or len(timeout_term_names) != len(set(timeout_term_names))
        or not set(timeout_term_names) < set(term_order)
    ):
        raise StairTerminationCaptureError(
            "termination snapshot timeout classification is invalid"
        )
    cloned_done = _clone_boolean_vector(
        done_mask,
        field_name="termination manager dones",
    )
    cloned_terms = {
        name: _clone_boolean_vector(
            term_masks[name],
            field_name=f"termination term {name}",
            expected_shape=tuple(cloned_done.shape),
        )
        for name in term_order
    }
    time_out_mask = cloned_terms[timeout_term_names[0]]
    for name in timeout_term_names[1:]:
        time_out_mask = time_out_mask | cloned_terms[name]
    timeout_names = set(timeout_term_names)
    terminated_term_names = tuple(
        name for name in term_order if name not in timeout_names
    )
    terminated_non_timeout_mask = cloned_terms[terminated_term_names[0]]
    for name in terminated_term_names[1:]:
        terminated_non_timeout_mask = (
            terminated_non_timeout_mask | cloned_terms[name]
        )
    union = time_out_mask | terminated_non_timeout_mask
    if not _mask_equal(union, cloned_done):
        raise StairTerminationCaptureError(
            "termination manager done mask does not equal OR(active term masks)"
        )
    return StairTerminationMaskSnapshot(
        term_order=term_order,
        timeout_term_names=timeout_term_names,
        term_masks=cloned_terms,
        done_mask=cloned_done,
        time_out_mask=time_out_mask,
        terminated_non_timeout_mask=terminated_non_timeout_mask,
    )


def _active_term_order(termination_manager: Any) -> tuple[str, ...]:
    raw_terms = getattr(termination_manager, "active_terms", None)
    if not isinstance(raw_terms, list | tuple):
        raise StairTerminationCaptureError(
            "termination manager active_terms must be an ordered sequence"
        )
    term_order = tuple(raw_terms)
    if (
        not term_order
        or len(term_order) != len(set(term_order))
        or any(not isinstance(name, str) or not name for name in term_order)
    ):
        raise StairTerminationCaptureError(
            "termination manager active term set is invalid"
        )
    return term_order


def _current_step_count(env: Any) -> int:
    value = getattr(env, "common_step_counter", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StairTerminationCaptureError(
            "environment common_step_counter must be a non-negative integer"
        )
    return value


def _clone_boolean_vector(
    value: Any,
    *,
    field_name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> Any:
    shape = getattr(value, "shape", None)
    try:
        resolved_shape = tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        resolved_shape = ()
    if len(resolved_shape) != 1 or resolved_shape[0] <= 0:
        raise StairTerminationCaptureError(f"{field_name} must be a non-empty vector")
    if expected_shape is not None and resolved_shape != expected_shape:
        raise StairTerminationCaptureError(f"{field_name} shape drifted")
    if str(getattr(value, "dtype", "")) not in {"bool", "torch.bool", "bool_"}:
        raise StairTerminationCaptureError(f"{field_name} must have boolean dtype")
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise StairTerminationCaptureError(f"{field_name} cannot be detached")
    detached = detach()
    clone = getattr(detached, "clone", None)
    if not callable(clone):
        raise StairTerminationCaptureError(f"{field_name} cannot be cloned")
    return clone()


def _mask_equal(left: Any, right: Any) -> bool:
    try:
        result = (left == right).all()
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StairTerminationCaptureError(
            "termination masks do not support exact equality"
        ) from exc


def _canonical_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StairTerminationCaptureError(f"Stair {field_name} must be canonical text")
    return value


def _canonical_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise StairTerminationCaptureError(f"{field_name} must be a string sequence")
    result = tuple(_canonical_text(item, field_name) for item in value)
    if len(result) != len(set(result)):
        raise StairTerminationCaptureError(f"{field_name} must be unique")
    return result


def _canonical_int_sequence(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StairTerminationCaptureError(f"{field_name} must be an integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise StairTerminationCaptureError(f"{field_name} must contain integers")
    return tuple(value)


__all__ = [
    "STAIR_TERMINATION_CAPTURE_SCHEME_VERSION",
    "StairTerminationCaptureError",
    "StairTerminationMaskSnapshot",
    "build_termination_mask_snapshot",
    "capture_current_termination_masks",
    "capture_stair_termination_term",
    "install_stair_termination_capture",
    "is_stair_termination_capture_callable",
    "stair_termination_manager_resolution_contract",
    "verify_stair_termination_manager_resolution",
    "verify_stair_vnext_termination_manager_resolution",
]

"""Runner-owned adapter for Phase 8E callable training components.

Phase 8E-2 defines the callable contract and verifies it without launching the
formal Isaac/RSL-RL loop.  The adapter is intentionally runner-owned: candidate
code supplies callables, while this module validates structure, pre-evaluates
schedules, and guards reward/loss outputs before any later trainer can consume
them.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from humanoid_agent.code_sandbox.components import DiagnosticTerm, LossTerm, RewardInputs, RewardTerm, SchedulePayload, ScheduleTerm, TrainingComponents


DEFAULT_SCHEDULE_ALLOWED_KEYS = (
    "push_magnitude_scale",
    "push_start_fraction",
    "push_interval_s",
    "domain_randomization_scale",
    "reward_weight_scale",
)
FORBIDDEN_SCHEDULE_KEY_RE = re.compile(r"(?:seed|eval|hidden|iid|budget|runner|checkpoint|metric)", re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
MAX_DIAGNOSTIC_TERMS = 32
MAX_DIAGNOSTIC_FIELDS_PER_TERM = 128
MAX_TOTAL_DIAGNOSTIC_FIELDS = 512


class ComponentContractError(ValueError):
    """Raised when candidate components violate the runner-owned contract."""


@dataclass(frozen=True)
class GuardedOutputSummary:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    num_envs: int
    finite: bool
    clip_min: float | None = None
    clip_max: float | None = None
    clip_applied: bool = False

    def to_public_summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedVectorResult:
    summary: GuardedOutputSummary
    value: Any


def validate_training_components(components: TrainingComponents) -> dict[str, Any]:
    """Validate component structure without executing candidate callables."""

    if not isinstance(components, TrainingComponents):
        raise ComponentContractError("candidate entrypoint must return TrainingComponents")
    _validate_name(components.name, "components.name")
    _validate_term_list(components.reward_terms, RewardTerm, "reward_terms")
    _validate_term_list(components.loss_terms, LossTerm, "loss_terms")
    _validate_term_list(components.schedules, ScheduleTerm, "schedules")
    _validate_term_list(components.diagnostics, DiagnosticTerm, "diagnostics", max_items=MAX_DIAGNOSTIC_TERMS)
    if components.train_observation_transforms:
        raise ComponentContractError("train_observation_transforms are not enabled in Phase 8E")
    names: set[str] = set()
    for family, terms in (
        ("reward_terms", components.reward_terms),
        ("loss_terms", components.loss_terms),
        ("schedules", components.schedules),
        ("diagnostics", components.diagnostics),
    ):
        for term in terms:
            if term.name in names:
                raise ComponentContractError(f"duplicate component name: {term.name}")
            names.add(term.name)
            _validate_name(term.name, f"{family}.{term.name}")
    return components.to_public_summary()


def build_reward_inputs(mapping: dict[str, Any]) -> RewardInputs:
    """Construct ``RewardInputs`` from trusted runner tensors.

    This is a shape-agnostic constructor; per-output vector shape is guarded
    when a candidate term is evaluated.
    """

    missing = [name for name in RewardInputs.field_names() if name not in mapping]
    if missing:
        raise ComponentContractError(f"missing RewardInputs fields: {missing}")
    extra = sorted(set(mapping) - set(RewardInputs.field_names()))
    if extra:
        raise ComponentContractError(f"unexpected RewardInputs fields: {extra}")
    return RewardInputs(**{name: mapping[name] for name in RewardInputs.field_names()})


def build_synthetic_reward_inputs(
    contract: dict[str, Any],
    *,
    num_envs: int,
    num_joints: int,
    num_actions: int,
    prefer_torch: bool = True,
    tensor_backend: dict[str, Any] | None = None,
) -> RewardInputs:
    """Build fast-validation inputs from the machine-readable RewardInputs contract."""

    if num_envs <= 0 or num_joints <= 0 or num_actions <= 0:
        raise ComponentContractError("synthetic RewardInputs dimensions must be positive")
    backend = tensor_backend if tensor_backend is not None else synthetic_tensor_backend(prefer_torch=prefer_torch)
    fields = contract.get("fields")
    if not isinstance(fields, dict):
        raise ComponentContractError("RewardInputs contract missing fields mapping")
    mapping: dict[str, Any] = {}
    for name in RewardInputs.field_names():
        field = fields.get(name)
        if not isinstance(field, dict):
            raise ComponentContractError(f"RewardInputs contract missing field: {name}")
        shape = field.get("shape")
        if not isinstance(shape, list):
            raise ComponentContractError(f"RewardInputs contract field has no shape: {name}")
        dims = _resolve_contract_shape(shape, num_envs=num_envs, num_joints=num_joints, num_actions=num_actions)
        mapping[name] = _synthetic_field_value(name, dims)
    if backend.get("backend") == "torch":
        mapping = _torch_mapping(mapping)
    elif prefer_torch:
        mapping = {name: _SyntheticTensor(value) for name, value in mapping.items()}
    return build_reward_inputs(mapping)


def fast_validate_components(
    components: TrainingComponents,
    *,
    contract: dict[str, Any],
    num_envs: int,
    num_joints: int,
    num_actions: int,
    progress_points: Iterable[float] = (0.0, 0.5, 1.0),
    allowed_schedule_keys: Iterable[str] = DEFAULT_SCHEDULE_ALLOWED_KEYS,
    prefer_torch: bool = True,
) -> dict[str, Any]:
    """Run cheap callable shape checks over contract-derived synthetic inputs."""

    summary: dict[str, Any] = {
        "schema_version": "humanoid_agent.phase8e.dynamic_fast_validation.v1",
        "status": "pass",
        "synthetic_inputs": {
            "source": "reward_inputs_contract",
            "num_envs": num_envs,
            "num_joints": num_joints,
            "num_actions": num_actions,
            "field_order": list(contract.get("field_order", [])),
        },
        "component_counts": {},
        "schedules": [],
        "reward_terms": [],
        "loss_terms": [],
        "diagnostics": [],
        "issues": [],
    }
    issues: list[dict[str, Any]] = summary["issues"]
    tensor_backend = synthetic_tensor_backend(prefer_torch=prefer_torch)
    summary["synthetic_inputs"]["tensor_backend"] = tensor_backend
    try:
        component_summary = validate_training_components(components)
    except Exception as exc:  # noqa: BLE001 - public issue record, not a trust boundary.
        issues.append(_fast_validation_issue(kind="structure", code="component_structure_invalid", component_name=None, exc=exc))
        summary["status"] = "fail"
        return summary
    summary["component_counts"] = component_summary.get("counts", {})
    try:
        inputs = build_synthetic_reward_inputs(
            contract,
            num_envs=num_envs,
            num_joints=num_joints,
            num_actions=num_actions,
            prefer_torch=prefer_torch,
            tensor_backend=tensor_backend,
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(_fast_validation_issue(kind="synthetic_inputs", code="synthetic_inputs_invalid", component_name=None, exc=exc))
        summary["status"] = "fail"
        return summary

    progress_points_tuple = tuple(progress_points)
    allowed_keys_tuple = tuple(allowed_schedule_keys)
    for term in components.schedules:
        try:
            table = precompute_schedule_table(
                TrainingComponents(name=components.name, schedules=[term]),
                progress_points=progress_points_tuple,
                allowed_keys=allowed_keys_tuple,
            )
            summary["schedules"].append({"name": term.name, "table": table})
        except Exception as exc:  # noqa: BLE001
            issues.append(_fast_validation_issue(kind="schedule", code="schedule_invalid", component_name=getattr(term, "name", None), exc=exc))

    for term in components.reward_terms:
        try:
            guarded = guard_vector_output(
                name=term.name,
                kind="reward",
                value=term.fn(inputs),
                num_envs=num_envs,
                clip_min=term.clip_min,
                clip_max=term.clip_max,
            )
            summary["reward_terms"].append(guarded.to_public_summary())
        except Exception as exc:  # noqa: BLE001
            issues.append(_fast_validation_issue(kind="reward", code="reward_term_invalid", component_name=getattr(term, "name", None), exc=exc))

    for term in components.loss_terms:
        try:
            guarded = guard_vector_output(
                name=term.name,
                kind="loss",
                value=term.fn(inputs),
                num_envs=num_envs,
                clip_min=term.clip_min,
                clip_max=term.clip_max,
            )
            summary["loss_terms"].append(guarded.to_public_summary())
        except Exception as exc:  # noqa: BLE001
            issues.append(_fast_validation_issue(kind="loss", code="loss_term_invalid", component_name=getattr(term, "name", None), exc=exc))

    total_diagnostic_fields = 0
    for term in components.diagnostics:
        try:
            payload = term.fn(inputs)
            row = _validate_diagnostic_payload(term=term, payload=payload, num_envs=num_envs)
            total_diagnostic_fields += row["field_count"]
            _validate_total_diagnostic_fields(total_fields=total_diagnostic_fields, term=term)
            summary["diagnostics"].append(row)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                _fast_validation_issue(kind="diagnostic", code="diagnostic_term_invalid", component_name=getattr(term, "name", None), exc=exc)
            )

    if issues:
        summary["status"] = "fail"
    return summary


def precompute_schedule_table(
    components: TrainingComponents,
    *,
    progress_points: Iterable[float],
    allowed_keys: Iterable[str] = DEFAULT_SCHEDULE_ALLOWED_KEYS,
) -> dict[str, Any]:
    """Evaluate schedule callables on a fixed public progress grid."""

    validate_training_components(components)
    allowed = tuple(allowed_keys)
    _validate_allowed_schedule_keys(allowed)
    rows: list[dict[str, Any]] = []
    for progress in progress_points:
        progress_float = float(progress)
        if not math.isfinite(progress_float) or not 0.0 <= progress_float <= 1.0:
            raise ComponentContractError(f"schedule progress must be finite in [0, 1], got {progress!r}")
        row: dict[str, Any] = {"progress_fraction": progress_float, "values": {}}
        for term in components.schedules:
            payload = term.fn(progress_float)
            row["values"][term.name] = _validate_schedule_payload(
                term=term,
                payload=payload,
                runner_allowed_keys=allowed,
            )
        rows.append(row)
    return {
        "schema_version": "humanoid_agent.phase8e.schedule_table.v1",
        "progress_points": [row["progress_fraction"] for row in rows],
        "schedule_names": [term.name for term in components.schedules],
        "runner_allowed_keys": list(allowed),
        "rows": rows,
    }


def evaluate_reward_terms(
    components: TrainingComponents,
    inputs: RewardInputs,
    *,
    num_envs: int,
) -> list[GuardedOutputSummary]:
    """Run reward callables and guard their vector outputs."""

    validate_training_components(components)
    return [
        guard_vector_output(
            name=term.name,
            kind="reward",
            value=term.fn(inputs),
            num_envs=num_envs,
            clip_min=term.clip_min,
            clip_max=term.clip_max,
        )
        for term in components.reward_terms
    ]


def evaluate_loss_terms(
    components: TrainingComponents,
    inputs: RewardInputs,
    *,
    num_envs: int,
) -> list[GuardedOutputSummary]:
    """Run loss callables and guard their vector outputs."""

    validate_training_components(components)
    return [
        guard_vector_output(
            name=term.name,
            kind="loss",
            value=term.fn(inputs),
            num_envs=num_envs,
            clip_min=term.clip_min,
            clip_max=term.clip_max,
        )
        for term in components.loss_terms
    ]


def evaluate_diagnostic_terms(
    components: TrainingComponents,
    inputs: RewardInputs,
    *,
    num_envs: int,
) -> list[dict[str, Any]]:
    """Run public diagnostic callables and summarize numeric outputs."""

    validate_training_components(components)
    rows: list[dict[str, Any]] = []
    total_fields = 0
    for term in components.diagnostics:
        payload = term.fn(inputs)
        row = _validate_diagnostic_payload(term=term, payload=payload, num_envs=num_envs)
        total_fields += row["field_count"]
        _validate_total_diagnostic_fields(total_fields=total_fields, term=term)
        rows.append(row)
    return rows


def guard_vector_output(
    *,
    name: str,
    kind: str,
    value: Any,
    num_envs: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> GuardedOutputSummary:
    """Validate one candidate vector output.

    The value may be a torch tensor, numpy array, or a plain numeric sequence.
    This function intentionally uses duck-typing so contract tests do not need
    torch installed locally.
    """

    return guard_vector_value(
        name=name,
        kind=kind,
        value=value,
        num_envs=num_envs,
        clip_min=clip_min,
        clip_max=clip_max,
    ).summary


def guard_vector_value(
    *,
    name: str,
    kind: str,
    value: Any,
    num_envs: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> GuardedVectorResult:
    """Validate one vector output and return the guarded value for training.

    Torch tensors are guarded on-device with ``torch.isfinite`` and optional
    clipping.  Non-torch values fall back to materializing nested Python data,
    which is acceptable for local selftests and non-training contract checks.
    """

    if num_envs <= 0:
        raise ComponentContractError("num_envs must be positive")
    _validate_clip_bounds(kind=kind, name=name, clip_min=clip_min, clip_max=clip_max)
    torch_mod = _torch_module_for_tensor(value)
    if torch_mod is not None:
        return _guard_torch_vector(
            torch_mod=torch_mod,
            name=name,
            kind=kind,
            value=value,
            num_envs=num_envs,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    data = _to_nested_python(value)
    shape = _nested_shape(data)
    reported_shape = _reported_shape_of(value)
    if reported_shape is not None and reported_shape != shape:
        raise ComponentContractError(
            f"{kind} term {name!r} reported shape {reported_shape} but materialized to {shape}"
        )
    if shape != (num_envs,):
        raise ComponentContractError(f"{kind} term {name!r} must return shape ({num_envs},), got {shape}")
    if not _all_finite_numbers(data):
        raise ComponentContractError(f"{kind} term {name!r} returned non-finite or non-numeric values")
    guarded = _clip_nested(data, clip_min=clip_min, clip_max=clip_max)
    return GuardedVectorResult(
        summary=GuardedOutputSummary(
            name=name,
            kind=kind,
            shape=shape,
            dtype=str(getattr(value, "dtype", type(value).__name__)),
            num_envs=num_envs,
            finite=True,
            clip_min=clip_min,
            clip_max=clip_max,
            clip_applied=clip_min is not None or clip_max is not None,
        ),
        value=guarded,
    )


def _validate_term_list(values: Any, expected_type: type, family: str, *, max_items: int | None = None) -> None:
    if not isinstance(values, list):
        raise ComponentContractError(f"{family} must be a list")
    if max_items is not None and len(values) > max_items:
        raise ComponentContractError(f"{family} contains {len(values)} terms; max_diagnostic_terms={max_items}")
    for index, item in enumerate(values):
        if not isinstance(item, expected_type):
            raise ComponentContractError(f"{family}[{index}] must be {expected_type.__name__}")
        if not callable(item.fn):
            raise ComponentContractError(f"{family}[{index}].fn must be callable")
        if isinstance(item, RewardTerm | LossTerm):
            _validate_finite_scalar(item.weight, f"{family}[{index}].weight")
            if item.clip_min is not None:
                _validate_finite_scalar(item.clip_min, f"{family}[{index}].clip_min")
            if item.clip_max is not None:
                _validate_finite_scalar(item.clip_max, f"{family}[{index}].clip_max")
        if isinstance(item, ScheduleTerm):
            _validate_allowed_schedule_keys(item.allowed_keys)


def _validate_name(value: str, field: str) -> None:
    if not isinstance(value, str) or not SAFE_NAME_RE.match(value):
        raise ComponentContractError(f"{field} must match {SAFE_NAME_RE.pattern}")


def _validate_finite_scalar(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ComponentContractError(f"{field} must be a finite number")


def _validate_allowed_schedule_keys(keys: Iterable[str]) -> None:
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ComponentContractError("schedule keys must be non-empty strings")
        if FORBIDDEN_SCHEDULE_KEY_RE.search(key):
            raise ComponentContractError(f"schedule key is forbidden by public contract: {key}")


def _validate_schedule_payload(
    *,
    term: ScheduleTerm,
    payload: Any,
    runner_allowed_keys: tuple[str, ...],
) -> SchedulePayload:
    if not isinstance(payload, dict):
        raise ComponentContractError(f"schedule {term.name!r} must return a dict")
    term_allowed = set(term.allowed_keys)
    runner_allowed = set(runner_allowed_keys)
    out: SchedulePayload = {}
    for key, value in payload.items():
        if key not in term_allowed:
            raise ComponentContractError(f"schedule {term.name!r} returned undeclared key {key!r}")
        if key not in runner_allowed:
            raise ComponentContractError(f"schedule {term.name!r} returned runner-disallowed key {key!r}")
        _validate_schedule_value(value, f"{term.name}.{key}")
        out[str(key)] = _json_safe_schedule_value(value)
    return out


def _validate_diagnostic_payload(
    *,
    term: DiagnosticTerm,
    payload: Any,
    num_envs: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ComponentContractError(f"diagnostic {term.name!r} must return a dict")
    field_count = len(payload)
    if field_count > MAX_DIAGNOSTIC_FIELDS_PER_TERM:
        raise ComponentContractError(
            f"diagnostic {term.name!r} returned {field_count} fields; "
            f"max_fields_per_diagnostic={MAX_DIAGNOSTIC_FIELDS_PER_TERM}"
        )
    fields: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not SAFE_NAME_RE.match(key):
            raise ComponentContractError(f"diagnostic {term.name!r} returned unsafe key {key!r}")
        fields[key] = _summarize_diagnostic_value(f"{term.name}.{key}", value, num_envs=num_envs)
    return {
        "name": term.name,
        "kind": "diagnostic",
        "field_count": field_count,
        "limits": {
            "max_fields_per_diagnostic": MAX_DIAGNOSTIC_FIELDS_PER_TERM,
            "max_total_diagnostic_fields": MAX_TOTAL_DIAGNOSTIC_FIELDS,
        },
        "fields": fields,
    }


def _validate_total_diagnostic_fields(*, total_fields: int, term: DiagnosticTerm) -> None:
    if total_fields > MAX_TOTAL_DIAGNOSTIC_FIELDS:
        raise ComponentContractError(
            f"diagnostic field count exceeded after {term.name!r}: "
            f"{total_fields} fields; max_total_diagnostic_fields={MAX_TOTAL_DIAGNOSTIC_FIELDS}"
        )


def _summarize_diagnostic_value(field: str, value: Any, *, num_envs: int) -> dict[str, Any]:
    torch_mod = _torch_module_for_tensor(value)
    if torch_mod is not None:
        return _summarize_torch_diagnostic(torch_mod, field, value, num_envs=num_envs)
    data = _to_nested_python(value)
    if isinstance(data, bool):
        return {"kind": "bool", "value": data}
    if isinstance(data, int | float):
        if not math.isfinite(float(data)):
            raise ComponentContractError(f"diagnostic {field} must be finite")
        return {"kind": "scalar", "value": float(data)}
    shape = _nested_shape(data)
    if shape != (num_envs,):
        raise ComponentContractError(f"diagnostic {field} must be scalar or shape ({num_envs},), got {shape}")
    if not _all_finite_numbers(data):
        raise ComponentContractError(f"diagnostic {field} returned non-finite or non-numeric values")
    values = [float(item) for item in data]
    return {
        "kind": "vector_summary",
        "shape": [num_envs],
        "mean": sum(values) / len(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _summarize_torch_diagnostic(torch_mod: Any, field: str, value: Any, *, num_envs: int) -> dict[str, Any]:
    if getattr(value, "is_complex", lambda: False)():
        raise ComponentContractError(f"diagnostic {field} must not be complex")
    shape = tuple(int(item) for item in value.shape)
    if shape == ():
        if value.dtype == torch_mod.bool:
            return {"kind": "bool", "value": bool(value.detach().cpu().item())}
        if not bool(torch_mod.isfinite(value).all().item()):
            raise ComponentContractError(f"diagnostic {field} must be finite")
        return {"kind": "scalar", "value": float(value.detach().cpu().item())}
    if shape != (num_envs,):
        raise ComponentContractError(f"diagnostic {field} must be scalar or shape ({num_envs},), got {shape}")
    if value.dtype == torch_mod.bool:
        data = value.detach()
        return {
            "kind": "bool_vector_summary",
            "shape": [num_envs],
            "true_fraction": float(data.float().mean().detach().cpu().item()),
        }
    if not bool(torch_mod.isfinite(value).all().item()):
        raise ComponentContractError(f"diagnostic {field} returned non-finite values")
    data = value.detach().float()
    return {
        "kind": "vector_summary",
        "shape": [num_envs],
        "mean": float(data.mean().cpu().item()),
        "min": float(data.min().cpu().item()),
        "max": float(data.max().cpu().item()),
    }


def _validate_schedule_value(value: Any, field: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ComponentContractError(f"{field} must be finite")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_schedule_value(item, f"{field}[{index}]")
        return
    raise ComponentContractError(f"{field} must be numeric/bool or a nested list of those values")


def _json_safe_schedule_value(value: Any) -> ScheduleValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, tuple):
        return [_json_safe_schedule_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_schedule_value(item) for item in value]
    raise ComponentContractError(f"unsupported schedule value: {value!r}")


def _validate_clip_bounds(*, kind: str, name: str, clip_min: float | None, clip_max: float | None) -> None:
    if clip_min is not None:
        _validate_finite_scalar(clip_min, f"{kind} term {name!r}.clip_min")
    if clip_max is not None:
        _validate_finite_scalar(clip_max, f"{kind} term {name!r}.clip_max")
    if clip_min is not None and clip_max is not None and float(clip_min) > float(clip_max):
        raise ComponentContractError(f"{kind} term {name!r} has clip_min > clip_max")


def _guard_torch_vector(
    *,
    torch_mod: Any,
    name: str,
    kind: str,
    value: Any,
    num_envs: int,
    clip_min: float | None,
    clip_max: float | None,
) -> GuardedVectorResult:
    if getattr(value, "is_complex", lambda: False)():
        raise ComponentContractError(f"{kind} term {name!r} must not return complex values")
    if value.dtype == torch_mod.bool:
        raise ComponentContractError(f"{kind} term {name!r} must return numeric values, not bool")
    shape = tuple(int(item) for item in value.shape)
    if shape != (num_envs,):
        raise ComponentContractError(f"{kind} term {name!r} must return shape ({num_envs},), got {shape}")
    if not bool(torch_mod.isfinite(value).all().item()):
        raise ComponentContractError(f"{kind} term {name!r} returned non-finite values")
    guarded = value
    if clip_min is not None or clip_max is not None:
        kwargs: dict[str, float] = {}
        if clip_min is not None:
            kwargs["min"] = float(clip_min)
        if clip_max is not None:
            kwargs["max"] = float(clip_max)
        guarded = torch_mod.clamp(value, **kwargs)
    return GuardedVectorResult(
        summary=GuardedOutputSummary(
            name=name,
            kind=kind,
            shape=shape,
            dtype=str(value.dtype),
            num_envs=num_envs,
            finite=True,
            clip_min=clip_min,
            clip_max=clip_max,
            clip_applied=clip_min is not None or clip_max is not None,
        ),
        value=guarded,
    )


def _torch_module_for_tensor(value: Any) -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - torch is optional for local contract tests.
        return None
    return torch if isinstance(value, torch.Tensor) else None


def _clip_nested(value: Any, *, clip_min: float | None, clip_max: float | None) -> Any:
    if clip_min is None and clip_max is None:
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        out = float(value)
        if clip_min is not None:
            out = max(out, float(clip_min))
        if clip_max is not None:
            out = min(out, float(clip_max))
        return out
    if isinstance(value, list):
        return [_clip_nested(item, clip_min=clip_min, clip_max=clip_max) for item in value]
    if isinstance(value, tuple):
        return tuple(_clip_nested(item, clip_min=clip_min, clip_max=clip_max) for item in value)
    return value


def _reported_shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _to_nested_python(value: Any) -> Any:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    return current


def _resolve_contract_shape(shape: list[Any], *, num_envs: int, num_joints: int, num_actions: int) -> list[int]:
    aliases = {
        "N": num_envs,
        "num_envs": num_envs,
        "num_joints": num_joints,
        "num_actions": num_actions,
    }
    dims: list[int] = []
    for raw in shape:
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str) and raw in aliases:
            value = aliases[raw]
        else:
            raise ComponentContractError(f"unsupported RewardInputs synthetic shape token: {raw!r}")
        if value <= 0:
            raise ComponentContractError(f"RewardInputs synthetic shape dimension must be positive: {raw!r}")
        dims.append(value)
    return dims


def _synthetic_field_value(name: str, dims: list[int]) -> Any:
    if not dims:
        return 0.0
    if name == "base_height" and len(dims) == 1:
        return [0.8 for _ in range(dims[0])]
    if name == "projected_gravity" and len(dims) == 2 and dims[1] == 3:
        return [[0.0, 0.0, -1.0] for _ in range(dims[0])]
    if name == "progress_fraction" and len(dims) == 1:
        return [0.5 for _ in range(dims[0])]
    return _zeros_for_shape(dims)


def _zeros_for_shape(dims: list[int]) -> Any:
    if not dims:
        return 0.0
    return [_zeros_for_shape(dims[1:]) for _ in range(dims[0])]


def synthetic_tensor_backend(*, prefer_torch: bool = True) -> dict[str, Any]:
    """Return the tensor backend used for synthetic fast-validation inputs."""

    if not prefer_torch:
        return {
            "backend": "synthetic",
            "prefer_torch": False,
            "torch_available": False,
            "torch_version": None,
            "degraded": False,
            "degraded_reason": None,
        }
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - torch is optional for local contract tests.
        return {
            "backend": "synthetic",
            "prefer_torch": True,
            "torch_available": False,
            "torch_version": None,
            "degraded": True,
            "degraded_reason": "torch_import_failed",
            "torch_import_error_type": type(exc).__name__,
            "torch_import_error": str(exc)[:300],
        }
    return {
        "backend": "torch",
        "prefer_torch": True,
        "torch_available": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "degraded": False,
        "degraded_reason": None,
    }


def _torch_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    return {name: torch.as_tensor(value, dtype=torch.float32) for name, value in mapping.items()}


class _SyntheticTensor:
    def __init__(self, data: Any):
        self._data = data
        self.shape = _nested_shape(data)
        self.dtype = "synthetic_float"

    def detach(self) -> "_SyntheticTensor":
        return self

    def cpu(self) -> "_SyntheticTensor":
        return self

    def tolist(self) -> Any:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: Any) -> Any:
        return _SyntheticTensor(_index_nested(self._data, key))

    def __add__(self, other: Any) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(self._data, _unwrap_synthetic(other), lambda left, right: left + right))

    def __radd__(self, other: Any) -> "_SyntheticTensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(self._data, _unwrap_synthetic(other), lambda left, right: left - right))

    def __rsub__(self, other: Any) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(_unwrap_synthetic(other), self._data, lambda left, right: left - right))

    def __mul__(self, other: Any) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(self._data, _unwrap_synthetic(other), lambda left, right: left * right))

    def __rmul__(self, other: Any) -> "_SyntheticTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(self._data, _unwrap_synthetic(other), lambda left, right: left / right))

    def __neg__(self) -> "_SyntheticTensor":
        return _SyntheticTensor(_elementwise(self._data, 0.0, lambda left, _right: -left))


def _unwrap_synthetic(value: Any) -> Any:
    return value._data if isinstance(value, _SyntheticTensor) else value


def _elementwise(left: Any, right: Any, op) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_elementwise(left_item, right_item, op) for left_item, right_item in zip(left, right, strict=True)]
    if isinstance(left, list):
        return [_elementwise(item, right, op) for item in left]
    if isinstance(right, list):
        return [_elementwise(left, item, op) for item in right]
    return op(float(left), float(right))


def _index_nested(data: Any, key: Any) -> Any:
    if not isinstance(key, tuple):
        return data[key]
    if len(key) == 2 and isinstance(key[0], slice):
        rows = data[key[0]]
        return [row[key[1]] for row in rows]
    current = data
    for part in key:
        current = current[part]
    return current


def _fast_validation_issue(*, kind: str, code: str, component_name: Any, exc: Exception) -> dict[str, Any]:
    return {
        "code": code,
        "kind": kind,
        "component_name": component_name if isinstance(component_name, str) else None,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
    }


def _nested_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, list | tuple):
        if not value:
            return (0,)
        child_shape = _nested_shape(value[0])
        for item in value[1:]:
            if _nested_shape(item) != child_shape:
                raise ComponentContractError("candidate output must be rectangular")
        return (len(value), *child_shape)
    return ()


def _all_finite_numbers(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, list | tuple):
        return all(_all_finite_numbers(item) for item in value)
    return False

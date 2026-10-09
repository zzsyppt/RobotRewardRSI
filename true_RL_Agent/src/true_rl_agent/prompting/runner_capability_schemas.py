"""Schema previews for future runner-owned candidate capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.agent_core.public_safety import (
    assert_public_only_prompt_payload,
    contains_forbidden_public_value_marker,
    sanitize_prompt_public_value,
)

CURRICULUM_DESIGN_CAPABILITY_ID = "curriculum_design"
TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID = "train_environment_design"
RUNNER_CAPABILITY_READINESS_REQUIREMENTS = ("task_enabled", "compiler_ready", "verifier_ready")
RUNNER_CAPABILITY_FORBIDDEN_FIELD_PARTS = (
    "eval_protocol",
    "eval_patch",
    "success_metric",
    "training_budget",
    "budget",
    "seed_policy",
    "checkpoint",
    "ppo",
    "runner_patch",
    "runner_source",
    "runner_code",
    "raw_seed",
    "eval_seed",
    "trusted_eval",
    "trusted_seed",
    "seed",
    "path",
    "secret",
    "api_key",
    "token",
)
RUNNER_CAPABILITY_PUBLIC_FORBIDDEN_CATEGORIES = (
    "eval protocol changes",
    "success metric changes",
    "training budget changes",
    "seed policy changes",
    "checkpoint changes",
    "PPO config changes",
    "runner patch or source changes",
)


@dataclass(frozen=True)
class RunnerCapabilitySchemaDraft:
    capability_id: str
    kind: str
    output_root_key: str
    purpose: str
    allowed_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    train_only_scope: str
    properties: Mapping[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        schema = {
            "schema_version": f"true_rl_agent.runner_capability_schema.{self.capability_id}.v1",
            "capability_id": self.capability_id,
            "kind": self.kind,
            "status": "schema_preview_only",
            "provider_output_enabled": False,
            "candidate_writable_now": False,
            "output_root_key": self.output_root_key,
            "purpose": self.purpose,
            "allowed_fields": list(self.allowed_fields),
            "required_fields": list(self.required_fields),
            "train_only_scope": self.train_only_scope,
            "properties": dict(self.properties),
            "forbidden_field_categories": list(RUNNER_CAPABILITY_PUBLIC_FORBIDDEN_CATEGORIES),
            "runner_owned_invariants": [
                "public eval protocol is fixed",
                "success metric is fixed",
                "training budget is fixed",
                "seed policy is fixed",
                "checkpoint policy is fixed",
                "PPO configuration is fixed",
                "runner source is not provider-visible",
            ],
        }
        assert_public_only_prompt_payload(schema)
        return schema


RUNNER_CAPABILITY_SCHEMA_DRAFTS: dict[str, RunnerCapabilitySchemaDraft] = {
    CURRICULUM_DESIGN_CAPABILITY_ID: RunnerCapabilitySchemaDraft(
        capability_id=CURRICULUM_DESIGN_CAPABILITY_ID,
        kind="train_protocol_design",
        output_root_key="train_only_curriculum",
        purpose="Preview a future train-only curriculum design surface without changing runner or eval authority.",
        allowed_fields=("capability_id", "rationale", "train_only_curriculum"),
        required_fields=("capability_id", "rationale", "train_only_curriculum"),
        train_only_scope="training distribution schedule only; final distribution remains runner-owned target distribution",
        properties={
            "train_only_curriculum": {
                "type": "object",
                "required": ["stages", "final_distribution_policy"],
                "fields": {
                    "stages": {
                        "type": "array",
                        "item_fields": {
                            "name": "short public stage name",
                            "progress_range": "[start, end] fractions within [0, 1]",
                            "train_distribution_adjustment": "bounded train-only distribution intent",
                            "rationale": "public mechanism rationale",
                        },
                    },
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            }
        },
    ),
    TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID: RunnerCapabilitySchemaDraft(
        capability_id=TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
        kind="train_distribution_design",
        output_root_key="train_only_environment",
        purpose="Preview a future train-only environment variation design surface without exposing runner internals.",
        allowed_fields=("capability_id", "rationale", "train_only_environment"),
        required_fields=("capability_id", "rationale", "train_only_environment"),
        train_only_scope="training environment distribution only; public eval environment remains fixed",
        properties={
            "train_only_environment": {
                "type": "object",
                "required": ["variations", "final_distribution_policy"],
                "fields": {
                    "variations": {
                        "type": "array",
                        "item_fields": {
                            "name": "short public variation name",
                            "kind": "bounded train-only variation kind",
                            "range": "public bounded numeric range or enum values",
                            "rationale": "public mechanism rationale",
                        },
                    },
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            }
        },
    ),
}


def runner_capability_schema_by_id(capability_id: str) -> dict[str, Any]:
    try:
        schema = RUNNER_CAPABILITY_SCHEMA_DRAFTS[capability_id].to_public_dict()
    except KeyError as exc:
        raise KeyError(f"unknown runner capability schema: {capability_id}") from exc
    assert_public_only_prompt_payload(schema)
    return schema


def runner_capability_schema_ids() -> tuple[str, ...]:
    return tuple(RUNNER_CAPABILITY_SCHEMA_DRAFTS)


def runner_capability_readiness_gate(
    capability_id: str,
    readiness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _ = runner_capability_schema_by_id(capability_id)
    values = readiness if isinstance(readiness, Mapping) else {}
    requirements = {
        key: values.get(key) is True
        for key in RUNNER_CAPABILITY_READINESS_REQUIREMENTS
    }
    missing = [key for key in RUNNER_CAPABILITY_READINESS_REQUIREMENTS if not requirements[key]]
    gate = {
        "status": "ready" if not missing else "not_ready",
        "requirements": requirements,
        "missing_requirements": missing,
    }
    assert_public_only_prompt_payload(gate)
    return gate


def validate_runner_capability_payload(capability_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = runner_capability_schema_by_id(capability_id)
    if not isinstance(payload, Mapping):
        raise TypeError("runner capability payload must be a mapping")
    _reject_if_public_sanitizer_would_change(payload)
    _reject_forbidden_runner_fields(payload, path="$")
    _reject_private_or_path_values(payload, path="$")
    clean = sanitize_prompt_public_context(dict(payload))
    assert_public_only_prompt_payload(clean)
    if clean.get("capability_id") != capability_id:
        raise ValueError(f"runner capability payload capability_id must be {capability_id}")
    allowed_fields = set(schema["allowed_fields"])
    extra = sorted(set(clean) - allowed_fields)
    if extra:
        raise ValueError(f"unsupported runner capability field(s): {extra}")
    missing = [field for field in schema["required_fields"] if field not in clean]
    if missing:
        raise ValueError(f"missing runner capability field(s): {missing}")
    _validate_schema_shape(schema, clean)
    return clean


def _reject_if_public_sanitizer_would_change(value: Any) -> None:
    normalized = _normalize_public_value(value)
    sanitized = sanitize_prompt_public_context(value)
    if sanitized != normalized:
        raise ValueError("forbidden runner capability public scrub would change payload")


def _normalize_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_public_value(item) for item in value]
    return value


def _reject_forbidden_runner_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in RUNNER_CAPABILITY_FORBIDDEN_FIELD_PARTS):
                raise ValueError(f"forbidden runner capability field at {path}.{key}")
            _reject_forbidden_runner_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_runner_fields(item, path=f"{path}[{index}]")


def _reject_private_or_path_values(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_private_or_path_values(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_or_path_values(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if contains_forbidden_public_value_marker(value) or sanitize_prompt_public_value(value) != value:
            raise ValueError(f"forbidden runner capability value at {path}")


def _validate_schema_shape(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    root_key = schema.get("output_root_key")
    if not isinstance(root_key, str):
        raise ValueError("runner capability schema is missing output_root_key")
    root = payload.get(root_key)
    root_schema = _schema_mapping(schema.get("properties"), path="properties").get(root_key)
    if not isinstance(root_schema, Mapping):
        raise ValueError(f"runner capability schema for {root_key} is malformed")
    _validate_object_schema(root_schema, root, path=root_key)


def _validate_object_schema(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    if schema.get("type") != "object":
        raise ValueError(f"runner capability schema {path} must declare an object")
    if not isinstance(value, Mapping):
        raise ValueError(f"runner capability schema field {path} must be an object")
    fields = _schema_mapping(schema.get("fields"), path=f"{path}.fields")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError(f"runner capability schema {path}.required must be a list of strings")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"runner capability schema field {path} missing required field(s): {missing}")
    extra = sorted(str(key) for key in value if str(key) not in fields)
    if extra:
        raise ValueError(f"runner capability schema field {path} has unsupported field(s): {extra}")
    for field_name, field_schema in fields.items():
        if field_name not in value:
            continue
        if isinstance(field_schema, Mapping):
            _validate_nested_schema_field(field_schema, value[field_name], path=f"{path}.{field_name}")
        elif isinstance(field_schema, str) and value[field_name] != field_schema:
            raise ValueError(
                f"runner capability schema field {path}.{field_name} must equal {field_schema}"
            )


def _validate_nested_schema_field(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"runner capability schema field {path} must be an array")
        item_fields = _schema_mapping(schema.get("item_fields"), path=f"{path}.item_fields")
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, Mapping):
                raise ValueError(f"runner capability schema field {item_path} must be an object")
            extra = sorted(str(key) for key in item if str(key) not in item_fields)
            if extra:
                raise ValueError(
                    f"runner capability schema field {item_path} has unsupported field(s): {extra}"
                )
    elif schema_type == "object":
        _validate_object_schema(schema, value, path=path)


def _schema_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"runner capability schema {path} must be a mapping")
    return {str(key): item for key, item in value.items()}

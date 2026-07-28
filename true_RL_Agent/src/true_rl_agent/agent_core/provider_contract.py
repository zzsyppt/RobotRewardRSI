"""Provider-facing candidate proposal contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from true_rl_agent.candidates.package_contract import (
    PROVIDER_AUTHORED_CANDIDATE_FILES,
)

PROMPT_ADVERTISED_DECLARED_COMPONENTS = (
    "reward_callable",
    "diagnostic",
    "reward_weight",
)
DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES = PROMPT_ADVERTISED_DECLARED_COMPONENTS
AUTHORING_RESPONSE_MODES = ("candidate_draft", "source_request")
DEFAULT_AUTHORING_RESPONSE_MODE = "candidate_draft"
SOURCE_REQUEST_SERVICE_ENABLED = True
PROVIDER_JSON_OUTPUT_REQUIRED_KEYS = (
    "candidate_id",
    "rationale",
    "declared_changed_components",
    "components_py",
)
PROVIDER_JSON_OUTPUT_ALLOWED_KEYS = ("response_mode",) + PROVIDER_JSON_OUTPUT_REQUIRED_KEYS
SOURCE_REQUEST_OUTPUT_REQUIRED_KEYS = (
    "response_mode",
    "logical_ids",
    "rationale",
)
SOURCE_REQUEST_OUTPUT_ALLOWED_KEYS = SOURCE_REQUEST_OUTPUT_REQUIRED_KEYS
PROVIDER_JSON_OUTPUT_PROHIBITED_KEYS = (
    "runner_patch",
    "eval_patch",
    "success_metric",
    "budget",
    "seed_policy",
    "checkpoint",
    "ppo_config",
    "operation_diff",
    "diagnostics",
    "candidate_manifest",
    "files",
)
PROVIDER_MATERIALIZED_FILES = PROVIDER_AUTHORED_CANDIDATE_FILES


def provider_json_output_contract(
    *,
    advertised_declared_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    declared_components = _advertised_declared_components(
        advertised_declared_components
    )
    candidate_draft_contract = {
        "response_mode": "candidate_draft",
        "required_keys": list(PROVIDER_JSON_OUTPUT_REQUIRED_KEYS),
        "allowed_keys": list(PROVIDER_JSON_OUTPUT_ALLOWED_KEYS),
        "additional_keys_allowed": False,
    }
    source_request_contract = {
        "response_mode": "source_request",
        "enabled": SOURCE_REQUEST_SERVICE_ENABLED,
        "required_keys": list(SOURCE_REQUEST_OUTPUT_REQUIRED_KEYS),
        "allowed_keys": list(SOURCE_REQUEST_OUTPUT_ALLOWED_KEYS),
        "additional_keys_allowed": False,
        "purpose": "controlled source resolver request for prompt source_catalog logical ids",
    }
    return {
        "type": "object",
        "json_only": True,
        "no_prose_outside_json": True,
        "response_shape": "one_of",
        "one_of": {
            "candidate_draft": candidate_draft_contract,
            "source_request": source_request_contract,
        },
        "additional_keys_allowed": False,
        "prohibited_keys": list(PROVIDER_JSON_OUTPUT_PROHIBITED_KEYS),
        "authoring_response_protocol": {
            "schema_version": "true_rl_agent.authoring_provider_response_contract.v1",
            "response_mode_field": "response_mode",
            "default_response_mode": DEFAULT_AUTHORING_RESPONSE_MODE,
            "supported_response_modes": list(AUTHORING_RESPONSE_MODES),
            "source_request_service_enabled": SOURCE_REQUEST_SERVICE_ENABLED,
            "instructions": [
                "Omit response_mode or set response_mode='candidate_draft' when returning candidate fields.",
                "candidate_draft and source_request fields are mutually exclusive response shapes.",
                "Use source_request only for logical ids declared in the prompt source_catalog.",
                "After a source_followup prompt, return candidate_draft unless another bounded source read is necessary.",
            ],
        },
        "declared_changed_components_allowed_values": list(declared_components),
        "field_contracts": {
            "response_mode": {
                "type": "string",
                "optional": True,
                "default": DEFAULT_AUTHORING_RESPONSE_MODE,
                "allowed_values": list(AUTHORING_RESPONSE_MODES),
                "phase7_note": "source_request is serviced only through the prompt source_catalog allowlist",
            },
            "candidate_id": {
                "type": "string",
                "format": "short identifier; no path separators",
                "materialized_file": "candidate_manifest.yaml",
            },
            "rationale": {
                "type": "string",
                "purpose": "public explanation of the training-side mechanism",
                "materialized_file": "rationale.md",
            },
            "declared_changed_components": {
                "type": "array[string]",
                "allowed_values": list(declared_components),
                "purpose": "provider declaration; verifier-derived change set remains authoritative",
                "materialized_file": "operation_diff.json",
            },
            "components_py": {
                "type": "string",
                "purpose": "complete training_code/components.py source",
                "materialized_file": "training_code/components.py",
            },
        },
    }


def _advertised_declared_components(
    values: Sequence[str] | None,
) -> tuple[str, ...]:
    if values is None:
        return DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError("advertised declared components must be a sequence")
    selected = tuple(values)
    if any(not isinstance(item, str) for item in selected):
        raise TypeError("advertised declared components must contain strings")
    if len(set(selected)) != len(selected):
        raise ValueError("advertised declared components must be unique")
    unknown = sorted(
        set(selected) - set(DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES)
    )
    if unknown:
        raise ValueError(
            "advertised declared components contain unsupported values: "
            + ", ".join(unknown)
        )
    return tuple(
        item
        for item in DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES
        if item in selected
    )


def candidate_package_files_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.candidate_package_files.v1",
        "materialized_files": list(PROVIDER_MATERIALIZED_FILES),
        "llm_writable_provider_json_fields": list(PROVIDER_JSON_OUTPUT_ALLOWED_KEYS),
        "provider_materialized_files": {
            "candidate_manifest.yaml": {
                "source": "provider",
                "purpose": "fixed schema, candidate_id, and controlled entrypoint",
            },
            "operation_diff.json": {
                "source": "candidate_materializer",
                "derived_from": ["declared_changed_components"],
                "authority": "provider declaration is provisional; verifier-derived change set remains authoritative",
                "purpose": "public record of declared changed components; not an LLM-editable runner diff",
            },
            "diagnostics.json": {
                "source": "candidate_materializer",
                "derived_from": ["provider_response_parse", "candidate_materialization_status"],
                "purpose": "public candidate-package diagnostics placeholder for future extensions",
            },
        },
        "llm_must_not_output_files": [
            "runner patches",
            "eval patches",
            "budget or seed files",
            "checkpoint files",
            "PPO config files",
        ],
    }


def validate_provider_json_output_keys(proposal: Any) -> None:
    if not isinstance(proposal, dict):
        return
    unsupported = [str(key) for key in proposal if str(key) not in PROVIDER_JSON_OUTPUT_ALLOWED_KEYS]
    if not unsupported:
        return
    safe_names = [
        key if key in PROVIDER_JSON_OUTPUT_PROHIBITED_KEYS else "[unsupported_key]"
        for key in unsupported[:5]
    ]
    raise ValueError("unsupported provider output key(s): " + ", ".join(safe_names))


def provider_authoring_response_mode(proposal: Mapping[str, Any]) -> str:
    mode = proposal.get("response_mode", DEFAULT_AUTHORING_RESPONSE_MODE)
    if not isinstance(mode, str):
        raise ValueError("response_mode must be a string when provided")
    if mode not in AUTHORING_RESPONSE_MODES:
        raise ValueError("response_mode must be candidate_draft or source_request")
    return mode


def validate_source_request_json_output_keys(proposal: Any) -> None:
    if not isinstance(proposal, dict):
        return
    unsupported = [str(key) for key in proposal if str(key) not in SOURCE_REQUEST_OUTPUT_ALLOWED_KEYS]
    if unsupported:
        safe_names = [
            key if key in PROVIDER_JSON_OUTPUT_PROHIBITED_KEYS else "[unsupported_key]"
            for key in unsupported[:5]
        ]
        raise ValueError("unsupported source_request output key(s): " + ", ".join(safe_names))
    missing = [key for key in SOURCE_REQUEST_OUTPUT_REQUIRED_KEYS if key not in proposal]
    if missing:
        raise ValueError("source_request missing required key(s): " + ", ".join(missing))


def validate_declared_changed_components(value: Any) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("declared_changed_components must be a JSON array of strings")
    changed = frozenset(value)
    unsupported = sorted(changed - frozenset(DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES))
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"declared_changed_components contains unsupported values: {joined}")
    return changed

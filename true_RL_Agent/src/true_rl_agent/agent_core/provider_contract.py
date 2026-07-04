"""Provider-facing candidate proposal contract."""

from __future__ import annotations

from typing import Any


PROMPT_ADVERTISED_DECLARED_COMPONENTS = (
    "reward_callable",
    "diagnostic",
    "reward_weight",
)
LEGACY_PARSE_DECLARED_COMPONENTS = (
    "reward_weight_scale",
    "schedule",
    "control_logic",
    "recovery_logic",
    "failure_classifier",
    "logging",
    "public_projection",
)
MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES = (
    PROMPT_ADVERTISED_DECLARED_COMPONENTS + LEGACY_PARSE_DECLARED_COMPONENTS
)
DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES = MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES
PROVIDER_JSON_OUTPUT_REQUIRED_KEYS = (
    "candidate_id",
    "rationale",
    "declared_changed_components",
    "components_py",
)
PROVIDER_JSON_OUTPUT_ALLOWED_KEYS = PROVIDER_JSON_OUTPUT_REQUIRED_KEYS
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
PROVIDER_MATERIALIZED_FILES = (
    "candidate_manifest.yaml",
    "training_code/components.py",
    "rationale.md",
    "operation_diff.json",
    "diagnostics.json",
)


def provider_json_output_contract() -> dict[str, Any]:
    return {
        "type": "object",
        "json_only": True,
        "no_prose_outside_json": True,
        "required_keys": list(PROVIDER_JSON_OUTPUT_REQUIRED_KEYS),
        "allowed_keys": list(PROVIDER_JSON_OUTPUT_ALLOWED_KEYS),
        "additional_keys_allowed": False,
        "prohibited_keys": list(PROVIDER_JSON_OUTPUT_PROHIBITED_KEYS),
        "declared_changed_components_allowed_values": list(PROMPT_ADVERTISED_DECLARED_COMPONENTS),
        "field_contracts": {
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
                "allowed_values": list(PROMPT_ADVERTISED_DECLARED_COMPONENTS),
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
                "source": "provider",
                "purpose": "public record of declared changed components; not an LLM-editable runner diff",
            },
            "diagnostics.json": {
                "source": "provider",
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

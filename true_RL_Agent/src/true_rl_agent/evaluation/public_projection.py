"""Task-neutral public projection guards for eval and runner outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.agent_core.public_safety import (
    contains_public_path_marker,
    sanitize_prompt_public_value,
)

_FORBIDDEN_PUBLIC_PROJECTION_KEY_PARTS = (
    "private",
    "hidden",
    "secret",
    "api_key",
    "token",
    "trusted_eval",
    "trusted_seed",
    "checkpoint_selector",
    "raw_seed",
    "local_path",
    "source_latest_checkpoint",
    "checkpoint_output",
    "checkpoint_output_path",
    "summary_path",
    "work_dir",
    "config_path",
    "cwd",
    "per_episode",
    "log_dir",
    "host",
)
_FORBIDDEN_PUBLIC_VALUE_MARKERS = (
    "private_seed",
    "private seed",
    "hidden_eval",
    "hidden eval",
    "iid_hidden",
    "trusted_eval",
    "trusted eval",
    "trusted_seed",
    "trusted seed",
    "checkpoint_selector",
    "checkpoint selector",
    "api_key",
    "api key",
    "secret token",
    "access token",
)
def assert_public_projection_safe(payload: Any) -> None:
    """Reject projected public eval/runner payloads that still contain raw evidence."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(part in lowered for part in _FORBIDDEN_PUBLIC_PROJECTION_KEY_PARTS):
                    raise ValueError(f"public projection contains forbidden key: {path}.{key_text}")
                visit(item, f"{path}.{key_text}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            if _contains_raw_path(value):
                raise ValueError(f"public projection contains raw path at: {path}")
            if _contains_forbidden_value_marker(value):
                raise ValueError(f"public projection contains forbidden value at: {path}")

    visit(payload, "$")


def project_public_train_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project raw runner train summary into public-safe ledger evidence."""

    public: dict[str, Any] = {}
    for key in ("schema_version", "status", "formal_training_executed", "dry_run", "budget_steps"):
        if key in summary:
            public[key] = _public_value(summary[key])

    training = summary.get("training")
    if isinstance(training, Mapping):
        projected_training = _project_training_summary(training)
        if projected_training:
            public["training"] = projected_training

    candidate = summary.get("candidate")
    if isinstance(candidate, Mapping):
        projected_candidate = _project_candidate_summary(candidate)
        if projected_candidate:
            public["candidate"] = projected_candidate

    errors = summary.get("errors")
    if isinstance(errors, list | tuple):
        public["errors"] = _public_value(errors)

    assert_public_projection_safe(public)
    return public


def project_public_runner_error(exc: BaseException) -> dict[str, Any]:
    """Return a public-safe runner error summary."""

    projected = {
        "type": exc.__class__.__name__,
        "message": sanitize_prompt_public_value(str(exc)),
    }
    public_details = getattr(exc, "public_details", None)
    if isinstance(public_details, Mapping):
        projected["details"] = _public_value(public_details)
    assert_public_projection_safe(projected)
    return projected


def _project_training_summary(training: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key in (
        "budget_steps",
        "num_envs",
        "num_steps_per_env",
        "max_iterations",
        "planned_env_steps",
        "config_derived_total_timesteps",
        "latest_checkpoint_iteration",
    ):
        if key in training:
            projected[key] = _public_value(training[key])
    budget_accounting = training.get("budget_accounting")
    if isinstance(budget_accounting, Mapping):
        projected["budget_accounting"] = _public_value(budget_accounting)
    return projected


def _project_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key in ("schema_version", "enabled"):
        if key in candidate:
            projected[key] = _public_value(candidate[key])
    manifest = candidate.get("candidate")
    if isinstance(manifest, Mapping):
        projected["candidate"] = _public_value(manifest)
    components = candidate.get("components")
    if isinstance(components, Mapping):
        component_projection: dict[str, Any] = {}
        for key in ("name", "effective_hook_counts"):
            if key in components:
                component_projection[key] = _public_value(components[key])
        if component_projection:
            projected["components"] = component_projection
    return projected


def _public_value(value: Any) -> Any:
    return sanitize_prompt_public_value(value)


def _contains_forbidden_value_marker(value: str) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in _FORBIDDEN_PUBLIC_VALUE_MARKERS):
        return True
    if "raw_seed" in lowered and "no_raw_seed" not in lowered:
        return True
    if "raw-seed" in lowered:
        return True
    if "raw seed id" in lowered:
        return True
    return False


def _contains_raw_path(value: str) -> bool:
    return contains_public_path_marker(value)

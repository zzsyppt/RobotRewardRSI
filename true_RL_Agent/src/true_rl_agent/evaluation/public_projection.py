"""Task-neutral public projection guards for eval and runner outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.agent_core.public_safety import (
    contains_forbidden_public_key_marker,
    contains_forbidden_public_value_marker,
    contains_public_path_marker,
    public_key_tokens,
    redact_secret_key_markers,
    sanitize_prompt_public_value,
    sanitize_untrusted_public_diagnostic,
)

_FORBIDDEN_PUBLIC_PROJECTION_KEY_SEQUENCES = (
    ("hostname",),
    ("local", "path"),
    ("source", "latest", "checkpoint"),
    ("checkpoint", "output"),
    ("summary", "path"),
    ("work", "dir"),
    ("config", "path"),
    ("per", "episode"),
    ("log", "dir"),
)


def _contains_forbidden_projection_key(value: Any) -> bool:
    if contains_forbidden_public_key_marker(value):
        return True
    tokens = public_key_tokens(value)
    if tokens in {("cwd",), ("host",)} or "host" in tokens:
        return True
    return any(
        tokens[index : index + len(sequence)] == sequence
        for sequence in _FORBIDDEN_PUBLIC_PROJECTION_KEY_SEQUENCES
        for index in range(len(tokens) - len(sequence) + 1)
    )


def assert_public_projection_safe(payload: Any) -> None:
    """Reject projected public eval/runner payloads that still contain raw evidence."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if _contains_forbidden_projection_key(key_text):
                    raise ValueError(f"public projection contains forbidden key: {path}.{key_text}")
                visit(item, f"{path}.{key_text}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            if _contains_raw_path(value):
                raise ValueError(f"public projection contains raw path at: {path}")
            if contains_forbidden_public_value_marker(value):
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
        public["errors"] = _public_diagnostic_value(errors)

    assert_public_projection_safe(public)
    return public


def project_public_runner_error(exc: BaseException) -> dict[str, Any]:
    """Return a public-safe runner error summary."""

    projected = {
        "type": exc.__class__.__name__,
        "message": _public_diagnostic_value(str(exc)),
    }
    public_details = getattr(exc, "public_details", None)
    if isinstance(public_details, Mapping):
        projected["details"] = _public_diagnostic_value(public_details)
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
    if "numeric_evidence" in training:
        projected["numeric_evidence"] = project_public_training_sampling(training["numeric_evidence"])
    return projected


def project_public_training_sampling(value: Any) -> dict[str, Any]:
    """Bound only sampling coverage; task term meanings remain opaque references.

    No archive I/O is needed during ledger replay. ``complete`` describes the
    published attachment; each window separately says what was actually sampled.
    The same allowlist handles the raw receipt and its public projection.
    """
    if not isinstance(value, Mapping) or not value:
        return {"status": "not_available"}
    status = value.get("status")
    if status not in {"complete", "not_enabled", "not_started", "incomplete", "not_available"}:
        return {"status": "not_available"}
    if status != "complete":
        return {"status": status}
    coverage = value.get("coverage", value)
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get("windows"), list | tuple):
        return {"status": "incomplete"}
    result: dict[str, Any] = {
        "status": "complete",
        "scope": "fixed_windows_and_environment_subset_not_whole_training_distribution",
    }
    for key in ("planned_control_steps", "observed_control_steps", "num_envs", "steps_per_iteration", "source_step_offset_excluded"):
        item = coverage.get(key)
        if type(item) is int and item >= 0:
            result[key] = item
    if coverage.get("training_status") in {"pass", "fail"}:
        result["training_status"] = coverage["training_status"]
    env_ids = coverage.get("sampled_env_ids")
    sampled_count = len(env_ids) if isinstance(env_ids, list | tuple) else coverage.get("sampled_env_count")
    if type(sampled_count) is int and 0 <= sampled_count <= 16:
        result["sampled_env_count"] = sampled_count
    schema = coverage.get("schema_version", coverage.get("capture_schema_version"))
    if schema in {"true_rl_agent.training_numeric_capture.v1", "true_rl_agent.training_numeric_capture.v2"}:
        result["capture_schema_version"] = schema
    # Refer to the original task-owned term contract without copying term arrays,
    # source paths, or interpreting rate/step-reward meanings in shared code.
    from true_rl_agent.evaluation.binding import canonical_json_hash, is_sha256_digest
    semantics = coverage.get("semantics")
    semantics_hash = canonical_json_hash(semantics) if isinstance(semantics, Mapping) else coverage.get("term_semantics_hash")
    if is_sha256_digest(semantics_hash):
        result["term_semantics_hash"] = semantics_hash
    reference = value.get("reference")
    archive_hash = reference.get("manifest_sha256") if isinstance(reference, Mapping) else value.get("archive_manifest_sha256")
    if is_sha256_digest(archive_hash):
        result["archive_manifest_sha256"] = archive_hash
    windows = {w.get("name"): w for w in coverage["windows"] if isinstance(w, Mapping)}
    result["windows"] = []
    for name in ("early", "middle", "late"):
        raw = windows.get(name, {})
        window_status = raw.get("status")
        item = {"name": name, "status": window_status if window_status in {
            "sampled", "partial", "not_reached", "not_sampled", "not_scheduled", "not_available"
        } else "not_available"}
        for key in ("start_step", "end_step", "planned_sample_steps", "sampled_control_steps",
                    "sampled_env_transitions", "planned_sampled_env_transitions"):
            number = raw.get(key)
            if type(number) is int and number >= 0:
                item[key] = number
        result["windows"].append(item)
    return result


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
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_public_value(item) for item in value]
    if isinstance(value, str):
        sanitized = sanitize_prompt_public_value(value)
        return redact_secret_key_markers(sanitized)
    return value


def _public_diagnostic_value(value: Any) -> Any:
    return _public_value(sanitize_untrusted_public_diagnostic(value))


def _contains_raw_path(value: str) -> bool:
    return contains_public_path_marker(value)

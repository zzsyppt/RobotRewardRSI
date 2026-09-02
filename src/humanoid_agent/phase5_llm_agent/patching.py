"""Apply bounded Phase 5 LLM config operations."""

from __future__ import annotations

import copy
import fnmatch
import math
from typing import Any


class PatchContractError(ValueError):
    """Raised when an LLM patch violates the Phase 5 contract."""


def apply_llm_patch(
    *,
    phase1_config: dict[str, Any],
    phase2_config: dict[str, Any],
    response: dict[str, Any],
    phase5_config: dict[str, Any],
    reward_bounds: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return candidate configs after applying validated LLM operations."""

    candidate_phase1 = copy.deepcopy(phase1_config)
    candidate_phase2 = copy.deepcopy(phase2_config)
    operations = _extract_operations(response)
    contract = phase5_config["patch_contract"]
    max_operations = int(contract.get("max_operations", 16))
    if len(operations) > max_operations:
        raise PatchContractError(f"Too many patch operations: {len(operations)} > {max_operations}")

    applied: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        normalized = _normalize_operation(operation, index, contract)
        target = candidate_phase1 if normalized["document"] == "phase1" else candidate_phase2
        old_value = _get_path(target, normalized["path"])
        value = _validated_value(normalized, phase5_config, reward_bounds)
        _set_path(target, normalized["path"], value)
        applied.append(
            {
                "document": normalized["document"],
                "path": normalized["path"],
                "old": old_value,
                "new": value,
                "reason": normalized.get("reason"),
            }
        )
    return candidate_phase1, candidate_phase2, applied


def _extract_operations(response: dict[str, Any]) -> list[dict[str, Any]]:
    patch = response.get("patch")
    if isinstance(patch, dict) and isinstance(patch.get("operations"), list):
        operations = patch["operations"]
    elif isinstance(response.get("operations"), list):
        operations = response["operations"]
    else:
        raise PatchContractError("LLM response must contain patch.operations")
    if not all(isinstance(item, dict) for item in operations):
        raise PatchContractError("Every patch operation must be a JSON object")
    return operations


def _normalize_operation(operation: dict[str, Any], index: int, contract: dict[str, Any]) -> dict[str, Any]:
    op = str(operation.get("op", "set"))
    if op != "set":
        raise PatchContractError(f"Unsupported operation at index {index}: {op}")
    document = str(operation.get("document", operation.get("doc", "")))
    document = _normalize_document(document, contract)
    path = str(operation.get("path", ""))
    if not path:
        raise PatchContractError(f"Missing path at operation index {index}")
    allowed = contract.get("allowed_paths", {}).get(document, [])
    if not _path_allowed(path, allowed):
        raise PatchContractError(f"Patch path is not allowed: {document}.{path}")
    reason = operation.get("reason")
    if contract.get("require_operation_reason", True) and not isinstance(reason, str):
        raise PatchContractError(f"Missing operation reason at index {index}")
    if "value" not in operation:
        raise PatchContractError(f"Missing value at operation index {index}")
    return {
        "op": op,
        "document": document,
        "path": path,
        "value": operation["value"],
        "reason": reason,
    }


def _normalize_document(document: str, contract: dict[str, Any]) -> str:
    documents = contract.get("documents", {})
    if document in documents:
        return document
    normalized = document.replace("\\", "/")
    for name, path in documents.items():
        if normalized == path:
            return name
    raise PatchContractError(f"Unknown patch document: {document}")


def _path_allowed(path: str, allowed_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in allowed_patterns)


def _validated_value(
    operation: dict[str, Any],
    phase5_config: dict[str, Any],
    reward_bounds: dict[str, Any] | None,
) -> Any:
    document = operation["document"]
    path = operation["path"]
    value = operation["value"]
    if _contains_nonfinite(value):
        raise PatchContractError(f"Patch value is not finite: {document}.{path}")
    if document == "phase1" and path.startswith("profiles.phase2_train.push."):
        return _validate_push_value(path, value, phase5_config)
    if document == "phase2" and path.startswith("reward_weights."):
        return _validate_reward_weight(path, value, reward_bounds or {})
    if document == "phase2" and path.startswith("ppo."):
        if isinstance(value, bool) or value is None:
            return value
        if not isinstance(value, (int, float)):
            raise PatchContractError(f"PPO value must be numeric/bool/null: {path}")
        return float(value)
    if document == "phase2" and path.startswith("reward_params."):
        return value
    if document == "phase2" and path in {"baseline.name", "baseline.description"}:
        if not isinstance(value, str):
            raise PatchContractError(f"Baseline metadata must be strings: {path}")
        return value
    return value


def _validate_push_value(path: str, value: Any, phase5_config: dict[str, Any]) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise PatchContractError(f"Push field must be a two-number list: {path}")
    parsed = [float(value[0]), float(value[1])]
    if parsed[0] > parsed[1]:
        raise PatchContractError(f"Push range must be ordered: {path}")
    if path.endswith("interval_range_s"):
        low, high = [float(x) for x in phase5_config["patch_contract"].get("push_interval_s_bounds", [0.5, 4.0])]
        if not (low <= parsed[0] <= parsed[1] <= high):
            raise PatchContractError(f"Push interval out of bounds [{low}, {high}]: {path}")
        return parsed
    max_abs = float(phase5_config["patch_contract"].get("push_velocity_abs_max_mps", 0.6))
    if max(abs(parsed[0]), abs(parsed[1])) > max_abs:
        raise PatchContractError(f"Push velocity exceeds ±{max_abs} m/s: {path}")
    return parsed


def _validate_reward_weight(path: str, value: Any, reward_bounds: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatchContractError(f"Reward weight must be numeric: {path}")
    parsed = float(value)
    name = path.split(".", 1)[1]
    bounds = reward_bounds.get(name)
    if bounds is not None:
        low, high = float(bounds[0]), float(bounds[1])
        if not (low <= parsed <= high):
            raise PatchContractError(f"Reward weight {name} out of bounds [{low}, {high}]")
    return parsed


def _get_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return copy.deepcopy(current)


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = document
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise PatchContractError(f"Cannot descend into non-mapping at: {path}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise PatchContractError(f"Cannot set value under non-mapping: {path}")
    current[parts[-1]] = copy.deepcopy(value)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, int):
        return False
    if isinstance(value, str):
        return False
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False

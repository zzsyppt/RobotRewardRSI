"""Configuration helpers for Phase 1 push-recovery tooling.

The evaluator runs inside the Isaac/Unitree environment on the server, but the
configuration merge and hashing logic is intentionally pure Python so it can be
tested without Isaac imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    PyYAML is available in the IsaacLab environment. Keeping the import local
    makes pure module import usable in minimal environments.
    """

    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_for_profile(config: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    """Apply a named profile from ``profiles`` to the base config."""

    if not profile_name:
        return copy.deepcopy(config)
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"Unknown profile '{profile_name}'. Available profiles: {available}")
    merged = deep_merge(config, profiles[profile_name])
    merged["active_profile"] = profile_name
    return merged


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


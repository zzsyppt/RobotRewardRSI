"""Canonical execution identity for the Isaac Lab + RSL-RL backend.

The manifest is deliberately narrow.  It binds the Python ABI, a fixed set of
distribution versions, and the five Python modules that implement the policy
optimization hot path.  Machine hardware and driver state are diagnostics, not
experiment identity, and therefore do not appear here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from true_rl_agent.backends.base import (
    BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import read_bounded_regular_file

ISAAC_LAB_RSL_RL_BACKEND_ID = "isaac_lab_rsl_rl"
ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_SCHEMA_VERSION = (
    "true_rl_agent.backend_execution_environment_manifest.isaac_lab_rsl_rl.v1"
)
_PACKAGE_DISTRIBUTIONS = (
    "isaaclab",
    "isaaclab-rl",
    "isaaclab-tasks",
    "isaacsim",
    "isaacsim-core",
    "isaacsim-extscache-physics",
    "rsl-rl-lib",
    "torch",
)
MAX_BACKEND_EXECUTION_MODULE_SOURCE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class _ModuleSourceSpec:
    logical_name: str
    import_name: str
    symbol_name: str
    top_level_package: str
    package_relative_path: str
    distribution: str

    @property
    def distribution_relative_path(self) -> str:
        return f"{self.top_level_package}/{self.package_relative_path}"


_MODULE_SOURCE_SPECS = (
    _ModuleSourceSpec(
        logical_name="OnPolicyRunner",
        import_name="rsl_rl.runners.on_policy_runner",
        symbol_name="OnPolicyRunner",
        top_level_package="rsl_rl",
        package_relative_path="runners/on_policy_runner.py",
        distribution="rsl-rl-lib",
    ),
    _ModuleSourceSpec(
        logical_name="PPO",
        import_name="rsl_rl.algorithms.ppo",
        symbol_name="PPO",
        top_level_package="rsl_rl",
        package_relative_path="algorithms/ppo.py",
        distribution="rsl-rl-lib",
    ),
    _ModuleSourceSpec(
        logical_name="ActorCritic",
        import_name="rsl_rl.modules.actor_critic",
        symbol_name="ActorCritic",
        top_level_package="rsl_rl",
        package_relative_path="modules/actor_critic.py",
        distribution="rsl-rl-lib",
    ),
    _ModuleSourceSpec(
        logical_name="RolloutStorage",
        import_name="rsl_rl.storage.rollout_storage",
        symbol_name="RolloutStorage",
        top_level_package="rsl_rl",
        package_relative_path="storage/rollout_storage.py",
        distribution="rsl-rl-lib",
    ),
    _ModuleSourceSpec(
        logical_name="RslRlVecEnvWrapper",
        import_name="isaaclab_rl.rsl_rl.vecenv_wrapper",
        symbol_name="RslRlVecEnvWrapper",
        top_level_package="isaaclab_rl",
        package_relative_path="rsl_rl/vecenv_wrapper.py",
        distribution="isaaclab-rl",
    ),
)


@dataclass(frozen=True)
class IsaacLabExecutionEnvironmentManifest:
    """Validated, canonical backend execution identity."""

    python: Mapping[str, str]
    packages: tuple[Mapping[str, str], ...]
    modules: tuple[Mapping[str, Any], ...]
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        python_identity = _validate_python_identity(self.python)
        packages = _validate_package_entries(self.packages)
        modules = _validate_module_entries(self.modules)
        object.__setattr__(self, "python", MappingProxyType(python_identity))
        object.__setattr__(
            self,
            "packages",
            tuple(MappingProxyType(item) for item in packages),
        )
        object.__setattr__(
            self,
            "modules",
            tuple(MappingProxyType(item) for item in modules),
        )
        canonical_hash = canonical_json_hash(self._hash_payload())
        if self.manifest_hash is not None:
            require_sha256_digest(
                self.manifest_hash,
                "backend execution environment manifest hash",
            )
            if self.manifest_hash != canonical_hash:
                raise ValueError("backend execution environment manifest hash mismatch")
        object.__setattr__(self, "manifest_hash", canonical_hash)

    @classmethod
    def from_mapping(cls, value: Any) -> "IsaacLabExecutionEnvironmentManifest":
        if not isinstance(value, Mapping):
            raise ValueError("backend execution environment manifest must be an object")
        expected_keys = {
            "schema_version",
            "backend_id",
            "python",
            "packages",
            "modules",
            "manifest_hash",
        }
        if set(value) != expected_keys:
            raise ValueError("backend execution environment manifest fields mismatch")
        if (
            value.get("schema_version")
            != ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "backend execution environment manifest schema_version mismatch"
            )
        if value.get("backend_id") != ISAAC_LAB_RSL_RL_BACKEND_ID:
            raise ValueError(
                "backend execution environment manifest backend_id mismatch"
            )
        raw_packages = value.get("packages")
        raw_modules = value.get("modules")
        if not isinstance(raw_packages, list | tuple):
            raise ValueError("backend execution environment packages must be an array")
        if not isinstance(raw_modules, list | tuple):
            raise ValueError("backend execution environment modules must be an array")
        return cls(
            python=_require_mapping(value.get("python"), "python identity"),
            packages=tuple(
                _require_mapping(item, "package identity") for item in raw_packages
            ),
            modules=tuple(
                _require_mapping(item, "module source identity") for item in raw_modules
            ),
            manifest_hash=require_sha256_digest(
                value.get("manifest_hash"),
                "backend execution environment manifest hash",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.manifest_hash is not None
        return {
            **self._hash_payload(),
            "manifest_hash": self.manifest_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_SCHEMA_VERSION),
            "backend_id": ISAAC_LAB_RSL_RL_BACKEND_ID,
            "python": dict(self.python),
            "packages": [dict(item) for item in self.packages],
            "modules": [dict(item) for item in self.modules],
        }


def build_isaac_lab_execution_environment_manifest_from_components(
    *,
    python_identity: Mapping[str, str],
    package_versions: Mapping[str, str],
    module_sources: Mapping[str, bytes],
) -> IsaacLabExecutionEnvironmentManifest:
    """Build the canonical manifest from explicit components.

    This is also the stable, dependency-free construction seam used by tests.
    The accepted package and module sets are fixed; callers cannot broaden the
    identity to an accidental ``pip freeze`` or source-tree hash.
    """

    if set(package_versions) != set(_PACKAGE_DISTRIBUTIONS):
        raise ValueError("backend execution environment package set mismatch")
    expected_module_names = {item.logical_name for item in _MODULE_SOURCE_SPECS}
    if set(module_sources) != expected_module_names:
        raise ValueError("backend execution environment module set mismatch")
    packages = tuple(
        {
            "distribution": distribution,
            "version": _nonempty_text(
                package_versions[distribution],
                f"{distribution} package version",
            ),
        }
        for distribution in _PACKAGE_DISTRIBUTIONS
    )
    modules: list[dict[str, Any]] = []
    for spec in _MODULE_SOURCE_SPECS:
        source = module_sources[spec.logical_name]
        if not isinstance(source, bytes):
            raise TypeError(f"{spec.logical_name} module source must be bytes")
        modules.append(
            {
                "logical_name": spec.logical_name,
                "import_name": spec.import_name,
                "symbol_name": spec.symbol_name,
                "distribution": spec.distribution,
                "distribution_relative_path": spec.distribution_relative_path,
                "source_size_bytes": len(source),
                "source_sha256": hashlib.sha256(source).hexdigest(),
            }
        )
    return IsaacLabExecutionEnvironmentManifest(
        python=python_identity,
        packages=packages,
        modules=tuple(modules),
    )


def build_current_isaac_lab_execution_environment_manifest() -> (
    IsaacLabExecutionEnvironmentManifest
):
    """Resolve the current interpreter and fixed backend hot-path sources."""

    python_identity = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": _nonempty_text(
            sys.implementation.cache_tag,
            "Python cache tag",
        ),
    }
    package_versions: dict[str, str] = {}
    for distribution in _PACKAGE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"backend execution environment package is unavailable: {distribution}"
            ) from exc
        package_versions[distribution] = version
    module_sources = {
        spec.logical_name: _read_installed_module_source(spec)
        for spec in _MODULE_SOURCE_SPECS
    }
    return build_isaac_lab_execution_environment_manifest_from_components(
        python_identity=python_identity,
        package_versions=package_versions,
        module_sources=module_sources,
    )


def verify_current_isaac_lab_execution_environment(
    expected_manifest_hash: str,
    *,
    current_manifest_builder: Callable[
        [], IsaacLabExecutionEnvironmentManifest
    ] = build_current_isaac_lab_execution_environment_manifest,
) -> dict[str, str]:
    """Recompute backend identity and return the bounded summary receipt."""

    expected = require_sha256_digest(
        expected_manifest_hash,
        "expected backend execution environment manifest hash",
    )
    current = current_manifest_builder()
    if not isinstance(current, IsaacLabExecutionEnvironmentManifest):
        raise TypeError("current backend execution environment builder is invalid")
    actual = str(current.manifest_hash)
    if actual != expected:
        raise ValueError(
            f"backend execution environment drift: expected={expected}:actual={actual}"
        )
    return {
        "schema_version": (BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION),
        "backend_id": ISAAC_LAB_RSL_RL_BACKEND_ID,
        "status": "pass",
        "expected_manifest_hash": expected,
        "actual_manifest_hash": actual,
    }


def _read_installed_module_source(spec: _ModuleSourceSpec) -> bytes:
    package_spec = importlib.util.find_spec(spec.top_level_package)
    locations = (
        tuple(package_spec.submodule_search_locations or ())
        if package_spec is not None
        else ()
    )
    if len(locations) != 1:
        raise ValueError(
            "backend execution module package location is unavailable or ambiguous: "
            f"{spec.top_level_package}"
        )
    package_root = Path(locations[0]).resolve(strict=True)
    source_path = package_root.joinpath(*spec.package_relative_path.split("/"))
    resolved_source = source_path.resolve(strict=True)
    try:
        resolved_source.relative_to(package_root)
    except ValueError as exc:
        raise ValueError(
            f"backend execution module source escaped its package: {spec.import_name}"
        ) from exc
    return read_bounded_regular_file(
        source_path,
        max_bytes=MAX_BACKEND_EXECUTION_MODULE_SOURCE_BYTES,
        label=f"backend execution module source {spec.import_name}",
    )


def _validate_python_identity(value: Mapping[str, str]) -> dict[str, str]:
    mapping = _require_mapping(value, "python identity")
    expected = {"implementation", "version", "cache_tag"}
    if set(mapping) != expected:
        raise ValueError("backend execution environment Python fields mismatch")
    return {
        key: _nonempty_text(mapping[key], f"Python {key}") for key in sorted(expected)
    }


def _validate_package_entries(
    values: tuple[Mapping[str, str], ...],
) -> tuple[dict[str, str], ...]:
    if not isinstance(values, tuple):
        raise TypeError("backend execution environment packages must be a tuple")
    entries: list[dict[str, str]] = []
    for value in values:
        mapping = _require_mapping(value, "package identity")
        if set(mapping) != {"distribution", "version"}:
            raise ValueError("backend execution environment package fields mismatch")
        entries.append(
            {
                "distribution": _nonempty_text(
                    mapping.get("distribution"),
                    "package distribution",
                ),
                "version": _nonempty_text(mapping.get("version"), "package version"),
            }
        )
    if tuple(item["distribution"] for item in entries) != _PACKAGE_DISTRIBUTIONS:
        raise ValueError("backend execution environment package order mismatch")
    return tuple(entries)


def _validate_module_entries(
    values: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, tuple):
        raise TypeError("backend execution environment modules must be a tuple")
    expected_fields = {
        "logical_name",
        "import_name",
        "symbol_name",
        "distribution",
        "distribution_relative_path",
        "source_size_bytes",
        "source_sha256",
    }
    entries: list[dict[str, Any]] = []
    if len(values) != len(_MODULE_SOURCE_SPECS):
        raise ValueError("backend execution environment module count mismatch")
    for value, spec in zip(values, _MODULE_SOURCE_SPECS, strict=True):
        mapping = _require_mapping(value, "module source identity")
        if set(mapping) != expected_fields:
            raise ValueError("backend execution environment module fields mismatch")
        fixed_values = {
            "logical_name": spec.logical_name,
            "import_name": spec.import_name,
            "symbol_name": spec.symbol_name,
            "distribution": spec.distribution,
            "distribution_relative_path": spec.distribution_relative_path,
        }
        for field_name, expected in fixed_values.items():
            if mapping.get(field_name) != expected:
                raise ValueError(
                    "backend execution environment module identity mismatch: "
                    f"{field_name}"
                )
        size = mapping.get("source_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("module source_size_bytes must be non-negative")
        entries.append(
            {
                **fixed_values,
                "source_size_bytes": size,
                "source_sha256": require_sha256_digest(
                    mapping.get("source_sha256"),
                    f"{spec.logical_name} source SHA-256",
                ),
            }
        )
    return tuple(entries)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


__all__ = [
    "BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION",
    "ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_SCHEMA_VERSION",
    "ISAAC_LAB_RSL_RL_BACKEND_ID",
    "MAX_BACKEND_EXECUTION_MODULE_SOURCE_BYTES",
    "IsaacLabExecutionEnvironmentManifest",
    "build_current_isaac_lab_execution_environment_manifest",
    "build_isaac_lab_execution_environment_manifest_from_components",
    "verify_current_isaac_lab_execution_environment",
]

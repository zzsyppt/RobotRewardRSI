"""Task-owned fixed training distribution for Stair scratch learning."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)

STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.execution_source_preflight.v1"
)
_STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION_V1 = (
    "true_rl_agent.task.stair_climbing.training_distribution.v1"
)
STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.training_distribution.v2"
)
_STAIR_TRAINING_DISTRIBUTION_EVIDENCE_SCHEMA_VERSION_V1 = (
    "true_rl_agent.task.stair_climbing.training_distribution_evidence.v1"
)
STAIR_TRAINING_DISTRIBUTION_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.training_distribution_evidence.v2"
)
STAIR_TRAINING_DISTRIBUTION_UPSTREAM_SOURCE_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.training_distribution_upstream_sources.v3"
)

_REQUIRED_UPSTREAM_SOURCES = (
    (
        "isaaclab",
        "terrains/terrain_importer.py",
        ("TerrainImporter.update_env_origins",),
    ),
    (
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/g1/agents/"
        "rsl_rl_ppo_cfg.py",
        ("G1RoughPPORunnerCfg",),
    ),
    (
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/g1/rough_env_cfg.py",
        ("G1RoughEnvCfg",),
    ),
    (
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/mdp/curriculums.py",
        ("terrain_levels_vel",),
    ),
    (
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/velocity_env_cfg.py",
        ("CurriculumCfg.terrain_levels",),
    ),
)

_TERRAIN_HEIGHT_SOURCE = (
    "base_v1.terrain.training_distribution.step_height_range_m"
)
_FORWARD_COMMAND_SOURCE = (
    "uniform_zero_to_base_v1.training.forward_command_mps"
)
_LATERAL_COMMAND_SOURCE = "base_v1.training.lateral_command_mps"
_YAW_COMMAND_SOURCE = "base_v1.training.yaw_command_radps"
_TERRAIN_CURRICULUM_TERM = (
    "isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums."
    "terrain_levels_vel"
)
STAIR_PREVIOUS_TRAINING_PROTOCOL_HASH = (
    "d08200cc7458773f464f4fe78122f26d188153895f6d7268c7b5d5e209bc009d"
)


def validate_stair_training_distribution_contract(
    value: Any,
    *,
    base_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(value, "Stair training distribution")
    schema_version = contract.get("schema_version")
    legacy = schema_version == _STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION_V1
    if not legacy and schema_version != STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError("Stair training distribution schema_version mismatch")
    _exact_fields(
        contract,
        ({
            "schema_version",
            "authority",
            "candidate_writable",
            "prompt_visible",
            "formal_and_pilot_identical",
            "design_basis",
            "selection_provenance",
            "upstream_reference",
            "terrain",
            "command",
        } if legacy else {
            "schema_version",
            "authority",
            "candidate_writable",
            "prompt_visible",
            "formal_reference_and_candidates_identical",
            "historical_pilot_distribution_identical",
            "design_basis",
            "selection_provenance",
            "upstream_reference",
            "terrain",
            "command",
        }),
        "Stair training distribution",
    )
    expected_flags = {
        "schema_version": (
            _STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION_V1
            if legacy
            else STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION
        ),
        "authority": "task_runner",
        "candidate_writable": False,
        "prompt_visible": False,
    }
    if legacy:
        expected_flags["formal_and_pilot_identical"] = True
    else:
        expected_flags.update(
            {
                "formal_reference_and_candidates_identical": True,
                "historical_pilot_distribution_identical": False,
            }
        )
    for field_name, expected in expected_flags.items():
        if contract[field_name] != expected:
            raise ValueError(
                f"Stair training distribution {field_name} mismatch"
            )
    expected_design_basis = (
        "upstream_g1_rough_curriculum_alignment_under_frozen_stair_v1_bounds"
        if legacy
        else "formal_eval_target_alignment_with_single_ood_frontier_bin"
    )
    if contract["design_basis"] != expected_design_basis:
        raise ValueError("Stair training distribution design basis mismatch")
    provenance = _mapping(
        contract["selection_provenance"],
        "Stair training distribution selection provenance",
    )
    expected_provenance = (
        {
            "kind": "upstream_alignment_not_failure_result",
            "invalid_pre_capture_pilot_results_used": False,
        }
        if legacy
        else {
            "kind": "operator_selected_formal_target_alignment",
            "previous_training_protocol_hash": STAIR_PREVIOUS_TRAINING_PROTOCOL_HASH,
            "historical_pilot_distribution_used_as_evidence": False,
            "formal_evaluation_frontier_retained_out_of_distribution": True,
        }
    )
    _exact_fields(
        provenance,
        set(expected_provenance),
        "Stair training distribution selection provenance",
    )
    if dict(provenance) != expected_provenance:
        raise ValueError("Stair training distribution selection provenance mismatch")
    _validate_upstream_reference(contract["upstream_reference"])

    terrain = _mapping(contract["terrain"], "Stair training terrain")
    _exact_fields(
        terrain,
        ({
            "generation_mode",
            "height_range_source",
            "runtime_level_curriculum_enabled",
            "initial_max_terrain_level",
            "level_update_term",
        } if legacy else {
            "generation_mode",
            "step_height_range_m",
            "runtime_level_curriculum_enabled",
            "initial_max_terrain_level",
            "level_update_term",
        }),
        "Stair training terrain",
    )
    if terrain["generation_mode"] != "difficulty_linear_by_row":
        raise ValueError("Stair training terrain generation mode mismatch")
    if legacy:
        if terrain["height_range_source"] != _TERRAIN_HEIGHT_SOURCE:
            raise ValueError("Stair training terrain height source mismatch")
    else:
        _validate_revised_height_range(
            terrain["step_height_range_m"],
            base_protocol=base_protocol,
        )
    if terrain["runtime_level_curriculum_enabled"] is not True:
        raise ValueError("Stair training terrain curriculum must be enabled")
    if terrain["level_update_term"] != _TERRAIN_CURRICULUM_TERM:
        raise ValueError("Stair training terrain curriculum term mismatch")
    initial_level = _nonnegative_int(
        terrain["initial_max_terrain_level"],
        "Stair initial max terrain level",
    )
    base = _mapping(base_protocol, "Stair base protocol")
    base_terrain = _mapping(base["terrain"], "Stair base terrain")
    base_distribution = _mapping(
        base_terrain["training_distribution"],
        "Stair base training distribution",
    )
    rows = _positive_int(
        base_distribution["terrain_rows"],
        "Stair base terrain rows",
    )
    if initial_level >= rows:
        raise ValueError("Stair initial max terrain level is outside the grid")

    command = _mapping(contract["command"], "Stair training command")
    _exact_fields(
        command,
        {
            "forward_sampling",
            "lateral_source",
            "yaw_source",
            "standing_env_fraction",
            "heading_command_enabled",
        },
        "Stair training command",
    )
    if command["forward_sampling"] != _FORWARD_COMMAND_SOURCE:
        raise ValueError("Stair training forward command source mismatch")
    if command["lateral_source"] != _LATERAL_COMMAND_SOURCE:
        raise ValueError("Stair training lateral command source mismatch")
    if command["yaw_source"] != _YAW_COMMAND_SOURCE:
        raise ValueError("Stair training yaw command source mismatch")
    standing_fraction = _finite_number(
        command["standing_env_fraction"],
        "Stair training standing fraction",
    )
    if not 0.0 <= standing_fraction < 1.0:
        raise ValueError("Stair training standing fraction must be in [0, 1)")
    if command["heading_command_enabled"] is not False:
        raise ValueError("Stair training heading command must remain disabled")

    resolved_stair_training_distribution(
        contract,
        base_protocol=base,
    )
    return _json_copy(contract)


def resolved_stair_training_distribution(
    contract: Mapping[str, Any],
    *,
    base_protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve source references without copying V1 values into the V2 artifact."""

    base = _mapping(base_protocol, "Stair base protocol")
    terrain = _mapping(base["terrain"], "Stair base terrain")
    base_distribution = _mapping(
        terrain["training_distribution"],
        "Stair base training distribution",
    )
    training = _mapping(base["training"], "Stair base training")
    terrain_contract = _mapping(contract["terrain"], "Stair training terrain")
    command_contract = _mapping(contract["command"], "Stair training command")
    upstream = _mapping(
        contract["upstream_reference"],
        "Stair training distribution upstream reference",
    )

    schema_version = contract.get("schema_version")
    legacy = schema_version == _STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION_V1
    if not legacy and schema_version != STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError("Stair training distribution schema_version mismatch")
    heights = (
        base_distribution["step_height_range_m"]
        if legacy
        else terrain_contract["step_height_range_m"]
    )
    if (
        not isinstance(heights, Sequence)
        or isinstance(heights, (str, bytes))
        or len(heights) != 2
        or any(not _is_finite_number(item) for item in heights)
        or float(heights[0]) <= 0.0
        or float(heights[0]) > float(heights[1])
    ):
        raise ValueError("Stair base training height range is invalid")
    forward = _finite_number(
        training["forward_command_mps"],
        "Stair base forward command",
    )
    if forward <= 0.0:
        raise ValueError("Stair base forward command must be positive")
    lateral = _finite_number(
        training["lateral_command_mps"],
        "Stair base lateral command",
    )
    yaw = _finite_number(
        training["yaw_command_radps"],
        "Stair base yaw command",
    )
    payload = {
        "schema_version": (
            _STAIR_TRAINING_DISTRIBUTION_EVIDENCE_SCHEMA_VERSION_V1
            if legacy
            else STAIR_TRAINING_DISTRIBUTION_EVIDENCE_SCHEMA_VERSION
        ),
        "status": "pass",
        "authority": "task_runner",
        "candidate_writable": False,
        "prompt_visible": False,
        "design_basis": contract["design_basis"],
        "selection_provenance": _json_copy(contract["selection_provenance"]),
        "upstream_reference": _json_copy(upstream),
        "terrain": {
            "generator_curriculum": True,
            "runtime_level_curriculum_enabled": True,
            "initial_max_terrain_level": int(
                terrain_contract["initial_max_terrain_level"]
            ),
            "terrain_rows": int(base_distribution["terrain_rows"]),
            "terrain_cols": int(base_distribution["terrain_cols"]),
            "step_height_range_m": [float(heights[0]), float(heights[1])],
            "level_update_term": terrain_contract["level_update_term"],
        },
        "command": {
            "lin_vel_x_range_mps": [0.0, forward],
            "lin_vel_y_range_mps": [lateral, lateral],
            "ang_vel_z_range_radps": [yaw, yaw],
            "standing_env_fraction": float(
                command_contract["standing_env_fraction"]
            ),
            "heading_command_enabled": False,
        },
    }
    if legacy:
        payload["formal_and_pilot_identical"] = True
    else:
        payload["formal_reference_and_candidates_identical"] = True
        payload["historical_pilot_distribution_identical"] = False
    return MappingProxyType(payload)


def _validate_revised_height_range(
    value: Any,
    *,
    base_protocol: Mapping[str, Any],
) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not _is_finite_number(item) for item in value)
    ):
        raise ValueError("Stair revised training height range is invalid")
    lower, upper = (float(value[0]), float(value[1]))
    if lower <= 0.0 or lower >= upper:
        raise ValueError("Stair revised training height range is invalid")

    base = _mapping(base_protocol, "Stair base protocol")
    base_terrain = _mapping(base["terrain"], "Stair base terrain")
    historical = _mapping(
        base_terrain["training_distribution"],
        "Stair base training distribution",
    )["step_height_range_m"]
    bins = base_terrain["evaluation_bins"]
    if (
        not isinstance(historical, list)
        or len(historical) != 2
        or not isinstance(bins, list)
    ):
        raise ValueError("Stair base terrain height evidence is invalid")
    eval_heights = sorted(float(item["height_m"]) for item in bins)
    if (
        lower != float(historical[0])
        or len(eval_heights) < 2
        or upper != eval_heights[-2]
        or sum(height > upper for height in eval_heights) != 1
    ):
        raise ValueError(
            "Stair revised training range must cover all but one evaluation frontier bin"
        )
    return lower, upper


def stair_training_distribution_upstream_source_evidence(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the source manifest expected from the installed Isaac Lab runtime."""

    upstream = _validate_upstream_reference(contract["upstream_reference"])
    identity = {
        "project": upstream["project"],
        "revision": upstream["revision"],
        "source_files": _json_list_copy(upstream["source_files"]),
    }
    return {
        "schema_version": (
            STAIR_TRAINING_DISTRIBUTION_UPSTREAM_SOURCE_EVIDENCE_SCHEMA_VERSION
        ),
        "status": "pass",
        **identity,
        "source_manifest_hash": canonical_json_hash(identity),
    }


def build_stair_execution_source_preflight_summary(
    *,
    task_protocol_hash: str,
    source_manifest_hash: str,
) -> dict[str, str]:
    """Build the canonical public proof emitted after source-byte verification."""

    return {
        "schema_version": STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
        "status": "pass",
        "task_protocol_hash": require_sha256_digest(
            task_protocol_hash,
            "Stair execution source preflight task protocol hash",
        ),
        "source_manifest_hash": require_sha256_digest(
            source_manifest_hash,
            "Stair execution source preflight source manifest hash",
        ),
    }


def validate_stair_training_distribution_upstream_source_evidence(
    value: Any,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected = stair_training_distribution_upstream_source_evidence(contract)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError(
            "Stair training distribution upstream source evidence mismatch"
        )
    return expected


def verify_installed_stair_training_distribution_upstream_sources(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the actual installed upstream source bytes before Stair execution."""

    evidence = stair_training_distribution_upstream_source_evidence(contract)
    roots: dict[str, tuple[Path, Path]] = {}
    packages = sorted(
        {item["import_package"] for item in evidence["source_files"]}
    )
    for package in packages:
        spec = importlib.util.find_spec(package)
        search_locations = tuple(
            Path(item) for item in (spec.submodule_search_locations or ())
        ) if spec is not None else ()
        package_sources = [
            item
            for item in evidence["source_files"]
            if item["import_package"] == package
        ]
        matching_roots = [
            root
            for root in search_locations
            if all(
                (root / item["package_relative_path"]).is_file()
                for item in package_sources
            )
        ]
        if len(matching_roots) != 1:
            raise ValueError(
                "Stair training distribution upstream source root is not unique"
            )
        root = matching_roots[0]
        if root.is_symlink():
            raise ValueError(
                "Stair training distribution upstream package is a symlink"
            )
        roots[package] = (root, root.resolve(strict=True))
    for item in evidence["source_files"]:
        root, resolved_root = roots[item["import_package"]]
        relative = Path(item["package_relative_path"])
        source = root / relative
        cursor = source
        while cursor != root:
            if cursor.is_symlink():
                raise ValueError(
                    "Stair training distribution upstream source traverses a symlink"
                )
            if cursor.parent == cursor:
                raise ValueError(
                    "Stair training distribution upstream source escapes package root"
                )
            cursor = cursor.parent
        resolved = source.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "Stair training distribution upstream source escapes package root"
            ) from exc
        try:
            source_bytes = resolved.read_bytes()
        except OSError as exc:
            raise ValueError(
                "Stair training distribution upstream source cannot be read"
            ) from exc
        if (
            _source_bytes_sha256(source_bytes) != item["checksum"]
            or len(source_bytes) != item["size_bytes"]
        ):
            raise ValueError(
                "Stair training distribution installed upstream source bytes drifted"
            )
        if not set(item["covered_symbols"]).issubset(
            _defined_source_symbols(source_bytes, path=resolved)
        ):
            raise ValueError(
                "Stair training distribution upstream source symbol coverage drifted"
            )
    return evidence


def _validate_upstream_reference(value: Any) -> dict[str, Any]:
    upstream = _mapping(
        value,
        "Stair training distribution upstream reference",
    )
    _exact_fields(
        upstream,
        {"project", "revision", "source_files"},
        "Stair training distribution upstream reference",
    )
    if (
        upstream["project"] != "IsaacLab"
        or upstream["revision"] != "v2.3.0"
    ):
        raise ValueError("Stair training distribution upstream reference mismatch")
    raw_sources = upstream["source_files"]
    if (
        not isinstance(raw_sources, Sequence)
        or isinstance(raw_sources, (str, bytes))
        or len(raw_sources) != len(_REQUIRED_UPSTREAM_SOURCES)
    ):
        raise ValueError("Stair training distribution upstream source set mismatch")
    sources = []
    previous_identity: tuple[str, str] | None = None
    for raw in raw_sources:
        source = _mapping(raw, "Stair training distribution upstream source")
        _exact_fields(
            source,
            {
                "import_package",
                "package_relative_path",
                "covered_symbols",
                "checksum",
                "size_bytes",
            },
            "Stair training distribution upstream source",
        )
        import_package = source["import_package"]
        if import_package not in {"isaaclab", "isaaclab_tasks"}:
            raise ValueError("Stair upstream source import package is invalid")
        relative_text = source["package_relative_path"]
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("Stair upstream source path must be non-empty")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or ".." in relative.parts
            or (
                previous_identity is not None
                and (import_package, relative_text) <= previous_identity
            )
        ):
            raise ValueError("Stair upstream source path is not canonical")
        checksum = require_sha256_checksum(
            source["checksum"],
            "Stair training distribution upstream source checksum",
        )
        size_bytes = _positive_int(
            source["size_bytes"],
            "Stair training distribution upstream source size",
        )
        covered_symbols = source["covered_symbols"]
        if (
            not isinstance(covered_symbols, Sequence)
            or isinstance(covered_symbols, (str, bytes))
            or not covered_symbols
            or any(not isinstance(item, str) or not item for item in covered_symbols)
            or list(covered_symbols) != sorted(set(covered_symbols))
        ):
            raise ValueError("Stair upstream source symbol coverage is invalid")
        sources.append(
            {
                "import_package": import_package,
                "package_relative_path": relative_text,
                "covered_symbols": list(covered_symbols),
                "checksum": checksum,
                "size_bytes": size_bytes,
            }
        )
        previous_identity = (import_package, relative_text)
    if tuple(
        (
            item["import_package"],
            item["package_relative_path"],
            tuple(item["covered_symbols"]),
        )
        for item in sources
    ) != _REQUIRED_UPSTREAM_SOURCES:
        raise ValueError("Stair training distribution upstream source set mismatch")
    return {
        "project": upstream["project"],
        "revision": upstream["revision"],
        "source_files": sources,
    }


def _defined_source_symbols(source_bytes: bytes, *, path: Path) -> set[str]:
    try:
        tree = ast.parse(source_bytes, filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError(
            "Stair training distribution upstream source cannot be parsed"
        ) from exc

    symbols: set[str] = set()

    def collect(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                symbols.add(name)
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name)
            elif prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        symbols.add(f"{prefix}.{target.id}")

    collect(tree.body)
    return symbols


def _source_bytes_sha256(source_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(source_bytes).hexdigest()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field_name} fields mismatch")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _finite_number(value: Any, field_name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _json_copy(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            result[key] = [
                _json_copy(nested) if isinstance(nested, Mapping) else nested
                for nested in item
            ]
        else:
            result[key] = item
    return result


def _json_list_copy(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Stair upstream source files must be a sequence")
    return [_json_copy(_mapping(item, "Stair upstream source")) for item in value]


__all__ = [
    "STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION",
    "STAIR_TRAINING_DISTRIBUTION_EVIDENCE_SCHEMA_VERSION",
    "STAIR_PREVIOUS_TRAINING_PROTOCOL_HASH",
    "STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION",
    "STAIR_TRAINING_DISTRIBUTION_UPSTREAM_SOURCE_EVIDENCE_SCHEMA_VERSION",
    "build_stair_execution_source_preflight_summary",
    "resolved_stair_training_distribution",
    "stair_training_distribution_upstream_source_evidence",
    "validate_stair_training_distribution_contract",
    "validate_stair_training_distribution_upstream_source_evidence",
    "verify_installed_stair_training_distribution_upstream_sources",
]

"""Candidate package loader for code-level candidate packages.

This module validates and imports candidate code in the current process.  Use it
directly only for trusted human-authored no-op candidates and tests.  Phase 8C
uses ``isolated_loader`` so the parent runner does not import candidate
entrypoints.
"""

from __future__ import annotations

import importlib.util
import json
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

from .components import TrainingComponents


MANIFEST_FILENAME = "candidate_manifest.yaml"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "humanoid_agent.cold_start.candidate.v1"


class CandidateLoadError(RuntimeError):
    """Raised when a candidate package cannot be loaded safely."""


@dataclass(frozen=True)
class CandidateManifest:
    schema_version: str
    candidate_id: str
    description: str
    author: str
    llm_authored: bool
    entrypoint: str
    phase: str
    expected_behavior: str
    path: Path

    def to_public_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "description": self.description,
            "author": self.author,
            "llm_authored": self.llm_authored,
            "entrypoint": self.entrypoint,
            "phase": self.phase,
            "expected_behavior": self.expected_behavior,
            "path": str(self.path),
        }


def candidate_manifest_required_fields() -> tuple[str, ...]:
    return tuple(field.name for field in fields(CandidateManifest) if field.name != "path")


def candidate_manifest_template(
    *,
    schema_version: str = CANDIDATE_MANIFEST_SCHEMA_VERSION,
    required_entrypoint: str = "training_code.components:build_training_components",
    phase: str = "8E",
) -> str:
    return (
        f"schema_version: {schema_version}\n"
        "candidate_id: phase8e_candidate_short_snake_case\n"
        'description: "One sentence describing the training-side change."\n'
        "author: LLM\n"
        "llm_authored: true\n"
        f"entrypoint: {required_entrypoint}\n"
        f'phase: "{phase}"\n'
        'expected_behavior: "Public training-side behavior expected from this candidate."\n'
    )


def candidate_manifest_prompt_contract(
    *,
    schema_version: str = CANDIDATE_MANIFEST_SCHEMA_VERSION,
    required_entrypoint: str = "training_code.components:build_training_components",
    phase: str = "8E",
) -> dict[str, Any]:
    return {
        "file": MANIFEST_FILENAME,
        "format": "YAML or JSON object serialized as text",
        "schema_version": schema_version,
        "required_fields_in_order": list(candidate_manifest_required_fields()),
        "required_entrypoint": required_entrypoint,
        "template": candidate_manifest_template(
            schema_version=schema_version,
            required_entrypoint=required_entrypoint,
            phase=phase,
        ),
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise CandidateLoadError(f"{path} is not JSON-compatible and PyYAML is unavailable") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise CandidateLoadError(f"{path} must contain a mapping")
    return loaded


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CandidateLoadError(f"manifest field {key!r} must be a non-empty string")
    return value


def load_candidate_manifest(candidate_dir: str | Path, *, expected_schema_version: str | None = None) -> CandidateManifest:
    root = Path(candidate_dir).resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CandidateLoadError(f"missing candidate manifest: {manifest_path}")

    data = _load_mapping(manifest_path)
    schema_version = _require_str(data, "schema_version")
    if expected_schema_version is not None and schema_version != expected_schema_version:
        raise CandidateLoadError(f"unsupported candidate schema_version: {schema_version!r}")

    llm_authored = data.get("llm_authored")
    if not isinstance(llm_authored, bool):
        raise CandidateLoadError("manifest field 'llm_authored' must be boolean")

    manifest = CandidateManifest(
        schema_version=schema_version,
        candidate_id=_require_str(data, "candidate_id"),
        description=_require_str(data, "description"),
        author=_require_str(data, "author"),
        llm_authored=llm_authored,
        entrypoint=_require_str(data, "entrypoint"),
        phase=_require_str(data, "phase"),
        expected_behavior=_require_str(data, "expected_behavior"),
        path=root,
    )
    resolve_entrypoint_file(manifest)
    return manifest


def load_training_components(
    candidate_dir: str | Path,
    *,
    expected_schema_version: str | None = None,
    allow_llm_authored: bool = False,
    task_spec: Any = None,
    base_env_cfg: Any = None,
    base_agent_cfg: Any = None,
) -> tuple[CandidateManifest, TrainingComponents]:
    """Load candidate training components through the manifest entrypoint."""

    manifest = load_candidate_manifest(candidate_dir, expected_schema_version=expected_schema_version)
    if manifest.llm_authored and not allow_llm_authored:
        raise CandidateLoadError("LLM-authored candidate code requires the isolated Phase 8C+ loader")

    module_path, function_name = resolve_entrypoint_file(manifest)
    module = _import_module_from_path(module_path, manifest.candidate_id)
    builder = getattr(module, function_name, None)
    if not callable(builder):
        raise CandidateLoadError(f"entrypoint function not callable: {manifest.entrypoint}")

    components = builder(task_spec, base_env_cfg, base_agent_cfg)
    if isinstance(components, TrainingComponents):
        return manifest, components
    if isinstance(components, dict):
        return manifest, TrainingComponents(**components)
    raise CandidateLoadError("candidate entrypoint must return TrainingComponents or a compatible dict")


def resolve_entrypoint_file(manifest: CandidateManifest) -> tuple[Path, str]:
    """Resolve the manifest entrypoint without executing candidate code."""

    if ":" not in manifest.entrypoint:
        raise CandidateLoadError("entrypoint must use 'module:function' format")
    module_name, function_name = manifest.entrypoint.split(":", 1)
    if not module_name.startswith("training_code."):
        raise CandidateLoadError("entrypoint module must be under training_code")
    if not function_name.isidentifier():
        raise CandidateLoadError("entrypoint function must be a valid identifier")

    relative_parts = module_name.split(".")
    if any(part in {"", ".."} for part in relative_parts):
        raise CandidateLoadError("invalid entrypoint module path")
    module_path = (manifest.path / Path(*relative_parts)).with_suffix(".py").resolve()
    training_root = (manifest.path / "training_code").resolve()
    if not module_path.is_file():
        raise CandidateLoadError(f"entrypoint module file missing: {module_path}")
    if training_root not in module_path.parents:
        raise CandidateLoadError("entrypoint module escapes training_code")
    return module_path, function_name


def candidate_package_fingerprint(manifest: CandidateManifest) -> dict[str, Any]:
    """Return externally checkable candidate package fingerprints.

    This is intentionally based on files and paths observed by the runner, not
    on the candidate's self-declared provenance.
    """

    module_path, _function_name = resolve_entrypoint_file(manifest)
    manifest_path = manifest.path / MANIFEST_FILENAME
    training_code_root = manifest.path / "training_code"
    return {
        "candidate_id": manifest.candidate_id,
        "path": str(manifest.path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "entrypoint": manifest.entrypoint,
        "entrypoint_path": str(module_path),
        "entrypoint_sha256": _sha256_file(module_path),
        "training_code_tree_sha256": _sha256_tree(training_code_root),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__/" in relative or relative.endswith(".pyc"):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _import_module_from_path(module_path: Path, candidate_id: str) -> ModuleType:
    module_name = f"humanoid_agent_candidate_{candidate_id}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CandidateLoadError(f"cannot import candidate module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

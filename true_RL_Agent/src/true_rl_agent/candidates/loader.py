"""Candidate manifest and controlled component loader."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from true_rl_agent.candidates.components import TrainingComponents


CANDIDATE_SCHEMA_VERSION = "true_rl_agent.candidate.v1"
CONTROLLED_ENTRYPOINT = "training_code.components:build_training_components"


@dataclass(frozen=True)
class CandidateManifest:
    schema_version: str
    candidate_id: str
    entrypoint: str

    def to_public_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "entrypoint": self.entrypoint,
        }


def load_candidate_manifest(candidate_dir: Path) -> CandidateManifest:
    path = candidate_dir / "candidate_manifest.yaml"
    values = _parse_simple_yaml_mapping(path.read_text(encoding="utf-8"))
    manifest = CandidateManifest(
        schema_version=values["schema_version"],
        candidate_id=values["candidate_id"],
        entrypoint=values["entrypoint"],
    )
    _validate_manifest(manifest)
    return manifest


def load_training_components(candidate_dir: Path) -> TrainingComponents:
    manifest = load_candidate_manifest(candidate_dir)
    module_path, function_name = _resolve_entrypoint(candidate_dir, manifest)
    module = _load_module(module_path, f"true_rl_candidate_{manifest.candidate_id}")
    factory = getattr(module, function_name)
    components = factory()
    if not isinstance(components, TrainingComponents):
        raise TypeError("candidate entrypoint must return TrainingComponents")
    return components


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(module_path.parent.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _validate_manifest(manifest: CandidateManifest) -> None:
    if manifest.schema_version != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate manifest schema_version is not supported")
    if manifest.entrypoint != CONTROLLED_ENTRYPOINT:
        raise ValueError("candidate manifest entrypoint is not allowlisted")
    if not manifest.candidate_id:
        raise ValueError("candidate manifest candidate_id is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(char not in allowed for char in manifest.candidate_id):
        raise ValueError("candidate manifest candidate_id has invalid characters")


def _resolve_entrypoint(candidate_dir: Path, manifest: CandidateManifest) -> tuple[Path, str]:
    _validate_manifest(manifest)
    module_name, function_name = manifest.entrypoint.split(":", 1)
    module_path = candidate_dir / (module_name.replace(".", "/") + ".py")
    resolved_candidate_dir = candidate_dir.resolve()
    resolved_module_path = module_path.resolve()
    try:
        resolved_module_path.relative_to(resolved_candidate_dir)
    except ValueError as exc:
        raise ValueError("candidate entrypoint escapes candidate directory") from exc
    return resolved_module_path, function_name


def _parse_simple_yaml_mapping(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"manifest line is not a key-value pair: {raw_line}")
        values[key.strip()] = value.strip().strip("\"'")
    return values

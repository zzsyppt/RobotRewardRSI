"""Candidate manifest and controlled component loader."""

from __future__ import annotations

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
    expected_fields = {"schema_version", "candidate_id", "entrypoint"}
    if set(values) != expected_fields:
        raise ValueError("candidate manifest fields mismatch")
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
    if module_path.is_symlink() or not module_path.is_file():
        raise ValueError("candidate components source must be a regular file")
    return load_training_components_source_bytes(
        module_path.read_bytes(),
        module_name=f"true_rl_candidate_{manifest.candidate_id}",
        source_name=str(module_path),
        function_name=function_name,
    )


def load_training_components_source_bytes(
    payload: bytes,
    *,
    module_name: str,
    source_name: str,
    function_name: str = "build_training_components",
) -> TrainingComponents:
    """Execute the exact verified source bytes through the controlled factory."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("training components source bytes must be non-empty")
    for field_name, value in (
        ("module_name", module_name),
        ("source_name", source_name),
        ("function_name", function_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"training components {field_name} is required")
    try:
        source = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("training components source must be valid UTF-8") from exc
    module = ModuleType(module_name)
    module.__file__ = source_name
    code = compile(source, source_name, "exec", dont_inherit=True)
    exec(code, module.__dict__)  # noqa: S102 - exact bytes passed a prior verifier
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError("candidate entrypoint factory must be callable")
    components = factory()
    if not isinstance(components, TrainingComponents):
        raise TypeError("candidate entrypoint must return TrainingComponents")
    return components


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
        normalized_key = key.strip().lstrip("\ufeff")
        if normalized_key in values:
            raise ValueError(f"candidate manifest has duplicate key: {normalized_key}")
        values[normalized_key] = value.strip().strip("\"'")
    return values

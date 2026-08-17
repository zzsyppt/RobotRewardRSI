"""Task-neutral ownership and byte boundaries for candidate packages."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from true_rl_agent.json_contract import canonical_json_bytes

_KIB = 1024

CANDIDATE_PACKAGE_FILE_MAX_BYTES: Mapping[str, int] = MappingProxyType(
    {
        "candidate_manifest.yaml": 2 * _KIB,
        "training_code/components.py": 24 * _KIB,
        "rationale.md": 8 * _KIB,
        "operation_diff.json": 8 * _KIB,
        "diagnostics.json": 8 * _KIB,
    }
)
"""Maximum provider-authored bytes for each file in the fixed package."""

PROVIDER_AUTHORED_CANDIDATE_FILES = tuple(CANDIDATE_PACKAGE_FILE_MAX_BYTES)
CANDIDATE_PACKAGE_MAX_BYTES = sum(CANDIDATE_PACKAGE_FILE_MAX_BYTES.values())
LEGACY_SEALED_REQUIRED_CANDIDATE_FILES = frozenset(
    {"candidate_manifest.yaml", "training_code/components.py"}
)
CANDIDATE_PACKAGE_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.candidate_package_contract.v1"
)

SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME = (
    ".true_rl_agent_candidate_artifact.json"
)
SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES = 64 * _KIB


def candidate_package_contract_descriptor() -> dict[str, object]:
    """Return the single task-neutral contract enforced for new candidates."""

    return {
        "schema_version": CANDIDATE_PACKAGE_CONTRACT_SCHEMA_VERSION,
        "provider_authored_files": list(PROVIDER_AUTHORED_CANDIDATE_FILES),
        "file_max_bytes": dict(CANDIDATE_PACKAGE_FILE_MAX_BYTES),
        "max_total_bytes": CANDIDATE_PACKAGE_MAX_BYTES,
        "all_files_required": True,
        "additional_files_allowed": False,
        "regular_files_required": True,
        "symlinks_allowed": False,
        "scope": "new_unsealed_candidate_packages",
        "legacy_sealed_replay_enforces_new_size_limits": False,
    }


CANDIDATE_PACKAGE_CONTRACT_HASH = hashlib.sha256(
    canonical_json_bytes(candidate_package_contract_descriptor())
).hexdigest()


class CandidatePackageContractError(ValueError):
    """A package failed structural preflight before any content was read."""

    def __init__(self, message: str, *, issue: str) -> None:
        super().__init__(message)
        self.issue = issue


@dataclass(frozen=True)
class CandidatePackageFileState:
    relative_path: str
    path: Path
    size_bytes: int
    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True)
class CandidatePackagePreflight:
    """Immutable filesystem facts captured before candidate content access."""

    root: Path
    sealed_artifact: bool
    enforce_size_limits: bool
    files: tuple[CandidatePackageFileState, ...]
    sealed_manifest: CandidatePackageFileState | None = None

    def file_state(self, relative_path: str) -> CandidatePackageFileState:
        for item in self.files:
            if item.relative_path == relative_path:
                return item
        if (
            self.sealed_manifest is not None
            and self.sealed_manifest.relative_path == relative_path
        ):
            return self.sealed_manifest
        raise CandidatePackageContractError(
            f"candidate package file is not in the fixed contract: {relative_path}",
            issue=f"forbidden_file:{relative_path}",
        )


def preflight_candidate_package(
    candidate_root: str | Path,
    *,
    sealed_artifact: bool = False,
    enforce_size_limits: bool = True,
) -> CandidatePackagePreflight:
    """Validate the complete fixed layout using metadata only.

    This function must run before any provider-authored file is read, parsed,
    copied, or executed.  Sealed v1 artifacts use the same closed provider file
    set plus their one runner-owned manifest.  Callers replaying already-sealed
    artifacts may disable the newly introduced provider byte ceilings; this
    preserves immutable historical evidence without weakening new admission.
    The fixed-shape runner manifest is always capped and descriptor-bound before
    parsing, independently of those provider-file replay limits.
    """

    if not isinstance(sealed_artifact, bool):
        raise TypeError("sealed_artifact must be a boolean")
    if not isinstance(enforce_size_limits, bool):
        raise TypeError("enforce_size_limits must be a boolean")

    unresolved_root = Path(candidate_root)
    root_status = _lstat(unresolved_root, display_path=".")
    if stat.S_ISLNK(root_status.st_mode):
        raise CandidatePackageContractError(
            "candidate package root must not be a symlink",
            issue="symlink_path:.",
        )
    if not stat.S_ISDIR(root_status.st_mode):
        raise CandidatePackageContractError(
            "candidate package root must be a directory",
            issue="nonregular_directory:.",
        )
    root = unresolved_root.resolve(strict=True)

    root_files = {
        path
        for path in PROVIDER_AUTHORED_CANDIDATE_FILES
        if "/" not in path
    }
    if sealed_artifact:
        root_files.add(SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME)
    _preflight_directory(
        root,
        relative_directory="",
        expected_files=root_files,
        expected_directories={"training_code"},
        provider_root=not sealed_artifact,
        required_names=(
            {"candidate_manifest.yaml", SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME, "training_code"}
            if sealed_artifact
            else None
        ),
    )
    _preflight_directory(
        root / "training_code",
        relative_directory="training_code",
        expected_files={"components.py"},
        expected_directories=set(),
        provider_root=False,
        required_names={"components.py"},
    )

    states: list[CandidatePackageFileState] = []
    for relative_path in PROVIDER_AUTHORED_CANDIDATE_FILES:
        path = root / Path(relative_path)
        try:
            file_status = path.lstat()
        except FileNotFoundError:
            if (
                sealed_artifact
                and relative_path not in LEGACY_SEALED_REQUIRED_CANDIDATE_FILES
            ):
                continue
            raise CandidatePackageContractError(
                f"candidate package is missing required path: {relative_path}",
                issue=f"missing_file:{relative_path}",
            )
        if stat.S_ISLNK(file_status.st_mode):
            raise CandidatePackageContractError(
                f"candidate package does not allow symlinks: {relative_path}",
                issue=f"symlink_path:{relative_path}",
            )
        if not stat.S_ISREG(file_status.st_mode):
            raise CandidatePackageContractError(
                f"candidate package file must be regular: {relative_path}",
                issue=f"nonregular_file:{relative_path}",
            )
        maximum = CANDIDATE_PACKAGE_FILE_MAX_BYTES[relative_path]
        if enforce_size_limits and file_status.st_size > maximum:
            raise CandidatePackageContractError(
                "candidate package file exceeds its byte limit: "
                f"{relative_path} ({file_status.st_size} > {maximum})",
                issue=(
                    f"file_too_large:{relative_path}:"
                    f"{file_status.st_size}>{maximum}"
                ),
            )
        states.append(
            CandidatePackageFileState(
                relative_path=relative_path,
                path=path,
                size_bytes=file_status.st_size,
                device=file_status.st_dev,
                inode=file_status.st_ino,
                modified_ns=file_status.st_mtime_ns,
            )
        )

    sealed_manifest_state: CandidatePackageFileState | None = None
    if sealed_artifact:
        manifest_path = root / SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME
        manifest_status = _lstat(
            manifest_path,
            display_path=SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
        )
        if stat.S_ISLNK(manifest_status.st_mode):
            raise CandidatePackageContractError(
                "sealed candidate artifact manifest must not be a symlink",
                issue=(
                    "symlink_path:"
                    f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME}"
                ),
            )
        if not stat.S_ISREG(manifest_status.st_mode):
            raise CandidatePackageContractError(
                "sealed candidate artifact manifest must be a regular file",
                issue=(
                    "nonregular_file:"
                    f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME}"
                ),
            )
        if (
            manifest_status.st_size
            > SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES
        ):
            raise CandidatePackageContractError(
                "sealed candidate artifact manifest exceeds its byte limit: "
                f"{manifest_status.st_size} > "
                f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES}",
                issue=(
                    "file_too_large:"
                    f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME}:"
                    f"{manifest_status.st_size}>"
                    f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES}"
                ),
            )
        sealed_manifest_state = CandidatePackageFileState(
            relative_path=SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
            path=manifest_path,
            size_bytes=manifest_status.st_size,
            device=manifest_status.st_dev,
            inode=manifest_status.st_ino,
            modified_ns=manifest_status.st_mtime_ns,
        )

    return CandidatePackagePreflight(
        root=root,
        sealed_artifact=sealed_artifact,
        enforce_size_limits=enforce_size_limits,
        files=tuple(states),
        sealed_manifest=sealed_manifest_state,
    )


@contextmanager
def open_preflighted_candidate_file(
    preflight: CandidatePackagePreflight,
    relative_path: str,
) -> Iterator[BinaryIO]:
    """Open a preflighted candidate-package file without following a replacement link."""

    if not isinstance(preflight, CandidatePackagePreflight):
        raise TypeError("candidate package preflight is required")
    state = preflight.file_state(relative_path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(state.path, flags)
    except OSError as exc:
        raise CandidatePackageContractError(
            f"candidate package file changed after preflight: {relative_path}",
            issue=f"file_changed:{relative_path}",
        ) from exc
    stream: BinaryIO | None = None
    try:
        opened_status = os.fstat(descriptor)
        _require_same_regular_file(state, opened_status)
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        yield stream
        final_status = os.fstat(stream.fileno())
        _require_same_regular_file(state, final_status)
    finally:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)


def copy_preflighted_candidate_package(
    preflight: CandidatePackagePreflight,
    destination_root: str | Path,
) -> list[dict[str, str | int]]:
    """Stream the fixed package from bound file descriptors into a clean root."""

    if not isinstance(preflight, CandidatePackagePreflight):
        raise TypeError("candidate package preflight is required")
    destination = Path(destination_root)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("candidate package snapshot destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    evidence: list[dict[str, str | int]] = []
    for state in sorted(preflight.files, key=lambda item: item.relative_path):
        target = destination / Path(state.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        copied_bytes = 0
        with open_preflighted_candidate_file(
            preflight,
            state.relative_path,
        ) as source, target.open("xb") as output:
            while chunk := source.read(64 * 1024):
                output.write(chunk)
                digest.update(chunk)
                copied_bytes += len(chunk)
        if copied_bytes != state.size_bytes:
            raise CandidatePackageContractError(
                "candidate package file changed while it was being copied: "
                f"{state.relative_path}",
                issue=f"file_changed:{state.relative_path}",
            )
        evidence.append(
            {
                "path": state.relative_path,
                "checksum": "sha256:" + digest.hexdigest(),
                "size_bytes": copied_bytes,
            }
        )
    return evidence


def _preflight_directory(
    directory: Path,
    *,
    relative_directory: str,
    expected_files: set[str],
    expected_directories: set[str],
    provider_root: bool,
    required_names: set[str] | None = None,
) -> None:
    display_directory = relative_directory or "."
    directory_status = _lstat(directory, display_path=display_directory)
    if stat.S_ISLNK(directory_status.st_mode):
        raise CandidatePackageContractError(
            f"candidate package does not allow symlinks: {display_directory}",
            issue=f"symlink_path:{display_directory}",
        )
    if not stat.S_ISDIR(directory_status.st_mode):
        raise CandidatePackageContractError(
            f"candidate package directory is not regular: {display_directory}",
            issue=f"nonregular_directory:{display_directory}",
        )

    expected_names = expected_files | expected_directories
    seen_names: set[str] = set()
    with os.scandir(directory) as scanner:
        for entry in scanner:
            relative = _join_relative(relative_directory, entry.name)
            entry_status = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_status.st_mode):
                raise CandidatePackageContractError(
                    f"candidate package does not allow symlinks: {relative}",
                    issue=f"symlink_path:{relative}",
                )
            if entry.name not in expected_names:
                if (
                    provider_root
                    and entry.name
                    == SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME
                ):
                    message = "provider candidate contains reserved artifact manifest"
                else:
                    message = f"candidate package contains forbidden path: {relative}"
                raise CandidatePackageContractError(
                    message,
                    issue=f"forbidden_file:{relative}",
                )
            seen_names.add(entry.name)
            if entry.name in expected_files and not stat.S_ISREG(entry_status.st_mode):
                raise CandidatePackageContractError(
                    f"candidate package file must be regular: {relative}",
                    issue=f"nonregular_file:{relative}",
                )
            if entry.name in expected_directories and not stat.S_ISDIR(entry_status.st_mode):
                raise CandidatePackageContractError(
                    f"candidate package directory is not regular: {relative}",
                    issue=f"nonregular_directory:{relative}",
                )
    required = expected_names if required_names is None else required_names
    if not required <= expected_names:
        raise ValueError("required candidate package names must be allowlisted")
    for missing in sorted(required - seen_names):
        relative = _join_relative(relative_directory, missing)
        raise CandidatePackageContractError(
            f"candidate package is missing required path: {relative}",
            issue=f"missing_file:{relative}",
        )


def _lstat(path: Path, *, display_path: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise CandidatePackageContractError(
            f"candidate package is missing required path: {display_path}",
            issue=f"missing_file:{display_path}",
        ) from exc


def _require_same_regular_file(
    expected: CandidatePackageFileState,
    actual: os.stat_result,
) -> None:
    if (
        not stat.S_ISREG(actual.st_mode)
        or actual.st_dev != expected.device
        or actual.st_ino != expected.inode
        or actual.st_size != expected.size_bytes
        or actual.st_mtime_ns != expected.modified_ns
    ):
        raise CandidatePackageContractError(
            "candidate package file changed after preflight: "
            f"{expected.relative_path}",
            issue=f"file_changed:{expected.relative_path}",
        )


def _join_relative(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


__all__ = [
    "CANDIDATE_PACKAGE_FILE_MAX_BYTES",
    "CANDIDATE_PACKAGE_MAX_BYTES",
    "PROVIDER_AUTHORED_CANDIDATE_FILES",
    "SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME",
    "SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES",
    "CandidatePackageContractError",
    "CandidatePackageFileState",
    "CandidatePackagePreflight",
    "copy_preflighted_candidate_package",
    "open_preflighted_candidate_file",
    "preflight_candidate_package",
]

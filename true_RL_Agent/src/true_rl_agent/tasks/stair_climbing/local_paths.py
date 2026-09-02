"""Task-owned local path guards for Stair trusted runtime inputs and outputs."""

from __future__ import annotations

from pathlib import Path


def resolve_local_path(path: Path, *, field_name: str) -> Path:
    """Resolve a local path without allowing a symlink at the declared leaf."""

    path = _path(path, field_name)
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink")
    return path.resolve()


def resolve_existing_file(path: Path, *, field_name: str) -> Path:
    path = _path(path, field_name)
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a regular file")
    return resolved


def resolve_existing_directory(path: Path, *, field_name: str) -> Path:
    path = _path(path, field_name)
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must be a directory")
    return resolved


def resolve_fresh_path(path: Path, *, field_name: str) -> Path:
    path = _path(path, field_name)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{field_name} must be fresh")
    return path.resolve()


def _path(value: Path, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    return value


__all__ = [
    "resolve_existing_directory",
    "resolve_existing_file",
    "resolve_fresh_path",
    "resolve_local_path",
]

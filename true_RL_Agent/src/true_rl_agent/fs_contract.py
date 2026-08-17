"""Small, task-neutral filesystem publication and bounded-read primitives."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path


def read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read at most ``max_bytes`` from one non-symlink regular file."""

    target = Path(path)
    try:
        inspected = target.lstat()
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected") from exc
    if not stat.S_ISREG(inspected.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if inspected.st_size > max_bytes:
        raise ValueError(f"{label} exceeds maximum size of {max_bytes} bytes")
    try:
        with target.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"failed to read {label}") from exc
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds maximum size of {max_bytes} bytes")
    return payload


def create_once_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes only when the destination does not exist."""

    target = Path(path)
    temporary = _write_temporary_sibling(target, payload)
    try:
        os.link(temporary, target)
    finally:
        _unlink_if_present(temporary)


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a destination with fully written bytes."""

    target = Path(path)
    temporary = _write_temporary_sibling(target, payload)
    try:
        os.replace(temporary, target)
    finally:
        _unlink_if_present(temporary)


def _write_temporary_sibling(path: Path, payload: bytes) -> Path:
    # Keep the sibling name short and independent of the target name/content so
    # a valid target path does not become invalid only while it is published.
    temporary = path.with_name(f".{uuid.uuid4().hex[:12]}")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            _unlink_if_present(temporary)
        raise
    return temporary


def _unlink_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "atomic_replace_bytes",
    "create_once_bytes",
    "read_bounded_regular_file",
]

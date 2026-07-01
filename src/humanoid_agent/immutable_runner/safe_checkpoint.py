"""Runner-owned safe checkpoint loading for the trusted evaluation boundary.

Phase 8E candidates train inside an OS-isolated process and emit a checkpoint.
The only artifact that crosses back to the trusted side is that checkpoint, so
its *deserialization* is the last place a candidate could corrupt the evaluator:
RSL-RL's ``OnPolicyRunner.load`` calls ``torch.load(..., weights_only=False)``,
which executes arbitrary pickle reduce payloads.  A candidate that tampers with
the checkpoint bytes before it is sealed could therefore run code inside the
trusted eval process.

``assert_safe_checkpoint`` closes that vector.  ``torch.load(weights_only=True)``
uses a restricted unpickler that only accepts tensor/state-dict primitives; if a
checkpoint loads under it, the bytes provably contain no executable reduce
payload, so a *later* ``weights_only=False`` load (e.g. inside RSL-RL) cannot run
code either.  The contract is: any trusted process must gate a candidate
checkpoint through this module before loading or selecting it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class CheckpointSafetyError(RuntimeError):
    """Raised when a checkpoint cannot be proven free of executable payloads."""


def assert_safe_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load ``path`` with ``weights_only=True`` and return the safe object.

    Raises :class:`CheckpointSafetyError` if the file is missing, unreadable, or
    contains anything the restricted unpickler rejects (i.e. a potential code
    payload).  A successful return proves the checkpoint bytes reference only
    allowlisted tensor/primitive globals.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointSafetyError(f"checkpoint is missing: {checkpoint}")
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - torch absence is a trusted-side config error.
        raise CheckpointSafetyError("torch is required to validate checkpoint safety") from exc
    try:
        loaded = torch.load(str(checkpoint), weights_only=True, map_location=map_location)
    except Exception as exc:  # noqa: BLE001 - any failure means we cannot prove it is safe.
        raise CheckpointSafetyError(
            f"checkpoint failed weights_only=True validation (possible code payload): {checkpoint}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise CheckpointSafetyError(f"checkpoint must deserialize to a state dict, got {type(loaded).__name__}: {checkpoint}")
    return loaded


def safe_load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Return a candidate checkpoint's tensors after proving it carries no payload.

    Trusted eval code should use this instead of ``OnPolicyRunner.load`` (which
    loads with ``weights_only=False``).  When eval must reuse RSL-RL's loader,
    call :func:`assert_safe_checkpoint` first as the gate.
    """

    return assert_safe_checkpoint(path, map_location=map_location)


def file_sha256(path: str | Path) -> str | None:
    """Return the sha256 of ``path`` or ``None`` if it cannot be read."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def checkpoint_safety_report(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Non-raising trusted-side safety report for a candidate checkpoint.

    This is the authoritative gate helper for any *trusted* process (trusted
    eval, or a runner step that never executed candidate code).  It must NOT be
    relied upon when called inside a process that has loaded candidate code,
    because candidate code can monkeypatch the symbols it uses; in that case the
    result is advisory only and a trusted process must re-run this gate.
    """

    report: dict[str, Any] = {
        "path": str(path),
        "safe_load": False,
        "sha256": file_sha256(path),
        "error": None,
    }
    try:
        assert_safe_checkpoint(path, map_location=map_location)
        report["safe_load"] = True
    except CheckpointSafetyError as exc:
        report["error"] = str(exc)
    return report

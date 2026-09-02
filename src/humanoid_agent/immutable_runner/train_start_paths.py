"""Trusted path rules for Phase 8E train-start checkpoints.

The train-start child runs candidate-authored code, so every path it reports is
untrusted data.  Parent-side checkpoint gates may only inspect the runner-fixed
sealed checkpoint location, never arbitrary child-provided absolute paths.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from humanoid_agent.immutable_runner.checkpoints import checkpoint_iteration


TRAIN_START_CHECKPOINT_PREFIX = PurePosixPath("/runner_artifacts/train_start_smoke/sealed_checkpoints")


def resolve_train_start_checkpoint_path(isolated_path: str, runner_artifacts_root: Path) -> Path | None:
    """Map a sealed train-start checkpoint path from the isolated namespace.

    Only ``/runner_artifacts/train_start_smoke/sealed_checkpoints/model_<n>.pt``
    is accepted.  The function intentionally rejects ``/candidate_output``, host
    absolute paths, path traversal, nested files, and non-model checkpoint names.
    """

    try:
        candidate = PurePosixPath(isolated_path)
    except TypeError:
        return None
    if not candidate.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    expected_prefix = TRAIN_START_CHECKPOINT_PREFIX.parts
    if candidate.parts[: len(expected_prefix)] != expected_prefix:
        return None
    suffix = candidate.parts[len(expected_prefix) :]
    if len(suffix) != 1:
        return None
    checkpoint_name = suffix[0]
    if checkpoint_iteration(checkpoint_name) is None:
        return None

    base = (runner_artifacts_root / "train_start_smoke" / "sealed_checkpoints").resolve(strict=False)
    host_path = (base / checkpoint_name).resolve(strict=False)
    try:
        host_path.relative_to(base)
    except ValueError:
        return None
    return host_path

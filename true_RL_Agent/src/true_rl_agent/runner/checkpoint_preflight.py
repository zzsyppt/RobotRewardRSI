"""Runner-owned byte preflight for sealed checkpoint process inputs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from true_rl_agent.runner.argv_contract import (
    require_exact_option_value,
    require_option_absent,
    require_string_argv,
)
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.specs import CheckpointRef


def validate_checkpoint_input_artifact(
    checkpoint: CheckpointRef,
    *,
    local_path: str | Path,
) -> Path:
    """Verify the exact sealed checkpoint input bytes before process launch."""

    if checkpoint.role not in {"eval", "warm_start"}:
        raise ValueError(
            "checkpoint input role must be eval or warm_start: "
            f"{checkpoint.role}"
        )
    if not checkpoint.sealed:
        raise ValueError("checkpoint input must be sealed")
    resolved = Path(os.path.abspath(local_path))
    if resolved.is_symlink():
        raise ValueError(f"checkpoint input must not be a symlink: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"checkpoint input file does not exist: {resolved}")
    actual_checksum = file_sha256(resolved)
    if actual_checksum is None:
        raise ValueError(f"checkpoint input could not be read: {resolved}")
    if checkpoint.checksum != actual_checksum:
        raise ValueError(
            "checkpoint input checksum does not match bytes: "
            f"expected={checkpoint.checksum!r} actual={actual_checksum!r}"
        )
    return resolved


def validate_checkpoint_process_input(
    checkpoint: CheckpointRef | None,
    *,
    argv: Sequence[str],
    checkpoint_path_option: str,
    checkpoint_checksum_option: str,
    resolved_local_path: str | Path | None = None,
) -> Path | None:
    """Bind checkpoint bytes and digest to the exact argv that will be launched."""

    require_string_argv(argv, context="checkpoint argv")
    if checkpoint is None:
        require_option_absent(argv, checkpoint_path_option)
        require_option_absent(argv, checkpoint_checksum_option)
        if resolved_local_path is not None:
            raise ValueError(
                "resolved checkpoint path must be absent when no checkpoint is configured"
            )
        return None

    raw_local_path = checkpoint.metadata.get("local_path")
    if not isinstance(raw_local_path, str) or not raw_local_path:
        raise ValueError("checkpoint input metadata.local_path is required")
    if "://" in raw_local_path:
        raise ValueError("checkpoint input local_path must be a local file")
    if resolved_local_path is None:
        raise ValueError("resolved checkpoint input path is required")

    require_exact_option_value(
        argv,
        checkpoint_path_option,
        expected=raw_local_path,
    )
    require_exact_option_value(
        argv,
        checkpoint_checksum_option,
        expected=checkpoint.checksum,
    )
    return validate_checkpoint_input_artifact(
        checkpoint,
        local_path=resolved_local_path,
    )

__all__ = [
    "validate_checkpoint_input_artifact",
    "validate_checkpoint_process_input",
]

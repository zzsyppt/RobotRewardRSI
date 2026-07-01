"""Checkpoint selection utilities for immutable runner code.

The runner, not candidate code, owns checkpoint selection.  These helpers keep
that policy explicit and dependency-free so shell wrappers, training summaries,
and future code-sandbox runners all use the same rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointInfo:
    """A numerically indexed RSL-RL checkpoint."""

    path: Path
    iteration: int


def checkpoint_iteration(path: str | Path) -> int | None:
    """Return the numeric iteration from ``model_<iteration>.pt``.

    Non-standard filenames intentionally return ``None``.  They are ignored by
    immutable runner selection instead of being treated as fallback candidates.
    """

    match = _MODEL_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def list_model_checkpoints(log_dir: str | Path) -> list[CheckpointInfo]:
    """List valid ``model_<iteration>.pt`` checkpoints sorted numerically."""

    root = Path(log_dir)
    if not root.is_dir():
        return []
    checkpoints: list[CheckpointInfo] = []
    for path in root.glob("model_*.pt"):
        iteration = checkpoint_iteration(path)
        if iteration is not None:
            checkpoints.append(CheckpointInfo(path=path, iteration=iteration))
    return sorted(checkpoints, key=lambda item: (item.iteration, str(item.path)))


def find_latest_checkpoint(log_dir: str | Path) -> Path | None:
    """Return the checkpoint with the largest numeric iteration."""

    checkpoints = list_model_checkpoints(log_dir)
    if not checkpoints:
        return None
    return checkpoints[-1].path


def _load_summary(summary_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(summary_or_path, dict):
        return summary_or_path
    with Path(summary_or_path).open(encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint summary must be a JSON object")
    return loaded


def find_latest_checkpoint_from_summary(
    summary_or_path: str | Path | dict[str, Any],
    *,
    allow_legacy_latest_checkpoint: bool = False,
) -> Path | None:
    """Select the latest valid checkpoint using a training summary.

    Preferred source is ``training.log_dir`` because old summaries may contain a
    lexicographically selected ``training.latest_checkpoint``.  The summary
    field is intentionally ignored by default when no log directory can be
    inspected.  Set ``allow_legacy_latest_checkpoint=True`` only for manual
    recovery of trusted historical summaries.
    """

    summary = _load_summary(summary_or_path)
    training = summary.get("training", {})
    if not isinstance(training, dict):
        return None

    log_dir = training.get("log_dir")
    if log_dir:
        latest = find_latest_checkpoint(log_dir)
        if latest is not None:
            return latest

    latest_checkpoint = training.get("latest_checkpoint") if allow_legacy_latest_checkpoint else None
    if latest_checkpoint and checkpoint_iteration(latest_checkpoint) is not None:
        path = Path(latest_checkpoint)
        if path.is_file():
            return path

    return None

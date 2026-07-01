"""Immutable-runner utilities shared by training/evaluation wrappers."""

from .budget import build_budget_accounting, derive_max_iterations, parse_total_timesteps
from .checkpoints import checkpoint_iteration, find_latest_checkpoint, find_latest_checkpoint_from_summary
from .safe_checkpoint import (
    CheckpointSafetyError,
    assert_safe_checkpoint,
    checkpoint_safety_report,
    file_sha256,
    safe_load_checkpoint,
)

__all__ = [
    "build_budget_accounting",
    "checkpoint_iteration",
    "derive_max_iterations",
    "find_latest_checkpoint",
    "find_latest_checkpoint_from_summary",
    "parse_total_timesteps",
    "CheckpointSafetyError",
    "assert_safe_checkpoint",
    "checkpoint_safety_report",
    "file_sha256",
    "safe_load_checkpoint",
]

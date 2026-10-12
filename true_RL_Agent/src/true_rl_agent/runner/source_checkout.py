"""Source-checkout authority for formal execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


EXECUTION_SOURCE_ROOTS = ("scripts", "src", "configs")


@dataclass(frozen=True)
class SourceCheckout:
    """Observed Git state that can affect a formal execution."""

    head_commit: str
    tracked_modifications: bool
    untracked_execution_sources: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.tracked_modifications and not self.untracked_execution_sources


def inspect_source_checkout(project_root: Path) -> SourceCheckout:
    """Inspect tracked state plus untracked files under executable source roots."""

    root = project_root.resolve()
    head_commit = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "-z",
    )
    untracked = _git_output(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *EXECUTION_SOURCE_ROOTS,
    )
    return SourceCheckout(
        head_commit=head_commit,
        tracked_modifications=bool(tracked_status),
        untracked_execution_sources=tuple(
            item.decode("utf-8") for item in untracked.split(b"\0") if item
        ),
    )


def require_source_checkout(
    project_root: Path,
    *,
    expected_commit: str,
) -> SourceCheckout:
    """Require the exact committed source authority for a formal run."""

    checkout = inspect_source_checkout(project_root)
    if checkout.head_commit != expected_commit:
        raise ValueError("source checkout commit drift from expected commit")
    if checkout.tracked_modifications:
        raise ValueError("source checkout has tracked modifications")
    if checkout.untracked_execution_sources:
        raise ValueError("source checkout has untracked execution source")
    return checkout


def _git_output(project_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout

"""Process-exit handling for Isaac SimulationApp entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def bind_isaac_log_path(args: Any, summary_path: str | Path) -> Path:
    """Assign a job-local Kit log before SimulationApp is constructed."""

    path = Path(summary_path).resolve().with_suffix(".kit.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    option = f'--/log/file="{path.as_posix()}"'
    args.kit_args = " ".join(part for part in (getattr(args, "kit_args", ""), option) if part)
    return path


def close_isaac_app_on_success(
    simulation_app: Any,
    *,
    exit_code: int,
) -> int:
    """Close Isaac only after success so its hard exit cannot mask failures."""

    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise TypeError("Isaac process exit_code must be an integer")
    if not 0 <= exit_code <= 255:
        raise ValueError("Isaac process exit_code must be in [0, 255]")
    _flush_standard_streams()
    if exit_code != 0:
        return exit_code
    close = getattr(simulation_app, "close", None)
    if not callable(close):
        raise TypeError("Isaac simulation app must provide close()")
    try:
        close()
    except Exception as exc:  # pragma: no cover - real Isaac shutdown failure
        print(f"Isaac SimulationApp close failed: {exc}", file=sys.stderr)
        _flush_standard_streams()
        return 1
    _flush_standard_streams()
    return 0


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass


__all__ = ["bind_isaac_log_path", "close_isaac_app_on_success"]

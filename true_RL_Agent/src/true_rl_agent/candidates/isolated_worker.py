"""Task-neutral two-stage lifecycle for isolated candidate workers."""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable


class IsolatedWorkerStartupError(RuntimeError):
    """Raised when worker infrastructure does not become ready."""


class IsolatedWorkerProtocolError(RuntimeError):
    """Raised when a worker exits without following the ready/result protocol."""


@dataclass(frozen=True)
class WorkerReady:
    evidence: Any = None


@dataclass(frozen=True)
class WorkerStartupFailure:
    error_type: str


@dataclass(frozen=True)
class IsolatedWorkerOutcome:
    completed: bool
    result: Any = None
    ready_evidence: Any = None


def run_ready_worker(
    *,
    target: Callable[..., None],
    args: tuple[Any, ...],
    startup_timeout_s: float,
    execution_timeout_s: float,
) -> IsolatedWorkerOutcome:
    """Start execution timing only after the spawned worker declares ready."""

    startup_timeout_s = _positive_finite_timeout(
        startup_timeout_s,
        "worker startup timeout",
    )
    execution_timeout_s = _positive_finite_timeout(
        execution_timeout_s,
        "worker execution timeout",
    )
    ctx = mp.get_context("spawn")
    result_parent, result_child = ctx.Pipe(duplex=False)
    start_child, start_parent = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=target,
        args=(*args, result_child, start_child),
    )
    process.daemon = True
    try:
        process.start()
        result_child.close()
        start_child.close()
        ready = _receive_message(
            result_parent,
            timeout_s=startup_timeout_s,
        )
        if ready is None:
            raise IsolatedWorkerStartupError("isolated worker startup timed out")
        if isinstance(ready, WorkerStartupFailure):
            raise IsolatedWorkerStartupError(
                "isolated worker startup failed: " + ready.error_type
            )
        if not isinstance(ready, WorkerReady):
            raise IsolatedWorkerStartupError(
                "isolated worker did not publish readiness evidence"
            )

        start_parent.send("start")
        result = _receive_message(
            result_parent,
            timeout_s=execution_timeout_s,
        )
        if result is None:
            _terminate_worker(process)
            return IsolatedWorkerOutcome(
                completed=False,
                ready_evidence=ready.evidence,
            )
        process.join(1.0)
        if process.is_alive():
            _terminate_worker(process)
        return IsolatedWorkerOutcome(
            completed=True,
            result=result,
            ready_evidence=ready.evidence,
        )
    except BaseException:
        if process.pid is not None:
            _terminate_worker(process)
        raise
    finally:
        result_parent.close()
        result_child.close()
        start_child.close()
        start_parent.close()


def _receive_message(connection: Any, *, timeout_s: float) -> Any | None:
    if not connection.poll(timeout_s):
        return None
    try:
        return connection.recv()
    except EOFError as exc:
        raise IsolatedWorkerProtocolError(
            "isolated worker exited without evidence"
        ) from exc


def _terminate_worker(process: Any) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(1.0)


def _positive_finite_timeout(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


__all__ = [
    "IsolatedWorkerOutcome",
    "IsolatedWorkerProtocolError",
    "IsolatedWorkerStartupError",
    "WorkerReady",
    "WorkerStartupFailure",
    "run_ready_worker",
]

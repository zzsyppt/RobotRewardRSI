"""Small runner-owned process lifecycle primitive for long train/eval jobs."""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_LOG_TAIL_BYTES = 16 * 1024
OWNED_PROCESS_IDENTITY_SCHEMA_VERSION = "true_rl_agent.runner.owned_process_identity.v1"


class OwnedProcessLaunchError(OSError):
    """Raised when a process could not be launched under runner ownership."""


class OwnedProcessCleanupIncomplete(RuntimeError):
    """Raised when a Linux owned process group still has live members."""

    def __init__(self, process_group_id: int, live_process_ids: Sequence[int]) -> None:
        self.process_group_id = int(process_group_id)
        self.live_process_ids = tuple(int(pid) for pid in live_process_ids)
        super().__init__(
            "owned process group cleanup remained incomplete: "
            f"pgid={self.process_group_id} live_pids={list(self.live_process_ids)}"
        )


class OwnedProcessTimeout(TimeoutError):
    """Raised after a timed-out process tree has been cleaned up."""


@dataclass(frozen=True)
class OwnedProcessResult:
    returncode: int
    pid: int
    stdout_tail: str
    stderr_tail: str
    stdout_log: Path
    stderr_log: Path


def current_owned_process_identity() -> dict[str, Any]:
    """Project the current process identity used by runner-owned summaries."""

    process_id = int(os.getpid())
    get_process_group = getattr(os, "getpgrp", None)
    process_group_id = (
        int(get_process_group()) if callable(get_process_group) else process_id
    )
    return {
        "schema_version": OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
        "process_id": process_id,
        "process_group_id": process_group_id,
    }


def validate_owned_process_identity(
    value: Mapping[str, Any] | None,
    *,
    expected_process_id: int | None = None,
    require_process_group_leader: bool = False,
) -> dict[str, Any]:
    """Validate task-neutral child identity without interpreting task evidence."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "process_id",
        "process_group_id",
    }:
        raise ValueError("owned process identity fields mismatch")
    if value.get("schema_version") != OWNED_PROCESS_IDENTITY_SCHEMA_VERSION:
        raise ValueError("owned process identity schema mismatch")
    process_id = value.get("process_id")
    process_group_id = value.get("process_group_id")
    for raw, field_name in (
        (process_id, "process_id"),
        (process_group_id, "process_group_id"),
    ):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"owned process identity {field_name} is invalid")
    if expected_process_id is not None:
        if (
            isinstance(expected_process_id, bool)
            or not isinstance(expected_process_id, int)
            or expected_process_id <= 0
        ):
            raise ValueError("expected owned process id is invalid")
        if process_id != expected_process_id:
            raise ValueError("summary process id does not match runner-owned child")
    if require_process_group_leader and process_group_id != process_id:
        raise ValueError("runner-owned child is not its process-group leader")
    return {
        "schema_version": OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
        "process_id": process_id,
        "process_group_id": process_group_id,
    }


def run_owned_process(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    timeout_s: int,
    log_dir: Path,
    termination_grace_s: float = 5.0,
    environment_overrides: Mapping[str, str | None] | None = None,
    on_started: Callable[[int], None] | None = None,
) -> OwnedProcessResult:
    """Run one argv job and retain ownership of its process tree until reaped."""

    if not isinstance(argv, Sequence) or isinstance(argv, str | bytes) or not argv:
        raise ValueError("owned process argv must be a non-empty sequence")
    if any(not isinstance(part, str) or not part for part in argv):
        raise ValueError("owned process argv entries must be non-empty strings")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        raise ValueError("owned process timeout_s must be a positive integer")
    if (
        isinstance(termination_grace_s, bool)
        or not isinstance(termination_grace_s, int | float)
        or float(termination_grace_s) < 0.0
    ):
        raise ValueError("termination_grace_s must be a non-negative number")
    if on_started is not None and not callable(on_started):
        raise TypeError("owned process on_started must be callable")

    resolved_log_dir = Path(log_dir).resolve(strict=False)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = resolved_log_dir / "stdout.log"
    stderr_log = resolved_log_dir / "stderr.log"
    process: subprocess.Popen[bytes] | None = None
    windows_job: _WindowsJobObject | None = None
    with stdout_log.open("wb") as stdout_handle, stderr_log.open("wb") as stderr_handle:
        try:
            process = _spawn(
                list(argv),
                cwd=cwd,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                environment_overrides=environment_overrides,
            )
            if os.name == "nt":
                windows_job = _WindowsJobObject.create_and_assign(process)
            if on_started is not None:
                on_started(int(process.pid))
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                _cleanup_preserving_exception(
                    process,
                    windows_job=windows_job,
                    grace_s=float(termination_grace_s),
                    original_exception=exc,
                )
                raise OwnedProcessTimeout(
                    f"process timed out after {timeout_s}s"
                ) from exc
            _cleanup_after_normal_exit(
                process,
                windows_job=windows_job,
                grace_s=float(termination_grace_s),
            )
        except (
            OwnedProcessCleanupIncomplete,
            OwnedProcessTimeout,
            OwnedProcessLaunchError,
        ):
            raise
        except BaseException as exc:
            if process is not None:
                _cleanup_preserving_exception(
                    process,
                    windows_job=windows_job,
                    grace_s=float(termination_grace_s),
                    original_exception=exc,
                )
            raise
        finally:
            if windows_job is not None:
                windows_job.close()

    assert process is not None
    return OwnedProcessResult(
        returncode=int(returncode),
        pid=int(process.pid),
        stdout_tail=_read_log_tail(stdout_log),
        stderr_tail=_read_log_tail(stderr_log),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def _spawn(
    argv: list[str],
    *,
    cwd: Path | None,
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    environment_overrides: Mapping[str, str | None] | None,
) -> subprocess.Popen[bytes]:
    creationflags = 0
    kwargs: dict[str, object] = {}
    spawn_argv = argv
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
        spawn_argv = _linux_parent_death_guard_argv(argv)
    if environment_overrides is not None:
        environment = dict(os.environ)
        for name, value in environment_overrides.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        kwargs["env"] = environment
    try:
        return subprocess.Popen(
            spawn_argv,
            cwd=cwd,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            **kwargs,
        )
    except OSError as exc:
        raise OwnedProcessLaunchError(str(exc)) from exc


def _linux_parent_death_guard_argv(argv: list[str]) -> list[str]:
    """Wrap trusted absolute executables so Linux parent death stops the job."""

    if not sys.platform.startswith("linux"):
        return argv
    executable = Path(argv[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        # Preserve native Popen launch errors for missing/invalid relative
        # executables instead of turning them into an opaque guard exit.
        return argv
    guard = Path(__file__).with_name("process_guard.py")
    return [
        sys.executable,
        str(guard),
        "--parent-pid",
        str(os.getpid()),
        "--",
        *argv,
    ]


def _cleanup_after_normal_exit(
    process: subprocess.Popen[bytes],
    *,
    windows_job: _WindowsJobObject | None,
    grace_s: float,
) -> None:
    if os.name == "nt":
        # Closing a kill-on-close job removes any descendants that outlived the
        # direct child. The direct child has already been reaped above.
        if windows_job is not None:
            windows_job.close()
        return
    if not _signal_posix_group(process.pid, signal.SIGTERM):
        return
    # Preserve the long-standing normal-exit grace: descendants get at most
    # 100 ms to honor TERM before the whole group is made non-executable.
    time.sleep(min(grace_s, 0.1))
    _signal_posix_group(process.pid, signal.SIGKILL)
    _require_linux_process_group_quiescence(
        process.pid,
        timeout_s=max(grace_s, 1.0),
    )


def _cleanup_preserving_exception(
    process: subprocess.Popen[bytes],
    *,
    windows_job: _WindowsJobObject | None,
    grace_s: float,
    original_exception: BaseException,
) -> None:
    try:
        _terminate_process_tree(
            process,
            windows_job=windows_job,
            grace_s=grace_s,
        )
    except OwnedProcessCleanupIncomplete:
        raise
    except BaseException as cleanup_error:
        add_note = getattr(original_exception, "add_note", None)
        if callable(add_note):
            add_note(
                "runner process cleanup also failed: " + type(cleanup_error).__name__
            )


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    windows_job: _WindowsJobObject | None,
    grace_s: float,
) -> None:
    if process.poll() is None:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except (OSError, ValueError):
                pass
        else:
            _signal_posix_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass

    if os.name == "nt":
        if windows_job is not None:
            windows_job.terminate()
        elif process.poll() is None:
            _terminate_windows_process_tree(process.pid, grace_s=grace_s)
            if process.poll() is None:
                process.kill()
    else:
        _signal_posix_group(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()

    try:
        process.wait(timeout=max(grace_s, 1.0))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if os.name != "nt":
        _require_linux_process_group_quiescence(
            process.pid,
            timeout_s=max(grace_s, 1.0),
        )


def _signal_posix_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise
    return True


def _require_linux_process_group_quiescence(
    process_group_id: int,
    *,
    timeout_s: float,
) -> None:
    if not sys.platform.startswith("linux"):
        return
    if _wait_for_linux_process_group_quiescence(process_group_id, timeout_s):
        return
    live_process_ids = _linux_live_process_group_members(process_group_id)
    if not live_process_ids:
        time.sleep(0.02)
        live_process_ids = _linux_live_process_group_members(process_group_id)
        if not live_process_ids:
            return
    raise OwnedProcessCleanupIncomplete(
        process_group_id,
        live_process_ids,
    )


def _wait_for_linux_process_group_quiescence(
    process_group_id: int,
    timeout_s: float,
) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    consecutive_empty_scans = 0
    while True:
        if _linux_live_process_group_members(process_group_id):
            consecutive_empty_scans = 0
        else:
            consecutive_empty_scans += 1
            if consecutive_empty_scans >= 2:
                return True
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.02, remaining))


def _linux_live_process_group_members(process_group_id: int) -> tuple[int, ...]:
    """Return non-zombie Linux members that could still mutate inherited logs."""

    if not sys.platform.startswith("linux"):
        return ()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise OwnedProcessCleanupIncomplete(process_group_id, ()) from exc
    live: list[int] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # hidepid may conceal unrelated users. Owned same-UID group members
            # remain readable and are the only members relevant to this owner.
            continue
        except OSError as exc:
            raise OwnedProcessCleanupIncomplete(
                process_group_id,
                tuple(live),
            ) from exc
        closing_parenthesis = stat.rfind(b")")
        if closing_parenthesis < 0:
            raise OwnedProcessCleanupIncomplete(process_group_id, tuple(live))
        fields = stat[closing_parenthesis + 1 :].split()
        if len(fields) < 3:
            raise OwnedProcessCleanupIncomplete(process_group_id, tuple(live))
        try:
            pid = int(entry.name)
            member_group_id = int(fields[2])
        except ValueError as exc:
            raise OwnedProcessCleanupIncomplete(
                process_group_id,
                tuple(live),
            ) from exc
        state = fields[0]
        if member_group_id == process_group_id and state not in {b"Z", b"X", b"x"}:
            live.append(pid)
    return tuple(sorted(live))


def _terminate_windows_process_tree(pid: int, *, grace_s: float) -> None:
    """Best-effort fallback when the host forbids nested Job Objects."""

    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(grace_s, 1.0),
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        # The direct child is still killed and reaped by the caller below.
        return


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _LOG_TAIL_BYTES), os.SEEK_SET)
            payload = handle.read(_LOG_TAIL_BYTES)
    except OSError:
        return ""
    return payload.decode("utf-8", errors="replace")


class _WindowsJobObject:
    """Minimal kill-on-close Job Object wrapper; only constructed on Windows."""

    def __init__(self, handle: int) -> None:
        self._handle = handle

    @classmethod
    def create_and_assign(
        cls,
        process: subprocess.Popen[bytes],
    ) -> _WindowsJobObject | None:
        if os.name != "nt":
            return None
        if process.poll() is not None:
            return None
        try:
            job = cls(_create_kill_on_close_job())
            _assign_pid_to_job(job._handle, process.pid)
        except OSError as exc:
            if "job" in locals():
                job.close()
            # Codex and other process supervisors may already own the child in
            # a non-breakaway Job Object. In that environment a second Job
            # assignment is forbidden; retain ownership through the dedicated
            # process group and taskkill fallback instead of rejecting the job.
            if getattr(exc, "winerror", None) == 5 or exc.errno == 5:
                return None
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        return job

    def terminate(self) -> None:
        if self._handle == 0 or os.name != "nt":
            return
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.TerminateJobObject(self._handle, 1):
            error = ctypes.get_last_error()
            if error not in {0, 5, 87}:
                raise OSError(error, "TerminateJobObject failed")

    def close(self) -> None:
        if self._handle == 0 or os.name != "nt":
            return
        import ctypes

        handle = self._handle
        self._handle = 0
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)


def _create_kill_on_close_job() -> int:
    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        error = ctypes.get_last_error()
        raise OSError(error, "CreateJobObjectW failed")
    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise OSError(error, "SetInformationJobObject failed")
    return int(handle)


def _assign_pid_to_job(job_handle: int, pid: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    process_handle = kernel32.OpenProcess(0x0001 | 0x0100 | 0x1000, False, pid)
    if not process_handle:
        error = ctypes.get_last_error()
        raise OSError(error, "OpenProcess failed while assigning Job Object")
    try:
        if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
            error = ctypes.get_last_error()
            raise OSError(error, "AssignProcessToJobObject failed")
    finally:
        kernel32.CloseHandle(process_handle)


__all__ = [
    "OwnedProcessCleanupIncomplete",
    "OwnedProcessLaunchError",
    "OwnedProcessResult",
    "OwnedProcessTimeout",
    "run_owned_process",
]

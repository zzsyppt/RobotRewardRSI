"""Small runner-owned process lifecycle primitive for long train/eval jobs."""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence


_LOG_TAIL_BYTES = 16 * 1024


class OwnedProcessLaunchError(OSError):
    """Raised when a process could not be launched under runner ownership."""


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


def run_owned_process(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    timeout_s: int,
    log_dir: Path,
    termination_grace_s: float = 5.0,
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
            )
            if os.name == "nt":
                windows_job = _WindowsJobObject.create_and_assign(process)
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
        except (OwnedProcessTimeout, OwnedProcessLaunchError):
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
) -> subprocess.Popen[bytes]:
    creationflags = 0
    kwargs: dict[str, object] = {}
    spawn_argv = argv
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
        spawn_argv = _linux_parent_death_guard_argv(argv)
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
    if _signal_posix_group(process.pid, signal.SIGTERM):
        time.sleep(min(grace_s, 0.1))
        _signal_posix_group(process.pid, signal.SIGKILL)


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
    except BaseException as cleanup_error:
        add_note = getattr(original_exception, "add_note", None)
        if callable(add_note):
            add_note(
                "runner process cleanup also failed: "
                + type(cleanup_error).__name__
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
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )]

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
    "OwnedProcessLaunchError",
    "OwnedProcessResult",
    "OwnedProcessTimeout",
    "run_owned_process",
]

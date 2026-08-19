"""Small cross-platform process-held file locks for runner-owned state."""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path
from typing import BinaryIO


class ProcessFileLockUnavailableError(RuntimeError):
    """Raised when a non-blocking process file lock is already owned."""


class ProcessFileLock:
    """An OS lock whose ownership ends automatically when the process exits."""

    def __init__(
        self,
        *,
        path: Path,
        handle: BinaryIO,
        payload: bytes | None,
        thread_lock: threading.Lock,
    ) -> None:
        self.path = path
        self._handle = handle
        self._payload = payload
        self._thread_lock = thread_lock

    def assert_owned(self) -> None:
        if self._handle.closed:
            raise RuntimeError("process file lock is closed")
        if self._payload is None:
            return
        self._handle.seek(0)
        if self._handle.read() != self._payload:
            raise RuntimeError("process file lock ownership payload changed")

    def replace_payload(self, payload: bytes) -> None:
        """Replace metadata through the descriptor that owns the OS lock."""

        if self._handle.closed:
            raise RuntimeError("process file lock is closed")
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("process file lock payload must be non-empty bytes")
        self.assert_owned()
        self._handle.seek(0)
        self._handle.write(payload)
        self._handle.truncate()
        os.fsync(self._handle.fileno())
        self._payload = payload

    def release(self) -> None:
        if self._handle.closed:
            return
        try:
            _unlock_file(self._handle)
        finally:
            try:
                self._handle.close()
            finally:
                self._thread_lock.release()

    def __enter__(self) -> ProcessFileLock:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback
        self.release()


def acquire_process_file_lock(
    path: Path,
    *,
    blocking: bool,
    payload: bytes | None = None,
) -> ProcessFileLock:
    """Acquire one byte of an OS-locked sidecar file.

    The sidecar is intentionally durable. A stale file is harmless because lock
    ownership is held by the OS descriptor rather than by file existence.
    """

    resolved = path.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock_for(resolved)
    if not thread_lock.acquire(blocking=blocking):
        raise ProcessFileLockUnavailableError(
            f"process file lock is already owned: {resolved}"
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(resolved, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        thread_lock.release()
        raise
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"{}")
            os.fsync(handle.fileno())
        handle.seek(0)
        if not _try_lock_file(handle, blocking=blocking):
            raise ProcessFileLockUnavailableError(
                f"process file lock is already owned: {resolved}"
            )
        if payload is not None:
            handle.seek(0)
            handle.write(payload)
            handle.truncate()
            os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        thread_lock.release()
        raise
    return ProcessFileLock(
        path=resolved,
        handle=handle,
        payload=payload,
        thread_lock=thread_lock,
    )


def _try_lock_file(handle: BinaryIO, *, blocking: bool) -> bool:
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), mode, 1)
        except OSError as exc:
            if not blocking and exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
                errno.EDEADLK,
            }:
                return False
            raise
        return True

    import fcntl

    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError:
        return False
    return True


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


__all__ = [
    "ProcessFileLock",
    "ProcessFileLockUnavailableError",
    "acquire_process_file_lock",
]

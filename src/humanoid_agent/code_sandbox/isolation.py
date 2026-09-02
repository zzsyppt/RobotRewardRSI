"""Phase 8C process, environment, network, and filesystem isolation helpers.

The server does not currently allow Docker for the user and does not allow
unprivileged bind mounts.  Phase 8C therefore uses the primitives that are
actually available on the server:

* a separate subprocess for candidate-authored code;
* a sanitized environment allowlist;
* Linux ``unshare -Urnfp --mount-proc`` for user/network/PID namespaces when
  available;
* Linux Landlock for per-process filesystem restrictions when available.

If a required primitive is unavailable, callers must fail closed instead of
pretending that isolation was enforced.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13

LANDLOCK_READ_EXECUTE = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
LANDLOCK_WRITE_TREE = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM
)

PR_SET_NO_NEW_PRIVS = 38


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    timed_out: bool
    elapsed_seconds: float
    command: list[str]
    log_file: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "elapsed_seconds": self.elapsed_seconds,
            "command": self.command,
            "log_file": self.log_file,
        }


class IsolationError(RuntimeError):
    """Raised when required isolation cannot be applied."""


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def landlock_abi_version() -> int | None:
    """Return the Landlock ABI version, or ``None`` when unavailable."""

    if not is_linux():
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(_sys_landlock_create_ruleset(), 0, 0, LANDLOCK_CREATE_RULESET_VERSION)
    if result < 0:
        err = ctypes.get_errno()
        if err in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            return None
        return None
    return int(result)


def network_namespace_supported() -> bool:
    """Return whether the current user can create a no-network namespace."""

    if not is_linux() or not command_exists("unshare"):
        return False
    completed = subprocess.run(
        ["unshare", "-Urn", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def pid_namespace_supported() -> bool:
    """Return whether ``unshare`` can create a PID namespace for cleanup."""

    if not is_linux() or not command_exists("unshare"):
        return False
    completed = subprocess.run(
        ["unshare", "-Urnfp", "--mount-proc", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def probe_isolation_capabilities() -> dict[str, Any]:
    """Return a machine-readable capability report."""

    return {
        "platform": platform.platform(),
        "is_linux": is_linux(),
        "commands": {
            "unshare": shutil.which("unshare"),
            "docker": shutil.which("docker"),
            "bwrap": shutil.which("bwrap"),
            "firejail": shutil.which("firejail"),
        },
        "network_namespace_supported": network_namespace_supported(),
        "pid_namespace_supported": pid_namespace_supported(),
        "landlock_abi_version": landlock_abi_version(),
    }


def default_landlock_read_paths(*, repo_root: Path, candidate_dir: Path) -> list[Path]:
    """Return read/execute paths needed by Python and the candidate API.

    Do not grant the whole repository.  Repository-level configs contain
    protected evaluation profiles and seeds; candidates should only see the
    minimal API surface required to build ``TrainingComponents``.
    """

    candidates = [
        candidate_dir,
        repo_root / "src" / "humanoid_agent" / "__init__.py",
        repo_root / "src" / "humanoid_agent" / "code_sandbox",
        Path(sys.executable).resolve(),
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path("/usr"),
        Path("/bin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ld.so.cache"),
        *_runtime_device_read_paths(),
    ]
    paths: list[Path] = []
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in paths:
            paths.append(resolved)
    for item in sys.path:
        if not item:
            continue
        path = Path(item)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if _is_relative_to(resolved, repo_root):
            continue
        if resolved.exists() and resolved not in paths:
            paths.append(resolved)
    return paths


def _runtime_device_read_paths() -> list[Path]:
    """Return non-secret device nodes required by Python/torch initialization."""

    return [Path("/dev/urandom"), Path("/dev/random")]


def default_training_landlock_read_paths(
    *,
    repo_root: Path,
    candidate_dir: Path,
    allow_cuda_device_paths: bool = True,
) -> list[Path]:
    """Return read/execute paths for an isolated training process.

    This keeps the minimal candidate API policy from ``default_landlock_read_paths``
    and optionally adds NVIDIA device nodes needed by future CUDA training.  It
    does not grant whole-repository read access.
    """

    paths = default_landlock_read_paths(repo_root=repo_root, candidate_dir=candidate_dir)
    if allow_cuda_device_paths:
        for pattern in ("/dev/nvidia*", "/dev/dri"):
            for item in sorted(Path("/").glob(pattern.lstrip("/"))):
                try:
                    resolved = item.resolve()
                except OSError:
                    continue
                if resolved.exists() and resolved not in paths:
                    paths.append(resolved)
    for item in _runtime_device_read_paths():
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in paths:
            paths.append(resolved)
    return paths


def default_training_landlock_read_write_paths(*, run_dir: Path) -> list[Path]:
    """Return safe read/write paths for an isolated training process."""

    candidates = [run_dir, Path("/dev/null")]
    paths: list[Path] = []
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in paths:
            paths.append(resolved)
    return paths


def apply_landlock(*, read_paths: list[Path], read_write_paths: list[Path]) -> dict[str, Any]:
    """Restrict the current process using Landlock.

    The ruleset allows read/execute access to ``read_paths`` and read/write
    access to ``read_write_paths``.  Anything outside those paths is denied for
    filesystem operations handled by the active Landlock ABI.
    """

    abi = landlock_abi_version()
    if abi is None:
        raise IsolationError("Landlock is unavailable on this host")

    handled_access = LANDLOCK_READ_EXECUTE | LANDLOCK_WRITE_TREE
    if abi >= 2:
        handled_access |= LANDLOCK_ACCESS_FS_REFER

    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled_access)
    ruleset_fd = libc.syscall(
        _sys_landlock_create_ruleset(),
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        raise _errno_error("landlock_create_ruleset")

    added_rules: list[dict[str, Any]] = []
    try:
        for path in _dedupe_existing_paths(read_paths):
            _add_landlock_path_rule(libc, ruleset_fd, path, LANDLOCK_READ_EXECUTE)
            added_rules.append({"path": str(path), "access": "read_execute"})
        for path in _dedupe_existing_paths(read_write_paths):
            _add_landlock_path_rule(libc, ruleset_fd, path, handled_access)
            added_rules.append({"path": str(path), "access": "read_write"})

        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise _errno_error("prctl(PR_SET_NO_NEW_PRIVS)")
        if libc.syscall(_sys_landlock_restrict_self(), ruleset_fd, 0) != 0:
            raise _errno_error("landlock_restrict_self")
    finally:
        os.close(ruleset_fd)

    return {
        "enabled": True,
        "abi_version": abi,
        "handled_access_fs": handled_access,
        "rules": added_rules,
    }


def sanitized_environment(*, repo_root: Path, allowed_names: list[str] | None = None) -> dict[str, str]:
    """Return an environment that excludes secrets and keeps imports working."""

    default_allowed = [
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    ]
    allowed = set(allowed_names or default_allowed)
    env: dict[str, str] = {}
    for key in allowed:
        value = os.environ.get(key)
        if value is not None and not _looks_secret(key):
            env[key] = value
    src = str((repo_root / "src").resolve())
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [src]
    pythonpath_parts.extend(part for part in existing_pythonpath.split(os.pathsep) if part and part != src)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def isolated_command_prefix(*, require_network_namespace: bool, use_pid_namespace: bool = True) -> list[str]:
    """Return the command prefix for namespace isolation."""

    if not require_network_namespace and not use_pid_namespace:
        return []
    if not is_linux() or not command_exists("unshare"):
        raise IsolationError("unshare is unavailable")
    if require_network_namespace and use_pid_namespace:
        return ["unshare", "-Urnfp", "--mount-proc"]
    if require_network_namespace:
        return ["unshare", "-Urn"]
    return ["unshare", "-Urfp", "--mount-proc"]


def run_isolated_subprocess(
    command: list[str],
    *,
    cwd: Path,
    log_file: Path,
    env: dict[str, str],
    timeout_seconds: int,
    require_network_namespace: bool,
    use_pid_namespace: bool = True,
) -> CommandResult:
    """Run a command with namespace prefix, timeout, and process-group cleanup."""

    full_command = [
        *isolated_command_prefix(
            require_network_namespace=require_network_namespace,
            use_pid_namespace=use_pid_namespace,
        ),
        *command,
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            full_command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            exit_code = process.wait(timeout=5)
        finally:
            _terminate_process_group(process)
    elapsed = time.monotonic() - start
    return CommandResult(
        exit_code=int(exit_code),
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        command=full_command,
        log_file=str(log_file),
    )


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _add_landlock_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: Path, allowed_access: int) -> None:
    if path.is_file():
        allowed_access &= (
            LANDLOCK_ACCESS_FS_EXECUTE
            | LANDLOCK_ACCESS_FS_READ_FILE
            | LANDLOCK_ACCESS_FS_WRITE_FILE
            | LANDLOCK_ACCESS_FS_REMOVE_FILE
        )
    elif not path.is_dir():
        # Character/block devices such as /dev/nvidia* are not directories and
        # cannot accept directory/execute rights in Landlock path-beneath rules.
        allowed_access &= LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE
    flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        rule = _LandlockPathBeneathAttr(allowed_access=allowed_access, parent_fd=fd)
        if libc.syscall(
            _sys_landlock_add_rule(),
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            0,
        ) != 0:
            raise _errno_error(f"landlock_add_rule({path})")
    finally:
        os.close(fd)


def _dedupe_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def _looks_secret(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))


def _errno_error(label: str) -> OSError:
    return OSError(ctypes.get_errno(), f"{label}: {os.strerror(ctypes.get_errno())}")


def _sys_landlock_create_ruleset() -> int:
    return 444


def _sys_landlock_add_rule() -> int:
    return 445


def _sys_landlock_restrict_self() -> int:
    return 446

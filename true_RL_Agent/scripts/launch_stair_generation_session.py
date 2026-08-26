"""Launch one Stair generation attempt with immutable private evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_stair_generation_session import (  # noqa: E402
    parse_stair_generation_session_args,
)

from true_rl_agent.runner.process_handle import (  # noqa: E402
    OwnedProcessCleanupIncomplete,
    OwnedProcessResult,
    run_owned_process,
)
from true_rl_agent.tasks.stair_climbing.launch_context import (  # noqa: E402
    STAIR_GENERATION_LAUNCH_START_PATH_ENV,
    STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION,
    STAIR_GENERATION_LAUNCH_START_SHA256_ENV,
    stair_generation_command_hash,
)
from true_rl_agent.tasks.stair_climbing.launch_readiness import (  # noqa: E402
    STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION,
    STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
    load_stair_generation_launch_readiness_receipt,
    resolve_stair_nvidia_smi_executable,
)
from true_rl_agent.tasks.stair_climbing.source_checkout import (  # noqa: E402
    inspect_stair_source_checkout,
)

GENERATION_ENTRYPOINT = PROJECT_ROOT / "scripts" / "run_stair_generation_session.py"
ATTEMPT_EXIT_SCHEMA_VERSION = "true_rl_agent.stair.generation_launcher_attempt_exit.v3"
ATTEMPT_START_SCHEMA_VERSION = STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION
P6D_READINESS_CONSUMPTION_SCHEMA_VERSION = (
    "true_rl_agent.stair.p6d_launch_readiness_consumption.v1"
)


class _LauncherTerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"launcher received signal {signum}")
        self.signum = signum


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    args.launcher_argv = _launcher_argv(raw_argv, explicit=argv is not None)
    if not args.python_executable.is_absolute():
        raise ValueError("python_executable must be absolute")
    python_executable = args.python_executable.resolve(strict=True)
    if not python_executable.is_file():
        raise ValueError("python_executable must be a regular file")
    if not args.attempt_dir.is_absolute():
        raise ValueError("attempt_dir must be absolute")
    attempt_parent = args.attempt_dir.resolve(strict=False)
    generation_args = list(args.generation_args)
    if generation_args[:1] == ["--"]:
        generation_args = generation_args[1:]
    if not generation_args:
        raise ValueError("generation entrypoint arguments are required after --")
    source_identity = _source_identity()
    host_identity = _host_identity()
    formal_identity = _formal_generation_identity(generation_args)
    launch_readiness_receipt = _validated_launch_readiness_ref(
        receipt_path=args.launch_readiness_receipt,
        formal_identity=formal_identity,
        source_identity=source_identity,
        host_identity=host_identity,
        attempt_parent=attempt_parent,
        formal_generation_command_sha256=stair_generation_command_hash(
            [
                str(python_executable),
                "-u",
                str(GENERATION_ENTRYPOINT),
                *generation_args,
            ]
        ),
    )
    attempt_parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    with _graceful_sigterm():
        return _run_attempt(
            args=args,
            python_executable=python_executable,
            attempt_parent=attempt_parent,
            generation_args=generation_args,
            source_identity=source_identity,
            host_identity=host_identity,
            launch_readiness_receipt=launch_readiness_receipt,
            formal_identity=formal_identity,
        )


def _run_attempt(
    *,
    args: argparse.Namespace,
    python_executable: Path,
    attempt_parent: Path,
    generation_args: list[str],
    source_identity: dict[str, object],
    host_identity: dict[str, object],
    launch_readiness_receipt: dict[str, object] | None,
    formal_identity: dict[str, object] | None,
) -> int:
    attempt_id, attempt_path = _create_attempt_directory(attempt_parent)
    launcher_log_path = attempt_path / "launcher.log"
    launcher_log = _open_text_exclusive(launcher_log_path)
    child_log_dir = attempt_path / "child"
    command = [
        str(python_executable),
        "-u",
        str(GENERATION_ENTRYPOINT),
        *generation_args,
    ]
    command_sha256 = _command_hash(command)
    started_at = _utc_now()
    process_result: OwnedProcessResult | None = None
    launcher_exception: BaseException | None = None
    start_receipt: dict[str, object] | None = None
    try:
        start_receipt = _write_json_exclusive(
            attempt_path / "start.json",
            _build_start_receipt(
                attempt_id=attempt_id,
                started_at_utc=started_at,
                launcher_argv=list(args.launcher_argv),
                child_command=command,
                timeout_s=args.timeout_s,
                source_identity=source_identity,
                host_identity=host_identity,
                launch_readiness_receipt=launch_readiness_receipt,
            ),
        )
        _write_log_event(
            launcher_log,
            {
                "event": "launcher_attempt_started",
                "attempt_id": attempt_id,
                "started_at_utc": started_at,
                "entrypoint": str(GENERATION_ENTRYPOINT),
                "command_sha256": command_sha256,
                "start_receipt": start_receipt,
                "timeout_s": args.timeout_s,
            },
        )
        process_result = run_owned_process(
            command,
            cwd=PROJECT_ROOT,
            timeout_s=args.timeout_s,
            log_dir=child_log_dir,
            termination_grace_s=10.0,
            environment_overrides=_child_environment(
                start_receipt,
                formal=launch_readiness_receipt is not None,
            ),
        )
    except BaseException as exc:
        # run_owned_process does not return or re-raise until its owned process
        # tree has been terminated. The traceback is stable before any receipt
        # checksum is calculated below.
        launcher_exception = exc
        launcher_log.write("\n")
        traceback.print_exc(file=launcher_log)
    finally:
        ended_at = _utc_now()
        _write_log_event(
            launcher_log,
            {
                "event": "launcher_attempt_finished",
                "attempt_id": attempt_id,
                "ended_at_utc": ended_at,
                "returncode": _attempt_returncode(
                    process_result,
                    launcher_exception,
                ),
                "process_id": (
                    process_result.pid if process_result is not None else None
                ),
                "launcher_exception_type": (
                    type(launcher_exception).__name__
                    if launcher_exception is not None
                    else None
                ),
            },
        )
        launcher_log.close()

    stdout_log_path = child_log_dir / "stdout.log"
    stderr_log_path = child_log_dir / "stderr.log"
    returncode = _attempt_returncode(
        process_result,
        launcher_exception,
    )
    cleanup_incomplete = isinstance(
        launcher_exception,
        OwnedProcessCleanupIncomplete,
    )
    exit_path = attempt_path / "exit.json"
    receipt = {
        "schema_version": ATTEMPT_EXIT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "entrypoint": str(GENERATION_ENTRYPOINT),
        "command_sha256": command_sha256,
        "start_receipt": start_receipt,
        "timeout_s": args.timeout_s,
        "process_id": (process_result.pid if process_result is not None else None),
        "logs": {
            "launcher": _log_receipt(launcher_log_path),
            "stdout": _log_receipt(
                stdout_log_path,
                stable=not cleanup_incomplete,
            ),
            "stderr": _log_receipt(
                stderr_log_path,
                stable=not cleanup_incomplete,
            ),
        },
        "returncode": returncode,
        "status": (
            "cleanup_incomplete"
            if cleanup_incomplete
            else "terminated"
            if isinstance(launcher_exception, _LauncherTerminationRequested)
            else "launcher_exception"
            if launcher_exception is not None
            else "completed"
            if returncode == 0
            else "failed"
        ),
        "launcher_exception_type": (
            type(launcher_exception).__name__
            if launcher_exception is not None
            else None
        ),
    }
    _write_json_exclusive(exit_path, receipt)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "attempt_path": str(attempt_path),
                "launcher_log_path": str(launcher_log_path),
                "start_receipt_path": (
                    start_receipt["path"] if start_receipt is not None else None
                ),
                "stdout_log_path": str(stdout_log_path),
                "stderr_log_path": str(stderr_log_path),
                "exit_receipt_path": str(exit_path),
                "returncode": returncode,
            },
            sort_keys=True,
        )
    )
    if isinstance(launcher_exception, _LauncherTerminationRequested):
        return 128 + launcher_exception.signum
    if launcher_exception is not None:
        raise launcher_exception
    assert returncode is not None
    return returncode


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the tracked Stair generation entrypoint with one immutable "
            "attempt directory and exit receipt per physical attempt."
        )
    )
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument(
        "--launch-readiness-receipt",
        type=Path,
        default=None,
        help=(
            "Required for a fresh formal generation. The immutable receipt must "
            "bind this source, host, output, session, GPU pair, lease root, and "
            "output cap before an attempt directory or child process is created."
        ),
    )
    parser.add_argument("generation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return args


def _validated_launch_readiness_ref(
    *,
    receipt_path: Path | None,
    formal_identity: dict[str, object] | None,
    source_identity: dict[str, object],
    host_identity: dict[str, object],
    attempt_parent: Path,
    formal_generation_command_sha256: str,
) -> dict[str, object] | None:
    if formal_identity is None:
        if receipt_path is not None:
            raise ValueError(
                "--launch-readiness-receipt is only valid for a formal generation"
            )
        return None
    output_dir = formal_identity["output_dir"]
    assert isinstance(output_dir, Path)
    if _path_is_within(attempt_parent, output_dir):
        raise ValueError("formal launcher attempt directory must be outside output")
    if receipt_path is None:
        raise ValueError(
            "--launch-readiness-receipt is required for a formal generation"
        )
    if source_identity.get("commit") != formal_identity["expected_source_commit"]:
        raise ValueError("launcher source HEAD does not match the formal generation")
    if source_identity.get("tracked_clean") is not True:
        raise ValueError(
            "launcher execution source must be clean for a formal generation"
        )
    if "CUDA_VISIBLE_DEVICES" in os.environ or "NVIDIA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("formal generation launcher environment must not remap GPUs")
    subject_modes = formal_identity.get("launch_subject_modes")
    if not isinstance(subject_modes, tuple):
        raise ValueError("formal generation launch subject is invalid")
    if subject_modes == (STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,):
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("fresh formal generation output already exists")
    elif set(subject_modes).issubset(
        {
            STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
        }
    ):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise FileNotFoundError(
                "formal resume requires an existing real output directory"
            )
    else:
        raise ValueError("formal generation launch subject is invalid")
    hostname = host_identity.get("hostname")
    boot_id = host_identity.get("boot_id")
    machine_id = host_identity.get("machine_id")
    if (
        not isinstance(hostname, str)
        or not isinstance(boot_id, str)
        or not isinstance(machine_id, str)
    ):
        raise ValueError("formal generation host identity is unavailable")
    if machine_id != formal_identity["machine_id"]:
        raise ValueError("formal generation machine identity mismatch")
    receipt = load_stair_generation_launch_readiness_receipt(
        receipt_path,
        expected_formal_output_dir=output_dir,
        expected_source_root=PROJECT_ROOT,
        expected_session_id=str(formal_identity["session_id"]),
        expected_source_commit=str(formal_identity["expected_source_commit"]),
        expected_machine_id=str(formal_identity["machine_id"]),
        expected_gpu_uuids=(
            str(formal_identity["gpu0_uuid"]),
            str(formal_identity["gpu1_uuid"]),
        ),
        expected_lease_root=formal_identity["lease_root"],
        expected_max_output_bytes=int(formal_identity["max_output_bytes"]),
        expected_nvidia_smi_executable=formal_identity["nvidia_smi_executable"],
        expected_hostname=hostname,
        expected_boot_id=boot_id,
        expected_subject_mode=(
            str(subject_modes[0]) if len(subject_modes) == 1 else None
        ),
        expected_subject_modes=(subject_modes if len(subject_modes) > 1 else None),
        expected_formal_generation_command_sha256=(
            formal_generation_command_sha256 if subject_modes else None
        ),
    )
    receipt.assert_fresh()
    if (
        receipt.payload.get("schema_version")
        != STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION
    ):
        raise ValueError("formal launcher requires a P6-D readiness v3 receipt")
    receipt_ref = receipt.to_ref(receipt_path).to_dict()
    _consume_p6d_readiness_once(
        receipt_path=receipt_path,
        receipt_ref=receipt_ref,
        formal_generation_command_sha256=formal_generation_command_sha256,
    )
    return receipt_ref


def _consume_p6d_readiness_once(
    *,
    receipt_path: Path,
    receipt_ref: dict[str, object],
    formal_generation_command_sha256: str,
) -> Path:
    target = Path(str(receipt_path.resolve(strict=True)) + ".consumed.json")
    if target.exists() or target.is_symlink():
        raise FileExistsError("P6-D launch readiness receipt was already consumed")
    _write_json_exclusive(
        target,
        {
            "schema_version": P6D_READINESS_CONSUMPTION_SCHEMA_VERSION,
            "consumed_at_utc": _utc_now(),
            "readiness_receipt": dict(receipt_ref),
            "formal_generation_command_sha256": (
                formal_generation_command_sha256
            ),
            "launcher_process_id": os.getpid(),
            "formal_output_writes": 0,
            "ledger_writes": 0,
            "acceptance_authority": False,
        },
    )
    return target


def _formal_generation_identity(
    generation_args: list[str],
) -> dict[str, object] | None:
    if not _has_generation_option(generation_args, "--output-dir"):
        return None
    if any(
        _has_generation_option(generation_args, option)
        for option in (
            "--help",
            "-h",
            "--preflight-only",
            "--provider-readiness-only",
        )
    ):
        return None
    parsed = parse_stair_generation_session_args(generation_args)
    if parsed.terminal_audit_only and not parsed.formal_session:
        return None
    multi_generation_session = parsed.formal_session
    if parsed.resume_existing and not multi_generation_session:
        raise ValueError("formal resume requires a multi-generation session")
    for field_name in ("output_dir", "lease_root", "provider_env_file"):
        value = getattr(parsed, field_name)
        if not isinstance(value, Path) or not value.is_absolute():
            raise ValueError(
                f"formal generation --{field_name.replace('_', '-')} must be absolute"
            )
    return {
        "output_dir": parsed.output_dir.resolve(strict=False),
        "session_id": parsed.session_id,
        "expected_source_commit": parsed.expected_source_commit,
        "machine_id": parsed.machine_id,
        "gpu0_uuid": parsed.gpu0_uuid,
        "gpu1_uuid": parsed.gpu1_uuid,
        "lease_root": parsed.lease_root.resolve(strict=False),
        "max_output_bytes": parsed.max_output_bytes,
        "nvidia_smi_executable": resolve_stair_nvidia_smi_executable(
            parsed.nvidia_smi_executable
        ),
        "formal_session": parsed.formal_session,
        "launch_subject_modes": (
            (STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,)
            if parsed.terminal_audit_only
            else (
                STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
                STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            )
            if parsed.resume_existing
            else (STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,)
        ),
    }


def _has_generation_option(arguments: list[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in arguments)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _create_attempt_directory(parent: Path) -> tuple[str, Path]:
    for _ in range(8):
        attempt_id = (
            "attempt-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"-pid{os.getpid()}-"
            + secrets.token_hex(4)
        )
        path = parent / attempt_id
        try:
            path.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            continue
        return attempt_id, path
    raise FileExistsError("could not allocate a unique launcher attempt directory")


@contextmanager
def _graceful_sigterm() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)
    requested = False

    def request_termination(signum: int, _frame: FrameType | None) -> None:
        nonlocal requested
        if requested:
            return
        requested = True
        # Do not let repeated TERM re-enter the bounded process-group cleanup.
        # SIGKILL remains the supervisor's unconditional escape hatch.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise _LauncherTerminationRequested(signum)

    signal.signal(signal.SIGTERM, request_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _open_text_exclusive(path: Path):
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _attempt_returncode(
    result: OwnedProcessResult | None,
    launcher_exception: BaseException | None,
) -> int | None:
    if result is not None:
        return result.returncode
    if isinstance(launcher_exception, _LauncherTerminationRequested):
        return 128 + launcher_exception.signum
    return None


def _write_log_event(handle, event: dict[str, object]) -> None:
    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_json_exclusive(
    path: Path,
    value: dict[str, object],
) -> dict[str, object]:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write_count = handle.write(payload)
            if write_count != len(payload):
                raise OSError("short write while persisting launcher exit receipt")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _build_start_receipt(
    *,
    attempt_id: str,
    started_at_utc: str,
    launcher_argv: list[str],
    child_command: list[str],
    timeout_s: int,
    source_identity: dict[str, object],
    host_identity: dict[str, object],
    launch_readiness_receipt: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": ATTEMPT_START_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "started_at_utc": started_at_utc,
        "launcher": {
            "argv": launcher_argv,
            "cwd": str(Path.cwd()),
            "process_id": os.getpid(),
            "process_group_id": _process_identity_value("getpgid"),
            "session_id": _process_identity_value("getsid"),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "child": {
            "argv": child_command,
            "command_sha256": _command_hash(child_command),
            "cwd": str(PROJECT_ROOT),
            "timeout_s": timeout_s,
        },
        "host": host_identity,
        "source": source_identity,
        "launch_readiness_receipt": launch_readiness_receipt,
    }


def _launcher_argv(raw_argv: list[str], *, explicit: bool) -> list[str]:
    if not explicit:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return [str(value) for value in original]
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *raw_argv,
    ]


def _process_identity_value(function_name: str) -> int | None:
    function = getattr(os, function_name, None)
    if function is None:
        return None
    try:
        return int(function(0))
    except OSError:
        return None


def _host_identity() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "boot_id": _read_optional_identity(Path("/proc/sys/kernel/random/boot_id")),
        "machine_id": _read_optional_identity(Path("/etc/machine-id")),
    }


def _read_optional_identity(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    return value or None


def _source_identity() -> dict[str, object]:
    checkout = inspect_stair_source_checkout(PROJECT_ROOT)
    return {
        "commit": checkout.head_commit,
        "tracked_clean": checkout.clean,
        "files": {
            _source_path_label(path): {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in _critical_source_paths()
        },
    }


def _source_path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        # Test and diagnostic launchers may intentionally substitute a helper
        # entrypoint outside the checkout. Formal Stair launches still resolve
        # every critical path under PROJECT_ROOT.
        return str(path)


def _critical_source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        GENERATION_ENTRYPOINT.resolve(strict=True),
        (SRC_ROOT / "true_rl_agent" / "runner" / "process_handle.py").resolve(
            strict=True
        ),
        (
            SRC_ROOT
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "launch_context.py"
        ).resolve(strict=True),
        (
            SRC_ROOT
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "launch_readiness.py"
        ).resolve(strict=True),
        (
            SRC_ROOT
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "generation_session.py"
        ).resolve(strict=True),
        (
            SRC_ROOT
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "source_checkout.py"
        ).resolve(strict=True),
    )


def _log_receipt(
    path: Path,
    *,
    stable: bool = True,
) -> dict[str, str | None] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": _file_sha256(path) if stable else None,
    }


def _command_hash(command: list[str]) -> str:
    return stair_generation_command_hash(command)


def _child_environment(
    start_receipt: dict[str, object],
    *,
    formal: bool,
) -> dict[str, str | None]:
    environment: dict[str, str | None] = {
        "PYTHONUNBUFFERED": "1",
        STAIR_GENERATION_LAUNCH_START_PATH_ENV: None,
        STAIR_GENERATION_LAUNCH_START_SHA256_ENV: None,
    }
    if not formal:
        return environment
    start_path = start_receipt.get("path")
    start_sha256 = start_receipt.get("sha256")
    if not isinstance(start_path, str) or not isinstance(start_sha256, str):
        raise ValueError("launcher start receipt reference is invalid")
    environment[STAIR_GENERATION_LAUNCH_START_PATH_ENV] = start_path
    environment[STAIR_GENERATION_LAUNCH_START_SHA256_ENV] = start_sha256
    return environment


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

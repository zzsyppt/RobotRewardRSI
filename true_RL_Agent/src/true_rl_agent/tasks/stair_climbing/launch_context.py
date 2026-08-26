"""Consume launcher-owned provenance before a formal Stair session can run."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import require_sha256_checksum
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.tasks.stair_climbing.launch_readiness import (
    STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
    StairGenerationLaunchReadinessReceiptRef,
    load_stair_generation_launch_readiness_receipt,
    resolve_stair_nvidia_smi_executable,
)

STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_launcher_attempt_start.v1"
)
STAIR_GENERATION_LAUNCH_START_PATH_ENV = (
    "TRUE_RL_AGENT_STAIR_GENERATION_LAUNCH_START_PATH"
)
STAIR_GENERATION_LAUNCH_START_SHA256_ENV = (
    "TRUE_RL_AGENT_STAIR_GENERATION_LAUNCH_START_SHA256"
)
MAX_LAUNCH_START_RECEIPT_BYTES = 1024 * 1024

_ROOT_KEYS = {
    "schema_version",
    "attempt_id",
    "started_at_utc",
    "launcher",
    "child",
    "host",
    "source",
    "launch_readiness_receipt",
}
_LAUNCHER_KEYS = {
    "argv",
    "cwd",
    "process_id",
    "process_group_id",
    "session_id",
    "python_executable",
}
_CHILD_KEYS = {"argv", "command_sha256", "cwd", "timeout_s"}
_HOST_KEYS = {"hostname", "boot_id", "machine_id"}
_SOURCE_KEYS = {"commit", "tracked_clean", "files"}


def stair_generation_command_hash(command: Sequence[str]) -> str:
    if (
        not isinstance(command, Sequence)
        or isinstance(command, str | bytes)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("Stair generation command must contain non-empty text")
    payload = json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def consume_stair_generation_launch_context(
    *,
    process_argv: Sequence[str],
    project_root: Path,
    formal_output_dir: Path,
    session_id: str,
    expected_source_commit: str,
    expected_machine_id: str,
    expected_gpu_uuids: tuple[str, str],
    lease_root: Path,
    max_output_bytes: int,
    nvidia_smi_executable: Path | str,
    helper_mode: bool,
    resume_existing: bool,
    formal_session: bool = False,
    terminal_audit_only: bool = False,
    environ: MutableMapping[str, str] | None = None,
    expected_launcher_parent_pid: int | None = None,
) -> dict[str, Any] | None:
    """Validate and consume the internal launch receipt before side effects."""

    environment = os.environ if environ is None else environ
    if terminal_audit_only and not formal_session:
        raise ValueError("terminal audit-only requires a multi-generation session")
    multi_generation_session = formal_session
    path_text = environment.get(STAIR_GENERATION_LAUNCH_START_PATH_ENV)
    checksum = environment.get(STAIR_GENERATION_LAUNCH_START_SHA256_ENV)
    if helper_mode:
        if path_text is not None or checksum is not None:
            raise ValueError("Stair helper mode rejects formal launch context")
        return None
    if resume_existing and not multi_generation_session:
        raise ValueError("formal resume requires a multi-generation session")
    if path_text is None or checksum is None:
        raise ValueError(
            "formal Stair generation must be launched by the tracked launcher"
        )
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("Stair formal launch start path is invalid")
    start_path = Path(path_text)
    if not start_path.is_absolute():
        raise ValueError("Stair formal launch start path must be absolute")
    checksum = require_sha256_checksum(
        checksum,
        "Stair formal launch start receipt sha256",
    )
    payload = read_bounded_regular_file(
        start_path,
        max_bytes=MAX_LAUNCH_START_RECEIPT_BYTES,
        label="Stair formal launch start receipt",
    )
    if "sha256:" + hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("Stair formal launch start receipt checksum mismatch")
    start = strict_json_object_loads(
        payload,
        context="Stair formal launch start receipt",
    )
    if payload != canonical_json_bytes(start, trailing_newline=True):
        raise ValueError("Stair formal launch start receipt must be canonical")
    root = _exact_mapping(start, _ROOT_KEYS, "Stair formal launch start receipt")
    if root["schema_version"] != STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION:
        raise ValueError("Stair formal launch start receipt schema mismatch")
    _bounded_text(root["attempt_id"], "Stair formal launch attempt id")
    _bounded_text(root["started_at_utc"], "Stair formal launch start time")

    launcher = _exact_mapping(root["launcher"], _LAUNCHER_KEYS, "Stair launcher")
    launcher_pid = _positive_int(launcher["process_id"], "Stair launcher process id")
    expected_parent = (
        os.getppid()
        if expected_launcher_parent_pid is None
        else _positive_int(expected_launcher_parent_pid, "expected launcher parent id")
    )
    if launcher_pid != expected_parent:
        raise ValueError("Stair formal launch parent process mismatch")

    child = _exact_mapping(root["child"], _CHILD_KEYS, "Stair launcher child")
    argv = _text_list(process_argv, "Stair generation process argv")
    if child["argv"] != argv:
        raise ValueError("Stair formal launch child argv mismatch")
    if child["command_sha256"] != stair_generation_command_hash(argv):
        raise ValueError("Stair formal launch child command hash mismatch")
    resolved_project_root = Path(project_root).resolve(strict=True)
    if child["cwd"] != str(resolved_project_root):
        raise ValueError("Stair formal launch child working directory mismatch")
    _positive_int(child["timeout_s"], "Stair formal launch timeout")
    if Path.cwd().resolve(strict=True) != resolved_project_root:
        raise ValueError("Stair formal generation current working directory mismatch")

    source = _exact_mapping(root["source"], _SOURCE_KEYS, "Stair launcher source")
    if (
        source["commit"] != expected_source_commit
        or source["tracked_clean"] is not True
        or not isinstance(source["files"], Mapping)
        or not source["files"]
    ):
        raise ValueError("Stair formal launch source identity mismatch")
    host = _exact_mapping(root["host"], _HOST_KEYS, "Stair launcher host")
    if host["machine_id"] != expected_machine_id:
        raise ValueError("Stair formal launch machine identity mismatch")
    hostname = _bounded_text(host["hostname"], "Stair launcher hostname")
    boot_id = _bounded_text(host["boot_id"], "Stair launcher boot id")
    if socket.gethostname() != hostname:
        raise ValueError("Stair formal launch hostname mismatch")

    readiness_ref = StairGenerationLaunchReadinessReceiptRef.from_mapping(
        root["launch_readiness_receipt"]
    )
    resolved_output = Path(formal_output_dir).resolve(strict=False)
    if _path_is_within(start_path.resolve(strict=True), resolved_output):
        raise ValueError("Stair formal launch receipt must be outside formal output")
    expected_subject_modes = (
        (
            (STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,)
            if terminal_audit_only
            else (
                STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
                STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            )
            if resume_existing
            else (STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,)
        )
        if formal_session
        else ()
    )
    _assert_formal_output_subject(
        resolved_output,
        resume_existing=resume_existing,
    )
    readiness = load_stair_generation_launch_readiness_receipt(
        readiness_ref.path,
        expected_formal_output_dir=resolved_output,
        expected_source_root=resolved_project_root,
        expected_session_id=session_id,
        expected_source_commit=expected_source_commit,
        expected_machine_id=expected_machine_id,
        expected_gpu_uuids=expected_gpu_uuids,
        expected_lease_root=Path(lease_root).resolve(strict=False),
        expected_max_output_bytes=max_output_bytes,
        expected_nvidia_smi_executable=resolve_stair_nvidia_smi_executable(
            nvidia_smi_executable
        ),
        expected_hostname=hostname,
        expected_boot_id=boot_id,
        expected_subject_mode=(
            expected_subject_modes[0] if len(expected_subject_modes) == 1 else None
        ),
        expected_subject_modes=(
            expected_subject_modes if len(expected_subject_modes) > 1 else None
        ),
        expected_formal_generation_command_sha256=(
            child["command_sha256"] if expected_subject_modes else None
        ),
    )
    readiness.assert_fresh()
    if readiness.to_ref(readiness_ref.path).to_dict() != readiness_ref.to_dict():
        raise ValueError("Stair formal launch readiness reference mismatch")
    _assert_formal_output_subject(
        resolved_output,
        resume_existing=resume_existing,
    )

    environment.pop(STAIR_GENERATION_LAUNCH_START_PATH_ENV, None)
    environment.pop(STAIR_GENERATION_LAUNCH_START_SHA256_ENV, None)
    return root


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _text_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} is invalid")
    return list(value)


def _bounded_text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _assert_path_absent(path: Path, label: str) -> None:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"could not inspect {label}") from exc
    raise FileExistsError(f"{label} already exists")


def _assert_formal_output_subject(path: Path, *, resume_existing: bool) -> None:
    if not resume_existing:
        _assert_path_absent(path, "formal output")
        return
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(
            "formal resume requires an existing real output directory"
        )


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "MAX_LAUNCH_START_RECEIPT_BYTES",
    "STAIR_GENERATION_LAUNCH_START_PATH_ENV",
    "STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION",
    "STAIR_GENERATION_LAUNCH_START_SHA256_ENV",
    "consume_stair_generation_launch_context",
    "stair_generation_command_hash",
]

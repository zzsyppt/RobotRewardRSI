#!/usr/bin/env python3
# ruff: noqa: E402
"""Exercise the production Linux attempt retry and process-owner boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.fs_contract import create_once_bytes  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.runner.base import RunnerFailureKind  # noqa: E402
from true_rl_agent.runner.job_attempts import (  # noqa: E402
    JobAttemptJournal,
    JobAttemptScope,
)
from true_rl_agent.runner.process_handle import (
    _linux_live_process_group_members,  # noqa: E402
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner  # noqa: E402

_JOB_ID = "p6b-attempt-chain-probe"
_TIMEOUT_S = 30
_GRACE_S = 5
_SOURCE_FILES = (
    "scripts/run_p6b_attempt_chain_fault_probe.py",
    "src/true_rl_agent/runner/job_attempts.py",
    "src/true_rl_agent/runner/process_handle.py",
    "src/true_rl_agent/runner/subprocess_runner.py",
)


@dataclass(frozen=True)
class _ProbeSpec:
    job_id: str
    attempt_scope: JobAttemptScope
    metadata: Mapping[str, Any]
    identity_hash: str
    execution_device: str = "cpu"
    gpu_process_binding: None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "execution_device": "cpu",
            "identity_hash": self.identity_hash,
        }


@dataclass(frozen=True)
class _ProbeResult:
    summary: Mapping[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        return {"summary": dict(self.summary)}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _source_identity() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit), "source commit is invalid")
    _require(
        not _git("status", "--porcelain=v1", "--untracked-files=no"),
        "attempt-chain probe requires tracked-clean source",
    )
    for relative in _SOURCE_FILES:
        _require(
            bool(_git("ls-files", "--error-unmatch", "--", relative)),
            f"attempt-chain source is not tracked: {relative}",
        )
    return {
        "source_commit": commit,
        "tracked_clean": True,
        "source_file_sha256": {
            relative: _sha256_file(PROJECT_ROOT / relative)
            for relative in _SOURCE_FILES
        },
    }


def _host_identity() -> dict[str, Any]:
    boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    )
    _require(bool(boot_id), "Linux boot ID is empty")
    return {
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "kernel": platform.uname()._asdict(),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": sys.version,
    }


def _child_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args(list(argv))
    state_dir = args.state_dir.resolve(strict=True)
    summary_path = args.summary_path
    _require(
        summary_path.is_absolute()
        and summary_path.parent.resolve(strict=True) == state_dir,
        "child summary escaped the probe state directory",
    )

    try:
        create_once_bytes(state_dir / "fault_once.marker", b"summary-missing\n")
        attempt_index = 0
    except FileExistsError:
        attempt_index = 1

    ready = state_dir / f"attempt_{attempt_index}.grandchild_ready"
    grandchild_source = (
        "import os,pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
    )
    grandchild = subprocess.Popen(
        [sys.executable, "-c", grandchild_source, str(ready)],
        stdin=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while not ready.is_file():
        _require(grandchild.poll() is None, "probe grandchild exited before ready")
        if time.monotonic() >= deadline:
            raise TimeoutError("probe grandchild readiness timed out")
        time.sleep(0.01)

    identity = {
        "attempt_index": attempt_index,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "grandchild_pid": grandchild.pid,
        "grandchild_pgid": os.getpgid(grandchild.pid),
    }
    identity_path = state_dir / f"attempt_{attempt_index}.identity.json"
    create_once_bytes(
        identity_path, canonical_json_bytes(identity, trailing_newline=True)
    )
    if attempt_index == 1:
        create_once_bytes(
            summary_path,
            canonical_json_bytes(
                {"attempt_index": 1, "status": "completed"},
                trailing_newline=True,
            ),
        )
    print(json.dumps(identity, sort_keys=True), flush=True)
    return 0


def _validate_attempt_chain(
    attempts_root: Path, *, expected_output_hash: str
) -> dict[str, Any]:
    reservations = sorted(attempts_root.rglob("attempt_*.reservation.json"))
    results = sorted(attempts_root.rglob("attempt_*.result.json"))
    _require(
        len(reservations) == len(results) == 2,
        "probe did not produce exactly two reservations and results",
    )
    _require(
        len({path.parent for path in [*reservations, *results]}) == 1,
        "probe receipts span more than one journal",
    )
    journal, _ = JobAttemptJournal.from_reservation_path(reservations[0])
    reservation_0 = journal.load_reservation(0)
    reservation_1 = journal.load_reservation(1)
    result_0 = journal.load_result(0)
    result_1 = journal.load_result(1)
    _require(
        result_0.payload.get("outcome") == "infrastructure_failure"
        and result_0.payload.get("failure_kind")
        == RunnerFailureKind.SUMMARY_MISSING.value
        and result_0.payload.get("retryable") is True
        and result_0.payload.get("output_hash") is None,
        "first attempt is not retryable summary_missing",
    )
    _require(
        reservation_1.payload.get("previous_attempt_result_hash")
        == result_0.payload_hash,
        "retry does not bind the first result",
    )
    _require(
        result_1.payload.get("outcome") == "completed"
        and result_1.payload.get("retryable") is False
        and result_1.payload.get("output_hash") == expected_output_hash,
        "terminal attempt result is invalid",
    )

    before = {path: path.read_bytes() for path in results}
    republished = (
        journal.finish(
            reservation_0,
            outcome="infrastructure_failure",
            failure_kind=RunnerFailureKind.SUMMARY_MISSING.value,
            retryable=True,
            output_hash=None,
            returncode=None,
        ),
        journal.finish(
            reservation_1,
            outcome="completed",
            failure_kind=None,
            retryable=False,
            output_hash=expected_output_hash,
            returncode=None,
        ),
    )
    _require(
        not any(receipt.created for receipt in republished)
        and before == {path: path.read_bytes() for path in results},
        "attempt results failed create-once republish verification",
    )
    return {
        "attempt_result_count": 2,
        "result_files": [
            path.relative_to(attempts_root).as_posix() for path in results
        ],
        "first_result_hash": result_0.payload_hash,
        "retry_previous_result_hash": reservation_1.payload[
            "previous_attempt_result_hash"
        ],
        "terminal_chain_head_hash": result_1.payload_hash,
        "terminal_output_hash": result_1.payload["output_hash"],
        "republish_created": [receipt.created for receipt in republished],
        "republish_bytes_unchanged": True,
    }


def _file_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "report.json"
    }


def _run_probe(
    output_dir: Path,
    python_executable: Path,
    command_argv: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state_dir = output_dir / "state"
    attempts_root = output_dir / "attempts"
    state_dir.mkdir(mode=0o700)
    attempts_root.mkdir(mode=0o700)
    summary_path = state_dir / "summary.json"
    identity_hash = canonical_json_hash(
        {
            "source_commit": source["source_commit"],
            "source_file_sha256": source["source_file_sha256"],
            "attempt_budget_units": 2,
            "max_infrastructure_retries": 1,
        }
    )
    child_argv = [
        str(python_executable),
        str(Path(__file__).resolve(strict=True)),
        "--child",
        "--state-dir",
        str(state_dir),
        "--summary-path",
        str(summary_path),
    ]
    scope = JobAttemptScope(
        journal_root=attempts_root,
        session_genesis_hash=canonical_json_hash(
            {"identity_hash": identity_hash, "kind": "genesis"}
        ),
        session_contract_hash=identity_hash,
        generation_id="p6b-attempt-chain-probe",
        lane_id="probe-lane",
        invocation_segment_id="probe-segment",
        budget_contract_hash=identity_hash,
        attempt_budget_units=2,
        max_infrastructure_retries=1,
    )
    metadata = {
        "cli_argv": child_argv,
        "cli_command": " ".join(child_argv),
        "summary_path": str(summary_path),
        "work_dir": str(PROJECT_ROOT),
        "backend_id": "p6b_attempt_chain_probe",
        "backend_execution_environment_manifest_hash": canonical_json_hash(
            source["source_file_sha256"]
        ),
    }
    spec = _ProbeSpec(_JOB_ID, scope, metadata, identity_hash)
    runner = SubprocessRunner(
        (str(python_executable),),
        timeout_s=_TIMEOUT_S,
        cwd=PROJECT_ROOT,
        termination_grace_s=_GRACE_S,
    )
    observations: list[dict[str, Any]] = []

    def operation(
        log_dir: Path | None,
        execution_context: Any,
        attempt_timeout_s: int,
    ) -> _ProbeResult:
        _require(log_dir is not None and execution_context is None, "CPU scope drifted")
        attempt_index = len(observations)
        _require(attempt_index in {0, 1}, "probe exceeded its attempt budget")
        completed = runner._run_job(
            runner._validated_argv(metadata, job_id=spec.job_id),
            metadata,
            job_id=spec.job_id,
            process_log_dir=log_dir,
            attempt_timeout_s=attempt_timeout_s,
            execution_context=None,
        )
        identity_path = state_dir / f"attempt_{attempt_index}.identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        _require(
            completed.returncode == 0
            and identity.get("attempt_index") == attempt_index
            and identity.get("pid") == completed.pid
            and identity.get("pgid") == completed.pid
            and identity.get("grandchild_pgid") == completed.pid,
            "owned child or grandchild identity mismatch",
        )
        live = _linux_live_process_group_members(completed.pid)
        _require(not live, f"owned process group remains live: {list(live)}")
        observations.append(
            {
                "attempt_index": attempt_index,
                "pid": completed.pid,
                "pgid": completed.pid,
                "grandchild_pid": identity["grandchild_pid"],
                "live_pids_after_return": list(live),
                "stdout_log": str(completed.stdout_log.relative_to(output_dir)),
                "stderr_log": str(completed.stderr_log.relative_to(output_dir)),
            }
        )
        summary = runner._read_summary(metadata, job_id=spec.job_id)
        _require(
            summary == {"attempt_index": 1, "status": "completed"},
            "terminal summary mismatch",
        )
        return _ProbeResult(summary)

    result = runner._execute_with_attempts(spec, job_kind="train", operation=operation)
    _require(len(observations) == 2, "initial-plus-retry process count mismatch")
    chain = _validate_attempt_chain(
        attempts_root,
        expected_output_hash=SubprocessRunner._attempt_output_hash(result),
    )
    return {
        "schema_version": "true_rl_agent.p6b.attempt_chain_probe.v1",
        "status": "pass",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command_argv": list(command_argv),
        "host": _host_identity(),
        "source": dict(source),
        "owned_python_executable": str(python_executable),
        "probe_identity_hash": identity_hash,
        "attempt_chain": chain,
        "process_observations": observations,
        "checks": {
            "exactly_two_create_once_results": True,
            "retry_chain_closed": True,
            "zero_process_group_residual": True,
        },
        "file_sha256": _file_hashes(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "--child":
        return _child_main(raw_argv[1:])
    if not sys.platform.startswith("linux"):
        raise RuntimeError("attempt-chain probe requires Linux")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    args = parser.parse_args(raw_argv)
    _require(args.output.is_absolute(), "probe output must be absolute")
    output_dir = args.output.parent.resolve(strict=True) / args.output.name
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"probe output already exists: {output_dir}")
    _require(args.python_executable.is_absolute(), "probe Python must be absolute")
    python_executable = args.python_executable.resolve(strict=True)
    _require(
        python_executable.is_file() and os.access(python_executable, os.X_OK),
        "probe Python is not executable",
    )

    source = _source_identity()
    command = [
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        *raw_argv,
    ]
    output_dir.mkdir(mode=0o700, exist_ok=False)
    report = _run_probe(output_dir, python_executable, command, source)
    encoded = canonical_json_bytes(report, trailing_newline=True)
    report_path = output_dir / "report.json"
    create_once_bytes(report_path, encoded)
    print(
        json.dumps(
            {
                "status": "pass",
                "report_path": str(report_path),
                "report_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

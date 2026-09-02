#!/usr/bin/env python3
"""Replay a closed P6-B bundle through the current production report loader.

This is an offline compatibility gate.  It deliberately calls the real
foundation, Session Genesis, cost-journal, and P6-B report loaders.  It never
enters the provider/session execution path and uses a strict no-row
``nvidia-smi`` stand-in so that no GPU process is queried or started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
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
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity  # noqa: E402
from true_rl_agent.tasks.stair_climbing import (  # noqa: E402
    generation_session as session,
)


_P6B_REPORT_SCHEMA = "true_rl_agent.stair.generation_session_report.v2"
_SOURCE_FILES = (
    "scripts/audit_stair_p6b_historical_loader_replay.py",
    "src/true_rl_agent/tasks/stair_climbing/generation_session.py",
    "src/true_rl_agent/tasks/stair_climbing/generation_session_cost.py",
    "src/true_rl_agent/agent_core/session_genesis.py",
    "src/true_rl_agent/agent_core/generation_lifecycle.py",
    "src/true_rl_agent/agent_core/generation_record.py",
    "src/true_rl_agent/agent_core/generation_receipts.py",
    "src/true_rl_agent/runner/job_attempts.py",
)


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
        timeout=30,
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
        "historical loader replay requires tracked-clean source",
    )
    for relative in _SOURCE_FILES:
        _require(
            bool(_git("ls-files", "--error-unmatch", "--", relative)),
            f"historical loader replay source is not tracked: {relative}",
        )
    return {
        "source_commit": commit,
        "tracked_clean": True,
        "source_file_sha256": {
            relative: _sha256_file(PROJECT_ROOT / relative)
            for relative in _SOURCE_FILES
        },
    }


def _strict_json(path: Path, *, context: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{context} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{context} must be an object")
    _require(
        path.read_bytes() == canonical_json_bytes(value, trailing_newline=True),
        f"{context} is not canonical JSON",
    )
    return value


def _tree_snapshot(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), "bundle root is invalid")
    entries: list[dict[str, Any]] = []
    pending = [resolved]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(resolved).as_posix()
                _require(
                    not child.is_symlink(),
                    f"bundle contains a symlink: {relative}",
                )
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                _require(
                    child.is_file(follow_symlinks=False),
                    f"bundle contains a special file: {relative}",
                )
                details = child.stat(follow_symlinks=False)
                _require(
                    stat.S_ISREG(details.st_mode),
                    f"bundle file is invalid: {relative}",
                )
                entries.append(
                    {
                        "relative_path": relative,
                        "size_bytes": details.st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    entries.sort(key=lambda item: item["relative_path"])
    return {
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "tree_hash": canonical_json_hash(entries),
    }


def _install_replay_guard(
    bundle_root: Path,
    *,
    idle_stub: Path,
) -> dict[str, int]:
    """Fail closed on formal writes, network calls, or unexpected children."""

    root = bundle_root.resolve(strict=True)
    counters = {
        "blocked_write_attempt_count": 0,
        "blocked_network_attempt_count": 0,
        "blocked_unexpected_subprocess_count": 0,
        "allowed_gpu_idle_stub_count": 0,
        "allowed_existing_directory_mkdir_count": 0,
    }
    write_open_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    write_open_flags |= int(getattr(os, "O_TMPFILE", 0))
    path_mutation_events = {
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mknod",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "os.utime",
    }

    def guarded_path(value: Any) -> Path | None:
        if not isinstance(value, str | bytes | os.PathLike):
            return None
        try:
            candidate = Path(os.fsdecode(value)).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return None
        return candidate if candidate == root or root in candidate.parents else None

    def reject(event: str, path: Path) -> None:
        counters["blocked_write_attempt_count"] += 1
        raise PermissionError(
            f"historical P6-B write guard blocked {event}: {path}"
        )

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex"}:
            counters["blocked_network_attempt_count"] += 1
            raise PermissionError("historical P6-B replay forbids network access")
        if event == "subprocess.Popen":
            executable = Path(os.fsdecode(args[0])).resolve(strict=False)
            argv = list(args[1])
            expected_argv = [
                str(idle_stub),
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
            if executable != idle_stub or argv != expected_argv:
                counters["blocked_unexpected_subprocess_count"] += 1
                raise PermissionError(
                    "historical P6-B replay forbids this subprocess"
                )
            counters["allowed_gpu_idle_stub_count"] += 1
            return
        if event == "open" and args:
            path = guarded_path(args[0])
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            write_mode = isinstance(mode, str) and any(
                marker in mode for marker in ("w", "a", "x", "+")
            )
            write_flags = isinstance(flags, int) and bool(flags & write_open_flags)
            if write_mode or write_flags:
                reject(event, path)
            return
        if event == "os.mkdir" and args:
            path = guarded_path(args[0])
            if path is None:
                return
            if path.is_dir() and not path.is_symlink():
                counters["allowed_existing_directory_mkdir_count"] += 1
                return
            reject(event, path)
        if event in path_mutation_events:
            for value in args:
                path = guarded_path(value)
                if path is not None:
                    reject(event, path)

    sys.addaudithook(audit)
    return counters


def _write_gpu_idle_stub(path: Path) -> None:
    source = b"""#!/usr/bin/env python3
import sys

expected = [
    "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits",
]
if sys.argv[1:] != expected:
    raise SystemExit("unexpected nvidia-smi arguments")
"""
    create_once_bytes(path, source)
    path.chmod(0o700)


def _recorded_gpu_uuids(report: Mapping[str, Any]) -> tuple[str, str]:
    evidence = report.get("resource_evidence")
    _require(isinstance(evidence, Mapping), "P6-B report resource evidence is invalid")
    startup = evidence.get("startup_compute_idle")
    _require(isinstance(startup, Mapping), "P6-B startup evidence is invalid")
    values = startup.get("target_gpu_uuids")
    _require(
        isinstance(values, list)
        and len(values) == 2
        and all(isinstance(value, str) for value in values),
        "P6-B target GPU identities are invalid",
    )
    return values[0], values[1]


def _loader_replay(
    *,
    bundle_root: Path,
    evidence_root: Path,
    machine_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = bundle_root / session.STAIR_GENERATION_REPORT_FILENAME
    recorded = _strict_json(report_path, context="historical P6-B report")
    _require(
        recorded.get("schema_version") == _P6B_REPORT_SCHEMA,
        "not a P6-B v2 report",
    )
    _require(
        recorded.get("session_mode") == session.DEVELOPMENT_GENERATION_PILOT_MODE,
        "not a P6-B pilot",
    )
    _require(
        recorded.get("output_dir") == str(bundle_root),
        "bundle absolute path drifted",
    )

    source_commit = recorded.get("source_commit")
    session_id = recorded.get("session_id")
    _require(
        isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit),
        "recorded P6-B source commit is invalid",
    )
    _require(
        isinstance(session_id, str) and session_id,
        "recorded P6-B session ID is invalid",
    )
    gpu0_uuid, gpu1_uuid = _recorded_gpu_uuids(recorded)
    physical_cap = recorded.get("physical_cost_cap")
    _require(isinstance(physical_cap, Mapping), "P6-B physical cap is invalid")

    idle_stub = evidence_root / "nvidia_smi_idle_stub.py"
    _write_gpu_idle_stub(idle_stub)
    config = session.StairGenerationPilotConfig(
        project_root=PROJECT_ROOT,
        output_dir=bundle_root,
        lease_root=evidence_root / "leases",
        session_id=session_id,
        expected_source_commit=source_commit,
        machine_id=machine_id,
        gpu0_uuid=gpu0_uuid,
        gpu1_uuid=gpu1_uuid,
        provider_env_file=evidence_root / "provider-not-used.env",
        max_allocated_active_gpu_hours=float(
            physical_cap["allocated_active_gpu_hours"]
        ),
        max_simulated_control_steps=int(physical_cap["simulated_control_steps"]),
        max_output_bytes=int(physical_cap["output_bytes"]),
        nvidia_smi_executable=str(idle_stub),
        resume_existing=True,
    )

    before = _tree_snapshot(bundle_root)
    replay_guard = _install_replay_guard(bundle_root, idle_stub=idle_stub)
    try:
        foundation = session._create_foundation(
            config,
            source_commit=source_commit,
            inventory=(gpu0_uuid, gpu1_uuid),
        )
        recorded_bindings = recorded.get("gpu_process_bindings")
        _require(
            recorded_bindings
            == [binding.to_dict() for binding in foundation.gpu_bindings],
            "current foundation GPU bindings differ from P6-B",
        )
        _require(
            [
                GpuDeviceIdentity("cuda:0", machine_id, gpu0_uuid).identity_hash,
                GpuDeviceIdentity("cuda:1", machine_id, gpu1_uuid).identity_hash,
            ]
            == [item["resource_binding_hash"] for item in recorded_bindings],
            "machine/GPU identity does not reproduce P6-B bindings",
        )
        genesis = session._load_existing_session_genesis_for_terminal_audit(foundation)
        cost_journal = session.StairGenerationActualCostJournal(
            bundle_root,
            contract=foundation.cost_contract,
        )
        startup = recorded["resource_evidence"]["startup_compute_idle"]
        replayed = session._load_and_validate_existing_report(
            foundation,
            genesis=genesis,
            cost_journal=cost_journal,
            startup_resource_observation=startup,
        )
    finally:
        after = _tree_snapshot(bundle_root)

    _require(replayed == recorded, "production P6-B loader result differs from report")
    _require(before == after, "historical P6-B bundle bytes changed during replay")
    _require(
        replay_guard["blocked_write_attempt_count"] == 0
        and replay_guard["blocked_network_attempt_count"] == 0
        and replay_guard["blocked_unexpected_subprocess_count"] == 0
        and replay_guard["allowed_gpu_idle_stub_count"] == 1,
        "historical P6-B replay side-effect guard did not close cleanly",
    )
    checks = replayed.get("checks")
    _require(
        isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values()),
        "replayed P6-B report checks are not all true",
    )
    _require(
        replayed.get("acceptance_authority") is False,
        "P6-B replay gained authority",
    )
    return {
        "schema_version": "true_rl_agent.stair.p6b_historical_loader_replay.v1",
        "status": "pass",
        "finished_at_utc": (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "source": dict(source),
        "historical_bundle": {
            "root": str(bundle_root),
            "session_id": session_id,
            "science_source_commit": source_commit,
            "report_schema_version": replayed["schema_version"],
            "report_sha256": _sha256_file(report_path),
            "session_genesis_hash": replayed["session_genesis_hash"],
            "generation_session_contract_hash": replayed[
                "generation_session_contract_hash"
            ],
            "session_cost_contract_hash": replayed["session_cost_contract_hash"],
            "generation_id": replayed["generation_id"],
            "generation_outcome": replayed["generation_outcome"],
            "actual_cost_receipt_hash": replayed["actual_cost_receipt"][
                "payload_hash"
            ],
            "tree_before": before,
            "tree_after": after,
            "tree_unchanged": True,
        },
        "loader": {
            "entrypoint": (
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "_load_and_validate_existing_report"
            ),
            "foundation_entrypoint": (
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "_create_foundation"
            ),
            "genesis_entrypoint": (
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "_load_existing_session_genesis_for_terminal_audit"
            ),
            "loader_replaced_or_mocked": False,
            "formal_bundle_write_guard_installed": True,
            "replay_guard": dict(replay_guard),
            "source_checkout_wrapper_bypassed": True,
            "source_checkout_wrapper_bypass_reason": (
                "current loader must replay a bundle recorded under an older "
                "source commit"
            ),
        },
        "side_effect_boundary": {
            "provider_called": False,
            "gpu_compute_started": False,
            "gpu_inventory_or_compute_process_queried": False,
            "nvidia_smi_idle_stub_sha256": _sha256_file(idle_stub),
            "lease_root": str(config.lease_root),
            "lease_root_outside_historical_bundle": True,
            "historical_bundle_written": False,
        },
        "checks": {
            "current_production_loader_returned_exact_recorded_report": True,
            "current_foundation_reproduced_recorded_contracts_and_bindings": True,
            "current_genesis_loader_validated_private_reference_artifacts": True,
            "current_cost_journal_replayed_recorded_actual_cost_chain": True,
            "all_recorded_report_checks_true": True,
            "historical_bundle_tree_unchanged": True,
            "no_loader_or_foundation_mock": True,
            "no_provider_or_gpu_compute": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("historical P6-B loader replay requires Linux")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    _require(args.bundle_root.is_absolute(), "bundle root must be absolute")
    bundle_root = args.bundle_root.resolve(strict=True)
    _require(args.output.is_absolute(), "evidence output must be absolute")
    evidence_root = args.output.parent.resolve(strict=True) / args.output.name
    if evidence_root.exists() or evidence_root.is_symlink():
        raise FileExistsError(f"evidence output already exists: {evidence_root}")
    _require(
        bundle_root not in evidence_root.parents and evidence_root != bundle_root,
        "evidence output must be outside the historical bundle",
    )
    _require(
        isinstance(args.machine_id, str) and args.machine_id.strip(),
        "machine ID is required",
    )

    source = _source_identity()
    evidence_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = _loader_replay(
        bundle_root=bundle_root,
        evidence_root=evidence_root,
        machine_id=args.machine_id,
        source=source,
    )
    encoded = canonical_json_bytes(report, trailing_newline=True)
    report_path = evidence_root / "report.json"
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

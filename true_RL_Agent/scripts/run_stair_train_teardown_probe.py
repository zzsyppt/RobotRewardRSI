#!/usr/bin/env python3
"""Run one real, bounded Stair smoke and prove normal-exit teardown is clean.

This is an operational launch gate. It has no experiment, acceptance, or
ledger authority and delegates all training semantics to the existing smoke.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.fs_contract import (  # noqa: E402
    create_once_bytes,
    read_bounded_regular_file,
)
from true_rl_agent.json_contract import (  # noqa: E402
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.runner.checkpoint import file_sha256  # noqa: E402
from true_rl_agent.runner.gpu_lease import (  # noqa: E402
    GpuLeaseManager,
    PreGenesisGpuLeaseHolder,
)
from true_rl_agent.runner.process_handle import (  # noqa: E402
    _linux_live_process_group_members,
    run_owned_process,
)
from true_rl_agent.runner.resource_shadow import (  # noqa: E402
    NvidiaGpuInventoryEntry,
    discover_nvidia_gpu_inventory,
    sample_resource_snapshot,
)
from true_rl_agent.runner.specs import (  # noqa: E402
    GpuProcessBinding,
    require_execution_device,
)
from true_rl_agent.runner.gpu_process_binding import (  # noqa: E402
    is_full_gpu_uuid, validate_gpu_runtime_summary_evidence,
)

LAUNCH_SCHEMA = "true_rl_agent.p6b.stair_train_teardown_probe.launch.v1"
RESULT_SCHEMA = "true_rl_agent.p6b.stair_train_teardown_probe.result.v1"
FAILURE_SCHEMA = "true_rl_agent.p6b.stair_train_teardown_probe.failure.v1"
SMOKE_SCHEMA = "true_rl_agent.task.stair_climbing.phase_d_final_rsl_rl_smoke.v1"
TIMEOUT_S = 900
TERMINATION_GRACE_S = 5.0
IDLE_SAMPLE_GAP_S = 1.0
MAX_REPORT_BYTES = 8 * 1024 * 1024
WORKLOAD = {"num_envs": 16, "num_steps_per_env": 8, "max_iterations": 1, "env_steps": 128}
CRITICAL_SOURCE_PATHS = ("scripts/run_stair_train_teardown_probe.py", "scripts/isaaclab_stair_rsl_rl_smoke.py", "src/true_rl_agent/runner/process_handle.py")
AUTHORITY = dict(operational_only=True, scientific_claim_authority=False, acceptance_authority=False, ledger_authority=False)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_BOOT_ID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--host-execution-device", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nvidia-smi-executable", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    return parser

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    create_once_bytes(path, canonical_json_bytes(dict(value), trailing_newline=True))

def _path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False

def _assert_absent(path: Path, label: str) -> None:
    if not _path_absent(path):
        raise FileExistsError(f"{label} must remain absent: {path}")

def _fresh_absolute_path(raw: Path, label: str) -> Path:
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = raw.resolve(strict=False)
    _assert_absent(resolved, label)
    return resolved

def _absolute_executable(raw: Path, label: str) -> Path:
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} must be an executable regular file")
    return resolved

def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents

def _linux_boot_id() -> str:
    payload = read_bounded_regular_file(
        Path("/proc/sys/kernel/random/boot_id"), max_bytes=128, label="Linux boot ID"
    )
    boot_id = payload.decode("ascii").strip()
    if _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        raise ValueError("Linux boot ID is invalid")
    return boot_id

def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()

def _source_identity(expected_commit: str) -> dict[str, Any]:
    if _COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected source commit must be a full lowercase commit")
    actual_commit = _git("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError("checked-out source commit does not match expected commit")
    if _git("status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("tracked source tree is not clean")

    checksums: dict[str, str] = {}
    for relative in CRITICAL_SOURCE_PATHS:
        if _git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"critical source tracking mismatch: {relative}")
        checksum = file_sha256((PROJECT_ROOT / relative).resolve(strict=True))
        if checksum is None:
            raise ValueError(f"critical source checksum failed: {relative}")
        checksums[relative] = checksum
    return {"commit": actual_commit, "tracked_clean": True, "checksums": checksums}

def _resolve_target_gpu(
    *, host_execution_device: str, expected_uuid: str, nvidia_smi: Path
) -> NvidiaGpuInventoryEntry:
    device = require_execution_device(
        host_execution_device, "teardown probe host_execution_device"
    )
    if not device.startswith("cuda:"):
        raise ValueError("teardown probe host device must be CUDA")
    if not is_full_gpu_uuid(expected_uuid):
        raise ValueError("teardown probe requires one full physical GPU UUID")
    inventory = discover_nvidia_gpu_inventory(nvidia_smi=str(nvidia_smi))
    matches = tuple(item for item in inventory if item.execution_device == device)
    if len(matches) != 1 or matches[0].uuid != expected_uuid:
        raise ValueError("host CUDA ordinal does not match the expected GPU UUID")
    return matches[0]

def _sample_idle(
    *,
    target: NvidiaGpuInventoryEntry,
    output_dir: Path,
    nvidia_smi: Path,
    stage: str,
) -> dict[str, Any]:
    snapshot = sample_resource_snapshot(
        inventory=(target,), output_dir=output_dir, nvidia_smi=str(nvidia_smi)
    )
    rows = snapshot.get("compute_apps")
    if not isinstance(rows, list):
        raise ValueError("GPU snapshot compute_apps is invalid")
    if rows:
        raise RuntimeError(f"target GPU is not compute-idle at {stage}")
    return {"stage": stage, "snapshot": snapshot}

def _detached_processes_referencing(
    marker: Path, *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Find live argv entries that still cite the child-owned output subtree."""

    needle = os.fsencode(str(marker.resolve(strict=False)))
    matches: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("failed to inspect Linux process table") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            with (entry / "cmdline").open("rb") as handle:
                argv = handle.read(64 * 1024 + 1)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # hidepid can conceal unrelated users; this owner's child remains readable.
            continue
        except OSError as exc:
            raise RuntimeError("failed to inspect Linux process argv") from exc
        if len(argv) > 64 * 1024:
            raise RuntimeError("Linux process argv exceeded probe bound")
        if needle in argv:
            matches.append(int(entry.name))
    return tuple(sorted(matches))

def _validated_smoke_report(
    *,
    report_path: Path,
    checkpoint_path: Path,
    binding: GpuProcessBinding,
    gpu_uuid: str,
    expected_child_pid: int,
) -> dict[str, Any]:
    report = strict_json_object_loads(
        read_bounded_regular_file(
            report_path,
            max_bytes=MAX_REPORT_BYTES,
            label="Stair teardown smoke report",
        ),
        context="Stair teardown smoke report",
    )
    declared_hash = report.pop("report_hash", None)
    if declared_hash != canonical_json_hash(report):
        raise ValueError("Stair smoke report hash mismatch")
    if report.get("schema_version") != SMOKE_SCHEMA or report.get("status") != "pass":
        raise ValueError("Stair smoke report did not pass")
    training = report.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Stair smoke training evidence is missing")
    for field_name, expected in WORKLOAD.items():
        if training.get(field_name) != expected:
            raise ValueError(f"Stair smoke workload mismatch: {field_name}")
    if report.get("reward_manager_active_terms") != ["complete_reward"]:
        raise ValueError("Stair smoke reward runtime was not activated")
    runtime = report.get("runtime")
    dispatcher_calls = runtime.get("dispatcher_calls") if isinstance(runtime, Mapping) else None
    if (
        isinstance(dispatcher_calls, bool)
        or not isinstance(dispatcher_calls, int)
        or dispatcher_calls <= 0
    ):
        raise ValueError("Stair smoke dispatcher was not activated")

    if not stat.S_ISREG(checkpoint_path.lstat().st_mode):
        raise ValueError("Stair smoke checkpoint must be a regular file")
    checkpoint_checksum = file_sha256(checkpoint_path)
    checkpoint = report.get("checkpoint")
    if (
        checkpoint_checksum is None
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != str(checkpoint_path)
        or checkpoint.get("checksum") != checkpoint_checksum
    ):
        raise ValueError("Stair smoke checkpoint evidence mismatch")
    gpu_evidence = validate_gpu_runtime_summary_evidence(
        report,
        expected_binding=binding,
        expected_physical_gpu_uuid=gpu_uuid,
        require_compute_context_observation=True,
    )
    if gpu_evidence["owned_process_identity"]["process_id"] != expected_child_pid:
        raise ValueError("Stair smoke PID does not match runner-owned child")
    report_checksum = file_sha256(report_path)
    if report_checksum is None:
        raise ValueError("Stair smoke report checksum failed")
    return {
        "report_hash": declared_hash,
        "report_checksum": report_checksum,
        "checkpoint_checksum": checkpoint_checksum,
        "workload": dict(WORKLOAD),
        "reward_runtime_activated": True,
        "dispatcher_calls": dispatcher_calls,
        "gpu_runtime_evidence": gpu_evidence,
    }

def _holder(contract_hash: str, *, segment: str) -> PreGenesisGpuLeaseHolder:
    return PreGenesisGpuLeaseHolder(
        session_id="p6b-stair-train-teardown-probe",
        genesis_identity_hash=contract_hash,
        session_contract_hash=contract_hash,
        lane_id="reference",
        invocation_segment_id=segment,
        logical_job_id="stair-train-teardown",
    )


def _execute_probe(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    formal_output_dir: Path,
) -> dict[str, Any]:
    isaac_python = _absolute_executable(args.isaac_python, "Isaac Python")
    if Path(sys.executable).resolve(strict=True) != isaac_python:
        raise ValueError("probe must run under the declared Isaac Python")
    nvidia_smi = _absolute_executable(
        args.nvidia_smi_executable, "nvidia-smi executable"
    )
    if not args.lease_root.is_absolute():
        raise ValueError("shared GPU lease root must be absolute")
    lease_root = args.lease_root.resolve(strict=True)
    if not lease_root.is_dir() or any(
        _paths_overlap(lease_root, path) for path in (output_dir, formal_output_dir)
    ):
        raise ValueError("shared GPU lease root must be an existing disjoint directory")
    if "CUDA_VISIBLE_DEVICES" in os.environ or "NVIDIA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("probe parent must see the unremapped host GPU inventory")

    source = _source_identity(args.expected_source_commit)
    target = _resolve_target_gpu(
        host_execution_device=args.host_execution_device,
        expected_uuid=args.gpu_uuid,
        nvidia_smi=nvidia_smi,
    )
    host = {
        "hostname": socket.gethostname(),
        "boot_id": _linux_boot_id(),
        "parent_process_id": os.getpid(),
        "isaac_python": str(isaac_python),
        "nvidia_smi_executable": str(nvidia_smi),
        "target_gpu": target.to_dict(),
    }
    binding = GpuProcessBinding(
        host_execution_device=target.execution_device,
        process_execution_device="cuda:0",
        resource_binding_hash=target.identity_hash,
    )
    child_root = output_dir / "child"
    report_path = child_root / "smoke_report.json"
    checkpoint_path = child_root / "smoke_checkpoint.pt"
    child_argv = [
        str(isaac_python),
        "-u",
        str((PROJECT_ROOT / "scripts/isaaclab_stair_rsl_rl_smoke.py").resolve(strict=True)),
        "--output",
        str(report_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--device",
        "cuda:0",
        "--headless",
        "--nvidia-smi-executable",
        str(nvidia_smi),
        "--gpu-host-execution-device",
        binding.host_execution_device,
        "--gpu-process-execution-device",
        binding.process_execution_device,
        "--gpu-resource-binding-hash",
        binding.resource_binding_hash,
    ]
    child_environment: dict[str, str | None] = {
        "CUDA_VISIBLE_DEVICES": target.uuid,
        "NVIDIA_VISIBLE_DEVICES": None,
    }
    launch_identity = {
        "source": source,
        "host": host,
        "probe_argv": args.probe_argv,
        "formal_output_dir": str(formal_output_dir),
        "lease_root": str(lease_root),
        "child_argv": child_argv,
        "child_environment_overrides": child_environment,
        "timeout_s": TIMEOUT_S,
        "termination_grace_s": TERMINATION_GRACE_S,
        "workload": dict(WORKLOAD),
        **AUTHORITY,
    }
    contract_hash = canonical_json_hash(launch_identity)

    _assert_absent(formal_output_dir, "formal output directory")
    startup_idle = _sample_idle(
        target=target,
        output_dir=output_dir,
        nvidia_smi=nvidia_smi,
        stage="startup",
    )
    manager = GpuLeaseManager(
        lease_root=lease_root, devices={target.execution_device: target.identity}
    )
    with manager.acquire(
        target.execution_device, holder=_holder(contract_hash, segment="real-smoke")
    ) as lease:
        lease.assert_owned()
        launch = {
            "schema_version": LAUNCH_SCHEMA,
            "status": "launched",
            "created_at_utc": _utc_now(),
            "launch_identity_hash": contract_hash,
            **launch_identity,
            "startup_compute_idle": startup_idle,
            "lease_held": True,
        }
        _publish_json(output_dir / "launch.json", launch)
        result = run_owned_process(
            child_argv,
            cwd=PROJECT_ROOT,
            timeout_s=TIMEOUT_S,
            log_dir=child_root / "process",
            termination_grace_s=TERMINATION_GRACE_S,
            environment_overrides=child_environment,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"real Stair smoke exited with {result.returncode}: "
                f"{result.stderr_tail[-1000:]}"
            )
        smoke = _validated_smoke_report(
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            binding=binding,
            gpu_uuid=target.uuid,
            expected_child_pid=result.pid,
        )
        live_group_members = _linux_live_process_group_members(result.pid)
        if live_group_members:
            raise RuntimeError("real Stair smoke process group is not empty")
        detached = _detached_processes_referencing(child_root)
        if detached:
            raise RuntimeError(
                f"detached processes still reference child output: {list(detached)}"
            )
        post_exit_idle = [
            _sample_idle(
                target=target,
                output_dir=output_dir,
                nvidia_smi=nvidia_smi,
                stage="post-exit-1",
            )
        ]
        time.sleep(IDLE_SAMPLE_GAP_S)
        post_exit_idle.append(
            _sample_idle(
                target=target,
                output_dir=output_dir,
                nvidia_smi=nvidia_smi,
                stage="post-exit-2",
            )
        )
    with manager.acquire(
        target.execution_device,
        holder=_holder(contract_hash, segment="lease-reacquire"),
    ) as reacquired:
        reacquired.assert_owned()
        if reacquired.resource_binding_hash != binding.resource_binding_hash:
            raise RuntimeError("reacquired GPU lease identity changed")
    _assert_absent(formal_output_dir, "formal output directory")

    stdout_checksum = file_sha256(result.stdout_log)
    stderr_checksum = file_sha256(result.stderr_log)
    launch_checksum = file_sha256(output_dir / "launch.json")
    if stdout_checksum is None or stderr_checksum is None or launch_checksum is None:
        raise ValueError("probe evidence checksum failed")
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "created_at_utc": _utc_now(),
        "launch_identity_hash": contract_hash,
        "source": source,
        "host": host,
        "child": {
            "argv": child_argv,
            "environment_overrides": child_environment,
            "process_id": result.pid,
            "process_group_id": result.pid,
            "returncode": result.returncode,
            "stdout_log": str(result.stdout_log),
            "stdout_checksum": stdout_checksum,
            "stderr_log": str(result.stderr_log),
            "stderr_checksum": stderr_checksum,
            "smoke_report": str(report_path),
            "checkpoint": str(checkpoint_path),
            **smoke,
        },
        "startup_compute_idle": startup_idle,
        "post_exit_compute_idle_samples": post_exit_idle,
        "process_group_live_members_after_return": [],
        "detached_processes_referencing_child_output": [],
        "lease_reacquired": True,
        "formal_output_absent": True,
        "launch_checksum": launch_checksum,
        **AUTHORITY,
    }


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(effective_argv)
    args.probe_argv = list(sys.orig_argv) if argv is None else [
        str(Path(sys.executable).resolve(strict=True)), str(Path(__file__).resolve()), *effective_argv
    ]
    requested = {
        "probe_argv": args.probe_argv,
        "expected_source_commit": args.expected_source_commit,
        "host_execution_device": args.host_execution_device,
        "gpu_uuid": args.gpu_uuid,
    }
    try:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Stair train teardown probe requires Linux")
        output_dir = _fresh_absolute_path(args.output_dir, "probe output directory")
        formal_output_dir = _fresh_absolute_path(
            args.formal_output_dir, "formal output directory"
        )
        if _paths_overlap(output_dir, formal_output_dir):
            raise ValueError("probe and formal output directories must be disjoint")
        output_dir.mkdir(parents=True, exist_ok=False)
        result = _execute_probe(
            args, output_dir=output_dir, formal_output_dir=formal_output_dir
        )
        _publish_json(output_dir / "result.json", result)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist the launch-gate failure
        raw_output = Path(args.output_dir)
        output_dir = raw_output.resolve(strict=False) if raw_output.is_absolute() else None
        if output_dir is not None and output_dir.is_dir():
            formal_raw = Path(args.formal_output_dir)
            formal = formal_raw.resolve(strict=False) if formal_raw.is_absolute() else formal_raw
            launch_path = output_dir / "launch.json"
            failure = {
                "schema_version": FAILURE_SCHEMA,
                "status": "fail",
                "created_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:4000],
                "requested": requested,
                "formal_output_absent_at_failure": _path_absent(formal),
                "launch_checksum": file_sha256(launch_path) if launch_path.is_file() else None,
                **AUTHORITY,
            }
            _publish_json(output_dir / "failure.json", failure)
        error = {"status": "fail", "error": type(exc).__name__, "message": str(exc)}
        print(canonical_json_bytes(error).decode(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

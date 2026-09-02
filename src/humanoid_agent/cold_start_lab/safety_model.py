"""Phase 8E-1 safety-model probes.

The safety model has two independent boundaries:

* candidate code may run only inside an OS-isolated training process;
* trusted evaluation consumes checkpoints only and never receives candidate code.

These helpers implement executable probes for that contract without launching
formal Isaac/RSL-RL training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.code_sandbox.isolation import (
    IsolationError,
    pid_namespace_supported,
    probe_isolation_capabilities,
    run_isolated_subprocess,
    sanitized_environment,
)
from humanoid_agent.immutable_runner.safe_checkpoint import checkpoint_safety_report
from humanoid_agent.immutable_runner.train_start_paths import resolve_train_start_checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CHILD_SCRIPT = PROJECT_ROOT / "scripts" / "phase8e_training_safety_child.py"
MINIMAL_ROOTFS_SCRIPT = PROJECT_ROOT / "scripts" / "phase8e_minimal_rootfs_probe.py"
TRUSTED_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "phase8e_trusted_eval_probe.py"
REPO_READ_SECURITY_CHECK_KEYS = ("repo_hidden_config_read_blocked", "repo_config_read_blocked")
NON_SECURITY_BLOCKER_CHECK_KEYS = (
    "cuda_training_smoke_passed",
    "train_start_smoke_passed",
    "train_start_checkpoint_present",
)


class SafetyModelError(RuntimeError):
    """Raised when the Phase 8E-1 safety contract cannot be evaluated."""


def run_safety_model_probe(
    *,
    config_path: Path,
    candidate_dir: Path,
    checkpoint_path: Path,
    run_dir: Path,
    hidden_probe_file: Path | None = None,
    require_server_enforcement: bool = False,
) -> tuple[int, dict[str, Any]]:
    config = load_mapping(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    training_summary = run_training_boundary_probe(
        config=config,
        candidate_dir=candidate_dir.resolve(),
        run_dir=run_dir / "training_boundary",
        hidden_probe_file=hidden_probe_file.resolve() if hidden_probe_file else None,
        require_server_enforcement=require_server_enforcement,
    )
    eval_summary = run_trusted_eval_probe(
        config=config,
        checkpoint_path=checkpoint_path.resolve(),
        run_dir=run_dir / "trusted_eval",
    )
    eval_reject_summary = run_trusted_eval_probe(
        config=config,
        checkpoint_path=checkpoint_path.resolve(),
        run_dir=run_dir / "trusted_eval_negative",
        forbidden_candidate_dir=candidate_dir.resolve(),
    )
    eval_child = eval_summary.get("child_summary") if isinstance(eval_summary.get("child_summary"), dict) else {}
    training_security_enforced = _training_security_enforced(training_summary)
    cuda_required = _cuda_probe_required(config, training_summary)
    cuda_training_boundary_usable = not cuda_required or training_summary.get("status") == "pass"
    eval_checkpoint_safety = eval_child.get("checkpoint_safety") if isinstance(eval_child.get("checkpoint_safety"), dict) else {}
    checks = {
        "training_process_isolated": training_security_enforced,
        "training_parent_did_not_import_candidate": True,
        "cuda_training_boundary_usable": cuda_training_boundary_usable,
        "trusted_eval_passes_checkpoint_only": eval_summary.get("status") == "pass"
        and eval_child.get("candidate_code_loaded") is False,
        "trusted_eval_checkpoint_safe_load": eval_checkpoint_safety.get("safe_load") is True,
        "trusted_eval_rejects_candidate_dir": eval_reject_summary.get("status") == "reject",
        "formal_training_not_executed": True,
    }
    status = "pass" if all(checks.values()) else "fail"
    if status == "fail" and training_security_enforced and not cuda_training_boundary_usable:
        status = "blocked"
    summary = {
        "schema_version": str(config.get("summary_schema_version", "humanoid_agent.phase8e.safety_model.summary.v1")),
        "phase": str(config.get("phase", "8E-1-safety-model")),
        "status": status,
        "created_at_utc": _now(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir),
        "capabilities": probe_isolation_capabilities(),
        "checks": checks,
        "training_boundary": project_training_boundary(training_summary),
        "eval_boundary": project_eval_boundary(eval_summary),
        "eval_negative_boundary": project_eval_boundary(eval_reject_summary),
        "formal_training_executed": False,
    }
    summary_file = run_dir / "phase8e_safety_model_summary.json"
    write_json(summary_file, summary)
    print(json.dumps({"summary_file": str(summary_file), "status": status}, ensure_ascii=False))
    return (0 if status == "pass" else 2 if status == "blocked" else 1), summary


def run_training_boundary_probe(
    *,
    config: dict[str, Any],
    candidate_dir: Path,
    run_dir: Path,
    hidden_probe_file: Path | None,
    require_server_enforcement: bool,
) -> dict[str, Any]:
    training_cfg = _mapping(config.get("training_boundary", {}), "training_boundary")
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_artifacts_dir = run_dir / str(training_cfg.get("runner_artifacts_dirname", "runner_artifacts"))
    candidate_output_dir = run_dir / str(training_cfg.get("candidate_output_dirname", "candidate_output"))
    runner_artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    child_summary_file = candidate_output_dir / str(training_cfg.get("train_summary_filename", "public_train_boundary_summary.json"))
    log_file = runner_artifacts_dir / "phase8e_training_safety_child.log"
    timeout_seconds = int(training_cfg.get("timeout_seconds", 30))

    capabilities = probe_isolation_capabilities()
    require_network = bool(training_cfg.get("require_network_namespace", True))
    require_pid = bool(training_cfg.get("require_pid_namespace", True))
    require_landlock = bool(training_cfg.get("require_landlock", True))
    filesystem_policy = str(training_cfg.get("filesystem_policy", "landlock_minimal")).strip()
    hard_supported = bool(
        capabilities.get("network_namespace_supported")
        and capabilities.get("pid_namespace_supported")
        and (capabilities.get("landlock_abi_version") or not require_landlock)
    )
    if require_server_enforcement and not hard_supported:
        raise SafetyModelError("Phase8E-1 training boundary hard isolation is required but unavailable")
    if not hard_supported and (require_network or require_pid or require_landlock):
        return {
            "schema_version": "humanoid_agent.phase8e.training_boundary.summary.v1",
            "status": "pass" if not require_server_enforcement else "fail",
            "hard_isolation_enforced": False,
            "skipped_hard_checks": True,
            "capabilities": capabilities,
            "reason": "hard isolation requires Linux unshare + Landlock",
            "child_summary": None,
        }

    if filesystem_policy == "minimal_rootfs_pivot_root":
        return run_minimal_rootfs_training_boundary_probe(
            config=config,
            candidate_dir=candidate_dir,
            run_dir=run_dir,
            hidden_probe_file=hidden_probe_file,
            runner_artifacts_dir=runner_artifacts_dir,
            timeout_seconds=timeout_seconds,
            capabilities=capabilities,
            training_cfg=training_cfg,
        )
    if filesystem_policy not in {"landlock_minimal", "landlock"}:
        raise SafetyModelError(f"unsupported training filesystem_policy: {filesystem_policy}")

    command = [
        sys.executable,
        str(TRAINING_CHILD_SCRIPT),
        "--candidate-dir",
        str(candidate_dir),
        "--summary-file",
        str(child_summary_file),
        "--repo-root",
        str(PROJECT_ROOT),
        "--run-dir",
        str(candidate_output_dir),
        "--expected-schema-version",
        str(config.get("candidate_schema_version", "humanoid_agent.cold_start.candidate.v1")),
    ]
    if hidden_probe_file is not None:
        command.extend(["--hidden-probe-file", str(hidden_probe_file)])
    if require_landlock:
        command.append("--landlock")
    if bool(training_cfg.get("allow_cuda_device_paths", True)):
        command.append("--allow-cuda-device-paths")
    if bool(training_cfg.get("require_cuda_training_smoke", False)):
        command.extend(["--cuda-probe", "--cuda-device", str(training_cfg.get("cuda_device", "cuda:0"))])

    env = sanitized_environment(
        repo_root=PROJECT_ROOT,
        allowed_names=list(training_cfg.get("env_allowlist", [])) or None,
    )
    result = run_isolated_subprocess(
        command,
        cwd=candidate_output_dir,
        log_file=log_file,
        env=env,
        timeout_seconds=timeout_seconds,
        require_network_namespace=require_network,
        use_pid_namespace=require_pid and pid_namespace_supported(),
    )
    child_summary = read_json(child_summary_file)
    status = "pass" if result.exit_code == 0 and child_summary.get("status") == "pass" else "fail"
    if status == "fail" and _child_security_checks_passed(child_summary) and _child_cuda_probe_failed(child_summary):
        status = "blocked"
    return {
        "schema_version": "humanoid_agent.phase8e.training_boundary.summary.v1",
        "status": status,
        "hard_isolation_enforced": True,
        "filesystem_policy": filesystem_policy,
        "capabilities": capabilities,
        "command_result": result.to_summary(),
        "child_summary_file": str(child_summary_file),
        "child_summary": child_summary,
        "parent_imported_candidate_code": False,
        "candidate_output_dir": str(candidate_output_dir),
        "runner_artifacts_dir": str(runner_artifacts_dir),
    }


def run_minimal_rootfs_training_boundary_probe(
    *,
    config: dict[str, Any],
    candidate_dir: Path,
    run_dir: Path,
    hidden_probe_file: Path | None,
    runner_artifacts_dir: Path,
    timeout_seconds: int,
    capabilities: dict[str, Any],
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    minimal_run_dir = run_dir / "minimal_rootfs"
    child_summary_file = minimal_run_dir / "candidate_output" / str(
        training_cfg.get("train_summary_filename", "public_train_boundary_summary.json")
    )
    log_file = runner_artifacts_dir / "phase8e_minimal_rootfs_probe.log"
    command = [
        "unshare",
        "-Urmnfp",
        sys.executable,
        str(MINIMAL_ROOTFS_SCRIPT),
        "--candidate-dir",
        str(candidate_dir),
        "--run-dir",
        str(minimal_run_dir),
        "--summary-file",
        child_summary_file.name,
        "--repo-root",
        str(PROJECT_ROOT),
        "--device",
        str(training_cfg.get("cuda_device", "cuda:0")),
        "--expected-schema-version",
        str(config.get("candidate_schema_version", "humanoid_agent.cold_start.candidate.v1")),
    ]
    if hidden_probe_file is not None:
        command.extend(["--hidden-probe-file", str(hidden_probe_file)])
    env = sanitized_environment(
        repo_root=PROJECT_ROOT,
        allowed_names=list(training_cfg.get("env_allowlist", [])) or None,
    )
    start = datetime.now(timezone.utc)
    timed_out = False
    with log_file.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
    child_summary = read_json(child_summary_file)
    status = "pass" if exit_code == 0 and child_summary.get("status") == "pass" else "fail"
    parent_checkpoint_safety = _parent_train_start_checkpoint_safety(child_summary, minimal_run_dir)
    parent_checkpoint_unsafe = parent_checkpoint_safety is not None and parent_checkpoint_safety.get("safe_load") is not True
    if status == "pass" and parent_checkpoint_unsafe:
        status = "fail"
    if (
        status == "fail"
        and not parent_checkpoint_unsafe
        and _child_security_checks_passed(child_summary)
        and _child_cuda_probe_failed(child_summary)
    ):
        status = "blocked"
    return {
        "schema_version": "humanoid_agent.phase8e.training_boundary.summary.v1",
        "status": status,
        "hard_isolation_enforced": True,
        "filesystem_policy": "minimal_rootfs_pivot_root",
        "capabilities": capabilities,
        "command_result": {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": (datetime.now(timezone.utc) - start).total_seconds(),
            "command": command,
            "log_file": str(log_file),
        },
        "child_summary_file": str(child_summary_file),
        "child_summary": child_summary,
        "parent_checkpoint_safety": parent_checkpoint_safety,
        "parent_imported_candidate_code": False,
        "candidate_output_dir": str(minimal_run_dir / "candidate_output"),
        "runner_artifacts_dir": str(runner_artifacts_dir),
    }


def run_trusted_eval_probe(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    run_dir: Path,
    forbidden_candidate_dir: Path | None = None,
) -> dict[str, Any]:
    eval_cfg = _mapping(config.get("eval_boundary", {}), "eval_boundary")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / str(eval_cfg.get("summary_filename", "trusted_eval_probe_summary.json"))
    log_file = run_dir / "phase8e_trusted_eval_probe.log"
    timeout_seconds = int(eval_cfg.get("timeout_seconds", 30))
    command = [
        sys.executable,
        str(TRUSTED_EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint_path),
        "--summary-file",
        str(summary_file),
    ]
    if forbidden_candidate_dir is not None:
        command.extend(["--candidate-dir", str(forbidden_candidate_dir)])
    env = sanitized_environment(repo_root=PROJECT_ROOT)
    start = datetime.now(timezone.utc)
    with log_file.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    child_summary = read_json(summary_file)
    status = child_summary.get("status", "fail")
    return {
        "schema_version": "humanoid_agent.phase8e.eval_boundary.summary.v1",
        "status": status,
        "exit_code": completed.returncode,
        "elapsed_seconds": (datetime.now(timezone.utc) - start).total_seconds(),
        "summary_file": str(summary_file),
        "log_file": str(log_file),
        "child_summary": child_summary,
        "candidate_dir_supplied": forbidden_candidate_dir is not None,
    }


def project_training_boundary(summary: dict[str, Any]) -> dict[str, Any]:
    child = summary.get("child_summary") if isinstance(summary.get("child_summary"), dict) else {}
    probes = child.get("probes") if isinstance(child.get("probes"), dict) else {}
    return {
        "status": summary.get("status"),
        "hard_isolation_enforced": summary.get("hard_isolation_enforced"),
        "filesystem_policy": summary.get("filesystem_policy"),
        "skipped_hard_checks": summary.get("skipped_hard_checks", False),
        "parent_imported_candidate_code": summary.get("parent_imported_candidate_code", False),
        "checks": child.get("checks"),
        "cuda_training_smoke": probes.get("cuda_training_smoke") or probes.get("cuda"),
        "train_start_checkpoint_safety": summary.get("parent_checkpoint_safety")
        or (probes.get("train_start") or {}).get("checkpoint_safety"),
        "blocker": "cuda_training_smoke_failed" if summary.get("status") == "blocked" else None,
        "process_role": child.get("process_role"),
        "candidate_output_dir": summary.get("candidate_output_dir"),
        "runner_artifacts_dir": summary.get("runner_artifacts_dir"),
    }


def project_eval_boundary(summary: dict[str, Any]) -> dict[str, Any]:
    child = summary.get("child_summary") if isinstance(summary.get("child_summary"), dict) else {}
    return {
        "status": summary.get("status"),
        "exit_code": summary.get("exit_code"),
        "process_role": child.get("process_role"),
        "candidate_code_loaded": child.get("candidate_code_loaded"),
        "candidate_dir_supplied": summary.get("candidate_dir_supplied"),
        "eval_contract": child.get("eval_contract"),
        "checkpoint_safety": child.get("checkpoint_safety"),
        "reason": child.get("reason"),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SafetyModelError(f"{name} must be a mapping")
    return value


def _cuda_probe_required(config: dict[str, Any], training_summary: dict[str, Any]) -> bool:
    if training_summary.get("skipped_hard_checks"):
        return False
    training_cfg = _mapping(config.get("training_boundary", {}), "training_boundary")
    return bool(training_cfg.get("require_cuda_training_smoke", False))


def _training_security_enforced(training_summary: dict[str, Any]) -> bool:
    if training_summary.get("status") == "pass":
        return True
    if training_summary.get("status") != "blocked":
        return False
    child = training_summary.get("child_summary") if isinstance(training_summary.get("child_summary"), dict) else {}
    return _child_security_checks_passed(child)


def _child_security_checks_passed(child_summary: dict[str, Any]) -> bool:
    checks = child_summary.get("checks") if isinstance(child_summary.get("checks"), dict) else {}
    if not checks:
        return False
    if not any(checks.get(key) is True for key in REPO_READ_SECURITY_CHECK_KEYS):
        return False
    ignored = set(REPO_READ_SECURITY_CHECK_KEYS) | set(NON_SECURITY_BLOCKER_CHECK_KEYS)
    return all(value is True for key, value in checks.items() if key not in ignored)


def _child_cuda_probe_failed(child_summary: dict[str, Any]) -> bool:
    checks = child_summary.get("checks") if isinstance(child_summary.get("checks"), dict) else {}
    probes = child_summary.get("probes") if isinstance(child_summary.get("probes"), dict) else {}
    cuda_probe_value = probes.get("cuda_training_smoke") or probes.get("cuda")
    cuda_probe = cuda_probe_value if isinstance(cuda_probe_value, dict) else {}
    return checks.get("cuda_training_smoke_passed") is False and cuda_probe.get("status") == "fail"


def _parent_train_start_checkpoint_safety(child_summary: dict[str, Any], minimal_run_dir: Path) -> dict[str, Any] | None:
    """Trusted parent-side safe-load gate for a train-start checkpoint.

    The minimal-rootfs child may execute arbitrary candidate code while training.
    Its summary is useful as evidence, but checkpoint safety must be rechecked in
    this parent process, which never imports the candidate package.  The child
    reports paths from inside the pivot root; map those back to the host run tree
    before validating.
    """

    probes = child_summary.get("probes") if isinstance(child_summary.get("probes"), dict) else {}
    train_start = probes.get("train_start") if isinstance(probes.get("train_start"), dict) else {}
    latest_checkpoint = train_start.get("latest_checkpoint")
    if not isinstance(latest_checkpoint, str) or not latest_checkpoint:
        return None
    host_checkpoint = _isolated_training_path_to_host(latest_checkpoint, minimal_run_dir)
    if host_checkpoint is None:
        return {
            "safe_load": False,
            "isolated_path": latest_checkpoint,
            "host_path": None,
            "error": "checkpoint path is outside known isolated training artifact mounts",
        }
    report = checkpoint_safety_report(host_checkpoint)
    report["isolated_path"] = latest_checkpoint
    report["host_path"] = str(host_checkpoint)
    return report


def _isolated_training_path_to_host(path: str, minimal_run_dir: Path) -> Path | None:
    return resolve_train_start_checkpoint_path(path, minimal_run_dir / "runner_artifacts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

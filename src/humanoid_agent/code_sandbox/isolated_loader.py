"""Parent-side Phase 8C isolated candidate loader."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .isolation import (
    IsolationError,
    pid_namespace_supported,
    probe_isolation_capabilities,
    read_json_if_exists,
    run_isolated_subprocess,
    sanitized_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHILD_SCRIPT = PROJECT_ROOT / "scripts" / "phase8c_candidate_child.py"


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"{path} is not JSON-compatible and PyYAML is unavailable") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{path} must contain a mapping")
    return loaded


def run_isolated_component_load(
    *,
    candidate_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    allow_llm_authored: bool = False,
    require_noop_components: bool = False,
    fast_validation: Mapping[str, Any] | None = None,
    python_executable: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load candidate components in an isolated child process.

    The parent process never imports the candidate entrypoint.  It only launches
    ``phase8c_candidate_child.py`` with the configured isolation primitives and
    then reads the child summary.
    """

    isolation_cfg = config.get("isolation", {})
    candidate_policy = config.get("candidate_policy", {})
    if not isinstance(isolation_cfg, dict) or not isinstance(candidate_policy, dict):
        raise RuntimeError("Phase8C config must contain isolation and candidate_policy mappings")

    run_dir.mkdir(parents=True, exist_ok=True)
    runner_artifacts_dir = run_dir / "runner_artifacts"
    candidate_output_dir = run_dir / "candidate_output"
    runner_artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = candidate_output_dir / "phase8c_candidate_child_summary.json"
    log_file = runner_artifacts_dir / "phase8c_candidate_child.log"
    fast_validation_config_file: Path | None = None
    if fast_validation is not None and bool(fast_validation.get("enabled", False)):
        fast_validation_config_file = candidate_output_dir / "fast_validation_config.json"
        fast_validation_config_file.write_text(json.dumps(dict(fast_validation), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    timeout_seconds = int(isolation_cfg.get("timeout_seconds", 30))
    capabilities = probe_isolation_capabilities()
    child_python = str(python_executable or isolation_cfg.get("python_executable") or sys.executable)
    child_python_source = "argument" if python_executable else "isolation_config" if isolation_cfg.get("python_executable") else "sys.executable"

    require_network = bool(isolation_cfg.get("require_network_namespace", True))
    require_pid = bool(isolation_cfg.get("require_pid_namespace", True))
    require_landlock = bool(isolation_cfg.get("require_landlock", True))
    if require_network and not capabilities["network_namespace_supported"]:
        raise IsolationError("required network namespace is unavailable")
    if require_pid and not capabilities["pid_namespace_supported"]:
        raise IsolationError("required PID namespace is unavailable")
    if require_landlock and capabilities["landlock_abi_version"] is None:
        raise IsolationError("required Landlock filesystem isolation is unavailable")

    command = [
        child_python,
        str(CHILD_SCRIPT),
        "--candidate-dir",
        str(candidate_dir.resolve()),
        "--summary-file",
        str(summary_file),
        "--repo-root",
        str(PROJECT_ROOT),
        "--run-dir",
        str(candidate_output_dir.resolve()),
        "--expected-schema-version",
        str(candidate_policy.get("allowed_schema_version", "humanoid_agent.cold_start.candidate.v1")),
    ]
    if allow_llm_authored:
        command.append("--allow-llm-authored")
    if require_noop_components:
        command.append("--require-noop-components")
    if fast_validation_config_file is not None:
        command.extend(["--fast-validation-config", str(fast_validation_config_file)])
    if require_landlock:
        command.append("--landlock")

    env = sanitized_environment(
        repo_root=PROJECT_ROOT,
        allowed_names=list(isolation_cfg.get("env_allowlist", [])) or None,
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
    child_summary = read_json_if_exists(summary_file)
    status = "pass" if result.exit_code == 0 and child_summary and child_summary.get("status") == "pass" else "reject"
    summary = {
        "schema_version": "humanoid_agent.phase8c.isolated_component_load.summary.v1",
        "status": status,
        "capabilities": capabilities,
        "isolation": {
            "network_namespace": require_network,
            "pid_namespace": require_pid,
            "landlock": require_landlock,
            "environment": "sanitized_allowlist",
            "candidate_output_dir": str(candidate_output_dir),
            "runner_artifacts_dir": str(runner_artifacts_dir),
            "parent_imported_candidate_code": False,
            "dynamic_fast_validation": fast_validation_config_file is not None,
            "python_executable": child_python,
            "python_executable_source": child_python_source,
            "parent_python_executable": sys.executable,
        },
        "command_result": result.to_summary(),
        "child_summary_file": str(summary_file),
        "child_summary": child_summary,
        "errors": [] if status == "pass" else [{"type": "CandidateChildRejected", "message": "isolated child did not pass"}],
    }
    return (0 if status == "pass" else 1), summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def secret_present_in_environment(env: dict[str, str] | None = None) -> list[str]:
    """Return secret-looking variable names in an environment mapping."""

    env = dict(os.environ if env is None else env)
    return [
        key
        for key in sorted(env)
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
    ]

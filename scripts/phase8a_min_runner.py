#!/usr/bin/env python3
"""Phase 8A minimal immutable runner for a human-authored no-op candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.loader import (  # noqa: E402
    CandidateLoadError,
    candidate_package_fingerprint,
    load_candidate_manifest,
    load_training_components,
)
from humanoid_agent.immutable_runner.budget import build_budget_accounting, derive_max_iterations  # noqa: E402
from humanoid_agent.immutable_runner.checkpoints import checkpoint_iteration, find_latest_checkpoint_from_summary  # noqa: E402


def _load_mapping(path: Path) -> dict[str, Any]:
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


def _status_summary(
    status: str,
    *,
    schema_version: str = "humanoid_agent.phase8a.min_runner.summary.v1",
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "cwd": os.getcwd(),
            "argv": sys.argv,
        },
        **kwargs,
    }


def build_train_smoke_command(config: dict[str, Any], *, run_dir: Path) -> tuple[list[str], Path]:
    runner_cfg = effective_runner_config(config)
    train_summary = run_dir / runner_cfg.get("train_summary_filename", "train_start_smoke_summary.json")
    train_run_dir = run_dir / "train_start_smoke"
    command = [
        sys.executable,
        "scripts/phase1_push_recovery_train_smoke.py",
        "--headless",
        "--device",
        str(runner_cfg["device"]),
        "--task",
        str(runner_cfg["task"]),
        "--seed",
        str(runner_cfg["seed"]),
        "--profile",
        str(runner_cfg["train_profile"]),
        "--num-envs",
        str(runner_cfg["num_envs"]),
        "--max-iterations",
        str(runner_cfg["max_iterations"]),
        "--config",
        str(runner_cfg["phase1_config"]),
        "--phase2-config",
        str(runner_cfg["phase2_config"]),
        "--run-dir",
        str(train_run_dir),
        "--summary-file",
        str(train_summary),
    ]
    return command, train_summary


def effective_runner_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return runner config after applying immutable budget derivation."""

    runner_cfg = dict(config["runner"])
    budget_contract = config.get("budget_contract", {})
    if isinstance(budget_contract, dict) and budget_contract.get("derive_max_iterations_from_env_step_budget"):
        derived = derive_max_iterations(
            env_step_budget=int(budget_contract["env_step_budget"]),
            num_envs=int(runner_cfg["num_envs"]),
            steps_per_iteration=int(budget_contract["steps_per_iteration"]),
        )
        runner_cfg["max_iterations"] = int(derived["max_iterations"])
        runner_cfg["derived_budget"] = derived
    return runner_cfg


def write_commands_file(config: dict[str, Any], command: list[str], *, run_dir: Path) -> Path:
    filename = config["runner"].get("commands_filename", "commands.sh")
    path = run_dir / filename
    shell_command = " ".join(_shell_quote(part) for part in command)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "cd \"${PROJECT_ROOT:-/data/yuyangliu/AGENT}\"\n"
        "source scripts/activate_environment.sh\n"
        "export PYTHONPATH=\"$PWD/src${PYTHONPATH:+:$PYTHONPATH}\"\n\n"
        f"{shell_command}\n",
        encoding="utf-8",
    )
    return path


def _shell_quote(value: str) -> str:
    if value.replace("_", "").replace("-", "").replace(".", "").replace("/", "").replace(":", "").isalnum():
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def subprocess_environment() -> dict[str, str]:
    """Return a deterministic environment for runner-owned subprocesses."""

    env = os.environ.copy()
    src = str(SRC_ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    parts = [item for item in current_pythonpath.split(os.pathsep) if item]
    if src not in parts:
        parts.insert(0, src)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def validate_candidate_provenance(manifest, candidate_policy: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate provenance before executing candidate code.

    Phase 8A cannot safely infer provenance from ``manifest.llm_authored`` alone.
    The runner therefore requires an external human-candidate allowlist with
    resolved path and file hashes.
    """

    fingerprint = candidate_package_fingerprint(manifest)
    if manifest.llm_authored and not bool(candidate_policy.get("allow_llm_authored", False)):
        raise CandidateLoadError("LLM-authored candidate code is disabled before Phase 8C isolation")

    if not bool(candidate_policy.get("require_trusted_human_provenance", True)):
        return {
            "trusted": None,
            "policy": "trusted human provenance not required",
            "fingerprint": fingerprint,
        }

    trusted_candidates = candidate_policy.get("trusted_human_candidates", [])
    if not isinstance(trusted_candidates, list):
        raise CandidateLoadError("candidate_policy.trusted_human_candidates must be a list")

    for item in trusted_candidates:
        if not isinstance(item, dict):
            continue
        expected_path = item.get("path")
        if not isinstance(expected_path, str):
            continue
        expected_resolved = (PROJECT_ROOT / expected_path).resolve()
        if manifest.path != expected_resolved:
            continue
        if item.get("candidate_id") != manifest.candidate_id:
            continue
        if str(item.get("manifest_sha256", "")).lower() != fingerprint["manifest_sha256"]:
            continue
        if str(item.get("entrypoint_sha256", "")).lower() != fingerprint["entrypoint_sha256"]:
            continue
        expected_tree_sha256 = item.get("training_code_tree_sha256")
        if expected_tree_sha256 is not None and str(expected_tree_sha256).lower() != fingerprint["training_code_tree_sha256"]:
            continue
        return {
            "trusted": True,
            "policy": "trusted_human_candidates",
            "matched_candidate_id": manifest.candidate_id,
            "fingerprint": fingerprint,
        }

    raise CandidateLoadError(
        "untrusted candidate provenance: Phase 8A only runs externally allowlisted human no-op candidates"
    )


def run_phase8a(
    *,
    candidate_dir: Path,
    config_path: Path,
    run_dir: Path,
    execute_smoke: bool,
) -> tuple[int, dict[str, Any]]:
    config = _load_mapping(config_path)
    runner_cfg = config.get("runner", {})
    candidate_policy = config.get("candidate_policy", {})
    if not isinstance(runner_cfg, dict) or not isinstance(candidate_policy, dict):
        raise RuntimeError("Phase8A config must contain runner and candidate_policy mappings")
    effective_runner_cfg = effective_runner_config(config)
    budget_contract = config.get("budget_contract", {})
    phase_name = str(config.get("phase", "8A-min"))
    summary_schema_version = str(config.get("summary_schema_version", "humanoid_agent.phase8a.min_runner.summary.v1"))

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / runner_cfg.get("runner_summary_filename", "phase8a_runner_summary.json")

    try:
        manifest = load_candidate_manifest(
            candidate_dir,
            expected_schema_version=candidate_policy.get("allowed_schema_version"),
        )
        provenance = validate_candidate_provenance(manifest, candidate_policy)
        manifest, components = load_training_components(
            candidate_dir,
            expected_schema_version=candidate_policy.get("allowed_schema_version"),
            allow_llm_authored=bool(candidate_policy.get("allow_llm_authored", False)),
            task_spec={"task": effective_runner_cfg.get("task"), "phase": phase_name},
            base_env_cfg=None,
            base_agent_cfg=None,
        )
        if candidate_policy.get("require_noop_components", True) and not components.is_noop():
            raise CandidateLoadError("Phase 8A only accepts no-op components")

        command, train_summary = build_train_smoke_command(config, run_dir=run_dir)
        commands_file = write_commands_file(config, command, run_dir=run_dir)
        smoke_result: dict[str, Any] = {
            "executed": execute_smoke,
            "command": command,
            "commands_file": str(commands_file),
            "train_summary_file": str(train_summary),
        }
        status = "pass"
        exit_code = 0
        budget_accounting: dict[str, Any] | None = None
        if execute_smoke:
            log_file = run_dir / "train_start_smoke.log"
            timeout_seconds = int(effective_runner_cfg.get("timeout_seconds", 1800))
            start_time = datetime.now(timezone.utc)
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=subprocess_environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            smoke_result["exit_code"] = completed.returncode
            smoke_result["log_file"] = str(log_file)
            if completed.returncode != 0:
                status = "reject"
                exit_code = completed.returncode
                smoke_result["reject_reason"] = "train_start_smoke_failed"
            elif train_summary.is_file():
                train_summary_data = _load_mapping(train_summary)
                checkpoint = find_latest_checkpoint_from_summary(train_summary)
                smoke_result["latest_checkpoint"] = str(checkpoint) if checkpoint is not None else None
                smoke_result["latest_checkpoint_iteration"] = checkpoint_iteration(checkpoint) if checkpoint is not None else None
                if isinstance(budget_contract, dict) and budget_contract.get("primary_unit") == "env_steps" and "env_step_budget" in budget_contract:
                    budget_accounting = build_budget_accounting(
                        runner_config=effective_runner_cfg,
                        budget_contract=budget_contract,
                        train_summary=train_summary_data,
                        train_log_file=log_file,
                        elapsed_seconds=elapsed_seconds,
                        latest_checkpoint=checkpoint,
                    )
                    if budget_accounting["status"] != "pass":
                        status = "reject"
                        exit_code = 1
                        smoke_result["reject_reason"] = "budget_contract_violation"
                if checkpoint is None:
                    status = "reject"
                    exit_code = 1
                    smoke_result["reject_reason"] = "train_start_smoke_missing_checkpoint"
            else:
                status = "reject"
                exit_code = 1
                smoke_result["reject_reason"] = "train_start_smoke_missing_summary"

        summary = _status_summary(
            status,
            schema_version=summary_schema_version,
            phase=phase_name,
            config={
                "config_path": str(config_path),
                "runner": effective_runner_cfg,
                "source_runner": runner_cfg,
                "budget_contract": budget_contract,
                "candidate_policy": candidate_policy,
            },
            candidate=manifest.to_public_summary(),
            provenance=provenance,
            components=components.to_public_summary(),
            smoke=smoke_result,
            budget_accounting=budget_accounting,
            errors=[],
        )
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        summary = _status_summary(
            "reject",
            schema_version=summary_schema_version if "summary_schema_version" in locals() else "humanoid_agent.phase8a.min_runner.summary.v1",
            phase=phase_name if "phase_name" in locals() else "8A-min",
            config={"config_path": str(config_path)},
            candidate={"path": str(candidate_dir)},
            smoke={"executed": False},
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )

    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_file), "status": summary["status"]}, ensure_ascii=False))
    return exit_code, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--config", default="configs/cold_start/phase8a_min.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--execute-smoke", action="store_true", help="Run the 1-iteration train-start smoke.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = _load_mapping(config_path)
    default_run_root = Path(config["runner"].get("run_root", "runs/phase_8_cold_start_autoresearch"))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (default_run_root / "phase8a_min_noop").resolve()
    exit_code, _summary = run_phase8a(
        candidate_dir=Path(args.candidate_dir).resolve(),
        config_path=config_path,
        run_dir=run_dir,
        execute_smoke=args.execute_smoke,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

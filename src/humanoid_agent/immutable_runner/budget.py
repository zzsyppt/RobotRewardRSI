"""Budget accounting for immutable runner executions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .checkpoints import list_model_checkpoints


def derive_max_iterations(*, env_step_budget: int, num_envs: int, steps_per_iteration: int) -> dict[str, Any]:
    """Derive a max-iteration count that does not exceed ``env_step_budget``."""

    if env_step_budget <= 0:
        raise ValueError("env_step_budget must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if steps_per_iteration <= 0:
        raise ValueError("steps_per_iteration must be positive")

    env_steps_per_iteration = num_envs * steps_per_iteration
    max_iterations = env_step_budget // env_steps_per_iteration
    if max_iterations < 1:
        raise ValueError("env_step_budget is smaller than one training iteration")
    planned_env_steps = max_iterations * env_steps_per_iteration
    return {
        "env_step_budget": env_step_budget,
        "num_envs": num_envs,
        "steps_per_iteration": steps_per_iteration,
        "env_steps_per_iteration": env_steps_per_iteration,
        "max_iterations": max_iterations,
        "planned_env_steps": planned_env_steps,
        "unused_env_steps": env_step_budget - planned_env_steps,
        "exact_budget_match": planned_env_steps == env_step_budget,
    }


def parse_total_timesteps(log_file: str | Path) -> int | None:
    """Deprecated stdout fallback for trusted historical debugging only."""

    import re

    total_timesteps_re = re.compile(r"Total timesteps:\s*([0-9]+)")

    path = Path(log_file)
    if not path.is_file():
        return None
    latest: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = total_timesteps_re.search(line)
        if match:
            latest = int(match.group(1))
    return latest


def checkpoint_accounting(log_dir: str | Path, *, latest_checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Return checkpoint count, sizes, and numeric latest selection."""

    infos = list_model_checkpoints(log_dir)
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for info in infos:
        size_bytes = info.path.stat().st_size if info.path.is_file() else 0
        total_bytes += size_bytes
        rows.append(
            {
                "path": str(info.path),
                "iteration": info.iteration,
                "size_bytes": size_bytes,
            }
        )
    return {
        "checkpoint_count": len(rows),
        "checkpoint_total_bytes": total_bytes,
        "checkpoints": rows,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else (rows[-1]["path"] if rows else None),
        "latest_checkpoint_iteration": rows[-1]["iteration"] if rows else None,
    }


def build_budget_accounting(
    *,
    runner_config: dict[str, Any],
    budget_contract: dict[str, Any],
    train_summary: dict[str, Any] | None = None,
    train_log_file: str | Path | None = None,
    elapsed_seconds: float | None = None,
    latest_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build an audit summary for one immutable-runner train execution."""

    num_envs = int(runner_config["num_envs"])
    max_iterations = int(runner_config["max_iterations"])
    steps_per_iteration = int(budget_contract["steps_per_iteration"])
    env_steps_per_iteration = num_envs * steps_per_iteration
    planned_env_steps = max_iterations * env_steps_per_iteration
    env_step_budget = int(budget_contract["env_step_budget"])

    config_derived_total_timesteps = _read_nested_int(
        train_summary or {},
        ("training_accounting", "config_derived_total_timesteps"),
    )
    config_derived_total_timesteps_source = (
        "training_summary.training_accounting.config_derived_total_timesteps"
        if config_derived_total_timesteps is not None
        else None
    )
    if (
        config_derived_total_timesteps is None
        and train_log_file is not None
        and bool(budget_contract.get("allow_stdout_total_timesteps_fallback", False))
    ):
        config_derived_total_timesteps = parse_total_timesteps(train_log_file)
        config_derived_total_timesteps_source = "stdout_total_timesteps_fallback" if config_derived_total_timesteps is not None else None

    training = train_summary.get("training", {}) if isinstance(train_summary, dict) else {}
    log_dir = training.get("log_dir") if isinstance(training, dict) else None
    checkpoint_summary = (
        checkpoint_accounting(log_dir, latest_checkpoint=latest_checkpoint)
        if isinstance(log_dir, str) and log_dir
        else {
            "checkpoint_count": None,
            "checkpoint_total_bytes": None,
            "checkpoints": [],
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
            "latest_checkpoint_iteration": None,
        }
    )

    secondary_caps = budget_contract.get("secondary_caps", {})
    if not isinstance(secondary_caps, dict):
        secondary_caps = {}

    violations: list[dict[str, Any]] = []
    _check_cap(violations, "planned_env_steps", planned_env_steps, env_step_budget)
    require_config_derived_total_timesteps = bool(budget_contract.get("require_config_derived_total_timesteps", True))
    if require_config_derived_total_timesteps and config_derived_total_timesteps is None:
        violations.append({"key": "config_derived_total_timesteps", "actual": None, "required": True})
    if config_derived_total_timesteps is not None:
        _check_cap(violations, "config_derived_total_timesteps", config_derived_total_timesteps, env_step_budget)
        tolerance = int(budget_contract.get("config_derived_planned_tolerance_env_steps", 0))
        require_config_derived_matches_planned = bool(budget_contract.get("require_config_derived_matches_planned", True))
        if require_config_derived_matches_planned and abs(config_derived_total_timesteps - planned_env_steps) > tolerance:
            violations.append(
                {
                    "key": "config_derived_total_timesteps",
                    "actual": config_derived_total_timesteps,
                    "expected": planned_env_steps,
                    "tolerance": tolerance,
                }
            )
    summary_steps_per_iteration = _read_nested_int(train_summary or {}, ("training_accounting", "num_steps_per_env"))
    if (
        budget_contract.get("require_summary_steps_per_iteration_matches_budget", True)
        and summary_steps_per_iteration is not None
        and summary_steps_per_iteration != steps_per_iteration
    ):
        violations.append(
            {
                "key": "num_steps_per_env",
                "actual": summary_steps_per_iteration,
                "expected": steps_per_iteration,
            }
        )
    policy_update_iterations = _read_nested_int(train_summary or {}, ("training_accounting", "policy_update_iterations"))
    estimated_optimizer_steps = _read_nested_int(train_summary or {}, ("training_accounting", "estimated_optimizer_steps"))
    estimated_gradient_steps = _read_nested_int(train_summary or {}, ("training_accounting", "estimated_gradient_steps"))
    estimated_forward_backward_calls = _read_nested_int(
        train_summary or {},
        ("training_accounting", "estimated_forward_backward_calls"),
    )
    _check_optional_cap(violations, "policy_update_iterations", policy_update_iterations, secondary_caps, "max_policy_update_iterations")
    _check_optional_cap(violations, "estimated_optimizer_steps", estimated_optimizer_steps, secondary_caps, "max_optimizer_steps")
    _check_optional_cap(violations, "estimated_gradient_steps", estimated_gradient_steps, secondary_caps, "max_gradient_steps")
    _check_optional_cap(
        violations,
        "estimated_forward_backward_calls",
        estimated_forward_backward_calls,
        secondary_caps,
        "max_forward_backward_calls",
    )
    if elapsed_seconds is not None and "wall_clock_seconds" in secondary_caps:
        _check_cap(violations, "elapsed_seconds", elapsed_seconds, float(secondary_caps["wall_clock_seconds"]))
    if checkpoint_summary["checkpoint_count"] is not None and "max_checkpoints" in secondary_caps:
        _check_cap(violations, "checkpoint_count", checkpoint_summary["checkpoint_count"], int(secondary_caps["max_checkpoints"]))
    if checkpoint_summary["checkpoint_total_bytes"] is not None and "max_checkpoint_total_bytes" in secondary_caps:
        _check_cap(
            violations,
            "checkpoint_total_bytes",
            checkpoint_summary["checkpoint_total_bytes"],
            int(secondary_caps["max_checkpoint_total_bytes"]),
        )

    return {
        "schema_version": "humanoid_agent.immutable_runner.budget_accounting.v1",
        "primary_unit": "env_steps",
        "status": "pass" if not violations else "fail",
        "env_step_budget": env_step_budget,
        "num_envs": num_envs,
        "steps_per_iteration": steps_per_iteration,
        "env_steps_per_iteration": env_steps_per_iteration,
        "max_iterations": max_iterations,
        "planned_env_steps": planned_env_steps,
        "config_derived_total_timesteps": config_derived_total_timesteps,
        "config_derived_total_timesteps_source": config_derived_total_timesteps_source,
        "total_timesteps_semantics": "config_derived_not_runtime_observed",
        "policy_update_iterations": policy_update_iterations,
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "estimated_gradient_steps": estimated_gradient_steps,
        "estimated_forward_backward_calls": estimated_forward_backward_calls,
        "elapsed_seconds": elapsed_seconds,
        "secondary_caps": secondary_caps,
        "checkpoint_accounting": checkpoint_summary,
        "violations": violations,
        "created_unix_time": time.time(),
    }


def _check_cap(violations: list[dict[str, Any]], key: str, actual: float | int, cap: float | int) -> None:
    if actual > cap:
        violations.append({"key": key, "actual": actual, "cap": cap})


def _check_optional_cap(
    violations: list[dict[str, Any]],
    key: str,
    actual: int | None,
    caps: dict[str, Any],
    cap_key: str,
) -> None:
    if cap_key not in caps:
        return
    if actual is None:
        violations.append({"key": key, "actual": None, "cap_key": cap_key, "required": True})
        return
    _check_cap(violations, key, actual, int(caps[cap_key]))


def _read_nested_int(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return int(current)
    except (TypeError, ValueError):
        return None

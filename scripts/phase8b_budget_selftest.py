#!/usr/bin/env python3
"""Self-test for Phase 8B env-step budget accounting."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.immutable_runner.budget import (  # noqa: E402
    build_budget_accounting,
    derive_max_iterations,
    parse_total_timesteps,
)
from humanoid_agent.immutable_runner.checkpoints import find_latest_checkpoint_from_summary  # noqa: E402
from scripts.phase8a_min_runner import _load_mapping, build_train_smoke_command, run_phase8a  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8b_budget.yaml"
NOOP_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "noop_candidate"


def _touch(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    config = _load_mapping(CONFIG_PATH)
    runner = config["runner"]
    budget = config["budget_contract"]

    derived = derive_max_iterations(
        env_step_budget=int(budget["env_step_budget"]),
        num_envs=int(runner["num_envs"]),
        steps_per_iteration=int(budget["steps_per_iteration"]),
    )
    _assert(derived["max_iterations"] == 500, "4096-env budget should derive 500 iterations")
    _assert(derived["planned_env_steps"] == int(budget["env_step_budget"]), "derived budget should match exactly")

    command, _summary = build_train_smoke_command(config, run_dir=Path("/tmp/phase8b_budget_selftest"))
    _assert(command[command.index("--num-envs") + 1] == "4096", "Phase8B runner must keep 4096 envs")
    _assert(command[command.index("--max-iterations") + 1] == "500", "Phase8B runner must derive 500 iterations")

    with tempfile.TemporaryDirectory(prefix="phase8b_budget_selftest_") as tmp:
        tmp_root = Path(tmp)
        log_file = tmp_root / "train.log"
        log_file.write_text(
            "Total timesteps: 98304\n"
            f"Total timesteps: {budget['env_step_budget']}\n",
            encoding="utf-8",
        )
        _assert(parse_total_timesteps(log_file) == int(budget["env_step_budget"]), "must parse latest total timesteps")

        log_dir = tmp_root / "rsl_rl"
        _touch(log_dir / "model_1.pt")
        _touch(log_dir / "model_900.pt")
        _touch(log_dir / "model_1999.pt")
        _touch(log_dir / "model_final.pt")
        train_summary = {
            "status": "pass",
            "training": {
                "num_envs": 4096,
                "max_iterations": 500,
                "log_dir": str(log_dir),
                "latest_checkpoint": str(log_dir / "model_900.pt"),
            },
            "training_accounting": {
                "source": "runner_owned_training_script",
                "num_envs": 4096,
                "max_iterations": 500,
                "num_steps_per_env": 24,
                "env_steps_per_iteration": 98304,
                "config_derived_total_timesteps": int(budget["env_step_budget"]),
                "total_timesteps_semantics": "config_derived_not_runtime_observed",
                "policy_update_iterations": 500,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "estimated_optimizer_steps": 10000,
                "estimated_gradient_steps": 10000,
                "estimated_forward_backward_calls": 10000,
            },
        }
        latest = find_latest_checkpoint_from_summary(train_summary)
        _assert(latest == log_dir / "model_1999.pt", "checkpoint policy must still select numeric latest")

        accounting = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=budget,
            train_summary=train_summary,
            train_log_file=log_file,
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(accounting["status"] == "pass", "exact budget should pass")
        _assert(accounting["config_derived_total_timesteps"] == int(budget["env_step_budget"]), "derived env steps mismatch")
        _assert(
            accounting["config_derived_total_timesteps_source"]
            == "training_summary.training_accounting.config_derived_total_timesteps",
            "derived env steps source mismatch",
        )
        _assert(accounting["estimated_optimizer_steps"] == 10000, "optimizer-step estimate should be preserved")
        _assert(accounting["checkpoint_accounting"]["latest_checkpoint_iteration"] == 1999, "checkpoint audit mismatch")

        forged_overrun_log = tmp_root / "forged_overrun.log"
        forged_overrun_log.write_text(f"Total timesteps: {int(budget['env_step_budget']) + 1}\n", encoding="utf-8")
        forged_ignored = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=budget,
            train_summary=train_summary,
            train_log_file=forged_overrun_log,
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(forged_ignored["status"] == "pass", "stdout forged overrun should be ignored when structured summary exists")
        _assert(
            forged_ignored["config_derived_total_timesteps_source"]
            == "training_summary.training_accounting.config_derived_total_timesteps",
            "derived env steps must come from structured training summary",
        )

        overrun_summary = dict(train_summary)
        overrun_accounting = dict(train_summary["training_accounting"])
        overrun_accounting["config_derived_total_timesteps"] = int(budget["env_step_budget"]) + 1
        overrun_summary["training_accounting"] = overrun_accounting
        overrun = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=budget,
            train_summary=overrun_summary,
            train_log_file=log_file,
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(overrun["status"] == "fail", "env-step overrun must fail")
        _assert(any(item["key"] == "config_derived_total_timesteps" for item in overrun["violations"]), "missing overrun violation")

        missing_derived = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=budget,
            train_summary={"status": "pass", "training": train_summary["training"]},
            train_log_file=tmp_root / "missing.log",
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(missing_derived["status"] == "fail", "missing derived total timesteps must fail closed")

        mismatched_steps_summary = dict(train_summary)
        mismatched_accounting = dict(train_summary["training_accounting"])
        mismatched_accounting["num_steps_per_env"] = 23
        mismatched_steps_summary["training_accounting"] = mismatched_accounting
        mismatched_steps = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=budget,
            train_summary=mismatched_steps_summary,
            train_log_file=log_file,
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(mismatched_steps["status"] == "fail", "num_steps_per_env mismatch must fail")

        tight_budget = dict(budget)
        tight_budget["secondary_caps"] = dict(budget["secondary_caps"])
        tight_budget["secondary_caps"]["max_checkpoints"] = 1
        checkpoint_fail = build_budget_accounting(
            runner_config={"num_envs": 4096, "max_iterations": 500},
            budget_contract=tight_budget,
            train_summary=train_summary,
            train_log_file=log_file,
            elapsed_seconds=12.0,
            latest_checkpoint=latest,
        )
        _assert(checkpoint_fail["status"] == "fail", "checkpoint count cap must fail")

        dry_run_dir = tmp_root / "dry_run"
        exit_code, summary = run_phase8a(
            candidate_dir=NOOP_CANDIDATE,
            config_path=CONFIG_PATH,
            run_dir=dry_run_dir,
            execute_smoke=False,
        )
        _assert(exit_code == 0, "Phase8B dry-run should pass")
        _assert(summary["phase"] == "8B-budget", "Phase8B dry-run phase mismatch")
        _assert(summary["config"]["runner"]["max_iterations"] == 500, "summary must expose derived iterations")

    print(json.dumps({"status": "pass", "test": "phase8b_budget_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

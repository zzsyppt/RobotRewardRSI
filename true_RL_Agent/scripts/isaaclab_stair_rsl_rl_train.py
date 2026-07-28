#!/usr/bin/env python3
"""Preflight a production Stair candidate train job for Phase B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    StairTrainEntrypointContext,
    validate_stair_train_invocation,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "--job-id",
        "--round",
        "--task-id",
        "--subject-kind",
        "--subject-id",
        "--subject-manifest-hash",
        "--program-source-checksum",
        "--candidate",
        "--candidate-artifact-hash",
        "--init",
        "--training-binding-hash",
        "--policy-architecture-hash",
        "--optimizer-training-profile-hash",
        "--agent-configuration-hash",
        "--agent-configuration-json",
        "--initialization-seed-schedule-hash",
        "--capability-profile-hash",
        "--summary-path",
        "--checkpoint-output",
        "--profile",
        "--runtime-dynamics-hash",
        "--config",
    ):
        parser.add_argument(option, required=True)
    for option in (
        "--budget-steps",
        "--num-envs",
        "--num-steps-per-env",
        "--max-iterations",
        "--seed",
    ):
        parser.add_argument(option, type=int, required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-checkpoint-sha256", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        context = validate_stair_train_invocation(args, project_root=PROJECT_ROOT)
        if args.headless is not True:
            raise ValueError("stair production train must be headless")
    except (TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary_path = _local_output_path(args.summary_path, "summary_path")
    summary = _phase_b_fail_closed_summary(
        context,
        round_id=args.round,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": summary["status"], "summary_path": str(summary_path)},
            sort_keys=True,
        )
    )
    return 1


def _phase_b_fail_closed_summary(
    context: StairTrainEntrypointContext,
    *,
    round_id: str,
) -> dict[str, Any]:
    binding = context.training_binding
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.train_summary.v1",
        **context.identity.to_public_dict(),
        "training_binding_hash": binding.binding_hash,
        "policy_architecture_hash": binding.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            binding.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": binding.agent_configuration_hash,
        "initialization_seed_schedule_hash": (
            binding.initialization_seed_schedule_hash
        ),
        "capability_profile_hash": binding.capability_profile_hash,
        "program_source_checksum": context.candidate_artifact.program_source_checksum,
        "status": "fail",
        "formal_training_executed": False,
        "runtime_activation_summary": {
            "status": "fail",
            "reason": "complete_reward_runtime_not_activated",
            "runtime_status": STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        },
        "training": {
            "budget_accounting": dict(context.budget_accounting),
            "num_envs": context.config.train_num_envs,
            "num_steps_per_env": context.config.train_steps_per_iteration,
            "max_iterations": context.config.train_max_iterations,
            "budget_steps": context.config.budget_steps,
        },
        "run": {"round_id": round_id},
        "errors": [
            {
                "type": "RuntimeError",
                "message": "stair complete reward runtime is not activated until Phase C",
            }
        ],
    }


def _local_output_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field_name} must be a local path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())

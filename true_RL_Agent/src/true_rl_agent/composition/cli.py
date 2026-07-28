"""Two-stage CLI helpers for task-owned defaults."""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

from true_rl_agent.composition.base import TaskCompositionProvider
from true_rl_agent.composition.registry import TaskCompositionRegistry

TASK_RUNTIME_DEFAULT_KEYS = (
    "isaac_python",
    "train_entrypoint",
    "eval_entrypoint",
    "budget_steps",
    "public_seed_count",
    "public_accept_seed_count",
    "sealed_validation_seed_count",
    "eval_protocol_version",
    "train_seed",
    "runtime_config",
    "train_profile",
    "eval_profile",
    "train_num_envs",
    "train_max_iterations",
    "train_steps_per_iteration",
    "eval_num_envs",
    "runner_timeout_s",
)


def select_task_composition_provider(
    argv: Sequence[str] | None,
    registry: TaskCompositionRegistry,
) -> TaskCompositionProvider:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-id", default=registry.default_task_id)
    selected, _ = parser.parse_known_args(argv)
    try:
        return registry.get(selected.task_id)
    except KeyError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error must exit") from exc


def add_task_selection_argument(
    parser: argparse.ArgumentParser,
    *,
    registry: TaskCompositionRegistry,
    provider: TaskCompositionProvider,
) -> None:
    parser.add_argument(
        "--task-id",
        choices=registry.task_ids(),
        default=provider.task_id,
    )


def add_task_runtime_arguments(
    parser: argparse.ArgumentParser,
    defaults: Mapping[str, Any],
) -> None:
    missing = [key for key in TASK_RUNTIME_DEFAULT_KEYS if key not in defaults]
    if missing:
        raise ValueError(
            "task composition CLI defaults missing: " + ", ".join(missing)
        )
    parser.add_argument("--isaac-python", default=defaults["isaac_python"])
    parser.add_argument("--train-entrypoint", default=defaults["train_entrypoint"])
    parser.add_argument("--eval-entrypoint", default=defaults["eval_entrypoint"])
    parser.add_argument(
        "--budget-steps",
        type=int,
        default=defaults["budget_steps"],
    )
    parser.add_argument(
        "--public-seed-count",
        type=int,
        default=defaults["public_seed_count"],
    )
    parser.add_argument(
        "--public-accept-seed-count",
        type=int,
        default=defaults["public_accept_seed_count"],
    )
    parser.add_argument(
        "--sealed-validation-seed-count",
        "--trusted-seed-count",
        dest="sealed_validation_seed_count",
        type=int,
        default=defaults["sealed_validation_seed_count"],
    )
    parser.add_argument(
        "--eval-protocol-version",
        default=defaults["eval_protocol_version"],
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=defaults["train_seed"],
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=defaults["runtime_config"],
        help=(
            "Optional task-owned config override; omitted uses the selected "
            "task's bound runtime protocol."
        ),
    )
    parser.add_argument("--train-profile", default=defaults["train_profile"])
    parser.add_argument("--eval-profile", default=defaults["eval_profile"])
    parser.add_argument(
        "--train-num-envs",
        type=int,
        default=defaults["train_num_envs"],
    )
    parser.add_argument(
        "--train-max-iterations",
        type=int,
        default=defaults["train_max_iterations"],
    )
    parser.add_argument(
        "--train-steps-per-iteration",
        type=int,
        default=defaults["train_steps_per_iteration"],
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=defaults["eval_num_envs"],
    )
    parser.add_argument(
        "--runner-timeout-s",
        type=int,
        default=defaults["runner_timeout_s"],
    )

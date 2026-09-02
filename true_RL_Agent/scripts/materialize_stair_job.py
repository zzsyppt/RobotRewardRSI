"""Materialize a production Stair candidate without launching PPO."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.materialize_job import main as materialize_generic_job  # noqa: E402

from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
)


def main(argv: list[str] | None = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_owned_options(user_argv)
    registry = build_application_composition_registry()
    return materialize_generic_job(
        [
            "--task-id",
            STAIR_TASK_ID,
            "--job-kind",
            "candidate",
            *user_argv,
        ],
        registry=registry,
    )


def _reject_owned_options(argv: list[str]) -> None:
    for option in ("--task-id", "--job-kind"):
        if option in argv or any(value.startswith(option + "=") for value in argv):
            raise SystemExit(f"{option} is owned by the Stair materializer entrypoint")


if __name__ == "__main__":
    raise SystemExit(main())

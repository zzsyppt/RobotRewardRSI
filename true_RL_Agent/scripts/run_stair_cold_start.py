"""Run Stair with no reference reward source and a bootstrap Genesis."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_loop import main as run_generic_loop  # noqa: E402
from scripts.run_stair_live import (  # noqa: E402
    _require_closed_stair_live_summary,
)

from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
)
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_REWARD_COLD_START_MODE,
)


def main(argv: list[str] | None = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_owned_options(user_argv)
    return run_generic_loop(
        [
            "--task-id",
            STAIR_TASK_ID,
            "--mode",
            "live",
            "--start-mode",
            STAIR_REWARD_COLD_START_MODE,
            *user_argv,
        ],
        registry=build_application_composition_registry(),
        require_existing_genesis=False,
        live_summary_validator=_require_closed_stair_live_summary,
    )


def _reject_owned_options(argv: list[str]) -> None:
    for option in ("--task-id", "--mode", "--start-mode"):
        if option in argv or any(value.startswith(option + "=") for value in argv):
            raise SystemExit(f"{option} is owned by the Stair cold-start entrypoint")


if __name__ == "__main__":
    raise SystemExit(main())

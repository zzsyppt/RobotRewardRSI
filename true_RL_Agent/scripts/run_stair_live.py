"""Run the production Stair live loop from an existing reference Genesis."""

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
    return run_generic_loop(
        ["--task-id", STAIR_TASK_ID, "--mode", "live", *user_argv],
        registry=registry,
        require_existing_genesis=True,
        live_summary_validator=_require_closed_stair_live_summary,
    )


def _reject_owned_options(argv: list[str]) -> None:
    for option in (
        "--task-id",
        "--mode",
        "--start-mode",
        "--session-genesis-dir",
        "--ledger-epoch-id",
    ):
        if option in argv or any(value.startswith(option + "=") for value in argv):
            raise SystemExit(f"{option} is owned by the Stair live entrypoint")


def _require_closed_stair_live_summary(summary: dict) -> None:
    requested = summary.get("rounds_requested")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("Stair live summary has an invalid requested round count")
    counts = {
        "rounds_completed": summary.get("rounds_completed"),
        "trained_candidate_count": summary.get("trained_candidate_count"),
        "full_eval_completed_count": summary.get("full_eval_completed_count"),
        "sealed_validation_completed_count": summary.get(
            "sealed_validation_completed_count"
        ),
    }
    invalid = [
        name
        for name, value in counts.items()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0
    ]
    if invalid:
        raise ValueError(
            "Stair live summary has invalid completion counts: "
            + ", ".join(invalid)
        )
    if counts["rounds_completed"] != requested:
        raise ValueError(
            "Stair live requires every requested round to produce a ledger record"
        )
    trained = counts["trained_candidate_count"]
    if trained > requested:
        raise ValueError(
            "Stair live trained candidate count exceeds requested rounds"
        )
    incomplete = [
        name
        for name in (
            "full_eval_completed_count",
            "sealed_validation_completed_count",
        )
        if counts[name] != trained
    ]
    if incomplete:
        raise ValueError(
            "Stair live requires every successfully trained candidate to complete "
            "full eval and sealed validation; mismatched: "
            + ", ".join(incomplete)
        )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create the Stair never-fed final confirmation plan before Session Genesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.composition.cli import add_task_runtime_arguments  # noqa: E402
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionConfig,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (  # noqa: E402
    create_stair_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_REFERENCE_START_MODE,
)


def _build_parser() -> argparse.ArgumentParser:
    defaults = StairClimbingCompositionConfig.defaults(
        project_root=PROJECT_ROOT
    ).to_task_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    add_task_runtime_arguments(parser, defaults)
    parser.add_argument(
        "--start-mode",
        choices=(STAIR_REFERENCE_START_MODE,),
        default=STAIR_REFERENCE_START_MODE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    composition = StairClimbingCompositionConfig.from_values(vars(args))
    config = composition.to_task_config(project_root=PROJECT_ROOT)
    plan, path = create_stair_final_confirmation_plan(
        output_dir=args.output_dir,
        session_id=args.session_id,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": "created",
                "path": str(path),
                "plan_hash": plan.plan_hash,
                "session_id": plan.session_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

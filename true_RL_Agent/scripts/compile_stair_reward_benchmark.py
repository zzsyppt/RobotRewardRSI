#!/usr/bin/env python3
"""Compile isolated Stair reward benchmark cases into one strict report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.json_contract import load_strict_json_object  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_performance import (  # noqa: E402
    compile_stair_reward_performance_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
    )
    reference = resolve_official_stair_reference_program(
        config,
        project_root=PROJECT_ROOT,
    )
    cases = [
        load_strict_json_object(Path(path))
        for path in args.case_report
    ]
    report = compile_stair_reward_performance_report(
        cases,
        protocol=config.protocol_v2,
        reference_program=reference.reward_program,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "status": report["status"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

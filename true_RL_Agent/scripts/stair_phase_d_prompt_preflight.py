#!/usr/bin/env python3
"""Validate the frozen historical Stair Phase D prompt evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.local_paths import (  # noqa: E402
    resolve_existing_file,
)
from true_rl_agent.tasks.stair_climbing.phase_d_prompt_preflight import (  # noqa: E402
    load_stair_phase_d_prompt_preflight_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="PLAN/stair/phase_d_prompt_preflight_report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report_path = _project_report_path(args.report)
        report = load_stair_phase_d_prompt_preflight_report(report_path)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed
        print(str(exc), file=sys.stderr)
        return 1
    print(
        "stair Phase D frozen prompt evidence is valid "
        f"report_hash={report['report_hash']}",
        flush=True,
    )
    return 0


def _project_report_path(value: object) -> Path:
    path = Path(str(value))
    unresolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolve_existing_file(
        unresolved,
        field_name="Phase D frozen report",
    )
    try:
        resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("Phase D report must remain inside the project root") from exc
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())

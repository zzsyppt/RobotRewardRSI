#!/usr/bin/env python3
"""Phase 8B budget-accounting runner wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.phase8a_min_runner import _load_mapping, run_phase8a  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", default="examples/cold_start_candidates/noop_candidate")
    parser.add_argument("--config", default="configs/cold_start/phase8b_budget.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--execute-smoke", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = _load_mapping(config_path)
    default_run_root = Path(config["runner"].get("run_root", "runs/phase_8_cold_start_autoresearch"))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (default_run_root / "phase8b_budget_noop").resolve()
    exit_code, _summary = run_phase8a(
        candidate_dir=Path(args.candidate_dir).resolve(),
        config_path=config_path,
        run_dir=run_dir,
        execute_smoke=args.execute_smoke,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

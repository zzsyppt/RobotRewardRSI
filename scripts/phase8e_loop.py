#!/usr/bin/env python3
"""Prepare or verify one Phase 8E code-level cold-start loop round.

This command does not launch formal training.  It writes a public-only prompt
package, optionally verifies a candidate through Phase 8D, and optionally
records a public decision/ledger entry when an external public result is
provided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e import Phase8EError, run_phase8e_round  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_loop.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--lab-dir", default=None, help="Stable experiment dir for the ledger and accepted-recipe snapshots; required for multi-round inheritance.")
    parser.add_argument("--candidate-dir", default=None, help="Optional candidate package to verify through Phase 8D.")
    parser.add_argument("--current-recipe-dir", default=None, help="Override current best recipe for prompt construction.")
    parser.add_argument("--public-result", default=None, help="Optional public-only training/eval result JSON.")
    parser.add_argument("--append-ledger", action="store_true", help="Append candidate decision to the Phase 8E ledger.")
    parser.add_argument("--inherit-if-accepted", action="store_true", help="Copy accepted candidate as current best recipe.")
    parser.add_argument("--require-server-enforcement", action="store_true", help="Require Phase 8C hard isolation inside Phase 8D.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = PROJECT_ROOT / "runs" / "phase_8_cold_start_autoresearch" / "phase8e_loop"
    try:
        exit_code, _summary = run_phase8e_round(
            config_path=config_path,
            run_dir=run_dir,
            candidate_dir=Path(args.candidate_dir).resolve() if args.candidate_dir else None,
            current_recipe_dir=Path(args.current_recipe_dir).resolve() if args.current_recipe_dir else None,
            public_result_path=Path(args.public_result).resolve() if args.public_result else None,
            append_ledger=args.append_ledger,
            inherit_if_accepted=args.inherit_if_accepted,
            require_server_enforcement=args.require_server_enforcement,
            lab_dir=Path(args.lab_dir).resolve() if args.lab_dir else None,
        )
    except Phase8EError as exc:
        print(f"phase8e_error={exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

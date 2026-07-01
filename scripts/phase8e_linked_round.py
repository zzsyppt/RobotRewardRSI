#!/usr/bin/env python3
"""Run Phase 8E-6-2 linked trusted eval and decision for one trained candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e import Phase8EError  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_linked import Phase8ELinkedError, run_phase8e_linked_round  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_loop.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lab-dir", default=None, help="Stable experiment dir for the ledger and accepted-recipe snapshots.")
    parser.add_argument("--candidate-dir", required=True, help="Candidate package already produced by the Agent/provider.")
    parser.add_argument("--train-summary", required=True, help="Runner-owned training summary containing train_start.latest_checkpoint.")
    parser.add_argument(
        "--runner-artifacts-root",
        required=True,
        help="Runner-owned artifacts root that contains train_start_smoke/sealed_checkpoints; never infer this from candidate output.",
    )
    parser.add_argument("--public-dev-summary", action="append", required=True, help="Trusted public-dev rollout summary. Repeatable.")
    parser.add_argument("--public-accept-summary", action="append", default=[], help="Trusted public-accept rollout summary. Repeatable.")
    parser.add_argument("--curriculum-lineage-path", default=None, help="Runner curriculum compiler lineage JSON for accepted ledger/snapshot attribution.")
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--inherit-if-accepted", action="store_true")
    parser.add_argument("--require-server-enforcement", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        exit_code, _summary = run_phase8e_linked_round(
            config_path=Path(args.config).resolve(),
            run_dir=Path(args.run_dir).resolve(),
            lab_dir=Path(args.lab_dir).resolve() if args.lab_dir else None,
            candidate_dir=Path(args.candidate_dir).resolve(),
            train_summary_path=Path(args.train_summary).resolve(),
            runner_artifacts_root=Path(args.runner_artifacts_root).resolve(),
            public_dev_summary_paths=[Path(path).resolve() for path in args.public_dev_summary],
            public_accept_summary_paths=[Path(path).resolve() for path in args.public_accept_summary],
            curriculum_lineage_path=Path(args.curriculum_lineage_path).resolve() if args.curriculum_lineage_path else None,
            append_ledger=args.append_ledger,
            inherit_if_accepted=args.inherit_if_accepted,
            require_server_enforcement=args.require_server_enforcement,
        )
    except (Phase8ELinkedError, Phase8EError) as exc:
        print(f"phase8e_linked_error={exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

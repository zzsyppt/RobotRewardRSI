#!/usr/bin/env python3
"""Run a capped Phase 8E self-evolution loop (default 3 rounds, then stop).

Each round generates a candidate from the provider, runs fixed-budget isolated
training, trusted public eval, and a decision.  Rounds share one lab_dir so the
provider prompt inherits accumulated public feedback and the latest accepted
recipe.  Policy weights cold-start every round; only recipe code is inherited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e_evolution_loop import (  # noqa: E402
    Phase8EEvolutionLoopError,
    run_phase8e_evolution_loop,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_loop.yaml")
    parser.add_argument("--run-root", required=True, help="Root directory; each round lands in run-root/round_NNN.")
    parser.add_argument("--lab-dir", default=None, help="Shared ledger/accepted-recipe dir. Defaults to run-root/lab.")
    parser.add_argument("--max-rounds", type=int, default=15, help="Run up to this many rounds, then stop (default 15).")
    parser.add_argument("--stop-after-no-improve", type=int, default=None, help="Optional early stop after N consecutive rounds with no new best public-accept mean.")
    parser.add_argument("--provider", default=None, help="Provider name from config. Defaults to config.provider.default.")
    parser.add_argument("--replay-response", default=None, help="Replay-provider JSON response file (reused every round).")
    parser.add_argument("--allow-api-spend", action="store_true", help="Required for network/API providers (e.g. deepseek).")
    parser.add_argument(
        "--no-require-server-enforcement",
        action="store_true",
        help="Disable hard server-boundary enforcement. Intended only for parser/local debugging.",
    )
    parser.add_argument("--python-executable", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        exit_code, aggregate = run_phase8e_evolution_loop(
            config_path=Path(args.config),
            run_root=Path(args.run_root),
            lab_dir=Path(args.lab_dir) if args.lab_dir else None,
            max_rounds=args.max_rounds,
            provider_name=args.provider,
            replay_response=Path(args.replay_response) if args.replay_response else None,
            allow_api_spend=args.allow_api_spend,
            require_server_enforcement=not args.no_require_server_enforcement,
            python_executable=args.python_executable,
            stop_after_no_improve=args.stop_after_no_improve,
        )
    except Phase8EEvolutionLoopError as exc:
        print(f"phase8e_evolution_loop_error={exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "summary_file": aggregate.get("summary_file"),
                "rounds_executed": aggregate.get("rounds_executed"),
                "accepted_rounds": aggregate.get("accepted_rounds"),
                "best_public_accept_mean": aggregate.get("best_public_accept_mean"),
                "stop_reason": aggregate.get("stop_reason"),
            },
            ensure_ascii=False,
        )
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

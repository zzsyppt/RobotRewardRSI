#!/usr/bin/env python3
"""Run one complete Phase 8E code-level cold-start round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e_full_round import (  # noqa: E402
    Phase8EFullRoundError,
    run_phase8e_full_round,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_loop.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lab-dir", default=None, help="Stable ledger/accepted-recipe directory. Defaults to run-dir.")
    parser.add_argument("--candidate-dir", default=None, help="Preexisting candidate package. If omitted, the provider is called.")
    parser.add_argument("--provider", default=None, help="Provider name from config. Defaults to config.provider.default.")
    parser.add_argument("--replay-response", default=None, help="Replay-provider JSON response file.")
    parser.add_argument("--allow-api-spend", action="store_true", help="Required for network/API providers.")
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--inherit-if-accepted", action="store_true")
    parser.add_argument(
        "--no-require-server-enforcement",
        action="store_true",
        help="Disable hard server-boundary enforcement. Intended only for parser/local debugging, not real code-level rounds.",
    )
    parser.add_argument("--python-executable", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        exit_code, summary = run_phase8e_full_round(
            config_path=Path(args.config),
            run_dir=Path(args.run_dir),
            lab_dir=Path(args.lab_dir) if args.lab_dir else None,
            candidate_dir=Path(args.candidate_dir) if args.candidate_dir else None,
            provider_name=args.provider,
            replay_response=Path(args.replay_response) if args.replay_response else None,
            allow_api_spend=args.allow_api_spend,
            append_ledger=args.append_ledger,
            inherit_if_accepted=args.inherit_if_accepted,
            require_server_enforcement=not args.no_require_server_enforcement,
            python_executable=args.python_executable,
        )
    except Phase8EFullRoundError as exc:
        print(f"phase8e_full_round_error={exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "summary_file": summary.get("summary_file"),
                "status": summary.get("status"),
            },
            ensure_ascii=False,
        )
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

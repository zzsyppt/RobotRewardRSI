#!/usr/bin/env python3
"""Verify one Phase 8D code-level candidate package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.verifier import verify_candidate_package, write_summary  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--config", default="configs/cold_start/phase8d_verifier.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--require-server-enforcement", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else PROJECT_ROOT / "runs" / "phase_8_cold_start_autoresearch" / "phase8d_verify_candidate"
    )
    summary = verify_candidate_package(
        candidate_dir=candidate_dir,
        config_path=config_path,
        run_dir=run_dir,
        require_server_enforcement=args.require_server_enforcement,
        python_executable=args.python_executable,
    )
    summary_file = Path(args.summary_file).resolve() if args.summary_file else run_dir / "phase8d_verifier_summary.json"
    write_summary(summary_file, summary)
    print(f"summary_file={summary_file}")
    print(f"status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

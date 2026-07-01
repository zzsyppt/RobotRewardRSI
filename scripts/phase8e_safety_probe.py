#!/usr/bin/env python3
"""Run the Phase 8E-1 safety-model probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.safety_model import SafetyModelError, run_safety_model_probe  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_safety_model.yaml")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--hidden-probe-file", default=None)
    parser.add_argument("--require-server-enforcement", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else PROJECT_ROOT / "runs" / "phase_8_cold_start_autoresearch" / "phase8e_safety_probe"
    )
    try:
        exit_code, _summary = run_safety_model_probe(
            config_path=config_path,
            candidate_dir=Path(args.candidate_dir).resolve(),
            checkpoint_path=Path(args.checkpoint).resolve(),
            run_dir=run_dir,
            hidden_probe_file=Path(args.hidden_probe_file).resolve() if args.hidden_probe_file else None,
            require_server_enforcement=args.require_server_enforcement,
        )
    except SafetyModelError as exc:
        print(f"phase8e_safety_error={exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

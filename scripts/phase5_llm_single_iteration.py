#!/usr/bin/env python3
"""Generate one Phase 5 LLM-agent config proposal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase5_llm_agent.runner import run_phase5_iteration


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--phase1-config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--phase2-config", default="configs/phase2_baseline.yaml")
    parser.add_argument("--phase5-config", default="configs/phase5_llm_agent.yaml")
    parser.add_argument("--verifier-config", default="configs/phase3_verifier.yaml")
    parser.add_argument("--provider", default=None, help="Provider profile from configs/phase5_llm_agent.yaml")
    parser.add_argument("--replay-response", default=None)
    parser.add_argument("--output-dir", default="agent_outputs/phase5_llm_single_iteration/latest")
    parser.add_argument("--allow-api-spend", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_phase5_iteration(
        repo_root=REPO_ROOT,
        summary_file=Path(args.summary_file),
        phase1_config_file=Path(args.phase1_config),
        phase2_config_file=Path(args.phase2_config),
        phase5_config_file=Path(args.phase5_config),
        verifier_config_file=Path(args.verifier_config),
        output_dir=Path(args.output_dir),
        provider_name=args.provider,
        replay_response=Path(args.replay_response) if args.replay_response else None,
        allow_api_spend=args.allow_api_spend,
        skip_verify=args.skip_verify,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

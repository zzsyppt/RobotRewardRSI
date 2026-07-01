#!/usr/bin/env python3
"""Build a Phase 6 public-only accept/reject decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - CLI YAML fallback.
from humanoid_agent.phase6_loop.decision import (
    build_phase6_public_decision,
    public_feedback_from_decision,
    write_json,
)
from humanoid_agent.phase6_loop.feedback import assert_public_only_payload
from humanoid_agent.phase6_loop.summaries import load_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--operation-diff", default=None)
    parser.add_argument("--phase6-config", default="configs/phase6_loop.yaml")
    parser.add_argument("--output-file", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    phase6_config_path = Path(args.phase6_config)
    if not phase6_config_path.is_absolute():
        phase6_config_path = REPO_ROOT / phase6_config_path
    best_summary_path = Path(args.best_summary)
    candidate_summary_path = Path(args.candidate_summary)
    operation_diff = load_json(Path(args.operation_diff)) if args.operation_diff else None
    phase6_cfg = _load_structured_file(phase6_config_path)
    decision = build_phase6_public_decision(
        best_summary=load_json(best_summary_path),
        candidate_summary=load_json(candidate_summary_path),
        best_summary_path=best_summary_path,
        candidate_summary_path=candidate_summary_path,
        operation_diff=operation_diff,
        seed_policy=phase6_cfg.get("seed_policy", {}),
    )
    assert_public_only_payload(public_feedback_from_decision(decision))
    write_json(Path(args.output_file), decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect a Phase 6 public-only candidate summary from per-seed artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase6_loop.summaries import collect_public_candidate_summary, write_json


def _parse_seeds(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase1-config", required=True)
    parser.add_argument("--phase2-config", required=True)
    parser.add_argument("--seeds", required=True, help="Comma- or space-separated training seeds.")
    parser.add_argument("--output-file", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    summary = collect_public_candidate_summary(
        run_dir=Path(args.run_dir),
        phase1_config=Path(args.phase1_config),
        phase2_config=Path(args.phase2_config),
        seeds=seeds,
    )
    output_file = Path(args.output_file) if args.output_file else Path(args.run_dir) / "summary.json"
    write_json(output_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

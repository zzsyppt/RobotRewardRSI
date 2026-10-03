#!/usr/bin/env python3
"""Offline CLI for the production trajectory artifact functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.tasks.stair_climbing.trajectory_artifacts import build_evidence
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode", action="append", help="optional run:env; validates the complete source before selecting")
    args = parser.parse_args()
    selection = {tuple(map(int, v.split(":"))) for v in args.episode} if args.episode else None
    if selection is not None and any(len(pair) != 2 or pair[0] < 1 or pair[1] < 0 for pair in selection):
        parser.error("episode must be a one-based run:zero-based env")
    result = build_evidence({"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
        "manifest_path": str(args.manifest), "manifest_sha256": args.manifest_sha256}, args.output, args.source_id, selection)
    print(json.dumps({k: result[k] for k in ("complete", "scanned_episodes", "emitted_episodes", "emitted_valid_rows", "record_counts", "evidence")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline CLI for the production trajectory artifact functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.tasks.stair_climbing.trajectory_artifacts import build_card_artifacts as build, peak_memory_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="full-cohort P1a manifest")
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--example-input", nargs=2, action="append", default=[], metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input, args.input_sha256, [(Path(p), h) for p, h in args.example_input],
                   args.selection, args.selection_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("complete", "primary_episodes", "primary_valid_rows", "claim_count", "selected_example_count", "cost", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()

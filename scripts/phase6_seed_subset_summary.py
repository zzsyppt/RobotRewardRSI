#!/usr/bin/env python3
"""Project an existing Phase 2/6 summary to a public-only paired seed subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase6_loop.summaries import load_json, subset_public_summary, write_json


def _parse_seeds(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--seeds", required=True, help="Comma- or space-separated training seeds.")
    parser.add_argument("--output-file", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source = Path(args.summary_file)
    summary = subset_public_summary(load_json(source), seeds=_parse_seeds(args.seeds), summary_path=source)
    write_json(Path(args.output_file), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

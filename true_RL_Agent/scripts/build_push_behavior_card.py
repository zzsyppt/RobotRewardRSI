#!/usr/bin/env python3
"""Build an offline Push root card from an explicit public source lock, on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.push_recovery.trajectory_card import build_push_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    group = next(g for g in lock["groups"] if g["source_id"] == args.group)
    sources = [{"reference": {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
        "manifest_path": str((ROOT / source["manifest_path"]).resolve()),
        "manifest_sha256": source["manifest_sha256"]}, "identity": source["identity"]} for source in group["sources"]]
    result = build_push_artifacts(sources, source_id=args.group, output=args.output)
    print(json.dumps({k: result[k] for k in ("source_id", "episodes", "wall_seconds", "model_calls", "gpu_runs")}, indent=2))


if __name__ == "__main__":
    main()

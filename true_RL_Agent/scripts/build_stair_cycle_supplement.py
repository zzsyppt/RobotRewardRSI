"""Build only public-dev offline v3 M06 evidence, with optional bounded preview."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import build_cycle_supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-public", type=Path)
    args = parser.parse_args()
    reference = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
                 "manifest_path": str(args.manifest.resolve()), "manifest_sha256": args.manifest_sha256}
    result = build_cycle_supplement(reference, args.output, source_id=args.source_id,
        base_public=json.loads(args.base_public.read_text(encoding="utf-8")) if args.base_public else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

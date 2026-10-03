"""Build offline nominal USD M09 evidence from qualified public-dev archives."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.stair_climbing.trajectory_joint_limits import build_joint_limit_supplement


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--source-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-public", type=Path)
    a = p.parse_args()
    ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(a.manifest.resolve()),
           "manifest_sha256": a.manifest_sha256}
    result = build_joint_limit_supplement(ref, a.output, source_id=a.source_id,
        base_public=json.loads(a.base_public.read_text(encoding="utf-8")) if a.base_public else None)
    print(json.dumps(result, indent=2))

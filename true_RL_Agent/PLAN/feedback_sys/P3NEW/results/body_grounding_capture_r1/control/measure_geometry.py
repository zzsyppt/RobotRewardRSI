"""CPU-only production geometry measurements for the original run-1 capture."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "src"))
    from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
    from true_rl_agent.tasks.push_recovery.body_geometry import bind_push_body_anchors, push_body_geometry_s0

    inputs = {}
    def read(path):
        payload = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)
    sidecar = read(args.capture / "video/recordings.json")
    row, = [r for r in sidecar["recordings"] if r["run_index"] == 1 and r["env_index"] == 19]
    snapshot_path = args.capture / "video" / row["file"]
    snapshot = read(snapshot_path)
    if snapshot != row["manifest"] or inputs[str(snapshot_path)] != row["sha256"]:
        raise ValueError("snapshot differs from original recordings")
    cell = ResolvedCellManifestV1.from_mapping(read(args.capture / "composition/resolved_cell_manifest.json"))
    summary = read(args.capture / "evaluation/public_dev/summary.json")
    if cell.manifest_hash != summary["resolved_cell_manifest_hash"] or cell.manifest_hash != sidecar["resolved_cell_manifest_hash"]:
        raise ValueError("original run composition proof differs")
    profile_id = cell.robot_profile["robot_profile_id"]
    profile_path = args.source / "configs/robots" / (profile_id + ".json")
    binding = bind_push_body_anchors(read(profile_path.with_suffix(".body_anchors.json")),
        runtime_projection={"robot_profile_id": profile_id, "robot_profile_hash": cell.robot_profile["profile_hash"],
                            "body_names": snapshot["articulation_body_names"]}, robot_profile=read(profile_path))
    window = SimpleNamespace(identity={"robot_id": profile_id, "env_index": 19, "runner_seed": row["seed"]},
        images=[SimpleNamespace(displayed_steps=(sample["control_step"],), view_id=view["view_id"])
                for sample in snapshot["samples"] for view in sample["views"]])
    measured = push_body_geometry_s0(window, snapshot, binding=binding)
    source_path = args.source / "src/true_rl_agent/tasks/push_recovery/body_geometry.py"
    result = {"schema_version": "p3new.original_run_body_geometry.v1", "scope": "all selected run1/env19 native samples",
        "projection_or_visibility_claim": False, "resolved_cell_manifest_hash": cell.manifest_hash,
        "production_function": "push_body_geometry_s0", "production_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs, "body_anchor_binding": binding, "measurements": measured}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "angles": [{"name": r["name"], "step": r["step"], "value": r["value"]} for r in measured]}))


if __name__ == "__main__":
    main()

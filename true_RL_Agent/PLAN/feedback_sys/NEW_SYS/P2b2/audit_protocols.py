"""K4-only CPU delivery/protocol and freeze checks, no simulator or holdout access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tests.unit.test_push_trajectory_pipeline import seed_from_manifest
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA, NumericEvidenceError
from true_rl_agent.tasks.push_recovery.trajectory_pipeline import build_phase_trajectory, replay_full_report


if __name__ == "__main__":
    output = ROOT / "outputs/p2b2/final/protocols"
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = ROOT / "PLAN/feedback_sys/NEW_SYS/P2b1/RULES_CODE_FREEZE.json"
    assert file_sha256(freeze_path) == "ec8158315d79fba1ff2f9122016de5f1eb5af5f7a0de96cc6f09dc5cda8ff5e8"
    freeze = json.loads(freeze_path.read_bytes())
    for name, expected in freeze["files_sha256"].items():
        assert file_sha256(ROOT / name) == expected, name
    lock = json.loads((ROOT / "PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json").read_bytes())
    results, groups = [], []
    for group in lock["groups"]:
        assert group["source_id"] in {"k4_force_v1", "k4_velocity_v2"}
        seeds = []
        for item in group["sources"]:
            path = ROOT / item["manifest_path"]
            assert file_sha256(path) == item["manifest_sha256"]
            manifest = json.loads(path.read_bytes())
            for chunk in manifest["chunks"]:
                assert file_sha256(path.parent / chunk["file"]) == chunk["sha256"]
            ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(path), "manifest_sha256": item["manifest_sha256"]}
            seeds.append(seed_from_manifest(manifest, ref))
        identity = {k: v for k, v in group["sources"][0]["identity"].items()
                    if k not in {"run_index", "split", "task_numeric_schema", "mechanism", "resolved_protocol_hash"}}
        context = {"cell_variant_id": "push.go2_12dof.recovery_v2", "resolved_cell_manifest_hash": identity["resolved_cell_manifest_hash"],
                   "resolved_protocol_hash": group["sources"][0]["identity"]["resolved_protocol_hash"]}
        full, public, receipt = build_phase_trajectory(seeds, expected_identity=identity, context=context,
            expected_seed_count=3, output_dir=output / group["source_id"])
        assert receipt["episodes"] == 192 and len(receipt["required_claim_ids"]) == 20
        assert replay_full_report(full, expected_sources=full["sources"], expected_identity=identity) == public
        for name, value in (("full", full), ("public", public), ("receipt", receipt)):
            (output / group["source_id"] / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append({"source_id": group["source_id"], "card_id": public["card_id"], "episodes": receipt["episodes"],
            "bytes": report_bytes(public), "projection": public["report_projection"], "omitted": receipt["omitted_claims"],
            "build_wall_seconds": receipt["build_wall_seconds"], "schema_capability": public["domain"]["root_signed_xy"],
            "mechanism": public["domain"]["mechanism"], "claims": public["report"]})
        groups.append((seeds, identity, context))
    mixed = [groups[0][0][0], *groups[1][0][1:]]
    try:
        build_phase_trajectory(mixed, expected_identity=groups[0][1], context=groups[0][2],
            expected_seed_count=3, output_dir=output / "mixed")
    except (ValueError, NumericEvidenceError) as error:
        rejection = str(error)
    else:
        raise AssertionError("mixed force/velocity group accepted")
    (output / "verification.json").write_text(json.dumps({"status": "PASS", "freeze_sha256": file_sha256(freeze_path),
        "frozen_files_checked": len(freeze["files_sha256"]), "groups": results, "mixed_protocol_rejection": rejection,
        "scope": "original K4 source-lock through CPU group pipeline; no new-policy/normal-generation velocity claim",
        "model_calls": 0, "gpu_runs": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "groups": [{k: r[k] for k in ("source_id", "episodes", "bytes", "schema_capability")} for r in results]}))

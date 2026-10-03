"""Build each frozen real source through the normal CPU artifact/projection path."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import build_trial_trajectory_card, validate_build_receipt, TRAJECTORY_PROFILE
from true_rl_agent.tasks.stair_climbing.trajectory_report import build_stair_trajectory_report
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["k1", "k2", "k3"], required=True)
    args = parser.parse_args()
    lock_path = Path(__file__).with_name("source_lock.json")
    source = next(s for s in json.loads(lock_path.read_text(encoding="utf-8"))["sources"] if s["source_id"] == args.source)
    output = ROOT / "outputs/p2a" / args.source
    ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
           "manifest_path": str(ROOT / source["manifest_relative_path"]), "manifest_sha256": source["manifest_sha256"]}
    started = time.perf_counter()
    card_ref, receipt = build_trial_trajectory_card(ref, expected_identity=source["identity"], output_dir=output)
    full = build_stair_trajectory_report(card_ref, numeric_reference=ref, expected_identity=source["identity"])
    public = project_stair_behavior(full)
    assert public == full["public_report"]
    receipt["projection"] = {"profile": TRAJECTORY_PROFILE, "max_report_bytes": 8192,
        "card_id": public["card_id"], "public_content_hash": public["report_projection"]["content_hash"],
        "required_claim_ids": full["audit"]["required_claim_ids"], "report_bytes": full["audit"]["report_bytes"]}
    validate_build_receipt(receipt, public)
    write(output / "behavior_full.json", full)
    write(output / "public.json", public)
    write(output / "receipt.json", receipt)
    report = {"source_id": args.source, "source_lock_sha256": file_sha256(lock_path), "card_reference": card_ref,
        "numeric_manifest": str(Path(card_ref["manifest_path"]).parent.parent / "numeric/manifest.json"),
        "wall_seconds": time.perf_counter() - started, "public_bytes": full["audit"]["report_bytes"],
        "mandatory_bytes": full["audit"]["required_report_bytes"], "mandatory_count": len(full["audit"]["required_claim_ids"]),
        "public_sha256": file_sha256(output / "public.json"), "full_sha256": file_sha256(output / "behavior_full.json"),
        "stored_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "external_model_calls": 0, "gpu_runs": 0, "training_runs": 0}
    write(output / "replay.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

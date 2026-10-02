"""Read-only old-development v4 capture audit; no GPU/model requests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "baseline", "candidate", "card", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source / "src"))
    spec = importlib.util.spec_from_file_location("exact_capture_audit", args.source / "PLAN/feedback_sys/P3NEW/verify_multi_env_capture.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    from true_rl_agent.evaluation.body_projection import project_snapshot_body_links
    from true_rl_agent.tasks.push_recovery.visual_observation import align_push_snapshot_samples
    from PIL import Image

    result = {"schema_version": "p3new.body_grounding_capture_audit.v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": {}, "runs": [],
              "scope": "old_development_sparse_engineering_not_N5", "provider_requests": 0,
              "projection_pixel_accuracy": "pending_independent_image_review",
              "limits": ["All first-episode valid numeric bytes and terminal rows are compared; padding is excluded.",
                         "Projection and numeric alignment use production code, not an independent physical oracle.",
                         "Projected points do not assert visibility. Original PNG pixels are not modified."]}
    code = 0
    try:
        evidence = result["input_sha256"]
        card = audit.read(args.card, evidence)
        left, _ = audit.read_runs(args.baseline, "push", evidence)
        right, sidecar = audit.read_runs(args.candidate, "push", evidence)
        audit.require(len(sidecar["recordings"]) == 3, "expected one env across exactly three seeds")
        seen_images = set()
        for run in (1, 2, 3):
            a, b = left[run], right[run]
            audit.require(a["seed"] == b["seed"], "runner seed changed")
            audit.require(all(a["manifest"]["identity"].get(k) == b["manifest"]["identity"].get(k)
                              for k in audit.IDENTITY), "numeric identity changed")
            la, ls, _ = audit.load_arrays(a, run, "push", evidence)
            ra, rs, _ = audit.load_arrays(b, run, "push", evidence)
            comparison = audit.compare_run(a, b, la, ra, ls, rs)
            row = {"run_index": run, "runner_seed": b["seed"], "numeric_comparison": comparison}
            result["runs"].append(row)
            audit.require(len(b["records"]) == 1 and b["records"][0]["env_index"] == 19, "unexpected recorded env")
            recording = b["records"][0]
            path = args.candidate / "video" / recording["file"]
            snapshot = audit.read(path, evidence)
            audit.require(evidence[str(path.resolve())] == recording["sha256"] and snapshot == recording["manifest"],
                          "snapshot/recordings binding differs")
            terminal = b["outcomes"][19]
            t = terminal["episode_steps"]
            audit.require(recording["episode_outcome"] == terminal and snapshot["episode_control_step_count"] == t,
                          "snapshot terminal binding differs")
            audit.require(snapshot["schema_version"] == "native_rgb_snapshots.v4" and snapshot["env_index"] == 19
                          and snapshot["seed"] == b["seed"], "wrong snapshot identity/version")
            steps = card["capture_profile"]["snapshot_steps"]
            actual = [s["control_step"] for s in snapshot["samples"]]
            audit.require(snapshot["selected_control_steps"] == steps and actual == [s for s in steps if s <= t],
                          "captured steps differ from registered sparse selection")
            audit.require(recording["includes_terminal_frame"] is (t in actual), "terminal-frame flag differs")
            valid = ra["valid"][:, 19]
            row["numeric_overlap"] = align_push_snapshot_samples(snapshot, control_step=ra["control_step"][valid, 19],
                pose=ra["pose"][valid, 19], planar_motion=ra["planar_motion"][valid, 19])
            row.update(env_index=19, T=t, actual_steps=actual, snapshot_manifest=str(path), images=[])
            for sample in snapshot["samples"]:
                state = sample["state_before"]
                audit.require(state == sample["state_after"], "render changed captured state/clock")
                audit.require([v["view_id"] for v in sample["views"]] == ["v0", "v1"], "view inventory differs")
                audit.require(sample["views"][0]["camera_params"]["cameraViewTransform"] !=
                              sample["views"][1]["camera_params"]["cameraViewTransform"], "view transforms are identical")
                for view in sample["views"]:
                    image_path = path.parent / view["file"]
                    audit.require(image_path not in seen_images and audit.sha(image_path) == view["sha256"], "PNG identity/hash differs")
                    seen_images.add(image_path)
                    with Image.open(image_path) as image:
                        audit.require(image.size == (1920, 1080) and image.mode == "RGB" and image.format == "PNG", "PNG geometry differs")
                        pixels_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                    points = project_snapshot_body_links(snapshot, control_step=sample["control_step"],
                        view_id=view["view_id"], body_names=card["review_body_names"])
                    row["images"].append({"control_step": sample["control_step"], "view_id": view["view_id"],
                        "file": str(image_path), "sha256": view["sha256"], "decoded_rgb_sha256": pixels_hash,
                        "camera_params": view["camera_params"], "body_image_points": points})
        audit.require(seen_images == set((args.candidate / "video").glob("*.png")), "orphan/missing PNGs")
        result["image_count"] = len(seen_images)
        result["all_valid_numeric_fields_and_terminals_identical"] = all(r["numeric_comparison"]["identical"] for r in result["runs"])
        result["all_media_numeric_overlaps_aligned"] = all(r["numeric_overlap"]["status"] == "overlap_aligned" for r in result["runs"])
        result["status"] = "pass" if result["all_valid_numeric_fields_and_terminals_identical"] and result["all_media_numeric_overlaps_aligned"] else "difference_observed"
        code = 0 if result["status"] == "pass" else 1
    except Exception as exc:
        result.update(status="incomplete_or_failed", error_type=type(exc).__name__, error=str(exc))
        code = 2
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "output": str(args.output), "error": result.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

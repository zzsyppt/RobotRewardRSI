"""Bind the missing medium replay observation to its measured source (offline)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = HERE / "results/checkpoint_evidence_r1/video_medium19_r2"
    raw = read(HERE / "results/push_numeric_independent.json")
    sidecar = root / "video/recordings.json"
    videos = read(sidecar)
    result = read(root / "result.json")
    checks = []
    for recording in videos["recordings"]:
        source_index = recording["run_index"] - 1
        source = raw["sources"][source_index]["identity"]
        episode = next(e for e in raw["episodes"] if (e["source"], e["env_id"]) == (source_index, 19))
        outcome = recording["episode_outcome"]
        expected = {
            "checkpoint": videos["checkpoint"]["checkpoint_hash"] == source["checkpoint_hash"],
            "manifest": videos["resolved_cell_manifest_hash"] == source["resolved_cell_manifest_hash"],
            "protocol": recording["resolved_protocol_hash"] == source["resolved_protocol_hash"],
            "slot": recording["env_index"] == outcome["env_id"] == 19,
            "stratum": outcome["stratum"] == episode["stratum"] == "medium",
            "steps": outcome["episode_steps"] == episode["valid_steps"],
            "outcome": outcome["failure_label"] == episode["terminal_label"],
            "final_speed": abs(outcome["final_planar_speed_mps"] - episode["terminal"]["planar_speed_mps"]) < 1e-7,
            "recovery_latch": outcome["push_events"][0]["recovery_control_step"] == episode["recoveries"][0]["recovery_control_step"],
        }
        assert all(expected.values()), expected
        checks.append({"run_index": recording["run_index"], "checks": expected,
                       "frame_count": recording["frame_count"], "episode_steps": episode["valid_steps"],
                       "scope": "source/protocol/outcome and selected numeric checkpoints match, not per-step trajectory equivalence"})
    assert result["status"] == "PASS" and result["checkpoint_runner_admission"] == "passed"
    grid = root / "video/medium19_run1_grid.png"
    facts = read(HERE / "r0b_evidence/push-02_Tc-num.json")
    augmented = deepcopy(facts)
    augmented["visual_observations"] = [{
        "episode": "source0.env19", "body": "whole trunk and visible legs",
        "visible_control_steps": [1,68,70,71,95,109,122,179,250,350,400,411],
        "observation": "The trunk remains visibly upright in these sampled frames. After the perturbation, the visible leg configurations and relative foot positions change, including in later frames. No whole-body collapse or rollover is visible in this sampled sequence.",
        "source": {"grid": grid.name, "grid_sha256": digest(grid),
                   "recordings_sha256": digest(sidecar), "video_sha256": videos["recordings"][0]["sha256"]},
        "observer": "model visual inspection of source-bound replay frames, 2026-09-29",
        "modality_delivered_to_author": "text observation, not image pixels",
        "visibility_limits": "Single oblique camera, relatively small robot and sparse frames. Do not infer exact step count, foot contact timing, ankle/hip strategy, force or friction. Frame 412 is not recorded. This does not establish that reward damping caused the observed motion.",
    }]
    (HERE / "r0b_evidence/push-02_Tc-vis.json").write_text(json.dumps(augmented,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report = {"status": "PASS", "checks": checks, "observed_run": 1,
              "media_result_sha256": digest(root / "result.json"), "recordings_sha256": digest(sidecar),
              "gpu_capacity_seconds": result["gpu_capacity_seconds"],
              "failed_prelaunch_gpu_capacity_seconds": read(root.parent / "video_medium19_r1/result.json")["gpu_capacity_seconds"],
              "qualification": "development visual material only; no new training or semantic/value PASS"}
    (HERE / "results/medium_video_checks.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"runs_checked":len(checks),"identity_checks":sum(len(row['checks']) for row in checks),"gpu_seconds":report['gpu_capacity_seconds']}))


if __name__ == "__main__":main()

"""Offline arithmetic and identity checks for the fixed random re-audit sample."""
import ast
import hashlib
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent
REVIEW = P3 / "results/author_final_bundle/author_blind_review/reviewer"
VIDEO = P3 / "review_video_20260924"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


manifest = read(HERE / "sample_manifest.json")
assert sorted(random.Random(manifest["seed"]).sample(manifest["pool"], 6)) == manifest["selected"]
stair = read(REVIEW / "evidence-stair.json")
push = read(REVIEW / "evidence-push.json")
out = {"sample_reproduced": True, "input_hashes": {}, "changed_functions": {}, "video_identity": []}
for name in ["evidence-stair.json", "evidence-push.json", "reference-stair.py", "reference-push.py", "rubric.json", "review_sheet.json"]:
    out["input_hashes"][name] = hashlib.sha256((REVIEW/name).read_bytes()).hexdigest()
for bid in manifest["selected"]:
    for path in [REVIEW / f"{bid}.py", REVIEW / ("seen-" + bid[-3:] + ".json")]:
        out["input_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

for num, task in [("015", "stair"), ("025", "push")]:
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)}
    old = functions(REVIEW / f"reference-{task}.py")
    new = functions(REVIEW / f"proposal-{num}.py")
    out["changed_functions"][num] = [name for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)]

for slot in [64, 156]:
    rec = read(VIDEO / f"slot{slot}_recordings.json")
    video = rec["recordings"][0]
    outcome = video["episode_outcome"]
    episode = next(e for e in stair["episodes"] if e["episode_id"] == f"r1.e{slot}")
    checks = {
        "checkpoint": rec["execution"]["checkpoint_hash"] == stair["domain"]["checkpoint_hash"],
        "cell": outcome["evaluation_cell_id"] == episode["cell"],
        "steps": outcome["episode_steps"] == episode["steps"],
        "outcome": outcome["primary_reason"] == episode["primary_reason"],
    }
    assert all(checks.values()), checks
    out["video_identity"].append({"task": "stair", "episode": episode["episode_id"], "checks": checks,
        "ranges": episode["ranges"], "steps": episode["steps"], "cell": episode["cell"],
        "root_events": [e for e in episode["events"] if e["body"] in ["root", "forbidden_role", "all_support"] and e["end"] > episode["steps"] - 70]})

for slot in [12, 55]:
    rec = read(VIDEO / f"push{slot}_recordings.json")
    video = rec["recordings"][0]
    outcome = video["episode_outcome"]
    candidates = [e for e in push["transcripts"] if e["env_id"] == slot]
    # Source naming remains explicit; first transcript is the first frozen run.
    episode = candidates[0]
    checks = {
        "checkpoint": rec["checkpoint"]["checkpoint_hash"] == push["binding"]["checkpoint_hash"],
        "protocol": video["resolved_protocol_hash"] == push["binding"]["resolved_protocol_hash"],
        "cell_manifest": rec["resolved_cell_manifest_hash"] == push["binding"]["resolved_cell_manifest_hash"],
        "steps": outcome["episode_steps"] == episode["valid_steps"],
        "outcome": outcome["failure_label"] == episode["terminal_label"],
        "final_speed": abs(outcome["final_planar_speed_mps"] - episode["final_pose"]["planar_speed_mps"]) < 1e-6,
    }
    assert all(checks.values()), checks
    out["video_identity"].append({"task": "push", "env_id": slot, "source": episode["source"], "checks": checks,
        "terminal_label": episode["terminal_label"], "final_failed_conditions": episode["final_failed_conditions"],
        "final_pose": episode["final_pose"], "max_abs_pitch_rad": outcome["max_abs_pitch_rad"],
        "max_abs_roll_rad": outcome["max_abs_roll_rad"]})

group = [e for e in stair["episodes"] if e["stratum"] == "c1_conditional_frontier"]
out["stair_frontier"] = {
    reason: {"n": len(rows), "tilt_before_terminal": sum(e["before_terminal"]["tilt"] for e in rows),
             "min_pitch_below_minus_065": sum(e["ranges"]["pitch_rad"][0] < -.65 for e in rows)}
    for reason in ["success", "forbidden_contact"]
    for rows in [[e for e in group if e["primary_reason"] == reason]]
}
nr = [e for e in push["transcripts"] if e["terminal_label"] == "not_recovered"]
out["push_not_recovered"] = {"n": len(nr),
    "max_abs_final_roll": max(abs(e["final_pose"]["roll_rad"]) for e in nr),
    "max_abs_final_pitch": max(abs(e["final_pose"]["pitch_rad"]) for e in nr),
    "recovery_windows_with_pose_violation": sum(any(r["condition_violation_spans"][k] for k in ["roll", "pitch", "height"]) for e in nr for r in e["recoveries"] if r.get("condition_violation_spans"))}

out["015_pitch_penalty"] = [{"pitch_rad": p, "old_weighted_cost": 2.5 * math.sin(p)**2,
    "new_weighted_cost": 2.5 * max(abs(math.sin(p))-.5, 0)} for p in [.3, .5, .65, 1.0, 1.57]]
def huber(x):
    return .5*x*x if x <= 1 else x-.5
out["025_upright_planar_penalty"] = [{"scaled_speed": v, "old_cost": .25*huber(v/.08),
    "new_cost": .25*huber(v/.25)} for v in [.04, .08, .16, .25]]
for name in ["g156_r1_zoom.png", "g64_r1_zoom.png", "gp55_r1.png", "gp12_r1_end.png"]:
    out["input_hashes"][name] = hashlib.sha256((VIDEO/name).read_bytes()).hexdigest()
(HERE / "sample_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))

"""Compare H native replays with their frozen numeric runs and draw frame sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
# Run with the bundled Python for NumPy/Pillow; retain the project's encoder.
sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
import imageio_ffmpeg

DEFAULT_ROOT = ROOT / "PLAN/2new_world/P6/short_force_01"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
SCORE_FIELDS = ("env_id", "stratum", "episode_steps", "planned_push_count", "success",
                "failure_label", "sequence_recovered", "sequence_recovery_control_step",
                "invalid_injection_count")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sheet(case, steps, pictures, facts, title, outcome, path):
    width, height, columns = 360, 372, 3
    canvas = Image.new("RGB", (width*columns, height*math.ceil(len(steps)/columns)+60), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=FONT, fill="white")
    draw.text((12, 32), f"{case['case_id']} | {case['selection_reason']} | outcome: {outcome}",
              font=SMALL, fill="#d5dde5")
    for index, step in enumerate(steps):
        x, y = (index % columns)*width, (index // columns)*height+60
        # Same fixed native-frame center crop as G; original MP4 is preserved.
        canvas.paste(pictures[step].crop((280, 90, 1000, 630)).resize((width, 270)), (x, y))
        f = facts[step-1]
        lines = [f"step {step} / {f['time_s']:.2f}s | force {f['force_n']:.1f} N",
                 f"tilt {f['tilt_deg']:.1f} deg | height {f['height_m']:.3f} m",
                 f"speed {f['speed_mps']:.3f} m/s | omega {f['angular_radps']:.3f}",
                 f"stable {f['stable']}"]
        for j, line in enumerate(lines):
            draw.text((x+8, y+275+22*j), line, font=FONT,
                      fill="#ffa552" if j == 0 and f["force_n"] > 0 else "white")
    canvas.save(path, quality=92)


def review_case(root, case):
    folder, source = root / "videos" / case["case_id"], root / "short"
    index, env, seed = case["seed_index"], case["env_index"], case["seed"]
    meta = read(folder / "video.json")
    original_path, replay_path = source / f"seed_{index:02d}.json", folder / f"seed_{index:02d}.json"
    original, replay = read(original_path), read(replay_path)
    original_raw, replay_raw = source / f"seed_{index:02d}.npz", folder / f"seed_{index:02d}.npz"
    assert sha(original_raw) == original["raw_sha256"] and sha(replay_raw) == replay["raw_sha256"]
    assert (meta["seed_index"], meta["seed"], meta["env_index"]) == (index, seed, env)
    assert original["seed"] == replay["seed"] == seed
    assert len(original["episodes"]) == len(replay["episodes"]) == 64
    prefix_seeds = []
    for previous in range(1, index+1):
        prefix = read(folder / f"seed_{previous:02d}.json")
        assert (folder / f"seed_{previous:02d}.npz").is_file()
        assert len(prefix["episodes"]) == 64
        prefix_seeds.append(prefix["seed"])
    assert prefix_seeds == [read(source / f"seed_{previous:02d}.json")["seed"]
                            for previous in range(1, index+1)]
    assert [row["env_id"] for row in original["episodes"]] == list(range(64))
    assert [row["env_id"] for row in replay["episodes"]] == list(range(64))
    assert case["expected_episode"] == original["episodes"][env]
    identity, original_identity = read(folder / "source_identity.json"), read(source / "source_identity.json")
    for field in ("checkpoint_sha256", "source_manifest_hash", "source_training_binding_hash", "source_environment_hash"):
        assert identity[field] == original_identity[field], field
    p, definition = read(folder / "parameters.json"), read(folder / "definition.json")
    assert p == read(source / "parameters.json") and definition == read(source / "definition.json")
    assert original["definition_hash"] == replay["definition_hash"] == definition["definition_hash"]
    a, b = arrays(original_raw), arrays(replay_raw)
    assert set(a) == set(b)
    assert all(np.isfinite(values).all() for values in a.values())
    assert all(np.isfinite(values).all() for values in b.values())
    # Compare masks before using either mask to select first-episode samples.
    for key in ("valid", "steps", "physics_valid"):
        assert np.array_equal(a[key], b[key]), key
    dt = definition["control_dt_s"]
    decimation = round(dt / definition["physics_dt_s"])
    assert abs(dt-.02) < 1e-12
    assert np.array_equal(b["physics_valid"], np.repeat(b["valid"], decimation, axis=0))
    assert np.array_equal((b["physics_control_step_before"]+1)[b["physics_valid"]],
                          np.repeat(b["steps"], decimation, axis=0)[b["physics_valid"]])
    outcomes = [all(x[k] == y[k] for k in SCORE_FIELDS)
                for x, y in zip(original["episodes"], replay["episodes"], strict=True)]
    full_rows = [x == y for x, y in zip(original["episodes"], replay["episodes"], strict=True)]
    differences = {}
    for key in a:
        x, y = a[key], b[key]
        assert x.shape == y.shape, key
        mask = b["physics_valid"] if key.startswith("physics_") else b["valid"]
        if x.shape[:2] == mask.shape:
            x, y = x[mask], y[mask]
        differences[key] = {"exact": bool(np.array_equal(x, y)),
            "max_abs_difference": float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.0}
    initial_equal = original["initial_xy"] == replay["initial_xy"] and original["origins"] == replay["origins"]
    for env_id, row in enumerate(replay["episodes"]):
        valid_steps = b["steps"][b["valid"][:, env_id], env_id]
        assert np.array_equal(valid_steps, np.arange(1, row["episode_steps"]+1))
    selected = replay["episodes"][env]
    n = selected["episode_steps"]
    assert selected == meta["selected_episode"] and n == meta["frame_count"] > 0
    assert meta["first_control_step"] == 1 and meta["last_control_step"] == n
    assert meta["capture_point"] == "after_physics_before_termination_and_reset"
    assert abs(meta["fps"]-1/dt) < 1e-6
    q = b["quaternion_xyzw"][:n, env].astype(float)
    x, y, z, w = q.T
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    height = b["position"][:n, env, 2] - replay["origins"][env][2]
    speed = np.linalg.norm(b["linear"][:n, env, :2], axis=1)
    angular = np.linalg.norm(b["angular"][:n, env], axis=1)
    stable = ((tilt <= p["recovery_tilt_rad"]) & (height >= p["recovery_height_m"])
              & (speed <= p["recovery_velocity_mps"]) & (angular <= p["recovery_angular_velocity_radps"]))
    force = np.linalg.norm(b["physics_requested_force_w"].reshape(-1, decimation, 64, 3)[:n, 0, env], axis=1)
    facts = [{"step": i+1, "time_s": (i+1)*dt, "force_n": float(force[i]),
              "tilt_deg": float(np.degrees(tilt[i])), "height_m": float(height[i]),
              "speed_mps": float(speed[i]), "angular_radps": float(angular[i]), "stable": bool(stable[i])}
             for i in range(n)]
    keys = {1, n, int(tilt.argmax())+1, int(height.argmin())+1, int(speed.argmax())+1}
    for event in selected["force_events"]:
        keys.update((max(1, event["start_control_step"]), min(n, event["end_control_step"])))
    recovery = selected["sequence_recovery_control_step"]
    if recovery is not None:
        keys.add(recovery)
    assert all(1 <= step <= n for step in keys)
    keys = sorted(keys)
    timeline = sorted(set(range(1, n+1, 8)) | {n})
    wanted, pictures = set(keys) | set(timeline), {}
    video = folder / "video" / Path(meta["file"]).name
    assert sha(video) == meta["sha256"]
    stream = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    header = next(stream)
    assert tuple(header["size"]) == (1280, 720) and abs(header["fps"]-50) < 1e-6
    count = 0
    for count, frame in enumerate(stream, 1):
        assert len(frame) == 1280*720*3
        if count in wanted:
            pictures[count] = Image.frombytes("RGB", header["size"], frame)
    assert count == n and set(pictures) == wanted
    exact = all(v["exact"] for v in differences.values())
    passed = all(outcomes) and all(full_rows) and exact and initial_equal
    out = root / "visual_review" / case["case_id"]
    out.mkdir(parents=True, exist_ok=True)
    title = "Native simulation - pulses / worst posture / sequence recovery / terminal"
    if not passed:
        title = "Numeric replay DIFFERENCES - inspect before interpreting this video"
    sheet(case, keys, pictures, facts, title, selected["failure_label"], out / "keyframes.jpg")
    for offset in range(0, len(timeline), 12):
        sheet(case, timeline[offset:offset+12], pictures, facts,
              "Native simulation - chronological frames every 0.16 seconds",
              selected["failure_label"], out / f"timeline_{offset//12+1:02d}.jpg")
    report = {"case_id": case["case_id"], "status": "PASS" if passed else "DIFFERENCES",
        "all_64_outcomes_equal": all(outcomes), "equal_outcome_count": sum(outcomes),
        "all_64_episode_records_equal": all(full_rows), "initial_xy_and_origins_exact": initial_equal,
        "replay_seeds_including_preceding": prefix_seeds,
        "replayed_episodes_including_preceding": 64*len(prefix_seeds),
        "all_valid_trajectory_arrays_exact": exact, "differences": differences,
        "decoded_frames": count, "metadata": header, "selected_episode": selected,
        "keyframe_steps": keys, "frame_facts": facts, "native_video_sha256": sha(video),
        "source_sha256": {str(path): sha(path) for path in
            (original_path, replay_path, original_raw, replay_raw, folder / "video.json")},
        "scope": "All 64 first episodes compared; decoded frame 1 corresponds to pre-reset control sample 1."}
    write(out / "comparison.json", report)
    print(case["case_id"], sum(outcomes), "exact", exact, "frames", count, flush=True)
    return {k: v for k, v in report.items() if k not in
            ("differences", "frame_facts", "selected_episode", "metadata", "source_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    selection_path = args.evidence_dir / "records/video_selection.json"
    cases = read(selection_path)["cases"]
    assert 1 <= len(cases) <= 2 and len({case["case_id"] for case in cases}) == len(cases)
    reports = []
    for case in cases:
        try:
            reports.append(review_case(args.evidence_dir, case))
        except FileNotFoundError as error:
            reports.append({"case_id": case["case_id"], "status": "INCOMPLETE", "error": str(error)})
        except Exception as error:
            reports.append({"case_id": case["case_id"], "status": "FAIL",
                            "error": f"{type(error).__name__}: {error}"})
    statuses = {report["status"] for report in reports}
    status = ("FAIL" if "FAIL" in statuses else "DIFFERENCES" if "DIFFERENCES" in statuses
              else "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS")
    result = {"status": status, "expected_case_count": len(cases), "cases": reports,
        "complete_case_count": sum("decoded_frames" in row for row in reports),
        "compared_episodes_not_new_samples": 64*sum("decoded_frames" in row for row in reports),
        "replayed_episodes_not_new_samples": sum(row.get("replayed_episodes_including_preceding", 0) for row in reports),
        "total_decoded_frames": sum(row.get("decoded_frames", 0) for row in reports),
        "selection_sha256": sha(selection_path), "review_script_sha256": sha(Path(__file__)),
        "limitations": ["Native frame sheets are sparse views; full MP4 is decoded and retained.",
                        "Matching trajectories do not establish independent live state.closed capture or reference qualification."]}
    write(args.evidence_dir / "video_comparison.json", result)
    print(json.dumps({"status": status, "cases": len(reports),
                      "complete": result["complete_case_count"], "frames": result["total_decoded_frames"]}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

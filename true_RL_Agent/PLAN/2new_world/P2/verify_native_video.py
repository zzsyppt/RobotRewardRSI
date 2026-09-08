"""Decode P2 videos and verify timing against the task-owned episode evidence."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("job_dir", type=Path)
args = parser.parse_args()
video_dir = args.job_dir / "video"
sidecar = json.loads((video_dir / "recordings.json").read_text())
summary = json.loads((args.job_dir / "evaluation/public_dev/summary.json").read_text())
assert sidecar["evaluation_summaries"]["public_dev"] == summary["task_evidence"]["summary"]
assert sidecar["execution"] == summary["task_evidence"]["execution_identity"]
assert sidecar["checkpoint_loading"] == summary["checkpoint_loading"]
assert not sidecar["reclassified_by_video_exporter"]
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
records = []
for item in sidecar["recordings"]:
    path = video_dir / item["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    expected_frames = item["episode_outcome"]["episode_steps"] - 1
    assert item["start_control_step"] == 1 and not item["starts_at_reset"]
    assert item["frame_count"] == expected_frames > 0
    decoded = subprocess.run(
        [ffmpeg, "-hide_banner", "-threads", "1", "-i", str(path),
         "-progress", "pipe:1", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    frames = int(re.findall(r"^frame=(\d+)$", decoded.stdout, re.M)[-1])
    duration_s = int(re.findall(r"^out_time_us=(\d+)$", decoded.stdout, re.M)[-1]) / 1e6
    fps = float(re.search(r"(\d+(?:\.\d+)?) fps", decoded.stderr).group(1))
    assert item["encoder"] == "imageio_ffmpeg_raw_rgb"
    assert frames == expected_frames
    assert abs(fps - 1 / item["control_dt_s"]) < 1e-6
    assert abs(duration_s - item["simulated_duration_s"]) <= item["control_dt_s"]
    images = []
    for name, index in (("first", 0), ("middle", frames // 2), ("last", frames - 1)):
        image = path.with_name(path.stem + "_" + name + ".png")
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
             "-i", str(path), "-vf", f"select=eq(n\\,{index})",
             "-frames:v", "1", "-update", "1", "-y", str(image)], check=True,
        )
        images.append(image.name)
    records.append({"file": path.name, "sha256": item["sha256"], "frames": frames,
                    "native_frames": expected_frames, "frame_delta": frames - expected_frames,
                    "fps": fps, "duration_s": duration_s, "images": images})
report = {"status": "PASS", "job_dir": str(args.job_dir),
          "sidecar_equals_task_evidence": True,
          "checkpoint_loading": summary["checkpoint_loading"], "recordings": records}
(video_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))

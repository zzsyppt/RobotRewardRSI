"""Offline worst-duration development capacity; no calls, scoring or N5 data."""
from pathlib import Path
import json
import sys
import hashlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
ROOT = PHASE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PHASE))
import export_confirmation_raw as raw
import prepare_confirmation_media as media
import prepare_confirmation_materials as packing


def main():
    full_path = PHASE / "results/stair_progress_process_development_r1/full.json"
    full = json.loads(full_path.read_bytes())
    row = sorted(full["episodes"], key=lambda r: (-r["steps"], r["episode_id"]))[0]
    run, env = map(int, row["episode_id"].replace("r", "").split(".e"))
    manifest_path = ROOT / "PLAN/feedback_sys/P3R/results/stair_numeric_dev_r1/numeric_evidence/manifest.json"
    manifest = raw.read_manifest(manifest_path)
    original, = [r for r in raw.source_rows("stair", manifest, "capacity") if (r["run"], r["env"]) == (run, env)]
    episode = {"numeric_reference": {"manifest_path": str(manifest_path), "manifest_sha256": raw.sha(manifest_path)},
        "run_index": run, "env_index": env, "episode_outcome": original["terminal"]}
    target = HERE / "raw"
    if target.exists():
        raise ValueError("preserve the old capacity export")
    media.numeric_episode(target, "stair", episode, "longest-development")
    shared = {"domain": full["domain"], "semantics": full["semantics"], "contract": full["semantic"]["contract"],
              "actionability": full["semantic"]["actionability"]}
    value = {"shared": shared, "complete_episode_derivation": packing.review_episode_derivation(row, task="stair", material_profile="related_v1")}
    raw.write(HERE / "full_episode.json", value)
    result = {"status": "DEVELOPMENT_CAPACITY_ONLY", "selection": "maximum observed steps then episode_id across all 864 old episodes",
        "original_episode_id": row["episode_id"], "steps": row["steps"], "events": len(row["events"]),
        "new_api_calls": 0, "new_gpu_runs": 0, "source_manifest_sha256": raw.sha(manifest_path),
        "source_full_sha256": raw.sha(full_path), "source_code_sha256": raw.sha(Path(__file__)),
        "inputs": [{"file": str(p.relative_to(HERE)), "sha256": raw.sha(p), "bytes": p.stat().st_size}
                   for p in (target / "related_numeric.json", HERE / "full_episode.json")]}
    raw.write(HERE / "preparation.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Re-derive one historical archive; no simulator, provider or acceptance run."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from true_rl_agent.evaluation.binding import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.behavior import build_stair_behavior
from true_rl_agent.tasks.stair_climbing.process_report import derive_stair_process_behavior, project_stair_process_report

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "PLAN/feedback_sys/P3NEW/results/delivery_offline_r1/stair"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = read(OLD / "raw_sources.json")
    item = sources[0]["manifest"]
    reference = read(OLD / "public_dev_result.json")["task_evidence"]["numeric_evidence"]
    assert reference["manifest_path"] == item["path"] and reference["manifest_sha256"] == item["sha256"]
    current = build_stair_behavior(reference)
    prior = read(OLD / "measured_v2.json")
    restored = deepcopy(current)
    restored["semantics"].pop("progress_process")
    for row in restored["episodes"]:
        row.pop("progress_process")
    assert restored == prior, "existing measurements or outcomes changed"
    measured_path = OUT / "measured_with_progress.json"
    measured_path.write_bytes(canonical_json_bytes(current))
    full = derive_stair_process_behavior(current)
    (OUT / "full.json").write_bytes(canonical_json_bytes(full))
    public = project_stair_process_report(full)
    (OUT / "public.json").write_bytes(canonical_json_bytes(public))
    receipt = {"status": "DEVELOPMENT_REDERIVATION_AND_PROJECTION_PASS_NOT_N5",
        "source": reference, "old_measured_sha256": sha(OLD / "measured_v2.json"),
        "episodes": len(current["episodes"]), "all_old_measurements_equal": True,
        "excursions": sum(len(row["progress_process"]["excursions"]) for row in current["episodes"]),
        "unreturned_at_terminal": sum(row["recovery_step"] is None for e in current["episodes"]
            for row in e["progress_process"]["excursions"]),
        "outputs": {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size}
            for name in ("measured_with_progress.json", "full.json", "public.json")},
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
            (ROOT / "src/true_rl_agent/tasks/stair_climbing/behavior.py",
             ROOT / "src/true_rl_agent/tasks/stair_climbing/process_report.py")},
        "new_api_calls": 0, "gpu_jobs": 0, "new_confirmation_data": False}
    (OUT / "rebuild_receipt.json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps({k: receipt[k] for k in ("status", "episodes", "excursions", "unreturned_at_terminal", "outputs")}))


if __name__ == "__main__":
    main()

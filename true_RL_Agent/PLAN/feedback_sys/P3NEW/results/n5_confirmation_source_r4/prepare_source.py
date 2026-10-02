"""One R4 source snapshot and unexecuted commands; no capture or model dispatch."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PRIOR = HERE.parent / "n5_confirmation_source_r3"
PHASE = "PLAN/feedback_sys/P3NEW/"
REMOTE_BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
COHORT = "n5_r4_20261001"
PUSH_SEED = 1800000000


def read(path):
    return json.loads(Path(path).read_bytes())


def record(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    previous = read(PRIOR / "source_manifest.final.json")
    assert all(record(PRIOR / "source_final" / name) == item for name, item in previous.items())
    additions = [PHASE + name for name in (
        "N5_RUN_CARD_R4_ZH.md", "N5_REVIEW_ASSIGNMENT_R4.json", "N5_SOL_CLI_POLICY_R4.json",
        "prepare_desktop_review.py", "test_sol_cli_carrier.py")]
    additions.append("tests/unit/test_stair_body_projection.py")
    allowed_changed = {
        "PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md",
        PHASE + "N5_FACT_REVIEW_PROTOCOL_ZH.md", PHASE + "REVIEW_PROTOCOL_ZH.md",
        PHASE + "prepare_native_observer.py", PHASE + "run_confirmation_review.py", PHASE + "run_sol_review.py",
        "configs/robots/unitree_go2_12dof_isaaclab_v1.body_frame.json",
        "src/true_rl_agent/tasks/stair_climbing/visual_observation.py"}
    actual_changed = {name for name, item in previous.items() if record(ROOT / name) != item}
    assert actual_changed == allowed_changed, actual_changed.symmetric_difference(allowed_changed)
    paths = sorted(set(previous) | set(additions))
    assert all((ROOT / name).is_file() for name in paths)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / PHASE)]
    import run_confirmation_review as review
    allocation_path = ROOT / PHASE / "N5_REVIEW_ASSIGNMENT_R4.json"
    allocation = read(allocation_path)
    assert allocation["status"] == "FROZEN"
    policy_path = ROOT / PHASE / "N5_SOL_CLI_POLICY_R4.json"
    policy = read(policy_path)
    assert policy["status"] == "FROZEN" and policy["model"] == "gpt-6.1-sol"
    assert policy["reasoning_effort"] == "high" and policy["model_context_window"] == 512000
    assert policy["model_auto_compact_token_limit"] == 450000 and policy["output_reservation_tokens"] == 32768
    ds = []
    for packet in allocation["packets"]:
        fields = review.assignment_fields(allocation_path, packet)
        reviewers = fields["review_assignment"]["reviewers"]
        assert reviewers["A"] == {"judge_id": "gpt6.1-sol-a", "model": "gpt-6.1-sol", "carrier": "fresh_cli"}
        if reviewers["B"]["judge_id"] != "deepseek":
            assert reviewers["B"] == {"judge_id": "gpt6.1-sol-b", "model": "gpt-6.1-sol", "carrier": "fresh_cli"}
        if reviewers["B"]["judge_id"] == "deepseek":
            ds.append(packet)
    assert len(allocation["packets"]) == 40 and sorted(ds) == [
        "episode-01", "episode-13", "focus-01", "focus-07", "media-01", "media-02", "media-03", "media-04"]
    current = {name: record(ROOT / name) for name in paths}
    destination = HERE / "source"
    destination.mkdir(exist_ok=False)
    for name in paths:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        assert record(target) == current[name], name
    assert all(record(ROOT / name) == current[name] for name in paths), "workspace changed during snapshot"
    save("source_manifest.json", current)
    save("source_delta.json", [{"path": name, "kind": "added" if name not in previous else "changed",
        "previous": previous.get(name), "current": current[name]} for name in paths if previous.get(name) != current[name]])
    with tarfile.open(HERE / "source.tar.gz", "x:gz") as archive:
        for name in paths:
            archive.add(destination / name, arcname=name, recursive=False)
    commands = deepcopy(read(PRIOR / "commands.candidate.final.json"))
    def rewrite(value):
        if isinstance(value, str):
            return value.replace("n5_confirmation_source_r3_final", "n5_confirmation_source_r4_final").replace(
                "n5_confirmation_preflight_r3_final", "n5_confirmation_preflight_r4_final").replace(
                "n5_confirmation_20260930_r3", "n5_confirmation_20261001_r4").replace(
                "n5_execution_20260930_r3", "n5_execution_20261001_r4")
        if isinstance(value, list): return [rewrite(item) for item in value]
        if isinstance(value, dict): return {key: rewrite(item) for key, item in value.items()}
        return value
    commands = rewrite(commands)
    differences = {}
    for task, row in commands["tasks"].items():
        for kind in ("preflight_argv", "capture_argv_not_executed"):
            argv = row[kind]
            option = "--diagnostic-cohort" if task == "stair" else "--eval-base-seed"
            argv[argv.index(option) + 1] = COHORT if task == "stair" else str(PUSH_SEED)
        before, after = row["preflight_argv"], row["capture_argv_not_executed"]
        differences[task] = [{"index": i, "option": before[i-1], "preflight": a, "capture": b}
                             for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(before) == len(after) and {d["option"] for d in differences[task]} == {"--mode", "--output", "--video-dir"}
        assert before[before.index("--mode")+1] == "preflight"
    commands.update(mode="CPU_PREPARATION_ONLY", capture_authorized_or_executed=False,
                    diagnostic_cohort=COHORT, push_eval_base_seed=PUSH_SEED)
    save("commands.candidate.json", commands)
    save("capture_argv_diff.json", differences)
    save("source_preparation.json", {"status": "SOURCE_SNAPSHOT_CREATED_CPU_PREFLIGHT_PENDING_NOT_RELEASED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_parent_manifest": str(PRIOR / "source_manifest.final.json"),
        "source_parent_manifest_sha256": record(PRIOR / "source_manifest.final.json")["sha256"],
        "source_parent_files": len(previous), "source_files": len(current),
        "source_manifest_sha256": record(HERE / "source_manifest.json")["sha256"],
        "bundle": record(HERE / "source.tar.gz"), "changed": [n for n in paths if n in previous and previous[n] != current[n]],
        "added": [n for n in paths if n not in previous],
        "review_assignment": {"path": PHASE + "N5_REVIEW_ASSIGNMENT_R4.json", **record(allocation_path)},
        "run_contract": {"path": PHASE + "N5_RUN_CARD_R4_ZH.md", **record(ROOT / PHASE / "N5_RUN_CARD_R4_ZH.md")},
        "cli_policy": {"path": PHASE + "N5_SOL_CLI_POLICY_R4.json", **record(policy_path)},
        "cpu_test_modules": ["tests.unit.test_stair_body_projection", "tests.unit.test_task_visual_observation"],
        "phase_test_patterns": ["test_reference_runner.py", "test_sol_cli_carrier.py"],
        "test_selection_basis": "Delta-only: Stair named projection/task observer wiring; assigned identities/own seal and current CLI reconstruction/actual access extraction. Existing unrelated arithmetic/history suites are not rerun.",
        "model_calls": 0, "gpu_runs": 0, "capture_runs": 0, "prior_source_unchanged": True,
        "limits": ["Run contract is NOT_RELEASED; no capture card or launch authorization is created here.",
                   "Reviewer roster binds roles only; raw membership and actual first-item task routing remain postcapture checks."]})
    print(json.dumps({"files": len(current), "changed": sum(previous.get(n) != current[n] for n in paths),
                      "source_manifest_sha256": record(HERE / "source_manifest.json")["sha256"], "capture_or_models": 0}))


if __name__ == "__main__":
    main()

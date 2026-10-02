"""One local source snapshot and NOT_RELEASED static-preflight card; no remote work."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
PHASE = PROJECT / "PLAN/feedback_sys/P3NEW"
BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
REMOTE_SOURCE = BASE + "/source_n5_preflight_r4"
REMOTE_OUTPUT = BASE + "/n5_preflight_r4"
HELPERS = tuple("PLAN/feedback_sys/P3NEW/" + name for name in (
    "extract_dual_view_evidence.py", "prepare_native_observer.py",
    "prepare_n5_media_selection.py", "verify_multi_env_capture.py"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path):
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def inventory_paths():
    paths = [p for folder in ("src", "scripts", "configs") for p in (PROJECT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths.extend(PROJECT / relative for relative in HELPERS)
    return {p.relative_to(PROJECT).as_posix(): p for p in sorted(paths)}


def main():
    old_card_path = PHASE / "N5_CAPTURE_CURRENT_DRAFT.json"
    old_manifest_path = PHASE / "results/n5_source_preflight_r3/source_manifest.json"
    preserved = {str(path): sha(path) for path in (old_card_path, old_manifest_path)}
    old = json.loads(old_card_path.read_bytes())
    old_files = json.loads(old_manifest_path.read_bytes())["files"]
    paths = inventory_paths()
    before = {relative: sha(path) for relative, path in paths.items()}
    source = HERE / "source"
    source.mkdir(exist_ok=False)
    for relative, path in paths.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == before[relative], relative
    assert {p: sha(f) for p, f in inventory_paths().items()} == before, "workspace changed during copy"
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = HERE / "source_manifest.json"
    save(manifest_path, {"created_at_utc": now,
        "purpose": "Local snapshot for N5 static preflight r4 only; no capture/review release",
        "files": before})
    changes = {"baseline": ref(old_manifest_path), "current_manifest": ref(manifest_path),
        "old_file_count": len(old_files), "new_file_count": len(before),
        "added": [{"path": p, "sha256": before[p]} for p in sorted(before.keys() - old_files.keys())],
        "removed": [{"path": p, "sha256": old_files[p]} for p in sorted(old_files.keys() - before.keys())],
        "modified": [{"path": p, "old_sha256": old_files[p], "new_sha256": before[p]}
                     for p in sorted(before.keys() & old_files.keys()) if before[p] != old_files[p]],
        "unchanged_count": sum(before[p] == old_files[p] for p in before.keys() & old_files.keys())}
    save(HERE / "source_changes.json", changes)
    bundle_path = HERE / "source.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as archive:
        for relative in before:
            archive.add(source / relative, arcname=relative, recursive=False)
    with tarfile.open(bundle_path, "r:gz") as archive:
        assert {row.name for row in archive.getmembers()} == set(before)
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
                   for name, digest in before.items())
    commands, tasks = {}, {}
    old_roots = {old["source"]["preflight_only_root"], old["source"]["final_frozen_source_root"]}
    for task, row in old["tasks"].items():
        argv = list(row["executed_preflight_argv"])
        for i, value in enumerate(argv):
            for prefix in old_roots:
                if value.startswith(prefix + "/"):
                    argv[i] = REMOTE_SOURCE + value[len(prefix):]
        for flag, value in (("--mode", "preflight"), ("--output", REMOTE_OUTPUT + "/" + task),
                            ("--video-dir", REMOTE_OUTPUT + "/" + task + "/video")):
            argv[argv.index(flag) + 1] = value
        assert argv[argv.index("--mode") + 1] == "preflight"
        assert argv[argv.index("--video-snapshot-interval-steps") + 1] == "1"
        assert "--video-snapshot-steps" not in argv
        assert argv[1].startswith(REMOTE_SOURCE + "/scripts/")
        for option in ("--checkpoint", "--checkpoint-summary", "--source-manifest", "--train-summary", "--eval-base-seed", "--cohort"):
            if option in argv:
                original = row["executed_preflight_argv"]
                assert argv[argv.index(option) + 1] == original[original.index(option) + 1]
        commands[task] = {"preflight_argv_not_executed": argv,
                          "environment": {"PYTHONPATH": REMOTE_SOURCE + "/src", "PYTHONDONTWRITEBYTECODE": "1"}}
        tasks[task] = {"server_verification_status": "PENDING_SERVER_CHECK",
            "inherited_checkpoint_reference": deepcopy(row["checkpoint"]),
            "inherited_source_input_references": deepcopy(row["source_input_refs"]),
            "inherited_training_binding_hash": row["preserved_training_binding_hash"],
            "inherited_source_training_manifest_hash": row["source_training_manifest_hash"],
            "prior_evaluation_binding_reference_not_current_proof": deepcopy(row["final_source_evaluation_binding"]),
            "cohort": deepcopy(row["cohort"]), "numeric_eval_envs": row["numeric_eval_envs"],
            "public_dev_seed_runs": row["numeric_public_dev_seed_runs"],
            "media_env_selection": deepcopy(row["media_env_selection"]),
            "selected_envs": [selection["selected_env"] for selection in row["media_env_selection"]],
            "env_mapping_status": "same preselected env IDs; mapping must be checked under this source in static preflight"}
        if "--protocol" in argv:
            relative = argv[argv.index("--protocol") + 1][len(REMOTE_SOURCE) + 1:]
            tasks[task]["current_local_protocol_reference"] = {"path": REMOTE_SOURCE + "/" + relative,
                "sha256": before[relative], "verification": "local snapshot bytes only; remote pending"}
    card_path = HERE / "preflight_card.json"
    save(card_path, {"schema_version": "p3new.n5_static_preflight_draft.v1", "status": "NOT_RELEASED",
        "purpose": "preflight-only local preparation; no server execution and no N5 capture release",
        "created_at_utc": now, "inherited_draft": ref(old_card_path),
        "source": {"remote_root": REMOTE_SOURCE, "manifest": ref(manifest_path), "bundle": ref(bundle_path),
                   "file_count": len(before), "local_snapshot_verified": True, "remote_verified": False,
                   "scope": "current src/scripts/configs excluding caches plus four existing capture helpers",
                   "is_final_n5_experiment_freeze": False},
        "commands": commands, "tasks": tasks, "capture_profile_reference_not_release": deepcopy(old["capture_profile"]),
        "deferred_freeze": ["review protocols and final model/observer/reviewer role configuration",
                            "independent arithmetic scripts and responsibility coverage",
                            "final facts/report/media packets, execution capacity and call budgets"],
        "prior_gpu_disk_process_snapshot_status": "STALE_REFERENCE_ONLY_PENDING_REFRESH",
        "prior_seed_freshness_status": "R3_EXPLICIT_HISTORY_SCOPE_INHERITED_NOT_RESCANNED",
        "prior_qualification_status": "INHERITED_REFERENCES_ONLY_PENDING_SERVER_CHECK",
        "release_gate": "Root authorized static preflight only; formal source freeze and capture remain NOT_RELEASED; no run argv exists here",
        "remote_commands_executed": 0, "provider_requests": 0, "gpu_jobs": 0, "training_jobs": 0,
        "formal_n5_evaluation_started": False})
    assert {p: sha(f) for p, f in inventory_paths().items()} == before, "workspace changed before freeze completion"
    assert all(sha(source / p) == digest for p, digest in before.items())
    assert all(sha(Path(path)) == digest for path, digest in preserved.items()), "old card or manifest changed"
    receipt = {"status": "LOCAL_SNAPSHOT_VERIFIED_NOT_RELEASED", "created_at_utc": now,
        "source_manifest": ref(manifest_path), "source_bundle": ref(bundle_path), "draft_card": ref(card_path),
        "source_changes": ref(HERE / "source_changes.json"), "freeze_script": ref(Path(__file__)),
        "source_file_count": len(before), "workspace_before_after_all_hashes_equal": True,
        "snapshot_all_hashes_equal": True, "bundle_all_file_hashes_equal": True,
        "old_card_and_manifest_unchanged": preserved,
        "change_counts": {key: len(changes[key]) for key in ("added", "removed", "modified")},
        "unchanged_count": changes["unchanged_count"], "remote_commands_executed": 0,
        "provider_requests": 0, "gpu_jobs": 0, "training_jobs": 0}
    save(HERE / "local_freeze_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Seal R2 from verified source/preflight receipts; launches nothing."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "n5_confirmation_source_r2"
OLD = HERE.parent / "n5_final_freeze_r1/capture_card.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    path = path.resolve()
    return {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}


source_receipt = read(SOURCE / "source_freeze_receipt.json")
manifest_path = SOURCE / "source_manifest.json"
manifest = read(manifest_path)
assert digest(manifest_path) == source_receipt["source_manifest_sha256"]
assert source_receipt["source_unchanged_after_execution"]
assert source_receipt["preflight_exit_codes"] == {"stair": 0, "push": 0}
for name, record in manifest.items():
    path = SOURCE / "source" / name
    assert digest(path) == record["sha256"] and path.stat().st_size == record["size_bytes"], name
verification = read(SOURCE / "downloaded/control/verification.json")
assert verification["status"] == "CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE"
commands = read(SOURCE / "commands.candidate.json")
cohorts = read(SOURCE / "final_verification.stdout.json")["tasks"]
storage = read(HERE / "storage_preflight.json")
resource = read(HERE / "resource_prestart.json")
assert not resource["capture_root_exists"] and not resource["control_root_exists"]
assert resource["disk_free_bytes"] >= storage["tasks"]["stair"]["required_before_task_bytes"]

card = read(OLD)
for name in ("history_scope", "pre_data_metadata_clarification"):
    card.pop(name)
card.update(status="FROZEN", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose="Fresh reset-trajectory confirmation after original N5 INCONCLUSIVE; no new terrain or qualification claim",
    main_plan_version="1.4.20", execution_status="READY_NOT_STARTED", review_status="NOT_STARTED_UNQUALIFIED",
    output_root=commands["capture_root"], original_card=ref(OLD), original_round_status="INCONCLUSIVE")
card["source"] = {"remote_root": commands["source_root"], "manifest": ref(manifest_path),
    "manifest_layout": "relative_path -> {sha256,size_bytes}", "file_count": len(manifest),
    "bundle": ref(SOURCE / "source.tar.gz"), "freeze_receipt": ref(SOURCE / "source_freeze_receipt.json"),
    "local_snapshot_verified": True, "remote_verified": True, "is_final_n5_experiment_freeze": True}
card["static_preflight"] = ref(SOURCE / "downloaded/control/verification.json")
card["experiment_files"] = ref(manifest_path)
card["run_contract"] = ref(ROOT / "PLAN/feedback_sys/P3NEW/N5_RUN_CARD_R2_ZH.md")
assert manifest["PLAN/feedback_sys/P3NEW/N5_RUN_CARD_R2_ZH.md"]["sha256"] == card["run_contract"]["sha256"]
assert manifest["PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md"]["sha256"] == card["review_protocol"]["sha256"]
card["cohort_verification"] = ref(SOURCE / "final_verification.stdout.json")
card["commands"] = {}
for task in ("stair", "push"):
    spec = card["tasks"][task]
    result_path = SOURCE / "downloaded/preflight" / task / "result.json"
    result = read(result_path)
    assert result["mode"] == "preflight"
    assert result["status"] == {"stair": "PREFLIGHT_PASS", "push": "PREPARED"}[task]
    assert result["checkpoint_sha256"] == "sha256:" + spec["checkpoint"]["sha256"]
    assert result["evaluation_binding"] == cohorts[task]["actual_evaluation_binding"]
    assert cohorts[task]["new_cohort_hash_absent_from_recorded_history"]
    assert not set(cohorts[task]["runner_seeds"]) & set(cohorts[task]["old_runner_seeds"])
    spec.update(current_evaluation_binding=result["evaluation_binding"],
        preserved_training_binding_hash=result["preserved_training_binding_hash"],
        source_training_manifest_hash=result["source_training_manifest_hash"],
        static_preflight_result=ref(result_path), runner_seeds=cohorts[task]["runner_seeds"])
    spec["cohort"] = ({"cli": "diagnostic", "diagnostic_cohort": "n5_r2_20260930",
        "namespace": cohorts[task]["namespace"], "fixed_terrain_new_reset_trajectories": True}
        if task == "stair" else {"eval_base_seed": 1600000000})
    command = commands["tasks"][task]
    before, after = command["preflight_argv"], command["capture_argv_not_executed"]
    differences = [{"index": i, "before": a, "after": b} for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(before) == len(after) and {before[d["index"]-1] for d in differences} == {"--mode", "--output", "--video-dir"}
    card["commands"][task] = {"cwd": command["cwd"], "argv": after,
        "actual_preflight_argv": before, "only_changes": differences,
        "environment_set": command["capture_environment"],
        "environment_unset_process_local": command["capture_environment_unset"], "executed": False}
card["model_roles"]["stair_observer"] = copy.deepcopy(card["model_roles"]["push_observer"])
card["model_roles"]["observer_input_transport"] = {
    "generator": "prepare_native_observer.py unchanged frozen blind text/S0/original PNGs",
    "api_envelope": "offline template only; DeepSeek fields in prepared artifacts are not the actual SOL carrier configuration",
    "actual_execution": "separate fresh SOL invocation/whitelist/image hashes/raw answer/access receipt; no provider dispatch of the template"}
card["resources"] = {"gpu": "one available 32GB GPU under existing runner lease", "sequence": ["stair", "push"],
    "minimum_disk_before_task_bytes": {t: storage["tasks"][t]["required_before_task_bytes"] for t in ("stair", "push")},
    "estimate_formula": storage["formula"], "receipt": ref(HERE / "storage_preflight.json"),
    "png_export": {"argv": ["--png-storage", "hardlink"], "shared_inode": True, "independent_backup": False,
        "source_and_export_immutable": True, "metadata_npz_separate": True, "copy_fallback": False},
    "no_gpu_hour_quota": True, "before_each_task": ["GPU/lease available", "disk meets this task estimate", "fresh output", "source hashes unchanged"],
    "space_fallback": "Pause before next task if needed; retain verified originals and all source bindings; no historical evidence deletion"}
card["latest_resource_prestart"] = ref(HERE / "resource_prestart.json")
card["budget"].update(history_peak_usage_ceiling_usd=0.468395796,
    new_stair_observer_calls=0, reserved_usd_per_stair_observer=0,
    new_api_reservation_total_usd=12, history_plus_planned_reserve_usd=12.468395796,
    initial_ledger=ref(HERE.parent / "n5_confirmation_execution_r1/remote_execution/budget_ledger.working.json"))
card["materials"]["explicit_numeric_preparation_options"] = ["--profile", "related_v1", "--max-output-tokens", "32768",
    "--capture-card", commands["execution_control_root"] + "/capture_card.json", "--protocol",
    commands["source_root"] + "/PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md"]
card["materials"]["task_definition_binding"] = "All task/run/numeric manifests explicitly supplied using --task-definition and --push-resolved-cell-manifest; bound private source index required"
card["known_limits"] = source_receipt["known_limits"]
with (HERE / "capture_card.json").open("x", encoding="utf-8") as stream:
    json.dump(card, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
(HERE / "runtime_source_manifest.json").write_bytes(manifest_path.read_bytes())
print(json.dumps({"card_sha256": digest(HERE / "capture_card.json"), "source_files": len(manifest),
    "model_calls": 0, "gpu_runs": 0, "output_root": card["output_root"]}))

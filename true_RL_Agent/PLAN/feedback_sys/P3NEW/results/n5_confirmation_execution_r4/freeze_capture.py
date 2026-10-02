"""Prepare one R4 capture card and the inherited single-task driver; launch nothing.

Run only after root approves the final 492-file CPU preflight and supplies this
freeze's read-only resource record and the current downloaded ledger bytes.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "n5_confirmation_source_r4"
SNAPSHOT = SOURCE / "source"
PREVIOUS = HERE.parent / "n5_confirmation_execution_r3"
PHASE = "PLAN/feedback_sys/P3NEW/"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def copy_bytes(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source != target:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    assert digest(source) == digest(target)


def main(resource_path, ledger_path):
    old_path = PREVIOUS / "capture_card.json"
    old = read(old_path)
    freeze_path = SOURCE / "source_freeze_receipt.json"
    freeze = read(freeze_path)
    # Earlier 491/492 failures remain evidence; only completed receipt names win.
    manifest_path = Path(freeze["source_manifest"]["path"]).resolve()
    bundle_path = Path(freeze["source_bundle"]["path"]).resolve()
    verification_path = Path(freeze["verification"]["path"]).resolve()
    assert ref(manifest_path) == freeze["source_manifest"]
    assert ref(bundle_path) == freeze["source_bundle"]
    assert ref(verification_path) == freeze["verification"]
    manifest = read(manifest_path)
    verification = read(verification_path)
    commands = read(SOURCE / "commands.candidate.json")
    assert digest(SOURCE / "commands.candidate.json") == freeze["commands_sha256"]
    assert commands["source_root"] == freeze["remote_source"]
    assert len(manifest) == freeze["source_files"] == verification["source_files"] == 492
    assert digest(manifest_path) == freeze["source_manifest_sha256"] == verification["source_manifest_sha256"]
    assert freeze["source_unchanged_after_execution"] and freeze["preflight_exit_codes"] == {"stair": 0, "push": 0}
    assert freeze["run_contract_status"] == "NOT_RELEASED"
    assert verification["status"] == "CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE"
    assert freeze["verification_sha256"] == digest(verification_path)
    assert verification["source_unchanged_after_tests_and_preflights"]
    assert commands["capture_authorized_or_executed"] is False
    assert commands["diagnostic_cohort"] == "n5_r4_20261001" and commands["push_eval_base_seed"] == 1800000000
    for required in ("run_sol_review.py", "run_confirmation_review.py", "prepare_desktop_review.py", "test_sol_cli_carrier.py"):
        assert PHASE + required in manifest
    assert "tests/unit/test_stair_body_projection.py" in manifest and "tests/unit/test_body_projection.py" in manifest
    for name, item in manifest.items():
        assert ref(SNAPSHOT / name)["sha256"] == item["sha256"] and (SNAPSHOT / name).stat().st_size == item["size_bytes"], name

    resource_path, ledger_path = Path(resource_path).resolve(), Path(ledger_path).resolve()
    resource, ledger = read(resource_path), read(ledger_path)
    assert resource["read_only"] and resource["observed_at_utc"] and resource["gpu_jobs"] == resource["api_calls"] == 0
    paths = {item["path"]: item["exists"] for item in resource["paths"].values()}
    assert paths[commands["capture_root"]] is False and paths[commands["execution_control_root"]] is False
    assert paths[commands["source_root"]] is True
    assert all(resource["disk_free_bytes"] >= required for required in old["resources"]["minimum_disk_before_task_bytes"].values())
    assert resource["gpu"]["returncode"] == 0
    gpu_rows = [[int(part.strip()) for part in line.split(",")] for line in resource["gpu"]["stdout"].splitlines()]
    assert any(total >= 32000 and used < 1024 for _, used, total in gpu_rows)
    assert digest(ledger_path) == resource["ledger"]["sha256"] and ledger_path.stat().st_size == resource["ledger"]["size_bytes"]
    assert ledger["budget_id"] == resource["ledger"]["budget_id"] == old["budget"]["budget_id"]
    assert ledger["limit_usd"] == old["budget"]["api_phase_limit_usd"] == 15
    assert len(ledger["entries"]) == resource["ledger"]["entries_count"]
    prior_ledger = read(PREVIOUS / "budget_ledger.working.json")
    assert ledger["entries"][:len(prior_ledger["entries"])] == prior_ledger["entries"], "preserve historical entries"
    history = sum(Decimal(str(row.get("charged_usd", row.get("reserved_usd", 0)))) for row in ledger["entries"])
    reserve = Decimal(str(old["budget"]["new_api_reservation_total_usd"]))
    assert history + reserve < Decimal(str(ledger["limit_usd"]))

    calibration_dir = ROOT / PHASE / "review_calibration_r3_gpt61"
    calibration_path, carrier_path = calibration_dir / "raw_comparison.json", calibration_dir / "carrier_receipt.json"
    calibration, carrier = read(calibration_path), read(carrier_path)
    assert calibration["carrier_receipt_sha256"] == digest(carrier_path)
    assert calibration["n"] == 20 and calibration["verdict_agreement_gate_passed"]
    assert calibration["verdict_agreement"] >= old["acceptance"]["calibration_agreement_min"]
    assert carrier["model"] == "gpt-6.1-sol" and carrier["reasoning_effort"] == "high" and carrier["fork_turns"] == "none"
    cli_proof_path = HERE.parent / "cli_clean_media_dev_r1/final_access_reaudit.json"
    cli_proof = read(cli_proof_path)
    assert cli_proof["status"] == "PASS_ENGINEERING_ACCESS_AND_ANSWER_STRUCTURE" and all(cli_proof["checks"].values())
    assert not cli_proof["formal_facts_sealed"] and not cli_proof["semantic_scoring_performed"]
    assert cli_proof["bindings"]["current_extractor"]["sha256"] == manifest[PHASE + "run_sol_review.py"]["sha256"]
    dev_dir = HERE.parent / "stair_body_identity_dev_r1"
    development = {}
    for name in ("root_module_review_receipt.json", "sol_audit_receipt.json", "cpu_preparation_receipt.json",
                 "offline_delivery/execution_receipt.json", "offline_delivery/local_download_verification.json"):
        path = dev_dir / name
        item = read(path)
        assert "development" in item["scope"].lower()
        assert item.get("confirmation_admitted", False) is False
        development[name] = {"source": ref(path), "status": item.get("status"), "scope": item["scope"]}
    assert development["sol_audit_receipt.json"]["status"] == "AUDITED_NOT_RELEASED"
    assert development["offline_delivery/execution_receipt.json"]["status"] == "DEVELOPMENT_OFFLINE_DELIVERY_VERIFIED_NOT_SENT"
    assert development["offline_delivery/local_download_verification.json"]["status"] == "LOCAL_SMALL_ARTIFACTS_VERIFIED"

    def frozen_ref(name):
        assert digest(SNAPSHOT / name) == manifest[name]["sha256"]
        return ref(SNAPSHOT / name)
    policy_path = PHASE + "N5_SOL_CLI_POLICY_R4.json"
    assignment_path = PHASE + "N5_REVIEW_ASSIGNMENT_R4.json"
    policy = read(SNAPSHOT / policy_path)
    assert policy["status"] == "FROZEN" and policy["model"] == "gpt-6.1-sol" and policy["reasoning_effort"] == "high"
    assert policy["model_context_window"] == 512000 and policy["model_auto_compact_token_limit"] == 450000
    assert policy["output_reservation_tokens"] == 32768
    assignment = read(SNAPSHOT / assignment_path)
    assert assignment["status"] == "FROZEN" and len(assignment["packets"]) == 40
    assert sum(row[slot]["carrier"] == "fresh_cli" for row in assignment["packets"].values() for slot in ("A", "B")) == 72

    card = deepcopy(old)
    card.update(status="FROZEN", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="R4 unused reset-trajectory confirmation after Stair development regression; frozen independent CLI formal reviewers",
        main_plan_version=re.search(r"版本：v([\d.]+)", (SNAPSHOT / "PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md").read_text(encoding="utf-8")).group(1),
        execution_status="READY_NOT_STARTED", review_status="NOT_STARTED_UNQUALIFIED",
        output_root=commands["capture_root"], execution_control_root=commands["execution_control_root"],
        original_card=ref(old_path), static_preflight=ref(verification_path), cohort_verification=ref(verification_path),
        experiment_files=ref(manifest_path), run_contract=frozen_ref(PHASE + "N5_RUN_CARD_R4_ZH.md"),
        review_protocol=frozen_ref(PHASE + "N5_FACT_REVIEW_PROTOCOL_ZH.md"),
        dual_review_protocol=frozen_ref(PHASE + "REVIEW_PROTOCOL_ZH.md"),
        main_plan=frozen_ref("PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md"),
        review_assignment={**frozen_ref(assignment_path), "remote_path": commands["source_root"] + "/" + assignment_path},
        cli_policy={**frozen_ref(policy_path), "remote_path": commands["source_root"] + "/" + policy_path},
        latest_resource_prestart=ref(resource_path), module_review_status="PENDING_NEW_CAPTURE",
        prior_cohort_disposition=ref(PREVIOUS / "cohort_disposition.json"),
        development_regression_evidence=development, cli_transport_proof=ref(cli_proof_path))
    card["source"] = {"remote_root": commands["source_root"], "manifest": ref(manifest_path),
        "manifest_layout": old["source"]["manifest_layout"], "file_count": 492,
        "bundle": ref(bundle_path), "freeze_receipt": ref(freeze_path),
        "local_snapshot_verified": True, "remote_verified": True, "is_final_n5_experiment_freeze": True,
        "initial_491_failure_preserved": True, "fixture_only_492_failure_preserved": True,
        "preflight_corrections": ["Added existing tests/unit/test_body_projection.py fixture dependency",
                                  "Clarified DeepSeek assignment model/carrier error text; rejection logic unchanged"]}
    for task, spec in card["tasks"].items():
        result_path = verification_path.parent.parent / f"preflight/{task}/result.json"
        result, cohort = read(result_path), verification["tasks"][task]
        assert result["mode"] == "preflight" and result["evaluation_binding"] == cohort["actual_evaluation_binding"]
        assert result["checkpoint_sha256"] == "sha256:" + spec["checkpoint"]["sha256"]
        assert cohort["new_cohort_hash_absent_from_recorded_history"]
        assert len(set(cohort["runner_seeds"])) == 3 and all(not set(cohort["runner_seeds"]) & set(prior) for prior in cohort["prior_runner_seeds"].values())
        assert not set(cohort["runner_seeds"]) & set(old["tasks"][task]["runner_seeds"])
        spec.update(current_evaluation_binding=result["evaluation_binding"],
            preserved_training_binding_hash=result["preserved_training_binding_hash"],
            source_training_manifest_hash=result["source_training_manifest_hash"],
            static_preflight_result=ref(result_path), runner_seeds=cohort["runner_seeds"],
            cohort=({"cli": "diagnostic", "diagnostic_cohort": commands["diagnostic_cohort"],
                     "namespace": cohort["namespace"], "fixed_terrain_new_reset_trajectories": True}
                    if task == "stair" else {"eval_base_seed": commands["push_eval_base_seed"]}))
        row = commands["tasks"][task]
        before, after = row["preflight_argv"], row["capture_argv_not_executed"]
        differences = [{"index": i, "before": a, "after": b} for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(before) == len(after) and {before[d["index"] - 1] for d in differences} == {"--mode", "--output", "--video-dir"}
        assert before[before.index("--mode") + 1] == "preflight" and after[after.index("--mode") + 1] == "run"
        card["commands"][task] = {"cwd": row["cwd"], "argv": after, "actual_preflight_argv": before,
            "only_changes": differences, "environment_set": row["capture_environment"],
            "environment_unset_process_local": row["capture_environment_unset"], "executed": False}

    for name in ("gpt61_sol_a_judge", "gpt61_sol_b_judge"):
        role = card["model_roles"][name]
        role.pop("fork_turns", None)
        role.pop("observed_calibration_context_window", None)
        role.update(carrier="fresh_cli", model=policy["model"], reasoning=policy["reasoning_effort"],
            configured_context_window=policy["model_context_window"], configured_auto_compact_threshold=policy["model_auto_compact_token_limit"],
            new_empty_runtime_home_per_call=True, policy=card["cli_policy"],
            actual_model_usage_compaction_and_access_required=True, hidden_transport_retries="unobserved")
    card["model_roles"]["observer_input_transport"]["generator"] = (
        "Current frozen prepare_native_observer.py with task-owned named body image positions; original S0 and PNG bytes retained. Observer carrier remains fresh_subagent.")
    card["materials"].update(review_assignment_sha256=card["review_assignment"]["sha256"],
        actual_packet_task_routing="PENDING: after R4 packing verify fixed first task episode/focus IDs",
        explicit_numeric_preparation_options=["--profile", "related_v2", "--max-output-tokens", "32768", "--capture-card",
            commands["execution_control_root"] + "/capture_card.json", "--protocol", commands["source_root"] + "/" + PHASE + "N5_FACT_REVIEW_PROTOCOL_ZH.md",
            "--review-assignment", card["review_assignment"]["remote_path"]],
        explicit_media_preparation_options=["--review-assignment", card["review_assignment"]["remote_path"]])
    card["acceptance"].update(calibration_source=ref(calibration_path), calibration_carrier=ref(carrier_path),
        calibration_carrier_scope="Inherited GPT-6.1/high fresh_subagent calibration; fork_turns=none and observed 258400 window belong only to that historical carrier. CLI probe proves transport/structure, not new calibration or accuracy.")
    card["known_limits"][-1] = "No R4 module review, formal facts or report has been completed; old-data regressions are development evidence only."
    card["known_limits"].append("Fresh CLI calls retain roles as JSON fields; native API message-role equivalence and hidden backend behavior are not claimed.")
    card["resources"].update(receipt=ref(HERE / "storage_preflight.json") if (HERE / "storage_preflight.json").exists() else None,
        preparation_disk_free_bytes=resource["disk_free_bytes"], preparation_disk_gate_passed={task: True for task in card["tasks"]})
    card["budget"].update(history_peak_usage_ceiling_usd=float(history), history_plus_planned_reserve_usd=float(history + reserve),
        initial_ledger=ref(ledger_path), working_ledger_remote_path=commands["execution_control_root"] + "/budget_ledger.working.json")
    # Preserve task selection, thresholds and all original budget rates/counts.
    assert card["capture_profile"] == old["capture_profile"] and card["selection"] == old["selection"]
    assert all(card["acceptance"][k] == v for k, v in old["acceptance"].items())
    for task in card["tasks"]:
        for key in ("checkpoint", "selected_envs", "media_env_selection", "numeric_eval_envs", "numeric_public_dev_seed_runs"):
            assert card["tasks"][task][key] == old["tasks"][task][key]
    for key in old["budget"]:
        if key not in {"history_peak_usage_ceiling_usd", "history_plus_planned_reserve_usd", "initial_ledger", "working_ledger_remote_path", "ledger_inheritance"}:
            assert card["budget"][key] == old["budget"][key]

    for name in ("capture_card.json", "capture_driver.py", "runtime_source_manifest.json", "budget_ledger.working.json",
                 "ledger_inheritance.json", "storage_preflight.json", "launch_registration.json", "card_preparation_receipt.json"):
        assert not (HERE / name).exists(), "preserve prior output: " + name
    copy_bytes(ledger_path, HERE / "budget_ledger.working.json")
    save("ledger_inheritance.json", {"source": resource["ledger"], "download": ref(ledger_path),
        "working_initial": ref(HERE / "budget_ledger.working.json"), "history_charged_usd": float(history),
        "entries_preserved_byte_for_byte": True, "single_writer_sequential_execution_required": True})
    storage = deepcopy(read(PREVIOUS / "storage_preflight.json"))
    storage.update(capture_root=commands["capture_root"], free_bytes=resource["disk_free_bytes"], observed_at_utc=resource["observed_at_utc"],
        inherited_estimate=ref(PREVIOUS / "storage_preflight.json"), applicability="Same envs/horizon/views/resolution; original disk thresholds retained")
    save("storage_preflight.json", storage)
    card["resources"]["receipt"] = ref(HERE / "storage_preflight.json")
    card["budget"]["ledger_inheritance"] = ref(HERE / "ledger_inheritance.json")
    save("capture_card.json", card)
    copy_bytes(manifest_path, HERE / "runtime_source_manifest.json")
    old_sha, new_sha = digest(old_path), digest(HERE / "capture_card.json")
    driver_path = PREVIOUS / "capture_driver.py"
    driver = driver_path.read_text(encoding="utf-8")
    assert driver.count(old_sha) == 1
    driver = driver.replace(old_sha, new_sha)
    compile(driver, str(HERE / "capture_driver.py"), "exec")
    with (HERE / "capture_driver.py").open("x", encoding="utf-8", newline="") as stream:
        stream.write(driver)
    save("launch_registration.json", {"card_sha256": new_sha, "driver_sha256": digest(HERE / "capture_driver.py"),
        "source_manifest_sha256": digest(manifest_path), "inherited_driver": ref(driver_path),
        "changes": ["Only exact R4 card SHA commitment; original single-task runtime checks unchanged"],
        "driver_import_or_execution_performed": False, "launched": False, "gpu_jobs": 0, "api_calls": 0})
    save("card_preparation_receipt.json", {"status": "FROZEN_CARD_AND_DRIVER_PREPARED_NOT_LAUNCHED",
        "card": ref(HERE / "capture_card.json"), "driver": ref(HERE / "capture_driver.py"), "generator": ref(__file__),
        "source_manifest": ref(manifest_path), "source_files": 492, "source_preflight": ref(verification_path),
        "resource": ref(resource_path), "resource_observed_at_utc": resource["observed_at_utc"],
        "current_downloaded_ledger": ref(ledger_path), "cli_transport_proof": ref(cli_proof_path),
        "historical_calibration_carrier": "fresh_subagent", "formal_carrier": "fresh_cli",
        "formal_n5_pass_claimed": False, "launched": False, "api_calls": 0, "gpu_jobs": 0})
    print(json.dumps({"status": "FROZEN_CARD_AND_DRIVER_PREPARED_NOT_LAUNCHED", "card_sha256": new_sha,
                      "source_files": 492, "launched": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--ledger-json", type=Path, required=True)
    args = parser.parse_args()
    main(args.resource_json, args.ledger_json)

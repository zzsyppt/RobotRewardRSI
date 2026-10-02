"""Build one R3 capture card and reuse the R2 single-task driver; launch nothing."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "n5_confirmation_source_r3"
SNAPSHOT = SOURCE / "source_final"
PREVIOUS = HERE.parent / "n5_confirmation_execution_r2"
CALIBRATION = ROOT / "PLAN/feedback_sys/P3NEW/review_calibration_r3_gpt61"


def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path): return {"path": str(path.resolve()), "sha256": digest(path), "size_bytes": path.stat().st_size}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    old = read(PREVIOUS / "capture_card.json")
    manifest_path = SOURCE / "source_manifest.final.json"
    manifest = read(manifest_path)
    receipt = read(SOURCE / "source_freeze_receipt.json")
    assert digest(manifest_path) == receipt["source_manifest_sha256"]
    assert receipt["source_unchanged_after_execution"] and receipt["preflight_exit_codes"] == {"stair": 0, "push": 0}
    for name, item in manifest.items():
        path = SNAPSHOT / name
        assert digest(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], name
    verification_path = SOURCE / "downloaded_final/control/verification.json"
    verification = read(verification_path)
    assert verification["status"] == "CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE"
    commands = read(SOURCE / "commands.candidate.final.json")
    assert len(sys.argv) == 2, "Supply the newly verified read-only resource JSON after storage preparation"
    resource_path = Path(sys.argv[1]).resolve()
    resource = read(resource_path)
    assert not resource["paths"]["n5_confirmation_20260930_r3"]["exists"]
    assert not resource["paths"]["n5_execution_20260930_r3"]["exists"]
    assert all(resource["disk_free_bytes"] >= required for required in old["resources"]["minimum_disk_before_task_bytes"].values()), "Keep pre-resource draft; wait for storage gates before writing the final artifacts"
    calibration = read(CALIBRATION / "raw_comparison.json")
    carrier_path = CALIBRATION / "carrier_receipt.json"
    carrier = read(carrier_path)
    assert calibration["carrier_receipt_sha256"] == digest(carrier_path)
    assert calibration["n"] == 20 and calibration["verdict_agreement_gate_passed"]
    assert calibration["verdict_agreement"] >= old["acceptance"]["calibration_agreement_min"]
    assert carrier["model"] == "gpt-6.1-sol" and carrier["reasoning_effort"] == "high" and carrier["fork_turns"] == "none"
    ledger_path = HERE / "budget_ledger.inherited.json"
    ledger = read(ledger_path)
    assert digest(ledger_path) == resource["ledger"]["sha256"]
    assert len(ledger["entries"]) == 6 and ledger["budget_id"] == "p3new-20260929-total-api" and ledger["limit_usd"] == 15
    history_cost = sum(Decimal(str(item.get("charged_usd", item.get("reserved_usd", 0)))) for item in ledger["entries"])
    assert abs(history_cost - Decimal("0.723248796")) < Decimal("0.000000000001")
    assert history_cost + Decimal("2.4") < Decimal("15")
    with (HERE / "budget_ledger.working.json").open("xb") as stream:
        stream.write(ledger_path.read_bytes())
    save(HERE / "ledger_inheritance.json", {"source": resource["ledger"], "download": ref(ledger_path),
         "working_initial": ref(HERE / "budget_ledger.working.json"), "history_charged_usd": float(history_cost),
         "entries_preserved_byte_for_byte": True, "single_writer_sequential_execution_required": True})
    storage = deepcopy(read(PREVIOUS / "storage_preflight.json"))
    storage.update(capture_root=commands["capture_root"], free_bytes=resource["disk_free_bytes"],
        observed_at_utc=resource["observed_at_utc"], inherited_estimate=ref(PREVIOUS / "storage_preflight.json"),
        applicability="Same frozen runtime/horizon/envs/dual views/resolution/interval; retained R2 estimates, no disk threshold relaxation")
    save(HERE / "storage_preflight.json", storage)
    with (HERE / "resource_prestart.json").open("xb") as stream:
        stream.write(resource_path.read_bytes())

    handled = set()
    def inherit(name):
        handled.add(name)
        return deepcopy(old[name])
    def replace(name): handled.add(name)
    def frozen_ref(name):
        path = SNAPSHOT / name
        assert digest(path) == manifest[name]["sha256"]
        return ref(path)

    card = {}
    for name in ("schema_version", "capture_profile", "selection", "original_failures_preserved", "original_round_status"):
        card[name] = inherit(name)
    card["five_questions"] = inherit("five_questions")
    assert digest(ROOT / card["five_questions"]["repository_relative_path"]) == card["five_questions"]["sha256"]
    updates = {
        "status": "FROZEN", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "R3 fresh reset-trajectory confirmation; related_v2/reference_blocks.v2 and frozen mixed independent review roles",
        "main_plan_version": "1.4.23", "execution_status": "READY_NOT_STARTED",
        "review_status": "NOT_STARTED_UNQUALIFIED", "output_root": commands["capture_root"],
        "source": {"remote_root": commands["source_root"], "manifest": ref(manifest_path),
                   "manifest_layout": "relative_path -> {sha256,size_bytes}", "file_count": len(manifest),
                   "bundle": ref(SOURCE / "source.final.tar.gz"), "freeze_receipt": ref(SOURCE / "source_freeze_receipt.json"),
                   "local_snapshot_verified": True, "remote_verified": True, "is_final_n5_experiment_freeze": True},
        "static_preflight": ref(verification_path), "experiment_files": ref(manifest_path),
        "run_contract": frozen_ref("PLAN/feedback_sys/P3NEW/N5_RUN_CARD_R3_ZH.md"),
        "review_protocol": frozen_ref("PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md"),
        "latest_resource_prestart": ref(HERE / "resource_prestart.json"),
        "original_card": ref(PREVIOUS / "capture_card.json"), "cohort_verification": ref(verification_path),
        "known_limits": verification["limitations"] + [
            "Raw calibration agreement passed the existing gate; it is not accuracy or formal N5 qualification.",
            "Fresh subagents share a filesystem; whitelist and tool-trace audit are required, not OS isolation.",
            "Recorded resource availability is a point-in-time observation; the driver repeats per-task disk/GPU checks and the runner owns its lease.",
            "No new observer module review, facts or report has been completed for R3."],
        "provider_requests_by_this_freeze": 0, "gpu_jobs_by_this_freeze": 0,
        "formal_n5_evaluation_started": False, "module_review_sealed_before_any_formal_facts_request": True
    }
    for name, value in updates.items():
        replace(name)
        card[name] = value
    card["main_plan"] = frozen_ref("PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md")
    card["dual_review_protocol"] = frozen_ref("PLAN/feedback_sys/P3NEW/REVIEW_PROTOCOL_ZH.md")
    assignment_relative = "PLAN/feedback_sys/P3NEW/N5_REVIEW_ASSIGNMENT_R3.json"
    card["review_assignment"] = frozen_ref(assignment_relative)
    assert card["review_assignment"]["sha256"] == "b898502e9fa33caf415cdcaf58f7b862de1b68473a898d535a07660ccd1cd893"
    card["review_assignment"]["remote_path"] = commands["source_root"] + "/" + assignment_relative
    card["execution_control_root"] = commands["execution_control_root"]
    card["module_review_status"] = "PENDING_NEW_CAPTURE"
    card["freeze_is_not_process_launch"] = True

    card["tasks"] = inherit("tasks")
    replace("commands")
    card["commands"] = {}
    for task in ("stair", "push"):
        spec = card["tasks"][task]
        result_path = SOURCE / "downloaded_final/preflight" / task / "result.json"
        result = read(result_path)
        cohort = verification["tasks"][task]
        assert result["mode"] == "preflight" and result["evaluation_binding"] == cohort["actual_evaluation_binding"]
        assert result["checkpoint_sha256"] == "sha256:" + spec["checkpoint"]["sha256"]
        assert cohort["new_cohort_hash_absent_from_recorded_history"]
        assert all(not set(cohort["runner_seeds"]) & set(prior) for prior in cohort["prior_runner_seeds"].values())
        spec.update(current_evaluation_binding=result["evaluation_binding"],
            preserved_training_binding_hash=result["preserved_training_binding_hash"],
            source_training_manifest_hash=result["source_training_manifest_hash"],
            static_preflight_result=ref(result_path), runner_seeds=cohort["runner_seeds"],
            cohort=({"cli": "diagnostic", "diagnostic_cohort": "n5_r3_20260930", "namespace": cohort["namespace"],
                     "fixed_terrain_new_reset_trajectories": True} if task == "stair" else {"eval_base_seed": 1700000000}))
        command = commands["tasks"][task]
        before, after = command["preflight_argv"], command["capture_argv_not_executed"]
        differences = [{"index": i, "before": first, "after": second} for i, (first, second) in enumerate(zip(before, after)) if first != second]
        assert len(before) == len(after) and {before[item["index"] - 1] for item in differences} == {"--mode", "--output", "--video-dir"}
        card["commands"][task] = {"cwd": command["cwd"], "argv": after, "actual_preflight_argv": before,
            "only_changes": differences, "environment_set": command["capture_environment"],
            "environment_unset_process_local": command["capture_environment_unset"], "executed": False}

    replace("model_roles")
    observer = {"carrier": "fresh_subagent", "model": "gpt-6.1-sol", "reasoning": "high", "fork_turns": "none", "calls": 2,
                "server_fingerprint_usage_cost": "unavailable unless actually supplied"}
    sol = {"carrier": "fresh_subagent", "model": "gpt-6.1-sol", "reasoning": "high", "fork_turns": "none",
           "memory": False, "resume": False, "observed_calibration_context_window": carrier["desktop_usage_last_record"]["model_context_window"],
           "output_32768_is_reservation_not_carrier_enforced": True, "desktop_cost_usd": None, "fingerprint": None}
    card["model_roles"] = {
        "stair_observer": deepcopy(observer), "push_observer": deepcopy(observer),
        "module_semantic_reviewer": deepcopy(old["model_roles"]["module_semantic_reviewer"]),
        "deepseek_judge": dict(deepcopy(old["model_roles"]["deepseek_judge"]), judge_id="deepseek", calls=16, units=8, role="B"),
        "gpt61_sol_a_judge": dict(sol, judge_id="gpt6.1-sol-a", calls=80, units=40, role="A"),
        "gpt61_sol_b_judge": dict(sol, judge_id="gpt6.1-sol-b", calls=64, units=32, role="B"),
        "formal_judges_share_no_development_or_observer_context": True,
        "a_b_share_no_context_or_answers": True, "automatic_retries": 0,
        "same_family_observer_judge_correlation_disclosed": True,
        "cross_family_units": 8, "same_family_independent_context_units": 32,
        "observer_input_transport": deepcopy(old["model_roles"]["observer_input_transport"])}
    card["materials"] = inherit("materials")
    card["materials"].update(numeric_profile="related_v2", reference_contract="p3new.reference_blocks.v2",
        review_assignment_sha256=card["review_assignment"]["sha256"], all_80_facts_sealed_before_any_report=True,
        actual_packet_task_routing="PENDING: after R3 packing confirm episode01/13 and focus01/07 are each task first IDs",
        explicit_numeric_preparation_options=["--profile", "related_v2", "--max-output-tokens", "32768", "--capture-card",
            commands["execution_control_root"] + "/capture_card.json", "--protocol", commands["source_root"] + "/PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md",
            "--review-assignment", card["review_assignment"]["remote_path"]],
        explicit_media_preparation_options=["--review-assignment", card["review_assignment"]["remote_path"]])
    card["acceptance"] = inherit("acceptance")
    card["acceptance"].update(calibration_source=ref(CALIBRATION / "raw_comparison.json"),
        calibration_carrier=ref(carrier_path), calibration_status="AGREEMENT_GATE_PASSED_NOT_ACCURACY",
        calibration_verdict_agreement=calibration["verdict_agreement"], calibration_kind_agreement=calibration["kind_agreement"])
    card["resources"] = inherit("resources")
    card["resources"].update(receipt=ref(HERE / "storage_preflight.json"),
        preparation_disk_free_bytes=resource["disk_free_bytes"],
        preparation_disk_gate_passed={task: resource["disk_free_bytes"] >= required for task, required in card["resources"]["minimum_disk_before_task_bytes"].items()})
    assert all(card["resources"]["preparation_disk_gate_passed"].values()), "Keep pre-resource draft; wait for storage gates before final freeze"
    replace("budget")
    card["budget"] = {
        "api_phase_limit_usd": 15, "budget_id": ledger["budget_id"], "history_peak_usage_ceiling_usd": float(history_cost),
        "new_review_calls": 16, "new_api_observer_calls": 0, "reserved_usd_per_review": 0.15,
        "new_api_reservation_total_usd": 2.4, "history_plus_planned_reserve_usd": float(history_cost + Decimal("2.4")),
        "prices_usd_per_million": deepcopy(old["budget"]["prices_usd_per_million"]),
        "pricing_source": old["budget"]["pricing_source"], "pricing_checked_date": old["budget"]["pricing_checked_date"],
        "pricing_basis": "Inherited registered rates for consistent accounting, not a new quote or invoice",
        "initial_ledger": ref(ledger_path), "working_ledger_remote_path": commands["execution_control_root"] + "/budget_ledger.working.json",
        "ledger_inheritance": ref(HERE / "ledger_inheritance.json"), "desktop_cost": "unknown_not_zero",
        "desktop_formal_review_stage_calls": 144, "desktop_observer_calls": 4,
        "each_actual_request_requires_separate_budget_and_material_release": True,
        "single_writer_sequential_execution_required": True}
    assert handled == set(old), {"unhandled_old_fields": sorted(set(old) - handled), "unknown_old_fields": sorted(handled - set(old))}
    semantic_fields = {key: value for key, value in card.items() if key not in ("original_card", "original_round_status", "five_questions")}
    semantic_text = json.dumps(semantic_fields)
    assert card["materials"]["numeric_profile"] == "related_v2"
    for stale in ("gpt-6-sol", "gpt6-sol", "n5_r2_20260930", "1600000000", "n5_confirmation_source_r2", "n5_confirmation_20260930_r2"):
        assert stale not in semantic_text, stale
    assert card["model_roles"]["deepseek_judge"]["calls"] == card["budget"]["new_review_calls"] == 16
    save(HERE / "capture_card.json", card)
    with (HERE / "runtime_source_manifest.json").open("xb") as stream:
        stream.write(manifest_path.read_bytes())
    driver_path = PREVIOUS / "capture_driver.py"
    driver = driver_path.read_text()
    old_sha, new_sha = digest(PREVIOUS / "capture_card.json"), digest(HERE / "capture_card.json")
    assert driver.count(old_sha) == 1
    updated_driver = driver.replace(old_sha, new_sha)
    compile(updated_driver, str(HERE / "capture_driver.py"), "exec")
    with (HERE / "capture_driver.py").open("x", encoding="utf-8", newline="") as stream:
        stream.write(updated_driver)
    save(HERE / "launch_registration.json", {"card_sha256": new_sha, "driver_sha256": digest(HERE / "capture_driver.py"),
        "source_manifest_sha256": digest(manifest_path), "inherited_driver": ref(driver_path),
        "changes": ["Only exact R3 card SHA commitment; single-task capture logic unchanged"],
        "driver_import_or_execution_performed": False, "launched": False,
        "readiness": "READY_NOT_STARTED_RECHECK_RESOURCES_AT_LAUNCH", "gpu_jobs": 0, "api_calls": 0})
    save(HERE / "card_preparation_receipt.json", {"status": "FROZEN_CARD_AND_DRIVER_PREPARED_NOT_LAUNCHED",
        "card": ref(HERE / "capture_card.json"), "driver": ref(HERE / "capture_driver.py"),
        "old_top_level_fields_explicitly_handled": sorted(handled), "legacy_execution_values_absent": True,
        "source_files_verified": len(manifest), "calibration_gate_passed": True, "ledger_entries": len(ledger["entries"]),
        "history_charged_usd": float(history_cost), "new_api_reserve_usd": 2.4,
        "resource": ref(HERE / "resource_prestart.json"), "launched": False, "api_calls": 0, "gpu_jobs": 0})
    print(json.dumps({"status": "FROZEN_CARD_AND_DRIVER_PREPARED_NOT_LAUNCHED", "card_sha256": new_sha,
                      "driver_sha256": digest(HERE / "capture_driver.py"), "ledger_entries": len(ledger["entries"])}))


if __name__ == "__main__":
    main()

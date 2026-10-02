"""Prepare the two final N4 slots from their own old development runs; no send."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.insert(0, str(BASE))
import true_rl_agent
import prepare_native_author_final_slots as P
from true_rl_agent.evaluation.visual_review import attach_visual_review
from true_rl_agent.tasks.stair_climbing.behavior import build_stair_behavior
from true_rl_agent.tasks.stair_climbing.process_report import derive_stair_process_behavior

D, require = P.D, P.N.require
INPUT = HERE / "source_inputs"
RESULTS = BASE / "results"
SHARED = "/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1"


def material(task, cutoff):
    source = INPUT / task
    phase_path, plan_path, old_full_path = [source / name for name in
        ("public_dev_result.json", "public_dev_job.json", "behavior_full.json")]
    phase, old = D.read(phase_path), D.read(old_full_path)
    binding = phase["task_evidence"]["behavior_full"]
    require(binding["sha256"] == P.N.digest(old_full_path)
            and binding["size_bytes"] == old_full_path.stat().st_size, "original phase/full mismatch")
    if task == "stair":
        packet = RESULTS / "diagnostic_module_probe_r10/control/media-02"
        audit_path = packet / "response/mechanical_audit.json"
        review_path = packet.parent / "root_semantic_review/media-02_semantic_review.json"
        observer_paths = [audit_path, review_path, packet / "private_sidecar.json", packet / "response/raw_answer.txt"]
        reference = deepcopy(phase["task_evidence"]["numeric_evidence"])
        reference["manifest_path"] = str(source / "numeric_evidence/manifest.json")
        measured = build_stair_behavior(reference)
        full = derive_stair_process_behavior(measured)
        audit = D.read(audit_path)
        require(all(row["media_identity"]["source_id"] == full["domain"]["archive_hash"]
                    and row["media_identity"]["session_id"] == phase["job_id"]
                    for row in audit["observations"]), "Stair observer is from a different run")
        observer_version = P.SCHEMA_VERSION + ".r10.supplied_nonblind_review"
        unreviewed_full = P.attach_stair_visual_observations(full, audit, observer_version=observer_version)
        unreviewed = P.project_stair_process_report(unreviewed_full)
        unreviewed_ids = {row["id"] for row in unreviewed["report"]["visual"]}
        require(not any(row.get("kind") == "S3" and row["id"] in unreviewed_ids
                        for row in D.sentences(task, unreviewed_full)), "unreviewed Stair S3 exposed")
        full = P.attach_stair_visual_observations(full, attach_visual_review(audit, D.read(review_path)),
            observer_version=observer_version)
        public = P.project_stair_process_report(full)
        require(full["episodes"] == measured["episodes"], "visual attachment changed measured episodes")
        method = "Current production build_stair_behavior over r10's own manifest and all declared NPZ chunks, then current process derivation and original supplied review attachment."
        source_checks = {"original_numeric_manifest": D.ref(source / "numeric_evidence/manifest.json"),
            "observer_archive_matches": True, "all_declared_npz_reopened_by_current_builder": True,
            "progress_recomputed_from_same_run": True,
            "visual_attachment_preserves_recomputed_episodes": True, "unreviewed_s3_count": 0}
    else:
        control = RESULTS / "model_carrier_probe_r12/control"
        projected = control / "reviewed_projection"
        audit_path, review_path = control / "mechanical_audit.json", control / "root_semantic_review.json"
        observer_paths = [audit_path, review_path, projected / "receipt.json",
            projected / "behavior_full.json", projected / "public_report.json"]
        receipt = D.read(projected / "receipt.json")
        require(all(P.N.digest(P.SOURCE_ROOT / name) == sha for name, sha in receipt["loaded_current_source_sha256"].items()),
                "Push production source changed since r12 reviewed projection")
        full = D.read(projected / "behavior_full.json")
        require(all(full[key] == old[key] for key in ("binding", "source_bindings", "sources", "transcripts")),
                "r12 full is not the same r2 numeric identity")
        public = P.project_push_process_behavior(full)
        require(public == D.read(projected / "public_report.json"), "current Push projection drift")
        method = "Current production reserialization of r12 reviewed projection over its own r2 numerical v3; same-source stored v3 template, not L0/v2 recomputation. Original SOL answer has no provider HTTP receipt."
        source_checks = {"r12_current_source_hashes_match": True, "r2_binding_and_transcripts_match": True,
            "current_public_projection_equals_r12": True, "L0_recomputed_here": False,
            "original_projection_receipt": D.ref(projected / "receipt.json")}
    require(not full["observer_source_binding"]["rejected_observation_indices"], "observer source rejected")
    rows = D.sentences(task, full)
    require(not any(row.get("kind") == "S4" for row in rows), "unverified parent comparison")
    shown_ids = {row["id"] for row in public["report"]["visual"]}
    require(any(row.get("kind") == "S3" and row["id"] in shown_ids for row in rows),
            "no reviewed visual sentence in bounded report")
    out = HERE / "materials" / task
    out.mkdir(parents=True, exist_ok=False)
    for name, path in (("source_phase.json", phase_path), ("source_full.json", old_full_path), ("source_plan.json", plan_path)):
        (out / name).write_bytes(path.read_bytes())
    full_ref = D.write(out / "behavior_full.json", full)
    derived = deepcopy(phase)
    derived["task_evidence"]["behavior"] = public
    derived["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": full_ref["sha256"], "size_bytes": full_ref["size_bytes"]}
    derived["task_evidence_hash"] = D.canonical_json_hash(derived["task_evidence"])
    phase_ref = D.write(out / "public_dev_result.json", derived)
    D.write(out / "public_report.json", public)
    categories = {row.get("evidence_category") for row in rows if row["id"] in shown_ids}
    provenance = {"original_phase_full_plan": [D.ref(p) for p in (phase_path, plan_path, old_full_path)],
        "observer_sources": [D.ref(p) for p in observer_paths], "method": method,
        "source_checks": source_checks, "remote_source_index": D.ref(INPUT / "remote_source_index.json"),
        "original_sources_unchanged": True, "derived_phase_is_not_new_qualification": True,
        "supplied_nonblind_review_reused_without_rescoring": True,
        "exposure": {"real_visual": "EXERCISED_IN_PREPARED_INPUT",
            "visual_only": "EXERCISED_IN_PREPARED_INPUT" if "visual_only" in categories else "NOT_EXERCISED",
            "conflict": "EXERCISED_IN_PREPARED_INPUT" if "conflict" in categories else "NOT_EXERCISED",
            "S4": "NOT_EXERCISED"}, "actual_author_reading": "NOT_RUN",
        "as_of_cutoff": cutoff, "cutoff_basis": "offline preparation time; no new simulation or model observation",
        "derived_phase": phase_ref}
    D.write(out / "provenance.json", provenance)
    return full, public, provenance


def main():
    cutoff = datetime.now(timezone.utc).isoformat()
    index = D.read(INPUT / "remote_source_index.json")
    for relative, reference in index["files"].items():
        path = INPUT / relative
        require(P.N.digest(path) == reference["sha256"] and path.stat().st_size == reference["size_bytes"],
                "downloaded original differs: " + relative)
    old_stage = D.read(INPUT / "shared_journal_readonly/execution_stage.released.json")
    usage_path = INPUT / "shared_journal_readonly/latest_native_usage.json"
    usage = D.read(usage_path)
    live = D.read(HERE / "shared_journal_validation.json")
    require(live["usage"] == usage and live["all_prior_json_hashes_unchanged"]
            and not live["generation004_exists"] and usage["attempts"] == 6 and not usage["violations"],
            "six-call history differs from original read-only journal check")
    episodes = []
    for task, lane in (("stair", "lane-a"), ("push", "lane-b")):
        full, public, provenance = material(task, cutoff)
        comp, builder, _ = P.context_for(task, HERE / "materials" / task, provenance, HERE / "contexts" / task)
        first, parent_hash = P.delivery(task, full, public, comp, builder, HERE / "prompts" / task)
        generation = "p3new-text-smoke-20260929-generation-004"
        episodes.append({"case_id": task + "-native-dev", "task": task, "generation_id": generation,
            "lane_id": lane, "round_id": generation + "-" + lane,
            "qualified_identity_source_case": P.QUALIFIED_CASES[task], "not_a_repeat_of_that_text_case": True,
            "qualified_parent_source_sha256": parent_hash,
            "material": D.ref(HERE / "materials" / task / "provenance.json"),
            "initial_http_body": first["request"], "initial_payload": first["payload"],
            "exposure": provenance["exposure"], "review_status": "NOT_RUN", "actual_final_seen_input": None})
        print(json.dumps({"task": task, "prepared": True, "draft_budget": first["budget"]}), flush=True)
    snapshot = D.write(HERE / "production_source_snapshot.json", P.project_source_snapshot(P.SOURCE_ROOT))
    _, frozen = D.bound_reader()
    r0b = ROOT / "PLAN/feedback_sys/P3R/r0b_inputs_local.json"
    frozen.extend(D.ref(path) for path in (r0b, r0b.parent / D.read(r0b)["p3_cases"]["path"],
        D.ORIGINALS / "source_index.json", ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_stair.json",
        ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_push.json", ROOT / "PLAN/feedback_sys/P3/author_experiment.py"))
    frozen.extend(D.ref(path) for path in (Path(__file__), Path(P.__file__), Path(P.N.__file__), Path(P.T.__file__),
        Path(D.__file__), Path(P.T.D.__file__), Path(P.C.__file__), Path(P.R.__file__), HERE / "driver/send_native_author_compat.py"))
    for episode in episodes:
        provenance = D.read(P.checked(episode["material"]))
        frozen.extend(provenance["observer_sources"])
    frozen.extend(D.ref(path) for path in sorted(HERE.rglob("*")) if path.is_file()
        and path.suffix != ".gz" and path.name not in ("stage.draft.json",))
    frozen = list({item["path"]: item for item in frozen}.values())
    prices = old_stage["budget"]["prices_usd_per_million"]
    require(prices == {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}, "verified peak prices differ")
    worst = (40000 * prices["cache_miss"] + 65536 * prices["output"]) / 1e6
    historical = usage["usd_peak_actual_or_reserved"]
    require(historical + 2 * worst < 4.0, "existing author budget insufficient")
    card = {"schema_version": "p3new.native_author_smoke.v1", "status": "PREPARED_NOT_RELEASED",
        "observer_schema_version": P.SCHEMA_VERSION, "scope": "old_development_not_n5",
        "purpose": "Last two of eight N4 episodes: actual author transport closure for current Stair progress and Push complete recovery outline; no training or A/T.",
        "episodes": episodes, "episode_budget": {"historical_text": 4, "historical_native": 2,
            "new_visual": 2, "total_if_run": 8, "limit": 8, "unused": 0},
        "provider": deepcopy(old_stage["provider"]), "pricing_status": "VERIFIED_FOR_PREPARATION_NOT_RELEASED",
        "pricing_evidence": {"official_url": "https://api-docs.deepseek.com/quick_start/pricing/",
            "checked_date_utc": "2026-09-30", "basis": "peak cache-miss input and full output reservation; legacy deepseek-v4-flash alias maps to current Flash"},
        "budget": {"stage_usd_cap": 4.0, "historical_actual_calls": 6, "historical_usd_ceiling": historical,
            "historical_usage_ref": D.ref(usage_path), "actual_invoice_usd": None,
            "historical_price_basis": prices, "prices_usd_per_million": prices,
            "new_actual_calls_cap": 2, "max_attempts_per_new_episode": 1,
            "author_total_actual_calls_if_both_sent": 8, "overall_n4_actual_calls_limit": 40,
            "input_reserve": 40000, "max_output_tokens": 65536,
            "two_request_peak_reservation_usd": 2 * worst,
            "remaining_usd_before_new_calls": 4.0 - historical,
            "total_with_two_peak_reservations_usd": historical + 2 * worst,
            "production_provider_caps": old_stage["budget"]["production_provider_caps"],
            "shared_journal_directory": SHARED, "max_generations": 4, "no_retry_or_replacement_episode": True},
        "runner": {"entry": "existing CandidateAuthoringLoop + CostGuardedProvider + RoundAuthoringJournal",
            "construction": "max_attempts=1; max_source_requests_per_round=0; no retry or replacement",
            "order": ["stair-native-dev", "push-native-dev"]},
        "release_remaining": ["Parent reviews and freezes a released card; preparation is not authorization to send.",
            "Deploy/reprepare in the existing remote material repository and run each sender --check-only against original shared journals before release; local path-bound bytes are not a remote execution receipt.",
            "After each unique call preserve actual HTTP request, response, terminal, and original shared-journal accounting."],
        "review_contract": P.T.review_contract(), "production_source_snapshot": snapshot,
        "material_repository_root": str(P.ROOT), "production_source_root": str(P.SOURCE_ROOT),
        "historical_accounting_check": D.ref(HERE / "shared_journal_validation.json"),
        "frozen_inputs": frozen, "api_calls_during_preparation": 0, "gpu_work": "NOT_RUN",
        "AT_value_gate": "NOT_APPLICABLE", "semantic_acceptance": "NOT_ESTABLISHED"}
    card["review_contract"]["exposure_limits"] = {row["case_id"]: row["exposure"] for row in episodes}
    D.write(HERE / "stage.draft.json", card)
    for reference in frozen:
        P.checked(reference)
    print(json.dumps({"status": card["status"], "episodes_if_sent": 8, "new_calls": 0,
        "usd_total_peak_ceiling": historical + 2 * worst, "card": str(HERE / "stage.draft.json")}))


if __name__ == "__main__":
    main()

"""Prepare four N4 text episodes without API calls, candidate execution or GPU.

This is a fixed four-case adapter to the existing production authoring loop.
The optional one-case run entry reuses the existing loop and journal accounting;
the saved draft card grants no API release.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
BASE = Path(__file__).parent
CASE_IDS = ("stair-01", "stair-03", "push-02", "push-04")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load(BASE / "audit_delivery_offline.py", "smoke_delivery_evidence")
C = load(ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", "smoke_confirmation_context")
R = load(ROOT / "PLAN/feedback_sys/P3R/author_ceiling_probe.py", "smoke_existing_author_entry")


def artifact(path):
    receipt = D.ref(path)
    receipt["path"] = Path(receipt["path"]).relative_to(ROOT).as_posix()
    return receipt


def check(reference):
    path = ROOT / reference["path"]
    assert D.digest(path.read_bytes()) == reference["sha256"], reference["path"]
    return path


def four_cases():
    r0b_path = ROOT / "PLAN/feedback_sys/P3R/r0b_inputs_local.json"
    r0b = D.read(r0b_path)
    old_path = r0b_path.parent / r0b["p3_cases"]["path"]
    assert D.digest(old_path.read_bytes()) == r0b["p3_cases"]["sha256"]
    old = {row["case_id"]: row for row in D.read(old_path)["cases"]}
    assert {row["case_id"] for row in r0b["cases"]} == set(CASE_IDS)
    return {name: old[name] for name in CASE_IDS}


def material_dir(task):
    return (BASE / "text_author_smoke_r3/current_push_materials" if task == "push"
            else BASE / "results/delivery_offline_r1/stair")


def verify_current_push_materials():
    from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior

    directory = material_dir("push")
    provenance = D.read(directory / "phase_provenance.json")
    check(provenance["production_process_source"])
    phase, full, public = [D.read(directory / name) for name in
                           ("public_dev_result.json", "behavior_full.json", "public_report.json")]
    reference = phase["task_evidence"]["behavior_full"]
    actual = artifact(directory / "behavior_full.json")
    assert reference == {"schema_version": "true_rl_agent.process_full_ref.v1", "file": "behavior_full.json",
                         "sha256": actual["sha256"], "size_bytes": actual["size_bytes"]}
    assert project_push_process_behavior(full) == public == phase["task_evidence"]["behavior"]
    assert D.canonical_json_hash(full) == public["full_derivation_sha256"]
    assert D.canonical_json_hash(phase["task_evidence"]) == phase["task_evidence_hash"]


def context_for(case_id, output):
    """Original qualification, with explicitly frozen current author materials."""
    case = four_cases()[case_id]
    task = case["task"]
    exp = C._p3(ROOT)
    exp.bound_read, _ = D.bound_reader()
    spec_path = ROOT / f"PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_{task}.json"
    assert D.digest(spec_path.read_bytes()) == case["author_context"]["sha256"]
    spec = D.read(spec_path)
    assert case["as_of_cutoff"] == spec["diagnostic_as_of"]
    extra = {}
    if task == "push":
        verify_current_push_materials()
        extra = {"diagnostic_eval_base_seed": spec["composition_values"]["eval_base_seed"],
            "source_manifest_ref": D.ref(D.ORIGINALS / "push/source_manifest.json"),
            "training_summary_ref": D.ref(D.ORIGINALS / "push/training_summary.json")}
    return C.build_context(case, "T", spec, root=ROOT, base=ROOT, output=output,
        diagnostic_phase_ref=D.ref(material_dir(task) / "public_dev_result.json"),
        diagnostic_plan_ref=D.ref(D.ORIGINALS / task / "public_dev_job.json"),
        allowed_seed_cohort_hash=spec["validated_evaluation_binding"]["seed_cohort_hash"],
        development_fixture=True, exp=exp, **extra)


def episode_loop(stage, case_id, output):
    """Construct the existing loop after release; constructing does not call it.

    Caller uses loop.run_round(round_id=round_id, records=[]), saves its terminal
    result, and reviews the authoritative journal before starting the next case.
    No alternative scheduling, retry or model-routing implementation lives here.
    """
    from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
    from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    assert stage["status"] == "FROZEN_API_RELEASED" and stage["pricing_status"] == "VERIFIED_FOR_AUTHOR_MODEL"
    for reference in stage["frozen_inputs"]:
        check(reference)
    assert project_source_snapshot(ROOT) == D.read(check(stage["production_source_snapshot"])), "production source drift"
    row = next(row for row in stage["episodes"] if row["case_id"] == case_id)
    comp, builder, _ = context_for(case_id, output / "contexts" / case_id)
    p, budget = stage["provider"], stage["budget"]
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model=p["requested_model"], base_url=p["base_url"], endpoint=p["endpoint"],
        api_key_env=p["api_key_env"], env_file=Path(p["env_file"]),
        output_dir=output / "provider_candidates" / row["generation_id"] / row["lane_id"],
        max_tokens=65536, reasoning_mode="adaptive", reasoning_effort="high", temperature=p["temperature"],
        timeout_s=p["timeout_s"], max_response_bytes=p["max_response_bytes"], empty_final_reasoning_retry=False))
    request = provider._chat_completion_request(R.first_prompt(comp, builder))
    assert D.digest(json.dumps(request, ensure_ascii=False).encode("utf-8")) == row["initial_http_body"]["sha256"]
    guarded = CostGuardedProvider(delegate=provider, output_dir=output, generation_id=row["generation_id"],
        lane_id=row["lane_id"], provider_caps=budget["production_provider_caps"], max_generations=2)
    loop = CandidateAuthoringLoop(context_builder=builder, provider=guarded,
        candidate_admission_policy=comp.candidate_admission_policy,
        round_dir=output / "authoring" / row["generation_id"] / row["lane_id"], max_attempts=5,
        max_source_requests_per_round=2, component_smoke_adapter=comp.task_plugin.component_smoke_adapter(),
        authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping())
    return loop, row["round_id"]


def run_case(stage_path, case_id, output):
    """One released episode, using the old runner's loop/accounting interfaces."""
    from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
    stage = D.read(stage_path)
    assert stage["status"] == "FROZEN_API_RELEASED" and stage["pricing_status"] == "VERIFIED_FOR_AUTHOR_MODEL"
    assert sys.platform == "linux", "run only on the registered humanoid-server host"
    assert stage["provider"]["env_file"] == "/data/yuyangliu/secrets/deepseek.env"
    assert stage["execution"]["python"] == str(Path(sys.executable).absolute())
    assert output == (ROOT / stage["execution"]["output_relative"]).resolve(), "use the single frozen journal directory"
    expected_model = stage["provider"]["verified_response_model"]
    expected_fingerprint = stage["provider"]["verified_system_fingerprint"]
    budget = stage["budget"]
    assert len(stage["episodes"]) == 4 and budget["production_provider_caps"] == {
        "attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 40000,
        "output_units_per_attempt_cap": 65536}
    prices = budget["prices_usd_per_million"]  # Added only after exact-model price verification.
    worst = (40000 * prices["cache_miss"] + 65536 * prices["output"]) / 1e6
    assert worst * 20 <= budget["stage_usd_cap"] == 4.0
    # Existing P3-R receipt parser expects its historical budget key, not a new parser.
    accounting_stage = deepcopy(stage)
    accounting_stage["budget"]["stage_usd"] = budget["stage_usd_cap"]
    exp = C._p3(ROOT)
    row = next(item for item in stage["episodes"] if item["case_id"] == case_id)
    terminal_path = output / "results" / f"{row['round_id']}.json"
    with acquire_process_file_lock(output / ".text_author_smoke.lock", blocking=False):
        if terminal_path.exists():
            return D.read(terminal_path)  # A terminal is never replaced by another episode.
        index = stage["episodes"].index(row)
        assert all((output / "results" / f"{prior['round_id']}.json").exists()
                   for prior in stage["episodes"][:index]), "run the frozen four-case order"
        usage = R.usage_report(output, accounting_stage, exp)
        assert not usage["violations"] and usage["usd_peak_actual_or_reserved"] + worst <= 4.0
        assert not any(item.get("usage_known") is False for item in usage["attempts_detail"]), "unknown prior usage; preserve and inspect"
        assert not usage["response_models"] or usage["response_models"] == [expected_model], "prior response model drift"
        assert all(item.get("system_fingerprint") == expected_fingerprint for item in usage["attempts_detail"] if item.get("usage_known")), "prior fingerprint drift"
        loop, round_id = episode_loop(stage, case_id, output)
        loop.provider.delegate.validate_credential()  # Local remote-host file read, no validation API.
        try:
            result = loop.run_round(round_id=round_id, records=[])
        finally:
            D.write(output / "usage.json", R.usage_report(output, accounting_stage, exp))
        terminal = {**row, "stop_reason": result.stop_reason,
            "candidate_private_path": result.candidate.path if result.candidate else None,
            "candidate": result.candidate.to_public_dict() if result.candidate else None,
            "authoring_checks": result.attempt_records, "prompt_payload_ref": result.prompt_payload_ref.to_public_dict(),
            "authoring_evidence_ref": result.authoring_evidence_ref.to_private_dict(),
            "static_failure": result.static_failure, "provider_failure": result.provider_failure,
            "train_status": "not_run", "review_status": "PENDING", "AT_value_gate": "NOT_APPLICABLE"}
        D.write(terminal_path, terminal)
        usage = D.read(output / "usage.json")
        assert not usage["violations"], "receipt violation; terminal and journals retained"
        assert not any(item.get("usage_known") is False for item in usage["attempts_detail"]), "unknown usage; stop later episodes"
        assert usage["response_models"] == [expected_model], "response model differs from verified release"
        fingerprints = {item.get("system_fingerprint") for item in usage["attempts_detail"] if item.get("usage_known")}
        assert fingerprints == {expected_fingerprint}, "fingerprint drift; stop later episodes"
        return terminal


def review_contract():
    return {
        "scope": "development read-smoke only; no author-value, semantic-qualification or training-improvement gate",
        "answer_location": "Existing candidate_draft contract: TrainingComponents.rationale plus all factual claims in components.py/docstrings; no new provider output schema.",
        "questions_per_focus": [
            {"id": "Q1", "question": "What is the cohort/focus denominator and observed outcome distribution? Keep score and success fraction distinct."},
            {"id": "Q2", "question": "Which measured processes are common or uncommon among failures, and is the stated contrast actually discriminating?"},
            {"id": "Q3", "question": "Does a compatible successful control share the process? State its matching domain, or the missing control."},
            {"id": "Q4", "question": "Which prompt-visible localized process relation or group measurement supports the proposed change? Preserve body, window and temporal overlap."},
            {"id": "Q5", "question": "What observation or sampling gap limits the inference? Separate observation, hypothesis, intervention and falsifiable prediction."},
        ],
        "questions_are_reviewer_checklist_not_extra_factual_answers_in_author_prompt": True,
        "hard_error_categories": [
            "episode_count_as_cell_count", "subgroup_fraction_as_cohort_fraction", "terminal_same_frame_as_strictly_before",
            "whole_episode_quantity_as_tail_window", "score_as_success_rate", "aggregate_interruptions_as_per_episode_threshold",
            "tensor_axis_error", "reward_bound_as_pointwise_dominance", "code_changes_claimed_preserved_behavior",
            "multiple_observations_or_conflict_as_multi_mechanism_permission", "visual_only_as_numeric_measurement",
            "S4_difference_as_causation", "invented_unobserved_fact", "unauthorized_input_or_component",
        ],
        "hard_error_rule": "Count only directly established contradiction/unsupported assertion, with output quote, final-seen evidence pointer, bound full/raw reference and numeric/code derivation. Hypotheses remain hypotheses; unresolved judgments stay pending, never zero.",
        "citation_record_fields": ["output_location", "claim_quote", "claim_kind", "report_sentence_id", "visible_in_actual_final_request",
            "seen_json_pointer", "full_evidence_refs", "quote_or_paraphrase", "scope_window_body_match", "judgment", "reason"],
        "citation_rules": [
            "Accept an explicit sentence ID, a matching bounded quote, or an unambiguous paraphrase mapped by the reviewer to a visible sentence; count these modes separately.",
            "An ID/number match is a location aid, not proof of correct meaning, unit, denominator, temporal relation or action relevance.",
            "Use each episode's actual final journal prompt, including repairs/source followup; do not assume the prepared draft was the final seen input.",
            "Record supported/contradicted/unsupported/unanswerable and pending separately. A correct fact absent from the actual input is not grounded use.",
        ],
        "existing_machine_helpers": ["P3/blind_export.py:rationale", "P3/blind_export.py:reference_locations", "P3/blind_export.py:reward_checks"],
        "helper_limits": "reference_locations locates literals only; natural-language sentence-ID/paraphrase matching and semantic errors require recorded review. It does not calculate new correctness scores.",
        "exposure_limits": {"visual_citation": "NOT_EXERCISED: no real observer content in these four text inputs",
            "actual_S4_causality_reading": "NOT_EXERCISED: genuine phase has no bound measured parent-child difference",
            "conflict_line_reading": "NOT_EXERCISED where no actual conflict line was supplied"},
        "all_terminals_retained": True, "no_candidate_or_service_error": "retain episode and journal; not success, zero errors, or permission to replace the episode",
        "not_a_gate": True, "no_AT_comparison": True,
    }


def prepare(output, source_update=None):
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    assert output.resolve().is_relative_to(BASE)
    assert not (output / "stage.draft.json").exists(), "use a fresh draft directory"
    delivery = D.read(BASE / "results/delivery_offline_r1/result.json")
    assert delivery["status"] == "PASS_OFFLINE_ENGINEERING"
    approved = D.read(source_update) if source_update else None
    if approved:
        previous = D.read(check(approved["previous_source_snapshot"]))["files"]
        current = project_source_snapshot(ROOT)["files"]
        changes = [{"path": name, "old_sha256": previous.get(name), "new_sha256": current.get(name)}
                   for name in sorted(previous.keys() | current.keys()) if previous.get(name) != current.get(name)]
        assert changes == approved["source_changes"]
        allowed = {"src/true_rl_agent/evaluation/visual_observer.py", "src/true_rl_agent/runner/native_video.py",
                   "scripts/stair_numeric_dev_replay.py", "scripts/push_numeric_replay.py",
                   "src/true_rl_agent/tasks/push_recovery/process_report.py"}
        assert changes and all(change["path"] in allowed for change in changes)
    for reference in delivery["production_sources"]:
        path = Path(reference["path"])
        current_hash = D.digest(path.read_bytes())
        if current_hash != reference["sha256"]:
            assert approved and {"path": path.relative_to(ROOT).as_posix(),
                "old_sha256": reference["sha256"], "new_sha256": current_hash} in approved["source_changes"]
    runtime_path = ROOT / "PLAN/feedback_sys/P3R/runtime_probe_r2/stage.json"
    runtime = D.read(runtime_path)
    provider = deepcopy(runtime["provider"])
    assert provider["reasoning_effort"] == "high" and provider["max_output_tokens"] == 65536 and provider["max_input_tokens"] == 40000
    cases = four_cases()
    contract = review_contract()
    contract_ref = artifact_after_write(output / "review_contract.json", contract)
    episodes, review_rows, inputs = [], [], [contract_ref, artifact(runtime_path), artifact(Path(__file__))]
    if source_update:
        inputs.extend([artifact(source_update), approved["previous_source_snapshot"]])
    for task in ("stair", "push"):
        inputs.append(artifact(ROOT / f"PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_{task}.json"))
        for filename in ("public_report.json", "public_dev_result.json", "behavior_full.json", "phase_provenance.json", "raw_sources.json"):
            inputs.append(artifact(material_dir(task) / filename))
    inputs.append(artifact(D.ORIGINALS / "source_index.json"))
    _, original_refs = D.bound_reader()
    inputs.extend(artifact(Path(reference["path"])) for reference in original_refs)
    for i, case_id in enumerate(CASE_IDS):
        case, task = cases[case_id], cases[case_id]["task"]
        comp, builder, feedback = context_for(case_id, output / "contexts" / case_id)
        payload = R.first_prompt(comp, builder)
        assert case["common_focus"] in payload["selection_guidance"]
        report = D.behavior_in(payload, task)
        assert report == D.read(material_dir(task) / "public_report.json")
        assert not any("image_url" in str(message) for message in [payload])
        source, source_hash, role = select_authoring_anchor_source(payload)
        assert source and role == "current_best"
        body, user, delivered = D.capture_request(payload, output)
        directory = output / "prompts" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "user_payload.json").write_bytes(user)
        (directory / "request.http_body.json").write_bytes(body)
        frame = delivered["authoring_context_frame"]["budget_metadata"]
        assert frame["actual_prompt_payload_context_units"] <= 40000
        full = D.read(material_dir(task) / "behavior_full.json")
        catalog = {"case_id": case_id, "audience": "reviewer_only", "task": task,
            "common_focus": case["common_focus"], "scope": case["scope"],
            "full_ref": artifact(material_dir(task) / "behavior_full.json"),
            "sentences": D.sentences(task, full), "actionability": report["report"]["actionability"]}
        catalog_ref = artifact_after_write(output / "reviewer" / f"{case_id}.json", catalog)
        generation, lane = f"p3new-text-smoke-20260929-generation-{i // 2 + 1:03d}", "lane-a" if i % 2 == 0 else "lane-b"
        row = {"case_id": case_id, "task": task, "scope": case["scope"], "repetition": 1, "view": "new_process_text_only",
            "generation_id": generation, "lane_id": lane, "round_id": generation + "-" + lane,
            "common_focus": case["common_focus"], "original_case_sha256": D.canonical_json_hash(case),
            "qualified_checkpoint_hash": case["checkpoint_hash"], "observation_as_of": case["as_of_cutoff"],
            "report_input": artifact(material_dir(task) / "public_report.json"),
            "initial_payload": artifact(directory / "user_payload.json"), "initial_http_body": artifact(directory / "request.http_body.json"),
            "system_sha256": D.digest(json.loads(body)["messages"][0]["content"].encode("utf-8")),
            "parent_source_sha256": source_hash, "provider_task_sections": frame["provider_task_sections"],
            "estimated_input_context_units": frame["actual_prompt_payload_context_units"], "reviewer_catalog": catalog_ref,
            "final_seen_input": "PENDING: authoritative final journal PromptPayloadRef after actual episode"}
        episodes.append(row)
        inputs.extend([row["initial_payload"], row["initial_http_body"], catalog_ref])
        review_rows.append({"case_id": case_id, "status": "NOT_RUN", "terminal": None, "actual_requests": None,
            "claims": [], "question_bundle_use": {question["id"]: None for question in contract["questions_per_focus"]},
            "hard_error_categories": {key: None for key in contract["hard_error_categories"]},
            "process_citations": None, "visual_citations": "NOT_EXERCISED", "reviewer": None})
    for path in (ROOT / "PLAN/feedback_sys/P3R/r0b_inputs_local.json", ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/case_inputs.json",
                 ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", ROOT / "PLAN/feedback_sys/P3R/author_ceiling_probe.py",
                 ROOT / "PLAN/feedback_sys/P3/author_experiment.py", ROOT / "PLAN/feedback_sys/P3/blind_export.py", BASE / "audit_delivery_offline.py"):
        inputs.append(artifact(path))
    snapshot_ref = artifact_after_write(output / "production_source_snapshot.json", project_source_snapshot(ROOT))
    inputs.append(snapshot_ref)
    prices = runtime["budget"]["prices_usd_per_million"]
    worst = (40000 * prices["cache_miss"] + 65536 * prices["output"]) / 1e6
    stage = {"schema_version": "feedback.p3new.text_author_smoke.v1", "status": "PREPARED_NOT_RELEASED",
        "pricing_status": "PENDING_VERIFY_FOR_EXACT_AUTHOR_MODEL", "provider": provider,
        "execution": {"host": "humanoid-server", "python": "/data/yuyangliu/envs/isaaclab6_validation/bin/python",
            "python_version_verified_by_root": "3.12.14", "repository_paths": "relative to this script's repository root",
            "output_relative": "PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1",
            "credentials": "only the registered remote env file; never export its value or copy it into this bundle"},
        "budget": {"stage_usd_cap": 4.0, "author_episodes": 4, "max_actual_requests_per_episode": 5,
            "max_actual_requests_total": 20, "source_requests_per_episode_included_in_five": 2,
            "input_reserve": 40000, "max_output_tokens": 65536, "automatic_transport_retries": 0,
            "historical_peak_prices_usd_per_million": prices, "price_snapshot_ref": artifact(runtime_path),
            "historical_snapshot_worst_request_usd": worst, "historical_snapshot_worst_total_usd": 20 * worst,
            "historical_price_is_not_current_model_pricing_guarantee": True,
            "production_provider_caps": {"attempts_per_lane_per_generation": 5,
                "input_units_per_attempt_reserve": 40000, "output_units_per_attempt_cap": 65536}},
        "runner": {"context_entry": "prepare_text_author_smoke.context_for", "construction_entry": "prepare_text_author_smoke.episode_loop",
            "execution_entry": "prepare_text_author_smoke.run_case -> CandidateAuthoringLoop.run_round(round_id=round_id, records=[])",
            "reuse": ["P3R/author_ceiling_probe.first_prompt", "CandidateAuthoringLoop", "CostGuardedProvider", "RoundAuthoringJournal"],
            "fixed_slots": "2 generations x 2 existing storage lanes x 5 attempts = 20; lanes are storage slots, not A/T arms",
            "do_not_run_old_top_level_cli": "P3R author_ceiling_probe schedules 32 episodes/four views and is outside this smoke scope",
            "component_checks": "existing static verifier and bounded CPU component fixture; no simulator/training backend"},
        "stop_rules": ["No API before explicit frozen release, model-specific price verification and unchanged source/input hashes.",
            "The existing per-request guard stops unknown usage/input-cap violations. At each episode terminal, inspect response model/fingerprint and usage before admitting the next episode.",
            "Charge actual output/usage in full even above requested output cap; no truncation of accounting and no automatic replacement episode.",
            "Stop when any episode spends five attempts, total spends twenty, or the US$4 reservation cannot cover a next request.",
            "Inspect existing durable journals on resume; never restart an interrupted episode under a fresh identity to regain attempts."],
        "release_remaining": ["Verify current pricing/capabilities of requested_model deepseek-v4-flash and its current canonical mapping; conservative next-call reservation <= remaining US$4.",
            "Add budget.prices_usd_per_million and provider.verified_response_model / verified_system_fingerprint from reviewed current evidence; historical values are not automatically adopted.",
            "Verify the execution host's registered credential file is readable without exporting its value, and deploy the hash-frozen source/input bundle.",
            "Check stage is explicitly FROZEN_API_RELEASED only after the four-case card review; this preparation does not grant that state."],
        "input_notes": {"new_report_versions": {"stair": "full.v3/public.v4", "push": "behavior.v3/public.v3"},
            "two_tasks_two_observation_packages": True, "four_cases_are_not_four_independent_policies": True,
            "no_curated_old_Tc_num_or_Tc_vis_in_author_input": True, "no_confirmation_labels": True,
            "observation_readback_does_not_establish_N3_semantic_pass": True},
        "review_contract": contract_ref, "episodes": episodes, "frozen_inputs": inputs,
        "source_only_update": artifact(source_update) if source_update else None,
        "production_source_snapshot": snapshot_ref, "network_calls_during_preparation": 0,
        "training": "NOT_RUN", "gpu_work": "NOT_RUN", "AT_value_gate": "NOT_APPLICABLE", "N5_confirmation": "NOT_STARTED"}
    assert stage["budget"]["historical_snapshot_worst_total_usd"] <= 4
    D.write(output / "review_sheet_template.json", {"status": "NOT_RUN", "items": review_rows})
    D.write(output / "stage.draft.json", stage)
    return stage


def artifact_after_write(path, value):
    D.write(path, value)
    return artifact(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "text_author_smoke_r1")
    parser.add_argument("--stage", type=Path, help="Reviewed released card; only used with --run-case")
    parser.add_argument("--source-update", type=Path, help="Explicit approved observer/camera source update receipt; preparation only")
    parser.add_argument("--run-case", choices=CASE_IDS, help="Run exactly one released text episode on humanoid-server")
    args = parser.parse_args()
    if args.run_case:
        assert args.stage is not None, "--run-case needs an explicitly released --stage"
        assert args.source_update is None, "source updates belong to preparation, not execution"
        terminal = run_case(args.stage.resolve(), args.run_case, args.output.resolve())
        print(json.dumps({"case_id": args.run_case, "stop_reason": terminal["stop_reason"], "train_status": "not_run"}))
    else:
        assert args.stage is None, "--stage is only for an explicit --run-case"
        card = prepare(args.output.resolve(), args.source_update.resolve() if args.source_update else None)
        print(json.dumps({"status": card["status"], "episodes": len(card["episodes"]),
            "max_requests": 20, "historical_price_worst_usd": card["budget"]["historical_snapshot_worst_total_usd"],
            "pricing_status": card["pricing_status"], "api_calls": 0}))

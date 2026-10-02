"""Prepare two same-run native-development author inputs offline; never send.

--inputs JSON --output NEW_DIR (under P3NEW). Input JSON has schema_version
p3new.native_author_inputs.v1, scope=old_development_not_n5, purpose, and tasks:
{"stair": {"phase": REF, "plan": REF, "full": REF, "observer_dir": PATH},
 "push": {"phase": REF, "plan": REF, "full": REF, "observer_dir": PATH}}.
Each REF is {"path": PATH, "sha256": SHA256}; size_bytes is optional.
Use the completed replay's public_dev_result.json, public_dev_job.json, and
the original behavior_full.json named by its task_evidence.behavior_full.
observer_dir is the received send_native_observer packet, including its export.
No observations, conflicts, parent measurements, or qualification are invented.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

# Honor the frozen PYTHONPATH package before legacy PLAN helpers prepend their
# own repository's src. The old repository still owns qualification and journals.
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
import true_rl_agent
import prepare_native_observer as N
import prepare_text_author_smoke as T
from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
from true_rl_agent.evaluation.visual_observer import SCHEMA_VERSION, audit_observer_output
from true_rl_agent.runner.source_snapshot import project_source_snapshot
from true_rl_agent.tasks.stair_climbing.process_report import project_stair_process_report
from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior
from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations

ROOT, BASE, C, R = T.ROOT, T.BASE, T.C, T.R
# Final N4 slots explicitly select the new helper; legacy T.D stays untouched.
import audit_delivery_offline_final_slots as D
SOURCE_ROOT = Path(true_rl_agent.__file__).resolve().parents[2]
QUALIFIED_CASES = {"stair": "stair-01", "push": "push-02"}
HISTORY = BASE / "results/text_author_smoke_r1"


def checked(reference):
    path = Path(reference["path"])
    if not path.is_absolute():
        path = ROOT / path
    N.require(N.digest(path) == reference["sha256"], "source hash differs: " + str(path))
    if "size_bytes" in reference:
        N.require(path.stat().st_size == reference["size_bytes"], "source size differs")
    return path.resolve()


def history_accounting():
    """Verify archived usage against four original receipts, without rebasing them.

    Their PromptPayloadRef paths remain those of the original remote execution.
    Actual sending must re-run production journal validation at that location.
    """
    path = HISTORY / "execution_stage.released.json"
    stage = D.read(path)
    N.require(stage["status"] == "FROZEN_API_RELEASED" and len(stage["episodes"]) == 4,
              "four historical released text episodes are required")
    usage = D.read(HISTORY / "usage.json")
    N.require(not usage["violations"] and usage["attempts"] == 4
              and all(row.get("usage_known") for row in usage["attempts_detail"]),
              "historical usage is not the four known author calls; inspect journals")
    attempts = [D.read(p) for p in sorted((HISTORY / "authoring").rglob("attempt_result.json"))]
    by_slot = {(row["round_id"], row["sequence"]): row for row in attempts}
    N.require(len(attempts) == len(by_slot) == 4, "historical attempt receipts differ from four occupied slots")
    price = stage["budget"]["prices_usd_per_million"]
    for row in usage["attempts_detail"]:
        original = by_slot[(row["round_id"], row["sequence"])]["provider_attempt"]
        units, receipt = original["provider_usage"], original["provider_response_receipt"]
        hit, miss = units["request_cache_hit_units"], units["request_cache_miss_units"]
        N.require(hit + miss == units["request_units"] == row["input_tokens"]
                  and units["response_units"] == row["output_tokens"]
                  and receipt["response_model"] == row["response_model"]
                  and receipt["system_fingerprint"] == row["system_fingerprint"], "historical receipt/usage differs")
        cost = (hit * price["cache_hit"] + miss * price["cache_miss"] + units["response_units"] * price["output"]) / 1e6
        N.require(abs(cost - row["usd_peak"]) < 1e-12, "historical cost derivation differs")
    N.require(abs(sum(row["usd_peak"] for row in usage["attempts_detail"])
                  - usage["usd_peak_actual_or_reserved"]) < 1e-12, "historical total differs")
    refs = [D.ref(path), D.ref(HISTORY / "usage.json")]
    for row in stage["episodes"]:
        terminal = HISTORY / "results" / (row["round_id"] + ".json")
        N.require(D.read(terminal)["case_id"] == row["case_id"], "historical terminal identity differs")
        refs.append(D.ref(terminal))
    refs.extend(D.ref(p) for p in sorted((HISTORY / "authoring").rglob("*.json")))
    return stage, usage, refs


def received_observer(packet, original_builder_ref):
    """Rebuild source checks, then independently audit the unchanged real answer."""
    packet = Path(packet).resolve()
    saved_audit = D.read(packet / "response/mechanical_audit.json")
    N.require(saved_audit.get("schema_version") == SCHEMA_VERSION,
              "observer contract differs; historical audits require their frozen source")
    private = D.read(packet / "private_sidecar.json")
    original_builder_path = checked(original_builder_ref)
    N.require(N.digest(original_builder_path) == D.read(packet / "card.json")["driver_sha256"],
              "original observer preparer differs from the recorded driver")
    original_builder = T.load(original_builder_path, "native_observer_original_preparer")
    built = original_builder.build(Path(private["source_export"]), packet / "selection.json")
    N.require((packet / "request.json").read_bytes() == built["prepared"]["request_bytes"], "observer request drift")
    for name in ("card", "private_sidecar", "mechanical_audit", "alignment"):
        N.require(D.read(packet / (name + ".json")) == built[name], "observer preparation drift: " + name)
    receipt = D.read(packet / "receipt.json")
    raw = D.read(packet / "response/raw_response.json")
    N.require(receipt["status"] == "received" and receipt["scope"] == "old_development_not_n5"
              and receipt["formal_confirmation"] is False
              and receipt["request_sha256"] == built["card"]["request_sha256"]
              and receipt["request"] == built["prepared"]["payload"]
              and receipt["response"] == raw and receipt["fingerprint"] == N.FINGERPRINT,
              "received observer receipt does not bind the sealed development request/answer")
    content = raw["choices"][0]["message"]["content"]
    N.require((packet / "response/raw_answer.txt").read_bytes() == content.encode("utf-8"), "observer answer bytes differ")
    audited = audit_observer_output(json.loads(content), built["windows"], numeric_check=built["numeric_check"])
    N.require(audited == saved_audit, "observer numeric audit differs")
    refs = [D.ref(packet / name) for name in (
        "request.json", "selection.json", "card.json", "private_sidecar.json", "alignment.json",
        "receipt.json", "receipt_journal.jsonl", "response/raw_response.json",
        "response/raw_answer.txt", "response/mechanical_audit.json")]
    export = Path(private["source_export"])
    extraction = D.read(export / "extraction.json")
    refs.append(D.ref(export / "extraction.json"))
    refs.extend(D.ref(export / name) for name in extraction["source_files"])
    refs.extend(D.ref(export / episode["numeric_file"]) for episode in extraction["episodes"])
    # Reconstruct the private identity from the same sealed raw manifest. The
    # historical Stair preparer confused resolved-cell and base-task protocols.
    # Keep its receipt/audit untouched; never spend a new call for private labels.
    current = N.build(export, packet / "selection.json")
    N.require(current["prepared"]["request_bytes"] == built["prepared"]["request_bytes"],
              "rebinding changed what the observer actually received")
    old_identity, new_identity = built["windows"][0].identity, current["windows"][0].identity
    changed = {key: {"old": old_identity.get(key), "new": new_identity.get(key)}
               for key in old_identity.keys() | new_identity.keys()
               if old_identity.get(key) != new_identity.get(key)}
    N.require(set(changed) <= {"protocol_id", "numeric_protocol_id"},
              "unexpected source change while reconstructing private protocol identity")
    corrected_audit = audit_observer_output(json.loads(content), current["windows"],
                                           numeric_check=current["numeric_check"])
    before, after = deepcopy(audited), deepcopy(corrected_audit)
    for value in (before, after):
        for observation in value["observations"]:
            observation.pop("media_identity", None)
    N.require(before == after, "private rebinding changed observation or numeric verdict")
    reconstruction = {"original_preparer": D.ref(original_builder_path),
        "current_preparer": D.ref(Path(N.__file__)), "changed_private_fields": changed,
        "reason": "Stair base task semantics hash is distinct from resolved-cell evaluation binding hash",
        "original_audit": D.ref(packet / "response/mechanical_audit.json"),
        "request_bytes_identical": True, "original_receipt_unchanged": True,
        "raw_answer_unchanged": True, "new_api_calls": 0,
        "numeric_checks_recomputed": True, "is_new_observation": False}
    refs.extend([D.ref(original_builder_path), D.ref(Path(N.__file__))])
    return current, corrected_audit, receipt, refs, reconstruction


def make_material(task, inputs, out):
    phase_path, plan_path, full_path = [checked(inputs[key]) for key in ("phase", "plan", "full")]
    phase, full = D.read(phase_path), D.read(full_path)
    original_refs = [D.ref(path) for path in (phase_path, plan_path, full_path)]
    bound = phase["task_evidence"]["behavior_full"]
    N.require(bound["sha256"] == N.digest(full_path) and bound["size_bytes"] == full_path.stat().st_size,
              "original phase/full binding differs")
    project, attach = ((project_stair_process_report, attach_stair_visual_observations) if task == "stair"
                       else (project_push_process_behavior, attach_push_visual_observations))
    N.require(project(full) == phase["task_evidence"]["behavior"], "original public report differs from full")
    built, audit, receipt, observer_refs, reconstruction = received_observer(
        inputs["observer_dir"], inputs["observer_preparer"])
    N.require(built["card"]["task"] == task, "observer task differs")
    summary = D.read(Path(built["private_sidecar"]["source_export"]) / "source_summary.json")
    N.require(summary["job_id"] == phase["job_id"]
              and summary["task_evidence"] == phase["task_evidence"], "observer and full belong to different evaluation runs")
    augmented = attach(full, audit, observer_version=SCHEMA_VERSION + ".native_development." + N.FINGERPRINT)
    N.require(not augmented["observer_source_binding"]["rejected_observation_indices"], "task rejected observer source binding")
    public = project(augmented)
    full_rows = D.sentences(task, augmented)
    shown = next((row for row in full_rows if row["id"] == "visual.observation"), None)
    N.require(shown is not None and any(row["id"] == shown["id"] for row in public["report"]["visual"]),
              "no mechanically usable real visual sentence; preserve observer output without spending an author slot")
    N.require(not any(row.get("kind") == "S4" for row in full_rows), "this replay adapter has no verified parent-child lineage")
    out.mkdir(parents=True, exist_ok=False)
    D.write(out / "observer_identity_reconstruction.json", reconstruction)
    D.write(out / "observer_rebound_audit.json", audit)
    for name, path in (("source_phase.json", phase_path), ("source_full.json", full_path), ("source_plan.json", plan_path)):
        (out / name).write_bytes(path.read_bytes())
    full_ref = D.write(out / "behavior_full.json", augmented)
    derived = deepcopy(phase)
    derived["task_evidence"]["behavior"] = public
    derived["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": full_ref["sha256"], "size_bytes": full_ref["size_bytes"]}
    derived["task_evidence_hash"] = D.canonical_json_hash(derived["task_evidence"])
    phase_ref = D.write(out / "public_dev_result.json", derived)
    D.write(out / "public_report.json", public)
    provenance = {"original_phase_full_plan": original_refs, "observer_sources": observer_refs,
        "method": "task attach of unchanged real observations to their own full, then deterministic public projection",
        "observer_identity_reconstruction": reconstruction,
        "original_sources_unchanged": True, "derived_phase_is_not_new_qualification": True,
        "numerical_episodes_unchanged": augmented["episodes" if task == "stair" else "transcripts"]
            == full["episodes" if task == "stair" else "transcripts"],
        "exposure": {"real_visual": "EXERCISED_IN_PREPARED_INPUT",
            "visual_only": "EXERCISED_IN_PREPARED_INPUT" if shown["evidence_category"] == "visual_only" else "NOT_EXERCISED",
            "conflict": "EXERCISED_IN_PREPARED_INPUT" if shown["evidence_category"] == "conflict" else "NOT_EXERCISED",
            "S4": "NOT_EXERCISED"}, "actual_author_reading": "NOT_RUN",
        "alignment_not_checked": built["alignment"]["not_checked"],
        "as_of_cutoff": receipt["recorded_at_utc"], "derived_phase": phase_ref}
    N.require(provenance["numerical_episodes_unchanged"], "visual attachment changed numerical episodes")
    D.write(out / "provenance.json", provenance)
    return augmented, public, provenance


def context_for(task, material, provenance, out):
    case = deepcopy(T.four_cases()[QUALIFIED_CASES[task]])
    spec_path = ROOT / f"PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_{task}.json"
    N.require(N.digest(spec_path) == case["author_context"]["sha256"], "qualified source context differs")
    spec = D.read(spec_path)
    case["as_of_cutoff"] = provenance["as_of_cutoff"]
    exp = C._p3(ROOT)
    exp.bound_read, _ = D.bound_reader()
    extra = {} if task == "stair" else {
        "diagnostic_eval_base_seed": spec["composition_values"]["eval_base_seed"],
        "source_manifest_ref": D.ref(D.ORIGINALS / "push/source_manifest.json"),
        "training_summary_ref": D.ref(D.ORIGINALS / "push/training_summary.json")}
    return C.build_context(case, "T", spec, root=ROOT, base=ROOT, output=out,
        diagnostic_phase_ref=D.ref(material / "public_dev_result.json"),
        diagnostic_plan_ref=D.ref(material / "source_plan.json"),
        allowed_seed_cohort_hash=spec["validated_evaluation_binding"]["seed_cohort_hash"],
        development_fixture=True, exp=exp, **extra)


def delivery(task, full, public, comp, builder, out):
    """Use production serialization; record true observer references, not synthetic origins."""
    out.mkdir(parents=True, exist_ok=False)
    def build_prompt(*args, **kwargs):
        kwargs.setdefault("authoring_capability_readiness", comp.authoring_capability_readiness.to_prompt_mapping())
        kwargs.setdefault("candidate_verifier_prompt_rules", comp.candidate_admission_policy.authoring_prompt_rules())
        return builder.build_prompt(*args, **kwargs)
    frames = D.make_frames(SimpleNamespace(build_prompt=build_prompt), comp, out)
    first = R.first_prompt(comp, builder)
    frames["draft"] = first
    source, source_hash, role = select_authoring_anchor_source(first)
    N.require(source and role == "current_best", "actual qualified authoring anchor missing")
    full_rows = {row["id"]: row for row in D.sentences(task, full)}
    full_bytes, public_bytes = D.canonical_json_bytes(full), D.canonical_json_bytes(public)
    source_receipt = D.ref(out.parents[1] / "materials" / task / "provenance.json")
    core = [{"section": section, **row} for section in D.SECTIONS for row in public["report"][section]]
    N.require({key for key, row in full_rows.items() if row.get("required", True)} <= {row["id"] for row in core},
              "required full sentence lost in bounded report")
    ledger, receipts = [], []
    for frame, payload in frames.items():
        body, user, delivered = D.capture_request(payload, out)
        N.require(D.behavior_in(delivered, task) == public, "behavior changed in serialized request")
        budget = delivered["authoring_context_frame"]["budget_metadata"]
        N.require(budget["actual_prompt_payload_context_units"] <= 40000, "author input exceeded existing reservation")
        target = out / frame
        target.mkdir()
        (target / "request.http_body.json").write_bytes(body)
        (target / "user_payload.json").write_bytes(user)
        receipts.append({"frame": frame, "request": D.ref(target / "request.http_body.json"),
            "payload": D.ref(target / "user_payload.json"), "budget": budget})
        for row in core:
            original = full_rows[row["id"]]
            N.require(original["text"] == row["text"], "full/public sentence drift")
            refs = []
            for reference in original.get("evidence_refs", []):
                # Push stores observation entries in its catalog, but nested image
                # refs are actual full JSON paths; do not invent catalog entries.
                resolved = (D.resolve_stair(full, reference)
                            if reference == "observer_evidence" or reference.startswith("observer_evidence/")
                            else D.resolve_refs(task, full, [reference])[0]["resolved"])
                N.require(resolved, "unresolved full evidence reference")
                refs.extend(resolved)
            fragment = D.string_fragment(row["text"])
            locations = {"full": D.byte_evidence(full_bytes, fragment),
                "public": D.byte_evidence(public_bytes, fragment),
                "payload": D.byte_evidence(user, fragment),
                "request": D.byte_evidence(body, D.string_fragment(fragment.decode("utf-8")))}
            N.require(all(item["present"] for item in locations.values()), "report sentence lost before transport")
            ledger.append({"frame": frame, "id": row["id"], "section": row["section"],
                "origin": "real_observer_and_same_run_numeric_check" if row["section"] == "visual" else "task_numeric_or_contract_derivation",
                "raw": {"source_receipts": source_receipt,
                    "numeric_manifest_sha256s": ([full["domain"]["archive_hash"]] if task == "stair" else
                        [row["manifest_sha256"] for row in full["sources"]]),
                    "raw_observer_and_png_hashes": "bound original answer and image_evidence in observer_evidence"
                        if row["section"] == "visual" else "not an observer claim",
                    "original_numeric_chunks_reopened_here": False,
                    "independent_semantic_support": "NOT_ESTABLISHED"},
                "full_evidence_refs": original.get("evidence_refs", []), "resolved": refs, "locations": locations})
        action = full["semantic"]["actionability"] if task == "stair" else full["actionability"]
        N.require(public["report"]["actionability"] == action, "actionability changed")
    D.write(out / "sentence_delivery_ledger.json", ledger)
    D.write(out / "receipt.json", {"status": "PASS_OFFLINE_DELIVERY", "frames": receipts,
        "lost_required_sentences": 0, "network_calls": 0, "branches_are_offline_fixtures": True,
        "first_request_is_not_actual_final_seen_input": True,
        "after_send_truth": "authoritative final RoundAuthoringJournal PromptPayloadRef and serialized provider request"})
    return receipts[0], source_hash


def prepare(inputs_path, output):
    output = output.resolve()
    N.require(output.is_relative_to(BASE) and not output.exists(), "use a fresh directory under P3NEW")
    inputs = D.read(inputs_path)
    N.require(inputs.get("schema_version") == "p3new.native_author_inputs.v1"
              and inputs.get("scope") == "old_development_not_n5"
              and isinstance(inputs.get("purpose"), str) and inputs["purpose"].strip()
              and set(inputs["tasks"]) == {"stair", "push"}, "two-task old-development inputs required")
    old_stage, old_usage, frozen = history_accounting()
    frozen.append(D.ref(inputs_path))
    # Freeze the existing qualification graph as well as the new diagnostic.
    # Reading these refs never creates or refreshes a qualification receipt.
    _, qualification_refs = D.bound_reader()
    frozen.extend(qualification_refs)
    r0b_path = ROOT / "PLAN/feedback_sys/P3R/r0b_inputs_local.json"
    r0b = D.read(r0b_path)
    frozen.extend(D.ref(path) for path in (
        r0b_path, r0b_path.parent / r0b["p3_cases"]["path"], D.ORIGINALS / "source_index.json",
        ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_stair.json",
        ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_push.json",
        ROOT / "PLAN/feedback_sys/P3/author_experiment.py"))
    episodes = []
    for task, lane in (("stair", "lane-a"), ("push", "lane-b")):
        material = output / "materials" / task
        full, public, provenance = make_material(task, inputs["tasks"][task], material)
        comp, builder, _ = context_for(task, material, provenance, output / "contexts" / task)
        first, source_hash = delivery(task, full, public, comp, builder, output / "prompts" / task)
        generation = "p3new-text-smoke-20260929-generation-003"
        episodes.append({"case_id": task + "-native-dev", "task": task, "generation_id": generation,
            "lane_id": lane, "round_id": generation + "-" + lane,
            "qualified_identity_source_case": QUALIFIED_CASES[task], "not_a_repeat_of_that_text_case": True,
            "qualified_parent_source_sha256": source_hash,
            "material": D.ref(material / "provenance.json"), "initial_http_body": first["request"],
            "initial_payload": first["payload"], "exposure": provenance["exposure"],
            "review_status": "NOT_RUN", "actual_final_seen_input": None})
        frozen.extend(provenance["original_phase_full_plan"] + provenance["observer_sources"])
    snapshot = D.write(output / "production_source_snapshot.json", project_source_snapshot(SOURCE_ROOT))
    frozen.extend(D.ref(p) for p in sorted(output.rglob("*")) if p.is_file())
    frozen.extend(D.ref(path) for path in (Path(__file__), Path(N.__file__), Path(T.__file__),
        Path(D.__file__), Path(C.__file__), Path(R.__file__)))
    prices = old_stage["budget"]["prices_usd_per_million"]
    worst = (40000 * prices["cache_miss"] + 65536 * prices["output"]) / 1e6
    cap = old_stage["budget"]["stage_usd_cap"]
    N.require(cap == 4.0 and old_usage["usd_peak_actual_or_reserved"] + 2 * worst <= cap,
              "historical author allocation cannot cover two first requests")
    card = {"schema_version": "p3new.native_author_smoke.v1", "status": "PREPARED_NOT_RELEASED",
        "observer_schema_version": SCHEMA_VERSION,
        "scope": inputs["scope"], "purpose": inputs["purpose"], "episodes": episodes,
        "episode_budget": {"historical_text": 4, "new_visual": 2, "total_if_run": 6, "limit": 8, "unused": 2},
        "provider": deepcopy(old_stage["provider"]), "pricing_status": "REVERIFY_BEFORE_RELEASE",
        "budget": {"stage_usd_cap": cap, "historical_actual_calls": old_usage["attempts"],
            "historical_usd_ceiling": old_usage["usd_peak_actual_or_reserved"], "actual_invoice_usd": None,
            "historical_price_basis": prices, "new_actual_calls_cap": 2, "max_attempts_per_new_episode": 1,
            "author_total_actual_calls_if_both_sent": 6, "overall_n4_actual_calls_limit": 40,
            "input_reserve": 40000, "max_output_tokens": 65536,
            "historical_price_two_request_reservation_usd": 2 * worst,
            "remaining_usd_before_new_calls": cap - old_usage["usd_peak_actual_or_reserved"],
            "production_provider_caps": old_stage["budget"]["production_provider_caps"],
            "shared_journal_directory": str(HISTORY), "max_generations": 3,
            "no_retry_or_replacement_episode": True},
        "runner": {"entry": "existing CandidateAuthoringLoop + CostGuardedProvider + RoundAuthoringJournal",
            "reuse_shared_generation_prefix": "p3new-text-smoke-20260929",
            "construction": "max_attempts=1; retain old lane caps for historical accounting; no second provider call",
            "on_source_request_or_invalid_candidate": "retain terminal; do not service a followup or replace the episode",
            "not_implemented_here": "API release/execution; this script only constructs evidence and serialized inputs"},
        "release_remaining": ["Review same-run pixel readability, mechanical alignment, and genuine observer readback.",
            "Record the explicit N4 plan clarification for 4 historical text + 2 same-source development visual episodes.",
            "Recheck provider model/pricing, original shared journals and remaining US$4; freeze source/input hashes.",
            "Use the existing author loop once per episode; preserve final actual request and every terminal."],
        "review_contract": T.review_contract(), "production_source_snapshot": snapshot,
        "material_repository_root": str(ROOT), "production_source_root": str(SOURCE_ROOT),
        "historical_accounting_check": "archived receipt/usage consistency; live original journal validation remains required before sending",
        "frozen_inputs": frozen, "api_calls_during_preparation": 0, "gpu_work": "NOT_RUN",
        "AT_value_gate": "NOT_APPLICABLE", "semantic_acceptance": "NOT_ESTABLISHED"}
    card["review_contract"]["exposure_limits"] = {row["case_id"]: row["exposure"] for row in episodes}
    D.write(output / "stage.draft.json", card)
    for reference in frozen:
        checked(reference)
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card = prepare(args.inputs.resolve(), args.output.resolve())
    print(json.dumps({"status": card["status"], "episode_budget": card["episode_budget"],
                      "api_calls": 0, "gpu_work": "NOT_RUN"}))

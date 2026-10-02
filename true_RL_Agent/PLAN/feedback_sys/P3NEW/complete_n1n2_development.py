"""Refresh metadata and audit three fixed historical cases; no API, GPU or L0 re-run."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.push_recovery.process_report import (
    build_push_process_behavior, project_push_process_behavior,
)

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "PLAN/feedback_sys/P3NEW"
OUT = BASE / "results/n1n2_completion_r1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(name, value, *, canonical=False):
    path = OUT / name
    path.write_bytes(canonical_json_bytes(value) if canonical else
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return ref(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    delivery = BASE / "results/delivery_offline_r1"
    old_push_path = delivery / "push/behavior_full.json"
    old_push = read(old_push_path)
    measured_path = delivery / "push/measured_v2.json"
    phase_path = delivery / "push/public_dev_result.json"
    phase = read(phase_path)
    refreshed = build_push_process_behavior(legacy_full=read(measured_path),
        seed_summaries=phase["task_evidence"]["seed_summaries"])
    assert {k: v for k, v in old_push.items() if k != "lexicon"} == {
        k: v for k, v in refreshed.items() if k != "lexicon"}, "non-lexicon derivation changed"
    changed = []
    for before, after in zip(old_push["lexicon"], refreshed["lexicon"], strict=True):
        clean = deepcopy(after)
        if "units" not in before["measurable_definition"]:
            assert clean["measurable_definition"].pop("units")
            changed.append(before["id"])
        assert before == clean, "more than units changed"
    assert len(changed) == 9
    prior_public, refreshed_public = map(project_push_process_behavior, (old_push, refreshed))
    assert {k: v for k, v in prior_public.items() if k != "full_derivation_sha256"} == {
        k: v for k, v in refreshed_public.items() if k != "full_derivation_sha256"}
    refreshed_ref = write("push_full_units_current.json", refreshed, canonical=True)
    assert refreshed_ref["sha256"] == refreshed_public["full_derivation_sha256"]
    cards_ref = write("push_term_cards.json", refreshed["lexicon"])
    public_ref = write("push_public_units_current.json", refreshed_public)

    v1_path = ROOT / "PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/reviewer/evidence-stair.json"
    current_path = delivery / "stair/behavior_full.json"
    raw_oracle_path = BASE / "results/stair_raw_development.json"
    v1, current, oracle = read(v1_path), read(current_path), read(raw_oracle_path)
    assert {k: v for k, v in v1["domain"].items() if k != "rule_version"} == {
        k: v for k, v in current["domain"].items() if k != "rule_version"}
    fixture_path = BASE / "stair_retrospective_cases.json"
    fixture = read(fixture_path)
    cases, candidates = [], []
    for case in fixture["cases"]:
        before = next(row for row in v1["episodes"] if row["episode_id"] == case["episode_id"])
        after = next(row for row in current["episodes"] if row["episode_id"] == case["episode_id"])
        for key in ("run", "slot", "cell", "primary_reason", "steps"):
            assert before[key] == after[key]
        residual = next(row for row in current["semantic"]["residuals"]["episodes"] if row["episode_id"] == case["episode_id"])
        independent = next(row for row in oracle["residuals"] if row["episode_id"] == case["episode_id"])
        assert residual["status"] == independent["state"]
        assert residual["success_n"] == independent["matched_success_n"]
        assert residual["matching_success_n"] == independent["matching_signature_n"]
        recordings_path = ROOT / case["recordings"]
        recordings = read(recordings_path)
        media = recordings["recordings"][0]
        assert media["file"] == "public_dev_seed_1.mp4" and media["env_index"] == case["slot"]
        assert recordings["execution"]["checkpoint_hash"] == current["domain"]["checkpoint_hash"]
        outcome = media["episode_outcome"]
        assert (outcome["episode_steps"], outcome["primary_reason"], outcome["bin_id"]) == (
            after["steps"], after["primary_reason"], after["cell"])
        assert max(case["shown_steps"]) <= media["frame_count"]
        replay = {"recordings": ref(recordings_path), "image": ref(ROOT / case["image"]),
            "recording": media, "numeric_relation": "separate_replay_not_synchronized",
            "same_trajectory_verified": False, "formal_visual_confirmation": False}
        relevant = {"tilt", "low_progress", "unloaded_low_motion", "no_instant_support", "forbidden_manager"}
        def snapshot(episode):
            return {"before_terminal": episode["before_terminal"],
                "confirmed_relevant_events": [event for event in episode["events"]
                    if event["code"] in relevant and event["confirmed"] is not None],
                "root_motion": episode.get("root_motion"), "root_motion_missing": "root_motion" not in episode}
        result = {**case, "numeric_domain": current["domain"],
            "numeric_identity": {key: after[key] for key in ("episode_id", "run", "slot", "cell", "primary_reason", "steps")},
            "v1": {"rule_version": v1["domain"]["rule_version"], **snapshot(before),
                "existing_before_terminal_predicate_hit": any(before["before_terminal"].values()),
                "n2_discrimination_state": "not_defined_in_v1",
                "named_visual_shape_rule": "not_defined"},
            "current": {"lexicon_version": current["semantic"]["residuals"]["lexicon_version"],
                **snapshot(after), "n2_residual": residual, "independent_oracle": independent,
                "named_visual_shape_rule": "not_defined"}, "replay": replay,
            "visual_numeric_alignment": "unverified; numeric changes cannot certify this replay's individual frames"}
        cases.append(result)
        candidates.append({"id": case["case_id"], "task": "stair_climbing", "status": "visual_candidate",
            "original_names": case["original_names"], "normalized_name": case["normalized_name"],
            "body": "visible robot trunk and limbs; numeric root/foot are not assumed identical",
            "shown_steps": case["observation_steps"], "observation": case["visible_description"],
            "not_established": case["not_established"], "sources": [replay], "occurrence_count": 1,
            "observer_version": "historical_development_review.20260929",
            "numeric_sources": {"v1": ref(v1_path), "current": ref(current_path),
                "episode_id": case["episode_id"], "oracle": ref(raw_oracle_path),
                "relation_to_media": "separate_replay_not_synchronized"},
            "count_basis": "one existing recording and sampled window; aliases and repeated readings are not new occurrences",
            "merge_records": [{"actor": fixture["visual_review"]["reviewer"], "operation": "scope_narrowing",
                "from": case["original_names"], "to": case["normalized_name"],
                "reason": "preserve historical names, restrict displayed claim to supported sampled shape; no assertion of physical equivalence",
                "original_description_source": ref(ROOT / fixture["historical_description_source"])}],
            "known_numeric_relation": case["known_numeric_relation"],
            "numeric_residual_state": residual["status"], "named_shape_measurability": "not_established",
            "promotion": {"status": "not_promoted", "decision": case["promotion_disposition"], "automatic": False}})
    retrospective_ref = write("stair_retrospective.json", {"schema_version": "feedback.p3new.stair_retrospective.v1",
        "purpose": "historical development exercise, not N5 or same-run visual confirmation",
        "input_refs": [ref(p) for p in (v1_path, current_path, raw_oracle_path, fixture_path)],
        "visual_review": fixture["visual_review"], "cases": cases})
    ledger_ref = write("candidate_lexicon.json", {"schema_version": "feedback.candidate_lexicon.dev.v1",
        "automatic_promotion": False, "promoted_count": 0, "tasks": {
            "stair_climbing": {"entries": candidates},
            "push_recovery": {"entries": [], "reason": "No new Push candidate established in this fixed Stair retrospective; not a claim of exhaustive semantic coverage"}}})
    result = {"status": "PASS_BOUNDED_DEVELOPMENT_REPAIR", "not_phase_acceptance": True,
        "source_refs": [ref(ROOT / "src/true_rl_agent/tasks/push_recovery/process_report.py"), ref(Path(__file__))],
        "push": {"changed_units_only": changed, "public_text_and_metrics_unchanged": True,
            "public_changed_fields": ["full_derivation_sha256"],
            "all_non_lexicon_fields_unchanged": True, "prior_full": ref(old_push_path),
            "measured_v2": ref(measured_path), "prior_phase": ref(phase_path),
            "refreshed_full": refreshed_ref, "cards": cards_ref, "public": public_ref, "L0_recomputed": False},
        "retrospective": retrospective_ref, "candidate_ledger": ledger_ref,
        "semantic_limits": "All three visual names remain unpromoted; v1 has no N2 classifier; old media not synchronized",
        "API_calls": 0, "GPU_hours": 0, "historical_results_overwritten": False}
    write("verification.json", result)
    print(json.dumps({"units_fixed": len(changed), "historical_cases": len(cases),
        "promoted": 0, "public_text_and_metrics_unchanged": True}))


if __name__ == "__main__":
    main()

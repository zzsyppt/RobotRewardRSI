"""Offline current-template rebuild from unchanged old development v2 data.

No provider, simulator, SSH or new confirmation data. Existing files are read
only; every output is created exclusively in this script's new result folder.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parent / "delivery_offline_r1/push"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.evaluation.visual_observer import (
    MAX_OBSERVATION_CHARS, MAX_UNCERTAINTY_CHARS, audit_observer_output,
)
from true_rl_agent.tasks.push_recovery.process_report import (
    build_push_process_behavior, compare_push_process_behavior, project_push_process_behavior,
)
from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations
from tests.unit.test_task_visual_observation import IDENTITY, observation, window
from tests.unit.test_visual_review import synthetic_review


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw = path.read_bytes()
    return {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def save(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))
    return ref(path)


def main():
    baseline_path = HERE.parent / "n1n2_completion_r1/push_full_units_current.json"
    watched = [OLD / name for name in ("behavior_full.json", "measured_v2.json", "public_dev_result.json", "raw_sources.json")]
    watched.append(baseline_path)
    for frozen in ("diagnostic_module_probe_r10/source", "body_grounding_observer_r11/control/source"):
        path = HERE.parent / frozen / "src/true_rl_agent/tasks/push_recovery/process_report.py"
        assert path.is_file()
        watched.append(path)
    before = [ref(path) for path in watched]
    old, v2, phase = (read(OLD / name) for name in ("behavior_full.json", "measured_v2.json", "public_dev_result.json"))
    assert canonical_json_hash(v2) == old["source_derivation_hash"]
    full = build_push_process_behavior(legacy_full=v2, seed_summaries=phase["task_evidence"]["seed_summaries"])
    baseline = read(baseline_path)
    assert {k: v for k, v in baseline.items() if k != "sentences"} == {
        k: v for k, v in full.items() if k != "sentences"}
    assert full["transcripts"] == v2["transcripts"] == old["transcripts"]
    assert full["source_bindings"] == old["source_bindings"]
    public = project_push_process_behavior(full)
    assert public["report_projection"]["omitted_statement_ids"] == []
    assert report_bytes(public) <= 12288
    assert public["full_derivation_sha256"] == canonical_json_hash(full)
    old_public = phase["task_evidence"]["behavior"]
    for section in ("results", "counterexample", "visual", "parent_diff", "actionability"):
        assert public["report"][section] == old_public["report"][section]
    assert public["report"]["process"][1:] == old_public["report"]["process"][1:]
    assert public["report"]["gaps"][:-1] == old_public["report"]["gaps"][:-1]
    example = next(row for row in full["sentences"] if row["id"] == "push.process.example")
    assert example["required"]
    row = full["transcripts"][51]
    assert (row["source"], row["env_id"]) == (0, 51)
    first, second = row["recoveries"]
    assert first["recovery_control_step"] is None and second["recovery_control_step"] == 324
    post = second["process"]["post_latch"]
    assert [(span["start_step"], span["end_step_inclusive"]) for span in post["unstable_spans"]] == [(366, 374), (385, 412)]
    assert post["unstable_steps"] == 37 and post["observed_through_step"] == 412
    assert row["final_failed_conditions"] == ["planar_speed"]
    assert row["final_pose"]["planar_speed_mps"] > full["definitions"]["stable_bounds"]["planar_speed_max_mps"]
    for part in ("10/14 compatible episodes", "This episode's sequence only:", "event_0:", "latch not observed",
                 "event_1:", "latch 324", "37 observed steps in 2 interval(s)", "first 366-374", "last 385-412",
                 "last interval reaches observation cutoff", "Detail event_0:", "outcome not_recovered"):
        assert part in example["text"], part
    assert "Detail event_1:" not in example["text"]

    # Reuse the ordinary audit/review/attachment and whole-statement capacity path.
    source, episode = full["source_bindings"][0], full["transcripts"][0]
    identity = {**IDENTITY, "source_id": source["manifest_sha256"], "checkpoint_id": full["binding"]["checkpoint_hash"],
        "protocol_id": full["binding"]["resolved_protocol_hash"], "run_id": str(source["actual_run_index"]),
        "episode_id": f"r{source['actual_run_index']}.e{episode['env_id']}"}
    steps = [1 + (episode["final_control_step"] - 1) * i // 5 for i in range(6)]
    claim = "root_small_xy_endpoint_delta"
    media = window(steps, ({"id": claim, "body": "root", "definition": "Synthetic capacity only."},),
                   identity=identity, views=("v0", "v1"))
    raw = {"episodes": [{"alias": "e1", "not_visible": [], "observations": [
        {**observation(claim, steps=steps, views=("v0", "v1")),
         "text": "\U00020000" * MAX_OBSERVATION_CHARS, "uncertainty": "\U00020001" * MAX_UNCERTAINTY_CHARS}
        for _ in range(3)]}]}
    audit = synthetic_review(audit_observer_output(raw, [media], numeric_check=lambda *_: {"status": "conflict"}))
    augmented = attach_push_visual_observations(full, audit, observer_version="synthetic_capacity_only")
    assert not augmented["observer_source_binding"]["rejected_observation_indices"]
    assert augmented["transcripts"] == full["transcripts"]
    visual = project_push_process_behavior(augmented)
    parent = deepcopy(public)
    parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
    counts = parent["comparison_basis"]["terminal_counts"]
    counts["success_recovered"] -= 1
    counts[next(k for k in counts if k != "success_recovered")] += 1
    difference = compare_push_process_behavior(public, parent, expected_parent_checkpoint_hash=parent["binding"]["checkpoint_hash"])
    assert difference["status"] == "available"
    combined_input = deepcopy(visual)
    combined_input["parent_diff"] = difference
    combined = project_push_process_behavior(combined_input)
    parent_first = deepcopy(augmented)
    parent_first["parent_diff"] = difference
    assert project_push_process_behavior(parent_first)["report"] == combined["report"]
    assert project_push_process_behavior(combined) == combined
    assert combined["report_projection"]["omitted_statement_ids"] == ["visual.observation"]
    for section in ("results", "process", "counterexample", "actionability"):
        assert combined["report"][section] == public["report"][section]
    assert combined["report"]["gaps"][:-1] == public["report"]["gaps"]
    coverage = combined["report"]["visual"][0]["text"]
    assert "0 shown" in coverage and "3 numeric conflicts" in coverage
    assert report_bytes(combined) <= 12288
    assert combined["full_derivation_sha256"] == canonical_json_hash(augmented)

    full_ref = save("behavior_full.json", full)
    assert full_ref["sha256"] == public["full_derivation_sha256"]
    public_ref = save("public_report.json", public)
    phase["task_evidence"]["behavior"] = public
    phase["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": full_ref["sha256"], "size_bytes": full_ref["size_bytes"]}
    phase_ref = save("public_dev_result.json", phase)
    augmented_ref = save("pressure/behavior_full.json", augmented)
    pressure_ref = save("pressure/public_report_with_synthetic_parent.json", combined)
    save("pressure/synthetic_observer_audit.json", audit)
    save("pressure/synthetic_parent_fixture.json", parent)
    assert before == [ref(path) for path in watched]
    result = {"status": "PASS_OFFLINE_PROJECTION_ONLY", "actual_model_calls": 0, "gpu_runs": 0, "ssh_calls": 0,
        "new_rollouts": 0, "L0_to_v2_recomputed": False, "actual_author_delivery": False,
        "method": "Current production build_push_process_behavior from hash-bound old measured_v2 and local seed manifests, then project_push_process_behavior. Not a new rollout or author request.",
        "old_sources_and_frozen_templates_unchanged": before,
        "source_derivation_hash": full["source_derivation_hash"], "transcripts_canonical_sha256": canonical_json_hash(full["transcripts"]),
        "all_numeric_fields_unchanged": True, "source_bindings_unchanged": True,
        "changed_fields_vs_prior_units_refreshed_full": ["sentences"],
        "note_vs_delivery_r1": "Current builder also retains the previously completed units-only lexicon metadata refresh, identical to n1n2_completion_r1; this change did not introduce it.",
        "full": full_ref, "public": public_ref, "phase": phase_ref, "full_public_phase_hash_binding_verified": True,
        "example": {"source_pointer": "/transcripts/51", "source": 0, "env_id": 51, "text": example["text"],
            "event_0_latch": first["recovery_control_step"], "event_1_latch": second["recovery_control_step"],
            "event_1_post_latch": post, "final_failed_conditions": row["final_failed_conditions"],
            "final_planar_speed_mps": row["final_pose"]["planar_speed_mps"],
            "planar_speed_bound_mps": full["definitions"]["stable_bounds"]["planar_speed_max_mps"]},
        "capacity": {"limit_bytes": 12288, "plain_bytes": report_bytes(public), "visual_bytes": report_bytes(visual),
            "visual_and_parent_bytes": report_bytes(combined), "omitted": combined["report_projection"]["omitted_statement_ids"],
            "numeric_counterexample_residual_and_outline_preserved": True, "both_parent_attachment_orders_and_replay_pass": True,
            "unicode_utf8_bytes_per_character": 4, "observation_characters": MAX_OBSERVATION_CHARS,
            "uncertainty_characters": MAX_UNCERTAINTY_CHARS, "raw_observation_count": 3,
            "cited_steps": 6, "views": 2, "image_references_per_observation": 12,
            "synthetic_only_not_semantic_qualification": True, "full": augmented_ref, "public": pressure_ref},
        "code": [ref(ROOT / name) for name in ("src/true_rl_agent/tasks/push_recovery/process_report.py",
            "src/true_rl_agent/tasks/push_recovery/behavior.py", "src/true_rl_agent/evaluation/process_report.py",
            "src/true_rl_agent/evaluation/visual_observer.py", "src/true_rl_agent/tasks/push_recovery/visual_observation.py",
            "tests/unit/test_push_process_report.py")], "verification_script": ref(Path(__file__))}
    save("verification.json", result)
    print(json.dumps({"status": result["status"], "full_sha256": full_ref["sha256"], "public_sha256": public_ref["sha256"],
        "plain_bytes": report_bytes(public), "combined_bytes": report_bytes(combined), "omitted": result["capacity"]["omitted"]}))


if __name__ == "__main__":
    main()

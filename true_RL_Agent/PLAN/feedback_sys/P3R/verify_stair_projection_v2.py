"""CPU-only regression on the frozen 864-episode Stair development archive."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig,
)
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.authoring_frame import build_authoring_context_frame, with_final_prompt_budget
from true_rl_agent.prompting.payload import PROMPT_PAYLOAD_SECTION_ORDER
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
from true_rl_agent.tasks.stair_climbing.behavior import MAX_PUBLIC_BYTES, compare_stair_behavior, project_stair_behavior

HERE = Path(__file__).resolve().parent
FULL_PATH = HERE.parent / "P3/results/author_final_bundle/author_blind_review/reviewer/evidence-stair.json"
OLD_PATH = HERE / "runtime_probe_r1/prompts/stair-01_T.json"
SOURCE_PATH = ROOT / "src/true_rl_agent/tasks/stair_climbing/behavior.py"
VIEW_PATH = HERE / "results/stair_projection_v2_real_view.json"
REPORT_PATH = HERE / "results/stair_projection_v2_real_validation.json"
ALIASES = {"loaded_point_motion_any": "slip_any", "loaded_point_motion_body": "slip_body"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    before = {path: digest(path) for path in (FULL_PATH, OLD_PATH, SOURCE_PATH)}
    assert before[FULL_PATH] == "50e0c0e19415d5fab68bcbf0e28272a5a3782d7dade51781cc4c4f4ca813eb89"
    assert before[OLD_PATH] == "95bf3950f3a93aa86d6d85bb79098ed8224b7ce416fd5423496f5056feef2a13"
    full, prompt = read(FULL_PATH), read(OLD_PATH)
    old = prompt["public_feedback_summary"]["behavior"]["profile"]
    assert project_stair_behavior(old) == old
    view = project_stair_behavior(full)
    assert project_stair_behavior(view) == view
    assert old["domain"] == view["domain"] == full["domain"]
    assert canonical_prompt_json_bytes(view) <= MAX_PUBLIC_BYTES
    episodes = full["episodes"]
    assert len(episodes) == view["profile"]["episode_count"] == 864
    by_stratum = defaultdict(list)
    for episode in episodes:
        by_stratum[episode["stratum"]].append(episode)
    totals = view["profile"]["stratum_totals"]
    assert len(totals["rows"]) == len(by_stratum) == 8
    for values in totals["rows"]:
        row = dict(zip(totals["columns"], values, strict=True))
        group = by_stratum[row["stratum"]]
        assert row["n"] == len(group)
        assert row["success"] == sum(e["primary_reason"] == "success" for e in group)
        assert row["root_anchor_crossed"] == sum(e["anchor"]["status"] == "crossed" for e in group)
        for code, count in row.items():
            if code in {"stratum", "n", "success", "root_anchor_crossed"}:
                continue
            window = code.startswith("window_")
            field = code.removeprefix("window_")
            field = ALIASES.get(field, field)
            assert count == sum(e["window_present" if window else "before_terminal"][field] for e in group)
    for row in view["contrasts"]["rows"]:
        for side, reason in (("failed", row["reason"]), ("success", "success")):
            group = [e for e in episodes if (e["cell"], e["run"], e["primary_reason"])
                     == (row["cell"], row["run_ordinal"], reason)]
            assert row[side]["n"] == len(group)
            for code, count in row[side]["confirmed"].items():
                assert count == sum(e["before_terminal"][ALIASES.get(code, code)] for e in group)
    by_id = {e["episode_id"]: e for e in episodes}
    transcripts = {e["episode_id"]: e for e in view["transcripts"]}
    assert set(transcripts) == {"r1.e71", "r1.e60"}
    required = {"r1.e71": {("low_progress", 110, 141, 110), ("forbidden_manager", 407, 407, 407)},
                "r1.e60": {("low_progress", 105, 131, 105), ("tilt", 423, 436, 432)}}
    for identity, transcript in transcripts.items():
        emitted = {(e["code"], e["start"], e["end"], e["confirmed"]) for e in transcript["events"]}
        assert required[identity] <= emitted, f"lost root process evidence: {identity}"
        for event in transcript["events"]:
            original = {**event, "code": ALIASES.get(event["code"], event["code"])}
            assert original in by_id[identity]["events"]
            if event["code"] == "forbidden_history":
                assert ("forbidden_manager", event["start"], event["end"], event["confirmed"]) not in emitted
    assert transcripts["r1.e60"]["relationship"] == "same_cell_run"
    assert transcripts["r1.e60"]["cell"] == transcripts["r1.e71"]["cell"]
    comparison = compare_stair_behavior(old, view, expected_parent_checkpoint_hash=old["domain"]["checkpoint_hash"])
    assert comparison["status"] == "available"
    assert all(delta == 0 for row in comparison["rows"] for delta in row["rate_delta"])
    semantics = json.dumps(view["semantics"])
    assert not any(name in semantics for name in ("slip_mps", "slip_steps", "slip_body", "slip_any"))
    assert view["semantics"]["thresholds"]["loaded_point_speed_mps"] == full["semantics"]["thresholds"]["slip_mps"]
    assert view["semantics"]["thresholds"]["loaded_point_steps"] == full["semantics"]["thresholds"]["slip_steps"]

    # Size probe only: the historical source_public_result_hash is not rebound.
    temporary = deepcopy(prompt)
    temporary["public_feedback_summary"]["behavior"]["profile"] = view
    old_frame = temporary.pop("authoring_context_frame")
    temporary["authoring_context_frame"] = build_authoring_context_frame(
        frame_kind=old_frame["frame_kind"], section_values=temporary,
        section_order=PROMPT_PAYLOAD_SECTION_ORDER, omitted_sections=old_frame["omitted_sections"])
    temporary["authoring_context_frame"] = with_final_prompt_budget(
        temporary["authoring_context_frame"], prompt_payload=temporary)
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model="offline-serialization-only", output_dir=HERE / "results/unused-provider-output"))
    old_text = provider._chat_completion_request(prompt)["messages"][1]["content"]
    new_text = provider._chat_completion_request(temporary)["messages"][1]["content"]
    delivered = json.loads(new_text)
    budget = delivered["authoring_context_frame"]["budget_metadata"]
    assert budget["actual_prompt_payload_bytes"] == len(new_text.encode("utf-8"))
    validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
    assert before == {path: digest(path) for path in before}
    write(VIEW_PATH, view)
    report = {
        "status": "PASS", "scope": "offline representation and prompt JSON size; not semantic or author-value acceptance",
        "paid_calls": 0, "new_capture": False,
        "full_input": {"path": FULL_PATH.relative_to(ROOT).as_posix(), "sha256": before[FULL_PATH],
                       "file_bytes": FULL_PATH.stat().st_size, "episodes": len(episodes)},
        "old_prompt": {"path": OLD_PATH.relative_to(ROOT).as_posix(), "sha256": before[OLD_PATH]},
        "projection_source_sha256": before[SOURCE_PATH], "regression_script_sha256": digest(Path(__file__)),
        "domain_unchanged": True, "old_public_v1_idempotent": True, "old_files_unchanged": True,
        "production_source_unchanged_during_validation": True,
        "old": {"view_bytes": canonical_prompt_json_bytes(old), "serialized_request_bytes": len(old_text.encode("utf-8")),
                "reason_rows": len(old["profile"]["rows"]),
                "matched_forbidden_cell_run_groups": old["contrasts"]["matched_cell_run_count"],
                "contrast_rows_shown": len(old["contrasts"]["rows"]),
                "examples": [{"episode_id": e["episode_id"], "reason": e["primary_reason"], "event_count": len(e["events"])}
                             for e in old["transcripts"]]},
        "new": {"view_bytes": canonical_prompt_json_bytes(view), "view_path": VIEW_PATH.relative_to(ROOT).as_posix(),
                "view_file_sha256": digest(VIEW_PATH), "serialized_request_size_probe_bytes": len(new_text.encode("utf-8")),
                "task_sections_bytes": budget["provider_task_sections"]["serialized_bytes"],
                "request_is_size_probe_only": True, "source_public_result_hash_not_rebound": True,
                "complete_episode_denominator": len(episodes), "complete_strata": len(by_stratum),
                "reason_rows_shown": len(view["profile"]["rows"]), "reason_rows_omitted": view["profile"]["omitted_group_rows"],
                "reason_group_episodes_omitted": view["profile"]["omitted_group_episodes"], "contrasts": view["contrasts"],
                "examples": [{"episode_id": e["episode_id"], "reason": e["primary_reason"], "event_count": len(e["events"]),
                              "omitted_events": e["omitted_events"], "events": e["events"]} for e in view["transcripts"]]},
        "checks": {name: True for name in ("stratum_totals_independent_recount", "retained_contrast_independent_recount",
            "every_retained_interval_in_full_archive", "no_dangling_slip_identifiers_in_v2_semantics",
            "renamed_threshold_values_unchanged", "common_v1_v2_metrics_equal", "confirmed_root_process_retained_under_real_budget",
            "duplicate_same_span_forbidden_history_omitted", "new_public_v2_idempotent")},
        "version_comparison": comparison, "value_gate": "NOT_TESTED",
    }
    write(REPORT_PATH, report)
    print(json.dumps({"status": "PASS", "episodes": 864, "view_bytes": canonical_prompt_json_bytes(view),
        "prompt_json_bytes": len(new_text.encode("utf-8")), "source_sha256": digest(SOURCE_PATH)}))


if __name__ == "__main__":
    main()

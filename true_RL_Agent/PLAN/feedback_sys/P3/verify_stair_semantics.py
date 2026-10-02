"""Independent P3 audit against frozen P2 event counters and saved history operands.

This acceptance-only tool reads already exported full views. It does not call the
Stair decoder/tracker or build_stair_behavior again, and never modifies inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


# SHA-256 values are from the committed P2 acceptance cards' full_report fields.
PROTOTYPES = {
    "stair_parent_v1": ("p2", "parent_pair_verification.json", "c5cd3d6580ec02a6d53ad949058f8ecfcc18e1110319bcb7b5332f084c65d2f7"),
    "stair_gen8_v1": ("p2", "gen8_pair_verification.json", "5f68dc820e1eec989375613a80651f6d3e8474cc9c7572e647fbbbd4653fc37d"),
    "stair_g1_v1": ("p2", "g1_pair_verification.json", "1d788ca424ff32c4ad48d319492e1980a97420467bff100b97e241d8e17705d9"),
    "stair_parent_v2": ("audit", "stair_parent_verification.json", "da16e56f610595767e5451141d76d2375da13a167b9f524dacf1c61b6e9cc74a"),
    "stair_g1_v2": ("audit", "stair_g1_verification.json", "1d4b0c8cc865e286168c4efc3f011a0ad120eec69568fdd208b342de419ef680"),
}
EVENTS = {
    "tilt": "base_orientation_condition",
    "slip_any": "slip_condition_any_support",
    "low_clearance_any": "low_clearance_condition_any_noncontact_support",
}
ANCHORS = {"crossed": "observed_forward_crossing",
           "already_beyond_first_sample": "already_beyond_at_first_sample",
           "not_reached": "not_reached"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_counts(episode, code):
    """Derive booleans from interval operands, not the full view's saved flags."""
    terminal = episode["steps"]
    first = max(1, terminal - int(math.ceil(1.0 / episode["dt_s"])) + 1)
    intervals = [event for event in episode["events"] if event["code"] == code]
    return {
        "sustained_strictly_before_terminal": any(
            event["confirmed"] is not None and event["confirmed"] < terminal for event in intervals),
        "window_present_strictly_before_terminal": any(
            event["start"] < terminal and event["end"] >= first for event in intervals),
    }


def audit_events(full, prototype):
    """All episodes and all same-cell/run/outcome groups, including no-anchor failures."""
    rows = full["episodes"]
    old_rows = prototype["behavior"]["episodes"]
    old_by_id = {(row["run_index"], row["env_index"]): row for row in old_rows}
    require(len(old_by_id) == len(rows), "P2/P3 episode identity count differs")
    require(len({(row["run"], row["slot"]) for row in rows}) == len(rows), "duplicate P3 episode")
    groups = defaultdict(list)
    event_checks = interval_count = anchor_checks = 0
    for row in rows:
        old = old_by_id[(row["run"], row["slot"])]
        require((row["cell"], row["primary_reason"], row["steps"]) ==
                (old["cell_id"], old["outcome"]["primary_reason"], old["outcome"]["episode_steps"]),
                "P2/P3 episode outcome/time identity differs")
        require(row["coverage"]["first_step"] == 1 and row["coverage"]["last_step"] == row["steps"]
                and row["coverage"]["terminal_included"] is True, "P3 episode coverage differs")
        anchor = old["first_riser_root_anchor"]
        require(ANCHORS[row["anchor"]["status"]] == anchor["status"]
                and row["anchor"]["step"] == anchor["first_observed_crossing_step"], "root anchor differs")
        anchor_checks += 1
        first = max(1, row["steps"] - int(math.ceil(1.0/row["dt_s"])) + 1)
        require(row["coverage"]["window_first_step"] == first, "terminal window origin differs")
        for event in row["events"]:
            require(1 <= event["start"] <= event["end"] <= row["steps"], "interval escapes first episode")
            require(event["confirmed"] is None or event["start"] <= event["confirmed"] <= event["end"],
                    "confirmation escapes interval")
            interval_count += 1
        for code, old_code in EVENTS.items():
            counts = event_counts(row, code)
            require(row["before_terminal"][code] == counts["sustained_strictly_before_terminal"],
                    "saved whole-episode flag disagrees with intervals")
            require(row["window_present"][code] == counts["window_present_strictly_before_terminal"],
                    "saved window flag disagrees with intervals")
            for key, measured in counts.items():
                require(measured == old["events"][old_code][key], "P2/P3 per-episode event flag differs")
                event_checks += 1
        groups[(row["run"], row["cell"], row["primary_reason"])].append(row)
    old_groups = {(g["run_index"], g["cell_id"], g["primary_reason"]): g for g in prototype["behavior"]["groups"]}
    require(set(groups) == set(old_groups), "P2/P3 group population differs")
    group_checks = 0
    for key, group in groups.items():
        old = old_groups[key]
        require(len(group) == old["denominator"], "P2/P3 group denominator differs")
        require(dict(Counter(ANCHORS[row["anchor"]["status"]] for row in group)) ==
                old["first_riser_anchor_status_counts"], "P2/P3 anchor denominator differs")
        for code, old_code in EVENTS.items():
            for condition in ("sustained_strictly_before_terminal", "window_present_strictly_before_terminal"):
                expected = old["event_rates"][old_code][condition]
                require(expected["denominator"] == len(group), "P2 event denominator inconsistent")
                require(sum(event_counts(row, code)[condition] for row in group) == expected["count"],
                        "P2/P3 grouped event count differs")
                group_checks += 1
    paired = sum(any(row["primary_reason"] == "success" for row in group)
                 and any(row["primary_reason"] == "forbidden_contact" for row in group)
                 for group in _cell_groups(rows).values())
    return {"status": "PASS", "episodes": len(rows), "groups": len(groups),
            "per_episode_event_checks": event_checks, "group_event_checks": group_checks,
            "bounded_intervals_checked": interval_count, "root_anchor_checks": anchor_checks,
            "anchor_status_counts": dict(Counter(row["anchor"]["status"] for row in rows)),
            "same_cell_run_success_forbidden_pairs": paired,
            "outcomes": dict(Counter(row["primary_reason"] for row in rows))}


def _cell_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["cell"])].append(row)
    return groups


def frontier_counterexample(full):
    selected = [row for row in full["episodes"] if row["run"] == 1
                and row["cell"] == "public_dev.c1_conditional_frontier_level_5"]
    counts = {}
    for label, n in (("forbidden_contact", 10), ("success", 2)):
        group = [row for row in selected if row["primary_reason"] == label]
        require(len(group) == n, "frontier counterexample denominator differs")
        counts[label] = {"n": n}
        for code in EVENTS:
            counts[label][code] = {key: sum(event_counts(row, code)[key] for row in group) for key in
                ("sustained_strictly_before_terminal", "window_present_strictly_before_terminal")}
            require(all(value == (n if code == "slip_any" else 0) for value in counts[label][code].values()),
                    "frontier slip counterexample differs")
    return {"status": "PASS", "cell": "public_dev.c1_conditional_frontier_level_5", "run": 1,
            "groups": counts, "interpretation": "Both outcomes contain slip; these counts do not identify a failure cause."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--p2-audit-root", type=Path, required=True)
    parser.add_argument("--history-verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root / "src"))
    from true_rl_agent.tasks.stair_climbing.behavior import compare_stair_behavior, project_stair_behavior
    from true_rl_agent.evaluation.binding import canonical_json_hash
    loader = importlib.util.spec_from_file_location("p2_contact_history_audit", args.history_verifier)
    history_tool = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(history_tool)
    start = time.time()
    report = {"schema_version": "true_rl_agent.feedback_p3.stair_independent_semantics.v1",
              "status": "FAIL", "command": [sys.executable, *sys.argv],
              "verifier_sha256": sha(__file__), "history_verifier_sha256": sha(args.history_verifier),
              "inventory_sha256": sha(args.inventory), "datasets": {}}
    try:
        inventory, publics, fulls = read(args.inventory), {}, {}
        for name, (root_key, filename, expected_hash) in PROTOTYPES.items():
            p2_path = (args.p2_root if root_key == "p2" else args.p2_audit_root) / filename
            require(sha(p2_path) == expected_hash, "frozen P2 full report hash differs: " + name)
            prototype = read(p2_path)
            require(prototype["status"] == "PASS", "P2 was not a passing acceptance")
            directory = args.derived_root / name
            full, public, provenance = (read(directory / filename) for filename in ("full.json", "public.json", "provenance.json"))
            require(sha(directory / "full.json") == provenance["full"]["sha256"], "export full hash differs")
            require(sha(directory / "public.json") == provenance["public"]["sha256"], "export public hash differs")
            item = inventory[name]
            expected_checkpoint = item["expected_checkpoint_hash"]
            require(full["domain"]["checkpoint_hash"] == prototype["checkpoint_hash"] == expected_checkpoint,
                    "frozen checkpoint identity differs")
            require(full["domain"]["archive_hash"] == prototype["archive"]["manifest_sha256"], "P2/P3 archive differs")
            require(project_stair_behavior(full) == public == project_stair_behavior(public), "public projection differs")
            phase_path = Path(item["run_root"]) / "public_dev_result.json"
            phase = read(phase_path)
            require(sha(phase_path) == provenance["phase_file_sha256"], "source phase file changed after export")
            require(canonical_json_hash(phase["task_evidence"]) == phase["task_evidence_hash"], "phase evidence hash differs")
            require(phase["checkpoint_hash"] == expected_checkpoint, "source phase checkpoint differs")
            require(phase["task_evidence"]["numeric_evidence"]["manifest_sha256"] == full["domain"]["archive_hash"],
                    "source phase archive differs")
            summary = audit_events(full, prototype)
            require(summary["episodes"] == item["expected_episodes"], "inventory denominator differs")
            for label, n in item["expected_outcomes"].items():
                require(summary["outcomes"].get(label, 0) == n, "inventory terminal count differs")
            history_codes = {"contact_history", "forbidden_history"}
            recorded = name.endswith("_v2")
            require(full["domain"]["history_status"] == ("recorded" if recorded else "not_recorded"),
                    "history availability differs")
            if not recorded:
                require(not any(event["code"] in history_codes for row in full["episodes"] for event in row["events"]),
                        "v1 contains invented history events")
                history = {"status": "NOT_RECORDED"}
            else:
                history = history_tool.verify_contact_history(phase["task_evidence"]["numeric_evidence"])
                require(history["status"] == "PASS", "independent force-history audit failed")
                expected_contacts = 72 if name == "stair_parent_v2" else 197
                expected_zero = 26 if name == "stair_parent_v2" else 11
                require(history["counts"]["manager_contact_terminal_frames"] == expected_contacts
                        and history["counts"]["instantaneous_zero_history_nonzero_terminal_frames"] == expected_zero,
                        "history/instantaneous terminal count differs from P2")
                for code in ("forbidden_manager", "forbidden_history"):
                    observed = sum(any(event["code"] == code and event["end"] == row["steps"]
                                       for event in row["events"]) for row in full["episodes"])
                    require(observed == expected_contacts, "P3 terminal event differs from independent history")
                # Keep the full check data, with its original operand reference,
                # in this private acceptance artifact rather than the prompt.
            if "g1" in name:
                require(full["semantics"]["support_bodies"] == ["left_ankle_roll_link", "right_ankle_roll_link"]
                        and full["semantics"]["forbidden_bodies"] == ["torso_link"]
                        and full["semantics"]["kinematic_body"] == "articulation_root", "G1 body roles differ")
                require(full["semantics"]["actionability"]["contact_edge"] == ["not_in_reward_ABI"],
                        "G1 ABI incorrectly exposes first-contact")
            summary.update(p2_full_report={"path": str(p2_path), "sha256": expected_hash},
                           derived_full_sha256=provenance["full"]["sha256"],
                           derived_public_sha256=provenance["public"]["sha256"], history=history)
            report["datasets"][name] = summary
            fulls[name], publics[name] = full, public
            print(json.dumps({"dataset": name, "status": "PASS", "episodes": summary["episodes"]}), flush=True)
        report["frontier_counterexample"] = frontier_counterexample(fulls["stair_parent_v1"])
        parent_hash = inventory["stair_parent_v1"]["expected_checkpoint_hash"]
        report["true_parent_candidate_diff"] = compare_stair_behavior(publics["stair_parent_v1"], publics["stair_gen8_v1"],
                                                                    expected_parent_checkpoint_hash=parent_hash)
        require(report["true_parent_candidate_diff"]["status"] == "available", "known common-domain pair is not comparable")
        report["different_source_comparison"] = compare_stair_behavior(publics["stair_parent_v2"], publics["stair_gen8_v1"],
                                                                     expected_parent_checkpoint_hash=parent_hash)
        require(report["different_source_comparison"]["status"] == "not_comparable"
                and "source_hash" in report["different_source_comparison"].get("fields", []),
                "different execution sources were treated as the same experiment")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["elapsed_seconds"] = time.time() - start
    report["boundaries"] = [
        "These are real saved evaluations, not newly trained policies or independent policy lineages.",
        "P2 counters and P3 intervals share the original decoder; this check independently validates the new behavior reduction, not simulator ground truth.",
        "A buffered contact predicate does not locate a contact surface or prove a failure cause; no physical-substep timestamps are invented.",
        "The 26/11 counts refer to terminal instantaneous-zero/history-positive observations, not all-sample force absence.",
        "All observed first episodes remain in the denominator, including failures without the first root-riser anchor.",
        "No video validation, author improvement, causal intervention, or post-first-arrival behavior is claimed.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

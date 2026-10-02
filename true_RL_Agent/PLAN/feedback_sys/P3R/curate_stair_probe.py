"""Reproducible development-only R.0-b facts from the frozen full Stair report.

This does not produce a production behavior schema or a new acceptance label.
Raw-array and visual supplementation is recorded separately before API freeze.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "P3"
SOURCE = P3 / "results/author_final_bundle/author_blind_review/reviewer/evidence-stair.json"
LABELS = {
    "tilt": "sustained_tilt",
    "low_progress": "low_forward_gain",
    "slip_any": "loaded_point_fast_motion",
    "low_clearance_any": "unloaded_low_point",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def group_facts(episodes):
    n = len(episodes)
    return {
        "n_episodes": n,
        "step_range": [min(e["steps"] for e in episodes), max(e["steps"] for e in episodes)],
        "confirmed_before_terminal": {
            LABELS[key]: {"n": sum(e["before_terminal"][key] for e in episodes),
                          "fraction": round(sum(e["before_terminal"][key] for e in episodes) / n, 4)}
            for key in LABELS
        },
    }


def representative(episode):
    result = {key: deepcopy(episode[key]) for key in (
        "episode_id", "cell", "run", "slot", "steps", "dt_s", "primary_reason", "anchor", "ranges")}
    # Preserve full confirmed root state intervals, including earlier recurrences.
    # Foot motion proxies remain aggregate facts, not invented physical labels.
    result["root_and_support_intervals"] = [
        {**event, "duration_s": round((event["end"] - event["start"] + 1) * episode["dt_s"], 4),
         "end_at_terminal": event["end"] == episode["steps"]}
        for event in episode["events"]
        if event["code"] in {"tilt", "low_progress", "forbidden_manager"}
        or (event["code"] == "no_instant_support" and event["end"] - event["start"] >= 4)
    ]
    return result


def raw_view(episode):
    """Keep declared checkpoints and the longest unloaded body, not a raw dump."""
    result = deepcopy(episode)
    checkpoints = {
        "r1.e48": {1, 100, 150, 200, 588},
        "r1.e64": {1, 346, 360, 369, 385, 394, 414, 426},
        "r1.e69": {1, 300, 350, 400, 450, 500, 556},
        "r1.e156": {1, 170, 185, 205, 225, 232},
        "r1.e158": {1, 150, 200, 250, 300, 350, 381},
    }
    result["samples"] = [row for row in result["samples"] if row["step"] in checkpoints[result["episode_id"]]]
    result["longest_unloaded_interval_across_four_feet"] = max(
        result.pop("longest_unloaded_intervals"), key=lambda row: row["duration_s"])
    return result


def main():
    raw = SOURCE.read_bytes()
    full = json.loads(raw)
    raw_path = HERE / "results/stair_raw_independent.json"
    raw_measurements = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_measurements["manifest_sha256"] == full["domain"]["archive_hash"]
    raw_by_id = {e["episode_id"]: e for e in raw_measurements["episodes"]}
    cases = [("stair-01", "c1_conditional_frontier", "r1.e64"),
             ("stair-03", "height_ood", "r1.e156")]
    for case, stratum, primary_id in cases:
        episodes = [e for e in full["episodes"] if e["stratum"] == stratum]
        by_reason, by_cell_run = defaultdict(list), defaultdict(list)
        for e in episodes:
            by_reason[e["primary_reason"]].append(e)
            by_cell_run[(e["cell"], e["run"])].append(e)
        contrasts = []
        for (cell, run), cohort in sorted(by_cell_run.items()):
            successes = [e for e in cohort if e["primary_reason"] == "success"]
            failures = {}
            for reason in sorted(set(e["primary_reason"] for e in cohort) - {"success"}):
                failed = [e for e in cohort if e["primary_reason"] == reason]
                failures[reason] = group_facts(failed)
            if failures:
                contrasts.append({"cell": cell, "run": run, "failure_by_reason": failures,
                                  "success": group_facts(successes) if successes else {"status": "missing", "n_episodes": 0}})
        primary = next(e for e in episodes if e["episode_id"] == primary_id)
        matched = [e for e in episodes if e["primary_reason"] == "success"
                   and (e["cell"], e["run"]) == (primary["cell"], primary["run"])]
        # Deterministic median-duration matched success; hand-picked failure is
        # explicitly development material previously inspected on video.
        selected = [primary]
        if matched:
            selected.append(sorted(matched, key=lambda e: (e["steps"], e["episode_id"]))[len(matched) // 2])
        timeouts = [e for e in episodes if e["primary_reason"] == "timeout"]
        if case == "stair-01" and timeouts:
            selected.append(min(timeouts, key=lambda e: (e["run"], e["slot"])))
        facts = {
            "schema_version": "feedback_p3r.curated_numeric.v1",
            "case_id": case, "stratum": stratum,
            "provenance": {"source": "P3 frozen full report; raw-array verified in P3 audit",
                           "file_sha256": hashlib.sha256(raw).hexdigest(), "domain": full["domain"]},
            "scope": {"episodes": len(episodes), "outcomes": dict(Counter(e["primary_reason"] for e in episodes)),
                      "selection": "development: pre-inspected failure and median-duration same-cell/run success; optional r1.e48 is a level4 timeout supplement, not a matched control for the level5 pair; examples do not estimate prevalence"},
            "definitions": {
                "time": "20 ms/control step; duration counts both interval endpoints; terminal end is censoring, not natural recovery",
                "tilt": "intervals mark abs roll OR pitch >0.65 rad; confirmed=null means <10 consecutive steps. Only sustained_tilt counts require >=10 steps; Euler extrema do not establish event order",
                "count_denominator": "Every confirmed_before_terminal count/fraction uses its enclosing n_episodes; categories can overlap",
                "before_terminal": "confirmed onset strictly before terminal; a last-frame coincidence is not an antecedent",
                "loaded_point_fast_motion": "any foot with instantaneous contact force and world xy reference-point speed >0.5m/s for 5 consecutive control steps; different feet may alternate; not contact-patch slip or friction",
                "unloaded_low_point": "each step any unloaded foot point clearance <0.018m for 3 consecutive steps; different feet may alternate; not toe collision",
                "no_instant_support": "all four foot instantaneous net force norms <=1N; does not by itself establish physical flight",
                "low_forward_gain": "max(root x over full 0.5s window including endpoints)-window initial x <0.02m; not endpoint net displacement or whole-body stillness",
                "comparison": "association; success/failure exposure durations differ. Aggregated strata mix difficulty; same-cell/run rows retain that boundary",
                "ranges": "whole-episode extrema, not their order; units from field names",
            },
            "by_outcome": {reason: group_facts(rows) for reason, rows in sorted(by_reason.items())},
            "same_cell_run_contrasts": contrasts,
            "representative_processes": [representative(e) for e in selected],
            "independent_raw_samples": {
                "file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "definitions": raw_measurements["definitions"],
                "selection": "declared sparse checkpoints; longest unloaded interval across four feet per episode; full extraction retained by file hash",
                "episodes": [raw_view(raw_by_id[e["episode_id"]]) for e in selected],
            },
            "limitations": ["No reward change or causal conclusion is supplied by these observations.",
                            "Forbidden-contact termination does not identify a collision object or exact body contact point.",
                            "No confirmed tilt does not imply adequate trunk clearance or absence of a transient tilt.",
                            "Root height above environment origin is not trunk clearance above the stair surface.",
                            "Longest unloaded intervals include actual speed and position range; no controller-freeze or motor-intent label is inferred."],
        }
        write(HERE / "r0b_evidence" / f"{case}_Tc-num.json", facts)
        visual = deepcopy(facts)
        visual["schema_version"] = "feedback_p3r.curated_numeric_plus_visual_text.v1"
        image = P3 / "review_video_20260924" / ("g64_r1_zoom.png" if case == "stair-01" else "g156_r1_zoom.png")
        recording = P3 / "review_video_20260924" / ("slot64_recordings.json" if case == "stair-01" else "slot156_recordings.json")
        visual["visual_observations"] = [{
            "episode_id": primary_id,
            "source": {"grid": image.name, "grid_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                       "recordings": recording.name, "recordings_sha256": hashlib.sha256(recording.read_bytes()).hexdigest()},
            "observer": "model visual inspection, 2026-09-29; source-bound P3 replay frames",
            "modality_delivered_to_author": "text observations only, not image pixels",
            "visible_control_steps": [360,380,390,398,405,410,414,418,421,423,424,425] if case == "stair-01" else [170,185,195,205,212,218,222,225,227,229,230,231],
            "body": "whole robot/trunk and visible legs",
            "observation": ("The front of the trunk is elevated in the earlier listed frames. The trunk then rotates sideways/backwards; in frames 421-425 the underside and several raised feet face the camera while the trunk falls toward the surface. This is a visible rollover sequence."
                            if case == "stair-01" else
                            "Across these sampled frames the robot continues moving along the stairs with repeated leg flexion/extension. The trunk stays visually close to the stair surface in multiple frames; no whole-body rollover is visible in these sampled frames."),
            "visibility_limits": "Oblique single camera with shadows/partial occlusion. Cannot identify a precise collision point, contact force, friction or action intent. Last displayed frame is before terminal. Replay matches recorded source/outcome checks, not a proof of per-step bitwise trajectory identity.",
        }]
        write(HERE / "r0b_evidence" / f"{case}_Tc-vis.json", visual)
        print(case, len(json.dumps(facts, ensure_ascii=False).encode()), "bytes", "selected", [e["episode_id"] for e in selected])


if __name__ == "__main__":
    main()

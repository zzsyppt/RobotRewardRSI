"""Independent L0 arithmetic oracle for a user-supplied Stair development pool.

The oracle does not call production signature, residual, sampling or selection
helpers. The producer is invoked only to obtain the output under comparison.
No simulator, network or provider is used. Archived task outcomes supply the
frozen outcome labels; this is not a second success/termination implementation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from true_rl_agent.tasks.stair_climbing.process_report import derive_stair_process_behavior, project_stair_process_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_confirmed(mask, required, terminal):
    streak = 0
    for step, active in enumerate(mask, 1):
        streak = streak + 1 if active else 0
        if streak >= required:
            return step if step < terminal else None
    return None


def verify(manifest_path, measured_path, output_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    produced = derive_stair_process_behavior(measured)
    semantic = produced["semantic"]
    profile = manifest["metadata"]
    signals = profile["semantics"]["episode_outcomes"]["signals"]
    template_strata = {row["template_id"]: row["stratum_id"] for row in profile["evaluation_suite"]["cell_templates"]}
    by_run = defaultdict(list)
    for chunk in manifest["chunks"]:
        path = manifest_path.parent / chunk["file"]
        assert digest(path) == chunk["sha256"].removeprefix("sha256:")
        by_run[chunk["metadata"]["run_index"]].append(path)
    oracle = {}
    valid_steps = 0
    for run, paths in sorted(by_run.items()):
        buffers = defaultdict(list)
        origins = None
        for path in paths:
            with np.load(path, allow_pickle=False) as arrays:
                if "terrain_env_origins_w" in arrays:
                    origins = arrays["terrain_env_origins_w"].copy()
                for key in ("base_pos_w", "base_quat_w", "support_lin_vel_w", "support_contact_forces_w", "valid", "episode_length"):
                    buffers[key].append(arrays[key])
        arrays = {key: np.concatenate(values) for key, values in buffers.items()}
        run_record = profile["seeds"][run-1]
        dt = run_record["step_dt"]
        for slot in range(arrays["valid"].shape[1]):
            valid = arrays["valid"][:, slot]
            a = {key: values[:, slot][valid] for key, values in arrays.items()}
            steps = len(a["episode_length"])
            assert np.array_equal(a["episode_length"], np.arange(1, steps+1))
            outcome = run_record["outcomes"][slot]
            assert steps == outcome["episode_steps"]
            valid_steps += steps
            positions = a["base_pos_w"][:, 0].astype(float) - float(origins[slot, 0])
            w, x, y, z = a["base_quat_w"].astype(float).T
            roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
            pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
            up = 1-2*(x*x+y*y)
            tilt = np.maximum(np.abs(roll), np.abs(pitch)) > signals["base_orientation_instability"]["threshold"]
            span = int(np.ceil(.5/dt))
            progress = np.zeros(steps, dtype=bool)
            for index in range(span, steps):
                progress[index] = np.max(positions[index-span:index+1]) - positions[index-span] < .02
            slow = ((np.linalg.norm(a["support_contact_forces_w"].astype(float), axis=-1) <= signals["slip"]["contact_force_threshold"])
                    & (np.linalg.norm(a["support_lin_vel_w"].astype(float), axis=-1) <= .05))
            slow_first = [first_confirmed(slow[:, body], span, steps) for body in range(slow.shape[1])]
            first = {
                "tilt": first_confirmed(tilt, signals["base_orientation_instability"]["consecutive_control_steps"], steps),
                "low_progress": first_confirmed(progress, 1, steps),
                "unloaded_low_motion": min((value for value in slow_first if value is not None), default=None),
                "up_z_negative": first_confirmed(up < 0, 1, steps),
            }
            signature = []
            for term in ("tilt", "low_progress", "unloaded_low_motion", "up_z_negative"):
                if first[term] is not None:
                    # Integer control-step comparison independently avoids binary
                    # floating-point boundary ambiguity at 0.5 and 2 seconds.
                    distance = steps - first[term]
                    time_bin = "le_0.5s" if distance <= round(.5/dt) else "le_2s" if distance <= round(2/dt) else "gt_2s"
                    signature.append((term, time_bin))
            cell = outcome["evaluation_cell_id"]
            eid = f"r{run}.e{slot}"
            oracle[eid] = {"episode_id": eid, "cell": cell, "run": run, "steps": steps,
                "stratum": template_strata[cell.removeprefix("public_dev.")], "reason": outcome["primary_reason"],
                "signature": tuple(signature), "first_confirmation": first}
    assert len(oracle) == len(measured["episodes"])
    for episode in measured["episodes"]:
        raw = oracle[episode["episode_id"]]
        assert (episode["cell"], episode["run"], episode["stratum"], episode["primary_reason"], episode["steps"]) == (
            raw["cell"], raw["run"], raw["stratum"], raw["reason"], raw["steps"])
    controls, groups = defaultdict(list), defaultdict(list)
    for raw in oracle.values():
        if raw["reason"] == "success":
            controls[(raw["cell"], raw["run"])].append(raw)
        else:
            groups[(raw["reason"], raw["cell"], raw["run"])].append(raw)
    expected_domains, expected_states = [], []
    for (reason, cell, run), failed in sorted(groups.items()):
        success = controls[(cell, run)]
        frequencies = Counter(row["signature"] for row in success)
        counts = Counter()
        for raw in sorted(failed, key=lambda row: row["episode_id"]):
            match = frequencies[raw["signature"]]
            status = ("comparison_missing" if len(success) < 3 else "rule_negative"
                      if not raw["signature"] or 2*match >= len(success) else "rule_positive")
            counts[status] += 1
            expected_states.append({"episode_id": raw["episode_id"], "cell": cell, "run": run, "reason": reason,
                "status": status, "signature": [list(pair) for pair in raw["signature"]],
                "matching_success_n": match, "success_n": len(success)})
        expected_domains.append({"reason": reason, "cell": cell, "run": run, "stratum": failed[0]["stratum"],
            "failure_n": len(failed), "success_n": len(success),
            **{key: counts[key] for key in ("rule_positive", "rule_negative", "comparison_missing")},
            "episode_ids": sorted(row["episode_id"] for row in failed)})
    assert expected_states == semantic["residuals"]["episodes"]
    assert expected_domains == semantic["residuals"]["domains"]
    totals = Counter(row["status"] for row in expected_states)
    assert all(semantic["residuals"][key] == totals[key] for key in ("rule_positive", "rule_negative", "comparison_missing"))

    def order(row):
        return hashlib.sha256(f"p3new-20260929|stair|{row['episode_id']}".encode()).hexdigest()

    strata = defaultdict(list)
    for raw in oracle.values():
        strata[(raw["stratum"], "success" if raw["reason"] == "success" else "failure")].append(raw)
    selected = defaultdict(set)
    for key, members in sorted(strata.items()):
        first = min(members, key=order)
        selected[first["episode_id"]].add("rule_independent_baseline")
        if key[1] == "success":
            selected[first["episode_id"]].add("success_control")
    eligible = [oracle[row["episode_id"]] for row in expected_states if row["status"] != "rule_positive"]
    supplement = sorted(eligible, key=order)[:4]
    for raw in supplement:
        selected[raw["episode_id"]].add("residual_or_missing_supplement")
    expected_windows, time_coverage = [], []
    for eid in sorted(selected):
        raw = oracle[eid]
        steps = raw["steps"]
        covered = set()
        for anchor in (1, 1+(steps-1)//2, steps):
            covered.update(range(max(1, anchor-25), min(steps, anchor+25)+1))
        spans = []
        for step in sorted(covered):
            if spans and spans[-1][1]+1 == step:
                spans[-1][1] = step
            else:
                spans.append([step, step])
        time_coverage.append({"episode_id": eid, "covered_steps": len(covered), "total_steps": steps,
                              "fraction": len(covered)/steps, "union": spans})
        for start, end in spans:
            expected_windows.append({"episode_id": eid, "cell": raw["cell"], "run": raw["run"], "start": start, "end": end,
                "sources": sorted(selected[eid]), "checkpoint_hash": manifest["identity"]["checkpoint_hash"],
                "archive_hash": measured["domain"]["archive_hash"], "media_observation_status": "not_observed"})
    sample = semantic["sampling"]
    assert sample["windows"] == expected_windows
    assert sample["time_coverage"] == time_coverage
    expected_group_coverage = [{"stratum": key[0], "outcome": key[1], "selected_n": sum(row["episode_id"] in selected for row in group),
                               "episode_n": len(group)} for key, group in sorted(strata.items())]
    assert sample["episode_coverage"] == expected_group_coverage
    assert sample["selected_episode_ids"] == sorted(selected)
    assert sample["unselected_episode_ids"] == sorted(set(oracle)-set(selected))
    assert sample["supplemental_episode_ids"] == [row["episode_id"] for row in supplement]
    assert sample["residual_unselected_episode_ids"] == sorted(row["episode_id"] for row in eligible if row["episode_id"] not in selected)
    assert sample["residual_selected_n"] == sum(row["episode_id"] in selected for row in eligible)
    fractions = [row["fraction"] for row in time_coverage]
    assert sample["time_coverage_median"] == median(fractions) and sample["time_coverage_minimum"] == min(fractions)
    public = project_stair_process_report(produced)
    result = {"status": "PASS", "scope": "development arithmetic only; not N5 qualification or observer validation",
        "oracle_script_sha256": digest(__file__), "numeric_manifest_sha256": digest(manifest_path),
        "measured_full_sha256": digest(measured_path), "production_source_sha256": digest(Path(__file__).resolve().parents[2]/"src/true_rl_agent/tasks/stair_climbing/process_report.py"),
        "raw_chunks_checked": len(manifest["chunks"]), "valid_control_frames": valid_steps, "episode_n": len(oracle),
        "failure_n": len(expected_states), "domain_n": len(expected_domains), "residual_totals": dict(totals),
        "every_failure_signature_status_and_domain_equal": True, "every_sampling_window_and_union_equal": True,
        "public_bytes": len(json.dumps(public, ensure_ascii=False, separators=(",", ":")).encode()),
        "full_v3_bytes": len(json.dumps(produced, ensure_ascii=False, separators=(",", ":")).encode()),
        "numeric_selected_episode_n": len(selected), "residual_eligible_n": len(eligible),
        "supplemental_selected_n": len(supplement), "residual_union_selected_n": sample["residual_selected_n"],
        "supplemental_episode_ids": sample["supplemental_episode_ids"],
        "time_coverage_median": median(fractions), "time_coverage_minimum": min(fractions),
        "visual_observed_episode_n": 0, "residual_domains": expected_domains, "residual_episodes": expected_states,
        "sampling_windows": expected_windows, "episode_coverage": expected_group_coverage, "time_coverage": time_coverage,
        "residual_unselected_episode_ids": sample["residual_unselected_episode_ids"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {
        "residual_domains", "residual_episodes", "sampling_windows", "episode_coverage", "time_coverage", "residual_unselected_episode_ids"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measured-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.manifest, args.measured_full, args.output)

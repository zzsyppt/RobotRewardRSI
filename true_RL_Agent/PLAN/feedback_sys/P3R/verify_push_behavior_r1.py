"""Offline R1/R2 evidence closure; no provider, simulator, or candidate execution.

Local archive locators bind by original manifest hashes. Original phase JSON is
never rewritten. The prior independent extraction predates this implementation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from true_rl_agent.tasks.push_recovery.behavior import (
    build_push_behavior, compare_push_behavior, project_push_behavior,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = ROOT / "results/push_numeric_dev_r1"
    phase_path = archive / "public_dev_result.json"
    phase = read(phase_path)
    historical = read(ROOT.parent / "P3/results/push_qualified_reference_full_r3.json")
    independent_path = ROOT / "results/push_numeric_independent.json"
    independent = read(independent_path)
    assert independent["phase_file_sha256"] == sha(phase_path)
    context = {**historical["binding"], "reward_input_fields": list(PushRecoveryV2RewardInputs.field_names())}
    seeds = deepcopy(phase["task_evidence"]["seed_summaries"])
    for index, seed in enumerate(seeds, 1):
        seed["numeric_evidence_ref"]["manifest_path"] = str(archive / f"numeric/public_dev-{index:02d}/manifest.json")
    full = build_push_behavior(seeds, context=context)
    public = project_push_behavior(full)
    assert project_push_behavior(public) == public
    with_diff = deepcopy(public)
    with_diff["parent_diff"] = compare_push_behavior(full, full, expected_parent_checkpoint_hash=context["checkpoint_hash"])
    with_diff = project_push_behavior(with_diff)
    # Historical sealed v1 remains reproducible, not upgraded by guessing facts.
    assert project_push_behavior(historical) == phase["task_evidence"]["behavior"]
    truth = {(row["source"], row["env_id"]): row for row in independent["episodes"]}
    measured = 0
    for row in full["transcripts"]:
        old = truth[row["source"], row["env_id"]]
        for key in ("terminal_label", "valid_steps", "final_failed_conditions", "all_required_recoveries_latched", "unstable_steps_after_all_recoveries"):
            assert row[key] == old[key], (row["source"], row["env_id"], key)
        for unit, previous in zip(row["recoveries"], old["recoveries"], strict=True):
            for key in ("anchor_control_step", "anchor_verified", "observed_window_steps", "stable_window_steps", "max_dwell_steps", "recovery_control_step", "failed_condition_steps"):
                assert unit[key] == previous[key], (row["source"], row["env_id"], unit["identity"], key)
            process, prior = unit["process"], previous.get("process")
            if not prior:
                assert process["status"] == "no_verified_observed_window"
                continue
            measured += 1
            for key in ("window_first", "window_speed_peak", "window_last"):
                for name, value in prior[key].items():
                    np.testing.assert_allclose(process[key][name], value, rtol=0, atol=1e-12)
            np.testing.assert_allclose(process["window_net_displacement_world_xy_m"], prior["window_net_displacement_world_xy_m"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(process["window_net_displacement_parallel_m"], prior["window_net_displacement_parallel_to_this_push_m"], rtol=0, atol=1e-12)
            assert process["window_reaches_planned_end"] == prior["window_reaches_planned_end"]
    # Independent array arithmetic also checks intervals added after R0:
    # every span is contiguous, maximal, bounded by that event's observation,
    # and sums to the actual mask. No production interval helper is used here.
    condition_spans = stable_spans = post_spans = 0
    for source, seed in enumerate(seeds):
        ref = seed["numeric_evidence_ref"]
        manifest = load_numeric_evidence_manifest(ref)
        chunks = [arrays for _, arrays in iter_numeric_evidence(ref)]
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in ("valid", "control_step", "conditions")}
        for row in (item for item in full["transcripts"] if item["source"] == source):
            env = row["env_id"]
            valid = arrays["valid"][:, env]
            steps, cond = arrays["control_step"][valid, env], arrays["conditions"][valid, env]
            for unit in row["recoveries"]:
                anchor = unit["anchor_control_step"]
                if anchor is None:
                    continue
                window = (steps > anchor) & (steps <= anchor + manifest["metadata"]["time"]["recovery_window_steps"])
                masks = [(unit["condition_violation_spans"][name], window & ~cond[:, i]) for i, name in enumerate(manifest["metadata"]["condition_fields"])]
                process = unit["process"]
                if process["status"] == "available":
                    masks.append((process["stable_spans"], window & cond.all(axis=1)))
                    stable_spans += len(process["stable_spans"])
                    post = process["post_latch"]
                    latch = unit["recovery_control_step"]
                    post_mask = ((steps > latch) & (steps <= post["observed_through_step"]) & ~cond.all(axis=1)
                                 if latch is not None else np.zeros(len(steps), dtype=bool))
                    masks.append((post["unstable_spans"], post_mask))
                    post_spans += len(post["unstable_spans"])
                for spans, mask in masks:
                    expected = []
                    for step, enabled in zip(steps.tolist(), mask.tolist()):
                        if enabled:
                            if expected and expected[-1][1] == step - 1:
                                expected[-1][1] = step
                            else:
                                expected.append([step, step])
                    assert [[span["start_step"], span["end_step_inclusive"]] for span in spans] == expected
                    assert sum(span["observed_steps"] for span in spans) == int(mask.sum())
                condition_spans += sum(len(values) for values in unit["condition_violation_spans"].values())
    outputs = {"push_behavior_r1_full.json": full, "push_behavior_r1_public.json": public,
               "push_behavior_r1_public_with_parent_diff.json": with_diff}
    for name, value in outputs.items():
        (ROOT / "results" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PASS", "scope": "historical 192-episode public-dev derivation; no new policy/causal/generalization evidence",
        "phase_sha256": sha(phase_path), "independent_extraction_sha256": sha(independent_path),
        "manifest_sha256": [seed["numeric_evidence_ref"]["manifest_sha256"] for seed in seeds],
        "episodes": len(full["transcripts"]), "planned_units": full["profile"]["all"]["recovery_units"],
        "independently_matched_windows": measured, "checked_condition_spans": condition_spans,
        "checked_all_five_stable_spans": stable_spans, "checked_post_latch_spans": post_spans,
        "public_serialized_bytes": canonical_prompt_json_bytes(public),
        "with_parent_diff_serialized_bytes": canonical_prompt_json_bytes(with_diff),
        "old_v1_projection_unchanged": True, "projection_idempotent": True,
        "strata": {key: value["episodes"] for key, value in public["profile"]["by_stratum"].items()},
        "first_run_episodes": public["profile"]["first_run"]["episodes"],
        "examples": [{key: row[key] for key in ("source", "env_id", "stratum", "terminal_label")} for row in public["representative_transcripts"]],
        "outputs": {name: sha(ROOT / "results" / name) for name in outputs},
        "production_sha256": sha(ROOT.parents[2] / "src/true_rl_agent/tasks/push_recovery/behavior.py")}
    (ROOT / "results/push_behavior_r1_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

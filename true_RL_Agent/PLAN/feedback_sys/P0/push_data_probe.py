"""Read-only Push protocol/ABI and bounded existing-artifact inventory for P0.

Does not instantiate a simulator, load a checkpoint, call a provider or alter evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}

def main() -> None:
    import numpy as np
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
    from true_rl_agent.tasks.push_recovery.prompt_adapter import _v2_reward_inputs_contract, _v2_dev_summary
    from true_rl_agent.tasks.push_recovery.metrics import project_push_split
    from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest

    rows = []
    for cell in ("push.go2_12dof.recovery_v2", "push.g1_37dof.recovery_v2",
                 "push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
        config = PushRecoveryTaskConfig.for_cell(cell)
        plugin = PushRecoveryTaskPlugin(config)
        manifest = resolve_push_cell_manifest(cell_variant_id=cell, config=config, plugin=plugin)
        projection = manifest.robot_task_binding["task_projection"]
        abi = projection["push_abi_facts"]
        evaluation = projection.get("push_resolved_evaluation_protocol", projection["push_resolved_protocol"])
        dt = abi["control"]["control_dt_s"]
        timing = {"control_dt_s": dt, "window_steps": math.ceil(evaluation["recovery_window_s"] / dt),
                  "required_dwell_steps": math.ceil(evaluation["stable_dwell_s"] / dt),
                  "nominal_push_start_steps": [math.ceil(value / dt) for value in evaluation["push_times_s"]]}
        if "pulse_duration_s" in evaluation:
            pulse = math.ceil(evaluation["pulse_duration_s"] / dt)
            gap = math.ceil(evaluation["repeated_gap_s"] / dt)
            first = timing["nominal_push_start_steps"][0]
            timing.update(pulse_steps=pulse, gap_steps=gap, actual_pulse_duration_s=pulse*dt,
                          repeated_start_steps=[first, first+pulse+gap],
                          single_recovery_zero_step=first+pulse,
                          repeated_recovery_zero_step=first+2*pulse+gap)
        rows.append({"cell": cell, "configuration_basis": "current for_cell defaults; historical manifests may differ",
                     "resolved_cell_manifest_hash": manifest.manifest_hash,
                     "resolved_task_protocol_hash": manifest.resolved_cell_task_protocol_hash,
                     "runtime_adapter_id": config.runtime_adapter_id,
                     "role_bindings": manifest.robot_task_binding["role_bindings"],
                     "abi": {k: abi[k] for k in ("policy_observation_dimension", "action_dimension", "control",
                              "binding_facts_hash", "policy_io_schema_hash", "sensor_schema_hash", "sensors")},
                     "training": projection["push_resolved_protocol"], "evaluation": evaluation,
                     "quantized_timing": timing})

    timing_artifacts = []
    for family in ("independent_timing_evidence_01", "stationary_v3_evidence_01"):
        for robot in ("go2", "g1"):
            for profile in ("strong_push", "no_push"):
                rel = Path("PLAN/2new_world/P6") / family / robot
                if family == "stationary_v3_evidence_01":
                    rel /= "timing"
                directory = ROOT / rel / profile
                if not (directory / "raw.npz").exists():
                    continue
                meta, independent = read(directory / "capture.json"), read(directory / "independent.json")
                records = meta["task_records"]
                item = {"raw": identity(directory / "raw.npz"), "capture": identity(directory / "capture.json"),
                        "independent": identity(directory / "independent.json"), "robot": robot, "profile": profile,
                        "diagnostic_only": meta.get("diagnostic_only"), "enters_qualification": meta.get("enters_qualification"),
                        "checkpoint_sha256": meta["checkpoint_sha256"], "manifest_hash": meta["manifest_hash"],
                        "mechanism": meta["parameters"]["mechanism"],
                        "resolved_protocol_hash": meta["parameters"]["resolved_protocol_hash"],
                        "control_dt_s": meta["dt"], "capture_point": meta.get("capture_point"),
                        "quaternion_order": meta.get("quaternion_order"),
                        "raw_sha256_matches_capture": sha(directory / "raw.npz") == meta["raw_sha256"].removeprefix("sha256:"),
                        "saved_independent_report_status": independent.get("status"),
                        "saved_report_matches_task": independent.get("independent_timing_matches_task"),
                        "saved_formal_first_seed_records_equal": meta.get("formal_first_seed_task_records_equal"),
                        "episode_count": len(records), "failure_labels": dict(Counter(r["failure_label"] for r in records)),
                        "episode_fields": sorted({key for record in records for key in record}),
                        "event_fields": sorted({key for record in records for event in record.get("push_events", []) for key in event})}
                with np.load(directory / "raw.npz", allow_pickle=False) as raw:
                    item["arrays"] = {name: {"shape": list(raw[name].shape), "dtype": str(raw[name].dtype)} for name in raw.files}
                    valid = raw["valid"]
                    lengths = valid.sum(axis=0)
                    item["valid_length_range"] = [int(lengths.min()), int(lengths.max())]
                    item["contiguous_first_episode_steps"] = all(np.array_equal(raw["steps"][valid[:, i], i], np.arange(1, int(lengths[i])+1)) for i in range(valid.shape[1]))
                timing_artifacts.append(item)

    public_artifacts = []
    paths = (
        "PLAN/2new_world/P6/g1_qualified_01/formal/runner/p6-g1-stationary-v3-reference-20260912/eval/public_dev/summary.json",
        "PLAN/2new_world/P6/go2_pose_01/native/eval/public_dev/summary.json",
        "PLAN/2new_world/Go2_push/attempt_08/session/runner/go2-push-5rounds-20260923-a08-generation-004-lane-a/eval/public_dev/summary.json",
    )
    last_public = None
    for relative in paths:
        path = ROOT / relative
        payload = read(path)
        assert payload["evaluation_phase"] == "public_dev"
        seeds = payload["task_evidence"]["seed_summaries"]
        episodes = [row for seed in seeds for row in seed["per_episode"]]
        errors = []
        projected = project_push_split(seeds, split_name="public_dev", errors=errors)
        last_public = _v2_dev_summary(projected)
        public_artifacts.append({
            "artifact": identity(path), "outer_schema": payload["schema_version"],
            "resolved_cell_identity": payload.get("resolved_cell_identity"),
            "rollout_schemas": sorted({seed["schema_version"] for seed in seeds}),
            "mechanisms": sorted({seed["task"]["mechanism"] for seed in seeds}),
            "resolved_protocol_hashes": sorted({seed["evidence"]["resolved_protocol_hash"] for seed in seeds}),
            "episode_count": len(episodes), "failure_labels": dict(Counter(row["failure_label"] for row in episodes)),
            "episode_fields": sorted({key for row in episodes for key in row}),
            "push_event_fields": sorted({key for row in episodes for event in row.get("push_events", []) for key in event}),
            "force_event_fields": sorted({key for row in episodes for event in row.get("force_events", []) for key in event}),
            "not_recovered_split": dict(Counter(
                "dwell_completed_but_final_unstable" if row.get("sequence_recovered") else "no_sequence_dwell_recorded"
                for row in episodes if row["failure_label"] == "not_recovered")) if "sequence_recovered" in episodes[0] else None,
            "public_projection_errors": errors, "author_dev_projection": last_public,
        })
    prompt_path = ROOT / "PLAN/2new_world/Go2_push/attempt_08/session/authoring/go2-push-5rounds-20260923-a08-generation-005/lane-b/authoring_attempts/attempt_000001/prompt_payload.json"
    prompt = read(prompt_path)
    def key_present(value, key):
        if isinstance(value, dict):
            return key in value or any(key_present(child, key) for child in value.values())
        if isinstance(value, list):
            return any(key_present(child, key) for child in value)
        return False
    feedback, runtime = prompt["public_feedback_summary"], prompt["runtime_feedback_summary"]
    prompt_artifact = {"artifact": identity(prompt_path), "schema_version": prompt["schema_version"],
                      "public_feedback_summary": feedback, "runtime_feedback_summary": runtime,
                      "public_dev_matches_current_projection_of_bound_saved_rollout": feedback["public_dev"] == last_public,
                      "runtime_and_behavior_same_trial": all(feedback.get(key) == runtime.get(key) for key in (
                          "source_round_id", "source_candidate_id", "program_source_checksum")),
                      "structured_field_presence": {key: key_present(prompt, key) for key in (
                          "per_episode", "force_events", "push_events", "sequence_recovery_control_step",
                          "min_base_height_m", "dwell_break_reason", "input_provenance",
                          "max_abs_candidate_reward_frame", "public_accept", "sealed_validation")}}
    train_path = ROOT / "PLAN/2new_world/Go2_push/attempt_08/session/runner/go2-push-5rounds-20260923-a08-generation-004-lane-a/train/summary.json"
    train = read(train_path)
    saved_runtime = train["runtime_activation_summary"]
    train_artifact = {"artifact": identity(train_path), "schema_version": train["schema_version"],
                      "program_source_checksum": train["program_source_checksum"],
                      "matches_prompt_runtime_source": train["program_source_checksum"] == runtime["program_source_checksum"],
                      "runtime_fields": sorted(saved_runtime), "input_provenance": saved_runtime["input_provenance"],
                      "step_calls": saved_runtime["step_calls"], "reward_term_calls": saved_runtime["reward_term_calls"],
                      "diagnostic_calls": saved_runtime["diagnostic_calls"],
                      "reward_peak_frame": saved_runtime.get("max_abs_candidate_reward_frame")}

    source_paths = [ROOT / "src/true_rl_agent/tasks/push_recovery" / name for name in (
        "config.py", "protocol.py", "force_protocol.py", "runtime.py", "force_runtime.py", "rollout.py",
        "robot_binding.py", "plugin.py", "reward_inputs.py", "runtime_adapter.py", "prompt_adapter.py",
        "metrics.py", "phased_adapter.py", "failure_classifier.py", "memory.py", "exploration.py")]
    source_paths.extend([ROOT / "configs/tasks/push_recovery_robot_policy_abi_v2.json",
                         ROOT / "PLAN/2new_world/P6/audit_push_episode_timing.py"])
    result = {"schema_version": "feedback_p0.push_data_probe.v1", "scope": "deterministic local metadata only",
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_identities": [identity(path) for path in source_paths],
              "protocol_rows": rows, "reward_contract": _v2_reward_inputs_contract(),
              "saved_timing_artifacts": timing_artifacts, "saved_public_dev_artifacts": public_artifacts,
              "saved_author_prompt": prompt_artifact, "saved_training_artifact": train_artifact}
    (OUT / "push_data_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"rows": [{"cell": r["cell"], "L": r["evaluation"]["robot_scale_m"], "timing": r["quantized_timing"]} for r in rows],
                      "saved_timing": [{"robot": r["robot"], "profile": r["profile"], "path": r["raw"]["path"],
                                        "labels": r["failure_labels"], "hash_valid": r["raw_sha256_matches_capture"],
                                        "contiguous": r["contiguous_first_episode_steps"]} for r in timing_artifacts],
                      "saved_public_dev": [{"path": r["artifact"]["path"], "schemas": r["rollout_schemas"],
                                            "labels": r["failure_labels"], "not_recovered_split": r["not_recovered_split"],
                                            "projection_errors": r["public_projection_errors"]} for r in public_artifacts],
                      "prompt_checks": {k: prompt_artifact[k] for k in (
                          "public_dev_matches_current_projection_of_bound_saved_rollout", "runtime_and_behavior_same_trial", "structured_field_presence")}}, ensure_ascii=False))

if __name__ == "__main__":
    main()

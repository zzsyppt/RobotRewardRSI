"""One fixed-checkpoint paired timing diagnostic, outside formal qualification."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from push_sequence_scoring import raw_conditions, sequence_outcome
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider

CHECKPOINT = "sha256:1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff"
ARMS = ("original", "continuous")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args):
    source = ROOT / "outputs/p6_go2_matched_dev_01"
    record = read(source / "result.json")
    train_path = source / "composition/runner/p6-go2-matched2-dev-20260912/train/summary.json"
    train = read(train_path)
    checkpoint = Path(train["checkpoint"]["metadata"]["local_path"])
    assert record["status"] == "PASS" and record["checkpoint_sha256"] == CHECKPOINT
    assert train["checkpoint"]["checksum"] == file_sha256(checkpoint) == CHECKPOINT
    provider = PushRecoveryCompositionProvider()
    composition = provider.resolve(values={**provider.cli_defaults(), **record["budget"],
        "cell_variant_id": record["cell"], "execution_device": args.device,
        "isaac_python": sys.executable, "reference_program_id": "push_standing_recovery_stationary_reference_v3",
        "push_protocol_path": str(Path(__file__).parent / "calibration_protocols_01/scale_2p0.json")},
        context=TaskCompositionContext(project_root=ROOT, output_dir=source / "composition", read_only=True))
    manifest = composition.resolved_cell_manifest
    assert manifest.manifest_hash == train["resolved_cell_manifest_hash"]
    old_environment = read(source / "backend_execution_environment.json")
    composition.backend_adapter.validate_session_execution_environment(execution_environment=old_environment)
    current_environment = composition.backend_adapter.resolve_session_execution_environment(
        session_id="p6-sequence-probe", session_dir=args.output, allow_create=True)
    assert current_environment["manifest_hash"] == old_environment["manifest_hash"]
    write(args.output / "backend_execution_environment.json", current_environment)
    projection = manifest.robot_task_binding["task_projection"]
    parameters = dict(projection["push_resolved_protocol"])
    dt = projection["push_abi_facts"]["control"]["control_dt_s"]
    definition = {"schema_version": "true_rl_agent.p6.push_sequence_diagnostic.v1",
        "source_resolved_protocol_hash": parameters["resolved_protocol_hash"],
        "checkpoint_role": "existing_D_inference_only_not_new_protocol_training",
        "base_seed": 42, "derived_seeds": [10042, 10043, 10044], "envs_per_seed": 64,
        "affected_stratum": "repeated", "arms": list(ARMS),
        "first_push_time_scale": 5.0, "original_second_push_time_scale": 17.0,
        "continuous_second_push_time_scale": 6.0,
        "step_rounding": "ceil_each_absolute_time_divided_by_control_dt",
        "control_dt_s": dt, "sequence_recovery": "continuous_dwell_after_last_verified_push",
        "first_recovery_sample": "last_injection_step_plus_one", "window_inclusive": True,
        "require_two_verified_pushes": True, "require_final_stable": True,
        "require_no_fall_or_boundary_exit_during_entire_episode": True,
        "old_event_recovery_values_used_for_sequence_success": False,
        "window_steps": math.ceil(parameters["recovery_window_s"] / dt),
        "dwell_steps": math.ceil(parameters["stable_dwell_s"] / dt),
        "train_jobs": 0, "holdout_jobs": 0, "provider_calls": 0, "automatic_retries": 0,
        "wall_limit_s": 1800}
    definition["definition_hash"] = canonical_json_hash(definition)
    write(args.output / "definition.json", definition)
    arms = {}
    for arm in ARMS:
        actual = deepcopy(parameters)
        if arm == "continuous":
            actual["push_times_s"][1] = 6.0 * actual["time_scale_s"]
        # A diagnostic overlay has its own identity; never label it as the source manifest.
        actual.update(schema_version="true_rl_agent.p6.push_sequence_parameters.v1",
                      source_protocol_hash=actual.pop("protocol_hash"),
                      protocol_hash=definition["definition_hash"], arm=arm)
        actual.pop("resolved_protocol_hash")
        actual["resolved_protocol_hash"] = canonical_json_hash(actual)
        arms[arm] = actual
    write(args.output / "parameters.json", arms)
    identity = {"source_manifest_hash": manifest.manifest_hash,
        "source_training_binding_hash": train["training_binding_hash"],
        "checkpoint_sha256": CHECKPOINT, "source_environment_hash": current_environment["manifest_hash"],
        "source_train_summary_sha256": file_sha256(train_path),
        "source_dev_result_sha256": file_sha256(source / "public_dev_result.json"),
        "script_sha256": file_sha256(__file__),
        "scoring_sha256": file_sha256(Path(__file__).with_name("push_sequence_scoring.py")),
        "definition_hash": definition["definition_hash"]}
    write(args.output / "source_identity.json", identity)
    return composition, checkpoint, arms, definition, read(source / "public_dev_result.json"), identity


def run_arm(args, composition, checkpoint, arm, parameters, definition, old):
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.isaac_lab_arrays import as_torch
    from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint
    from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
    from true_rl_agent.tasks.push_recovery.runtime import (physical_runtime_config_from_manifest,
        instantiate_physical_push_configs, make_physical_push_environment, push_policy_io_evidence,
        push_runtime_state)
    from true_rl_agent.tasks.push_recovery.rollout import run_physical_push_rollout

    output = args.output / arm
    output.mkdir()
    manifest = composition.resolved_cell_manifest
    config = physical_runtime_config_from_manifest(manifest, "strong_push")
    rollout_args = SimpleNamespace(seed=42, num_envs=64, device=args.device, max_steps=None,
        public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
        evaluation_subject=ExperimentSubject.baseline("p6-D-sequence-probe"),
        task_id=composition.task_id, round=args.output.name)
    torch.manual_seed(42)
    env_cfg, agent_cfg = instantiate_physical_push_configs(config, rollout_args, evaluation=True)
    env_cfg.push_protocol_parameters = deepcopy(parameters)
    base = make_physical_push_environment(env_cfg)
    wrapped = None
    try:
        abi = push_policy_io_evidence(base, manifest)
        agent_cfg.device = args.device
        wrapped = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        learner = rsl_rl_agent_configuration(config["runtime_adapter_id"]).verify_runner(runner)
        loaded = load_inference_checkpoint(runner, str(checkpoint))
        write(output / "runtime_admission.json", {"policy_io": abi, "learner": learner, "checkpoint_loading": loaded})
        policy = runner.get_inference_policy(device=base.device)
        state, robot = push_runtime_state(base), base.scene["robot"]
        assert list(state.event_times) == ([51, 172] if arm == "original" else [51, 61])
        cpu = lambda value: as_torch(value).detach().cpu().numpy().copy()
        all_rows = []
        controls_mismatched = []
        old_sequences_mismatched = []
        for index in range(1, 4):
            frames = []
            original_compute = base.termination_manager.compute

            def capture_compute():
                frames.append({"valid": cpu(~state.closed), "steps": cpu(base.episode_length_buf),
                    "position": cpu(robot.data.root_pos_w), "quaternion_xyzw": cpu(robot.data.root_quat_w),
                    "linear": cpu(robot.data.root_lin_vel_w), "angular": cpu(robot.data.root_ang_vel_w)})
                return original_compute()

            base.termination_manager.compute = capture_compute
            try:
                rollout = run_physical_push_rollout(wrapped, config, rollout_args, inference_policy=policy,
                    split_name="public_dev", run_index=index)
            finally:
                base.termination_manager.compute = original_compute
            assert rollout["status"] == "pass", "invalid injection, finite state or episode closure"
            arrays = {key: np.stack([row[key] for row in frames]) for key in frames[0]}
            raw_path = output / f"seed_{index:02d}.npz"
            np.savez_compressed(raw_path, **arrays)
            initial_xy, origins = cpu(state.initial_xy), cpu(base.scene.env_origins)
            result_rows = []
            for row, before in zip(rollout["per_episode"], old["task_evidence"]["seed_summaries"][index-1]["per_episode"], strict=True):
                env_id = row["env_id"]
                if row["stratum"] != "repeated":
                    if row != before:
                        controls_mismatched.append({"seed_index": index, "env_id": env_id,
                            "fields": [k for k in row if row[k] != before[k]]})
                    result_rows.append(row)
                    continue
                valid = arrays["valid"][:, env_id]
                steps = arrays["steps"][valid, env_id].tolist()
                stable, failure = raw_conditions(**{name: arrays[name][valid, env_id]
                    for name in ("position", "quaternion_xyzw", "linear", "angular")},
                    initial_xy=initial_xy[env_id], origin=origins[env_id], parameters=parameters)
                sequence = sequence_outcome(steps, stable, row["push_events"], failure_label=failure,
                    final_stable=stable[-1], window_steps=definition["window_steps"], dwell_steps=definition["dwell_steps"])
                clean = {k: v for k, v in row.items() if k not in {"push_events", "recovered_event_count"}}
                clean["push_events"] = [{k: v for k, v in e.items() if k != "recovered" and not k.startswith("recovery_")}
                                        for e in row["push_events"]]
                clean.update(sequence, recovery_scope="sequence_after_last_push")
                clean["sequence_recovery_latency_s"] = ((sequence["sequence_recovery_control_step"] - row["push_events"][-1]["control_step"]) * base.step_dt
                                                        if sequence["sequence_recovered"] else None)
                if arm == "original" and sequence["success"] != before["success"]:
                    old_sequences_mismatched.append({"seed_index": index, "env_id": env_id})
                result_rows.append(clean)
            write(output / f"seed_{index:02d}.json", {"diagnostic_only": True, "arm": arm,
                "definition_hash": definition["definition_hash"], "seed": 10041 + index,
                "parameters_hash": parameters["resolved_protocol_hash"], "raw_sha256": file_sha256(raw_path),
                "initial_xy": initial_xy.tolist(), "origins": origins.tolist(),
                "capture_point": "pre_reset_before_termination_compute_after_physics",
                "episodes": result_rows, "vector_env_step_calls": rollout["execution_cost"]["vector_env_step_calls"]})
            all_rows.extend(result_rows)
        report = {"arm": arm, "controls_mismatched_with_original_D": controls_mismatched,
                  "baseline_sequence_success_mismatches": old_sequences_mismatched,
                  "episodes": len(all_rows), "strata": {}}
        for stratum in ("low", "medium", "high", "repeated"):
            rows = [r for r in all_rows if r["stratum"] == stratum]
            report["strata"][stratum] = {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
                "failure_counts": dict(Counter(r["failure_label"] for r in rows))}
        report["successes"] = sum(r["success"] for r in all_rows)
        report["observed_pushes"] = sum(len(r["push_events"]) for r in all_rows)
        report["planned_pushes"] = sum(r["planned_push_count"] for r in all_rows)
        report["invalid_injections"] = sum(r["invalid_injection_count"] for r in all_rows)
        write(output / "summary.json", report)
        assert not controls_mismatched and not old_sequences_mismatched, "unchanged control or baseline mismatch"
        assert report["invalid_injections"] == 0
        return report
    finally:
        (wrapped or base).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-definition-hash")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "diagnostic_only": True, "mode": args.mode,
        "started_at_unix": time.time(), "arms_started": [], "arms_completed": [],
        "train_jobs": 0, "holdout_jobs": 0, "provider_calls": 0, "automatic_retries": 0}
    app, code = None, 1
    try:
        composition, checkpoint, parameters, definition, old, identity = prepare(args)
        if args.expected_definition_hash is not None:
            assert definition["definition_hash"] == args.expected_definition_hash
        report.update(source_identity=identity)
        if args.mode == "run":
            assert args.expected_definition_hash is not None
            from isaaclab.app import AppLauncher
            from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path
            launch = argparse.Namespace(headless=True, device=args.device)
            bind_isaac_log_path(launch, args.output / "result.json")
            app = AppLauncher(launch, multi_gpu=False).app
            report["arms"] = {}
            for arm in ARMS:
                report["arms_started"].append(arm)
                write(args.output / "result.json", report)
                report["arms"][arm] = run_arm(args, composition, checkpoint, arm, parameters[arm], definition, old)
                report["arms_completed"].append(arm)
        report["status"] = "PASS" if args.mode == "run" else "PREPARED"
        code = 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["ended_at_unix"] = time.time()
        write(args.output / "result.json", report)
    if app is not None:
        from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success
        code = close_isaac_app_on_success(app, exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

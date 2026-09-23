"""One original dev seed, two diagnostic profiles; never qualification evidence.

Capture reads physical state at TerminationManager.compute, before task observers.
Recompute uses NumPy and actual velocity writes, not the task's stable/dwell state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flag(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f"source job requires exactly one {name}")
    return argv[argv.index(name) + 1]


def recompute(directory):
    import numpy as np

    directory = Path(directory)
    meta = read(directory / "capture.json")
    if sha(directory / "raw.npz") != meta["raw_sha256"]:
        raise ValueError("raw trajectory hash mismatch")
    with np.load(directory / "raw.npz", allow_pickle=False) as source:
        raw = {key: source[key] for key in source.files}
    p, dt = meta["parameters"], meta["dt"]
    window, dwell = math.ceil(p["recovery_window_s"] / dt), math.ceil(p["stable_dwell_s"] / dt)
    strata = [name for name in ("low", "medium", "high", "repeated")
              for _ in range(p["evaluation_cohort_counts"][name])]
    results, mismatches = [], []
    for env_id in range(64):
        mask = raw["valid"][:, env_id]
        steps = raw["steps"][mask, env_id]
        pos, quat = raw["position"][mask, env_id], raw["quaternion_xyzw"][mask, env_id]
        linear, angular = raw["linear"][mask, env_id], raw["angular"][mask, env_id]
        if not len(steps) or not np.array_equal(steps, np.arange(1, len(steps) + 1)):
            raise ValueError(f"non-contiguous first episode for env {env_id}")
        # Native Isaac Lab 3/Sim 6 buffers are xyzw. Do not use the product conversion helper.
        x, y, z, w = quat.astype(np.float64).T
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
        height = pos[:, 2] - meta["origins"][env_id][2]
        drift = np.linalg.norm(pos[:, :2] - np.array(meta["initial_xy"])[env_id], axis=1)
        speed, omega = np.linalg.norm(linear[:, :2], axis=1), np.linalg.norm(angular, axis=1)
        stable = ((speed <= p["recovery_velocity_mps"]) & (omega <= p["recovery_angular_velocity_radps"])
                  & (np.abs(roll) <= p["recovery_tilt_rad"]) & (np.abs(pitch) <= p["recovery_tilt_rad"])
                  & (height >= p["recovery_height_m"]))
        failed = ((height < p["fall_height_m"]) | (np.abs(roll) > p["fall_tilt_rad"])
                  | (np.abs(pitch) > p["fall_tilt_rad"]) | (drift > p["max_displacement_m"]))
        expected_done = failed | (steps >= meta["max_episode_length"])
        if not np.array_equal(expected_done, raw["done"][mask, env_id]) or not expected_done[-1]:
            mismatches.append({"env_id": env_id, "field": "independent_termination"})
        events = [event for event in meta["writes"] if event["env_id"] == env_id]
        recovery_steps = []
        injection_validity = []
        for index, event in enumerate(events):
            before, desired, after = (np.array(event[key]) for key in ("before", "desired", "after"))
            tolerance = p["injection_tolerance_mps"]
            magnitude = p["velocity_magnitudes_mps"]["medium" if strata[env_id] == "repeated" else strata[env_id]]
            valid_injection = bool(index < len(p["push_times_s"]) and np.isfinite(after).all()
                and np.max(abs(after - desired)) <= tolerance and np.max(abs(after[2:] - before[2:])) <= tolerance
                and abs(np.linalg.norm(after[:2] - before[:2]) - magnitude) <= tolerance
                and event["step"] == math.ceil(p["push_times_s"][index] / dt))
            injection_validity.append(valid_injection)
            within = (steps > event["step"]) & (steps <= event["step"] + window)
            # A full window of D true samples is independent of the runtime's mutable counter.
            samples = stable[within]
            hits = np.flatnonzero(np.convolve(samples.astype(int), np.ones(dwell, dtype=int), "valid") == dwell) if len(samples) >= dwell else []
            recovery = int(steps[within][int(hits[0]) + dwell - 1]) if len(hits) and valid_injection else None
            recovery_steps.append(recovery)
        control = meta["profile"] == "no_push"
        planned = 0 if control else (2 if strata[env_id] == "repeated" else 1)
        fall = ("fall_forward" if max(pitch) > p["fall_tilt_rad"] else
                "fall_backward" if min(pitch) < -p["fall_tilt_rad"] else
                "fall_side" if max(abs(roll)) > p["fall_tilt_rad"] else
                "height_collapse" if min(height) < p["fall_height_m"] else None)
        outside = max(drift) > p["max_displacement_m"]
        invalid = not all(injection_validity)
        invalid |= control and bool(events)
        invalid |= not control and not fall and not outside and len(events) != planned
        recovered = len(events) == planned and planned > 0 and all(step is not None for step in recovery_steps)
        label = ("invalid_push_injection" if invalid else "no_push_control" if control else fall or
                 ("out_of_bounds" if outside else "success_recovered" if recovered and stable[-1] else "not_recovered"))
        own = {"env_id": env_id, "episode_steps": int(steps[-1]), "failure_label": label,
               "success": label == "success_recovered", "final_recovery_condition": bool(stable[-1]),
               "planned_push_count": planned, "push_event_count": len(events),
               "recovery_control_steps": recovery_steps, "stable_sample_count": int(stable.sum()),
               "final_speed_mps": float(speed[-1]), "final_angular_speed_radps": float(omega[-1]),
               "fall_label": fall, "out_of_bounds": bool(outside)}
        task = meta["task_records"][env_id]
        for key in ("episode_steps", "failure_label", "success", "final_recovery_condition", "planned_push_count", "push_event_count"):
            if own[key] != task[key]:
                mismatches.append({"env_id": env_id, "field": key, "independent": own[key], "task": task[key]})
        task_steps = [e["recovery_control_step"] for e in task["push_events"]]
        if recovery_steps != task_steps:
            mismatches.append({"env_id": env_id, "field": "recovery_control_steps", "independent": recovery_steps, "task": task_steps})
        if [e["step"] for e in events] != [e["control_step"] for e in task["push_events"]]:
            mismatches.append({"env_id": env_id, "field": "actual_write_event_steps"})
        results.append(own)
    reproduced = meta.get("formal_first_seed_task_records_equal")
    report = {"diagnostic_only": True, "enters_qualification": False, "profile": meta["profile"],
              "status": "PASS" if not mismatches and reproduced is not False else "FAIL",
              "formal_first_seed_task_records_equal": reproduced,
              "capture_sha256": sha(directory / "capture.json"), "raw_sha256": meta["raw_sha256"],
              "independent_timing_matches_task": not mismatches, "mismatches": mismatches,
              "stationary_at_end_n_over_N": [sum(row["final_recovery_condition"] for row in results), 64],
              "fall_count": sum(row["fall_label"] is not None for row in results),
              "out_of_bounds_count": sum(row["out_of_bounds"] for row in results),
              "complete_recovery_n_over_N": None if meta["profile"] == "no_push" else [sum(row["success"] for row in results), 64],
              "episodes": results}
    write(directory / "independent.json", report)
    return report


def capture_profile(args, manifest, original, profile, source_seed, checkpoint_hash):
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.isaac_lab_arrays import as_torch
    from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint
    from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
    from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed
    from true_rl_agent.tasks.push_recovery.runtime import (physical_runtime_config_from_manifest,
        instantiate_physical_push_configs, make_physical_push_environment, push_runtime_state, push_policy_io_evidence)

    output = args.output / profile
    output.mkdir()
    config = physical_runtime_config_from_manifest(manifest, profile)
    if config is None:
        raise ValueError("diagnostic requires Push v2")
    cfg_args = SimpleNamespace(seed=source_seed, num_envs=64, device=args.device)
    torch.manual_seed(source_seed)
    env_cfg, agent_cfg = instantiate_physical_push_configs(config, cfg_args, evaluation=True)
    base = make_physical_push_environment(env_cfg)
    wrapped = None
    try:
        abi = push_policy_io_evidence(base, manifest)
        agent_cfg.device = args.device
        wrapped = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        learner = rsl_rl_agent_configuration(config["runtime_adapter_id"]).verify_runner(runner)
        loaded = load_inference_checkpoint(runner, str(args.checkpoint))
        policy = runner.get_inference_policy(device=base.device)
        seed = derive_runner_seed(base_seed=source_seed, split_name="public_dev", run_index=1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        base.seed(seed)
        state = push_runtime_state(base)
        state.begin_evaluation("public_dev", seed)
        obs = wrapped.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        robot = base.scene["robot"]
        cpu = lambda value: as_torch(value).detach().cpu().numpy().copy()
        initial_xy, origins = cpu(robot.data.root_pos_w)[:, :2], cpu(base.scene.env_origins)
        alive = np.ones(64, dtype=bool)
        snapshots, writes = [], []
        interval = False
        original_compute, original_apply = base.termination_manager.compute, base.event_manager.apply
        original_write = robot.write_root_velocity_to_sim

        def compute():
            row = {"valid": alive.copy(), "steps": cpu(base.episode_length_buf),
                   "position": cpu(robot.data.root_pos_w), "quaternion_xyzw": cpu(robot.data.root_quat_w),
                   "linear": cpu(robot.data.root_lin_vel_w), "angular": cpu(robot.data.root_ang_vel_w)}
            result = original_compute()
            row["done"] = cpu(result).astype(bool)
            snapshots.append(row)
            alive[:] &= ~row["done"]
            return result

        def apply(mode, *values, **kwargs):
            nonlocal interval
            interval = mode == "interval"
            try:
                return original_apply(mode, *values, **kwargs)
            finally:
                interval = False

        def velocity_write(value, env_ids=None, **kwargs):
            if not interval:
                return original_write(value, env_ids=env_ids, **kwargs)
            ids = np.arange(64) if env_ids is None else np.arange(64)[env_ids] if isinstance(env_ids, slice) else cpu(env_ids).astype(int)
            before = np.concatenate((cpu(robot.data.root_lin_vel_w), cpu(robot.data.root_ang_vel_w)), axis=1)[ids]
            steps, desired = cpu(base.episode_length_buf)[ids], cpu(value)
            result = original_write(value, env_ids=env_ids, **kwargs)
            after = np.concatenate((cpu(robot.data.root_lin_vel_w), cpu(robot.data.root_ang_vel_w)), axis=1)[ids]
            for j, env_id in enumerate(ids):
                if alive[env_id]:
                    writes.append({"env_id": int(env_id), "step": int(steps[j]), "before": before[j].tolist(),
                                   "desired": desired[j].tolist(), "after": after[j].tolist(),
                                   "verified": bool(np.isfinite(after[j]).all() and np.max(abs(after[j] - desired[j])) <= state.parameters["injection_tolerance_mps"])})
            return result

        base.termination_manager.compute, base.event_manager.apply = compute, apply
        robot.write_root_velocity_to_sim = velocity_write
        try:
            for _ in range(int(base.max_episode_length)):
                with torch.inference_mode():
                    actions = policy(obs)
                obs = wrapped.step(actions)[0]
                if not alive.any():
                    break
        finally:
            base.termination_manager.compute, base.event_manager.apply = original_compute, original_apply
            robot.write_root_velocity_to_sim = original_write
        np.savez_compressed(output / "raw.npz", **{key: np.stack([row[key] for row in snapshots]) for key in snapshots[0]})
        records = state.episode_records()
        write(output / "capture.json", {"diagnostic_only": True, "enters_qualification": False,
            "profile": profile, "checkpoint_sha256": checkpoint_hash, "checkpoint_loading": loaded,
            "manifest_hash": manifest.manifest_hash, "policy_io": abi, "constructed_learner": learner,
            "run_index": 1, "base_seed": source_seed, "derived_public_dev_seed": seed,
            "dt": float(base.step_dt), "max_episode_length": int(base.max_episode_length),
            "parameters": dict(base.cfg.push_protocol_parameters), "initial_xy": initial_xy.tolist(), "origins": origins.tolist(),
            "quaternion_order": "xyzw", "capture_point": "before TerminationManager.compute; after physics and counter increment",
            "first_episode_mask": "own mask from returned termination flags; independently checked offline",
            "raw_sha256": sha(output / "raw.npz"), "writes": writes, "task_records": records,
            "formal_first_seed_task_records_equal": all(record == {key: old[key] for key in record}
                for record, old in zip(records, original, strict=True)) if profile != "no_push" else None})
        return recompute(output)
    finally:
        (wrapped or base).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="mode", required=True)
    offline = commands.add_parser("recompute")
    offline.add_argument("--capture-dir", type=Path, required=True)
    run = commands.add_parser("capture")
    for name in ("source-job", "formal-dev-summary", "manifest", "checkpoint", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "recompute":
        report = recompute(args.capture_dir)
        print(json.dumps({key: value for key, value in report.items() if key != "episodes"}))
        return 0 if report["status"] == "PASS" else 1
    job, formal = read(args.source_job), read(args.formal_dev_summary)
    argv = job["argv"]
    if (flag(argv, "--evaluation-phase") != "public_dev" or int(flag(argv, "--num-envs")) != 64
            or flag(argv, "--profile") not in {"strong_push", "recovery_v2"}):
        raise ValueError("requires an original 64-env public_dev job")
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    manifest = load_resolved_cell_manifest(args.manifest.resolve(), expected_manifest_hash=flag(argv, "--resolved-cell-manifest-hash"),
        expected_task_id="unitree_push_recovery", expected_base_task_protocol_hash=flag(argv, "--base-task-protocol-hash"))
    checkpoint_hash = flag(argv, "--checkpoint-sha256")
    if (sha(args.checkpoint) != checkpoint_hash or formal["checkpoint_safety_report"]["checkpoint_hash"] != checkpoint_hash
            or formal["resolved_cell_manifest_hash"] != manifest.manifest_hash or formal["evaluation_phase"] != "public_dev"
            or flag(argv, "--evaluation-binding-hash") != formal["evaluation_binding"]["evaluation_binding_hash"]):
        raise ValueError("checkpoint, original public_dev, and manifest identities differ")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "sources.json", {"diagnostic_only": True, "enters_qualification": False,
        "sources": {name: {"path": str(getattr(args, name).resolve()), "sha256": sha(getattr(args, name))}
                    for name in ("source_job", "formal_dev_summary", "manifest", "checkpoint")}, "script_sha256": sha(__file__)})
    from isaaclab.app import AppLauncher
    from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path, close_isaac_app_on_success
    launch = argparse.Namespace(headless=True, device=args.device)
    bind_isaac_log_path(launch, args.output / "result.json")
    app = AppLauncher(launch, multi_gpu=False).app
    code = 1
    try:
        from true_rl_agent.runner.checkpoint import build_checkpoint_safety_report
        safety = build_checkpoint_safety_report(args.checkpoint, expected_checkpoint_hash=checkpoint_hash)
        if safety.status != "pass":
            raise ValueError("checkpoint safety check failed")
        original = formal["task_evidence"]["seed_summaries"][0]["per_episode"]
        reports = [capture_profile(args, manifest, original, profile, int(flag(argv, "--seed")), checkpoint_hash)
                   for profile in (flag(argv, "--profile"), "no_push")]
        code = 0 if all(report["status"] == "PASS" for report in reports) else 1
        write(args.output / "result.json", {"diagnostic_only": True, "enters_qualification": False,
              "status": "PASS" if code == 0 else "FAIL",
              "profiles": [{k: v for k, v in report.items() if k != "episodes"} for report in reports]})
    except Exception:
        import traceback
        traceback.print_exc()
        write(args.output / "error.json", {"diagnostic_only": True, "traceback": traceback.format_exc()})
    return close_isaac_app_on_success(app, exit_code=code)


if __name__ == "__main__":
    raise SystemExit(main())

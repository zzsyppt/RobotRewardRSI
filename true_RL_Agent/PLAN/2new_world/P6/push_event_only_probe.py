"""G1 event wiring only, under an explicit independent diagnostic timing protocol."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isaaclab.app import AppLauncher
from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path, close_isaac_app_on_success

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.num_envs, args.seed = 64, 731
args.output.mkdir(parents=True, exist_ok=False)
bind_isaac_log_path(args, args.output / "result.json")
report = {"schema_version": "true_rl_agent.p6.push_event_only_diagnostic.v1",
    "purpose": "G1 velocity-write and next-physics evidence; no task success rate or qualification",
    "policy": "zero_action", "diagnostic_seed": args.seed, "started_at_unix": time.time(),
    "formal_protocol_modified": False, "training_or_holdout_consumed": False}
app = AppLauncher(args, multi_gpu=False).app
env, code = None, 1
try:
    import torch
    from true_rl_agent.composition import TaskCompositionContext
    from true_rl_agent.runner.checkpoint import file_sha256
    from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
    from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol, physical_public_runtime_dynamics
    from true_rl_agent.tasks.push_recovery.runtime import (
        physical_runtime_config_from_manifest, instantiate_physical_push_configs,
        make_physical_push_environment, push_policy_io_evidence, push_runtime_state,
    )

    provider = PushRecoveryCompositionProvider()
    composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": "push.g1_37dof.recovery_v2",
        "execution_device": args.device}, context=TaskCompositionContext(project_root=ROOT, output_dir=args.output / "composition"))
    manifest = composition.resolved_cell_manifest
    config = physical_runtime_config_from_manifest(manifest, "recovery_v2")
    original = PushRecoveryProtocol.from_mapping(config["push_protocol"])
    resolved = original.resolve(config["push_robot_scale_m"])
    dt = manifest.robot_task_binding["task_projection"]["push_abi_facts"]["control"]["control_dt_s"]
    unit = dt / resolved["time_scale_s"]
    diagnostic = replace(original, protocol_id="p6_g1_event_only_early_timing_diagnostic_v1", protocol_state="draft",
        first_push_time_scale=2 * unit, second_push_time_scale=6 * unit,
        recovery_window_time_scale=2 * unit, stable_dwell_time_scale=unit)
    config = {**config, "push_protocol": diagnostic.to_dict()}
    report.update(source_manifest_hash=manifest.manifest_hash, source_protocol_hash=original.protocol_hash,
        diagnostic_protocol_hash=diagnostic.protocol_hash, diagnostic_protocol=diagnostic.to_dict(),
        overrides={key: {"source": original.to_dict()[key], "diagnostic": value}
            for key, value in diagnostic.to_dict().items() if original.to_dict()[key] != value},
        diagnostic_runtime=physical_public_runtime_dynamics(config), probe_source_sha256=file_sha256(Path(__file__)))
    torch.manual_seed(args.seed)
    env_cfg, _ = instantiate_physical_push_configs(config, args, evaluation=True)
    env = make_physical_push_environment(env_cfg)
    report["policy_io"] = push_policy_io_evidence(env, manifest)
    state = push_runtime_state(env.unwrapped)
    state.begin_evaluation("public_dev", args.seed)
    env.reset()
    assert max(state.event_times) + 2 <= 12, "diagnostic events must permit next-physics observation within the bound"
    action = torch.zeros((args.num_envs, env.unwrapped.action_manager.total_action_dim), device=args.device)
    with torch.inference_mode():
        for _ in range(12):
            env.step(action)
    episodes = state.episode_records()
    event_fields = ("event_index", "control_step", "time_s", "requested_delta_v_world_mps", "observed_delta_v_world_mps",
        "velocity_before_world_6d", "velocity_after_write_world_6d", "velocity_after_next_physics_step_world_6d", "injection_verified")
    events = [{"env_id": episode["env_id"], "episode_index": episode["episode_index"], "stratum": episode["stratum"],
        **{key: event[key] for key in event_fields}} for episode in episodes for event in episode["push_events"]]
    repeated = sum(episode["stratum"] == "repeated" and len(episode["push_events"]) == 2 for episode in episodes)
    planned = sum(episode["planned_push_count"] for episode in episodes)
    checks = {"at_least_one_episode_received_two_events": repeated > 0,
        "all_injections_verified": bool(events) and all(event["injection_verified"] for event in events),
        "all_next_physics_observed": bool(events) and all(event["velocity_after_next_physics_step_world_6d"] is not None for event in events),
        "z_and_angular_components_preserved": bool(events) and all(event["velocity_before_world_6d"][2:] == event["velocity_after_write_world_6d"][2:] for event in events),
        "all_next_physics_finite": bool(events) and all(event["velocity_after_next_physics_step_world_6d"] is not None
            and all(math.isfinite(value) for value in event["velocity_after_next_physics_step_world_6d"]) for event in events)}
    report.update(checks=checks, events=events, scheduled_control_steps=list(state.event_times),
        event_coverage={"planned": planned, "observed": len(events), "repeated_episodes_with_two_events": repeated,
            "status": "complete" if len(events) == planned else "insufficient"},
        execution_cost={"num_envs": args.num_envs, "vector_env_step_calls": 12, "control_steps": args.num_envs * 12},
        recovery_measurements_excluded="Diagnostic window/dwell do not qualify original-task recovery")
    assert all(checks.values()), "event wiring diagnostic failed; inspect recorded checks and actual event evidence"
    report["status"], code = "PASS", 0
except Exception:
    report.update(status="FAIL", error=traceback.format_exc())
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    report["ended_at_unix"] = time.time()
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    sys.exit(close_isaac_app_on_success(app, exit_code=code))

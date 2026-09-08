"""Same-state/action migration trace through each checkout's training factory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--initial-state", type=Path)
parser.add_argument("--device", required=True)
parser.add_argument("--nominal-mass", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / "src"))
from isaaclab.app import AppLauncher
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    stair_runtime_projection_from_manifest,
    resolve_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)
from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success

filename = (
    "stair_unified_protocol_v2.json"
    if args.robot == "g1"
    else "stair_go2_12dof_unified_protocol_v2.json"
)
config = StairClimbingTaskConfig.defaults(args.source_root / "configs/tasks" / filename)
cell = resolve_stair_cell_manifest(
    cell_variant_id=config.cell_variant_id,
    config=config,
    plugin=StairClimbingTaskPlugin(config),
)
config = replace(
    config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash
)
projection = stair_runtime_projection_from_manifest(cell)
official = resolve_official_stair_reference_program(
    config, project_root=args.source_root, resolved_cell_manifest=cell
)
current = importlib.metadata.version("rsl-rl-lib").startswith("5.")
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(
    headless=True,
    device=args.device,
    multi_gpu=False,
    kit_args="--/log/file=" + str((args.output / "result.kit.log").resolve()),
).app
env, code = None, 1
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_train_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
        apply_unified_initial_terrain_assignment,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_TRAIN_GYM_ID

    def tensor(x):
        return getattr(x, "torch", x)

    def cpu(x):
        value = tensor(x)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        return value.numpy().copy()  # Native PhysX view methods return Warp arrays.

    cfg = build_stair_train_env_cfg(
        None,
        seed=config.train_seed,
        device=args.device,
        reward_program=official.reward_program,
        unified_protocol=config.unified_protocol,
        runtime_projection=projection,
    )
    cfg.scene.num_envs = 16
    # Isolate the deterministic observation ABI from independent RNG streams.
    # This is a diagnostic configuration; training keeps its frozen noise.
    cfg.observations.policy.enable_corruption = False
    # The nominal comparison isolates physics; the default records the actual
    # bound startup distribution and its same-seed random draws as well.
    if args.nominal_mass:
        cfg.events.add_base_mass = None
    register_production_stair_envs()
    env = gym.make(STAIR_TRAIN_GYM_ID, cfg=cfg)
    base = env.unwrapped
    apply_unified_initial_terrain_assignment(
        base, seed=config.train_seed, profile=config.unified_training_profile
    )
    env.reset()
    robot = base.scene["robot"]
    contact = base.scene[projection.contact_sensor_name]
    resolved = resolve_stair_runtime_projection(
        projection, robot=robot, contact_sensor=contact
    )
    physics = robot.root_physx_view
    physics_facts = {
        "masses": cpu(physics.get_masses()),
        "inertias": cpu(physics.get_inertias()),
        "materials": cpu(physics.get_material_properties()),
    }
    np.savez(args.output / "physics_facts.npz", **physics_facts)

    def capture():
        quat = cpu(robot.data.root_quat_w)
        if current:
            quat = quat[:, [3, 0, 1, 2]]
        return {
            "root_pos": cpu(robot.data.root_pos_w),
            "root_quat_wxyz": quat,
            "root_lin_vel": cpu(robot.data.root_lin_vel_w),
            "root_ang_vel": cpu(robot.data.root_ang_vel_w),
            "joint_pos": cpu(robot.data.joint_pos),
            "joint_vel": cpu(robot.data.joint_vel),
            "support_pos": cpu(robot.data.body_pos_w)[:, resolved.support_body_ids],
            "support_force": cpu(contact.data.net_forces_w)[
                :, resolved.contact_support_body_ids
            ],
            "commanded_base_vel": cpu(
                base.command_manager.get_command("base_velocity")
            ),
        }

    if args.initial_state is not None:
        initial = np.load(args.initial_state)
        quat = initial["root_quat_wxyz"]
        if current:
            quat = quat[:, [1, 2, 3, 0]]
        pose = torch.tensor(
            np.concatenate((initial["root_pos"], quat), axis=1), device=args.device
        )
        velocity = torch.tensor(
            np.concatenate((initial["root_lin_vel"], initial["root_ang_vel"]), axis=1),
            device=args.device,
        )
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(velocity)
        robot.write_joint_state_to_sim(
            torch.tensor(initial["joint_pos"], device=args.device),
            torch.tensor(initial["joint_vel"], device=args.device),
        )
        tensor(base.command_manager.get_command("base_velocity")).copy_(
            torch.tensor(initial["commanded_base_vel"], device=args.device)
        )
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.scene.update(0.0)
    np.savez(args.output / "initial.npz", **capture())
    traces = {name: [value] for name, value in capture().items()}
    rewards, dones, policy_observations, reward_terms, reward_inputs = (
        [],
        [],
        [],
        {},
        {},
    )
    with torch.inference_mode():
        for index in range(32):
            # Fixed small actions in the shared joint order, independent of RNG.
            actions = torch.full(
                (16, base.action_manager.total_action_dim),
                0.02 * np.sin(index / 4.0),
                device=args.device,
            )
            observations, reward, terminated, truncated, _ = env.step(actions)
            policy_observations.append(cpu(observations["policy"]))
            term = base._true_rl_stair_complete_reward_term
            for name, value in term.dispatcher.last_evaluation.weighted_terms.items():
                reward_terms.setdefault(name, []).append(cpu(value))
            for name in term.last_inputs.field_names:
                reward_inputs.setdefault(name, []).append(
                    cpu(getattr(term.last_inputs, name))
                )
            for name, value in capture().items():
                traces[name].append(value)
            rewards.append(cpu(reward))
            dones.append(cpu(terminated) | cpu(truncated))
    np.savez(
        args.output / "trace.npz",
        **{k: np.stack(v) for k, v in traces.items()},
        reward=np.stack(rewards),
        done=np.stack(dones),
        policy_observation=np.stack(policy_observations),
    )
    np.savez(
        args.output / "reward_terms.npz",
        **{k: np.stack(v) for k, v in reward_terms.items()},
    )
    np.savez(
        args.output / "reward_inputs.npz",
        **{k: np.stack(v) for k, v in reward_inputs.items()},
    )
    report = {
        "status": "PASS",
        "robot": args.robot,
        "current_runtime": current,
        "physics_dt": cfg.sim.dt,
        "control_dt": base.step_dt,
        "decimation": cfg.decimation,
        "train_seed": config.train_seed,
        "source_root": str(args.source_root),
        "observation_corruption_disabled_for_probe": True,
        "startup_mass_randomization_disabled_for_probe": args.nominal_mass,
        "cell_manifest_hash": cell.manifest_hash,
        "robot_asset": cfg.scene.robot.spawn.usd_path,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "resolved_projection": resolved.to_summary_dict(),
        "terrain_origins": cpu(base.scene.terrain.env_origins).tolist(),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", flush=True)
    code = 0
except Exception:
    report = {"status": "FAIL", "error": traceback.format_exc()}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    sys.exit(close_isaac_app_on_success(app, exit_code=code))

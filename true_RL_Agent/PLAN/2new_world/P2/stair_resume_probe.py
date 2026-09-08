"""Real native checkpoint round trip and continuation on both Stair robots."""

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isaaclab.app import AppLauncher
from true_rl_agent.runner.isaac_app_lifecycle import (
    bind_isaac_log_path,
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
filename = (
    "stair_unified_protocol_v2.json"
    if args.robot == "g1"
    else "stair_go2_12dof_unified_protocol_v2.json"
)
config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / filename)
cell = resolve_stair_cell_manifest(
    cell_variant_id=config.cell_variant_id,
    config=config,
    plugin=StairClimbingTaskPlugin(config),
)
config = replace(
    config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash
)
official = resolve_official_stair_reference_program(
    config, project_root=ROOT, resolved_cell_manifest=cell
)
bind_isaac_log_path(args, args.output / "result.json")
app = AppLauncher(args, multi_gpu=False).app
env, code = None, 1
report = {
    "robot": args.robot,
    "purpose": "P2 diagnostic native resume",
    "cell": cell.to_dict(),
}
try:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.rsl_rl_checkpoint import (
        load_training_checkpoint,
        load_inference_checkpoint,
    )
    from true_rl_agent.robots.isaaclab_locomotion import (
        isaaclab_locomotion_runtime_spec,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_train_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_training import (
        configure_stair_rsl_rl_agent,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
        apply_unified_initial_terrain_assignment,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_TRAIN_GYM_ID
    from true_rl_agent.tasks.stair_climbing.runtime_projection import (
        stair_runtime_projection_from_manifest,
    )

    projection = stair_runtime_projection_from_manifest(cell)
    cfg = build_stair_train_env_cfg(
        None,
        seed=config.train_seed,
        device=args.device,
        reward_program=official.reward_program,
        unified_protocol=config.unified_protocol,
        runtime_projection=projection,
    )
    cfg.scene.num_envs = 16
    register_production_stair_envs()
    env = gym.make(STAIR_TRAIN_GYM_ID, cfg=cfg)
    apply_unified_initial_terrain_assignment(
        env.unwrapped, seed=config.train_seed, profile=config.unified_training_profile
    )
    env.reset()
    agent = isaaclab_locomotion_runtime_spec(
        projection.runtime_adapter_id
    ).instantiate_runner_cfg()
    configuration = configure_stair_rsl_rl_agent(
        agent,
        configuration=config.training_agent_configuration(),
        seed=config.train_seed,
        device=args.device,
        num_steps_per_env=8,
        max_iterations=2,
        save_interval=1,
        run_name=args.output.name,
        experiment_name="p2_resume",
        execution_contract=config.training_execution_contract(),
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(
        wrapped, agent.to_dict(), log_dir=str(args.output / "train"), device=args.device
    )
    report["constructed_learner"] = config.training_agent_configuration().verify_runner(runner)
    original_lr = runner.alg.learning_rate
    runner.alg.learning_rate = original_lr * 2
    try:
        config.training_agent_configuration().verify_runner(runner)
    except ValueError:
        report["constructed_learner_drift_rejected"] = True
    else:
        raise AssertionError("learner drift was not rejected")
    runner.alg.learning_rate = original_lr
    runner.learn(2)
    checkpoint = args.output / "initial.pt"
    runner.save(str(checkpoint))
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    observations = wrapped.get_observations()
    with torch.inference_mode():
        expected = runner.get_inference_policy()(observations).clone()
    resumed = OnPolicyRunner(
        wrapped,
        agent.to_dict(),
        log_dir=str(args.output / "resume"),
        device=args.device,
    )
    load_training_checkpoint(resumed, checkpoint)
    assert resumed.current_learning_iteration == saved["iter"]
    torch.testing.assert_close(
        resumed.alg.actor.state_dict(), saved["actor_state_dict"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        resumed.alg.critic.state_dict(), saved["critic_state_dict"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        resumed.alg.optimizer.state_dict(),
        saved["optimizer_state_dict"],
        atol=0,
        rtol=0,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            resumed.get_inference_policy()(observations), expected, atol=0, rtol=0
        )
    resumed.learn(1)
    resumed.save(str(args.output / "resumed.pt"))
    load_inference_checkpoint(runner, args.output / "resumed.pt")
    with torch.inference_mode():
        obs, _ = wrapped.reset()
        for _ in range(16):
            obs, _, _, _ = wrapped.step(runner.get_inference_policy()(obs))
    report.update(
        status="PASS",
        train_control_steps=16 * 8 * 3,
        evaluation_vector_steps=16,
        optimizer_exact_roundtrip=True,
        actor_critic_normalizer_exact_roundtrip=True,
        saved_iteration=int(saved["iter"]),
        action_dim=wrapped.num_actions,
        observation_shapes={k: list(v.shape) for k, v in obs.items()},
        configuration=configuration,
    )
    code = 0
except Exception:
    report.update(status="FAIL", error=traceback.format_exc())
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(report["status"], flush=True)
    sys.exit(close_isaac_app_on_success(app, exit_code=code))

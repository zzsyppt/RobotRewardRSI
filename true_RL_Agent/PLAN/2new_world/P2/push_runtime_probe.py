"""G1-29 target-runtime training/resume and standing-push diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isaaclab.app import AppLauncher
from true_rl_agent.runner.isaac_app_lifecycle import (
    bind_isaac_log_path,
    close_isaac_app_on_success,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (
    build_current_isaac_lab_execution_environment_manifest,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.num_envs, args.seed = 16, 731
args.output.mkdir(parents=True, exist_ok=True)
bind_isaac_log_path(args, args.output / "result.json")
report = {
    "environment": build_current_isaac_lab_execution_environment_manifest().to_dict()
}
app = AppLauncher(args, multi_gpu=False).app
code, env = 1, None
try:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.isaac_lab_arrays import as_torch
    from true_rl_agent.backends.rsl_rl_checkpoint import (
        load_training_checkpoint,
        load_inference_checkpoint,
    )
    from true_rl_agent.robots.unitree_locomotion import (
        instantiate_g1_29_environment_cfg,
        instantiate_g1_29_runner_cfg,
    )
    from true_rl_agent.robots.rsl_rl_runtime import (
        G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION,
    )
    from true_rl_agent.candidates.components import TrainingComponents, DiagnosticTerm
    from true_rl_agent.tasks.push_recovery.runtime_adapter import (
        CandidateRuntimeAdapter,
        CandidateRewardVecEnvWrapper,
    )
    from true_rl_agent.tasks.push_recovery.isaac_config import load_runtime_config
    from true_rl_agent.tasks.push_recovery.isaac_overlay import (
        apply_push_recovery_overlay,
    )

    cfg = instantiate_g1_29_environment_cfg(device=args.device, num_envs=args.num_envs)
    config = load_runtime_config(ROOT / "configs/tasks/push_recovery.yaml")
    apply_push_recovery_overlay(cfg, config, args)
    env = gym.make(config["task"]["gym_id"], cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    adapter = CandidateRuntimeAdapter(
        manifest=None,
        components=TrainingComponents(
            name="p2_runtime_probe",
            diagnostics=[DiagnosticTerm("height_probe", lambda x: x.base_height)],
        ),
        num_envs=args.num_envs,
        planned_training_steps=24,
        diagnostic_interval_steps=1,
    )
    wrapped = CandidateRewardVecEnvWrapper(wrapped, adapter)
    agent = instantiate_g1_29_runner_cfg()
    G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION.apply_to(agent)
    agent.seed, agent.device, agent.num_steps_per_env = args.seed, args.device, 8
    agent.experiment_name = "p2_push_runtime_probe"
    runner = OnPolicyRunner(
        wrapped, agent.to_dict(), log_dir=str(args.output / "train"), device=args.device
    )
    report["constructed_learner"] = G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION.verify_runner(runner)
    runner.learn(2)
    checkpoint = args.output / "initial.pt"
    runner.save(str(checkpoint))
    obs = wrapped.get_observations()
    with torch.inference_mode():
        before = runner.get_inference_policy()(obs).clone()
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    resumed = OnPolicyRunner(
        wrapped,
        agent.to_dict(),
        log_dir=str(args.output / "resume"),
        device=args.device,
    )
    load_training_checkpoint(resumed, checkpoint)
    assert resumed.current_learning_iteration == saved["iter"]
    torch.testing.assert_close(
        resumed.alg.optimizer.state_dict(),
        saved["optimizer_state_dict"],
        atol=0,
        rtol=0,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            resumed.get_inference_policy()(obs), before, atol=0, rtol=0
        )
    resumed.learn(1)
    resumed.save(str(args.output / "resumed.pt"))
    load_inference_checkpoint(runner, args.output / "resumed.pt")
    policy = runner.get_inference_policy()
    velocities = []
    with torch.inference_mode():
        obs, _ = wrapped.reset()
        for _ in range(100):
            obs, reward, done, extras = wrapped.step(policy(obs))
            velocities.append(
                as_torch(env.unwrapped.scene["robot"].data.root_lin_vel_w).clone()
            )
    velocity = torch.stack(velocities)
    report.update(
        status="PASS",
        action_dim=wrapped.num_actions,
        observation_shapes={k: list(v.shape) for k, v in obs.items()},
        train_control_steps=16 * 8 * 3,
        evaluation_vector_steps=100,
        velocity_delta_near_push_max=float(
            (velocity[48:55, :, :2] - velocity[47:54, :, :2]).norm(dim=-1).max()
        ),
        adapter=adapter.public_summary(),
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

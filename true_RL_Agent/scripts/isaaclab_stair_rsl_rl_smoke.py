#!/usr/bin/env python3
"""Run one bounded RSL-RL update through the production Stair reward path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.runner.checkpoint import file_sha256  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (  # noqa: E402
    StairGpuRuntimeBinding,
    add_stair_gpu_process_binding_arguments,
    attach_stair_gpu_runtime_evidence,
    capture_stair_gpu_compute_context,
    validate_stair_gpu_process_binding_invocation,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)

SMOKE_NUM_ENVS = 16
SMOKE_NUM_STEPS_PER_ENV = 8
SMOKE_MAX_ITERATIONS = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    output_action = parser.add_argument("--output", type=Path)
    checkpoint_action = parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--nvidia-smi-executable")
    AppLauncher.add_app_launcher_args(parser)
    add_stair_gpu_process_binding_arguments(parser)
    output_action.required = True
    checkpoint_action.required = True
    parser.set_defaults(headless=True)
    return parser


args = _build_parser().parse_args()
gpu_runtime_binding = validate_stair_gpu_process_binding_invocation(
    args,
    environment=os.environ,
)


def _bound_nvidia_smi(
    runtime: StairGpuRuntimeBinding | None,
) -> str | None:
    if runtime is None:
        return None
    raw = args.nvidia_smi_executable
    if not isinstance(raw, str) or not raw:
        raise ValueError("bound Stair smoke requires --nvidia-smi-executable")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("bound Stair smoke nvidia-smi must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("bound Stair smoke nvidia-smi must be executable")
    return str(resolved)


bound_nvidia_smi = _bound_nvidia_smi(gpu_runtime_binding)
config = StairClimbingTaskConfig.defaults(
    PROJECT_ROOT / STAIR_PROTOCOL_V2_RELATIVE_PATH
)
official = resolve_official_stair_reference_program(
    config,
    project_root=PROJECT_ROOT,
)
simulation_app = (
    AppLauncher(args).app
    if gpu_runtime_binding is None
    else AppLauncher(args, multi_gpu=False).app
)


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    G1RoughPPORunnerCfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (  # noqa: E402
    get_stair_complete_reward_term,
    run_stair_reward_activation_preflight,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (  # noqa: E402
    build_stair_train_env_cfg,
    register_production_stair_envs,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_training import (  # noqa: E402
    configure_stair_rsl_rl_agent,
    verify_stair_rsl_rl_observation_groups,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_TRAIN_GYM_ID,
)


def main() -> int:
    output_path = args.output.resolve()
    checkpoint_output = args.checkpoint_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    env = None
    compute_context_observation = None
    report: dict[str, Any]
    try:
        if checkpoint_output.exists():
            raise FileExistsError("Stair RSL-RL smoke checkpoint must be fresh")
        execution = config.protocol_v2.to_public_dict()["training_execution"]
        torch.backends.cuda.matmul.allow_tf32 = bool(
            execution["torch_matmul_allow_tf32"]
        )
        torch.backends.cudnn.allow_tf32 = bool(execution["cudnn_allow_tf32"])
        torch.backends.cudnn.deterministic = bool(
            execution["cudnn_deterministic"]
        )
        torch.backends.cudnn.benchmark = bool(execution["cudnn_benchmark"])

        register_production_stair_envs()
        env_cfg = build_stair_train_env_cfg(
            config.protocol_v1.to_dict(),
            seed=config.train_seed,
            device=str(args.device),
            reward_program=official.reward_program,
            runner_owned_training_distribution=(
                config.protocol_v2.runner_owned_training_distribution
            ),
        )
        env_cfg.scene.num_envs = SMOKE_NUM_ENVS
        if env_cfg.recorders is not None:
            raise RuntimeError("Stair RSL-RL smoke unexpectedly installed a recorder")

        agent_cfg = G1RoughPPORunnerCfg()
        effective_agent = configure_stair_rsl_rl_agent(
            agent_cfg,
            configuration=config.training_agent_configuration(),
            seed=config.train_seed,
            device=str(args.device),
            num_steps_per_env=SMOKE_NUM_STEPS_PER_ENV,
            max_iterations=SMOKE_MAX_ITERATIONS,
            save_interval=1,
            run_name="phase_d_final_smoke",
            experiment_name="true_rl_agent_stair_phase_d_final_smoke",
            execution_contract=execution,
        )

        env = gym.make(STAIR_TRAIN_GYM_ID, cfg=env_cfg, render_mode=None)
        env.reset()
        base_env = env.unwrapped
        if list(base_env.reward_manager.active_terms) != ["complete_reward"]:
            raise RuntimeError("Stair RSL-RL smoke found a second reward authority")
        term_cfg = base_env.reward_manager.get_term_cfg("complete_reward")
        if float(term_cfg.weight) != 1.0:
            raise RuntimeError("Stair RSL-RL smoke reward-manager weight drifted")
        complete_term = get_stair_complete_reward_term(base_env)
        determinism = run_stair_reward_activation_preflight(
            env,
            reward_program=official.reward_program,
        )

        run_dir = output_path.parent / f"stair-rsl-smoke-{uuid.uuid4().hex}"
        log_dir = run_dir / "rsl_rl"
        wrapped_env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        resolved_observation_groups = verify_stair_rsl_rl_observation_groups(
            runner,
            execution_contract=execution,
        )
        effective_agent["constructed_learner"] = config.training_agent_configuration().verify_runner(runner)
        effective_agent["resolved_observation_groups"] = (
            resolved_observation_groups
        )
        runner.add_git_repo_to_log(__file__)
        ppo_evidence: dict[str, Any] = {
            "process_env_step_calls": 0,
            "all_rewards_equal_dispatcher_result": True,
            "first_reward_sha256": None,
            "first_reward_mean": None,
            "episode_reward_log_checks": 0,
            "episode_reward_log_values": [],
            "episode_reward_log_max_abs_error": 0.0,
        }
        ppo_episode_sums = torch.zeros(
            SMOKE_NUM_ENVS,
            dtype=torch.float32,
            device=agent_cfg.device,
        )
        original_process_env_step = runner.alg.process_env_step

        def audited_process_env_step(
            observations: Any,
            rewards: Any,
            dones: Any,
            extras: Any,
        ) -> Any:
            evaluation = complete_term.dispatcher.last_evaluation
            if evaluation is None:
                raise RuntimeError("PPO received a reward without dispatcher evidence")
            expected = evaluation.total_rate * float(base_env.step_dt)
            if not torch.equal(rewards, expected):
                ppo_evidence["all_rewards_equal_dispatcher_result"] = False
                raise RuntimeError("PPO consumed a reward outside the Stair authority")
            if ppo_evidence["first_reward_sha256"] is None:
                reward_bytes = rewards.detach().cpu().contiguous().numpy().tobytes()
                ppo_evidence["first_reward_sha256"] = hashlib.sha256(
                    reward_bytes
                ).hexdigest()
                ppo_evidence["first_reward_mean"] = round(
                    float(rewards.detach().mean().cpu().item()),
                    8,
                )
            ppo_episode_sums.add_(rewards)
            done_mask = dones.reshape(-1) > 0
            if bool(done_mask.any().item()):
                log_payload = extras.get("log") if isinstance(extras, dict) else None
                logged_reward = (
                    log_payload.get("Episode_Reward/complete_reward")
                    if isinstance(log_payload, dict)
                    else None
                )
                if logged_reward is None:
                    raise RuntimeError("done transition omitted complete reward log")
                expected_episode_reward = (
                    ppo_episode_sums[done_mask].mean()
                    / float(base_env.max_episode_length_s)
                )
                actual_episode_reward = torch.as_tensor(
                    logged_reward,
                    dtype=torch.float32,
                    device=rewards.device,
                ).reshape(-1).mean()
                error = float(
                    (actual_episode_reward - expected_episode_reward)
                    .abs()
                    .detach()
                    .cpu()
                    .item()
                )
                if error > 1.0e-5:
                    raise RuntimeError("RSL episode reward log differs from PPO sum")
                ppo_evidence["episode_reward_log_checks"] += 1
                ppo_evidence["episode_reward_log_values"].append(
                    round(float(actual_episode_reward.detach().cpu().item()), 8)
                )
                ppo_evidence["episode_reward_log_max_abs_error"] = max(
                    ppo_evidence["episode_reward_log_max_abs_error"],
                    error,
                )
                ppo_episode_sums[done_mask] = 0.0
            ppo_evidence["process_env_step_calls"] += 1
            return original_process_env_step(observations, rewards, dones, extras)

        runner.alg.process_env_step = audited_process_env_step
        base_env.episode_length_buf[0] = int(base_env.max_episode_length) - 1
        runner.learn(
            num_learning_iterations=SMOKE_MAX_ITERATIONS,
            init_at_random_ep_len=False,
        )
        if hasattr(runner, "writer") and runner.writer is not None:
            runner.writer.flush()

        expected_control_steps = SMOKE_NUM_STEPS_PER_ENV * SMOKE_MAX_ITERATIONS
        runtime = complete_term.runtime_summary()
        expected_dispatcher_calls = expected_control_steps + int(
            determinism["real_transition_dispatcher_calls"]
        )
        if runtime["dispatcher_calls"] != expected_dispatcher_calls:
            raise RuntimeError("RSL-RL smoke dispatcher call count did not close")
        if ppo_evidence["process_env_step_calls"] != expected_control_steps:
            raise RuntimeError("RSL-RL smoke PPO reward call count did not close")
        if ppo_evidence["episode_reward_log_checks"] <= 0:
            raise RuntimeError("RSL-RL smoke observed no complete episode reward log")
        expected_env_steps = SMOKE_NUM_ENVS * expected_control_steps
        if int(runner.tot_timesteps) != expected_env_steps:
            raise RuntimeError("RSL-RL smoke timestep accounting did not close")

        if gpu_runtime_binding is not None:
            assert bound_nvidia_smi is not None
            compute_context_observation = capture_stair_gpu_compute_context(
                gpu_runtime_binding,
                nvidia_smi=bound_nvidia_smi,
            )

        checkpoints = sorted(log_dir.rglob("model_*.pt"))
        if not checkpoints:
            raise RuntimeError("RSL-RL smoke produced no checkpoint")
        source_checkpoint = checkpoints[-1]
        shutil.copy2(source_checkpoint, checkpoint_output)
        source_hash = file_sha256(source_checkpoint)
        checkpoint_hash = file_sha256(checkpoint_output)
        if source_hash is None or source_hash != checkpoint_hash:
            raise RuntimeError("RSL-RL smoke checkpoint copy hash mismatch")

        event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            raise RuntimeError("RSL-RL smoke produced no TensorBoard event file")
        event_accumulator = EventAccumulator(str(log_dir))
        event_accumulator.Reload()
        scalar_tags = sorted(event_accumulator.Tags().get("scalars", []))
        required_tags = {"Perf/total_fps", "Loss/learning_rate"}
        if not required_tags.issubset(scalar_tags):
            raise RuntimeError("RSL-RL smoke log is missing required scalar tags")
        episode_reward_events = event_accumulator.Scalars(
            "Episode_Reward/complete_reward"
        )
        if len(episode_reward_events) != 1:
            raise RuntimeError("RSL-RL smoke episode reward event count drifted")
        expected_event_reward = sum(
            ppo_evidence["episode_reward_log_values"]
        ) / len(ppo_evidence["episode_reward_log_values"])
        tensorboard_reward = float(episode_reward_events[0].value)
        if abs(tensorboard_reward - expected_event_reward) > 1.0e-5:
            raise RuntimeError("TensorBoard reward differs from episode reward log")

        report = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.phase_d_final_rsl_rl_smoke.v1"
            ),
            "status": "pass",
            "protocol_v2_hash": config.protocol_v2.protocol_hash,
            "reference_program": official.reward_program.to_public_dict(),
            "random_initialization": {
                "mode": "random",
                "seed": config.train_seed,
                "checkpoint_input": None,
            },
            "training": {
                "num_envs": SMOKE_NUM_ENVS,
                "num_steps_per_env": SMOKE_NUM_STEPS_PER_ENV,
                "max_iterations": SMOKE_MAX_ITERATIONS,
                "env_steps": expected_env_steps,
                "effective_agent": effective_agent,
                "training_recorder_enabled": False,
                "episode_phase_probe": {
                    "forced_timeout_env_count": 1,
                    "runner_random_episode_phase_disabled": True,
                    "production_random_episode_phase": bool(
                        execution["init_at_random_episode_length"]
                    ),
                    "reward_log_normalization_s": float(
                        base_env.max_episode_length_s
                    ),
                },
            },
            "reward_manager_active_terms": list(
                base_env.reward_manager.active_terms
            ),
            "determinism": determinism,
            "runtime": runtime,
            "ppo_consumption": ppo_evidence,
            "logging": {
                "log_dir": str(log_dir),
                "event_files": [str(path) for path in event_files],
                "scalar_tags": scalar_tags,
                "complete_reward_scalar": tensorboard_reward,
                "expected_complete_reward_scalar": expected_event_reward,
            },
            "checkpoint": {
                "path": str(checkpoint_output),
                "checksum": checkpoint_hash,
                "source_path": str(source_checkpoint),
                "source_checksum": source_hash,
            },
        }
    except Exception as exc:  # noqa: BLE001 - probe must preserve one report
        report = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.phase_d_final_rsl_rl_smoke.v1"
            ),
            "status": "fail",
            "protocol_v2_hash": config.protocol_v2.protocol_hash,
            "reference_program": official.reward_program.to_public_dict(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    if gpu_runtime_binding is not None:
        report = attach_stair_gpu_runtime_evidence(
            report,
            gpu_runtime_binding,
            compute_context_observation=compute_context_observation,
        )
    report["report_hash"] = canonical_json_hash(report)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "status": report["status"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    exit_code = main()
    exit_code = close_isaac_app_on_success(
        simulation_app,
        exit_code=exit_code,
    )
    os._exit(exit_code)

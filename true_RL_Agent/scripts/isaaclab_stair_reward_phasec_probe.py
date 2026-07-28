#!/usr/bin/env python3
"""Run a bounded real-Isaac smoke for the Stair complete-reward runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-num-envs", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


args = _build_parser().parse_args()
if args.probe_num_envs <= 0 or args.probe_steps <= 0:
    raise ValueError("probe envs and steps must be positive")
config = StairClimbingTaskConfig.defaults(
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
)
official = resolve_official_stair_reference_program(
    config,
    project_root=PROJECT_ROOT,
)
simulation_app = AppLauncher(args).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import RewardTermCfg  # noqa: E402

from true_rl_agent.tasks.stair_climbing import (  # noqa: E402
    isaaclab_runtime_env as runtime_env_module,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (  # noqa: E402
    get_stair_complete_reward_term,
    run_stair_reward_activation_preflight,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (  # noqa: E402
    build_stair_train_env_cfg,
    register_production_stair_envs,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_TRAIN_GYM_ID,
)


def _sentinel_base_reward(env):
    return torch.full(
        (int(env.num_envs),),
        1.0e6,
        dtype=torch.float32,
        device=env.device,
    )


def main() -> int:
    env = None
    output_path = Path(args.output)
    try:
        register_production_stair_envs()
        upstream_builder = runtime_env_module.build_stair_env_cfg

        def upstream_with_sentinel(*builder_args, **builder_kwargs):
            cfg = upstream_builder(*builder_args, **builder_kwargs)
            cfg.rewards.phase_c_sentinel_base_reward = RewardTermCfg(
                func=_sentinel_base_reward,
                weight=1.0e6,
            )
            return cfg

        with patch.object(
            runtime_env_module,
            "build_stair_env_cfg",
            side_effect=upstream_with_sentinel,
        ):
            env_cfg = build_stair_train_env_cfg(
                config.protocol_v1.to_dict(),
                seed=config.train_seed,
                device=str(args.device),
                reward_program=official.reward_program,
            )
        sentinel_removed = not hasattr(
            env_cfg.rewards,
            "phase_c_sentinel_base_reward",
        )
        if not sentinel_removed:
            raise RuntimeError("sentinel base reward survived complete replacement")
        env_cfg.scene.num_envs = args.probe_num_envs
        env = gym.make(STAIR_TRAIN_GYM_ID, cfg=env_cfg, render_mode=None)
        env.reset()
        base = env.unwrapped
        term = get_stair_complete_reward_term(base)
        determinism = run_stair_reward_activation_preflight(
            env,
            reward_program=official.reward_program,
        )
        equivalence = official.reward_program.contract.to_public_dict()[
            "reference_equivalence"
        ]
        absolute_tolerance = float(equivalence["absolute_tolerance"])
        relative_tolerance = float(equivalence["relative_tolerance"])
        action_dim = int(
            config.protocol_v1.to_dict()["policy_compatibility"]["action"][
                "total_dim"
            ]
        )
        actions = torch.zeros(
            (args.probe_num_envs, action_dim),
            dtype=torch.float32,
            device=base.device,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        first_step_checks = None
        for step in range(args.probe_steps):
            _, reward, terminated, truncated, _ = env.step(actions)
            evaluation = term.dispatcher.last_evaluation
            if evaluation is None:
                raise RuntimeError("dispatcher did not retain its authoritative result")
            expected_reward = evaluation.total_rate * float(base.step_dt)
            step_reward = base.reward_manager._step_reward[:, 0]
            reward_equal = torch.equal(reward, expected_reward)
            step_rate_close = torch.allclose(
                step_reward,
                evaluation.total_rate,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
            if not reward_equal or not step_rate_close:
                reward_error = float(
                    (reward - expected_reward).abs().max().detach().cpu().item()
                )
                step_rate_error = float(
                    (step_reward - evaluation.total_rate)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                    .item()
                )
                raise RuntimeError(
                    "Stair reward outlets differ on one transition: "
                    f"step={step} reward_equal={reward_equal} "
                    f"step_rate_close={step_rate_close} "
                    f"reward_max_abs_error={reward_error} "
                    f"step_rate_max_abs_error={step_rate_error}"
                )
            if step == 0:
                done = terminated | truncated
                if bool(done.any().item()):
                    raise RuntimeError("first probe transition unexpectedly terminated")
                episode_sum = base.reward_manager._episode_sums[
                    "complete_reward"
                ]
                if not torch.equal(episode_sum, reward):
                    raise RuntimeError(
                        "Stair episode accumulator differs from returned reward"
                    )
                first_step_checks = {
                    "returned_reward_equals_dispatcher_rate_times_dt": True,
                    "reward_manager_step_rate_within_protocol_tolerance": True,
                    "episode_accumulator_equals_returned_reward": True,
                }
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        runtime = term.runtime_summary()
        expected_dispatcher_calls = args.probe_steps + int(
            determinism["real_transition_dispatcher_calls"]
        )
        if runtime["dispatcher_calls"] != expected_dispatcher_calls:
            raise RuntimeError("dispatcher call count differs from probe steps")
        if runtime["term_calls"] != (
            expected_dispatcher_calls * runtime["declared_term_count"]
        ):
            raise RuntimeError("reward term call count is not closed")
        report = {
            "schema_version": "true_rl_agent.task.stair_climbing.phase_c_smoke.v1",
            "status": "pass",
            "protocol_v2_hash": config.protocol_v2.protocol_hash,
            "reference_program": official.reward_program.to_public_dict(),
            "num_envs": args.probe_num_envs,
            "control_steps": args.probe_steps,
            "env_steps_per_second": round(
                args.probe_num_envs * args.probe_steps / elapsed,
                3,
            ),
            "reward_manager_active_terms": list(
                base.reward_manager.active_terms
            ),
            "sentinel_base_reward_removed_before_env": sentinel_removed,
            "training_recorder_enabled": env_cfg.recorders is not None,
            "first_step_checks": first_step_checks,
            "determinism": determinism,
            "runtime": runtime,
        }
        report["report_hash"] = canonical_json_hash(report)
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema_version": "true_rl_agent.task.stair_climbing.phase_c_smoke.v1",
            "status": "fail",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "status": report["status"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    exit_code = close_isaac_app_on_success(
        simulation_app,
        exit_code=main(),
    )
    os._exit(exit_code)

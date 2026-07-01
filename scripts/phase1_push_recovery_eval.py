#!/usr/bin/env python3
"""Evaluate the Phase 1 standing push-recovery environment.

This script launches Isaac Sim, applies a project-local standing/push overlay to
the official Unitree G1 task, runs a zero or random policy, classifies failures,
and writes a rich JSON summary for later LLM-driven diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--profile", default="strong_push")
    parser.add_argument("--task", default=None)
    parser.add_argument("--policy", default="random", choices=("random", "zero", "checkpoint"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--status-file", default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from humanoid_agent.phase1_push_recovery.config import (  # noqa: E402
    config_for_profile,
    load_config,
    sha256_file,
)
from humanoid_agent.phase1_push_recovery.diagnostics import build_diagnostic_hints  # noqa: E402
from humanoid_agent.phase1_push_recovery.failure_classifier import (  # noqa: E402
    classify_episode,
    summarize_condition_flags,
    summarize_failures,
)
from humanoid_agent.phase1_push_recovery.isaac_overlay import (  # noqa: E402
    apply_phase1_overlay,
    as_pair,
)
from humanoid_agent.immutable_runner.safe_checkpoint import assert_safe_checkpoint  # noqa: E402


def _all_finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_tree(item) for item in value)
    return True


def _float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _make_actions(policy: str, num_envs: int, action_dim: int, device: torch.device, scale: float) -> torch.Tensor:
    if policy == "zero":
        return torch.zeros((num_envs, action_dim), device=device)
    if policy == "random":
        return (2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0) * scale
    raise ValueError(f"Unsupported policy: {policy}")


def _action_dim(env: gym.Env) -> int:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_manager"):
        return int(unwrapped.action_manager.total_action_dim)
    return int(gym.spaces.flatdim(unwrapped.single_action_space))


def _reset_env(env: Any) -> tuple[Any, dict[str, Any]]:
    return env.reset()


def _step_env(env: Any, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    step_result = env.step(actions)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, extras = step_result
        return obs, reward, terminated.bool(), truncated.bool(), extras
    if len(step_result) == 4:
        obs, reward, dones, extras = step_result
        dones_bool = dones.bool()
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
        if time_outs is None:
            truncated = torch.zeros_like(dones_bool, dtype=torch.bool)
        else:
            truncated = time_outs.bool()
        terminated = dones_bool & ~truncated
        return obs, reward, terminated, truncated, extras
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _has_nonzero_push(config: dict[str, Any]) -> bool:
    velocity_range = config["push"]["velocity_range_mps"]
    return any(abs(float(bound)) > 1.0e-9 for axis in ("x", "y") for bound in velocity_range[axis])


def _apply_eval_overrides(env_cfg: Any, config: dict[str, Any]) -> None:
    evaluation_cfg = config.get("evaluation", {})
    if bool(evaluation_cfg.get("disable_vendor_fall_terminations", False)):
        for name in ("base_height", "bad_orientation"):
            if hasattr(env_cfg.terminations, name):
                setattr(env_cfg.terminations, name, None)


def _run_rollout(env: Any, config: dict[str, Any], args: argparse.Namespace, inference_policy: Any = None) -> dict[str, Any]:
    unwrapped = env.unwrapped
    device = torch.device(unwrapped.device)
    num_envs = int(unwrapped.num_envs)
    action_dim = _action_dim(env)
    step_dt = float(unwrapped.step_dt)
    max_steps = args.max_steps if args.max_steps is not None else int(round(config["environment"]["episode_length_s"] / step_dt))
    max_steps = int(max_steps)

    policy_defaults = config.get("policy_defaults", {})
    if args.action_scale is not None:
        action_scale = float(args.action_scale)
    elif args.policy == "random":
        action_scale = float(policy_defaults.get("random_action_scale", 1.0))
    else:
        action_scale = float(policy_defaults.get("zero_action_scale", 0.0))

    obs, reset_extras = _reset_env(env)
    observation_finite = _all_finite_tree(obs)
    reward_finite = True

    robot = unwrapped.scene["robot"]
    initial_xy = robot.data.root_pos_w[:, :2].clone()

    inf = torch.full((num_envs,), float("inf"), device=device)
    neg_inf = torch.full((num_envs,), float("-inf"), device=device)
    min_base_height = inf.clone()
    max_pitch = neg_inf.clone()
    min_pitch = inf.clone()
    max_abs_pitch = torch.zeros(num_envs, device=device)
    max_abs_roll = torch.zeros(num_envs, device=device)
    final_xy_drift = torch.zeros(num_envs, device=device)
    cumulative_reward = torch.zeros(num_envs, device=device)
    done_step = torch.full((num_envs,), max_steps, dtype=torch.long, device=device)
    terminated_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    timed_out_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)

    max_pre_push_speed = torch.zeros(num_envs, device=device)
    max_post_push_speed = torch.zeros(num_envs, device=device)
    push_interval = as_pair(config["push"]["interval_range_s"])
    push_start_s = push_interval[0]
    push_evidence_cfg = config["push"]["evidence"]
    pre_window_s = float(push_evidence_cfg["pre_window_s"])
    post_window_s = float(push_evidence_cfg["post_window_s"])
    single_step_window_s = as_pair(push_evidence_cfg.get("single_step_window_s", [push_start_s, push_start_s]))
    max_push_step_velocity_delta = torch.zeros(num_envs, device=device)
    push_step_of_max_delta = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    nonzero_push_configured = _has_nonzero_push(config)

    last_action = torch.zeros((num_envs, action_dim), device=device)
    action_l2_sum = torch.zeros(num_envs, device=device)
    action_rate_l2_sum = torch.zeros(num_envs, device=device)
    action_samples = torch.zeros(num_envs, device=device)

    first_step_seen = False
    for step in range(max_steps):
        t = step * step_dt
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) > 0:
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w
            root_vel = robot.data.root_lin_vel_w
            pre_step_root_lin_vel_xy = root_vel[:, :2].clone()
            roll, pitch, _ = euler_xyz_from_quat(root_quat)
            planar_speed = torch.linalg.vector_norm(root_vel[:, :2], dim=1)
            drift = torch.linalg.vector_norm(root_pos[:, :2] - initial_xy, dim=1)

            min_base_height[active_idx] = torch.minimum(min_base_height[active_idx], root_pos[active_idx, 2])
            max_pitch[active_idx] = torch.maximum(max_pitch[active_idx], pitch[active_idx])
            min_pitch[active_idx] = torch.minimum(min_pitch[active_idx], pitch[active_idx])
            max_abs_pitch[active_idx] = torch.maximum(max_abs_pitch[active_idx], torch.abs(pitch[active_idx]))
            max_abs_roll[active_idx] = torch.maximum(max_abs_roll[active_idx], torch.abs(roll[active_idx]))
            final_xy_drift[active_idx] = drift[active_idx]

            if t <= max(push_start_s, pre_window_s):
                max_pre_push_speed[active_idx] = torch.maximum(max_pre_push_speed[active_idx], planar_speed[active_idx])
            if push_start_s <= t <= push_start_s + post_window_s:
                max_post_push_speed[active_idx] = torch.maximum(max_post_push_speed[active_idx], planar_speed[active_idx])
        else:
            pre_step_root_lin_vel_xy = robot.data.root_lin_vel_w[:, :2].clone()

        if inference_policy is not None:
            with torch.inference_mode():
                actions = inference_policy(obs)
        else:
            actions = _make_actions(args.policy, num_envs, action_dim, device, action_scale)
        action_norm = torch.linalg.vector_norm(actions, dim=1)
        action_rate = torch.linalg.vector_norm(actions - last_action, dim=1)
        if not first_step_seen:
            action_rate = torch.zeros_like(action_rate)
            first_step_seen = True
        last_action = actions

        active_float = active.to(dtype=torch.float32)
        action_l2_sum += action_norm * active_float
        action_rate_l2_sum += action_rate * active_float
        action_samples += active_float

        active_before_step = active.clone()
        obs, reward, terminated, truncated, _extras = _step_env(env, actions)
        observation_finite = observation_finite and _all_finite_tree(obs)
        reward_finite = reward_finite and bool(torch.isfinite(reward).all().item())
        cumulative_reward += reward * active_float

        if single_step_window_s[0] <= t <= single_step_window_s[1]:
            post_step_root_lin_vel_xy = robot.data.root_lin_vel_w[:, :2]
            step_delta = torch.linalg.vector_norm(post_step_root_lin_vel_xy - pre_step_root_lin_vel_xy, dim=1)
            update_mask = active_before_step & (step_delta > max_push_step_velocity_delta)
            max_push_step_velocity_delta[update_mask] = step_delta[update_mask]
            push_step_of_max_delta[update_mask] = step + 1

        done = (terminated | truncated) & active
        if bool(done.any().item()):
            done_step[done] = step + 1
            terminated_once |= terminated & done
            timed_out_once |= truncated & done
            active[done] = False
        if not bool(active.any().item()):
            break

    # Final state for environments that never terminated.
    active_idx = active.nonzero(as_tuple=False).squeeze(-1)
    if len(active_idx) > 0:
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        roll, pitch, _ = euler_xyz_from_quat(root_quat)
        drift = torch.linalg.vector_norm(root_pos[:, :2] - initial_xy, dim=1)
        min_base_height[active_idx] = torch.minimum(min_base_height[active_idx], root_pos[active_idx, 2])
        max_pitch[active_idx] = torch.maximum(max_pitch[active_idx], pitch[active_idx])
        min_pitch[active_idx] = torch.minimum(min_pitch[active_idx], pitch[active_idx])
        max_abs_pitch[active_idx] = torch.maximum(max_abs_pitch[active_idx], torch.abs(pitch[active_idx]))
        max_abs_roll[active_idx] = torch.maximum(max_abs_roll[active_idx], torch.abs(roll[active_idx]))
        final_xy_drift[active_idx] = drift[active_idx]

    thresholds = config["success_thresholds"]
    max_records = int(config.get("metrics", {}).get("max_per_episode_records", 24))
    per_episode: list[dict[str, Any]] = []
    labels: list[str] = []
    classifications: list[dict[str, Any]] = []
    successes = 0

    for env_id in range(num_envs):
        samples = max(1.0, _float(action_samples[env_id]))
        metrics = {
            "env_id": env_id,
            "episode_steps": int(done_step[env_id].detach().cpu().item()),
            "terminated": bool(terminated_once[env_id].detach().cpu().item()),
            "timed_out": bool(timed_out_once[env_id].detach().cpu().item()),
            "min_base_height_m": _round_float(_float(min_base_height[env_id])),
            "max_pitch_rad": _round_float(_float(max_pitch[env_id])),
            "min_pitch_rad": _round_float(_float(min_pitch[env_id])),
            "max_abs_pitch_rad": _round_float(_float(max_abs_pitch[env_id])),
            "max_abs_roll_rad": _round_float(_float(max_abs_roll[env_id])),
            "final_xy_drift_m": _round_float(_float(final_xy_drift[env_id])),
            "return": _round_float(_float(cumulative_reward[env_id])),
            "mean_action_l2": _round_float(_float(action_l2_sum[env_id]) / samples),
            "mean_action_rate_l2": _round_float(_float(action_rate_l2_sum[env_id]) / samples),
            "pre_push_peak_speed_mps": _round_float(_float(max_pre_push_speed[env_id])),
            "post_push_peak_speed_mps": _round_float(_float(max_post_push_speed[env_id])),
            "observed_push_delta_mps": _round_float(
                _float(max_post_push_speed[env_id] - max_pre_push_speed[env_id])
            ),
        }
        classification = classify_episode(metrics, thresholds)
        classifications.append(classification)
        metrics["failure_label"] = classification["label"]
        metrics["success"] = classification["success"]
        metrics["classification_rules"] = classification["ordered_rules"]
        labels.append(classification["label"])
        successes += int(classification["success"])
        if len(per_episode) < max_records:
            per_episode.append(metrics)

    failure_summary = summarize_failures(labels)
    condition_flag_summary = summarize_condition_flags(classifications)
    deltas = (max_post_push_speed - max_pre_push_speed).detach().cpu()
    min_delta = float(push_evidence_cfg["min_observed_delta_mps"])
    min_fraction = float(push_evidence_cfg["min_env_fraction_with_delta"])
    fraction_with_delta = float((deltas >= min_delta).to(dtype=torch.float32).mean().item())
    legacy_peak_window_gate_pass = fraction_with_delta >= min_fraction
    single_step_deltas = max_push_step_velocity_delta.detach().cpu()
    min_single_step_delta = float(push_evidence_cfg["min_single_step_delta_mps"])
    min_single_step_fraction = float(push_evidence_cfg["min_env_fraction_with_single_step_delta"])
    single_step_fraction = float((single_step_deltas >= min_single_step_delta).to(dtype=torch.float32).mean().item())
    single_step_gate_pass = single_step_fraction >= min_single_step_fraction
    push_gate_pass = single_step_gate_pass if nonzero_push_configured else None

    episode_steps_values = [int(done_step[i].detach().cpu().item()) for i in range(num_envs)]
    returns = [_float(cumulative_reward[i]) for i in range(num_envs)]
    action_l2_values = [_float(action_l2_sum[i]) / max(1.0, _float(action_samples[i])) for i in range(num_envs)]
    action_rate_values = [
        _float(action_rate_l2_sum[i]) / max(1.0, _float(action_samples[i])) for i in range(num_envs)
    ]

    summary = {
        "schema_version": "humanoid_agent.phase1.push_recovery.summary.v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "seed": args.seed,
        },
        "task": {
            "gym_id": args.task,
            "robot": config["task"]["robot"],
            "implementation_strategy": config["task"]["implementation_strategy"],
            "vendor_env": config["task"]["vendor_env"],
        },
        "policy": {
            "kind": args.policy,
            "checkpoint": args.checkpoint,
            "action_scale": action_scale,
            "action_dim": action_dim,
        },
        "configuration": {
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(args.config),
            "active_profile": config.get("active_profile"),
            "standing_command": config["standing_command"],
            "push": config["push"],
            "success_thresholds": thresholds,
        },
        "environment": {
            "num_envs": num_envs,
            "device": str(device),
            "step_dt": step_dt,
            "max_steps": max_steps,
            "episode_length_s": config["environment"]["episode_length_s"],
        },
        "evidence": {
            "reset_ok": True,
            "observation_finite": observation_finite,
            "reward_finite": reward_finite,
            "push_effect": {
                "mechanism": config["push"]["mechanism"],
                "nonzero_push_configured": nonzero_push_configured,
                "expected_interval_range_s": list(push_interval),
                "single_step_window_s": list(single_step_window_s),
                "mean_single_step_delta_mps": _round_float(float(single_step_deltas.mean().item())),
                "max_single_step_delta_mps": _round_float(float(single_step_deltas.max().item())),
                "min_single_step_delta_mps": _round_float(min_single_step_delta),
                "env_fraction_with_single_step_delta": _round_float(single_step_fraction),
                "min_env_fraction_with_single_step_delta": _round_float(min_single_step_fraction),
                "single_step_gate_pass": single_step_gate_pass,
                "mean_step_index_of_max_delta": _round_float(float(push_step_of_max_delta.to(dtype=torch.float32).mean().item())),
                "gate_pass": push_gate_pass,
                "gate_reason": (
                    "single_step_velocity_jump_detected"
                    if nonzero_push_configured
                    else "nonzero push is not configured; push-effect gate intentionally skipped"
                ),
                "legacy_peak_speed_window": {
                    "pre_window_s": pre_window_s,
                    "post_window_s": post_window_s,
                    "mean_pre_push_peak_speed_mps": _round_float(float(max_pre_push_speed.mean().item())),
                    "mean_post_push_peak_speed_mps": _round_float(float(max_post_push_speed.mean().item())),
                    "mean_observed_delta_mps": _round_float(float(deltas.mean().item())),
                    "min_observed_delta_mps": _round_float(min_delta),
                    "env_fraction_with_delta": _round_float(fraction_with_delta),
                    "min_env_fraction_with_delta": _round_float(min_fraction),
                    "legacy_gate_pass": legacy_peak_window_gate_pass,
                },
            },
        },
        "aggregate": {
            "episodes": num_envs,
            "success_count": successes,
            "success_rate": _round_float(successes / num_envs),
            "failure_breakdown": failure_summary,
            "condition_flag_breakdown": condition_flag_summary,
        },
        "secondary_metrics": {
            "mean_episode_steps": _round_float(_mean([float(x) for x in episode_steps_values])),
            "mean_return": _round_float(_mean(returns)),
            "mean_action_l2": _round_float(_mean(action_l2_values)),
            "mean_action_rate_l2": _round_float(_mean(action_rate_values)),
            "mean_min_base_height_m": _round_float(float(min_base_height.mean().item())),
            "mean_max_abs_pitch_rad": _round_float(float(max_abs_pitch.mean().item())),
            "mean_max_abs_roll_rad": _round_float(float(max_abs_roll.mean().item())),
            "mean_final_xy_drift_m": _round_float(float(final_xy_drift.mean().item())),
        },
        "per_episode": per_episode,
        "reset_extras_keys": sorted(reset_extras.keys()) if isinstance(reset_extras, dict) else [],
        "errors": [],
    }
    summary["diagnostic_hints"] = build_diagnostic_hints(summary)
    return summary


def main() -> int:
    config_path = Path(args_cli.config)
    config = config_for_profile(load_config(config_path), args_cli.profile)
    task = args_cli.task or config["task"]["gym_id"]
    args_cli.task = task

    summary_path = Path(args_cli.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args_cli.status_file:
        Path(args_cli.status_file).parent.mkdir(parents=True, exist_ok=True)

    env = None
    base_env = None
    summary: dict[str, Any]
    try:
        if args_cli.policy == "checkpoint" and not args_cli.checkpoint:
            raise ValueError("--checkpoint is required when --policy checkpoint")
        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="env_cfg_entry_point")
        apply_phase1_overlay(env_cfg, config, args_cli)
        _apply_eval_overrides(env_cfg, config)
        base_env = gym.make(task, cfg=env_cfg, render_mode=None)
        if isinstance(base_env.unwrapped, DirectMARLEnv):
            base_env = multi_agent_to_single_agent(base_env)
        inference_policy = None
        if args_cli.policy == "checkpoint":
            agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            if hasattr(agent_cfg, "device") and args_cli.device is not None:
                agent_cfg.device = args_cli.device
            env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            assert_safe_checkpoint(args_cli.checkpoint)
            runner.load(args_cli.checkpoint)
            inference_policy = runner.get_inference_policy(device=env.unwrapped.device)
        else:
            env = base_env
        summary = _run_rollout(env, config, args_cli, inference_policy=inference_policy)
    except Exception as exc:  # noqa: BLE001 - top-level summary must capture failures.
        summary = {
            "schema_version": "humanoid_agent.phase1.push_recovery.summary.v1",
            "status": "fail",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": {"cwd": os.getcwd(), "argv": sys.argv, "seed": args_cli.seed},
            "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        }
    finally:
        if env is not None:
            env.close()
        elif base_env is not None:
            base_env.close()

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args_cli.status_file:
        Path(args_cli.status_file).write_text(str(summary.get("status", "fail")) + "\n", encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_path), "status": summary.get("status")}, ensure_ascii=False))
    exit_code = 0 if summary.get("status") == "pass" else 1
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

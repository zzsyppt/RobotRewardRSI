#!/usr/bin/env python3
"""Verify canonical Stair R0 against official Isaac Lab reward terms."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

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
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-num-envs", type=int, default=8)
    parser.add_argument("--normal-steps", type=int, default=4)
    output_action = parser.add_argument("--output")
    AppLauncher.add_app_launcher_args(parser)
    output_action.required = True
    parser.set_defaults(headless=True)
    return parser


args = _build_parser().parse_args()
if args.probe_num_envs < 4 or args.normal_steps <= 0:
    raise ValueError("oracle probe requires at least four envs and one normal step")
config = StairClimbingTaskConfig.defaults(
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
)
official = resolve_official_stair_reference_program(
    config,
    project_root=PROJECT_ROOT,
)
simulation_app = AppLauncher(args).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from true_rl_agent.tasks.stair_climbing.isaaclab_reward_oracle import (  # noqa: E402
    STAIR_REWARD_ORACLE_EXTRAS_KEY,
    STAIR_REWARD_ORACLE_GYM_ID,
    STAIR_REWARD_ORACLE_SCHEMA_VERSION,
    build_stair_reward_oracle_env_cfg,
    register_stair_reward_oracle_env,
)


def _new_metric() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "mismatch_count": 0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
    }


def _update_metric(
    metric: dict[str, Any],
    observed: Any,
    expected: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if tuple(observed.shape) != tuple(expected.shape):
        raise RuntimeError("oracle comparison tensor shapes differ")
    absolute_error = (observed - expected).abs()
    relative_error = absolute_error / expected.abs().clamp(min=1.0e-12)
    close = torch.isclose(
        observed,
        expected,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )
    metric["sample_count"] += int(observed.numel())
    metric["mismatch_count"] += int((~close).sum().detach().cpu().item())
    metric["max_absolute_error"] = max(
        float(metric["max_absolute_error"]),
        float(absolute_error.max().detach().cpu().item()),
    )
    metric["max_relative_error"] = max(
        float(metric["max_relative_error"]),
        float(relative_error.max().detach().cpu().item()),
    )


def _scenario_masks(payload: dict[str, Any]) -> dict[str, Any]:
    masks = payload["termination_masks"]
    timeout = masks["time_out"].to(dtype=torch.bool)
    contact = masks["base_contact"].to(dtype=torch.bool)
    terminated = masks["terminated_non_timeout"].to(dtype=torch.bool)
    if not torch.equal(contact, terminated):
        raise RuntimeError("oracle non-timeout mask is not exactly base_contact")
    return {
        "normal": ~(timeout | contact),
        "timeout": timeout & ~contact,
        "base_contact": contact & ~timeout,
        "timeout_and_base_contact": timeout & contact,
    }


def _force_terminal_mix(base: Any, *, z_offset: float) -> None:
    timeout_only = torch.tensor([0], dtype=torch.long, device=base.device)
    contact_only = torch.tensor([1], dtype=torch.long, device=base.device)
    concurrent = torch.tensor([2], dtype=torch.long, device=base.device)
    base.episode_length_buf[timeout_only] = int(base.max_episode_length) - 1
    base.episode_length_buf[concurrent] = int(base.max_episode_length) - 1
    low_ids = torch.cat((contact_only, concurrent))
    robot = base.scene["robot"]
    pose = robot.data.root_state_w[low_ids, :7].detach().clone()
    origins = base.scene.terrain.env_origins[low_ids]
    pose[:, :2] = origins[:, :2]
    pose[:, 2] = origins[:, 2] + float(z_offset)
    robot.write_root_pose_to_sim(pose, env_ids=low_ids)
    robot.write_root_velocity_to_sim(
        torch.zeros((len(low_ids), 6), dtype=pose.dtype, device=pose.device),
        env_ids=low_ids,
    )
    base.sim.forward()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    env = None
    output_path = Path(args.output)
    report: dict[str, Any]
    try:
        register_stair_reward_oracle_env()
        env_cfg = build_stair_reward_oracle_env_cfg(
            config.protocol_v1.to_dict(),
            seed=config.train_seed,
            device=str(args.device),
            reward_program=official.reward_program,
        )
        env_cfg.scene.num_envs = args.probe_num_envs
        env = gym.make(STAIR_REWARD_ORACLE_GYM_ID, cfg=env_cfg, render_mode=None)
        env.reset()
        base = env.unwrapped
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
        equivalence = official.reward_program.contract.to_public_dict()[
            "reference_equivalence"
        ]
        absolute_tolerance = float(equivalence["absolute_tolerance"])
        relative_tolerance = float(equivalence["relative_tolerance"])
        active_names = tuple(equivalence["active_term_names"])
        term_metrics = {
            name: {
                "raw_r0_vs_official": _new_metric(),
                "weighted_r0_vs_official": _new_metric(),
                "manager_vs_official": _new_metric(),
            }
            for name in active_names
        }
        total_metrics = {
            "r0_vs_manager_reward": _new_metric(),
            "r0_vs_manager_term_sum": _new_metric(),
            "manager_reward_vs_term_sum": _new_metric(),
        }
        scenario_counts = {
            "normal": 0,
            "timeout": 0,
            "base_contact": 0,
            "timeout_and_base_contact": 0,
        }
        post_reset_current_step = {
            "observed_env_count": 0,
            "unexpected_done_count": 0,
        }
        transition_count = 0
        manager_term_order: tuple[str, ...] | None = None

        def consume_transition() -> dict[str, Any]:
            nonlocal transition_count, manager_term_order
            payload = base.extras.get(STAIR_REWARD_ORACLE_EXTRAS_KEY)
            if not isinstance(payload, dict):
                raise RuntimeError("Stair reward oracle transition evidence is missing")
            if payload.get("schema_version") != STAIR_REWARD_ORACLE_SCHEMA_VERSION:
                raise RuntimeError("Stair reward oracle transition schema drifted")
            observed_order = tuple(payload["manager_term_order"])
            if manager_term_order is None:
                manager_term_order = observed_order
            elif manager_term_order != observed_order:
                raise RuntimeError("official manager reward order changed during probe")
            if tuple(payload["active_term_names"]) != active_names:
                raise RuntimeError("oracle active term order differs from the contract")
            masks = _scenario_masks(payload)
            for name, mask in masks.items():
                scenario_counts[name] += int(mask.sum().detach().cpu().item())
            for name in active_names:
                values = payload["terms"][name]
                _update_metric(
                    term_metrics[name]["raw_r0_vs_official"],
                    values["r0_raw"],
                    values["official_raw"],
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _update_metric(
                    term_metrics[name]["weighted_r0_vs_official"],
                    values["r0_weighted_rate"],
                    values["official_weighted_rate"],
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _update_metric(
                    term_metrics[name]["manager_vs_official"],
                    values["manager_weighted_rate"],
                    values["official_weighted_rate"],
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            for name, rates in payload["zero_weight_manager_rates"].items():
                if bool((rates != 0.0).any().detach().cpu().item()):
                    raise RuntimeError(
                        f"official zero-weight term produced manager reward: {name}"
                    )
            totals = payload["totals"]
            _update_metric(
                total_metrics["r0_vs_manager_reward"],
                totals["r0_total_rate"],
                totals["manager_reward_rate"],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            _update_metric(
                total_metrics["r0_vs_manager_term_sum"],
                totals["r0_total_rate"],
                totals["manager_term_sum_rate"],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            _update_metric(
                total_metrics["manager_reward_vs_term_sum"],
                totals["manager_reward_rate"],
                totals["manager_term_sum_rate"],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            transition_count += 1
            return {
                name: payload["termination_masks"][name]
                .to(dtype=torch.bool)
                .detach()
                .clone()
                for name in ("time_out", "base_contact", "terminated_non_timeout")
            }

        for _ in range(args.normal_steps):
            env.step(actions)
            consume_transition()

        for z_offset in (-0.35, -0.75, -1.25):
            _force_terminal_mix(base, z_offset=z_offset)
            env.step(actions)
            terminal_masks = consume_transition()
            contact_reset = terminal_masks["base_contact"]
            if bool(contact_reset.any().detach().cpu().item()):
                env.step(actions)
                post_reset_masks = consume_transition()
                unexpected_done = contact_reset & (
                    post_reset_masks["time_out"]
                    | post_reset_masks["base_contact"]
                    | post_reset_masks["terminated_non_timeout"]
                )
                observed = int(contact_reset.sum().detach().cpu().item())
                unexpected = int(unexpected_done.sum().detach().cpu().item())
                post_reset_current_step["observed_env_count"] += observed
                post_reset_current_step["unexpected_done_count"] += unexpected
                if unexpected > 0:
                    raise RuntimeError(
                        "oracle termination masks leaked across auto-reset"
                    )
            if all(count > 0 for count in scenario_counts.values()):
                break

        all_metrics = [
            metric
            for channels in term_metrics.values()
            for metric in channels.values()
        ] + list(total_metrics.values())
        mismatch_count = sum(int(item["mismatch_count"]) for item in all_metrics)
        missing_scenarios = [
            name for name, count in scenario_counts.items() if count <= 0
        ]
        if manager_term_order is None:
            raise RuntimeError("oracle did not record any manager transition")
        post_reset_checked = post_reset_current_step["observed_env_count"] > 0
        status = (
            "pass"
            if mismatch_count == 0 and not missing_scenarios and post_reset_checked
            else "fail"
        )
        report = {
            "schema_version": "true_rl_agent.task.stair_climbing.reward_equivalence_report.v1",
            "status": status,
            "task_id": STAIR_TASK_ID,
            "protocol_v2_hash": config.protocol_v2.protocol_hash,
            "base_protocol_hash": config.protocol_v1.protocol_hash,
            "reference_program": official.reward_program.to_public_dict(),
            "verified_reference_artifact": official.artifact.to_public_dict(),
            "upstream": {
                "project": equivalence["upstream_project"],
                "revision": equivalence["upstream_revision"],
                "source_path": equivalence["upstream_source_path"],
            },
            "tolerances": {
                "absolute": absolute_tolerance,
                "relative": relative_tolerance,
            },
            "probe": {
                "num_envs": args.probe_num_envs,
                "transition_count": transition_count,
                "scenario_sample_counts": scenario_counts,
                "missing_scenarios": missing_scenarios,
                "manager_term_order": list(manager_term_order),
                "post_reset_current_step": post_reset_current_step,
            },
            "term_metrics": term_metrics,
            "total_metrics": total_metrics,
            "mismatch_count": mismatch_count,
            "runtime_versions": {
                "isaac_lab": _version("isaaclab"),
                "isaaclab_tasks": _version("isaaclab-tasks"),
                "torch": torch.__version__,
            },
        }
        report["report_hash"] = canonical_json_hash(report)
        if not math.isfinite(
            max(
                float(item["max_absolute_error"])
                for item in all_metrics
            )
        ):
            raise RuntimeError("oracle report contains a non-finite error metric")
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema_version": "true_rl_agent.task.stair_climbing.reward_equivalence_report.v1",
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

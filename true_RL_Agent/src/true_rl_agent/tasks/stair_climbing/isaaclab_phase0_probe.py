"""Isaac Lab runtime evidence for the stair-climbing Phase 0 probe.

This module is imported only after Isaac Sim has started.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

import torch

from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    PHASE0_STAIR_GYM_ID,
    PHASE0_TRANSITION_EXTRAS_KEY,
    canonical_json_sha256,
    phase0_manager_term_groups,
    phase0_terminal_reason_names,
    resolve_phase0_terminal_reason,
)


def build_runtime_environment_evidence(
    *,
    raw_env: Any,
    policy_observation: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, Any]:
    """Export the resolved runtime contract, not config-level declarations."""

    robot = raw_env.scene["robot"]
    robot_schema = {
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
    }
    robot_schema["schema_hash"] = canonical_json_sha256(robot_schema)

    observation_schema = _runtime_observation_schema(raw_env)
    action_schema = _runtime_action_schema(raw_env)
    termination_schema = _runtime_termination_schema(raw_env)
    terrain = raw_env.scene.terrain
    terrain_grid_origins = _xyz_rows(terrain.terrain_origins.reshape(-1, 3))
    env_origins = _xyz_rows(terrain.env_origins)
    body_names = robot_schema["body_names"]

    return {
        "gym_id": PHASE0_STAIR_GYM_ID,
        "device": str(raw_env.device),
        "num_envs": int(raw_env.num_envs),
        "simulation_dt_s": float(raw_env.physics_dt),
        "control_dt_s": float(raw_env.step_dt),
        "policy_observation_shape": list(policy_observation.shape),
        "action_shape": list(action.shape),
        "robot_schema": robot_schema,
        "policy_observation_schema": observation_schema,
        "action_schema": action_schema,
        "termination_schema": termination_schema,
        "foot_contact_body_names": [
            name for name in body_names if name.endswith("_ankle_roll_link")
        ],
        "terrain_types": [
            int(value) for value in terrain.terrain_types.detach().cpu().tolist()
        ],
        "terrain_levels": [
            int(value) for value in terrain.terrain_levels.detach().cpu().tolist()
        ],
        "terrain_grid_origins_m": terrain_grid_origins,
        "env_origins_m": env_origins,
        "observation_corruption_enabled": bool(
            raw_env.cfg.observations.policy.enable_corruption
        ),
    }


def assert_runtime_environment_matches(
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Fail before rollout if any resolved runtime identity drifts."""

    policy = protocol["policy_compatibility"]
    evaluation = protocol["evaluation"]
    terrain = protocol["terrain"]
    expected_num_envs = int(evaluation["protocol"]["eval_num_envs"])
    expected = {
        "num_envs": expected_num_envs,
        "simulation_dt_s": policy["simulation_dt_s"],
        "control_dt_s": policy["control_dt_s"],
        "policy_observation_shape": [
            expected_num_envs,
            policy["policy_observation"]["total_dim"],
        ],
        "action_shape": [expected_num_envs, policy["action"]["total_dim"]],
        "robot_schema": policy["robot_schema"],
        "policy_observation_schema": policy["policy_observation"],
        "action_schema": policy["action"],
        "termination_schema": protocol["phase0_probe"]["termination_schema"],
        "foot_contact_body_names": [
            name
            for name in policy["robot_schema"]["body_names"]
            if name.endswith("_ankle_roll_link")
        ],
        "terrain_types": expected_terrain_types(protocol),
        "terrain_levels": [0] * expected_num_envs,
        "terrain_grid_origins_m": terrain["evaluation_grid_origins_m"],
        "env_origins_m": [
            terrain["evaluation_grid_origins_m"][terrain_type]
            for terrain_type in expected_terrain_types(protocol)
        ],
        "observation_corruption_enabled": False,
    }
    for key, expected_value in expected.items():
        if evidence[key] != expected_value:
            raise RuntimeError(f"runtime environment contract mismatch: {key}")


def run_first_episode_probe(
    *,
    env: Any,
    raw_env: Any,
    policy: Callable[[Any], torch.Tensor],
    observation: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Run exactly the first episode of every env through the frozen horizon."""

    probe = protocol["phase0_probe"]
    control_steps_limit = int(probe["control_steps"])
    terrain_types = expected_terrain_types(protocol)
    terrain_type_tensor = torch.tensor(
        terrain_types,
        dtype=torch.long,
        device=raw_env.device,
    )
    env_origins = raw_env.scene.terrain.env_origins
    robot = raw_env.scene["robot"]
    num_envs = int(raw_env.num_envs)

    terrain = protocol["terrain"]
    outcomes = protocol["episode_outcomes"]
    goal_progress = float(terrain["goal"]["progress_goal_m_from_origin"])
    fall_signal = outcomes["signals"]["fall"]
    fall_clearance_threshold = float(fall_signal["threshold"])
    fall_required_steps = int(fall_signal["consecutive_control_steps"])
    success_hold_steps = int(outcomes["success"]["top_hold_control_steps"])
    first_step_progress = float(probe["first_step_progress_m_from_origin"])
    timeout_manager_terms, terminated_manager_terms = phase0_manager_term_groups(
        protocol
    )
    timeout_manager_term_set = set(timeout_manager_terms)
    terminated_manager_term_set = set(terminated_manager_terms)
    step_heights = torch.tensor(
        [item["height_m"] for item in terrain["evaluation_bins"]],
        dtype=torch.float32,
        device=raw_env.device,
    )[terrain_type_tensor]

    active = torch.ones(num_envs, dtype=torch.bool, device=raw_env.device)
    initial_progress, initial_root_clearance = _progress_and_root_clearance(
        root_pos_w=robot.data.root_pos_w,
        env_origins=env_origins,
        step_heights=step_heights,
        protocol=protocol,
    )
    _assert_finite(initial_progress, "initial root progress")
    _assert_finite(initial_root_clearance, "initial root clearance")
    max_progress = initial_progress.clone()
    top_hold = torch.zeros(num_envs, dtype=torch.long, device=raw_env.device)
    max_top_hold = torch.zeros_like(top_hold)
    fall_streak = torch.zeros_like(top_hold)
    max_fall_streak = torch.zeros_like(top_hold)
    records: list[dict[str, Any] | None] = [None] * num_envs

    start = time.perf_counter()
    control_steps_executed = 0
    for step_index in range(control_steps_limit):
        with torch.inference_mode():
            actions = policy(observation)
            _assert_finite(actions, f"actions at step {step_index}")
            observation, reward, dones, extras = env.step(actions)
        control_steps_executed = step_index + 1
        _assert_finite(observation["policy"], f"observation at step {step_index}")
        _assert_finite(reward, f"reward at step {step_index}")
        transition = extras.pop(PHASE0_TRANSITION_EXTRAS_KEY, None)
        if not isinstance(transition, Mapping):
            raise TypeError("Phase 0 recorder did not publish post-transition evidence")
        root_pos_w = transition.get("root_pos_w")
        episode_lengths = transition.get("episode_length")
        raw_termination_masks = transition.get("termination_masks")
        if not isinstance(root_pos_w, torch.Tensor):
            raise TypeError("Phase 0 transition root position is missing")
        if not isinstance(episode_lengths, torch.Tensor):
            raise TypeError("Phase 0 transition episode length is missing")
        if not isinstance(raw_termination_masks, Mapping):
            raise TypeError("Phase 0 transition termination masks are missing")
        if tuple(root_pos_w.shape) != (num_envs, 3):
            raise RuntimeError("Phase 0 transition root position shape drifted")
        if tuple(episode_lengths.shape) != (num_envs,):
            raise RuntimeError("Phase 0 transition episode length shape drifted")
        _assert_finite(root_pos_w, f"terminal-safe root position at step {step_index}")

        termination_term_order = [
            item["manager_term"]
            for item in probe["termination_schema"]["terms"]
        ]
        if set(raw_termination_masks) != set(termination_term_order):
            raise RuntimeError("Phase 0 transition termination term set drifted")
        termination_masks: dict[str, torch.Tensor] = {}
        for name in termination_term_order:
            raw_mask = raw_termination_masks[name]
            if not isinstance(raw_mask, torch.Tensor):
                raise TypeError(f"termination mask is not a tensor: {name}")
            if tuple(raw_mask.shape) != (num_envs,):
                raise RuntimeError(f"termination mask shape drifted: {name}")
            termination_masks[name] = raw_mask.to(
                device=raw_env.device,
                dtype=torch.bool,
            )
        manager_done_mask = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=raw_env.device,
        )
        for mask in termination_masks.values():
            manager_done_mask |= mask
        wrapper_done_mask = dones.to(device=raw_env.device, dtype=torch.bool)
        if not torch.equal(manager_done_mask, wrapper_done_mask):
            raise RuntimeError("wrapper done mask does not match termination term masks")

        progress, root_clearance = _progress_and_root_clearance(
            root_pos_w=root_pos_w,
            env_origins=env_origins,
            step_heights=step_heights,
            protocol=protocol,
        )
        _assert_finite(progress, f"root progress at step {step_index}")
        _assert_finite(root_clearance, f"root clearance at step {step_index}")
        max_progress = torch.where(active, torch.maximum(max_progress, progress), max_progress)

        fallen_now = active & (root_clearance < fall_clearance_threshold)
        fall_streak = torch.where(
            active,
            torch.where(fallen_now, fall_streak + 1, torch.zeros_like(fall_streak)),
            fall_streak,
        )
        max_fall_streak = torch.maximum(max_fall_streak, fall_streak)
        on_goal = (
            active
            & (progress >= goal_progress)
            & (root_clearance >= fall_clearance_threshold)
        )
        top_hold = torch.where(
            active,
            torch.where(on_goal, top_hold + 1, torch.zeros_like(top_hold)),
            top_hold,
        )
        max_top_hold = torch.maximum(max_top_hold, top_hold)

        done_mask = manager_done_mask & active
        for env_index in done_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            terminal_terms = [
                name
                for name in termination_term_order
                if bool(termination_masks[name][env_index].item())
            ]
            terminal_reason = resolve_phase0_terminal_reason(protocol, terminal_terms)
            episode_steps = int(episode_lengths[env_index].item())
            if episode_steps != control_steps_executed:
                raise RuntimeError("first-episode transition length drifted")
            observed_progress = float(max_progress[env_index].item())
            observed_max_top_hold = int(max_top_hold[env_index].item())
            observed_max_fall = int(max_fall_streak[env_index].item())
            fall_detected = observed_max_fall >= fall_required_steps
            active_terminal_terms = set(terminal_terms)
            success = (
                bool(active_terminal_terms & timeout_manager_term_set)
                and not bool(active_terminal_terms & terminated_manager_term_set)
                and observed_progress >= goal_progress
                and observed_max_top_hold >= success_hold_steps
                and not fall_detected
            )
            records[env_index] = {
                "env_index": int(env_index),
                "terminal_reason": terminal_reason,
                "terminal_terms": terminal_terms,
                "episode_steps": episode_steps,
                "max_progress_m": round(observed_progress, 8),
                "reached_first_step": observed_progress >= first_step_progress,
                "max_top_hold_steps": observed_max_top_hold,
                "success": success,
                "max_fall_consecutive_steps": observed_max_fall,
                "fall_detected": fall_detected,
            }
        active &= ~done_mask
        if not bool(active.any().item()):
            break

    elapsed_s = time.perf_counter() - start
    if bool(active.any().item()) or any(record is None for record in records):
        missing = active.nonzero(as_tuple=False).flatten().cpu().tolist()
        raise RuntimeError(
            f"first-episode probe did not complete every env: incomplete={missing}"
        )

    bins = protocol["terrain"]["evaluation_bins"]
    first_episodes = {item["bin_id"]: [] for item in bins}
    for env_index, raw_record in enumerate(records):
        record = raw_record
        if record is None:
            raise RuntimeError("first-episode record unexpectedly missing")
        bin_id = bins[terrain_types[env_index]]["bin_id"]
        first_episodes[bin_id].append(record)
    summary_per_bin = {
        bin_id: _summarize_first_episodes(
            records_for_bin,
            terminal_reason_names=phase0_terminal_reason_names(protocol),
        )
        for bin_id, records_for_bin in first_episodes.items()
    }

    return {
        "seed": int(probe["seed"]),
        "control_steps_limit": control_steps_limit,
        "control_steps_executed": control_steps_executed,
        "envs_per_bin": int(
            protocol["evaluation"]["env_to_bin_mapping"]["envs_per_bin"]
        ),
        "first_episodes": first_episodes,
        "summary_per_bin": summary_per_bin,
        "elapsed_s": elapsed_s,
        "aggregate_env_steps_per_s": num_envs * control_steps_executed / elapsed_s,
        "all_values_finite": True,
    }


def phase0_task_viability_passed(
    protocol: Mapping[str, Any],
    first_episodes: Mapping[str, list[Mapping[str, Any]]],
) -> bool:
    """Evaluate the frozen warm-start viability floor from first episodes."""

    probe = protocol["phase0_probe"]
    minimum_fraction = float(
        probe["minimum_reached_first_step_fraction_per_bin"]
    )
    for bin_id in probe["in_distribution_bin_ids"]:
        records = first_episodes[bin_id]
        reached = sum(bool(record["reached_first_step"]) for record in records)
        if reached / len(records) < minimum_fraction:
            return False
    return True


def expected_terrain_types(protocol: Mapping[str, Any]) -> list[int]:
    mapping = protocol["evaluation"]["env_to_bin_mapping"]
    envs_per_bin = int(mapping["envs_per_bin"])
    return [
        terrain_type
        for terrain_type in range(len(protocol["terrain"]["evaluation_bins"]))
        for _ in range(envs_per_bin)
    ]


def _runtime_observation_schema(raw_env: Any) -> dict[str, Any]:
    manager = raw_env.observation_manager
    group_name = "policy"
    term_names = list(manager.active_terms[group_name])
    term_dims = list(manager.group_obs_term_dim[group_name])
    group_cfg = getattr(raw_env.cfg.observations, group_name)
    terms: list[dict[str, Any]] = []
    for name, shape in zip(term_names, term_dims, strict=True):
        term_cfg = getattr(group_cfg, name)
        terms.append(
            {
                "manager_term": name,
                "callable": _qualified_name(term_cfg.func),
                "shape": [int(value) for value in shape],
                "parameters": _public_observation_parameters(term_cfg.params),
            }
        )
    schema = {
        "group": group_name,
        "concatenate_terms": bool(manager.group_obs_concatenate[group_name]),
        "terms": terms,
        "total_dim": int(math.prod(manager.group_obs_dim[group_name])),
    }
    schema["schema_hash"] = canonical_json_sha256(schema)
    return schema


def _runtime_action_schema(raw_env: Any) -> dict[str, Any]:
    manager = raw_env.action_manager
    terms: list[dict[str, Any]] = []
    for name in manager.active_terms:
        term = manager.get_term(name)
        cfg = term.cfg
        resolved_joint_names = list(getattr(term, "_joint_names", ()))
        terms.append(
            {
                "manager_term": name,
                "implementation": _qualified_name(type(term)),
                "dim": int(term.action_dim),
                "asset_name": str(cfg.asset_name),
                "joint_selector": list(cfg.joint_names),
                "preserve_order": bool(cfg.preserve_order),
                "scale": float(cfg.scale),
                "use_default_offset": bool(cfg.use_default_offset),
                "resolved_joint_names": resolved_joint_names,
            }
        )
    schema = {
        "terms": terms,
        "total_dim": int(manager.total_action_dim),
    }
    schema["schema_hash"] = canonical_json_sha256(schema)
    return schema


def _runtime_termination_schema(raw_env: Any) -> dict[str, Any]:
    manager = raw_env.termination_manager
    terms: list[dict[str, Any]] = []
    for name in manager.active_terms:
        cfg = manager.get_term_cfg(name)
        terms.append(
            {
                "manager_term": name,
                "callable": _qualified_name(cfg.func),
                "time_out": bool(cfg.time_out),
                "parameters": _public_termination_parameters(cfg.params),
            }
        )
    schema = {"terms": terms}
    schema["schema_hash"] = canonical_json_sha256(schema)
    return schema


def _public_observation_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, bool | int | float | str):
            result[str(key)] = value
        elif hasattr(value, "name"):
            result[f"{key}_name"] = str(value.name)
        else:
            raise RuntimeError(
                f"observation parameter cannot be canonically exported: {key}"
            )
    return result


def _public_termination_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, bool | int | float | str):
            result[str(key)] = value
            continue
        if not hasattr(value, "name"):
            raise RuntimeError(
                f"termination parameter cannot be canonically exported: {key}"
            )
        entity = {"name": str(value.name)}
        for selector_name in ("body_names", "joint_names"):
            selector = getattr(value, selector_name, None)
            if selector is not None:
                entity[selector_name] = (
                    [selector] if isinstance(selector, str) else list(selector)
                )
        result[str(key)] = entity
    return result


def _progress_and_root_clearance(
    *,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    step_heights: torch.Tensor,
    protocol: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    terrain = protocol["terrain"]
    progress = root_pos_w[:, 0] - env_origins[:, 0]
    local_origin_x = float(terrain["start_platform_length_m"]) / 2.0
    local_x = progress + local_origin_x
    step_level = torch.ceil(
        (local_x - float(terrain["start_platform_length_m"]))
        / float(terrain["tread_depth_m"])
    )
    step_level = torch.clamp(
        step_level,
        min=0,
        max=int(terrain["step_count"]),
    )
    surface_height = step_level * step_heights
    root_clearance = root_pos_w[:, 2] - env_origins[:, 2] - surface_height
    return progress, root_clearance


def _summarize_first_episodes(
    records: list[Mapping[str, Any]],
    *,
    terminal_reason_names: tuple[str, ...],
) -> dict[str, Any]:
    episode_count = len(records)
    return {
        "episode_count": episode_count,
        "terminal_reason_counts": {
            reason: sum(record["terminal_reason"] == reason for record in records)
            for reason in terminal_reason_names
        },
        "reached_first_step_count": sum(
            bool(record["reached_first_step"]) for record in records
        ),
        "fall_detected_count": sum(
            bool(record["fall_detected"]) for record in records
        ),
        "success_count": sum(bool(record["success"]) for record in records),
        "mean_max_progress_m": round(
            sum(float(record["max_progress_m"]) for record in records)
            / episode_count,
            8,
        ),
    }


def _qualified_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _xyz_rows(value: torch.Tensor) -> list[list[float]]:
    return [
        [round(float(component), 8) for component in row]
        for row in value.detach().cpu().tolist()
    ]


def _assert_finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{context} contains NaN or Inf")


__all__ = [
    "assert_runtime_environment_matches",
    "build_runtime_environment_evidence",
    "expected_terrain_types",
    "phase0_task_viability_passed",
    "run_first_episode_probe",
]

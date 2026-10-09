"""Push v2 environment, exact velocity events and pre-reset episode evidence.

Isaac and torch are imported only when constructing or executing the runtime.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.push_recovery.protocol import physical_protocol_from_config, physical_public_runtime_dynamics


def physical_runtime_config_from_manifest(manifest: Any, profile: str, *, evaluation: bool = False) -> dict[str, Any] | None:
    if manifest is None:
        return None
    projection = manifest.robot_task_binding["task_projection"]
    if "push_protocol" not in projection:
        return None
    from true_rl_agent.robots.isaaclab_locomotion import isaaclab_locomotion_runtime_spec
    adapter_id = manifest.runtime_adapter["runtime_adapter_id"]
    spec = isaaclab_locomotion_runtime_spec(adapter_id)
    config = {"push_protocol": dict(projection["push_protocol"]),
        "push_robot_scale_m": projection["push_robot_scale_m"],
        "runtime_adapter_id": adapter_id, "active_profile": profile,
        "task": {"runtime_adapter_id": adapter_id,
            "environment_cfg_entrypoint": spec.environment_cfg_entrypoint,
            "runner_cfg_entrypoint": spec.runner_cfg_entrypoint,
            "agent_configuration_hash": spec.agent_configuration.agent_configuration_hash}}
    _, resolved = physical_protocol_from_config(config)
    if projection.get("push_resolved_protocol") != resolved:
        raise ValueError("Push manifest dimensional protocol does not match its source")
    if evaluation and "push_evaluation_protocol" in projection:
        config["push_evaluation_protocol"] = dict(projection["push_evaluation_protocol"])
        config["force_body_name"] = manifest.robot_task_binding["role_bindings"]["base"][0]
        _, evaluation_resolved = physical_protocol_from_config(config)
        if projection.get("push_resolved_evaluation_protocol") != evaluation_resolved:
            raise ValueError("Push force evaluation protocol does not match its source")
    physical_public_runtime_dynamics(config)
    return config


def instantiate_physical_push_configs(config: Mapping[str, Any], args: Any, *, evaluation: bool) -> tuple[Any, Any]:
    from true_rl_agent.robots.isaaclab_locomotion import isaaclab_locomotion_runtime_spec
    physical_public_runtime_dynamics(config)
    spec = isaaclab_locomotion_runtime_spec(config["runtime_adapter_id"])
    env_cfg = spec.instantiate_environment_cfg()
    apply_physical_push_overlay(env_cfg, config, args, evaluation=evaluation)
    agent_cfg = spec.instantiate_runner_cfg()
    spec.agent_configuration.apply_to(agent_cfg)
    return env_cfg, agent_cfg


def apply_physical_push_overlay(env_cfg: Any, config: Mapping[str, Any], args: Any, *, evaluation: bool) -> None:
    from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
    from isaaclab.utils import configclass
    force_evaluation = "push_evaluation_protocol" in config
    if force_evaluation and not evaluation:
        raise ValueError("Push force protocol is evaluation-only")
    _, p = physical_protocol_from_config(config)
    env_cfg.seed, env_cfg.scene.num_envs, env_cfg.sim.device = args.seed, args.num_envs, args.device
    env_cfg.episode_length_s = p["episode_length_s"]
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.height_scanner = None
    env_cfg.scene.contact_forces = None
    for name in ("policy", "critic"):
        group = getattr(env_cfg.observations, name, None)
        if group is not None:
            if hasattr(group, "height_scan"):
                group.height_scan = None
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = not evaluation
    for name in vars(env_cfg.curriculum):
        if not name.startswith("_"):
            setattr(env_cfg.curriculum, name, None)
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (p["episode_length_s"] + 1.0,) * 2
    command.rel_standing_envs, command.rel_heading_envs = 1.0, 0.0
    command.heading_command, command.debug_vis = False, False
    for ranges in (command.ranges, getattr(command, "limit_ranges", None)):
        if ranges is not None:
            ranges.lin_vel_x = ranges.lin_vel_y = ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    reset = p["reset"]
    env_cfg.events.reset_base.params["pose_range"] = {key: tuple(value) for key, value in reset["root_pose_range"].items()}
    env_cfg.events.reset_base.params["velocity_range"] = {key: tuple(value) for key, value in reset["root_velocity_range"].items()}
    env_cfg.events.reset_robot_joints.params["position_range"] = tuple(reset["joint_position_scale"])
    env_cfg.events.reset_robot_joints.params["velocity_range"] = tuple(reset["joint_velocity_range"])
    env_cfg.push_protocol_parameters, env_cfg.push_runtime_config = p, dict(config)
    env_cfg.push_evaluation = evaluation
    env_cfg.push_force_evaluation = force_evaluation
    env_cfg.push_control_only = config["active_profile"] == "no_push"
    dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.events.push_reset = EventTermCfg(func=reset_push_state, mode="reset")
    env_cfg.events.push_velocity_increment = None if force_evaluation else EventTermCfg(
        func=apply_push_velocity_increment, mode="interval", interval_range_s=(dt, dt))

    @configclass
    class PushTerminations:
        time_out = TerminationTermCfg(func=push_timeout, time_out=True)
        task_failure = TerminationTermCfg(func=capture_push_transition)

    @configclass
    class PushRewards:
        full_program = RewardTermCfg(func=push_program_reward, weight=1.0)

    env_cfg.terminations, env_cfg.rewards = PushTerminations(), PushRewards()


def make_physical_push_environment(env_cfg: Any, *, render_mode: str | None = None) -> Any:
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    if getattr(env_cfg, "push_force_evaluation", False):
        push_runtime_state(env).install_force_hooks()
    return env


def _tensor(value: Any) -> Any:
    from true_rl_agent.backends.isaac_lab_arrays import as_torch
    return as_torch(value)


def push_runtime_state(env: Any) -> "PushRuntimeState":
    if not hasattr(env, "_push_runtime_state"):
        if getattr(env.cfg, "push_force_evaluation", False):
            from true_rl_agent.tasks.push_recovery.force_runtime import ForcePushRuntimeState
            env._push_runtime_state = ForcePushRuntimeState(env)
        else:
            env._push_runtime_state = PushRuntimeState(env)
    return env._push_runtime_state


def reset_push_state(env: Any, env_ids: Any) -> None:
    push_runtime_state(env).reset(env_ids)


def apply_push_velocity_increment(env: Any, env_ids: Any) -> None:
    push_runtime_state(env).inject(env_ids)


def push_timeout(env: Any) -> Any:
    return _tensor(env.episode_length_buf) >= int(env.max_episode_length)


def capture_push_transition(env: Any) -> Any:
    capture = getattr(env, "_push_pre_reset_capture", None)
    if capture is not None:
        capture()
    return push_runtime_state(env).observe()


def push_program_reward(env: Any) -> Any:
    import torch
    if env.cfg.push_evaluation:
        return torch.zeros(env.num_envs, device=env.device)
    adapter = getattr(env, "_push_reward_adapter", None)
    if adapter is None or not adapter.components.reward_terms:
        raise RuntimeError("Push v2 requires a nonempty full reward program before stepping")
    env._push_full_program_runtime = True
    return adapter.apply_step_rewards(env=env, actions=_tensor(env.action_manager.action),
        rewards=torch.zeros(env.num_envs, device=env.device), reset_mask=_tensor(env.termination_manager.dones))


class PushRuntimeState:
    """Vectorized state; evaluation freezes each first episode before auto-reset."""
    def __init__(self, env: Any) -> None:
        import torch
        self.env, self.parameters = env, dict(env.cfg.push_protocol_parameters)
        self.n, self.device, self.dt = int(env.num_envs), env.device, float(env.step_dt)
        self.evaluation, self.control_only = bool(env.cfg.push_evaluation), bool(env.cfg.push_control_only)
        self.split, self.phase_angle = "public_dev", 0.0
        # Optional task-owned numerical archive; never participates in decisions.
        self.numeric_capture = None
        self.closed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.episode_index = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        for name in ("stratum", "planned", "steps"):
            setattr(self, name, torch.zeros(self.n, dtype=torch.long, device=self.device))
        self.initial_xy = torch.zeros((self.n, 2), device=self.device)
        for name in ("min_height", "min_pitch"):
            setattr(self, name, torch.full((self.n,), float("inf"), device=self.device))
        self.max_pitch = torch.full((self.n,), -float("inf"), device=self.device)
        for name in ("max_roll", "max_drift", "final_speed", "public_active"):
            setattr(self, name, torch.zeros(self.n, device=self.device))
        self.final_stable = torch.zeros_like(self.closed)
        self.event_steps = torch.full((self.n, 2), -1, dtype=torch.long, device=self.device)
        self.recovered_step = torch.full_like(self.event_steps, -1)
        self.dwell = torch.zeros_like(self.event_steps)
        self.requested = torch.zeros((self.n, 2, 2), device=self.device)
        self.observed = torch.zeros_like(self.requested)
        self.before = torch.zeros((self.n, 2, 6), device=self.device)
        self.after, self.next_physics_velocity = torch.zeros_like(self.before), torch.zeros_like(self.before)
        self.verified = torch.zeros((self.n, 2), dtype=torch.bool, device=self.device)
        self.next_physics_recorded = torch.zeros_like(self.verified)
        # Lifetime event evidence survives episode resets; transfer to CPU only at summary.
        self.event_counts = torch.zeros(2, dtype=torch.long, device=self.device)
        self.public_delta = torch.zeros((self.n, 2), device=self.device)
        self.angle = torch.zeros_like(self.public_delta)
        self.event_times = tuple(math.ceil(value / self.dt) for value in self.parameters["push_times_s"])
        self.window = math.ceil(self.parameters["recovery_window_s"] / self.dt)
        self.required_dwell = math.ceil(self.parameters["stable_dwell_s"] / self.dt)

    def begin_evaluation(self, split: str, seed: int) -> None:
        import random
        if split not in {"public_dev", "public_accept", "sealed_validation"}:
            raise ValueError("unsupported Push evaluation split")
        self.split = split
        lower, upper = self.parameters["evaluation_direction_policy"]["angular_range_rad"]
        self.phase_angle = lower + random.Random(seed).random() * (upper - lower)
        self.closed.zero_()

    def reset(self, env_ids: Any) -> None:
        import torch
        from true_rl_agent.tasks.push_recovery.protocol import PUSH_STRATA
        ids = torch.arange(self.n, device=self.device) if env_ids is None else _tensor(env_ids).long()
        if self.evaluation:
            ids = ids[~self.closed[ids]]
        if ids.numel() == 0:
            return
        self.episode_index[ids] += 1
        self.steps[ids] = 0
        self.initial_xy[ids] = _tensor(self.env.scene["robot"].data.root_pos_w)[ids, :2]
        self.min_height[ids], self.min_pitch[ids], self.max_pitch[ids] = float("inf"), float("inf"), -float("inf")
        for name in ("max_roll", "max_drift", "final_speed", "final_stable", "requested", "observed",
                     "before", "after", "verified", "dwell", "next_physics_velocity", "next_physics_recorded", "public_active", "public_delta"):
            getattr(self, name)[ids] = 0
        self.event_steps[ids], self.recovered_step[ids] = -1, -1
        if self.evaluation:
            counts = torch.tensor([self.parameters["evaluation_cohort_counts"][name] for name in PUSH_STRATA], device=self.device)
            ends = counts.cumsum(0)
            cohort_ids = ids % int(ends[-1])
            stratum_indices = torch.bucketize(cohort_ids, ends, right=True)
            offsets = torch.cat((torch.zeros(1, dtype=counts.dtype, device=self.device), ends[:-1]))
            policy = self.parameters["evaluation_direction_policy"]
            lower, upper = policy["angular_range_rad"]
            self.stratum[ids] = stratum_indices + 1
            self.angle[ids, 0] = ((cohort_ids - offsets[stratum_indices]) % policy["count"]) * ((upper - lower) / policy["count"]) + self.phase_angle
            self.angle[ids, 1] = self.angle[ids, 0] + policy["second_event_offset_rad"]
        else:
            sampling = self.parameters["training_sampling"]
            probabilities = torch.tensor([sampling["stratum_probabilities"][name] for name in PUSH_STRATA], device=self.device)
            if sampling["event_directions"] != "independent_uniform":
                raise ValueError("unsupported Push training direction sampling")
            self.stratum[ids] = torch.multinomial(probabilities, len(ids), replacement=True) + 1
            lower, upper = sampling["angular_range_rad"]
            self.angle[ids] = lower + torch.rand((len(ids), 2), device=self.device) * (upper - lower)
        self.planned[ids] = torch.where(self.stratum[ids] == 4, 2, 1)
        if self.control_only:
            self.stratum[ids], self.planned[ids] = 0, 0

    def inject(self, env_ids: Any) -> None:
        import torch
        self.public_active.zero_()
        self.public_delta.zero_()
        steps = _tensor(self.env.episode_length_buf)
        allowed = torch.ones(self.n, dtype=torch.bool, device=self.device)
        if env_ids is not None:
            allowed.zero_()
            allowed[_tensor(env_ids).long()] = True
        if self.evaluation:
            allowed &= ~self.closed
        magnitudes, robot = self.parameters["velocity_magnitudes_mps"], self.env.scene["robot"]
        for index, scheduled in enumerate(self.event_times):
            mask = allowed & (self.planned > index) & (self.event_steps[:, index] < 0) & (steps >= scheduled)
            ids = mask.nonzero(as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            scale = torch.full((len(ids),), magnitudes["medium"], device=self.device)
            scale[self.stratum[ids] == 1], scale[self.stratum[ids] == 3] = magnitudes["low"], magnitudes["high"]
            angle = self.angle[ids, index]
            delta = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1) * scale[:, None]
            before = torch.cat((_tensor(robot.data.root_lin_vel_w)[ids], _tensor(robot.data.root_ang_vel_w)[ids]), dim=1).clone()
            desired = before.clone()
            desired[:, :2] += delta
            robot.write_root_velocity_to_sim(desired, env_ids=ids)
            after = torch.cat((_tensor(robot.data.root_lin_vel_w)[ids], _tensor(robot.data.root_ang_vel_w)[ids]), dim=1).clone()
            observed = after[:, :2] - before[:, :2]
            valid = torch.isfinite(after).all(dim=1) & ((after - desired).abs().max(dim=1).values <= self.parameters["injection_tolerance_mps"])
            self.event_steps[ids, index] = steps[ids]
            self.before[ids, index], self.after[ids, index] = before, after
            self.requested[ids, index], self.observed[ids, index], self.verified[ids, index] = delta, observed, valid
            self.public_active[ids], self.public_delta[ids] = 1, observed
            self.event_counts[0] += ids.numel()
            self.event_counts[1] += valid.sum()

    def _observe_pose(self) -> tuple[Any, Any, Any]:
        import torch
        from true_rl_agent.backends.isaac_lab_arrays import quaternion_wxyz
        robot, p = self.env.scene["robot"], self.parameters
        pos, velocity = _tensor(robot.data.root_pos_w), _tensor(robot.data.root_lin_vel_w)
        angular, quat = _tensor(robot.data.root_ang_vel_w), quaternion_wxyz(robot.data.root_quat_w)
        w, x, y, z = quat.unbind(dim=1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        height = pos[:, 2] - _tensor(self.env.scene.env_origins)[:, 2]
        displacement = pos[:, :2] - self.initial_xy
        drift = torch.linalg.vector_norm(displacement, dim=1)
        speed = torch.linalg.vector_norm(velocity[:, :2], dim=1)
        angular_speed = torch.linalg.vector_norm(angular, dim=1)
        stable = ((speed <= p["recovery_velocity_mps"]) & (angular_speed <= p["recovery_angular_velocity_radps"])
            & (roll.abs() <= p["recovery_tilt_rad"]) & (pitch.abs() <= p["recovery_tilt_rad"]) & (height >= p["recovery_height_m"]))
        failure = (height < p["fall_height_m"]) | (roll.abs() > p["fall_tilt_rad"]) | (pitch.abs() > p["fall_tilt_rad"])
        # A null task parameter disables spatial failure; drift remains a diagnostic.
        if p["max_displacement_m"] is not None:
            failure |= drift > p["max_displacement_m"]
        active = ~self.closed if self.evaluation else torch.ones_like(self.closed)
        ids = active.nonzero(as_tuple=False).flatten()
        self.steps[ids] = _tensor(self.env.episode_length_buf)[ids]
        self.min_height[ids] = torch.minimum(self.min_height[ids], height[ids])
        self.max_pitch[ids], self.min_pitch[ids] = torch.maximum(self.max_pitch[ids], pitch[ids]), torch.minimum(self.min_pitch[ids], pitch[ids])
        self.max_roll[ids], self.max_drift[ids] = torch.maximum(self.max_roll[ids], roll[ids].abs()), torch.maximum(self.max_drift[ids], drift[ids])
        self.final_speed[ids], self.final_stable[ids] = speed[ids], stable[ids]
        if self.numeric_capture is not None:
            self._numeric_pose = torch.stack((height, roll, pitch, speed, angular_speed, drift), dim=1)
            # Preserve signed operands from this same pre-reset observation. The
            # norm columns above remain the exact operands used by task tests.
            self._numeric_planar_motion = torch.cat((velocity[:, :2], displacement), dim=1)
            self._numeric_conditions = torch.stack((speed <= p["recovery_velocity_mps"],
                angular_speed <= p["recovery_angular_velocity_radps"], roll.abs() <= p["recovery_tilt_rad"],
                pitch.abs() <= p["recovery_tilt_rad"], height >= p["recovery_height_m"]), dim=1)
        return stable, failure, active

    def observe(self) -> Any:
        import torch
        stable, failure, active = self._observe_pose()
        robot = self.env.scene["robot"]
        velocity, angular = _tensor(robot.data.root_lin_vel_w), _tensor(robot.data.root_ang_vel_w)
        for index in range(2):
            elapsed = self.steps - self.event_steps[:, index]
            injected = active & (self.event_steps[:, index] >= 0) & (elapsed > 0)
            first = injected & ~self.next_physics_recorded[:, index]
            self.next_physics_velocity[first, index] = torch.cat((velocity[first], angular[first]), dim=1)
            self.next_physics_recorded[first, index] = True
            observing = injected & (elapsed <= self.window) & (self.recovered_step[:, index] < 0)
            self.dwell[observing, index] = torch.where(stable[observing], self.dwell[observing, index] + 1, 0)
            recovered = observing & (self.dwell[:, index] >= self.required_dwell) & self.verified[:, index]
            self.recovered_step[recovered, index] = self.steps[recovered]
        if self.numeric_capture is not None:
            self.numeric_capture.record(self, failure, active)
        if self.evaluation:
            self.closed |= active & (failure | (_tensor(self.env.episode_length_buf) >= self.env.max_episode_length))
        return failure

    def exposure_summary(self) -> dict[str, Any]:
        """Write attempts and verified read-backs; not next-physics or recovery counts."""
        injected, verified = self.event_counts.detach().cpu().tolist()
        return {"schema_version": "true_rl_agent.push_recovery.exposure.v1",
                "scope": "this_runtime_instance_all_episodes",
                "evaluation": self.evaluation, "control_only": self.control_only,
                "injection_attempt_count": injected, "verified_event_count": verified}

    def episode_records(self) -> list[dict[str, Any]]:
        from true_rl_agent.tasks.push_recovery.failure_classifier import classify_recovery_episode
        names = ("episode_index", "stratum", "planned", "steps", "min_height", "max_pitch", "min_pitch", "max_roll", "max_drift",
            "final_speed", "final_stable", "event_steps", "requested", "observed", "before", "after", "verified", "recovered_step",
            "next_physics_velocity", "next_physics_recorded")
        v = {name: getattr(self, name).detach().cpu().tolist() for name in names}
        strata, records = ("control", "low", "medium", "high", "repeated"), []
        for env_id in range(self.n):
            events = []
            for index in range(2):
                step, recovery = v["event_steps"][env_id][index], v["recovered_step"][env_id][index]
                if step < 0:
                    continue
                events.append({"event_index": index, "control_step": step, "time_s": step * self.dt,
                    "requested_delta_v_world_mps": v["requested"][env_id][index], "observed_delta_v_world_mps": v["observed"][env_id][index],
                    "velocity_before_world_6d": v["before"][env_id][index], "velocity_after_write_world_6d": v["after"][env_id][index],
                    "velocity_after_next_physics_step_world_6d": v["next_physics_velocity"][env_id][index] if v["next_physics_recorded"][env_id][index] else None,
                    "injection_verified": v["verified"][env_id][index], "recovered": recovery >= 0,
                    "recovery_control_step": recovery if recovery >= 0 else None,
                    "recovery_latency_s": (recovery - step) * self.dt if recovery >= 0 else self.window * self.dt,
                    "recovery_censored": recovery < 0})
            record = {"env_id": env_id, "episode_index": v["episode_index"][env_id], "episode_steps": v["steps"][env_id],
                "stratum": strata[v["stratum"][env_id]], "control_only": self.control_only, "planned_push_count": v["planned"][env_id],
                "push_events": events, "min_base_height_m": v["min_height"][env_id], "max_pitch_rad": v["max_pitch"][env_id],
                "min_pitch_rad": v["min_pitch"][env_id], "max_abs_roll_rad": v["max_roll"][env_id],
                "max_abs_pitch_rad": max(abs(v["max_pitch"][env_id]), abs(v["min_pitch"][env_id])),
                "max_xy_drift_m": v["max_drift"][env_id], "final_planar_speed_mps": v["final_speed"][env_id],
                "final_recovery_condition": v["final_stable"][env_id]}
            record.update(classify_recovery_episode(record, self.parameters))
            record["failure_label"] = record.pop("label")
            records.append(record)
        return records


def _plain_configuration(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_configuration(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _plain_configuration(value.to_dict())
    raise TypeError(f"unsupported Push ABI configuration value: {type(value).__name__}")


def push_policy_io_evidence(env: Any, manifest: Any = None) -> dict[str, Any]:
    base = env.unwrapped
    actions = []
    effective_actions = []
    actual_scale_consistent = True
    for name in base.action_manager.active_terms:
        term = base.action_manager.get_term(name)
        cfg = term.cfg
        actions.append({"name": name, "class": type(cfg).__module__ + "." + type(cfg).__qualname__,
            "asset_name": cfg.asset_name, "joint_names": list(cfg.joint_names),
            "scale": _plain_configuration(cfg.scale), "use_default_offset": cfg.use_default_offset})
        scale = _plain_configuration(term._scale)
        if isinstance(scale, list) and scale and isinstance(scale[0], list):
            if any(row != scale[0] for row in scale):
                raise ValueError("Push action scale differs across environments")
            scale = scale[0]
        effective_actions.append({"name": name, "joint_names": list(term._joint_names), "scale": scale})
        if isinstance(cfg.scale, (int, float)):
            actual_scale_consistent &= (all(value == cfg.scale for value in scale) if isinstance(scale, list) else scale == cfg.scale)
        elif isinstance(cfg.scale, Mapping):
            import re
            expected_scale = []
            for joint in term._joint_names:
                matches = [value for pattern, value in cfg.scale.items() if re.fullmatch(pattern, joint)]
                expected_scale.append(matches[0] if len(matches) == 1 else 1.0)
            actual_scale_consistent &= scale == expected_scale
        else:
            actual_scale_consistent = False
    group = base.cfg.observations.policy
    terms = []
    for name in base.observation_manager.active_terms["policy"]:
        cfg = getattr(group, name)
        noise = getattr(cfg, "noise", None)
        noise_projection = None if noise is None else {
            "class": type(noise).__module__ + "." + type(noise).__qualname__,
            **{key: _plain_configuration(getattr(noise, key)) for key in ("n_min", "n_max", "operation")}}
        terms.append({"name": name, "callable": _plain_configuration(cfg.func),
            "params": _plain_configuration(cfg.params), "scale": _plain_configuration(cfg.scale),
            "clip": _plain_configuration(cfg.clip), "noise": noise_projection})
    observations = {"concatenate_terms": group.concatenate_terms,
        "enable_corruption_by_mode": {"train": True, "eval": False},
        "history_length": group.history_length, "terms": terms}
    sensors = {name: {"class": type(sensor).__module__ + "." + type(sensor).__qualname__,
        "prim_path": sensor.cfg.prim_path} for name, sensor in base.scene.sensors.items()}
    payload = {"schema_version": "true_rl_agent.task.push_recovery.runtime_policy_io.v2",
        "action_dim": int(base.action_manager.total_action_dim),
        "observation_group_dims": {key: list(value) for key, value in base.observation_manager.group_obs_dim.items()},
        "observation_group_terms": {key: list(value) for key, value in base.observation_manager.active_terms.items()},
        "action_terms": list(base.action_manager.active_terms), "control_dt_s": float(base.step_dt),
        "control": {"control_dt_s": float(base.step_dt), "decimation": int(base.cfg.decimation), "simulation_dt_s": float(base.cfg.sim.dt)},
        "joint_names": list(base.scene["robot"].joint_names), "actions": actions,
        "effective_actions": effective_actions, "policy_observation": observations,
        "sensors": sensors,
        "policy_observation_term_dimensions": {name: math.prod(shape) for name, shape in zip(
            base.observation_manager.active_terms["policy"], base.observation_manager.group_obs_term_dim["policy"], strict=True)},
        "effective_observation_corruption": group.enable_corruption,
        "height_scanner_present": getattr(base.cfg.scene, "height_scanner", None) is not None}
    if manifest is not None:
        expected = manifest.robot_task_binding["task_projection"]["push_abi_facts"]
        checks = {"joint_names": payload["joint_names"] == expected["joint_names"],
            "actions": actions == expected["actions"], "control": payload["control"] == expected["control"],
            "policy_observation": observations == expected["policy_observation"],
            "action_dimension": payload["action_dim"] == expected["action_dimension"],
            "policy_observation_dimension": payload["observation_group_dims"]["policy"] == [expected["policy_observation_dimension"]],
            "term_dimensions": payload["policy_observation_term_dimensions"] == expected["policy_observation_term_dimensions"],
            "sensors": sensors == expected["sensors"] == {},
            "effective_action_scales": actual_scale_consistent,
            "resolved_action_joint_order": [joint for item in effective_actions for joint in item["joint_names"]] == expected["joint_names"],
            "observation_corruption": group.enable_corruption == (not base.cfg.push_evaluation),
            "height_scanner_removed": not payload["height_scanner_present"]}
        payload["manifest_checks"] = checks
        if not all(checks.values()):
            raise ValueError("Push actual policy ABI mismatch: " + ", ".join(key for key, passed in checks.items() if not passed))
        payload["verified_declared_content_hashes"] = {key: expected[key] for key in (
            "action_content_hash", "observation_content_hash", "sensor_schema_hash", "policy_io_schema_hash")}
    payload["evidence_hash"] = canonical_json_hash(payload)
    return payload

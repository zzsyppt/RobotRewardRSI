"""Finite world forces and sequence recovery for Push evaluation only.

The task installs instance-local hooks at the articulation write boundary.  Each
physics substep receives a fresh wrench; verification inspects the actual local
force passed to PhysX, not just the requested composer input.
"""
from __future__ import annotations

import math
from typing import Any

from true_rl_agent.tasks.push_recovery.runtime import PushRuntimeState, _tensor


def _world_vector(vector: Any, quaternion_xyzw: Any) -> Any:
    import torch
    xyz, w = quaternion_xyzw[..., :3], quaternion_xyzw[..., 3:4]
    cross = 2 * torch.linalg.cross(xyz, vector)
    return vector + w * cross + torch.linalg.cross(xyz, cross)


def _physics_tensor(value: Any) -> Any:
    import torch
    result = _tensor(value)
    if torch.is_tensor(result):
        return result
    import warp
    return warp.to_torch(result)


class ForcePushRuntimeState(PushRuntimeState):
    """One frozen first episode per environment; no per-pulse recovery claim."""

    def __init__(self, env: Any) -> None:
        import torch
        super().__init__(env)
        if not self.evaluation:
            raise ValueError("finite-force Push runtime is evaluation-only")
        if getattr(env, "_physics_handles_decimation", False):
            raise ValueError("finite-force Push requires a write at every physics substep")
        self.robot = env.scene["robot"]
        self.body_name = env.cfg.push_runtime_config["force_body_name"]
        body_ids, names = self.robot.find_bodies(self.body_name)
        if len(body_ids) != 1 or list(names) != [self.body_name]:
            raise ValueError("Push force body must resolve to one exact bound base link")
        self.body_index = int(body_ids[0])
        self.body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
        self.physics_dt = float(env.physics_dt)
        self.decimation = int(env.cfg.decimation)
        if not math.isclose(self.physics_dt * self.decimation, self.dt, rel_tol=1e-9):
            raise ValueError("Push force physics and control time steps disagree")
        self.pulse_steps = math.ceil(self.parameters["pulse_duration_s"] / self.dt)
        self.gap_steps = math.ceil(self.parameters["repeated_gap_s"] / self.dt)
        first = self.event_times[0]
        self.event_times = (first, first + self.pulse_steps + self.gap_steps)
        self.expected_substeps = self.pulse_steps * self.decimation
        if self.event_times[-1] + self.pulse_steps + self.window >= env.max_episode_length:
            raise ValueError("quantized Push force sequence and recovery window exceed the episode")
        self.total_mass = torch.zeros(self.n, device=self.device)
        self.force_world = torch.zeros((self.n, 2, 3), device=self.device)
        self.delivered_impulse = torch.zeros_like(self.force_world)
        self.actual_substeps = torch.zeros_like(self.event_steps)
        self.event_ends = torch.full_like(self.event_steps, -1)
        self.max_force_error = torch.zeros((self.n, 2), device=self.device)
        self.unplanned_wrench = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.sequence_dwell = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.sequence_recovery_step = torch.full_like(self.sequence_dwell, -1)
        self._hooks_installed = False

    def reset(self, env_ids: Any) -> None:
        import torch
        ids = torch.arange(self.n, device=self.device) if env_ids is None else _tensor(env_ids).long()
        ids = ids[~self.closed[ids]]
        super().reset(ids)
        if ids.numel() == 0:
            return
        for name in ("delivered_impulse", "actual_substeps", "max_force_error", "unplanned_wrench", "sequence_dwell"):
            getattr(self, name)[ids] = 0
        self.event_ends[ids], self.sequence_recovery_step[ids] = -1, -1
        masses = _tensor(self.robot.data.body_mass)[ids]
        total = masses.sum(dim=1)
        if not bool(torch.isfinite(masses).all() & (total > 0).all()):
            raise ValueError("Push force requires finite positive actual total robot masses")
        self.total_mass[ids] = total
        magnitudes = self.parameters["impulse_velocity_mps"]
        magnitude = torch.full((len(ids),), magnitudes["medium"], device=self.device)
        magnitude[self.stratum[ids] == 1] = magnitudes["low"]
        magnitude[self.stratum[ids] == 3] = magnitudes["high"]
        magnitude[self.stratum[ids] == 4] = magnitudes["high"] * self.parameters["repeated_high_fraction"]
        direction = torch.stack((torch.cos(self.angle[ids]), torch.sin(self.angle[ids]), torch.zeros_like(self.angle[ids])), dim=-1)
        self.force_world[ids] = direction * (total * magnitude / (self.pulse_steps * self.dt))[:, None, None]

    def inject(self, env_ids: Any) -> None:
        raise RuntimeError("finite forces are applied at the physics write boundary, not interval events")

    def install_force_hooks(self) -> None:
        import torch
        if self._hooks_installed:
            return
        robot = self.robot
        original_write = robot.write_data_to_sim
        original_apply = robot.root_view.apply_forces_and_torques_at_position
        pending: dict[str, Any] = {}

        def apply_wrench(*args: Any, **kwargs: Any) -> Any:
            force = _physics_tensor(kwargs["force_data"]).reshape(self.n, robot.num_bodies, 3)
            torque = _physics_tensor(kwargs["torque_data"]).reshape_as(force)
            # The target composer uses link axes, even when inertial COM axes differ.
            quat = _tensor(robot.data.body_link_quat_w)[:, self.body_index]
            delivered = _world_vector(force[:, self.body_index], quat)
            error = (delivered - pending["force"]).abs().amax(dim=1)
            tolerance = self.parameters["force_tolerance_n"]
            valid = (torch.isfinite(force).all(dim=(1, 2)) & torch.isfinite(torque).all(dim=(1, 2))
                & torch.isfinite(quat).all(dim=1) & ((quat.square().sum(dim=1) - 1).abs() <= 1e-4)
                & (error <= tolerance) & (torque.abs().amax(dim=(1, 2)) <= tolerance))
            other = [index for index in range(robot.num_bodies) if index != self.body_index]
            if other:
                valid &= force[:, other].abs().amax(dim=(1, 2)) <= tolerance
            if kwargs.get("position_data") is not None or kwargs.get("is_global") is not False:
                valid.zero_()
            # Exactly one actual PhysX apply call must correspond to each write.
            if pending["called"]:
                valid.zero_()
            pending["called"] = True
            event = pending["event"]
            active = ~self.closed
            self.unplanned_wrench |= active & (event < 0) & ~valid
            for index in range(2):
                mask = active & (event == index)
                self.actual_substeps[mask, index] += 1
                self.verified[mask, index] &= valid[mask]
                self.event_ends[mask, index] = pending["steps"][mask] + 1
                self.delivered_impulse[mask, index] += delivered[mask] * self.physics_dt
                self.max_force_error[mask, index] = torch.maximum(self.max_force_error[mask, index], error[mask])
            return original_apply(*args, **kwargs)

        def inject_write() -> Any:
            steps = _tensor(self.env.episode_length_buf)
            force = torch.zeros((self.n, 3), device=self.device)
            event = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
            for index, start in enumerate(self.event_times):
                mask = (~self.closed) & (self.planned > index) & (steps >= start) & (steps < start + self.pulse_steps)
                first = mask & (self.event_steps[:, index] < 0)
                self.event_steps[first, index] = steps[first]
                self.verified[first, index] = True
                force[mask] = self.force_world[mask, index]
                event[mask] = index
            pending.update(steps=steps.clone(), force=force, event=event, called=False)
            robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=force[:, None, :], torques=torch.zeros_like(force[:, None, :]),
                positions=None, body_ids=self.body_ids, is_global=True)
            result = original_write()
            if not pending["called"]:
                for index in range(2):
                    self.verified[event == index, index] = False
            return result

        robot.write_data_to_sim = inject_write
        robot.root_view.apply_forces_and_torques_at_position = apply_wrench
        self._hooks_installed = True

    def observe(self) -> Any:
        import torch
        stable, failure, active = self._observe_pose()
        complete = (self.actual_substeps == self.expected_substeps) & self.verified
        complete &= self.event_steps == torch.tensor(self.event_times, device=self.device)
        complete &= self.event_ends == torch.tensor(self.event_times, device=self.device) + self.pulse_steps
        all_complete = complete[:, 0] & ((self.planned == 1) | complete[:, 1]) & (self.planned > 0)
        last_end = torch.where(self.planned == 2, self.event_times[1], self.event_times[0]) + self.pulse_steps
        elapsed = self.steps - last_end
        observing = active & all_complete & (elapsed > 0) & (elapsed <= self.window) & (self.sequence_recovery_step < 0)
        self.sequence_dwell[observing] = torch.where(stable[observing], self.sequence_dwell[observing] + 1, 0)
        recovered = observing & (self.sequence_dwell >= self.required_dwell) & ~self.unplanned_wrench
        self.sequence_recovery_step[recovered] = self.steps[recovered]
        if self.numeric_capture is not None:
            self.numeric_capture.record(self, failure, active)
        self.closed |= active & (failure | (_tensor(self.env.episode_length_buf) >= self.env.max_episode_length))
        return failure

    def episode_records(self) -> list[dict[str, Any]]:
        from true_rl_agent.tasks.push_recovery.failure_classifier import classify_force_recovery_episode
        names = ("episode_index", "stratum", "planned", "steps", "min_height", "max_pitch", "min_pitch", "max_roll", "max_drift",
            "final_speed", "final_stable", "event_steps", "event_ends", "verified", "actual_substeps", "force_world",
            "delivered_impulse", "total_mass", "max_force_error", "unplanned_wrench", "sequence_recovery_step")
        v = {name: getattr(self, name).detach().cpu().tolist() for name in names}
        strata, records = ("control", "low", "medium", "high", "repeated"), []
        for env_id in range(self.n):
            events = []
            for index in range(2):
                start, end = v["event_steps"][env_id][index], v["event_ends"][env_id][index]
                if start < 0:
                    continue
                events.append({"event_index": index, "start_control_step": start, "end_control_step": end,
                    "actual_substeps": v["actual_substeps"][env_id][index], "expected_substeps": self.expected_substeps,
                    "completed": v["actual_substeps"][env_id][index] == self.expected_substeps
                        and start == self.event_times[index] and end == self.event_times[index] + self.pulse_steps,
                    "injection_verified": v["verified"][env_id][index],
                    "force_world_n": v["force_world"][env_id][index],
                    "delivered_impulse_world_ns": v["delivered_impulse"][env_id][index],
                    "max_force_error_n": v["max_force_error"][env_id][index]})
            planned = v["planned"][env_id]
            recovery = v["sequence_recovery_step"][env_id]
            record = {"env_id": env_id, "episode_index": v["episode_index"][env_id], "episode_steps": v["steps"][env_id],
                "stratum": strata[v["stratum"][env_id]], "control_only": self.control_only, "planned_push_count": planned,
                "force_events": events,
                "sequence_recovered": recovery >= 0, "sequence_recovery_control_step": recovery if recovery >= 0 else None,
                "min_base_height_m": v["min_height"][env_id], "max_pitch_rad": v["max_pitch"][env_id],
                "min_pitch_rad": v["min_pitch"][env_id], "max_abs_roll_rad": v["max_roll"][env_id],
                "max_abs_pitch_rad": max(abs(v["max_pitch"][env_id]), abs(v["min_pitch"][env_id])),
                "max_xy_drift_m": v["max_drift"][env_id], "final_planar_speed_mps": v["final_speed"][env_id],
                "final_recovery_condition": v["final_stable"][env_id], "actual_total_mass_kg": v["total_mass"][env_id]}
            record.update(classify_force_recovery_episode(
                {**record, "unplanned_wrench": v["unplanned_wrench"][env_id]}, self.parameters))
            record["failure_label"] = record.pop("label")
            records.append(record)
        return records

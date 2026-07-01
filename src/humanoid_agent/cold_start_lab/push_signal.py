"""Runner-owned push event instrumentation for Phase 8E reward inputs."""

from __future__ import annotations

from typing import Any


PUSH_SIGNAL_SCHEMA = "humanoid_agent.phase8e.push_signal.v1"
BUFFER_ATTR = "phase8e_push_signal_buffer"


class Phase8EPushSignalBuffer:
    """One-step public buffer for actual push events observed during training."""

    def __init__(self, *, num_envs: int | None = None) -> None:
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.trigger_count = 0
        self.last_env_ids: Any = None
        self.last_vector_xy: Any = None
        self.last_error: str | None = None

    def record(self, *, env_ids: Any, vector_xy: Any, num_envs: int | None = None) -> None:
        if num_envs is not None:
            self.num_envs = int(num_envs)
        self.trigger_count += 1
        self.last_env_ids = _clone_public_value(env_ids)
        self.last_vector_xy = _clone_public_value(vector_xy)
        self.last_error = None

    def record_error(self, reason: str) -> None:
        self.last_error = reason

    def consume(self, rewards: Any) -> dict[str, Any]:
        active, vector = _zeros_for_rewards(rewards)
        env_ids = self.last_env_ids
        vector_xy = self.last_vector_xy
        provenance = "zero_fallback_no_push_event_observed"
        if env_ids is not None and vector_xy is not None:
            try:
                active, vector = _scatter_push(rewards, env_ids, vector_xy)
                provenance = "runner_owned_push_event_instrumentation"
            except Exception as exc:  # noqa: BLE001 - public fallback should not crash reward evaluation.
                self.record_error(f"consume_failed:{type(exc).__name__}")
                active, vector = _zeros_for_rewards(rewards)
                provenance = "zero_fallback_push_signal_consume_failed"
        self.last_env_ids = None
        self.last_vector_xy = None
        return {
            "schema_version": PUSH_SIGNAL_SCHEMA,
            "active": active,
            "vector": vector,
            "provenance": provenance,
            "trigger_count": self.trigger_count,
            "last_error": self.last_error,
        }

    def public_summary(self) -> dict[str, Any]:
        return {
            "schema_version": PUSH_SIGNAL_SCHEMA,
            "enabled": True,
            "source": "runner_wrapped_push_robot_event",
            "num_envs": self.num_envs,
            "trigger_count": self.trigger_count,
            "last_error": self.last_error,
        }


def install_phase8e_push_signal_probe(env: Any) -> dict[str, Any]:
    """Wrap IsaacLab's ``push_robot`` interval event to expose public push state."""

    base = _unwrap_env(env)
    buffer = getattr(base, BUFFER_ATTR, None)
    if not isinstance(buffer, Phase8EPushSignalBuffer):
        buffer = Phase8EPushSignalBuffer(num_envs=_num_envs(base))
        setattr(base, BUFFER_ATTR, buffer)

    event_manager = getattr(base, "event_manager", None)
    if event_manager is None or not callable(getattr(event_manager, "get_term_cfg", None)):
        return _install_report("not_installed", "event_manager_missing", buffer)
    try:
        term_cfg = event_manager.get_term_cfg("push_robot")
    except Exception as exc:  # noqa: BLE001
        return _install_report("not_installed", f"push_robot_term_missing:{type(exc).__name__}", buffer)
    if getattr(term_cfg, "_phase8e_push_signal_wrapped", False):
        return _install_report("already_installed", "push_robot_already_wrapped", buffer)

    original_func = getattr(term_cfg, "func", None)
    if not callable(original_func):
        return _install_report("not_installed", "push_robot_func_not_callable", buffer)

    def phase8e_recording_push(env_obj: Any, env_ids: Any, velocity_range: dict[str, Any], **kwargs: Any) -> None:
        return _recording_push_by_setting_velocity(
            env_obj=env_obj,
            env_ids=env_ids,
            velocity_range=velocity_range,
            kwargs=kwargs,
            original_func=original_func,
            buffer=buffer,
        )

    phase8e_recording_push.__name__ = "phase8e_recording_push_by_setting_velocity"
    term_cfg.func = phase8e_recording_push
    setattr(term_cfg, "_phase8e_push_signal_wrapped", True)
    try:
        event_manager.set_term_cfg("push_robot", term_cfg)
    except Exception as exc:  # noqa: BLE001
        term_cfg.func = original_func
        setattr(term_cfg, "_phase8e_push_signal_wrapped", False)
        return _install_report("not_installed", f"set_term_cfg_failed:{type(exc).__name__}", buffer)
    return _install_report("installed", "push_robot_wrapped", buffer)


def push_signal_public_summary(env: Any) -> dict[str, Any]:
    base = _unwrap_env(env)
    buffer = getattr(base, BUFFER_ATTR, None)
    if isinstance(buffer, Phase8EPushSignalBuffer):
        return buffer.public_summary()
    return {
        "schema_version": PUSH_SIGNAL_SCHEMA,
        "enabled": False,
        "source": "not_installed",
    }


def _recording_push_by_setting_velocity(
    *,
    env_obj: Any,
    env_ids: Any,
    velocity_range: dict[str, Any],
    kwargs: dict[str, Any],
    original_func: Any,
    buffer: Phase8EPushSignalBuffer,
) -> None:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        from isaaclab.utils import math as math_utils  # type: ignore[import-not-found]  # noqa: PLC0415

        asset_cfg = kwargs.get("asset_cfg")
        asset_name = getattr(asset_cfg, "name", "robot")
        asset = env_obj.scene[asset_name]
        if env_ids is None:
            env_ids = torch.arange(env_obj.scene.num_envs, device=asset.device)
        vel_w = asset.data.root_vel_w[env_ids]
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        sampled = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        buffer.record(env_ids=env_ids, vector_xy=sampled[:, 0:2], num_envs=int(env_obj.scene.num_envs))
        asset.write_root_velocity_to_sim(vel_w + sampled, env_ids=env_ids)
    except Exception as exc:  # noqa: BLE001
        buffer.record_error(f"recording_push_failed:{type(exc).__name__}")
        original_func(env_obj, env_ids, velocity_range=velocity_range, **kwargs)


def _install_report(status: str, reason: str, buffer: Phase8EPushSignalBuffer) -> dict[str, Any]:
    return {
        "schema_version": PUSH_SIGNAL_SCHEMA,
        "status": status,
        "reason": reason,
        "buffer": buffer.public_summary(),
    }


def _zeros_for_rewards(rewards: Any) -> tuple[Any, Any]:
    torch_mod = _torch_for(rewards)
    if torch_mod is not None:
        return (
            torch_mod.zeros_like(rewards),
            torch_mod.zeros((int(rewards.shape[0]), 2), device=rewards.device, dtype=rewards.dtype),
        )
    if isinstance(rewards, list | tuple):
        return [0.0 for _ in rewards], [[0.0, 0.0] for _ in rewards]
    return 0.0, [[0.0, 0.0]]


def _scatter_push(rewards: Any, env_ids: Any, vector_xy: Any) -> tuple[Any, Any]:
    torch_mod = _torch_for(rewards)
    if torch_mod is not None:
        active, vector = _zeros_for_rewards(rewards)
        ids = env_ids.to(device=rewards.device, dtype=torch_mod.long) if _torch_for(env_ids) is not None else torch_mod.tensor(env_ids, dtype=torch_mod.long, device=rewards.device)
        values = vector_xy.to(device=rewards.device, dtype=rewards.dtype) if _torch_for(vector_xy) is not None else torch_mod.tensor(vector_xy, dtype=rewards.dtype, device=rewards.device)
        active[ids] = 1.0
        vector[ids, :] = values[:, 0:2]
        return active, vector

    active, vector = _zeros_for_rewards(rewards)
    ids = _to_list(env_ids)
    values = _to_list(vector_xy)
    if not isinstance(active, list) or not isinstance(vector, list):
        return active, vector
    for index, row in zip(ids, values, strict=True):
        env_index = int(index)
        active[env_index] = 1.0
        vector[env_index] = [float(row[0]), float(row[1])]
    return active, vector


def _to_list(value: Any) -> list[Any]:
    if _torch_for(value) is not None:
        return value.detach().cpu().tolist()
    if callable(getattr(value, "tolist", None)):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clone_public_value(value: Any) -> Any:
    if value is None:
        return None
    if _torch_for(value) is not None:
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_public_value(item) for item in value)
    return value


def _torch_for(value: Any) -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return torch if isinstance(value, torch.Tensor) else None


def _unwrap_env(env: Any) -> Any:
    current = env
    for _ in range(8):
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped
    return current


def _num_envs(env: Any) -> int | None:
    for path in (("scene", "num_envs"), ("num_envs",)):
        current = env
        for name in path:
            current = getattr(current, name, None)
            if current is None:
                break
        if current is not None:
            try:
                return int(current)
            except (TypeError, ValueError):
                pass
    return None

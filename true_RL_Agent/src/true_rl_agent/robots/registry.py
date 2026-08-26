"""Unique registry for exact robot profiles and runtime adapters."""

from __future__ import annotations

from true_rl_agent.robots.base import RobotProfile, RobotRuntimeAdapter


class RobotRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, RobotProfile] = {}
        self._runtime_adapters: dict[str, RobotRuntimeAdapter] = {}

    def register_profile(self, profile: RobotProfile) -> None:
        if not isinstance(profile, RobotProfile):
            raise TypeError("robot registry profile must be RobotProfile")
        profile_id = profile.robot_profile_id
        if profile_id in self._profiles:
            raise ValueError(f"robot profile already registered: {profile_id}")
        self._profiles[profile_id] = profile

    def register_runtime_adapter(self, adapter: RobotRuntimeAdapter) -> None:
        if not isinstance(adapter, RobotRuntimeAdapter):
            raise TypeError("robot registry adapter must be RobotRuntimeAdapter")
        adapter_id = adapter.runtime_adapter_id
        if adapter_id in self._runtime_adapters:
            raise ValueError(f"robot runtime adapter already registered: {adapter_id}")
        if adapter.robot_profile_id not in self._profiles:
            raise ValueError(
                "robot runtime adapter references an unregistered profile: "
                + adapter.robot_profile_id
            )
        self._runtime_adapters[adapter_id] = adapter

    def profile(self, robot_profile_id: str) -> RobotProfile:
        try:
            return self._profiles[robot_profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot profile: {robot_profile_id}") from exc

    def runtime_adapter(self, runtime_adapter_id: str) -> RobotRuntimeAdapter:
        try:
            return self._runtime_adapters[runtime_adapter_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown robot runtime adapter: {runtime_adapter_id}"
            ) from exc

    def execution(
        self,
        *,
        robot_profile_id: str,
        runtime_adapter_id: str,
    ) -> tuple[RobotProfile, RobotRuntimeAdapter]:
        profile = self.profile(robot_profile_id)
        adapter = self.runtime_adapter(runtime_adapter_id)
        if adapter.robot_profile_id != profile.robot_profile_id:
            raise ValueError("robot runtime adapter/profile identity mismatch")
        return profile, adapter

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def runtime_adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtime_adapters))


__all__ = ["RobotRegistry"]

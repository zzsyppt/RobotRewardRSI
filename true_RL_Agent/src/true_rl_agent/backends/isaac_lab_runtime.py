"""The single Isaac Sim 6 / Lab 3 PhysX environment configuration boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_ISAAC_LAB_PYTHON = "/data/yuyangliu/envs/isaaclab6_validation/bin/python"


def resolve_physx_environment(config: Any) -> Any:
    """Resolve upstream presets once, before applying task-owned settings."""

    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sensors import ContactSensorCfg
    from isaaclab_tasks.utils.hydra import resolve_presets
    from true_rl_agent.backends.isaac_lab_contact_sensor import SampledContactSensor

    config = resolve_presets(config)
    if config.sim.physics is None:
        config.sim.physics = PhysxCfg()
    if not isinstance(config.sim.physics, PhysxCfg):
        raise ValueError("the production runtime requires Isaac Lab PhysX")
    # The 4096-env workload exhausts the upstream 2**21 aggregate-pair buffer.
    # Capacity changes allocation only; an overflow drops physical interactions.
    config.sim.physics.gpu_total_aggregate_pairs_capacity = max(
        config.sim.physics.gpu_total_aggregate_pairs_capacity, 2**23,
    )
    for sensor in vars(config.scene).values():
        if isinstance(sensor, ContactSensorCfg):
            sensor.class_type = SampledContactSensor
            # Serialize PhysX's effective defaults, rather than unresolved None.
            if sensor.force_threshold is None:
                sensor.force_threshold = 1.0
            if sensor.max_contact_data_count_per_prim is None:
                sensor.max_contact_data_count_per_prim = 4
    return config


def require_physx_success(kit_log_path: str | Path) -> None:
    """PhysX can log an error and keep stepping; such a run is not valid evidence."""
    with Path(kit_log_path).open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if "[Error] [omni.physx.plugin]" in line:
                raise RuntimeError(f"invalid PhysX execution ({kit_log_path}): {line.strip()}")

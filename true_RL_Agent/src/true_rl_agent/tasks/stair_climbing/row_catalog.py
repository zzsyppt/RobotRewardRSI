"""Thin wiring catalog for robot rows of the Stair task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.composition.cell_manifest import RobotTaskBinding
from true_rl_agent.json_contract import load_strict_json_object

STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID = (
    "stair.g1_37dof.clean_legacy_v1"
)
STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID = "stair.g1_37dof.unified_v1"
STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID = "stair.go2_12dof.unified_v1"


@dataclass(frozen=True)
class StairRobotRow:
    cell_variant_id: str
    robot_task_cell_id: str
    protocol_relative_path: str
    binding_relative_path: str
    environment_id: str
    unified: bool

    def load_binding(self, *, project_root: Path | None = None) -> RobotTaskBinding:
        root = project_root or Path(__file__).resolve().parents[4]
        binding = RobotTaskBinding.from_mapping(
            load_strict_json_object(
                root / self.binding_relative_path,
                context=f"Stair row binding {self.cell_variant_id}",
                require_canonical=True,
                trailing_newline=True,
            )
        )
        if binding.cell_variant_id != self.cell_variant_id:
            raise ValueError("Stair row catalog/binding cell identity mismatch")
        return binding


_ROWS = {
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.g1_37dof_v1",
        protocol_relative_path="configs/tasks/stair_protocol_v2.json",
        binding_relative_path=(
            "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json"
        ),
        environment_id="stair_clean_legacy_environment_v1",
        unified=False,
    ),
    STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.g1_37dof_v1",
        protocol_relative_path="configs/tasks/stair_unified_protocol_v1.json",
        binding_relative_path=(
            "configs/tasks/stair_g1_37dof_unified_binding_v1.json"
        ),
        environment_id="stair_unified_c0_c1_environment_v1",
        unified=True,
    ),
    STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.go2_12dof_v1",
        protocol_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
        ),
        binding_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_binding_v1.json"
        ),
        environment_id="stair_go2_12dof_unified_environment_v1",
        unified=True,
    ),
}


def stair_robot_row(cell_variant_id: str) -> StairRobotRow:
    try:
        return _ROWS[cell_variant_id]
    except KeyError as exc:
        raise KeyError("unknown Stair robot row: " + str(cell_variant_id)) from exc


def active_unified_stair_cell_variant_ids() -> tuple[str, ...]:
    return tuple(row.cell_variant_id for row in _ROWS.values() if row.unified)


__all__ = [
    "STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID",
    "STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID",
    "StairRobotRow",
    "active_unified_stair_cell_variant_ids",
    "stair_robot_row",
]

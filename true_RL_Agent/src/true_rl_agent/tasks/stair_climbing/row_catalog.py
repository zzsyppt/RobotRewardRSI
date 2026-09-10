"""Thin wiring catalog for robot rows of the Stair task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.composition.cell_manifest import RobotTaskBinding
from true_rl_agent.json_contract import load_strict_json_object

STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID = (
    "stair.g1_37dof.clean_legacy_v1"
)
STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID = "stair.g1_37dof.unified_v1"
STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID = "stair.go2_12dof.unified_v1"
STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID = "stair.g1_37dof.unified_v2"
STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID = "stair.go2_12dof.unified_v2"
DEFAULT_STAIR_CELL_VARIANT_ID = "stair.g1_37dof.unified_v3"


@dataclass(frozen=True)
class StairRobotRow:
    cell_variant_id: str
    robot_task_cell_id: str
    protocol_relative_path: str
    binding_relative_path: str
    environment_id: str
    unified: bool
    active: bool
    frozen_task_protocol_hash: str | None = None
    frozen_binding_hash: str | None = None

    def require_frozen_protocol_identity(self, task_protocol_hash: str) -> None:
        if (
            self.frozen_task_protocol_hash is not None
            and task_protocol_hash != self.frozen_task_protocol_hash
        ):
            raise ValueError("frozen Stair predecessor protocol identity drifted")

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
        if (
            self.frozen_binding_hash is not None
            and binding.binding_hash != self.frozen_binding_hash
        ):
            raise ValueError("frozen Stair predecessor binding identity drifted")
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
        active=False,
    ),
    STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.g1_37dof_v1",
        protocol_relative_path="configs/tasks/stair_unified_protocol_v1.json",
        binding_relative_path=(
            "configs/tasks/stair_g1_37dof_unified_binding_v1.json"
        ),
        environment_id="stair_unified_c0_c1_environment_v1",
        unified=True,
        active=False,
        frozen_task_protocol_hash=(
            "e65239927cbf98f6cf12b0ddb959c6bd1390bdf7fd1c6a262fbbab7f1f04dcf8"
        ),
        frozen_binding_hash=(
            "ca8b1a7f91e81e10d637253590ba4d6860d5fe3723b1dd7a08266dd7a6691c97"
        ),
    ),
    STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.go2_12dof_v1",
        protocol_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
        ),
        binding_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_binding_v1.json"
        ),
        environment_id="stair_go2_12dof_unified_environment_v1",
        unified=True,
        active=False,
        frozen_task_protocol_hash=(
            "fae31b3d9dd9b971e870a4ee5379e24d25a843929a8f2d20944887efc0610e59"
        ),
        frozen_binding_hash=(
            "97de2d11aa349dec4bc75d34f9ad153eb4aaf49dfd1b71e7dcd9aa4e659dec71"
        ),
    ),
    STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.g1_37dof_v1",
        protocol_relative_path="configs/tasks/stair_unified_protocol_v2.json",
        binding_relative_path=(
            "configs/tasks/stair_g1_37dof_unified_binding_v2.json"
        ),
        environment_id="stair_unified_c0_c1_environment_v2",
        unified=True,
        active=True,
    ),
    STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID: StairRobotRow(
        cell_variant_id=STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID,
        robot_task_cell_id="unitree_stair_climbing.go2_12dof_v1",
        protocol_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_protocol_v2.json"
        ),
        binding_relative_path=(
            "configs/tasks/stair_go2_12dof_unified_binding_v2.json"
        ),
        environment_id="stair_go2_12dof_unified_environment_v2",
        unified=True,
        active=True,
    ),
    "stair.g1_37dof.unified_v3": StairRobotRow(
        cell_variant_id="stair.g1_37dof.unified_v3",
        robot_task_cell_id="unitree_stair_climbing.g1_37dof_v1",
        protocol_relative_path="configs/tasks/stair_unified_protocol_v3.json",
        binding_relative_path="configs/tasks/stair_g1_37dof_unified_binding_v3.json",
        environment_id="stair_unified_c0_c1_environment_v3",
        unified=True,
        active=True,
    ),
    "stair.go2_12dof.unified_v3": StairRobotRow(
        cell_variant_id="stair.go2_12dof.unified_v3",
        robot_task_cell_id="unitree_stair_climbing.go2_12dof_v1",
        protocol_relative_path="configs/tasks/stair_go2_12dof_unified_protocol_v3.json",
        binding_relative_path="configs/tasks/stair_go2_12dof_unified_binding_v3.json",
        environment_id="stair_go2_12dof_unified_environment_v3",
        unified=True,
        active=True,
    ),
}


def stair_robot_row(cell_variant_id: str) -> StairRobotRow:
    try:
        return _ROWS[cell_variant_id]
    except KeyError as exc:
        raise KeyError("unknown Stair robot row: " + str(cell_variant_id)) from exc


def frozen_stair_predecessor_row(cell_variant_id: str) -> StairRobotRow | None:
    """Return a registered immutable predecessor, without owning protocol parsing."""

    row = _ROWS.get(cell_variant_id)
    if row is None or row.frozen_task_protocol_hash is None:
        return None
    return row


def active_unified_stair_cell_variant_ids() -> tuple[str, ...]:
    return tuple(
        row.cell_variant_id for row in _ROWS.values() if row.unified and row.active
    )


__all__ = [
    "DEFAULT_STAIR_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID",
    "STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID",
    "STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID",
    "StairRobotRow",
    "active_unified_stair_cell_variant_ids",
    "frozen_stair_predecessor_row",
    "stair_robot_row",
]

"""Admission for nominal measurements, separate from physical runtime claims.

The v3 interface is checked on CPU here. R1b must populate it from the actual
recorder; accepting this contract does not prove that a new rollout took place.
"""
from __future__ import annotations

import json
from pathlib import Path

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.numeric_evidence import file_sha256
from .trajectory_joint_limits import QUALIFIED_SOURCES, nominal_binding, NominalSourceUnavailable

CAPTURE_PROFILE = "stair.unified_first_episode.v3"
CONTRACT_VERSION = "stair.nominal_measurement_inputs.v1"
GEOMETRY_FILES = ("rough_surface.py", "ordered_evaluation.py", "unified_evaluation.py",
                  "unified_training.py", "isaaclab_irregular_terrain.py",
                  "isaaclab_rough_terrain.py", "isaaclab_evaluation_terrain.py")
HISTORY_FILE = Path(__file__).with_name("trajectory_historical_nominal.json")


def geometry_sources() -> dict:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in GEOMETRY_FILES}


def cell_bindings(cells) -> list[dict]:
    return [{"cell_id": cell.cell_id, "geometry_digest": cell.geometry_digest,
             "terrain_type": cell.terrain_type_index, "origin_world_m": list(cell.expected_origin_m)}
            for cell in cells]


def validate_measurement_contract(contract: dict, projection) -> None:
    """Validate recorded names/order/coordinate provenance, never infer it from q."""
    if contract.get("schema_version") != CONTRACT_VERSION:
        raise ValueError("nominal measurement capture contract version mismatch")
    if contract.get("sample_phase") != "pre_reset_control_step_end":
        raise ValueError("nominal measurement sample phase mismatch")
    q = contract.get("joint_position")
    if q is not None:
        expected = {"source": "robot.data.joint_pos[:, resolved.joint_ids]", "unit": "rad",
                    "coordinate_basis": "USD_joint_local_frames_no_added_zero_offset",
                    "joint_names": list(projection.joint_names),
                    "runtime_projection_hash": projection.projection_hash}
        if any(q.get(k) != v for k, v in expected.items()):
            raise ValueError("nominal q source/unit/coordinate/entity contract mismatch")
        names, ids = q.get("articulation_joint_names", []), q.get("resolved_joint_ids", [])
        if (len(ids) != len(projection.joint_names) or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < len(names) for i in ids)
                or [names[i] for i in ids] != list(projection.joint_names)):
            raise ValueError("nominal q resolved articulation order mismatch")
    terrain = contract.get("nominal_terrain")
    if terrain is not None:
        if (terrain.get("frame") != "world_z_up_translation_only"
                or terrain.get("placement") != "nominal_expected_layout_with_recorded_origins"):
            raise ValueError("nominal terrain source/frame/placement contract mismatch")


def resolve_capabilities(archive, reference: dict) -> tuple[dict, dict | None]:
    """Missing qualification disables that capability; contradictory bindings fail."""
    context = archive.manifest["metadata"]
    result = {"cycle": {"status": "available", "basis": "validated_current_marker_capture_contract"},
              "nominal_joint_limits": {"status": "unavailable", "reason": "q_coordinate_provenance_not_recorded"},
              "nominal_terrain": {"status": "unavailable", "reason": "nominal_geometry_source_and_axes_not_recorded"},
              "effective_limits": "unknown_not_recorded", "runtime_surface": "unknown_not_recorded"}
    bounds = None
    historical = reference["manifest_sha256"] in QUALIFIED_SOURCES
    contract = context.get("measurement_contract") if context["profile"] == CAPTURE_PROFILE else None
    if contract is not None:
        validate_measurement_contract(contract, archive.projection)
    if historical or contract is not None and contract.get("joint_position") is not None:
        try:
            bounds, binding = nominal_binding(archive)
        except NominalSourceUnavailable:
            if historical:
                raise
            result["nominal_joint_limits"] = {"status": "unavailable", "reason": "asset_outside_qualified_nominal_model"}
        else:
            result["nominal_joint_limits"] = {"status": "available", "binding": binding,
                "basis": "historical_audit" if historical else "structurally_validated_capture_contract",
                "runtime_execution": "historical_only" if historical else "not_verified_by_CPU_validation"}
    if historical:
        history = json.loads(HISTORY_FILE.read_bytes())
        if (set(history["manifest_sha256"]) != set(QUALIFIED_SOURCES)
                or history["cells"] != cell_bindings(archive.cells)):
            raise ValueError("historical nominal geometry/source binding mismatch")
        result["nominal_terrain"] = {"status": "available", "basis": "historical_nominal_only",
            "qualification_sha256": history["qualification_sha256"],
            "capsule_sha256": file_sha256(HISTORY_FILE),
            "placement": "nominal_expected_layout_with_recorded_origins; historical_importer_bytes_unverified"}
    elif contract is not None and contract.get("nominal_terrain") is not None:
        terrain = contract["nominal_terrain"]
        if terrain.get("cells") != cell_bindings(archive.cells):
            raise ValueError("nominal terrain cell/geometry/type/origin contract mismatch")
        result["nominal_terrain"] = {"status": "available", "basis": "structurally_validated_capture_contract",
            "contract_sha256": canonical_json_hash(terrain),
            "runtime_execution": "not_verified_by_CPU_validation", "placement": terrain["placement"]}
    return result, bounds

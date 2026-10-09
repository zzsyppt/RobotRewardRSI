"""Qualified nominal USD limits and coordinate provenance."""
import json
import hashlib
import math
from pathlib import Path

ASSET_SEMANTIC_SHA = '76edd27804bf710f15e1e56313df4541c4ba78bdfce2cb51c0fa927d58be3ab0'
ASSET_FILE = Path(__file__).with_name("go2_nominal_usd.json")
ASSET_SHA = "b8a1ab9846cca1b282258cd9b8c871d5f6746bd9bbb85e3d38a34921d90614bb"
# Historical q semantics and complete base contents reviewed for this bounded stage.
# New archives need their own coordinate/source qualification, not an asset-name guess.
QUALIFIED_SOURCES = {
    "0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3": "b7247d0c5408b8b0d1578089dea784a60dbca83aef94b5147b2e4f00499c57d4",
    "fe488cb848dbe61454c0b9cc4fa6d8f8fc278deb1157065170577d9e91be5f20": "cfa9059e0dbfca1a2d6932da5f1f841c878316a5f25e412befadc53b19f4f27c",
    "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83": "8a66d7d0da7540cb10ce45fd51edadc66e700434e52f94fd2bdef77d9bb44f3a",
}


class NominalSourceUnavailable(ValueError):
    """The archive is valid but this nominal asset model does not cover it."""


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def nominal_binding(archive) -> tuple[dict, dict]:
    """Only this reviewed asset closure/coordinate mapping is qualified."""
    asset = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    if _canonical_hash(asset) != ASSET_SEMANTIC_SHA:
        raise ValueError("nominal asset extraction differs from reviewed content")
    profile = archive.manifest["metadata"]["resolved_cell_manifest"]["robot_profile"]
    closure = profile["asset_closure"]
    entries = {e["sha256"]: e for e in closure["entries"]}
    root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
    identities = [root, root.rsplit("/", 1)[0]+"/Props/instanceable_meshes.usd"]
    if (closure["root_identity"] != root or closure["unresolved"] or len(entries) != 2
            or any(layer["sha256"] not in entries or entries[layer["sha256"]]["size_bytes"] != layer["bytes"]
                   or entries[layer["sha256"]]["identity"] != identity
                   for layer, identity in zip(asset["layers"], identities, strict=True))):
        raise NominalSourceUnavailable("archive nominal USD closure is not qualified")
    joints = {j["name"]: j for j in asset["joints"]}
    if set(archive.projection.joint_names) != set(joints):
        raise NominalSourceUnavailable("archive joints differ from qualified USD joint names")
    bounds = {name: {"lower": math.radians(joints[name]["physics:lowerLimit"]),
                     "upper": math.radians(joints[name]["physics:upperLimit"]),
                     "unit": "rad", "coordinate_status": "qualified_nominal"}
              for name in archive.projection.joint_names}
    binding = {"asset_extraction_sha256": ASSET_SHA, "closure": closure,
               "joint_names": list(archive.projection.joint_names), "q_unit": "rad",
               "q_semantics": "recorded_articulation_joint_position_not_default_pose_error",
               "coordinate_basis": "USD_joint_local_frames_no_added_zero_offset",
               "limit_kind": "nominal_USD_only", "effective_limits": "unknown_not_recorded"}
    return bounds, binding

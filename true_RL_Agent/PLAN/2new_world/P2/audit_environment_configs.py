"""Inventory every old/corrected config difference; this is not physics equivalence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"


def flatten(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            result.update(flatten(item, key))
        else:
            result[key] = item
    return result


def disposition(path):
    if path == "scene.contact_forces.class_type":
        return "contact_sampling", "PhysX kernel retained; sampled sensor restores history/timing cadence. See audit_contact_sampling_comparison.json."
    if path.endswith(".rot"):
        return "rotation_representation", "Same identity rotation, WXYZ to XYZW. Root/body policy projection and fixed-state traces verified."
    if "._usd_" in path:
        return "usd_schema_metadata", "Lab 3 property serialization declares USD API/namespace; shared physical field values are unchanged."
    if path.endswith(".compliant_contact_damping") or path.endswith(".compliant_contact_stiffness"):
        return "physx_default", "None leaves the PhysX rigid-material default at zero; compliant contacts are not enabled."
    if path == "events.add_base_mass.params.distribution":
        return "explicit_default", "Old function default uniform is now explicit. Distribution parameters are unchanged; CPU/GPU RNG sampling change remains declared."
    if path.startswith("scene.height_scanner.spawn."):
        return "ray_frame", "Lab 3 creates a FrameView for the same ray attachment. Ray layout and same-state height scan were measured."
    if path in {"scene.height_scanner.attach_yaw_only", "scene.height_scanner.history_length"}:
        return "retired_inactive_field", "Old unused alias / zero history removed; ray_alignment and update_period are unchanged."
    if path.startswith("commands.base_velocity.") and "visualizer" not in path:
        return "command_metadata", "New names/thresholds describe command metrics. Command ranges, resampling and reward formulas are unchanged."
    if path == "scene.terrain.terrain_generator.color_scheme":
        return "declared_visual_change", "Uniform material replaces vertex height colors: target trimesh lacks Lab's jet map. No RGB policy input; geometry and collision meshes unchanged."
    if any(token in path for token in ("visualizer", "visual_material", "sky_light.spawn.texture_file")):
        return "presentation", "Renderer assets/options; debug markers disabled, no policy RGB input. Native recordings use the same scene geometry."
    if path.startswith(("sim.render.", "video_recorder.")) or path == "num_rerenders_on_reset":
        return "render_configuration", "New optional renderer fields; upstream video recorder not invoked. Viewport RGB needs its own annotator warmup; scene RTX reset loop is inactive."
    if path in {"sim.log_dir", "sim.logging_level", "sim.save_logs_to_file", "io_descriptors_output_dir", "isaac_teleop"}:
        return "execution_metadata", "Logging/IO metadata or disabled teleoperation; no dynamics or workload change."
    if path == "sim.physics.enable_external_forces_every_iteration":
        return "explicit_physics_default", "False equals the measured legacy USD setting; see timeout_bootstrap_source_audit and physics probes."
    if path == "sim.physics.gpu_total_aggregate_pairs_capacity":
        return "measured_capacity_fix", "Raise GPU allocation from 2**21 to 2**23 after the 4096-env job exhausted aggregate pairs. Overflowed runs are invalid; task dynamics and layout stay fixed."
    if path == "sim.device":
        return "execution_placement", "Probe uses the logical device exposed by its GPU lease; this is not a physics parameter."
    if path in {"sim.physics.class_type", "sim.physics.enable_scene_query_support", "sim.use_newton_actuators"}:
        return "backend_selection", "Explicit sole PhysX backend; Newton actuators and scene query support disabled as before."
    if path == "scene.terrain.use_terrain_origins":
        return "explicit_scene_default", "True preserves terrain-origin placement; recorded initial terrain assignment is unchanged."
    if path == "scene.contact_forces.track_friction_forces":
        return "inactive_sensor_feature", "New tangential-force feature remains disabled."
    if path in {"recorders.dataset_compression", "recorders.export_in_close"}:
        return "inactive_dataset_export", "Both configs select EXPORT_NONE; dataset compression/close export do not affect the task's transition recorder."
    if path.endswith((".spawn.spawn_path", ".disable_shape_checks", ".spawn.physics_material", ".spawn.physics_material_path")):
        return "inactive_spawn_option", "New optional spawn/validation fields; no material override or alternate spawn path selected."
    return "UNRESOLVED", "Requires source-level review."


report = {
    "scope": "Complete config-difference inventory, not a numerical-equivalence certificate",
    "normalization": "Only sim.physx container is renamed to sim.physics before leaf comparison",
    "configurations": {},
}
for robot in ("g1", "go2"):
    for mode in ("train", "eval"):
        name = f"{robot}_{mode}"
        old = json.loads((ROOT / "environment_cfg_old" / f"{name}.json").read_text())
        new = json.loads((ROOT / "audit_final_environment_cfg" / f"{name}.json").read_text())
        old["sim"]["physics"] = old["sim"].pop("physx")
        left, right = flatten(old), flatten(new)
        changes = []
        for path in sorted(left.keys() | right.keys()):
            a, b = left.get(path, "MISSING"), right.get(path, "MISSING")
            if a != b:
                owner, reason = disposition(path)
                changes.append({"path": path, "old": a, "new": b, "disposition": owner, "reason": reason})
        fields = ("update_period", "force_threshold", "max_contact_data_count_per_prim", "history_length", "track_air_time")
        report["configurations"][name] = {
            "contact_effective_fields_equal": {
                field: old["scene"]["contact_forces"][field] == new["scene"]["contact_forces"][field]
                for field in fields
            },
            "changes": changes,
            "unresolved_paths": [item["path"] for item in changes if item["disposition"] == "UNRESOLVED"],
        }
report["inventory_complete"] = all(not item["unresolved_paths"] for item in report["configurations"].values())
(ROOT / "audit_environment_config_disposition.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({name: {"changes": len(item["changes"]), "unresolved": item["unresolved_paths"], "contact_fields_equal": all(item["contact_effective_fields_equal"].values())} for name, item in report["configurations"].items()}))

"""R5 control copy: frozen Stair operand arithmetic; exact terminal projection binding."""
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_stair_confirmation_arithmetic import encoded, decode_column
from terminal_projection_check import load_contract, verify_episode_projection
FROZEN_SOURCE_ROOT = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r5_final')
SOURCE_MANIFEST = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5/runtime_source_manifest.json')

def verify_stair_operand_sources(raw_numeric, *, episode_header_path, raw_export_manifest_path, packing_receipt_path):
    """Check five supplied force/surface columns against a bound original NPZ.

Force reductions are recomputed here. Surface uses the task's normative geometry
primitive, never an event reducer/exporter. This supplements, without rewriting,
the conditional coverage recorded by ``verify_stair_episode``.
"""
    from dataclasses import fields
    import sys

    raw = raw_numeric
    header_path, manifest_path, packing_path = map(Path, (episode_header_path, raw_export_manifest_path, packing_receipt_path))
    header, manifest, packing = [json.loads(p.read_bytes()) for p in (header_path, manifest_path, packing_path)]
    checks, unchecked, watched = [], [], {}
    file_sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def record(name, passed, **detail):
        checks.append({"check": name, "status": "pass" if passed else "fail", **detail})
    def file_ref(path, reference, label):
        sha = file_sha(path)
        watched[str(path.resolve())] = sha
        record(label, sha == reference["sha256"] and path.stat().st_size == reference["bytes"],
               path=str(path.resolve()), expected_sha256=reference["sha256"], actual_sha256=sha)
    for path in (header_path, manifest_path, packing_path): watched[str(path.resolve())] = file_sha(path)
    header_name = f"episodes/{raw['episode']['episode_id']}/episode.json"
    references = [r for r in raw["source_materials"] if r["file"] == header_name]
    listed = [r for r in manifest["readable_materials"] if r["file"] == header_name]
    record("header_reference_unique", len(references) == len(listed) == 1, locator="raw_numeric/data/source_materials")
    if len(references) == len(listed) == 1:
        file_ref(header_path, references[0], "header_bytes_match_model_material_reference")
        file_ref(header_path, listed[0], "header_bytes_match_export_manifest")
    record("packing_raw_export_manifest_binding",
           file_sha(manifest_path) == packing["source_sha256"]["raw_manifest"]
           and Path(packing["source"]["raw_export"]).resolve() == manifest_path.parent.resolve()
           and header_path.resolve() == (manifest_path.parent / header_name).resolve(),
           packing_receipt_sha256=file_sha(packing_path), raw_manifest_sha256=file_sha(manifest_path))
    for key, header_key in (("definitions", "definitions"), ("static_values", "static_values"),
                            ("source_fields", "raw_fields"), ("source_direct_fields", "direct_curve_fields"),
                            ("original_values_npz", "original_values_npz")):
        record("header_mapping_" + key, encoded(raw[key]) == encoded(header[header_key]), locator="raw_numeric/data/" + key)
    projection = verify_episode_projection(raw["episode"], header["episode"], task="stair", contract=load_contract(FROZEN_SOURCE_ROOT, SOURCE_MANIFEST))
    record("episode_metadata_mapping", projection["status"] == "PASS_EXACT_RETAINED_FIELDS", projection=projection)
    npz_path = (header_path.parent / header["original_values_npz"]["file"]).resolve()
    if not npz_path.is_relative_to(header_path.parent.resolve()): raise ValueError("NPZ reference escaped episode directory")
    file_ref(npz_path, raw["original_values_npz"], "original_npz_matches_material_and_header")
    if any(c["status"] == "fail" for c in checks):
        return {"status": "FAIL_SOURCE_BINDING", "checks": checks, "unchecked": ["operand arithmetic not run after source binding failure"], "input_sha256": watched}
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    for prefix, declarations in (("raw__", header["raw_fields"]), ("static__", header["static_fields"])):
        record(prefix + "declared_field_set", {k[len(prefix):] for k in arrays if k.startswith(prefix)} == set(declarations))
        for name, declaration in declarations.items():
            a = arrays.get(prefix + name)
            passed = (a is not None and a.dtype.str == declaration["dtype"] and list(a.shape) == declaration["shape"]
                      and hashlib.sha256(a.tobytes()).hexdigest() == declaration["c_order_bytes_sha256"])
            record("npz_declared_field_" + prefix + name, passed, locator=f"episode.json/{'raw_fields' if prefix == 'raw__' else 'static_fields'}/{name}")
    if any(c["status"] == "fail" for c in checks):
        return {"status": "FAIL_DECLARED_NPZ_MAPPING", "checks": checks, "unchecked": ["operand arithmetic not run after field mapping failure"], "input_sha256": watched}
    n, definitions = raw["clock"]["last_step"], header["definitions"]
    record("complete_raw_clock", np.array_equal(arrays["raw__episode_length"], np.arange(1, n+1)))
    for section in ("raw_columns", "raw_discrete"):
        for name, value in raw[section].items():
            restored = decode_column(value, n).astype(np.dtype(value["dtype"]))
            a = arrays.get("raw__" + name)
            record("model_raw_column_" + name, a is not None and restored.shape == a.shape and restored.dtype == a.dtype and restored.tobytes() == a.tobytes(),
                   locator=f"raw_numeric/data/{section}/{name}")
    for name, value in header["static_values"].items():
        a = arrays["static__" + name]
        restored = np.asarray(value, dtype=a.dtype)
        record("static_operand_" + name, restored.shape == a.shape and restored.tobytes() == a.tobytes())
    record("terrain_type_mapping", int(arrays["static__terrain_types"]) == definitions["terrain_type_index"])
    outputs, operand_map = {}, {}
    for role, body_role in (("support", "support_effector"), ("forbidden", "forbidden_contact")):
        count = len(definitions["bodies"][body_role])
        for history in (False, True):
            field = role + "_contact_forces_w" + ("_history" if history else "")
            target = role + ("_buffer_max_force_norm_n" if history else "_force_norm_n")
            a = arrays.get("raw__" + field)
            if a is None:
                unchecked.append({"locator": "raw_numeric/data/direct_columns/" + target, "reason": "NPZ field missing", "field": field})
                continue
            shape = (n, definitions["contact_history"]["history_length"], count, 3) if history else (n, count, 3)
            record("axes_" + field, a.shape == shape, expected_shape=list(shape), actual_shape=list(a.shape), bodies=definitions["bodies"][body_role])
            if a.shape != shape: continue
            # Explicit vector-square/sum/sqrt, then history-slot reduction. No exporter call.
            norms = np.sqrt(np.sum(np.square(a.astype(np.float64)), axis=-1))
            outputs[target] = norms.max(axis=1) if history else norms
            operand_map[target] = "original_values.npz/raw__" + field
    geometry_sources = {}
    try:
        repo = FROZEN_SOURCE_ROOT
        sys.path.insert(0, str(repo / "src"))
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import UnifiedStairCellSpec
        from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
        for name in ("terrain_evidence", "rough_surface", "ordered_evaluation", "unified_evaluation"):
            module = sys.modules["true_rl_agent.tasks.stair_climbing." + name]
            path = Path(module.__file__).resolve()
            relative = path.relative_to(repo).as_posix()
            geometry_sources[relative] = file_sha(path)
            record("normative_geometry_source_" + name, geometry_sources[relative] == manifest["source_hashes"].get(relative),
                   current_sha256=geometry_sources[relative], export_source_sha256=manifest["source_hashes"].get(relative))
        geometry = definitions["geometry"]
        parameters = {f.name: geometry[f.name] for f in fields(UnifiedStairCellSpec)}
        for name in ("riser_heights_m", "tread_depths_m", "size_m"): parameters[name] = tuple(parameters[name])
        surface = stair_surface_query_for_ordered_cell(UnifiedStairCellSpec(**parameters))
        positions, origin = arrays["raw__support_pos_w"], arrays["static__terrain_env_origins_w"]
        shape = (n, len(definitions["bodies"]["support_effector"]), 3)
        record("surface_position_axes", positions.shape == shape and origin.shape == (3,), actual_shape=list(positions.shape), expected_shape=list(shape))
        if positions.shape == shape and origin.shape == (3,):
            relative = positions.astype(np.float64) - origin.astype(np.float64)
            outputs["support_surface_height_terrain_m"] = np.asarray([
                [surface.sample(progress_m=float(point[0]), lateral_m=float(point[1])).height_m for point in row] for row in relative], dtype=np.float64)
            operand_map["support_surface_height_terrain_m"] = "NPZ support_pos_w minus terrain_env_origins_w; header geometry; normative task surface.sample(x,y)"
    except (KeyError, TypeError, ValueError) as error:
        unchecked.append({"locator": "raw_numeric/data/direct_columns/support_surface_height_terrain_m", "reason": "geometry input/query unavailable", "detail": str(error)})
    comparisons = []
    for name, expected in outputs.items():
        if name not in raw["direct_columns"]:
            unchecked.append({"locator": "raw_numeric/data/direct_columns/" + name, "reason": "model column unavailable"})
            continue
        actual = decode_column(raw["direct_columns"][name], n)
        metadata = header["direct_curve_fields"][name]
        same_shape = actual.shape == expected.shape
        equal = same_shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        declaration_ok = (raw["direct_columns"][name]["dtype"] == metadata["dtype"]
                          and raw["direct_columns"][name]["component_shape"] == metadata["shape"][1:]
                          and list(actual.shape) == metadata["shape"] and actual.dtype.str == metadata["dtype"]
                          and hashlib.sha256(actual.tobytes()).hexdigest() == metadata["c_order_bytes_sha256"])
        detail = {"field": name, "locator": "raw_numeric/data/direct_columns/" + name,
                  "operand": operand_map[name], "shape": list(expected.shape), "value_count": int(expected.size),
                  "exact_declared_dtype_bytes_equal": equal, "header_direct_declaration_equal": declaration_ok,
                  "max_abs_error": float(np.max(np.abs(actual-expected))) if same_shape else None,
                  "recomputed_c_order_bytes_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                  "basis": "normative_geometry_export_consistency_not_independent_collision_physics" if "surface" in name else "independent_original_force_vector_arithmetic"}
        comparisons.append(detail)
        record("direct_operand_" + name, equal and declaration_ok, **detail)
    record("source_files_unchanged", all(file_sha(p) == h for p, h in watched.items()))
    failed = sum(c["status"] == "fail" for c in checks)
    return {"schema_version": "p3new.stair_source_operand_link.v1",
            "status": "FAIL" if failed else "PASS_SOURCE_OPERAND_LINK" if len(comparisons) == 5 and not unchecked else "PASS_CHECKED_WITH_UNCHECKED",
            "scope": "bound selected NPZ to supplied force/surface columns only; not asset identity, collision physics, termination classification, or N5 acceptance",
            "production_event_reducer_or_exporter_calls": 0, "checks": checks, "failed_n": failed, "unchecked": unchecked,
            "input_sha256": watched, "geometry_primitive_source_sha256": geometry_sources, "direct_operand_comparisons": comparisons,
            "remaining_boundaries": ["Task normative surface query reused; no independent collision/geometry implementation",
                                     "Body names/geometry are verified as declared header inputs, not reverified against simulator assets",
                                     "Original outcome and broader metadata semantics are not reclassified or endorsed"]}

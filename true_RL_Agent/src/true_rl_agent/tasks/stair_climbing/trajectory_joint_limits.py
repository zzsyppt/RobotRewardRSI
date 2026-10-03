"""Bounded offline Go2 nominal USD diagnostic; no normal delivery entry point."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import time

from true_rl_agent.evaluation.trajectory_translation.cohort import distribution
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.evaluation.trajectory_translation.joint_limits import (
    BASE_CATALOG_VERSION, CATALOG_VERSION, CONFIG, measure_joint_limits,
)
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.evaluation.process_report import report_bytes
from .trajectory_adapter import _load_contract, iter_stair_episodes
from .trajectory_artifacts import peak_memory_bytes
from .trajectory_report import validate_stair_trajectory_report

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
    if file_sha256(ASSET_FILE) != ASSET_SHA:
        raise ValueError("nominal asset extraction differs from reviewed bytes")
    asset = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
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


def summarize(episodes: list[dict]) -> dict:
    groups = {}
    for name in episodes[0]["joint_names"]:
        values = [next(r["value"] for r in e["records"] if r["entity_ids"] == [name]) for e in episodes]
        counts = {k: sum(v["counts"][k] for v in values) for k in values[0]["counts"]}
        seconds = {k: sum(v["sample_cell_seconds"][k] for v in values) for k in values[0]["sample_cell_seconds"]}
        groups[name] = {"episodes": len(values), "known_episodes": sum(v["status"] == "measured" for v in values),
            "counts": counts, "sample_cell_seconds": seconds,
            "episode_equal": {key: distribution([v[key] for v in values if v[key] is not None]) for key in
                ("minimum_signed_margin_rad", "range_occupancy_fraction", "inside_edge_band_fraction", "outside_fraction")},
            "pooled_sample_fraction": {key: counts[key+"_rows"]/counts["finite_rows"] if counts["finite_rows"] else None
                                       for key in ("inside_edge_band", "outside")},
            "pooled_sample_cell_time_fraction": {key: seconds[key]/seconds["finite"] if seconds["finite"] else None
                                                 for key in ("inside_edge_band", "outside")}}
    return {"episode_count": len(episodes), "groups": groups,
            "scope": "all recorded public-dev episodes; no outcome filtering",
            "weighting": "episode equal distributions; pooled samples and sample-cell time separately"}


def statements(binding: dict, summary: dict) -> list[dict]:
    result = []
    for name, group in summary["groups"].items():
        c = group["counts"]
        margin = group["episode_equal"]["minimum_signed_margin_rad"]["median"]
        occupancy = group["episode_equal"]["range_occupancy_fraction"]["median"]
        text = (f"{name}: {c['inside_edge_band_rows']}/{c['finite_rows']} finite recorded q samples lie inside the nominal USD 5% edge band; "
                f"{c['outside_rows']}/{c['finite_rows']} lie outside nominal bounds. "
                + (f"Episode-equal median minimum signed margin {margin:.6g}rad; median range occupancy {occupancy:.6g}. "
                   if margin is not None else "Margin and range occupancy unknown. ")
                + f"Unknown samples {c['unknown_rows']}/{c['total_rows']}. Effective runtime limits unknown; no saturation or cause claim.")
        result.append({"id": stable_id("nominal-limit-claim", binding, CONFIG, name, group), "text": text})
    return result


def _base_report(base_public: dict, archive, reference: dict) -> dict:
    base = validate_stair_trajectory_report(base_public, max_bytes=8192)
    identity = archive.manifest["identity"]
    domain = {"archive_hash": reference["manifest_sha256"], "evaluation_job_id": identity["job_id"],
              "projection_hash": archive.projection.projection_hash, "acceptance_split": "public_dev",
              "catalog_version": BASE_CATALOG_VERSION,
              **{k: identity[k] for k in ("task_id", "checkpoint_hash", "seed_namespace")},
              **{k: identity["evaluation_binding"].get(k) for k in ("task_protocol_hash", "evaluation_binding_hash")}}
    if "evaluation_plan_hash" in identity:
        domain["evaluation_plan_hash"] = identity["evaluation_plan_hash"]
    if any(base["domain"].get(k) != v for k, v in domain.items()):
        raise ValueError("nominal limit base source/domain mismatch")
    if _canonical_hash(base) != QUALIFIED_SOURCES.get(reference["manifest_sha256"]):
        raise ValueError("base content differs from qualified historical public report")
    return base


def build_joint_limit_supplement(reference: dict, output: Path, *, source_id: str, base_public: dict | None = None) -> dict:
    """Accept raw archive refs only; arbitrary reports/records/text cannot enter preview."""
    started, cpu = time.perf_counter(), time.process_time()
    archive = _load_contract(reference)
    if reference["manifest_sha256"] not in QUALIFIED_SOURCES:
        raise ValueError("unqualified historical joint coordinate source")
    bounds, asset_binding = nominal_binding(archive)
    base = _base_report(base_public, archive, reference) if base_public is not None else None
    binding = {"source_manifest_sha256": reference["manifest_sha256"],
               "identity": archive.manifest["identity"], "asset": asset_binding,
               "runtime_projection_hash": archive.projection.projection_hash, "catalog_version": CATALOG_VERSION}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    with (output/"joint_limits.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for ep in iter_stair_episodes(reference, source_id=source_id):
            records = measure_joint_limits(ep, bounds, q_unit="rad", asset_id=ASSET_SHA)
            row = {"episode_id": ep.episode_id, "run_index": ep.run_index, "env_index": ep.env_index,
                   "runner_seed": ep.source["runner_seed"], "dt_s": ep.dt_s,
                   "joint_names": list(ep.entities["joints"]), "records": records}
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            episodes.append(row)
    summary = summarize(episodes)
    claims = statements(binding, summary)
    report = {"schema_version": "stair.nominal_joint_limit_supplement.v1", "binding": binding,
              "catalog_version": CATALOG_VERSION, "base_catalog_version": BASE_CATALOG_VERSION,
              "measurement_config": CONFIG, "summary": summary, "optional_statements": claims,
              "status": "offline_not_production_delivery", "effective_limits": "unknown_not_recorded"}
    outputs = {"report.json": report}
    if base is not None:
        included, omitted, preview = [], [], base
        for claim in claims:
            candidate = {"schema_version": "stair.nominal_joint_limit_preview.v1", "status": "offline_not_production_delivery",
                         "base_report": base, "optional_nominal_joint_limit_statements": [*included, claim]}
            if report_bytes(candidate) <= 8192:
                included.append(claim)
                preview = candidate
            else:
                omitted.append({"id": claim["id"], "reason": "optional_whole_sentence_combined_budget"})
        outputs.update({"preview.json": preview, "preview_receipt.json": {
            "status": "offline_not_production_delivery", "included_ids": [c["id"] for c in included], "omitted": omitted,
            "source_manifest_sha256": reference["manifest_sha256"], "asset_extraction_sha256": ASSET_SHA,
            "base_content_hash": base["report_projection"]["content_hash"],
            "base_required_claim_ids_hash": base["report_projection"]["required_claim_ids_hash"],
            "combined_bytes": report_bytes(preview), "max_bytes": 8192}})
    for name, value in outputs.items():
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        (output/name).write_bytes(data.encode("utf-8"))
    package = Path(__file__).resolve().parents[2]
    dependencies = [Path(__file__), ASSET_FILE, *sorted((package/"evaluation/trajectory_translation").glob("*.py")),
                    *(Path(__file__).with_name(n) for n in ("trajectory_adapter.py", "trajectory_report.py", "runtime_projection.py")),
                    package/"runner/numeric_evidence.py"]
    manifest = {"schema_version": "stair.nominal_joint_limit_manifest.v1", "complete": True,
                "source_manifest_ref": reference, "binding": binding,
                "artifacts": {name: {"sha256": file_sha256(output/name), "bytes": (output/name).stat().st_size}
                              for name in ["joint_limits.jsonl", *outputs]},
                "code_sha256": {p.relative_to(package).as_posix(): file_sha256(p) for p in dependencies},
                "episode_count": len(episodes), "wall_seconds": time.perf_counter()-started,
                "cpu_seconds": time.process_time()-cpu, "peak_working_set_bytes": peak_memory_bytes(),
                "external_model_calls": 0, "gpu_runs": 0, "training_runs": 0,
                "status": "offline_pending_independent_gate"}
    (output/"manifest.json").write_bytes((json.dumps(manifest, indent=2)+"\n").encode())
    return manifest

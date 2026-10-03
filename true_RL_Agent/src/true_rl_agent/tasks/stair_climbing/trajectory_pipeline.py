"""Task-owned automatic numeric feedback, once per bound public-dev archive."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.trajectory_artifacts import build_evidence, build_card_artifacts
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import ARCHIVE_CAPABILITIES

LEGACY_PROFILE = "legacy_process_v1"
TRAJECTORY_PROFILE = "trajectory_numeric_v1"
PRODUCTION_PROFILE = "trajectory_numeric_v2"
PROFILES = (LEGACY_PROFILE, TRAJECTORY_PROFILE, PRODUCTION_PROFILE)
SUPPORTED_CELLS = ("stair.go2_12dof.unified_v2", "stair.go2_12dof.unified_v3")


def validate_build_receipt(receipt: Mapping[str, Any], public: Mapping[str, Any]) -> None:
    """Validate sealed private receipt against public bytes without disk access."""
    from true_rl_agent.evaluation.process_report import report_bytes
    production = public.get("schema_version") == "true_rl_agent.stair.behavior.public.v6"
    profile = PRODUCTION_PROFILE if production else TRAJECTORY_PROFILE
    if not isinstance(receipt, Mapping) or public.get("schema_version") not in {"true_rl_agent.stair.behavior.public.v5", "true_rl_agent.stair.behavior.public.v6"}:
        raise ValueError("trajectory build receipt requires trajectory public behavior")
    inputs, projection = receipt.get("inputs"), receipt.get("projection")
    if not isinstance(inputs, Mapping) or not isinstance(projection, Mapping):
        raise ValueError("trajectory build receipt inputs/projection missing")
    domain = public["domain"]
    identity = inputs.get("identity", {})
    expected = {"job_id": domain["evaluation_job_id"], "checkpoint_hash": domain["checkpoint_hash"],
                "evaluation_plan_hash": domain["evaluation_plan_hash"], "seed_namespace": domain["seed_namespace"],
                "task_id": domain["task_id"], "splits": ["public_dev"]}
    budget = projection.get("max_report_bytes")
    if (receipt.get("schema_version") != "stair.trajectory_build_receipt.v1" or receipt.get("complete") is not True
            or inputs.get("profile") != profile or projection.get("profile") != profile
            or receipt.get("cache_key") != canonical_json_hash(inputs)
            or inputs.get("archive_sha256") != domain["archive_hash"]
            or receipt.get("source_id") != inputs.get("source_id")
            or receipt.get("source_id") != "s" + domain["archive_hash"][:12]
            or not isinstance(identity, Mapping) or any(identity.get(k) != v for k, v in expected.items())
            or identity.get("evaluation_binding", {}).get("evaluation_binding_hash") != domain["evaluation_binding_hash"]
            or projection.get("card_id") != public["card_id"]
            or projection.get("public_content_hash") != public["report_projection"]["content_hash"]
            or not isinstance(projection.get("required_claim_ids"), list)
            or (len(set(projection["required_claim_ids"])) < 21 if production else len(set(projection["required_claim_ids"])) != 18)
            or canonical_json_hash(projection.get("required_claim_ids")) != public["report_projection"]["required_claim_ids_hash"]
            or type(budget) is not int or not 0 < budget <= (12288 if production else 8192)
            or projection.get("report_bytes") != report_bytes(public) or report_bytes(public) > budget):
        raise ValueError("trajectory build receipt binding mismatch")
    if production:
        from .trajectory_production import VERSIONS
        if inputs.get("production_versions") != VERSIONS or len(projection.get("base_semantic_mapping", {})) != 18:
            raise ValueError("production build semantic/version binding mismatch")


def validate_feedback_profile(profile: str, max_bytes: int, cell_variant_id: str) -> None:
    if profile not in PROFILES:
        raise ValueError("unsupported Stair behavior feedback profile")
    limit = 12288 if profile == PRODUCTION_PROFILE else 8192
    if type(max_bytes) is not int or not 0 < max_bytes <= limit:
        raise ValueError(f"Stair trajectory report byte budget must be in 1..{limit}")
    if profile == LEGACY_PROFILE and max_bytes != 8192:
        raise ValueError("trajectory report budget requires trajectory_numeric_v1")
    if profile != LEGACY_PROFILE and cell_variant_id not in SUPPORTED_CELLS:
        raise ValueError("trajectory_numeric_v1 supports only Stair Go2 unified v2/v3 qualified numeric v1/v2 archives")


def _code_hashes(*, production: bool = False) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    paths = [*(package / "evaluation/trajectory_translation").glob("*.py"),
             Path(__file__), Path(__file__).with_name("trajectory_artifacts.py"), Path(__file__).with_name("trajectory_adapter.py"),
             *(Path(__file__).with_name(name) for name in ("runtime_projection.py", "evaluation_semantics.py",
                 "unified_training.py", "unified_evaluation.py")),
             package / "composition/cell_manifest.py", package / "runner/numeric_evidence.py"]
    if production:
        paths.extend(Path(__file__).with_name(name) for name in (
            "trajectory_report.py", "trajectory_production.py", "trajectory_capabilities.py",
            "trajectory_cycles.py", "trajectory_joint_limits.py", "trajectory_terrain.py",
            "trajectory_historical_nominal.json", "go2_nominal_usd.json", "isaaclab_runtime_env.py",
            "rough_surface.py", "ordered_evaluation.py", "isaaclab_irregular_terrain.py",
            "isaaclab_rough_terrain.py", "isaaclab_evaluation_terrain.py"))
    return {p.relative_to(package).as_posix(): file_sha256(p) for p in sorted(paths)}


def _restart_incomplete(path: Path, root: Path) -> None:
    # These two directories are owned by this exact cache key. Never delete a
    # published manifest or a directory outside the phase's pipeline tree.
    if path.exists() and not (path / "manifest.json").exists():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("trajectory incomplete output is outside its phase directory")
        shutil.rmtree(path)


def build_trial_trajectory_card(reference: dict, *, expected_identity: Mapping[str, Any],
                                output_dir: Path, profile: str = TRAJECTORY_PROFILE) -> tuple[dict, dict]:
    """Reuse complete P1a/P1b artifacts; projection budget is deliberately separate.

    Completed phase summaries are replayed by the runner without coming here.
    A CPU retry may reuse its completed numeric/card stages, but cannot reuse a
    sibling, changed archive, implementation or numeric configuration.
    """
    archive = load_numeric_evidence_manifest(reference)
    if profile not in (TRAJECTORY_PROFILE, PRODUCTION_PROFILE):
        raise ValueError("automatic trajectory profile required")
    production = profile == PRODUCTION_PROFILE
    if (expected_identity.get("splits") != ["public_dev"]
            or any(archive["identity"].get(k) != v for k, v in expected_identity.items())):
        raise ValueError("trajectory evaluation/source identity mismatch")
    if archive["metadata"].get("profile") not in ARCHIVE_CAPABILITIES:
        raise ValueError("trajectory_numeric_v1 requires qualified Stair numeric v1/v2 archive")
    if archive["metadata"]["resolved_cell_manifest"].get("cell_variant_id") not in SUPPORTED_CELLS:
        raise ValueError("trajectory_numeric_v1 archive robot/cell is unsupported")
    from true_rl_agent.evaluation.trajectory_translation.measurement import NUMERIC_CONFIG
    source_id = "s" + reference["manifest_sha256"][:12]
    key = {"profile": profile, "archive_sha256": reference["manifest_sha256"],
           "identity": dict(expected_identity), "source_id": source_id,
           "selection": "current_archive_deterministic_examples.v1",
           "numeric_config": NUMERIC_CONFIG, "code_sha256": _code_hashes(production=production)}
    if production:
        from .trajectory_production import VERSIONS
        key["production_versions"] = VERSIONS
    cache_key = canonical_json_hash(key)
    root = Path(output_dir).resolve() / "trajectory" / cache_key
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    numeric, card = root / "numeric", root / "card"
    _restart_incomplete(numeric, root)
    numeric_reused = (numeric / "manifest.json").is_file()
    if not numeric_reused:
        build_evidence(reference, numeric, source_id)
    _restart_incomplete(card, root)
    card_reused = (card / "manifest.json").is_file()
    if not card_reused:
        build_card_artifacts(numeric / "manifest.json", file_sha256(numeric / "manifest.json"),
                             [], None, None, card)
    card_ref = {"manifest_path": str(card / "manifest.json"),
                "manifest_sha256": file_sha256(card / "manifest.json")}
    receipt = {"schema_version": "stair.trajectory_build_receipt.v1", "complete": True,
               "cache_key": cache_key, "inputs": key, "card_reference": card_ref,
               "numeric_manifest_sha256": file_sha256(numeric / "manifest.json"),
               "source_id": source_id, "numeric_reused": numeric_reused, "card_reused": card_reused,
               "build_wall_seconds": time.perf_counter() - started}
    if production:
        from .trajectory_production import build_measurements, load_measurements
        measurements = root / "measurements"
        _restart_incomplete(measurements, root)
        reused = (measurements / "manifest.json").is_file()
        measurements_ref = ({"manifest_path": str(measurements / "manifest.json"),
                             "manifest_sha256": file_sha256(measurements / "manifest.json")} if reused else
                            build_measurements(reference, measurements, source_id=source_id))
        measured = load_measurements(measurements_ref)
        if measured["archive_sha256"] != reference["manifest_sha256"]:
            raise ValueError("production cached measurements belong to another archive")
        receipt.update(measurements_reference=measurements_ref, measurements_reused=reused,
                       build_wall_seconds=time.perf_counter() - started)
    # This marks artifact construction only. The phase's existing atomic summary
    # and runner receipt mark successful projection and finalization.
    marker = root / "complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (previous.get("complete") is not True or previous.get("inputs") != key
                or previous.get("card_reference") != card_ref
                or (production and previous.get("measurements_reference") != receipt["measurements_reference"])
                or previous.get("numeric_manifest_sha256") != receipt["numeric_manifest_sha256"]):
            raise ValueError("trajectory completed artifact binding changed")
    else:
        marker.with_suffix(".partial").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        marker.with_suffix(".partial").replace(marker)
    return card_ref, receipt

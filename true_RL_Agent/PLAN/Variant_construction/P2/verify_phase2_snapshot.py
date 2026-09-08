from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.robots.base import (  # noqa: E402
    SUPPORT_EFFECTOR_ROLE,
    SemanticRole,
)
from true_rl_agent.robots.defaults import (  # noqa: E402
    G1_37DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.registry import RobotRegistry  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    build_stair_cell_catalog,
    compile_stair_cell_manifest,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row  # noqa: E402
from true_rl_agent.tasks.stair_climbing.runtime_projection import (  # noqa: E402
    stair_runtime_projection_from_execution,
    stair_runtime_projection_from_manifest,
)

SNAPSHOT_PATH = Path(__file__).with_name("P2_MORPHOLOGY_DECOUPLING_SNAPSHOT.json")
RUNTIME_RECEIPT_PATH = Path(__file__).with_name("PHASE2_RUNTIME_PROBE_RECEIPT.json")
SNAPSHOT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase2_morphology_receipt.v1"
)
RUNTIME_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase2_runtime_receipt.v1"
)
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase2_morphology_verify.v1"

_PRODUCT_PATHS = (
    "scripts/isaaclab_stair_rsl_rl_eval.py",
    "scripts/isaaclab_stair_rsl_rl_train.py",
    "src/true_rl_agent/composition/cell_manifest.py",
    "src/true_rl_agent/robots/base.py",
    "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/legacy_reward_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/legacy_runtime_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/legacy_transition_reader.py",
    "src/true_rl_agent/tasks/stair_climbing/plugin.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/resource_shadow.py",
    "src/true_rl_agent/tasks/stair_climbing/reward_inputs.py",
    "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_semantics.py",
    "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
    "configs/tasks/stair_evaluation_semantics_v7.json",
    "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json",
    "tests/integration/test_stair_evaluation_runtime_pipeline.py",
    "tests/unit/test_stair_episode_evidence.py",
    "tests/unit/test_stair_isaaclab_runtime_env.py",
    "tests/unit/test_stair_morphology_projection.py",
    "tests/unit/test_stair_terrain_evidence.py",
    "PLAN/Variant_construction/P2/capture_phase2_runtime_receipt.py",
)
_GENERIC_STAIR_PATHS = (
    "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_semantics.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/plugin.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
)
_BEHAVIORAL_MARKERS = (
    "_FOOT_BODY_NAMES",
    "_BASE_CONTACT_BODY_NAMES",
    "range(2)",
    "maximum_count=2",
    "Unitree G1 Stair Climbing",
    '["policy_compatibility"]["robot_schema"]',
)


def _normalized_text_hash(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    return value


def _without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "receipt_hash"}


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AssertionError(f"{context} mismatch: {actual!r} != {expected!r}")


def _validate_recorded_source_closure(
    source_hashes: Any, *, context: str
) -> dict[str, str]:
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise AssertionError(f"{context} is missing")
    normalized = {str(path): str(digest) for path, digest in source_hashes.items()}
    if any(
        not path
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in normalized.items()
    ):
        raise AssertionError(f"{context} contains an invalid recorded digest")
    return normalized


def _task_context() -> tuple[Any, StairClimbingTaskPlugin]:
    row = stair_robot_row(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / row.protocol_relative_path,
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    )
    return config, StairClimbingTaskPlugin(config)


def _product_tree_receipt() -> dict[str, Any]:
    hashes = {
        path: _normalized_text_hash(PROJECT_ROOT / path) for path in _PRODUCT_PATHS
    }
    return {
        "hash_mode": "normalized_lf_sha256",
        "file_count": len(hashes),
        "file_hashes": hashes,
        "product_tree_hash": canonical_json_hash(hashes),
    }


def _legacy_cell_receipt(
    config: Any, plugin: StairClimbingTaskPlugin
) -> dict[str, Any]:
    if plugin.task_spec.name != "Stair Climbing":
        raise AssertionError("Stair task display name contains robot identity")
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        config=config,
        plugin=plugin,
    )
    projection = stair_runtime_projection_from_manifest(manifest)
    requirement = next(
        item
        for item in plugin.robot_requirements().roles
        if item.role_id == SUPPORT_EFFECTOR_ROLE
    )
    return {
        "cell_variant_id": manifest.cell_variant_id,
        "task_display_name": plugin.task_spec.name,
        "manifest_hash": manifest.manifest_hash,
        "resolved_cell_task_protocol_hash": manifest.resolved_cell_task_protocol_hash,
        "legacy_training_binding_v1_grant_hash": (
            manifest.legacy_training_binding_v1_grant_hash
        ),
        "projection_hash": projection.projection_hash,
        "reward_input_schema_hash_vnext": projection.reward_input_schema_hash_vnext,
        "reward_capability_hash_vnext": projection.reward_capability_hash_vnext,
        "legacy_reward_adapter_id": projection.legacy_reward_adapter_id,
        "support_count": projection.support_count,
        "joint_count": projection.joint_count,
        "action_count": projection.action_count,
        "forbidden_contact_count": projection.forbidden_contact_count,
        "support_requirement": requirement.to_dict(),
    }


def _cardinality_receipt(
    config: Any, plugin: StairClimbingTaskPlugin
) -> dict[str, Any]:
    default_registry = build_default_robot_registry()
    definition = build_stair_cell_catalog(
        config=config,
        plugin=plugin,
        robot_registry=default_registry,
    ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
    original = default_registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
    original_supports = tuple(original.semantic_roles[SUPPORT_EFFECTOR_ROLE].targets)
    adapter = default_registry.runtime_adapter(definition.runtime_adapter_id)
    rows: list[dict[str, Any]] = []
    for support_count in (2, 4, 6):
        supports = original_supports + tuple(
            f"phase2_fixture_support_{index}_link"
            for index in range(len(original_supports), support_count)
        )
        facts = copy.deepcopy(dict(original.runtime_facts))
        facts["body_names"] = list(dict.fromkeys([*facts["body_names"], *supports]))
        facts["support_body_names"] = list(supports)
        facts["runtime_facts_hash"] = canonical_json_hash(
            {key: value for key, value in facts.items() if key != "runtime_facts_hash"}
        )
        roles = dict(original.semantic_roles)
        roles[SUPPORT_EFFECTOR_ROLE] = SemanticRole(
            SUPPORT_EFFECTOR_ROLE,
            "body",
            supports,
        )
        profile = replace(
            original,
            runtime_facts=facts,
            semantic_roles=roles,
            profile_hash=None,
        )
        binding = replace(
            definition.robot_task_binding,
            role_bindings={
                **definition.robot_task_binding.role_bindings,
                SUPPORT_EFFECTOR_ROLE: supports,
            },
            binding_hash=None,
        )
        projection = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=binding,
        )
        registry = RobotRegistry()
        registry.register_profile(profile)
        registry.register_runtime_adapter(adapter)
        fixture_definition = replace(
            definition,
            robot_task_binding=binding,
            contract_hashes={
                **definition.contract_hashes,
                "reward_input_vnext_schema_hash": (
                    projection.reward_input_schema_hash_vnext
                ),
                "reward_capability_vnext_hash": (
                    projection.reward_capability_hash_vnext
                ),
            },
            legacy_training_binding_v1_grant_hash=None,
        )
        manifest = compile_stair_cell_manifest(
            definition=fixture_definition,
            robot_registry=registry,
            requirements=plugin.robot_requirements(),
            base_task_protocol_hash=config.task_protocol_hash,
        ).require_supported()
        incomplete = replace(
            binding,
            role_bindings={
                **binding.role_bindings,
                SUPPORT_EFFECTOR_ROLE: supports[:-1],
            },
            binding_hash=None,
        )
        rejected = compile_stair_cell_manifest(
            definition=replace(fixture_definition, robot_task_binding=incomplete),
            robot_registry=registry,
            requirements=plugin.robot_requirements(),
            base_task_protocol_hash=config.task_protocol_hash,
        )
        rows.append(
            {
                "support_count": support_count,
                "resolved_support_count": len(
                    manifest.resolved_roles[SUPPORT_EFFECTOR_ROLE]
                ),
                "projection_hash": projection.projection_hash,
                "reward_input_schema_hash_vnext": (
                    projection.reward_input_schema_hash_vnext
                ),
                "reward_capability_hash_vnext": (
                    projection.reward_capability_hash_vnext
                ),
                "incomplete_selection_status": rejected.status,
                "incomplete_selection_reasons": list(rejected.reasons),
            }
        )
    if len({item["reward_input_schema_hash_vnext"] for item in rows}) != 3:
        raise AssertionError("2/4/6 Stair VNext schema identities are not distinct")
    return {"rows": rows}


def _registered_robot_names() -> set[str]:
    registry = build_default_robot_registry()
    role_vocabulary = {
        str(role_id)
        for profile_id in registry.profile_ids()
        for role_id in registry.profile(profile_id).semantic_roles
    }
    return {
        str(name)
        for profile_id in registry.profile_ids()
        for key in ("body_names", "joint_names")
        for name in registry.profile(profile_id).runtime_facts.get(key, ())
        if isinstance(name, str) and name and name not in role_vocabulary
    }


def _generic_stair_marker_hits(sources: Mapping[str, str]) -> list[str]:
    concrete_robot_names = _registered_robot_names()
    marker_hits: list[str] = []
    for relative_path, source in sources.items():
        marker_hits.extend(
            f"{relative_path}:{marker}"
            for marker in _BEHAVIORAL_MARKERS
            if marker in source
        )
        string_literals = {
            node.value
            for node in ast.walk(ast.parse(source, filename=relative_path))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        marker_hits.extend(
            f"{relative_path}:robot_name:{name}"
            for name in sorted(concrete_robot_names)
            if any(name in literal for literal in string_literals)
        )
    return marker_hits


def _boundary_receipt() -> dict[str, Any]:
    public_paths = tuple(
        sorted((PROJECT_ROOT / "src/true_rl_agent/robots").glob("*.py"))
    ) + (PROJECT_ROOT / "src/true_rl_agent/composition/cell_manifest.py",)
    public_task_imports = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in public_paths
        if "true_rl_agent.tasks." in path.read_text(encoding="utf-8")
    ]
    marker_hits = _generic_stair_marker_hits(
        {
            relative_path: (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            for relative_path in _GENERIC_STAIR_PATHS
        }
    )
    if public_task_imports or marker_hits:
        raise AssertionError(
            "Phase 2 ownership boundary failed: "
            f"public_task_imports={public_task_imports!r}, marker_hits={marker_hits!r}"
        )
    return {
        "public_task_imports": public_task_imports,
        "generic_stair_concrete_marker_hits": marker_hits,
        "registered_robot_name_count": len(_registered_robot_names()),
        "generic_file_count": len(_GENERIC_STAIR_PATHS),
    }


def _runtime_receipt(config: Any, plugin: StairClimbingTaskPlugin) -> dict[str, Any]:
    receipt = _load_json(RUNTIME_RECEIPT_PATH)
    _assert_equal(
        receipt.get("schema_version"),
        RUNTIME_RECEIPT_SCHEMA_VERSION,
        "Phase 2 runtime receipt schema",
    )
    _assert_equal(receipt.get("status"), "pass", "Phase 2 runtime status")
    _assert_equal(
        canonical_json_hash(_without_hash(receipt)),
        receipt.get("receipt_hash"),
        "Phase 2 runtime receipt hash",
    )
    if receipt.get("product_source_hash_mode") != "normalized_lf_sha256":
        raise AssertionError("Phase 2 runtime source hash mode changed")
    _validate_recorded_source_closure(
        receipt.get("product_source_hashes"),
        context="Phase 2 runtime source closure",
    )

    manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        config=config,
        plugin=plugin,
    )
    projection = stair_runtime_projection_from_manifest(manifest)
    runtime_projection = receipt["runtime_projection"]
    static_projection = {
        key: value
        for key, value in runtime_projection.items()
        if key != "resolved_runtime"
    }
    current_projection = projection.to_summary_dict()
    stable_projection_fields = (
        "schema_version",
        "cell_variant_id",
        "robot_profile_id",
        "runtime_adapter_id",
        "robot_profile_hash",
        "runtime_adapter_hash",
        "robot_task_binding_hash",
        "resolved_roles",
        "joint_names",
        "action_joint_names",
        "contact_sensor_name",
        "height_sensor_name",
        "contact_history_length",
        "policy_io_schema_hash",
        "sensor_schema_hash",
        "binding_facts_hash",
        "kinematic_base_source",
        "legacy_reward_adapter_id",
    )
    _assert_equal(
        {key: static_projection.get(key) for key in stable_projection_fields},
        {key: current_projection.get(key) for key in stable_projection_fields},
        "Phase 2 live legacy projection semantics",
    )
    parity = receipt.get("kinematic_base_parity")
    if not isinstance(parity, Mapping):
        raise AssertionError("Phase 2 kinematic-base parity evidence is missing")
    _assert_equal(
        parity.get("source"),
        "articulation_root",
        "Phase 2 G1 kinematic-base source",
    )
    _assert_equal(parity.get("status"), "exact", "Phase 2 base parity status")
    differences = parity.get("maximum_absolute_differences")
    if not isinstance(differences, Mapping) or not differences or any(
        value != 0.0 for value in differences.values()
    ):
        raise AssertionError("Phase 2 legacy/VNext kinematic-base parity drifted")
    resolved_runtime = runtime_projection["resolved_runtime"]
    expected_lengths = {
        "joint_ids": projection.joint_count,
        "base_body_ids": 1,
        "support_body_ids": projection.support_count,
        "contact_support_body_ids": projection.support_count,
        "contact_forbidden_body_ids": projection.forbidden_contact_count,
    }
    for name, expected in expected_lengths.items():
        values = resolved_runtime.get(name)
        if (
            not isinstance(values, list)
            or len(values) != expected
            or len(values) != len(set(values))
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in values
            )
        ):
            raise AssertionError(f"Phase 2 resolved runtime IDs are invalid: {name}")

    reference = resolve_official_stair_reference_program(
        config,
        project_root=PROJECT_ROOT,
    ).reward_program
    _assert_equal(
        receipt["cell"]["manifest_hash"],
        static_projection["resolved_cell_manifest_hash"],
        "captured runtime cell closure",
    )
    _assert_equal(
        receipt["action"]["shape"], [1, projection.action_count], "action shape"
    )
    _assert_equal(
        receipt["transition"]["support_pos_w"]["shape"],
        [1, projection.support_count, 3],
        "transition support shape",
    )
    _assert_equal(
        receipt["transition"]["source_schema_version"],
        "true_rl_agent.task.stair_climbing.transition.v2",
        "legacy transition schema",
    )
    _assert_equal(
        receipt["reward"]["input_type"], "StairRewardInputsV2", "reward input type"
    )
    _assert_equal(
        receipt["reward"]["input_schema_hash"],
        reference.contract.input_schema_hash,
        "legacy reward input schema",
    )
    _assert_equal(receipt["reward"]["field_count"], 42, "legacy reward field count")
    _assert_equal(
        receipt["reward"]["term_names"], list(reference.term_names), "reference terms"
    )
    _assert_equal(
        receipt["reward"]["total_rate"]["finite"], True, "reference reward finite"
    )
    return {
        "runtime_receipt_hash": receipt["receipt_hash"],
        "capture_kind": receipt["capture_kind"],
        "captured_repository_head": receipt["captured_repository_head"],
        "head_identity_semantics": receipt["head_identity_semantics"],
        "device": receipt["environment"]["device"],
        "cuda_device_name": receipt["environment"]["cuda_device_name"],
        "projection_hash": static_projection["projection_hash"],
        "current_projection_hash": projection.projection_hash,
        "kinematic_base_parity": dict(parity),
        "resolved_runtime": resolved_runtime,
        "observation_policy_shape": receipt["observation_after_step"]["policy"][
            "shape"
        ],
        "action_shape": receipt["action"]["shape"],
        "transition_source_schema": receipt["transition"]["source_schema_version"],
        "reward_input_schema_hash": receipt["reward"]["input_schema_hash"],
        "reward_term_count": len(receipt["reward"]["term_names"]),
        "status": receipt["status"],
    }


def _current_contract() -> dict[str, Any]:
    config, plugin = _task_context()
    return {
        "product_tree": _product_tree_receipt(),
        "legacy_cell": _legacy_cell_receipt(config, plugin),
        "cardinality": _cardinality_receipt(config, plugin),
        "ownership_boundary": _boundary_receipt(),
        "runtime": _runtime_receipt(config, plugin),
    }


def _validate_frozen_product_tree(product_tree: Any) -> None:
    if not isinstance(product_tree, Mapping):
        raise AssertionError("Phase 2 frozen product tree is missing")
    _assert_equal(
        product_tree.get("hash_mode"),
        "normalized_lf_sha256",
        "Phase 2 frozen product hash mode",
    )
    hashes = _validate_recorded_source_closure(
        product_tree.get("file_hashes"),
        context="Phase 2 frozen product source closure",
    )
    _assert_equal(
        product_tree.get("file_count"),
        len(hashes),
        "Phase 2 frozen product file count",
    )
    _assert_equal(
        product_tree.get("product_tree_hash"),
        canonical_json_hash(hashes),
        "Phase 2 frozen product tree hash",
    )


def _semantic_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    legacy = contract["legacy_cell"]
    legacy_fields = (
        "cell_variant_id",
        "task_display_name",
        "legacy_reward_adapter_id",
        "support_count",
        "joint_count",
        "action_count",
        "forbidden_contact_count",
        "support_requirement",
    )
    cardinality_rows = [
        {
            key: row[key]
            for key in (
                "support_count",
                "resolved_support_count",
                "incomplete_selection_status",
                "incomplete_selection_reasons",
            )
        }
        for row in contract["cardinality"]["rows"]
    ]
    boundary = contract["ownership_boundary"]
    runtime = contract["runtime"]
    runtime_fields = (
        "runtime_receipt_hash",
        "capture_kind",
        "kinematic_base_parity",
        "resolved_runtime",
        "observation_policy_shape",
        "action_shape",
        "transition_source_schema",
        "reward_input_schema_hash",
        "reward_term_count",
        "status",
    )
    return {
        "legacy_cell": {key: legacy[key] for key in legacy_fields},
        "cardinality": {"rows": cardinality_rows},
        "ownership_boundary": {
            "public_task_imports": boundary["public_task_imports"],
            "generic_stair_concrete_marker_hits": boundary[
                "generic_stair_concrete_marker_hits"
            ],
            "generic_file_count": boundary["generic_file_count"],
        },
        "runtime": {key: runtime[key] for key in runtime_fields},
    }


def verify(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    snapshot = _load_json(snapshot_path)
    _assert_equal(
        snapshot.get("schema_version"), SNAPSHOT_SCHEMA_VERSION, "snapshot schema"
    )
    _assert_equal(
        canonical_json_hash(_without_hash(snapshot)),
        snapshot.get("receipt_hash"),
        "Phase 2 snapshot hash",
    )
    contract = snapshot["verification_contract"]
    _validate_frozen_product_tree(contract.get("product_tree"))
    current_contract = _current_contract()
    _assert_equal(
        _semantic_contract(contract),
        _semantic_contract(current_contract),
        "Phase 2 current semantic contract",
    )
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "snapshot_receipt_hash": snapshot["receipt_hash"],
        "product_file_count": contract["product_tree"]["file_count"],
        "support_cardinalities": [
            row["support_count"] for row in contract["cardinality"]["rows"]
        ],
        "legacy_projection_hash": contract["legacy_cell"]["projection_hash"],
        "current_legacy_projection_hash": current_contract["legacy_cell"][
            "projection_hash"
        ],
        "runtime_receipt_hash": contract["runtime"]["runtime_receipt_hash"],
        "source_identity_semantics": "historical_provenance",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()
    payload = verify(args.snapshot)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

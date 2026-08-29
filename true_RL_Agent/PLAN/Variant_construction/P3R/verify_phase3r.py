from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (  # noqa: E402
    STAIR_ORDERED_PHASED_EVAL_COST_SCHEMA_VERSION,
    stair_phased_evaluation_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (  # noqa: E402
    StairClimbingPromptAdapter,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    build_stair_cell_catalog,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (  # noqa: E402
    build_unified_score_projection,
    build_unified_stair_summary,
    validate_unified_stair_summary,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (  # noqa: E402
    STAIR_TERRAIN_REGIME_C0,
    STAIR_TERRAIN_REGIME_C1,
    STAIR_TERRAIN_REGIME_C2,
    unified_assignment_summary,
    unified_initial_terrain_assignment,
)

SNAPSHOT_PATH = Path(__file__).with_name("PHASE3R_CONTRACT_SNAPSHOT.json")
RUNTIME_SMOKE_PATH = Path(__file__).with_name("PHASE3R_RUNTIME_SMOKE_RECEIPT.json")
SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3r_snapshot.v1"
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3r_verify.v1"
RUNTIME_SMOKE_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3r_runtime_smoke.v1"
)

_PRODUCT_PATHS = (
    "configs/tasks/stair_unified_acceptance_protocol_v6.json",
    "configs/tasks/stair_unified_evaluation_suite_v1.json",
    "configs/tasks/stair_unified_protocol_v1.json",
    "configs/tasks/stair_unified_training_profile_v1.json",
    "configs/tasks/stair_vnext_reward_authority_v1.json",
    "src/true_rl_agent/agent_core/decision.py",
    "src/true_rl_agent/composition/cell_manifest.py",
    "src/true_rl_agent/evaluation/acceptance_contract.py",
    "src/true_rl_agent/evaluation/acceptance_protocol.py",
    "src/true_rl_agent/evaluation/acceptance_replay.py",
    "src/true_rl_agent/evaluation/current_best.py",
    "src/true_rl_agent/evaluation/decision_record_contract.py",
    "src/true_rl_agent/tasks/stair_climbing/candidate_admission.py",
    "src/true_rl_agent/tasks/stair_climbing/component_smoke.py",
    "src/true_rl_agent/tasks/stair_climbing/composition.py",
    "src/true_rl_agent/tasks/stair_climbing/config.py",
    "src/true_rl_agent/tasks/stair_climbing/entrypoint_contract.py",
    "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_semantics.py",
    "src/true_rl_agent/tasks/stair_climbing/exploration.py",
    "src/true_rl_agent/tasks/stair_climbing/generation_session.py",
    "src/true_rl_agent/tasks/stair_climbing/generation_session_cost.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_irregular_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_unified_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/memory.py",
    "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/ordered_summary.py",
    "src/true_rl_agent/tasks/stair_climbing/phased_cost.py",
    "src/true_rl_agent/tasks/stair_climbing/phased_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/plugin.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/protocol_v2.py",
    "src/true_rl_agent/tasks/stair_climbing/public_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_program.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_vnext_v1/training_code/components.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_qualification.py",
    "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
    "src/true_rl_agent/tasks/stair_climbing/runtime_profile.py",
    "src/true_rl_agent/tasks/stair_climbing/selection.py",
    "src/true_rl_agent/tasks/stair_climbing/termination_capture.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_protocol.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_scoring.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_training.py",
    "src/true_rl_agent/tasks/stair_climbing/vnext_reward_authority.py",
    "src/true_rl_agent/tasks/stair_climbing/vnext_reward_contract.py",
    "PLAN/Variant_construction/P3R/capture_phase3r_runtime_smoke.py",
    "PLAN/Variant_construction/P3R/verify_phase3r.py",
)

_GENERIC_ACCEPTANCE_PATHS = (
    "src/true_rl_agent/agent_core/decision.py",
    "src/true_rl_agent/evaluation/acceptance_contract.py",
    "src/true_rl_agent/evaluation/acceptance_protocol.py",
    "src/true_rl_agent/evaluation/acceptance_replay.py",
    "src/true_rl_agent/evaluation/current_best.py",
    "src/true_rl_agent/evaluation/decision_record_contract.py",
)

_RETIRED_PHASE3A_PATHS = (
    "configs/tasks/stair_g1_37dof_irregular_acceptance_protocol_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_calibration_protocol_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_evaluation_suite_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_reference_gate_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_training_profile_v1.json",
    "src/true_rl_agent/tasks/stair_climbing/irregular_calibration.py",
    "src/true_rl_agent/tasks/stair_climbing/irregular_reference_gate.py",
    "src/true_rl_agent/tasks/stair_climbing/irregular_training.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_programs/g1_irregular_v1/training_code/components.py",
    "tests/unit/test_stair_phase3a_contract.py",
)

_RETIRED_ACTIVE_MARKERS = (
    "stair.g1_37dof.irregular_v1",
    "fixed_stair_ordered_c1",
    "g1_irregular_v1",
    "stair_g1_37dof_irregular",
    "unseen_final_holdout",
)


def _normalized_text_hash(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    return value


def _without_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "receipt_hash"}


def _resolve_default() -> Any:
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    with tempfile.TemporaryDirectory(prefix="p3r-contract-") as temporary:
        return resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=Path(temporary),
            ),
        )


def _assert_draft_formal_rejection() -> dict[str, str]:
    provider = StairClimbingCompositionProvider()
    failures: dict[str, str] = {}
    for mode in ("live", "formal_generation_session"):
        values = dict(provider.cli_defaults())
        values.update(
            {
                "mode": mode,
                "session_id": f"p3r-draft-{mode}",
                "rounds": 15,
                "max_generations": 15,
            }
        )
        with tempfile.TemporaryDirectory(prefix="p3r-draft-") as temporary:
            try:
                provider.resolve(
                    values=values,
                    context=TaskCompositionContext(
                        project_root=PROJECT_ROOT,
                        output_dir=Path(temporary),
                    ),
                )
            except ValueError as exc:
                message = str(exc)
                if "protocol is draft" not in message:
                    raise AssertionError(
                        f"draft {mode} failed for the wrong reason: {message}"
                    ) from exc
                failures[mode] = message
            else:
                raise AssertionError(f"draft Stair unexpectedly allowed {mode}")
    return failures


def _synthetic_score_receipt(config: Any) -> dict[str, Any]:
    suite = config.ordered_evaluation_suite
    progress_by_stratum = {
        "c0_id": 0.8,
        "c1_id": 0.6,
        "height_ood": 0.5,
        "tread_ood": 0.4,
        "joint_ood": 0.3,
        "phase_composition_ood": 0.7,
        "c1_conditional_frontier": 0.2,
    }
    per_seed: list[list[StairEpisodeOutcome]] = []
    for _ in range(config.public_seed_count):
        outcomes: list[StairEpisodeOutcome] = []
        for env_index, cell in enumerate(suite.private_cell_assignment("public_dev")):
            fraction = progress_by_stratum[cell.stratum_id]
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=cell.cell_id,
                    episode_steps=cell.episode_horizon_control_steps,
                    primary_reason="timeout",
                    success=False,
                    diagnostic_flags=(),
                    max_progress_m=fraction * cell.progress_goal_m,
                    progress_fraction=fraction,
                    active_terminal_terms=("time_out",),
                    evaluation_cell_id=cell.cell_id,
                )
            )
        per_seed.append(outcomes)
    summary = validate_unified_stair_summary(
        suite=suite,
        semantics=config.evaluation_semantics,
        split="public_dev",
        summary=build_unified_stair_summary(
            suite=suite,
            semantics=config.evaluation_semantics,
            split="public_dev",
            per_seed_outcomes=per_seed,
        ),
    )
    expected = {
        "j_id": 0.35,
        "j_ood": 0.2125,
        "j_primary": (2.0 * 0.2125 + 0.35) / 3.0,
    }
    for name, value in expected.items():
        if not math.isclose(float(summary[name]), value, abs_tol=1.0e-12):
            raise AssertionError(f"unified Stair {name} formula drifted")
    projection = build_unified_score_projection(summary)
    if set(projection) != {"j_primary", "j_ood", "c0_id", "c1_id"}:
        raise AssertionError("unified Stair score projection fields drifted")
    return {
        **expected,
        "score_projection_hash": canonical_json_hash(projection),
        "paired_seed_count": config.public_seed_count,
    }


def _contract_receipt() -> dict[str, Any]:
    composition = _resolve_default()
    manifest = composition.resolved_cell_manifest
    if manifest is None:
        raise AssertionError("unified Stair composition has no resolved manifest")
    config = composition.task_plugin.config
    suite = config.ordered_evaluation_suite
    profile = config.unified_training_profile
    catalog = build_stair_cell_catalog(
        config=config,
        plugin=composition.task_plugin,
    )
    if catalog.ids() != ("stair.g1_37dof.unified_v1",):
        raise AssertionError("Stair registry does not contain exactly one unified cell")
    definition = catalog.resolve(config.cell_variant_id)
    if definition.protocol_state != "draft":
        raise AssertionError("Phase 3R unified Stair protocol must remain draft")
    try:
        catalog.require_active(config.cell_variant_id)
    except ValueError:
        pass
    else:
        raise AssertionError("draft Stair catalog entry was treated as active")

    terrain_types, terrain_levels = unified_initial_terrain_assignment(
        num_envs=config.train_num_envs,
        seed=config.train_seed,
        profile=profile,
    )
    assignment = unified_assignment_summary(
        types=terrain_types,
        levels=terrain_levels,
        profile=profile,
    )
    if assignment["regime_counts"] != {
        STAIR_TERRAIN_REGIME_C0: 2048,
        STAIR_TERRAIN_REGIME_C1: 2048,
    }:
        raise AssertionError("unified Stair assignment is not exactly 50/50")
    if assignment["maximum_initial_level"] != {
        STAIR_TERRAIN_REGIME_C0: 5,
        STAIR_TERRAIN_REGIME_C1: 1,
    }:
        raise AssertionError("unified Stair initial level bounds drifted")
    if STAIR_TERRAIN_REGIME_C2 in assignment["regime_counts"]:
        raise AssertionError("C2 leaked into the Phase 3R training distribution")

    pattern_id = profile.pattern_ids[0]
    lower_tread = profile.c1_sequence(
        riser_base_m=0.1,
        tread_base_m=0.37,
        pattern_id=pattern_id,
        phase_steps=0,
    )
    upper_tread = profile.c1_sequence(
        riser_base_m=0.1,
        tread_base_m=0.43,
        pattern_id=pattern_id,
        phase_steps=0,
    )
    upper_riser = profile.c1_sequence(
        riser_base_m=0.2,
        tread_base_m=0.37,
        pattern_id=pattern_id,
        phase_steps=0,
    )
    if lower_tread["riser_heights_m"] != upper_tread["riser_heights_m"]:
        raise AssertionError("C1 tread choice still changes the riser sequence")
    if lower_tread["tread_depths_m"] != upper_riser["tread_depths_m"]:
        raise AssertionError("C1 riser choice still changes the tread sequence")

    expected_strata = {
        "c0_id",
        "c1_id",
        "height_ood",
        "tread_ood",
        "joint_ood",
        "phase_composition_ood",
        "c1_conditional_frontier",
    }
    minimum_cells = int(
        suite.qualification_rule["minimum_cells_per_stratum_per_cohort"]
    )
    cohort_counts: dict[str, dict[str, int]] = {}
    cohort_geometry_digests: dict[str, set[str]] = {}
    phase_labels: set[str] = set()
    for cohort in ("public_dev", "public_accept", "sealed_validation"):
        cells = suite.cells_for_cohort(cohort)
        counts = Counter(cell.stratum_id for cell in cells)
        if (
            set(counts) != expected_strata
            or set(counts.values()) != {minimum_cells}
            or len(cells) != len(expected_strata) * minimum_cells
        ):
            raise AssertionError(f"unified Stair stratum coverage drifted: {cohort}")
        expected_envs = len(cells) * suite.episodes_per_seed_per_cell
        if len(suite.private_cell_assignment(cohort)) != expected_envs:
            raise AssertionError(f"unified Stair evaluation cost drifted: {cohort}")
        digests = {cell.geometry_digest for cell in cells}
        if len(digests) != len(cells):
            raise AssertionError(
                f"unified Stair duplicate geometry within cohort: {cohort}"
            )
        if any(
            not math.isclose(
                cell.floor_depth_m,
                float(profile.terrain["floor_depth_m"]),
                abs_tol=1.0e-12,
            )
            for cell in cells
        ):
            raise AssertionError("evaluation terrain facts diverged from training")
        for cell in cells:
            if "cell_id" in cell.geometry_payload():
                raise AssertionError("geometry identity contains a cohort label")
            if cell.stratum_id == "phase_composition_ood":
                phase = cell.factor_levels["phase"]
                phase_labels.add(phase)
                if not profile.phase_changes_training_sequence(
                    pattern_id=cell.factor_levels["sequence_pattern"],
                    phase_steps=int(phase[1:]),
                ):
                    raise AssertionError("phase OOD reuses a training geometry")
        cohort_geometry_digests[cohort] = digests
        cohort_counts[cohort] = dict(sorted(counts.items()))
    cohorts = tuple(cohort_geometry_digests)
    for index, left in enumerate(cohorts):
        for right in cohorts[index + 1 :]:
            if cohort_geometry_digests[left] & cohort_geometry_digests[right]:
                raise AssertionError(f"formal Stair geometries overlap: {left}/{right}")
    if phase_labels != {"p1", "p2", "p4", "p6"}:
        raise AssertionError("phase OOD coverage is narrower than declared")
    formal_geometry_pool_size = len(set().union(*cohort_geometry_digests.values()))
    formal_evaluation_episode_count = sum(
        suite.seed_count(cohort)
        * len(suite.cells_for_cohort(cohort))
        * suite.episodes_per_seed_per_cell
        for cohort in cohorts
    )
    if formal_geometry_pool_size != 63 or formal_evaluation_episode_count != 1260:
        raise AssertionError("formal Stair geometry/cost contract drifted")
    if any(cell.terrain_regime_id == STAIR_TERRAIN_REGIME_C2 for cell in suite.cells):
        raise AssertionError("C2 appeared before Phase 3B")
    legacy_stair_010_guard_hits = [
        cell.cell_id
        for cell in suite.cells
        if cell.stratum_id == "c0_id"
        and all(
            math.isclose(value, 0.10, abs_tol=1.0e-12) for value in cell.riser_heights_m
        )
        and all(
            math.isclose(value, 0.40, abs_tol=1.0e-12) for value in cell.tread_depths_m
        )
    ]
    if legacy_stair_010_guard_hits:
        raise AssertionError("legacy stair_010 geometry leaked into the C0-ID guard")

    prompt_text = json.dumps(
        StairClimbingPromptAdapter(
            config,
            resolved_cell_manifest=manifest,
        ).build_prompt_capacity_preflight_sections(),
        sort_keys=True,
    )
    sealed_prompt_hits = [
        cell.cell_id
        for cell in suite.cells_for_cohort("sealed_validation")
        if cell.cell_id in prompt_text
    ]
    if sealed_prompt_hits or "cohort_geometry_overrides" in prompt_text:
        raise AssertionError("sealed Stair geometry leaked into the LLM prompt")

    command = config.unified_protocol.training["command"]
    evaluation_command = config.unified_protocol.runtime_controls["evaluation_command"]
    if (
        tuple(command["forward_mps_range"]) != (0.0, 0.8)
        or not math.isclose(
            float(command["forward_mps_range"][-1]),
            float(evaluation_command["forward_mps"]),
            abs_tol=1.0e-12,
        )
        or "floor_depth_m"
        in config.unified_protocol.evaluation_suite.task_projection()["geometry_policy"]
    ):
        raise AssertionError("unified Stair command/terrain authority diverged")

    acceptance = config.acceptance_protocol
    guard_names = tuple(guard.name for guard in acceptance.named_guards)
    if guard_names != ("j_ood", "c0_id", "c1_id"):
        raise AssertionError("acceptance v6 named guards drifted")
    if acceptance.score_source != "unified_score.j_primary.score_stats.mean":
        raise AssertionError("acceptance v6 primary score source drifted")

    authority = config.vnext_reward_authority
    active_terms = tuple(
        authority.runtime_authority_payload()["reference_equivalence"][
            "active_term_names"
        ]
    )
    if len(active_terms) != 15 or len(set(active_terms)) != 15:
        raise AssertionError("official VNext reference is not the 15-term program")
    reference = resolve_official_stair_reference_program(
        config,
        project_root=PROJECT_ROOT,
        resolved_cell_manifest=manifest,
    )
    if tuple(reference.reward_program.term_names) != active_terms:
        raise AssertionError("qualified reference terms differ from VNext authority")

    training_binding = config.formal_training_binding()
    if manifest.uses_legacy_training_binding_v1:
        raise AssertionError("unified Stair received the legacy training grant")
    if training_binding.task_protocol_hash != manifest.resolved_cell_task_protocol_hash:
        raise AssertionError("unified Stair TrainingBindingV2 identity diverged")
    evaluation_binding = composition.backend_adapter.config.evaluation_binding
    if (
        evaluation_binding.task_protocol_hash
        != manifest.resolved_cell_task_protocol_hash
    ):
        raise AssertionError("unified Stair EvaluationBinding identity diverged")

    task_projection_text = json.dumps(
        config.unified_protocol.task_projection(), sort_keys=True
    ).lower()
    robot_markers = (
        "torso_link",
        "pelvis",
        "ankle_roll_link",
        ".usd",
        "37dof",
    )
    leaked_markers = [
        marker for marker in robot_markers if marker in task_projection_text
    ]
    if leaked_markers:
        raise AssertionError(
            f"robot morphology leaked into Stair task projection: {leaked_markers}"
        )

    phased_cost = stair_phased_evaluation_cost_contract(config)
    if (
        phased_cost["schema_version"] != STAIR_ORDERED_PHASED_EVAL_COST_SCHEMA_VERSION
        or phased_cost["evaluation_suite_hash"] != suite.suite_hash
        or phased_cost["episodes_per_seed_unit"] != 84
    ):
        raise AssertionError("unified Stair phased cost is not suite-bound")

    return {
        "cell_variant_id": config.cell_variant_id,
        "protocol_state": config.protocol_state,
        "base_task_protocol_hash": config.base_task_protocol_hash,
        "resolved_cell_task_protocol_hash": manifest.resolved_cell_task_protocol_hash,
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "training_binding_hash": training_binding.binding_hash,
        "evaluation_binding_hash": evaluation_binding.binding_hash,
        "training_profile_hash": profile.profile_hash,
        "initial_assignment": assignment,
        "independent_c1_axes": True,
        "evaluation_suite_hash": suite.suite_hash,
        "formal_eval_envs_per_seed": 84,
        "formal_geometry_pool_size": formal_geometry_pool_size,
        "formal_evaluation_episode_count": formal_evaluation_episode_count,
        "formal_geometry_isolation": "cohort_disjoint",
        "shared_floor_depth_m": float(profile.terrain["floor_depth_m"]),
        "phase_ood_labels": sorted(phase_labels),
        "cohort_stratum_counts": cohort_counts,
        "legacy_stair_010_guard_hits": legacy_stair_010_guard_hits,
        "sealed_geometry_prompt_hits": sealed_prompt_hits,
        "score": _synthetic_score_receipt(config),
        "acceptance_protocol_hash": acceptance.protocol_hash,
        "named_guards": list(guard_names),
        "vnext_reward_authority_hash": authority.authority_hash,
        "official_reference_manifest_hash": reference.artifact.manifest.manifest_hash,
        "official_reference_term_count": len(active_terms),
        "phased_cost_contract_hash": phased_cost["contract_hash"],
        "draft_formal_rejections": _assert_draft_formal_rejection(),
    }


def _boundary_receipt() -> dict[str, Any]:
    retired_hits: list[str] = []
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if "is_irregular_cell" in path.read_text(encoding="utf-8"):
                retired_hits.append(str(path.relative_to(PROJECT_ROOT)))
    if retired_hits:
        raise AssertionError(f"retired terrain-cell switch remains: {retired_hits}")

    retired_paths_present = [
        relative_path
        for relative_path in _RETIRED_PHASE3A_PATHS
        if (PROJECT_ROOT / relative_path).exists()
    ]
    if retired_paths_present:
        raise AssertionError(f"retired Phase 3A paths remain: {retired_paths_present}")

    retired_marker_hits: dict[str, list[str]] = {}
    for root_name in ("src", "tests", "scripts", "configs"):
        root = PROJECT_ROOT / root_name
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            hits = [marker for marker in _RETIRED_ACTIVE_MARKERS if marker in text]
            if hits:
                retired_marker_hits[str(path.relative_to(PROJECT_ROOT))] = hits
    if retired_marker_hits:
        raise AssertionError(f"retired Phase 3A markers remain: {retired_marker_hits}")

    forbidden = ("c0_id", "c1_id", "c2_rough", "riser", "tread", "stair")
    generic_hits: dict[str, list[str]] = {}
    for relative_path in _GENERIC_ACCEPTANCE_PATHS:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
        hits = [marker for marker in forbidden if marker in text]
        if hits:
            generic_hits[relative_path] = hits
    if generic_hits:
        raise AssertionError(
            f"task semantics leaked into generic acceptance: {generic_hits}"
        )
    return {
        "retired_is_irregular_cell_hits": retired_hits,
        "retired_phase3a_paths_present": retired_paths_present,
        "retired_phase3a_marker_hits": retired_marker_hits,
        "generic_acceptance_task_marker_hits": generic_hits,
    }


def _product_tree_receipt() -> dict[str, Any]:
    hashes = {
        relative_path: _normalized_text_hash(PROJECT_ROOT / relative_path)
        for relative_path in _PRODUCT_PATHS
    }
    return {
        "hash_mode": "normalized_lf_sha256",
        "file_count": len(hashes),
        "file_hashes": hashes,
        "product_tree_hash": canonical_json_hash(hashes),
    }


def _runtime_receipt(
    contract: Mapping[str, Any],
    path: Path = RUNTIME_SMOKE_PATH,
) -> dict[str, Any]:
    receipt = _load_object(path)
    if receipt.get("schema_version") != RUNTIME_SMOKE_SCHEMA_VERSION:
        raise AssertionError("Phase 3R runtime smoke schema mismatch")
    if receipt.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(receipt)
    ):
        raise AssertionError("Phase 3R runtime smoke receipt hash mismatch")
    if receipt.get("status") != "pass":
        raise AssertionError("Phase 3R runtime smoke did not pass")

    source_hashes = receipt.get("product_source_hashes")
    if (
        receipt.get("product_source_hash_mode") != "normalized_lf_sha256"
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
    ):
        raise AssertionError("Phase 3R runtime source closure is invalid")
    for relative_path, expected_hash in source_hashes.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise AssertionError("Phase 3R runtime source entry is invalid")
        if _normalized_text_hash(PROJECT_ROOT / relative_path) != expected_hash:
            raise AssertionError(f"Phase 3R runtime source drifted: {relative_path}")

    source_preflight = receipt.get("source_preflight")
    if not isinstance(source_preflight, Mapping):
        raise AssertionError("Phase 3R execution source preflight is missing")
    source_summary = source_preflight.get("summary")
    source_evidence = source_preflight.get("evidence")
    if not isinstance(source_summary, Mapping) or not isinstance(
        source_evidence, Mapping
    ):
        raise AssertionError("Phase 3R execution source preflight is invalid")
    evidence_payload = {
        key: value
        for key, value in source_evidence.items()
        if key != "source_manifest_hash"
    }
    if (
        source_summary.get("status") != "pass"
        or source_summary.get("task_protocol_hash")
        != contract.get("resolved_cell_task_protocol_hash")
        or source_summary.get("source_manifest_hash")
        != source_evidence.get("source_manifest_hash")
        or source_evidence.get("source_manifest_hash")
        != canonical_json_hash(evidence_payload)
    ):
        raise AssertionError("Phase 3R execution source preflight identity drifted")
    source_files = source_evidence.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise AssertionError("Phase 3R execution source file evidence is missing")
    for entry in source_files:
        if not isinstance(entry, Mapping):
            raise AssertionError("Phase 3R execution source entry is invalid")
        source_path = PROJECT_ROOT / str(entry.get("repository_path"))
        payload = source_path.read_bytes()
        if len(payload) != entry.get("size_bytes") or hashlib.sha256(
            payload
        ).hexdigest() != entry.get("sha256"):
            raise AssertionError(f"Phase 3R execution source drifted: {source_path}")

    cell = receipt.get("cell")
    if not isinstance(cell, Mapping):
        raise AssertionError("Phase 3R runtime cell identity is missing")
    identity_pairs = {
        "cell_variant_id": "cell_variant_id",
        "protocol_state": "protocol_state",
        "manifest_hash": "resolved_cell_manifest_hash",
        "base_task_protocol_hash": "base_task_protocol_hash",
        "resolved_cell_task_protocol_hash": "resolved_cell_task_protocol_hash",
        "training_profile_hash": "training_profile_hash",
        "evaluation_suite_hash": "evaluation_suite_hash",
        "vnext_reward_authority_hash": "vnext_reward_authority_hash",
    }
    for runtime_name, contract_name in identity_pairs.items():
        if cell.get(runtime_name) != contract.get(contract_name):
            raise AssertionError(
                f"Phase 3R runtime cell identity drifted: {runtime_name}"
            )

    training_smoke = receipt.get("training_smoke")
    if not isinstance(training_smoke, Mapping):
        raise AssertionError("Phase 3R training smoke evidence is missing")
    training = training_smoke.get("training")
    activation = training_smoke.get("runtime_activation_summary")
    if (
        training_smoke.get("status") != "pass"
        or training_smoke.get("training_executed") is not True
        or not isinstance(training, Mapping)
        or not isinstance(activation, Mapping)
    ):
        raise AssertionError("Phase 3R real PPO smoke did not execute")
    assignment = training.get("initial_terrain_assignment")
    if not isinstance(assignment, Mapping) or assignment.get("regime_counts") != {
        STAIR_TERRAIN_REGIME_C0: 8,
        STAIR_TERRAIN_REGIME_C1: 8,
    }:
        raise AssertionError("Phase 3R runtime C0/C1 assignment drifted")
    if (
        training.get("budget_steps") != 128
        or activation.get("declared_term_count") != 15
        or int(activation.get("dispatcher_calls", 0)) <= 1
        or activation.get("nonzero_reward_contribution") is not True
        or not math.isfinite(float(activation.get("last_total_rate_mean", math.nan)))
    ):
        raise AssertionError("Phase 3R runtime reward/PPO evidence is invalid")

    curriculum = receipt.get("curriculum_probe")
    if (
        not isinstance(curriculum, Mapping)
        or curriculum.get("status") != "pass"
        or curriculum.get("terrain_type_immutable") is not True
        or curriculum.get("levels_before") != [2, 0, 1]
        or curriculum.get("levels_after") != [3, 0, 0]
        or curriculum.get("terrain_types_before")
        != curriculum.get("terrain_types_after")
    ):
        raise AssertionError("Phase 3R runtime curriculum evidence is invalid")

    geometry = receipt.get("evaluation_geometry_probe")
    representatives = (
        geometry.get("representatives") if isinstance(geometry, Mapping) else None
    )
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("status") != "pass"
        or not isinstance(representatives, list)
        or [item.get("cohort") for item in representatives]
        != [
            "public_dev",
            "public_accept",
            "sealed_validation",
        ]
        or len({item.get("geometry_digest") for item in representatives}) != 3
        or any(
            item.get("mesh_count") != 10
            or not math.isclose(
                float(item.get("floor_depth_m", math.nan)),
                float(contract["shared_floor_depth_m"]),
                abs_tol=1.0e-12,
            )
            for item in representatives
        )
    ):
        raise AssertionError("Phase 3R runtime evaluation geometry evidence is invalid")

    eligibility = receipt.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility != {
        "reference_qualification": False,
        "current_best": False,
        "candidate_acceptance": False,
        "reason": "phase3r_draft_smoke_only",
    }:
        raise AssertionError("Phase 3R draft smoke eligibility drifted")
    cost = receipt.get("cost")
    if (
        not isinstance(cost, Mapping)
        or cost.get("smoke_budget_steps") != 128
        or cost.get("full_training_budget_steps") != 196_608_000
        or float(cost.get("budget_fraction", math.inf))
        > float(cost.get("maximum_authorized_fraction", -math.inf))
        or float(cost.get("maximum_authorized_fraction", math.nan)) != 0.05
    ):
        raise AssertionError("Phase 3R runtime smoke budget evidence is invalid")

    return {
        "receipt_hash": receipt["receipt_hash"],
        "source_closure_hash": canonical_json_hash(dict(source_hashes)),
        "source_file_count": len(source_hashes),
        "execution_source_manifest_hash": source_summary["source_manifest_hash"],
        "device": receipt["environment"]["device"],
        "cuda_device_name": receipt["environment"]["cuda_device_name"],
        "smoke_budget_steps": cost["smoke_budget_steps"],
        "budget_fraction": cost["budget_fraction"],
        "runtime_assignment": assignment["regime_counts"],
        "dispatcher_calls": activation["dispatcher_calls"],
        "reward_term_calls": activation["reward_term_calls"],
        "curriculum_status": curriculum["status"],
        "evaluation_geometry_status": geometry["status"],
        "eligibility": dict(eligibility),
    }


def emit_current() -> dict[str, Any]:
    contract = _contract_receipt()
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract": contract,
        "boundaries": _boundary_receipt(),
        "product_tree": _product_tree_receipt(),
        "runtime_smoke": _runtime_receipt(contract),
    }
    payload["receipt_hash"] = canonical_json_hash(payload)
    return payload


def verify(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    expected = _load_object(snapshot_path)
    if expected.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise AssertionError("Phase 3R snapshot schema mismatch")
    if expected.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(expected)
    ):
        raise AssertionError("Phase 3R snapshot receipt hash mismatch")
    current = emit_current()
    for section in ("contract", "boundaries", "product_tree", "runtime_smoke"):
        if current[section] != expected.get(section):
            raise AssertionError(f"Phase 3R current {section} differs from snapshot")
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "snapshot_path": str(snapshot_path.resolve()),
        "receipt_hash": expected["receipt_hash"],
        "product_tree_hash": expected["product_tree"]["product_tree_hash"],
        "runtime_smoke_receipt_hash": expected["runtime_smoke"]["receipt_hash"],
        "protocol_state": expected["contract"]["protocol_state"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Phase 3R contract snapshot"
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--emit-current", action="store_true")
    args = parser.parse_args(argv)
    payload = emit_current() if args.emit_current else verify(args.snapshot)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

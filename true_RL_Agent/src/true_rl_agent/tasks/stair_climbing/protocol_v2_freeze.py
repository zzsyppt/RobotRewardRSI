"""Derive the production Stair protocol from Phase D and operator-owned revisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.tasks.stair_climbing.pilot_artifact_store import (
    validate_stair_phase_d_pilot_artifact_manifest,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
    PREVIOUS_STAIR_PROTOCOL_V2_HASH,
    STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH,
    StairProtocolV2Draft,
)
from true_rl_agent.tasks.stair_climbing.reference_pilot import (
    build_stair_pilot_config,
    build_stair_pilot_evaluation_binding,
    build_stair_pilot_training_binding,
    stair_pilot_training_seed_domain_hash,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_PROFILE,
    STAIR_TRAIN_PROFILE,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_TRAIN_SEED_BASE,
    StairPrivateTrainingSeedResolver,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION,
)
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)


def build_stair_protocol_v2_payload(
    *,
    draft_path: Path,
    phase_d_report: Mapping[str, Any],
    pilot_artifact_manifest: Mapping[str, Any],
    agent_configuration: RslRlAgentConfiguration,
) -> dict[str, Any]:
    """Build one deterministic final payload from an already replayed pass report."""

    if not isinstance(draft_path, Path):
        raise TypeError("Stair V2 freeze draft_path must be a Path")
    if not isinstance(phase_d_report, Mapping):
        raise TypeError("Stair V2 freeze report must be an object")
    if not isinstance(pilot_artifact_manifest, Mapping):
        raise TypeError("Stair V2 freeze artifact manifest must be an object")
    if not isinstance(agent_configuration, RslRlAgentConfiguration):
        raise TypeError("Stair V2 freeze agent configuration is invalid")
    pilot_artifact_manifest = validate_stair_phase_d_pilot_artifact_manifest(
        manifest=pilot_artifact_manifest
    )

    draft = StairProtocolV2Draft.load(draft_path)
    if draft.protocol_hash != EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH:
        raise ValueError("Stair V2 freeze requires the trusted pilot draft")
    if phase_d_report.get("status") != "pass":
        raise ValueError("Stair V2 freeze requires a passing Phase D report")
    if phase_d_report.get("draft_protocol_hash") != draft.protocol_hash:
        raise ValueError("Stair V2 freeze report draft hash mismatch")
    if (
        pilot_artifact_manifest.get("draft_protocol_hash") != draft.protocol_hash
        or pilot_artifact_manifest.get("phase_d_report_hash")
        != phase_d_report.get("report_hash")
    ):
        raise ValueError("Stair V2 freeze artifact manifest lineage mismatch")
    pilot_evidence = phase_d_report.get("pilot_evidence")
    if not isinstance(pilot_evidence, Mapping) or pilot_evidence.get(
        "status"
    ) != "pass":
        raise ValueError("Stair V2 freeze requires passing capability evidence")

    pilot_config = build_stair_pilot_config(draft_path)
    if agent_configuration != pilot_config.training_agent_configuration():
        raise ValueError("Stair V2 freeze agent configuration differs from pilot")
    pilot_training = build_stair_pilot_training_binding(pilot_config)
    pilot_evaluation = build_stair_pilot_evaluation_binding(pilot_config)
    if (
        pilot_evidence.get("training_seed_index") != 0
        or pilot_evidence.get("training_binding")
        != pilot_training.to_public_dict()
        or pilot_evidence.get("evaluation_binding")
        != pilot_evaluation.to_public_dict()
    ):
        raise ValueError("Stair V2 freeze pilot evidence binding mismatch")

    seed_resolver = StairPrivateTrainingSeedResolver(
        base_seed=STAIR_TRAIN_SEED_BASE,
        task_protocol_hash=draft.protocol_hash,
    )
    seed_resolver.assert_namespaces_disjoint(
        pilot_seed_count=1
    )
    formal_seed = seed_resolver.derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
    formal_training = TrainingBinding(
        initialization_mode=PolicyInitialization.random().kind,
        initialization_checkpoint_hash=None,
        policy_architecture_hash=str(agent_configuration.policy_architecture_hash),
        agent_configuration_hash=str(agent_configuration.agent_configuration_hash),
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(
            formal_seed
        ),
        optimizer_training_profile_hash=str(
            agent_configuration.optimizer_training_profile_hash
        ),
        budget_steps=pilot_config.budget_steps,
        private_training_seed_resolver_hash=fixed_training_seed_resolver_hash(
            formal_seed
        ),
        capability_profile_hash=draft.capability_profile.profile_hash,
    )
    _assert_training_binding_only_seed_domain_differs(
        pilot=pilot_training,
        formal=formal_training,
    )

    evaluation_resolver = pilot_config.seed_resolver()
    pilot_eval_cohort = evaluation_resolver.namespace_cohort_hash(
        STAIR_PILOT_EVAL_NAMESPACE
    )
    formal_eval_cohort = evaluation_resolver.namespace_cohort_hash(
        STAIR_FORMAL_EVAL_NAMESPACE
    )
    if pilot_eval_cohort == formal_eval_cohort:
        raise ValueError("Stair V2 freeze formal and pilot eval cohorts overlap")

    draft_payload = draft.to_public_dict()
    base_training = pilot_config.protocol_v1.to_dict()["training"]
    formal_draft = dict(draft_payload["formal_protocol"])
    payload: dict[str, Any] = {
        "schema_version": "true_rl_agent.task.stair_climbing.protocol_v2.v3",
        "protocol_id": "unitree_stair_climbing_v2",
        "task_id": draft_payload["task_id"],
        "status": "production_revised_for_formal_target_training_distribution",
        "base_protocol": draft_payload["base_protocol"],
        "genesis_anchor_policy": draft_payload["genesis_anchor_policy"],
        "reference_program": draft_payload["reference_program"],
        "candidate_capability_profile": draft_payload[
            "candidate_capability_profile"
        ],
        "production_training": {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.production_training.v1"
            ),
            "budget_steps": pilot_config.budget_steps,
            "num_envs": pilot_config.train_num_envs,
            "steps_per_env": pilot_config.train_steps_per_iteration,
            "max_iterations": pilot_config.train_max_iterations,
            "runner_timeout_s": int(base_training["runner_timeout_s"]),
            "train_profile": STAIR_TRAIN_PROFILE,
            "eval_profile": STAIR_EVAL_PROFILE,
            "policy_initialization": "random",
            "warm_start_allowed": False,
            "checkpoint_selector": draft_payload["training_execution"][
                "checkpoint_selector"
            ],
            "policy_architecture_hash": agent_configuration.policy_architecture_hash,
            "optimizer_training_profile_hash": (
                agent_configuration.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": agent_configuration.agent_configuration_hash,
            "initialization_seed_schedule_hash": (
                formal_training.initialization_seed_schedule_hash
            ),
            "private_training_seed_resolver_hash": (
                formal_training.private_training_seed_resolver_hash
            ),
            "formal_training_binding_hash": formal_training.binding_hash,
            "capability_profile_hash": draft.capability_profile.profile_hash,
        },
        "reference_capability_gate": draft_payload["reference_capability_gate"],
        "formal_protocol": {
            **formal_draft,
            "evaluation_protocol_hash": (
                pilot_config.evaluation_protocol().protocol_hash
            ),
            "acceptance_protocol_hash": (
                pilot_config.protocol_v1.acceptance_protocol.protocol_hash
            ),
            "formal_evaluation_seed_cohort_hash": formal_eval_cohort,
            "pilot_evaluation_seed_cohort_hash": pilot_eval_cohort,
            "task_protocol_binding": "final_protocol_hash_at_materialization",
        },
        "runner_owned_training_distribution": _production_training_distribution(
            draft_payload["runner_owned_training_distribution"],
            base_protocol=pilot_config.protocol_v1.to_dict(),
        ),
        "training_execution": draft_payload["training_execution"],
        "complete_reward_runtime": draft_payload["complete_reward_runtime"],
        "inheritance": draft_payload["inheritance"],
        "pilot_lineage": {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.phase_d_freeze_lineage.v4"
            ),
            "draft_protocol_hash": draft.protocol_hash,
            "phase_d_report_hash": phase_d_report["report_hash"],
            "pilot_evidence_hash": pilot_evidence["seed_evidence_hash"],
            "fixed_budget_policy_hash": canonical_json_hash(
                phase_d_report["budget_policy"]
            ),
            "pilot_training_binding_hash": pilot_training.binding_hash,
            "pilot_training_seed_count": 1,
            "pilot_training_seed_domain_hash": (
                stair_pilot_training_seed_domain_hash(pilot_config)
            ),
            "pilot_evaluation_binding_hash": pilot_evaluation.binding_hash,
            "pilot_artifact_uri": pilot_artifact_manifest["artifact_uri"],
            "pilot_artifact_content_hash": pilot_artifact_manifest["content_hash"],
            "pilot_artifact_manifest_hash": pilot_artifact_manifest[
                "manifest_hash"
            ],
            "pilot_artifact_manifest_repository_path": (
                STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH
            ),
            "boundaries": dict(phase_d_report["boundaries"]),
        },
    }
    payload["protocol_hash"] = canonical_json_hash(payload)
    return payload


def _production_training_distribution(
    historical: Mapping[str, Any],
    *,
    base_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep Phase D evidence historical while defining the current runtime authority."""

    terrain = historical["terrain"]
    base_terrain = base_protocol["terrain"]
    historical_range = base_terrain["training_distribution"]["step_height_range_m"]
    evaluation_heights = sorted(
        float(item["height_m"]) for item in base_terrain["evaluation_bins"]
    )
    if len(evaluation_heights) < 2:
        raise ValueError("Stair formal evaluation needs an OOD frontier bin")
    production_range = [float(historical_range[0]), evaluation_heights[-2]]
    return {
        "schema_version": STAIR_TRAINING_DISTRIBUTION_SCHEMA_VERSION,
        "authority": "task_runner",
        "candidate_writable": False,
        "prompt_visible": False,
        "formal_reference_and_candidates_identical": True,
        "historical_pilot_distribution_identical": False,
        "design_basis": "formal_eval_target_alignment_with_single_ood_frontier_bin",
        "selection_provenance": {
            "kind": "operator_selected_formal_target_alignment",
            "previous_training_protocol_hash": PREVIOUS_STAIR_PROTOCOL_V2_HASH,
            "historical_pilot_distribution_used_as_evidence": False,
            "formal_evaluation_frontier_retained_out_of_distribution": True,
        },
        "upstream_reference": historical["upstream_reference"],
        "terrain": {
            "generation_mode": terrain["generation_mode"],
            "step_height_range_m": production_range,
            "runtime_level_curriculum_enabled": terrain[
                "runtime_level_curriculum_enabled"
            ],
            "initial_max_terrain_level": terrain["initial_max_terrain_level"],
            "level_update_term": terrain["level_update_term"],
        },
        "command": historical["command"],
    }


def _assert_training_binding_only_seed_domain_differs(
    *,
    pilot: TrainingBinding,
    formal: TrainingBinding,
) -> None:
    pilot_public = pilot.to_public_dict()
    formal_public = formal.to_public_dict()
    stable_fields = {
        "schema_version",
        "initialization_mode",
        "initialization_checkpoint_hash",
        "policy_architecture_hash",
        "agent_configuration_hash",
        "optimizer_training_profile_hash",
        "budget_steps",
        "capability_profile_hash",
    }
    if any(pilot_public[key] != formal_public[key] for key in stable_fields):
        raise ValueError("Stair V2 pilot/formal TrainingBinding differs beyond seed")
    if pilot.initialization_seed_schedule_hash == formal.initialization_seed_schedule_hash:
        raise ValueError("Stair V2 pilot/formal initialization seed schedules overlap")
    if pilot.private_training_seed_resolver_hash == (
        formal.private_training_seed_resolver_hash
    ):
        raise ValueError("Stair V2 pilot/formal training seed resolvers overlap")
    if pilot.binding_hash == formal.binding_hash:
        raise ValueError("Stair V2 pilot/formal TrainingBinding overlap")


__all__ = ["build_stair_protocol_v2_payload"]

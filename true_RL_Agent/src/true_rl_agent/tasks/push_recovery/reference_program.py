"""Task-owned standing-recovery qualification and the frozen G1-29 reader."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import DEFAULT_MOTION_LIMITS, attach_push_budget_gate
from true_rl_agent.tasks.push_recovery.metrics import recovery_validation_contract
from true_rl_agent.tasks.reference_qualification import (
    REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
    ReferenceQualificationResult,
)
from true_rl_agent.training_programs import (
    VerifiedReferenceProgramArtifact,
    resolve_reference_program_artifact,
)

REFERENCE_ID = "push_locomotion_overlay_reference_v1"
REFERENCE_SOURCE = (
    "src/true_rl_agent/tasks/push_recovery/reference_programs/"
    "locomotion_overlay_v1/training_code/components.py"
)
RECOVERY_REFERENCE_ID = "push_standing_recovery_reference_v2"
RECOVERY_REFERENCE_SOURCE = (
    "src/true_rl_agent/tasks/push_recovery/reference_programs/"
    "standing_recovery_v2/training_code/components.py"
)
STATIONARY_REFERENCE_ID = "push_standing_recovery_stationary_reference_v3"
STATIONARY_REFERENCE_SOURCE = (
    "src/true_rl_agent/tasks/push_recovery/reference_programs/"
    "standing_recovery_stationary_v3/training_code/components.py"
)
STATIONARY_REFERENCE_UPSTREAM_SOURCE = (
    "PLAN/2new_world/P6/dev_reference_stationary_v3/training_code/components.py"
)
PUSH_REFERENCE_IDS = (REFERENCE_ID, RECOVERY_REFERENCE_ID, STATIONARY_REFERENCE_ID)


def reference_program_id(config: PushRecoveryTaskConfig) -> str:
    default = RECOVERY_REFERENCE_ID if config.push_protocol is not None else REFERENCE_ID
    selected = config.reference_program_id
    if selected is None:
        return default
    allowed = (RECOVERY_REFERENCE_ID, STATIONARY_REFERENCE_ID) if config.push_protocol is not None else (REFERENCE_ID,)
    if selected not in allowed:
        raise ValueError("reference_program_id is not supported by this Push protocol")
    return selected


def reference_qualification_contract(config: PushRecoveryTaskConfig) -> dict[str, Any]:
    if config.push_protocol is not None:
        contract = {
            "schema_version": "true_rl_agent.task.push_recovery.reference_capability.v2",
            "scope": "standing_recovery_task_and_reference",
            "score_source": config.acceptance_score_source,
            "minimum_score": config.absolute_accept_floor,
            "public_dev_reference_band": [0.2, 0.9],
            "band_purpose": "basic_learnability_and_nonzero_improvement_headroom",
            "required_gates": ["accept_evidence_gate", "budget_accounting",
                               "recovery_evidence_gate", "sealed_validation_gate"],
            "train_budget_steps": config.budget_steps,
            "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
            "recovery_protocol_hash": (config.push_evaluation_protocol.protocol_hash
                if config.push_evaluation_protocol is not None else config.push_protocol.protocol_hash),
            "independent_validation": recovery_validation_contract(),
        }
        if config.reference_admission_policy is not None:
            for name in ("minimum_score", "public_dev_reference_band", "band_purpose"):
                contract.pop(name)
            contract.update(
                schema_version="true_rl_agent.task.push_recovery.reference_capability.v3",
                admission_policy=config.reference_admission_policy,
                automatic_capability_evidence="observed_public_dev_successes_and_failures",
                public_accept_capability_evidence="observed_successful_recovery",
                saturation_assessment="inspect_public_dev_strata_and_outcomes_without_a_fixed_rate_band",
            )
        return contract
    return {
        "schema_version": "true_rl_agent.task.push_recovery.reference_capability.v1",
        "scope": "g1_29_existing_task_basic_capability",
        "score_source": config.acceptance_score_source,
        "minimum_score": config.absolute_accept_floor,
        "required_gates": ["accept_evidence_gate", "budget_accounting",
                           "motion_quality_gate", "behavior_quality_gate",
                           "sealed_validation_gate"],
        "motion_limits": dict(DEFAULT_MOTION_LIMITS),
        "train_budget_steps": config.budget_steps,
        "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
    }


def resolve_push_reference_program(
    *, project_root: Path, capability_profile_hash: str,
    config: PushRecoveryTaskConfig | None = None,
) -> VerifiedReferenceProgramArtifact:
    reference_id = reference_program_id(config) if config is not None else REFERENCE_ID
    source = {
        REFERENCE_ID: REFERENCE_SOURCE,
        RECOVERY_REFERENCE_ID: RECOVERY_REFERENCE_SOURCE,
        STATIONARY_REFERENCE_ID: STATIONARY_REFERENCE_SOURCE,
    }[reference_id]
    stationary = reference_id == STATIONARY_REFERENCE_ID
    return resolve_reference_program_artifact(
        project_root / source,
        reference_id=reference_id,
        task_id="unitree_push_recovery",
        authorship_origin="llm_generated" if stationary else "human_authored",
        upstream_project="true_RL_Agent",
        upstream_revision="standing_recovery_v2_stationary_only_v3" if stationary else reference_id,
        upstream_source_path=STATIONARY_REFERENCE_UPSTREAM_SOURCE if stationary else source,
        capability_profile_hash=capability_profile_hash,
    )


def reference_dev_evidence_passes(config: PushRecoveryTaskConfig, dev: Mapping[str, Any]) -> bool:
    """Apply the frozen admission policy before holdout and at qualification.

    The current policy only rejects degenerate all-failure/all-success evidence.
    It does not turn a new percentage into a proxy for qualitative headroom.
    """
    if config.reference_admission_policy is not None:
        counts = [row.get("recovery_counts", {}) for row in dev.get("seed_results", [])]
        if not counts or any("successes" not in row or "eligible_episode_count" not in row for row in counts):
            return False
        successes = sum(row["successes"] for row in counts)
        episodes = sum(row["eligible_episode_count"] for row in counts)
        return successes > 0 and episodes - successes > 0
    score = dev.get("score_stats", {}).get("mean")
    low, high = reference_qualification_contract(config)["public_dev_reference_band"]
    return score is not None and low <= score <= high


def qualify_push_reference(
    *, config: PushRecoveryTaskConfig, public_result: PublicEvalResult,
    train_summary: Mapping[str, Any], task_protocol_hash: str,
) -> ReferenceQualificationResult:
    contract = reference_qualification_contract(config)
    training = train_summary.get("training", train_summary)
    projected = attach_push_budget_gate(public_result, budget_summary=training)
    metrics = copy.deepcopy(dict(projected.metrics))
    statuses = {
        gate: metrics.get(gate, {}).get("status", "missing")
        for gate in contract["required_gates"]
    }
    score = metrics.get("public_accept", {}).get("score_stats", {}).get("mean")
    passed = (
        projected.status == "pass"
        and all(status == "pass" for status in statuses.values())
        and score is not None
        and (score > 0.0 if config.reference_admission_policy is not None
             else score >= contract["minimum_score"])
    )
    dev_score = None
    if config.push_protocol is not None:
        dev_score = metrics.get("public_dev", {}).get("score_stats", {}).get("mean")
        passed = passed and reference_dev_evidence_passes(config, metrics.get("public_dev", {}))
    evidence = {
        "schema_version": ("true_rl_agent.task.push_recovery.reference_evidence.v3"
                           if config.reference_admission_policy is not None
                           else "true_rl_agent.task.push_recovery.reference_evidence.v2"
                           if config.push_protocol is not None
                           else "true_rl_agent.task.push_recovery.reference_evidence.v1"),
        "protocol_hash": task_protocol_hash,
        "qualification_contract_hash": canonical_json_hash(contract),
        "gate_statuses": statuses,
        "public_accept_score": score,
        "scope": contract["scope"],
    }
    if config.push_protocol is not None:
        evidence["public_dev_score"] = dev_score
        if config.reference_admission_policy is None:
            evidence["public_dev_reference_band"] = contract["public_dev_reference_band"]
        else:
            evidence["reference_admission_policy"] = config.reference_admission_policy
    evidence["evidence_hash"] = canonical_json_hash(evidence)
    gate = {
        "schema_version": REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "protocol_hash": evidence["protocol_hash"],
        "evidence_hash": evidence["evidence_hash"],
    }
    metrics["reference_capability_gate"] = gate
    required = metrics["acceptance"]["required_pass_gates"]
    if "reference_capability_gate" not in required:
        required.append("reference_capability_gate")
    return ReferenceQualificationResult(
        public_result=PublicEvalResult(status=projected.status,
            public_score=projected.public_score, metrics=metrics),
        gate=gate, private_evidence=evidence,
    )

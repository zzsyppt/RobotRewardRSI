"""Reference contract for the existing G1-29 Push task.

This basic capability gate supports generation orchestration. It does not
qualify the future Go2 task or claim that every episode received a push.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import DEFAULT_MOTION_LIMITS, attach_push_budget_gate
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


def reference_qualification_contract(config: PushRecoveryTaskConfig) -> dict[str, Any]:
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
) -> VerifiedReferenceProgramArtifact:
    return resolve_reference_program_artifact(
        project_root / REFERENCE_SOURCE,
        reference_id=REFERENCE_ID,
        task_id="unitree_push_recovery",
        authorship_origin="human_authored",
        upstream_project="true_RL_Agent",
        upstream_revision=REFERENCE_ID,
        upstream_source_path=REFERENCE_SOURCE,
        capability_profile_hash=capability_profile_hash,
    )


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
        and score is not None and score >= contract["minimum_score"]
    )
    evidence = {
        "schema_version": "true_rl_agent.task.push_recovery.reference_evidence.v1",
        "protocol_hash": task_protocol_hash,
        "qualification_contract_hash": canonical_json_hash(contract),
        "gate_statuses": statuses,
        "public_accept_score": score,
        "scope": contract["scope"],
    }
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

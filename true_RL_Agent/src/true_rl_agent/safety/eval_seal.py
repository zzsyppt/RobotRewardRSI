"""Trusted eval seal summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import (
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.runner.checkpoint import validate_checkpoint_safety_report
from true_rl_agent.runner.specs import EvalJobSpec, EvaluationJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec

TRUSTED_EVAL_SEAL_SCHEMA_VERSION = "true_rl_agent.sealed_eval_seal.v3"


@dataclass(frozen=True)
class TrustedEvalSeal:
    subject_kind: str
    subject_id: str
    subject_manifest_hash: str | None
    program_source_checksum: str | None
    training_binding_hash: str | None
    capability_profile_hash: str
    eval_config_hash: str
    evaluation_binding_hash: str
    checkpoint_hash: str
    task_invariant_hash: str
    safe_load_status: str
    checkpoint_safety_status: str
    checkpoint_safety_report_hash: str
    candidate_isolation_status: str
    seal_hash: str
    candidate_code_loaded: bool = False

    def __post_init__(self) -> None:
        if self.subject_manifest_hash is not None:
            require_sha256_digest(
                self.subject_manifest_hash,
                "subject_manifest_hash",
            )
        if self.subject_kind in {"baseline", "reference"} and self.subject_manifest_hash is None:
            raise ValueError("trusted eval seal subject_manifest_hash is required")
        if self.subject_kind in {"candidate", "reference"}:
            require_sha256_checksum(
                self.program_source_checksum,
                "program_source_checksum",
            )
            require_sha256_digest(
                self.training_binding_hash,
                "training_binding_hash",
            )
        elif self.training_binding_hash is not None or self.program_source_checksum is not None:
            raise ValueError("baseline trusted eval seal cannot carry program lineage")
        require_sha256_digest(
            self.capability_profile_hash,
            "capability_profile_hash",
        )
        for field_name in (
            "eval_config_hash",
            "evaluation_binding_hash",
            "task_invariant_hash",
            "checkpoint_safety_report_hash",
            "seal_hash",
        ):
            require_sha256_digest(getattr(self, field_name), field_name)
        require_sha256_checksum(self.checkpoint_hash, "checkpoint_hash")

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": TRUSTED_EVAL_SEAL_SCHEMA_VERSION,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "eval_config_hash": self.eval_config_hash,
            "evaluation_binding_hash": self.evaluation_binding_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "task_invariant_hash": self.task_invariant_hash,
            "safe_load_status": self.safe_load_status,
            "checkpoint_safety_status": self.checkpoint_safety_status,
            "checkpoint_safety_report_hash": self.checkpoint_safety_report_hash,
            "candidate_isolation_status": self.candidate_isolation_status,
            "seal_hash": self.seal_hash,
            "candidate_code_loaded": self.candidate_code_loaded,
        }
        if self.subject_manifest_hash is not None:
            result["subject_manifest_hash"] = self.subject_manifest_hash
        if self.training_binding_hash is not None:
            result["training_binding_hash"] = self.training_binding_hash
        if self.program_source_checksum is not None:
            result["program_source_checksum"] = self.program_source_checksum
        result["capability_profile_hash"] = self.capability_profile_hash
        return result


def build_trusted_eval_seal(
    *,
    eval_spec: EvaluationJobSpec,
    task_invariants: TaskInvariantSpec,
    checkpoint_safety_report: Any,
    candidate_isolation_status: str,
) -> TrustedEvalSeal:
    checkpoint = eval_spec.execution_checkpoint()
    if not checkpoint.sealed:
        raise ValueError("trusted eval seal requires a sealed checkpoint")
    report = validate_checkpoint_safety_report(
        checkpoint_safety_report,
        expected_checkpoint_hash=checkpoint.checksum,
    )
    if candidate_isolation_status != "pass":
        raise ValueError("candidate isolation failed")
    subject = eval_spec.evaluation_subject()
    subject_manifest_hash = eval_spec.evaluation_manifest_hash()
    training_binding_hash = (
        eval_spec.training_binding_hash
        if isinstance(eval_spec, EvalJobSpec)
        else None
    )
    program_source_checksum = (
        eval_spec.program_source_checksum
        if isinstance(eval_spec, EvalJobSpec)
        else None
    )
    capability_profile_hash = eval_spec.capability_profile_hash
    eval_protocol = eval_spec.evaluation_protocol().to_public_dict()
    evaluation_binding = eval_spec.evaluation_binding.to_public_dict()
    eval_config = {
        "job_id": eval_spec.job_id,
        "task_id": eval_spec.task_id,
        "subject": subject.to_public_dict(),
        "subject_manifest_hash": subject_manifest_hash,
        "program_source_checksum": program_source_checksum,
        "training_binding_hash": training_binding_hash,
        "capability_profile_hash": capability_profile_hash,
        "checkpoint": checkpoint.to_public_dict(),
        "evaluation_protocol": eval_protocol,
        "evaluation_binding": evaluation_binding,
    }
    eval_config_hash = _stable_hash(eval_config)
    task_invariant_hash = _stable_hash(dict(task_invariants.values))
    checkpoint_safety_report_hash = _stable_hash(report)
    seal_payload = {
        "schema_version": TRUSTED_EVAL_SEAL_SCHEMA_VERSION,
        "subject": subject.to_public_dict(),
        "subject_manifest_hash": subject_manifest_hash,
        "program_source_checksum": program_source_checksum,
        "training_binding_hash": training_binding_hash,
        "capability_profile_hash": capability_profile_hash,
        "eval_config_hash": eval_config_hash,
        "evaluation_binding_hash": eval_spec.evaluation_binding.binding_hash,
        "checkpoint_hash": checkpoint.checksum,
        "task_invariant_hash": task_invariant_hash,
        "checkpoint_safety_status": report["status"],
        "checkpoint_safety_report_hash": checkpoint_safety_report_hash,
        "candidate_isolation_status": candidate_isolation_status,
        "eval_protocol_hash": eval_protocol["eval_protocol_hash"],
        "candidate_code_loaded": False,
    }
    return TrustedEvalSeal(
        subject_kind=subject.kind,
        subject_id=subject.subject_id,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        eval_config_hash=eval_config_hash,
        evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
        checkpoint_hash=checkpoint.checksum,
        task_invariant_hash=task_invariant_hash,
        safe_load_status=report["status"],
        checkpoint_safety_status=report["status"],
        checkpoint_safety_report_hash=checkpoint_safety_report_hash,
        candidate_isolation_status=candidate_isolation_status,
        seal_hash=_stable_hash(seal_payload),
        candidate_code_loaded=False,
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

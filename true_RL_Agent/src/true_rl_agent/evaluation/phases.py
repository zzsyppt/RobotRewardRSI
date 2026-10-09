"""Task-neutral identities and results for fixed phased evaluation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import ExperimentSubject


EVALUATION_PHASE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.evaluation_phase_result.v1"
)
EVALUATION_PHASE_RESULT_GPU_BINDING_SCHEMA_VERSION = (
    "true_rl_agent.evaluation_phase_result.v2"
)
EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.evaluation_phase_summary.v1"
)
EVALUATION_ASSEMBLY_SCHEMA_VERSION = (
    "true_rl_agent.evaluation_assembly.v1"
)
EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.trial_eval_phase_receipt.v1"
)
EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION = (
    "true_rl_agent.trial_receipt_ref.v1"
)
MAX_EVALUATION_PHASE_RECEIPT_BYTES = 8 * 1024 * 1024
_CUDA_EXECUTION_DEVICE_PATTERN = re.compile(r"cuda:(?:0|[1-9][0-9]*)\Z")


class EvaluationPhase(str, Enum):
    """The only split phases supported by the fixed evaluation protocol."""

    PUBLIC_DEV = "public_dev"
    PUBLIC_ACCEPT = "public_accept"
    SEALED_VALIDATION = "sealed_validation"

    @classmethod
    def parse(cls, value: Any) -> "EvaluationPhase":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("evaluation phase must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unsupported evaluation phase: {value!r}") from exc


@dataclass(frozen=True)
class EvaluationPhaseResult:
    """A validated private phase result that has no acceptance authority."""

    job_id: str
    task_id: str
    phase: EvaluationPhase
    evaluation_plan_hash: str
    subject: ExperimentSubject
    subject_manifest_hash: str | None
    program_source_checksum: str
    training_binding_hash: str
    capability_profile_hash: str
    checkpoint_hash: str
    eval_protocol_hash: str
    evaluation_binding_hash: str
    execution_device: str
    resource_binding_hash: str | None
    task_evidence: Mapping[str, Any]
    execution_cost: Mapping[str, Any]
    environment_source_preflight: Mapping[str, Any]
    backend_execution_environment_verification: Mapping[str, Any]
    checkpoint_safety_report: Mapping[str, Any]
    gpu_process_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in ("job_id", "task_id", "execution_device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"evaluation phase result {field_name} is required")
        if not isinstance(self.phase, EvaluationPhase):
            raise TypeError("evaluation phase result phase is invalid")
        if not isinstance(self.subject, ExperimentSubject):
            raise TypeError("evaluation phase result subject is invalid")
        if self.subject_manifest_hash is not None:
            require_sha256_digest(
                self.subject_manifest_hash,
                "evaluation phase result subject_manifest_hash",
            )
        require_sha256_checksum(
            self.program_source_checksum,
            "evaluation phase result program_source_checksum",
        )
        require_sha256_checksum(
            self.checkpoint_hash,
            "evaluation phase result checkpoint_hash",
        )
        for field_name in (
            "evaluation_plan_hash",
            "training_binding_hash",
            "capability_profile_hash",
            "eval_protocol_hash",
            "evaluation_binding_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"evaluation phase result {field_name}",
            )
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "evaluation phase result resource_binding_hash",
            )
        for field_name in (
            "task_evidence",
            "execution_cost",
            "environment_source_preflight",
            "backend_execution_environment_verification",
            "checkpoint_safety_report",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"evaluation phase result {field_name} must be an object")
            object.__setattr__(self, field_name, copy.deepcopy(dict(value)))
        if self.gpu_process_binding is not None:
            binding = _validated_gpu_process_binding_mapping(
                self.gpu_process_binding
            )
            if binding["process_execution_device"] != self.execution_device:
                raise ValueError(
                    "evaluation phase GPU process binding execution device mismatch"
                )
            if binding["resource_binding_hash"] != self.resource_binding_hash:
                raise ValueError(
                    "evaluation phase GPU process binding resource hash mismatch"
                )
            object.__setattr__(self, "gpu_process_binding", binding)

    @property
    def task_evidence_hash(self) -> str:
        return canonical_json_hash(self.task_evidence)

    def to_private_dict(self) -> dict[str, Any]:
        """Serialize the complete result for a private create-once receipt."""

        result = {
            "schema_version": (
                EVALUATION_PHASE_RESULT_GPU_BINDING_SCHEMA_VERSION
                if self.gpu_process_binding is not None
                else EVALUATION_PHASE_RESULT_SCHEMA_VERSION
            ),
            "job_id": self.job_id,
            "task_id": self.task_id,
            "phase": self.phase.value,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "subject": self.subject.to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "eval_protocol_hash": self.eval_protocol_hash,
            "evaluation_binding_hash": self.evaluation_binding_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "task_evidence": copy.deepcopy(dict(self.task_evidence)),
            "task_evidence_hash": self.task_evidence_hash,
            "execution_cost": copy.deepcopy(dict(self.execution_cost)),
            "environment_source_preflight": copy.deepcopy(
                dict(self.environment_source_preflight)
            ),
            "backend_execution_environment_verification": copy.deepcopy(
                dict(self.backend_execution_environment_verification)
            ),
            "checkpoint_safety_report": copy.deepcopy(
                dict(self.checkpoint_safety_report)
            ),
            "acceptance_authority": False,
        }
        if self.gpu_process_binding is not None:
            result["gpu_process_binding"] = copy.deepcopy(
                dict(self.gpu_process_binding)
            )
        return result

    @classmethod
    def from_private_mapping(cls, value: Any) -> "EvaluationPhaseResult":
        if not isinstance(value, Mapping):
            raise ValueError("evaluation phase result must be an object")
        expected_fields = {
            "schema_version",
            "job_id",
            "task_id",
            "phase",
            "evaluation_plan_hash",
            "subject",
            "subject_manifest_hash",
            "program_source_checksum",
            "training_binding_hash",
            "capability_profile_hash",
            "checkpoint_hash",
            "eval_protocol_hash",
            "evaluation_binding_hash",
            "execution_device",
            "resource_binding_hash",
            "task_evidence",
            "task_evidence_hash",
            "execution_cost",
            "environment_source_preflight",
            "backend_execution_environment_verification",
            "checkpoint_safety_report",
            "acceptance_authority",
        }
        actual_fields = set(value)
        schema_version = value.get("schema_version")
        expected_schema_fields = (
            expected_fields
            if schema_version == EVALUATION_PHASE_RESULT_SCHEMA_VERSION
            else expected_fields | {"gpu_process_binding"}
            if schema_version
            == EVALUATION_PHASE_RESULT_GPU_BINDING_SCHEMA_VERSION
            else None
        )
        if expected_schema_fields is None or actual_fields != expected_schema_fields:
            raise ValueError("evaluation phase result fields mismatch")
        if value.get("acceptance_authority") is not False:
            raise ValueError("evaluation phase result contract mismatch")
        task_evidence = value.get("task_evidence")
        if canonical_json_hash(task_evidence) != value.get("task_evidence_hash"):
            raise ValueError("evaluation phase result task evidence hash mismatch")
        return cls(
            job_id=_required_text(value.get("job_id"), "job_id"),
            task_id=_required_text(value.get("task_id"), "task_id"),
            phase=EvaluationPhase.parse(value.get("phase")),
            evaluation_plan_hash=require_sha256_digest(
                value.get("evaluation_plan_hash"), "evaluation_plan_hash"
            ),
            subject=ExperimentSubject.from_mapping(value.get("subject")),
            subject_manifest_hash=value.get("subject_manifest_hash"),
            program_source_checksum=require_sha256_checksum(
                value.get("program_source_checksum"), "program_source_checksum"
            ),
            training_binding_hash=require_sha256_digest(
                value.get("training_binding_hash"), "training_binding_hash"
            ),
            capability_profile_hash=require_sha256_digest(
                value.get("capability_profile_hash"), "capability_profile_hash"
            ),
            checkpoint_hash=require_sha256_checksum(
                value.get("checkpoint_hash"), "checkpoint_hash"
            ),
            eval_protocol_hash=require_sha256_digest(
                value.get("eval_protocol_hash"), "eval_protocol_hash"
            ),
            evaluation_binding_hash=require_sha256_digest(
                value.get("evaluation_binding_hash"), "evaluation_binding_hash"
            ),
            execution_device=_required_text(
                value.get("execution_device"), "execution_device"
            ),
            resource_binding_hash=value.get("resource_binding_hash"),
            task_evidence=_required_mapping(task_evidence, "task_evidence"),
            execution_cost=_required_mapping(
                value.get("execution_cost"), "execution_cost"
            ),
            environment_source_preflight=_required_mapping(
                value.get("environment_source_preflight"),
                "environment_source_preflight",
            ),
            backend_execution_environment_verification=_required_mapping(
                value.get("backend_execution_environment_verification"),
                "backend_execution_environment_verification",
            ),
            checkpoint_safety_report=_required_mapping(
                value.get("checkpoint_safety_report"),
                "checkpoint_safety_report",
            ),
            gpu_process_binding=(
                _required_mapping(
                    value.get("gpu_process_binding"),
                    "gpu_process_binding",
                )
                if schema_version
                == EVALUATION_PHASE_RESULT_GPU_BINDING_SCHEMA_VERSION
                else None
            ),
        )


@dataclass(frozen=True)
class EvaluationPhaseReceiptBinding:
    """One immutable phase receipt referenced by an assembled evaluation."""

    phase: EvaluationPhase
    job_id: str
    path: str
    checksum: str
    size_bytes: int
    ref_schema_version: str
    job_spec_hash: str
    phase_result_hash: str
    resource_binding_hash: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, EvaluationPhase):
            raise TypeError("evaluation phase receipt phase is invalid")
        for field_name in ("job_id", "path", "ref_schema_version"):
            _required_text(getattr(self, field_name), field_name)
        require_sha256_checksum(
            self.checksum,
            "evaluation phase receipt checksum",
        )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_EVALUATION_PHASE_RECEIPT_BYTES
        ):
            raise ValueError("evaluation phase receipt size is invalid")
        require_sha256_digest(
            self.job_spec_hash,
            "evaluation phase receipt job_spec_hash",
        )
        require_sha256_digest(
            self.phase_result_hash,
            "evaluation phase receipt phase_result_hash",
        )
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "evaluation phase receipt resource_binding_hash",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "job_id": self.job_id,
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "ref_schema_version": self.ref_schema_version,
            "job_spec_hash": self.job_spec_hash,
            "phase_result_hash": self.phase_result_hash,
            "resource_binding_hash": self.resource_binding_hash,
        }


@dataclass(frozen=True)
class EvaluationAssembly:
    """Task-validated summary bound to selection and executed phase receipts."""

    evaluation_plan_hash: str
    selected_trial_id: str
    selection: Mapping[str, Any]
    phase_receipts: tuple[EvaluationPhaseReceiptBinding, ...]
    summary: Mapping[str, Any]
    assembly_hash: str | None = None

    def __post_init__(self) -> None:
        require_sha256_digest(
            self.evaluation_plan_hash,
            "evaluation assembly evaluation_plan_hash",
        )
        _required_text(self.selected_trial_id, "selected_trial_id")
        if not isinstance(self.selection, Mapping):
            raise TypeError("evaluation assembly selection must be an object")
        selection = copy.deepcopy(dict(self.selection))
        if (
            selection.get("selected_trial_id") != self.selected_trial_id
            or selection.get("acceptance_authority") is not False
        ):
            raise ValueError("evaluation assembly selection binding mismatch")
        if (
            not isinstance(self.phase_receipts, tuple)
            or any(
                not isinstance(item, EvaluationPhaseReceiptBinding)
                for item in self.phase_receipts
            )
            or {item.phase for item in self.phase_receipts} != set(EvaluationPhase)
            or len(self.phase_receipts) != len(EvaluationPhase)
        ):
            raise ValueError("evaluation assembly requires exactly three phase receipts")
        if len({item.path for item in self.phase_receipts}) != len(EvaluationPhase):
            raise ValueError("evaluation assembly phase receipt paths must be unique")
        if not isinstance(self.summary, Mapping):
            raise TypeError("evaluation assembly summary must be an object")
        summary = copy.deepcopy(dict(self.summary))
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "summary", summary)
        expected = canonical_json_hash(self._hash_payload())
        if self.assembly_hash is not None:
            require_sha256_digest(
                self.assembly_hash,
                "evaluation assembly hash",
            )
            if self.assembly_hash != expected:
                raise ValueError("evaluation assembly hash mismatch")
        object.__setattr__(self, "assembly_hash", expected)

    @property
    def selection_hash(self) -> str:
        return canonical_json_hash(self.selection)

    @property
    def summary_hash(self) -> str:
        return canonical_json_hash(self.summary)

    def to_private_dict(self) -> dict[str, Any]:
        assert self.assembly_hash is not None
        return {**self._hash_payload(), "assembly_hash": self.assembly_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_ASSEMBLY_SCHEMA_VERSION,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "selected_trial_id": self.selected_trial_id,
            "selection": copy.deepcopy(dict(self.selection)),
            "selection_hash": self.selection_hash,
            "phase_receipts": [
                item.to_dict()
                for item in sorted(
                    self.phase_receipts,
                    key=lambda receipt: receipt.phase.value,
                )
            ],
            "summary": copy.deepcopy(dict(self.summary)),
            "summary_hash": self.summary_hash,
            "acceptance_authority": False,
        }


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation phase result {field_name} is required")
    return value


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation phase result {field_name} must be an object")
    return value


def _validated_gpu_process_binding_mapping(value: Any) -> dict[str, str]:
    expected_fields = {
        "host_execution_device",
        "process_execution_device",
        "resource_binding_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("evaluation phase GPU process binding fields mismatch")
    result = dict(value)
    for field_name in (
        "host_execution_device",
        "process_execution_device",
    ):
        field_value = result.get(field_name)
        if (
            not isinstance(field_value, str)
            or _CUDA_EXECUTION_DEVICE_PATTERN.fullmatch(field_value) is None
        ):
            raise ValueError(
                f"evaluation phase GPU process binding {field_name} is invalid"
            )
    result["resource_binding_hash"] = require_sha256_digest(
        result.get("resource_binding_hash"),
        "evaluation phase GPU process binding resource_binding_hash",
    )
    return result


__all__ = [
    "EVALUATION_ASSEMBLY_SCHEMA_VERSION",
    "EVALUATION_PHASE_RESULT_SCHEMA_VERSION",
    "EVALUATION_PHASE_RESULT_GPU_BINDING_SCHEMA_VERSION",
    "EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION",
    "EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION",
    "EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION",
    "MAX_EVALUATION_PHASE_RECEIPT_BYTES",
    "EvaluationAssembly",
    "EvaluationPhase",
    "EvaluationPhaseReceiptBinding",
    "EvaluationPhaseResult",
]

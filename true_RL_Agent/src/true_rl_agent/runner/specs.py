"""Runner-owned train/eval job specifications."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

from true_rl_agent.candidates.artifact import VerifiedCandidateArtifactRef
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.reference_artifact import VerifiedReferenceProgramArtifactRef
from true_rl_agent.runner.candidate_failure import CandidateFailureContract
from true_rl_agent.runner.initial_policy import (
    verified_initial_policy_state_pairing_hash,
)
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.training_binding import TrainingBinding


def require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


_EXECUTION_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda:(?:0|[1-9][0-9]*))\Z")


def require_execution_device(value: Any, field_name: str = "execution_device") -> str:
    """Require an explicit canonical CPU or indexed CUDA device."""

    if not isinstance(value, str) or _EXECUTION_DEVICE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be canonical 'cpu' or 'cuda:<non-negative index>'"
        )
    return value


def _resolved_sealed_validation_seed_count(
    declared: Any,
    legacy: Any,
) -> int:
    if declared is None:
        if legacy is None:
            raise ValueError("sealed_validation_seed_count must be provided")
        return require_positive_integer(legacy, "trusted_seed_count")
    resolved = require_positive_integer(
        declared,
        "sealed_validation_seed_count",
    )
    if legacy is not None:
        legacy_count = require_positive_integer(legacy, "trusted_seed_count")
        if legacy_count != resolved:
            raise ValueError(
                "trusted_seed_count legacy alias does not match "
                "sealed_validation_seed_count"
            )
    return resolved


@dataclass(frozen=True)
class CheckpointRef:
    """Byte-addressed checkpoint reference with an explicit lifecycle role."""

    uri: str
    checksum: str
    role: str
    sealed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("checkpoint uri must be a non-empty string")
        require_sha256_checksum(self.checksum, "checkpoint checksum")
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("checkpoint role must be a non-empty string")
        if self.role not in {"warm_start", "eval", "train_output"}:
            raise ValueError(f"unsupported checkpoint role: {self.role}")
        if not isinstance(self.sealed, bool):
            raise ValueError("checkpoint sealed must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("checkpoint metadata must be an object")

    def to_public_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        data = {
            "uri": self.uri,
            "checksum": self.checksum,
            "role": self.role,
            "sealed": self.sealed,
        }
        if include_metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class CheckpointOutputPlan:
    """Planned checkpoint output location before any bytes or digest exist."""

    uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("checkpoint output plan uri must be a non-empty string")

    def to_public_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "state": "planned_output",
        }


@dataclass(frozen=True)
class TrainJobSpec:
    """Materialized fixed-budget training job."""

    job_id: str
    task_id: str
    subject: ExperimentSubject
    subject_manifest_hash: str | None
    program_source_checksum: str
    policy_initialization: PolicyInitialization
    training_binding: TrainingBinding
    budget_steps: int
    execution_device: str
    expected_initial_policy_state_hash: str | None = None
    candidate_failure_contract: CandidateFailureContract | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_artifact: VerifiedCandidateArtifactRef | None = None
    reference_artifact: VerifiedReferenceProgramArtifactRef | None = None
    attempt_scope: JobAttemptScope | None = None

    def __post_init__(self) -> None:
        self.job_identity()
        if self.subject.kind not in {"candidate", "reference"}:
            raise ValueError("train job subject must be candidate or reference")
        require_sha256_checksum(
            self.program_source_checksum,
            "train job program_source_checksum",
        )
        if not isinstance(self.training_binding, TrainingBinding):
            raise TypeError("train job training_binding must be a TrainingBinding")
        if self.training_binding.initialization_mode != self.policy_initialization.kind:
            raise ValueError("train job policy initialization does not match training binding")
        if self.training_binding.budget_steps != self.budget_steps:
            raise ValueError("train job budget does not match training binding")
        require_execution_device(self.execution_device, "train job execution_device")
        if self.expected_initial_policy_state_hash is not None:
            require_sha256_digest(
                self.expected_initial_policy_state_hash,
                "train job expected initial policy state hash",
            )
            if self.subject.kind != "candidate":
                raise ValueError(
                    "only candidate train jobs may carry an expected initial policy state"
                )
            if self.policy_initialization.kind != "random":
                raise ValueError(
                    "initial policy state pairing requires random initialization"
                )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("train job metadata must be an object")
        if self.attempt_scope is not None and not isinstance(
            self.attempt_scope,
            JobAttemptScope,
        ):
            raise TypeError("train job attempt_scope must be a JobAttemptScope")
        if self.subject.kind == "candidate":
            if not isinstance(
                self.candidate_artifact,
                VerifiedCandidateArtifactRef,
            ):
                raise TypeError(
                    "candidate train job requires a verified candidate_artifact"
                )
            if self.candidate_artifact.candidate_id != self.subject.subject_id:
                raise ValueError("train job candidate artifact subject mismatch")
            if (
                self.candidate_artifact.subject_manifest_hash
                != self.subject_manifest_hash
            ):
                raise ValueError("train job candidate artifact manifest mismatch")
            if (
                self.candidate_artifact.program_source_checksum
                != self.program_source_checksum
            ):
                raise ValueError("train job candidate artifact source mismatch")
            if self.reference_artifact is not None:
                raise ValueError(
                    "candidate train job cannot carry a reference_artifact"
                )
            if self.candidate_failure_contract is not None and not isinstance(
                self.candidate_failure_contract,
                CandidateFailureContract,
            ):
                raise TypeError(
                    "candidate train job failure contract must be a "
                    "CandidateFailureContract"
                )
        else:
            if self.candidate_failure_contract is not None:
                raise ValueError(
                    "reference train job cannot carry a candidate failure contract"
                )
            if self.candidate_artifact is not None:
                raise ValueError(
                    "reference train job cannot carry a candidate_artifact"
                )
            if not isinstance(
                self.reference_artifact,
                VerifiedReferenceProgramArtifactRef,
            ):
                raise TypeError(
                    "reference train job requires a verified reference_artifact"
                )
            if self.reference_artifact.reference_id != self.subject.subject_id:
                raise ValueError("train job reference artifact subject mismatch")
            if self.reference_artifact.task_id != self.task_id:
                raise ValueError("train job reference artifact task mismatch")
            if (
                self.reference_artifact.subject_manifest_hash
                != self.subject_manifest_hash
            ):
                raise ValueError("train job reference artifact manifest mismatch")
            if (
                self.reference_artifact.program_source_checksum
                != self.program_source_checksum
            ):
                raise ValueError("train job reference artifact source mismatch")
            if (
                self.reference_artifact.program.manifest.capability_profile_hash
                != self.training_binding.capability_profile_hash
            ):
                raise ValueError("train job reference artifact capability mismatch")
        require_positive_integer(self.budget_steps, "budget_steps")

    def job_identity(self) -> JobIdentity:
        return JobIdentity(
            job_id=self.job_id,
            task_id=self.task_id,
            subject=self.subject,
            subject_manifest_hash=self.subject_manifest_hash,
        )

    def training_start_checkpoint(self) -> CheckpointRef | None:
        if self.policy_initialization.baseline is None:
            return None
        checkpoint = self.policy_initialization.baseline.checkpoint
        if not isinstance(checkpoint, CheckpointRef):
            raise TypeError("baseline checkpoint must be a CheckpointRef")
        if checkpoint.role != "warm_start":
            raise ValueError(
                "training start checkpoint requires role='warm_start'"
            )
        if not checkpoint.sealed:
            raise ValueError("training start checkpoint must be sealed")
        if (
            self.training_binding.initialization_checkpoint_hash
            != checkpoint.checksum
        ):
            raise ValueError(
                "training start checkpoint does not match training binding"
            )
        return checkpoint

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "subject": self.subject.to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "policy_initialization": self.policy_initialization.kind,
            "training_binding": self.training_binding.to_public_dict(),
            "budget_steps": self.budget_steps,
            "execution_device": self.execution_device,
            "candidate_artifact": (
                self.candidate_artifact.to_public_dict()
                if self.candidate_artifact is not None
                else None
            ),
            "reference_artifact": (
                self.reference_artifact.to_public_dict()
                if self.reference_artifact is not None
                else None
            ),
            "training_start_checkpoint": (
                self.training_start_checkpoint().to_public_dict(include_metadata=False)
                if self.training_start_checkpoint() is not None
                else None
            ),
        }
        if self.expected_initial_policy_state_hash is not None:
            result["expected_initial_policy_state_hash"] = (
                self.expected_initial_policy_state_hash
            )
        if self.candidate_failure_contract is not None:
            result["candidate_failure_contract_hash"] = (
                self.candidate_failure_contract.contract_hash
            )
        return result


@dataclass(frozen=True)
class EvalJobSpec:
    """Candidate eval job, either planned or bound to a sealed train output."""

    job_id: str
    task_id: str
    subject: ExperimentSubject
    subject_manifest_hash: str | None
    program_source_checksum: str
    training_binding_hash: str
    capability_profile_hash: str
    checkpoint: CheckpointRef | None
    public_seed_count: int
    evaluation_binding: EvaluationBindingV1
    execution_device: str
    initial_policy_state_pairing_hash: str | None = None
    sealed_validation_seed_count: int | None = None
    public_accept_seed_count: int = 3
    eval_num_envs: int = 64
    eval_protocol_version: str = DEFAULT_EVAL_PROTOCOL_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_output_plan: CheckpointOutputPlan | None = None
    attempt_scope: JobAttemptScope | None = None
    trusted_seed_count: InitVar[int | None] = None

    def __post_init__(self, trusted_seed_count: int | None = None) -> None:
        self.job_identity()
        if self.subject.kind not in {"candidate", "reference"}:
            raise ValueError("eval job subject must be candidate or reference")
        require_sha256_checksum(
            self.program_source_checksum,
            "eval job program_source_checksum",
        )
        require_sha256_digest(
            self.training_binding_hash,
            "eval job training_binding_hash",
        )
        require_sha256_digest(
            self.capability_profile_hash,
            "eval job capability_profile_hash",
        )
        if self.checkpoint is not None and not isinstance(
            self.checkpoint,
            CheckpointRef,
        ):
            raise TypeError("eval job checkpoint must be a CheckpointRef")
        if self.checkpoint_output_plan is not None and not isinstance(
            self.checkpoint_output_plan,
            CheckpointOutputPlan,
        ):
            raise TypeError(
                "eval job checkpoint_output_plan must be a CheckpointOutputPlan"
            )
        if (self.checkpoint is None) == (self.checkpoint_output_plan is None):
            raise ValueError(
                "EvalJobSpec requires exactly one of checkpoint or "
                "checkpoint_output_plan"
            )
        if self.checkpoint is None:
            if self.initial_policy_state_pairing_hash is not None:
                raise ValueError(
                    "planned eval job cannot declare an initial policy pairing"
                )
        else:
            checkpoint_pairing_hash = (
                verified_initial_policy_state_pairing_hash(
                    self.checkpoint.metadata.get(
                        "initial_policy_state_pairing"
                    )
                )
            )
            if checkpoint_pairing_hash is None:
                if self.initial_policy_state_pairing_hash is not None:
                    raise ValueError(
                        "eval job initial policy pairing is not present in checkpoint lineage"
                    )
            else:
                require_sha256_digest(
                    self.initial_policy_state_pairing_hash,
                    "eval job initial_policy_state_pairing_hash",
                )
                if self.initial_policy_state_pairing_hash != checkpoint_pairing_hash:
                    raise ValueError(
                        "eval job initial policy pairing does not match checkpoint lineage"
                    )
        if not isinstance(self.evaluation_binding, EvaluationBindingV1):
            raise TypeError("eval job evaluation_binding must be EvaluationBindingV1")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("eval job metadata must be an object")
        if self.attempt_scope is not None and not isinstance(
            self.attempt_scope,
            JobAttemptScope,
        ):
            raise TypeError("eval job attempt_scope must be a JobAttemptScope")
        sealed_validation_seed_count = _resolved_sealed_validation_seed_count(
            self.sealed_validation_seed_count,
            trusted_seed_count,
        )
        object.__setattr__(
            self,
            "sealed_validation_seed_count",
            sealed_validation_seed_count,
        )
        if self.checkpoint is not None:
            if self.checkpoint.role != "eval":
                raise ValueError(
                    "EvalJobSpec requires a checkpoint with role='eval'"
                )
            if not self.checkpoint.sealed:
                raise ValueError(
                    "EvalJobSpec requires a sealed checkpoint once checkpoint "
                    "bytes are bound"
                )
        require_positive_integer(self.public_seed_count, "public_seed_count")
        require_positive_integer(
            self.public_accept_seed_count,
            "public_accept_seed_count",
        )
        require_positive_integer(self.eval_num_envs, "eval_num_envs")
        require_execution_device(self.execution_device, "eval job execution_device")
        if not self.eval_protocol_version:
            raise ValueError("eval_protocol_version must be non-empty")
        if self.evaluation_binding.eval_protocol_hash != self.evaluation_protocol().protocol_hash:
            raise ValueError("eval job evaluation binding does not match evaluation protocol")

    def evaluation_subject(self) -> ExperimentSubject:
        return self.subject

    def evaluation_manifest_hash(self) -> str | None:
        return self.subject_manifest_hash

    def execution_checkpoint(self) -> CheckpointRef:
        checkpoint = self.checkpoint
        if checkpoint is None:
            raise ValueError(
                "eval checkpoint bytes have not been bound to the job"
            )
        if checkpoint.role != "eval" or not checkpoint.sealed:
            raise ValueError(
                "eval execution requires a sealed checkpoint with "
                "role='eval'"
            )
        return checkpoint

    def job_identity(self) -> JobIdentity:
        return JobIdentity(
            job_id=self.job_id,
            task_id=self.task_id,
            subject=self.subject,
            subject_manifest_hash=self.subject_manifest_hash,
        )

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "subject": self.evaluation_subject().to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "execution_device": self.execution_device,
            **self.evaluation_protocol().to_public_dict(),
            "evaluation_binding": self.evaluation_binding.to_public_dict(),
        }
        if self.checkpoint is not None:
            result["checkpoint"] = self.checkpoint.to_public_dict(
                include_metadata=False
            )
        else:
            assert self.checkpoint_output_plan is not None
            result["checkpoint_plan"] = (
                self.checkpoint_output_plan.to_public_dict()
            )
        if self.initial_policy_state_pairing_hash is not None:
            result["initial_policy_state_pairing_hash"] = (
                self.initial_policy_state_pairing_hash
            )
        return result


@dataclass(frozen=True)
class EvalPhaseJobSpec:
    """One split execution bound to a complete immutable candidate eval plan."""

    job_id: str
    evaluation_plan: EvalJobSpec
    phase: EvaluationPhase
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attempt_scope: JobAttemptScope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_plan, EvalJobSpec):
            raise TypeError("phase eval job requires an EvalJobSpec plan")
        self.evaluation_plan.execution_checkpoint()
        if not isinstance(self.phase, EvaluationPhase):
            raise TypeError("phase eval job phase is invalid")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("phase eval job metadata must be an object")
        if self.attempt_scope is not None and not isinstance(
            self.attempt_scope,
            JobAttemptScope,
        ):
            raise TypeError("phase eval job attempt_scope must be a JobAttemptScope")
        self.job_identity()

    @property
    def task_id(self) -> str:
        return self.evaluation_plan.task_id

    @property
    def subject(self) -> ExperimentSubject:
        return self.evaluation_plan.subject

    @property
    def subject_manifest_hash(self) -> str | None:
        return self.evaluation_plan.subject_manifest_hash

    @property
    def program_source_checksum(self) -> str:
        return self.evaluation_plan.program_source_checksum

    @property
    def training_binding_hash(self) -> str:
        return self.evaluation_plan.training_binding_hash

    @property
    def capability_profile_hash(self) -> str:
        return self.evaluation_plan.capability_profile_hash

    @property
    def evaluation_binding(self) -> EvaluationBindingV1:
        return self.evaluation_plan.evaluation_binding

    @property
    def execution_device(self) -> str:
        return self.evaluation_plan.execution_device

    @property
    def sealed_validation_seed_count(self) -> int:
        return require_positive_integer(
            self.evaluation_plan.sealed_validation_seed_count,
            "sealed_validation_seed_count",
        )

    @property
    def evaluation_plan_hash(self) -> str:
        return canonical_json_hash(self.evaluation_plan.to_public_dict())

    def evaluation_subject(self) -> ExperimentSubject:
        return self.subject

    def evaluation_manifest_hash(self) -> str | None:
        return self.subject_manifest_hash

    def execution_checkpoint(self) -> CheckpointRef:
        return self.evaluation_plan.execution_checkpoint()

    def evaluation_protocol(self) -> EvaluationProtocol:
        return self.evaluation_plan.evaluation_protocol()

    def job_identity(self) -> JobIdentity:
        return JobIdentity(
            job_id=self.job_id,
            task_id=self.task_id,
            subject=self.subject,
            subject_manifest_hash=self.subject_manifest_hash,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.eval_phase_job_spec.v1",
            "job_id": self.job_id,
            "phase": self.phase.value,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "evaluation_plan": self.evaluation_plan.to_public_dict(),
        }


def require_eval_phase_result_matches_spec(
    result: EvaluationPhaseResult,
    spec: EvalPhaseJobSpec,
    *,
    expected_resource_binding_hash: str | None = None,
) -> None:
    """Validate the sole task-neutral phase-result/job-spec identity contract."""

    if not isinstance(result, EvaluationPhaseResult):
        raise TypeError("eval phase result is invalid")
    if not isinstance(spec, EvalPhaseJobSpec):
        raise TypeError("eval phase job spec is invalid")
    if expected_resource_binding_hash is None and spec.attempt_scope is not None:
        expected_resource_binding_hash = spec.attempt_scope.resource_binding_hash
    checkpoint = spec.execution_checkpoint()
    expected = {
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "phase": spec.phase,
        "evaluation_plan_hash": spec.evaluation_plan_hash,
        "subject": spec.subject,
        "subject_manifest_hash": spec.subject_manifest_hash,
        "program_source_checksum": spec.program_source_checksum,
        "training_binding_hash": spec.training_binding_hash,
        "capability_profile_hash": spec.capability_profile_hash,
        "checkpoint_hash": checkpoint.checksum,
        "eval_protocol_hash": spec.evaluation_protocol().protocol_hash,
        "evaluation_binding_hash": spec.evaluation_binding.binding_hash,
        "execution_device": spec.execution_device,
        "resource_binding_hash": expected_resource_binding_hash,
    }
    if any(getattr(result, name) != value for name, value in expected.items()):
        raise ValueError("eval phase result does not match its job spec")


@dataclass(frozen=True)
class BaselineEvalJobSpec:
    """Eval-only job for a sealed warm-start baseline, with no candidate identity."""

    job_id: str
    task_id: str
    subject: ExperimentSubject
    subject_manifest_hash: str
    capability_profile_hash: str
    checkpoint: CheckpointRef
    public_seed_count: int
    evaluation_binding: EvaluationBindingV1
    execution_device: str
    sealed_validation_seed_count: int | None = None
    public_accept_seed_count: int = 3
    eval_num_envs: int = 64
    eval_protocol_version: str = DEFAULT_EVAL_PROTOCOL_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attempt_scope: JobAttemptScope | None = None
    trusted_seed_count: InitVar[int | None] = None

    def __post_init__(self, trusted_seed_count: int | None = None) -> None:
        self.job_identity()
        if self.subject.kind != "baseline":
            raise ValueError("baseline eval job subject must be baseline")
        require_sha256_digest(
            self.subject_manifest_hash,
            "subject_manifest_hash",
        )
        require_sha256_digest(
            self.capability_profile_hash,
            "baseline eval job capability_profile_hash",
        )
        if not isinstance(self.evaluation_binding, EvaluationBindingV1):
            raise TypeError(
                "baseline eval job evaluation_binding must be EvaluationBindingV1"
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("baseline eval job metadata must be an object")
        if self.attempt_scope is not None and not isinstance(
            self.attempt_scope,
            JobAttemptScope,
        ):
            raise TypeError(
                "baseline eval job attempt_scope must be a JobAttemptScope"
            )
        sealed_validation_seed_count = _resolved_sealed_validation_seed_count(
            self.sealed_validation_seed_count,
            trusted_seed_count,
        )
        object.__setattr__(
            self,
            "sealed_validation_seed_count",
            sealed_validation_seed_count,
        )
        if self.checkpoint.role != "warm_start":
            raise ValueError(
                "BaselineEvalJobSpec requires a checkpoint with role='warm_start'"
            )
        if not self.checkpoint.sealed:
            raise ValueError("BaselineEvalJobSpec requires a sealed checkpoint")
        require_positive_integer(self.public_seed_count, "public_seed_count")
        require_positive_integer(
            self.public_accept_seed_count,
            "public_accept_seed_count",
        )
        require_positive_integer(self.eval_num_envs, "eval_num_envs")
        require_execution_device(
            self.execution_device,
            "baseline eval job execution_device",
        )
        if not self.eval_protocol_version:
            raise ValueError("eval_protocol_version must be non-empty")
        if (
            self.evaluation_binding.eval_protocol_hash
            != self.evaluation_protocol().protocol_hash
        ):
            raise ValueError(
                "baseline eval job evaluation binding does not match evaluation protocol"
            )

    def evaluation_subject(self) -> ExperimentSubject:
        return self.subject

    def evaluation_manifest_hash(self) -> str:
        return self.subject_manifest_hash

    def execution_checkpoint(self) -> CheckpointRef:
        if self.checkpoint.role != "warm_start" or not self.checkpoint.sealed:
            raise ValueError(
                "baseline eval execution requires a sealed checkpoint with "
                "role='warm_start'"
            )
        return self.checkpoint

    def job_identity(self) -> JobIdentity:
        return JobIdentity(
            job_id=self.job_id,
            task_id=self.task_id,
            subject=self.subject,
            subject_manifest_hash=self.subject_manifest_hash,
        )

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "subject": self.evaluation_subject().to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "execution_device": self.execution_device,
            "checkpoint": self.checkpoint.to_public_dict(include_metadata=False),
            **self.evaluation_protocol().to_public_dict(),
            "evaluation_binding": self.evaluation_binding.to_public_dict(),
        }


EvaluationJobSpec = EvalJobSpec | EvalPhaseJobSpec | BaselineEvalJobSpec

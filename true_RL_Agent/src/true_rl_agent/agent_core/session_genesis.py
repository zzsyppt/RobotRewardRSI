"""Create-once session trust root for acceptance replay."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationSessionContractRef,
    load_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    _canonical_uuid_hex,
    _current_process_group_id,
    _positive_int,
    _process_group_is_alive,
    _process_incarnation_token,
    _timestamp,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    _load_json as _load_generation_json,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    _publish_json_once as _publish_generation_json_once,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    _reservation_ref as _generation_reservation_ref,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    _validate_reservation_ref as _validate_generation_reservation_ref,
)
from true_rl_agent.agent_core.process_lock import (
    ProcessFileLock,
    ProcessFileLockUnavailableError,
    acquire_process_file_lock,
)
from true_rl_agent.agent_core.trial_receipts import (
    TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
    LoadedTrialEvalPhaseReceipt,
    TrialReceiptJournal,
    TrialReceiptRef,
    load_trial_eval_phase_receipt,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    is_sha256_checksum,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseReceiptBinding,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import (
    ExperimentSubject,
    validate_experiment_subject_id,
)
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_loads,
)
from true_rl_agent.memory.records import PrivateArtifactRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.reference_artifact import (
    VerifiedReferenceProgramArtifactRef,
    load_reference_program_artifact,
)
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.runner.job_attempts import (
    JobAttemptJournal,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
    require_execution_device,
)
from true_rl_agent.safety.eval_seal import TrustedEvalSeal, build_trusted_eval_seal
from true_rl_agent.tasks.base import TaskPlugin, TaskSpec
from true_rl_agent.tasks.reference_qualification import (
    REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
    ReferenceQualificationResult,
)
from true_rl_agent.training_binding import TrainingBinding
from true_rl_agent.training_programs import (
    ReferenceProgramManifest,
    VerifiedReferenceProgramArtifact,
)

SESSION_GENESIS_SCHEMA_VERSION = "true_rl_agent.session_genesis.v3"
GENERATION_SESSION_GENESIS_SCHEMA_VERSION = "true_rl_agent.session_genesis.v4"
LEGACY_SESSION_GENESIS_SCHEMA_VERSION = "true_rl_agent.session_genesis.v2"
BASELINE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.baseline_eval_artifact.v1"
)
REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.reference_eval_artifact.v2"
)
GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.reference_eval_artifact.v3"
)
PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.reference_eval_artifact.v4"
)
SESSION_GENESIS_FILENAME = "session_genesis.json"
BASELINE_EVAL_ARTIFACT_FILENAME = "baseline_eval_artifact.json"
REFERENCE_EVAL_ARTIFACT_FILENAME = "reference_eval_artifact.json"
SESSION_GENESIS_CREATION_LEASE_FILENAME = ".session_genesis.create.lease"
_SESSION_GENESIS_CREATION_LEASE_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.creation_lease.v2"
)
_PRE_GENESIS_CREATION_LEASE_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.creation_lease.v3"
)
_PRE_GENESIS_UNCLEAN_RECOVERY_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.unclean_recovery.v1"
)
_PRE_GENESIS_RECOVERY_DIRNAME = "session_genesis_recoveries"
_SESSION_GENESIS_RECOVERY_MAX_BYTES = 128 * 1024


class SessionGenesisCreationInProgressError(RuntimeError):
    """Raised when another writer owns creation of the same session Genesis."""


class ReferencePhasedEvaluationAdapter(Protocol):
    """Task-owned assembly hook for a split reference evaluation."""

    def assemble_reference_full_summary(
        self,
        *,
        eval_plan: EvalJobSpec,
        phase_results: Sequence[EvaluationPhaseResult],
        phase_receipt_refs: Mapping[
            EvaluationPhase | str,
            Mapping[str, Any],
        ],
    ) -> EvaluationAssembly: ...


def _reference_eval_phase_role(phase: EvaluationPhase) -> str:
    return f"reference_eval_{phase.value}"


def _normalize_reference_eval_phase_attempt_scopes(
    value: Mapping[EvaluationPhase | str, PreGenesisExecutionScope] | None,
) -> Mapping[EvaluationPhase, PreGenesisExecutionScope] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("reference_eval_phase_attempt_scopes must be an object")
    normalized: dict[EvaluationPhase, PreGenesisExecutionScope] = {}
    for raw_phase, scope in value.items():
        phase = EvaluationPhase.parse(raw_phase)
        if phase in normalized:
            raise ValueError("reference eval phase attempt scopes repeat a phase")
        if not isinstance(scope, PreGenesisExecutionScope):
            raise TypeError(
                "reference eval phase attempt scope must be a "
                "PreGenesisExecutionScope"
            )
        normalized[phase] = scope
    if set(normalized) != set(EvaluationPhase):
        raise ValueError(
            "reference eval phase attempt scopes must cover every evaluation phase"
        )
    return MappingProxyType(dict(normalized))


@dataclass
class _SessionGenesisCreationLease:
    payload: bytes
    lock: ProcessFileLock
    previous_payload: bytes | None = None
    closed_payload: bytes | None = None
    active: bool = True

    def assert_owned(self) -> None:
        self.lock.assert_owned()

    def activate(self) -> None:
        if self.active:
            self.assert_owned()
            return
        self.lock.replace_payload(self.payload)
        self.active = True

    def release(self) -> None:
        try:
            if self.active and self.closed_payload is not None:
                self.lock.replace_payload(self.closed_payload)
        finally:
            self.lock.release()
            self.active = False


@dataclass(frozen=True)
class BootstrapGenesisAnchor:
    anchor_id: str
    initial_best_score: float
    initial_policy_state_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise ValueError("bootstrap anchor_id must be non-empty")
        object.__setattr__(
            self,
            "initial_best_score",
            _finite_number(
                self.initial_best_score,
                "bootstrap initial_best_score",
            ),
        )
        if self.initial_policy_state_hash is not None:
            require_sha256_digest(
                self.initial_policy_state_hash,
                "bootstrap initial policy state hash",
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "BootstrapGenesisAnchor":
        if not isinstance(value, Mapping):
            raise TypeError("bootstrap genesis anchor must be an object")
        fields = set(value)
        legacy_fields = {"kind", "anchor_id", "initial_best_score"}
        anchored_fields = legacy_fields | {"initial_policy_state_hash"}
        if fields not in (legacy_fields, anchored_fields):
            raise ValueError("bootstrap genesis anchor fields mismatch")
        mapping = value
        if mapping.get("kind") != "bootstrap":
            raise ValueError("bootstrap genesis anchor kind mismatch")
        return cls(
            anchor_id=_nonempty_text(mapping.get("anchor_id"), "bootstrap anchor_id"),
            initial_best_score=_finite_number(
                mapping.get("initial_best_score"),
                "bootstrap initial_best_score",
            ),
            initial_policy_state_hash=(
                require_sha256_digest(
                    mapping.get("initial_policy_state_hash"),
                    "bootstrap initial policy state hash",
                )
                if "initial_policy_state_hash" in mapping
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": "bootstrap",
            "anchor_id": self.anchor_id,
            "initial_best_score": self.initial_best_score,
        }
        if self.initial_policy_state_hash is not None:
            payload["initial_policy_state_hash"] = self.initial_policy_state_hash
        return payload


@dataclass(frozen=True)
class BaselineGenesisAnchor:
    baseline_id: str
    baseline_manifest_hash: str
    checkpoint_hash: str
    public_result: Mapping[str, Any]
    public_result_hash: str
    trusted_eval_seal: Mapping[str, Any]
    private_eval_artifact: PrivateArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id:
            raise ValueError("baseline genesis baseline_id must be non-empty")
        require_sha256_digest(
            self.baseline_manifest_hash,
            "baseline genesis manifest hash",
        )
        require_sha256_checksum(
            self.checkpoint_hash,
            "baseline genesis checkpoint hash",
        )
        if canonical_json_hash(self.public_result) != self.public_result_hash:
            raise ValueError("baseline genesis public result hash mismatch")
        assert_public_projection_safe(self.public_result)
        _validate_private_artifact_ref(
            self.private_eval_artifact,
            expected_kind="baseline_eval_artifact",
            expected_reason="session_genesis_baseline_evidence",
            field_prefix="baseline Genesis",
        )
        seal_hash = self.trusted_eval_seal.get("seal_hash")
        require_sha256_digest(seal_hash, "baseline genesis trusted eval seal hash")
        object.__setattr__(
            self,
            "public_result",
            _freeze_json_value(_json_object(self.public_result, "public_result")),
        )
        object.__setattr__(
            self,
            "trusted_eval_seal",
            _freeze_json_value(
                _json_object(self.trusted_eval_seal, "trusted_eval_seal")
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "BaselineGenesisAnchor":
        mapping = _exact_mapping(
            value,
            {
                "kind",
                "baseline_id",
                "baseline_manifest_hash",
                "checkpoint_hash",
                "public_result",
                "public_result_hash",
                "trusted_eval_seal",
                "private_eval_artifact",
            },
            "baseline genesis anchor",
        )
        if mapping.get("kind") != "policy_baseline":
            raise ValueError("baseline genesis anchor kind mismatch")
        artifact = _exact_mapping(
            mapping.get("private_eval_artifact"),
            {"kind", "uri", "checksum", "reason"},
            "baseline private eval artifact",
        )
        return cls(
            baseline_id=_nonempty_text(
                mapping.get("baseline_id"),
                "baseline genesis baseline_id",
            ),
            baseline_manifest_hash=require_sha256_digest(
                mapping.get("baseline_manifest_hash"),
                "baseline genesis manifest hash",
            ),
            checkpoint_hash=require_sha256_checksum(
                mapping.get("checkpoint_hash"),
                "baseline genesis checkpoint hash",
            ),
            public_result=_json_object(
                mapping.get("public_result"),
                "baseline genesis public_result",
            ),
            public_result_hash=require_sha256_digest(
                mapping.get("public_result_hash"),
                "baseline genesis public result hash",
            ),
            trusted_eval_seal=_json_object(
                mapping.get("trusted_eval_seal"),
                "baseline genesis trusted_eval_seal",
            ),
            private_eval_artifact=PrivateArtifactRef(
                kind=_nonempty_text(
                    artifact.get("kind"),
                    "baseline private artifact kind",
                ),
                uri=_nonempty_text(
                    artifact.get("uri"),
                    "baseline private artifact uri",
                ),
                checksum=require_sha256_checksum(
                    artifact.get("checksum"),
                    "baseline private artifact checksum",
                ),
                reason=_nonempty_text(
                    artifact.get("reason"),
                    "baseline private artifact reason",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "policy_baseline",
            "baseline_id": self.baseline_id,
            "baseline_manifest_hash": self.baseline_manifest_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "public_result": _json_copy(self.public_result),
            "public_result_hash": self.public_result_hash,
            "trusted_eval_seal": _json_copy(self.trusted_eval_seal),
            "private_eval_artifact": self.private_eval_artifact.to_private_dict(),
        }


@dataclass(frozen=True)
class ReferenceProgramGenesisAnchor:
    reference_manifest: ReferenceProgramManifest
    training_binding: TrainingBinding
    initial_policy_state_hash: str
    checkpoint_hash: str
    checkpoint_artifact: PrivateArtifactRef
    public_result: Mapping[str, Any]
    public_result_hash: str
    trusted_eval_seal: Mapping[str, Any]
    qualification_gate: Mapping[str, Any]
    private_eval_artifact: PrivateArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.reference_manifest, ReferenceProgramManifest):
            raise TypeError("reference Genesis manifest is invalid")
        if not isinstance(self.training_binding, TrainingBinding):
            raise TypeError("reference Genesis training binding is invalid")
        if self.training_binding.initialization_mode != "random":
            raise ValueError("reference Genesis training must use random initialization")
        if (
            self.training_binding.capability_profile_hash
            != self.reference_manifest.capability_profile_hash
        ):
            raise ValueError("reference Genesis capability profile mismatch")
        require_sha256_digest(
            self.initial_policy_state_hash,
            "reference Genesis initial policy state hash",
        )
        require_sha256_checksum(
            self.checkpoint_hash,
            "reference Genesis checkpoint hash",
        )
        _validate_private_artifact_ref(
            self.checkpoint_artifact,
            expected_kind="reference_checkpoint",
            expected_reason="session_genesis_reference_checkpoint",
            field_prefix="reference Genesis checkpoint",
        )
        if self.checkpoint_artifact.checksum != self.checkpoint_hash:
            raise ValueError("reference Genesis checkpoint artifact hash mismatch")
        if canonical_json_hash(self.public_result) != self.public_result_hash:
            raise ValueError("reference Genesis public result hash mismatch")
        assert_public_projection_safe(self.public_result)
        _validate_reference_qualification_gate(self.qualification_gate)
        _validate_reference_seal(
            self.trusted_eval_seal,
            reference_manifest=self.reference_manifest,
            training_binding=self.training_binding,
            checkpoint_hash=self.checkpoint_hash,
        )
        _validate_private_artifact_ref(
            self.private_eval_artifact,
            expected_kind="reference_eval_artifact",
            expected_reason="session_genesis_reference_evidence",
            field_prefix="reference Genesis",
        )
        object.__setattr__(
            self,
            "public_result",
            _freeze_json_value(_json_object(self.public_result, "public_result")),
        )
        object.__setattr__(
            self,
            "trusted_eval_seal",
            _freeze_json_value(
                _json_object(self.trusted_eval_seal, "trusted_eval_seal")
            ),
        )
        object.__setattr__(
            self,
            "qualification_gate",
            _freeze_json_value(
                _json_object(self.qualification_gate, "qualification_gate")
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "ReferenceProgramGenesisAnchor":
        mapping = _exact_mapping(
            value,
            {
                "kind",
                "reference_manifest",
                "training_binding",
                "initial_policy_state_hash",
                "checkpoint_hash",
                "checkpoint_artifact",
                "public_result",
                "public_result_hash",
                "trusted_eval_seal",
                "qualification_gate",
                "private_eval_artifact",
            },
            "reference genesis anchor",
        )
        if mapping.get("kind") != "reference_program":
            raise ValueError("reference genesis anchor kind mismatch")
        artifact = _exact_mapping(
            mapping.get("private_eval_artifact"),
            {"kind", "uri", "checksum", "reason"},
            "reference private eval artifact",
        )
        checkpoint_artifact = _exact_mapping(
            mapping.get("checkpoint_artifact"),
            {"kind", "uri", "checksum", "reason"},
            "reference checkpoint artifact",
        )
        return cls(
            reference_manifest=ReferenceProgramManifest.from_mapping(
                mapping.get("reference_manifest")
            ),
            training_binding=TrainingBinding.from_mapping(
                mapping.get("training_binding")
            ),
            initial_policy_state_hash=require_sha256_digest(
                mapping.get("initial_policy_state_hash"),
                "reference Genesis initial policy state hash",
            ),
            checkpoint_hash=require_sha256_checksum(
                mapping.get("checkpoint_hash"),
                "reference Genesis checkpoint hash",
            ),
            checkpoint_artifact=PrivateArtifactRef(
                kind=_nonempty_text(
                    checkpoint_artifact.get("kind"),
                    "reference checkpoint artifact kind",
                ),
                uri=_nonempty_text(
                    checkpoint_artifact.get("uri"),
                    "reference checkpoint artifact uri",
                ),
                checksum=require_sha256_checksum(
                    checkpoint_artifact.get("checksum"),
                    "reference checkpoint artifact checksum",
                ),
                reason=_nonempty_text(
                    checkpoint_artifact.get("reason"),
                    "reference checkpoint artifact reason",
                ),
            ),
            public_result=_json_object(
                mapping.get("public_result"),
                "reference Genesis public_result",
            ),
            public_result_hash=require_sha256_digest(
                mapping.get("public_result_hash"),
                "reference Genesis public result hash",
            ),
            trusted_eval_seal=_json_object(
                mapping.get("trusted_eval_seal"),
                "reference Genesis trusted_eval_seal",
            ),
            qualification_gate=_json_object(
                mapping.get("qualification_gate"),
                "reference Genesis qualification_gate",
            ),
            private_eval_artifact=PrivateArtifactRef(
                kind=_nonempty_text(
                    artifact.get("kind"),
                    "reference private artifact kind",
                ),
                uri=_nonempty_text(
                    artifact.get("uri"),
                    "reference private artifact uri",
                ),
                checksum=require_sha256_checksum(
                    artifact.get("checksum"),
                    "reference private artifact checksum",
                ),
                reason=_nonempty_text(
                    artifact.get("reason"),
                    "reference private artifact reason",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "reference_program",
            "reference_manifest": self.reference_manifest.to_public_dict(),
            "training_binding": self.training_binding.to_private_dict(),
            "initial_policy_state_hash": self.initial_policy_state_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_artifact": self.checkpoint_artifact.to_private_dict(),
            "public_result": _json_copy(self.public_result),
            "public_result_hash": self.public_result_hash,
            "trusted_eval_seal": _json_copy(self.trusted_eval_seal),
            "qualification_gate": _json_copy(self.qualification_gate),
            "private_eval_artifact": self.private_eval_artifact.to_private_dict(),
        }


GenesisAnchor = (
    BootstrapGenesisAnchor
    | BaselineGenesisAnchor
    | ReferenceProgramGenesisAnchor
)


@dataclass(frozen=True)
class SessionGenesisArtifact:
    session_id: str
    task_id: str
    backend_id: str
    policy_initialization_kind: str
    runtime_snapshot_hash: str
    evaluation_binding: EvaluationBindingV1
    backend_execution_environment: Mapping[str, Any] | None
    anchor: GenesisAnchor
    generation_session_contract_ref: GenerationSessionContractRef | None = None
    session_genesis_hash: str | None = None
    schema_version: str = SESSION_GENESIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "session_id",
            "task_id",
            "backend_id",
            "policy_initialization_kind",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self,
                field_name,
            ):
                raise ValueError(f"session genesis {field_name} must be non-empty")
        validate_experiment_subject_id(self.session_id)
        if self.policy_initialization_kind not in {"random", "warm_start"}:
            raise ValueError("session genesis policy initialization kind is invalid")
        require_sha256_digest(
            self.runtime_snapshot_hash,
            "session genesis runtime_snapshot_hash",
        )
        if self.schema_version not in {
            GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
            SESSION_GENESIS_SCHEMA_VERSION,
            LEGACY_SESSION_GENESIS_SCHEMA_VERSION,
        }:
            raise ValueError("session genesis schema_version mismatch")
        if self.schema_version == LEGACY_SESSION_GENESIS_SCHEMA_VERSION:
            if self.backend_execution_environment is not None:
                raise ValueError(
                    "legacy session Genesis cannot carry backend environment identity"
                )
        elif self.backend_execution_environment is not None:
            normalized_environment = _json_object(
                self.backend_execution_environment,
                "session Genesis backend execution environment",
            )
            require_sha256_digest(
                normalized_environment.get("manifest_hash"),
                "session Genesis backend environment manifest hash",
            )
            object.__setattr__(
                self,
                "backend_execution_environment",
                _freeze_json_value(normalized_environment),
            )
        if self.schema_version == GENERATION_SESSION_GENESIS_SCHEMA_VERSION:
            if not isinstance(
                self.generation_session_contract_ref,
                GenerationSessionContractRef,
            ):
                raise TypeError(
                    "generation session Genesis requires a generation contract ref"
                )
        elif self.generation_session_contract_ref is not None:
            raise ValueError(
                "legacy/width-one session Genesis cannot carry a generation contract ref"
            )
        if not isinstance(self.evaluation_binding, EvaluationBindingV1):
            raise TypeError("session genesis evaluation_binding must be EvaluationBindingV1")
        if not isinstance(
            self.anchor,
            BootstrapGenesisAnchor
            | BaselineGenesisAnchor
            | ReferenceProgramGenesisAnchor,
        ):
            raise TypeError("session genesis anchor type is invalid")
        canonical_hash = canonical_json_hash(self._hash_payload())
        if self.session_genesis_hash is not None:
            require_sha256_digest(
                self.session_genesis_hash,
                "session_genesis_hash",
            )
            if self.session_genesis_hash != canonical_hash:
                raise ValueError("session genesis hash mismatch")
        object.__setattr__(self, "session_genesis_hash", canonical_hash)

    @property
    def genesis_kind(self) -> str:
        if isinstance(self.anchor, BaselineGenesisAnchor):
            return "policy_baseline"
        if isinstance(self.anchor, ReferenceProgramGenesisAnchor):
            return "reference_program"
        return "bootstrap"

    @classmethod
    def from_mapping(cls, value: Any) -> "SessionGenesisArtifact":
        raw_mapping = _json_object(value, "session genesis")
        schema_version = raw_mapping.get("schema_version")
        if schema_version == LEGACY_SESSION_GENESIS_SCHEMA_VERSION:
            expected_keys = {
                "schema_version",
                "session_id",
                "task_id",
                "backend_id",
                "policy_initialization_kind",
                "runtime_snapshot_hash",
                "evaluation_binding",
                "anchor",
                "session_genesis_hash",
            }
        elif schema_version in {
            SESSION_GENESIS_SCHEMA_VERSION,
            GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
        }:
            expected_keys = {
                "schema_version",
                "session_id",
                "task_id",
                "backend_id",
                "policy_initialization_kind",
                "runtime_snapshot_hash",
                "evaluation_binding",
                "backend_execution_environment",
                "anchor",
                "session_genesis_hash",
            }
            if schema_version == GENERATION_SESSION_GENESIS_SCHEMA_VERSION:
                expected_keys.add("generation_session_contract_ref")
        else:
            raise ValueError("session genesis schema_version mismatch")
        mapping = _exact_mapping(
            raw_mapping,
            expected_keys,
            "session genesis",
        )
        raw_anchor = _json_object(mapping.get("anchor"), "session genesis anchor")
        anchor_kind = raw_anchor.get("kind")
        if anchor_kind == "policy_baseline":
            anchor: GenesisAnchor = BaselineGenesisAnchor.from_mapping(raw_anchor)
        elif anchor_kind == "reference_program":
            anchor = ReferenceProgramGenesisAnchor.from_mapping(raw_anchor)
        elif anchor_kind == "bootstrap":
            anchor = BootstrapGenesisAnchor.from_mapping(raw_anchor)
        else:
            raise ValueError("session genesis anchor kind is invalid")
        return cls(
            session_id=_nonempty_text(mapping.get("session_id"), "session_id"),
            task_id=_nonempty_text(mapping.get("task_id"), "task_id"),
            backend_id=_nonempty_text(mapping.get("backend_id"), "backend_id"),
            policy_initialization_kind=_nonempty_text(
                mapping.get("policy_initialization_kind"),
                "policy_initialization_kind",
            ),
            runtime_snapshot_hash=require_sha256_digest(
                mapping.get("runtime_snapshot_hash"),
                "runtime_snapshot_hash",
            ),
            evaluation_binding=EvaluationBindingV1.from_mapping(
                mapping.get("evaluation_binding")
            ),
            backend_execution_environment=(
                _json_object(
                    mapping.get("backend_execution_environment"),
                    "session Genesis backend execution environment",
                )
                if schema_version in {
                    SESSION_GENESIS_SCHEMA_VERSION,
                    GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
                }
                and mapping.get("backend_execution_environment") is not None
                else None
            ),
            anchor=anchor,
            generation_session_contract_ref=(
                GenerationSessionContractRef.from_mapping(
                    mapping.get("generation_session_contract_ref")
                )
                if schema_version == GENERATION_SESSION_GENESIS_SCHEMA_VERSION
                else None
            ),
            session_genesis_hash=require_sha256_digest(
                mapping.get("session_genesis_hash"),
                "session_genesis_hash",
            ),
            schema_version=str(schema_version),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.session_genesis_hash is not None
        return {
            **self._hash_payload(),
            "session_genesis_hash": self.session_genesis_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "backend_id": self.backend_id,
            "policy_initialization_kind": self.policy_initialization_kind,
            "runtime_snapshot_hash": self.runtime_snapshot_hash,
            "evaluation_binding": self.evaluation_binding.to_public_dict(),
            "anchor": self.anchor.to_dict(),
        }
        if self.schema_version in {
            SESSION_GENESIS_SCHEMA_VERSION,
            GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
        }:
            payload["backend_execution_environment"] = (
                _json_copy(self.backend_execution_environment)
                if self.backend_execution_environment is not None
                else None
            )
        if self.schema_version == GENERATION_SESSION_GENESIS_SCHEMA_VERSION:
            assert self.generation_session_contract_ref is not None
            payload["generation_session_contract_ref"] = (
                self.generation_session_contract_ref.to_dict()
            )
        return payload


class VerifiedSessionGenesis:
    """Opaque result of contextual Genesis verification."""

    __slots__ = ("_artifact", "_artifact_path", "_initial_current_best")
    _TOKEN = object()

    def __init__(
        self,
        *,
        artifact: SessionGenesisArtifact,
        artifact_path: Path,
        initial_current_best: CurrentBestEvalStats,
        _token: object,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError("VerifiedSessionGenesis must come from the Genesis validator")
        object.__setattr__(self, "_artifact", artifact)
        object.__setattr__(self, "_artifact_path", artifact_path)
        object.__setattr__(self, "_initial_current_best", initial_current_best)

    def __setattr__(self, name: str, value: object) -> None:
        _ = name, value
        raise AttributeError("VerifiedSessionGenesis is immutable")

    @classmethod
    def _validated(
        cls,
        *,
        artifact: SessionGenesisArtifact,
        artifact_path: Path,
        initial_current_best: CurrentBestEvalStats,
    ) -> "VerifiedSessionGenesis":
        return cls(
            artifact=artifact,
            artifact_path=artifact_path,
            initial_current_best=initial_current_best,
            _token=cls._TOKEN,
        )

    @property
    def artifact(self) -> SessionGenesisArtifact:
        return self._artifact

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def session_genesis_hash(self) -> str:
        assert self._artifact.session_genesis_hash is not None
        return self._artifact.session_genesis_hash

    @property
    def evaluation_binding(self) -> EvaluationBindingV1:
        return self._artifact.evaluation_binding

    @property
    def backend_execution_environment(self) -> Mapping[str, Any] | None:
        environment = self._artifact.backend_execution_environment
        return _freeze_json_value(_json_copy(environment)) if environment else None

    @property
    def backend_execution_environment_manifest_hash(self) -> str | None:
        environment = self._artifact.backend_execution_environment
        if environment is None:
            return None
        return require_sha256_digest(
            environment.get("manifest_hash"),
            "session Genesis backend environment manifest hash",
        )

    @property
    def generation_session_contract_ref(
        self,
    ) -> GenerationSessionContractRef | None:
        return self._artifact.generation_session_contract_ref

    @property
    def initial_current_best(self) -> CurrentBestEvalStats:
        return self._initial_current_best

    @property
    def expected_initial_policy_state_hash(self) -> str | None:
        """Return the trusted reference initialization anchor, when present."""

        anchor = self._artifact.anchor
        if isinstance(anchor, ReferenceProgramGenesisAnchor):
            return anchor.initial_policy_state_hash
        if isinstance(anchor, BootstrapGenesisAnchor):
            return anchor.initial_policy_state_hash
        return None

    def assert_file_unchanged(self, output_dir: Path) -> None:
        expected = output_dir.resolve() / SESSION_GENESIS_FILENAME
        if self._artifact_path.resolve() != expected:
            raise ValueError("verified session genesis belongs to a different output directory")
        try:
            current_bytes = expected.read_bytes()
        except OSError as exc:
            raise ValueError("session genesis could not be read after verification") from exc
        expected_bytes = canonical_json_bytes(
            self._artifact.to_dict(),
            trailing_newline=True,
        )
        if current_bytes != expected_bytes:
            raise ValueError("session genesis bytes changed after verification")
        current = SessionGenesisArtifact.from_mapping(_read_json(expected))
        if current.to_dict() != self._artifact.to_dict():
            raise ValueError("session genesis changed after verification")

    def baseline_anchor_public(self) -> dict[str, Any] | None:
        if not isinstance(self._artifact.anchor, BaselineGenesisAnchor):
            return None
        return {
            "anchor_kind": "baseline",
            "baseline_id": self._artifact.anchor.baseline_id,
            "public_score": self._initial_current_best.score_for_comparison(),
            "task_protocol_hash": self.evaluation_binding.task_protocol_hash,
            "eval_protocol_hash": self.evaluation_binding.eval_protocol_hash,
            "acceptance_protocol_hash": (
                self.evaluation_binding.acceptance_protocol_hash
            ),
            "evaluation_binding_hash": self.evaluation_binding.binding_hash,
        }

    def reference_anchor_public(self) -> dict[str, Any] | None:
        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        manifest = anchor.reference_manifest
        return {
            "anchor_kind": "reference_program",
            "reference_id": manifest.reference_id,
            "reference_manifest_hash": manifest.manifest_hash,
            "program_source_checksum": manifest.source_checksum,
            "training_binding_hash": anchor.training_binding.binding_hash,
            "capability_profile_hash": manifest.capability_profile_hash,
            "public_score": self._initial_current_best.score_for_comparison(),
            "task_protocol_hash": self.evaluation_binding.task_protocol_hash,
            "eval_protocol_hash": self.evaluation_binding.eval_protocol_hash,
            "acceptance_protocol_hash": (
                self.evaluation_binding.acceptance_protocol_hash
            ),
            "evaluation_binding_hash": self.evaluation_binding.binding_hash,
        }

    def initial_anchor_public_evidence(self) -> dict[str, Any] | None:
        """Return the validated public evaluation that established the session head."""

        anchor = self._artifact.anchor
        if isinstance(anchor, BaselineGenesisAnchor):
            anchor_kind = "baseline"
        elif isinstance(anchor, ReferenceProgramGenesisAnchor):
            anchor_kind = "reference_program"
        else:
            return None
        return {
            "anchor_kind": anchor_kind,
            "public_result": _json_copy(anchor.public_result),
            "public_result_hash": anchor.public_result_hash,
        }

    def reference_program_source(self) -> str | None:
        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        return _reference_program_source_from_private_artifact(
            anchor=anchor,
            expected_path=(
                self._artifact_path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME
            ),
        )

    def reference_eval_plan(self) -> EvalJobSpec | None:
        """Return the already-validated reference EvalJobSpec used by Genesis."""

        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        private_payload = _read_bound_reference_private_artifact(
            anchor=anchor,
            expected_path=(
                self._artifact_path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME
            ),
        )
        return _reference_eval_spec_from_private_artifact(
            private_payload,
            anchor=anchor,
            evaluation_binding=self.evaluation_binding,
        )

    def reference_eval_phase_receipt(
        self,
        phase: EvaluationPhase | str,
    ) -> LoadedTrialEvalPhaseReceipt | None:
        """Reload one receipt from a verified phased reference evaluation."""

        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        resolved_phase = EvaluationPhase.parse(phase)
        private_payload = _read_bound_reference_private_artifact(
            anchor=anchor,
            expected_path=(
                self._artifact_path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME
            ),
        )
        eval_spec = _reference_eval_spec_from_private_artifact(
            private_payload,
            anchor=anchor,
            evaluation_binding=self.evaluation_binding,
        )
        assembly = _validate_reference_evaluation_assembly(
            private_payload,
            eval_spec=eval_spec,
            expected_phased=True,
        )
        assert assembly is not None
        binding = next(
            item
            for item in assembly.phase_receipts
            if item.phase is resolved_phase
        )
        loaded = load_trial_eval_phase_receipt(
            TrialReceiptRef(
                kind="eval_phase",
                path=binding.path,
                checksum=binding.checksum,
                size_bytes=binding.size_bytes,
                schema_version=binding.ref_schema_version,
            )
        )
        if (
            loaded.job_spec_hash != binding.job_spec_hash
            or loaded.result.job_id != binding.job_id
            or loaded.result.resource_binding_hash != binding.resource_binding_hash
        ):
            raise ValueError("reference evaluation phase receipt binding mismatch")
        return loaded


def prepare_session_genesis(
    *,
    output_dir: Path,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    runner: Runner,
    policy_initialization: PolicyInitialization,
    genesis_identity_snapshot: Mapping[str, Any],
    baseline_manifest: BaselineManifest | None = None,
    reference_program_artifact: VerifiedReferenceProgramArtifact | None = None,
    bootstrap_initial_best_score: float | None = None,
    bootstrap_initial_policy_state_hash: str | None = None,
    require_existing: bool = False,
    bind_backend_execution_environment: bool = False,
    generation_session_contract_ref: GenerationSessionContractRef | None = None,
    gpu_process_binding: GpuProcessBinding | None = None,
    reference_train_attempt_scope: PreGenesisExecutionScope | None = None,
    reference_eval_attempt_scope: PreGenesisExecutionScope | None = None,
    reference_phased_evaluation_adapter: (
        ReferencePhasedEvaluationAdapter | None
    ) = None,
    reference_eval_phase_attempt_scopes: (
        Mapping[EvaluationPhase | str, PreGenesisExecutionScope] | None
    ) = None,
    baseline_eval_attempt_scope: PreGenesisExecutionScope | None = None,
) -> VerifiedSessionGenesis:
    """Create a Genesis once, or validate the existing one without rerunning eval."""

    if not isinstance(genesis_identity_snapshot, Mapping):
        raise TypeError("genesis_identity_snapshot must be an object")
    session_id = validate_experiment_subject_id(session_id)
    if (
        baseline_manifest is not None
        and reference_program_artifact is not None
    ):
        raise ValueError("Genesis cannot carry baseline and reference manifests")
    if baseline_manifest is not None and bootstrap_initial_best_score is not None:
        raise ValueError(
            "baseline Genesis cannot also declare a bootstrap initial score"
        )
    if (
        reference_program_artifact is not None
        and bootstrap_initial_best_score is not None
    ):
        raise ValueError(
            "reference Genesis cannot also declare a bootstrap initial score"
        )
    normalized_bootstrap_score = (
        _finite_number(
            bootstrap_initial_best_score,
            "bootstrap initial_best_score",
        )
        if bootstrap_initial_best_score is not None
        else None
    )
    normalized_bootstrap_initial_policy_hash = (
        require_sha256_digest(
            bootstrap_initial_policy_state_hash,
            "bootstrap initial policy state hash",
        )
        if bootstrap_initial_policy_state_hash is not None
        else None
    )
    if (
        normalized_bootstrap_initial_policy_hash is not None
        and normalized_bootstrap_score is None
    ):
        raise ValueError(
            "bootstrap initial policy state hash requires an initial best score"
        )
    if (
        normalized_bootstrap_initial_policy_hash is not None
        and (baseline_manifest is not None or reference_program_artifact is not None)
    ):
        raise ValueError(
            "baseline/reference Genesis cannot carry a bootstrap initial policy hash"
        )
    if not isinstance(require_existing, bool):
        raise TypeError("require_existing must be boolean")
    if not isinstance(bind_backend_execution_environment, bool):
        raise TypeError("bind_backend_execution_environment must be boolean")
    if gpu_process_binding is not None and not isinstance(
        gpu_process_binding,
        GpuProcessBinding,
    ):
        raise TypeError("gpu_process_binding must be a GpuProcessBinding")
    if gpu_process_binding is not None and generation_session_contract_ref is None:
        raise ValueError(
            "GPU-bound Genesis requires a generation session contract"
        )
    phase_attempt_scopes = _normalize_reference_eval_phase_attempt_scopes(
        reference_eval_phase_attempt_scopes
    )
    phased_reference_eval = reference_phased_evaluation_adapter is not None
    if phased_reference_eval != (phase_attempt_scopes is not None):
        raise ValueError(
            "phased reference evaluation requires both an adapter and all phase scopes"
        )
    if phased_reference_eval and not callable(
        getattr(
            reference_phased_evaluation_adapter,
            "assemble_reference_full_summary",
            None,
        )
    ):
        raise TypeError("reference phased evaluation adapter is invalid")
    if phased_reference_eval and reference_eval_attempt_scope is not None:
        raise ValueError(
            "reference evaluation cannot use full and phased attempt scopes together"
        )
    for field_name, scope in (
        ("reference_train_attempt_scope", reference_train_attempt_scope),
        ("reference_eval_attempt_scope", reference_eval_attempt_scope),
        ("baseline_eval_attempt_scope", baseline_eval_attempt_scope),
    ):
        if scope is not None and not isinstance(scope, PreGenesisExecutionScope):
            raise TypeError(
                f"{field_name} must be a PreGenesisExecutionScope"
            )
    if (
        reference_train_attempt_scope is not None
        or reference_eval_attempt_scope is not None
        or phase_attempt_scopes is not None
    ) and reference_program_artifact is None:
        raise ValueError(
            "reference pre-Genesis attempt scopes require a reference artifact"
        )
    if baseline_eval_attempt_scope is not None and baseline_manifest is None:
        raise ValueError(
            "baseline pre-Genesis attempt scope requires a baseline manifest"
        )
    output_dir = output_dir.resolve()
    genesis_path = output_dir / SESSION_GENESIS_FILENAME
    if require_existing and not genesis_path.is_file():
        raise FileNotFoundError(
            f"existing session Genesis is required: {genesis_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_contract = None
    if generation_session_contract_ref is not None:
        if not isinstance(
            generation_session_contract_ref,
            GenerationSessionContractRef,
        ):
            raise TypeError(
                "generation_session_contract_ref must be a GenerationSessionContractRef"
            )
        generation_contract = load_generation_session_contract(
            output_dir,
            generation_session_contract_ref,
        )
        expected_generation_identity = {
            "session_id": session_id,
            "task_id": task_plugin.task_spec.task_id,
            "backend_id": backend_adapter.backend_id,
        }
        actual_generation_identity = {
            "session_id": generation_contract.session_id,
            "task_id": generation_contract.task_id,
            "backend_id": generation_contract.backend_id,
        }
        if actual_generation_identity != expected_generation_identity:
            raise ValueError("generation session contract identity mismatch")
        if (
            gpu_process_binding is not None
            or reference_train_attempt_scope is not None
            or reference_eval_attempt_scope is not None
            or phase_attempt_scopes is not None
            or baseline_eval_attempt_scope is not None
        ) and (
            generation_contract.schema_version
            != GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION
            or "session_cost" not in generation_contract.bound_contract_hashes
        ):
            raise ValueError(
                "GPU-bound/pre-Genesis execution requires a generation session "
                "contract v3 with session_cost"
            )
    genesis_identity_hash = session_genesis_identity_hash(
        genesis_identity_snapshot
    )
    for field_name, scope in (
        ("reference_train_attempt_scope", reference_train_attempt_scope),
        ("reference_eval_attempt_scope", reference_eval_attempt_scope),
        ("baseline_eval_attempt_scope", baseline_eval_attempt_scope),
        *(
            (
                f"reference_eval_phase_attempt_scopes[{phase.value}]",
                scope,
            )
            for phase, scope in (phase_attempt_scopes or {}).items()
        ),
    ):
        if scope is not None:
            if generation_contract is None:
                raise ValueError(
                    f"{field_name} requires a generation session contract"
                )
            _validate_pre_genesis_execution_scope(
                scope,
                field_name=field_name,
                session_id=session_id,
                genesis_identity_hash=genesis_identity_hash,
                session_contract_hash=generation_contract.contract_hash,
                budget_contract_hash=(
                    generation_contract.bound_contract_hashes["session_cost"]
                ),
                resource_binding_hash=(
                    gpu_process_binding.resource_binding_hash
                    if gpu_process_binding is not None
                    else None
                ),
            )
    pre_genesis_scopes = {
        role: scope
        for role, scope in (
            ("reference_train", reference_train_attempt_scope),
            ("reference_eval", reference_eval_attempt_scope),
            ("baseline_eval", baseline_eval_attempt_scope),
            *(
                (_reference_eval_phase_role(phase), scope)
                for phase, scope in (phase_attempt_scopes or {}).items()
            ),
        )
        if scope is not None
    }
    pre_genesis_creation_identity = (
        _pre_genesis_creation_identity(
            session_id=session_id,
            genesis_identity_hash=genesis_identity_hash,
            session_contract_hash=generation_contract.contract_hash,
            budget_contract_hash=(
                generation_contract.bound_contract_hashes["session_cost"]
            ),
            resource_binding_hash=(
                gpu_process_binding.resource_binding_hash
                if gpu_process_binding is not None
                else None
            ),
            scopes=pre_genesis_scopes,
        )
        if pre_genesis_scopes and generation_contract is not None
        else None
    )

    def resolve_context(*, allow_create: bool) -> _GenesisContext:
        if genesis_path.exists():
            existing_artifact = SessionGenesisArtifact.from_mapping(
                _read_json(genesis_path)
            )
            execution_environment = (
                existing_artifact.backend_execution_environment
            )
        elif bind_backend_execution_environment:
            execution_environment = (
                backend_adapter.resolve_session_execution_environment(
                    session_id=session_id,
                    session_dir=output_dir,
                    allow_create=allow_create,
                )
            )
        else:
            execution_environment = None
        if execution_environment is not None and not isinstance(
            execution_environment,
            Mapping,
        ):
            raise TypeError(
                "backend session execution environment must be an object or None"
            )
        evaluation_binding = backend_adapter.resolve_session_evaluation_binding(
            session_id=session_id,
            session_dir=output_dir,
            task_spec=task_plugin.task_spec,
            allow_create=allow_create,
        )
        if not isinstance(evaluation_binding, EvaluationBindingV1):
            raise TypeError(
                "backend session evaluation binding must be EvaluationBindingV1"
            )
        return _GenesisContext(
            session_id=session_id,
            task_plugin=task_plugin,
            backend_adapter=backend_adapter,
            policy_initialization=policy_initialization,
            genesis_identity_hash=genesis_identity_hash,
            baseline_manifest=baseline_manifest,
            reference_program_artifact=reference_program_artifact,
            bootstrap_initial_best_score=normalized_bootstrap_score,
            bootstrap_initial_policy_state_hash=(
                normalized_bootstrap_initial_policy_hash
            ),
            evaluation_binding=evaluation_binding,
            backend_execution_environment=(
                _json_copy(execution_environment)
                if execution_environment is not None
                else None
            ),
            generation_session_contract_ref=generation_session_contract_ref,
            gpu_process_binding=gpu_process_binding,
            reference_train_attempt_scope=reference_train_attempt_scope,
            reference_eval_attempt_scope=reference_eval_attempt_scope,
            reference_phased_evaluation_adapter=(
                reference_phased_evaluation_adapter
            ),
            reference_eval_phase_attempt_scopes=phase_attempt_scopes,
            baseline_eval_attempt_scope=baseline_eval_attempt_scope,
        )

    if genesis_path.exists():
        return _load_verified_session_genesis(
            genesis_path,
            context=resolve_context(allow_create=False),
        )

    if require_existing:
        raise FileNotFoundError(
            f"existing session Genesis is required: {genesis_path}"
        )

    try:
        creation_lease = _acquire_session_genesis_creation_lease(
            output_dir=output_dir,
            session_id=session_id,
            pre_genesis_identity=pre_genesis_creation_identity,
        )
    except SessionGenesisCreationInProgressError:
        if genesis_path.exists():
            return _load_verified_session_genesis(
                genesis_path,
                context=resolve_context(allow_create=False),
            )
        raise

    try:
        if genesis_path.exists():
            return _load_verified_session_genesis(
                genesis_path,
                context=resolve_context(allow_create=False),
            )
        allow_pre_genesis_state = False
        if pre_genesis_creation_identity is not None:
            allow_pre_genesis_state = _recover_pre_genesis_execution(
                output_dir=output_dir,
                identity=pre_genesis_creation_identity,
                scopes=pre_genesis_scopes,
                previous_lease_payload=creation_lease.previous_payload,
            )
            creation_lease.activate()
        _reject_partial_session_without_genesis(
            output_dir,
            session_id=session_id,
            allow_pre_genesis_state=allow_pre_genesis_state,
        )
        context = resolve_context(allow_create=True)
        if context.genesis_anchor_kind == "reference_program":
            artifact = _create_reference_genesis(
                output_dir=output_dir,
                context=context,
                runner=runner,
            )
        elif context.genesis_anchor_kind == "policy_baseline":
            artifact = _create_baseline_genesis(
                output_dir=output_dir,
                context=context,
                runner=runner,
            )
        else:
            artifact = _create_bootstrap_genesis(
                context=context,
            )
        creation_lease.assert_owned()
        _atomic_create_json(genesis_path, artifact.to_dict())
        return _load_verified_session_genesis(genesis_path, context=context)
    finally:
        creation_lease.release()


def verify_session_genesis_for_runtime(
    genesis: VerifiedSessionGenesis,
    *,
    output_dir: Path,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    policy_initialization: PolicyInitialization,
) -> None:
    """Cheap live-loop guard after contextual preparation has succeeded."""

    if not isinstance(genesis, VerifiedSessionGenesis):
        raise TypeError("run requires a VerifiedSessionGenesis")
    artifact = genesis.artifact
    if artifact.session_id != session_id:
        raise ValueError("session Genesis session_id mismatch")
    if artifact.task_id != task_plugin.task_spec.task_id:
        raise ValueError("session Genesis task_id mismatch")
    if artifact.backend_id != backend_adapter.backend_id:
        raise ValueError("session Genesis backend_id mismatch")
    if artifact.policy_initialization_kind != policy_initialization.kind:
        raise ValueError("session Genesis policy initialization mismatch")
    expected_anchor_kind = task_plugin.task_spec.genesis_anchor_kind_for(
        policy_initialization.kind
    )
    if artifact.genesis_kind != expected_anchor_kind:
        raise ValueError("session Genesis anchor policy mismatch")
    binding = backend_adapter.resolve_session_evaluation_binding(
        session_id=session_id,
        session_dir=output_dir.resolve(),
        task_spec=task_plugin.task_spec,
        allow_create=False,
    )
    if not isinstance(binding, EvaluationBindingV1):
        raise TypeError(
            "backend session evaluation binding must be EvaluationBindingV1"
        )
    backend_adapter.validate_session_evaluation_binding(
        task_spec=task_plugin.task_spec,
        evaluation_binding=binding,
    )
    if artifact.evaluation_binding != binding:
        raise ValueError("session Genesis evaluation binding mismatch")
    execution_environment = artifact.backend_execution_environment
    backend_adapter.validate_runtime_session_execution_environment(
        execution_environment=execution_environment,
    )
    generation_ref = artifact.generation_session_contract_ref
    if generation_ref is not None:
        contract = load_generation_session_contract(output_dir, generation_ref)
        if {
            "session_id": contract.session_id,
            "task_id": contract.task_id,
            "backend_id": contract.backend_id,
        } != {
            "session_id": session_id,
            "task_id": task_plugin.task_spec.task_id,
            "backend_id": backend_adapter.backend_id,
        }:
            raise ValueError("runtime generation session contract identity mismatch")
    _assert_policy_initialization_matches_anchor(policy_initialization, artifact.anchor)
    genesis.assert_file_unchanged(output_dir)


@dataclass(frozen=True)
class _GenesisContext:
    session_id: str
    task_plugin: TaskPlugin
    backend_adapter: BackendAdapter
    policy_initialization: PolicyInitialization
    genesis_identity_hash: str
    baseline_manifest: BaselineManifest | None
    reference_program_artifact: VerifiedReferenceProgramArtifact | None
    bootstrap_initial_best_score: float | None
    bootstrap_initial_policy_state_hash: str | None
    evaluation_binding: EvaluationBindingV1
    backend_execution_environment: Mapping[str, Any] | None
    generation_session_contract_ref: GenerationSessionContractRef | None
    gpu_process_binding: GpuProcessBinding | None
    reference_train_attempt_scope: PreGenesisExecutionScope | None
    reference_eval_attempt_scope: PreGenesisExecutionScope | None
    reference_phased_evaluation_adapter: (
        ReferencePhasedEvaluationAdapter | None
    )
    reference_eval_phase_attempt_scopes: (
        Mapping[EvaluationPhase, PreGenesisExecutionScope] | None
    )
    baseline_eval_attempt_scope: PreGenesisExecutionScope | None

    def __post_init__(self) -> None:
        if self.gpu_process_binding is not None and not isinstance(
            self.gpu_process_binding,
            GpuProcessBinding,
        ):
            raise TypeError("Genesis GPU process binding is invalid")
        for field_name in (
            "reference_train_attempt_scope",
            "reference_eval_attempt_scope",
            "baseline_eval_attempt_scope",
        ):
            scope = getattr(self, field_name)
            if scope is not None and not isinstance(
                scope,
                PreGenesisExecutionScope,
            ):
                raise TypeError(f"Genesis {field_name} is invalid")
        phased_reference_eval = self.reference_phased_evaluation_adapter is not None
        phase_scopes = self.reference_eval_phase_attempt_scopes
        if phased_reference_eval != (phase_scopes is not None):
            raise ValueError("Genesis phased reference evaluation is incomplete")
        if phased_reference_eval:
            if self.reference_eval_attempt_scope is not None:
                raise ValueError("Genesis reference eval scopes are ambiguous")
            if set(phase_scopes or {}) != set(EvaluationPhase):
                raise ValueError("Genesis reference eval phase scopes are incomplete")
        if (
            self.gpu_process_binding is not None
            and self.generation_session_contract_ref is None
        ):
            raise ValueError(
                "Genesis GPU process binding requires a generation contract"
            )
        binding = self.evaluation_binding
        if binding.task_protocol_hash != self.task_spec.resolved_task_protocol_hash():
            raise ValueError("Genesis binding task protocol mismatch")
        if binding.acceptance_protocol_hash != self.acceptance_protocol.protocol_hash:
            raise ValueError("Genesis binding acceptance protocol mismatch")
        self.backend_adapter.validate_session_evaluation_binding(
            task_spec=self.task_spec,
            evaluation_binding=binding,
        )
        self.backend_adapter.validate_session_execution_environment(
            execution_environment=self.backend_execution_environment,
        )
        if self.genesis_anchor_kind == "bootstrap":
            if self.baseline_manifest is not None or self.reference_program_artifact is not None:
                raise ValueError("bootstrap Genesis cannot carry a manifest")
            if self.bootstrap_initial_policy_state_hash is not None:
                require_sha256_digest(
                    self.bootstrap_initial_policy_state_hash,
                    "bootstrap initial policy state hash",
                )
        elif self.genesis_anchor_kind == "policy_baseline":
            if self.baseline_manifest is None or self.reference_program_artifact is not None:
                raise ValueError("policy-baseline Genesis requires only a baseline manifest")
        elif self.genesis_anchor_kind == "reference_program":
            if self.reference_program_artifact is None or self.baseline_manifest is not None:
                raise ValueError("reference Genesis requires only a reference artifact")
            manifest = self.reference_program_artifact.manifest
            if manifest.task_id != self.task_spec.task_id:
                raise ValueError("reference Genesis manifest task_id mismatch")
            if (
                manifest.capability_profile_hash
                != self.task_spec.candidate_capability_profile.profile_hash
            ):
                raise ValueError("reference Genesis manifest capability profile mismatch")
        else:
            raise ValueError("Genesis anchor policy is invalid")

    @property
    def task_spec(self) -> TaskSpec:
        return self.task_plugin.task_spec

    @property
    def acceptance_protocol(self) -> AcceptanceProtocol:
        protocol = self.task_spec.acceptance_protocol
        if protocol is None:
            raise ValueError("session Genesis requires a fixed acceptance protocol")
        return protocol

    @property
    def genesis_anchor_kind(self) -> str:
        return self.task_spec.genesis_anchor_kind_for(
            self.policy_initialization.kind
        )

    @property
    def reference_program_manifest(self) -> ReferenceProgramManifest | None:
        if self.reference_program_artifact is None:
            return None
        return self.reference_program_artifact.manifest

def _create_bootstrap_genesis(
    *,
    context: _GenesisContext,
) -> SessionGenesisArtifact:
    if context.genesis_anchor_kind != "bootstrap":
        raise ValueError("task policy does not allow a bootstrap Genesis")
    if context.policy_initialization.kind != "random":
        raise ValueError("bootstrap Genesis requires random policy initialization")
    if context.bootstrap_initial_best_score is None:
        raise ValueError("bootstrap Genesis requires initial_best_score")
    return SessionGenesisArtifact(
        session_id=context.session_id,
        task_id=context.task_spec.task_id,
        backend_id=context.backend_adapter.backend_id,
        policy_initialization_kind=context.policy_initialization.kind,
        runtime_snapshot_hash=context.genesis_identity_hash,
        evaluation_binding=context.evaluation_binding,
        backend_execution_environment=context.backend_execution_environment,
        anchor=BootstrapGenesisAnchor(
            anchor_id=context.session_id,
            initial_best_score=context.bootstrap_initial_best_score,
            initial_policy_state_hash=(
                context.bootstrap_initial_policy_state_hash
            ),
        ),
        generation_session_contract_ref=context.generation_session_contract_ref,
        schema_version=_context_session_genesis_schema_version(context),
    )


def _create_baseline_genesis(
    *,
    output_dir: Path,
    context: _GenesisContext,
    runner: Runner,
) -> SessionGenesisArtifact:
    manifest = context.baseline_manifest
    if manifest is None:
        raise ValueError("baseline Genesis requires a baseline manifest")
    eval_spec = _materialize_baseline_eval(context, manifest)
    public_result = runner.run_baseline_eval(eval_spec)
    seal = _build_baseline_seal(
        eval_spec=eval_spec,
        task_spec=context.task_spec,
        public_result=public_result,
    )
    current_best = CurrentBestEvalStats.from_baseline_public_result_mapping(
        baseline_id=manifest.baseline_id,
        baseline_manifest_hash=str(manifest.manifest_hash),
        checkpoint_hash=manifest.checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        acceptance_protocol=context.acceptance_protocol,
        expected_evaluation_binding=context.evaluation_binding,
        capability_profile_hash=(
            context.task_spec.candidate_capability_profile.profile_hash
        ),
    )
    if not current_best.has_full_eval_stats():
        raise ValueError("baseline evaluation cannot establish a full current-best")

    private_payload = _baseline_eval_artifact_payload(
        session_id=context.session_id,
        backend_id=context.backend_adapter.backend_id,
        eval_spec=eval_spec,
        public_result=public_result.to_public_dict(),
        trusted_eval_seal=seal.to_public_dict(),
    )
    private_path = output_dir / BASELINE_EVAL_ARTIFACT_FILENAME
    _atomic_create_json(private_path, private_payload)
    private_checksum = file_sha256(private_path)
    if not is_sha256_checksum(private_checksum):
        raise ValueError("baseline eval artifact checksum could not be computed")
    anchor = BaselineGenesisAnchor(
        baseline_id=manifest.baseline_id,
        baseline_manifest_hash=str(manifest.manifest_hash),
        checkpoint_hash=manifest.checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        public_result_hash=canonical_json_hash(public_result.to_public_dict()),
        trusted_eval_seal=seal.to_public_dict(),
        private_eval_artifact=PrivateArtifactRef(
            kind="baseline_eval_artifact",
            uri=str(private_path),
            checksum=private_checksum,
            reason="session_genesis_baseline_evidence",
        ),
    )
    return SessionGenesisArtifact(
        session_id=context.session_id,
        task_id=context.task_spec.task_id,
        backend_id=context.backend_adapter.backend_id,
        policy_initialization_kind=context.policy_initialization.kind,
        runtime_snapshot_hash=context.genesis_identity_hash,
        evaluation_binding=context.evaluation_binding,
        backend_execution_environment=context.backend_execution_environment,
        anchor=anchor,
        generation_session_contract_ref=context.generation_session_contract_ref,
        schema_version=_context_session_genesis_schema_version(context),
    )


def _create_reference_genesis(
    *,
    output_dir: Path,
    context: _GenesisContext,
    runner: Runner,
) -> SessionGenesisArtifact:
    reference = context.reference_program_artifact
    if reference is None:
        raise ValueError("reference Genesis requires a verified reference artifact")
    materialization_kwargs: dict[str, Any] = {}
    environment_hash = _context_backend_execution_environment_manifest_hash(
        context
    )
    if environment_hash is not None:
        materialization_kwargs[
            "backend_execution_environment_manifest_hash"
        ] = environment_hash
    if context.gpu_process_binding is not None:
        if (
            context.reference_train_attempt_scope is None
            or (
                context.reference_eval_attempt_scope is None
                and context.reference_eval_phase_attempt_scopes is None
            )
        ):
            raise ValueError(
                "GPU-bound reference Genesis requires train and eval attempt scopes"
            )
        materialization_kwargs["gpu_process_binding"] = (
            context.gpu_process_binding
        )
    if context.reference_train_attempt_scope is not None:
        materialization_kwargs["train_attempt_scope"] = (
            context.reference_train_attempt_scope
        )
    if context.reference_eval_attempt_scope is not None:
        materialization_kwargs["eval_attempt_scope"] = (
            context.reference_eval_attempt_scope
        )
    jobs = context.backend_adapter.materialize_reference_jobs(
        session_id=context.session_id,
        task_spec=context.task_spec,
        reference=reference,
        artifact_root=output_dir / "reference_program_artifacts",
        policy_initialization=context.policy_initialization,
        evaluation_binding=context.evaluation_binding,
        **materialization_kwargs,
    )
    _validate_reference_job_bundle(context=context, jobs=jobs)
    train_result = runner.run_train(jobs.train_spec)
    _validate_reference_train_result(jobs.train_spec, train_result)
    initial_policy_state_hash = _reference_initial_policy_state_hash(train_result)
    checkpoint_path = _reference_checkpoint_path(train_result)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != train_result.checkpoint.checksum:
        raise ValueError("reference train checkpoint bytes changed after Runner sealing")

    eval_spec = bind_eval_spec_to_checkpoint(
        jobs.eval_spec,
        train_result.checkpoint,
    )
    evaluation_assembly: EvaluationAssembly | None = None
    if context.reference_phased_evaluation_adapter is None:
        raw_public_result = runner.run_eval(eval_spec)
    else:
        evaluation_assembly, raw_public_result = (
            _run_reference_phased_evaluation(
                output_dir=output_dir,
                context=context,
                runner=runner,
                eval_plan=eval_spec,
            )
        )
    qualification = context.task_plugin.qualify_reference_result(
        public_result=raw_public_result,
        train_summary=train_result.summary,
    )
    if not isinstance(qualification, ReferenceQualificationResult):
        raise TypeError("task reference qualification result is invalid")
    if (
        qualification.gate["protocol_hash"]
        != context.evaluation_binding.task_protocol_hash
    ):
        raise ValueError("reference qualification task protocol mismatch")
    if not qualification.passed:
        raise ValueError("reference program failed the task-owned capability gate")
    public_result = qualification.public_result
    seal = _build_reference_seal(
        eval_spec=eval_spec,
        task_spec=context.task_spec,
        public_result=public_result,
    )
    current_best = CurrentBestEvalStats.from_reference_public_result_mapping(
        reference_id=reference.manifest.reference_id,
        reference_manifest_hash=str(reference.manifest.manifest_hash),
        program_source_checksum=reference.manifest.source_checksum,
        training_binding_hash=str(jobs.train_spec.training_binding.binding_hash),
        capability_profile_hash=reference.manifest.capability_profile_hash,
        checkpoint_hash=train_result.checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        acceptance_protocol=context.acceptance_protocol,
        expected_evaluation_binding=context.evaluation_binding,
    )
    if not current_best.has_full_eval_stats():
        raise ValueError("reference evaluation cannot establish a full current-best")

    private_payload = _reference_eval_artifact_payload(
        session_id=context.session_id,
        backend_id=context.backend_adapter.backend_id,
        reference=reference,
        reference_artifact=jobs.train_spec.reference_artifact,
        train_spec=jobs.train_spec,
        train_result=train_result,
        eval_spec=eval_spec,
        public_result=public_result,
        trusted_eval_seal=seal,
        qualification=qualification,
        evaluation_assembly=evaluation_assembly,
    )
    private_path = output_dir / REFERENCE_EVAL_ARTIFACT_FILENAME
    _atomic_create_json(private_path, private_payload)
    private_checksum = file_sha256(private_path)
    if not is_sha256_checksum(private_checksum):
        raise ValueError("reference eval artifact checksum could not be computed")
    checkpoint_artifact = PrivateArtifactRef(
        kind="reference_checkpoint",
        uri=str(checkpoint_path),
        checksum=train_result.checkpoint.checksum,
        reason="session_genesis_reference_checkpoint",
    )
    anchor = ReferenceProgramGenesisAnchor(
        reference_manifest=reference.manifest,
        training_binding=jobs.train_spec.training_binding,
        initial_policy_state_hash=initial_policy_state_hash,
        checkpoint_hash=train_result.checkpoint.checksum,
        checkpoint_artifact=checkpoint_artifact,
        public_result=public_result.to_public_dict(),
        public_result_hash=canonical_json_hash(public_result.to_public_dict()),
        trusted_eval_seal=seal.to_public_dict(),
        qualification_gate=dict(qualification.gate),
        private_eval_artifact=PrivateArtifactRef(
            kind="reference_eval_artifact",
            uri=str(private_path),
            checksum=private_checksum,
            reason="session_genesis_reference_evidence",
        ),
    )
    return SessionGenesisArtifact(
        session_id=context.session_id,
        task_id=context.task_spec.task_id,
        backend_id=context.backend_adapter.backend_id,
        policy_initialization_kind=context.policy_initialization.kind,
        runtime_snapshot_hash=context.genesis_identity_hash,
        evaluation_binding=context.evaluation_binding,
        backend_execution_environment=context.backend_execution_environment,
        anchor=anchor,
        generation_session_contract_ref=context.generation_session_contract_ref,
        schema_version=_context_session_genesis_schema_version(context),
    )


def _run_reference_phased_evaluation(
    *,
    output_dir: Path,
    context: _GenesisContext,
    runner: Runner,
    eval_plan: EvalJobSpec,
) -> tuple[EvaluationAssembly, PublicEvalResult]:
    adapter = context.reference_phased_evaluation_adapter
    phase_scopes = context.reference_eval_phase_attempt_scopes
    if adapter is None or phase_scopes is None:
        raise ValueError("reference phased evaluation context is incomplete")
    manifest = context.reference_program_manifest
    if manifest is None:
        raise ValueError("reference phased evaluation has no manifest")
    eval_plan_hash = canonical_json_hash(eval_plan.to_public_dict())
    receipt_commitment = {
        "schema_version": (
            "true_rl_agent.session_genesis.reference_phase_receipts.v1"
        ),
        "session_id": context.session_id,
        "genesis_identity_hash": context.genesis_identity_hash,
        "reference_manifest_hash": manifest.manifest_hash,
        "evaluation_plan_hash": eval_plan_hash,
    }
    journal = TrialReceiptJournal(
        output_dir / "reference_program_artifacts" / "evaluation_receipts",
        contract={
            "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "round_id": context.session_id,
            "session_genesis_hash": context.genesis_identity_hash,
            "ledger_epoch_hash": canonical_json_hash(receipt_commitment),
            "purpose": "session_genesis_reference_phased_evaluation",
            "commitment": receipt_commitment,
        },
    )
    journal.bind_contract()
    phase_results: list[EvaluationPhaseResult] = []
    phase_receipt_refs: dict[EvaluationPhase, Mapping[str, Any]] = {}
    for phase in EvaluationPhase:
        scope = phase_scopes[phase]
        phase_spec = context.backend_adapter.materialize_eval_phase(
            eval_plan=eval_plan,
            phase=phase,
            attempt_scope=scope,
        )
        if (
            phase_spec.evaluation_plan != eval_plan
            or phase_spec.phase is not phase
            or phase_spec.attempt_scope != scope
            or phase_spec.gpu_process_binding != context.gpu_process_binding
        ):
            raise ValueError("reference eval phase job identity mismatch")
        loaded = journal.load_eval_phase(phase_spec)
        if loaded is None:
            phase_result = runner.run_eval_phase(phase_spec)
            receipt_ref = journal.publish_eval_phase(phase_spec, phase_result)
        else:
            phase_result, receipt_ref = loaded
        phase_results.append(phase_result)
        phase_receipt_refs[phase] = receipt_ref.to_dict()
    assembly = adapter.assemble_reference_full_summary(
        eval_plan=eval_plan,
        phase_results=phase_results,
        phase_receipt_refs=phase_receipt_refs,
    )
    if not isinstance(assembly, EvaluationAssembly):
        raise TypeError("reference phased evaluation assembly is invalid")
    raw_result = runner.project_assembled_eval_summary(eval_plan, assembly)
    if not isinstance(raw_result, PublicEvalResult):
        raise TypeError("reference phased evaluation projection is invalid")
    return assembly, raw_result


def _context_session_genesis_schema_version(context: _GenesisContext) -> str:
    return (
        GENERATION_SESSION_GENESIS_SCHEMA_VERSION
        if context.generation_session_contract_ref is not None
        else SESSION_GENESIS_SCHEMA_VERSION
    )


def _validate_reference_job_bundle(*, context: _GenesisContext, jobs: Any) -> None:
    train_spec = getattr(jobs, "train_spec", None)
    eval_spec = getattr(jobs, "eval_spec", None)
    if not isinstance(train_spec, TrainJobSpec) or not isinstance(eval_spec, EvalJobSpec):
        raise TypeError("reference backend returned an invalid job bundle")
    manifest = context.reference_program_manifest
    if manifest is None:
        raise ValueError("reference job bundle has no expected manifest")
    expected = {
        "task_id": context.task_spec.task_id,
        "subject": ExperimentSubject.reference(manifest.reference_id),
        "subject_manifest_hash": manifest.manifest_hash,
        "program_source_checksum": manifest.source_checksum,
        "capability_profile_hash": manifest.capability_profile_hash,
    }
    actual_train = {
        "task_id": train_spec.task_id,
        "subject": train_spec.subject,
        "subject_manifest_hash": train_spec.subject_manifest_hash,
        "program_source_checksum": train_spec.program_source_checksum,
        "capability_profile_hash": train_spec.training_binding.capability_profile_hash,
    }
    actual_eval = {
        "task_id": eval_spec.task_id,
        "subject": eval_spec.subject,
        "subject_manifest_hash": eval_spec.subject_manifest_hash,
        "program_source_checksum": eval_spec.program_source_checksum,
        "capability_profile_hash": eval_spec.capability_profile_hash,
    }
    if actual_train != expected or actual_eval != expected:
        raise ValueError("reference backend job identity chain mismatch")
    if train_spec.policy_initialization != context.policy_initialization:
        raise ValueError("reference train policy initialization mismatch")
    if eval_spec.training_binding_hash != train_spec.training_binding.binding_hash:
        raise ValueError("reference eval TrainingBinding mismatch")
    if eval_spec.evaluation_binding != context.evaluation_binding:
        raise ValueError("reference eval binding mismatch")
    if (
        train_spec.gpu_process_binding != context.gpu_process_binding
        or eval_spec.gpu_process_binding != context.gpu_process_binding
    ):
        raise ValueError("reference job GPU process binding mismatch")
    if (
        train_spec.attempt_scope != context.reference_train_attempt_scope
        or eval_spec.attempt_scope != context.reference_eval_attempt_scope
    ):
        raise ValueError("reference job attempt scope mismatch")
    if jobs.task_invariants is not None:
        context.task_spec.invariants.assert_compatible_with(jobs.task_invariants)


def _validate_reference_train_result(
    train_spec: TrainJobSpec,
    train_result: TrainJobResult,
) -> None:
    if not isinstance(train_result, TrainJobResult):
        raise TypeError("reference Runner returned an invalid train result")
    expected = {
        "job_id": train_spec.job_id,
        "subject": train_spec.subject,
        "subject_manifest_hash": train_spec.subject_manifest_hash,
        "program_source_checksum": train_spec.program_source_checksum,
        "training_binding_hash": train_spec.training_binding.binding_hash,
        "capability_profile_hash": train_spec.training_binding.capability_profile_hash,
        "execution_device": train_spec.execution_device,
        "gpu_process_binding": train_spec.gpu_process_binding,
    }
    actual = {
        "job_id": train_result.job_id,
        "subject": train_result.subject,
        "subject_manifest_hash": train_result.subject_manifest_hash,
        "program_source_checksum": train_result.program_source_checksum,
        "training_binding_hash": train_result.training_binding_hash,
        "capability_profile_hash": train_result.capability_profile_hash,
        "execution_device": train_result.execution_device,
        "gpu_process_binding": train_result.gpu_process_binding,
    }
    if actual != expected:
        raise ValueError("reference train result identity chain mismatch")
    if train_result.summary.get("formal_training_executed") is not True:
        raise ValueError("reference train did not execute formal training")


def _reference_initial_policy_state_hash(train_result: TrainJobResult) -> str:
    training = _json_object(
        train_result.summary.get("training"),
        "reference train summary training",
    )
    return require_sha256_digest(
        training.get("initial_policy_state_hash"),
        "reference initial policy state hash",
    )


def _reference_checkpoint_path(train_result: TrainJobResult) -> Path:
    raw_path = train_result.checkpoint.metadata.get("local_path")
    if not isinstance(raw_path, str) or not raw_path or "://" in raw_path:
        raise ValueError("reference checkpoint has no local byte path")
    unresolved = Path(raw_path)
    if unresolved.is_symlink():
        raise ValueError("reference checkpoint path must not be a symlink")
    path = unresolved.resolve(strict=True)
    if not path.is_file():
        raise ValueError("reference checkpoint path is not a file")
    return path


def _load_verified_session_genesis(
    path: Path,
    *,
    context: _GenesisContext,
) -> VerifiedSessionGenesis:
    artifact = SessionGenesisArtifact.from_mapping(_read_json(path))
    expected_schema = _context_session_genesis_schema_version(context)
    if artifact.schema_version != expected_schema:
        if artifact.schema_version == LEGACY_SESSION_GENESIS_SCHEMA_VERSION:
            raise ValueError(
                "legacy session Genesis is historical-only; a new formal session "
                "must freeze backend execution environment identity"
            )
        raise ValueError(
            "session Genesis capability/schema does not match the requested session"
        )
    if artifact.session_id != context.session_id:
        raise ValueError("session Genesis session_id drift")
    if artifact.task_id != context.task_spec.task_id:
        raise ValueError("session Genesis task_id drift")
    if artifact.backend_id != context.backend_adapter.backend_id:
        raise ValueError("session Genesis backend_id drift")
    if artifact.policy_initialization_kind != context.policy_initialization.kind:
        raise ValueError("session Genesis policy initialization drift")
    if artifact.runtime_snapshot_hash != context.genesis_identity_hash:
        raise ValueError("session Genesis identity drift")
    if artifact.evaluation_binding != context.evaluation_binding:
        raise ValueError("session Genesis evaluation binding drift")
    if _json_copy(artifact.backend_execution_environment) != _json_copy(
        context.backend_execution_environment
    ):
        raise ValueError("session Genesis backend execution environment drift")
    if artifact.generation_session_contract_ref != (
        context.generation_session_contract_ref
    ):
        raise ValueError("session Genesis generation contract ref drift")
    if artifact.genesis_kind != context.genesis_anchor_kind:
        raise ValueError("session Genesis anchor policy drift")
    _assert_policy_initialization_matches_anchor(
        context.policy_initialization,
        artifact.anchor,
    )

    if isinstance(artifact.anchor, BootstrapGenesisAnchor):
        if context.baseline_manifest is not None:
            raise ValueError("bootstrap Genesis cannot replace a baseline manifest")
        if context.genesis_anchor_kind != "bootstrap":
            raise ValueError("task policy does not allow a bootstrap Genesis")
        if context.bootstrap_initial_best_score is None:
            raise ValueError("bootstrap Genesis requires initial_best_score")
        if (
            artifact.anchor.initial_best_score
            != context.bootstrap_initial_best_score
        ):
            raise ValueError("session Genesis bootstrap initial score drift")
        if (
            artifact.anchor.initial_policy_state_hash
            != context.bootstrap_initial_policy_state_hash
        ):
            raise ValueError("session Genesis bootstrap initial policy hash drift")
        initial_current_best = CurrentBestEvalStats.bootstrap(
            initial_best_score=artifact.anchor.initial_best_score,
            acceptance_protocol=context.acceptance_protocol,
            evaluation_binding=artifact.evaluation_binding,
            anchor_id=artifact.anchor.anchor_id,
        )
    elif isinstance(artifact.anchor, BaselineGenesisAnchor):
        manifest = context.baseline_manifest
        if manifest is None:
            raise ValueError("baseline Genesis requires the expected baseline manifest")
        _assert_manifest_matches_anchor(manifest, artifact.anchor)
        eval_spec = _materialize_baseline_eval(context, manifest)
        public_result = PublicEvalResult(
            status=artifact.anchor.public_result.get("status"),
            public_score=artifact.anchor.public_result.get("public_score"),
            metrics=_json_object(
                artifact.anchor.public_result.get("metrics"),
                "baseline Genesis public result metrics",
            ),
        )
        expected_seal = _build_baseline_seal(
            eval_spec=eval_spec,
            task_spec=context.task_spec,
            public_result=public_result,
        )
        if artifact.anchor.trusted_eval_seal != expected_seal.to_public_dict():
            raise ValueError("baseline Genesis trusted eval seal mismatch")
        _validate_private_baseline_artifact(
            anchor=artifact.anchor,
            expected_path=path.parent / BASELINE_EVAL_ARTIFACT_FILENAME,
            expected_payload=_baseline_eval_artifact_payload(
                session_id=context.session_id,
                backend_id=context.backend_adapter.backend_id,
                eval_spec=eval_spec,
                public_result=public_result.to_public_dict(),
                trusted_eval_seal=expected_seal.to_public_dict(),
            ),
        )
        initial_current_best = CurrentBestEvalStats.from_baseline_public_result_mapping(
            baseline_id=manifest.baseline_id,
            baseline_manifest_hash=str(manifest.manifest_hash),
            checkpoint_hash=manifest.checkpoint.checksum,
            public_result=public_result.to_public_dict(),
            acceptance_protocol=context.acceptance_protocol,
            expected_evaluation_binding=context.evaluation_binding,
            capability_profile_hash=(
                context.task_spec.candidate_capability_profile.profile_hash
            ),
        )
        if not initial_current_best.has_full_eval_stats():
            raise ValueError("baseline Genesis current-best is incomplete")
    else:
        manifest = context.reference_program_manifest
        if manifest is None:
            raise ValueError(
                "reference Genesis requires the expected reference manifest"
            )
        anchor = artifact.anchor
        if anchor.reference_manifest != manifest:
            raise ValueError("session Genesis reference manifest drift")
        if (
            anchor.training_binding.capability_profile_hash
            != context.task_spec.candidate_capability_profile.profile_hash
        ):
            raise ValueError("session Genesis reference capability profile drift")
        public_result = PublicEvalResult(
            status=anchor.public_result.get("status"),
            public_score=anchor.public_result.get("public_score"),
            metrics=_json_object(
                anchor.public_result.get("metrics"),
                "reference Genesis public result metrics",
            ),
        )
        _validate_reference_public_result(
            anchor=anchor,
            public_result=public_result,
            evaluation_binding=context.evaluation_binding,
        )
        private_payload = _validate_private_reference_artifact(
            anchor=anchor,
            expected_path=path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME,
            context=context,
        )
        eval_spec = _reference_eval_spec_from_private_artifact(
            private_payload,
            anchor=anchor,
            evaluation_binding=context.evaluation_binding,
        )
        _validate_reference_evaluation_assembly(
            private_payload,
            eval_spec=eval_spec,
            expected_phased=(
                context.reference_phased_evaluation_adapter is not None
            ),
        )
        if eval_spec.gpu_process_binding != context.gpu_process_binding:
            raise ValueError("reference private eval GPU process binding mismatch")
        if (
            private_payload["train_result"].get("execution_device")
            != eval_spec.execution_device
        ):
            raise ValueError("reference Genesis train/eval execution device mismatch")
        if (
            public_result.metrics.get("execution_device")
            != eval_spec.execution_device
        ):
            raise ValueError("reference Genesis public execution device mismatch")
        expected_seal = _build_reference_seal(
            eval_spec=eval_spec,
            task_spec=context.task_spec,
            public_result=public_result,
        )
        if anchor.trusted_eval_seal != expected_seal.to_public_dict():
            raise ValueError("reference Genesis trusted eval seal mismatch")
        initial_current_best = CurrentBestEvalStats.from_reference_public_result_mapping(
            reference_id=manifest.reference_id,
            reference_manifest_hash=str(manifest.manifest_hash),
            program_source_checksum=manifest.source_checksum,
            training_binding_hash=str(anchor.training_binding.binding_hash),
            capability_profile_hash=manifest.capability_profile_hash,
            checkpoint_hash=anchor.checkpoint_hash,
            public_result=public_result.to_public_dict(),
            acceptance_protocol=context.acceptance_protocol,
            expected_evaluation_binding=context.evaluation_binding,
        )
        if not initial_current_best.has_full_eval_stats():
            raise ValueError("reference Genesis current-best is incomplete")
    return VerifiedSessionGenesis._validated(
        artifact=artifact,
        artifact_path=path.resolve(),
        initial_current_best=initial_current_best,
    )


def _materialize_baseline_eval(
    context: _GenesisContext,
    manifest: BaselineManifest,
) -> BaselineEvalJobSpec:
    materialization_kwargs: dict[str, Any] = {}
    environment_hash = _context_backend_execution_environment_manifest_hash(
        context
    )
    if environment_hash is not None:
        materialization_kwargs[
            "backend_execution_environment_manifest_hash"
        ] = environment_hash
    if context.gpu_process_binding is not None:
        if context.baseline_eval_attempt_scope is None:
            raise ValueError(
                "GPU-bound baseline Genesis requires an eval attempt scope"
            )
        materialization_kwargs["gpu_process_binding"] = (
            context.gpu_process_binding
        )
    if context.baseline_eval_attempt_scope is not None:
        materialization_kwargs["eval_attempt_scope"] = (
            context.baseline_eval_attempt_scope
        )
    spec = context.backend_adapter.materialize_baseline_eval(
        session_id=context.session_id,
        task_spec=context.task_spec,
        baseline_manifest=manifest,
        evaluation_binding=context.evaluation_binding,
        **materialization_kwargs,
    )
    if spec.evaluation_binding != context.evaluation_binding:
        raise ValueError("baseline eval spec evaluation binding drift")
    if spec.checkpoint != manifest.checkpoint:
        raise ValueError("baseline eval spec checkpoint drift")
    if spec.gpu_process_binding != context.gpu_process_binding:
        raise ValueError("baseline eval spec GPU process binding drift")
    if spec.attempt_scope != context.baseline_eval_attempt_scope:
        raise ValueError("baseline eval spec attempt scope drift")
    return spec


def _context_backend_execution_environment_manifest_hash(
    context: _GenesisContext,
) -> str | None:
    environment = context.backend_execution_environment
    if environment is None:
        return None
    return require_sha256_digest(
        environment.get("manifest_hash"),
        "Genesis backend execution environment manifest hash",
    )


def _validate_pre_genesis_execution_scope(
    scope: PreGenesisExecutionScope,
    *,
    field_name: str,
    session_id: str,
    genesis_identity_hash: str,
    session_contract_hash: str,
    budget_contract_hash: str,
    resource_binding_hash: str | None,
) -> None:
    expected = {
        "session_id": session_id,
        "genesis_identity_hash": genesis_identity_hash,
        "session_contract_hash": session_contract_hash,
        "budget_contract_hash": budget_contract_hash,
        "resource_binding_hash": resource_binding_hash,
    }
    actual = {
        "session_id": scope.session_id,
        "genesis_identity_hash": scope.genesis_identity_hash,
        "session_contract_hash": scope.session_contract_hash,
        "budget_contract_hash": scope.budget_contract_hash,
        "resource_binding_hash": scope.resource_binding_hash,
    }
    if actual != expected:
        raise ValueError(f"{field_name} identity mismatch")


def _build_baseline_seal(
    *,
    eval_spec: BaselineEvalJobSpec,
    task_spec: TaskSpec,
    public_result: PublicEvalResult,
) -> TrustedEvalSeal:
    metrics = public_result.metrics
    sealed_gate = _json_object(
        metrics.get("sealed_validation_gate"),
        "baseline sealed_validation_gate",
    )
    return build_trusted_eval_seal(
        eval_spec=eval_spec,
        task_invariants=task_spec.invariants,
        checkpoint_safety_report=metrics.get("checkpoint_safety_report"),
        candidate_isolation_status=_nonempty_text(
            sealed_gate.get("candidate_isolation_status"),
            "baseline candidate isolation status",
        ),
    )


def _baseline_eval_artifact_payload(
    *,
    session_id: str,
    backend_id: str,
    eval_spec: BaselineEvalJobSpec,
    public_result: Mapping[str, Any],
    trusted_eval_seal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_EVAL_ARTIFACT_SCHEMA_VERSION,
        "session_id": session_id,
        "task_id": eval_spec.task_id,
        "backend_id": backend_id,
        "eval_job": _job_spec_to_private_dict(eval_spec),
        "public_result": _json_copy(public_result),
        "trusted_eval_seal": _json_copy(trusted_eval_seal),
    }


def _job_spec_to_private_dict(
    spec: TrainJobSpec | EvalJobSpec | BaselineEvalJobSpec,
) -> dict[str, Any]:
    """Preserve trusted host binding only inside a private Genesis artifact."""

    payload = spec.to_public_dict()
    if spec.gpu_process_binding is not None:
        payload["gpu_process_binding"] = spec.gpu_process_binding.to_dict()
    return payload


def _validate_private_baseline_artifact(
    *,
    anchor: BaselineGenesisAnchor,
    expected_path: Path,
    expected_payload: Mapping[str, Any],
) -> None:
    ref = anchor.private_eval_artifact
    path = Path(ref.uri)
    if path.resolve() != expected_path.resolve():
        raise ValueError("baseline Genesis private artifact path drift")
    actual_checksum = file_sha256(path)
    if actual_checksum != ref.checksum:
        raise ValueError("baseline Genesis private artifact checksum mismatch")
    if _read_json(path) != expected_payload:
        raise ValueError("baseline Genesis private artifact content mismatch")


def _validate_reference_qualification_gate(value: Mapping[str, Any]) -> None:
    gate = _exact_mapping(
        value,
        {"schema_version", "status", "protocol_hash", "evidence_hash"},
        "reference qualification gate",
    )
    if gate.get("schema_version") != REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION:
        raise ValueError("reference qualification gate schema_version mismatch")
    if gate.get("status") != "pass":
        raise ValueError("reference qualification gate must pass")
    require_sha256_digest(
        gate.get("protocol_hash"),
        "reference qualification protocol hash",
    )
    require_sha256_digest(
        gate.get("evidence_hash"),
        "reference qualification evidence hash",
    )


def _validate_reference_seal(
    value: Mapping[str, Any],
    *,
    reference_manifest: ReferenceProgramManifest,
    training_binding: TrainingBinding,
    checkpoint_hash: str,
) -> None:
    seal = _json_object(value, "reference trusted eval seal")
    expected = {
        "subject_kind": "reference",
        "subject_id": reference_manifest.reference_id,
        "subject_manifest_hash": reference_manifest.manifest_hash,
        "program_source_checksum": reference_manifest.source_checksum,
        "training_binding_hash": training_binding.binding_hash,
        "capability_profile_hash": reference_manifest.capability_profile_hash,
        "checkpoint_hash": checkpoint_hash,
        "safe_load_status": "pass",
        "checkpoint_safety_status": "pass",
        "candidate_isolation_status": "pass",
        "candidate_code_loaded": False,
    }
    for field_name, expected_value in expected.items():
        if seal.get(field_name) != expected_value:
            raise ValueError(
                f"reference trusted eval seal {field_name} mismatch"
            )
    require_execution_device(
        seal.get("execution_device"),
        "reference trusted eval seal execution_device",
    )
    for field_name in (
        "eval_config_hash",
        "evaluation_binding_hash",
        "task_invariant_hash",
        "checkpoint_safety_report_hash",
        "seal_hash",
    ):
        require_sha256_digest(
            seal.get(field_name),
            f"reference trusted eval seal {field_name}",
        )


def _validate_reference_public_result(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    public_result: PublicEvalResult,
    evaluation_binding: EvaluationBindingV1,
) -> None:
    metrics = public_result.metrics
    if EvaluationBindingV1.from_mapping(
        metrics.get("evaluation_binding")
    ) != evaluation_binding:
        raise ValueError("reference Genesis public evaluation binding mismatch")
    expected_subject = ExperimentSubject.reference(
        anchor.reference_manifest.reference_id
    )
    if ExperimentSubject.from_mapping(
        metrics.get("evaluation_subject")
    ) != expected_subject:
        raise ValueError("reference Genesis public subject mismatch")
    expected_lineage = {
        "subject_manifest_hash": anchor.reference_manifest.manifest_hash,
        "program_source_checksum": anchor.reference_manifest.source_checksum,
        "training_binding_hash": anchor.training_binding.binding_hash,
        "capability_profile_hash": (
            anchor.reference_manifest.capability_profile_hash
        ),
    }
    for field_name, expected_value in expected_lineage.items():
        if metrics.get(field_name) != expected_value:
            raise ValueError(
                f"reference Genesis public {field_name} mismatch"
            )
    gate = _json_object(
        metrics.get("reference_capability_gate"),
        "reference public capability gate",
    )
    if gate != _json_copy(anchor.qualification_gate):
        raise ValueError("reference Genesis qualification gate projection mismatch")
    if gate.get("protocol_hash") != evaluation_binding.task_protocol_hash:
        raise ValueError("reference Genesis qualification protocol mismatch")
    acceptance = _json_object(
        metrics.get("acceptance"),
        "reference public acceptance",
    )
    required_gates = acceptance.get("required_pass_gates")
    if (
        not isinstance(required_gates, list)
        or "reference_capability_gate" not in required_gates
    ):
        raise ValueError(
            "reference Genesis acceptance must require capability gate"
        )


def _reference_eval_artifact_payload(
    *,
    session_id: str,
    backend_id: str,
    reference: VerifiedReferenceProgramArtifact,
    reference_artifact: VerifiedReferenceProgramArtifactRef | None,
    train_spec: TrainJobSpec,
    train_result: TrainJobResult,
    eval_spec: EvalJobSpec,
    public_result: PublicEvalResult,
    trusted_eval_seal: TrustedEvalSeal,
    qualification: ReferenceQualificationResult,
    evaluation_assembly: EvaluationAssembly | None = None,
) -> dict[str, Any]:
    if reference_artifact is None:
        raise ValueError("reference private evidence requires a materialized artifact")
    gpu_process_binding = train_result.gpu_process_binding
    if gpu_process_binding != train_spec.gpu_process_binding:
        raise ValueError("reference train result GPU process binding mismatch")
    if eval_spec.gpu_process_binding != gpu_process_binding:
        raise ValueError("reference eval GPU process binding mismatch")
    train_result_payload = {
        "job_id": train_result.job_id,
        "subject": train_result.subject.to_public_dict(),
        "subject_manifest_hash": train_result.subject_manifest_hash,
        "program_source_checksum": train_result.program_source_checksum,
        "training_binding_hash": train_result.training_binding_hash,
        "capability_profile_hash": train_result.capability_profile_hash,
        "execution_device": train_result.execution_device,
        "checkpoint": train_result.checkpoint.to_public_dict(
            include_metadata=True
        ),
        "summary": _json_copy(train_result.summary),
    }
    if gpu_process_binding is not None:
        train_result_payload["gpu_process_binding"] = (
            gpu_process_binding.to_dict()
        )
    payload = {
        "schema_version": (
            PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
            if evaluation_assembly is not None
            else GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
            if gpu_process_binding is not None
            else REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
        ),
        "session_id": session_id,
        "task_id": train_spec.task_id,
        "backend_id": backend_id,
        "reference_manifest": reference.manifest.to_public_dict(),
        "reference_artifact": {
            **reference_artifact.to_public_dict(),
            "path": reference_artifact.path,
        },
        "training_binding": train_spec.training_binding.to_private_dict(),
        "train_job": {
            "spec": _job_spec_to_private_dict(train_spec),
            "metadata": _json_copy(train_spec.metadata),
        },
        "train_result": train_result_payload,
        "eval_job": {
            "spec": _job_spec_to_private_dict(eval_spec),
            "metadata": _json_copy(eval_spec.metadata),
        },
        "checkpoint_hash": train_result.checkpoint.checksum,
        "checkpoint_artifact": {
            "path": str(_reference_checkpoint_path(train_result)),
            "checksum": train_result.checkpoint.checksum,
        },
        "public_result": public_result.to_public_dict(),
        "trusted_eval_seal": trusted_eval_seal.to_public_dict(),
        "qualification_gate": dict(qualification.gate),
        "qualification_evidence": dict(qualification.private_evidence),
        "program_source": reference.source_text,
    }
    if evaluation_assembly is not None:
        payload["evaluation_assembly"] = evaluation_assembly.to_private_dict()
    return payload


def _reference_program_source_from_private_artifact(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    expected_path: Path,
) -> str:
    ref = anchor.private_eval_artifact
    path = Path(ref.uri)
    if path.resolve() != expected_path.resolve():
        raise ValueError("reference Genesis private artifact path drift")
    if file_sha256(path) != ref.checksum:
        raise ValueError("reference Genesis private artifact checksum mismatch")
    payload = _read_json(path)
    source = payload.get("program_source")
    if not isinstance(source, str) or not source:
        raise ValueError("reference private artifact program_source is invalid")
    anchor.reference_manifest.verify_source_bytes(source.encode("utf-8"))
    return source


def _read_bound_reference_private_artifact(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    expected_path: Path,
) -> dict[str, Any]:
    ref = anchor.private_eval_artifact
    path = Path(ref.uri)
    if path.resolve() != expected_path.resolve():
        raise ValueError("reference Genesis private artifact path drift")
    actual_checksum = file_sha256(path)
    if actual_checksum != ref.checksum:
        raise ValueError("reference Genesis private artifact checksum mismatch")
    raw_payload = _read_json(path)
    schema_version = raw_payload.get("schema_version")
    expected_fields = {
            "schema_version",
            "session_id",
            "task_id",
            "backend_id",
            "reference_manifest",
            "reference_artifact",
            "training_binding",
            "train_job",
            "train_result",
            "eval_job",
            "checkpoint_hash",
            "checkpoint_artifact",
            "public_result",
            "trusted_eval_seal",
            "qualification_gate",
            "qualification_evidence",
            "program_source",
        }
    if schema_version == PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION:
        expected_fields.add("evaluation_assembly")
    payload = _exact_mapping(
        raw_payload,
        expected_fields,
        "reference Genesis private artifact",
    )
    if payload.get("schema_version") not in {
        REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
        GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
        PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError("reference private artifact schema_version mismatch")
    return dict(payload)


def _validate_reference_evaluation_assembly(
    payload: Mapping[str, Any],
    *,
    eval_spec: EvalJobSpec,
    expected_phased: bool,
) -> EvaluationAssembly | None:
    raw_assembly = payload.get("evaluation_assembly")
    if expected_phased != (raw_assembly is not None):
        raise ValueError("reference private phased evaluation mode mismatch")
    if raw_assembly is None:
        return None
    assembly_payload = _exact_mapping(
        raw_assembly,
        {
            "schema_version",
            "evaluation_plan_hash",
            "selected_trial_id",
            "selection",
            "selection_hash",
            "phase_receipts",
            "summary",
            "summary_hash",
            "acceptance_authority",
            "assembly_hash",
        },
        "reference evaluation assembly",
    )
    raw_receipts = assembly_payload.get("phase_receipts")
    if not isinstance(raw_receipts, list):
        raise ValueError("reference evaluation assembly receipts are invalid")
    bindings: list[EvaluationPhaseReceiptBinding] = []
    for raw_binding in raw_receipts:
        binding_payload = _exact_mapping(
            raw_binding,
            {
                "phase",
                "job_id",
                "path",
                "checksum",
                "size_bytes",
                "ref_schema_version",
                "job_spec_hash",
                "phase_result_hash",
                "resource_binding_hash",
            },
            "reference evaluation phase receipt binding",
        )
        binding = EvaluationPhaseReceiptBinding(
            phase=EvaluationPhase.parse(binding_payload.get("phase")),
            job_id=binding_payload.get("job_id"),
            path=binding_payload.get("path"),
            checksum=binding_payload.get("checksum"),
            size_bytes=binding_payload.get("size_bytes"),
            ref_schema_version=binding_payload.get("ref_schema_version"),
            job_spec_hash=binding_payload.get("job_spec_hash"),
            phase_result_hash=binding_payload.get("phase_result_hash"),
            resource_binding_hash=binding_payload.get("resource_binding_hash"),
        )
        loaded = load_trial_eval_phase_receipt(
            TrialReceiptRef(
                kind="eval_phase",
                path=binding.path,
                checksum=binding.checksum,
                size_bytes=binding.size_bytes,
                schema_version=binding.ref_schema_version,
            )
        )
        if (
            loaded.job_spec_hash != binding.job_spec_hash
            or loaded.result.job_id != binding.job_id
            or loaded.result.phase is not binding.phase
            or loaded.result.resource_binding_hash != binding.resource_binding_hash
            or canonical_json_hash(loaded.result.to_private_dict())
            != binding.phase_result_hash
        ):
            raise ValueError("reference evaluation phase receipt binding mismatch")
        bindings.append(binding)
    assembly = EvaluationAssembly(
        evaluation_plan_hash=assembly_payload.get("evaluation_plan_hash"),
        selected_trial_id=assembly_payload.get("selected_trial_id"),
        selection=_json_object(
            assembly_payload.get("selection"),
            "reference evaluation assembly selection",
        ),
        phase_receipts=tuple(bindings),
        summary=_json_object(
            assembly_payload.get("summary"),
            "reference evaluation assembly summary",
        ),
        assembly_hash=assembly_payload.get("assembly_hash"),
    )
    if assembly.to_private_dict() != dict(assembly_payload):
        raise ValueError("reference evaluation assembly canonical form mismatch")
    if assembly.evaluation_plan_hash != canonical_json_hash(eval_spec.to_public_dict()):
        raise ValueError("reference evaluation assembly plan mismatch")
    if assembly.selected_trial_id != eval_spec.subject.subject_id:
        raise ValueError("reference evaluation assembly subject mismatch")
    return assembly


def _validate_private_reference_artifact(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    expected_path: Path,
    context: _GenesisContext,
) -> dict[str, Any]:
    payload = _read_bound_reference_private_artifact(
        anchor=anchor,
        expected_path=expected_path,
    )
    expected_values = {
        "session_id": context.session_id,
        "task_id": context.task_spec.task_id,
        "backend_id": context.backend_adapter.backend_id,
        "reference_manifest": anchor.reference_manifest.to_public_dict(),
        "training_binding": anchor.training_binding.to_private_dict(),
        "checkpoint_hash": anchor.checkpoint_hash,
        "public_result": _json_copy(anchor.public_result),
        "trusted_eval_seal": _json_copy(anchor.trusted_eval_seal),
        "qualification_gate": _json_copy(anchor.qualification_gate),
    }
    for field_name, expected_value in expected_values.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(
                f"reference private artifact {field_name} mismatch"
            )
    expected_reference = context.reference_program_artifact
    if expected_reference is None:
        raise ValueError("reference private artifact has no expected source")
    reference_artifact = _reference_artifact_from_private_payload(
        payload,
        expected_reference=expected_reference,
        anchor=anchor,
    )
    train_spec = _reference_train_spec_from_private_artifact(
        payload,
        anchor=anchor,
        reference_artifact=reference_artifact,
    )
    if train_spec.gpu_process_binding != context.gpu_process_binding:
        raise ValueError("reference private train GPU process binding mismatch")
    train_result = _reference_train_result_from_private_artifact(
        payload,
        anchor=anchor,
        train_spec=train_spec,
    )
    checkpoint_payload = _exact_mapping(
        payload.get("checkpoint_artifact"),
        {"path", "checksum"},
        "reference private checkpoint artifact",
    )
    if checkpoint_payload != {
        "path": anchor.checkpoint_artifact.uri,
        "checksum": anchor.checkpoint_hash,
    }:
        raise ValueError("reference private checkpoint artifact mismatch")
    checkpoint_path = _reference_checkpoint_path(train_result)
    if checkpoint_path != Path(anchor.checkpoint_artifact.uri).resolve():
        raise ValueError("reference checkpoint path drift")
    if file_sha256(checkpoint_path) != anchor.checkpoint_hash:
        raise ValueError("reference checkpoint bytes changed after Genesis")

    public_result = PublicEvalResult(
        status=anchor.public_result.get("status"),
        public_score=anchor.public_result.get("public_score"),
        metrics=_json_object(
            anchor.public_result.get("metrics"),
            "reference private public result metrics",
        ),
    )
    qualification = ReferenceQualificationResult(
        public_result=public_result,
        gate=_json_object(
            payload.get("qualification_gate"),
            "reference private qualification gate",
        ),
        private_evidence=_json_object(
            payload.get("qualification_evidence"),
            "reference private qualification evidence",
        ),
    )
    if not qualification.passed:
        raise ValueError("reference private qualification gate does not pass")
    source = _reference_program_source_from_private_artifact(
        anchor=anchor,
        expected_path=expected_path,
    )
    if source != expected_reference.source_text:
        raise ValueError("reference private artifact source differs from expected")
    return dict(payload)


def _reference_artifact_from_private_payload(
    payload: Mapping[str, Any],
    *,
    expected_reference: VerifiedReferenceProgramArtifact,
    anchor: ReferenceProgramGenesisAnchor,
) -> VerifiedReferenceProgramArtifactRef:
    raw = _exact_mapping(
        payload.get("reference_artifact"),
        {
            "reference_id",
            "task_id",
            "subject_manifest_hash",
            "program_source_checksum",
            "program_source_size_bytes",
            "artifact_hash",
            "verification_hash",
            "path",
        },
        "reference private materialized artifact",
    )
    artifact = load_reference_program_artifact(
        _nonempty_text(raw.get("path"), "reference artifact path"),
        expected_program=expected_reference,
        expected_artifact_hash=require_sha256_digest(
            raw.get("artifact_hash"),
            "reference artifact hash",
        ),
        expected_reference_id=anchor.reference_manifest.reference_id,
        expected_task_id=anchor.reference_manifest.task_id,
        expected_subject_manifest_hash=str(anchor.reference_manifest.manifest_hash),
        expected_program_source_checksum=anchor.reference_manifest.source_checksum,
    )
    if {**artifact.to_public_dict(), "path": artifact.path} != dict(raw):
        raise ValueError("reference materialized artifact projection mismatch")
    return artifact


def _reference_train_spec_from_private_artifact(
    payload: Mapping[str, Any],
    *,
    anchor: ReferenceProgramGenesisAnchor,
    reference_artifact: VerifiedReferenceProgramArtifactRef,
) -> TrainJobSpec:
    wrapper = _exact_mapping(
        payload.get("train_job"),
        {"spec", "metadata"},
        "reference private train job",
    )
    raw = _json_object(wrapper.get("spec"), "reference private train job spec")
    metadata = _json_object(
        wrapper.get("metadata"),
        "reference private train job metadata",
    )
    spec = TrainJobSpec(
        job_id=_nonempty_text(raw.get("job_id"), "reference train job_id"),
        task_id=_nonempty_text(raw.get("task_id"), "reference train task_id"),
        subject=ExperimentSubject.from_mapping(raw.get("subject")),
        subject_manifest_hash=require_sha256_digest(
            raw.get("subject_manifest_hash"),
            "reference train subject_manifest_hash",
        ),
        program_source_checksum=require_sha256_checksum(
            raw.get("program_source_checksum"),
            "reference train source checksum",
        ),
        policy_initialization=PolicyInitialization.random(),
        training_binding=anchor.training_binding,
        budget_steps=raw.get("budget_steps"),
        execution_device=_nonempty_text(
            raw.get("execution_device"),
            "reference train execution_device",
        ),
        gpu_process_binding=(
            GpuProcessBinding.from_mapping(raw.get("gpu_process_binding"))
            if "gpu_process_binding" in raw
            else None
        ),
        metadata=metadata,
        reference_artifact=reference_artifact,
    )
    if _job_spec_to_private_dict(spec) != raw:
        raise ValueError("reference private train job canonical projection mismatch")
    argv = metadata.get("cli_argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("reference private train argv is invalid")
    validate_train_job_argv(spec, argv)
    return spec


def _reference_train_result_from_private_artifact(
    payload: Mapping[str, Any],
    *,
    anchor: ReferenceProgramGenesisAnchor,
    train_spec: TrainJobSpec,
) -> TrainJobResult:
    expected_result_fields = {
        "job_id",
        "subject",
        "subject_manifest_hash",
        "program_source_checksum",
        "training_binding_hash",
        "capability_profile_hash",
        "execution_device",
        "checkpoint",
        "summary",
    }
    if train_spec.gpu_process_binding is not None:
        expected_result_fields.add("gpu_process_binding")
    raw = _exact_mapping(
        payload.get("train_result"),
        expected_result_fields,
        "reference private train result",
    )
    result = TrainJobResult(
        job_id=_nonempty_text(raw.get("job_id"), "reference train result job_id"),
        subject=ExperimentSubject.from_mapping(raw.get("subject")),
        subject_manifest_hash=require_sha256_digest(
            raw.get("subject_manifest_hash"),
            "reference train result manifest hash",
        ),
        program_source_checksum=require_sha256_checksum(
            raw.get("program_source_checksum"),
            "reference train result source checksum",
        ),
        training_binding_hash=require_sha256_digest(
            raw.get("training_binding_hash"),
            "reference train result binding hash",
        ),
        capability_profile_hash=require_sha256_digest(
            raw.get("capability_profile_hash"),
            "reference train result capability hash",
        ),
        execution_device=_nonempty_text(
            raw.get("execution_device"),
            "reference train result execution device",
        ),
        gpu_process_binding=(
            GpuProcessBinding.from_mapping(raw.get("gpu_process_binding"))
            if "gpu_process_binding" in raw
            else None
        ),
        checkpoint=_reference_train_checkpoint_from_private_artifact(payload),
        summary=_json_object(raw.get("summary"), "reference train result summary"),
    )
    _validate_reference_train_result(train_spec, result)
    if _reference_initial_policy_state_hash(result) != anchor.initial_policy_state_hash:
        raise ValueError("reference initial policy state hash drift")
    return result


def _reference_train_checkpoint_from_private_artifact(
    payload: Mapping[str, Any],
) -> CheckpointRef:
    train_result = _json_object(
        payload.get("train_result"),
        "reference private train result",
    )
    checkpoint = _exact_mapping(
        train_result.get("checkpoint"),
        {"uri", "checksum", "role", "sealed", "metadata"},
        "reference private train checkpoint",
    )
    return CheckpointRef(
        uri=_nonempty_text(checkpoint.get("uri"), "reference checkpoint uri"),
        checksum=require_sha256_checksum(
            checkpoint.get("checksum"),
            "reference checkpoint checksum",
        ),
        role=_nonempty_text(checkpoint.get("role"), "reference checkpoint role"),
        sealed=checkpoint.get("sealed"),
        metadata=_json_object(
            checkpoint.get("metadata"),
            "reference checkpoint metadata",
        ),
    )


def _reference_eval_spec_from_private_artifact(
    payload: Mapping[str, Any],
    *,
    anchor: ReferenceProgramGenesisAnchor,
    evaluation_binding: EvaluationBindingV1,
) -> EvalJobSpec:
    wrapper = _exact_mapping(
        payload.get("eval_job"),
        {"spec", "metadata"},
        "reference private eval job",
    )
    expected_eval_fields = {
        "job_id",
        "task_id",
        "subject",
        "subject_manifest_hash",
        "program_source_checksum",
        "training_binding_hash",
        "capability_profile_hash",
        "execution_device",
        "schema_version",
        "eval_protocol_version",
        "public_seed_count",
        "public_accept_seed_count",
        "sealed_validation_seed_count",
        "eval_num_envs",
        "eval_protocol_hash",
        "evaluation_binding",
        "checkpoint",
    }
    raw_eval_spec = _json_object(
        wrapper.get("spec"),
        "reference private eval job",
    )
    if payload.get("schema_version") == (
        GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
    ) or (
        payload.get("schema_version") == PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
        and "gpu_process_binding" in raw_eval_spec
    ):
        expected_eval_fields.add("gpu_process_binding")
    raw = _exact_mapping(
        raw_eval_spec,
        expected_eval_fields,
        "reference private eval job",
    )
    metadata = _json_object(
        wrapper.get("metadata"),
        "reference private eval job metadata",
    )
    protocol = EvaluationProtocol.from_mapping(
        {
            key: raw[key]
            for key in (
                "schema_version",
                "eval_protocol_version",
                "public_seed_count",
                "public_accept_seed_count",
                "sealed_validation_seed_count",
                "eval_num_envs",
                "eval_protocol_hash",
            )
        }
    )
    if EvaluationBindingV1.from_mapping(
        raw.get("evaluation_binding")
    ) != evaluation_binding:
        raise ValueError("reference private eval job binding mismatch")
    public_checkpoint = _exact_mapping(
        raw.get("checkpoint"),
        {"uri", "checksum", "role", "sealed"},
        "reference private eval checkpoint",
    )
    checkpoint = _reference_train_checkpoint_from_private_artifact(payload)
    if checkpoint.to_public_dict(include_metadata=False) != dict(public_checkpoint):
        raise ValueError(
            "reference private eval checkpoint differs from train result"
        )
    if checkpoint.checksum != anchor.checkpoint_hash:
        raise ValueError("reference private eval checkpoint hash mismatch")
    spec = EvalJobSpec(
        job_id=_nonempty_text(raw.get("job_id"), "reference eval job_id"),
        task_id=_nonempty_text(raw.get("task_id"), "reference eval task_id"),
        subject=ExperimentSubject.from_mapping(raw.get("subject")),
        subject_manifest_hash=require_sha256_digest(
            raw.get("subject_manifest_hash"),
            "reference eval subject_manifest_hash",
        ),
        program_source_checksum=require_sha256_checksum(
            raw.get("program_source_checksum"),
            "reference eval program_source_checksum",
        ),
        training_binding_hash=require_sha256_digest(
            raw.get("training_binding_hash"),
            "reference eval training_binding_hash",
        ),
        capability_profile_hash=require_sha256_digest(
            raw.get("capability_profile_hash"),
            "reference eval capability_profile_hash",
        ),
        checkpoint=checkpoint,
        public_seed_count=protocol.public_seed_count,
        public_accept_seed_count=protocol.public_accept_seed_count,
        sealed_validation_seed_count=protocol.sealed_validation_seed_count,
        eval_num_envs=protocol.eval_num_envs,
        eval_protocol_version=protocol.eval_protocol_version,
        execution_device=_nonempty_text(
            raw.get("execution_device"),
            "reference eval execution_device",
        ),
        gpu_process_binding=(
            GpuProcessBinding.from_mapping(raw.get("gpu_process_binding"))
            if "gpu_process_binding" in raw
            else None
        ),
        evaluation_binding=evaluation_binding,
        metadata=metadata,
    )
    expected = {
        "subject": ExperimentSubject.reference(
            anchor.reference_manifest.reference_id
        ),
        "subject_manifest_hash": anchor.reference_manifest.manifest_hash,
        "program_source_checksum": anchor.reference_manifest.source_checksum,
        "training_binding_hash": anchor.training_binding.binding_hash,
        "capability_profile_hash": (
            anchor.reference_manifest.capability_profile_hash
        ),
        "checkpoint_hash": anchor.checkpoint_hash,
    }
    actual = {
        "subject": spec.subject,
        "subject_manifest_hash": spec.subject_manifest_hash,
        "program_source_checksum": spec.program_source_checksum,
        "training_binding_hash": spec.training_binding_hash,
        "capability_profile_hash": spec.capability_profile_hash,
        "checkpoint_hash": spec.execution_checkpoint().checksum,
    }
    if actual != expected:
        raise ValueError("reference private eval job lineage mismatch")
    if _job_spec_to_private_dict(spec) != dict(raw):
        raise ValueError("reference private eval job canonical projection mismatch")
    argv = metadata.get("cli_argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("reference private eval argv is invalid")
    validate_eval_job_argv(spec, argv)
    return spec


def _build_reference_seal(
    *,
    eval_spec: EvalJobSpec,
    task_spec: TaskSpec,
    public_result: PublicEvalResult,
) -> TrustedEvalSeal:
    metrics = public_result.metrics
    sealed_gate = _json_object(
        metrics.get("sealed_validation_gate"),
        "reference sealed_validation_gate",
    )
    return build_trusted_eval_seal(
        eval_spec=eval_spec,
        task_invariants=task_spec.invariants,
        checkpoint_safety_report=metrics.get("checkpoint_safety_report"),
        candidate_isolation_status=_nonempty_text(
            sealed_gate.get("candidate_isolation_status"),
            "reference candidate isolation status",
        ),
    )


def _assert_manifest_matches_anchor(
    manifest: BaselineManifest,
    anchor: BaselineGenesisAnchor,
) -> None:
    if manifest.baseline_id != anchor.baseline_id:
        raise ValueError("session Genesis baseline_id drift")
    if manifest.manifest_hash != anchor.baseline_manifest_hash:
        raise ValueError("session Genesis baseline manifest hash drift")
    if manifest.checkpoint.checksum != anchor.checkpoint_hash:
        raise ValueError("session Genesis checkpoint hash drift")


def _assert_policy_initialization_matches_anchor(
    policy_initialization: PolicyInitialization,
    anchor: GenesisAnchor,
) -> None:
    if policy_initialization.kind == "random":
        return
    baseline = policy_initialization.baseline
    if baseline is None or not isinstance(anchor, BaselineGenesisAnchor):
        raise ValueError("warm-start policy initialization requires baseline Genesis")
    if (
        baseline.baseline_id != anchor.baseline_id
        or baseline.baseline_manifest_hash != anchor.baseline_manifest_hash
        or baseline.checkpoint.checksum != anchor.checkpoint_hash
    ):
        raise ValueError("warm-start policy initialization differs from Genesis")


def _validate_private_artifact_ref(
    ref: PrivateArtifactRef,
    *,
    expected_kind: str,
    expected_reason: str,
    field_prefix: str,
) -> None:
    if ref.kind != expected_kind:
        raise ValueError(f"{field_prefix} private artifact kind mismatch")
    if not isinstance(ref.uri, str) or not ref.uri:
        raise ValueError(f"{field_prefix} private artifact uri must be non-empty")
    if not (
        PurePosixPath(ref.uri).is_absolute()
        or PureWindowsPath(ref.uri).is_absolute()
    ):
        raise ValueError(f"{field_prefix} private artifact uri must be absolute")
    require_sha256_checksum(
        ref.checksum,
        f"{field_prefix} private artifact checksum",
    )
    if ref.reason != expected_reason:
        raise ValueError(f"{field_prefix} private artifact reason mismatch")


def _reject_partial_session_without_genesis(
    output_dir: Path,
    *,
    session_id: str,
    allow_pre_genesis_state: bool = False,
) -> None:
    partial_paths = [output_dir / "ledger.jsonl"]
    if not allow_pre_genesis_state:
        partial_paths.extend(
            (
                output_dir / BASELINE_EVAL_ARTIFACT_FILENAME,
                output_dir / REFERENCE_EVAL_ARTIFACT_FILENAME,
            )
        )
    for partial_path in partial_paths:
        if partial_path.exists():
            raise ValueError(
                "session has durable state but no Genesis; refusing implicit rebuild: "
                + str(partial_path)
            )
    runner_session_dir = output_dir / "runner" / session_id
    if runner_session_dir.is_dir() and not allow_pre_genesis_state:
        partial_runner_file = next(
            (path for path in runner_session_dir.rglob("*") if path.is_file()),
            None,
        )
        if partial_runner_file is not None:
            raise ValueError(
                "session has runner state but no Genesis; refusing implicit rebuild: "
                + str(partial_runner_file)
            )


def _pre_genesis_creation_identity(
    *,
    session_id: str,
    genesis_identity_hash: str,
    session_contract_hash: str,
    budget_contract_hash: str,
    resource_binding_hash: str | None,
    scopes: Mapping[str, PreGenesisExecutionScope],
) -> dict[str, Any]:
    return _validate_pre_genesis_creation_identity({
        "session_id": session_id,
        "genesis_identity_hash": genesis_identity_hash,
        "session_contract_hash": session_contract_hash,
        "budget_contract_hash": budget_contract_hash,
        "resource_binding_hash": resource_binding_hash,
        "attempt_scope_hashes": {
            role: canonical_json_hash(scope.binding_payload())
            for role, scope in sorted(scopes.items())
        },
    })


def _validate_pre_genesis_creation_identity(value: Any) -> dict[str, Any]:
    mapping = _exact_mapping(
        value,
        {
            "session_id",
            "genesis_identity_hash",
            "session_contract_hash",
            "budget_contract_hash",
            "resource_binding_hash",
            "attempt_scope_hashes",
        },
        "pre-Genesis creation identity",
    )
    _nonempty_text(mapping["session_id"], "pre-Genesis creation session_id")
    for field_name in ("genesis_identity_hash", "session_contract_hash", "budget_contract_hash"):
        require_sha256_digest(mapping[field_name], f"pre-Genesis {field_name}")
    resource_binding_hash = mapping.get("resource_binding_hash")
    if resource_binding_hash is not None:
        require_sha256_digest(resource_binding_hash, "pre-Genesis resource binding")
    scope_hashes = mapping["attempt_scope_hashes"]
    allowed_roles = {
        "reference_train",
        "reference_eval",
        "baseline_eval",
        *(_reference_eval_phase_role(phase) for phase in EvaluationPhase),
    }
    if (
        not isinstance(scope_hashes, Mapping)
        or not scope_hashes
        or not set(scope_hashes).issubset(allowed_roles)
        or len(set(scope_hashes.values())) != len(scope_hashes)
    ):
        raise ValueError("pre-Genesis creation attempt scope roles mismatch")
    for role, scope_hash in scope_hashes.items():
        require_sha256_digest(scope_hash, f"pre-Genesis {role} scope hash")
    return _json_copy(mapping)


def _recover_pre_genesis_execution(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
    previous_lease_payload: bytes | None,
) -> bool:
    """Recover only exact pre-Genesis reservations from a dead creation owner."""

    expected_identity = _validate_pre_genesis_creation_identity(identity)
    _validate_pre_genesis_scope_set(expected_identity, scopes=scopes)
    if previous_lease_payload is None:
        return False
    previous_lease = _parse_pre_genesis_creation_lease(previous_lease_payload)
    if previous_lease is None:
        return False
    if previous_lease["identity"] != expected_identity:
        raise ValueError("pre-Genesis creation lease identity mismatch")

    observed_incarnation: str | None = None
    process_id_reused = False
    if previous_lease["state"] == "active":
        process_id = previous_lease["process_id"]
        observed_incarnation = _process_incarnation_token(process_id)
        if observed_incarnation == previous_lease["process_incarnation_token"]:
            if os.name != "nt" or _process_group_is_alive(
                previous_lease["process_group_id"]
            ):
                raise ValueError("pre-Genesis creation process is still alive")
            # Windows may keep a terminated process object queryable briefly;
            # the zero-time wait used by the group probe is the liveness gate.
            observed_incarnation = None
        if observed_incarnation is not None:
            require_sha256_digest(
                observed_incarnation,
                "observed pre-Genesis creation process incarnation",
            )
        process_id_reused = observed_incarnation is not None
        if (
            not process_id_reused
            and _process_group_is_alive(previous_lease["process_group_id"])
        ):
            raise ValueError("pre-Genesis creation process group is still alive")

    recoveries = _load_pre_genesis_recoveries(
        output_dir=output_dir,
        identity=expected_identity,
        scopes=scopes,
    )
    recovery_bindings = _pre_genesis_recovery_bindings(recoveries)
    scan = _scan_pre_genesis_attempts(
        identity=expected_identity,
        scopes=scopes,
        recovery_bindings=recovery_bindings,
    )
    if previous_lease["state"] == "closed":
        if scan["open_reservations"]:
            raise ValueError(
                "closed pre-Genesis creation lease has an open attempt reservation"
            )
        return scan["reservation_count"] > 0

    for recovery in recoveries:
        _finish_pre_genesis_recovery(
            recovery,
            recovery_hash=canonical_json_hash(recovery),
        )
    recovered_paths = {
        path for paths in recovery_bindings.values() for path in paths
    }
    open_reservations = [
        path for path in scan["open_reservations"] if path not in recovered_paths
    ]
    if open_reservations:
        recovery = _publish_or_load_pre_genesis_recovery(
            output_dir=output_dir,
            identity=expected_identity,
            scopes=scopes,
            unclean_lease=previous_lease,
            observed_incarnation=observed_incarnation,
            process_id_reused=process_id_reused,
            reservations=open_reservations,
        )
        recovery_hash = canonical_json_hash(recovery)
        _finish_pre_genesis_recovery(recovery, recovery_hash=recovery_hash)
    return scan["reservation_count"] > 0


def _validate_pre_genesis_scope_set(
    identity: Mapping[str, Any],
    *,
    scopes: Mapping[str, PreGenesisExecutionScope],
) -> None:
    if not all(isinstance(scope, PreGenesisExecutionScope) for scope in scopes.values()):
        raise TypeError("pre-Genesis recovery scope is invalid")
    actual_hashes = {
        role: canonical_json_hash(scope.binding_payload())
        for role, scope in sorted(scopes.items())
    }
    if actual_hashes != identity["attempt_scope_hashes"]:
        raise ValueError("pre-Genesis recovery attempt scope set mismatch")
    for scope in scopes.values():
        if any(
            getattr(scope, field_name) != identity[field_name]
            for field_name in (
                "session_id",
                "genesis_identity_hash",
                "session_contract_hash",
                "budget_contract_hash",
                "resource_binding_hash",
            )
        ):
            raise ValueError("pre-Genesis recovery scope identity mismatch")


def _parse_pre_genesis_creation_lease(payload: bytes) -> dict[str, Any] | None:
    try:
        parsed = strict_json_loads(
            payload,
            context="session Genesis creation lease",
        )
    except ValueError as exc:
        raise ValueError("session Genesis creation lease is invalid") from exc
    if parsed == {}:
        return None
    if not isinstance(parsed, Mapping):
        raise ValueError("session Genesis creation lease must be an object")
    schema = parsed.get("schema_version")
    if schema == _SESSION_GENESIS_CREATION_LEASE_SCHEMA_VERSION:
        return None
    if schema != _PRE_GENESIS_CREATION_LEASE_SCHEMA_VERSION:
        raise ValueError("session Genesis creation lease schema mismatch")
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError("session Genesis creation lease is not canonical JSON")
    return _validate_pre_genesis_creation_lease_mapping(parsed)


def _validate_pre_genesis_creation_lease_mapping(value: Any) -> dict[str, Any]:
    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "identity",
            "owner_token",
            "process_id",
            "process_group_id",
            "process_incarnation_token",
            "ownership_model",
            "state",
        },
        "pre-Genesis creation lease",
    )
    if (
        mapping.get("schema_version") != _PRE_GENESIS_CREATION_LEASE_SCHEMA_VERSION
        or mapping.get("ownership_model") != "process_held_os_file_lock"
        or mapping.get("state") not in {"active", "closed"}
    ):
        raise ValueError("pre-Genesis creation lease binding mismatch")
    mapping["identity"] = _validate_pre_genesis_creation_identity(mapping["identity"])
    _canonical_uuid_hex(mapping["owner_token"], "pre-Genesis creation owner token")
    for field_name in ("process_id", "process_group_id"):
        _positive_int(mapping[field_name], f"pre-Genesis creation {field_name}")
    require_sha256_digest(
        mapping["process_incarnation_token"],
        "pre-Genesis creation process_incarnation_token",
    )
    return _json_copy(mapping)


def _load_pre_genesis_recoveries(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
) -> list[dict[str, Any]]:
    root = output_dir / _PRE_GENESIS_RECOVERY_DIRNAME
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("pre-Genesis recovery root must be a regular directory")
    recoveries: list[dict[str, Any]] = []
    for path in sorted(root.glob("recovery-*.json")):
        parsed = _load_generation_json(
            path,
            max_bytes=_SESSION_GENESIS_RECOVERY_MAX_BYTES,
            label="pre-Genesis recovery receipt",
        )
        recovery = _validate_pre_genesis_recovery(
            parsed,
            identity=identity,
            scopes=scopes,
        )
        expected_name = (
            "recovery-"
            + canonical_json_hash(recovery["unclean_creation_lease"])[:24]
            + ".json"
        )
        if path.name != expected_name:
            raise ValueError("pre-Genesis recovery receipt filename mismatch")
        recoveries.append(recovery)
    return recoveries


def _validate_pre_genesis_recovery(
    value: Any,
    *,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
) -> dict[str, Any]:
    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "unclean_creation_lease",
            "observed_process_incarnation_token",
            "process_id_reused",
            "recovered_at_utc",
            "abandoned_attempt_reservations",
            "ledger_authority",
        },
        "pre-Genesis recovery receipt",
    )
    lease = _validate_pre_genesis_creation_lease_mapping(
        mapping.get("unclean_creation_lease")
    )
    if (
        mapping.get("schema_version")
        != _PRE_GENESIS_UNCLEAN_RECOVERY_SCHEMA_VERSION
        or mapping.get("ledger_authority") is not False
        or lease["identity"] != identity
        or lease["state"] != "active"
    ):
        raise ValueError("pre-Genesis recovery receipt binding mismatch")
    observed_incarnation = mapping.get("observed_process_incarnation_token")
    if observed_incarnation is not None:
        require_sha256_digest(
            observed_incarnation,
            "observed pre-Genesis recovery process incarnation",
        )
    expected_reused = (
        observed_incarnation is not None
        and observed_incarnation != lease["process_incarnation_token"]
    )
    if (
        mapping.get("process_id_reused") is not expected_reused
        or observed_incarnation == lease["process_incarnation_token"]
    ):
        raise ValueError("pre-Genesis recovery process incarnation mismatch")
    _timestamp(mapping["recovered_at_utc"], "pre-Genesis recovered_at_utc")
    refs = mapping.get("abandoned_attempt_reservations")
    if not isinstance(refs, list) or not refs:
        raise ValueError("pre-Genesis abandoned reservation refs must be non-empty")
    seen_paths: set[Path] = set()
    for ref in refs:
        _validate_generation_reservation_ref(ref)
        _, reservation = _validate_pre_genesis_reservation(
            Path(ref["path"]),
            identity=identity,
            scopes=scopes,
        )
        path = reservation.path.resolve(strict=True)
        if reservation.payload_hash != ref["reservation_hash"]:
            raise ValueError("pre-Genesis recovery reservation hash mismatch")
        if path in seen_paths:
            raise ValueError("pre-Genesis recovery repeats an attempt reservation")
        seen_paths.add(path)
    return _json_copy(mapping)


def _pre_genesis_recovery_bindings(
    recoveries: list[dict[str, Any]],
) -> dict[str, set[Path]]:
    bindings: dict[str, set[Path]] = {}
    seen_paths: set[Path] = set()
    for recovery in recoveries:
        paths = {
            Path(ref["path"]).resolve(strict=True)
            for ref in recovery["abandoned_attempt_reservations"]
        }
        if seen_paths.intersection(paths):
            raise ValueError(
                "pre-Genesis reservation appears in multiple recovery receipts"
            )
        seen_paths.update(paths)
        bindings[canonical_json_hash(recovery)] = paths
    return bindings


def _scan_pre_genesis_attempts(
    *,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
    recovery_bindings: Mapping[str, set[Path]],
) -> dict[str, Any]:
    roots: set[Path] = set()
    for scope in scopes.values():
        if scope.journal_root.is_symlink():
            raise ValueError("pre-Genesis attempt journal root cannot be a symlink")
        roots.add(scope.journal_root.resolve(strict=False))
    reservations: list[Path] = []
    result_paths: set[Path] = set()
    for root in sorted(roots, key=str):
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("pre-Genesis attempt journal root must be a directory")
        for path in sorted(root.glob("*/attempt_*.reservation.json")):
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("pre-Genesis attempt reservation cannot be a symlink")
            resolved = path.resolve(strict=True)
            if resolved.parent.parent != root:
                raise ValueError("pre-Genesis attempt reservation escapes its root")
            reservations.append(resolved)
        for path in sorted(root.glob("*/attempt_*.result.json")):
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("pre-Genesis attempt result cannot be a symlink")
            result_paths.add(path.resolve(strict=True))

    expected_result_paths = {_attempt_result_path(path) for path in reservations}
    if not result_paths.issubset(expected_result_paths):
        raise ValueError("pre-Genesis attempt result has no reservation")

    open_reservations: list[Path] = []
    logical_jobs: dict[str, set[str]] = {}
    for path in reservations:
        journal, reservation = _validate_pre_genesis_reservation(
            path,
            identity=identity,
            scopes=scopes,
        )
        scope_hash = canonical_json_hash(journal.scope.binding_payload())
        logical_jobs.setdefault(scope_hash, set()).add(journal.logical_job_id)
        attempt_index = reservation.payload["attempt_index"]
        if attempt_index > 0:
            previous = journal.load_result(attempt_index - 1)
            verified = journal.reserve(attempt_index, previous_result=previous)
            if verified.payload_hash != reservation.payload_hash:
                raise ValueError("pre-Genesis retry reservation chain mismatch")
        result_path = _attempt_result_path(path)
        if not result_path.exists():
            open_reservations.append(path)
            continue
        result = journal.load_result(attempt_index)
        recovery_hash = result.payload.get("recovery_receipt_hash")
        if (
            recovery_hash is not None
            and path not in recovery_bindings.get(recovery_hash, set())
        ):
            raise ValueError("pre-Genesis recovered result has no bound recovery receipt")
    if any(len(job_ids) != 1 for job_ids in logical_jobs.values()):
        raise ValueError("pre-Genesis attempt scope contains multiple logical jobs")
    return {
        "reservation_count": len(reservations),
        "open_reservations": open_reservations,
    }


def _publish_or_load_pre_genesis_recovery(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
    unclean_lease: Mapping[str, Any],
    observed_incarnation: str | None,
    process_id_reused: bool,
    reservations: list[Path],
) -> dict[str, Any]:
    lease_hash = canonical_json_hash(unclean_lease)
    path = output_dir / _PRE_GENESIS_RECOVERY_DIRNAME / (
        f"recovery-{lease_hash[:24]}.json"
    )
    recovery = {
        "schema_version": _PRE_GENESIS_UNCLEAN_RECOVERY_SCHEMA_VERSION,
        "unclean_creation_lease": _json_copy(unclean_lease),
        "observed_process_incarnation_token": observed_incarnation,
        "process_id_reused": process_id_reused,
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "abandoned_attempt_reservations": [
            _generation_reservation_ref(path) for path in reservations
        ],
        "ledger_authority": False,
    }
    _publish_generation_json_once(path, recovery)
    return _validate_pre_genesis_recovery(
        recovery,
        identity=identity,
        scopes=scopes,
    )


def _finish_pre_genesis_recovery(
    recovery: Mapping[str, Any],
    *,
    recovery_hash: str,
) -> None:
    for ref in recovery["abandoned_attempt_reservations"]:
        journal, reservation = JobAttemptJournal.from_reservation_path(
            Path(ref["path"])
        )
        result_path = _attempt_result_path(reservation.path)
        if result_path.exists():
            result = journal.load_result(reservation.payload["attempt_index"])
            if result.payload.get("recovery_receipt_hash") != recovery_hash:
                raise ValueError("pre-Genesis recovered result binding mismatch")
            continue
        journal.finish(
            reservation,
            outcome="infrastructure_failure",
            failure_kind="abandoned_unclean",
            retryable=(
                reservation.payload["attempt_index"]
                < journal.scope.max_infrastructure_retries
            ),
            output_hash=None,
            returncode=None,
            recovery_receipt_hash=recovery_hash,
        )


def _validate_pre_genesis_reservation(
    path: Path,
    *,
    identity: Mapping[str, Any],
    scopes: Mapping[str, PreGenesisExecutionScope],
) -> tuple[JobAttemptJournal, Any]:
    path = path.resolve(strict=True)
    journal, reservation = JobAttemptJournal.from_reservation_path(path)
    scope_hash = canonical_json_hash(journal.scope.binding_payload())
    role_by_hash = {
        scope_hash: role
        for role, scope_hash in identity["attempt_scope_hashes"].items()
    }
    role = role_by_hash.get(scope_hash)
    if not isinstance(journal.scope, PreGenesisExecutionScope) or role is None:
        raise ValueError("abandoned reservation is outside pre-Genesis authority")
    phase_roles = {_reference_eval_phase_role(phase) for phase in EvaluationPhase}
    expected_kind = (
        "eval_phase"
        if role in phase_roles
        else {
            "reference_train": "train",
            "reference_eval": "eval",
            "baseline_eval": "baseline_eval",
        }[role]
    )
    if journal.job_kind != expected_kind:
        raise ValueError("pre-Genesis abandoned reservation job kind mismatch")
    approved_root = scopes[role].journal_root.resolve(strict=False)
    if path.parent.parent != approved_root:
        raise ValueError("pre-Genesis abandoned reservation root mismatch")
    return journal, reservation


def _attempt_result_path(reservation_path: Path) -> Path:
    return reservation_path.with_name(
        reservation_path.name.replace(".reservation.", ".result.")
    )


def _acquire_session_genesis_creation_lease(
    *,
    output_dir: Path,
    session_id: str,
    pre_genesis_identity: Mapping[str, Any] | None = None,
) -> _SessionGenesisCreationLease:
    path = output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME
    if pre_genesis_identity is not None:
        identity = _validate_pre_genesis_creation_identity(pre_genesis_identity)
        process_id = os.getpid()
        process_group_id = _current_process_group_id()
        process_incarnation_token = _process_incarnation_token(process_id)
        if process_incarnation_token is None:
            raise RuntimeError("current process incarnation is unavailable")
        require_sha256_digest(
            process_incarnation_token,
            "pre-Genesis creation process incarnation",
        )
        owner_token = uuid.uuid4().hex
        ownership = {
            "schema_version": _PRE_GENESIS_CREATION_LEASE_SCHEMA_VERSION,
            "identity": identity,
            "owner_token": owner_token,
            "process_id": process_id,
            "process_group_id": process_group_id,
            "process_incarnation_token": process_incarnation_token,
            "ownership_model": "process_held_os_file_lock",
        }
        active_payload = canonical_json_bytes(
            {**ownership, "state": "active"},
            trailing_newline=True,
        )
        closed_payload = canonical_json_bytes(
            {**ownership, "state": "closed"},
            trailing_newline=True,
        )
        try:
            lock = acquire_process_file_lock(
                path,
                blocking=False,
                payload=None,
            )
        except ProcessFileLockUnavailableError as exc:
            raise SessionGenesisCreationInProgressError(
                f"session Genesis creation already in progress: {session_id}"
            ) from exc
        try:
            previous_payload = _read_owned_creation_lease_payload(lock)
            return _SessionGenesisCreationLease(
                payload=active_payload,
                lock=lock,
                previous_payload=previous_payload,
                closed_payload=closed_payload,
                active=False,
            )
        except BaseException:
            lock.release()
            raise
    payload = canonical_json_bytes(
        {
            "schema_version": _SESSION_GENESIS_CREATION_LEASE_SCHEMA_VERSION,
            "session_id": session_id,
            "process_id": os.getpid(),
            "owner_token": uuid.uuid4().hex,
            "ownership_model": "process_held_os_file_lock",
        },
        trailing_newline=True,
    )
    try:
        lock = acquire_process_file_lock(
            path,
            blocking=False,
            payload=payload,
        )
    except ProcessFileLockUnavailableError as exc:
        raise SessionGenesisCreationInProgressError(
            f"session Genesis creation already in progress: {session_id}"
        ) from exc
    return _SessionGenesisCreationLease(
        payload=payload,
        lock=lock,
    )


def _read_owned_creation_lease_payload(lock: ProcessFileLock) -> bytes:
    """Read stale metadata through the descriptor that owns the byte lock.

    Windows denies a second handle access to the locked byte, so the task-owned
    recovery transaction must inspect the prior bytes through the owner handle.
    """

    lock.assert_owned()
    handle = lock._handle
    size = os.fstat(handle.fileno()).st_size
    if size > _SESSION_GENESIS_RECOVERY_MAX_BYTES:
        raise ValueError("session Genesis creation lease exceeds byte limit")
    handle.seek(0)
    payload = handle.read(_SESSION_GENESIS_RECOVERY_MAX_BYTES + 1)
    if len(payload) > _SESSION_GENESIS_RECOVERY_MAX_BYTES:
        raise ValueError("session Genesis creation lease exceeds byte limit")
    return payload


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value, trailing_newline=True)
    create_once_bytes(path, encoded)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return load_strict_json_object(
            path,
            context=f"session Genesis artifact {path.name}",
            require_canonical=True,
            trailing_newline=True,
        )
    except ValueError as exc:
        raise ValueError(f"failed to read session Genesis artifact: {path.name}") from exc


def _exact_mapping(
    value: Any,
    expected_keys: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    mapping = _json_object(value, field_name)
    if set(mapping) != expected_keys:
        raise ValueError(f"{field_name} fields mismatch")
    return mapping


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return _json_copy(value)


def session_genesis_identity_hash(
    genesis_identity_snapshot: Mapping[str, Any],
) -> str:
    """Hash the canonical JSON projection used by Session Genesis."""

    return canonical_json_hash(_json_copy(genesis_identity_snapshot))


def _json_copy(value: Any) -> Any:
    normalized = _normalize_json_value(value)
    try:
        return strict_json_loads(
            canonical_json_bytes(normalized),
            context="canonical JSON value",
        )
    except ValueError as exc:
        raise ValueError("value must be canonical JSON data") from exc


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError("value must contain only canonical JSON types")


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    return parsed


__all__ = [
    "BASELINE_EVAL_ARTIFACT_FILENAME",
    "LEGACY_SESSION_GENESIS_SCHEMA_VERSION",
    "GENERATION_SESSION_GENESIS_SCHEMA_VERSION",
    "REFERENCE_EVAL_ARTIFACT_FILENAME",
    "REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION",
    "GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION",
    "PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION",
    "REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION",
    "SESSION_GENESIS_CREATION_LEASE_FILENAME",
    "SESSION_GENESIS_FILENAME",
    "SESSION_GENESIS_SCHEMA_VERSION",
    "BaselineGenesisAnchor",
    "BootstrapGenesisAnchor",
    "ReferenceProgramGenesisAnchor",
    "ReferencePhasedEvaluationAdapter",
    "SessionGenesisCreationInProgressError",
    "SessionGenesisArtifact",
    "VerifiedSessionGenesis",
    "prepare_session_genesis",
    "session_genesis_identity_hash",
    "verify_session_genesis_for_runtime",
]

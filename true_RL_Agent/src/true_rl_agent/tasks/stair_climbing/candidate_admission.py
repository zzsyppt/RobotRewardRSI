"""Stair-owned sealed candidate admission policy."""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.artifact import (
    CandidateExecutionAuthorization,
    SealedCandidateArtifactRef,
)
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.candidates.isolated_worker import (
    IsolatedWorkerProtocolError,
    IsolatedWorkerStartupError,
    WorkerReady,
    WorkerStartupFailure,
    run_ready_worker,
)
from true_rl_agent.candidates.package_contract import (
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.candidates.verifier import (
    CandidateAdmissionPolicyDescriptor,
    CandidateVerifier,
    StaticCandidateVerifier,
    StaticCandidateVerifierPolicy,
    VerificationResult,
    seal_and_verify_candidate,
    static_verifier_source_policy_hash,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.protocol_v2 import StairProtocolV2
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_artifact,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION,
    STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION,
    QualifiedStairRewardProgram,
    StairRewardEnvironmentAxisError,
    StairRewardProgramError,
    assert_stair_reward_program_deterministic,
    assert_stair_reward_program_environment_independent,
    qualify_stair_candidate_reward_program,
)
from true_rl_agent.tasks.stair_climbing.reward_source_semantics import (
    audit_stair_reward_source_semantics,
    stair_reward_source_semantics_policy_hash,
)
from true_rl_agent.training_programs import CANONICAL_TRAINING_COMPONENTS_PATH

STAIR_CANDIDATE_ADMISSION_POLICY_ID = (
    "true_rl_agent.stair_climbing.complete_reward_admission.v1"
)
STAIR_CANDIDATE_ADMISSION_POLICY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.candidate_admission_policy.v5"
)
STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.candidate_execution_authorization.v1"
)
DEFAULT_STAIR_QUALIFICATION_TIMEOUT_S = 8.0
DEFAULT_STAIR_QUALIFICATION_WORKER_STARTUP_TIMEOUT_S = 120.0
_QUALIFICATION_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.candidate_qualification.v1"
)
_QUALIFICATION_CANDIDATE_FAILURE_REASON = (
    "stair_candidate_reward_program_invalid"
)
_QUALIFICATION_TIMEOUT_REASON = "stair_candidate_qualification_timeout"
_QUALIFICATION_INFRASTRUCTURE_FAILURE_REASON = (
    "stair_candidate_qualification_infrastructure_failed"
)


def _require_positive_finite_timeout(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class VerifiedStairCandidateExecutionAuthorization:
    """Task-owned interpretation of one generic execution authorization."""

    authorization: CandidateExecutionAuthorization
    static_verifier_policy: StaticCandidateVerifierPolicy
    qualification_hash: str

    def verify_program(
        self,
        program: QualifiedStairRewardProgram,
    ) -> QualifiedStairRewardProgram:
        if not isinstance(program, QualifiedStairRewardProgram):
            raise TypeError("Stair execution authorization requires a qualified program")
        expected_source_policy_hash = static_verifier_source_policy_hash(
            policy=self.static_verifier_policy,
        )
        if (
            program.subject.kind != "candidate"
            or program.qualification_hash != self.qualification_hash
            or program.static_verifier_policy_hash
            != expected_source_policy_hash
        ):
            raise ValueError(
                "Stair reward program does not match execution authorization"
            )
        return program


@dataclass(frozen=True)
class StairCandidateAdmissionPolicy(CandidateVerifier):
    """Admit complete rewards only after qualification of runner-owned bytes."""

    config: StairClimbingTaskConfig
    resolved_cell_manifest: ResolvedCellManifestV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    qualification_timeout_s: float = DEFAULT_STAIR_QUALIFICATION_TIMEOUT_S
    worker_startup_timeout_s: float = (
        DEFAULT_STAIR_QUALIFICATION_WORKER_STARTUP_TIMEOUT_S
    )
    static_verifier: StaticCandidateVerifier = field(
        default_factory=StaticCandidateVerifier,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("Stair admission requires StairClimbingTaskConfig")
        if not (
            self.config.is_unified_protocol
            or isinstance(self.config.protocol_v2, StairProtocolV2)
        ):
            raise ValueError("Stair admission requires the final V2 protocol")
        if self.config.is_unified_protocol:
            if not isinstance(self.resolved_cell_manifest, ResolvedCellManifestV1):
                raise ValueError("Stair VNext admission requires the resolved manifest")
            if (
                self.resolved_cell_manifest.cell_variant_id
                != self.config.cell_variant_id
                or self.resolved_cell_manifest.resolved_cell_task_protocol_hash
                != self.config.task_protocol_hash
            ):
                raise ValueError("Stair admission manifest/config identity mismatch")
        _require_positive_finite_timeout(
            self.qualification_timeout_s,
            "Stair qualification timeout",
        )
        _require_positive_finite_timeout(
            self.worker_startup_timeout_s,
            "Stair qualification worker startup timeout",
        )
        if not isinstance(self.static_verifier, StaticCandidateVerifier):
            raise TypeError("Stair admission static verifier is invalid")

    def admission_policy_descriptor(self) -> CandidateAdmissionPolicyDescriptor:
        return CandidateAdmissionPolicyDescriptor(
            policy_id=STAIR_CANDIDATE_ADMISSION_POLICY_ID,
            policy_hash=self.policy_hash,
            capability_ids=frozenset(
                {COMPLETE_REWARD_PROGRAM_CAPABILITY_ID}
            ),
            execution_authorization_schema_version=(
                STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
            ),
        )

    def authoring_prompt_rules(self) -> Mapping[str, Any]:
        return {**self.static_verifier.authoring_prompt_rules(),
                "reward_research_scope": "One structural reward mechanism; numeric-only modifications and numeric-only renames require repair."}

    def verify_authoring_delta(
        self,
        *,
        candidate: CandidatePackageRef,
        verification: VerificationResult,
        parent_source: str | None,
    ) -> VerificationResult:
        if verification.status != "pass":
            return verification
        try:
            _verified_mutable_candidate_source(
                candidate,
                verification,
            )
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            return _authoring_verification_failure(
                verification,
                reason="stair_candidate_authoring_source_invalid",
                issue=(
                    "authoring_source_identity_invalid:"
                    + type(exc).__name__
                ),
            )
        with tempfile.TemporaryDirectory(
            prefix=".stair-authoring-admission-"
        ) as artifact_root:
            temporary_verification = seal_and_verify_candidate(
                candidate,
                artifact_root=artifact_root,
                verifier=self,
            )
        task_evidence = temporary_verification.result.details.get(
            "task_qualification"
        )
        if (
            temporary_verification.result.status != "pass"
            and isinstance(task_evidence, Mapping)
            and task_evidence.get("error_type")
            == StairRewardEnvironmentAxisError.__name__
        ):
            return _authoring_verification_failure(
                verification,
                reason="stair_candidate_environment_axis_invalid",
                issue="reward_environment_axis_independence_failed",
            )
        if temporary_verification.result.status != "pass":
            return verification
        if (
            temporary_verification.result.program_identity
            != verification.program_identity
        ):
            return _authoring_verification_failure(
                verification,
                reason="stair_candidate_authoring_source_invalid",
                issue="authoring_source_identity_changed_during_admission",
            )
        return verification

    @property
    def policy_hash(self) -> str:
        adapter = StairClimbingComponentSmokeAdapter(
            self.config,
            resolved_cell_manifest=self.resolved_cell_manifest,
        )
        static_policy = self.static_verifier.admission_policy_descriptor()
        reference_artifact, _ = resolve_official_stair_reference_artifact(
            self.config,
            resolved_cell_manifest=self.resolved_cell_manifest,
        )
        return canonical_json_hash(
            {
                "schema_version": (
                    STAIR_CANDIDATE_ADMISSION_POLICY_SCHEMA_VERSION
                ),
                "policy_id": STAIR_CANDIDATE_ADMISSION_POLICY_ID,
                "capability_id": COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
                "task_protocol_hash": self.config.task_protocol_hash,
                "input_schema_hash": adapter.contract.input_schema_hash,
                "smoke_fixture_hash": adapter.fixture_hash,
                "static_verifier_policy_hash": static_policy.policy_hash,
                "task_source_semantics_policy_hash": (
                    stair_reward_source_semantics_policy_hash(
                        adapter.contract,
                        trusted_reference_source_text=(
                            reference_artifact.source_text
                        ),
                    )
                ),
                "execution_authorization_schema_version": (
                    STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
                ),
                "qualification_timeout_s": float(
                    self.qualification_timeout_s
                ),
            }
        )

    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    ) -> VerificationResult:
        static_result = self.static_verifier.verify(candidate)
        if static_result.status != "pass":
            return static_result
        if isinstance(candidate, CandidatePackageRef):
            reference_artifact, _ = resolve_official_stair_reference_artifact(
                self.config,
                resolved_cell_manifest=self.resolved_cell_manifest,
            )
            source_text = _verified_mutable_candidate_source(
                candidate,
                static_result,
            )
            task_issues = audit_stair_reward_source_semantics(
                source_text,
                contract=StairClimbingComponentSmokeAdapter(
                    self.config,
                    resolved_cell_manifest=self.resolved_cell_manifest,
                ).contract,
                trusted_reference_source_text=reference_artifact.source_text,
            )
            if task_issues:
                return VerificationResult(
                    "fail",
                    "stair_candidate_source_semantics_invalid",
                    {
                        "issues": list(task_issues),
                        "derived_change_set": list(
                            static_result.details.get(
                                "derived_change_set",
                                (),
                            )
                        ),
                    },
                    program_identity=static_result.program_identity,
                )
            # Mutable bytes receive repairable task checks. Trusted identity is
            # still minted only after the runner seals and rechecks exact bytes.
            return static_result
        if not isinstance(candidate, SealedCandidateArtifactRef):
            return VerificationResult(
                "fail",
                "stair_candidate_artifact_type_invalid",
            )

        task_evidence = _run_qualification_worker(
            artifact=candidate,
            protocol_v2_path=self.config.protocol_v2_path,
            cell_variant_id=self.config.cell_variant_id,
            resolved_cell_task_protocol_hash=(
                self.config.resolved_cell_task_protocol_hash
            ),
            resolved_cell_manifest=self.resolved_cell_manifest,
            static_verifier_policy=self.static_verifier.policy,
            worker_startup_timeout_s=float(self.worker_startup_timeout_s),
            candidate_execution_timeout_s=float(
                self.qualification_timeout_s
            ),
        )
        if task_evidence.get("status") != "pass":
            return VerificationResult(
                "fail",
                str(
                    task_evidence.get(
                        "reason",
                        "stair_candidate_qualification_failed",
                    )
                ),
                {
                    "issues": [
                        str(
                            task_evidence.get(
                                "reason",
                                "stair_candidate_qualification_failed",
                            )
                        )
                    ],
                    "task_qualification": dict(task_evidence),
                },
            )
        adapter = StairClimbingComponentSmokeAdapter(
            self.config,
            resolved_cell_manifest=self.resolved_cell_manifest,
        )
        _assert_task_evidence_matches(
            candidate,
            task_evidence,
            expected_input_schema_hash=adapter.contract.input_schema_hash,
            expected_smoke_fixture_hash=adapter.fixture_hash,
            expected_static_verifier_source_policy_hash=(
                static_verifier_source_policy_hash(
                    policy=self.static_verifier.policy,
                )
            ),
        )
        descriptor = self.admission_policy_descriptor()
        details = {
            "derived_change_set": list(
                static_result.details.get("derived_change_set", ())
            ),
            "verification_policy_id": descriptor.policy_id,
            "verification_policy_hash": descriptor.policy_hash,
            "capability_ids": sorted(descriptor.capability_ids),
            "static_verification_hash": canonical_json_hash(
                static_result.to_public_dict()
            ),
            "task_qualification": dict(task_evidence),
        }
        return VerificationResult(
            "pass",
            "stair_candidate_admission_passed",
            details,
            program_identity=static_result.program_identity,
            candidate_artifact_hash=candidate.artifact_hash,
        )

    def build_execution_authorization_payload(
        self,
        *,
        artifact: SealedCandidateArtifactRef,
        result: VerificationResult,
    ) -> Mapping[str, Any]:
        if not isinstance(artifact, SealedCandidateArtifactRef):
            raise TypeError("Stair execution authorization artifact is invalid")
        if result.status != "pass":
            raise ValueError("Stair execution authorization requires a passing result")
        descriptor = self.admission_policy_descriptor()
        if (
            result.details.get("verification_policy_id") != descriptor.policy_id
            or result.details.get("verification_policy_hash")
            != descriptor.policy_hash
        ):
            raise ValueError("Stair execution authorization policy identity mismatch")
        evidence = result.details.get("task_qualification")
        if not isinstance(evidence, Mapping):
            raise ValueError("Stair execution authorization evidence is missing")
        adapter = StairClimbingComponentSmokeAdapter(
            self.config,
            resolved_cell_manifest=self.resolved_cell_manifest,
        )
        source_policy_hash = static_verifier_source_policy_hash(
            policy=self.static_verifier.policy,
        )
        _assert_task_evidence_matches(
            artifact,
            evidence,
            expected_input_schema_hash=adapter.contract.input_schema_hash,
            expected_smoke_fixture_hash=adapter.fixture_hash,
            expected_static_verifier_source_policy_hash=source_policy_hash,
        )
        qualification = evidence["qualification"]
        assert isinstance(qualification, Mapping)
        return {
            "schema_version": (
                STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
            ),
            "task_protocol_hash": self.config.task_protocol_hash,
            "candidate_id": artifact.candidate_id,
            "candidate_artifact_hash": artifact.artifact_hash,
            "subject_manifest_hash": artifact.subject_manifest_hash,
            "program_source_checksum": artifact.program_source_checksum,
            "program_source_size_bytes": artifact.program_source_size_bytes,
            "qualification_schema_version": (
                STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION
            ),
            "qualification_hash": require_sha256_digest(
                qualification.get("qualification_hash"),
                "Stair candidate qualification hash",
            ),
            "input_schema_hash": adapter.contract.input_schema_hash,
            "smoke_fixture_hash": adapter.fixture_hash,
            "static_verifier_source_policy_hash": source_policy_hash,
            "qualification_timeout_s": float(self.qualification_timeout_s),
            "static_verifier_policy_snapshot": (
                self.static_verifier.policy.to_snapshot_payload()
            ),
        }


def validate_stair_candidate_execution_authorization(
    authorization: CandidateExecutionAuthorization,
    *,
    artifact: SealedCandidateArtifactRef,
    config: StairClimbingTaskConfig,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> VerifiedStairCandidateExecutionAuthorization:
    """Rebuild the exact policy and bind it to Stair runtime identities."""

    if not isinstance(authorization, CandidateExecutionAuthorization):
        raise TypeError("Stair candidate execution authorization is invalid")
    if not isinstance(artifact, SealedCandidateArtifactRef):
        raise TypeError("Stair candidate execution artifact is invalid")
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair candidate execution config is invalid")
    payload = authorization.payload
    expected_fields = {
        "schema_version",
        "task_protocol_hash",
        "candidate_id",
        "candidate_artifact_hash",
        "subject_manifest_hash",
        "program_source_checksum",
        "program_source_size_bytes",
        "qualification_schema_version",
        "qualification_hash",
        "input_schema_hash",
        "smoke_fixture_hash",
        "static_verifier_source_policy_hash",
        "qualification_timeout_s",
        "static_verifier_policy_snapshot",
    }
    if set(payload) != expected_fields:
        raise ValueError("Stair candidate execution authorization fields mismatch")
    if (
        payload.get("schema_version")
        != STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
    ):
        raise ValueError("Stair candidate execution authorization schema mismatch")
    if (
        require_sha256_digest(
            payload.get("task_protocol_hash"),
            "Stair candidate authorization task protocol hash",
        )
        != config.task_protocol_hash
    ):
        raise ValueError("Stair candidate authorization task protocol mismatch")
    source_size = payload.get("program_source_size_bytes")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise ValueError("Stair candidate authorization source size is invalid")
    if (
        payload.get("candidate_id") != artifact.candidate_id
        or require_sha256_digest(
            payload.get("candidate_artifact_hash"),
            "Stair candidate authorization artifact hash",
        )
        != artifact.artifact_hash
        or require_sha256_digest(
            payload.get("subject_manifest_hash"),
            "Stair candidate authorization subject manifest hash",
        )
        != artifact.subject_manifest_hash
        or require_sha256_checksum(
            payload.get("program_source_checksum"),
            "Stair candidate authorization source checksum",
        )
        != artifact.program_source_checksum
        or source_size != artifact.program_source_size_bytes
    ):
        raise ValueError("Stair candidate authorization artifact identity mismatch")
    if (
        payload.get("qualification_schema_version")
        != STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("Stair candidate authorization qualification schema mismatch")

    adapter = StairClimbingComponentSmokeAdapter(
        config,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    if (
        require_sha256_digest(
            payload.get("input_schema_hash"),
            "Stair candidate authorization input schema hash",
        )
        != adapter.contract.input_schema_hash
        or require_sha256_digest(
            payload.get("smoke_fixture_hash"),
            "Stair candidate authorization smoke fixture hash",
        )
        != adapter.fixture_hash
    ):
        raise ValueError("Stair candidate authorization runtime contract mismatch")
    static_policy = StaticCandidateVerifierPolicy.from_snapshot_payload(
        payload.get("static_verifier_policy_snapshot")
    )
    source_policy_hash = static_verifier_source_policy_hash(policy=static_policy)
    if require_sha256_digest(
        payload.get("static_verifier_source_policy_hash"),
        "Stair candidate authorization source policy hash",
    ) != source_policy_hash:
        raise ValueError("Stair candidate authorization source policy mismatch")
    qualification_timeout_s = _require_positive_finite_timeout(
        payload.get("qualification_timeout_s"),
        "Stair candidate authorization qualification timeout",
    )
    reconstructed = StairCandidateAdmissionPolicy(
        config,
        resolved_cell_manifest=resolved_cell_manifest,
        qualification_timeout_s=qualification_timeout_s,
        static_verifier=StaticCandidateVerifier(policy=static_policy),
    )
    descriptor = reconstructed.admission_policy_descriptor()
    if (
        authorization.admission_policy_id != descriptor.policy_id
        or authorization.admission_policy_hash != descriptor.policy_hash
    ):
        raise ValueError("Stair candidate authorization admission identity mismatch")
    return VerifiedStairCandidateExecutionAuthorization(
        authorization=authorization,
        static_verifier_policy=static_policy,
        qualification_hash=require_sha256_digest(
            payload.get("qualification_hash"),
            "Stair candidate authorization qualification hash",
        ),
    )


def _run_qualification_worker(
    *,
    artifact: SealedCandidateArtifactRef,
    protocol_v2_path: Path,
    cell_variant_id: str,
    resolved_cell_task_protocol_hash: str | None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
    static_verifier_policy: StaticCandidateVerifierPolicy,
    worker_startup_timeout_s: float,
    candidate_execution_timeout_s: float,
) -> Mapping[str, Any]:
    try:
        outcome = run_ready_worker(
            target=_qualification_worker,
            args=(
                artifact,
                str(protocol_v2_path),
                cell_variant_id,
                resolved_cell_task_protocol_hash,
                (
                    resolved_cell_manifest.to_dict()
                    if resolved_cell_manifest is not None
                    else None
                ),
                static_verifier_policy,
            ),
            startup_timeout_s=worker_startup_timeout_s,
            execution_timeout_s=candidate_execution_timeout_s,
        )
    except (IsolatedWorkerProtocolError, IsolatedWorkerStartupError) as exc:
        raise RuntimeError(
            "Stair candidate qualification infrastructure failed"
        ) from exc
    if not outcome.completed:
        result = {
            "schema_version": _QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
            "status": "fail",
            "reason": _QUALIFICATION_TIMEOUT_REASON,
            "failure_owner": "candidate",
            "error_type": "TimeoutError",
        }
    else:
        result = outcome.result
    return _validate_qualification_worker_result(result)


def _validate_qualification_worker_result(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise RuntimeError("Stair candidate qualification evidence is invalid")
    evidence = dict(result)
    if evidence.get("status") != "fail":
        if evidence.get("status") != "pass":
            raise RuntimeError("Stair candidate qualification status is invalid")
        return evidence
    expected_fields = {
        "schema_version",
        "status",
        "reason",
        "failure_owner",
        "error_type",
    }
    if (
        set(evidence) != expected_fields
        or evidence.get("schema_version")
        != _QUALIFICATION_EVIDENCE_SCHEMA_VERSION
        or not isinstance(evidence.get("error_type"), str)
        or not evidence["error_type"]
    ):
        raise RuntimeError("Stair candidate qualification failure evidence is invalid")
    owner = evidence.get("failure_owner")
    reason = evidence.get("reason")
    if owner == "infrastructure":
        if reason != _QUALIFICATION_INFRASTRUCTURE_FAILURE_REASON:
            raise RuntimeError(
                "Stair candidate qualification infrastructure evidence is invalid"
            )
        raise RuntimeError("Stair candidate qualification infrastructure failed")
    if owner != "candidate" or reason not in {
        _QUALIFICATION_CANDIDATE_FAILURE_REASON,
        _QUALIFICATION_TIMEOUT_REASON,
    }:
        raise RuntimeError("Stair candidate qualification failure owner is invalid")
    return evidence


def _qualification_failure_evidence(exc: Exception) -> Mapping[str, Any]:
    candidate_owned = isinstance(exc, StairRewardProgramError)
    return {
        "schema_version": _QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
        "status": "fail",
        "reason": (
            _QUALIFICATION_CANDIDATE_FAILURE_REASON
            if candidate_owned
            else _QUALIFICATION_INFRASTRUCTURE_FAILURE_REASON
        ),
        "failure_owner": "candidate" if candidate_owned else "infrastructure",
        "error_type": type(exc).__name__,
    }


def _qualification_worker(
    artifact: SealedCandidateArtifactRef,
    protocol_v2_path: str,
    cell_variant_id: str,
    resolved_cell_task_protocol_hash: str | None,
    resolved_cell_manifest_payload: Mapping[str, Any] | None,
    static_verifier_policy: StaticCandidateVerifierPolicy,
    result_connection: Any,
    start_connection: Any,
) -> None:
    try:
        config = StairClimbingTaskConfig.defaults(
            Path(protocol_v2_path),
            cell_variant_id=cell_variant_id,
            resolved_cell_task_protocol_hash=resolved_cell_task_protocol_hash,
        )
        resolved_cell_manifest = (
            ResolvedCellManifestV1.from_mapping(resolved_cell_manifest_payload)
            if resolved_cell_manifest_payload is not None
            else None
        )
        if not isinstance(static_verifier_policy, StaticCandidateVerifierPolicy):
            raise TypeError("Stair qualification worker policy is invalid")
        adapter = StairClimbingComponentSmokeAdapter(
            config,
            resolved_cell_manifest=resolved_cell_manifest,
        )
        adapter.build_reward_inputs()
        reference_artifact, _ = resolve_official_stair_reference_artifact(
            config,
            resolved_cell_manifest=resolved_cell_manifest,
        )
    except Exception as exc:  # noqa: BLE001 - private detail is not projected.
        result_connection.send(
            WorkerStartupFailure(error_type=type(exc).__name__)
        )
        return
    result_connection.send(
        WorkerReady(
            evidence={"smoke_fixture_hash": adapter.fixture_hash},
        )
    )
    try:
        if start_connection.recv() != "start":
            return
    except EOFError:
        return
    try:
        program = qualify_stair_candidate_reward_program(
            artifact=artifact,
            contract=adapter.contract,
            static_verifier_policy=static_verifier_policy,
            trusted_reference_artifact=reference_artifact,
        )
        determinism = assert_stair_reward_program_deterministic(
            program,
            adapter.build_reward_inputs,
        )
        environment_axis = (
            assert_stair_reward_program_environment_independent(
                program,
                adapter.build_reward_inputs,
            )
        )
    except Exception as exc:  # noqa: BLE001 - ownership is closed below.
        result_connection.send(_qualification_failure_evidence(exc))
        return
    result_connection.send(
        {
            "schema_version": _QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
            "status": "pass",
            "reason": "stair_candidate_reward_program_qualified",
            "qualification": program.to_public_dict(),
            "determinism": determinism,
            "environment_axis": environment_axis,
            "smoke_fixture_hash": adapter.fixture_hash,
        }
    )


def _assert_task_evidence_matches(
    artifact: SealedCandidateArtifactRef,
    evidence: Mapping[str, Any],
    *,
    expected_input_schema_hash: str,
    expected_smoke_fixture_hash: str,
    expected_static_verifier_source_policy_hash: str,
) -> None:
    qualification = evidence.get("qualification")
    if not isinstance(qualification, Mapping):
        raise RuntimeError("Stair candidate qualification identity is missing")
    environment_axis = evidence.get("environment_axis")
    subject = qualification.get("subject")
    expected_subject = {
        "schema_version": "true_rl_agent.experiment.subject.v2",
        "kind": "candidate",
        "id": artifact.candidate_id,
    }
    if (
        subject != expected_subject
        or qualification.get("subject_manifest_hash")
        != artifact.subject_manifest_hash
        or qualification.get("program_source_checksum")
        != artifact.program_source_checksum
        or qualification.get("program_source_size_bytes")
        != artifact.program_source_size_bytes
        or qualification.get("schema_version")
        != STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION
        or not isinstance(qualification.get("qualification_hash"), str)
        or qualification.get("input_schema_hash")
        != expected_input_schema_hash
        or qualification.get("static_verifier_policy_hash")
        != expected_static_verifier_source_policy_hash
        or not isinstance(environment_axis, Mapping)
        or environment_axis.get("schema_version")
        != STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION
        or environment_axis.get("status") != "pass"
        or environment_axis.get("term_names")
        != qualification.get("term_names")
        or environment_axis.get("reverse_permutation_checks")
        != len(qualification.get("term_names", ()))
        or environment_axis.get("single_row_locality_checks")
        != len(qualification.get("term_names", ()))
        or evidence.get("smoke_fixture_hash")
        != expected_smoke_fixture_hash
    ):
        raise RuntimeError("Stair candidate qualification identity mismatch")


def _verified_mutable_candidate_source(
    candidate: CandidatePackageRef,
    verification: VerificationResult,
) -> str:
    if not isinstance(candidate, CandidatePackageRef):
        raise TypeError("Stair authoring verification requires a mutable candidate")
    if verification.program_identity is None:
        raise ValueError("Stair authoring verification has no program identity")
    preflight = preflight_candidate_package(
        candidate.path,
        sealed_artifact=False,
        enforce_size_limits=True,
    )
    source_state = preflight.file_state(CANONICAL_TRAINING_COMPONENTS_PATH)
    with open_preflighted_candidate_file(
        preflight,
        CANONICAL_TRAINING_COMPONENTS_PATH,
    ) as source:
        source_bytes = source.read(source_state.size_bytes + 1)
    if len(source_bytes) != source_state.size_bytes:
        raise ValueError("candidate program source size changed after preflight")
    identity = verification.program_identity
    source_checksum = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    if (
        identity.candidate_id != candidate.candidate_id
        or identity.source_checksum != source_checksum
        or identity.source_size_bytes != len(source_bytes)
    ):
        raise ValueError("candidate program identity changed after static verification")
    return source_bytes.decode("utf-8-sig")


def _authoring_verification_failure(
    verification: VerificationResult,
    *,
    reason: str,
    issue: str,
) -> VerificationResult:
    return VerificationResult(
        "fail",
        reason,
        {
            "issues": [issue],
            "derived_change_set": list(
                verification.details.get("derived_change_set", ())
            ),
        },
        program_identity=verification.program_identity,
    )


__all__ = [
    "DEFAULT_STAIR_QUALIFICATION_TIMEOUT_S",
    "DEFAULT_STAIR_QUALIFICATION_WORKER_STARTUP_TIMEOUT_S",
    "STAIR_CANDIDATE_ADMISSION_POLICY_ID",
    "STAIR_CANDIDATE_ADMISSION_POLICY_SCHEMA_VERSION",
    "STAIR_CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION",
    "StairCandidateAdmissionPolicy",
    "VerifiedStairCandidateExecutionAuthorization",
    "validate_stair_candidate_execution_authorization",
]

"""Candidate verifier interface for replay and fake-round tests."""

from __future__ import annotations

import ast
import inspect
import math
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping

from true_rl_agent.candidates.artifact import (
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    CandidateExecutionAuthorization,
    SealedCandidateArtifactRef,
    VerifiedCandidateArtifactRef,
    seal_candidate_package,
    verify_sealed_candidate_artifact,
)
from true_rl_agent.candidates.authoring_policy import (
    component_name_contract,
    require_component_name,
)
from true_rl_agent.candidates.components import (
    DiagnosticTerm,
    RewardTerm,
    TrainingComponents,
)
from true_rl_agent.candidates.isolated_worker import (
    WorkerReady,
    run_ready_worker,
)
from true_rl_agent.candidates.loader import (
    load_candidate_manifest,
    load_training_components,
)
from true_rl_agent.candidates.package_contract import (
    CANDIDATE_PACKAGE_CONTRACT_HASH,
    CANDIDATE_PACKAGE_FILE_MAX_BYTES,
    CANDIDATE_PACKAGE_MAX_BYTES,
    PROVIDER_AUTHORED_CANDIDATE_FILES,
    CandidatePackageContractError,
    copy_preflighted_candidate_package,
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schedule_contract import (
    allowed_schedule_keys as candidate_allowed_schedule_keys,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.training_programs import (
    CANONICAL_TRAINING_COMPONENTS_PATH,
    CandidateProgramIdentity,
    resolve_candidate_program_identity,
)

STATIC_CANDIDATE_VERIFIER_POLICY_ID = (
    "true_rl_agent.static_candidate_verifier.pure_function.v2"
)
DEFAULT_STATIC_COMPONENT_LOAD_TIMEOUT_S = 2.0
STATIC_CANDIDATE_VERIFIER_POLICY_SNAPSHOT_SCHEMA_VERSION = (
    "true_rl_agent.static_candidate_verifier_policy_snapshot.v1"
)


def _positive_finite_timeout(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


def _immutable_string_set(value: Any, field_name: str) -> frozenset[str]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a collection of strings")
    try:
        normalized = frozenset(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a collection of strings") from exc
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return normalized


def _immutable_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a collection of strings")
    try:
        normalized = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a collection of strings") from exc
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return normalized


@dataclass(frozen=True)
class _FrozenAllowedImports(Mapping[str, frozenset[str]]):
    """Immutable mapping that remains serializable across spawned workers."""

    entries: tuple[tuple[str, frozenset[str]], ...]

    def __getitem__(self, key: str) -> frozenset[str]:
        for module, names in self.entries:
            if module == key:
                return names
        raise KeyError(key)

    def __iter__(self):
        return (module for module, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _immutable_allowed_imports(value: Any) -> Mapping[str, frozenset[str]]:
    if not isinstance(value, Mapping):
        raise TypeError("allowed_imports must be an object")
    normalized: dict[str, frozenset[str]] = {}
    for module, names in value.items():
        if not isinstance(module, str) or not module:
            raise ValueError("allowed_imports keys must be non-empty strings")
        normalized[module] = _immutable_string_set(names, f"allowed_imports[{module}]")
    return _FrozenAllowedImports(tuple(sorted(normalized.items())))


@dataclass(frozen=True)
class StaticCandidateVerifierPolicy:
    """Immutable rules consumed by verification, identity, and prompt projection."""

    provider_authored_files: frozenset[str] = frozenset(PROVIDER_AUTHORED_CANDIDATE_FILES)
    runner_internal_files: frozenset[str] = frozenset(
        {CANDIDATE_ARTIFACT_MANIFEST_FILENAME}
    )
    forbidden_tokens: tuple[str, ...] = (
        "import os",
        "subprocess",
        "open(",
        "optimizer",
        "checkpoint_selector",
        "eval_config",
        "success_metric",
        "runner.",
        "env.",
    )
    allowed_imports: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: {
            "__future__": frozenset({"annotations"}),
            "true_rl_agent.candidates.components": frozenset(
                {"DiagnosticTerm", "RewardInputs", "RewardTerm", "ScheduleTerm", "TrainingComponents"}
            ),
        }
    )
    forbidden_names: frozenset[str] = frozenset(
        {
            "env",
            "runner",
            "optimizer",
            "checkpoint_selector",
            "eval_config",
            "success_metric",
        }
    )
    forbidden_calls: frozenset[str] = frozenset(
        {
            "__import__",
            "__setattr__",
            "__setitem__",
            "breakpoint",
            "compile",
            "delattr",
            "dir",
            "eval",
            "exec",
            "exit",
            "getattr",
            "globals",
            "help",
            "input",
            "locals",
            "open",
            "print",
            "quit",
            "setattr",
            "vars",
        }
    )
    forbidden_call_roots: frozenset[str] = frozenset(
        {"os", "pathlib", "shutil", "socket", "subprocess", "sys"}
    )
    forbidden_mutating_methods: frozenset[str] = frozenset(
        {
            "append",
            "clear",
            "extend",
            "insert",
            "pop",
            "remove",
            "reverse",
            "setdefault",
            "sort",
            "update",
        }
    )
    forbidden_random_calls: frozenset[str] = frozenset(
        {
            "bernoulli",
            "beta",
            "choice",
            "choices",
            "cauchy",
            "exponential",
            "fork_rng",
            "gamma",
            "geometric",
            "get_rng_state",
            "initial_seed",
            "log_normal",
            "manual_seed",
            "multinomial",
            "normal",
            "permutation",
            "poisson",
            "rand",
            "randint",
            "randn",
            "random",
            "random_sample",
            "seed",
            "set_rng_state",
            "shuffle",
            "standard_normal",
            "uniform",
        }
    )
    forbidden_state_attributes: frozenset[str] = frozenset(
        {
            "__class__",
            "__dict__",
            "__globals__",
            "data",
            "storage",
            "untyped_storage",
        }
    )
    forbidden_name_prefixes: tuple[str, ...] = ("__",)
    forbidden_attribute_prefixes: tuple[str, ...] = ("_",)
    allowed_top_level_ast_nodes: frozenset[str] = frozenset(
        {"module_docstring", "ImportFrom", "FunctionDef"}
    )
    forbidden_ast_nodes: frozenset[str] = frozenset(
        {"Import", "Global", "Nonlocal", "AugAssign", "Delete", "NamedExpr"}
    )
    forbidden_assignment_target_nodes: frozenset[str] = frozenset({"Attribute", "Subscript"})
    function_decorators_allowed: bool = False
    nested_functions_allowed: bool = False
    mutable_or_dynamic_defaults_allowed: bool = False
    additional_files_allowed: bool = False
    import_aliases_allowed: bool = False
    relative_imports_allowed: bool = False
    forbidden_mutating_call_suffix: str = "_"
    allowed_schedule_keys: frozenset[str] = frozenset(candidate_allowed_schedule_keys())
    component_load_timeout_s: float = DEFAULT_STATIC_COMPONENT_LOAD_TIMEOUT_S

    def __post_init__(self) -> None:
        for field_name in (
            "provider_authored_files",
            "runner_internal_files",
            "forbidden_names",
            "forbidden_calls",
            "forbidden_call_roots",
            "forbidden_mutating_methods",
            "forbidden_random_calls",
            "forbidden_state_attributes",
            "allowed_top_level_ast_nodes",
            "forbidden_ast_nodes",
            "forbidden_assignment_target_nodes",
            "allowed_schedule_keys",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_string_set(getattr(self, field_name), field_name),
            )
        for field_name in (
            "forbidden_tokens",
            "forbidden_name_prefixes",
            "forbidden_attribute_prefixes",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_string_tuple(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "allowed_imports", _immutable_allowed_imports(self.allowed_imports))
        for field_name in (
            "function_decorators_allowed",
            "nested_functions_allowed",
            "mutable_or_dynamic_defaults_allowed",
            "additional_files_allowed",
            "import_aliases_allowed",
            "relative_imports_allowed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.provider_authored_files != frozenset(
            PROVIDER_AUTHORED_CANDIDATE_FILES
        ):
            raise ValueError("provider_authored_files is a fixed package contract")
        if self.runner_internal_files != frozenset(
            {CANDIDATE_ARTIFACT_MANIFEST_FILENAME}
        ):
            raise ValueError("runner_internal_files is a fixed package contract")
        if self.additional_files_allowed:
            raise ValueError("additional candidate package files are not supported")
        if (
            not isinstance(self.forbidden_mutating_call_suffix, str)
            or not self.forbidden_mutating_call_suffix
        ):
            raise ValueError("forbidden_mutating_call_suffix must be a non-empty string")
        object.__setattr__(
            self,
            "component_load_timeout_s",
            _positive_finite_timeout(self.component_load_timeout_s, "component_load_timeout_s"),
        )

    def source_policy_facts(self) -> dict[str, Any]:
        """Preserve the source-policy identity used by frozen reference programs."""

        return {
            "policy_id": STATIC_CANDIDATE_VERIFIER_POLICY_ID,
            "provider_authored_files": sorted(self.provider_authored_files),
            "allowed_imports": {
                module: sorted(names)
                for module, names in sorted(self.allowed_imports.items())
            },
            "forbidden_tokens": list(self.forbidden_tokens),
            "forbidden_names": sorted(self.forbidden_names),
            "forbidden_name_prefixes": list(self.forbidden_name_prefixes),
            "forbidden_calls": sorted(self.forbidden_calls),
            "forbidden_call_roots": sorted(self.forbidden_call_roots),
            "forbidden_mutating_methods": sorted(self.forbidden_mutating_methods),
            "forbidden_random_calls": sorted(self.forbidden_random_calls),
            "forbidden_state_attributes": sorted(self.forbidden_state_attributes),
            "forbidden_attribute_prefixes": list(self.forbidden_attribute_prefixes),
            "allowed_top_level_ast_nodes": sorted(self.allowed_top_level_ast_nodes),
            "forbidden_ast_nodes": sorted(self.forbidden_ast_nodes),
            "forbidden_assignment_target_nodes": sorted(
                self.forbidden_assignment_target_nodes
            ),
            "function_decorators_allowed": self.function_decorators_allowed,
            "nested_functions_allowed": self.nested_functions_allowed,
            "mutable_or_dynamic_defaults_allowed": self.mutable_or_dynamic_defaults_allowed,
            "additional_files_allowed": self.additional_files_allowed,
            "import_aliases_allowed": self.import_aliases_allowed,
            "relative_imports_allowed": self.relative_imports_allowed,
            "forbidden_mutating_call_suffix": self.forbidden_mutating_call_suffix,
            "allowed_schedule_keys": sorted(self.allowed_schedule_keys),
            "component_callable_contract": {
                "callable_type": "python_function",
                "defined_in_canonical_components_source": True,
                "positional_argument_count": 1,
                "closures_allowed": False,
                "component_name_contract": component_name_contract(),
            },
        }

    def executable_policy_facts(self) -> dict[str, Any]:
        facts = self.source_policy_facts()
        facts["runner_internal_files"] = sorted(self.runner_internal_files)
        facts["component_load_timeout_s"] = self.component_load_timeout_s
        return facts

    def to_snapshot_payload(self) -> dict[str, Any]:
        """Serialize every executable rule from this exact immutable policy."""

        facts = self.executable_policy_facts()
        policy_field_names = {item.name for item in dataclass_fields(type(self))}
        if set(facts) != policy_field_names | {
            "policy_id",
            "component_callable_contract",
        }:
            raise RuntimeError(
                "static candidate verifier policy facts do not cover all fields"
            )
        return {
            "schema_version": (
                STATIC_CANDIDATE_VERIFIER_POLICY_SNAPSHOT_SCHEMA_VERSION
            ),
            "policy_facts": facts,
            "source_policy_hash": canonical_json_hash(self.source_policy_facts()),
            "executable_policy_hash": canonical_json_hash(facts),
        }

    @classmethod
    def from_snapshot_payload(
        cls,
        value: Mapping[str, Any],
    ) -> "StaticCandidateVerifierPolicy":
        if not isinstance(value, Mapping):
            raise TypeError("static verifier policy snapshot must be an object")
        if set(value) != {
            "schema_version",
            "policy_facts",
            "source_policy_hash",
            "executable_policy_hash",
        }:
            raise ValueError("static verifier policy snapshot fields mismatch")
        if (
            value.get("schema_version")
            != STATIC_CANDIDATE_VERIFIER_POLICY_SNAPSHOT_SCHEMA_VERSION
        ):
            raise ValueError("static verifier policy snapshot schema mismatch")
        facts_value = value.get("policy_facts")
        if not isinstance(facts_value, Mapping):
            raise ValueError("static verifier policy facts must be an object")
        facts = dict(facts_value)
        policy_field_names = {item.name for item in dataclass_fields(cls)}
        if set(facts) != policy_field_names | {
            "policy_id",
            "component_callable_contract",
        }:
            raise ValueError("static verifier policy facts fields mismatch")
        try:
            policy = cls(
                **{
                    item.name: facts[item.name]
                    for item in dataclass_fields(cls)
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("static verifier policy snapshot is invalid") from exc
        if policy.executable_policy_facts() != facts:
            raise ValueError("static verifier policy snapshot facts mismatch")
        if require_sha256_digest(
            value.get("source_policy_hash"),
            "static verifier source policy snapshot hash",
        ) != canonical_json_hash(policy.source_policy_facts()):
            raise ValueError("static verifier source policy snapshot hash mismatch")
        if require_sha256_digest(
            value.get("executable_policy_hash"),
            "static verifier executable policy snapshot hash",
        ) != canonical_json_hash(policy.executable_policy_facts()):
            raise ValueError("static verifier executable policy snapshot hash mismatch")
        return policy


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)
    program_identity: CandidateProgramIdentity | None = None
    candidate_artifact_hash: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "reason": self.reason,
            "details": dict(self.details),
        }
        if self.program_identity is not None:
            result["verified_program_identity"] = {
                "candidate_id": self.program_identity.candidate_id,
                "subject_manifest_hash": self.program_identity.manifest_hash,
                "program_source_checksum": self.program_identity.source_checksum,
                "program_source_size_bytes": self.program_identity.source_size_bytes,
                "candidate_artifact_hash": self.candidate_artifact_hash,
            }
        return result


@dataclass(frozen=True)
class CandidateAdmissionPolicyDescriptor:
    """Stable identity and executable capability set for one admission policy."""

    policy_id: str
    policy_hash: str
    capability_ids: frozenset[str] = frozenset()
    execution_authorization_schema_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("candidate admission policy ID is required")
        require_sha256_digest(
            self.policy_hash,
            "candidate admission policy hash",
        )
        if not isinstance(self.capability_ids, frozenset):
            raise TypeError("candidate admission capability IDs must be a frozenset")
        if any(not isinstance(item, str) or not item for item in self.capability_ids):
            raise ValueError(
                "candidate admission capability IDs must be non-empty strings"
            )
        if self.execution_authorization_schema_version is not None and (
            not isinstance(self.execution_authorization_schema_version, str)
            or not self.execution_authorization_schema_version
        ):
            raise ValueError(
                "candidate execution authorization schema must be a non-empty string"
            )

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "capability_ids": sorted(self.capability_ids),
        }
        if self.execution_authorization_schema_version is not None:
            result["execution_authorization_schema_version"] = (
                self.execution_authorization_schema_version
            )
        return result


@dataclass(frozen=True)
class CandidateArtifactVerification:
    result: VerificationResult
    sealed_artifact: SealedCandidateArtifactRef | None = None
    verified_artifact: VerifiedCandidateArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.verified_artifact is not None:
            if self.sealed_artifact is None:
                raise ValueError("verified candidate result has no sealed artifact")
            if self.verified_artifact.sealed_artifact != self.sealed_artifact:
                raise ValueError("verified and sealed candidate artifacts differ")
        if self.result.status == "pass" and self.verified_artifact is None:
            raise ValueError("passing candidate verification has no execution capability")

    @property
    def passed(self) -> bool:
        return self.result.status == "pass" and self.verified_artifact is not None

    def require_verified(self) -> VerifiedCandidateArtifactRef:
        if self.verified_artifact is None:
            raise ValueError(
                "candidate artifact verification failed: " + self.result.reason
            )
        return self.verified_artifact


class CandidateVerifier(ABC):
    def admission_policy_descriptor(self) -> CandidateAdmissionPolicyDescriptor:
        raise TypeError(
            "candidate verifier does not expose an admission policy descriptor"
        )

    def authoring_prompt_rules(self) -> Mapping[str, Any]:
        raise TypeError("candidate verifier does not expose authoring prompt rules")

    def build_execution_authorization_payload(
        self,
        *,
        artifact: SealedCandidateArtifactRef,
        result: VerificationResult,
    ) -> Mapping[str, Any] | None:
        """Return task-owned execution facts, or None when no replay is needed."""

        return None

    @abstractmethod
    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    ) -> VerificationResult:
        raise NotImplementedError


def seal_and_verify_candidate(
    candidate: CandidatePackageRef,
    *,
    artifact_root: str | Path,
    verifier: CandidateVerifier,
) -> CandidateArtifactVerification:
    """Create the only execution capability accepted by production backends."""

    try:
        sealed = seal_candidate_package(candidate, artifact_root=artifact_root)
    except (TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_sealing_failed",
                {"issues": [f"artifact_sealing_failed:{type(exc).__name__}"]},
            )
        )
    return verify_candidate_artifact(sealed, verifier=verifier)


def verify_candidate_artifact(
    artifact: SealedCandidateArtifactRef,
    *,
    verifier: CandidateVerifier,
) -> CandidateArtifactVerification:
    try:
        sealed_before = verify_sealed_candidate_artifact(artifact)
    except (TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_invalid",
                {"issues": [f"artifact_invalid:{type(exc).__name__}"]},
            )
        )

    result = verifier.verify(sealed_before)
    try:
        sealed_after = verify_sealed_candidate_artifact(sealed_before)
    except (TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_changed_during_verification",
                {"issues": [f"artifact_changed:{type(exc).__name__}"]},
            )
        )
    if sealed_after != sealed_before:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_changed_during_verification",
                {"issues": ["artifact_reference_changed"]},
            )
        )
    if result.status != "pass":
        return CandidateArtifactVerification(
            result=result,
            sealed_artifact=sealed_after,
        )
    if not _verified_artifact_matches(result, sealed_after):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verified_identity_mismatch",
                {"issues": ["verified_identity_does_not_match_sealed_artifact"]},
            ),
            sealed_artifact=sealed_after,
        )

    effective_change_set = candidate_change_set_from_verification(
        fallback=sealed_after.change_set,
        details=result.details,
    )
    bound_artifact = replace(
        sealed_after,
        change_set=effective_change_set,
    )
    verification_policy = result.details.get("verification_policy_id")
    verification_policy_hash = result.details.get("verification_policy_hash")
    if not isinstance(verification_policy, str) or not verification_policy:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verification_policy_identity_missing",
            ),
            sealed_artifact=sealed_after,
        )
    try:
        verification_policy_hash = require_sha256_digest(
            verification_policy_hash,
            "candidate verification policy hash",
        )
    except (TypeError, ValueError):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verification_policy_identity_invalid",
            ),
            sealed_artifact=sealed_after,
        )
    try:
        descriptor = verifier.admission_policy_descriptor()
    except (TypeError, ValueError):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verification_policy_identity_invalid",
            ),
            sealed_artifact=sealed_after,
        )
    if (
        descriptor.policy_id != verification_policy
        or descriptor.policy_hash != verification_policy_hash
    ):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verification_policy_identity_mismatch",
            ),
            sealed_artifact=sealed_after,
        )
    try:
        authorization_payload = verifier.build_execution_authorization_payload(
            artifact=bound_artifact,
            result=result,
        )
        execution_authorization = (
            None
            if authorization_payload is None
            else CandidateExecutionAuthorization.from_payload(
                admission_policy_id=verification_policy,
                admission_policy_hash=verification_policy_hash,
                payload=authorization_payload,
            )
        )
        expected_authorization_schema = (
            descriptor.execution_authorization_schema_version
        )
        if expected_authorization_schema is None:
            if execution_authorization is not None:
                raise ValueError("unexpected candidate execution authorization")
        elif (
            execution_authorization is None
            or execution_authorization.payload_schema_version
            != expected_authorization_schema
        ):
            raise ValueError("candidate execution authorization schema mismatch")
        verification_evidence = result.to_public_dict()
        if execution_authorization is not None:
            verification_evidence = {
                **verification_evidence,
                "execution_authorization": (
                    execution_authorization.to_public_dict()
                ),
            }
        verification_hash = canonical_json_hash(verification_evidence)
    except (TypeError, ValueError):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_execution_authorization_invalid",
            ),
            sealed_artifact=sealed_after,
        )
    return CandidateArtifactVerification(
        result=result,
        sealed_artifact=bound_artifact,
        verified_artifact=VerifiedCandidateArtifactRef._validated(
            sealed_artifact=bound_artifact,
            verification_policy=verification_policy,
            verification_policy_hash=verification_policy_hash,
            verification_hash=verification_hash,
            execution_authorization=execution_authorization,
        ),
    )


@dataclass(frozen=True, init=False)
class StaticCandidateVerifier(CandidateVerifier):
    DEFAULT_COMPONENT_LOAD_TIMEOUT_S = DEFAULT_STATIC_COMPONENT_LOAD_TIMEOUT_S
    DEFAULT_COMPONENT_WORKER_STARTUP_TIMEOUT_S = 60.0

    policy: StaticCandidateVerifierPolicy
    worker_startup_timeout_s: float

    def __init__(
        self,
        *,
        policy: StaticCandidateVerifierPolicy | None = None,
        component_load_timeout_s: float | None = None,
        worker_startup_timeout_s: float = DEFAULT_COMPONENT_WORKER_STARTUP_TIMEOUT_S,
    ) -> None:
        if policy is not None and not isinstance(policy, StaticCandidateVerifierPolicy):
            raise TypeError("static candidate verifier policy is invalid")
        if policy is not None and component_load_timeout_s is not None:
            raise ValueError(
                "component_load_timeout_s is already owned by the supplied policy"
            )
        resolved_policy = policy or StaticCandidateVerifierPolicy(
            component_load_timeout_s=(
                self.DEFAULT_COMPONENT_LOAD_TIMEOUT_S
                if component_load_timeout_s is None
                else component_load_timeout_s
            )
        )
        object.__setattr__(self, "policy", resolved_policy)
        object.__setattr__(
            self,
            "worker_startup_timeout_s",
            _positive_finite_timeout(worker_startup_timeout_s, "worker_startup_timeout_s"),
        )

    @property
    def component_load_timeout_s(self) -> float:
        return self.policy.component_load_timeout_s

    def admission_policy_descriptor(self) -> CandidateAdmissionPolicyDescriptor:
        return CandidateAdmissionPolicyDescriptor(
            policy_id=STATIC_CANDIDATE_VERIFIER_POLICY_ID,
            policy_hash=canonical_json_hash(self.policy.executable_policy_facts()),
        )

    def authoring_prompt_rules(self) -> Mapping[str, Any]:
        return static_verifier_prompt_rules(policy=self.policy)

    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef | Path,
    ) -> VerificationResult:
        try:
            candidate_dir = self._candidate_dir(candidate)
        except ValueError as exc:
            return VerificationResult("fail", "candidate_path_invalid", {"issues": [str(exc)]})
        sealed_artifact = isinstance(candidate, SealedCandidateArtifactRef)
        try:
            preflight = preflight_candidate_package(
                candidate_dir,
                sealed_artifact=sealed_artifact,
                enforce_size_limits=not sealed_artifact,
            )
        except CandidatePackageContractError as exc:
            return VerificationResult(
                "fail",
                "static_candidate_verifier_failed",
                {"issues": [exc.issue]},
            )
        if sealed_artifact:
            return self._verify_trusted_candidate(
                candidate,
                candidate_dir=preflight.root,
                sealed_artifact=True,
            )
        with tempfile.TemporaryDirectory(prefix=".candidate-verify-") as tmp:
            trusted_candidate_dir = Path(tmp)
            try:
                copy_preflighted_candidate_package(
                    preflight,
                    trusted_candidate_dir,
                )
                if preflight_candidate_package(candidate_dir) != preflight:
                    raise CandidatePackageContractError(
                        "candidate package changed while it was being verified",
                        issue="package_changed_during_verification",
                    )
                preflight_candidate_package(trusted_candidate_dir)
            except (OSError, TypeError, ValueError) as exc:
                return VerificationResult(
                    "fail",
                    "static_candidate_verifier_failed",
                    {
                        "issues": [
                            exc.issue
                            if isinstance(exc, CandidatePackageContractError)
                            else f"snapshot_failed:{type(exc).__name__}"
                        ]
                    },
                )
            result = self._verify_trusted_candidate(
                candidate,
                candidate_dir=trusted_candidate_dir,
                sealed_artifact=False,
            )
            try:
                unchanged = preflight_candidate_package(candidate_dir) == preflight
            except (OSError, TypeError, ValueError):
                unchanged = False
            if not unchanged:
                return VerificationResult(
                    "fail",
                    "candidate_program_identity_changed",
                    {"issues": ["package_changed_during_verification"]},
                )
            return result

    def _verify_trusted_candidate(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef | Path,
        *,
        candidate_dir: Path,
        sealed_artifact: bool,
    ) -> VerificationResult:
        issues: list[str] = []
        rel = CANONICAL_TRAINING_COMPONENTS_PATH
        try:
            trusted_preflight = preflight_candidate_package(
                candidate_dir,
                sealed_artifact=sealed_artifact,
                enforce_size_limits=not sealed_artifact,
            )
            with open_preflighted_candidate_file(trusted_preflight, rel) as source:
                source_bytes = source.read()
            text = source_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            issues.append(f"invalid_utf8:{rel}")
        else:
            for token in self.policy.forbidden_tokens:
                if token in text:
                    issues.append(f"forbidden_token:{rel}:{token}")
            issues.extend(self._audit_python_source(rel, text))
        try:
            manifest = load_candidate_manifest(candidate_dir)
            if isinstance(candidate, CandidatePackageRef) and manifest.candidate_id != candidate.candidate_id:
                issues.append("manifest_candidate_id_mismatch")
        except OSError:
            raise
        except Exception as exc:  # noqa: BLE001
            issues.append(f"manifest_load_failed:{type(exc).__name__}")
        if issues:
            return VerificationResult("fail", "static_candidate_verifier_failed", {"issues": issues})
        try:
            if isinstance(candidate, SealedCandidateArtifactRef):
                verify_sealed_candidate_artifact(candidate)
            identity_before = resolve_candidate_program_identity(
                candidate_dir,
                expected_candidate_id=(
                    candidate.candidate_id
                    if isinstance(
                        candidate,
                        CandidatePackageRef | SealedCandidateArtifactRef,
                    )
                    else manifest.candidate_id
                ),
                sealed_artifact=sealed_artifact,
                enforce_package_size_limits=not sealed_artifact,
            )
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                "fail",
                "candidate_program_identity_invalid",
                {"issues": [f"program_identity_invalid:{type(exc).__name__}"]},
            )
        loaded = _load_components_summary_with_timeout(
            candidate_dir=candidate_dir,
            allowed_schedule_keys=self.policy.allowed_schedule_keys,
            worker_startup_timeout_s=self.worker_startup_timeout_s,
            component_load_timeout_s=self.component_load_timeout_s,
        )
        if loaded["status"] == "timeout":
            return VerificationResult(
                "fail",
                "training_components_load_timeout",
                {"issues": ["component_load_timeout"]},
            )
        if loaded["status"] == "fail":
            return VerificationResult(
                "fail",
                "training_components_load_failed",
                {"issues": [f"component_load_failed:{loaded['error_type']}"]},
            )
        component_issues = list(loaded["component_issues"])
        if component_issues:
            return VerificationResult(
                "fail",
                "training_components_invalid",
                {"issues": component_issues},
            )
        if loaded["is_noop"]:
            return VerificationResult("fail", "candidate_has_no_effective_components")
        try:
            if isinstance(candidate, SealedCandidateArtifactRef):
                verify_sealed_candidate_artifact(candidate)
            identity_after = resolve_candidate_program_identity(
                candidate_dir,
                expected_candidate_id=identity_before.candidate_id,
                sealed_artifact=sealed_artifact,
                enforce_package_size_limits=not sealed_artifact,
            )
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                "fail",
                "candidate_program_identity_changed",
                {"issues": [f"program_identity_changed:{type(exc).__name__}"]},
            )
        if identity_after != identity_before:
            return VerificationResult(
                "fail",
                "candidate_program_identity_changed",
                {"issues": ["program_identity_changed_during_verification"]},
            )
        return VerificationResult(
            "pass",
            "static_candidate_verifier_passed",
            {
                "derived_change_set": list(loaded["derived_change_set"]),
                "verification_policy_id": STATIC_CANDIDATE_VERIFIER_POLICY_ID,
                "verification_policy_hash": (
                    self.admission_policy_descriptor().policy_hash
                ),
                "candidate_package_contract_hash": (
                    CANDIDATE_PACKAGE_CONTRACT_HASH
                ),
            },
            program_identity=identity_after,
            candidate_artifact_hash=(
                candidate.artifact_hash
                if isinstance(candidate, SealedCandidateArtifactRef)
                else None
            ),
        )

    def _candidate_dir(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef | Path,
    ) -> Path:
        if isinstance(candidate, CandidatePackageRef | SealedCandidateArtifactRef):
            if "://" in candidate.path:
                raise ValueError("candidate_path_must_be_local_directory")
            return Path(candidate.path)
        return Path(candidate)

    def _audit_python_source(self, rel: str, text: str) -> list[str]:
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            return [f"syntax_error:{rel}:{exc.lineno}"]

        issues: list[str] = []
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                if "module_docstring" not in self.policy.allowed_top_level_ast_nodes:
                    issues.append(f"forbidden_top_level:{rel}:Expr")
                continue
            if isinstance(node, ast.ImportFrom):
                if "ImportFrom" in self.policy.allowed_top_level_ast_nodes:
                    issues.extend(self._audit_import_from(rel, node))
                else:
                    issues.append(f"forbidden_top_level:{rel}:ImportFrom")
                continue
            if isinstance(node, ast.FunctionDef):
                if "FunctionDef" not in self.policy.allowed_top_level_ast_nodes:
                    issues.append(f"forbidden_top_level:{rel}:FunctionDef")
                    continue
                if node.decorator_list and not self.policy.function_decorators_allowed:
                    issues.append(
                        f"forbidden_function_decorator:{rel}:{node.name}"
                    )
                if (
                    _has_mutable_or_dynamic_defaults(node)
                    and not self.policy.mutable_or_dynamic_defaults_allowed
                ):
                    issues.append(
                        f"forbidden_function_default:{rel}:{node.name}"
                    )
                if (
                    not self.policy.nested_functions_allowed
                    and any(
                        isinstance(child, ast.FunctionDef)
                        for child in ast.walk(node)
                        if child is not node
                    )
                ):
                    issues.append(
                        f"forbidden_nested_function:{rel}:{node.name}"
                    )
                continue
            issues.append(f"forbidden_top_level:{rel}:{type(node).__name__}")

        for node in ast.walk(tree):
            node_name = type(node).__name__
            if node_name in self.policy.forbidden_ast_nodes:
                issues.extend(_forbidden_ast_node_issues(rel, node))
            elif isinstance(node, ast.ImportFrom):
                issues.extend(self._audit_import_from(rel, node))
            elif isinstance(node, ast.Name):
                if node.id in self.policy.forbidden_names:
                    issues.append(f"forbidden_name:{rel}:{node.id}")
                if node.id.startswith(self.policy.forbidden_name_prefixes):
                    issues.append(f"forbidden_name_prefix:{rel}:{node.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr in self.policy.forbidden_names:
                    issues.append(f"forbidden_attribute:{rel}:{node.attr}")
                if node.attr in self.policy.forbidden_state_attributes or node.attr.startswith(
                    self.policy.forbidden_attribute_prefixes
                ):
                    issues.append(f"forbidden_state_attribute:{rel}:{node.attr}")
            elif isinstance(node, ast.Assign):
                if any(
                    _is_forbidden_assignment_target(
                        target,
                        self.policy.forbidden_assignment_target_nodes,
                    )
                    for target in node.targets
                ):
                    issues.append(f"forbidden_state_assignment:{rel}")
            elif isinstance(node, ast.AnnAssign):
                if _is_forbidden_assignment_target(
                    node.target,
                    self.policy.forbidden_assignment_target_nodes,
                ):
                    issues.append(f"forbidden_state_assignment:{rel}")
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in self.policy.forbidden_calls:
                    issues.append(f"forbidden_call:{rel}:{call_name}")
                call_root = call_name.split(".", 1)[0] if call_name else ""
                if call_root in self.policy.forbidden_call_roots:
                    issues.append(f"forbidden_call:{rel}:{call_name}")
                call_leaf = call_name.rsplit(".", 1)[-1] if call_name else ""
                if (
                    call_leaf.endswith(self.policy.forbidden_mutating_call_suffix)
                    and not call_leaf.startswith("__")
                ):
                    issues.append(f"forbidden_mutating_call:{rel}:{call_name}")
                if call_leaf in self.policy.forbidden_mutating_methods:
                    issues.append(f"forbidden_mutating_call:{rel}:{call_name}")
                if call_leaf in self.policy.forbidden_random_calls:
                    issues.append(f"forbidden_random_call:{rel}:{call_name}")
        return sorted(set(issues))

    def _audit_import_from(self, rel: str, node: ast.ImportFrom) -> list[str]:
        if (
            (node.level != 0 and not self.policy.relative_imports_allowed)
            or node.module not in self.policy.allowed_imports
        ):
            return [f"forbidden_import:{rel}:{node.module or ''}"]
        allowed_names = self.policy.allowed_imports[node.module]
        issues = []
        for alias in node.names:
            if alias.name not in allowed_names:
                issues.append(f"forbidden_import_name:{rel}:{node.module}.{alias.name}")
            if alias.asname is not None and not self.policy.import_aliases_allowed:
                issues.append(f"forbidden_import_alias:{rel}:{alias.name}")
        return issues

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""


def candidate_change_set_from_verification(
    *,
    fallback: CandidateChangeSet,
    details: Mapping[str, Any],
) -> CandidateChangeSet:
    raw = details.get("derived_change_set")
    if not isinstance(raw, list | tuple) or not all(
        isinstance(item, str) for item in raw
    ):
        return fallback
    return CandidateChangeSet(frozenset(raw))


def _verified_artifact_matches(
    verification: VerificationResult,
    artifact: SealedCandidateArtifactRef,
) -> bool:
    identity = verification.program_identity
    return (
        identity is not None
        and identity.candidate_id == artifact.candidate_id
        and identity.manifest_hash == artifact.subject_manifest_hash
        and identity.source_checksum == artifact.program_source_checksum
        and identity.source_size_bytes == artifact.program_source_size_bytes
        and verification.candidate_artifact_hash == artifact.artifact_hash
    )


def static_verifier_prompt_rules(
    *,
    policy: StaticCandidateVerifierPolicy | None = None,
) -> dict[str, Any]:
    """Return the prompt-safe static rules enforced by StaticCandidateVerifier."""

    policy_facts = static_verifier_policy_facts(policy=policy)
    return {
        "schema_version": "true_rl_agent.static_candidate_verifier_prompt_rules.v1",
        "rule_source": "StaticCandidateVerifier",
        "verification_policy_id": STATIC_CANDIDATE_VERIFIER_POLICY_ID,
        "verification_policy_hash": canonical_json_hash(policy_facts),
        "candidate_package_contract_hash": CANDIDATE_PACKAGE_CONTRACT_HASH,
        "allowed_files": policy_facts["provider_authored_files"],
        "file_max_bytes": dict(CANDIDATE_PACKAGE_FILE_MAX_BYTES),
        "max_total_bytes": CANDIDATE_PACKAGE_MAX_BYTES,
        "allowed_imports": policy_facts["allowed_imports"],
        "blocked_text_fragments": [
            _prompt_safe_verifier_rule_text(item)
            for item in policy_facts["forbidden_tokens"]
        ],
        "blocked_names": [
            _prompt_safe_verifier_rule_text(item)
            for item in policy_facts["forbidden_names"]
        ],
        "blocked_name_prefixes": policy_facts["forbidden_name_prefixes"],
        "blocked_calls": policy_facts["forbidden_calls"],
        "blocked_call_roots": policy_facts["forbidden_call_roots"],
        "blocked_mutating_methods": policy_facts["forbidden_mutating_methods"],
        "blocked_mutating_call_suffix": policy_facts["forbidden_mutating_call_suffix"],
        "blocked_random_calls": policy_facts["forbidden_random_calls"],
        "blocked_state_attributes": policy_facts["forbidden_state_attributes"],
        "blocked_attribute_prefixes": policy_facts["forbidden_attribute_prefixes"],
        "blocked_ast_nodes": policy_facts["forbidden_ast_nodes"],
        "blocked_assignment_target_nodes": policy_facts[
            "forbidden_assignment_target_nodes"
        ],
        "allowed_top_level_ast_nodes": policy_facts["allowed_top_level_ast_nodes"],
        "allowed_schedule_keys": policy_facts["allowed_schedule_keys"],
        "function_decorators_allowed": policy_facts["function_decorators_allowed"],
        "nested_functions_allowed": policy_facts["nested_functions_allowed"],
        "mutable_or_dynamic_defaults_allowed": policy_facts[
            "mutable_or_dynamic_defaults_allowed"
        ],
        "additional_files_allowed": policy_facts["additional_files_allowed"],
        "import_aliases_allowed": policy_facts["import_aliases_allowed"],
        "relative_imports_allowed": policy_facts["relative_imports_allowed"],
        "component_callable_contract": policy_facts["component_callable_contract"],
        "component_load_timeout_s": policy_facts["component_load_timeout_s"],
    }


def static_verifier_source_policy_facts(
    *,
    policy: StaticCandidateVerifierPolicy | None = None,
) -> dict[str, Any]:
    """Return the task-neutral source rules shared by audit and qualification."""

    return (policy or StaticCandidateVerifierPolicy()).source_policy_facts()


def static_verifier_source_policy_hash(
    *,
    policy: StaticCandidateVerifierPolicy | None = None,
) -> str:
    """Hash source-purity rules independently of verifier runtime limits."""

    return canonical_json_hash(static_verifier_source_policy_facts(policy=policy))


def static_verifier_policy_facts(
    *,
    policy: StaticCandidateVerifierPolicy | None = None,
    component_load_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Return the executable facts for one static-verifier instance."""

    if policy is not None and component_load_timeout_s is not None:
        raise ValueError(
            "component_load_timeout_s is already owned by the supplied policy"
        )
    resolved_policy = policy or StaticCandidateVerifierPolicy(
        component_load_timeout_s=(
            StaticCandidateVerifier.DEFAULT_COMPONENT_LOAD_TIMEOUT_S
            if component_load_timeout_s is None
            else component_load_timeout_s
        )
    )
    return resolved_policy.executable_policy_facts()


def static_verifier_policy_hash(
    *,
    policy: StaticCandidateVerifierPolicy | None = None,
    component_load_timeout_s: float | None = None,
) -> str:
    """Hash one executable static admission policy instance."""

    return canonical_json_hash(
        static_verifier_policy_facts(
            policy=policy,
            component_load_timeout_s=component_load_timeout_s,
        )
    )


def audit_candidate_python_source(
    source: str,
    *,
    relative_path: str = "training_code/components.py",
    policy: StaticCandidateVerifierPolicy | None = None,
) -> tuple[str, ...]:
    """Apply the same AST policy to runner-owned reference source bytes."""

    if not isinstance(source, str):
        raise TypeError("candidate source must be text")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("candidate source relative_path is required")
    verifier = StaticCandidateVerifier(policy=policy)
    issues = [
        f"forbidden_token:{relative_path}:{token}"
        for token in verifier.policy.forbidden_tokens
        if token in source
    ]
    issues.extend(verifier._audit_python_source(relative_path, source))
    return tuple(sorted(set(issues)))


def _prompt_safe_verifier_rule_text(value: str) -> str:
    replacements = {
        "checkpoint_selector": "checkpoint policy selector surface",
        "eval_config": "eval configuration surface",
        "runner.": "runner attribute access",
        "env.": "environment attribute access",
        "success_metric": "success metric surface",
    }
    return replacements.get(value, value)


def _forbidden_ast_node_issues(rel: str, node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [f"forbidden_import:{rel}:import"]
    if isinstance(node, ast.Global):
        return [f"forbidden_global:{rel}:{','.join(node.names)}"]
    if isinstance(node, ast.Nonlocal):
        return [f"forbidden_nonlocal:{rel}:{','.join(node.names)}"]
    if isinstance(node, ast.AugAssign):
        return [f"forbidden_augmented_assignment:{rel}"]
    if isinstance(node, ast.Delete):
        return [f"forbidden_delete:{rel}"]
    if isinstance(node, ast.NamedExpr):
        return [f"forbidden_named_expression:{rel}"]
    return [f"forbidden_ast_node:{rel}:{type(node).__name__}"]


def _is_forbidden_assignment_target(
    node: ast.AST,
    forbidden_node_names: frozenset[str],
) -> bool:
    return type(node).__name__ in forbidden_node_names


def _has_mutable_or_dynamic_defaults(node: ast.FunctionDef) -> bool:
    defaults = list(node.args.defaults) + [
        value for value in node.args.kw_defaults if value is not None
    ]
    return any(not _is_immutable_default(value) for value in defaults)


def _is_immutable_default(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes, int, float, bool, type(None)))
    if isinstance(node, ast.Tuple):
        return all(_is_immutable_default(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value,
            (int, float),
        )
    return False


def _load_components_summary_with_timeout(
    *,
    candidate_dir: Path,
    allowed_schedule_keys: frozenset[str],
    worker_startup_timeout_s: float,
    component_load_timeout_s: float,
) -> dict[str, Any]:
    outcome = run_ready_worker(
        target=_component_summary_worker,
        args=(
            str(candidate_dir),
            tuple(sorted(allowed_schedule_keys)),
        ),
        startup_timeout_s=worker_startup_timeout_s,
        execution_timeout_s=component_load_timeout_s,
    )
    if not outcome.completed:
        return {"status": "timeout"}
    if isinstance(outcome.result, dict):
        return outcome.result
    return {"status": "fail", "error_type": "WorkerResultInvalid"}


def _component_summary_worker(
    candidate_dir: str,
    allowed_schedule_keys: tuple[str, ...],
    result_connection: Any,
    start_connection: Any,
) -> None:
    result_connection.send(WorkerReady())
    try:
        if start_connection.recv() != "start":
            return
    except EOFError:
        return
    try:
        candidate_path = Path(candidate_dir)
        manifest = load_candidate_manifest(candidate_path)
        components = load_training_components(candidate_path)
        allowed = frozenset(allowed_schedule_keys)
        result_connection.send(
            {
                "status": "pass",
                "component_issues": training_component_purity_issues(
                    components,
                    expected_module_name=(
                        f"true_rl_candidate_{manifest.candidate_id}"
                    ),
                    expected_source_name=str(
                        (
                            candidate_path / CANONICAL_TRAINING_COMPONENTS_PATH
                        ).resolve()
                    ),
                    allowed_schedule_keys=allowed,
                ),
                "is_noop": components.is_noop(),
                "derived_change_set": sorted(components.to_change_set().changed_components),
            }
        )
    except Exception as exc:  # noqa: BLE001 - verifier exposes only the public error type.
        result_connection.send(
            {"status": "fail", "error_type": type(exc).__name__}
        )
def training_component_purity_issues(
    components: Any,
    *,
    expected_module_name: str,
    expected_source_name: str,
    allowed_schedule_keys: frozenset[str] | None = None,
) -> list[str]:
    """Validate that component callables are stateless functions from exact source."""

    issues: list[str] = []
    if type(components) is not TrainingComponents:
        return ["invalid_training_components_type"]
    for field_name in ("reward_terms", "diagnostics", "schedules"):
        if type(getattr(components, field_name)) is not list:
            issues.append(f"invalid_component_container:{field_name}")
    if issues:
        return issues
    try:
        require_component_name(
            components.name,
            "TrainingComponents.name",
        )
    except ValueError:
        issues.append("invalid_training_components_name")
    for reward in components.reward_terms:
        if type(reward) is not RewardTerm:
            issues.append("invalid_reward_term_type")
            continue
        try:
            require_component_name(reward.name, "reward term name")
        except ValueError:
            issues.append("invalid_reward_term_name")
            continue
        if not callable(reward.fn):
            issues.append(f"non_callable_reward_term:{reward.name}")
            continue
        issues.extend(
            _component_callable_issues(
                reward.fn,
                kind="reward",
                name=reward.name,
                expected_module_name=expected_module_name,
                expected_source_name=expected_source_name,
            )
        )
    for diagnostic in components.diagnostics:
        if type(diagnostic) is not DiagnosticTerm:
            issues.append("invalid_diagnostic_term_type")
            continue
        try:
            require_component_name(diagnostic.name, "diagnostic term name")
        except ValueError:
            issues.append("invalid_diagnostic_term_name")
            continue
        if not callable(diagnostic.fn):
            issues.append(f"non_callable_diagnostic_term:{diagnostic.name}")
            continue
        issues.extend(
            _component_callable_issues(
                diagnostic.fn,
                kind="diagnostic",
                name=diagnostic.name,
                expected_module_name=expected_module_name,
                expected_source_name=expected_source_name,
            )
        )
    for schedule in components.schedules:
        try:
            require_component_name(schedule.name, "schedule term name")
        except ValueError:
            issues.append("invalid_schedule_term_name")
            continue
        if allowed_schedule_keys is not None and schedule.key not in allowed_schedule_keys:
            issues.append(f"schedule_key_not_allowlisted:{schedule.name}:{schedule.key}")
    return issues


def _component_callable_issues(
    fn: Any,
    *,
    kind: str,
    name: str,
    expected_module_name: str,
    expected_source_name: str,
) -> list[str]:
    if type(fn) is not FunctionType:
        return [f"non_source_defined_{kind}_term:{name}"]
    issues = []
    if fn.__module__ != expected_module_name or fn.__code__.co_filename != expected_source_name:
        issues.append(f"foreign_source_{kind}_term:{name}")
    if fn.__closure__ is not None:
        issues.append(f"stateful_{kind}_term_closure:{name}")
    parameters = tuple(inspect.signature(fn).parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].kind
        not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        or parameters[0].default is not inspect.Parameter.empty
    ):
        issues.append(f"invalid_{kind}_term_signature:{name}")
    return issues

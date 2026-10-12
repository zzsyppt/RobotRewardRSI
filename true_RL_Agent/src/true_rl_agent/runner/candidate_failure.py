"""Task-neutral contract for closed candidate-owned execution failures."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)

CANDIDATE_FAILURE_SCHEMA_VERSION = "true_rl_agent.candidate_failure.v1"
CANDIDATE_FAILURE_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.candidate_failure_contract.v1"
)

_CONTRACT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_INFRASTRUCTURE_ERROR_MARKERS = (
    "cuda error",
    "cuda out of memory",
    "cublas",
    "cudnn",
    "cusolver",
    "device-side assert",
    "driver shutting down",
    "hip error",
    "illegal memory access",
    "nccl",
    "out of memory",
)


@dataclass(frozen=True)
class CandidateFailureRule:
    """One task-declared failure stage and its closed reason vocabulary."""

    stage: str
    reason_codes: tuple[str, ...]
    requires_initial_policy_state: bool

    def __post_init__(self) -> None:
        _require_code(self.stage, "candidate failure stage")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or tuple(sorted(set(self.reason_codes))) != self.reason_codes
        ):
            raise ValueError(
                "candidate failure reason_codes must be a sorted unique tuple"
            )
        for reason_code in self.reason_codes:
            _require_code(reason_code, "candidate failure reason code")
        if not isinstance(self.requires_initial_policy_state, bool):
            raise TypeError(
                "candidate failure requires_initial_policy_state must be boolean"
            )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason_codes": list(self.reason_codes),
            "requires_initial_policy_state": self.requires_initial_policy_state,
        }


@dataclass(frozen=True)
class CandidateFailureContract:
    """A task-owned allowlist interpreted by the task-neutral Runner."""

    contract_id: str
    rules: tuple[CandidateFailureRule, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contract_id, str)
            or _CONTRACT_ID_PATTERN.fullmatch(self.contract_id) is None
        ):
            raise ValueError("candidate failure contract_id is invalid")
        if (
            not isinstance(self.rules, tuple)
            or not self.rules
            or any(not isinstance(rule, CandidateFailureRule) for rule in self.rules)
        ):
            raise TypeError(
                "candidate failure contract rules must be a non-empty tuple"
            )
        stages = tuple(rule.stage for rule in self.rules)
        if tuple(sorted(set(stages))) != stages:
            raise ValueError(
                "candidate failure contract rules must have sorted unique stages"
            )

    @property
    def contract_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def rule_for(self, *, stage: str, reason_code: str) -> CandidateFailureRule:
        _require_code(stage, "candidate failure stage")
        _require_code(reason_code, "candidate failure reason code")
        for rule in self.rules:
            if rule.stage == stage:
                if reason_code not in rule.reason_codes:
                    raise ValueError(
                        "candidate failure reason code is not allowed for its stage"
                    )
                return rule
        raise ValueError("candidate failure stage is not allowed by the contract")

    def evidence(
        self,
        *,
        stage: str,
        reason_code: str,
    ) -> "CandidateFailureEvidence":
        self.rule_for(stage=stage, reason_code=reason_code)
        return CandidateFailureEvidence(
            contract_hash=self.contract_hash,
            failure_stage=stage,
            reason_code=reason_code,
        )

    def validate_evidence(self, value: Any) -> "CandidateFailureEvidence":
        evidence = CandidateFailureEvidence.from_mapping(value)
        if evidence.contract_hash != self.contract_hash:
            raise ValueError("candidate failure contract hash mismatch")
        self.rule_for(
            stage=evidence.failure_stage,
            reason_code=evidence.reason_code,
        )
        return evidence

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "contract_hash": self.contract_hash,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "CandidateFailureContract":
        if not isinstance(value, Mapping):
            raise TypeError("candidate failure contract must be an object")
        expected_fields = {
            "schema_version",
            "contract_id",
            "rules",
            "contract_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("candidate failure contract fields mismatch")
        if value.get("schema_version") != CANDIDATE_FAILURE_CONTRACT_SCHEMA_VERSION:
            raise ValueError("candidate failure contract schema mismatch")
        raw_rules = value.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("candidate failure contract rules must be a non-empty list")
        rules: list[CandidateFailureRule] = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, Mapping):
                raise TypeError(
                    f"candidate failure contract rule {index} must be an object"
                )
            if set(raw_rule) != {
                "stage",
                "reason_codes",
                "requires_initial_policy_state",
            }:
                raise ValueError(
                    f"candidate failure contract rule {index} fields mismatch"
                )
            reason_codes = raw_rule.get("reason_codes")
            if not isinstance(reason_codes, list):
                raise TypeError(
                    f"candidate failure contract rule {index} reason_codes "
                    "must be a list"
                )
            rules.append(
                CandidateFailureRule(
                    stage=raw_rule.get("stage"),
                    reason_codes=tuple(reason_codes),
                    requires_initial_policy_state=raw_rule.get(
                        "requires_initial_policy_state"
                    ),
                )
            )
        contract = cls(
            contract_id=value.get("contract_id"),
            rules=tuple(rules),
        )
        if value.get("contract_hash") != contract.contract_hash:
            raise ValueError("candidate failure contract hash mismatch")
        return contract

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_FAILURE_CONTRACT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "rules": [rule.to_public_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class CandidateFailureEvidence:
    """Closed public-safe evidence emitted by a task-owned subprocess."""

    contract_hash: str
    failure_stage: str
    reason_code: str

    def __post_init__(self) -> None:
        require_sha256_digest(
            self.contract_hash,
            "candidate failure contract hash",
        )
        _require_code(self.failure_stage, "candidate failure stage")
        _require_code(self.reason_code, "candidate failure reason code")

    @classmethod
    def from_mapping(cls, value: Any) -> "CandidateFailureEvidence":
        if not isinstance(value, Mapping):
            raise TypeError("candidate failure evidence must be an object")
        expected_fields = {
            "schema_version",
            "contract_hash",
            "failure_owner",
            "failure_stage",
            "reason_code",
        }
        if set(value) != expected_fields:
            raise ValueError("candidate failure evidence fields mismatch")
        if value.get("schema_version") != CANDIDATE_FAILURE_SCHEMA_VERSION:
            raise ValueError("candidate failure evidence schema mismatch")
        if value.get("failure_owner") != "candidate":
            raise ValueError("candidate failure owner must be candidate")
        return cls(
            contract_hash=value.get("contract_hash"),
            failure_stage=value.get("failure_stage"),
            reason_code=value.get("reason_code"),
        )

    def to_public_dict(self) -> dict[str, str]:
        return {
            "schema_version": CANDIDATE_FAILURE_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "failure_owner": "candidate",
            "failure_stage": self.failure_stage,
            "reason_code": self.reason_code,
        }


def validate_candidate_runtime_failure_details(value: Any) -> dict[str, Any]:
    """Validate the only public detail shape a CandidateRuntimeError may carry."""

    if not isinstance(value, Mapping):
        raise TypeError("candidate runtime failure details must be an object")
    expected_fields = {
        "status",
        "subprocess_returncode",
        "candidate_failure_contract",
        "candidate_failure",
    }
    if set(value) - {"training_sampling"} != expected_fields:
        raise ValueError("candidate runtime failure detail fields mismatch")
    if value.get("status") != "fail":
        raise ValueError("candidate runtime failure status must be fail")
    returncode = value.get("subprocess_returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode <= 0:
        raise ValueError(
            "candidate runtime failure subprocess_returncode must be positive"
        )
    contract = CandidateFailureContract.from_mapping(
        value.get("candidate_failure_contract")
    )
    evidence = contract.validate_evidence(value.get("candidate_failure"))
    result = {
        "status": "fail",
        "subprocess_returncode": returncode,
        "candidate_failure_contract": contract.to_public_dict(),
        "candidate_failure": evidence.to_public_dict(),
    }
    if "training_sampling" in value:
        from true_rl_agent.evaluation.public_projection import project_public_training_sampling
        sampled = project_public_training_sampling(value["training_sampling"])
        if sampled != value["training_sampling"]:
            raise ValueError("candidate failure training sampling is not a public projection")
        result["training_sampling"] = sampled
    return result


def is_infrastructure_runtime_exception(exc: BaseException) -> bool:
    """Conservatively keep resource and operating-system failures out of denial."""

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (MemoryError, OSError)):
            return True
        error_type = type(current)
        qualified_name = f"{error_type.__module__}.{error_type.__name__}".lower()
        message = str(current).lower()
        if "outofmemoryerror" in qualified_name or any(
            marker in message for marker in _INFRASTRUCTURE_ERROR_MARKERS
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _require_code(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CODE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase code")
    return value


__all__ = [
    "CANDIDATE_FAILURE_CONTRACT_SCHEMA_VERSION",
    "CANDIDATE_FAILURE_SCHEMA_VERSION",
    "CandidateFailureContract",
    "CandidateFailureEvidence",
    "CandidateFailureRule",
    "is_infrastructure_runtime_exception",
    "validate_candidate_runtime_failure_details",
]

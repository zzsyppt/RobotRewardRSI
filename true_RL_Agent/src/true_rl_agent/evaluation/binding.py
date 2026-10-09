"""Task-neutral binding for one comparable evaluation domain."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import canonical_json_bytes

EVALUATION_BINDING_SCHEMA_VERSION = "true_rl_agent.evaluation.binding.v1"


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def require_sha256_digest(value: object, field_name: str) -> str:
    if not is_sha256_digest(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def is_sha256_checksum(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and is_sha256_digest(value.removeprefix("sha256:"))
    )


def require_sha256_checksum(value: object, field_name: str) -> str:
    if not is_sha256_checksum(value):
        raise ValueError(
            f"{field_name} must use sha256:<lowercase SHA-256 digest>"
        )
    assert isinstance(value, str)
    return value


@dataclass(frozen=True)
class EvaluationBindingV1:
    """Hashes that must match before two policy evaluations are comparable."""

    eval_protocol_hash: str
    acceptance_protocol_hash: str
    task_protocol_hash: str
    seed_cohort_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "task_protocol_hash",
            "seed_cohort_hash",
        ):
            require_sha256_digest(getattr(self, field_name), field_name)

    @classmethod
    def from_mapping(cls, value: Any) -> "EvaluationBindingV1":
        if not isinstance(value, Mapping):
            raise ValueError("evaluation binding must be an object")
        expected_keys = {
            "schema_version",
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "task_protocol_hash",
            "seed_cohort_hash",
            "evaluation_binding_hash",
        }
        if set(value) != expected_keys:
            raise ValueError("evaluation binding fields mismatch")
        if value.get("schema_version") != EVALUATION_BINDING_SCHEMA_VERSION:
            raise ValueError("evaluation binding schema_version mismatch")
        binding = cls(
            eval_protocol_hash=value.get("eval_protocol_hash"),
            acceptance_protocol_hash=value.get("acceptance_protocol_hash"),
            task_protocol_hash=value.get("task_protocol_hash"),
            seed_cohort_hash=value.get("seed_cohort_hash"),
        )
        if value.get("evaluation_binding_hash") != binding.binding_hash:
            raise ValueError("evaluation binding hash mismatch")
        return binding

    @property
    def binding_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def to_public_dict(self) -> dict[str, str]:
        return {
            **self._hash_payload(),
            "evaluation_binding_hash": self.binding_hash,
        }

    def _hash_payload(self) -> dict[str, str]:
        return {
            "schema_version": EVALUATION_BINDING_SCHEMA_VERSION,
            "eval_protocol_hash": self.eval_protocol_hash,
            "acceptance_protocol_hash": self.acceptance_protocol_hash,
            "task_protocol_hash": self.task_protocol_hash,
            "seed_cohort_hash": self.seed_cohort_hash,
        }


def attach_evaluation_binding(
    public_result: PublicEvalResult,
    *,
    binding: EvaluationBindingV1,
    subject: ExperimentSubject,
    subject_manifest_hash: str,
    program_source_checksum: str,
    training_binding_hash: str,
    capability_profile_hash: str,
) -> PublicEvalResult:
    """Attach the runner-owned comparison domain without allowing projection drift."""

    metrics = dict(public_result.metrics)
    expected_binding = binding.to_public_dict()
    expected_subject = subject.to_public_dict()
    for key, expected in (
        ("evaluation_binding", expected_binding),
        ("evaluation_subject", expected_subject),
        ("subject_manifest_hash", subject_manifest_hash),
        ("program_source_checksum", program_source_checksum),
        ("training_binding_hash", training_binding_hash),
        ("capability_profile_hash", capability_profile_hash),
    ):
        existing = metrics.get(key)
        if existing is not None and existing != expected:
            raise ValueError(f"public eval result {key} mismatch")
        metrics[key] = expected
    return PublicEvalResult(
        status=public_result.status,
        public_score=public_result.public_score,
        metrics=metrics,
    )


__all__ = [
    "EVALUATION_BINDING_SCHEMA_VERSION",
    "EvaluationBindingV1",
    "attach_evaluation_binding",
    "canonical_json_hash",
    "is_sha256_checksum",
    "is_sha256_digest",
    "require_sha256_checksum",
    "require_sha256_digest",
]

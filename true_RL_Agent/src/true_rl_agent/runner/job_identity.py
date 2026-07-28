"""Task-neutral identity binding for runner-owned job summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.evaluation.subject import ExperimentSubject

ALTERNATE_JOB_IDENTITY_FIELDS = (
    "candidate_id",
    "baseline_id",
    "reference_id",
    "candidate_manifest_hash",
    "baseline_manifest_hash",
    "reference_manifest_hash",
    "evaluation_subject",
    "subject",
    "evaluation_identity",
    "job_identity",
    "identity",
)


@dataclass(frozen=True)
class JobIdentity:
    """Sole identity shared by train/eval specs, subprocesses, and summaries."""

    job_id: str
    task_id: str
    subject: ExperimentSubject
    subject_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("job_id", "task_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"job identity {field_name} must be non-empty")
            if value != value.strip():
                raise ValueError(
                    f"job identity {field_name} must not contain surrounding whitespace"
                )
        if not isinstance(self.subject, ExperimentSubject):
            raise TypeError("job identity subject must be an ExperimentSubject")
        if self.subject.kind in {"baseline", "reference"}:
            require_sha256_digest(
                self.subject_manifest_hash,
                "job identity subject_manifest_hash",
            )
        elif self.subject_manifest_hash is not None:
            require_sha256_digest(
                self.subject_manifest_hash,
                "job identity subject_manifest_hash",
            )

    def to_public_dict(self) -> dict[str, str]:
        identity = {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "subject_kind": self.subject.kind,
            "subject_id": self.subject.subject_id,
        }
        if self.subject_manifest_hash is not None:
            identity["subject_manifest_hash"] = self.subject_manifest_hash
        return identity


def validate_job_summary_identity(
    summary: Any,
    *,
    expected: JobIdentity,
    summary_name: str,
) -> JobIdentity:
    """Require a summary to belong to the exact runner-owned job identity."""

    if not isinstance(summary, Mapping):
        raise ValueError(f"{summary_name} must be an object")
    for field_name, expected_value in expected.to_public_dict().items():
        actual = summary.get(field_name)
        if actual != expected_value:
            raise ValueError(
                f"{summary_name} binding mismatch for {field_name}: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    if (
        expected.subject_manifest_hash is None
        and "subject_manifest_hash" in summary
    ):
        raise ValueError(
            f"{summary_name} must not carry an unbound subject_manifest_hash"
        )
    for alternate_field in ALTERNATE_JOB_IDENTITY_FIELDS:
        if alternate_field in summary:
            raise ValueError(
                f"{summary_name} must not carry alternate job identity field "
                f"{alternate_field}"
            )
    return expected


__all__ = [
    "ALTERNATE_JOB_IDENTITY_FIELDS",
    "JobIdentity",
    "validate_job_summary_identity",
]

"""Baseline policy manifests and warm-start compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.subject import validate_experiment_subject_id
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.base import TaskSpec

BASELINE_MANIFEST_SCHEMA_VERSION = "true_rl_agent.baseline_manifest.v2"


@dataclass(frozen=True)
class CheckpointProvenance:
    source_run_id: str
    source_task_id: str
    created_by: str

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            self,
            ("source_run_id", "source_task_id", "created_by"),
            prefix="checkpoint provenance",
        )

    def to_public_dict(self) -> dict[str, str]:
        return {
            "source_run_id": self.source_run_id,
            "source_task_id": self.source_task_id,
            "created_by": self.created_by,
        }


@dataclass(frozen=True)
class BaselineCompatibilitySpec:
    backend_id: str
    robot: str
    observation_schema_hash: str
    action_schema_hash: str
    policy_architecture: str

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            self,
            (
                "backend_id",
                "robot",
                "observation_schema_hash",
                "action_schema_hash",
                "policy_architecture",
            ),
            prefix="baseline compatibility",
        )

    def to_public_dict(self) -> dict[str, str]:
        return {
            "backend_id": self.backend_id,
            "robot": self.robot,
            "observation_schema_hash": self.observation_schema_hash,
            "action_schema_hash": self.action_schema_hash,
            "policy_architecture": self.policy_architecture,
        }


@dataclass(frozen=True)
class BaselineSourceBinding:
    """Bind a runtime manifest to its task-owned source and admission roots."""

    source_schema_version: str
    source_manifest_hash: str
    trust_anchor_hash: str
    admission_evidence_hash: str

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            self,
            ("source_schema_version",),
            prefix="baseline source binding",
        )
        for field_name in (
            "source_manifest_hash",
            "trust_anchor_hash",
            "admission_evidence_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"baseline source binding {field_name}",
            )

    def to_public_dict(self) -> dict[str, str]:
        return {
            "source_schema_version": self.source_schema_version,
            "source_manifest_hash": self.source_manifest_hash,
            "trust_anchor_hash": self.trust_anchor_hash,
            "admission_evidence_hash": self.admission_evidence_hash,
        }


@dataclass(frozen=True)
class BaselineManifest:
    baseline_id: str
    task_id: str
    checkpoint: CheckpointRef
    provenance: CheckpointProvenance
    compatibility: BaselineCompatibilitySpec
    source_binding: BaselineSourceBinding | None = None
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        validate_experiment_subject_id(self.baseline_id)
        _require_nonempty_strings(self, ("task_id",), prefix="baseline")
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError("baseline checkpoint must be CheckpointRef")
        if not isinstance(self.provenance, CheckpointProvenance):
            raise TypeError(
                "baseline provenance must be CheckpointProvenance"
            )
        if not isinstance(self.compatibility, BaselineCompatibilitySpec):
            raise TypeError(
                "baseline compatibility must be BaselineCompatibilitySpec"
            )
        if self.source_binding is not None and not isinstance(
            self.source_binding,
            BaselineSourceBinding,
        ):
            raise TypeError(
                "baseline source_binding must be BaselineSourceBinding"
            )
        if (
            "://" not in self.checkpoint.uri
            or self.checkpoint.uri.lower().startswith("file://")
        ):
            raise ValueError(
                "baseline checkpoint uri must be a non-local artifact URI"
            )
        if self.checkpoint.role != "warm_start":
            raise ValueError("baseline checkpoint role must be warm_start")
        canonical_hash = canonical_json_hash(self._hash_payload())
        if self.manifest_hash is not None:
            require_sha256_digest(self.manifest_hash, "baseline manifest_hash")
            if self.manifest_hash != canonical_hash:
                raise ValueError("baseline manifest hash mismatch")
        object.__setattr__(self, "manifest_hash", canonical_hash)

    def to_policy_initialization(self) -> PolicyInitialization:
        return PolicyInitialization.warm_start(
            BaselinePolicyRef(
                baseline_id=self.baseline_id,
                task_id=self.task_id,
                checkpoint=self.checkpoint,
                compatibility_tags=frozenset(
                    {
                        self.compatibility.backend_id,
                        self.compatibility.robot,
                        self.compatibility.policy_architecture,
                    }
                ),
                baseline_manifest_hash=self.manifest_hash,
            )
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "manifest_hash": self.manifest_hash,
        }

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": BASELINE_MANIFEST_SCHEMA_VERSION,
            "baseline_id": self.baseline_id,
            "task_id": self.task_id,
            "checkpoint": self.checkpoint.to_public_dict(include_metadata=False),
            "provenance": self.provenance.to_public_dict(),
            "compatibility": self.compatibility.to_public_dict(),
            "source_binding": (
                self.source_binding.to_public_dict()
                if self.source_binding is not None
                else None
            ),
        }


@dataclass(frozen=True)
class BaselineCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]
    checkpoint_reference_status: str


def validate_baseline_compatibility(
    *,
    manifest: BaselineManifest,
    task_spec: TaskSpec,
    backend_spec: BackendSpec,
    robot: str,
    observation_schema_hash: str,
    action_schema_hash: str,
    policy_architecture: str,
) -> BaselineCompatibilityReport:
    reasons: list[str] = []

    if manifest.task_id != task_spec.task_id:
        reasons.append("task_id_mismatch")
    if not task_spec.supports_policy_initialization("warm_start"):
        reasons.append("task_does_not_support_warm_start")
    if task_spec.task_id not in backend_spec.supported_tasks:
        reasons.append("backend_does_not_support_task")
    if "warm_start" not in backend_spec.supported_policy_initializations:
        reasons.append("backend_does_not_support_warm_start")
    if not backend_spec.supports_baseline_eval:
        reasons.append("backend_does_not_support_baseline_eval")
    if manifest.compatibility.backend_id != backend_spec.backend_id:
        reasons.append("backend_id_mismatch")
    if manifest.compatibility.robot != robot:
        reasons.append("robot_mismatch")
    if manifest.compatibility.observation_schema_hash != observation_schema_hash:
        reasons.append("observation_schema_hash_mismatch")
    if manifest.compatibility.action_schema_hash != action_schema_hash:
        reasons.append("action_schema_hash_mismatch")
    if manifest.compatibility.policy_architecture != policy_architecture:
        reasons.append("policy_architecture_mismatch")

    reference_status, reference_reason = _validate_warm_start_checkpoint_reference(
        manifest.checkpoint
    )
    if reference_status != "pass":
        reasons.append(reference_reason)

    return BaselineCompatibilityReport(
        compatible=not reasons,
        reasons=tuple(reasons),
        checkpoint_reference_status=reference_status,
    )


def _validate_warm_start_checkpoint_reference(
    checkpoint: CheckpointRef,
) -> tuple[str, str]:
    if checkpoint.role != "warm_start":
        return "fail", "baseline_checkpoint_role_not_warm_start"
    if not checkpoint.sealed:
        return "fail", "baseline_checkpoint_not_sealed"
    checksum = checkpoint.checksum.removeprefix("sha256:")
    try:
        require_sha256_digest(checksum, "baseline checkpoint checksum")
    except ValueError:
        return "fail", "baseline_checkpoint_hash_missing"
    return "pass", "baseline_checkpoint_reference_valid"


def _require_nonempty_strings(
    value: object,
    field_names: tuple[str, ...],
    *,
    prefix: str,
) -> None:
    for field_name in field_names:
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{prefix} {field_name} must be non-empty")
        if field_value != field_value.strip():
            raise ValueError(
                f"{prefix} {field_name} must not contain surrounding whitespace"
            )

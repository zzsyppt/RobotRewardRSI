"""Baseline policy manifests and warm-start compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.base import TaskSpec


@dataclass(frozen=True)
class CheckpointProvenance:
    source_run_id: str
    source_task_id: str
    created_by: str

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

    def to_public_dict(self) -> dict[str, str]:
        return {
            "backend_id": self.backend_id,
            "robot": self.robot,
            "observation_schema_hash": self.observation_schema_hash,
            "action_schema_hash": self.action_schema_hash,
            "policy_architecture": self.policy_architecture,
        }


@dataclass(frozen=True)
class BaselineManifest:
    baseline_id: str
    task_id: str
    checkpoint: CheckpointRef
    provenance: CheckpointProvenance
    compatibility: BaselineCompatibilitySpec

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
            )
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "baseline_id": self.baseline_id,
            "task_id": self.task_id,
            "checkpoint": self.checkpoint.to_public_dict(),
            "provenance": self.provenance.to_public_dict(),
            "compatibility": self.compatibility.to_public_dict(),
        }


@dataclass(frozen=True)
class BaselineCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]
    safe_load_status: str


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

    safe_load_status, safe_load_reason = _safe_load_warm_start_checkpoint(manifest.checkpoint)
    if safe_load_status != "pass":
        reasons.append(safe_load_reason)

    return BaselineCompatibilityReport(
        compatible=not reasons,
        reasons=tuple(reasons),
        safe_load_status=safe_load_status,
    )


def _safe_load_warm_start_checkpoint(checkpoint: CheckpointRef) -> tuple[str, str]:
    if checkpoint.role != "warm_start":
        return "fail", "baseline_checkpoint_role_not_warm_start"
    if not checkpoint.sealed:
        return "fail", "baseline_checkpoint_not_runner_sealed"
    if not checkpoint.checksum.startswith("sha256:"):
        return "fail", "baseline_checkpoint_hash_missing"
    return "pass", "baseline_checkpoint_safe_to_load"


"""Task-neutral application composition contracts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.capability_readiness import (
    AuthoringCapabilityReadiness,
)
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin
from true_rl_agent.tasks.compatibility import check_task_backend_compatibility
from true_rl_agent.training_programs import (
    ReferenceProgramManifest,
    VerifiedReferenceProgramArtifact,
)


class TaskCompositionError(ValueError):
    """Raised before execution when an application composition is invalid."""


@dataclass(frozen=True)
class TaskCompositionContext:
    project_root: Path
    output_dir: Path
    read_only: bool = False


@dataclass(frozen=True)
class ResolvedTaskComposition:
    """Resolved task wiring with a stable, Genesis-only identity boundary."""

    task_id: str
    task_plugin: TaskPlugin
    task_memory_adapter: TaskMemoryAdapter
    backend_adapter: BackendAdapter
    runner: Runner
    candidate_admission_policy: CandidateVerifier
    policy_initialization: PolicyInitialization
    genesis_anchor_kind: str
    task_defaults: Mapping[str, Any] = field(default_factory=dict)
    scientific_parameters: Mapping[str, Any] = field(default_factory=dict)
    execution_placement: Mapping[str, Any] = field(default_factory=dict)
    baseline_manifest: BaselineManifest | None = None
    reference_program_artifact: VerifiedReferenceProgramArtifact | None = None
    bootstrap_initial_best_score: float | None = None
    bootstrap_initial_policy_state_hash: str | None = None
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None
    authoring_capability_readiness: AuthoringCapabilityReadiness = field(
        init=False,
    )

    def __post_init__(self) -> None:
        if self.task_plugin.task_spec.task_id != self.task_id:
            raise TaskCompositionError(
                "composition task_id does not match TaskPlugin task_spec"
            )
        if self.task_memory_adapter.task_id != self.task_id:
            raise TaskCompositionError(
                "composition task_id does not match task memory adapter"
            )
        if self.resolved_cell_manifest is not None:
            manifest = self.resolved_cell_manifest
            if not isinstance(manifest, ResolvedCellManifestV1):
                raise TaskCompositionError("composition cell manifest is invalid")
            if manifest.task_id != self.task_id:
                raise TaskCompositionError("composition cell manifest task_id mismatch")
            expected_task_protocol_hash = (
                manifest.base_task_protocol_hash
                if manifest.uses_legacy_training_binding_v1
                else manifest.resolved_cell_task_protocol_hash
            )
            if expected_task_protocol_hash != (
                self.task_plugin.task_spec.resolved_task_protocol_hash()
            ):
                raise TaskCompositionError(
                    "composition cell manifest task protocol mismatch"
                )
        expected_anchor_kind = self.task_plugin.task_spec.genesis_anchor_kind_for(
            self.policy_initialization.kind
        )
        if self.genesis_anchor_kind != expected_anchor_kind:
            raise TaskCompositionError(
                "composition Genesis anchor kind does not match TaskSpec policy"
            )
        report = check_task_backend_compatibility(
            task_plugin=self.task_plugin,
            backend_adapter=self.backend_adapter,
            policy_initialization=self.policy_initialization.kind,
        )
        if not report.compatible:
            raise TaskCompositionError(
                "task/backend compatibility failed: " + ", ".join(report.reasons)
            )
        if not isinstance(self.candidate_admission_policy, CandidateVerifier):
            raise TaskCompositionError(
                "composition candidate admission policy is invalid"
            )
        admission = (
            self.candidate_admission_policy.admission_policy_descriptor()
        )
        if self.backend_adapter.candidate_admission_policy_identity != (
            admission.policy_id,
            admission.policy_hash,
        ):
            raise TaskCompositionError(
                "composition admission policy does not match backend"
            )
        if (
            self.backend_adapter.candidate_execution_authorization_schema
            != admission.execution_authorization_schema_version
        ):
            raise TaskCompositionError(
                "composition execution authorization schema does not match backend"
            )
        smoke_adapter = self.task_plugin.component_smoke_adapter()
        if smoke_adapter.task_id != self.task_id:
            raise TaskCompositionError(
                "composition task smoke adapter task_id mismatch"
            )
        smoke_capability_ids = smoke_adapter.candidate_capability_ids
        readiness = AuthoringCapabilityReadiness.derive(
            profile=self.task_plugin.task_spec.candidate_capability_profile,
            admission_policy_id=admission.policy_id,
            admission_policy_hash=admission.policy_hash,
            admission_capability_ids=admission.capability_ids,
            materializer_runtime_capability_ids=(
                self.backend_adapter.candidate_execution_capability_ids
            ),
            task_smoke_capability_ids=smoke_capability_ids,
        )
        try:
            readiness.require_selected_capabilities_ready()
        except ValueError as exc:
            raise TaskCompositionError(str(exc)) from exc
        object.__setattr__(self, "authoring_capability_readiness", readiness)
        if self.baseline_manifest is not None:
            if self.baseline_manifest.task_id != self.task_id:
                raise TaskCompositionError(
                    "composition baseline manifest task_id mismatch"
                )
            if not self.backend_adapter.backend_spec.supports_baseline_eval:
                raise TaskCompositionError(
                    "composition backend does not support baseline eval"
                )
        if self.reference_program_artifact is not None:
            if not isinstance(
                self.reference_program_artifact,
                VerifiedReferenceProgramArtifact,
            ):
                raise TaskCompositionError(
                    "composition reference program is not a verified artifact"
                )
            manifest = self.reference_program_artifact.manifest
            if manifest.task_id != self.task_id:
                raise TaskCompositionError(
                    "composition reference program task_id mismatch"
                )
            if (
                manifest.capability_profile_hash
                != self.task_plugin.task_spec.candidate_capability_profile.profile_hash
            ):
                raise TaskCompositionError(
                    "composition reference program capability profile mismatch"
                )
        if self.policy_initialization.kind == "warm_start":
            if self.baseline_manifest is None:
                raise TaskCompositionError(
                    "warm_start composition requires a baseline manifest"
                )
            initialization_baseline = self.policy_initialization.baseline
            if initialization_baseline is None:
                raise TaskCompositionError(
                    "warm_start composition initialization baseline is missing"
                )
            if (
                initialization_baseline.baseline_id
                != self.baseline_manifest.baseline_id
                or initialization_baseline.task_id
                != self.baseline_manifest.task_id
                or initialization_baseline.checkpoint
                != self.baseline_manifest.checkpoint
                or initialization_baseline.baseline_manifest_hash
                != self.baseline_manifest.manifest_hash
            ):
                raise TaskCompositionError(
                    "warm_start initialization does not match baseline manifest"
                )
        if self.genesis_anchor_kind == "bootstrap":
            if self.policy_initialization.kind != "random":
                raise TaskCompositionError(
                    "bootstrap composition requires random initialization"
                )
            if (
                self.baseline_manifest is not None
                or self.reference_program_artifact is not None
            ):
                raise TaskCompositionError(
                    "bootstrap composition cannot carry a baseline or reference manifest"
                )
            if self.bootstrap_initial_best_score is not None:
                score = self.bootstrap_initial_best_score
                if (
                    isinstance(score, bool)
                    or not isinstance(score, int | float)
                    or not math.isfinite(float(score))
                ):
                    raise TaskCompositionError(
                        "composition bootstrap initial best score must be finite"
                    )
            if self.bootstrap_initial_policy_state_hash is not None:
                if self.bootstrap_initial_best_score is None:
                    raise TaskCompositionError(
                        "composition bootstrap initial policy hash requires a score"
                    )
                require_sha256_digest(
                    self.bootstrap_initial_policy_state_hash,
                    "composition bootstrap initial policy state hash",
                )
        elif self.genesis_anchor_kind == "policy_baseline":
            if (
                self.bootstrap_initial_best_score is not None
                or self.bootstrap_initial_policy_state_hash is not None
            ):
                raise TaskCompositionError(
                    "policy-baseline composition cannot carry bootstrap state"
                )
            if self.policy_initialization.kind != "warm_start":
                raise TaskCompositionError(
                    "policy-baseline composition requires warm_start initialization"
                )
            if self.baseline_manifest is None:
                raise TaskCompositionError(
                    "policy-baseline composition requires a baseline manifest"
                )
            if self.reference_program_artifact is not None:
                raise TaskCompositionError(
                    "policy-baseline composition cannot carry a reference manifest"
                )
        elif self.genesis_anchor_kind == "reference_program":
            if (
                self.bootstrap_initial_best_score is not None
                or self.bootstrap_initial_policy_state_hash is not None
            ):
                raise TaskCompositionError(
                    "reference composition cannot carry bootstrap state"
                )
            if self.policy_initialization.kind != "random":
                raise TaskCompositionError(
                    "reference-program composition requires random initialization"
                )
            if self.reference_program_artifact is None:
                raise TaskCompositionError(
                    "reference-program composition requires a verified reference artifact"
                )
            if self.baseline_manifest is not None:
                raise TaskCompositionError(
                    "reference-program composition cannot carry a baseline manifest"
                )
        else:
            raise TaskCompositionError("composition Genesis anchor kind is invalid")
        for name in ("task_defaults", "scientific_parameters", "execution_placement"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def reference_program_manifest(self) -> ReferenceProgramManifest | None:
        artifact = self.reference_program_artifact
        return artifact.manifest if artifact is not None else None

    def genesis_identity_snapshot(self) -> Mapping[str, Any]:
        """Return the immutable Phase E execution identity bound into Genesis.

        Candidate admission and authoring readiness are deliberately excluded:
        they are validated independently when a live authoring loop starts and
        may evolve without changing an already sealed reference anchor.
        """

        snapshot = {
            "task_id": self.task_id,
            "task_protocol_hash": (
                self.task_plugin.task_spec.resolved_task_protocol_hash()
            ),
            "genesis_anchor_kind": self.genesis_anchor_kind,
            "candidate_capability_profile": (
                self.task_plugin.task_spec.candidate_capability_profile.to_public_dict()
            ),
            "fixed_train_protocol": dict(self.task_plugin.task_spec.fixed_train_protocol),
            "fixed_eval_protocol": dict(self.task_plugin.task_spec.fixed_eval_protocol),
            "parameters": dict(self.scientific_parameters),
        }
        if self.resolved_cell_manifest is not None:
            manifest = self.resolved_cell_manifest
            snapshot["resolved_cell"] = {
                "manifest_hash": manifest.manifest_hash,
                "identity": manifest.summary_identity(),
                "resolved_cell_task_protocol_hash": (
                    manifest.resolved_cell_task_protocol_hash
                ),
            }
        if self.reference_program_artifact is not None:
            manifest = self.reference_program_artifact.manifest
            snapshot["reference_program"] = {
                "reference_id": manifest.reference_id,
                "manifest_hash": manifest.manifest_hash,
                "source_checksum": manifest.source_checksum,
                "source_size_bytes": manifest.source_size_bytes,
                "verification_hash": (
                    self.reference_program_artifact.verification_hash
                ),
            }
        if self.bootstrap_initial_policy_state_hash is not None:
            snapshot["bootstrap_initial_policy_state_hash"] = (
                self.bootstrap_initial_policy_state_hash
            )
        if self.bootstrap_initial_best_score is not None:
            snapshot["bootstrap_initial_best_score"] = float(
                self.bootstrap_initial_best_score
            )
        return MappingProxyType({
            "schema_version": "true_rl_agent.runtime_composition_snapshot.v3",
            "scientific": snapshot,
            "execution_placement": dict(self.execution_placement),
        })


class TaskCompositionProvider(ABC):
    """Task-owned factory selected by the application composition root."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def cli_defaults(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def add_cli_arguments(self, parser: Any) -> None:
        """Allow a task provider to expose task-owned options without shared semantics."""

        _ = parser

    @abstractmethod
    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        raise NotImplementedError

    def run_audit(
        self,
        *,
        composition: ResolvedTaskComposition,
        output_dir: Path,
        rounds: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        _ = composition
        raise TaskCompositionError(
            f"task does not provide a dry-run audit capability: {self.task_id}"
        )


def resolve_task_composition(
    *,
    provider: TaskCompositionProvider,
    values: Mapping[str, Any],
    context: TaskCompositionContext,
) -> ResolvedTaskComposition:
    """Resolve once and bind the returned snapshot to the selected provider."""

    composition = provider.resolve(values=values, context=context)
    if not isinstance(composition, ResolvedTaskComposition):
        raise TaskCompositionError(
            "task composition provider must return ResolvedTaskComposition"
        )
    if composition.task_id != provider.task_id:
        raise TaskCompositionError(
            "resolved composition task_id does not match selected provider"
        )
    return composition

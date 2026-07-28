"""Task-neutral application composition contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin
from true_rl_agent.tasks.compatibility import check_task_backend_compatibility
from true_rl_agent.training_programs import ReferenceProgramManifest


class TaskCompositionError(ValueError):
    """Raised before execution when an application composition is invalid."""


@dataclass(frozen=True)
class TaskCompositionContext:
    project_root: Path
    output_dir: Path


@dataclass(frozen=True)
class ResolvedTaskComposition:
    """One immutable runtime snapshot selected by task ID."""

    task_id: str
    task_plugin: TaskPlugin
    task_memory_adapter: TaskMemoryAdapter
    backend_adapter: BackendAdapter
    runner: Runner
    policy_initialization: PolicyInitialization
    genesis_anchor_kind: str
    task_defaults: Mapping[str, Any] = field(default_factory=dict)
    baseline_manifest: BaselineManifest | None = None
    reference_program_manifest: ReferenceProgramManifest | None = None

    def __post_init__(self) -> None:
        if self.task_plugin.task_spec.task_id != self.task_id:
            raise TaskCompositionError(
                "composition task_id does not match TaskPlugin task_spec"
            )
        if self.task_memory_adapter.task_id != self.task_id:
            raise TaskCompositionError(
                "composition task_id does not match task memory adapter"
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
        if self.baseline_manifest is not None:
            if self.baseline_manifest.task_id != self.task_id:
                raise TaskCompositionError(
                    "composition baseline manifest task_id mismatch"
                )
            if not self.backend_adapter.backend_spec.supports_baseline_eval:
                raise TaskCompositionError(
                    "composition backend does not support baseline eval"
                )
        if self.reference_program_manifest is not None:
            manifest = self.reference_program_manifest
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
            if self.baseline_manifest is not None or self.reference_program_manifest is not None:
                raise TaskCompositionError(
                    "bootstrap composition cannot carry a baseline or reference manifest"
                )
        elif self.genesis_anchor_kind == "policy_baseline":
            if self.policy_initialization.kind != "warm_start":
                raise TaskCompositionError(
                    "policy-baseline composition requires warm_start initialization"
                )
            if self.baseline_manifest is None:
                raise TaskCompositionError(
                    "policy-baseline composition requires a baseline manifest"
                )
            if self.reference_program_manifest is not None:
                raise TaskCompositionError(
                    "policy-baseline composition cannot carry a reference manifest"
                )
        elif self.genesis_anchor_kind == "reference_program":
            if self.policy_initialization.kind != "random":
                raise TaskCompositionError(
                    "reference-program composition requires random initialization"
                )
            if self.reference_program_manifest is None:
                raise TaskCompositionError(
                    "reference-program composition requires a reference manifest"
                )
            if self.baseline_manifest is not None:
                raise TaskCompositionError(
                    "reference-program composition cannot carry a baseline manifest"
                )
        else:
            raise TaskCompositionError("composition Genesis anchor kind is invalid")
        object.__setattr__(
            self,
            "task_defaults",
            MappingProxyType(dict(self.task_defaults)),
        )

    def runtime_snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": "true_rl_agent.runtime_composition_snapshot.v1",
                "task_id": self.task_id,
                "task_protocol_hash": (
                    self.task_plugin.task_spec.resolved_task_protocol_hash()
                ),
                "genesis_anchor_kind": self.genesis_anchor_kind,
                "candidate_capability_profile": (
                    self.task_plugin.task_spec.candidate_capability_profile.to_public_dict()
                ),
                "task_defaults": dict(self.task_defaults),
            }
        )


class TaskCompositionProvider(ABC):
    """Task-owned factory selected by the application composition root."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def cli_defaults(self) -> Mapping[str, Any]:
        raise NotImplementedError

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

"""Resolved execution dependencies shared by task-specific session assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from true_rl_agent.agent_core.generation_contract import (
    GenerationSessionContract, GenerationSessionContractRef,
)
from true_rl_agent.agent_core.generation_cost import GenerationCostContract
from true_rl_agent.agent_core.completed_reference_training import VerifiedCompletedReferenceTraining
from true_rl_agent.agent_core.generation_lifecycle import GenerationContinuationPolicy
from true_rl_agent.composition import ResolvedTaskComposition
from true_rl_agent.evaluation.phased_adapter import PhasedEvaluationAdapter
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


class GenerationSessionLocation(Protocol):
    session_id: str
    output_dir: Path
    gpu0_uuid: str
    gpu1_uuid: str
    nvidia_smi_executable: str


@dataclass(frozen=True)
class GenerationSessionFoundation:
    config: GenerationSessionLocation
    source_commit: str
    composition: ResolvedTaskComposition
    runner: SubprocessRunner
    phased_adapter: PhasedEvaluationAdapter
    continuation_policy: GenerationContinuationPolicy
    cost_contract: GenerationCostContract
    session_contract: GenerationSessionContract
    session_contract_ref: GenerationSessionContractRef
    gpu_identities: tuple[GpuDeviceIdentity, GpuDeviceIdentity]
    gpu_bindings: tuple[GpuProcessBinding, GpuProcessBinding]
    gpu_lease_manager: GpuLeaseManager
    provider_policy: Mapping[str, Any]
    final_confirmation_policy: Mapping[str, Any]
    # The task assembles and interprets its final experiment. The execution
    # foundation carries it without interpreting terrain, scores or seed sets.
    final_confirmation_plan: Any | None
    completed_reference_training: VerifiedCompletedReferenceTraining | None = None

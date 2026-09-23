"""Task-owned projections consumed by the shared generation lifecycle.

Screens and selections keep their task schemas. The host owns phase execution
and immutable receipts; it does not interpret score or robot semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, TYPE_CHECKING

from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import EvalJobSpec
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.evaluation.public_accept_scores import PublicAcceptScores

if TYPE_CHECKING:
    from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1


@dataclass(frozen=True)
class GenerationWorkload:
    """Task-owned work units for two dev trials and one confirmed selection.

    Counts describe executed jobs. Retry and wall-time/output reserves belong
    to the session's separate physical cost contract.
    """

    task_protocol_hash: str
    phased_evaluation_cost_contract_hash: str
    train_steps_per_job: int
    public_dev_seed_units_per_job: int
    public_accept_seed_units_per_job: int
    sealed_validation_seed_units_per_job: int
    max_eval_control_steps_per_seed_unit: int

    def __post_init__(self) -> None:
        for name in ("task_protocol_hash", "phased_evaluation_cost_contract_hash"):
            require_sha256_digest(getattr(self, name), name)
        for name in self.__dataclass_fields__:
            if name.endswith("_hash"):
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"generation workload {name} must be positive")

    @property
    def train_jobs_per_generation(self) -> int:
        return 2

    @property
    def reuses_reference_public_dev_receipt(self) -> bool:
        return True

    @property
    def reference_eval_seed_units(self) -> int:
        return sum(self.seed_units_for_phase(phase.value) for phase in EvaluationPhase)

    @property
    def eval_seed_units_per_generation(self) -> int:
        return self.public_dev_seed_units_per_job + self.reference_eval_seed_units

    def seed_units_for_phase(self, phase: str) -> int:
        return {
            "public_dev": self.public_dev_seed_units_per_job,
            "public_accept": self.public_accept_seed_units_per_job,
            "sealed_validation": self.sealed_validation_seed_units_per_job,
        }[phase]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.generation_workload.v1",
            "topology": "two_dev_selected_accept_selected_sealed",
            **asdict(self),
            "train_jobs_per_generation": self.train_jobs_per_generation,
            "reference_eval_seed_units": self.reference_eval_seed_units,
            "eval_seed_units_per_generation": self.eval_seed_units_per_generation,
        }


class PublicDevScreen(Protocol):
    phase_result_hash: str

    @property
    def evidence_hash(self) -> str: ...

    def to_selector_dict(self) -> dict[str, Any]: ...


class GenerationSelection(Protocol):
    selected_trial_id: str | None
    reason: str

    def to_public_dict(self) -> dict[str, Any]: ...


class ReferencePhasedEvaluationAdapter(Protocol):
    """Task-owned admission and assembly for a split reference evaluation."""

    def project_reference_public_dev_gate(
        self, *, phase_result: EvaluationPhaseResult,
    ) -> Mapping[str, Any] | None: ...

    def assemble_reference_full_summary(
        self,
        *,
        eval_plan: EvalJobSpec,
        phase_results: Sequence[EvaluationPhaseResult],
        phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
    ) -> EvaluationAssembly: ...


class PhasedEvaluationAdapter(ReferencePhasedEvaluationAdapter, Protocol):
    """The semantic capability required for a task to run generations."""

    @property
    def task_protocol_hash(self) -> str: ...

    @property
    def parent_public_dev_source(self) -> Literal["genesis_receipt", "evaluate"]: ...

    def selection_contract(self) -> Mapping[str, Any]: ...

    def cost_contract(self) -> Mapping[str, Any]: ...

    def generation_workload(
        self, *, cell_manifest: "ResolvedCellManifestV1",
    ) -> GenerationWorkload: ...

    def project_public_dev_screen(
        self,
        *,
        trial_id: str,
        phase_result: EvaluationPhaseResult,
        screen_role: str,
        train_result: TrainJobResult | None,
        train_receipt_checksum: str | None,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> PublicDevScreen: ...

    def project_public_phase_summary(
        self, *, phase_result: EvaluationPhaseResult
    ) -> Mapping[str, Any]: ...

    def project_public_accept_scores(
        self, *, phase_result: EvaluationPhaseResult,
    ) -> PublicAcceptScores | None: ...

    def select(
        self,
        screens: Sequence[PublicDevScreen],
        *,
        guard_reference: PublicDevScreen,
    ) -> GenerationSelection: ...

    def assemble_full_summary(
        self,
        *,
        eval_plan: EvalJobSpec,
        trial_id: str,
        selection_result: GenerationSelection,
        selection_screens: Sequence[PublicDevScreen],
        guard_reference: PublicDevScreen,
        phase_results: Sequence[EvaluationPhaseResult],
        phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
    ) -> EvaluationAssembly: ...

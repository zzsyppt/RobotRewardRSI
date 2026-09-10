"""Execution-facing cost capabilities, independent of task/version classes."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phased_adapter import GenerationWorkload


class GenerationScientificCost(Protocol):
    train_steps_per_job: int
    train_jobs_per_generation: int
    reuses_reference_public_dev_receipt: bool
    max_eval_control_steps_per_seed_unit: int
    reference_eval_seed_units: int
    eval_seed_units_per_generation: int

    def seed_units_for_phase(self, phase: str) -> int: ...


class GenerationCostContract(Protocol):
    """Already frozen cost authority consumed by admission and accounting."""

    contract_hash: str
    max_generations: int
    scientific_cost_basis: GenerationScientificCost
    imports_reference_evidence: bool
    imported_reference_timeout_envelope: tuple[int, int] | None
    final_confirmation_seed_units: int
    finalization_output_reserve_bytes: int
    max_allocated_active_gpu_hours: float
    max_allocated_gpu_capacity_ns: int
    max_output_bytes: int
    max_simulated_control_steps: int
    physical_train_job_cap: int
    physical_train_step_cap: int
    physical_eval_seed_unit_cap: int
    uses_operation_reserves: bool
    reference_phase_reserve_rule: str
    actual_cost_schema_version: str
    operational_evidence_namespace: str
    retry_train_jobs: int
    retry_eval_seed_units: int
    max_infrastructure_retries_per_eligible_job: int

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OperationCostReserve:
    """Use the frozen allocation, including any explicit timeout override."""

    operation_class: str
    budget_class: str
    multiplicity: int
    source_allocated_gpu_capacity_ns: int
    source_output_bytes: int
    allocated_gpu_capacity_ns_per_attempt: int
    output_bytes_per_attempt: int
    allocated_gpu_capacity_authority: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperationCostReserve":
        return cls(
            operation_class=value["operation_class"],
            budget_class=value["budget_class"],
            multiplicity=value["multiplicity"],
            source_allocated_gpu_capacity_ns=value["source_allocated_gpu_capacity_ns_per_attempt"],
            source_output_bytes=value["source_output_bytes_per_attempt"],
            allocated_gpu_capacity_ns_per_attempt=value["allocated_gpu_capacity_ns_per_attempt"],
            output_bytes_per_attempt=value["output_bytes_per_attempt"],
            allocated_gpu_capacity_authority=value.get("allocated_gpu_capacity_authority"),
        )

    @property
    def allocated_gpu_capacity_ns(self) -> int:
        return self.multiplicity * self.allocated_gpu_capacity_ns_per_attempt

    @property
    def output_bytes(self) -> int:
        return self.multiplicity * self.output_bytes_per_attempt

    def to_dict(self) -> dict[str, Any]:
        value = {
            "operation_class": self.operation_class,
            "budget_class": self.budget_class,
            "multiplicity": self.multiplicity,
            "source_allocated_gpu_capacity_ns_per_attempt": self.source_allocated_gpu_capacity_ns,
            "source_output_bytes_per_attempt": self.source_output_bytes,
            "allocated_gpu_capacity_ns_per_attempt": self.allocated_gpu_capacity_ns_per_attempt,
            "output_bytes_per_attempt": self.output_bytes_per_attempt,
            "allocated_gpu_capacity_ns": self.allocated_gpu_capacity_ns,
            "output_bytes": self.output_bytes,
        }
        if self.allocated_gpu_capacity_authority is not None:
            value["allocated_gpu_capacity_authority"] = self.allocated_gpu_capacity_authority
        return value

    @classmethod
    def from_source(
        cls, *, operation_class: str, budget_class: str, multiplicity: int,
        source_allocated_gpu_capacity_ns: int, source_output_bytes: int,
        headroom_numerator: int, headroom_denominator: int,
    ) -> "OperationCostReserve":
        def reserve(value: int) -> int:
            return (value * headroom_numerator + headroom_denominator - 1) // headroom_denominator

        return cls(operation_class, budget_class, multiplicity,
            source_allocated_gpu_capacity_ns, source_output_bytes,
            reserve(source_allocated_gpu_capacity_ns), reserve(source_output_bytes))


@dataclass(frozen=True)
class GenerationProviderBudget:
    attempts_per_lane_per_generation: int
    input_units_per_attempt_reserve: int
    output_units_per_attempt_cap: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"provider budget {name} must be positive")

    def for_generations(self, count: int) -> dict[str, Any]:
        attempts = 2 * count * self.attempts_per_lane_per_generation
        input_units = attempts * self.input_units_per_attempt_reserve
        output_units = attempts * self.output_units_per_attempt_cap
        return {
            **asdict(self),
            "generation_width": 2,
            "attempt_cap": attempts,
            "input_unit_cap": input_units,
            "output_unit_cap": output_units,
            "total_unit_cap": input_units + output_units,
        }


def generation_operation_layout(
    *, max_generations: int, imports_reference: bool, final_jobs: int,
) -> dict[str, tuple[str, int]]:
    layout = {
        "lane_a_train": ("successful_session", max_generations),
        "lane_b_train": ("successful_session", max_generations),
        "lane_a_public_dev": ("successful_session", max_generations),
        "lane_b_public_dev": ("successful_session", max_generations),
        "selected_public_accept": ("successful_session", max_generations),
        "sealed_validation": ("successful_session", max_generations + final_jobs),
        "sealed_validation_retry": ("retry_reserve", 1),
    }
    if not imports_reference:
        layout.update({
            "reference_train": ("successful_session", 1),
            "reference_qualification": ("successful_session", 1),
            "reference_train_retry": ("retry_reserve", 1),
        })
    return layout


@dataclass(frozen=True)
class FrozenGenerationCostContract:
    """One new-writer cost shape, with no task or historical-version dispatch.

    The supplied operation entries retain their measured/timeout source and
    headroom. Scientific work, reserved capacity and actual usage stay separate.
    Historical cost readers retain their original schemas and arithmetic.
    """

    max_generations: int
    scientific_cost_basis: GenerationWorkload
    provider_budget: GenerationProviderBudget
    operation_reserves: tuple[OperationCostReserve, ...]
    imports_reference_evidence: bool
    final_confirmation_seed_units: int
    finalization_output_reserve_bytes: int
    session_output_reserve_bytes: int
    headroom_numerator: int
    headroom_denominator: int
    imported_reference_timeout_envelope: tuple[int, int] | None = None

    uses_operation_reserves = True
    reference_phase_reserve_rule = "apportion_frozen_totals"
    actual_cost_schema_version = "true_rl_agent.generation_actual_cost.v1"
    operational_evidence_namespace = "true_rl_agent.generation"
    max_infrastructure_retries_per_eligible_job = 1

    def __post_init__(self) -> None:
        for name in ("max_generations", "finalization_output_reserve_bytes", "session_output_reserve_bytes",
                     "headroom_numerator", "headroom_denominator"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"generation cost {name} must be positive")
        if self.headroom_numerator < self.headroom_denominator:
            raise ValueError("operation headroom cannot reduce its source reservation")
        expected_final = 2 * self.scientific_cost_basis.sealed_validation_seed_units_per_job
        if self.final_confirmation_seed_units not in (0, expected_final):
            raise ValueError("final work must match the paired phase jobs")
        if not isinstance(self.imports_reference_evidence, bool):
            raise TypeError("reference import mode must be boolean")
        expected = generation_operation_layout(
            max_generations=self.max_generations,
            imports_reference=self.imports_reference_evidence,
            final_jobs=2 if self.final_confirmation_seed_units else 0,
        )
        actual = {entry.operation_class: (entry.budget_class, entry.multiplicity)
                  for entry in self.operation_reserves}
        if actual != expected or len(actual) != len(self.operation_reserves):
            raise ValueError("operation reserve entries differ from the session topology")
        for entry in self.operation_reserves:
            if min(entry.source_allocated_gpu_capacity_ns, entry.source_output_bytes,
                   entry.allocated_gpu_capacity_ns_per_attempt, entry.output_bytes_per_attempt) <= 0:
                raise ValueError("operation resource amounts must be positive")

    @classmethod
    def freeze(
        cls, *, max_generations: int, workload: GenerationWorkload,
        provider_budget: GenerationProviderBudget,
        operation_sources: Mapping[str, OperationCostReserve],
        imports_reference: bool, final_confirmation: bool,
        finalization_output_reserve_bytes: int,
        session_output_reserve_bytes: int,
        headroom_numerator: int, headroom_denominator: int,
        imported_reference_timeout_envelope: tuple[int, int] | None = None,
    ) -> "FrozenGenerationCostContract":
        layout = generation_operation_layout(
            max_generations=max_generations, imports_reference=imports_reference,
            final_jobs=2 if final_confirmation else 0,
        )
        if set(operation_sources) != set(layout):
            raise ValueError("operation sources must match the selected session work")
        return cls(
            max_generations=max_generations, scientific_cost_basis=workload,
            provider_budget=provider_budget,
            operation_reserves=tuple(
                replace(operation_sources[name], operation_class=name,
                        budget_class=budget_class, multiplicity=count)
                for name, (budget_class, count) in sorted(layout.items())
            ),
            imports_reference_evidence=imports_reference,
            final_confirmation_seed_units=(
                2 * workload.sealed_validation_seed_units_per_job
                if final_confirmation else 0
            ),
            finalization_output_reserve_bytes=finalization_output_reserve_bytes,
            session_output_reserve_bytes=session_output_reserve_bytes,
            headroom_numerator=headroom_numerator,
            headroom_denominator=headroom_denominator,
            imported_reference_timeout_envelope=imported_reference_timeout_envelope,
        )

    @property
    def retry_train_jobs(self) -> int:
        return int(not self.imports_reference_evidence)

    @property
    def retry_eval_seed_units(self) -> int:
        return self.scientific_cost_basis.sealed_validation_seed_units_per_job

    @property
    def physical_train_job_cap(self) -> int:
        return 2 * self.max_generations + 2 * int(not self.imports_reference_evidence)

    @property
    def physical_train_step_cap(self) -> int:
        return self.physical_train_job_cap * self.scientific_cost_basis.train_steps_per_job

    @property
    def physical_eval_seed_unit_cap(self) -> int:
        work = self.scientific_cost_basis
        return (self.max_generations * work.eval_seed_units_per_generation
                + int(not self.imports_reference_evidence) * work.reference_eval_seed_units
                + self.final_confirmation_seed_units + self.retry_eval_seed_units)

    @property
    def max_simulated_control_steps(self) -> int:
        return (self.physical_train_step_cap + self.physical_eval_seed_unit_cap
                * self.scientific_cost_basis.max_eval_control_steps_per_seed_unit)

    @property
    def max_allocated_active_gpu_hours(self) -> float:
        return self.max_allocated_gpu_capacity_ns / 3_600_000_000_000

    @property
    def max_allocated_gpu_capacity_ns(self) -> int:
        return sum(entry.allocated_gpu_capacity_ns for entry in self.operation_reserves)

    @property
    def max_output_bytes(self) -> int:
        return (sum(entry.output_bytes for entry in self.operation_reserves)
                + self.finalization_output_reserve_bytes + self.session_output_reserve_bytes)

    @property
    def contract_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.generation_cost_contract.v1",
            "max_generations": self.max_generations,
            "scientific_cost_basis": self.scientific_cost_basis.to_dict(),
            "provider": self.provider_budget.for_generations(self.max_generations),
            "imports_reference_evidence": self.imports_reference_evidence,
            "imported_reference_timeout_envelope": (
                list(self.imported_reference_timeout_envelope)
                if self.imported_reference_timeout_envelope is not None else None
            ),
            "final_confirmation_seed_units": self.final_confirmation_seed_units,
            "finalization_output_reserve_bytes": self.finalization_output_reserve_bytes,
            "session_output_reserve_bytes": self.session_output_reserve_bytes,
            "operation_reserves": {
                "derivation": {
                    "headroom_numerator": self.headroom_numerator,
                    "headroom_denominator": self.headroom_denominator,
                    "reference_phase_reserve_rule": self.reference_phase_reserve_rule,
                },
                "entries": [entry.to_dict() for entry in self.operation_reserves],
            },
            "physical_cap": {
                "train_jobs": self.physical_train_job_cap,
                "train_steps": self.physical_train_step_cap,
                "eval_seed_units": self.physical_eval_seed_unit_cap,
                "simulated_control_steps": self.max_simulated_control_steps,
                "allocated_active_gpu_hours": self.max_allocated_active_gpu_hours,
                "output_bytes": self.max_output_bytes,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        return {**payload, "contract_hash": canonical_json_hash(payload)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenGenerationCostContract":
        work = value["scientific_cost_basis"]
        sources = value["operation_reserves"]
        envelope = value["imported_reference_timeout_envelope"]
        contract = cls(
            max_generations=value["max_generations"],
            scientific_cost_basis=GenerationWorkload(**{
                name: work[name] for name in GenerationWorkload.__dataclass_fields__
            }),
            provider_budget=GenerationProviderBudget(**{
                name: value["provider"][name] for name in GenerationProviderBudget.__dataclass_fields__
            }),
            operation_reserves=tuple(OperationCostReserve.from_mapping(entry)
                                     for entry in sources["entries"]),
            imports_reference_evidence=value["imports_reference_evidence"],
            final_confirmation_seed_units=value["final_confirmation_seed_units"],
            finalization_output_reserve_bytes=value["finalization_output_reserve_bytes"],
            session_output_reserve_bytes=value["session_output_reserve_bytes"],
            headroom_numerator=sources["derivation"]["headroom_numerator"],
            headroom_denominator=sources["derivation"]["headroom_denominator"],
            imported_reference_timeout_envelope=tuple(envelope) if envelope is not None else None,
        )
        if contract.to_dict() != dict(value):
            raise ValueError("generation cost contract differs from its frozen projection")
        return contract

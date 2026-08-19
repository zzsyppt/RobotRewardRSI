"""Task-owned complete-session cost contract for Stair generation runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    build_stair_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)

STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v1"
)
STAIR_GENERATION_SESSION_COST_FILENAME = "generation_session_cost.json"
STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_actual_cost.v1"
)
STAIR_GENERATION_ACTUAL_COST_DIRNAME = "generation_actual_cost"
STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_scientific_cost_basis.v1"
)
STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES = 1024 * 1024
STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE = 5
STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT = 300_000
STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT = 32_768
STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB = 1
STAIR_GENERATION_RETRY_TRAIN_JOBS = 1
STAIR_GENERATION_RETRY_EVAL_SEED_UNITS = 7
_EMPTY_COST_EVIDENCE_HASH = canonical_json_hash([])

DEVELOPMENT_GENERATION_PILOT_MODE = "development_generation_pilot"
DEVELOPMENT_GENERATION_VALIDATION_MODE = "development_generation_validation"
FORMAL_GENERATION_SESSION_MODE = "formal_generation_session"

_MODE_GENERATIONS = {
    DEVELOPMENT_GENERATION_PILOT_MODE: 1,
    DEVELOPMENT_GENERATION_VALIDATION_MODE: 3,
    FORMAL_GENERATION_SESSION_MODE: 15,
}

_REFERENCE_TRAIN_JOBS = 1
_GENERATION_WIDTH = 2
_MONEY_GOVERNOR = "waived_by_user_20260820"


@dataclass(frozen=True)
class StairGenerationScientificCostBasis:
    """Exact task-contract projection used by the session cost envelope."""

    task_protocol_hash: str
    phased_evaluation_cost_contract_hash: str
    train_steps_per_job: int
    train_jobs_per_generation: int
    reference_eval_seed_units: int
    parent_bootstrap_seed_units: int
    eval_seed_units_per_generation: int
    public_dev_seed_units_per_job: int
    public_accept_seed_units_per_job: int
    sealed_validation_seed_units_per_job: int
    episodes_per_seed_unit: int
    max_eval_control_steps_per_seed_unit: int
    final_confirmation_seed_units: int
    schema_version: str = STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION:
            raise ValueError("Stair generation scientific cost basis schema mismatch")
        for field_name in (
            "task_protocol_hash",
            "phased_evaluation_cost_contract_hash",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        for field_name in (
            "train_steps_per_job",
            "train_jobs_per_generation",
            "reference_eval_seed_units",
            "parent_bootstrap_seed_units",
            "eval_seed_units_per_generation",
            "public_dev_seed_units_per_job",
            "public_accept_seed_units_per_job",
            "sealed_validation_seed_units_per_job",
            "episodes_per_seed_unit",
            "max_eval_control_steps_per_seed_unit",
            "final_confirmation_seed_units",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.train_jobs_per_generation != _GENERATION_WIDTH:
            raise ValueError("scientific train-job count differs from generation width")
        if self.reference_eval_seed_units != (
            self.public_dev_seed_units_per_job
            + self.public_accept_seed_units_per_job
            + self.sealed_validation_seed_units_per_job
        ):
            raise ValueError("reference evaluation seed-unit projection mismatch")
        if self.parent_bootstrap_seed_units != self.public_dev_seed_units_per_job:
            raise ValueError("parent bootstrap seed-unit projection mismatch")
        if self.eval_seed_units_per_generation != (
            _GENERATION_WIDTH * self.public_dev_seed_units_per_job
            + _GENERATION_WIDTH * self.public_accept_seed_units_per_job
            + self.sealed_validation_seed_units_per_job
        ):
            raise ValueError("generation evaluation seed-unit projection mismatch")

    def seed_units_for_phase(self, phase: str) -> int:
        values = {
            "public_dev": self.public_dev_seed_units_per_job,
            "public_accept": self.public_accept_seed_units_per_job,
            "sealed_validation": self.sealed_validation_seed_units_per_job,
        }
        try:
            return values[phase]
        except KeyError as exc:
            raise ValueError("unknown Stair evaluation phase") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StairGenerationScientificCostBasis:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation scientific cost basis must be an object")
        expected_fields = set(cls.__dataclass_fields__)
        if set(value) != expected_fields:
            raise ValueError("Stair generation scientific cost basis fields mismatch")
        return cls(**{field_name: value[field_name] for field_name in expected_fields})


def build_stair_generation_scientific_cost_basis(
    *,
    session_id: str,
    config: StairClimbingTaskConfig,
) -> StairGenerationScientificCostBasis:
    """Project all scientific counters from the existing Stair contracts."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("scientific cost basis session_id is required")
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("scientific cost basis requires StairClimbingTaskConfig")
    phased = stair_phased_evaluation_cost_contract(config)
    phased_hash = phased.get("contract_hash")
    if (
        not isinstance(phased_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", phased_hash) is None
        or phased_hash
        != canonical_json_hash(
            {key: value for key, value in phased.items() if key != "contract_hash"}
        )
    ):
        raise ValueError("Stair phased evaluation cost hash is invalid")
    if (
        phased.get("task_id") != STAIR_TASK_ID
        or phased.get("evaluation_protocol_hash")
        != config.evaluation_protocol().protocol_hash
        or phased.get("evaluation_profile_hash")
        != config.evaluation_profile.profile_hash
    ):
        raise ValueError("Stair phased evaluation cost identity differs from config")
    phases = phased.get("per_generation_phases")
    bootstrap = phased.get("session_bootstrap")
    if not isinstance(phases, Mapping) or set(phases) != {
        "public_dev",
        "public_accept",
        "sealed_validation",
    }:
        raise ValueError("Stair phased evaluation cost topology is invalid")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"parent_public_dev"}:
        raise ValueError("Stair phased bootstrap cost topology is invalid")

    def phase(name: str) -> tuple[int, int]:
        item = phases[name]
        if not isinstance(item, Mapping):
            raise TypeError("Stair phased evaluation cost phase is invalid")
        return (
            _positive_int(item.get("max_jobs"), f"{name} max_jobs"),
            _positive_int(
                item.get("seed_count_per_job"),
                f"{name} seed_count_per_job",
            ),
        )

    public_dev_jobs, public_dev_seeds = phase("public_dev")
    public_accept_jobs, public_accept_seeds = phase("public_accept")
    sealed_jobs, sealed_seeds = phase("sealed_validation")
    parent = bootstrap["parent_public_dev"]
    if not isinstance(parent, Mapping):
        raise TypeError("Stair phased bootstrap cost entry is invalid")
    parent_jobs = _positive_int(parent.get("max_jobs"), "parent bootstrap max_jobs")
    parent_seeds = _positive_int(
        parent.get("seed_count_per_job"),
        "parent bootstrap seed_count_per_job",
    )
    trial_count = _positive_int(
        phased.get("scientific_trial_count"),
        "scientific_trial_count",
    )
    sealed_query_count = _positive_int(
        phased.get("sealed_query_count"),
        "sealed_query_count",
    )
    if (
        public_dev_jobs != trial_count
        or public_accept_jobs != trial_count
        or sealed_jobs != sealed_query_count
        or public_dev_seeds != config.public_seed_count
        or public_accept_seeds != config.public_accept_seed_count
        or sealed_seeds != config.sealed_validation_seed_count
        or parent_seeds != config.public_seed_count
        or phased.get("episodes_per_seed_unit")
        != len(config.evaluation_profile.private_cell_assignment())
    ):
        raise ValueError("Stair phased evaluation cost differs from task config")
    computed_generation_seed_units = (
        public_dev_jobs * public_dev_seeds
        + public_accept_jobs * public_accept_seeds
        + sealed_jobs * sealed_seeds
    )
    episodes_per_seed = _positive_int(
        phased.get("episodes_per_seed_unit"),
        "episodes_per_seed_unit",
    )
    if (
        phased.get("max_seed_units_per_generation") != computed_generation_seed_units
        or phased.get("max_session_bootstrap_seed_units") != parent_jobs * parent_seeds
        or phased.get("max_episode_executions_per_generation")
        != computed_generation_seed_units * episodes_per_seed
        or phased.get("max_session_bootstrap_episode_executions")
        != parent_jobs * parent_seeds * episodes_per_seed
        or phased.get("execution_retry_reserve_included") is not False
    ):
        raise ValueError("Stair phased evaluation cost arithmetic mismatch")

    final_plan = build_stair_final_confirmation_plan(
        session_id=session_id,
        config=config,
    )
    comparison = final_plan.to_dict().get("comparison")
    if not isinstance(comparison, Mapping):
        raise TypeError("Stair final confirmation comparison is invalid")
    execution_order = comparison.get("execution_order")
    if not isinstance(execution_order, list) or not execution_order:
        raise ValueError("Stair final confirmation execution order is invalid")
    if final_plan.seed_count != config.sealed_validation_seed_count:
        raise ValueError("Stair final confirmation seed count differs from config")

    return StairGenerationScientificCostBasis(
        task_protocol_hash=config.task_protocol_hash,
        phased_evaluation_cost_contract_hash=phased_hash,
        train_steps_per_job=config.budget_steps,
        train_jobs_per_generation=trial_count,
        reference_eval_seed_units=(
            public_dev_seeds + public_accept_seeds + sealed_seeds
        ),
        parent_bootstrap_seed_units=parent_jobs * parent_seeds,
        eval_seed_units_per_generation=computed_generation_seed_units,
        public_dev_seed_units_per_job=public_dev_seeds,
        public_accept_seed_units_per_job=public_accept_seeds,
        sealed_validation_seed_units_per_job=sealed_seeds,
        episodes_per_seed_unit=episodes_per_seed,
        max_eval_control_steps_per_seed_unit=(
            episodes_per_seed
            * config.evaluation_profile.max_episode_horizon_control_steps
        ),
        final_confirmation_seed_units=(len(execution_order) * final_plan.seed_count),
    )


@dataclass(frozen=True)
class StairGenerationSessionCostContract:
    """One immutable budget spanning Genesis, generations, and final diagnostic."""

    session_mode: str
    max_generations: int
    scientific_cost_basis: StairGenerationScientificCostBasis
    retry_train_jobs: int
    retry_eval_seed_units: int
    max_allocated_active_gpu_hours: float
    max_simulated_control_steps: int
    max_output_bytes: int
    schema_version: str = STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION:
            raise ValueError("Stair generation session cost schema mismatch")
        expected_generations = _MODE_GENERATIONS.get(self.session_mode)
        if expected_generations is None:
            raise ValueError("Stair generation session mode is unsupported")
        if self.max_generations != expected_generations:
            raise ValueError("Stair generation count does not match session mode")
        if not isinstance(
            self.scientific_cost_basis,
            StairGenerationScientificCostBasis,
        ):
            raise TypeError("Stair generation scientific cost basis is invalid")
        _nonnegative_int(self.retry_train_jobs, "retry_train_jobs")
        _nonnegative_int(self.retry_eval_seed_units, "retry_eval_seed_units")
        if (
            isinstance(self.max_allocated_active_gpu_hours, bool)
            or not isinstance(self.max_allocated_active_gpu_hours, int | float)
            or float(self.max_allocated_active_gpu_hours) <= 0.0
        ):
            raise ValueError("max_allocated_active_gpu_hours must be positive")
        _positive_int(
            self.max_simulated_control_steps,
            "max_simulated_control_steps",
        )
        _positive_int(self.max_output_bytes, "max_output_bytes")

    @property
    def final_confirmation_seed_units(self) -> int:
        return (
            self.scientific_cost_basis.final_confirmation_seed_units
            if self.session_mode == FORMAL_GENERATION_SESSION_MODE
            else 0
        )

    @property
    def provider_attempt_cap(self) -> int:
        return (
            self.max_generations
            * _GENERATION_WIDTH
            * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
        )

    @property
    def successful_train_jobs(self) -> int:
        return _REFERENCE_TRAIN_JOBS + (
            self.max_generations * self.scientific_cost_basis.train_jobs_per_generation
        )

    @property
    def successful_train_steps(self) -> int:
        return (
            self.successful_train_jobs * self.scientific_cost_basis.train_steps_per_job
        )

    @property
    def successful_eval_seed_units(self) -> int:
        return (
            self.scientific_cost_basis.reference_eval_seed_units
            + self.scientific_cost_basis.parent_bootstrap_seed_units
            + self.max_generations
            * self.scientific_cost_basis.eval_seed_units_per_generation
            + self.final_confirmation_seed_units
        )

    @property
    def physical_train_job_cap(self) -> int:
        return self.successful_train_jobs + self.retry_train_jobs

    @property
    def physical_train_step_cap(self) -> int:
        return self.successful_train_steps + (
            self.retry_train_jobs * self.scientific_cost_basis.train_steps_per_job
        )

    @property
    def physical_eval_seed_unit_cap(self) -> int:
        return self.successful_eval_seed_units + self.retry_eval_seed_units

    @property
    def contract_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        provider_attempts = self.provider_attempt_cap
        successful_eval = self.successful_eval_seed_units
        physical_eval = self.physical_eval_seed_unit_cap
        return {
            "schema_version": self.schema_version,
            "session_mode": self.session_mode,
            "max_generations": self.max_generations,
            "scientific_cost_basis": self.scientific_cost_basis.to_dict(),
            "provider": {
                "generation_width": _GENERATION_WIDTH,
                "attempts_per_lane_per_generation": (
                    STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
                ),
                "attempt_cap": provider_attempts,
                "input_units_per_attempt_reserve": (
                    STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                ),
                "output_units_per_attempt_cap": (
                    STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                ),
                "input_unit_cap": (
                    provider_attempts
                    * STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                ),
                "output_unit_cap": (
                    provider_attempts
                    * STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                ),
                "total_unit_cap": provider_attempts
                * (
                    STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                    + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                ),
                "money_governor": _MONEY_GOVERNOR,
            },
            "successful_session": {
                "reference_qualification": {
                    "train_jobs": _REFERENCE_TRAIN_JOBS,
                    "train_steps": (self.scientific_cost_basis.train_steps_per_job),
                    "eval_seed_units": (
                        self.scientific_cost_basis.reference_eval_seed_units
                    ),
                },
                "parent_public_dev_bootstrap_seed_units": (
                    self.scientific_cost_basis.parent_bootstrap_seed_units
                ),
                "generation": {
                    "train_jobs": (
                        self.max_generations
                        * self.scientific_cost_basis.train_jobs_per_generation
                    ),
                    "train_steps": (
                        self.max_generations
                        * self.scientific_cost_basis.train_jobs_per_generation
                        * self.scientific_cost_basis.train_steps_per_job
                    ),
                    "eval_seed_units": (
                        self.max_generations
                        * self.scientific_cost_basis.eval_seed_units_per_generation
                    ),
                },
                "final_confirmation_seed_units": (self.final_confirmation_seed_units),
                "train_jobs": self.successful_train_jobs,
                "train_steps": self.successful_train_steps,
                "eval_seed_units": successful_eval,
                "episodes": (
                    successful_eval * self.scientific_cost_basis.episodes_per_seed_unit
                ),
            },
            "retry_reserve": {
                "train_jobs": self.retry_train_jobs,
                "train_steps": (
                    self.retry_train_jobs
                    * self.scientific_cost_basis.train_steps_per_job
                ),
                "eval_seed_units": self.retry_eval_seed_units,
                "episodes": (
                    self.retry_eval_seed_units
                    * self.scientific_cost_basis.episodes_per_seed_unit
                ),
            },
            "physical_cap": {
                "train_jobs": self.physical_train_job_cap,
                "train_steps": self.physical_train_step_cap,
                "eval_seed_units": physical_eval,
                "episodes": (
                    physical_eval * self.scientific_cost_basis.episodes_per_seed_unit
                ),
                "allocated_active_gpu_hours": float(
                    self.max_allocated_active_gpu_hours
                ),
                "simulated_control_steps": self.max_simulated_control_steps,
                "output_bytes": self.max_output_bytes,
                "conservative_eval_control_steps_per_seed_unit": (
                    self.scientific_cost_basis.max_eval_control_steps_per_seed_unit
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "contract_hash": self.contract_hash}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StairGenerationSessionCostContract:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation session cost must be an object")
        physical = value.get("physical_cap")
        retry = value.get("retry_reserve")
        scientific_cost_basis = value.get("scientific_cost_basis")
        if (
            not isinstance(physical, Mapping)
            or not isinstance(retry, Mapping)
            or not isinstance(scientific_cost_basis, Mapping)
        ):
            raise TypeError("Stair generation session cost sections are invalid")
        contract = cls(
            schema_version=value.get("schema_version"),
            session_mode=value.get("session_mode"),
            max_generations=value.get("max_generations"),
            scientific_cost_basis=(
                StairGenerationScientificCostBasis.from_mapping(scientific_cost_basis)
            ),
            retry_train_jobs=retry.get("train_jobs"),
            retry_eval_seed_units=retry.get("eval_seed_units"),
            max_allocated_active_gpu_hours=physical.get("allocated_active_gpu_hours"),
            max_simulated_control_steps=physical.get("simulated_control_steps"),
            max_output_bytes=physical.get("output_bytes"),
        )
        if dict(value) != contract.to_dict():
            raise ValueError("Stair generation session cost projection mismatch")
        return contract


def create_stair_generation_session_cost_contract(
    output_dir: Path,
    contract: StairGenerationSessionCostContract,
) -> Path:
    if not isinstance(contract, StairGenerationSessionCostContract):
        raise TypeError("Stair generation session cost contract is invalid")
    output_dir = Path(output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STAIR_GENERATION_SESSION_COST_FILENAME
    create_once_bytes(
        path,
        canonical_json_bytes(contract.to_dict(), trailing_newline=True),
    )
    loaded = load_stair_generation_session_cost_contract(path)
    if loaded != contract:
        raise ValueError("Stair generation session cost create-once mismatch")
    return path


def load_stair_generation_session_cost_contract(
    path: Path,
) -> StairGenerationSessionCostContract:
    payload = load_strict_json_object(
        Path(path),
        context="Stair generation session cost contract",
    )
    return StairGenerationSessionCostContract.from_mapping(payload)


@dataclass(frozen=True)
class StairGenerationActualCostUsage:
    """Cumulative task-owned measurements used by one cost-chain edge."""

    provider_attempts: int = 0
    provider_request_units: int = 0
    provider_response_units: int = 0
    provider_total_units: int = 0
    provider_reasoning_units: int = 0
    provider_conservative_attempts: int = 0
    job_attempts: int = 0
    train_jobs: int = 0
    train_steps: int = 0
    eval_seed_units: int = 0
    simulated_control_steps: int = 0
    resource_lease_wall_duration_ns: int = 0
    unmeasured_bound_attempts: int = 0
    observed_output_bytes: int = 0
    observed_output_files: int = 0
    job_cost_evidence_hash: str = _EMPTY_COST_EVIDENCE_HASH

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "job_cost_evidence_hash":
                continue
            _nonnegative_int(getattr(self, field_name), field_name)
        if re.fullmatch(r"[0-9a-f]{64}", self.job_cost_evidence_hash) is None:
            raise ValueError("job_cost_evidence_hash must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class StairGenerationActualCostReceiptRef:
    path: Path
    payload_hash: str
    payload: Mapping[str, Any]


class StairGenerationActualCostJournal:
    """Small append-only actual-cost chain for the Stair session driver."""

    def __init__(
        self,
        output_dir: Path,
        *,
        contract: StairGenerationSessionCostContract,
    ) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("actual-cost output_dir must be a Path")
        if not isinstance(contract, StairGenerationSessionCostContract):
            raise TypeError("actual-cost contract is invalid")
        self.output_dir = output_dir.resolve(strict=False)
        self.root = self.output_dir / STAIR_GENERATION_ACTUAL_COST_DIRNAME
        self.contract = contract

    def publish_reservation(self) -> StairGenerationActualCostReceiptRef:
        existing = self.receipt_for_event("session-reservation")
        if existing is not None:
            return existing
        return self.publish_snapshot(
            event_id="session-reservation",
            previous_cost_chain_head_hash=self.contract.contract_hash,
            usage=StairGenerationActualCostUsage(),
            conservative_charges=(),
            finalization_output_reserve_bytes=0,
        )

    def publish_snapshot(
        self,
        *,
        event_id: str,
        previous_cost_chain_head_hash: str,
        usage: StairGenerationActualCostUsage,
        conservative_charges: tuple[str, ...] = (),
        finalization_output_reserve_bytes: int = (
            STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES
        ),
    ) -> StairGenerationActualCostReceiptRef:
        event_id = _event_id(event_id)
        if not isinstance(usage, StairGenerationActualCostUsage):
            raise TypeError("actual-cost usage is invalid")
        if not isinstance(previous_cost_chain_head_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", previous_cost_chain_head_hash
        ):
            raise ValueError("actual-cost previous chain head is invalid")
        _nonnegative_int(
            finalization_output_reserve_bytes,
            "finalization_output_reserve_bytes",
        )
        if not isinstance(conservative_charges, tuple) or any(
            not isinstance(item, str) or not item or len(item) > 128
            for item in conservative_charges
        ):
            raise ValueError("actual-cost conservative charges are invalid")

        loaded = self.load_chain()
        matching_indexes = [
            index
            for index, item in enumerate(loaded)
            if item.payload["event_id"] == event_id
        ]
        if len(matching_indexes) > 1:
            raise ValueError("actual-cost event is duplicated")
        if matching_indexes and matching_indexes[0] != len(loaded) - 1:
            raise ValueError("actual-cost event was already superseded")
        previous_task_hash = (
            loaded[-2].payload_hash
            if matching_indexes and len(loaded) > 1
            else (
                None
                if matching_indexes
                else (loaded[-1].payload_hash if loaded else None)
            )
        )
        reservation, actual, checks = _actual_cost_projection(
            self.contract,
            usage=usage,
            finalization_output_reserve_bytes=finalization_output_reserve_bytes,
        )
        payload = {
            "schema_version": STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION,
            "session_cost_contract_hash": self.contract.contract_hash,
            "event_id": event_id,
            "previous_cost_chain_head_hash": previous_cost_chain_head_hash,
            "previous_task_cost_receipt_hash": previous_task_hash,
            "reservation": reservation,
            "actual": actual,
            "conservative_charges": list(conservative_charges),
            "cap_checks": checks,
            "acceptance_authority": False,
        }
        if matching_indexes:
            existing = loaded[-1]
            if dict(existing.payload) != payload:
                raise ValueError(
                    "existing actual-cost event differs from recomputation"
                )
            return existing
        sequence = len(loaded)
        path = self.root / f"{sequence:04d}-{event_id}.json"
        encoded = canonical_json_bytes(payload, trailing_newline=True)
        self.root.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, encoded)
        return StairGenerationActualCostReceiptRef(
            path=path,
            payload_hash=canonical_json_hash(payload),
            payload=payload,
        )

    def receipt_for_event(
        self,
        event_id: str,
    ) -> StairGenerationActualCostReceiptRef | None:
        event_id = _event_id(event_id)
        matches = [
            item for item in self.load_chain() if item.payload["event_id"] == event_id
        ]
        if len(matches) > 1:
            raise ValueError("actual-cost event is duplicated")
        return matches[0] if matches else None

    def load_chain(self) -> list[StairGenerationActualCostReceiptRef]:
        if not self.root.exists():
            return []
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("actual-cost receipt root is invalid")
        result: list[StairGenerationActualCostReceiptRef] = []
        previous_task_hash: str | None = None
        for expected_index, path in enumerate(sorted(self.root.glob("*.json"))):
            match = re.fullmatch(r"([0-9]{4})-([a-z0-9-]+)\.json", path.name)
            if match is None or int(match.group(1)) != expected_index:
                raise ValueError("actual-cost receipt sequence is invalid")
            payload = load_strict_json_object(
                path,
                context="Stair actual-cost receipt",
            )
            _validate_actual_cost_receipt(
                payload,
                contract=self.contract,
                expected_event_id=match.group(2),
                previous_task_hash=previous_task_hash,
            )
            if expected_index == 0:
                if (
                    payload["event_id"] != "session-reservation"
                    or payload["previous_cost_chain_head_hash"]
                    != self.contract.contract_hash
                    or payload["actual"]
                    != _actual_cost_projection(
                        self.contract,
                        usage=StairGenerationActualCostUsage(),
                        finalization_output_reserve_bytes=0,
                    )[1]
                ):
                    raise ValueError("actual-cost reservation receipt mismatch")
            elif payload["event_id"] == "session-reservation":
                raise ValueError("actual-cost reservation receipt is duplicated")
            encoded = canonical_json_bytes(payload, trailing_newline=True)
            if path.read_bytes() != encoded:
                raise ValueError("Stair actual-cost receipt is not canonical JSON")
            ref = StairGenerationActualCostReceiptRef(
                path=path,
                payload_hash=canonical_json_hash(payload),
                payload=payload,
            )
            result.append(ref)
            previous_task_hash = ref.payload_hash
        return result


def _validate_actual_cost_receipt(
    value: Mapping[str, Any],
    *,
    contract: StairGenerationSessionCostContract,
    expected_event_id: str,
    previous_task_hash: str | None,
) -> None:
    expected_fields = {
        "schema_version",
        "session_cost_contract_hash",
        "event_id",
        "previous_cost_chain_head_hash",
        "previous_task_cost_receipt_hash",
        "reservation",
        "actual",
        "conservative_charges",
        "cap_checks",
        "acceptance_authority",
    }
    if set(value) != expected_fields:
        raise ValueError("Stair actual-cost receipt fields mismatch")
    if (
        value.get("schema_version") != STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION
        or value.get("session_cost_contract_hash") != contract.contract_hash
        or value.get("event_id") != expected_event_id
        or value.get("previous_task_cost_receipt_hash") != previous_task_hash
        or value.get("acceptance_authority") is not False
    ):
        raise ValueError("Stair actual-cost receipt identity mismatch")
    previous = value.get("previous_cost_chain_head_hash")
    if not isinstance(previous, str) or re.fullmatch(r"[0-9a-f]{64}", previous) is None:
        raise ValueError("Stair actual-cost prior chain head is invalid")
    reservation = value.get("reservation")
    actual = value.get("actual")
    checks = value.get("cap_checks")
    conservative = value.get("conservative_charges")
    if (
        not isinstance(reservation, Mapping)
        or not isinstance(actual, Mapping)
        or not isinstance(checks, Mapping)
        or not isinstance(conservative, list)
        or any(not isinstance(item, str) or not item for item in conservative)
    ):
        raise ValueError("Stair actual-cost receipt sections are invalid")
    # Re-project the payload through the publishing logic's immutable fields.
    usage_fields = set(StairGenerationActualCostUsage.__dataclass_fields__)
    extra_actual = {
        "allocated_gpu_capacity_hours_charged",
        "finalization_output_reserve_bytes",
        "accounted_output_bytes",
    }
    if set(actual) != usage_fields | extra_actual:
        raise ValueError("Stair actual-cost usage fields mismatch")
    usage = StairGenerationActualCostUsage(
        **{field_name: actual[field_name] for field_name in usage_fields}
    )
    reserve = actual.get("finalization_output_reserve_bytes")
    _nonnegative_int(reserve, "finalization_output_reserve_bytes")
    expected_reservation, expected_actual, expected_checks = _actual_cost_projection(
        contract,
        usage=usage,
        finalization_output_reserve_bytes=reserve,
    )
    if (
        dict(reservation) != expected_reservation
        or dict(actual) != expected_actual
        or dict(checks) != expected_checks
    ):
        raise ValueError("Stair actual-cost projection mismatch")


def _actual_cost_projection(
    contract: StairGenerationSessionCostContract,
    *,
    usage: StairGenerationActualCostUsage,
    finalization_output_reserve_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    provider_caps = contract.to_dict()["provider"]
    charged_gpu_hours = (
        usage.resource_lease_wall_duration_ns / 3_600_000_000_000
        + usage.unmeasured_bound_attempts
        * float(contract.max_allocated_active_gpu_hours)
    )
    accounted_output_bytes = (
        usage.observed_output_bytes + finalization_output_reserve_bytes
    )
    reservation = {
        "provider_attempt_cap": provider_caps["attempt_cap"],
        "provider_input_unit_cap": provider_caps["input_unit_cap"],
        "provider_output_unit_cap": provider_caps["output_unit_cap"],
        "provider_total_unit_cap": provider_caps["total_unit_cap"],
        "train_job_cap": contract.physical_train_job_cap,
        "train_step_cap": contract.physical_train_step_cap,
        "eval_seed_unit_cap": contract.physical_eval_seed_unit_cap,
        "allocated_gpu_capacity_hours_cap": float(
            contract.max_allocated_active_gpu_hours
        ),
        "simulated_control_step_cap": contract.max_simulated_control_steps,
        "output_byte_cap": contract.max_output_bytes,
    }
    actual = {
        **usage.to_dict(),
        "allocated_gpu_capacity_hours_charged": charged_gpu_hours,
        "finalization_output_reserve_bytes": (finalization_output_reserve_bytes),
        "accounted_output_bytes": accounted_output_bytes,
    }
    checks = {
        "provider_attempts_within_cap": (
            usage.provider_attempts <= provider_caps["attempt_cap"]
        ),
        "provider_request_units_within_cap": (
            usage.provider_request_units <= provider_caps["input_unit_cap"]
        ),
        "provider_response_units_within_cap": (
            usage.provider_response_units <= provider_caps["output_unit_cap"]
        ),
        "provider_total_units_within_cap": (
            usage.provider_total_units <= provider_caps["total_unit_cap"]
        ),
        "train_jobs_within_cap": (usage.train_jobs <= contract.physical_train_job_cap),
        "train_steps_within_cap": (
            usage.train_steps <= contract.physical_train_step_cap
        ),
        "eval_seed_units_within_cap": (
            usage.eval_seed_units <= contract.physical_eval_seed_unit_cap
        ),
        "allocated_gpu_capacity_hours_within_cap": (
            charged_gpu_hours <= float(contract.max_allocated_active_gpu_hours)
        ),
        "simulated_control_steps_within_cap": (
            usage.simulated_control_steps <= contract.max_simulated_control_steps
        ),
        "output_bytes_within_cap": (
            accounted_output_bytes <= contract.max_output_bytes
        ),
        "resource_lease_measurement_complete": (usage.unmeasured_bound_attempts == 0),
    }
    checks["all_passed"] = all(checks.values())
    return reservation, actual, checks


def _event_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 96
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
    ):
        raise ValueError("actual-cost event_id is invalid")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


__all__ = [
    "DEVELOPMENT_GENERATION_PILOT_MODE",
    "DEVELOPMENT_GENERATION_VALIDATION_MODE",
    "FORMAL_GENERATION_SESSION_MODE",
    "STAIR_GENERATION_ACTUAL_COST_DIRNAME",
    "STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION",
    "STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES",
    "STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB",
    "STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE",
    "STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT",
    "STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT",
    "STAIR_GENERATION_RETRY_EVAL_SEED_UNITS",
    "STAIR_GENERATION_RETRY_TRAIN_JOBS",
    "STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_FILENAME",
    "STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION",
    "StairGenerationActualCostJournal",
    "StairGenerationActualCostReceiptRef",
    "StairGenerationActualCostUsage",
    "StairGenerationScientificCostBasis",
    "StairGenerationSessionCostContract",
    "build_stair_generation_scientific_cost_basis",
    "create_stair_generation_session_cost_contract",
    "load_stair_generation_session_cost_contract",
]

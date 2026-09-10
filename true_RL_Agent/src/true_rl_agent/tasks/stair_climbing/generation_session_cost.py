"""Task-owned complete-session cost contract for Stair generation runs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_cost_journal import (
    GenerationActualCostUsage as StairGenerationActualCostUsageV2,
    LegacyGenerationActualCostUsage as StairGenerationActualCostUsage,
    GenerationActualCostReceiptRef as StairGenerationActualCostReceiptRef,
    GenerationActualCostJournal as StairGenerationActualCostJournal,
    replay_generation_combined_cost_chain as replay_stair_generation_combined_cost_chain,
    replay_generation_final_cost_chain as replay_stair_generation_formal_cost_chain,
    project_actual_generation_cost,
    require_cost_event_id,
    require_nonnegative_cost_integer,
)
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
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    build_stair_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)

STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v1"
)
STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v2"
)
STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v3"
)
STAIR_GENERATION_SESSION_COST_V4_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v4"
)
STAIR_GENERATION_SESSION_COST_V5_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_cost.v5"
)
STAIR_GENERATION_SESSION_COST_FILENAME = "generation_session_cost.json"
STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_actual_cost.v1"
)
STAIR_GENERATION_ACTUAL_COST_DIRNAME = "generation_actual_cost"
STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_scientific_cost_basis.v1"
)
STAIR_GENERATION_OPERATION_RESERVES_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_operation_reserves.v1"
)
STAIR_GENERATION_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_operation_reserves.v2"
)
STAIR_GENERATION_IMPORTED_REFERENCE_OPERATION_RESERVES_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_operation_reserves.v3"
)
STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_operation_reserves.v5"
)
STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES = 1024 * 1024
STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE = 5
STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT = 300_000
STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT = 32_768
STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB = 1
STAIR_GENERATION_RETRY_TRAIN_JOBS = 1
STAIR_GENERATION_RETRY_EVAL_SEED_UNITS = 7
STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS = 11.0
STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS = 1.5
STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES = 3 * 1024**3
STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES = 512 * 1024**2
STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS = (
    13.6
)
STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS = 0.2
STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES = 3 * 1024**3
STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES = 512 * 1024**2
STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS = 49.0
STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS = 1.5
STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES = 13 * 1024**3
STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES = 512 * 1024**2
STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS = (
    68.1
)
STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS = 0.2
STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES = 13 * 1024**3
STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES = 512 * 1024**2
_EMPTY_COST_EVIDENCE_HASH = canonical_json_hash([])

DEVELOPMENT_GENERATION_PILOT_MODE = "development_generation_pilot"
DEVELOPMENT_GENERATION_VALIDATION_MODE = "development_generation_validation"
FORMAL_GENERATION_SESSION_MODE = "formal_generation_session"
STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES = (3, 6, 9, 12)

_MODE_GENERATIONS = {
    DEVELOPMENT_GENERATION_PILOT_MODE: 1,
    DEVELOPMENT_GENERATION_VALIDATION_MODE: 3,
    FORMAL_GENERATION_SESSION_MODE: STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
}

_REFERENCE_TRAIN_JOBS = 1
_GENERATION_WIDTH = 2
_MONEY_GOVERNOR = "waived_by_user_20260820"
_P6B_OPERATION_RESERVE_SOURCE_COMMIT = "5f2d7557ca51b73f1f16911e117a9401a103c171"
_P6B_OPERATION_RESERVE_SOURCE_SESSION_ID = "p6b-5f2d755-20260822T025737Z"
_P6B_OPERATION_RESERVE_SOURCE_COST_RECEIPT_SHA256 = (
    "ad500f719c55657cf5e9a4e37243e3de2699af5288df007af9c1e26270032bb5"
)
_P6B_SUCCESS_NON_JOB_OUTPUT_BYTES = 3_100_927
_OPERATION_RESERVE_HEADROOM_NUMERATOR = 5
_OPERATION_RESERVE_HEADROOM_DENOMINATOR = 4
_P6C_FORMAL_RESERVE_SOURCE_COMMIT = "415bfad6eb6e85c6cf0cf94851c6b39d1067116b"
_P6C_FORMAL_RESERVE_SOURCE_SESSION_ID = "p6c-415bfad-20260823T042422Z"
_P6C_FORMAL_RESERVE_G4_AUDIT_SHA256 = (
    "ad691299c0ca5b08383f984a68f514d987c1831cdd3b24d367a9ca34d7266a93"
)
_P6C_FORMAL_SOURCE_NON_JOB_OUTPUT_BYTES = 13_448_831
_IMPORTED_REFERENCE_FORMAL_PUBLIC_DEV_PROCESS_TIMEOUT_S = 420


def _ceil_fraction(value: int, *, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class StairGenerationOperationCostReserve:
    """One frozen Stair operation class in the P6-C physical-cost envelope."""

    operation_class: str
    budget_class: str
    multiplicity: int
    source_allocated_gpu_capacity_ns: int
    source_output_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operation_class, str)
            or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", self.operation_class) is None
        ):
            raise ValueError("operation reserve class is invalid")
        if self.budget_class not in {"successful_session", "retry_reserve"}:
            raise ValueError("operation reserve budget class is invalid")
        for field_name in (
            "multiplicity",
            "source_allocated_gpu_capacity_ns",
            "source_output_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"operation reserve {field_name} must be positive")

    @property
    def allocated_gpu_capacity_ns_per_attempt(self) -> int:
        return _ceil_fraction(
            self.source_allocated_gpu_capacity_ns,
            numerator=_OPERATION_RESERVE_HEADROOM_NUMERATOR,
            denominator=_OPERATION_RESERVE_HEADROOM_DENOMINATOR,
        )

    @property
    def output_bytes_per_attempt(self) -> int:
        return _ceil_fraction(
            self.source_output_bytes,
            numerator=_OPERATION_RESERVE_HEADROOM_NUMERATOR,
            denominator=_OPERATION_RESERVE_HEADROOM_DENOMINATOR,
        )

    @property
    def allocated_gpu_capacity_ns(self) -> int:
        return self.multiplicity * self.allocated_gpu_capacity_ns_per_attempt

    @property
    def output_bytes(self) -> int:
        return self.multiplicity * self.output_bytes_per_attempt

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_class": self.operation_class,
            "budget_class": self.budget_class,
            "multiplicity": self.multiplicity,
            "source_allocated_gpu_capacity_ns_per_attempt": (
                self.source_allocated_gpu_capacity_ns
            ),
            "source_output_bytes_per_attempt": self.source_output_bytes,
            "allocated_gpu_capacity_ns_per_attempt": (
                self.allocated_gpu_capacity_ns_per_attempt
            ),
            "output_bytes_per_attempt": self.output_bytes_per_attempt,
            "allocated_gpu_capacity_ns": self.allocated_gpu_capacity_ns,
            "output_bytes": self.output_bytes,
        }


@dataclass(frozen=True)
class StairGenerationTimeoutBoundOperationCostReserve(
    StairGenerationOperationCostReserve
):
    """Historical cost evidence with a protocol-owned hard-timeout allocation."""

    timeout_bound_allocated_gpu_capacity_ns_per_attempt: int

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.timeout_bound_allocated_gpu_capacity_ns_per_attempt
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "timeout_bound_allocated_gpu_capacity_ns_per_attempt must be positive"
            )

    @property
    def allocated_gpu_capacity_ns_per_attempt(self) -> int:
        return self.timeout_bound_allocated_gpu_capacity_ns_per_attempt

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "allocated_gpu_capacity_authority": "runner_timeout_plus_cleanup",
        }


_P6C_OPERATION_COST_RESERVES = (
    StairGenerationOperationCostReserve(
        "reference_train",
        "successful_session",
        1,
        3_513_400_399_270,
        333_735_635,
    ),
    StairGenerationOperationCostReserve(
        "reference_qualification",
        "successful_session",
        1,
        853_002_812_172,
        285_030,
    ),
    StairGenerationOperationCostReserve(
        "parent_public_dev",
        "successful_session",
        1,
        186_233_373_218,
        82_395,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_train",
        "successful_session",
        3,
        3_687_999_590_412,
        333_737_533,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_train",
        "successful_session",
        3,
        3_735_431_307_277,
        333_738_027,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_public_dev",
        "successful_session",
        3,
        195_538_158_574,
        82_511,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_public_dev",
        "successful_session",
        3,
        193_248_428_714,
        82_480,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_public_accept",
        "successful_session",
        3,
        303_143_509_683,
        111_568,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_public_accept",
        "successful_session",
        3,
        305_984_678_125,
        111_448,
    ),
    StairGenerationOperationCostReserve(
        "sealed_validation",
        "successful_session",
        3,
        437_052_835_681,
        128_855,
    ),
    StairGenerationOperationCostReserve(
        "reference_train_retry",
        "retry_reserve",
        1,
        3_735_431_307_277,
        325_894_966,
    ),
    StairGenerationOperationCostReserve(
        "sealed_validation_retry",
        "retry_reserve",
        1,
        437_052_835_681,
        128_855,
    ),
)


_P6D_OPERATION_COST_RESERVES = (
    StairGenerationOperationCostReserve(
        "reference_train",
        "successful_session",
        1,
        3_713_516_361_162,
        334_112_409,
    ),
    StairGenerationOperationCostReserve(
        "reference_qualification",
        "successful_session",
        1,
        872_050_072_750,
        285_034,
    ),
    StairGenerationOperationCostReserve(
        "parent_public_dev",
        "successful_session",
        1,
        193_973_260_733,
        82_398,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_train",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        3_727_860_244_139,
        334_112_409,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_train",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        3_791_222_586_112,
        334_112_409,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_public_dev",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        204_580_836_528,
        82_636,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_public_dev",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        196_154_850_372,
        82_636,
    ),
    StairGenerationOperationCostReserve(
        "lane_a_public_accept",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        334_870_033_966,
        111_727,
    ),
    StairGenerationOperationCostReserve(
        "lane_b_public_accept",
        "successful_session",
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        314_647_793_917,
        111_727,
    ),
    StairGenerationOperationCostReserve(
        "sealed_validation",
        "successful_session",
        17,
        411_671_242_846,
        129_068,
    ),
    StairGenerationOperationCostReserve(
        "reference_train_retry",
        "retry_reserve",
        1,
        3_791_222_586_112,
        334_112_409,
    ),
    StairGenerationOperationCostReserve(
        "sealed_validation_retry",
        "retry_reserve",
        1,
        411_671_242_846,
        129_068,
    ),
)


def stair_generation_operation_cost_reserves() -> tuple[
    StairGenerationOperationCostReserve, ...
]:
    """Return the immutable P6-C job-operation reserve table."""

    return _P6C_OPERATION_COST_RESERVES


def stair_formal_generation_operation_cost_reserves() -> tuple[
    StairGenerationOperationCostReserve, ...
]:
    """Return the immutable P6-D operation table derived from final P6-C."""

    return _P6D_OPERATION_COST_RESERVES


def _operation_reserve_plan_payload(
    *,
    include_parent_public_dev: bool = True,
) -> dict[str, Any]:
    if not isinstance(include_parent_public_dev, bool):
        raise TypeError("parent public-dev reserve inclusion must be boolean")
    active_reserves = tuple(
        item
        for item in _P6C_OPERATION_COST_RESERVES
        if include_parent_public_dev or item.operation_class != "parent_public_dev"
    )
    entries = [item.to_dict() for item in active_reserves]
    totals: dict[str, dict[str, int]] = {}
    for budget_class in ("successful_session", "retry_reserve"):
        selected = [
            item
            for item in active_reserves
            if item.budget_class == budget_class
        ]
        totals[budget_class] = {
            "source_allocated_gpu_capacity_ns": sum(
                item.multiplicity * item.source_allocated_gpu_capacity_ns
                for item in selected
            ),
            "source_output_bytes": sum(
                item.multiplicity * item.source_output_bytes for item in selected
            ),
            "reserved_allocated_gpu_capacity_ns": sum(
                item.allocated_gpu_capacity_ns for item in selected
            ),
            "reserved_output_bytes": sum(item.output_bytes for item in selected),
        }
    totals["successful_session"]["source_non_job_output_bytes"] = (
        _P6B_SUCCESS_NON_JOB_OUTPUT_BYTES
    )
    totals["successful_session"]["source_total_output_bytes"] = (
        totals["successful_session"]["source_output_bytes"]
        + _P6B_SUCCESS_NON_JOB_OUTPUT_BYTES
    )
    physical_gpu_ns = int(
        (
            STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            + STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        )
        * 3_600_000_000_000
    )
    physical_output_bytes = (
        STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
        + STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES
    )
    reserved_gpu_ns = sum(
        item.allocated_gpu_capacity_ns for item in active_reserves
    )
    reserved_output_bytes = sum(
        item.output_bytes for item in active_reserves
    )
    return {
        "schema_version": STAIR_GENERATION_OPERATION_RESERVES_SCHEMA_VERSION,
        "source": {
            "session_id": _P6B_OPERATION_RESERVE_SOURCE_SESSION_ID,
            "source_commit": _P6B_OPERATION_RESERVE_SOURCE_COMMIT,
            "actual_cost_receipt_sha256": (
                _P6B_OPERATION_RESERVE_SOURCE_COST_RECEIPT_SHA256
            ),
        },
        "derivation": {
            "rule": "ceil_per_operation",
            "headroom_numerator": _OPERATION_RESERVE_HEADROOM_NUMERATOR,
            "headroom_denominator": _OPERATION_RESERVE_HEADROOM_DENOMINATOR,
            "capacity_time_unit": "allocated_gpu_capacity_ns",
            "output_unit": "bytes",
        },
        "entries": entries,
        "totals": totals,
        "physical_unallocated_headroom": {
            "allocated_gpu_capacity_ns": physical_gpu_ns - reserved_gpu_ns,
            "output_bytes": physical_output_bytes - reserved_output_bytes,
        },
    }


def imported_reference_operation_cost_reserves(
    *,
    candidate_train_process_timeout_s: int,
    termination_cleanup_margin_s: int,
    session_mode: str = DEVELOPMENT_GENERATION_VALIDATION_MODE,
) -> tuple[StairGenerationOperationCostReserve, ...]:
    for field_name, value in (
        ("candidate_train_process_timeout_s", candidate_train_process_timeout_s),
        ("termination_cleanup_margin_s", termination_cleanup_margin_s),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be positive")
    if session_mode not in {
        DEVELOPMENT_GENERATION_VALIDATION_MODE,
        FORMAL_GENERATION_SESSION_MODE,
    }:
        raise ValueError("imported-reference operation reserve mode is unsupported")
    excluded = {
        "reference_train",
        "reference_qualification",
        "parent_public_dev",
        "reference_train_retry",
    }
    train_allocation_ns = (
        candidate_train_process_timeout_s + termination_cleanup_margin_s
    ) * 1_000_000_000
    active: list[StairGenerationOperationCostReserve] = []
    source_reserves = (
        _P6D_OPERATION_COST_RESERVES
        if session_mode == FORMAL_GENERATION_SESSION_MODE
        else _P6C_OPERATION_COST_RESERVES
    )
    formal = session_mode == FORMAL_GENERATION_SESSION_MODE
    selected_public_accept_added = False
    selected_public_accept_sources = tuple(
        item
        for item in source_reserves
        if item.operation_class
        in {"lane_a_public_accept", "lane_b_public_accept"}
    )
    for item in source_reserves:
        if item.operation_class in excluded:
            continue
        if item.operation_class in {"lane_a_train", "lane_b_train"}:
            active.append(
                StairGenerationTimeoutBoundOperationCostReserve(
                    operation_class=item.operation_class,
                    budget_class=item.budget_class,
                    multiplicity=item.multiplicity,
                    source_allocated_gpu_capacity_ns=(
                        item.source_allocated_gpu_capacity_ns
                    ),
                    source_output_bytes=item.source_output_bytes,
                    timeout_bound_allocated_gpu_capacity_ns_per_attempt=(
                        train_allocation_ns
                    ),
                )
            )
        elif formal and item.operation_class in {
            "lane_a_public_dev",
            "lane_b_public_dev",
        }:
            active.append(
                StairGenerationTimeoutBoundOperationCostReserve(
                    operation_class=item.operation_class,
                    budget_class=item.budget_class,
                    multiplicity=item.multiplicity,
                    source_allocated_gpu_capacity_ns=(
                        item.source_allocated_gpu_capacity_ns
                    ),
                    source_output_bytes=item.source_output_bytes,
                    timeout_bound_allocated_gpu_capacity_ns_per_attempt=(
                        (
                            _IMPORTED_REFERENCE_FORMAL_PUBLIC_DEV_PROCESS_TIMEOUT_S
                            + termination_cleanup_margin_s
                        )
                        * 1_000_000_000
                    ),
                )
            )
        elif formal and item.operation_class in {
            "lane_a_public_accept",
            "lane_b_public_accept",
        }:
            if selected_public_accept_added:
                continue
            active.append(
                StairGenerationOperationCostReserve(
                    operation_class="selected_public_accept",
                    budget_class=item.budget_class,
                    multiplicity=item.multiplicity,
                    source_allocated_gpu_capacity_ns=max(
                        reserve.source_allocated_gpu_capacity_ns
                        for reserve in selected_public_accept_sources
                    ),
                    source_output_bytes=max(
                        reserve.source_output_bytes
                        for reserve in selected_public_accept_sources
                    ),
                )
            )
            selected_public_accept_added = True
        else:
            active.append(item)
    return tuple(active)


def _imported_reference_operation_reserve_plan_payload(
    *,
    candidate_train_process_timeout_s: int,
    termination_cleanup_margin_s: int,
    session_mode: str = DEVELOPMENT_GENERATION_VALIDATION_MODE,
) -> dict[str, Any]:
    """Project only work executed by a session with imported Genesis.

    The P5 reference train and qualification receipts are immutable inputs to the
    new session, not operations that the Phase 6 process can execute or retry.
    Historical per-operation measurements remain useful as evidence. Candidate
    train entries are instead allocated from the protocol timeout plus the
    runner-owned cleanup margin, so the plan and admission accounting agree.
    """

    active_reserves = imported_reference_operation_cost_reserves(
        candidate_train_process_timeout_s=candidate_train_process_timeout_s,
        termination_cleanup_margin_s=termination_cleanup_margin_s,
        session_mode=session_mode,
    )
    entries = [item.to_dict() for item in active_reserves]
    totals: dict[str, dict[str, int]] = {}
    for budget_class in ("successful_session", "retry_reserve"):
        selected = tuple(
            item for item in active_reserves if item.budget_class == budget_class
        )
        totals[budget_class] = {
            "source_allocated_gpu_capacity_ns": sum(
                item.multiplicity * item.source_allocated_gpu_capacity_ns
                for item in selected
            ),
            "source_output_bytes": sum(
                item.multiplicity * item.source_output_bytes for item in selected
            ),
            "reserved_allocated_gpu_capacity_ns": sum(
                item.allocated_gpu_capacity_ns for item in selected
            ),
            "reserved_output_bytes": sum(item.output_bytes for item in selected),
        }
    formal = session_mode == FORMAL_GENERATION_SESSION_MODE
    successful_gpu_hours = (
        STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        if formal
        else STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
    )
    retry_gpu_hours = (
        STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        if formal
        else STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
    )
    successful_output_bytes = (
        STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES
        if formal
        else STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
    )
    retry_output_bytes = (
        STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES
        if formal
        else STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES
    )
    physical_gpu_ns = int(
        successful_gpu_hours * 3_600_000_000_000
    ) + int(retry_gpu_hours * 3_600_000_000_000)
    physical_output_bytes = successful_output_bytes + retry_output_bytes
    reserved_gpu_ns = sum(
        item.allocated_gpu_capacity_ns for item in active_reserves
    )
    reserved_output_bytes = sum(item.output_bytes for item in active_reserves)
    source = (
        {
            "session_id": _P6C_FORMAL_RESERVE_SOURCE_SESSION_ID,
            "source_commit": _P6C_FORMAL_RESERVE_SOURCE_COMMIT,
            "g4_audit_sha256": _P6C_FORMAL_RESERVE_G4_AUDIT_SHA256,
        }
        if formal
        else {
            "session_id": _P6B_OPERATION_RESERVE_SOURCE_SESSION_ID,
            "source_commit": _P6B_OPERATION_RESERVE_SOURCE_COMMIT,
            "actual_cost_receipt_sha256": (
                _P6B_OPERATION_RESERVE_SOURCE_COST_RECEIPT_SHA256
            ),
        }
    )
    return {
        "schema_version": (
            STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION
            if formal
            else STAIR_GENERATION_IMPORTED_REFERENCE_OPERATION_RESERVES_SCHEMA_VERSION
        ),
        "source": source,
        "derivation": {
            "rule": (
                "imported_reference_excluded_timeout_bound_train_"
                "formal_public_dev_selected_public_accept"
                if formal
                else "imported_reference_excluded_timeout_bound_train"
            ),
            "headroom_numerator": _OPERATION_RESERVE_HEADROOM_NUMERATOR,
            "headroom_denominator": _OPERATION_RESERVE_HEADROOM_DENOMINATOR,
            "candidate_train_process_timeout_s": (
                candidate_train_process_timeout_s
            ),
            "termination_cleanup_margin_s": termination_cleanup_margin_s,
            **(
                {
                    "formal_public_dev_process_timeout_s": (
                        _IMPORTED_REFERENCE_FORMAL_PUBLIC_DEV_PROCESS_TIMEOUT_S
                    ),
                    "formal_public_accept_execution": "selected_lane_only",
                }
                if formal
                else {}
            ),
            "capacity_time_unit": "allocated_gpu_capacity_ns",
            "output_unit": "bytes",
        },
        "entries": entries,
        "totals": totals,
        "physical_unallocated_headroom": {
            "allocated_gpu_capacity_ns": physical_gpu_ns - reserved_gpu_ns,
            "output_bytes": physical_output_bytes - reserved_output_bytes,
        },
    }


def _formal_operation_reserve_plan_payload(
    *,
    include_parent_public_dev: bool = True,
) -> dict[str, Any]:
    if not isinstance(include_parent_public_dev, bool):
        raise TypeError("parent public-dev reserve inclusion must be boolean")
    active_reserves = tuple(
        item
        for item in _P6D_OPERATION_COST_RESERVES
        if include_parent_public_dev or item.operation_class != "parent_public_dev"
    )
    entries = [item.to_dict() for item in active_reserves]
    totals: dict[str, dict[str, int]] = {}
    for budget_class in ("successful_session", "retry_reserve"):
        selected = [
            item
            for item in active_reserves
            if item.budget_class == budget_class
        ]
        totals[budget_class] = {
            "source_allocated_gpu_capacity_ns": sum(
                item.multiplicity * item.source_allocated_gpu_capacity_ns
                for item in selected
            ),
            "source_output_bytes": sum(
                item.multiplicity * item.source_output_bytes for item in selected
            ),
            "reserved_allocated_gpu_capacity_ns": sum(
                item.allocated_gpu_capacity_ns for item in selected
            ),
            "reserved_output_bytes": sum(item.output_bytes for item in selected),
        }
    totals["successful_session"]["source_non_job_output_bytes"] = (
        _P6C_FORMAL_SOURCE_NON_JOB_OUTPUT_BYTES
    )
    totals["successful_session"]["source_total_output_bytes"] = (
        totals["successful_session"]["source_output_bytes"]
        + _P6C_FORMAL_SOURCE_NON_JOB_OUTPUT_BYTES
    )
    physical_gpu_ns = int(
        (
            STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            + STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        )
        * 3_600_000_000_000
    )
    physical_output_bytes = (
        STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES
        + STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES
    )
    reserved_gpu_ns = sum(
        item.allocated_gpu_capacity_ns for item in active_reserves
    )
    reserved_output_bytes = sum(
        item.output_bytes for item in active_reserves
    )
    return {
        "schema_version": STAIR_GENERATION_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION,
        "source": {
            "session_id": _P6C_FORMAL_RESERVE_SOURCE_SESSION_ID,
            "source_commit": _P6C_FORMAL_RESERVE_SOURCE_COMMIT,
            "g4_audit_sha256": _P6C_FORMAL_RESERVE_G4_AUDIT_SHA256,
        },
        "derivation": {
            "rule": "ceil_per_operation",
            "headroom_numerator": _OPERATION_RESERVE_HEADROOM_NUMERATOR,
            "headroom_denominator": _OPERATION_RESERVE_HEADROOM_DENOMINATOR,
            "capacity_time_unit": "allocated_gpu_capacity_ns",
            "output_unit": "bytes",
        },
        "entries": entries,
        "totals": totals,
        "physical_unallocated_headroom": {
            "allocated_gpu_capacity_ns": physical_gpu_ns - reserved_gpu_ns,
            "output_bytes": physical_output_bytes - reserved_output_bytes,
        },
    }


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
            "eval_seed_units_per_generation",
            "public_dev_seed_units_per_job",
            "public_accept_seed_units_per_job",
            "sealed_validation_seed_units_per_job",
            "episodes_per_seed_unit",
            "max_eval_control_steps_per_seed_unit",
            "final_confirmation_seed_units",
        ):
            _positive_int(getattr(self, field_name), field_name)
        require_nonnegative_cost_integer(
            self.parent_bootstrap_seed_units,
            "parent_bootstrap_seed_units",
        )
        if self.train_jobs_per_generation != _GENERATION_WIDTH:
            raise ValueError("scientific train-job count differs from generation width")
        if self.reference_eval_seed_units != (
            self.public_dev_seed_units_per_job
            + self.public_accept_seed_units_per_job
            + self.sealed_validation_seed_units_per_job
        ):
            raise ValueError("reference evaluation seed-unit projection mismatch")
        if self.parent_bootstrap_seed_units not in {
            0,
            self.public_dev_seed_units_per_job,
        }:
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

    @property
    def reuses_reference_public_dev_receipt(self) -> bool:
        """Whether Genesis already owns the parent public-dev guard evidence."""

        return self.parent_bootstrap_seed_units == 0

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
    ordered_suite = config.ordered_evaluation_suite_config
    if ordered_suite is None:
        evaluation_identity_matches = (
            phased.get("evaluation_profile_hash")
            == config.evaluation_profile.profile_hash
        )
        expected_episodes_per_seed = len(
            config.evaluation_profile.private_cell_assignment()
        )
        max_episode_horizon_control_steps = (
            config.evaluation_profile.max_episode_horizon_control_steps
        )
    else:
        evaluation_identity_matches = (
            phased.get("evaluation_suite_hash") == ordered_suite.suite_hash
        )
        expected_episodes_per_seed = len(
            ordered_suite.private_cell_assignment("public_dev")
        )
        max_episode_horizon_control_steps = (
            ordered_suite.max_episode_horizon_control_steps
        )
    if (
        phased.get("task_id") != STAIR_TASK_ID
        or phased.get("evaluation_protocol_hash")
        != config.evaluation_protocol().protocol_hash
        or not evaluation_identity_matches
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
    expected_bootstrap_keys = (
        set() if ordered_suite is not None else {"parent_public_dev"}
    )
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != expected_bootstrap_keys:
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
    if ordered_suite is None:
        parent = bootstrap["parent_public_dev"]
        if not isinstance(parent, Mapping):
            raise TypeError("Stair phased bootstrap cost entry is invalid")
        parent_jobs = _positive_int(parent.get("max_jobs"), "parent bootstrap max_jobs")
        parent_seeds = _positive_int(
            parent.get("seed_count_per_job"),
            "parent bootstrap seed_count_per_job",
        )
    else:
        parent_jobs = 0
        parent_seeds = 0
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
        or (
            ordered_suite is None
            and parent_seeds != config.public_seed_count
        )
        or phased.get("episodes_per_seed_unit") != expected_episodes_per_seed
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

    if ordered_suite is None:
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
        final_confirmation_seed_units = (
            len(execution_order) * final_plan.seed_count
        )
    else:
        # Unified Stair final confirmation compares the frozen Genesis reference
        # with the terminal policy on the sealed cohort.
        final_confirmation_seed_units = 2 * config.sealed_validation_seed_count

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
            episodes_per_seed * max_episode_horizon_control_steps
        ),
        final_confirmation_seed_units=final_confirmation_seed_units,
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

    @property
    def max_infrastructure_retries_per_eligible_job(self) -> int:
        return STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB

    @property
    def uses_operation_reserves(self) -> bool:
        return "operation_reserves" in self.to_dict()

    @property
    def actual_cost_schema_version(self) -> str:
        return STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION

    @property
    def operational_evidence_namespace(self) -> str:
        return "true_rl_agent.stair"

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
        require_nonnegative_cost_integer(self.retry_train_jobs, "retry_train_jobs")
        require_nonnegative_cost_integer(self.retry_eval_seed_units, "retry_eval_seed_units")
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


@dataclass(frozen=True)
class StairGenerationSessionCostContractV2:
    """Strict P6-C budget with separately bound success and retry reserves."""

    reference_phase_reserve_rule = "headroom_on_source_shares"

    @property
    def max_infrastructure_retries_per_eligible_job(self) -> int:
        return STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB

    @property
    def uses_operation_reserves(self) -> bool:
        return "operation_reserves" in self.to_dict()

    @property
    def actual_cost_schema_version(self) -> str:
        return STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION

    @property
    def operational_evidence_namespace(self) -> str:
        return "true_rl_agent.stair"

    @property
    def finalization_output_reserve_bytes(self) -> int:
        return STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES

    session_mode: str
    max_generations: int
    scientific_cost_basis: StairGenerationScientificCostBasis
    retry_train_jobs: int
    retry_eval_seed_units: int
    successful_allocated_active_gpu_hours: float
    retry_allocated_active_gpu_hours: float
    successful_output_bytes: int
    retry_output_bytes: int
    schema_version: str = STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION:
            raise ValueError("Stair generation session cost v2 schema mismatch")
        if self.session_mode != DEVELOPMENT_GENERATION_VALIDATION_MODE:
            raise ValueError("Stair generation session cost v2 mode is unsupported")
        if self.max_generations != _MODE_GENERATIONS[self.session_mode]:
            raise ValueError("Stair generation count does not match session mode")
        if not isinstance(
            self.scientific_cost_basis,
            StairGenerationScientificCostBasis,
        ):
            raise TypeError("Stair generation scientific cost basis is invalid")
        require_nonnegative_cost_integer(self.retry_train_jobs, "retry_train_jobs")
        require_nonnegative_cost_integer(self.retry_eval_seed_units, "retry_eval_seed_units")
        if (
            self.retry_train_jobs != STAIR_GENERATION_RETRY_TRAIN_JOBS
            or self.retry_eval_seed_units != STAIR_GENERATION_RETRY_EVAL_SEED_UNITS
        ):
            raise ValueError("Stair generation session-wide retry reserve mismatch")
        _positive_float(
            self.successful_allocated_active_gpu_hours,
            "successful_allocated_active_gpu_hours",
        )
        _positive_float(
            self.retry_allocated_active_gpu_hours,
            "retry_allocated_active_gpu_hours",
        )
        _positive_int(self.successful_output_bytes, "successful_output_bytes")
        _positive_int(self.retry_output_bytes, "retry_output_bytes")
        if (
            self.successful_allocated_active_gpu_hours
            != STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            or self.retry_allocated_active_gpu_hours
            != STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            or self.successful_output_bytes
            != STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
            or self.retry_output_bytes != STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES
        ):
            raise ValueError("Stair generation validation physical reserve mismatch")
        source_operation_plan = _operation_reserve_plan_payload()
        source_operation_totals = source_operation_plan["totals"]
        source_successful = source_operation_totals["successful_session"]
        source_retry = source_operation_totals["retry_reserve"]
        if (
            source_successful["source_allocated_gpu_capacity_ns"]
            != 31_127_832_110_058
            or source_successful["source_total_output_bytes"] != 2_341_181_253
            or source_retry["source_allocated_gpu_capacity_ns"]
            != 4_172_484_142_958
            or source_retry["source_output_bytes"] != 326_023_821
        ):
            raise ValueError("Stair operation reserve source evidence drifted")
        operation_plan = _operation_reserve_plan_payload(
            include_parent_public_dev=(
                not self.scientific_cost_basis.reuses_reference_public_dev_receipt
            )
        )
        operation_totals = operation_plan["totals"]
        successful = operation_totals["successful_session"]
        retry = operation_totals["retry_reserve"]
        if (
            successful["reserved_allocated_gpu_capacity_ns"]
            > int(self.successful_allocated_active_gpu_hours * 3_600_000_000_000)
            or successful["reserved_output_bytes"] > self.successful_output_bytes
            or retry["reserved_allocated_gpu_capacity_ns"]
            > int(self.retry_allocated_active_gpu_hours * 3_600_000_000_000)
            or retry["reserved_output_bytes"] > self.retry_output_bytes
        ):
            raise ValueError("Stair operation reserves exceed the physical budget")
        if any(
            value < 0
            for value in operation_plan["physical_unallocated_headroom"].values()
        ):
            raise ValueError("Stair operation reserve headroom is negative")

    @property
    def final_confirmation_seed_units(self) -> int:
        return 0

    @property
    def imports_reference_evidence(self) -> bool:
        return False

    @property
    def imported_reference_timeout_envelope(self) -> tuple[int, int] | None:
        return None

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
        )

    @property
    def successful_episodes(self) -> int:
        return (
            self.successful_eval_seed_units
            * self.scientific_cost_basis.episodes_per_seed_unit
        )

    @property
    def successful_simulated_control_steps(self) -> int:
        return self.successful_train_steps + (
            self.successful_eval_seed_units
            * self.scientific_cost_basis.max_eval_control_steps_per_seed_unit
        )

    @property
    def retry_train_steps(self) -> int:
        return self.retry_train_jobs * self.scientific_cost_basis.train_steps_per_job

    @property
    def retry_episodes(self) -> int:
        return (
            self.retry_eval_seed_units
            * self.scientific_cost_basis.episodes_per_seed_unit
        )

    @property
    def retry_simulated_control_steps(self) -> int:
        return self.retry_train_steps + (
            self.retry_eval_seed_units
            * self.scientific_cost_basis.max_eval_control_steps_per_seed_unit
        )

    @property
    def physical_train_job_cap(self) -> int:
        return self.successful_train_jobs + self.retry_train_jobs

    @property
    def physical_train_step_cap(self) -> int:
        return self.successful_train_steps + self.retry_train_steps

    @property
    def physical_eval_seed_unit_cap(self) -> int:
        return self.successful_eval_seed_units + self.retry_eval_seed_units

    @property
    def physical_episode_cap(self) -> int:
        return self.successful_episodes + self.retry_episodes

    @property
    def max_allocated_active_gpu_hours(self) -> float:
        return (
            self.successful_allocated_active_gpu_hours
            + self.retry_allocated_active_gpu_hours
        )

    @property
    def max_allocated_gpu_capacity_ns(self) -> int:
        # Historical contracts retain the exact admission limit they emitted.
        return int(self.max_allocated_active_gpu_hours * 3_600_000_000_000)

    @property
    def max_simulated_control_steps(self) -> int:
        return (
            self.successful_simulated_control_steps + self.retry_simulated_control_steps
        )

    @property
    def max_output_bytes(self) -> int:
        return self.successful_output_bytes + self.retry_output_bytes

    @property
    def contract_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        provider_attempts = self.provider_attempt_cap
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
                    "train_steps": self.scientific_cost_basis.train_steps_per_job,
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
                "final_confirmation_seed_units": self.final_confirmation_seed_units,
                "train_jobs": self.successful_train_jobs,
                "train_steps": self.successful_train_steps,
                "eval_seed_units": self.successful_eval_seed_units,
                "episodes": self.successful_episodes,
                "simulated_control_steps": (self.successful_simulated_control_steps),
                "allocated_active_gpu_hours": float(
                    self.successful_allocated_active_gpu_hours
                ),
                "output_bytes": self.successful_output_bytes,
            },
            "retry_reserve": {
                "train_jobs": self.retry_train_jobs,
                "train_steps": self.retry_train_steps,
                "eval_seed_units": self.retry_eval_seed_units,
                "episodes": self.retry_episodes,
                "simulated_control_steps": self.retry_simulated_control_steps,
                "allocated_active_gpu_hours": float(
                    self.retry_allocated_active_gpu_hours
                ),
                "output_bytes": self.retry_output_bytes,
            },
            "operation_reserves": _operation_reserve_plan_payload(
                include_parent_public_dev=(
                    not self.scientific_cost_basis.reuses_reference_public_dev_receipt
                )
            ),
            "physical_cap": {
                "train_jobs": self.physical_train_job_cap,
                "train_steps": self.physical_train_step_cap,
                "eval_seed_units": self.physical_eval_seed_unit_cap,
                "episodes": self.physical_episode_cap,
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
    ) -> StairGenerationSessionCostContractV2:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation session cost v2 must be an object")
        successful = value.get("successful_session")
        retry = value.get("retry_reserve")
        scientific_cost_basis = value.get("scientific_cost_basis")
        if (
            not isinstance(successful, Mapping)
            or not isinstance(retry, Mapping)
            or not isinstance(scientific_cost_basis, Mapping)
        ):
            raise TypeError("Stair generation session cost v2 sections are invalid")
        contract = cls(
            schema_version=value.get("schema_version"),
            session_mode=value.get("session_mode"),
            max_generations=value.get("max_generations"),
            scientific_cost_basis=(
                StairGenerationScientificCostBasis.from_mapping(scientific_cost_basis)
            ),
            retry_train_jobs=retry.get("train_jobs"),
            retry_eval_seed_units=retry.get("eval_seed_units"),
            successful_allocated_active_gpu_hours=successful.get(
                "allocated_active_gpu_hours"
            ),
            retry_allocated_active_gpu_hours=retry.get("allocated_active_gpu_hours"),
            successful_output_bytes=successful.get("output_bytes"),
            retry_output_bytes=retry.get("output_bytes"),
        )
        if dict(value) != contract.to_dict():
            raise ValueError("Stair generation session cost v2 projection mismatch")
        return contract


@dataclass(frozen=True)
class StairGenerationSessionCostContractV3(
    StairGenerationSessionCostContractV2
):
    """Strict P6-D budget including final confirmation and P6-C evidence."""

    schema_version: str = STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION:
            raise ValueError("Stair generation session cost v3 schema mismatch")
        if self.session_mode != FORMAL_GENERATION_SESSION_MODE:
            raise ValueError("Stair generation session cost v3 mode is unsupported")
        if self.max_generations != _MODE_GENERATIONS[self.session_mode]:
            raise ValueError("Stair generation count does not match session mode")
        if not isinstance(
            self.scientific_cost_basis,
            StairGenerationScientificCostBasis,
        ):
            raise TypeError("Stair generation scientific cost basis is invalid")
        require_nonnegative_cost_integer(self.retry_train_jobs, "retry_train_jobs")
        require_nonnegative_cost_integer(self.retry_eval_seed_units, "retry_eval_seed_units")
        if (
            self.retry_train_jobs != STAIR_GENERATION_RETRY_TRAIN_JOBS
            or self.retry_eval_seed_units != STAIR_GENERATION_RETRY_EVAL_SEED_UNITS
        ):
            raise ValueError("Stair generation session-wide retry reserve mismatch")
        _positive_float(
            self.successful_allocated_active_gpu_hours,
            "successful_allocated_active_gpu_hours",
        )
        _positive_float(
            self.retry_allocated_active_gpu_hours,
            "retry_allocated_active_gpu_hours",
        )
        _positive_int(self.successful_output_bytes, "successful_output_bytes")
        _positive_int(self.retry_output_bytes, "retry_output_bytes")
        if (
            self.successful_allocated_active_gpu_hours
            != STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            or self.retry_allocated_active_gpu_hours
            != STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            or self.successful_output_bytes
            != STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES
            or self.retry_output_bytes != STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES
        ):
            raise ValueError("Stair generation formal physical reserve mismatch")
        source_operation_plan = _formal_operation_reserve_plan_payload()
        source_successful = source_operation_plan["totals"]["successful_session"]
        source_retry = source_operation_plan["totals"]["retry_reserve"]
        if (
            source_successful["reserved_allocated_gpu_capacity_ns"]
            != 175_397_494_998_204
            or source_successful["reserved_output_bytes"] != 12_957_346_478
            or source_retry["reserved_allocated_gpu_capacity_ns"]
            != 5_253_617_286_198
            or source_retry["reserved_output_bytes"] != 417_801_847
        ):
            raise ValueError("Stair formal operation reserve evidence drifted")
        operation_plan = _formal_operation_reserve_plan_payload(
            include_parent_public_dev=(
                not self.scientific_cost_basis.reuses_reference_public_dev_receipt
            )
        )
        successful = operation_plan["totals"]["successful_session"]
        retry = operation_plan["totals"]["retry_reserve"]
        if (
            successful["reserved_allocated_gpu_capacity_ns"]
            > int(self.successful_allocated_active_gpu_hours * 3_600_000_000_000)
            or successful["reserved_output_bytes"] > self.successful_output_bytes
            or retry["reserved_allocated_gpu_capacity_ns"]
            > int(self.retry_allocated_active_gpu_hours * 3_600_000_000_000)
            or retry["reserved_output_bytes"] > self.retry_output_bytes
        ):
            raise ValueError("Stair formal operation reserves exceed physical budget")
        if any(
            value < 0
            for value in operation_plan["physical_unallocated_headroom"].values()
        ):
            raise ValueError("Stair formal operation reserve headroom is negative")

    @property
    def final_confirmation_seed_units(self) -> int:
        return self.scientific_cost_basis.final_confirmation_seed_units

    @property
    def successful_eval_seed_units(self) -> int:
        return (
            super().successful_eval_seed_units
            + self.final_confirmation_seed_units
        )

    def _hash_payload(self) -> dict[str, Any]:
        payload = super()._hash_payload()
        payload["operation_reserves"] = _formal_operation_reserve_plan_payload(
            include_parent_public_dev=(
                not self.scientific_cost_basis.reuses_reference_public_dev_receipt
            )
        )
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StairGenerationSessionCostContractV3:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation session cost v3 must be an object")
        successful = value.get("successful_session")
        retry = value.get("retry_reserve")
        scientific_cost_basis = value.get("scientific_cost_basis")
        if (
            not isinstance(successful, Mapping)
            or not isinstance(retry, Mapping)
            or not isinstance(scientific_cost_basis, Mapping)
        ):
            raise TypeError("Stair generation session cost v3 sections are invalid")
        contract = cls(
            schema_version=value.get("schema_version"),
            session_mode=value.get("session_mode"),
            max_generations=value.get("max_generations"),
            scientific_cost_basis=(
                StairGenerationScientificCostBasis.from_mapping(scientific_cost_basis)
            ),
            retry_train_jobs=retry.get("train_jobs"),
            retry_eval_seed_units=retry.get("eval_seed_units"),
            successful_allocated_active_gpu_hours=successful.get(
                "allocated_active_gpu_hours"
            ),
            retry_allocated_active_gpu_hours=retry.get("allocated_active_gpu_hours"),
            successful_output_bytes=successful.get("output_bytes"),
            retry_output_bytes=retry.get("output_bytes"),
        )
        if dict(value) != contract.to_dict():
            raise ValueError("Stair generation session cost v3 projection mismatch")
        return contract


@dataclass(frozen=True)
class StairGenerationSessionCostContractV4(
    StairGenerationSessionCostContractV2
):
    """Validation budget for a session importing an already-qualified reference."""

    candidate_train_process_timeout_s: int = 7_200
    termination_cleanup_margin_s: int = 16
    schema_version: str = STAIR_GENERATION_SESSION_COST_V4_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SESSION_COST_V4_SCHEMA_VERSION:
            raise ValueError("Stair generation session cost v4 schema mismatch")
        if self.session_mode != DEVELOPMENT_GENERATION_VALIDATION_MODE:
            raise ValueError("Stair generation session cost v4 mode is unsupported")
        if self.max_generations != _MODE_GENERATIONS[self.session_mode]:
            raise ValueError("Stair generation count does not match session mode")
        if not isinstance(
            self.scientific_cost_basis,
            StairGenerationScientificCostBasis,
        ):
            raise TypeError("Stair generation scientific cost basis is invalid")
        if not self.scientific_cost_basis.reuses_reference_public_dev_receipt:
            raise ValueError("Stair cost v4 requires imported reference evidence")
        require_nonnegative_cost_integer(self.retry_train_jobs, "retry_train_jobs")
        require_nonnegative_cost_integer(self.retry_eval_seed_units, "retry_eval_seed_units")
        if (
            self.retry_train_jobs != 0
            or self.retry_eval_seed_units
            != self.scientific_cost_basis.sealed_validation_seed_units_per_job
        ):
            raise ValueError("Stair imported-reference retry reserve mismatch")
        _positive_int(
            self.candidate_train_process_timeout_s,
            "candidate_train_process_timeout_s",
        )
        _positive_int(
            self.termination_cleanup_margin_s,
            "termination_cleanup_margin_s",
        )
        _positive_float(
            self.successful_allocated_active_gpu_hours,
            "successful_allocated_active_gpu_hours",
        )
        _positive_float(
            self.retry_allocated_active_gpu_hours,
            "retry_allocated_active_gpu_hours",
        )
        _positive_int(self.successful_output_bytes, "successful_output_bytes")
        _positive_int(self.retry_output_bytes, "retry_output_bytes")
        if (
            self.successful_allocated_active_gpu_hours
            != STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            or self.retry_allocated_active_gpu_hours
            != STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            or self.successful_output_bytes
            != STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
            or self.retry_output_bytes
            != STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES
        ):
            raise ValueError(
                "Stair imported-reference validation physical reserve mismatch"
            )

        operation_plan = _imported_reference_operation_reserve_plan_payload(
            candidate_train_process_timeout_s=(
                self.candidate_train_process_timeout_s
            ),
            termination_cleanup_margin_s=self.termination_cleanup_margin_s,
        )
        successful_entries = tuple(
            entry
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "successful_session"
        )
        retry_entries = tuple(
            entry
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "retry_reserve"
        )
        successful_train_multiplicity = sum(
            int(entry["multiplicity"])
            for entry in successful_entries
            if str(entry["operation_class"]).endswith("_train")
        )
        if successful_train_multiplicity != self.successful_train_jobs:
            raise ValueError(
                "Stair imported-reference train reserve multiplicity mismatch"
            )
        nontrain_success_ns = sum(
            int(entry["allocated_gpu_capacity_ns"])
            for entry in successful_entries
            if not str(entry["operation_class"]).endswith("_train")
        )
        train_lease_ceiling_ns = (
            self.successful_train_jobs
            * (
                self.candidate_train_process_timeout_s
                + self.termination_cleanup_margin_s
            )
            * 1_000_000_000
        )
        successful_cap_ns = int(
            self.successful_allocated_active_gpu_hours * 3_600_000_000_000
        )
        retry_cap_ns = int(
            self.retry_allocated_active_gpu_hours * 3_600_000_000_000
        )
        if train_lease_ceiling_ns + nontrain_success_ns > successful_cap_ns:
            raise ValueError(
                "Stair candidate-train timeout envelope exceeds successful budget"
            )
        if (
            sum(int(entry["allocated_gpu_capacity_ns"]) for entry in retry_entries)
            > retry_cap_ns
        ):
            raise ValueError("Stair evaluation retry reserve exceeds retry budget")
        if (
            operation_plan["totals"]["successful_session"]["reserved_output_bytes"]
            > self.successful_output_bytes
            or operation_plan["totals"]["retry_reserve"]["reserved_output_bytes"]
            > self.retry_output_bytes
        ):
            raise ValueError("Stair imported-reference output reserves exceed budget")
        if any(
            value < 0
            for value in operation_plan["physical_unallocated_headroom"].values()
        ):
            raise ValueError(
                "Stair imported-reference operation reserve headroom is negative"
            )

    @property
    def successful_train_jobs(self) -> int:
        return self.max_generations * self.scientific_cost_basis.train_jobs_per_generation

    @property
    def imports_reference_evidence(self) -> bool:
        return True

    @property
    def imported_reference_timeout_envelope(self) -> tuple[int, int]:
        return (
            self.candidate_train_process_timeout_s,
            self.termination_cleanup_margin_s,
        )

    @property
    def successful_eval_seed_units(self) -> int:
        return (
            self.max_generations
            * self.scientific_cost_basis.eval_seed_units_per_generation
        )

    @property
    def max_allocated_active_gpu_hours(self) -> float:
        capacity_ns = int(
            self.successful_allocated_active_gpu_hours * 3_600_000_000_000
        ) + int(self.retry_allocated_active_gpu_hours * 3_600_000_000_000)
        return capacity_ns / 3_600_000_000_000

    def _hash_payload(self) -> dict[str, Any]:
        payload = super()._hash_payload()
        successful = payload["successful_session"]
        successful.pop("reference_qualification")
        successful["imported_reference_evidence"] = {
            "execution": "imported_verified_receipts",
            "local_train_jobs": 0,
            "local_train_steps": 0,
            "local_eval_seed_units": 0,
        }
        operation_plan = _imported_reference_operation_reserve_plan_payload(
            candidate_train_process_timeout_s=(
                self.candidate_train_process_timeout_s
            ),
            termination_cleanup_margin_s=self.termination_cleanup_margin_s,
        )
        payload["operation_reserves"] = operation_plan
        nontrain_success_ns = sum(
            int(entry["allocated_gpu_capacity_ns"])
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "successful_session"
            and not str(entry["operation_class"]).endswith("_train")
        )
        train_lease_ceiling_s = (
            self.candidate_train_process_timeout_s
            + self.termination_cleanup_margin_s
        )
        train_lease_ceiling_ns = (
            self.successful_train_jobs * train_lease_ceiling_s * 1_000_000_000
        )
        success_cap_ns = int(
            self.successful_allocated_active_gpu_hours * 3_600_000_000_000
        )
        payload["candidate_train_timeout_envelope"] = {
            "timeout_authority": "runner_timeout_s",
            "process_timeout_s": self.candidate_train_process_timeout_s,
            "termination_cleanup_margin_s": self.termination_cleanup_margin_s,
            "lease_ceiling_s_per_attempt": train_lease_ceiling_s,
            "attempt_count": self.successful_train_jobs,
            "allocated_gpu_capacity_ns": train_lease_ceiling_ns,
            "nontrain_success_reserve_ns": nontrain_success_ns,
            "successful_budget_headroom_ns": (
                success_cap_ns - train_lease_ceiling_ns - nontrain_success_ns
            ),
        }
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StairGenerationSessionCostContractV4:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation session cost v4 must be an object")
        successful = value.get("successful_session")
        retry = value.get("retry_reserve")
        scientific_cost_basis = value.get("scientific_cost_basis")
        timeout_envelope = value.get("candidate_train_timeout_envelope")
        if (
            not isinstance(successful, Mapping)
            or not isinstance(retry, Mapping)
            or not isinstance(scientific_cost_basis, Mapping)
            or not isinstance(timeout_envelope, Mapping)
        ):
            raise TypeError("Stair generation session cost v4 sections are invalid")
        contract = cls(
            schema_version=value.get("schema_version"),
            session_mode=value.get("session_mode"),
            max_generations=value.get("max_generations"),
            scientific_cost_basis=(
                StairGenerationScientificCostBasis.from_mapping(scientific_cost_basis)
            ),
            retry_train_jobs=retry.get("train_jobs"),
            retry_eval_seed_units=retry.get("eval_seed_units"),
            successful_allocated_active_gpu_hours=successful.get(
                "allocated_active_gpu_hours"
            ),
            retry_allocated_active_gpu_hours=retry.get("allocated_active_gpu_hours"),
            successful_output_bytes=successful.get("output_bytes"),
            retry_output_bytes=retry.get("output_bytes"),
            candidate_train_process_timeout_s=timeout_envelope.get(
                "process_timeout_s"
            ),
            termination_cleanup_margin_s=timeout_envelope.get(
                "termination_cleanup_margin_s"
            ),
        )
        if dict(value) != contract.to_dict():
            raise ValueError("Stair generation session cost v4 projection mismatch")
        return contract


@dataclass(frozen=True)
class StairGenerationSessionCostContractV5(
    StairGenerationSessionCostContractV3
):
    """Formal budget for a session importing an already-qualified reference."""

    candidate_train_process_timeout_s: int = 7_200
    termination_cleanup_margin_s: int = 16
    schema_version: str = STAIR_GENERATION_SESSION_COST_V5_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_SESSION_COST_V5_SCHEMA_VERSION:
            raise ValueError("Stair generation session cost v5 schema mismatch")
        if self.session_mode != FORMAL_GENERATION_SESSION_MODE:
            raise ValueError("Stair generation session cost v5 mode is unsupported")
        if self.max_generations != _MODE_GENERATIONS[self.session_mode]:
            raise ValueError("Stair generation count does not match session mode")
        if not isinstance(
            self.scientific_cost_basis,
            StairGenerationScientificCostBasis,
        ):
            raise TypeError("Stair generation scientific cost basis is invalid")
        if not self.scientific_cost_basis.reuses_reference_public_dev_receipt:
            raise ValueError("Stair cost v5 requires imported reference evidence")
        require_nonnegative_cost_integer(self.retry_train_jobs, "retry_train_jobs")
        require_nonnegative_cost_integer(self.retry_eval_seed_units, "retry_eval_seed_units")
        if (
            self.retry_train_jobs != 0
            or self.retry_eval_seed_units
            != self.scientific_cost_basis.sealed_validation_seed_units_per_job
        ):
            raise ValueError("Stair imported-reference retry reserve mismatch")
        _positive_int(
            self.candidate_train_process_timeout_s,
            "candidate_train_process_timeout_s",
        )
        _positive_int(
            self.termination_cleanup_margin_s,
            "termination_cleanup_margin_s",
        )
        _positive_float(
            self.successful_allocated_active_gpu_hours,
            "successful_allocated_active_gpu_hours",
        )
        _positive_float(
            self.retry_allocated_active_gpu_hours,
            "retry_allocated_active_gpu_hours",
        )
        _positive_int(self.successful_output_bytes, "successful_output_bytes")
        _positive_int(self.retry_output_bytes, "retry_output_bytes")
        if (
            self.successful_allocated_active_gpu_hours
            != STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            or self.retry_allocated_active_gpu_hours
            != STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            or self.successful_output_bytes
            != STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES
            or self.retry_output_bytes
            != STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES
        ):
            raise ValueError(
                "Stair imported-reference formal physical reserve mismatch"
            )

        operation_plan = _imported_reference_operation_reserve_plan_payload(
            candidate_train_process_timeout_s=(
                self.candidate_train_process_timeout_s
            ),
            termination_cleanup_margin_s=self.termination_cleanup_margin_s,
            session_mode=self.session_mode,
        )
        successful_entries = tuple(
            entry
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "successful_session"
        )
        retry_entries = tuple(
            entry
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "retry_reserve"
        )
        successful_train_multiplicity = sum(
            int(entry["multiplicity"])
            for entry in successful_entries
            if str(entry["operation_class"]).endswith("_train")
        )
        if successful_train_multiplicity != self.successful_train_jobs:
            raise ValueError(
                "Stair imported-reference train reserve multiplicity mismatch"
            )
        nontrain_success_ns = sum(
            int(entry["allocated_gpu_capacity_ns"])
            for entry in successful_entries
            if not str(entry["operation_class"]).endswith("_train")
        )
        train_lease_ceiling_ns = (
            self.successful_train_jobs
            * (
                self.candidate_train_process_timeout_s
                + self.termination_cleanup_margin_s
            )
            * 1_000_000_000
        )
        successful_cap_ns = int(
            self.successful_allocated_active_gpu_hours * 3_600_000_000_000
        )
        retry_cap_ns = int(
            self.retry_allocated_active_gpu_hours * 3_600_000_000_000
        )
        if train_lease_ceiling_ns + nontrain_success_ns > successful_cap_ns:
            raise ValueError(
                "Stair candidate-train timeout envelope exceeds successful budget"
            )
        if (
            sum(int(entry["allocated_gpu_capacity_ns"]) for entry in retry_entries)
            > retry_cap_ns
        ):
            raise ValueError("Stair evaluation retry reserve exceeds retry budget")
        if (
            operation_plan["totals"]["successful_session"]["reserved_output_bytes"]
            > self.successful_output_bytes
            or operation_plan["totals"]["retry_reserve"]["reserved_output_bytes"]
            > self.retry_output_bytes
        ):
            raise ValueError("Stair imported-reference output reserves exceed budget")
        if any(
            value < 0
            for value in operation_plan["physical_unallocated_headroom"].values()
        ):
            raise ValueError(
                "Stair imported-reference operation reserve headroom is negative"
            )

    @property
    def successful_train_jobs(self) -> int:
        return self.max_generations * self.scientific_cost_basis.train_jobs_per_generation

    @property
    def successful_eval_seed_units(self) -> int:
        return (
            self.max_generations
            * self.scientific_cost_basis.eval_seed_units_per_generation
            + self.final_confirmation_seed_units
        )

    @property
    def imports_reference_evidence(self) -> bool:
        return True

    @property
    def imported_reference_timeout_envelope(self) -> tuple[int, int]:
        return (
            self.candidate_train_process_timeout_s,
            self.termination_cleanup_margin_s,
        )

    def _hash_payload(self) -> dict[str, Any]:
        payload = StairGenerationSessionCostContractV2._hash_payload(self)
        successful = payload["successful_session"]
        successful.pop("reference_qualification")
        successful["imported_reference_evidence"] = {
            "execution": "imported_verified_receipts",
            "local_train_jobs": 0,
            "local_train_steps": 0,
            "local_eval_seed_units": 0,
        }
        operation_plan = _imported_reference_operation_reserve_plan_payload(
            candidate_train_process_timeout_s=(
                self.candidate_train_process_timeout_s
            ),
            termination_cleanup_margin_s=self.termination_cleanup_margin_s,
            session_mode=self.session_mode,
        )
        payload["operation_reserves"] = operation_plan
        nontrain_success_ns = sum(
            int(entry["allocated_gpu_capacity_ns"])
            for entry in operation_plan["entries"]
            if entry["budget_class"] == "successful_session"
            and not str(entry["operation_class"]).endswith("_train")
        )
        train_lease_ceiling_s = (
            self.candidate_train_process_timeout_s
            + self.termination_cleanup_margin_s
        )
        train_lease_ceiling_ns = (
            self.successful_train_jobs * train_lease_ceiling_s * 1_000_000_000
        )
        success_cap_ns = int(
            self.successful_allocated_active_gpu_hours * 3_600_000_000_000
        )
        payload["candidate_train_timeout_envelope"] = {
            "timeout_authority": "runner_timeout_s",
            "process_timeout_s": self.candidate_train_process_timeout_s,
            "termination_cleanup_margin_s": self.termination_cleanup_margin_s,
            "lease_ceiling_s_per_attempt": train_lease_ceiling_s,
            "attempt_count": self.successful_train_jobs,
            "allocated_gpu_capacity_ns": train_lease_ceiling_ns,
            "nontrain_success_reserve_ns": nontrain_success_ns,
            "successful_budget_headroom_ns": (
                success_cap_ns - train_lease_ceiling_ns - nontrain_success_ns
            ),
        }
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StairGenerationSessionCostContractV5:
        if not isinstance(value, Mapping):
            raise TypeError("Stair generation session cost v5 must be an object")
        successful = value.get("successful_session")
        retry = value.get("retry_reserve")
        scientific_cost_basis = value.get("scientific_cost_basis")
        timeout_envelope = value.get("candidate_train_timeout_envelope")
        if (
            not isinstance(successful, Mapping)
            or not isinstance(retry, Mapping)
            or not isinstance(scientific_cost_basis, Mapping)
            or not isinstance(timeout_envelope, Mapping)
        ):
            raise TypeError("Stair generation session cost v5 sections are invalid")
        contract = cls(
            schema_version=value.get("schema_version"),
            session_mode=value.get("session_mode"),
            max_generations=value.get("max_generations"),
            scientific_cost_basis=(
                StairGenerationScientificCostBasis.from_mapping(scientific_cost_basis)
            ),
            retry_train_jobs=retry.get("train_jobs"),
            retry_eval_seed_units=retry.get("eval_seed_units"),
            successful_allocated_active_gpu_hours=successful.get(
                "allocated_active_gpu_hours"
            ),
            retry_allocated_active_gpu_hours=retry.get("allocated_active_gpu_hours"),
            successful_output_bytes=successful.get("output_bytes"),
            retry_output_bytes=retry.get("output_bytes"),
            candidate_train_process_timeout_s=timeout_envelope.get(
                "process_timeout_s"
            ),
            termination_cleanup_margin_s=timeout_envelope.get(
                "termination_cleanup_margin_s"
            ),
        )
        if dict(value) != contract.to_dict():
            raise ValueError("Stair generation session cost v5 projection mismatch")
        return contract


StairGenerationSessionCostContractLike = (
    StairGenerationSessionCostContract
    | StairGenerationSessionCostContractV2
    | StairGenerationSessionCostContractV3
    | StairGenerationSessionCostContractV4
    | StairGenerationSessionCostContractV5
)


def create_stair_generation_session_cost_contract(
    output_dir: Path,
    contract: StairGenerationSessionCostContractLike,
) -> Path:
    if not isinstance(contract, StairGenerationSessionCostContractLike):
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
) -> StairGenerationSessionCostContractLike:
    payload = load_strict_json_object(
        Path(path),
        context="Stair generation session cost contract",
    )
    schema_version = payload.get("schema_version")
    if schema_version == STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION:
        return StairGenerationSessionCostContract.from_mapping(payload)
    if schema_version == STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION:
        return StairGenerationSessionCostContractV2.from_mapping(payload)
    if schema_version == STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION:
        return StairGenerationSessionCostContractV3.from_mapping(payload)
    if schema_version == STAIR_GENERATION_SESSION_COST_V4_SCHEMA_VERSION:
        return StairGenerationSessionCostContractV4.from_mapping(payload)
    if schema_version == STAIR_GENERATION_SESSION_COST_V5_SCHEMA_VERSION:
        return StairGenerationSessionCostContractV5.from_mapping(payload)
    raise ValueError("Stair generation session cost schema is unsupported")




















def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value




def _positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite float")
    return value


__all__ = [
    "DEVELOPMENT_GENERATION_PILOT_MODE",
    "DEVELOPMENT_GENERATION_VALIDATION_MODE",
    "FORMAL_GENERATION_SESSION_MODE",
    "STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES",
    "STAIR_GENERATION_ACTUAL_COST_DIRNAME",
    "STAIR_GENERATION_ACTUAL_COST_SCHEMA_VERSION",
    "STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES",
    "STAIR_GENERATION_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION",
    "STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES",
    "STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES",
    "STAIR_GENERATION_IMPORTED_REFERENCE_OPERATION_RESERVES_SCHEMA_VERSION",
    "STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_OPERATION_RESERVES_SCHEMA_VERSION",
    "STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES",
    "STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES",
    "STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES",
    "STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES",
    "STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB",
    "STAIR_GENERATION_OPERATION_RESERVES_SCHEMA_VERSION",
    "STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE",
    "STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT",
    "STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT",
    "STAIR_GENERATION_RETRY_EVAL_SEED_UNITS",
    "STAIR_GENERATION_RETRY_TRAIN_JOBS",
    "STAIR_GENERATION_SCIENTIFIC_COST_BASIS_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_FILENAME",
    "STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_V4_SCHEMA_VERSION",
    "STAIR_GENERATION_SESSION_COST_V5_SCHEMA_VERSION",
    "STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES",
    "STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS",
    "STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES",
    "StairGenerationActualCostJournal",
    "StairGenerationActualCostReceiptRef",
    "StairGenerationActualCostUsage",
    "StairGenerationActualCostUsageV2",
    "StairGenerationOperationCostReserve",
    "StairGenerationTimeoutBoundOperationCostReserve",
    "StairGenerationScientificCostBasis",
    "StairGenerationSessionCostContract",
    "StairGenerationSessionCostContractLike",
    "StairGenerationSessionCostContractV2",
    "StairGenerationSessionCostContractV3",
    "StairGenerationSessionCostContractV4",
    "StairGenerationSessionCostContractV5",
    "build_stair_generation_scientific_cost_basis",
    "create_stair_generation_session_cost_contract",
    "imported_reference_operation_cost_reserves",
    "load_stair_generation_session_cost_contract",
    "replay_stair_generation_combined_cost_chain",
    "replay_stair_generation_formal_cost_chain",
    "stair_formal_generation_operation_cost_reserves",
    "stair_generation_operation_cost_reserves",
]

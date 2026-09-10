"""Allocate fixed generation jobs from explicit frozen operation costs.

Job-attempt authority owns process/retry identity. Task-specific metrics and
cost measurements enter only through the resolved cost contract.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_cost import GenerationCostContract, OperationCostReserve
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.job_attempts import (
    RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptJournal, JobAttemptScope, PreGenesisExecutionScope,
)


@dataclass(frozen=True)
class OperationAllocationState:
    reserved_success_slots: frozenset[str]
    reserved_retry_classes: frozenset[str]
    attempt_results: Mapping[str, Mapping[str, Any] | None]
    measured_gpu_capacity_ns: int
    conservatively_charged_gpu_capacity_ns: int
    unmeasured_bound_attempts: int
    open_attempt_output_reserve_bytes: int



def _strict_operation_reserves_by_class(
    cost_contract: GenerationCostContract,
) -> dict[str, OperationCostReserve]:
    frozen = cost_contract.to_dict()["operation_reserves"]["entries"]
    result = {item["operation_class"]: OperationCostReserve.from_mapping(item)
              for item in frozen}
    if len(result) != len(frozen):
        raise ValueError("operation reserve classes are duplicated")
    return result



def _reference_qualification_phase_reserves(
    cost_contract: GenerationCostContract,
) -> dict[EvaluationPhase, OperationCostReserve]:
    """Split one frozen reference-eval reserve across its three processes."""

    combined = _strict_operation_reserves_by_class(cost_contract)[
        "reference_qualification"
    ]
    weights = {
        phase: cost_contract.scientific_cost_basis.seed_units_for_phase(phase.value)
        for phase in EvaluationPhase
    }
    total_weight = sum(weights.values())

    def shares(total: int) -> dict[EvaluationPhase, int]:
        consumed_weight = 0
        previous = 0
        result: dict[EvaluationPhase, int] = {}
        for phase in EvaluationPhase:
            consumed_weight += weights[phase]
            current = total * consumed_weight // total_weight
            result[phase] = current - previous
            previous = current
        if sum(result.values()) != total or any(
            value <= 0 for value in result.values()
        ):
            raise ValueError("reference qualification reserve cannot be split")
        return result

    gpu_shares = shares(combined.source_allocated_gpu_capacity_ns)
    output_shares = shares(combined.source_output_bytes)
    if cost_contract.reference_phase_reserve_rule == "apportion_frozen_totals":
        # Partition the already frozen reserve. Applying headroom a second time
        # to rounded source shares can exceed the combined slot by a nanosecond.
        allocated = shares(combined.allocated_gpu_capacity_ns_per_attempt)
        outputs = shares(combined.output_bytes_per_attempt)
        return {phase: OperationCostReserve(
            operation_class=f"reference_qualification_{phase.value}",
            budget_class=combined.budget_class, multiplicity=1,
            source_allocated_gpu_capacity_ns=gpu_shares[phase],
            source_output_bytes=output_shares[phase],
            allocated_gpu_capacity_ns_per_attempt=allocated[phase],
            output_bytes_per_attempt=outputs[phase],
            allocated_gpu_capacity_authority=combined.allocated_gpu_capacity_authority,
        ) for phase in EvaluationPhase}
    return {
        phase: OperationCostReserve.from_source(
            operation_class=f"reference_qualification_{phase.value}",
            budget_class=combined.budget_class,
            multiplicity=1,
            source_allocated_gpu_capacity_ns=gpu_shares[phase],
            source_output_bytes=output_shares[phase],
            headroom_numerator=cost_contract.to_dict()["operation_reserves"]["derivation"]["headroom_numerator"],
            headroom_denominator=cost_contract.to_dict()["operation_reserves"]["derivation"]["headroom_denominator"],
        )
        for phase in EvaluationPhase
    }



def read_operation_slot(
    *,
    session_id: str,
    scope: JobAttemptScope | PreGenesisExecutionScope,
    logical_job_id: str,
    job_kind: str,
    attempt_index: int,
    cost_contract: GenerationCostContract,
) -> tuple[str, OperationCostReserve]:
    reuses_reference_public_dev = (
        cost_contract.scientific_cost_basis.reuses_reference_public_dev_receipt
    )
    reserves = _strict_operation_reserves_by_class(cost_contract)
    if cost_contract.imports_reference_evidence and isinstance(
        scope, PreGenesisExecutionScope
    ):
        raise ValueError(
            "imported-reference session cannot execute pre-Genesis work"
        )
    max_generations = cost_contract.max_generations
    allowed_retries = frozenset({0})
    retry_class: str | None = None
    if logical_job_id == f"{session_id}-train":
        if (
            not isinstance(scope, PreGenesisExecutionScope)
            or scope.session_id != session_id
            or scope.lane_id != "lane-a"
            or job_kind != "train"
        ):
            raise ValueError("bounded generation reference-train attempt binding is invalid")
        success_slot = "reference_train"
        success_class = "reference_train"
        retry_class = "reference_train_retry"
        allowed_retries = frozenset({1})
    elif logical_job_id == f"{session_id}-eval":
        if (
            reuses_reference_public_dev
            or not isinstance(scope, PreGenesisExecutionScope)
            or scope.session_id != session_id
            or scope.lane_id != "lane-a"
            or job_kind != "eval"
        ):
            raise ValueError("bounded generation reference-eval attempt binding is invalid")
        success_slot = "reference_qualification"
        success_class = "reference_qualification"
    elif (
        reuses_reference_public_dev
        and isinstance(scope, PreGenesisExecutionScope)
        and logical_job_id.startswith(f"{session_id}-eval-")
    ):
        suffix = logical_job_id.removeprefix(f"{session_id}-eval-")
        phase_by_suffix = {
            phase.value.replace("_", "-"): phase for phase in EvaluationPhase
        }
        phase = phase_by_suffix.get(suffix)
        if (
            phase is None
            or scope.session_id != session_id
            or scope.lane_id != "lane-a"
            or scope.invocation_segment_id
            != "session-genesis-" + phase.value.replace("_", "-")
            or job_kind != "eval_phase"
        ):
            raise ValueError(
                "bounded generation phased reference-eval binding is invalid"
            )
        if scope.max_infrastructure_retries != 0 or attempt_index != 0:
            raise ValueError(
                "bounded generation phased reference-eval retry is invalid"
            )
        return (
            f"reference_qualification:{phase.value}",
            _reference_qualification_phase_reserves(cost_contract)[phase],
        )
    elif (
        not reuses_reference_public_dev
        and logical_job_id == f"{session_id}-eval-public-dev"
    ):
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != "generation-parent-public-dev-bootstrap"
            or scope.lane_id != "lane-a"
            or job_kind != "eval_phase"
        ):
            raise ValueError("bounded generation parent-bootstrap attempt binding is invalid")
        success_slot = "parent_public_dev"
        success_class = "parent_public_dev"
    elif (
        (cost_contract.final_confirmation_seed_units > 0)
        and logical_job_id
        in {
            f"{session_id}-final-confirmation-genesis-eval-sealed-validation",
            f"{session_id}-final-confirmation-final-eval-sealed-validation",
        }
    ):
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != "generation-final-confirmation"
            or scope.lane_id != "lane-a"
            or job_kind != "eval_phase"
        ):
            raise ValueError("formal final confirmation attempt binding is invalid")
        role = (
            "genesis" if "-genesis-" in logical_job_id else "final"
        )
        success_slot = f"final-confirmation:{role}"
        success_class = "sealed_validation"
        retry_class = "sealed_validation_retry"
        allowed_retries = frozenset({0, 1})
    else:
        match = re.fullmatch(
            re.escape(session_id) + r"-generation-([0-9]{3})-lane-(a|b)-"
            r"(train|eval-public-dev|eval-public-accept|eval-sealed-validation)",
            logical_job_id,
        )
        if match is None or not isinstance(scope, JobAttemptScope):
            raise ValueError("bounded generation attempt has no frozen operation allocation")
        generation_text, lane_suffix, operation_suffix = match.groups()
        generation_index = int(generation_text)
        if not 1 <= generation_index <= max_generations:
            raise ValueError("generation attempt exceeds its frozen session")
        lane_id = f"lane-{lane_suffix}"
        generation_id = f"{session_id}-generation-{generation_index:03d}"
        if scope.generation_id != generation_id or scope.lane_id != lane_id:
            raise ValueError("bounded generation attempt scope is misbound")
        if operation_suffix == "train":
            expected_kind = "train"
            success_class = f"lane_{lane_suffix}_train"
            success_slot = f"generation-{generation_index:03d}:{lane_id}:train"
        elif operation_suffix == "eval-public-dev":
            expected_kind = "eval_phase"
            success_class = f"lane_{lane_suffix}_public_dev"
            success_slot = f"generation-{generation_index:03d}:{lane_id}:public_dev"
        elif operation_suffix == "eval-public-accept":
            expected_kind = "eval_phase"
            if "selected_public_accept" in reserves:
                success_class = "selected_public_accept"
                success_slot = (
                    f"generation-{generation_index:03d}:selected:public_accept"
                )
            else:
                success_class = f"lane_{lane_suffix}_public_accept"
                success_slot = (
                    f"generation-{generation_index:03d}:{lane_id}:public_accept"
                )
        else:
            expected_kind = "eval_phase"
            success_class = "sealed_validation"
            success_slot = f"generation-{generation_index:03d}:sealed_validation"
            retry_class = "sealed_validation_retry"
            allowed_retries = frozenset({0, 1})
        if job_kind != expected_kind:
            raise ValueError("bounded generation attempt job kind differs from its operation")
    if scope.max_infrastructure_retries not in allowed_retries:
        raise ValueError("bounded generation attempt retry authority differs from its operation")
    if attempt_index == 0:
        return success_slot, reserves[success_class]
    if (
        attempt_index == 1
        and retry_class is not None
        and scope.max_infrastructure_retries == 1
    ):
        return retry_class, reserves[retry_class]
    raise ValueError("bounded generation attempt index has no frozen operation allocation")



def read_operation_allocation_state(
    *,
    output_dir: Path,
    session_id: str,
    cost_contract: GenerationCostContract,
    session_contract_hash: str,
) -> OperationAllocationState:
    attempt_root = output_dir / "job_attempts"
    success_slots: set[str] = set()
    retry_classes: set[str] = set()
    results: dict[str, Mapping[str, Any] | None] = {}
    measured_ns = 0
    conservative_ns = 0
    unmeasured = 0
    open_output_reserve = 0
    if not attempt_root.exists():
        return OperationAllocationState(
            reserved_success_slots=frozenset(),
            reserved_retry_classes=frozenset(),
            attempt_results={},
            measured_gpu_capacity_ns=0,
            conservatively_charged_gpu_capacity_ns=0,
            unmeasured_bound_attempts=0,
            open_attempt_output_reserve_bytes=0,
        )
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("bounded generation job-attempt root is invalid")
    for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if (
            scope.budget_contract_hash != cost_contract.contract_hash
            or scope.session_contract_hash != session_contract_hash
        ):
            raise ValueError("bounded generation attempt belongs to another budget authority")
        attempt_index = reservation.payload["attempt_index"]
        slot, reserve = read_operation_slot(
            session_id=session_id,
            scope=scope,
            logical_job_id=journal.logical_job_id,
            job_kind=journal.job_kind,
            attempt_index=attempt_index,
            cost_contract=cost_contract,
        )
        if reserve.budget_class == "successful_session":
            if slot in success_slots:
                raise ValueError("bounded generation successful operation slot is duplicated")
            success_slots.add(slot)
        else:
            if slot in retry_classes:
                raise ValueError("bounded generation session-wide retry allocation is duplicated")
            retry_classes.add(slot)
        result_path = journal.root / f"attempt_{attempt_index:02d}.result.json"
        result_payload: Mapping[str, Any] | None = None
        if result_path.exists():
            result_payload = journal.load_result(attempt_index).payload
        else:
            open_output_reserve += reserve.output_bytes_per_attempt
        results[slot] = result_payload
        if (
            result_payload is not None
            and result_payload.get("schema_version")
            == RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
        ):
            measured_ns += result_payload["resource_lease_wall_duration_ns"]
        else:
            unmeasured += 1
            conservative_ns += reserve.allocated_gpu_capacity_ns_per_attempt
    return OperationAllocationState(
        reserved_success_slots=frozenset(success_slots),
        reserved_retry_classes=frozenset(retry_classes),
        attempt_results=results,
        measured_gpu_capacity_ns=measured_ns,
        conservatively_charged_gpu_capacity_ns=conservative_ns,
        unmeasured_bound_attempts=unmeasured,
        open_attempt_output_reserve_bytes=open_output_reserve,
    )



def _strict_success_operation_slots(
    cost_contract: GenerationCostContract,
) -> dict[str, OperationCostReserve]:
    reserves = _strict_operation_reserves_by_class(cost_contract)
    slots: dict[str, OperationCostReserve] = {}
    reuses_reference_public_dev = (
        cost_contract.scientific_cost_basis.reuses_reference_public_dev_receipt
    )
    if not cost_contract.imports_reference_evidence:
        slots["reference_train"] = reserves["reference_train"]
        if reuses_reference_public_dev:
            slots.update(
                {
                    f"reference_qualification:{phase.value}": reserve
                    for phase, reserve in _reference_qualification_phase_reserves(
                        cost_contract
                    ).items()
                }
            )
        else:
            slots["reference_qualification"] = reserves[
                "reference_qualification"
            ]
            slots["parent_public_dev"] = reserves["parent_public_dev"]
    max_generations = cost_contract.max_generations
    selected_public_accept = "selected_public_accept" in reserves
    for generation_index in range(1, max_generations + 1):
        for lane_suffix in ("a", "b"):
            lane_id = f"lane-{lane_suffix}"
            for suffix, operation_class in (
                ("train", f"lane_{lane_suffix}_train"),
                ("public_dev", f"lane_{lane_suffix}_public_dev"),
            ):
                slots[f"generation-{generation_index:03d}:{lane_id}:{suffix}"] = (
                    reserves[operation_class]
                )
            if not selected_public_accept:
                slots[
                    f"generation-{generation_index:03d}:{lane_id}:public_accept"
                ] = reserves[f"lane_{lane_suffix}_public_accept"]
        if selected_public_accept:
            slots[
                f"generation-{generation_index:03d}:selected:public_accept"
            ] = reserves["selected_public_accept"]
        slots[f"generation-{generation_index:03d}:sealed_validation"] = reserves[
            "sealed_validation"
        ]
    if (cost_contract.final_confirmation_seed_units > 0):
        slots["final-confirmation:genesis"] = reserves["sealed_validation"]
        slots["final-confirmation:final"] = reserves["sealed_validation"]
    expected_counts: dict[str, int] = {}
    phased_reference_classes: set[str] = set()
    for reserve in slots.values():
        if reserve.operation_class.startswith("reference_qualification_"):
            phased_reference_classes.add(reserve.operation_class)
            continue
        expected_counts[reserve.operation_class] = (
            expected_counts.get(reserve.operation_class, 0) + 1
        )
    if phased_reference_classes:
        if len(phased_reference_classes) != len(EvaluationPhase):
            raise ValueError("phased reference operation slot inventory is incomplete")
        expected_counts["reference_qualification"] = 1
    operation_reserves = cost_contract.to_dict()["operation_reserves"]["entries"]
    frozen_counts = {
        str(entry["operation_class"]): int(entry["multiplicity"])
        for entry in operation_reserves
        if entry["budget_class"] == "successful_session"
    }
    if expected_counts != frozen_counts:
        raise ValueError("bounded generation operation slot inventory differs from its cost contract")
    return slots



def _strict_retry_is_still_possible(
    *,
    operation_class: str,
    closed_generation_count: int,
    state: OperationAllocationState,
    cost_contract: GenerationCostContract,
) -> bool:
    if operation_class == "reference_train_retry":
        if operation_class in state.reserved_retry_classes:
            return False
        result = state.attempt_results.get("reference_train")
        return (
            "reference_train" not in state.reserved_success_slots
            or result is None
            or (
                result.get("outcome") == "infrastructure_failure"
                and result.get("retryable") is True
            )
        )
    if operation_class != "sealed_validation_retry":
        raise ValueError("bounded generation retry operation class is unknown")
    if operation_class in state.reserved_retry_classes:
        return False
    slots = _strict_success_operation_slots(cost_contract)
    future_slot_exists = any(
        slot.startswith("generation-")
        and slot.endswith(":sealed_validation")
        and slot not in state.reserved_success_slots
        and int(slot.split(":", 1)[0].split("-")[1]) > closed_generation_count
        for slot in slots
    )
    retryable_active_slot_exists = any(
        slot in slots
        and slots[slot].operation_class == "sealed_validation"
        and (
            result is None
            or (
                result.get("outcome") == "infrastructure_failure"
                and result.get("retryable") is True
            )
        )
        for slot, result in state.attempt_results.items()
    )
    final_slot_exists = any(
        slot.startswith("final-confirmation:")
        and slot not in state.reserved_success_slots
        for slot in slots
    )
    return (
        future_slot_exists
        or retryable_active_slot_exists
        or final_slot_exists
    )



def _strict_remaining_operation_reserves(
    *,
    closed_generation_count: int,
    state: OperationAllocationState,
    cost_contract: GenerationCostContract,
) -> tuple[OperationCostReserve, ...]:
    slots = _strict_success_operation_slots(cost_contract)
    remaining: list[OperationCostReserve] = []
    for slot, reserve in slots.items():
        if slot in state.reserved_success_slots:
            continue
        generation_match = re.match(r"generation-([0-9]{3}):", slot)
        if (
            generation_match is not None
            and int(generation_match.group(1)) <= closed_generation_count
        ):
            continue
        remaining.append(reserve)
    reserves = _strict_operation_reserves_by_class(cost_contract)
    retry_classes = tuple(
        operation_class
        for operation_class, reserve in reserves.items()
        if reserve.budget_class == "retry_reserve"
    )
    for retry_class in retry_classes:
        if _strict_retry_is_still_possible(
            operation_class=retry_class,
            closed_generation_count=closed_generation_count,
            state=state,
            cost_contract=cost_contract,
        ):
            remaining.append(reserves[retry_class])
    return tuple(remaining)



def validate_operation_reservation(
    *,
    output_dir: Path,
    session_id: str,
    cost_contract: GenerationCostContract,
    session_contract_hash: str,
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, bool], OperationAllocationState]:
    state = read_operation_allocation_state(
        output_dir=output_dir,
        session_id=session_id,
        cost_contract=cost_contract,
        session_contract_hash=session_contract_hash,
    )
    remaining = _strict_remaining_operation_reserves(
        closed_generation_count=len(ledger),
        state=state,
        cost_contract=cost_contract,
    )
    output_bytes, _ = output_tree_size(output_dir)
    checks = {
        "allocated_active_gpu_hours": (
            state.measured_gpu_capacity_ns
            + state.conservatively_charged_gpu_capacity_ns
            + sum(item.allocated_gpu_capacity_ns_per_attempt for item in remaining)
            <= cost_contract.max_allocated_gpu_capacity_ns
        ),
        "output_bytes": (
            output_bytes
            + state.open_attempt_output_reserve_bytes
            + cost_contract.finalization_output_reserve_bytes
            + sum(item.output_bytes_per_attempt for item in remaining)
            <= cost_contract.max_output_bytes
        ),
    }
    return checks, state



@dataclass(frozen=True)
class OperationAdmission:
    output_dir: Path
    session_id: str
    cost_contract: GenerationCostContract
    session_contract_hash: str
    termination_grace_s: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.termination_grace_s, bool)
            or not isinstance(self.termination_grace_s, int)
            or self.termination_grace_s <= 0
        ):
            raise ValueError("bounded generation operation termination grace must be positive")
        if self.cost_contract.imports_reference_evidence:
            timeout_envelope = (
                self.cost_contract.imported_reference_timeout_envelope
            )
            if timeout_envelope is None:
                raise ValueError("imported-reference timeout envelope is missing")
            _, cleanup_margin_s = timeout_envelope
            if cleanup_margin_s != 3 * self.termination_grace_s + 1:
                raise ValueError(
                    "imported-reference cleanup margin differs from the runner"
                )

    def __call__(
        self,
        scope: JobAttemptScope | PreGenesisExecutionScope,
        logical_job_id: str,
        job_kind: str,
        attempt_index: int,
    ) -> int | None:
        if (
            scope.budget_contract_hash != self.cost_contract.contract_hash
            or scope.session_contract_hash != self.session_contract_hash
        ):
            raise ValueError("bounded generation attempt admission authority is misbound")
        ledger_path = self.output_dir / "ledger.jsonl"
        ledger = load_ledger(ledger_path) if ledger_path.exists() else []
        slot, reserve = read_operation_slot(
            session_id=self.session_id,
            scope=scope,
            logical_job_id=logical_job_id,
            job_kind=job_kind,
            attempt_index=attempt_index,
            cost_contract=self.cost_contract,
        )
        checks, state = validate_operation_reservation(
            output_dir=self.output_dir,
            session_id=self.session_id,
            cost_contract=self.cost_contract,
            session_contract_hash=self.session_contract_hash,
            ledger=ledger,
        )
        if reserve.budget_class == "successful_session":
            if slot in state.reserved_success_slots:
                raise ValueError("bounded generation operation slot was already reserved")
            generation_match = re.match(r"generation-([0-9]{3}):", slot)
            if generation_match is not None and int(generation_match.group(1)) <= len(
                ledger
            ):
                raise ValueError("bounded generation closed generation cannot start new work")
        elif not _strict_retry_is_still_possible(
            operation_class=reserve.operation_class,
            closed_generation_count=len(ledger),
            state=state,
            cost_contract=self.cost_contract,
        ):
            raise ValueError("bounded generation session-wide retry reserve is unavailable")
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise RuntimeError(
                "bounded generation operation admission cannot reserve frozen work: "
                + ", ".join(failed)
            )
        # Timeout cleanup can spend one grace window on TERM, one after KILL,
        # and one proving process-group quiescence.  Keep those windows plus
        # one second inside this operation's frozen one-GPU wall-time slot.
        cleanup_margin_s = 3 * max(self.termination_grace_s, 1) + 1
        if self.cost_contract.imports_reference_evidence and job_kind == "train":
            return None
        slot_wall_time_s = (
            reserve.allocated_gpu_capacity_ns_per_attempt // 1_000_000_000
        )
        timeout_s = slot_wall_time_s - cleanup_margin_s
        if timeout_s <= 0:
            raise RuntimeError("bounded generation operation slot cannot fund process cleanup")
        return timeout_s



def output_tree_size(root: Path) -> tuple[int, int]:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("generation output root is invalid")
    total_bytes = 0
    file_count = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("generation output tree contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total_bytes += entry.stat(follow_symlinks=False).st_size
                    file_count += 1
                else:
                    raise ValueError("generation output tree contains a special file")
    return total_bytes, file_count

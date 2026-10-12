"""Recompute cumulative work from provider, trial and process receipts."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_cost_journal import (GenerationActualCostUsage, LegacyGenerationActualCostUsage,
    GenerationActualCostJournal, GenerationActualCostReceiptRef, assert_actual_cost_receipt_matches_usage)
from true_rl_agent.agent_core.generation_operations import output_tree_size, read_operation_allocation_state, validate_operation_reservation
from true_rl_agent.agent_core.generation_job_evidence import validated_job_cost_artifacts, eval_seed_units_for_job
from true_rl_agent.agent_core.generation_provider_budget import collect_provider_actuals
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope, RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
from true_rl_agent.agent_core.generation_contract import GENERATION_LANE_IDS
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal

def collect_actual_cost_usage(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis | None,
    ledger: list[dict[str, Any]],
) -> tuple[LegacyGenerationActualCostUsage, tuple[str, ...], tuple[Path, ...]]:
    """Recompute actual cost usage from authoritative session artifacts without mutation."""

    provider = collect_provider_actuals(foundation, ledger=ledger)
    scientific_cost = foundation.cost_contract.scientific_cost_basis
    if genesis is None:
        train_steps_by_job: dict[str, int] = {}
        eval_cost_by_job: dict[str, tuple[int, int]] = {}
        job_cost_evidence_hash = canonical_json_hash([])
    else:
        (
            train_steps_by_job,
            eval_cost_by_job,
            job_cost_evidence_hash,
        ) = validated_job_cost_artifacts(foundation, genesis=genesis)
    attempt_root = foundation.config.output_dir / "job_attempts"
    job_attempts = 0
    train_jobs = 0
    train_steps = 0
    eval_seed_units = 0
    eval_control_steps = 0
    lease_wall_ns = 0
    unmeasured_bound_attempts = 0
    conservative: list[str] = list(provider["conservative_charges"])
    open_reservations: list[Path] = []
    used_train_costs: set[str] = set()
    used_eval_costs: set[str] = set()
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            scope = journal.scope
            if (
                scope.budget_contract_hash != foundation.cost_contract.contract_hash
                or scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another cost contract")
            result_path = path.with_name(path.name.replace(".reservation.", ".result."))
            if not result_path.exists():
                open_reservations.append(path)
                job_attempts += 1
                if journal.resource_binding_hash is not None:
                    unmeasured_bound_attempts += 1
                    conservative.append(
                        f"open-bound-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                if journal.job_kind == "train":
                    train_jobs += 1
                    train_steps += scientific_cost.train_steps_per_job
                    conservative.append(
                        f"full-train-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                else:
                    seed_units = eval_seed_units_for_job(
                        journal.logical_job_id,
                        journal.job_kind,
                        cost_contract=foundation.cost_contract,
                    )
                    eval_seed_units += seed_units
                    eval_control_steps += (
                        seed_units
                        * scientific_cost.max_eval_control_steps_per_seed_unit
                    )
                    conservative.append(
                        f"full-eval-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                continue
            result = journal.load_result(reservation.payload["attempt_index"])
            payload = result.payload
            job_attempts += 1
            if journal.resource_binding_hash is not None:
                if (
                    payload.get("schema_version")
                    == RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
                ):
                    lease_wall_ns += payload["resource_lease_wall_duration_ns"]
                else:
                    unmeasured_bound_attempts += 1
                    conservative.append(
                        f"unmeasured-bound-attempt:{journal.logical_job_id}:"
                        f"{payload['attempt_index']}"
                    )
            if journal.job_kind == "train":
                train_jobs += 1
                observed_steps = train_steps_by_job.get(journal.logical_job_id)
                if payload.get("outcome") == "completed" and observed_steps is not None:
                    if journal.logical_job_id in used_train_costs:
                        raise ValueError("train cost artifact was consumed twice")
                    used_train_costs.add(journal.logical_job_id)
                    train_steps += observed_steps
                else:
                    train_steps += scientific_cost.train_steps_per_job
                    conservative.append(
                        f"full-train-attempt:{journal.logical_job_id}:"
                        f"{payload['attempt_index']}"
                    )
                continue
            if journal.job_kind not in {"eval", "eval_phase", "baseline_eval"}:
                raise ValueError("unknown job-attempt kind")
            seed_units = eval_seed_units_for_job(
                journal.logical_job_id,
                journal.job_kind,
                cost_contract=foundation.cost_contract,
            )
            eval_seed_units += seed_units
            observed_cost = eval_cost_by_job.get(journal.logical_job_id)
            if payload.get("outcome") == "completed" and observed_cost is not None:
                if journal.logical_job_id in used_eval_costs:
                    raise ValueError("eval cost artifact was consumed twice")
                observed_seed_units, observed_steps = observed_cost
                if observed_seed_units != seed_units:
                    raise ValueError("eval cost seed count differs from attempt plan")
                used_eval_costs.add(journal.logical_job_id)
                eval_control_steps += observed_steps
            else:
                eval_control_steps += (
                    seed_units * scientific_cost.max_eval_control_steps_per_seed_unit
                )
                conservative.append(
                    f"full-eval-attempt:{journal.logical_job_id}:"
                    f"{payload['attempt_index']}"
                )
    unused_train = set(train_steps_by_job) - used_train_costs
    unused_eval = set(eval_cost_by_job) - used_eval_costs
    if unused_train or unused_eval:
        raise ValueError("cost artifacts are not bound to completed job attempts")
    output_bytes, output_files = output_tree_size(foundation.config.output_dir)
    conservative_gpu_ns = 0
    usage_type: type[LegacyGenerationActualCostUsage] = LegacyGenerationActualCostUsage
    if foundation.cost_contract.uses_operation_reserves:
        allocation_state = read_operation_allocation_state(
            output_dir=foundation.config.output_dir,
            session_id=foundation.config.session_id,
            cost_contract=foundation.cost_contract,
            session_contract_hash=foundation.session_contract.contract_hash,
        )
        if (
            allocation_state.measured_gpu_capacity_ns != lease_wall_ns
            or allocation_state.unmeasured_bound_attempts != unmeasured_bound_attempts
        ):
            raise ValueError("bounded generation operation allocation accounting drifted")
        conservative_gpu_ns = allocation_state.conservatively_charged_gpu_capacity_ns
        usage_type = GenerationActualCostUsage
    usage_kwargs = {
        "provider_attempts": provider["attempts"],
        "provider_request_units": provider["request_units"],
        "provider_response_units": provider["response_units"],
        "provider_total_units": provider["total_units"],
        "provider_reasoning_units": provider["reasoning_units"],
        "provider_conservative_attempts": provider["conservative_attempts"],
        "job_attempts": job_attempts,
        "train_jobs": train_jobs,
        "train_steps": train_steps,
        "eval_seed_units": eval_seed_units,
        "simulated_control_steps": train_steps + eval_control_steps,
        "resource_lease_wall_duration_ns": lease_wall_ns,
        "unmeasured_bound_attempts": unmeasured_bound_attempts,
        "observed_output_bytes": output_bytes,
        "observed_output_files": output_files,
        "job_cost_evidence_hash": job_cost_evidence_hash,
    }
    if usage_type is GenerationActualCostUsage:
        usage_kwargs["conservatively_charged_gpu_capacity_ns"] = conservative_gpu_ns
    usage = usage_type(**usage_kwargs)
    return usage, tuple(sorted(set(conservative))), tuple(open_reservations)



def assert_generation_admission(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    generation_index: int,
) -> None:
    """Reserve remaining worst-case work before a bounded generation starts."""

    if generation_index != len(ledger) + 1:
        raise ValueError("bounded generation admission generation index is not contiguous")
    generation_id = f"{foundation.config.session_id}-generation-{generation_index:03d}"
    usage, _, _ = collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if not foundation.cost_contract.uses_operation_reserves:
        raise TypeError("bounded generation admission requires the v2 physical-cost contract")
    operation_checks, _ = validate_operation_reservation(
        output_dir=foundation.config.output_dir,
        session_id=foundation.config.session_id,
        cost_contract=foundation.cost_contract,
        session_contract_hash=foundation.session_contract.contract_hash,
        ledger=ledger,
    )
    completed_train_jobs, completed_eval_seed_units = (
        _completed_current_generation_operation_cost(
            foundation,
            generation_id=generation_id,
        )
    )
    scientific = foundation.cost_contract.scientific_cost_basis
    remaining_train_jobs = scientific.train_jobs_per_generation - completed_train_jobs
    remaining_eval_seed_units = (
        scientific.eval_seed_units_per_generation - completed_eval_seed_units
    )
    if remaining_train_jobs < 0 or remaining_eval_seed_units < 0:
        raise ValueError("bounded generation partial generation exceeds its scientific plan")
    provider_attempts_in_generation = _generation_provider_attempt_count(
        foundation.config.output_dir,
        generation_id=generation_id,
    )
    provider_plan = foundation.cost_contract.to_dict()["provider"]
    provider_attempts_per_generation = (
        len(GENERATION_LANE_IDS) * provider_plan["attempts_per_lane_per_generation"]
    )
    if provider_attempts_in_generation > provider_attempts_per_generation:
        raise ValueError("bounded generation partial generation exceeds its provider plan")
    remaining_provider_attempts = (
        provider_attempts_per_generation - provider_attempts_in_generation
    )
    remaining_train_steps = remaining_train_jobs * scientific.train_steps_per_job
    remaining_control_steps = remaining_train_steps + (
        remaining_eval_seed_units * scientific.max_eval_control_steps_per_seed_unit
    )
    checks = {
        "provider_attempts": (
            usage.provider_attempts + remaining_provider_attempts
            <= provider_plan["attempt_cap"]
        ),
        "provider_request_units": (
            usage.provider_request_units
            + remaining_provider_attempts
            * provider_plan["input_units_per_attempt_reserve"]
            <= provider_plan["input_unit_cap"]
        ),
        "provider_response_units": (
            usage.provider_response_units
            + remaining_provider_attempts
            * provider_plan["output_units_per_attempt_cap"]
            <= provider_plan["output_unit_cap"]
        ),
        "provider_total_units": (
            usage.provider_total_units
            + remaining_provider_attempts
            * (
                provider_plan["input_units_per_attempt_reserve"]
                + provider_plan["output_units_per_attempt_cap"]
            )
            <= provider_plan["total_unit_cap"]
        ),
        "train_jobs": (
            usage.train_jobs + remaining_train_jobs
            <= foundation.cost_contract.physical_train_job_cap
        ),
        "train_steps": (
            usage.train_steps + remaining_train_steps
            <= foundation.cost_contract.physical_train_step_cap
        ),
        "eval_seed_units": (
            usage.eval_seed_units + remaining_eval_seed_units
            <= foundation.cost_contract.physical_eval_seed_unit_cap
        ),
        "simulated_control_steps": (
            usage.simulated_control_steps + remaining_control_steps
            <= foundation.cost_contract.max_simulated_control_steps
        ),
        "allocated_active_gpu_hours": operation_checks["allocated_active_gpu_hours"],
        "output_bytes": operation_checks["output_bytes"],
    }
    remaining_sealed_retry_entitlement(foundation)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "bounded generation admission cannot reserve frozen work: "
            + ", ".join(failed)
        )



def _completed_current_generation_operation_cost(
    foundation: GenerationSessionFoundation,
    *,
    generation_id: str,
) -> tuple[int, int]:
    attempt_root = foundation.config.output_dir / "job_attempts"
    completed_train_jobs: set[str] = set()
    completed_eval_seed_units: dict[str, int] = {}
    if not attempt_root.exists():
        return 0, 0
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("job-attempt root is invalid")
    for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != generation_id
        ):
            continue
        if (
            scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or scope.session_contract_hash != foundation.session_contract.contract_hash
        ):
            raise ValueError("partial generation attempt belongs to another contract")
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.is_file():
            continue
        result = journal.load_result(reservation.payload["attempt_index"])
        if result.payload.get("outcome") != "completed":
            continue
        if journal.job_kind == "train":
            if journal.logical_job_id in completed_train_jobs:
                raise ValueError("partial generation train completed more than once")
            completed_train_jobs.add(journal.logical_job_id)
        elif journal.job_kind in {"eval", "eval_phase", "baseline_eval"}:
            if journal.logical_job_id in completed_eval_seed_units:
                raise ValueError("partial generation eval completed more than once")
            completed_eval_seed_units[journal.logical_job_id] = (
                eval_seed_units_for_job(
                    journal.logical_job_id,
                    journal.job_kind,
                    cost_contract=foundation.cost_contract,
                )
            )
        else:
            raise ValueError("unknown job-attempt kind")
    return len(completed_train_jobs), sum(completed_eval_seed_units.values())



def _generation_provider_attempt_count(
    output_dir: Path,
    *,
    generation_id: str,
) -> int:
    count = 0
    for lane_id in GENERATION_LANE_IDS:
        round_dir = output_dir / "authoring" / generation_id / lane_id
        if not round_dir.exists():
            continue
        if round_dir.is_symlink() or not round_dir.is_dir():
            raise ValueError("provider authoring directory is invalid")
        count += len(
            RoundAuthoringJournal(
                round_dir,
                round_id=f"{generation_id}-{lane_id}",
            )
            .snapshot()
            .attempts
        )
    return count



def remaining_sealed_retry_entitlement(
    foundation: GenerationSessionFoundation,
) -> int:
    """Return the one session-wide sealed retry only while it is unconsumed."""

    train_retry_jobs, eval_retry_seed_units = session_retry_usage(foundation)
    if train_retry_jobs > foundation.cost_contract.retry_train_jobs:
        raise ValueError("session-wide train retry reserve is exhausted")
    if eval_retry_seed_units > foundation.cost_contract.retry_eval_seed_units:
        raise ValueError("session-wide eval retry reserve is exhausted")
    if eval_retry_seed_units not in {
        0,
        foundation.cost_contract.retry_eval_seed_units,
    }:
        raise ValueError("sealed retry reserve was consumed fractionally")
    return (
        foundation.cost_contract.max_infrastructure_retries_per_eligible_job
        if eval_retry_seed_units == 0 and foundation.cost_contract.retry_eval_seed_units > 0
        else 0
    )



def session_retry_usage(
    foundation: GenerationSessionFoundation,
) -> tuple[int, int]:
    train_retry_jobs = 0
    eval_retry_seed_units = 0
    attempt_root = foundation.config.output_dir / "job_attempts"
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            scope = journal.scope
            if (
                scope.budget_contract_hash != foundation.cost_contract.contract_hash
                or scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another retry contract")
            attempt_index = reservation.payload["attempt_index"]
            if attempt_index == 0:
                continue
            if attempt_index != 1:
                raise ValueError("logical job exceeded its retry entitlement")
            if journal.job_kind == "train":
                train_retry_jobs += 1
            elif journal.job_kind in {"eval", "eval_phase", "baseline_eval"}:
                eval_retry_seed_units += eval_seed_units_for_job(
                    journal.logical_job_id,
                    journal.job_kind,
                    cost_contract=foundation.cost_contract,
                )
            else:
                raise ValueError("unknown job-attempt kind")
    return train_retry_jobs, eval_retry_seed_units


def settle_generation_cost(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: GenerationActualCostJournal,
    event_id: str,
    previous_cost_chain_head_hash: str,
) -> tuple[GenerationActualCostReceiptRef, tuple[Path, ...]]:
    usage, conservative_charges, open_reservations = collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    existing = cost_journal.receipt_for_event(event_id)
    if existing is not None:
        assert_actual_cost_receipt_matches_usage(
            existing,
            usage=usage,
            conservative_charges=conservative_charges,
            context="generation settlement",
        )
        return existing, open_reservations
    receipt = cost_journal.publish_snapshot(
        event_id=event_id,
        previous_cost_chain_head_hash=previous_cost_chain_head_hash,
        usage=usage,
        conservative_charges=conservative_charges,
    )
    return receipt, open_reservations


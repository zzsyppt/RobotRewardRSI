"""One formal execution host for every resolved generation-capable cell.

This layer owns starting, resuming, closing and replaying a session. Task code
owns the reference qualification, exploration rules and final score projection.
"""

from __future__ import annotations

import time
from collections import Counter

from true_rl_agent.agent_core.generation_accounting import collect_actual_cost_usage, settle_generation_cost
from true_rl_agent.agent_core.generation_cost_journal import (
    GenerationActualCostJournal, replay_generation_final_cost_chain, assert_actual_cost_receipt_matches_usage,
)
from true_rl_agent.agent_core.generation_finalization import (
    FINAL_INFORMATION_FLOW, bind_final_evaluation_job, select_final_evaluation_targets,
    run_final_evaluation_phase,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    audit_generation_lifecycle, build_generation_terminal_binding,
)
from true_rl_agent.agent_core.generation_resources import (
    assert_target_gpus_compute_idle, build_resource_closure_evidence,
    validate_stored_resource_closure_evidence,
)
from true_rl_agent.agent_core.generation_runtime import (
    generation_lifecycle_identity, generation_pause_result, generation_public_history,
    prepare_generation_genesis, prepare_generation_inputs, prevalidate_generation_terminal,
)
from true_rl_agent.agent_core.generation_session import (
    run_bounded_generation_session,
)
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.runner.checkpoint import file_sha256

REPORT_FILENAME = "generation_session_report.json"


def _publish_or_replay(path, value, *, read_only=False):
    if path.exists():
        stored = load_strict_json_object(path, require_canonical=True, trailing_newline=True)
        if stored != value:
            raise ValueError(f"stored {path.name} differs from authoritative replay")
    elif read_only:
        raise FileNotFoundError(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, canonical_json_bytes(value, trailing_newline=True))


def _cost_ref(receipt):
    return {"path": str(receipt.path), "payload_hash": receipt.payload_hash}


def _terminal_binding(foundation, *, genesis):
    return build_generation_terminal_binding(
        ledger_path=foundation.config.output_dir / "ledger.jsonl",
        output_dir=foundation.config.output_dir,
        attempt_journal_root=foundation.config.output_dir / "job_attempts",
        identity=generation_lifecycle_identity(foundation, genesis=genesis),
        policy=foundation.continuation_policy, session_contract=foundation.session_contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=foundation.composition.task_plugin.task_spec.acceptance_protocol,
        expected_initial_policy_state_hash=genesis.expected_initial_policy_state_hash)


def finalize_generation_session(foundation, *, genesis, ledger, cost_journal,
                                startup_resource_observation=None):
    config = foundation.config
    read_only = config.terminal_audit_only
    started = time.perf_counter_ns()
    # Full terminal replay belongs here, after the generation segment closes.
    # Pre-close cost/resource checks in the loop do not rebuild this proof.
    terminal = _terminal_binding(foundation, genesis=genesis)
    terminal_ns = time.perf_counter_ns() - started
    targets = select_final_evaluation_targets(genesis=genesis, ledger=ledger, terminal=terminal)
    plan = foundation.final_confirmation_plan
    eval_plans = {role: bind_final_evaluation_job(foundation, target=target, role=role)
                  for role, target in (("genesis", targets.reference), ("final", targets.terminal))}
    phase_specs = {role: foundation.composition.backend_adapter.materialize_eval_phase(
        eval_plan=job, phase=plan.phase) for role, job in eval_plans.items()}
    ledger_checksum = file_sha256(config.output_dir / "ledger.jsonl")
    reservation = {"schema_version": "true_rl_agent.generation_final_reservation.v1",
        "session_id": config.session_id, "plan_hash": plan.plan_hash,
        "session_genesis_hash": genesis.session_genesis_hash,
        "session_contract_hash": foundation.session_contract.contract_hash,
        "cost_contract_hash": foundation.cost_contract.contract_hash,
        "terminal_binding": terminal.to_dict(), "ledger_checksum": ledger_checksum,
        "source_commit": foundation.source_commit,
        "backend_execution_environment_manifest_hash": genesis.backend_execution_environment_manifest_hash,
        "jobs": {role: {"job_id": spec.job_id,
            "eval_phase_spec_hash": canonical_json_hash(spec.to_public_dict()),
            "eval_plan_hash": canonical_json_hash(eval_plans[role].to_public_dict()),
            "checkpoint_hash": eval_plans[role].execution_checkpoint().checksum}
            for role, spec in phase_specs.items()}}
    reservation_hash = canonical_json_hash(reservation)
    _publish_or_replay(config.output_dir / "final_confirmation" / "reservation.json",
                       reservation, read_only=read_only)
    final_started = time.perf_counter_ns()
    executions = {}
    for role, job in eval_plans.items():
        spec = phase_specs[role]
        if read_only and not (config.output_dir / "final_confirmation" / "phase_receipts" / role).is_dir():
            raise FileNotFoundError("terminal audit is missing final phase receipts")
        executions[role] = run_final_evaluation_phase(foundation, genesis=genesis,
            eval_plan=job, role=role, phase=plan.phase, plan_hash=plan.plan_hash,
            reservation_hash=reservation_hash, expected_job_id=spec.job_id,
            expected_spec_hash=canonical_json_hash(spec.to_public_dict()),
            require_existing=read_only)
    final_ns = time.perf_counter_ns() - final_started
    comparison = foundation.composition.task_plugin.summarize_final_comparison(
        plan=plan, plan_path=config.output_dir / "final_confirmation" / "plan.json",
        reference=executions["genesis"].result, final=executions["final"].result)
    result = {"schema_version": "true_rl_agent.generation_final_result.v1",
        "plan_hash": plan.plan_hash, "reservation_hash": reservation_hash,
        "execution": {role: {"phase_receipt": item.receipt_ref.to_dict(),
                             "attempt_result_payload_hash": item.attempt_result.payload_hash}
                      for role, item in executions.items()},
        "task_comparison": comparison, "information_flow": dict(FINAL_INFORMATION_FLOW)}
    _publish_or_replay(config.output_dir / "final_confirmation" / "result.json", result, read_only=read_only)
    if file_sha256(config.output_dir / "ledger.jsonl") != ledger_checksum:
        raise ValueError("final confirmation changed the generation ledger")
    if read_only and cost_journal.receipt_for_event("final-confirmation-settlement") is None:
        raise FileNotFoundError("terminal audit is missing final cost settlement")
    final_cost, open_reservations = settle_generation_cost(foundation, genesis=genesis,
        ledger=ledger, cost_journal=cost_journal, event_id="final-confirmation-settlement",
        previous_cost_chain_head_hash=terminal.cost_chain_head_hash)
    if open_reservations or not final_cost.payload["cap_checks"]["all_passed"]:
        raise ValueError("final confirmation has unsettled work or exceeded its budget")
    resource_path = config.output_dir / "final_confirmation" / "resource_closure.json"
    if resource_path.exists():
        resource = load_strict_json_object(resource_path, require_canonical=True, trailing_newline=True)
        validate_stored_resource_closure_evidence(foundation, resource_evidence=resource,
            generation_index=len(ledger), include_final_confirmation=True)
    elif read_only:
        raise FileNotFoundError(resource_path)
    else:
        resource = build_resource_closure_evidence(foundation, genesis=genesis,
            invocation_segment_id="formal-finalization", startup_observation=startup_resource_observation,
            generation_index=len(ledger), include_final_confirmation=True)
        _publish_or_replay(resource_path, resource)
    report_started = time.perf_counter_ns()
    report = build_generation_report(foundation, genesis=genesis, ledger=ledger,
        terminal=terminal, cost_journal=cost_journal, final_cost=final_cost,
        result=result, resource=resource)
    report_ns = time.perf_counter_ns() - report_started
    report_path = config.output_dir / REPORT_FILENAME
    _publish_or_replay(report_path, report, read_only=read_only)
    if not read_only:
        timing_path = config.output_dir / "final_confirmation" / "timing.json"
        if not timing_path.exists():
            _publish_or_replay(timing_path, {"schema_version": "true_rl_agent.generation_terminal_timing.v1",
                "clock_source": "time.perf_counter_ns", "terminal_replay_ns": terminal_ns,
                "final_phase_execution_or_reuse_ns": final_ns, "report_projection_ns": report_ns})
    return report


def build_generation_report(foundation, *, genesis, ledger, terminal, cost_journal,
                            final_cost, result, resource):
    # Each independent report replay obtains fresh verified views. Prefix users
    # within this call share those inputs, never a process-global cached verdict.
    history = generation_public_history(foundation, genesis=genesis, ledger=ledger)
    lifecycle = audit_generation_lifecycle(output_dir=foundation.config.output_dir,
        identity=generation_lifecycle_identity(foundation, genesis=genesis),
        policy=foundation.continuation_policy, require_closed=True)
    costs = cost_journal.load_snapshot()
    head = replay_generation_final_cost_chain(contract=foundation.cost_contract,
        actual_cost_receipts=tuple(costs.values()), lifecycle=lifecycle,
        require_finalization_settlement=True)
    usage, charges, opened = collect_actual_cost_usage(foundation, genesis=genesis, ledger=ledger)
    assert_actual_cost_receipt_matches_usage(final_cost, usage=usage,
        conservative_charges=charges, context="terminal report")
    if opened or head != final_cost.payload_hash:
        raise ValueError("terminal report cost chain is not closed")
    expected_head = canonical_json_hash(history.current_best.to_public_dict())
    if expected_head != terminal.current_head_hash:
        raise ValueError("terminal report public replay head differs")
    return {"schema_version": "true_rl_agent.generation_session_report.v1",
        "session_id": foundation.config.session_id, "task_id": foundation.composition.task_id,
        "cell_variant_id": foundation.config.cell_variant_id,
        "session_genesis_hash": genesis.session_genesis_hash,
        "session_contract": foundation.session_contract.to_dict(),
        "terminal_binding": terminal.to_dict(), "current_best": history.current_best.to_public_dict(),
        "generation_outcomes": dict(sorted(Counter(record["generation_outcome"] for record in ledger).items())),
        "closed_generation_count": len(ledger),
        "cost": {"contract": foundation.cost_contract.to_dict(),
                 "final_receipt": _cost_ref(final_cost), "actual": dict(final_cost.payload["actual"])},
        "final_confirmation": result, "resource_closure": resource,
        "checks": {"status": "PASS", "ledger_and_terminal_replayed": True,
            "actual_cost_replayed": True, "final_receipts_replayed": True,
            "resource_closure_replayed": True}, "acceptance_authority": False}


def run_generation_session(foundation):
    config = foundation.config
    journal = GenerationActualCostJournal(config.output_dir, contract=foundation.cost_contract)
    report_path = config.output_dir / REPORT_FILENAME
    if report_path.exists() and not config.terminal_audit_only:
        raise ValueError("completed session requires terminal audit")
    if config.terminal_audit_only:
        if not report_path.exists():
            raise FileNotFoundError("terminal audit requires a completed report")
        genesis = prepare_generation_genesis(foundation, require_existing=True)
        return finalize_generation_session(foundation, genesis=genesis,
            ledger=load_ledger(config.output_dir / "ledger.jsonl"), cost_journal=journal)
    startup = assert_target_gpus_compute_idle(config.nvidia_smi_executable,
        expected=(config.gpu0_uuid, config.gpu1_uuid), stage="startup",
        schema_version=f"{foundation.cost_contract.operational_evidence_namespace}.generation_compute_idle.v1")
    journal.publish_reservation()
    try:
        genesis = prepare_generation_genesis(foundation)
    except BaseException:
        usage, charges, _ = collect_actual_cost_usage(foundation, genesis=None, ledger=[])
        chain = journal.load_chain()
        journal.publish_snapshot(event_id=f"genesis-failed-{len(chain):04d}",
            previous_cost_chain_head_hash=chain[-1].payload_hash,
            usage=usage, conservative_charges=charges, finalization_output_reserve_bytes=0)
        raise
    if journal.receipt_for_event("post-genesis") is None:
        usage, charges, _ = collect_actual_cost_usage(foundation, genesis=genesis, ledger=[])
        journal.publish_snapshot(event_id="post-genesis",
            previous_cost_chain_head_hash=journal.load_chain()[-1].payload_hash,
            usage=usage, conservative_charges=charges, finalization_output_reserve_bytes=0)
    if not journal.receipt_for_event("post-genesis").payload["cap_checks"]["all_passed"]:
        raise ValueError("Genesis exhausted its reserved cost")
    identity = generation_lifecycle_identity(foundation, genesis=genesis)
    lifecycle = audit_generation_lifecycle(output_dir=config.output_dir, identity=identity,
        policy=foundation.continuation_policy, require_closed=False,
        require_recovery_attempt_results=False)
    if lifecycle.terminal:
        return finalize_generation_session(foundation, genesis=genesis,
            ledger=load_ledger(config.output_dir / "ledger.jsonl"), cost_journal=journal,
            startup_resource_observation=startup)
    return run_bounded_generation_session(foundation, genesis=genesis, cost_journal=journal,
        startup_resource_observation=startup, lifecycle_identity=identity,
        prepare_generation=prepare_generation_inputs, publish_pause=generation_pause_result,
        prevalidate_terminal=prevalidate_generation_terminal,
        finalize_terminal=finalize_generation_session)

"""One bounded generation lifecycle for resolved task/robot compositions.

The host owns execution, recovery, cost settlement and generation closure.
Assembly supplies the lane inputs and boundary reporting/final science hooks;
those hooks do not choose another lifecycle or drive the next generation.
"""
from __future__ import annotations
from collections.abc import Callable
from true_rl_agent.agent_core.generation_cost_journal import (
    GenerationActualCostJournal, GenerationActualCostReceiptRef,
    replay_generation_combined_cost_chain, assert_actual_cost_receipt_matches_usage,
)

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_accounting import collect_actual_cost_usage, assert_generation_admission
from true_rl_agent.agent_core.generation_resources import build_resource_closure_evidence
from true_rl_agent.agent_core.generation import execute_fixed_width_generation
from true_rl_agent.agent_core.generation_lifecycle import GenerationLifecycleAudit, GenerationLifecycleIdentity, GenerationSegmentJournal, GenerationTerminalBinding, audit_generation_lifecycle, build_generation_continuation_cursor
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.agent_core.generation_attempt_scopes import (
    recover_parent_bootstrap_attempt_scope,
    generation_job_attempt_scope as _job_attempt_scope,
)
from true_rl_agent.agent_core.generation_closure import load_existing_generation_execution, publish_or_validate_generation_closure
from true_rl_agent.agent_core.generation_accounting import settle_generation_cost
from true_rl_agent.agent_core.generation_guard import replay_generation_guard_chain, run_genesis_parent_public_dev


def run_bounded_generation_session(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: GenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
    lifecycle_identity: GenerationLifecycleIdentity,
    prepare_generation: Callable[..., Any],
    publish_pause: Callable[..., dict[str, Any]],
    prevalidate_terminal: Callable[..., None],
    finalize_terminal: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run the single bounded generation engine shared by P6-C and P6-D."""

    config = foundation.config
    composition = foundation.composition
    contract = foundation.session_contract
    if not foundation.cost_contract.uses_operation_reserves:
        raise ValueError("bounded session requires explicit operation reserves")
    ledger_path = config.output_dir / "ledger.jsonl"
    ledger = load_ledger(ledger_path)
    lifecycle = audit_generation_lifecycle(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
        require_recovery_attempt_results=False,
    )
    combined_cost_head = _replay_validation_combined_cost_chain(
        cost_journal,
        lifecycle,
    )
    requested_closed_boundary = (
        config.pause_after_generation is not None
        and len(ledger) == config.pause_after_generation
    )
    if (
        config.pause_after_generation is not None
        and len(ledger) > config.pause_after_generation
    ):
        raise ValueError("requested generation boundary is already closed")
    if requested_closed_boundary and lifecycle.active_segment_id is None:
        generation_index = len(ledger)
        closure_path = (
            config.output_dir
            / "generation_closures"
            / f"{ledger[-1]['generation_id']}.json"
        )
        if (
            cost_journal.receipt_for_event(
                f"generation-{generation_index:03d}-settlement"
            )
            is not None
            and closure_path.is_file()
            and _validation_boundary_was_already_paused(
                config.output_dir,
                generation_count=generation_index,
            )
        ):
            raise ValueError("requested generation boundary is already closed")
    if lifecycle.terminal:
        if len(ledger) != contract.max_generations:
            raise ValueError("terminal bounded generation lifecycle has the wrong ledger height")
        latest_cost = cost_journal.load_chain()[-1]
        if combined_cost_head != latest_cost.payload_hash:
            raise ValueError("terminal lifecycle actual-cost head mismatch")
        return finalize_terminal(
            foundation,
            genesis=genesis,
            ledger=ledger,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )

    segment = GenerationSegmentJournal.start(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=contract,
        closed_generation_count=len(ledger),
        cost_chain_head_hash=combined_cost_head,
        attempt_journal_root=config.output_dir / "job_attempts",
    )
    started_lifecycle = audit_generation_lifecycle(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
    )
    segment_cost_chain_head = _replay_validation_combined_cost_chain(
        cost_journal,
        started_lifecycle,
    )
    if segment.cost_chain_head_hash != segment_cost_chain_head:
        raise ValueError("bounded generation segment cost head differs from combined replay")
    segment_closed = False
    try:
        bootstrap_scope = None
        if foundation.phased_adapter.parent_public_dev_source == "evaluate":
            bootstrap_scope = _job_attempt_scope(
                foundation,
                genesis=genesis,
                generation_id="generation-parent-public-dev-bootstrap",
                lane_id="lane-a",
                invocation_segment_id=segment.segment_id,
                binding=foundation.gpu_bindings[0],
            )
            bootstrap_scope = recover_parent_bootstrap_attempt_scope(
                attempt_journal_root=config.output_dir / "job_attempts",
                session_id=config.session_id,
                current_invocation_segment_id=segment.segment_id,
                current_scope=bootstrap_scope,
            )
        initial_guard = run_genesis_parent_public_dev(
            genesis=genesis,
            adapter=foundation.phased_adapter,
            backend=composition.backend_adapter,
            runner=foundation.runner,
            receipt_root=config.output_dir / "parent_public_dev_receipts",
            generation_session_contract_hash=contract.contract_hash,
            attempt_scope=bootstrap_scope,
        )
        existing_bootstrap_cost = cost_journal.receipt_for_event(
            "post-parent-bootstrap"
        )
        bootstrap_cost = _publish_or_load_parent_bootstrap_cost_boundary(
            foundation,
            genesis=genesis,
            ledger=ledger,
            cost_journal=cost_journal,
            previous_cost_chain_head_hash=segment_cost_chain_head,
        )
        if existing_bootstrap_cost is None:
            segment_cost_chain_head = bootstrap_cost.payload_hash
        cursor = build_generation_continuation_cursor(
            ledger_path=ledger_path,
            output_dir=config.output_dir,
            identity=lifecycle_identity,
            policy=foundation.continuation_policy,
            session_contract=contract,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
        )
        parent, guard = replay_generation_guard_chain(
            generation_records=ledger,
            initial_parent=genesis.initial_current_best,
            initial_guard_reference=initial_guard,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
            adapter=foundation.phased_adapter,
            expected_generation_session_contract_hash=contract.contract_hash,
        )
        if (
            parent != cursor.current_best
            or cursor.next_generation_index != len(ledger) + 1
        ):
            raise ValueError("bounded generation guard replay differs from continuation cursor")

        closures: list[dict[str, Any]] = []
        latest_resource_evidence: Mapping[str, Any] | None = None
        for index, existing_record in enumerate(ledger, start=1):
            loaded_record, execution_ref = load_existing_generation_execution(
                foundation,
                record=existing_record,
            )
            event_id = f"generation-{index:03d}-settlement"
            cost_ref = cost_journal.receipt_for_event(event_id)
            cost_receipt_existed = cost_ref is not None
            closure_path = (
                config.output_dir
                / "generation_closures"
                / f"{loaded_record['generation_id']}.json"
            )
            if cost_ref is None or not closure_path.is_file():
                if index != len(ledger):
                    raise ValueError(
                        "bounded generation closed generation prefix lacks an automatic closure"
                    )
                latest_resource_evidence = build_resource_closure_evidence(
                    foundation,
                    genesis=genesis,
                    invocation_segment_id=segment.segment_id,
                    startup_observation=startup_resource_observation,
                    generation_index=index,
                )
                cost_ref, open_reservations = settle_generation_cost(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    cost_journal=cost_journal,
                    event_id=event_id,
                    previous_cost_chain_head_hash=segment_cost_chain_head,
                )
                if not cost_receipt_existed:
                    segment_cost_chain_head = cost_ref.payload_hash
                if open_reservations:
                    raise ValueError(
                        "bounded generation recovered generation has open job-attempt reservations"
                    )
            assert cost_ref is not None
            closure = publish_or_validate_generation_closure(
                foundation,
                ledger=ledger,
                generation_record=loaded_record,
                generation_execution_receipt=execution_ref.to_dict(),
                actual_cost_receipt=cost_ref,
                resource_evidence=latest_resource_evidence or {},
                segment_id=segment.segment_id,
            )
            closures.append(closure)

        if requested_closed_boundary:
            if not closures:
                raise ValueError("bounded generation recovered boundary has no completed closure")
            boundary_cost = _current_validation_boundary_cost_receipt(
                foundation,
                genesis=genesis,
                ledger=ledger,
                cost_journal=cost_journal,
                lifecycle=started_lifecycle,
                segment_id=segment.segment_id,
            )
            segment_cost_chain_head = boundary_cost.payload_hash
            segment.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=len(ledger),
                cost_chain_head_hash=segment_cost_chain_head,
            )
            segment_closed = True
            return publish_pause(
                foundation,
                genesis=genesis,
                ledger=ledger,
                segment_id=segment.segment_id,
                actual_cost_receipt=boundary_cost,
                closure=closures[-1],
            )

        while len(ledger) < contract.max_generations:
            generation_index = len(ledger) + 1
            assert_generation_admission(
                foundation,
                genesis=genesis,
                ledger=ledger,
                generation_index=generation_index,
            )
            generation_inputs, _ = prepare_generation(
                foundation,
                genesis=genesis,
                guard=guard,
                segment_id=segment.segment_id,
                generation_index=generation_index,
                parent=parent,
            )
            result = execute_fixed_width_generation(generation_inputs)
            next_ledger = load_ledger(ledger_path)
            if (
                next_ledger[: len(ledger)] != ledger
                or len(next_ledger) != generation_index
                or not (result.committed or result.reused_execution)
            ):
                raise ValueError("bounded generation did not append exactly one record")
            ledger = next_ledger
            generation_record = dict(result.record)
            if generation_record != ledger[-1]:
                raise ValueError("bounded generation execution result differs from its ledger record")
            latest_resource_evidence = build_resource_closure_evidence(
                foundation,
                genesis=genesis,
                invocation_segment_id=segment.segment_id,
                startup_observation=startup_resource_observation,
                generation_index=generation_index,
            )
            actual_cost, open_reservations = settle_generation_cost(
                foundation,
                genesis=genesis,
                ledger=ledger,
                cost_journal=cost_journal,
                event_id=f"generation-{generation_index:03d}-settlement",
                previous_cost_chain_head_hash=segment_cost_chain_head,
            )
            segment_cost_chain_head = actual_cost.payload_hash
            if open_reservations:
                raise ValueError("bounded generation left an open attempt reservation")
            if actual_cost.payload["cap_checks"]["all_passed"] is not True:
                segment.finish(
                    status="failed",
                    reason="budget_exhausted",
                    closed_generation_count=len(ledger),
                    cost_chain_head_hash=actual_cost.payload_hash,
                )
                segment_closed = True
                raise ValueError("bounded generation exhausted its frozen cost contract")
            closures.append(
                publish_or_validate_generation_closure(
                    foundation,
                    ledger=ledger,
                    generation_record=generation_record,
                    generation_execution_receipt=(
                        result.execution_receipt_ref.to_dict()
                    ),
                    actual_cost_receipt=actual_cost,
                    resource_evidence=latest_resource_evidence,
                    segment_id=segment.segment_id,
                )
            )
            cursor = build_generation_continuation_cursor(
                ledger_path=ledger_path,
                output_dir=config.output_dir,
                identity=lifecycle_identity,
                policy=foundation.continuation_policy,
                session_contract=contract,
                initial_current_best=genesis.initial_current_best,
                acceptance_protocol=(
                    composition.task_plugin.task_spec.acceptance_protocol
                ),
                expected_initial_policy_state_hash=(
                    genesis.expected_initial_policy_state_hash
                ),
            )
            parent, guard = replay_generation_guard_chain(
                generation_records=ledger,
                initial_parent=genesis.initial_current_best,
                initial_guard_reference=initial_guard,
                acceptance_protocol=(
                    composition.task_plugin.task_spec.acceptance_protocol
                ),
                adapter=foundation.phased_adapter,
                expected_generation_session_contract_hash=contract.contract_hash,
            )
            if parent != cursor.current_best:
                raise ValueError("bounded generation post-generation head replay diverged")
            if config.pause_after_generation == generation_index:
                boundary_cost = _current_validation_boundary_cost_receipt(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    cost_journal=cost_journal,
                    lifecycle=started_lifecycle,
                    segment_id=segment.segment_id,
                )
                segment_cost_chain_head = boundary_cost.payload_hash
                segment.finish(
                    status="paused",
                    reason="operator_pause",
                    closed_generation_count=len(ledger),
                    cost_chain_head_hash=segment_cost_chain_head,
                )
                segment_closed = True
                return publish_pause(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    segment_id=segment.segment_id,
                    actual_cost_receipt=boundary_cost,
                    closure=closures[-1],
                )

        if latest_resource_evidence is None:
            latest_resource_evidence = build_resource_closure_evidence(
                foundation,
                genesis=genesis,
                invocation_segment_id=segment.segment_id,
                startup_observation=startup_resource_observation,
                generation_index=len(ledger),
            )
        latest_cost = _current_validation_boundary_cost_receipt(
            foundation,
            genesis=genesis,
            ledger=ledger,
            cost_journal=cost_journal,
            lifecycle=started_lifecycle,
            segment_id=segment.segment_id,
        )
        segment_cost_chain_head = latest_cost.payload_hash
        prospective_terminal = GenerationTerminalBinding(
            identity_hash=lifecycle_identity.identity_hash,
            required_generation_count=contract.max_generations,
            closed_generation_count=len(ledger),
            ledger_snapshot_hash=cursor.ledger_snapshot_hash,
            current_head_hash=cursor.current_head_hash,
            lifecycle_head_hash="0" * 64,
            cost_chain_head_hash=segment_cost_chain_head,
            segment_count=segment.continuation_index + 1,
        )
        prevalidate_terminal(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=prospective_terminal.to_dict(),
            actual_cost_receipt=latest_cost,
            resource_evidence=latest_resource_evidence,
        )
        segment.finish(
            status="completed",
            reason="normal",
            closed_generation_count=len(ledger),
            cost_chain_head_hash=segment_cost_chain_head,
        )
        segment_closed = True
    except BaseException as exc:
        if not segment_closed:
            try:
                failed_ledger = load_ledger(ledger_path)
                usage, conservative_charges, open_reservations = (
                    collect_actual_cost_usage(
                        foundation,
                        genesis=genesis,
                        ledger=failed_ledger,
                    )
                )
                failed_cost = cost_journal.publish_snapshot(
                    event_id=(
                        f"segment-{segment.segment_id}-failed-"
                        f"generation-{len(failed_ledger):03d}"
                    ),
                    previous_cost_chain_head_hash=segment_cost_chain_head,
                    usage=usage,
                    conservative_charges=conservative_charges,
                )
                segment.finish(
                    status="failed",
                    reason="exception",
                    closed_generation_count=len(failed_ledger),
                    cost_chain_head_hash=failed_cost.payload_hash,
                    unclosed_attempt_reservations=open_reservations,
                    exception=exc,
                )
            except Exception as lifecycle_error:  # noqa: BLE001
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "generation lifecycle exit evidence failed: "
                        + type(lifecycle_error).__name__
                    )
        raise

    return finalize_terminal(
        foundation,
        genesis=genesis,
        ledger=ledger,
        cost_journal=cost_journal,
        startup_resource_observation=startup_resource_observation,
    )


def _latest_cost_head(journal: GenerationActualCostJournal) -> str:
    chain = journal.load_chain()
    return chain[-1].payload_hash if chain else journal.contract.contract_hash


def _replay_validation_combined_cost_chain(
    cost_journal: GenerationActualCostJournal,
    lifecycle: GenerationLifecycleAudit,
) -> str:
    return replay_generation_combined_cost_chain(
        contract=cost_journal.contract,
        actual_cost_receipts=cost_journal.load_chain(),
        lifecycle=lifecycle,
    )


def _current_validation_boundary_cost_receipt(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: GenerationActualCostJournal,
    lifecycle: GenerationLifecycleAudit,
    segment_id: str,
) -> GenerationActualCostReceiptRef:
    """Return an actual receipt at the current bounded-session cost-chain head.

    Generation closures intentionally retain their generation-settlement receipt.
    A pause or terminal boundary instead binds the current segment head, including
    later failure snapshots.  Lifecycle recovery can advance the combined head
    without creating an actual-cost receipt, so that one case is reconciled by a
    segment-unique, create-once snapshot.
    """

    combined_head = _replay_validation_combined_cost_chain(cost_journal, lifecycle)
    chain = cost_journal.load_chain()
    if not chain:
        raise ValueError("bounded generation boundary has no actual-cost chain")
    latest = chain[-1]
    usage, conservative_charges, open_reservations = collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if open_reservations:
        raise ValueError("bounded generation boundary has open job-attempt reservations")

    if latest.payload_hash == combined_head:
        current = latest
    else:
        current = cost_journal.publish_snapshot(
            event_id=f"segment-{segment_id}-boundary-reconciliation",
            previous_cost_chain_head_hash=combined_head,
            usage=usage,
            conservative_charges=conservative_charges,
        )
        if (
            _replay_validation_combined_cost_chain(cost_journal, lifecycle)
            != current.payload_hash
        ):
            raise ValueError(
                "bounded generation boundary reconciliation did not close the cost chain"
            )

    assert_actual_cost_receipt_matches_usage(
        current,
        usage=usage,
        conservative_charges=conservative_charges,
        context="bounded generation boundary",
    )
    if current.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("bounded generation boundary exhausted its frozen cost contract")
    return current


def _validation_boundary_was_already_paused(
    output_dir: Path,
    *,
    generation_count: int,
) -> bool:
    """Distinguish a delivered pause from a failed post-generation segment."""

    segment_root = output_dir / "generation_segments"
    segment_dirs = sorted(segment_root.glob("segment-*"))
    if not segment_dirs:
        return False
    exit_path = segment_dirs[-1] / "exit.json"
    if not exit_path.is_file():
        return False
    exit_record = load_strict_json_object(
        exit_path,
        context="bounded generation last segment exit",
    )
    return bool(
        exit_record.get("status") == "paused"
        and exit_record.get("reason") == "operator_pause"
        and exit_record.get("generations_at_end") == generation_count
    )


def _publish_or_load_parent_bootstrap_cost_boundary(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: GenerationActualCostJournal,
    previous_cost_chain_head_hash: str,
) -> GenerationActualCostReceiptRef:
    """Bind the one session-level parent bootstrap outside generation deltas."""

    event_id = "post-parent-bootstrap"
    existing = cost_journal.receipt_for_event(event_id)
    if existing is not None:
        if existing.payload["cap_checks"]["all_passed"] is not True:
            raise ValueError("bounded generation parent bootstrap exhausted its frozen cost contract")
        return existing
    if ledger:
        raise ValueError(
            "bounded generation closed generation prefix lacks its parent-bootstrap cost boundary"
        )
    usage, conservative_charges, open_reservations = collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=[],
    )
    if open_reservations:
        raise ValueError("bounded generation parent bootstrap left an open attempt reservation")
    receipt = cost_journal.publish_snapshot(
        event_id=event_id,
        previous_cost_chain_head_hash=previous_cost_chain_head_hash,
        usage=usage,
        conservative_charges=conservative_charges,
        finalization_output_reserve_bytes=0,
    )
    if receipt.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("bounded generation parent bootstrap exhausted its frozen cost contract")
    return receipt

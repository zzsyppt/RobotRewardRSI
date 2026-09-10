from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationSessionContract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GenerationContinuationPolicy,
    GenerationLifecycleIdentity,
    GenerationSegmentJournal,
    audit_generation_lifecycle,
    publish_generation_continuation_policy,
)
from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.agent_core.generation_session import (
    assert_actual_cost_receipt_matches_usage,
    _current_validation_boundary_cost_receipt,
    publish_or_validate_generation_closure,
    _replay_validation_combined_cost_chain,
    _validation_boundary_was_already_paused,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    StairGenerationActualCostJournal,
    StairGenerationActualCostUsage,
    StairGenerationActualCostUsageV2,
    StairGenerationSessionCostContractV2,
    build_stair_generation_scientific_cost_basis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StairGenerationCostRecoveryTests(unittest.TestCase):
    def test_pause_boundary_uses_later_failure_actual_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp).resolve())
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            generation = fixture.cost_journal.publish_snapshot(
                event_id="generation-001-settlement",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            failure = fixture.cost_journal.publish_snapshot(
                event_id=f"segment-{first.segment_id}-failed-generation-001",
                previous_cost_chain_head_hash=generation.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            first.finish(
                status="failed",
                reason="exception",
                closed_generation_count=1,
                cost_chain_head_hash=failure.payload_hash,
                exception=RuntimeError("synthetic post-generation failure"),
            )
            self.assertFalse(
                _validation_boundary_was_already_paused(
                    fixture.root,
                    generation_count=1,
                )
            )
            resumed = fixture.start_segment(
                closed_generation_count=1,
                cost_chain_head_hash=failure.payload_hash,
            )
            lifecycle = fixture.audit(require_closed=False)
            foundation = SimpleNamespace(provider=Mock(), runner=Mock())
            usage = StairGenerationActualCostUsageV2()

            with (
                patch(
                    'true_rl_agent.agent_core.generation_session.collect_actual_cost_usage',
                    return_value=(usage, (), ()),
                ),
                patch.object(
                    fixture.cost_journal,
                    "publish_snapshot",
                    wraps=fixture.cost_journal.publish_snapshot,
                ) as publish,
            ):
                current = _current_validation_boundary_cost_receipt(
                    foundation,
                    genesis=SimpleNamespace(),
                    ledger=[{"generation_index": 1}],
                    cost_journal=fixture.cost_journal,
                    lifecycle=lifecycle,
                    segment_id=resumed.segment_id,
                )

            self.assertEqual(current.payload_hash, failure.payload_hash)
            publish.assert_not_called()
            foundation.provider.assert_not_called()
            foundation.runner.assert_not_called()
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=current.payload_hash,
            )
            self.assertEqual(
                fixture.audit(require_closed=True).cost_chain_head_hash,
                failure.payload_hash,
            )

    def test_recovery_only_head_gets_one_boundary_actual_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp).resolve())
            reservation = fixture.cost_journal.publish_reservation()
            generation_segment = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            generation = fixture.cost_journal.publish_snapshot(
                event_id="generation-001-settlement",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            generation_segment.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=generation.payload_hash,
            )
            self.assertTrue(
                _validation_boundary_was_already_paused(
                    fixture.root,
                    generation_count=1,
                )
            )
            first = fixture.start_segment(
                closed_generation_count=1,
                cost_chain_head_hash=generation.payload_hash,
            )
            attempt = fixture.attempt_journal(
                first,
                logical_suffix="future-generation-crash",
                generation_id="generation-002",
            )
            attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()
            resumed = fixture.start_segment(
                closed_generation_count=1,
                cost_chain_head_hash=generation.payload_hash,
                recover=True,
            )
            lifecycle = fixture.audit(require_closed=False)
            recovery_head = _replay_validation_combined_cost_chain(
                fixture.cost_journal,
                lifecycle,
            )
            self.assertNotEqual(recovery_head, generation.payload_hash)
            foundation = SimpleNamespace(provider=Mock(), runner=Mock())
            usage = StairGenerationActualCostUsageV2()

            with (
                patch(
                    'true_rl_agent.agent_core.generation_session.collect_actual_cost_usage',
                    return_value=(usage, (), ()),
                ),
                patch.object(
                    fixture.cost_journal,
                    "publish_snapshot",
                    wraps=fixture.cost_journal.publish_snapshot,
                ) as publish,
            ):
                current = _current_validation_boundary_cost_receipt(
                    foundation,
                    genesis=SimpleNamespace(),
                    ledger=[{"generation_index": 1}],
                    cost_journal=fixture.cost_journal,
                    lifecycle=lifecycle,
                    segment_id=resumed.segment_id,
                )

            publish.assert_called_once()
            self.assertEqual(
                publish.call_args.kwargs["event_id"],
                f"segment-{resumed.segment_id}-boundary-reconciliation",
            )
            self.assertEqual(
                current.payload["previous_cost_chain_head_hash"],
                recovery_head,
            )
            self.assertEqual(
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    lifecycle,
                ),
                current.payload_hash,
            )
            foundation.provider.assert_not_called()
            foundation.runner.assert_not_called()
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=current.payload_hash,
            )

    def test_terminal_boundary_without_cost_change_reuses_generation_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp).resolve())
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            generation_1 = fixture.cost_journal.publish_snapshot(
                event_id="generation-001-settlement",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            first.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=generation_1.payload_hash,
            )
            second = fixture.start_segment(
                closed_generation_count=1,
                cost_chain_head_hash=generation_1.payload_hash,
            )
            generation_2 = fixture.cost_journal.publish_snapshot(
                event_id="generation-002-settlement",
                previous_cost_chain_head_hash=generation_1.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            second.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=2,
                cost_chain_head_hash=generation_2.payload_hash,
            )
            segment = fixture.start_segment(
                closed_generation_count=2,
                cost_chain_head_hash=generation_2.payload_hash,
            )
            generation = fixture.cost_journal.publish_snapshot(
                event_id="generation-003-settlement",
                previous_cost_chain_head_hash=generation_2.payload_hash,
                usage=StairGenerationActualCostUsageV2(),
            )
            lifecycle = fixture.audit(require_closed=False)
            foundation = SimpleNamespace(provider=Mock(), runner=Mock())
            usage = StairGenerationActualCostUsageV2()

            with (
                patch(
                    'true_rl_agent.agent_core.generation_session.collect_actual_cost_usage',
                    return_value=(usage, (), ()),
                ),
                patch.object(
                    fixture.cost_journal,
                    "publish_snapshot",
                    wraps=fixture.cost_journal.publish_snapshot,
                ) as publish,
            ):
                current = _current_validation_boundary_cost_receipt(
                    foundation,
                    genesis=SimpleNamespace(),
                    ledger=[
                        {"generation_index": 1},
                        {"generation_index": 2},
                        {"generation_index": 3},
                    ],
                    cost_journal=fixture.cost_journal,
                    lifecycle=lifecycle,
                    segment_id=segment.segment_id,
                )

            self.assertEqual(current.payload_hash, generation.payload_hash)
            publish.assert_not_called()
            self.assertEqual(len(fixture.cost_journal.load_chain()), 4)
            segment.finish(
                status="completed",
                reason="normal",
                closed_generation_count=3,
                cost_chain_head_hash=current.payload_hash,
            )
            terminal = fixture.audit(require_closed=True)
            self.assertTrue(terminal.terminal)
            self.assertEqual(terminal.cost_chain_head_hash, generation.payload_hash)

    def test_terminal_cost_replay_validates_conservative_charges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp).resolve())
            reservation = fixture.cost_journal.publish_reservation()
            segment = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            usage = StairGenerationActualCostUsageV2()
            charges = ("synthetic-conservative-charge",)
            terminal_cost = fixture.cost_journal.publish_snapshot(
                event_id="generation-003-settlement",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=usage,
                conservative_charges=charges,
            )
            segment.finish(
                status="completed",
                reason="normal",
                closed_generation_count=3,
                cost_chain_head_hash=terminal_cost.payload_hash,
            )
            lifecycle = fixture.audit(require_closed=True)

            self.assertEqual(_replay_validation_combined_cost_chain(
                fixture.cost_journal, lifecycle), terminal_cost.payload_hash)
            assert_actual_cost_receipt_matches_usage(
                terminal_cost, usage=usage, conservative_charges=charges,
                context="terminal cost replay",
            )

    def test_recovery_head_is_the_next_actual_settlement_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            attempt = fixture.attempt_journal(first, logical_suffix="crashed")
            attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()

            resumed = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
                recover=True,
            )
            after_recovery = fixture.audit(require_closed=False)
            recovery_head = resumed.cost_chain_head_hash
            self.assertEqual(
                after_recovery.recovery_cost_edges,
                (
                    (
                        reservation.payload_hash,
                        reservation.payload_hash,
                        recovery_head,
                    ),
                ),
            )
            self.assertNotEqual(recovery_head, reservation.payload_hash)

            settled = fixture.cost_journal.publish_snapshot(
                event_id="generation-001-settled",
                previous_cost_chain_head_hash=recovery_head,
                usage=StairGenerationActualCostUsage(),
                finalization_output_reserve_bytes=0,
            )
            self.assertEqual(
                settled.payload["previous_cost_chain_head_hash"],
                recovery_head,
            )
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=settled.payload_hash,
            )

            closed = fixture.audit(require_closed=True)
            self.assertEqual(
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    closed,
                ),
                settled.payload_hash,
            )

    def test_two_consecutive_recoveries_without_new_actual_do_not_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            attempt = fixture.attempt_journal(first, logical_suffix="first-crash")
            attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()

            second = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
                recover=True,
            )
            first_recovery_head = second.cost_chain_head_hash
            active_second = fixture.audit(require_closed=False)
            self.assertEqual(
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    active_second,
                ),
                first_recovery_head,
            )
            second.release_unclean_for_test()

            before_second_recovery = fixture.audit(require_closed=False)
            authoritative_head = _replay_validation_combined_cost_chain(
                fixture.cost_journal,
                before_second_recovery,
            )
            third = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=authoritative_head,
                recover=True,
            )
            after_second_recovery = fixture.audit(require_closed=False)

            self.assertEqual(authoritative_head, first_recovery_head)
            self.assertEqual(third.cost_chain_head_hash, first_recovery_head)
            self.assertEqual(len(after_second_recovery.recovery_cost_edges), 2)
            self.assertEqual(
                after_second_recovery.recovery_cost_edges[1],
                (
                    first_recovery_head,
                    first_recovery_head,
                    first_recovery_head,
                ),
            )
            self.assertEqual(
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    after_second_recovery,
                ),
                first_recovery_head,
            )
            third.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=0,
                cost_chain_head_hash=first_recovery_head,
            )

    def test_combined_replay_rejects_an_actual_cost_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            fixture.cost_journal.publish_reservation()
            fixture.cost_journal.publish_snapshot(
                event_id="forked-settlement",
                previous_cost_chain_head_hash=fixture.cost_contract.contract_hash,
                usage=StairGenerationActualCostUsage(),
                finalization_output_reserve_bytes=0,
            )

            with self.assertRaisesRegex(ValueError, "combined cost chain forks"):
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    fixture.audit(require_closed=False),
                )

    def test_combined_replay_rejects_a_settlement_that_skips_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            attempt = fixture.attempt_journal(first, logical_suffix="skip-recovery")
            attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()
            resumed = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
                recover=True,
            )
            try:
                fixture.cost_journal.publish_snapshot(
                    event_id="skipped-recovery-settlement",
                    previous_cost_chain_head_hash=reservation.payload_hash,
                    usage=StairGenerationActualCostUsage(),
                    finalization_output_reserve_bytes=0,
                )
                with self.assertRaisesRegex(ValueError, "combined cost chain forks"):
                    _replay_validation_combined_cost_chain(
                        fixture.cost_journal,
                        fixture.audit(require_closed=False),
                    )
            finally:
                resumed.release_unclean_for_test()

    def test_combined_replay_rejects_an_orphan_actual_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            fixture.cost_journal.publish_reservation()
            fixture.cost_journal.publish_snapshot(
                event_id="orphaned-settlement",
                previous_cost_chain_head_hash="f" * 64,
                usage=StairGenerationActualCostUsage(),
                finalization_output_reserve_bytes=0,
            )

            with self.assertRaisesRegex(ValueError, "combined cost chain has an orphan edge"):
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    fixture.audit(require_closed=False),
                )

    def test_gen3_settlement_then_recovery_only_terminal_keeps_settlement_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = _fixture(root)
            reservation = fixture.cost_journal.publish_reservation()
            first = fixture.start_segment(
                closed_generation_count=0,
                cost_chain_head_hash=reservation.payload_hash,
            )
            generation_3 = fixture.cost_journal.publish_snapshot(
                event_id="generation-003-settled",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsage(),
                finalization_output_reserve_bytes=0,
            )
            first.release_unclean_for_test()

            active = fixture.audit(require_closed=False)
            authoritative_head = _replay_validation_combined_cost_chain(
                fixture.cost_journal,
                active,
            )
            self.assertEqual(authoritative_head, generation_3.payload_hash)
            resumed = fixture.start_segment(
                closed_generation_count=3,
                cost_chain_head_hash=authoritative_head,
                recover=True,
            )
            resumed.finish(
                status="completed",
                reason="normal",
                closed_generation_count=3,
                cost_chain_head_hash=authoritative_head,
            )

            terminal = fixture.audit(require_closed=True)
            self.assertTrue(terminal.terminal)
            self.assertEqual(
                terminal.recovery_cost_edges,
                (
                    (
                        reservation.payload_hash,
                        generation_3.payload_hash,
                        generation_3.payload_hash,
                    ),
                ),
            )
            self.assertEqual(
                _replay_validation_combined_cost_chain(
                    fixture.cost_journal,
                    terminal,
                ),
                generation_3.payload_hash,
            )



class _Fixture:
    def __init__(
        self,
        *,
        root: Path,
        policy: GenerationContinuationPolicy,
        session_contract: GenerationSessionContract,
        identity: GenerationLifecycleIdentity,
        cost_contract: StairGenerationSessionCostContractV2,
    ) -> None:
        self.root = root
        self.policy = policy
        self.session_contract = session_contract
        self.identity = identity
        self.cost_contract = cost_contract
        self.cost_journal = StairGenerationActualCostJournal(
            root,
            contract=cost_contract,
        )

    def start_segment(
        self,
        *,
        closed_generation_count: int,
        cost_chain_head_hash: str,
        recover: bool = False,
    ) -> GenerationSegmentJournal:
        recovery_options = (
            {
                "recovery_process_incarnation": lambda _pid: None,
                "process_group_alive": lambda _group: False,
            }
            if recover
            else {}
        )
        return GenerationSegmentJournal.start(
            output_dir=self.root,
            identity=self.identity,
            policy=self.policy,
            session_contract=self.session_contract,
            closed_generation_count=closed_generation_count,
            cost_chain_head_hash=cost_chain_head_hash,
            attempt_journal_root=self.root / "attempts",
            **recovery_options,
        )

    def attempt_journal(
        self,
        segment: GenerationSegmentJournal,
        *,
        logical_suffix: str,
        generation_id: str = "generation-001",
    ) -> JobAttemptJournal:
        scope = JobAttemptScope(
            journal_root=self.root / "attempts",
            session_genesis_hash=self.identity.session_genesis_hash,
            session_contract_hash=self.session_contract.contract_hash,
            generation_id=generation_id,
            lane_id="lane-a",
            invocation_segment_id=segment.segment_id,
            budget_contract_hash=self.cost_contract.contract_hash,
            resource_binding_hash=None,
        )
        return JobAttemptJournal(
            scope=scope,
            logical_job_id=f"{generation_id}-lane-a-{logical_suffix}",
            job_kind="train",
            job_input_hash="c" * 64,
            environment_hash="d" * 64,
            execution_device="cpu",
            resource_binding_hash=None,
        )

    def audit(self, *, require_closed: bool):
        return audit_generation_lifecycle(
            output_dir=self.root,
            identity=self.identity,
            policy=self.policy,
            require_closed=require_closed,
        )


def _fixture(root: Path) -> _Fixture:
    session_id = "p6c-cost-recovery-test"
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
    )
    cost_contract = StairGenerationSessionCostContractV2(
        session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
        max_generations=3,
        scientific_cost_basis=build_stair_generation_scientific_cost_basis(
            session_id=session_id,
            config=config,
        ),
        retry_train_jobs=1,
        retry_eval_seed_units=7,
        successful_allocated_active_gpu_hours=(
            STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        retry_allocated_active_gpu_hours=(
            STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        successful_output_bytes=(STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES),
        retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    )
    policy = GenerationContinuationPolicy(
        max_segments=4,
        allowed_continuation_reasons=(
            "operator_pause",
            "typed_infrastructure_failure",
            "exception",
        ),
    )
    hashes = {
        "authoring_policy": "1" * 64,
        "continuation_policy": policy.policy_hash,
        "evaluation_cost": "2" * 64,
        "evaluation_protocol": "3" * 64,
        "exploration_projector": "e" * 64,
        "final_confirmation_plan": "4" * 64,
        "provider_policy": "5" * 64,
        "resource_plan": "6" * 64,
        "retry_policy": "7" * 64,
        "selector": "8" * 64,
        "session_cost": cost_contract.contract_hash,
    }
    session_contract = GenerationSessionContract(
        schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
        session_id=session_id,
        task_id="stair_climbing",
        backend_id="isaac_lab_rsl_rl",
        max_generations=3,
        max_trials=6,
        max_train_jobs=6,
        bound_contract_hashes=hashes,
    )
    identity = GenerationLifecycleIdentity(
        session_id=session_id,
        session_genesis_hash="9" * 64,
        generation_session_contract_hash=session_contract.contract_hash,
        final_confirmation_plan_hash=hashes["final_confirmation_plan"],
        ledger_epoch_hash="a" * 64,
        output_root=str(root),
    )
    publish_generation_continuation_policy(root, policy)
    return _Fixture(
        root=root,
        policy=policy,
        session_contract=session_contract,
        identity=identity,
        cost_contract=cost_contract,
    )


if __name__ == "__main__":
    unittest.main()

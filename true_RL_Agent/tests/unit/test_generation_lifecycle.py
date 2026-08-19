from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    make_bootstrap_current_best,
)
from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
from true_rl_agent.agent_core.generation_lifecycle import (
    GenerationContinuationPolicy,
    GenerationLifecycleIdentity,
    GenerationSegmentJournal,
    audit_generation_lifecycle,
    build_generation_continuation_cursor,
    load_generation_continuation_policy,
    publish_generation_continuation_policy,
    _process_group_is_alive,
    _process_incarnation_token,
    _unclosed_reservations,
    _validate_session_attempts_closed,
)
from true_rl_agent.agent_core.ledger import LEDGER_RECORD_SCHEMA_VERSION
from true_rl_agent.agent_core.process_lock import ProcessFileLockUnavailableError
from true_rl_agent.runner.job_attempts import (
    RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)


class GenerationLifecycleTests(unittest.TestCase):
    def test_segment_scans_ignore_pre_genesis_attempt_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _, contract, identity = _contracts(root)
            scope = PreGenesisExecutionScope(
                journal_root=root / "attempts",
                session_id=identity.session_id,
                genesis_identity_hash="1" * 64,
                session_contract_hash=contract.contract_hash,
                lane_id="lane-a",
                invocation_segment_id="segment-pre-genesis",
                budget_contract_hash="2" * 64,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
            )
            journal = JobAttemptJournal(
                scope=scope,
                logical_job_id="reference-train",
                job_kind="train",
                job_input_hash="3" * 64,
                environment_hash="4" * 64,
                execution_device="cpu",
                resource_binding_hash=None,
            )
            journal.reserve(0, previous_result=None)

            self.assertEqual(
                _unclosed_reservations(
                    root / "attempts",
                    identity=identity,
                    segment_id="segment-pre-genesis",
                ),
                [],
            )
            _validate_session_attempts_closed(
                root / "attempts",
                identity=identity,
            )

    def test_controlled_segments_form_one_terminal_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            publish_generation_continuation_policy(root, policy)
            self.assertEqual(load_generation_continuation_policy(root), policy)

            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            with self.assertRaises(ProcessFileLockUnavailableError):
                GenerationSegmentJournal.start(
                    output_dir=root,
                    identity=identity,
                    policy=policy,
                    session_contract=contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=root / "attempts",
                )
            first.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=2,
                cost_chain_head_hash="b" * 64,
            )

            second = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=2,
                cost_chain_head_hash="b" * 64,
                attempt_journal_root=root / "attempts",
            )
            second.finish(
                status="completed",
                reason="normal",
                closed_generation_count=15,
                cost_chain_head_hash="c" * 64,
            )
            audit = audit_generation_lifecycle(
                output_dir=root,
                identity=identity,
                policy=policy,
                require_closed=True,
            )
            self.assertEqual(audit.segment_count, 2)
            self.assertEqual(audit.closed_generation_count, 15)
            self.assertEqual(audit.cost_chain_head_hash, "c" * 64)
            self.assertTrue(audit.terminal)
            self.assertIsNone(audit.active_segment_id)
            with self.assertRaisesRegex(ValueError, "not continuation eligible"):
                GenerationSegmentJournal.start(
                    output_dir=root,
                    identity=identity,
                    policy=policy,
                    session_contract=contract,
                    closed_generation_count=15,
                    cost_chain_head_hash="c" * 64,
                    attempt_journal_root=root / "attempts",
                )

    def test_unclean_reservation_is_charged_then_uses_only_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            attempt = _attempt_journal(
                root,
                contract=contract,
                identity=identity,
                segment_id=first.segment_id,
            )
            initial = attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()

            resumed = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
                recovery_process_incarnation=lambda _pid: None,
                process_group_alive=lambda _group: False,
            )
            recovered = attempt.load_result(0)
            self.assertEqual(
                recovered.payload["schema_version"],
                RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION,
            )
            self.assertEqual(recovered.payload["failure_kind"], "abandoned_unclean")
            self.assertTrue(recovered.payload["retryable"])
            self.assertNotEqual(
                resumed.cost_chain_head_hash,
                "a" * 64,
            )
            retry_index, retry, previous = attempt.begin_or_resume()
            self.assertEqual(retry_index, 1)
            self.assertIsNotNone(previous)
            self.assertTrue(retry.created)
            attempt.finish(
                retry,
                outcome="completed",
                failure_kind=None,
                retryable=False,
                output_hash="d" * 64,
                returncode=0,
            )
            with self.assertRaisesRegex(ValueError, "exhausted its retry"):
                attempt.begin_or_resume()
            resumed.finish(
                status="completed",
                reason="normal",
                closed_generation_count=15,
                cost_chain_head_hash=resumed.cost_chain_head_hash,
            )
            audit = audit_generation_lifecycle(
                output_dir=root,
                identity=identity,
                policy=policy,
                require_closed=True,
            )
            self.assertTrue(audit.terminal)
            self.assertEqual(audit.segment_count, 2)
            self.assertTrue(initial.path.exists())

    def test_unclean_recovery_refuses_a_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            first.release_unclean_for_test()
            with self.assertRaisesRegex(ValueError, "still alive"):
                GenerationSegmentJournal.start(
                    output_dir=root,
                    identity=identity,
                    policy=policy,
                    session_contract=contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=root / "attempts",
                    recovery_process_incarnation=(
                        lambda _pid: first.start_record[
                            "process_incarnation_token"
                        ]
                    ),
                )
            with self.assertRaisesRegex(ValueError, "process group is still alive"):
                GenerationSegmentJournal.start(
                    output_dir=root,
                    identity=identity,
                    policy=policy,
                    session_contract=contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=root / "attempts",
                    recovery_process_incarnation=lambda _pid: None,
                    process_group_alive=lambda _group: True,
                )

    def test_unclean_recovery_distinguishes_a_reused_process_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            old_token = first.start_record["process_incarnation_token"]
            reused_token = "f" * 64 if old_token != "f" * 64 else "e" * 64
            first.release_unclean_for_test()

            resumed = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
                process_group_alive=lambda _group: True,
                recovery_process_incarnation=lambda _pid: reused_token,
            )
            recovery_path = next((root / "generation_recoveries").glob("*.json"))
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            self.assertTrue(recovery["process_id_reused"])
            self.assertEqual(
                recovery["observed_process_incarnation_token"],
                reused_token,
            )
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=0,
                cost_chain_head_hash=resumed.cost_chain_head_hash,
            )

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires real Linux procfs")
    def test_linux_process_incarnation_and_process_group_liveness(self) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        try:
            token = _process_incarnation_token(child.pid)
            self.assertIsNotNone(token)
            self.assertTrue(_process_group_is_alive(child.pid))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
        self.assertIsNone(_process_incarnation_token(child.pid))
        self.assertFalse(_process_group_is_alive(child.pid))

    def test_unclean_recovery_accepts_ledger_progress_made_in_dead_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            first.release_unclean_for_test()
            resumed = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=3,
                cost_chain_head_hash="b" * 64,
                attempt_journal_root=root / "attempts",
                recovery_process_incarnation=lambda _pid: None,
                process_group_alive=lambda _group: False,
            )
            self.assertEqual(resumed.start_record["generations_at_start"], 3)
            self.assertEqual(resumed.cost_chain_head_hash, resumed.start_record["cost_chain_head_hash"])
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=3,
                cost_chain_head_hash=resumed.cost_chain_head_hash,
            )

    def test_published_recovery_is_reconciled_after_recovery_process_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            first = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
            )
            attempt = _attempt_journal(
                root,
                contract=contract,
                identity=identity,
                segment_id=first.segment_id,
            )
            attempt.reserve(0, previous_result=None)
            first.release_unclean_for_test()
            with patch.object(
                JobAttemptJournal,
                "finish",
                side_effect=RuntimeError("simulated recovery crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated recovery crash"):
                    GenerationSegmentJournal.start(
                        output_dir=root,
                        identity=identity,
                        policy=policy,
                        session_contract=contract,
                        closed_generation_count=0,
                        cost_chain_head_hash="a" * 64,
                        attempt_journal_root=root / "attempts",
                        recovery_process_incarnation=lambda _pid: None,
                        process_group_alive=lambda _group: False,
                    )
            resumed = GenerationSegmentJournal.start(
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                attempt_journal_root=root / "attempts",
                recovery_process_incarnation=lambda _pid: None,
                process_group_alive=lambda _group: False,
            )
            self.assertEqual(
                attempt.load_result(0).payload["failure_kind"],
                "abandoned_unclean",
            )
            resumed.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=0,
                cost_chain_head_hash=resumed.cost_chain_head_hash,
            )

    def test_continuation_cursor_is_rebuilt_from_empty_ledger_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            cursor = build_generation_continuation_cursor(
                ledger_path=root / "ledger.jsonl",
                output_dir=root,
                identity=identity,
                policy=policy,
                session_contract=contract,
                initial_current_best=make_bootstrap_current_best(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            self.assertEqual(cursor.next_generation_index, 1)
            self.assertEqual(cursor.ledger_record_count, 0)
            self.assertIsNone(cursor.lifecycle_head_hash)
            self.assertEqual(cursor.cost_chain_head_hash, "0" * 64)

    def test_generation_cursor_rejects_a_width1_round_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            policy, contract, identity = _contracts(root)
            ledger_path = root / "ledger.jsonl"
            ledger_path.write_text(
                json.dumps(
                    {
                        "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
                        "record_kind": "round",
                        "authoring_mode": "provider_authoring",
                        "ledger_epoch_hash": identity.ledger_epoch_hash,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "requires generation ledger records",
            ):
                build_generation_continuation_cursor(
                    ledger_path=ledger_path,
                    output_dir=root,
                    identity=identity,
                    policy=policy,
                    session_contract=contract,
                    initial_current_best=make_bootstrap_current_best(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )


def _contracts(
    root: Path,
) -> tuple[
    GenerationContinuationPolicy,
    GenerationSessionContract,
    GenerationLifecycleIdentity,
]:
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
    }
    contract = GenerationSessionContract(
        session_id="generation-session-p5",
        task_id="fake-task",
        backend_id="fake-backend",
        max_generations=15,
        max_trials=30,
        max_train_jobs=30,
        bound_contract_hashes=hashes,
    )
    identity = GenerationLifecycleIdentity(
        session_id=contract.session_id,
        session_genesis_hash="9" * 64,
        generation_session_contract_hash=contract.contract_hash,
        final_confirmation_plan_hash=hashes["final_confirmation_plan"],
        ledger_epoch_hash="a" * 64,
        output_root=str(root),
    )
    return policy, contract, identity


def _attempt_journal(
    root: Path,
    *,
    contract: GenerationSessionContract,
    identity: GenerationLifecycleIdentity,
    segment_id: str,
) -> JobAttemptJournal:
    scope = JobAttemptScope(
        journal_root=root / "attempts",
        session_genesis_hash=identity.session_genesis_hash,
        session_contract_hash=contract.contract_hash,
        generation_id="generation-001",
        lane_id="lane-a",
        invocation_segment_id=segment_id,
        budget_contract_hash="b" * 64,
        resource_binding_hash=None,
    )
    return JobAttemptJournal(
        scope=scope,
        logical_job_id="generation-001-lane-a-train",
        job_kind="train",
        job_input_hash="c" * 64,
        environment_hash="d" * 64,
        execution_device="cpu",
        resource_binding_hash=None,
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.fakes.evaluation import make_test_candidate_package
from tests.fakes.fake_components import (
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_round_session_fields,
)
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    load_ledger,
)
from true_rl_agent.agent_core.round import (
    RoundInputs,
    commit_candidate_trial,
    execute_candidate_trial,
)
from true_rl_agent.agent_core.trial_receipts import (
    TrialReceiptJournal,
    load_trial_execution_receipt,
)
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.policy import PolicyInitialization


class TrialExecutorTests(unittest.TestCase):
    def _inputs(self, root: Path, *, runner: FakeRunner) -> RoundInputs:
        task = FakeTaskPlugin()
        candidate = make_test_candidate_package(
            "candidate-trial-executor",
            change_set=CandidateChangeSet(
                changed_components=frozenset({"reward_callable"})
            ),
            metadata={"fake_public_score": 0.20},
        )
        return RoundInputs(
            round_id="round-001",
            task_plugin=task,
            backend_adapter=FakeBackendAdapter(),
            runner=runner,
            candidate_admission_policy=FakeVerifier(),
            candidate=candidate,
            policy_initialization=PolicyInitialization.random(),
            ledger_path=root / "ledger.jsonl",
            trial_receipt_root=root / "trial_receipts",
            authoring_mode=NON_PROVIDER_AUTHORING_MODE,
            public_context={"previous_rounds": 0, "public_only": True},
            **fake_round_session_fields(task_plugin=task),
        )

    def test_execution_has_no_ledger_side_effect_and_commit_is_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())

            execution = execute_candidate_trial(inputs)

            self.assertFalse(inputs.ledger_path.exists())
            self.assertTrue(Path(execution.receipt_ref.path).is_file())
            first = commit_candidate_trial(inputs, execution)
            second = commit_candidate_trial(inputs, execution)
            ledger = load_ledger(inputs.ledger_path)

        self.assertEqual(first, second)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0], execution.record)

    def test_train_receipt_is_reused_after_eval_interruption(self) -> None:
        class InterruptEvalRunner(FakeRunner):
            def __init__(self) -> None:
                self.train_calls = 0

            def run_train(self, train_spec):
                self.train_calls += 1
                return super().run_train(train_spec)

            def run_eval(self, eval_spec):
                _ = eval_spec
                raise KeyboardInterrupt

        class TrackingRunner(FakeRunner):
            def __init__(self) -> None:
                self.train_calls = 0
                self.eval_calls = 0

            def run_train(self, train_spec):
                self.train_calls += 1
                return super().run_train(train_spec)

            def run_eval(self, eval_spec):
                self.eval_calls += 1
                return super().run_eval(eval_spec)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interrupted = InterruptEvalRunner()
            inputs = self._inputs(root, runner=interrupted)
            with self.assertRaises(KeyboardInterrupt):
                execute_candidate_trial(inputs)
            self.assertEqual(interrupted.train_calls, 1)
            self.assertFalse(inputs.ledger_path.exists())

            resumed = TrackingRunner()
            execution = execute_candidate_trial(
                replace(inputs, runner=resumed)
            )

            self.assertEqual(resumed.train_calls, 0)
            self.assertEqual(resumed.eval_calls, 1)
            self.assertFalse(execution.reused_receipt)
            commit_candidate_trial(replace(inputs, runner=resumed), execution)
            self.assertEqual(len(load_ledger(inputs.ledger_path)), 1)

    def test_complete_receipt_skips_runner_on_reexecution(self) -> None:
        class NoRunnerCalls(FakeRunner):
            def run_train(self, train_spec):
                raise AssertionError("completed trial must not retrain")

            def run_eval(self, eval_spec):
                raise AssertionError("completed trial must not reevaluate")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())
            first = execute_candidate_trial(inputs)
            second = execute_candidate_trial(
                replace(inputs, runner=NoRunnerCalls())
            )

        self.assertFalse(first.reused_receipt)
        self.assertTrue(second.reused_receipt)
        self.assertEqual(first.record, second.record)
        self.assertEqual(first.receipt_ref, second.receipt_ref)

    def test_eval_receipt_is_reused_when_final_publish_is_interrupted(
        self,
    ) -> None:
        class TrackingRunner(FakeRunner):
            def __init__(self) -> None:
                self.train_calls = 0
                self.eval_calls = 0

            def run_train(self, train_spec):
                self.train_calls += 1
                return super().run_train(train_spec)

            def run_eval(self, eval_spec):
                self.eval_calls += 1
                return super().run_eval(eval_spec)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_runner = TrackingRunner()
            inputs = self._inputs(root, runner=first_runner)
            with patch.object(
                TrialReceiptJournal,
                "publish_execution",
                side_effect=OSError("injected final receipt interruption"),
            ):
                with self.assertRaises(OSError):
                    execute_candidate_trial(inputs)
            self.assertEqual(first_runner.train_calls, 1)
            self.assertEqual(first_runner.eval_calls, 1)

            resumed_runner = TrackingRunner()
            execution = execute_candidate_trial(
                replace(inputs, runner=resumed_runner)
            )

        self.assertEqual(resumed_runner.train_calls, 0)
        self.assertEqual(resumed_runner.eval_calls, 0)
        self.assertFalse(execution.reused_receipt)

    def test_commit_rejects_tampered_final_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())
            execution = execute_candidate_trial(inputs)
            receipt_path = Path(execution.receipt_ref.path)
            receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                commit_candidate_trial(inputs, execution)
            self.assertFalse(inputs.ledger_path.exists())

    def test_commit_rejects_receipt_reuse_across_round_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())
            execution = execute_candidate_trial(inputs)
            mismatches = (
                (
                    replace(inputs, session_genesis_hash="a" * 64),
                    "session_genesis_hash",
                ),
                (
                    replace(inputs, ledger_epoch_hash="e" * 64),
                    "ledger_epoch_hash",
                ),
                (replace(inputs, round_id="round-002"), "round_id"),
            )

            for mismatched_inputs, field_name in mismatches:
                with self.subTest(field_name=field_name):
                    with self.assertRaisesRegex(ValueError, field_name):
                        commit_candidate_trial(mismatched_inputs, execution)
            self.assertFalse(inputs.ledger_path.exists())

    def test_commit_consumes_contract_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())
            execution = execute_candidate_trial(inputs)
            contract, loaded = load_trial_execution_receipt(
                execution.receipt_ref
            )
            forged_contract = {
                **contract,
                "session_genesis_hash": "a" * 64,
            }

            with patch(
                "true_rl_agent.agent_core.round.load_trial_execution_receipt",
                return_value=(forged_contract, loaded),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "contract session_genesis_hash",
                ):
                    commit_candidate_trial(inputs, execution)
            self.assertFalse(inputs.ledger_path.exists())

    def test_contract_rejects_a_record_from_another_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._inputs(root, runner=FakeRunner())
            execution = execute_candidate_trial(inputs)
            contract, loaded = load_trial_execution_receipt(
                execution.receipt_ref
            )
            forged_contract = {
                **contract,
                "session_genesis_hash": "a" * 64,
            }
            journal = TrialReceiptJournal(
                root / "forged_receipts",
                contract=forged_contract,
            )
            journal.bind_contract()

            with self.assertRaisesRegex(
                ValueError,
                "record field: session_genesis_hash",
            ):
                journal.publish_execution(
                    record=loaded.record,
                    round_stage=loaded.round_stage,
                    train_receipt_ref=None,
                    eval_receipt_ref=None,
                    mechanism_operation_delta=(
                        loaded.mechanism_operation_delta
                    ),
                )


if __name__ == "__main__":
    unittest.main()

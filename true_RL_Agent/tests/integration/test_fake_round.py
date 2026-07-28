from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_test_candidate_package,
)
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_round_session_fields,
)
from true_rl_agent.agent_core.decision import DecisionOutcome
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.memory import PromptPayloadRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec


class FakeRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = FakeTaskPlugin()
        self.backend = FakeBackendAdapter()
        self.runner = FakeRunner()
        self.verifier = FakeVerifier()

    def test_malformed_eval_score_is_recorded_as_eval_incomplete(self) -> None:
        class MalformedScoreRunner(FakeRunner):
            def run_eval(self, eval_spec):
                _ = eval_spec
                raise ValueError(
                    "public eval public_score must be a finite non-boolean number"
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-malformed-score",
                change_set=CandidateChangeSet(
                    changed_components=frozenset({"reward_callable"})
                ),
                metadata={"fake_public_score": 0.20},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-malformed-score",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=MalformedScoreRunner(),
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(result.decision.reason, "eval_incomplete")
        self.assertIsNone(result.public_result)
        self.assertEqual(ledger[0]["decision"]["reason"], "eval_incomplete")
        self.assertEqual(ledger[0]["runner_error"]["type"], "ValueError")

    def test_public_eval_fail_status_is_recorded_without_entering_comparison(self) -> None:
        class FailedPublicEvalRunner(FakeRunner):
            def run_eval(self, eval_spec):
                _ = eval_spec
                return PublicEvalResult(
                    status="fail",
                    public_score=0.0,
                    metrics={},
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-public-eval-fail",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20},
            )
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-public-eval-fail",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=FailedPublicEvalRunner(),
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(result.decision.reason, "public_result_status_not_pass")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["comparison"], {})
        self.assertEqual(ledger[0]["decision"]["consumed_gates"], [])
        self.assertTrue(
            validate_decision_record(
                ledger[0],
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
            ).valid
        )

    def _run_candidate(
        self,
        change_kind: str,
        public_score: float,
        *,
        verifier_status: str = "pass",
    ) -> tuple[DecisionOutcome, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                f"candidate-{change_kind}",
                change_set=CandidateChangeSet(changed_components=frozenset({change_kind})),
                metadata={"fake_public_score": public_score, "verifier_status": verifier_status},
            )
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            return result.decision.outcome, load_ledger(ledger_path)

    def test_code_level_candidate_can_policy_accept_and_append_ledger(self) -> None:
        outcome, ledger = self._run_candidate("reward_callable", public_score=0.20)

        self.assertIs(outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["outcome"], "policy_improvement_accept")
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-reward_callable")
        self.assertEqual(ledger[0]["public_result"]["public_score"], 0.20)
        self.assertEqual(
            ledger[0]["jobs"]["eval"]["checkpoint"]["uri"],
            ledger[0]["train_result"]["checkpoint"]["uri"],
        )
        self.assertEqual(ledger[0]["public_context"]["previous_rounds"], 0)
        self.assertFalse(ledger[0]["feedback_safe"]["private_eval_used"])

    def test_round_ledger_records_prompt_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.jsonl"
            prompt_ref = PromptPayloadRef(path=str(root / "round-001" / "prompt_payload.json"))
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20},
            )

            run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                    prompt_payload_ref=prompt_ref,
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(ledger[0]["prompt_payload"], prompt_ref.to_public_dict())
        self.assertEqual(ledger[0]["prompt_payload"]["visibility"], "public_prompt_payload")

    def test_parameter_only_candidate_rejects_even_with_high_score(self) -> None:
        outcome, ledger = self._run_candidate("reward_weight", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.REJECT)
        self.assertEqual(ledger[0]["decision"]["reason"], "parameter_only_candidate")
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_diagnostic_only_candidate_records_without_policy_accept(self) -> None:
        outcome, ledger = self._run_candidate("diagnostic", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.DIAGNOSTIC_IMPROVEMENT_RECORD)
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_engineering_only_candidate_records_without_policy_accept(self) -> None:
        outcome, ledger = self._run_candidate("failure_classifier", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.ENGINEERING_IMPROVEMENT_RECORD)
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_verifier_failure_rejects_without_training(self) -> None:
        outcome, ledger = self._run_candidate("reward_callable", public_score=0.95, verifier_status="fail")

        self.assertIs(outcome, DecisionOutcome.REJECT)
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_verifier_failed")
        self.assertEqual(ledger[0]["candidate"]["changed_components"], [])
        self.assertEqual(ledger[0]["candidate_surface"]["classification"], "no_code_change")
        self.assertIsNone(ledger[0]["train_result"])
        self.assertIsNone(ledger[0]["public_result"])

    def test_task_invariant_violation_rejects_before_training(self) -> None:
        class TamperedBackend(FakeBackendAdapter):
            def materialize_jobs(self, **kwargs):
                bundle = super().materialize_jobs(**kwargs)
                return BackendJobBundle(
                    train_spec=bundle.train_spec,
                    eval_spec=bundle.eval_spec,
                    task_invariants=TaskInvariantSpec({"fall_threshold": 0.10}),
                )

        class TrackingRunner(FakeRunner):
            def __init__(self) -> None:
                self.train_calls = 0

            def run_train(self, train_spec):
                self.train_calls += 1
                return super().run_train(train_spec)

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            runner = TrackingRunner()
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.95},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=TamperedBackend(),
                    runner=runner,
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )

            ledger = load_ledger(ledger_path)
            self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
            self.assertEqual(result.decision.reason, "task_invariant_violation")
            self.assertEqual(runner.train_calls, 0)
            self.assertIsNone(ledger[0]["train_result"])

    def test_round_decision_records_comparison_and_consumed_eval_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    verifier=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertIs(result.decision.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertIn("comparison", ledger[0]["decision"])
        self.assertEqual(ledger[0]["decision"]["comparison"]["mode"], "bootstrap")
        self.assertIn("sealed_validation_gate", ledger[0]["decision"]["consumed_gates"])
        self.assertEqual(ledger[0]["decision"]["gate_statuses"]["sealed_validation_gate"], "pass")

    def test_fake_runner_uses_eval_spec_public_accept_seed_count(self) -> None:
        eval_spec = EvalJobSpec(
            job_id="fake-eval",
            task_id="fake_push_recovery",
            subject=ExperimentSubject.candidate("candidate-reward_callable"),
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash="7" * 64,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=CheckpointRef(
                "artifact://fake/best.pt",
                "sha256:" + "f" * 64,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            public_accept_seed_count=2,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=2,
                sealed_validation_seed_count=8,
                acceptance_protocol_hash=FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
            ),
            metadata={"fake_public_score": 0.42, "changed_components": ["reward_callable"]},
        )

        result = self.runner.run_eval(eval_spec)

        self.assertEqual(result.metrics["public_accept"]["score_stats"]["count"], 2)
        self.assertEqual(
            result.metrics["acceptance"]["min_score_count"],
            FAKE_ACCEPTANCE_PROTOCOL.min_score_count,
        )
        self.assertEqual(result.metrics["accept_evidence_gate"]["observed_count"], 2)
        self.assertEqual(len(result.metrics["public_accept"]["seed_results"]), 2)


if __name__ == "__main__":
    unittest.main()

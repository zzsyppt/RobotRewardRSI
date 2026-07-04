from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.decision import DecisionOutcome
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.memory import PromptPayloadRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.tasks.base import TaskInvariantSpec

from tests.fakes.fake_components import FakeBackendAdapter, FakeRunner, FakeTaskPlugin, FakeVerifier


class FakeRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = FakeTaskPlugin()
        self.backend = FakeBackendAdapter()
        self.runner = FakeRunner()
        self.verifier = FakeVerifier()

    def _run_candidate(
        self,
        change_kind: str,
        public_score: float,
        *,
        verifier_status: str = "pass",
    ) -> tuple[DecisionOutcome, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = CandidatePackageRef(
                candidate_id=f"candidate-{change_kind}",
                path=f"memory://candidate-{change_kind}",
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
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
            candidate = CandidatePackageRef(
                candidate_id="candidate-reward_callable",
                path="memory://candidate-reward_callable",
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
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
            candidate = CandidatePackageRef(
                candidate_id="candidate-reward_callable",
                path="memory://candidate-reward_callable",
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            )

            ledger = load_ledger(ledger_path)
            self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
            self.assertEqual(result.decision.reason, "task_invariant_violation")
            self.assertEqual(runner.train_calls, 0)
            self.assertIsNone(ledger[0]["train_result"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.context import build_public_context, sanitize_public_context
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.loop import AgentLoop, LoopInputs
from true_rl_agent.agent_core.provider import ReplayProvider
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.agent_core.session import AgentSession
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import CheckpointRef, TrainJobSpec

from tests.fakes.fake_components import FakeBackendAdapter, FakeRunner, FakeTaskPlugin, FakeVerifier


class AgentCoreLoopTests(unittest.TestCase):
    def _candidate(self, candidate_id: str, component: str, public_score: float) -> CandidatePackageRef:
        return CandidatePackageRef(
            candidate_id=candidate_id,
            path=f"memory://{candidate_id}",
            change_set=CandidateChangeSet(frozenset({component})),
            metadata={"fake_public_score": public_score},
        )

    def test_replay_provider_runs_three_rounds_and_builds_public_context(self) -> None:
        candidates = [
            self._candidate("round-001-knob", "reward_weight_scale", 0.95),
            self._candidate("round-002-diagnostic", "diagnostic", 0.95),
            self._candidate("round-003-code", "reward_callable", 0.25),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "session" / "ledger.jsonl"
            loop = AgentLoop(
                LoopInputs(
                    session_id="session",
                    round_count=3,
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=FakeRunner(),
                    verifier=FakeVerifier(),
                    provider=ReplayProvider(candidates),
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    initial_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            )

            results = loop.run()
            ledger = load_ledger(ledger_path)
            next_context = build_public_context(ledger)

            self.assertEqual(len(results), 3)
            self.assertEqual(len(ledger), 3)
            self.assertEqual(ledger[0]["decision"]["outcome"], "reject")
            self.assertEqual(ledger[0]["anti_tuning_decision"]["pure_knob_streak"], 1)
            self.assertEqual(ledger[1]["decision"]["outcome"], "diagnostic_improvement_record")
            self.assertEqual(ledger[2]["decision"]["outcome"], "policy_improvement_accept")
            self.assertIn("curriculum", ledger[2])
            self.assertIn("train", ledger[2]["jobs"])
            self.assertIn("eval", ledger[2]["jobs"])
            self.assertEqual(ledger[2]["mechanism_attribution"]["status"], "pass")
            self.assertGreater(ledger[2]["runtime_activation_summary"]["reward_term_calls"], 0)
            self.assertTrue(ledger[2]["ablation_summary"]["triggered"])
            self.assertEqual(next_context["previous_rounds"], 3)
            self.assertEqual(next_context["best_public_score"], 0.25)
            self.assertNotIn("private_seed", str(next_context))

    def test_session_dirs_and_public_context_sanitizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = AgentSession.create(root_dir=Path(tmp), session_id="session-001")
            sanitized = sanitize_public_context(
                {
                    "public_metric": 0.5,
                    "private_seed": 123,
                    "nested": {"api_key": "sk-secret", "ok": True},
                    "note": "sk-secret-value",
                }
            )

            self.assertTrue(session.session_dir.is_dir())
            self.assertEqual(session.ledger_path, session.session_dir / "ledger.jsonl")
            self.assertEqual(session.round_dir("round-001"), session.session_dir / "rounds" / "round-001")
            self.assertEqual(sanitized["public_metric"], 0.5)
            self.assertEqual(sanitized["nested"], {"ok": True})
            self.assertEqual(sanitized["note"], "[redacted]")
            self.assertNotIn("private_seed", sanitized)

    def test_code_level_public_score_without_runtime_activation_is_not_accepted(self) -> None:
        class NoActivationRunner(FakeRunner):
            def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
                raw_checkpoint = CheckpointRef(
                    uri=f"artifact://{train_spec.job_id}/best.pt",
                    checksum=f"sha256:{train_spec.candidate_id}",
                    role="train_output",
                )
                return TrainJobResult(
                    job_id=train_spec.job_id,
                    checkpoint=seal_checkpoint(raw_checkpoint, sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt"),
                    summary={"formal_training_executed": False, "budget_steps": train_spec.budget_steps},
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-activation-missing",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=NoActivationRunner(),
                    verifier=FakeVerifier(),
                    candidate=self._candidate("activation-missing", "reward_callable", 0.95),
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True, "private_seed": 999},
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            )

            ledger = load_ledger(ledger_path)

            self.assertFalse(result.decision.accepted)
            self.assertEqual(result.decision.reason, "runner_activation_missing")
            self.assertEqual(ledger[0]["mechanism_attribution"]["status"], "attribution_unclear")
            self.assertNotIn("private_seed", ledger[0]["public_context"])


if __name__ == "__main__":
    unittest.main()

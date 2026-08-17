from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import make_test_candidate_package
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_round_session_fields,
    prepare_fake_session_genesis,
)
from true_rl_agent.agent_core.context import (
    build_public_context,
    sanitize_public_context,
)
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    append_ledger_record,
    load_ledger,
)
from true_rl_agent.agent_core.loop import AgentLoop, LoopInputs
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.agent_core.session import AgentSession
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import TrainJobSpec


class _AuthoringCapableProvider:
    def next_candidate(self, *, prompt_payload):
        raise AssertionError("legacy AgentLoop must not call authoring-capable providers")

    def next_authoring_response(self, *, prompt_payload):
        raise AssertionError("legacy AgentLoop must not call authoring-capable providers")


class _AliasMutatingReplayProvider:
    def __init__(self, candidates):
        self._remaining = list(candidates)

    def next_candidate(self, *, prompt_payload):
        prompt_payload["public_context"]["previous_rounds"] = 999
        if not self._remaining:
            raise StopIteration("replay provider has no remaining candidates")
        return self._remaining.pop(0)


class _NeverCalledReplayProvider:
    def __init__(self) -> None:
        self.called = False

    def next_candidate(self, *, prompt_payload):
        _ = prompt_payload
        self.called = True
        raise AssertionError("provider must not run after ledger integrity failure")


class AgentCoreLoopTests(unittest.TestCase):
    def _candidate(self, candidate_id: str, component: str, public_score: float) -> CandidatePackageRef:
        return make_test_candidate_package(
            candidate_id,
            change_set=CandidateChangeSet(frozenset({component})),
            metadata={"fake_public_score": public_score},
        )

    def test_replay_provider_runs_three_rounds_and_builds_public_context(self) -> None:
        candidates = [
            self._candidate("round-001-knob", "reward_weight_scale", 0.95),
            self._candidate("round-002-diagnostic", "diagnostic", 0.95),
            self._candidate("round-003-code", "reward_callable", 0.25),
        ]
        provider = _AliasMutatingReplayProvider(candidates)

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "session" / "ledger.jsonl"
            task = FakeTaskPlugin()
            backend = FakeBackendAdapter()
            runner = FakeRunner()
            genesis = prepare_fake_session_genesis(
                output_dir=ledger_path.parent,
                session_id="session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
            )
            loop = AgentLoop(
                LoopInputs(
                    session_id="session",
                    round_count=3,
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    session_genesis=genesis,
                    policy_accept_delta=0.05,
                )
            )

            results = loop.run()
            ledger = load_ledger(ledger_path)
            next_context = build_public_context(
                ledger,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                initial_current_best=genesis.initial_current_best,
                expected_session_genesis_hash=genesis.session_genesis_hash,
            )

            self.assertEqual(len(results), 3)
            self.assertEqual(len(ledger), 3)
            self.assertEqual(ledger[0]["decision"]["outcome"], "reject")
            self.assertEqual(ledger[0]["anti_tuning_decision"]["pure_knob_streak"], 1)
            self.assertEqual(ledger[1]["decision"]["outcome"], "diagnostic_improvement_record")
            self.assertEqual(ledger[2]["decision"]["outcome"], "policy_improvement_accept")
            self.assertIn("curriculum", ledger[2])
            self.assertIn("train", ledger[2]["jobs"])
            self.assertIn("eval", ledger[2]["jobs"])
            self.assertEqual(ledger[2]["runtime_activation_gate"]["status"], "pass")
            self.assertEqual(ledger[2]["mechanism_attribution"]["status"], "not_attributed")
            self.assertEqual(ledger[2]["mechanism_attribution"]["reason"], "shadow_ablation_not_run")
            self.assertEqual(
                [record["public_context"]["previous_rounds"] for record in ledger],
                [0, 1, 2],
            )
            self.assertGreater(ledger[2]["runtime_activation_summary"]["reward_term_calls"], 0)
            self.assertTrue(ledger[2]["ablation_summary"]["triggered"])
            self.assertEqual(next_context["previous_rounds"], 3)
            self.assertEqual(next_context["best_public_score"], 0.25)
            self.assertNotIn("private_seed", str(next_context))

    def test_legacy_agent_loop_rejects_authoring_capable_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_fake_session_genesis(
                output_dir=output_dir,
                session_id="session",
            )
            with self.assertRaisesRegex(TypeError, "replay-only"):
                AgentLoop(
                    LoopInputs(
                        session_id="session",
                        round_count=1,
                        task_plugin=FakeTaskPlugin(),
                        backend_adapter=FakeBackendAdapter(),
                        runner=FakeRunner(),
                        candidate_admission_policy=FakeVerifier(),
                        provider=_AuthoringCapableProvider(),
                        policy_initialization=PolicyInitialization.random(),
                        ledger_path=output_dir / "ledger.jsonl",
                        session_genesis=genesis,
                        policy_accept_delta=0.05,
                    )
                )

    def test_replay_loop_refuses_invalid_declared_accept_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            task = FakeTaskPlugin()
            backend = FakeBackendAdapter()
            runner = FakeRunner()
            genesis = prepare_fake_session_genesis(
                output_dir=ledger_path.parent,
                session_id="session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
            )
            first = run_candidate_round(
                RoundInputs(
                    round_id="round-valid",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    candidate=self._candidate(
                        "candidate-valid", "reward_callable", 0.25
                    ),
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    current_best_eval_stats=genesis.initial_current_best,
                    session_genesis_hash=genesis.session_genesis_hash,
                    evaluation_binding=genesis.evaluation_binding,
                )
            )
            self.assertTrue(first.decision.accepted)
            forged = copy.deepcopy(load_ledger(ledger_path)[0])
            forged["round_id"] = "round-forged-bootstrap-accept"
            append_ledger_record(ledger_path, forged)
            provider = _NeverCalledReplayProvider()
            loop = AgentLoop(
                LoopInputs(
                    session_id="session",
                    round_count=1,
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    session_genesis=genesis,
                    policy_accept_delta=0.05,
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "round-forged-bootstrap-accept",
            ):
                loop.run()

            self.assertFalse(provider.called)
            self.assertEqual(len(load_ledger(ledger_path)), 2)

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
                result = super().run_train(train_spec)
                return replace(
                    result,
                    summary={
                        **dict(result.summary),
                        "formal_training_executed": False,
                        "runtime_activation_summary": {},
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-activation-missing",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=NoActivationRunner(),
                    candidate_admission_policy=FakeVerifier(),
                    candidate=self._candidate("activation-missing", "reward_callable", 0.95),
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True, "private_seed": 999},
                    **fake_round_session_fields(),
                )
            )

            ledger = load_ledger(ledger_path)

            self.assertFalse(result.decision.accepted)
            self.assertEqual(result.decision.reason, "runner_activation_missing")
            self.assertEqual(ledger[0]["runtime_activation_gate"]["status"], "attribution_unclear")
            self.assertEqual(ledger[0]["mechanism_attribution"]["status"], "not_attributed")
            self.assertNotIn("private_seed", ledger[0]["public_context"])


if __name__ == "__main__":
    unittest.main()

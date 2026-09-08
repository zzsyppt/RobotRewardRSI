from __future__ import annotations

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
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_round_session_fields,
)
from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.ledger import NON_PROVIDER_AUTHORING_MODE, load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round, _private_artifacts_for_round
from true_rl_agent.runner.checkpoint import file_sha256
from types import SimpleNamespace
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import TrainJobSpec


class RoundPublicBoundariesTests(unittest.TestCase):
    def test_private_outputs_only_include_produced_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_summary = root / "train.json"
            train_summary.write_text('{"status":"failed"}', encoding="utf-8")
            jobs = SimpleNamespace(
                train_spec=SimpleNamespace(metadata={
                    "work_dir": str(root),
                    "summary_path": str(train_summary),
                    "checkpoint_output_path": str(root / "best.pt"),
                }),
                eval_spec=SimpleNamespace(metadata={"summary_path": str(root / "eval.json")}),
            )
            refs = _private_artifacts_for_round(
                candidate=make_test_candidate_package("failed-without-checkpoint"),
                jobs=jobs, train_result=None, executed_eval_spec=None,
            )
            outputs = [ref for ref in refs if ref.kind.startswith("runner_")]
            self.assertEqual([ref.kind for ref in outputs], ["runner_train_summary"])
            self.assertEqual(outputs[0].checksum, file_sha256(train_summary))
            train_summary.unlink()
            refs = _private_artifacts_for_round(
                candidate=make_test_candidate_package("failed-without-checkpoint"),
                jobs=jobs, train_result=None, executed_eval_spec=None,
            )
            self.assertFalse(any(ref.kind.startswith("runner_") for ref in refs))

    def test_public_context_sanitizer(self) -> None:
        sanitized = sanitize_public_context(
            {
                "public_metric": 0.5,
                "private_seed": 123,
                "nested": {"api_key": "sk-secret", "ok": True},
                "note": "sk-secret-value",
            }
        )

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
                    candidate=make_test_candidate_package(
                        "activation-missing",
                        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                        metadata={"fake_public_score": 0.95},
                    ),
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={
                        "previous_rounds": 0,
                        "public_only": True,
                        "private_seed": 999,
                    },
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

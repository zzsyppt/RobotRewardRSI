from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop  # noqa: E402
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError  # noqa: E402
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef  # noqa: E402
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402

from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier  # noqa: E402


class _ProviderErrorThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            raise ProviderCandidateError(
                "json_parse_error",
                "bad JSON at D:/humanoid_agent/private/raw.txt",
                details={"raw_excerpt": "sk-live-secret D:/humanoid_agent/private/raw.txt"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path="memory://candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30},
        )


class _VerifierFailThenSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def failed_static_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path="memory://candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
        )


class _VerifierFailThenProviderFailureProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def failed_static_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        raise ProviderCandidateError(
            "schema_error",
            "candidate schema failed at D:/humanoid_agent/private/raw.txt",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class _AlwaysProviderFailureProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        raise ProviderCandidateError(
            "schema_error",
            "candidate schema failed at /data/yuyangliu/secrets/key.env",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class CandidateAuthoringLoopTests(unittest.TestCase):
    def test_provider_parse_failure_repairs_with_static_frame_and_public_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _ProviderErrorThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(task_prompt_adapter=FakeTaskPlugin().prompt_adapter()),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["provider_status"], "fail")
        self.assertEqual(result.attempt_records[1]["provider_status"], "pass")
        self.assertEqual(provider.payloads[0]["authoring_context_frame"]["frame_kind"], "candidate_draft")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        self.assertEqual(repair_feedback["failure_category"], "json_parse_error")
        self.assertNotIn("D:/humanoid_agent", repair_text)
        self.assertNotIn("sk-live-secret", repair_text)

    def test_verifier_failure_repair_uses_latest_failed_source_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenSuccessProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(task_prompt_adapter=FakeTaskPlugin().prompt_adapter()),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        failed_sources = [
            item
            for item in provider.payloads[1]["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        self.assertEqual(feedback["failure_category"], "verifier_error")
        self.assertEqual(feedback["previous_candidate_id"], "candidate-001")
        self.assertEqual(len(failed_sources), 1)
        self.assertIn("failed_static_reward", failed_sources[0]["content"])

    def test_verifier_failure_then_provider_failure_keeps_attempt_records_public_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenProviderFailureProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(task_prompt_adapter=FakeTaskPlugin().prompt_adapter()),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        attempt_text = json.dumps(result.attempt_records, sort_keys=True)
        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertIsNone(result.candidate)
        self.assertEqual(result.provider_failure["category"], "schema_error")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["candidate_id"], "candidate-001")
        self.assertEqual(result.attempt_records[0]["verifier_status"], "fail")
        self.assertNotIn("candidate_path", result.attempt_records[0])
        self.assertEqual(result.attempt_records[0]["candidate_source_visibility"], "private_non_prompt_artifact")
        self.assertFalse(result.attempt_records[0]["candidate_source_included_in_prompt"])
        self.assertNotIn(str(root / "candidates"), attempt_text)
        self.assertNotIn("D:/humanoid_agent", attempt_text)
        self.assertNotIn("/data/yuyangliu", attempt_text)
        self.assertNotIn("sk-live-secret", attempt_text)

    def test_all_provider_attempts_fail_returns_public_safe_failure_without_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _AlwaysProviderFailureProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(task_prompt_adapter=FakeTaskPlugin().prompt_adapter()),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        result_text = json.dumps(result.provider_failure, sort_keys=True)
        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertIsNone(result.candidate)
        self.assertEqual(result.provider_failure["category"], "schema_error")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(len(result.attempt_prompt_payload_paths), 2)
        self.assertTrue(str(result.prompt_payload_path).endswith("prompt_payload_attempt_002.json"))
        self.assertNotIn("/data/yuyangliu", result_text)
        self.assertNotIn("sk-live-secret", result_text)


if __name__ == "__main__":
    unittest.main()

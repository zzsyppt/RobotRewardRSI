"""Offline boundary regressions with a task that has no robotics task semantics."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.openai_compatible_provider import (  # noqa: E402
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
)
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402
from true_rl_agent.tasks.prompt_adapter import PromptPublicHistoryProjection  # noqa: E402


TASK_ID = "fixture_orbit_tracking"
ROUND_ID = "round-orbit-01"
CANDIDATE_ID = "candidate-orbit-01"
CANARIES = ("PRIVATE_TRACE_CANARY", "ACCEPT_ONLY_CANARY", "SEALED_ONLY_CANARY")


def _record(radial_error=None):
    # Existing generic runner-decision fixture; the additional metric belongs
    # solely to this test task and has no shared interpretation.
    record = valid_accepted_ledger_record(
        round_id=ROUND_ID, candidate_id=CANDIDATE_ID, public_score=0.25
    )
    record["task"] = {"task_id": TASK_ID}
    metrics = record["public_result"]["metrics"]
    metrics["public_dev"]["score_stats"]["mean"] = 0.75
    metrics["public_dev"]["orbit_observations"] = {
        "expected_count": 6,
        "observed_count": 4,
        "mean_radial_error": radial_error,
        "evidence_refs": ["orbit-public-dev-evidence-01"],
        "private_trace": CANARIES[0],
    }
    metrics["public_accept"]["internal_note"] = CANARIES[1]
    # Do not alter the strict sealed-validation receipt used by the runner gate.
    metrics["sealed_rollout_details"] = {"internal_note": CANARIES[2]}
    return record


class _OrbitPromptAdapter:
    task_id = TASK_ID

    def project_prompt_public_history(
        self, *, public_records, initial_anchor_public_evidence
    ):
        # Deliberately edit the supplied dictionaries in place. The production
        # builder owns isolation; adapters need not manufacture another copy.
        for record in public_records:
            result = record["public_result"]
            dev = result["metrics"]["public_dev"]
            observations = dev["orbit_observations"]
            result["metrics"] = {
                "public_dev": {
                    "score_stats": dev["score_stats"],
                    "orbit_observations": {
                        key: observations[key]
                        for key in (
                            "expected_count",
                            "observed_count",
                            "mean_radial_error",
                            "evidence_refs",
                        )
                    },
                }
            }
            score = dev["score_stats"]["mean"]
            result["public_score"] = score
            record["decision"]["public_score"] = score
            record["decision"]["comparison"] = {
                "mode": "provider_feedback_absolute",
                "score_source": "task.public_dev.orbit_score",
                "candidate_score": score,
            }
        return PromptPublicHistoryProjection(
            records=tuple(public_records),
            public_score_source="task.public_dev.orbit_score",
        )

    def prompt_public_record_scrub_policy(self):
        return {"blocked_key_parts": (), "blocked_text_parts": ()}

    def build_prompt_sections(self, *, public_records, public_context):
        return {
            "task": {"task_id": self.task_id, "name": "Orbit tracking fixture"},
            "public_feedback_summary": {
                "task_id": self.task_id,
                "memory": public_context["task_public_summary"],
            },
        }


class _OrbitMemoryAdapter:
    task_id = TASK_ID

    def __init__(self):
        self.seen_records = []

    def build_public_summary(self, public_records):
        self.seen_records = copy.deepcopy(list(public_records))
        record = public_records[-1]
        return {
            "task_id": self.task_id,
            "round_id": record["round_id"],
            "candidate_id": record["candidate"]["candidate_id"],
            "observations": copy.deepcopy(
                record["public_result"]["metrics"]["public_dev"]["orbit_observations"]
            ),
        }


class _ObservedBuilder(PublicContextBuilder):
    """Observe the real validated history without replacing replay or projection."""

    def _project_public_history(self, experiment_records):
        history = super()._project_public_history(experiment_records)
        self.validated_history = history
        self.before_task_projection = copy.deepcopy(history.records)
        return history


def _builder(prompt_adapter=None):
    return _ObservedBuilder(
        task_prompt_adapter=prompt_adapter or _OrbitPromptAdapter(),
        task_memory_adapter=_OrbitMemoryAdapter(),
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        initial_current_best=make_bootstrap_current_best(),
        session_genesis_hash=TEST_SESSION_GENESIS_HASH,
    )


class FeedbackEvidenceBoundaryTests(unittest.TestCase):
    def test_third_task_evidence_survives_memory_and_offline_provider_request(self):
        outputs = []
        for radial_error in (None, 0.0):
            with self.subTest(radial_error=radial_error):
                record = _record(radial_error)
                original = copy.deepcopy(record)
                builder = _builder()
                built = builder.build_prompt([record])
                with tempfile.TemporaryDirectory() as tmp:
                    provider = OpenAICompatibleCandidateProvider(
                        OpenAICompatibleProviderConfig(
                            model="offline-fixture", output_dir=Path(tmp)
                        )
                    )
                    # Request construction only: no credential or transport call.
                    request = provider._chat_completion_request(built.provider_payload)
                    wire = json.dumps(request, allow_nan=False)
                    delivered = json.loads(request["messages"][1]["content"])
                outputs.append(delivered)

                memory = delivered["public_feedback_summary"]["memory"]
                self.assertEqual(delivered["task"]["task_id"], TASK_ID)
                self.assertEqual(memory["round_id"], ROUND_ID)
                self.assertEqual(memory["candidate_id"], CANDIDATE_ID)
                self.assertEqual(
                    memory["observations"],
                    {
                        "expected_count": 6,
                        "observed_count": 4,
                        "mean_radial_error": radial_error,
                        "evidence_refs": ["orbit-public-dev-evidence-01"],
                    },
                )
                self.assertEqual(
                    memory, built.round_public_context["task_public_summary"]
                )
                self.assertEqual(delivered, built.provider_payload)
                self.assertEqual(delivered["public_history"]["best_public_score"], 0.75)
                self.assertEqual(
                    delivered["public_history"]["public_score_source"],
                    "task.public_dev.orbit_score",
                )
                memory_record = builder.task_memory_adapter.seen_records[0]
                self.assertTrue(memory_record["decision"]["accepted"])
                self.assertEqual(memory_record["decision"]["validation_status"], "valid")
                self.assertEqual(record, original)
                self.assertEqual(
                    builder.validated_history.records, builder.before_task_projection
                )
                validated_text = json.dumps(builder.before_task_projection)
                # Generic privacy filtering acts before the task adapter; the
                # accept/sealed data then need the task's public-dev allowlist.
                self.assertNotIn(CANARIES[0], validated_text)
                for canary in CANARIES[1:]:
                    self.assertIn(canary, validated_text)
                for canary in CANARIES:
                    self.assertIn(canary, json.dumps(original))
                    self.assertNotIn(canary, wire)
                    self.assertNotIn(canary, json.dumps(memory_record))

        self.assertIsNone(
            outputs[0]["public_feedback_summary"]["memory"]["observations"]["mean_radial_error"]
        )
        self.assertIsInstance(
            outputs[1]["public_feedback_summary"]["memory"]["observations"]["mean_radial_error"],
            float,
        )

    def test_feedback_only_in_place_projection_is_isolated_and_repeatable(self):
        record = _record()
        original = copy.deepcopy(record)
        builder = _builder()
        first = builder.build_prompt([record])
        authority = copy.deepcopy(builder.validated_history.records)
        second = builder.build_prompt([record])

        self.assertEqual(first, second)
        self.assertEqual(record, original)
        self.assertEqual(builder.validated_history.records, authority)
        self.assertEqual(authority[0]["public_result"]["public_score"], 0.25)
        self.assertEqual(first.provider_payload["public_history"]["best_public_score"], 0.75)
        self.assertIn("public_accept", authority[0]["public_result"]["metrics"])
        self.assertNotIn(
            "public_accept",
            builder.task_memory_adapter.seen_records[0]["public_result"]["metrics"],
        )

    def test_in_place_authority_edits_are_rejected_without_mutating_validated_history(self):
        for field in ("candidate", "decision"):
            with self.subTest(field=field):
                class CorruptingAdapter(_OrbitPromptAdapter):
                    def project_prompt_public_history(
                        self, *, public_records, initial_anchor_public_evidence
                    ):
                        if field == "candidate":
                            public_records[0]["candidate"]["candidate_id"] = "wrong-candidate"
                        else:
                            public_records[0]["decision"]["accepted"] = False
                        return PromptPublicHistoryProjection(records=tuple(public_records))

                record = _record()
                original = copy.deepcopy(record)
                builder = _builder(CorruptingAdapter())
                with self.assertRaisesRegex(ValueError, "changed runner authority"):
                    builder.build_prompt([record])
                self.assertEqual(record, original)
                self.assertEqual(
                    builder.validated_history.records, builder.before_task_projection
                )
                self.assertEqual(builder.task_memory_adapter.seen_records, [])


if __name__ == "__main__":
    unittest.main()

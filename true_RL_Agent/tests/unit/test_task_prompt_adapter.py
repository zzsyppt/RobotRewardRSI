from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import TEST_CAPABILITY_PROFILE
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.memory import (
    ExperimentRecord,
    PublicContextBuilder,
    TaskMemoryAdapter,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.prompt_adapter import (
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    NeutralTaskPromptAdapter,
    sanitize_task_prompt_sections,
)


class NonPushTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="stair_climbing",
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={"random": "bootstrap"},
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec({"episode_length": 400}),
        )


class TaskPromptAdapterTests(unittest.TestCase):
    def test_neutral_prompt_adapter_carries_task_memory_in_canonical_feedback_only(self) -> None:
        class CustomTaskMemoryAdapter(TaskMemoryAdapter):
            task_id = "stair_climbing"

            def build_public_summary(self, public_records):
                return {
                    "schema_version": "custom.task_memory.v1",
                    "opaque_public_metric": len(public_records) + 7,
                }

        payload = PublicContextBuilder(
            task_memory_adapter=CustomTaskMemoryAdapter(),
            task_prompt_adapter=NonPushTaskPlugin().prompt_adapter(),
        ).build_prompt_payload([])

        self.assertEqual(
            payload["public_feedback_summary"]["task_public_summary"],
            {
                "schema_version": "custom.task_memory.v1",
                "opaque_public_metric": 7,
            },
        )
        self.assertNotIn("public_context", payload)

    def test_task_plugin_exposes_neutral_prompt_adapter_without_push_semantics(self) -> None:
        adapter = NonPushTaskPlugin().prompt_adapter()

        self.assertIsInstance(adapter, NeutralTaskPromptAdapter)
        sections = adapter.build_prompt_sections(public_records=[], public_context={"previous_rounds": 0})
        text = json.dumps(sections, sort_keys=True)

        self.assertEqual(sections["task"]["task_id"], "stair_climbing")
        self.assertEqual(sections["task"]["prompt_adapter"], "neutral_fallback")
        self.assertEqual(sections["reward_inputs_contract"]["status"], "not_available")
        self.assertEqual(sections["public_eval_contract"]["status"], "not_available")
        self.assertEqual(sections["known_good_examples"], [])
        self.assertIn("candidate package", " ".join(sections["selection_guidance"]).lower())
        self.assertNotIn("push_active", text)
        self.assertNotIn("push_vec_public", text)
        self.assertNotIn("push recovery", text.lower())
        self.assertNotIn("curriculum", text.lower())
        assert_public_only_prompt_payload(sections)

    def test_non_push_full_prompt_payload_does_not_expose_push_reward_inputs(self) -> None:
        payload = PublicContextBuilder(task_prompt_adapter=NonPushTaskPlugin().prompt_adapter()).build_prompt_payload([])
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            payload["task"]["task_id"],
            "stair_climbing",
        )
        self.assertNotIn("public_context", payload)
        self.assertNotIn("push_active", text)
        self.assertNotIn("push_vec_public", text)
        self.assertNotIn("push recovery", text.lower())
        assert_public_only_prompt_payload(payload)

    def test_public_context_builder_includes_sanitized_task_prompt_sections(self) -> None:
        class CustomTaskPromptAdapter:
            task_id = "custom_task"

            def build_prompt_sections(self, *, public_records, public_context):
                seen_round_counts.append(len(public_records))
                seen_previous_rounds.append(public_context["previous_rounds"])
                return {
                    "task": {
                        "schema_version": "custom.task_prompt.v1",
                        "task_id": "custom_task",
                        "name": "Custom Task",
                        "private_seed": 123,
                    },
                    "reward_inputs_contract": {
                        "schema_version": "custom.reward_inputs.v1",
                        "fields": [{"name": "height", "shape": "[num_envs]"}],
                        "note": "sk-secret-value",
                    },
                    "public_eval_contract": {"status": "custom_public_only"},
                    "public_feedback_summary": {"previous_rounds": public_context["previous_rounds"]},
                    "runtime_feedback_summary": {"status": "not_observed_yet"},
                    "known_good_examples": [],
                    "selection_guidance": ["Use custom public fields only."],
                }

        seen_round_counts: list[int] = []
        seen_previous_rounds: list[int] = []
        records = [
            ExperimentRecord.from_ledger_record(
                {
                    "round_id": "round-001",
                    "task": {"task_id": "custom_task"},
                    "decision": {"accepted": False, "outcome": "reject", "public_score": None},
                    "candidate_surface": {"classification": "code_level"},
                }
            )
        ]

        payload = PublicContextBuilder(task_prompt_adapter=CustomTaskPromptAdapter()).build_prompt_payload(records)
        sections = payload
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(sections["task"]["task_id"], "custom_task")
        self.assertEqual(sections["reward_inputs_contract"]["note"], "[redacted]")
        self.assertNotIn("public_context", payload)
        self.assertNotIn("private_seed", text)
        self.assertNotIn("sk-secret-value", text)
        self.assertEqual(seen_round_counts, [1])
        self.assertEqual(seen_previous_rounds, [1])
        assert_public_only_prompt_payload(payload)

    def test_task_prompt_sections_reject_curriculum_interfaces(self) -> None:
        with self.assertRaisesRegex(ValueError, "curriculum"):
            sanitize_task_prompt_sections(
                {
                    "task": {"task_id": "custom_task"},
                    "curriculum_design": {"stage_list": []},
                }
            )

    def test_task_prompt_sections_scrub_absolute_paths(self) -> None:
        sections = sanitize_task_prompt_sections(
            {
                "task": {"task_id": "custom_task"},
                "selection_guidance": [
                    "Do not inspect D:/humanoid_agent/private/raw.txt or /data/yuyangliu/secrets/key.env",
                ],
            }
        )

        text = json.dumps(sections, sort_keys=True)
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertIn("[path]", text)

    def test_task_prompt_sections_reject_context_explosion(self) -> None:
        oversized = {
            "task": {"task_id": "custom_task"},
            "selection_guidance": ["x" * (MAX_TASK_PROMPT_SECTIONS_BYTES + 1)],
        }

        with self.assertRaisesRegex(ValueError, "exceed"):
            sanitize_task_prompt_sections(oversized)


if __name__ == "__main__":
    unittest.main()

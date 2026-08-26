from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.prompting.capacity import (  # noqa: E402
    build_prompt_capacity_receipt,
    verify_prompt_capacity_receipt,
)
from true_rl_agent.memory.context_builder import (  # noqa: E402
    preflight_task_prompt_capacity,
)
from true_rl_agent.prompting.payload import build_prompt_payload  # noqa: E402
from true_rl_agent.prompting.serialization import (  # noqa: E402
    canonical_prompt_json_bytes,
)
from true_rl_agent.prompting.task_sections import (  # noqa: E402
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    PROVIDER_TASK_PROMPT_SECTION_KEYS,
    TASK_PROMPT_SECTION_KEYS,
    build_provider_task_prompt_receipt,
    provider_task_prompt_sections,
    sanitize_task_prompt_sections,
    validate_provider_task_prompt_receipt,
)
from true_rl_agent.tasks.push_recovery.prompt_adapter import (  # noqa: E402
    _minimum_core_push_prompt_sections,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (  # noqa: E402
    StairClimbingPromptAdapter,
    _compact_historical_comparison_entry,
    _compact_public_feedback_to_fit_task_sections,
    _minimum_core_stair_prompt_sections,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)


def _base_sections(feedback: dict) -> dict:
    return {
        "task": {"task_id": "test", "objective": "improve"},
        "reward_inputs_contract": {"schema_version": "reward.v1"},
        "public_eval_contract": {"schema_version": "eval.v1"},
        "public_feedback_summary": feedback,
        "runtime_feedback_summary": {"status": "available", "detail": "x"},
        "known_good_examples": [],
        "selection_guidance": ["Use public evidence."],
    }


class PromptCapacityReceiptTests(unittest.TestCase):
    def test_receipt_binds_visible_projection_and_is_deterministic(self) -> None:
        original = _base_sections(
            {"status": "available", "feedback_entries": [{"round_id": "r1"}]}
        )
        projected = _base_sections(
            {"status": "no_closed_outcome", "feedback_entries": []}
        )
        receipt = build_prompt_capacity_receipt(
            original=original,
            projected_content=projected,
            projection_id="test.minimum_core.v1",
            omitted=("runtime_detail", "history", "history"),
        )
        sections = copy.deepcopy(projected)
        sections["public_feedback_summary"]["capacity_projection_receipt"] = receipt

        clean = sanitize_task_prompt_sections(sections)
        self.assertEqual(clean, sections)
        self.assertEqual(receipt["omitted"], ["history", "runtime_detail"])
        self.assertEqual(
            verify_prompt_capacity_receipt(
                receipt,
                original=original,
                projected_content=projected,
            ),
            receipt,
        )
        self.assertEqual(
            build_prompt_capacity_receipt(
                original=original,
                projected_content=projected,
                projection_id="test.minimum_core.v1",
                omitted=("runtime_detail", "history"),
            ),
            receipt,
        )

    def test_receipt_or_visible_content_tamper_is_rejected(self) -> None:
        original = _base_sections({"status": "available"})
        projected = _base_sections({"status": "no_closed_outcome"})
        receipt = build_prompt_capacity_receipt(
            original=original,
            projected_content=projected,
            projection_id="test.minimum_core.v1",
            omitted=("history",),
        )
        sections = copy.deepcopy(projected)
        sections["public_feedback_summary"]["capacity_projection_receipt"] = receipt

        changed_content = copy.deepcopy(sections)
        changed_content["selection_guidance"] = ["forged"]
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            sanitize_task_prompt_sections(changed_content)

        changed_receipt = copy.deepcopy(sections)
        changed_receipt["public_feedback_summary"][
            "capacity_projection_receipt"
        ]["projected_content_without_receipt_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "content size mismatch"):
            sanitize_task_prompt_sections(changed_receipt)

        extra_field = copy.deepcopy(sections)
        extra_field["public_feedback_summary"][
            "capacity_projection_receipt"
        ]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            sanitize_task_prompt_sections(extra_field)


class TaskOwnedMinimumCoreTests(unittest.TestCase):
    def test_stair_bootstrap_minimum_core_has_explicit_sentinel(self) -> None:
        minimum = _minimum_core_stair_prompt_sections(
            _base_sections({"status": "not_available"}),
            target_bytes=MAX_TASK_PROMPT_SECTIONS_BYTES,
        )
        clean = sanitize_task_prompt_sections(minimum)

        self.assertEqual(
            clean["public_feedback_summary"]["status"],
            "no_closed_outcome",
        )
        self.assertEqual(clean["public_feedback_summary"]["feedback_entries"], [])
        self.assertEqual(
            clean["runtime_feedback_summary"]["status"],
            "omitted_for_capacity",
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(clean),
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )

    def test_stair_capacity_pressure_keeps_current_best_and_latest(self) -> None:
        entries = [
            {
                "source_kind": "candidate",
                "candidate_id": "accepted-head",
                "selection_basis": ["accepted_current_best"],
                "feedback": {
                    "schema_version": "unsupported.for.minimum-core-test",
                    "oversized_detail": "x" * 20000,
                },
            },
            {
                "source_kind": "candidate",
                "candidate_id": "latest-candidate",
                "selection_basis": ["latest_evaluated"],
                "feedback": {
                    "schema_version": "unsupported.for.minimum-core-test",
                    "oversized_detail": "y" * 20000,
                },
            },
        ]
        sections = _base_sections(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
                ),
                "status": "available",
                "feedback_entries": entries,
                "latest_generation_trial_comparisons": [
                    {
                        "generation_id": "generation-002",
                        "trial_id": "generation-002-lane-a",
                        "lane_id": "lane-a",
                        "trial_status": "qualified_not_selected",
                        "selection_status": "not_selected",
                        "observed_effect": "mixed",
                        "objective_delta": -0.1,
                        "objective_materiality_threshold": 0.02,
                        "objective_materiality_met": False,
                        "guard_delta": 0.05,
                        "guard_max_regression": 0.01,
                        "guard_met": True,
                        "reason_codes": [
                            "public_guard_improved_objective_not_improved"
                        ],
                    }
                ],
                "omitted_candidate_feedback_entry_count": 0,
            }
        )
        projected = _compact_public_feedback_to_fit_task_sections(
            sections,
            evaluation_profile=None,
        )
        clean = sanitize_task_prompt_sections(projected)
        candidate_ids = [
            entry["candidate_id"]
            for entry in clean["public_feedback_summary"]["feedback_entries"]
        ]

        self.assertEqual(candidate_ids, ["accepted-head", "latest-candidate"])
        self.assertEqual(
            clean["public_feedback_summary"]["projection_level"],
            "minimum_core",
        )
        self.assertNotIn(
            "oversized_detail",
            str(clean["public_feedback_summary"]),
        )
        self.assertEqual(
            clean["public_feedback_summary"][
                "latest_generation_trial_comparisons"
            ][0]["trial_id"],
            "generation-002-lane-a",
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(clean),
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )
        rerun = _compact_public_feedback_to_fit_task_sections(
            sections,
            evaluation_profile=None,
        )
        self.assertEqual(clean, sanitize_task_prompt_sections(rerun))

    def test_stair_nonfeedback_pressure_with_empty_entries_reaches_minimum_core(
        self,
    ) -> None:
        sections = _base_sections(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
                ),
                "status": "not_available",
                "feedback_entries": [],
            }
        )
        sections["runtime_feedback_summary"] = {
            "status": "available",
            "contract_valid_bounded_detail": "x" * 40000,
        }

        projected = _compact_public_feedback_to_fit_task_sections(
            sections,
            evaluation_profile=None,
        )
        clean = sanitize_task_prompt_sections(projected)

        self.assertEqual(
            clean["public_feedback_summary"]["projection_level"],
            "minimum_core",
        )
        self.assertEqual(
            clean["public_feedback_summary"]["status"],
            "no_closed_outcome",
        )
        self.assertNotIn("contract_valid_bounded_detail", str(clean))
        self.assertLessEqual(
            canonical_prompt_json_bytes(clean),
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )

    def test_stair_capacity_projection_normalizes_absent_feedback_entries(
        self,
    ) -> None:
        cases = (
            {"status": "available", "feedback_entries": []},
            {"status": "available", "feedback_entries": "oops"},
            {"status": "available", "feedback_entries": {}},
            {"status": "available", "feedback_entries": [1, 2, 3]},
            {"status": "available"},
            {"status": "available", "feedback_entries": None},
            None,
        )
        for feedback in cases:
            with self.subTest(feedback=feedback):
                sections = _base_sections({"status": "available"})
                if feedback is None:
                    sections.pop("public_feedback_summary")
                else:
                    sections["public_feedback_summary"] = feedback
                sections["runtime_feedback_summary"] = {
                    "status": "available",
                    "contract_valid_bounded_detail": "x" * 40000,
                }

                clean = sanitize_task_prompt_sections(
                    _compact_public_feedback_to_fit_task_sections(
                        sections,
                        evaluation_profile=None,
                    )
                )

                self.assertEqual(
                    clean["public_feedback_summary"]["projection_level"],
                    "minimum_core",
                )
                self.assertEqual(
                    clean["public_feedback_summary"]["feedback_entries"],
                    [],
                )

    def test_provider_section_keys_cover_all_but_task_only_metadata(self) -> None:
        self.assertEqual(
            TASK_PROMPT_SECTION_KEYS
            - frozenset(PROVIDER_TASK_PROMPT_SECTION_KEYS),
            {"task_capability_metadata"},
        )
        self.assertLessEqual(
            frozenset(PROVIDER_TASK_PROMPT_SECTION_KEYS),
            TASK_PROMPT_SECTION_KEYS,
        )

    def test_historical_compaction_normalizes_absent_detail_omissions(
        self,
    ) -> None:
        compacted = _compact_historical_comparison_entry(
            {
                "candidate_id": "historical",
                "feedback": {},
                "detail_omissions": None,
            }
        )
        self.assertIsInstance(compacted["detail_omissions"], list)

    def test_provider_task_receipt_measures_the_exact_shipped_sections(self) -> None:
        task_sections = sanitize_task_prompt_sections(
            _base_sections({"status": "no_closed_outcome"})
        )
        payload = build_prompt_payload(
            public_context={"task_prompt_sections": task_sections},
        )
        receipt = payload["authoring_context_frame"]["budget_metadata"][
            "provider_task_sections"
        ]

        self.assertEqual(provider_task_prompt_sections(payload), task_sections)
        self.assertIn("evidence_discipline", payload["loop_contract"])
        self.assertEqual(
            receipt,
            build_provider_task_prompt_receipt(payload),
        )
        self.assertEqual(
            validate_provider_task_prompt_receipt(
                receipt,
                prompt_payload=payload,
            ),
            receipt,
        )
        self.assertLessEqual(
            receipt["serialized_bytes"],
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )

    def test_stair_worst_case_preflight_is_deterministic_and_keeps_headroom(
        self,
    ) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        adapter = StairClimbingPromptAdapter(config)

        first = preflight_task_prompt_capacity(
            task_prompt_adapter=adapter,
            candidate_capability_profile=None,
        )
        second = preflight_task_prompt_capacity(
            task_prompt_adapter=adapter,
            candidate_capability_profile=None,
        )

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["headroom_bytes"], 512)

    def test_push_uses_its_own_minimum_core_without_stair_fields(self) -> None:
        original = _base_sections(
            {
                "schema_version": "push.feedback.v1",
                "status": "available",
                "round_count": 3,
                "latest_round": {"round_id": "r3", "status": "fail"},
                "large_task_detail": "x" * 40000,
            }
        )
        clean = sanitize_task_prompt_sections(
            _minimum_core_push_prompt_sections(original)
        )

        self.assertEqual(clean["public_feedback_summary"]["round_count"], 3)
        self.assertEqual(
            clean["public_feedback_summary"]["projection_level"],
            "minimum_core",
        )
        self.assertNotIn("large_task_detail", clean["public_feedback_summary"])
        self.assertNotIn("stable_success_summary", str(clean))
        self.assertLessEqual(
            canonical_prompt_json_bytes(clean),
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )

    def test_unknown_common_section_remains_fail_closed(self) -> None:
        sections = _base_sections({"status": "not_available"})
        sections["unexpected_section"] = {"value": 1}
        with self.assertRaisesRegex(ValueError, "unknown task prompt section"):
            sanitize_task_prompt_sections(sections)


if __name__ == "__main__":
    unittest.main()

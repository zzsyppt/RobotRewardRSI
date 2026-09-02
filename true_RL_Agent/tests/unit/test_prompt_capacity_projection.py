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
    _minimum_core_selection_guidance,
    _minimum_core_stair_prompt_sections,
    _minimum_core_task_section,
    _project_feedback_entry_for_prompt,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    resolve_stair_cell_manifest,
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
    def test_stair_minimum_core_preserves_mode_guidance_and_seed_counts(
        self,
    ) -> None:
        guidance = [f"mode-aware-guidance-{index}" for index in range(13)]
        self.assertEqual(_minimum_core_selection_guidance(guidance), guidance)

        seed_contract = {
            "strata": ["c0_id", "c1_id"],
            "cells_per_stratum": 3,
            "episodes_per_seed_per_cell": 3,
            "public_seed_count": 3,
            "public_accept_seed_count": 5,
            "sealed_validation_seed_count": 7,
            "seed_values_exposed": False,
            "statistics_unit": "cell_then_stratum_then_paired_seed",
            "success_criteria": {},
        }
        projected = _minimum_core_task_section(
            {"fixed_eval_protocol_summary": seed_contract}
        )["fixed_eval_protocol_summary"]
        self.assertEqual(projected["public_seed_count"], 3)
        self.assertEqual(projected["public_accept_seed_count"], 5)
        self.assertEqual(projected["sealed_validation_seed_count"], 7)
        self.assertFalse(projected["seed_values_exposed"])

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
                    "oversized_detail": "x" * 30000,
                },
            },
            {
                "source_kind": "candidate",
                "candidate_id": "latest-candidate",
                "selection_basis": ["latest_evaluated"],
                "feedback": {
                    "schema_version": "unsupported.for.minimum-core-test",
                    "oversized_detail": "y" * 30000,
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
            "contract_valid_bounded_detail": "x" * 70000,
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

    def test_stair_compacts_static_contract_before_measured_feedback(self) -> None:
        feedback = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_summary.v12"
            ),
            "status": "available",
            "feedback_entries": [
                {
                    "source_kind": "candidate",
                    "candidate_id": "latest",
                    "selection_basis": ["latest_evaluated"],
                    "feedback": {"stratum_diagnostics": "d" * 5000},
                }
            ],
        }
        sections = _base_sections(feedback)
        sections["reward_inputs_contract"]["fields"] = [
            {
                "name": f"field_{index}",
                "dtype": "float",
                "trailing_shape": [],
                "unit": "unitless",
                "description": "s" * 500,
            }
            for index in range(80)
        ]
        sections["runtime_feedback_summary"] = {
            "status": "available",
            "runtime_activation_detail": "r" * 5000,
        }
        self.assertGreater(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES - 512,
        )

        clean = sanitize_task_prompt_sections(
            _compact_public_feedback_to_fit_task_sections(
                sections,
                evaluation_profile=None,
            )
        )

        receipt = clean["public_feedback_summary"][
            "capacity_projection_receipt"
        ]
        self.assertEqual(
            receipt["projection_id"],
            "stair_climbing.static_contract_core.v1",
        )
        self.assertEqual(
            clean["public_feedback_summary"]["feedback_entries"][0]["feedback"][
                "stratum_diagnostics"
            ],
            "d" * 5000,
        )
        self.assertEqual(
            clean["runtime_feedback_summary"]["runtime_activation_detail"],
            "r" * 5000,
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(clean),
            MAX_TASK_PROMPT_SECTIONS_BYTES - 512,
        )

    def test_stair_minimum_core_retains_compacted_public_dev_feedback(self) -> None:
        sections = _base_sections(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.public_feedback_summary.v12"
                ),
                "status": "available",
                "feedback_entries": [
                    {
                        "source_kind": "reference_anchor",
                        "source_public_result_hash": "a" * 64,
                        "objective_score": 0.78,
                        "selection_basis": ["reference_current_best"],
                        "feedback": {
                            "schema_version": (
                                "true_rl_agent.task.stair_climbing.prompt_feedback.v9"
                            ),
                            "source_split": "public_dev",
                            "stable_success_summary": {
                                "stable_top_success_rate": 0.82
                            },
                            "dominant_failure": {
                                "reason": "forbidden_contact",
                                "rate": 0.16,
                            },
                            "generalization_summary": {
                                "public_dev_objective_score": 0.74,
                                "j_id": 0.98,
                                "j_ood": 0.62,
                                "named_guard_scores": {
                                    "c0_id": 0.97,
                                    "c1_id": 0.99,
                                    "j_ood": 0.62,
                                },
                                "family_scores": {
                                    "F_scalar": 0.62,
                                    "F_c1": 1.0,
                                    "F_c2": 1.0,
                                },
                                "qualified_ood_families": ["F_scalar"],
                                "objective_family_mask_hash": "b" * 64,
                            },
                            "progress_summary": {
                                "mean_progress_fraction": 0.89,
                                "mean_progress_fraction_by_cell": {
                                    "public_dev.height_ood_01": 0.65
                                },
                            },
                        },
                    }
                ],
            }
        )
        latest = copy.deepcopy(
            sections["public_feedback_summary"]["feedback_entries"][0]
        )
        latest.update(
            {
                "source_kind": "candidate",
                "source_public_result_hash": "c" * 64,
                "candidate_id": "latest-candidate",
                "selection_basis": ["latest_evaluated"],
            }
        )
        sections["public_feedback_summary"]["feedback_entries"].append(latest)
        # Force the full two-stage path: required-entry compaction first, then
        # task-owned minimum-core projection because non-feedback detail still
        # exceeds the reserved task-section budget.
        sections["runtime_feedback_summary"] = {
            "status": "available",
            "contract_valid_bounded_detail": "x" * 70000,
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
        entries = clean["public_feedback_summary"]["feedback_entries"]
        self.assertEqual(len(entries), 2)
        feedback = entries[0][
            "feedback"
        ]
        self.assertEqual(
            feedback["dominant_failure"],
            {"reason": "forbidden_contact", "rate": 0.16},
        )
        self.assertEqual(
            feedback["generalization_summary"]["public_dev_objective_score"],
            0.74,
        )
        self.assertEqual(
            feedback["generalization_summary"]["qualified_ood_families"],
            ["F_scalar"],
        )
        self.assertEqual(
            feedback["generalization_summary"]["objective_family_mask_hash"],
            "b" * 64,
        )
        self.assertEqual(
            feedback["generalization_summary"]["named_guard_scores"],
            {"c0_id": 0.97, "c1_id": 0.99, "j_ood": 0.62},
        )
        self.assertEqual(
            feedback["generalization_summary"]["family_scores"],
            {"F_scalar": 0.62, "F_c1": 1.0, "F_c2": 1.0},
        )
        self.assertEqual(
            feedback["progress_summary"]["mean_progress_fraction_by_cell"],
            {"public_dev.height_ood_01": 0.65},
        )
        latest_feedback = entries[1]["feedback"]
        self.assertEqual(
            latest_feedback["generalization_summary"],
            {"public_dev_objective_score": 0.74},
        )
        self.assertEqual(
            latest_feedback["progress_summary"],
            {"mean_progress_fraction": 0.89},
        )
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
                    "contract_valid_bounded_detail": "x" * 70000,
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

    def test_provider_task_receipt_keeps_v1_32k_history_replayable(self) -> None:
        task_sections = sanitize_task_prompt_sections(
            _base_sections({"status": "no_closed_outcome"})
        )
        payload = build_prompt_payload(
            public_context={"task_prompt_sections": task_sections},
        )
        current = build_provider_task_prompt_receipt(payload)
        legacy = {
            **current,
            "schema_version": "true_rl_agent.prompt.provider_task_sections_receipt.v1",
            "max_serialized_bytes": 32 * 1024,
            "headroom_bytes": 32 * 1024 - current["serialized_bytes"],
        }

        self.assertEqual(
            validate_provider_task_prompt_receipt(
                legacy,
                prompt_payload=payload,
            ),
            legacy,
        )

    def test_stair_worst_case_preflight_is_deterministic_and_keeps_headroom(
        self,
    ) -> None:
        for protocol_name in (
            "stair_unified_protocol_v2.json",
            "stair_go2_12dof_unified_protocol_v2.json",
        ):
            with self.subTest(protocol=protocol_name):
                protocol_path = ROOT / "configs" / "tasks" / protocol_name
                base = StairClimbingTaskConfig.defaults(protocol_path)
                manifest = resolve_stair_cell_manifest(
                    cell_variant_id=base.cell_variant_id,
                    config=base,
                    plugin=StairClimbingTaskPlugin(base),
                )
                config = StairClimbingTaskConfig.defaults(
                    protocol_path,
                    resolved_cell_task_protocol_hash=(
                        manifest.resolved_cell_task_protocol_hash
                    ),
                )
                adapter = StairClimbingPromptAdapter(
                    config,
                    resolved_cell_manifest=manifest,
                )

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
                sections = adapter.build_prompt_capacity_preflight_sections()
                feedback = sections["public_feedback_summary"]
                self.assertEqual(
                    feedback["projection_level"],
                    "minimum_core",
                )
                self.assertEqual(len(feedback["feedback_entries"]), 2)
                self.assertEqual(
                    [entry["selection_basis"] for entry in feedback["feedback_entries"]],
                    [["accepted_current_best"], ["latest_evaluated"]],
                )
                self.assertNotIn("feedback_entries_omitted_for_capacity", feedback)
                self.assertIn("capacity_projection_receipt", feedback)
                self.assertEqual(
                    sections["public_eval_contract"][
                        "completion_derived_diagnostic_rates"
                    ],
                    {
                        "first_valid_top_arrival_achieved": "success_rate",
                    },
                )

    def test_first_arrival_prompt_omits_only_exact_completion_diagnostics(
        self,
    ) -> None:
        raw = {
            "feedback": {
                "climb_completion_summary": {
                    "completion_mode": "first_valid_top_arrival_latched",
                    "success_rate": 0.75,
                    "first_valid_top_arrival_rate": 0.75,
                },
                "diagnostic_flag_rates": {
                    "first_valid_top_arrival_achieved": 0.75,
                    "goal_not_reached_detected": 0.1,
                    "slip_detected": 0.5,
                },
                "generalization_summary": {
                    "stratum_diagnostics": {
                        "height_ood": {
                            "score": 0.7,
                            "success_rate": 0.75,
                            "mean_progress_fraction": 0.8,
                            "primary_failure_breakdown": {"timeout": 0.25},
                            "diagnostic_flag_rates": {
                                "first_valid_top_arrival_achieved": 0.75,
                                "goal_not_reached_detected": 0.1,
                                "slip_detected": 0.5,
                            },
                        }
                    }
                },
            }
        }
        untouched = copy.deepcopy(raw)
        projected = _project_feedback_entry_for_prompt(raw)["feedback"]

        self.assertEqual(raw, untouched)
        self.assertEqual(
            projected["diagnostic_flag_rates"],
            {
                "goal_not_reached_detected": 0.1,
                "slip_detected": 0.5,
            },
        )
        table = projected["generalization_summary"]["stratum_diagnostics"]
        self.assertEqual(
            table["diagnostic_flag_rate_columns"],
            ["goal_not_reached_detected", "slip_detected"],
        )
        self.assertEqual(
            table["derived_diagnostic_rate_columns_omitted"],
            ["first_valid_top_arrival_achieved"],
        )
        self.assertEqual(table["rows"][0][5], [0.1, 0.5])

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

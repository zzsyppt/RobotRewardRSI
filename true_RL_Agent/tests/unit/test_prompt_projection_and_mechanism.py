from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.authoring_response import SourceRequestResponse
from true_rl_agent.agent_core.source_request_limits import (
    MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
)
from true_rl_agent.candidates.authoring_policy import (
    MAX_CANONICAL_COMPONENT_NAME_CHARS,
    MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES,
)
from true_rl_agent.candidates.mechanism import (
    MechanismEvidenceGate,
    RuntimeActivationEvidence,
    build_knob_only_shadow_candidate,
)
from true_rl_agent.candidates.prompt_projection import (
    PROMPT_PAYLOAD_SECTION_ORDER,
    assert_public_only_prompt_payload,
    build_prompt_payload,
    candidate_allowed_controls,
    candidate_component_contract,
    reward_inputs_projection,
    serialize_prompt_payload,
)
from true_rl_agent.candidates.schema import (
    CANDIDATE_ID_ALLOWED_CHARACTERS,
    MAX_CANDIDATE_ID_CHARS,
    CandidateChangeSet,
    candidate_id_contract,
)
from true_rl_agent.prompting.authoring_frame import (
    AUTHORING_CONTEXT_FRAME_KINDS,
    MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
    MAX_AUTHORING_SECTION_BYTES,
    with_final_prompt_budget,
)
from true_rl_agent.prompting.capabilities import build_candidate_capability_profile
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.source_catalog import (
    MAX_PROMPT_SOURCE_INLINE_BYTES,
    prompt_source_catalog_logical_ids,
)
from true_rl_agent.tasks.prompt_adapter import MAX_TASK_PROMPT_SECTIONS_BYTES
from true_rl_agent.tasks.push_recovery.candidate_surface import (
    PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY,
)


class PromptProjectionAndMechanismTests(unittest.TestCase):
    def test_task_profile_hides_unready_and_unselected_capabilities(self) -> None:
        profile = build_candidate_capability_profile(
            profile_id="phase_a_hidden_profile",
            capability_ids=("complete_reward_program",),
        )

        payload = build_prompt_payload(
            public_context={"public_only": True},
            candidate_capability_profile=profile,
        )

        capabilities = payload["candidate_capabilities"]
        self.assertEqual(capabilities["prompt_visible_capabilities"], [])
        self.assertEqual(capabilities["disabled_or_reserved_capabilities"], [])
        self.assertEqual(
            capabilities["provider_declared_components_prompt_advertised"],
            [],
        )
        self.assertEqual(payload["loop_contract"]["candidate_may"], [])
        component_contract = payload["required_candidate_package"][
            "component_contract"
        ]
        self.assertEqual(component_contract["status"], "unavailable")
        self.assertEqual(component_contract["classes"], {})
        self.assertEqual(
            payload["required_candidate_package"]["provider_json_output"][
                "declared_changed_components_allowed_values"
            ],
            [],
        )
        text = json.dumps(payload, sort_keys=True)
        for hidden_surface in (
            "complete_reward_program",
            "RewardTerm",
            "DiagnosticTerm",
            "ScheduleTerm",
        ):
            self.assertNotIn(hidden_surface, text)

    def test_prompt_projection_exposes_only_wired_controls_and_public_reward_inputs(self) -> None:
        payload = build_prompt_payload(public_context={"last_public_score": 0.1})

        self.assertIn("RewardTerm", candidate_allowed_controls())
        self.assertIn("DiagnosticTerm", candidate_allowed_controls())
        self.assertIn("ScheduleTerm", candidate_allowed_controls())
        self.assertNotIn("LossTerm", candidate_allowed_controls())
        self.assertNotIn("ObservationTransform", candidate_allowed_controls())
        self.assertEqual(reward_inputs_projection()["status"], "legacy_alias_deprecated")
        self.assertEqual(reward_inputs_projection()["canonical_section"], "reward_inputs_contract")
        self.assertEqual(reward_inputs_projection()["fields"], [])
        self.assertNotIn("env", reward_inputs_projection()["fields"])
        self.assertNotIn("push_active", reward_inputs_projection()["fields"])
        self.assertNotIn("push_vec_public", reward_inputs_projection()["fields"])
        contract = candidate_component_contract()
        self.assertEqual(contract["builder_signature"], "def build_training_components() -> TrainingComponents:")
        self.assertIn("RewardTerm(name: str, weight: float", contract["classes"]["RewardTerm"])
        self.assertEqual(contract["allowed_schedule_keys"], ["reward_weight_scale"])
        reward_semantics = contract["reward_term_runtime_semantics"]
        self.assertEqual(
            reward_semantics["composition_order"],
            [
                "raw_output = fn(reward_inputs)",
                "guarded_output = clip(raw_output, clip_min, clip_max)",
                "weighted_contribution = weight * guarded_output",
                "training_reward = training_reward + weighted_contribution",
            ],
        )
        self.assertIn("positive weighted contributions reinforce", reward_semantics["optimization_direction"])
        self.assertIn("positive cost rewards", reward_semantics["penalty_sign_rule"])
        self.assertIn("def stance_margin(inp: RewardInputs):", contract["minimal_components_py"])
        self.assertEqual(payload["required_candidate_package"]["component_contract"], contract)
        self.assertIn("static_verifier_rules", contract)
        verifier_rules = contract["static_verifier_rules"]
        self.assertEqual(
            verifier_rules["component_callable_contract"][
                "component_name_contract"
            ]["max_chars"],
            MAX_CANONICAL_COMPONENT_NAME_CHARS,
        )
        self.assertEqual(
            verifier_rules["component_callable_contract"][
                "component_name_contract"
            ]["max_utf8_bytes"],
            MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES,
        )
        self.assertIn("true_rl_agent.candidates.components", verifier_rules["allowed_imports"])
        self.assertIn("open", verifier_rules["blocked_calls"])
        self.assertIn("subprocess", verifier_rules["blocked_call_roots"])
        self.assertIn("checkpoint policy selector surface", verifier_rules["blocked_text_fragments"])
        self.assertNotIn("eval_config", str(payload))
        assert_public_only_prompt_payload(payload)

    def test_candidate_capabilities_are_canonical_prompt_authoring_surface(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})

        capabilities = payload["candidate_capabilities"]
        visible = capabilities["prompt_visible_capabilities"]
        visible_classes = [item["component_class"] for item in visible]

        self.assertEqual(
            capabilities["schema_version"],
            "true_rl_agent.candidate_authoring_capabilities.v1",
        )
        self.assertEqual(visible_classes, ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"])
        self.assertNotIn("candidate_allowed_controls", payload)

        component_contract = payload["required_candidate_package"]["component_contract"]
        effectiveness_contract = payload["control_effectiveness_contract"]
        for capability in visible:
            component_class = capability["component_class"]
            self.assertEqual(
                component_contract["classes"][component_class],
                capability["component_signature"],
            )
            self.assertEqual(
                effectiveness_contract["component_controls"][component_class]["effectiveness"],
                capability["effectiveness"],
            )
            self.assertEqual(
                effectiveness_contract["component_controls"][component_class]["training_effect"],
                capability["runtime_effect"],
            )

        self.assertEqual(component_contract["allowed_schedule_keys"], ["reward_weight_scale"])
        self.assertEqual(
            effectiveness_contract["schedule_controls"]["effective_keys"],
            ["reward_weight_scale"],
        )
        self.assertEqual(
            payload["required_candidate_package"]["component_contract"],
            component_contract,
        )
        self.assertIn("candidate_capabilities", payload["authoring_context_frame"]["included_sections"])
        self.assertIn("candidate_capabilities", payload["authoring_context_frame"]["stable_contract_refs"])
        assert_public_only_prompt_payload(payload)

    def test_candidate_capabilities_do_not_expose_reserved_surfaces(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        capabilities = payload["candidate_capabilities"]
        visible_ids = {item["capability_id"] for item in capabilities["prompt_visible_capabilities"]}

        self.assertNotIn("curriculum_design", visible_ids)
        self.assertNotIn("train_environment_design", visible_ids)
        self.assertEqual(capabilities["disabled_or_reserved_capabilities"], [])

        capabilities_text = json.dumps(capabilities, sort_keys=True)
        self.assertNotIn("curriculum_design", capabilities_text)
        self.assertNotIn("train_environment_design", capabilities_text)
        self.assertNotIn("observation_projection", capabilities_text)
        self.assertNotIn("feature_projection", capabilities_text)
        self.assertNotIn("push_active", capabilities_text)
        self.assertNotIn("push_vec_public", capabilities_text)
        self.assertNotIn("push recovery", capabilities_text.lower())

    def test_task_capability_metadata_stays_task_namespaced(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "task_prompt_sections": {
                    "task": {"schema_version": "custom.task.v1", "task_id": "custom_task"},
                    "task_capability_metadata": {
                        "schema_version": "custom.capability_metadata.v1",
                        "task_id": "custom_task",
                        "capability_notes": {
                            "reward_term": {
                                "task_namespace": "custom_task",
                                "candidate_guidance": "Use custom public height evidence.",
                            }
                        },
                        "debug_note": "scrub D:/humanoid_agent/private/raw.txt",
                    },
                },
            }
        )

        capabilities = payload["candidate_capabilities"]
        self.assertIn("task_specific_metadata", capabilities)
        self.assertEqual(capabilities["task_specific_metadata"]["task_id"], "custom_task")
        self.assertEqual(
            capabilities["task_specific_metadata"]["capability_notes"]["reward_term"][
                "candidate_guidance"
            ],
            "Use custom public height evidence.",
        )
        self.assertIn("[path]", capabilities["task_specific_metadata"]["debug_note"])
        self.assertNotIn(
            "Use custom public height evidence.",
            json.dumps(capabilities["prompt_visible_capabilities"], sort_keys=True),
        )
        visible_classes = [
            item["component_class"]
            for item in payload["candidate_capabilities"]["prompt_visible_capabilities"]
        ]
        self.assertEqual(visible_classes, ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"])
        assert_public_only_prompt_payload(payload)

    def test_prompt_payload_assembler_promotes_task_sections_to_stable_top_level_order(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "previous_rounds": 2,
                "best_public_score": 0.42,
                "last_outcome": "reject",
                "recent_outcomes": [{"round_id": "round-002", "outcome": "reject"}],
                "task_prompt_sections": {
                    "task": {"schema_version": "custom.task.v1", "task_id": "custom_task"},
                    "reward_inputs_contract": {
                        "schema_version": "custom.reward_inputs.v1",
                        "fields": [{"name": "height", "shape": "[num_envs]"}],
                    },
                    "public_eval_contract": {"schema_version": "custom.eval.v1", "status": "public_only"},
                    "public_feedback_summary": {"schema_version": "custom.feedback.v1", "latest": "reject"},
                    "runtime_feedback_summary": {"schema_version": "custom.runtime.v1", "status": "not_loaded"},
                    "known_good_examples": [{"name": "shape_safe_example"}],
                    "selection_guidance": ["Use public custom evidence only."],
                },
            }
        )

        expected_prefix = [
            section
            for section in PROMPT_PAYLOAD_SECTION_ORDER
            if section in payload
        ]
        self.assertEqual(list(payload)[: len(expected_prefix)], expected_prefix)
        self.assertEqual(payload["task"]["task_id"], "custom_task")
        self.assertEqual(payload["reward_inputs_contract"]["fields"][0]["name"], "height")
        self.assertEqual(payload["public_feedback_summary"]["latest"], "reject")
        self.assertEqual(payload["public_history"]["previous_rounds"], 2)
        common_guidance = " ".join(payload["loop_contract"]["evidence_discipline"])
        self.assertIn("public_history.experiment_memory", common_guidance)
        self.assertIn("Use public custom evidence only.", payload["selection_guidance"])
        self.assertNotIn("task_prompt_sections", payload["public_context"])
        self.assertIn("component_contract", payload["required_candidate_package"])
        self.assertNotIn("reward_inputs", payload)
        self.assertNotIn("candidate_component_contract", payload)
        assert_public_only_prompt_payload(payload)

    def test_serialized_prompt_payload_front_loads_task_boundary_feedback_and_package(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "previous_rounds": 3,
                "best_public_score": 0.25,
                "last_outcome": "reject",
                "task_prompt_sections": {
                    "task": {
                        "schema_version": "custom.task.v1",
                        "task_id": "custom_task",
                        "name": "Readable Custom Task",
                    },
                    "public_feedback_summary": {
                        "schema_version": "custom.feedback.v1",
                        "latest": "needs_code_level_change",
                    },
                    "runtime_feedback_summary": {
                        "schema_version": "custom.runtime.v1",
                        "status": "candidate_loaded",
                    },
                },
            }
        )

        text = serialize_prompt_payload(payload)
        first_screen = text[:2500]
        early_sections = [
            '"task"',
            '"loop_contract"',
            '"candidate_capabilities"',
            '"reward_inputs_contract"',
            '"required_candidate_package"',
            '"source_catalog"',
            '"candidate_lineage"',
        ]

        self.assertEqual(json.loads(text), payload)
        for section in early_sections:
            self.assertIn(section, text)
        self.assertLess(text.index('"schema_version"'), text.index('"task"'))
        self.assertLess(text.index('"task"'), text.index('"loop_contract"'))
        self.assertLess(text.index('"loop_contract"'), text.index('"candidate_capabilities"'))
        self.assertLess(text.index('"candidate_capabilities"'), text.index('"reward_inputs_contract"'))
        self.assertLess(text.index('"selection_guidance"'), text.index('"required_candidate_package"'))
        self.assertLess(text.index('"required_candidate_package"'), text.index('"authoring_context_frame"'))
        self.assertLess(text.index('"selection_guidance"'), text.index('"authoring_context_frame"'))
        self.assertLess(text.index('"reward_inputs_contract"'), text.index('"source_catalog"'))
        self.assertLess(text.index('"source_catalog"'), text.index('"candidate_lineage"'))
        self.assertLess(text.index('"candidate_lineage"'), text.index('"public_feedback_summary"'))
        self.assertLess(text.index('"public_feedback_summary"'), text.index('"runtime_feedback_summary"'))
        self.assertIn("Readable Custom Task", first_screen)
        self.assertNotIn('"public_context"', first_screen)

    def test_lane_specific_candidate_rules_start_after_the_shared_prompt_prefix(
        self,
    ) -> None:
        context = {"public_only": True}
        base_rules = candidate_component_contract()["static_verifier_rules"]
        lane_a = build_prompt_payload(
            public_context=context,
            candidate_verifier_prompt_rules={
                **base_rules,
                "generation_lane": "lane-a",
            },
        )
        lane_b = build_prompt_payload(
            public_context=context,
            candidate_verifier_prompt_rules={
                **base_rules,
                "generation_lane": "lane-b",
            },
        )

        marker = '"required_candidate_package"'
        prefix_a = serialize_prompt_payload(lane_a).partition(marker)[0]
        prefix_b = serialize_prompt_payload(lane_b).partition(marker)[0]
        self.assertEqual(prefix_a, prefix_b)
        self.assertGreater(len(prefix_a), 1000)
        self.assertNotEqual(
            lane_a["required_candidate_package"],
            lane_b["required_candidate_package"],
        )

    def test_provider_facing_payload_omits_legacy_alias_sections(self) -> None:
        generic_payload = build_prompt_payload(public_context={"public_only": True})

        for alias in (
            "reward_inputs",
            "candidate_allowed_controls",
            "candidate_component_contract",
            "anti_tuning_notice",
            "runner_owned_extensions",
        ):
            self.assertNotIn(alias, generic_payload)

        custom_payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "task_prompt_sections": {
                    "reward_inputs_contract": {
                        "schema_version": "custom.reward_inputs.v1",
                        "fields": {
                            "height": {"shape": ["num_envs"]},
                            "lateral_velocity": {"shape": ["num_envs"]},
                        },
                    },
                },
            }
        )

        self.assertEqual(list(custom_payload["reward_inputs_contract"]["fields"]), ["height", "lateral_velocity"])
        for alias in (
            "reward_inputs",
            "candidate_allowed_controls",
            "candidate_component_contract",
            "anti_tuning_notice",
            "runner_owned_extensions",
        ):
            self.assertNotIn(alias, custom_payload)

    def test_prompt_payload_assembler_scrubs_paths_from_plain_public_context(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "note": "review D:/humanoid_agent/private/raw.txt and /data/yuyangliu/secrets/key.env",
            }
        )

        text = json.dumps(payload, sort_keys=True)
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertNotIn("private/raw.txt", text)
        self.assertNotIn("/secrets/", text)
        self.assertNotIn("note", payload["public_context"])

    def test_prompt_payload_assembler_reuses_task_section_path_scrub(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "task_prompt_sections": {
                    "task": {"schema_version": "custom.task.v1", "task_id": "custom_task"},
                    "selection_guidance": [
                        "Do not inspect D:/humanoid_agent/private/raw.txt or /data/yuyangliu/secrets/key.env",
                    ],
                },
            }
        )

        text = json.dumps(payload, sort_keys=True)
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertIn("[path]", text)
        assert_public_only_prompt_payload(payload)

    def test_prompt_payload_assembler_reuses_task_section_size_limit(self) -> None:
        oversized = {
            "public_only": True,
            "task_prompt_sections": {
                "task": {"schema_version": "custom.task.v1", "task_id": "custom_task"},
                "selection_guidance": ["x" * (MAX_TASK_PROMPT_SECTIONS_BYTES + 1)],
            },
        }

        with self.assertRaisesRegex(ValueError, "exceed"):
            build_prompt_payload(public_context=oversized)

    def test_task_prompt_section_budget_uses_provider_serialization_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "task prompt sections exceed"):
            build_prompt_payload(
                public_context={
                    "public_only": True,
                    "task_prompt_sections": {
                        "selection_guidance": ["x"] * 3638,
                    },
                }
            )

    def test_prompt_payload_assembler_promotes_repair_feedback(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "candidate_repair_feedback": {
                    "schema_version": "true_rl_agent.provider_repair_feedback.v1",
                    "failure_category": "schema_error",
                    "reason": "components_py missing",
                },
            }
        )

        self.assertEqual(
            payload["provider_or_runtime_repair_feedback"]["failure_category"],
            "schema_error",
        )
        self.assertNotIn("candidate_repair_feedback", payload["public_context"])
        self.assertEqual(
            payload["public_context"]["candidate_repair_feedback_ref"],
            {
                "available": True,
                "canonical_section": "provider_or_runtime_repair_feedback",
            },
        )
        self.assertEqual(
            set(payload["public_context"]),
            {
                "schema_version",
                "deprecated",
                "public_only",
                "previous_rounds",
                "best_public_score",
                "last_outcome",
                "pure_knob_streak",
                "canonical_sections",
                "candidate_repair_feedback_ref",
            },
        )
        self.assertLess(canonical_prompt_json_bytes(payload["public_context"]), 1024)
        self.assertLess(
            list(payload).index("schema_version"),
            list(payload).index("provider_or_runtime_repair_feedback"),
        )

    def test_prompt_payload_records_candidate_draft_authoring_context_frame(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "previous_rounds": 1,
                "best_public_score": 0.31,
                "last_outcome": "reject",
                "recent_outcomes": [{"round_id": "round-001", "outcome": "reject"}],
            }
        )

        frame = payload["authoring_context_frame"]

        self.assertIn("candidate_draft", AUTHORING_CONTEXT_FRAME_KINDS)
        self.assertEqual(frame["schema_version"], "true_rl_agent.authoring_context_frame.v2")
        self.assertEqual(frame["frame_kind"], "candidate_draft")
        self.assertIn("task", frame["included_sections"])
        self.assertIn("public_history", frame["included_sections"])
        self.assertEqual(frame["omitted_sections"], [])
        self.assertEqual(frame["source_payloads"], [])
        self.assertEqual(frame["source_hashes"], [])
        self.assertGreater(frame["budget_metadata"]["estimated_prompt_bytes"], 0)
        self.assertGreater(frame["budget_metadata"]["estimated_prompt_context_units"], 0)
        self.assertEqual(
            frame["budget_metadata"]["hard_limits"]["max_prompt_payload_bytes"],
            MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
        )
        self.assertEqual(
            frame["budget_metadata"]["actual_prompt_payload_bytes"],
            len(serialize_prompt_payload(payload).encode("utf-8")),
        )
        self.assertIn("public_context", frame["included_sections"])
        self.assertIn("public_context", frame["legacy_alias_sections"])
        self.assertIn("public_context", frame["budget_metadata"]["section_serialized_bytes"])
        self.assertIn("public_context", frame["budget_metadata"]["legacy_alias_section_bytes"])
        for removed_alias in (
            "reward_inputs",
            "candidate_allowed_controls",
            "candidate_component_contract",
            "anti_tuning_notice",
            "runner_owned_extensions",
        ):
            self.assertNotIn(removed_alias, frame["included_sections"])
            self.assertNotIn(removed_alias, frame["legacy_alias_sections"])
            self.assertNotIn(removed_alias, frame["budget_metadata"]["section_serialized_bytes"])
            self.assertNotIn(removed_alias, frame["budget_metadata"]["legacy_alias_section_bytes"])
        self.assertIn("public_history", frame["budget_metadata"]["section_serialized_bytes"])
        self.assertLess(
            list(payload).index("selection_guidance"),
            list(payload).index("required_candidate_package"),
        )
        self.assertLess(
            list(payload).index("required_candidate_package"),
            list(payload).index("authoring_context_frame"),
        )
        self.assertLess(
            list(payload).index("selection_guidance"),
            list(payload).index("authoring_context_frame"),
        )
        self.assertLess(
            list(payload).index("authoring_context_frame"),
            list(payload).index("source_catalog"),
        )
        common_guidance = " ".join(payload["loop_contract"]["evidence_discipline"])
        self.assertIn("numeric-only", common_guidance)
        self.assertIn("historical-source gate", common_guidance)
        self.assertIn(
            "change exactly one named canonical prompt-visible",
            common_guidance,
        )
        self.assertNotIn("combine multiple rejected mechanisms", common_guidance)
        assert_public_only_prompt_payload(frame)
        assert_public_only_prompt_payload(payload)

    def test_static_repair_without_failed_source_keeps_draft_context(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "previous_rounds": 3,
                "best_public_score": 0.31,
                "recent_outcomes": [
                    {"round_id": "round-001", "outcome": "reject"},
                    {"round_id": "round-002", "outcome": "reject"},
                    {"round_id": "round-003", "outcome": "reject"},
                ],
                "candidate_repair_feedback": {
                    "schema_version": "true_rl_agent.provider_repair_feedback.v1",
                    "failure_category": "verifier_error",
                    "reason": "components_py failed static verification",
                },
            }
        )

        frame = payload["authoring_context_frame"]
        omitted = frame["omitted_sections"]

        self.assertEqual(frame["frame_kind"], "static_repair")
        self.assertIn("provider_or_runtime_repair_feedback", frame["included_sections"])
        self.assertEqual(len(payload["public_history"]["recent_outcomes"]), 3)
        self.assertIn("public_feedback_summary", payload)
        self.assertIn("runtime_feedback_summary", payload)
        self.assertIn("known_good_examples", payload)
        self.assertIn("selection_guidance", payload)
        self.assertIn("exploration_directive", payload)
        self.assertEqual(omitted, [])
        self.assertEqual(payload["source_payloads"], [])
        self.assertIn("source_payloads", frame["frame_contract"]["required_sections"])
        self.assertIn(
            "public_history",
            frame["frame_contract"]["optional_sections"],
        )
        for stable_contract in (
            "task",
            "loop_contract",
            "required_candidate_package",
            "candidate_capabilities",
            "reward_inputs_contract",
            "public_eval_contract",
            "control_effectiveness_contract",
        ):
            self.assertIn(stable_contract, frame["stable_contract_refs"])
        assert_public_only_prompt_payload(frame)
        assert_public_only_prompt_payload(payload)

    def test_legacy_public_context_alias_is_fixed_and_does_not_duplicate_history(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "previous_rounds": 2,
                "best_public_score": 0.42,
                "last_outcome": "reject",
                "pure_knob_streak": 1,
                "public_history_limits": {"max_recent_rounds": 99},
                "public_history_omitted_round_count": 7,
                "recent_outcomes": [
                    {"round_id": "round-001", "outcome": "reject"},
                    {"round_id": "round-002", "outcome": "reject"},
                ],
                "task_public_summary": {"opaque": "task-owned"},
            }
        )

        alias = payload["public_context"]

        self.assertEqual(
            alias,
            {
                "schema_version": "true_rl_agent.public_context_legacy_summary.v1",
                "deprecated": True,
                "public_only": True,
                "previous_rounds": 2,
                "best_public_score": 0.42,
                "last_outcome": "reject",
                "pure_knob_streak": 1,
                "canonical_sections": {
                    "history": "public_history",
                    "public_feedback": "public_feedback_summary",
                    "runtime_feedback": "runtime_feedback_summary",
                },
            },
        )
        self.assertEqual(len(payload["public_history"]["recent_outcomes"]), 2)
        self.assertLess(canonical_prompt_json_bytes(alias), 1024)

    def test_source_followup_frame_is_provider_constructible_and_omits_recent_history(self) -> None:
        self.assertIn("source_followup", AUTHORING_CONTEXT_FRAME_KINDS)

        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "recent_outcomes": [{"round_id": "round-001", "outcome": "reject"}],
                "candidate_repair_feedback": {
                    "schema_version": "true_rl_agent.source_request_feedback.v1",
                    "status": "source_payload_returned",
                    "failure_category": "source_payload_returned",
                    "returned_logical_ids": ["current_best/training_code/components.py"],
                },
            },
            frame_kind="source_followup",
        )

        frame = payload["authoring_context_frame"]

        self.assertEqual(frame["frame_kind"], "source_followup")
        self.assertTrue(frame["frame_contract"]["source_followup_enabled"])
        self.assertIn("candidate_capabilities", payload)
        self.assertIn("candidate_capabilities", frame["included_sections"])
        self.assertNotIn("candidate_allowed_controls", payload)
        self.assertNotIn("candidate_component_contract", payload)
        self.assertNotIn("runner_owned_extensions", payload)
        self.assertEqual(payload["public_history"]["recent_outcomes"], [])
        self.assertIn("provider_or_runtime_repair_feedback", frame["included_sections"])
        self.assertEqual(frame["feedback_items"][0]["failure_category"], "source_payload_returned")
        self.assertEqual(frame["omitted_sections"][0]["reason"], "omitted_for_source_followup_frame")

    def test_authoring_frame_reports_current_source_budgets_not_phase0_placeholder(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})

        hard_limits = payload["authoring_context_frame"]["budget_metadata"]["hard_limits"]

        self.assertEqual(hard_limits["max_source_inline_payload_bytes"], MAX_PROMPT_SOURCE_INLINE_BYTES)
        self.assertEqual(
            hard_limits["max_source_request_bytes_per_round"],
            MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
        )
        self.assertNotIn("max_source_payload_bytes_phase0", hard_limits)

    def test_prompt_payload_enforces_authoring_frame_payload_budget(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        section_payload = {
            "section_a": "x" * 90_000,
            "section_b": "y" * 90_000,
            "section_c": "z" * 90_000,
        }

        with self.assertRaisesRegex(ValueError, "authoring prompt payload exceeds"):
            with_final_prompt_budget(
                payload["authoring_context_frame"],
                prompt_payload=section_payload,
            )

    def test_prompt_payload_does_not_copy_oversized_unprojected_context_into_legacy_alias(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "note": "x" * (MAX_AUTHORING_SECTION_BYTES + 1),
            }
        )

        self.assertNotIn("note", payload["public_context"])
        self.assertLess(canonical_prompt_json_bytes(payload["public_context"]), 1024)

    def test_required_candidate_package_advertises_only_currently_wired_declared_components(self) -> None:
        from true_rl_agent.agent_core.provider_contract import (
            DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES,
            PROMPT_ADVERTISED_DECLARED_COMPONENTS,
        )

        payload = build_prompt_payload(public_context={"public_only": True})

        prompt_values = tuple(
            payload["required_candidate_package"]["provider_json_output"][
                "declared_changed_components_allowed_values"
            ]
        )
        self.assertEqual(prompt_values, PROMPT_ADVERTISED_DECLARED_COMPONENTS)
        self.assertEqual(prompt_values, DECLARED_CHANGED_COMPONENTS_ALLOWED_VALUES)
        for legacy_value in (
            "control_logic",
            "recovery_logic",
            "failure_classifier",
            "logging",
            "public_projection",
            "reward_weight_scale",
            "schedule",
        ):
            self.assertNotIn(legacy_value, prompt_values)

    def test_required_candidate_package_defines_provider_output_and_file_contract(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        package = payload["required_candidate_package"]
        output = package["provider_json_output"]
        files = package["candidate_files"]

        self.assertEqual(package["candidate_role"], "candidate_package_only")
        self.assertEqual(output["response_shape"], "one_of")
        self.assertNotIn("required_keys", output)
        self.assertNotIn("allowed_keys", output)
        draft = output["one_of"]["candidate_draft"]
        source_request = output["one_of"]["source_request"]
        self.assertEqual(
            draft["required_keys"],
            ["candidate_id", "declared_changed_components", "components_py"],
        )
        self.assertEqual(draft["allowed_keys"], ["response_mode", *draft["required_keys"]])
        self.assertEqual(source_request["required_keys"], ["response_mode", "logical_ids", "rationale"])
        self.assertEqual(source_request["allowed_keys"], ["response_mode", "logical_ids", "rationale"])
        self.assertNotIn("candidate_id", source_request["allowed_keys"])
        self.assertIn("logical_ids", source_request["allowed_keys"])
        self.assertFalse(output["additional_keys_allowed"])
        self.assertTrue(output["json_only"])
        self.assertTrue(output["no_prose_outside_json"])
        self.assertEqual(output["authoring_response_protocol"]["default_response_mode"], "candidate_draft")
        self.assertEqual(
            output["authoring_response_protocol"]["schema_version"],
            "true_rl_agent.authoring_provider_response_contract.v2",
        )
        self.assertTrue(output["authoring_response_protocol"]["source_request_service_enabled"])
        self.assertTrue(source_request["enabled"])
        self.assertIn("runner_patch", output["prohibited_keys"])
        self.assertIn("operation_diff", output["prohibited_keys"])
        self.assertEqual(output["field_contracts"]["components_py"]["materialized_file"], "training_code/components.py")
        self.assertEqual(
            output["field_contracts"]["components_py"]["derived_materialized_files"],
            {"rationale.md": "TrainingComponents.rationale"},
        )
        self.assertNotIn("rationale", draft["allowed_keys"])
        self.assertEqual(
            output["field_contracts"]["rationale"]["available_in_response_modes"],
            ["source_request"],
        )
        self.assertEqual(output["field_contracts"]["candidate_id"]["materialized_file"], "candidate_manifest.yaml")
        self.assertEqual(
            output["field_contracts"]["candidate_id"]["max_chars"],
            MAX_CANDIDATE_ID_CHARS,
        )
        self.assertEqual(
            output["field_contracts"]["candidate_id"]["allowed_characters"],
            CANDIDATE_ID_ALLOWED_CHARACTERS,
        )
        self.assertEqual(
            {
                key: output["field_contracts"]["candidate_id"][key]
                for key in candidate_id_contract()
            },
            candidate_id_contract(),
        )

        expected_files = {
            "candidate_manifest.yaml",
            "training_code/components.py",
            "rationale.md",
            "operation_diff.json",
            "diagnostics.json",
        }
        self.assertEqual(set(files["materialized_files"]), expected_files)
        self.assertEqual(files["llm_writable_provider_json_fields"], draft["allowed_keys"])
        self.assertNotIn("logical_ids", files["llm_writable_provider_json_fields"])
        self.assertEqual(files["provider_materialized_files"]["operation_diff.json"]["source"], "candidate_materializer")
        self.assertEqual(files["provider_materialized_files"]["operation_diff.json"]["derived_from"], ["declared_changed_components"])
        self.assertEqual(files["provider_materialized_files"]["diagnostics.json"]["source"], "candidate_materializer")
        self.assertEqual(
            files["provider_materialized_files"]["rationale.md"]["derived_from"],
            ["training_code/components.py:TrainingComponents.rationale"],
        )
        self.assertIn("success metric", package["candidate_must_not_change"])
        self.assertIn("runner code", package["candidate_must_not_change"])

    def test_public_only_prompt_payload_rejects_private_hidden_or_secret_keys(self) -> None:
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public": True, "private_seed": 123})
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"raw_seed_id": "public-dev-seed-001"}})
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"api_key": "sk-secret"})

    def test_source_request_response_allows_only_prompt_catalog_logical_ids(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        source_catalog_ids = tuple(sorted(prompt_source_catalog_logical_ids(payload)))

        response = SourceRequestResponse(
            logical_ids=("current_best/training_code/components.py",),
            rationale="Need current best source.",
            source_catalog_logical_ids=source_catalog_ids,
        )

        self.assertEqual(response.logical_ids, ("current_best/training_code/components.py",))
        with self.assertRaises(ValueError):
            SourceRequestResponse(
                logical_ids=("../secret",),
                rationale="Need source.",
                source_catalog_logical_ids=source_catalog_ids,
            )
        with self.assertRaises(ValueError):
            SourceRequestResponse(
                logical_ids=("unknown/source.py",),
                rationale="Need source.",
                source_catalog_logical_ids=source_catalog_ids,
            )

    def test_mechanism_gate_rejects_parameter_only_and_requires_runner_activation(self) -> None:
        gate = MechanismEvidenceGate()

        parameter_only = gate.evaluate(
            surface=PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY.classify(
                CandidateChangeSet(frozenset({"reward_weight"}))
            ),
            evidence=RuntimeActivationEvidence(reward_term_calls=10, nonzero_reward_contribution=True),
        )
        inactive_code = gate.evaluate(
            surface=PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY.classify(
                CandidateChangeSet(frozenset({"reward_callable"}))
            ),
            evidence=RuntimeActivationEvidence(reward_term_calls=0, nonzero_reward_contribution=False),
        )
        active_code = gate.evaluate(
            surface=PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY.classify(
                CandidateChangeSet(frozenset({"reward_callable"}))
            ),
            evidence=RuntimeActivationEvidence(reward_term_calls=3, nonzero_reward_contribution=True),
        )

        self.assertEqual(parameter_only.status, "reject")
        self.assertEqual(parameter_only.reason, "parameter_only_candidate")
        self.assertEqual(inactive_code.status, "attribution_unclear")
        self.assertEqual(active_code.status, "pass")

    def test_runtime_activation_mapping_rejects_weak_type_coercion(self) -> None:
        valid = RuntimeActivationEvidence.from_mapping(
            {
                "reward_term_calls": 3,
                "diagnostic_calls": 0,
                "nonzero_reward_contribution": True,
                "reward_output_stats": {},
                "diagnostic_outputs": {},
                "component_activation_records": [{"name": "reward"}],
                "feature_activation_records": [],
                "capability_activation_records": ["reward_program"],
            }
        )
        self.assertEqual(valid.reward_term_calls, 3)

        for field_name, invalid in (
            ("reward_term_calls", "3"),
            ("diagnostic_calls", 0.0),
            ("nonzero_reward_contribution", 1),
            ("component_activation_records", ["not-an-object"]),
            ("capability_activation_records", [1]),
        ):
            with self.subTest(field_name=field_name):
                payload = {
                    "reward_term_calls": 3,
                    "diagnostic_calls": 0,
                    "nonzero_reward_contribution": True,
                    "component_activation_records": [],
                    "capability_activation_records": [],
                }
                payload[field_name] = invalid
                with self.assertRaises((TypeError, ValueError)):
                    RuntimeActivationEvidence.from_mapping(payload)

    def test_knob_only_shadow_candidate_keeps_only_parameter_components(self) -> None:
        shadow = build_knob_only_shadow_candidate(
            CandidateChangeSet(frozenset({"reward_callable", "reward_weight", "terrain_range"})),
            surface_policy=PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY,
        )

        self.assertEqual(shadow.changed_components, frozenset({"reward_weight", "terrain_range"}))


if __name__ == "__main__":
    unittest.main()

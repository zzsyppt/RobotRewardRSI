from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.candidates.source_policy import SourceAccessPolicy
from true_rl_agent.prompting.authoring_frame import (
    AUTHORING_CONTEXT_FRAME_KINDS,
    MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
    MAX_AUTHORING_SECTION_BYTES,
    with_final_prompt_budget,
)
from true_rl_agent.tasks.prompt_adapter import MAX_TASK_PROMPT_SECTIONS_BYTES


class PromptProjectionAndMechanismTests(unittest.TestCase):
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
        self.assertIn("def stance_margin(inp: RewardInputs):", contract["minimal_components_py"])
        self.assertEqual(payload["candidate_component_contract"], contract)
        self.assertNotIn("eval_config", str(payload))
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
        self.assertNotIn("task_prompt_sections", payload["public_context"])
        self.assertEqual(
            payload["required_candidate_package"]["component_contract"],
            payload["candidate_component_contract"],
        )
        self.assertEqual(payload["reward_inputs"]["status"], "legacy_alias_deprecated")
        self.assertEqual(payload["reward_inputs"]["fields"], [])
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
            '"required_candidate_package"',
            '"reward_inputs_contract"',
            '"source_catalog"',
            '"candidate_lineage"',
        ]

        self.assertEqual(json.loads(text), payload)
        for section in early_sections:
            self.assertIn(section, text)
        self.assertLess(text.index('"task"'), text.index('"loop_contract"'))
        self.assertLess(text.index('"loop_contract"'), text.index('"required_candidate_package"'))
        self.assertLess(text.index('"required_candidate_package"'), text.index('"reward_inputs_contract"'))
        self.assertLess(text.index('"reward_inputs_contract"'), text.index('"source_catalog"'))
        self.assertLess(text.index('"source_catalog"'), text.index('"candidate_lineage"'))
        self.assertLess(text.index('"candidate_lineage"'), text.index('"public_feedback_summary"'))
        self.assertLess(text.index('"public_feedback_summary"'), text.index('"runtime_feedback_summary"'))
        self.assertIn("Readable Custom Task", first_screen)
        self.assertNotIn('"public_context"', first_screen)

    def test_legacy_reward_inputs_alias_never_reexports_task_owned_fields(self) -> None:
        generic_payload = build_prompt_payload(public_context={"public_only": True})

        self.assertEqual(generic_payload["reward_inputs"]["status"], "legacy_alias_deprecated")
        self.assertEqual(generic_payload["reward_inputs"]["canonical_section"], "reward_inputs_contract")
        self.assertEqual(generic_payload["reward_inputs"]["fields"], [])

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

        self.assertEqual(custom_payload["reward_inputs"]["fields"], [])
        self.assertEqual(list(custom_payload["reward_inputs_contract"]["fields"]), ["height", "lateral_velocity"])

    def test_prompt_payload_assembler_scrubs_paths_from_plain_public_context(self) -> None:
        payload = build_prompt_payload(
            public_context={
                "public_only": True,
                "note": "review D:/humanoid_agent/private/raw.txt and /data/yuyangliu/secrets/key.env",
            }
        )

        text = json.dumps(payload, sort_keys=True)
        self.assertIn("[path]", text)
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertNotIn("private/raw.txt", text)
        self.assertNotIn("/secrets/", text)

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
        self.assertEqual(
            payload["provider_or_runtime_repair_feedback"],
            payload["public_context"]["candidate_repair_feedback"],
        )
        self.assertLess(
            list(payload).index("provider_or_runtime_repair_feedback"),
            list(payload).index("schema_version"),
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
        self.assertEqual(frame["schema_version"], "true_rl_agent.authoring_context_frame.v1")
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
            len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")),
        )
        for alias_section in (
            "public_context",
            "reward_inputs",
            "candidate_allowed_controls",
            "candidate_component_contract",
            "anti_tuning_notice",
            "runner_owned_extensions",
        ):
            self.assertIn(alias_section, frame["included_sections"])
            self.assertIn(alias_section, frame["legacy_alias_sections"])
            self.assertIn(alias_section, frame["budget_metadata"]["section_serialized_bytes"])
            self.assertIn(alias_section, frame["budget_metadata"]["legacy_alias_section_bytes"])
        self.assertIn("public_history", frame["budget_metadata"]["section_serialized_bytes"])
        self.assertLess(
            list(payload).index("required_candidate_package"),
            list(payload).index("authoring_context_frame"),
        )
        self.assertLess(list(payload).index("authoring_context_frame"), list(payload).index("reward_inputs_contract"))
        assert_public_only_prompt_payload(frame)
        assert_public_only_prompt_payload(payload)

    def test_static_repair_frame_omits_recent_history_and_records_omission(self) -> None:
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
        self.assertEqual(payload["public_history"]["recent_outcomes"], [])
        self.assertEqual(payload["public_context"]["recent_outcomes"], [])
        self.assertGreaterEqual(payload["public_history"]["omitted_round_count"], 3)
        self.assertEqual(omitted[0]["section"], "public_history.recent_outcomes")
        self.assertEqual(omitted[0]["reason"], "omitted_for_static_repair_frame")
        self.assertEqual(omitted[0]["omitted_item_count"], 3)
        self.assertEqual(omitted[0]["recoverable_from"], "prior ledger public records")
        assert_public_only_prompt_payload(frame)
        assert_public_only_prompt_payload(payload)

    def test_reserved_source_followup_frame_is_not_provider_constructible_in_phase0(self) -> None:
        self.assertNotIn("source_followup", AUTHORING_CONTEXT_FRAME_KINDS)

        with self.assertRaisesRegex(ValueError, "disabled authoring frame kind: source_followup"):
            build_prompt_payload(public_context={"public_only": True}, frame_kind="source_followup")

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

    def test_prompt_payload_enforces_legacy_alias_section_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "authoring context section exceeds"):
            build_prompt_payload(
                public_context={
                    "public_only": True,
                    "note": "x" * (MAX_AUTHORING_SECTION_BYTES + 1),
                }
            )

    def test_required_candidate_package_advertises_only_currently_wired_declared_components(self) -> None:
        from true_rl_agent.agent_core.provider_contract import (
            MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES,
            PROMPT_ADVERTISED_DECLARED_COMPONENTS,
        )

        payload = build_prompt_payload(public_context={"public_only": True})

        prompt_values = tuple(
            payload["required_candidate_package"]["provider_json_output"][
                "declared_changed_components_allowed_values"
            ]
        )
        self.assertEqual(prompt_values, PROMPT_ADVERTISED_DECLARED_COMPONENTS)
        self.assertTrue(set(prompt_values).issubset(MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES))
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
        self.assertEqual(output["required_keys"], ["candidate_id", "rationale", "declared_changed_components", "components_py"])
        self.assertEqual(output["allowed_keys"], output["required_keys"])
        self.assertFalse(output["additional_keys_allowed"])
        self.assertTrue(output["json_only"])
        self.assertTrue(output["no_prose_outside_json"])
        self.assertIn("runner_patch", output["prohibited_keys"])
        self.assertIn("operation_diff", output["prohibited_keys"])
        self.assertEqual(output["field_contracts"]["components_py"]["materialized_file"], "training_code/components.py")
        self.assertEqual(output["field_contracts"]["candidate_id"]["materialized_file"], "candidate_manifest.yaml")

        expected_files = {
            "candidate_manifest.yaml",
            "training_code/components.py",
            "rationale.md",
            "operation_diff.json",
            "diagnostics.json",
        }
        self.assertEqual(set(files["materialized_files"]), expected_files)
        self.assertEqual(files["llm_writable_provider_json_fields"], output["required_keys"])
        self.assertEqual(files["provider_materialized_files"]["operation_diff.json"]["source"], "provider")
        self.assertEqual(files["provider_materialized_files"]["diagnostics.json"]["source"], "provider")
        self.assertIn("success metric", package["candidate_must_not_change"])
        self.assertIn("runner code", package["candidate_must_not_change"])

    def test_public_only_prompt_payload_rejects_private_hidden_or_secret_keys(self) -> None:
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public": True, "private_seed": 123})
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"raw_seed_id": "public-dev-seed-001"}})
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"api_key": "sk-secret"})

    def test_source_policy_allows_only_logical_allowlist_ids(self) -> None:
        policy = SourceAccessPolicy({"candidate_schema": "docs/candidate_schema.md"})

        self.assertEqual(policy.resolve("candidate_schema"), "docs/candidate_schema.md")
        with self.assertRaises(KeyError):
            policy.resolve("../secret")

    def test_mechanism_gate_rejects_parameter_only_and_requires_runner_activation(self) -> None:
        gate = MechanismEvidenceGate()

        parameter_only = gate.evaluate(
            change_set=CandidateChangeSet(frozenset({"reward_weight"})),
            evidence=RuntimeActivationEvidence(reward_term_calls=10, nonzero_reward_contribution=True),
        )
        inactive_code = gate.evaluate(
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            evidence=RuntimeActivationEvidence(reward_term_calls=0, nonzero_reward_contribution=False),
        )
        active_code = gate.evaluate(
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            evidence=RuntimeActivationEvidence(reward_term_calls=3, nonzero_reward_contribution=True),
        )

        self.assertEqual(parameter_only.status, "reject")
        self.assertEqual(parameter_only.reason, "parameter_only_candidate")
        self.assertEqual(inactive_code.status, "attribution_unclear")
        self.assertEqual(active_code.status, "pass")

    def test_knob_only_shadow_candidate_keeps_only_parameter_components(self) -> None:
        shadow = build_knob_only_shadow_candidate(
            CandidateChangeSet(frozenset({"reward_callable", "reward_weight", "terrain_range"}))
        )

        self.assertEqual(shadow.changed_components, frozenset({"reward_weight", "terrain_range"}))


if __name__ == "__main__":
    unittest.main()

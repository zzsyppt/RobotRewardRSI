from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.authoring_loop import build_authoring_prompt_payload
from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.prompting.runner_capability_schemas import (
    CURRICULUM_DESIGN_CAPABILITY_ID,
    TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
    runner_capability_schema_by_id,
    validate_runner_capability_payload,
)


class Phase9RunnerCapabilitySchemaTests(unittest.TestCase):
    def test_runner_owned_capability_schemas_default_to_non_prompt_writable_refs(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        capabilities = payload["candidate_capabilities"]
        visible_ids = {item["capability_id"] for item in capabilities["prompt_visible_capabilities"]}
        reserved = {item["capability_id"]: item for item in capabilities["disabled_or_reserved_capabilities"]}

        self.assertNotIn(CURRICULUM_DESIGN_CAPABILITY_ID, visible_ids)
        self.assertNotIn(TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID, visible_ids)
        for capability_id in (CURRICULUM_DESIGN_CAPABILITY_ID, TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID):
            capability = reserved[capability_id]
            self.assertFalse(capability["prompt_visible"])
            self.assertFalse(capability["candidate_writable"])
            self.assertFalse(capability["effective_now"])
            self.assertEqual(capability["schema_status"], "schema_draft_hidden_until_ready")
            self.assertNotIn("output_schema", capability)
            self.assertEqual(capability["readiness_gate"]["status"], "not_ready")

        visible_classes = [
            item["component_class"]
            for item in capabilities["prompt_visible_capabilities"]
        ]
        self.assertEqual(visible_classes, ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"])
        self.assertNotIn("candidate_allowed_controls", payload)

    def test_runner_owned_readiness_requires_strict_boolean_true(self) -> None:
        payload = build_prompt_payload(
            public_context={"public_only": True},
            runner_capability_readiness={
                CURRICULUM_DESIGN_CAPABILITY_ID: {
                    "task_enabled": "true",
                    "compiler_ready": "yes",
                    "verifier_ready": "1",
                },
            },
        )
        visible_ids = {
            item["capability_id"]
            for item in payload["candidate_capabilities"]["prompt_visible_capabilities"]
        }
        reserved = {
            item["capability_id"]: item
            for item in payload["candidate_capabilities"]["disabled_or_reserved_capabilities"]
        }
        self.assertNotIn(CURRICULUM_DESIGN_CAPABILITY_ID, visible_ids)
        self.assertEqual(reserved[CURRICULUM_DESIGN_CAPABILITY_ID]["readiness_gate"]["status"], "not_ready")
        self.assertEqual(
            reserved[CURRICULUM_DESIGN_CAPABILITY_ID]["readiness_gate"]["missing_requirements"],
            ["task_enabled", "compiler_ready", "verifier_ready"],
        )

    def test_runner_owned_schema_preview_requires_task_compiler_and_verifier_readiness(self) -> None:
        partial_payload = build_prompt_payload(
            public_context={"public_only": True},
            runner_capability_readiness={
                CURRICULUM_DESIGN_CAPABILITY_ID: {
                    "task_enabled": True,
                    "compiler_ready": True,
                    "verifier_ready": False,
                },
            },
        )
        partial_reserved = {
            item["capability_id"]: item
            for item in partial_payload["candidate_capabilities"]["disabled_or_reserved_capabilities"]
        }
        self.assertEqual(
            partial_reserved[CURRICULUM_DESIGN_CAPABILITY_ID]["readiness_gate"]["missing_requirements"],
            ["verifier_ready"],
        )

        ready_payload = build_prompt_payload(
            public_context={"public_only": True},
            runner_capability_readiness={
                CURRICULUM_DESIGN_CAPABILITY_ID: {
                    "task_enabled": True,
                    "compiler_ready": True,
                    "verifier_ready": True,
                },
                TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID: {
                    "task_enabled": True,
                    "compiler_ready": True,
                    "verifier_ready": True,
                },
            },
        )
        visible = {
            item["capability_id"]: item
            for item in ready_payload["candidate_capabilities"]["prompt_visible_capabilities"]
        }

        for capability_id in (CURRICULUM_DESIGN_CAPABILITY_ID, TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID):
            capability = visible[capability_id]
            self.assertTrue(capability["prompt_visible"])
            self.assertFalse(capability["candidate_writable"])
            self.assertFalse(capability["effective_now"])
            self.assertEqual(capability["exposure_mode"], "schema_preview_only")
            self.assertEqual(capability["schema_status"], "schema_preview_ready")
            self.assertEqual(capability["readiness_gate"]["status"], "ready")
            self.assertEqual(capability["output_schema"]["capability_id"], capability_id)

        ready_text = json.dumps(ready_payload, sort_keys=True)
        for forbidden in (
            "runner_source",
            "runner_code",
            "eval_seed",
            "checkpoint_path",
            "trusted_eval",
            "raw_seed",
        ):
            self.assertNotIn(forbidden, ready_text)
        provider_output = ready_payload["required_candidate_package"]["provider_json_output"]
        draft_allowed_keys = provider_output["one_of"]["candidate_draft"]["allowed_keys"]
        self.assertNotIn(CURRICULUM_DESIGN_CAPABILITY_ID, draft_allowed_keys)
        self.assertNotIn(TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID, draft_allowed_keys)
        self.assertNotIn("operation_diff", draft_allowed_keys)

    def test_runner_owned_schema_validator_rejects_eval_budget_seed_or_checkpoint_requests(self) -> None:
        schema = runner_capability_schema_by_id(CURRICULUM_DESIGN_CAPABILITY_ID)
        self.assertEqual(schema["capability_id"], CURRICULUM_DESIGN_CAPABILITY_ID)
        self.assertEqual(schema["status"], "schema_preview_only")

        valid_environment_payload = validate_runner_capability_payload(
            TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "Train-only terrain variation preview.",
                "train_only_environment": {
                    "variations": [
                        {
                            "name": "roughness_scale",
                            "kind": "range_scale",
                            "range": [0.0, 1.0],
                            "rationale": "Bounded train-only robustness preview.",
                        }
                    ],
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
        )
        self.assertEqual(valid_environment_payload["capability_id"], TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID)

        forbidden_payloads = [
            {
                "capability_id": CURRICULUM_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden eval change",
                "train_only_curriculum": {"eval_protocol": "change_public_eval"},
            },
            {
                "capability_id": CURRICULUM_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden budget change",
                "train_only_curriculum": {"training_budget": 123},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden seed policy",
                "train_only_environment": {"seed_policy": "choose_easy_seed"},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden checkpoint",
                "train_only_environment": {"checkpoint": "custom_checkpoint"},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden ppo",
                "train_only_environment": {"ppo_config": {"learning_rate": 1.0}},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden eval seed",
                "train_only_environment": {"eval_seed": 123},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden raw seed",
                "train_only_environment": {"raw_seed": 123},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden trusted eval",
                "train_only_environment": {"trusted_eval": "hidden"},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden secret",
                "train_only_environment": {"secret_key": "value"},
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "forbidden path",
                "train_only_environment": {"artifact_path": "D:/humanoid_agent/private/raw.txt"},
            },
        ]
        for payload in forbidden_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "forbidden runner capability"):
                    validate_runner_capability_payload(str(payload["capability_id"]), payload)

    def test_runner_owned_schema_validator_rejects_scrubbed_values_and_malformed_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "public scrub"):
            validate_runner_capability_payload(
                TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                {
                    "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                    "rationale": "See /data/yuyangliu/secrets/key.env",
                    "train_only_environment": {
                        "variations": [],
                        "final_distribution_policy": "runner_owned_target_distribution_required",
                    },
                },
            )

        scrubbed_key_payloads = [
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "private key must reject, not scrub",
                "private_note": "should not be silently deleted",
                "train_only_environment": {
                    "variations": [],
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "hidden key must reject, not scrub",
                "hidden_eval_hint": "should not be silently deleted",
                "train_only_environment": {
                    "variations": [],
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
        ]
        for payload in scrubbed_key_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "public scrub"):
                    validate_runner_capability_payload(str(payload["capability_id"]), payload)

        malformed_payloads = [
            {
                "capability_id": CURRICULUM_DESIGN_CAPABILITY_ID,
                "rationale": "bad root type",
                "train_only_curriculum": [],
            },
            {
                "capability_id": CURRICULUM_DESIGN_CAPABILITY_ID,
                "rationale": "missing stage list",
                "train_only_curriculum": {
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "bad variations type",
                "train_only_environment": {
                    "variations": {},
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "bad variation item",
                "train_only_environment": {
                    "variations": ["roughness"],
                    "final_distribution_policy": "runner_owned_target_distribution_required",
                },
            },
            {
                "capability_id": CURRICULUM_DESIGN_CAPABILITY_ID,
                "rationale": "candidate must not change target distribution",
                "train_only_curriculum": {
                    "stages": [],
                    "final_distribution_policy": "candidate_changes_target_distribution",
                },
            },
            {
                "capability_id": TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                "rationale": "candidate must not change target distribution",
                "train_only_environment": {
                    "variations": [],
                    "final_distribution_policy": "candidate_changes_target_distribution",
                },
            },
        ]
        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "runner capability schema"):
                    validate_runner_capability_payload(str(payload["capability_id"]), payload)

    def test_runner_owned_readiness_reaches_context_builder_and_authoring_prompt_entrypoints(self) -> None:
        readiness = {
            CURRICULUM_DESIGN_CAPABILITY_ID: {
                "task_enabled": True,
                "compiler_ready": True,
                "verifier_ready": True,
            },
        }
        builder_payload = PublicContextBuilder().build_prompt_payload(
            [],
            runner_capability_readiness=readiness,
        )
        authoring_payload = build_authoring_prompt_payload(
            context_builder=PublicContextBuilder(),
            records=[],
            repair_feedback=None,
            latest_failed_candidate=None,
            runner_capability_readiness=readiness,
        )

        for payload in (builder_payload, authoring_payload):
            visible = {
                item["capability_id"]: item
                for item in payload["candidate_capabilities"]["prompt_visible_capabilities"]
            }
            self.assertIn(CURRICULUM_DESIGN_CAPABILITY_ID, visible)
            self.assertEqual(visible[CURRICULUM_DESIGN_CAPABILITY_ID]["exposure_mode"], "schema_preview_only")
            self.assertFalse(visible[CURRICULUM_DESIGN_CAPABILITY_ID]["candidate_writable"])

        default_authoring_payload = build_authoring_prompt_payload(
            context_builder=PublicContextBuilder(),
            records=[],
            repair_feedback=None,
            latest_failed_candidate=None,
        )
        default_visible_ids = {
            item["capability_id"]
            for item in default_authoring_payload["candidate_capabilities"]["prompt_visible_capabilities"]
        }
        self.assertNotIn(CURRICULUM_DESIGN_CAPABILITY_ID, default_visible_ids)


if __name__ == "__main__":
    unittest.main()

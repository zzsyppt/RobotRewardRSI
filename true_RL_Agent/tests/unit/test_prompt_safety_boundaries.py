from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import TEST_CAPABILITY_PROFILE  # noqa: E402
from true_rl_agent.agent_core import context as public_context_module  # noqa: E402
from true_rl_agent.candidates.prompt_projection import (  # noqa: E402
    assert_public_only_prompt_payload,
    build_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402
from true_rl_agent.tasks.base import (  # noqa: E402
    TaskInvariantSpec,
    TaskPlugin,
    TaskSpec,
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


class PromptSafetyBoundaryTests(unittest.TestCase):
    def test_public_key_policy_is_boundary_aware_and_task_neutral(self) -> None:
        ordinary = {
            "tokenizer": "v1",
            "tokenizer_version": "v1",
            "token_count": 12,
            "secretary_score": 0.8,
            "privateer_progress": 0.4,
            "hostname": "public-worker-label",
            "hostname_validated": True,
            "authorization_status": "succeeded",
            "proxyAuthorizationStatus": "not_configured",
            "x_authorization": "public-diagnostic-label",
            "auth_header": "omitted",
        }
        assert_public_only_prompt_payload(ordinary)
        self.assertEqual(public_context_module.sanitize_public_context(ordinary), ordinary)

        for key in (
            "api_key",
            "raw_seed_id",
            "trusted_eval",
            "private_seed",
            "hidden_eval",
            "secret_token",
            "accessToken",
            "APIKey",
            "APIKEY",
            "apikey",
            "SECRETKEY",
            "PRIVATESEED",
            "authorization",
            "Authorization",
            "proxy_authorization",
            "Proxy-Authorization",
            "proxyauthorization",
            "PROXYAUTHORIZATION",
            "PROXYAuthorization",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_public_only_prompt_payload({key: "must-not-pass"})
            self.assertEqual(
                public_context_module.sanitize_public_context(
                    {key: "must-not-pass"}
                ),
                {},
            )

    def test_shared_public_history_scrub_rules_are_task_agnostic(self) -> None:
        shared_key_parts = set(public_context_module._PUBLIC_HISTORY_BLOCKED_KEY_PARTS)
        shared_text_parts = set(public_context_module._PUBLIC_HISTORY_BLOCKED_TEXT_PARTS)

        self.assertNotIn("push_active", shared_key_parts)
        self.assertNotIn("push_vec_public", shared_key_parts)
        self.assertNotIn("push_active", shared_text_parts)
        self.assertNotIn("push_vec_public", shared_text_parts)

    def test_public_prompt_rejects_and_scrubs_private_hidden_trusted_and_raw_seed_data(self) -> None:
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"raw_seed_id": "public-dev-seed-001"}})

        payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-raw-safety",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "private_seed": 123,
                    "hidden_eval": {"success_rate": 1.0},
                    "trusted_seed": 456,
                    "trusted_eval": {"score": 1.0},
                    "checkpoint_selector": "best_trusted_checkpoint",
                    "raw_seed_id": "public-dev-seed-001",
                    "api_key": "sk-should-not-survive",
                }
            ]
        )
        text = serialize_prompt_payload(payload)

        for forbidden in (
            "private_seed",
            "hidden_eval",
            "trusted_seed",
            "trusted_eval",
            "checkpoint_selector",
            "raw_seed_id",
            "public-dev-seed-001",
            "api_key",
            "sk-should-not-survive",
        ):
            self.assertNotIn(forbidden, text)
        assert_public_only_prompt_payload(payload)

    def test_public_prompt_rejects_forbidden_markers_embedded_in_text_values(self) -> None:
        for marker in (
            "private_seed=123",
            "private seed 123",
            "hidden_eval=1.0",
            "hidden eval 1.0",
            "trusted_eval=0.99",
            "trusted eval 0.99",
            "trusted_seed=456",
            "trusted seed 456",
            "checkpoint_selector=best",
            "checkpoint selector best",
            "api_key=not-sk-format",
            "api key leaked",
            "raw_seed=123",
            "raw-seed=123",
            "raw_seed_id=public-dev-seed-001",
            "raw seed id public-dev-seed-001",
            "secret token leaked",
            "access token leaked",
            "Bearer abc123",
            "bEaReR AbC123+/=",
            "Authorization: Bearer abc123",
            "Proxy-Authorization: Bearer abc123",
            "request failed: Authorization: Bearer abc123",
            'request headers={"Authorization":"Basic abcdef"}',
            r'request headers={\"Authorization\":\"Basic abcdef\"}',
        ):
            with self.subTest(marker=marker):
                with self.assertRaises(ValueError):
                    assert_public_only_prompt_payload({"public_context": {"message": marker}})

        assert_public_only_prompt_payload({"public_context": {"message": "task-owned prompt adapter"}})
        assert_public_only_prompt_payload(
            {
                "public_context": {
                    "messages": [
                        "bearer of good news; authorization succeeded",
                        "bearer is ready",
                        "the bearer status remains healthy",
                        "Bearer protocol version is documented",
                        "not a bearer credential",
                        "the bond bearer presented proof",
                        "upstream sent Bearer abc123",
                        "upstream sent Bearer abcdef",
                        "execution_authorization: state succeeded",
                    ]
                }
            }
        )
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"message": "provider key sk-live-secret"}})

        for tuple_payload in (
            ("Bearer abcdef",),
            ({"Authorization": "Bearer abcdef"},),
            {"messages": ("Bearer abcdef",)},
        ):
            with self.subTest(tuple_payload=tuple_payload), self.assertRaises(ValueError):
                assert_public_only_prompt_payload(tuple_payload)

        assert_public_only_prompt_payload(
            {
                "task": {
                    "seed_visibility": "counts_only_no_raw_seed_ids",
                    "checkpoint_policy": "runner_sealed_train_output",
                    "note": "raw seed identifiers are omitted",
                    "parse_error": "invalid token at line 1",
                }
            }
        )

    def test_public_history_redacts_forbidden_markers_embedded_in_text_values(self) -> None:
        payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-hidden-marker-value",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "raw_seed=123 marker in reason",
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "runner_error": {
                        "type": "CandidateRuntimeError",
                        "message": "hidden_eval checkpoint_selector trusted_seed raw-seed marker in message",
                        "details": {
                            "status": "trusted_eval detail marker",
                            "code": "hidden_eval_code",
                        },
                    },
                }
            ]
        )
        text = serialize_prompt_payload(payload)
        history = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(history["reason"], "decision_record_invalid")
        self.assertEqual(history["runner_error"]["message"], "[redacted]")
        self.assertEqual(history["runner_error"]["details"]["status"], "[redacted]")
        self.assertEqual(history["runner_error"]["details"]["code"], "[redacted]")
        for forbidden in (
            "raw_seed=123",
            "raw-seed",
            "hidden_eval",
            "checkpoint_selector",
            "trusted_seed",
        ):
            self.assertNotIn(forbidden, text)
        assert_public_only_prompt_payload(payload)

    def test_public_history_preserves_nonsensitive_error_tokens(self) -> None:
        payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-json-token-error",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "provider_failure": {
                        "category": "json_parse_error",
                        "reason": "invalid token at line 1",
                        "details": {"status": "parse_failed"},
                    },
                }
            ]
        )
        history = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(history["provider_failure"]["reason"], "invalid token at line 1")
        assert_public_only_prompt_payload(payload)

    def test_non_push_prompt_does_not_inherit_push_recovery_task_semantics(self) -> None:
        payload = PublicContextBuilder(task_prompt_adapter=NonPushTaskPlugin().prompt_adapter()).build_prompt_payload([])
        text = serialize_prompt_payload(payload).lower()

        self.assertEqual(payload["task"]["task_id"], "stair_climbing")
        self.assertEqual(payload["reward_inputs_contract"].get("fields", []), [])
        self.assertNotIn("reward_inputs", payload)
        for forbidden in (
            "unitree_push_recovery",
            "push recovery",
            "push_active",
            "push_vec_public",
            "push_recovery_drift",
            "fall_forward",
            "fall_backward",
            "success_predicate",
        ):
            self.assertNotIn(forbidden, text)
        assert_public_only_prompt_payload(payload)

    def test_effective_controls_exclude_unwired_or_runner_owned_surfaces(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        text = serialize_prompt_payload(payload)
        controls = payload["control_effectiveness_contract"]

        visible_controls = [
            item["component_class"]
            for item in payload["candidate_capabilities"]["prompt_visible_capabilities"]
        ]
        self.assertEqual(visible_controls, ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"])
        self.assertNotIn("candidate_allowed_controls", payload)
        self.assertEqual(controls["schedule_controls"]["effective_keys"], ["reward_weight_scale"])
        self.assertNotIn("runner_owned_extensions", controls)
        for unwired in (
            "LossTerm",
            "ObservationTransform",
            "train_observation_transforms",
            "not_instrumented",
            "observation_projection",
            "feature_projection",
            "curriculum",
        ):
            self.assertNotIn(unwired, text)
        assert_public_only_prompt_payload(payload)

    def test_runner_owned_surfaces_are_denied_not_candidate_writable(self) -> None:
        payload = build_prompt_payload(public_context={"public_only": True})
        denied_text = json.dumps(
            payload["loop_contract"]["candidate_must_not_change"]
            + payload["required_candidate_package"]["candidate_must_not_change"],
            sort_keys=True,
        ).lower()
        may_text = json.dumps(payload["loop_contract"]["candidate_may"], sort_keys=True).lower()
        guidance_text = json.dumps(payload["selection_guidance"], sort_keys=True).lower()

        for surface in (
            "runner",
            "eval protocol",
            "training budget",
            "seed policy",
            "checkpoints",
            "ppo",
        ):
            self.assertIn(surface, denied_text)
            self.assertNotIn(surface, may_text)
        self.assertIn(
            "do not modify runner, eval protocol, training budget, seed policy, checkpoints, ppo",
            guidance_text,
        )
        assert_public_only_prompt_payload(payload)


if __name__ == "__main__":
    unittest.main()

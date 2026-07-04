from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.prompt_projection import (  # noqa: E402
    assert_public_only_prompt_payload,
    build_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec  # noqa: E402


class NonPushTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="stair_climbing",
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            invariants=TaskInvariantSpec({"episode_length": 400}),
        )


class PromptSafetyBoundaryTests(unittest.TestCase):
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
        ):
            with self.subTest(marker=marker):
                with self.assertRaises(ValueError):
                    assert_public_only_prompt_payload({"public_context": {"message": marker}})

        assert_public_only_prompt_payload({"public_context": {"message": "task-owned prompt adapter"}})
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"message": "provider key sk-live-secret"}})

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

        self.assertEqual(history["reason"], "[redacted]")
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
        self.assertEqual(payload["reward_inputs"]["fields"], [])
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

        self.assertEqual(payload["candidate_allowed_controls"], ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"])
        self.assertEqual(controls["schedule_controls"]["effective_keys"], ["reward_weight_scale"])
        self.assertEqual(controls["runner_owned_extensions"]["curriculum"], "not_prompt_writable_in_this_prompt_upgrade")
        self.assertEqual(controls["runner_owned_extensions"]["observation_projection"], "not_prompt_writable")
        for unwired in (
            "LossTerm",
            "ObservationTransform",
            "train_observation_transforms",
            "not_instrumented",
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

from __future__ import annotations

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
    assert_public_only_prompt_payload,
    build_prompt_payload,
    candidate_allowed_controls,
    reward_inputs_projection,
)
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.candidates.source_policy import SourceAccessPolicy


class PromptProjectionAndMechanismTests(unittest.TestCase):
    def test_prompt_projection_exposes_only_wired_controls_and_public_reward_inputs(self) -> None:
        payload = build_prompt_payload(public_context={"last_public_score": 0.1})

        self.assertIn("RewardTerm", candidate_allowed_controls())
        self.assertIn("DiagnosticTerm", candidate_allowed_controls())
        self.assertIn("ScheduleTerm", candidate_allowed_controls())
        self.assertNotIn("LossTerm", candidate_allowed_controls())
        self.assertNotIn("ObservationTransform", candidate_allowed_controls())
        self.assertIn("base_height", reward_inputs_projection()["fields"])
        self.assertNotIn("env", reward_inputs_projection()["fields"])
        self.assertNotIn("eval_config", str(payload))
        assert_public_only_prompt_payload(payload)

    def test_public_only_prompt_payload_rejects_private_hidden_or_secret_keys(self) -> None:
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public": True, "private_seed": 123})
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

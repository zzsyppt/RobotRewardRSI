from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.tasks.stair_climbing.candidate_surface import (
    build_stair_candidate_surface_policy,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairPhase0Bundle


class StairExtensionGuideTests(unittest.TestCase):
    def test_stair_phase_b_is_staged_without_changing_default_runtime(self) -> None:
        guide = PROJECT_ROOT / "docs" / "task_extension_guides" / "stair_climbing.md"
        fixture = PROJECT_ROOT / "configs" / "tasks" / "stair_extension_schema.json"
        text = guide.read_text(encoding="utf-8")
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        bundle = StairPhase0Bundle.load(PROJECT_ROOT)
        activation = bundle.protocol.to_dict()["activation"]

        stair_package = SRC_ROOT / "true_rl_agent" / "tasks" / "stair_climbing"
        self.assertTrue(stair_package.is_dir())
        self.assertTrue((stair_package / "phase0_contract.py").is_file())
        self.assertTrue((stair_package / "plugin.py").is_file())
        self.assertFalse(activation["runtime_plugin_enabled"])
        self.assertFalse(activation["training_command_enabled"])
        registry = build_application_composition_registry()
        self.assertEqual(registry.default_task_id, "unitree_push_recovery")
        self.assertEqual(registry.task_ids(), ("unitree_push_recovery",))
        self.assertEqual(payload["task_id"], "unitree_stair_climbing")
        self.assertEqual(bundle.protocol.protocol_id, "unitree_stair_climbing_v1")
        for required in (
            "TaskPlugin",
            "task config",
            "historical baseline manifest",
            "failure classifier",
            "metrics",
            "BackendAdapter",
            "AgentCore",
            "Runner",
            "runner-owned approved extension",
            "RewardTerm",
            "DiagnosticTerm",
            "complete_reward_program",
            "fixed random initialization",
        ):
            self.assertIn(required, text)

    def test_stair_descriptor_points_to_canonical_protocol_without_copying_semantics(self) -> None:
        fixture = PROJECT_ROOT / "configs" / "tasks" / "stair_extension_schema.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        bundle = StairPhase0Bundle.load(PROJECT_ROOT)
        protocol = bundle.protocol.to_dict()

        self.assertEqual(
            payload["task_protocol"]["protocol_hash"],
            bundle.protocol.protocol_hash,
        )
        self.assertEqual(
            [item["height_m"] for item in protocol["terrain"]["evaluation_bins"]],
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        )
        self.assertFalse(bundle.baseline.to_dict()["accepted_current_best"])
        self.assertNotIn("fixed_eval_bins", payload)
        self.assertNotIn("future_failure_taxonomy", payload)

    def test_future_stair_tuning_surfaces_are_not_code_level_accepts(self) -> None:
        policy = build_stair_candidate_surface_policy(
            StairPhase0Bundle.load(PROJECT_ROOT).protocol
        )

        for component in (
            "stair_height_range",
            "existing_reward_scalar_weight",
            "schedule",
        ):
            assessment = policy.classify(CandidateChangeSet(frozenset({component})))
            self.assertEqual(assessment.classification, "parameter_only")
            self.assertFalse(assessment.eligible_for_policy_accept)


if __name__ == "__main__":
    unittest.main()

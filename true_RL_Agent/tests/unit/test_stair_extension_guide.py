from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.schema import CandidateChangeSet, CandidateSurfacePolicy


class StairExtensionGuideTests(unittest.TestCase):
    def test_stair_extension_is_documented_but_not_runtime_enabled(self) -> None:
        guide = PROJECT_ROOT / "docs" / "task_extension_guides" / "stair_climbing.md"
        fixture = PROJECT_ROOT / "configs" / "tasks" / "stair_extension_schema.json"
        text = guide.read_text(encoding="utf-8")
        payload = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertFalse((SRC_ROOT / "true_rl_agent" / "tasks" / "stair_climbing").exists())
        self.assertFalse(payload["runtime_plugin_enabled"])
        self.assertFalse(payload["training_command_enabled"])
        for required in (
            "TaskPlugin",
            "task config",
            "baseline manifest",
            "failure classifier",
            "metrics",
            "BackendAdapter",
            "AgentCore",
            "Runner",
            "DecisionEngine",
            "ledger",
            "runner-owned approved extension",
            "RewardTerm",
            "DiagnosticTerm",
        ):
            self.assertIn(required, text)

    def test_stair_schema_fixture_declares_baseline_eval_bins_and_runner_owned_projection(self) -> None:
        fixture = PROJECT_ROOT / "configs" / "tasks" / "stair_extension_schema.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["future_baseline"]["description"],
            "Isaac/Unitree G1 rough locomotion policy trained on stair_height 0.05-0.20 m",
        )
        self.assertEqual(
            [item["height_m"] for item in payload["fixed_eval_bins"]],
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        )
        self.assertEqual(
            payload["height_map_terrain_metadata"]["height_map_projection_owner"],
            "runner_owned_approved_extension",
        )
        self.assertFalse(payload["height_map_terrain_metadata"]["candidate_may_change_obs_action_space"])

    def test_future_stair_tuning_surfaces_are_not_code_level_accepts(self) -> None:
        policy = CandidateSurfacePolicy()

        for component in ("stair_height_range", "reward_weight", "schedule"):
            assessment = policy.classify(CandidateChangeSet(frozenset({component})))
            self.assertEqual(assessment.classification, "parameter_only")
            self.assertFalse(assessment.eligible_for_policy_accept)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)


class StairEvaluationSemanticsTests(unittest.TestCase):
    def test_active_and_historical_semantics_are_both_preserved(self) -> None:
        semantics_files = sorted(
            path.name
            for path in (ROOT / "configs" / "tasks").glob(
                "stair_evaluation_semantics_v*.json"
            )
        )
        self.assertEqual(
            semantics_files,
            [
                "stair_evaluation_semantics_v6.json",
                "stair_evaluation_semantics_v7.json",
            ],
        )

    def test_active_semantics_are_bound_without_rewriting_historical_protocols(
        self,
    ) -> None:
        protocol_path = ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        active = StairClimbingTaskConfig.defaults(protocol_path)

        self.assertEqual(
            active.protocol_v1.protocol_hash,
            "7732f94502b30cdd8524b1e9ad4a1dbb118bb8f6b7b2f60cd1d52e8901b61cde",
        )
        self.assertEqual(
            active.protocol_v2.protocol_hash,
            "97d9473e4ba6ad5b9829fab26ad47d6d4bb89e9ad711d198629b17ccc8e153f5",
        )
        self.assertIsNotNone(active.evaluation_semantics)
        assert active.evaluation_semantics is not None
        self.assertEqual(
            active.evaluation_semantics.protocol_hash,
            "df35aa0fb8c24983570e2c634fd0cc40f8d39beefc191d400c37f099aa8cb766",
        )
        self.assertEqual(
            active.task_protocol_hash,
            "d82aae04f46d98b02cbd4848f4692f370bb419af5b6ee16aa34e108b954a56c9",
        )
        self.assertEqual(
            active.acceptance_protocol.protocol_hash,
            "9fccce3984eed8f04602a07511d7afb79cedf8dc8732fe56b7a18d804ef26382",
        )
        self.assertEqual(
            active.evaluation_profile.profile_hash,
            "86d8eed0c8b9a8c32af49c69a81d983b3ec826abec408d5664717f61e16939d5",
        )
        self.assertEqual(active.acceptance_protocol.guard_max_regression, 0.05)
        self.assertEqual(active.acceptance_protocol.guard_regression_margin, 0.02)
        self.assertEqual(
            active.eval_protocol_version,
            "stair_anchor_and_tread_generalization_v5",
        )
        self.assertEqual(
            active.episode_outcomes()["signals"]["timeout"],
            {
                "deadline_authority": "stair_evaluation_profile",
                "source": "evaluation_profile_cell_episode_horizon",
                "unit": "s",
            },
        )
        self.assertEqual(
            active.episode_outcomes()["signals"]["fall"][
                "maximum_projected_surface_penetration_m"
            ],
            0.03,
        )
        self.assertNotEqual(active.task_protocol_hash, active.protocol_v2.protocol_hash)
        self.assertEqual(
            active.episode_outcomes()["success"]["completion_mode"],
            "stable_top_traversal_after_first_valid_top_arrival",
        )
        self.assertEqual(
            active.episode_outcomes()["top_platform_stability_requirement"][
                "acceptance_role"
            ],
            "primary_success_required",
        )
        self.assertIn(
            "first_valid_top_arrival_achieved",
            active.episode_outcomes()["diagnostic_flags"],
        )

        self.assertIn(
            "forbidden_contact",
            active.episode_outcomes()["primary_reasons"],
        )
        self.assertIn(
            "base_contact",
            active.protocol_v1.to_dict()["episode_outcomes"]["primary_reasons"],
        )

    def test_final_config_cannot_disable_active_evaluation_semantics(self) -> None:
        protocol_path = ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            StairClimbingTaskConfig.defaults(
                protocol_path,
                enable_revised_evaluation=False,
            )


if __name__ == "__main__":
    unittest.main()

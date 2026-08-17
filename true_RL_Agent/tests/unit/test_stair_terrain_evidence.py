from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.phase0_contract import (  # noqa: E402
    StairProtocolV1,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (  # noqa: E402
    expected_stair_surface_height_m,
    supporting_stair_surface_height_m,
)


class StairTerrainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = StairProtocolV1.load(
            ROOT / "configs" / "tasks" / "stair_protocol_v1.json"
        ).to_dict()

    def test_surface_levels_follow_frozen_bottom_origin_geometry(self) -> None:
        cases = {
            -0.1: 0.0,
            0.0: 0.0,
            0.75: 0.0,
            0.750001: 0.2,
            1.15: 0.2,
            1.150001: 0.4,
            3.95: 1.6,
            20.0: 1.6,
        }
        for progress, expected in cases.items():
            with self.subTest(progress=progress):
                self.assertAlmostEqual(
                    expected_stair_surface_height_m(
                        progress_m=progress,
                        step_height_m=0.2,
                        protocol=self.protocol,
                    ),
                    expected,
                )

    def test_nonfinite_or_nonpositive_geometry_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            expected_stair_surface_height_m(
                progress_m=float("nan"),
                step_height_m=0.2,
                protocol=self.protocol,
            )
        with self.assertRaises(ValueError):
            expected_stair_surface_height_m(
                progress_m=0.0,
                step_height_m=0.0,
                protocol=self.protocol,
            )

    def test_riser_crossing_uses_contacting_feet_not_root_projection(self) -> None:
        support_height = supporting_stair_surface_height_m(
            foot_progress_m=(3.14, 3.20),
            foot_contacts=(True, True),
            foot_surface_clearance_m=(0.0, 0.0),
            step_height_m=0.35,
            maximum_projected_surface_penetration_m=0.03,
            protocol=self.protocol,
        )

        self.assertAlmostEqual(support_height, 6 * 0.35)
        root_height = support_height + 0.74
        old_root_projected_surface = expected_stair_surface_height_m(
            progress_m=3.20,
            step_height_m=0.35,
            protocol=self.protocol,
        )
        self.assertAlmostEqual(root_height - support_height, 0.74)
        self.assertAlmostEqual(root_height - old_root_projected_surface, 0.39)

    def test_no_contact_fallback_uses_lowest_projected_foot_tread(self) -> None:
        self.assertAlmostEqual(
            supporting_stair_surface_height_m(
                foot_progress_m=(3.14, 3.20),
                foot_contacts=(False, False),
                foot_surface_clearance_m=(0.0, 0.0),
                step_height_m=0.35,
                maximum_projected_surface_penetration_m=0.03,
                protocol=self.protocol,
            ),
            6 * 0.35,
        )

    def test_riser_collision_cannot_promote_support_above_the_foot(self) -> None:
        self.assertAlmostEqual(
            supporting_stair_surface_height_m(
                foot_progress_m=(3.20, 3.22),
                foot_contacts=(True, True),
                foot_surface_clearance_m=(-0.35, -0.35),
                step_height_m=0.35,
                maximum_projected_surface_penetration_m=0.03,
                protocol=self.protocol,
            ),
            6 * 0.35,
        )


if __name__ == "__main__":
    unittest.main()

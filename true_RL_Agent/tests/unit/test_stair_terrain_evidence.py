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


if __name__ == "__main__":
    unittest.main()

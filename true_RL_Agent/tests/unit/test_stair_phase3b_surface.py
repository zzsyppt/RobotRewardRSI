from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig  # noqa: E402
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEnvTransition,
    StairEpisodeTracker,
)
from true_rl_agent.tasks.stair_climbing.rough_surface import (  # noqa: E402
    normalize_stair_surface_profile,
    rough_surface_offset_m,
    rough_surface_vertex_offset_m,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (  # noqa: E402
    IrregularFlatStairSurfaceQuery,
    RoughStairSurfaceQuery,
)


class StairPhase3BSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks/stair_unified_protocol_v1.json"
        )
        cls.suite = cls.config.ordered_evaluation_suite
        runtime = cls.suite.runtime_cells("public_dev")
        cls.flat = next(cell for cell in runtime if cell.template_id == "c1_id_02")
        cls.rough = next(
            cell for cell in runtime if cell.template_id == "c2_rough_02"
        )
        if (
            cls.flat.riser_heights_m != cls.rough.riser_heights_m
            or cls.flat.tread_depths_m != cls.rough.tread_depths_m
        ):
            raise AssertionError("paired Phase 3B cells must share their C1 skeleton")

    def test_profile_bound_and_piecewise_linear_vertex_identity(self) -> None:
        profile = dict(self.rough.surface_profile)
        amplitude = float(profile["amplitude_m"])
        nx = int(profile["longitudinal_vertices"])
        ny = int(profile["lateral_vertices"])
        for ix in range(nx):
            for iy in range(ny):
                vertex = rough_surface_vertex_offset_m(
                    profile,
                    tread_index=2,
                    longitudinal_index=ix,
                    lateral_index=iy,
                )
                interpolated = rough_surface_offset_m(
                    profile,
                    tread_index=2,
                    longitudinal_fraction=ix / (nx - 1),
                    lateral_fraction=iy / (ny - 1),
                )
                self.assertAlmostEqual(vertex, interpolated, places=12)
                self.assertLessEqual(abs(vertex), amplitude + 1.0e-12)
                if ix in {0, nx - 1} or iy in {0, ny - 1}:
                    self.assertAlmostEqual(vertex, 0.0, places=12)

        invalid = dict(profile)
        invalid["amplitude_m"] = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_stair_surface_profile(invalid)

    def test_surface_relative_clearance_keeps_failure_classification_exact(self) -> None:
        flat_query = IrregularFlatStairSurfaceQuery(self.flat.spec)
        rough_query = RoughStairSurfaceQuery(self.rough.spec)
        level = 2
        boundaries = self.flat.cumulative_tread_boundaries_m
        local_x = boundaries[level - 1] + 0.37 * self.flat.tread_depths_m[level - 1]
        progress = local_x - self.flat.start_platform_length_m / 2.0
        lateral = 0.17
        flat_surface = flat_query.sample(progress_m=progress, lateral_m=lateral)
        rough_surface = rough_query.sample(progress_m=progress, lateral_m=lateral)
        self.assertEqual(flat_surface.level, rough_surface.level)
        self.assertNotAlmostEqual(flat_surface.height_m, rough_surface.height_m, places=6)

        outcomes = []
        for cell, surface_height in (
            (self.flat, flat_surface.height_m),
            (self.rough, rough_surface.height_m),
        ):
            tracker = StairEpisodeTracker(
                None,
                self.config.evaluation_semantics,
                env_index=0,
                bin_id=cell.cell_id,
                evaluation_cell=cell,
            )
            outcome = None
            for step in range(1, 4):
                foot_z = surface_height + 0.02
                outcome = tracker.observe(
                    StairEnvTransition(
                        env_index=0,
                        bin_id=cell.cell_id,
                        episode_step=step,
                        progress_m=progress,
                        lateral_m=lateral,
                        base_height_m=surface_height + 0.75,
                        base_clearance_m=0.75,
                        abs_roll_rad=0.0,
                        abs_pitch_rad=0.0,
                        forbidden_contact=False,
                        support_contacts=(False, True),
                        support_progress_m=(progress, progress),
                        support_surface_levels=(level, level),
                        support_lateral_m=(lateral, lateral),
                        support_planar_speeds_mps=(0.0, 0.0),
                        support_surface_clearances_m=(
                            foot_z - surface_height,
                            0.08,
                        ),
                        support_upward_forces_n=(0.0, 100.0),
                        done=step == 3,
                        termination_terms=("time_out",) if step == 3 else (),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            self.assertIsNotNone(outcome)
            outcomes.append(outcome)
        self.assertEqual(outcomes[0].primary_reason, "insufficient_clearance")
        self.assertEqual(outcomes[0].primary_reason, outcomes[1].primary_reason)
        self.assertEqual(outcomes[0].diagnostic_flags, outcomes[1].diagnostic_flags)

    def test_fixed_world_clearance_drift_cannot_exceed_surface_amplitude(self) -> None:
        flat_query = IrregularFlatStairSurfaceQuery(self.flat.spec)
        rough_query = RoughStairSurfaceQuery(self.rough.spec)
        amplitude = float(self.rough.surface_profile["amplitude_m"])
        threshold = 0.03
        flips = 0
        for level in range(1, self.rough.step_count + 1):
            left = self.rough.cumulative_tread_boundaries_m[level - 1]
            depth = self.rough.tread_depths_m[level - 1]
            for u in (0.2, 0.4, 0.6, 0.8):
                progress = left + u * depth - self.rough.start_platform_length_m / 2.0
                for lateral in (-0.35, -0.1, 0.2, 0.4):
                    flat_height = flat_query.sample(
                        progress_m=progress, lateral_m=lateral
                    ).height_m
                    rough_height = rough_query.sample(
                        progress_m=progress, lateral_m=lateral
                    ).height_m
                    world_z = flat_height + threshold
                    flat_clearance = world_z - flat_height
                    rough_clearance = world_z - rough_height
                    self.assertLessEqual(
                        abs(rough_clearance - flat_clearance), amplitude + 1.0e-12
                    )
                    flips += (flat_clearance < threshold) != (
                        rough_clearance < threshold
                    )
        self.assertGreater(flips, 0)


if __name__ == "__main__":
    unittest.main()

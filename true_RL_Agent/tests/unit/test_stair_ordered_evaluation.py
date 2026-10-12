from __future__ import annotations

import copy
import json
import math
import statistics
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
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (  # noqa: E402
    STAIR_ORDERED_FORMAL_COHORTS,
)
from true_rl_agent.tasks.stair_climbing.ordered_summary import (  # noqa: E402
    build_ordered_stair_summary,
    validate_ordered_stair_summary,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_CALIBRATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
    StairOrderedEvaluationSeedResolver,
    assert_seed_namespaces_disjoint,
)
from true_rl_agent.tasks.stair_climbing.terrain_evidence import (  # noqa: E402
    IrregularFlatStairSurfaceQuery,
    RoughStairSurfaceQuery,
)


class StairOrderedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks/stair_unified_protocol_v1.json"
        )
        cls.suite = cls.config.ordered_evaluation_suite
        cls.semantics = cls.config.evaluation_semantics

    def test_unified_suite_has_one_bounded_ordered_runtime_view(self) -> None:
        self.assertEqual(len(self.suite.cells), 77)
        self.assertEqual(self.suite.evaluation_protocol.eval_num_envs, 72)
        self.assertEqual(self.suite.calibration_seed_count, 3)
        for cohort in STAIR_ORDERED_FORMAL_COHORTS:
            cells = self.suite.runtime_cells(cohort)
            self.assertEqual(len(cells), 24)
            self.assertEqual(
                tuple(cell.terrain_type_index for cell in cells), tuple(range(24))
            )
            self.assertEqual(len(self.suite.private_cell_assignment(cohort)), 72)
        cohorts = [
            {cell.cell_id for cell in self.suite.cells_for_cohort(cohort)}
            for cohort in ("calibration", *STAIR_ORDERED_FORMAL_COHORTS)
        ]
        for index, left in enumerate(cohorts):
            for right in cohorts[index + 1 :]:
                self.assertFalse(left & right)

    def test_c1_geometry_and_surface_query_follow_explicit_boundaries(self) -> None:
        cells = [
            cell
            for cell in self.suite.cells
            if cell.terrain_regime_id == "c1_irregular_flat"
        ]
        self.assertTrue(cells)
        for cell in cells:
            self.assertEqual(len(cell.riser_heights_m), 8)
            self.assertEqual(len(cell.tread_depths_m), 8)
            query = IrregularFlatStairSurfaceQuery(cell)
            origin_x = cell.start_platform_length_m / 2.0
            for level, boundary_x in enumerate(
                cell.cumulative_tread_boundaries_m[:-1]
            ):
                progress = boundary_x - origin_x
                self.assertEqual(
                    query.sample(progress_m=progress, lateral_m=0.0).level,
                    level,
                )
                self.assertEqual(
                    query.sample(
                        progress_m=progress + 1.0e-7, lateral_m=0.0
                    ).level,
                    level + 1,
                )
            self.assertAlmostEqual(
                query.sample(progress_m=100.0, lateral_m=0.0).height_m,
                cell.total_height_m,
            )

    def test_c2_query_is_bounded_and_returns_to_nominal_at_tread_edges(self) -> None:
        cells = [
            cell
            for cell in self.suite.cells
            if cell.terrain_regime_id == "c2_rough"
        ]
        self.assertTrue(cells)
        for cell in cells:
            query = RoughStairSurfaceQuery(cell)
            amplitude = float(cell.surface_profile["amplitude_m"])
            origin_x = cell.start_platform_length_m / 2.0
            observed_offsets = []
            for level in range(1, cell.step_count + 1):
                left = cell.cumulative_tread_boundaries_m[level - 1]
                right = cell.cumulative_tread_boundaries_m[level]
                nominal = cell.cumulative_riser_heights_m[level]
                for local_x in (left, (left + right) / 2.0, right):
                    observed = query.height_for_level_m(
                        level=level,
                        progress_m=local_x - origin_x,
                        lateral_m=0.13,
                    )
                    observed_offsets.append(observed - nominal)
                self.assertAlmostEqual(observed_offsets[-3], 0.0, places=12)
                self.assertAlmostEqual(observed_offsets[-1], 0.0, places=12)
            self.assertLessEqual(max(abs(value) for value in observed_offsets), amplitude)

    def test_seed_domain_binds_each_cohort(self) -> None:
        resolver = StairOrderedEvaluationSeedResolver(
            base_seed=12021,
            task_protocol_hash=self.config.task_protocol_hash,
            evaluation_suite=self.suite,
        )
        assert_seed_namespaces_disjoint(resolver)
        public = resolver.public_seed_record(
            split="public_accept",
            run_index=1,
            namespace=STAIR_FORMAL_EVAL_NAMESPACE,
        )
        sealed = resolver.public_seed_record(
            split="sealed_validation",
            run_index=1,
            namespace=STAIR_FORMAL_EVAL_NAMESPACE,
        )
        self.assertNotEqual(public["split_cohort_hash"], sealed["split_cohort_hash"])
        calibration = resolver.public_seed_record(
            split="calibration",
            run_index=1,
            namespace=STAIR_CALIBRATION_EVAL_NAMESPACE,
        )
        self.assertEqual(calibration["split_seed_count"], 3)
        self.assertEqual(calibration["cell_count"], 5)
        with self.assertRaisesRegex(ValueError, "namespace"):
            resolver.derive_seed(
                split="calibration",
                run_index=1,
                namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            )

    def test_ordered_summary_recomputes_cell_and_factor_evidence(self) -> None:
        assignment = self.suite.private_cell_assignment("public_dev")
        per_seed = []
        for seed_index in range(self.suite.evaluation_protocol.public_seed_count):
            outcomes = []
            for env_index, cell in enumerate(assignment):
                success = (env_index + seed_index) % 3 == 0
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success" if success else "timeout",
                        success=success,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",) if success else ()
                        ),
                        max_progress_m=cell.progress_goal_m if success else 0.5,
                        progress_fraction=1.0 if success else 0.1,
                        active_terminal_terms=() if success else ("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        summary = build_ordered_stair_summary(
            suite=self.suite,
            semantics=self.semantics,
            split="public_dev",
            per_seed_outcomes=per_seed,
        )
        self.assertEqual(summary["total_episode_count"], 3 * 24 * 3)
        self.assertEqual(
            set(summary["factor_diagnostics"]),
            {
                "phase",
                "riser_band",
                "tread_band",
                "sequence_pattern",
                "roughness_band",
                "surface_pattern",
            },
        )
        canonical_round_trip = json.loads(json.dumps(summary, sort_keys=True))
        self.assertEqual(
            validate_ordered_stair_summary(
                suite=self.suite,
                semantics=self.semantics,
                split="public_dev",
                summary=canonical_round_trip,
            ),
            canonical_round_trip,
        )
        one_ulp = copy.deepcopy(canonical_round_trip)
        one_ulp["score_stats"]["std"] = math.nextafter(
            one_ulp["score_stats"]["std"],
            math.inf,
        )
        with self.assertRaisesRegex(ValueError, "canonical score statistics"):
            validate_ordered_stair_summary(
                suite=self.suite,
                semantics=self.semantics,
                split="public_dev",
                summary=one_ulp,
            )
        # Read actual frozen v2 evidence instead of relabeling a current summary.
        frozen = json.loads(
            (
                ROOT / "tests/fixtures/stair_summaries/p5_legacy_public_dev.json"
            ).read_text(encoding="utf-8")
        )["g1"]
        legacy_config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks" / frozen["protocol"]
        )
        one_ulp = copy.deepcopy(frozen["summary"]["ordered_summary"])
        one_ulp["score_stats"]["std"] = math.nextafter(
            one_ulp["score_stats"]["std"], math.inf
        )
        self.assertEqual(
            validate_ordered_stair_summary(
                suite=legacy_config.ordered_evaluation_suite,
                semantics=legacy_config.evaluation_semantics,
                split="public_dev",
                summary=one_ulp,
            ),
            one_ulp,
        )
        changed_score_stats = copy.deepcopy(canonical_round_trip)
        changed_score_stats["score_stats"]["std"] += 2.0e-12
        with self.assertRaisesRegex(ValueError, "score statistics"):
            validate_ordered_stair_summary(
                suite=self.suite,
                semantics=self.semantics,
                split="public_dev",
                summary=changed_score_stats,
            )
        changed = copy.deepcopy(summary)
        first_cell = summary["cell_ids"][0]
        changed["aggregate_success_rate_by_cell"][first_cell] = 1.0
        with self.assertRaisesRegex(ValueError, "aggregate success rates"):
            validate_ordered_stair_summary(
                suite=self.suite,
                semantics=self.semantics,
                split="public_dev",
                summary=changed,
            )

        changed_diagnostics = copy.deepcopy(summary)
        changed_diagnostics["diagnostic_flag_counts_by_cell"][first_cell][
            "first_valid_top_arrival_achieved"
        ] += 1
        with self.assertRaisesRegex(ValueError, "aggregate diagnostic counts"):
            validate_ordered_stair_summary(
                suite=self.suite,
                semantics=self.semantics,
                split="public_dev",
                summary=changed_diagnostics,
            )

        legacy = copy.deepcopy(frozen["summary"]["ordered_summary"])
        legacy["schema_version"] = (
            "true_rl_agent.task.stair_climbing.ordered_evaluation_summary.v1"
        )
        legacy.pop("diagnostic_flag_counts_by_cell")
        for seed in legacy["seed_summaries"]:
            seed.pop("diagnostic_flag_counts_by_cell")
        self.assertEqual(
            validate_ordered_stair_summary(
                suite=legacy_config.ordered_evaluation_suite,
                semantics=legacy_config.evaluation_semantics,
                split="public_dev",
                summary=legacy,
            ),
            legacy,
        )

    def test_progress_aggregation_keeps_one_seed_macro_reduction(self) -> None:
        assignment = self.suite.private_cell_assignment("public_dev")
        groups = (
            tuple((index % 7) / 10.0 for index in range(len(assignment))),
            tuple(((index + 2) % 9) / 10.0 for index in range(len(assignment))),
            tuple(((index + 4) % 8) / 10.0 for index in range(len(assignment))),
        )
        per_seed = []
        for values in groups:
            outcomes = [
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=cell.cell_id,
                    episode_steps=cell.episode_horizon_control_steps,
                    primary_reason="timeout",
                    success=False,
                    diagnostic_flags=(),
                    max_progress_m=values[env_index] * cell.progress_goal_m,
                    progress_fraction=values[env_index],
                    active_terminal_terms=("time_out",),
                    evaluation_cell_id=cell.cell_id,
                )
                for env_index, cell in enumerate(assignment)
            ]
            per_seed.append(outcomes)
        summary = build_ordered_stair_summary(
            suite=self.suite,
            semantics=self.semantics,
            split="public_dev",
            per_seed_outcomes=per_seed,
        )
        first_cell = assignment[0].cell_id
        expected = statistics.fmean(
            statistics.fmean(values[:3]) for values in groups
        )
        self.assertEqual(
            summary["aggregate_mean_progress_fraction_by_cell"][first_cell],
            expected,
        )


if __name__ == "__main__":
    unittest.main()

"""Current arithmetic from episode facts, plus frozen reference readers."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.derived_statistics import (
    DERIVED_STATISTICS_VERSION,
    paired_standard_error,
    sample_mean,
    sample_statistics,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_score_projection,
    build_unified_stair_summary,
    validate_unified_stair_summary,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "tests/fixtures/stair_summaries/p5_legacy_public_dev.json"


def current_episode_identities() -> dict:
    """Portable probe: regenerate every current aggregate from identical episodes."""
    rows = {}
    for robot, fixture in json.loads(FROZEN.read_text(encoding="utf-8")).items():
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks" / fixture["protocol"]
        )
        suite = config.ordered_evaluation_suite
        seeds = []
        for seed_index in range(config.public_seed_count):
            outcomes = []
            for index, cell in enumerate(suite.private_cell_assignment("public_dev")):
                success = (index + seed_index) % 5 < 2
                progress = (
                    1.0 if success else ((index * 7 + seed_index * 3) % 19) / 19.0
                )
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success" if success else "timeout",
                        success=success,
                        diagnostic_flags=(),
                        max_progress_m=progress * cell.progress_goal_m,
                        progress_fraction=progress,
                        active_terminal_terms=() if success else ("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            seeds.append(outcomes)
        summary = build_unified_stair_summary(
            suite=suite,
            semantics=config.evaluation_semantics,
            split="public_dev",
            per_seed_outcomes=seeds,
        )
        restored = validate_unified_stair_summary(
            suite=suite,
            semantics=config.evaluation_semantics,
            split="public_dev",
            summary=json.loads(json.dumps(summary)),
        )
        if restored != summary:
            raise AssertionError("current summary changed during validation")
        projection = build_unified_score_projection(
            summary,
            objective_family_mask=fixture["objective_family_mask"],
        )
        rows[robot] = {
            "raw_episode_hash": canonical_json_hash(
                [[asdict(outcome) for outcome in outcomes] for outcomes in seeds]
            ),
            "summary_hash": canonical_json_hash(summary),
            "projection_hash": canonical_json_hash(projection),
            "score_stats": projection["j_primary"]["score_stats"],
        }
    return rows


class DerivedStatisticsTests(unittest.TestCase):
    def test_paired_error_uses_current_default_and_requires_samples(self) -> None:
        values = [0.1, 0.3, 0.8]
        self.assertEqual(
            paired_standard_error(values), sample_statistics(values)["standard_error"]
        )
        for version in (None, DERIVED_STATISTICS_VERSION):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    paired_standard_error([], statistics_version=version)

    def test_current_episode_pipeline_does_not_use_legacy_statistics(self) -> None:
        with (
            patch("statistics.fmean", side_effect=AssertionError("legacy mean")),
            patch("statistics.stdev", side_effect=AssertionError("legacy stdev")),
        ):
            rows = current_episode_identities()
        self.assertEqual(set(rows), {"g1", "go2"})
        self.assertEqual(sample_mean([1e16, 1.0, -1e16]), 1.0 / 3.0)

    def test_frozen_reference_summaries_preserve_identity(self) -> None:
        for robot, fixture in json.loads(FROZEN.read_text(encoding="utf-8")).items():
            with self.subTest(robot=robot):
                config = StairClimbingTaskConfig.defaults(
                    ROOT / "configs/tasks" / fixture["protocol"]
                )
                original = fixture["summary"]
                restored = validate_unified_stair_summary(
                    suite=config.ordered_evaluation_suite,
                    semantics=config.evaluation_semantics,
                    split="public_dev",
                    summary=original,
                )
                self.assertEqual(
                    canonical_json_hash(restored), canonical_json_hash(original)
                )
                unknown = json.loads(json.dumps(original))
                unknown["ordered_summary"]["schema_version"] = (
                    "unsupported-ordered-format"
                )
                with self.assertRaisesRegex(ValueError, "ordered summary schema"):
                    build_unified_score_projection(unknown)


if __name__ == "__main__":
    unittest.main()

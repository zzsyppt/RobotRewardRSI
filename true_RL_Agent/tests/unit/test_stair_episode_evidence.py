from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEnvTransition,
    StairEpisodeEvidenceError,
    StairEpisodeOutcome,
    StairEpisodeTracker,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (  # noqa: E402
    StairProtocolV1,
)


class StairEpisodeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = StairProtocolV1.load(
            ROOT / "configs" / "tasks" / "stair_protocol_v1.json"
        )
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        assert cls.config.evaluation_semantics is not None
        cls.semantics = cls.config.evaluation_semantics
        cls.bin_id = str(cls.protocol.eval_bins[0]["bin_id"])
        cls.goal_m = float(
            cls.protocol.to_dict()["terrain"]["goal"]["progress_goal_m_from_origin"]
        )

    def test_terminal_transition_state_is_counted_before_reset(self) -> None:
        tracker = self._tracker()

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                progress_m=0.8,
                done=True,
                termination_terms=("base_contact",),
                forbidden_contact=True,
            )
        )

        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.max_progress_m, 0.8)
        self.assertEqual(outcome.primary_reason, "forbidden_contact")

    def test_progress_is_floored_at_the_protocol_course_start(self) -> None:
        tracker = self._tracker()

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                progress_m=-0.08,
                done=True,
                termination_terms=("base_contact",),
                forbidden_contact=True,
            )
        )

        assert outcome is not None
        self.assertEqual(outcome.max_progress_m, 0.0)
        self.assertEqual(outcome.progress_fraction, 0.0)

    def test_done_and_term_masks_are_one_contract(self) -> None:
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "done mask does not match",
        ):
            _transition(bin_id=self.bin_id, step=1, done=True)
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "done mask does not match",
        ):
            _transition(
                bin_id=self.bin_id,
                step=1,
                done=False,
                termination_terms=("time_out",),
            )

    def test_forbidden_contact_signal_and_terminal_term_are_one_contract(self) -> None:
        inconsistent = (
            {
                "done": True,
                "termination_terms": ("base_contact",),
                "forbidden_contact": False,
            },
            {
                "done": True,
                "termination_terms": ("time_out",),
                "forbidden_contact": True,
            },
        )
        for overrides in inconsistent:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    StairEpisodeEvidenceError,
                    "forbidden-contact signal disagrees",
                ):
                    self._tracker().observe(
                        _transition(
                            bin_id=self.bin_id,
                            step=1,
                            **overrides,
                        )
                    )

    def test_simultaneous_timeout_and_contact_follow_frozen_priority(self) -> None:
        tracker = self._tracker()

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                done=True,
                termination_terms=("time_out", "base_contact"),
                forbidden_contact=True,
            )
        )

        assert outcome is not None
        self.assertEqual(outcome.primary_reason, "forbidden_contact")
        self.assertEqual(outcome.active_terminal_terms, ("time_out", "base_contact"))

    def test_unknown_manager_term_fails_closed(self) -> None:
        tracker = self._tracker()
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "unmapped terms",
        ):
            tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=1,
                    done=True,
                    termination_terms=("future_fall_term",),
                )
            )

    def test_stable_top_traversal_closes_after_twenty_steps(self) -> None:
        tracker = self._tracker()
        outcome = None
        for step in range(1, 21):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                )
            )

        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.primary_reason, "success")
        self.assertEqual(outcome.episode_steps, 20)
        self.assertEqual(outcome.active_terminal_terms, ())
        self.assertNotIn("goal_not_reached_detected", outcome.diagnostic_flags)
        self.assertIn(
            "first_valid_top_arrival_achieved",
            outcome.diagnostic_flags,
        )
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "already complete",
        ):
            tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=21,
                    progress_m=7.3,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                )
            )

    def test_first_arrival_v2_records_recovered_low_clearance_without_blocking(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_unified_protocol_v2.json"
        )
        assert config.evaluation_semantics is not None
        cell = config.ordered_evaluation_suite.runtime_cells("public_dev")[0]
        tracker = StairEpisodeTracker(
            None,
            config.evaluation_semantics,
            env_index=0,
            bin_id=cell.bin_id,
            evaluation_cell=cell,
        )

        for step in range(1, 6):
            self.assertIsNone(
                tracker.observe(
                    _transition(
                        bin_id=cell.bin_id,
                        step=step,
                        evaluation_cell_id=cell.cell_id,
                        root_height_m=0.4,
                        root_clearance_m=0.2,
                    )
                )
            )

        outcome = tracker.observe(
            _transition(
                bin_id=cell.bin_id,
                step=6,
                evaluation_cell_id=cell.cell_id,
                progress_m=cell.progress_goal_m,
                root_height_m=cell.total_height_m + 0.8,
                root_clearance_m=0.8,
                foot_progress_m=(cell.progress_goal_m, cell.progress_goal_m),
                foot_surface_levels=(cell.step_count, cell.step_count),
                done=True,
                termination_terms=("time_out",),
            )
        )

        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.primary_reason, "success")
        self.assertEqual(outcome.episode_steps, 6)
        self.assertNotIn("low_base_clearance_detected", outcome.diagnostic_flags)
        self.assertIn("first_valid_top_arrival_achieved", outcome.diagnostic_flags)
        self.assertNotIn(
            "post_arrival_safety_failure_detected",
            config.episode_outcomes()["diagnostic_flags"],
        )

        failed_tracker = StairEpisodeTracker(
            None,
            config.evaluation_semantics,
            env_index=0,
            bin_id=cell.bin_id,
            evaluation_cell=cell,
        )
        failed_outcome = None
        for step in range(1, 6):
            failed_outcome = failed_tracker.observe(
                _transition(
                    bin_id=cell.bin_id,
                    step=step,
                    evaluation_cell_id=cell.cell_id,
                    root_height_m=0.4,
                    root_clearance_m=0.2,
                    done=step == 5,
                    termination_terms=("time_out",) if step == 5 else (),
                )
            )
        assert failed_outcome is not None
        self.assertFalse(failed_outcome.success)
        self.assertEqual(failed_outcome.primary_reason, "timeout")
        self.assertNotIn("low_base_clearance_detected", failed_outcome.diagnostic_flags)

    def test_go2_legal_arrival_clearance_is_independent_of_running_gait(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT
            / "configs"
            / "tasks"
            / "stair_go2_12dof_unified_protocol_v2.json"
        )
        assert config.evaluation_semantics is not None
        cell = config.ordered_evaluation_suite.runtime_cells("public_dev")[0]
        tracker = StairEpisodeTracker(
            None,
            config.evaluation_semantics,
            env_index=0,
            bin_id=cell.bin_id,
            evaluation_cell=cell,
        )

        for step in range(1, 6):
            self.assertIsNone(
                tracker.observe(
                    _transition(
                        bin_id=cell.bin_id,
                        step=step,
                        evaluation_cell_id=cell.cell_id,
                        root_height_m=0.2,
                        root_clearance_m=0.2,
                    )
                )
            )
        outcome = tracker.observe(
            _transition(
                bin_id=cell.bin_id,
                step=6,
                evaluation_cell_id=cell.cell_id,
                progress_m=cell.progress_goal_m,
                root_height_m=cell.total_height_m + 0.3,
                root_clearance_m=0.3,
                foot_progress_m=(cell.progress_goal_m, cell.progress_goal_m),
                foot_surface_levels=(cell.step_count, cell.step_count),
            )
        )

        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertNotIn("low_base_clearance_detected", outcome.diagnostic_flags)

    def test_safety_failure_on_twentieth_stable_step_blocks_success(self) -> None:
        tracker = self._tracker()
        for step in range(1, 20):
            self.assertIsNone(
                tracker.observe(
                    _transition(
                        bin_id=self.bin_id,
                        step=step,
                        progress_m=self.goal_m,
                        root_height_m=1.55,
                        root_clearance_m=0.75,
                        foot_surface_levels=(8, 8),
                    )
                )
            )

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=20,
                progress_m=self.goal_m,
                root_height_m=1.55,
                root_clearance_m=0.75,
                foot_surface_levels=(8, 8),
                done=True,
                termination_terms=("base_contact",),
                forbidden_contact=True,
            )
        )

        assert outcome is not None
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.primary_reason, "forbidden_contact")
        self.assertIn(
            "first_valid_top_arrival_achieved",
            outcome.diagnostic_flags,
        )
        self.assertIn(
            "post_arrival_safety_failure_detected",
            outcome.diagnostic_flags,
        )

    def test_nineteen_stable_top_steps_at_deadline_is_timeout(self) -> None:
        tracker = self._tracker()
        outcome = None
        for step in range(1, 20):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                    done=step == 19,
                    termination_terms=("time_out",) if step == 19 else (),
                )
            )

        assert outcome is not None
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.primary_reason, "timeout")
        self.assertIn("first_valid_top_arrival_achieved", outcome.diagnostic_flags)

    def test_unstable_step_resets_stable_top_success_streak(self) -> None:
        tracker = self._tracker()
        for step in range(1, 31):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                    foot_planar_speeds_mps=((0.8, 0.8) if step == 11 else (0.0, 0.0)),
                )
            )
            self.assertIsNone(outcome)

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=31,
                progress_m=self.goal_m,
                root_height_m=1.55,
                root_clearance_m=0.75,
                foot_surface_levels=(8, 8),
            )
        )
        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.episode_steps, 31)

    def test_failure_signal_streaks_are_not_single_step_diagnostics(self) -> None:
        tracker = self._tracker()
        outcome = None
        for step in range(1, 6):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    root_clearance_m=0.4,
                    done=step == 5,
                    termination_terms=("time_out",) if step == 5 else (),
                )
            )

        assert outcome is not None
        self.assertEqual(outcome.primary_reason, "fall")

    def test_progress_alone_cannot_satisfy_physical_top_arrival(self) -> None:
        cases = (
            {"abs_pitch_rad": 0.9},
            {"foot_contacts": (False, False)},
            {"foot_upward_forces_n": (0.0, 0.0)},
            {"foot_upward_forces_n": (5.0, 5.0)},
            {"foot_surface_clearances_m": (-0.3, -0.3)},
            {"foot_surface_clearances_m": (0.5, 0.5)},
            {"root_height_m": 1.0},
            {"lateral_m": 2.1},
            {"foot_surface_levels": (7, 7)},
            {"foot_lateral_m": (2.1, 2.1)},
            {"foot_progress_m": (7.3, 7.3)},
            {"progress_m": 7.3},
        )
        for override in cases:
            with self.subTest(override=override):
                tracker = self._tracker()
                outcome = None
                for step in range(1, 21):
                    transition_kwargs = {
                        "bin_id": self.bin_id,
                        "step": step,
                        "progress_m": self.goal_m,
                        "root_height_m": 1.55,
                        "root_clearance_m": 0.75,
                        "foot_surface_levels": (8, 8),
                        "done": step == 20,
                        "termination_terms": (("time_out",) if step == 20 else ()),
                    }
                    transition_kwargs.update(override)
                    outcome = tracker.observe(_transition(**transition_kwargs))
                assert outcome is not None
                self.assertFalse(outcome.success)

    def test_top_arrival_without_stability_is_timeout(self) -> None:
        tracker = self._tracker()

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                progress_m=self.goal_m,
                root_height_m=1.55,
                root_clearance_m=0.75,
                foot_surface_levels=(8, 8),
                foot_planar_speeds_mps=(0.8, 0.8),
                done=True,
                termination_terms=("time_out",),
            )
        )

        assert outcome is not None
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.primary_reason, "timeout")
        self.assertNotIn("goal_not_reached_detected", outcome.diagnostic_flags)
        self.assertIn(
            "first_valid_top_arrival_achieved",
            outcome.diagnostic_flags,
        )

    def test_post_arrival_off_course_blocks_stable_success(self) -> None:
        tracker = self._tracker()
        self.assertIsNone(
            tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=1,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                )
            )
        )

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=2,
                progress_m=7.3,
                root_height_m=1.55,
                root_clearance_m=0.75,
                foot_surface_levels=(8, 8),
                done=True,
                termination_terms=("time_out",),
            )
        )

        assert outcome is not None
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.primary_reason, "off_course")
        self.assertIn("first_valid_top_arrival_achieved", outcome.diagnostic_flags)
        self.assertIn("off_course_detected", outcome.diagnostic_flags)
        self.assertIn(
            "post_arrival_safety_failure_detected",
            outcome.diagnostic_flags,
        )

    def test_pre_completion_off_course_still_blocks_success(self) -> None:
        tracker = self._tracker()
        self.assertIsNone(
            tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=1,
                    lateral_m=2.1,
                )
            )
        )

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=2,
                progress_m=self.goal_m,
                root_height_m=1.55,
                root_clearance_m=0.75,
                foot_surface_levels=(8, 8),
                done=True,
                termination_terms=("time_out",),
            )
        )

        assert outcome is not None
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.primary_reason, "off_course")

    def test_minimum_upward_support_force_is_inclusive(self) -> None:
        tracker = self._tracker()
        outcome = None
        for step in range(1, 21):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                    foot_upward_forces_n=(10.0, 10.0),
                )
            )
        assert outcome is not None
        self.assertTrue(outcome.success)

    def test_top_height_multiplier_is_applied_for_every_bin(self) -> None:
        for raw_bin in self.protocol.eval_bins:
            bin_id = str(raw_bin["bin_id"])
            step_height = float(raw_bin["height_m"])
            tracker = StairEpisodeTracker(
                self.protocol,
                self.semantics,
                env_index=0,
                bin_id=bin_id,
            )
            outcome = None
            for step in range(1, 21):
                outcome = tracker.observe(
                    _transition(
                        bin_id=bin_id,
                        step=step,
                        progress_m=self.goal_m,
                        root_height_m=8 * step_height + 0.55,
                        root_clearance_m=0.55,
                        foot_surface_levels=(8, 8),
                    )
                )
            assert outcome is not None
            self.assertTrue(outcome.success, bin_id)

    def test_leaving_own_terrain_patch_is_a_primary_failure(self) -> None:
        tracker = self._tracker()
        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                lateral_m=2.01,
                done=True,
                termination_terms=("time_out",),
            )
        )

        assert outcome is not None
        self.assertEqual(outcome.primary_reason, "off_course")
        self.assertIn("off_course_detected", outcome.diagnostic_flags)

    def test_profile_cell_timeout_uses_distance_scaled_control_step(self) -> None:
        cell = next(
            cell
            for cell in self.config.evaluation_profile.cells
            if cell.bin_id == self.bin_id and cell.tread_depth_m == 0.3
        )
        tracker = StairEpisodeTracker(
            self.protocol,
            self.semantics,
            env_index=0,
            bin_id=self.bin_id,
            evaluation_cell=cell,
        )

        outcome = None
        for step in range(1, cell.episode_horizon_control_steps + 1):
            outcome = tracker.observe(
                replace(
                    _transition(bin_id=self.bin_id, step=step),
                    evaluation_cell_id=cell.cell_id,
                )
            )
            if step < cell.episode_horizon_control_steps:
                self.assertIsNone(outcome)

        assert outcome is not None
        self.assertEqual(outcome.primary_reason, "timeout")
        self.assertEqual(outcome.episode_steps, 490)
        self.assertIn("goal_not_reached_detected", outcome.diagnostic_flags)

    def test_recovered_orientation_instability_is_visible_on_success(self) -> None:
        tracker = self._tracker()
        for step in range(1, 11):
            self.assertIsNone(
                tracker.observe(
                    _transition(
                        bin_id=self.bin_id,
                        step=step,
                        abs_roll_rad=0.9,
                    )
                )
            )
        outcome = None
        for step in range(11, 31):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    root_height_m=1.55,
                    root_clearance_m=0.75,
                    foot_surface_levels=(8, 8),
                )
            )
        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertIn(
            "base_orientation_instability_detected",
            outcome.diagnostic_flags,
        )

    def test_recovered_signals_are_diagnostics_but_timeout_remains_primary(
        self,
    ) -> None:
        cases = (
            (
                "base_orientation_instability",
                10,
                {"abs_roll_rad": 0.9},
                "base_orientation_instability_detected",
            ),
            (
                "slip",
                5,
                {"foot_planar_speeds_mps": (0.8, 0.0)},
                "slip_detected",
            ),
            (
                "insufficient_clearance",
                3,
                {
                    "foot_contacts": (False, True),
                    "foot_surface_clearances_m": (0.0, 0.05),
                },
                "insufficient_clearance_detected",
            ),
        )
        for name, active_steps, active_override, diagnostic in cases:
            with self.subTest(signal=name):
                tracker = self._tracker()
                for step in range(1, active_steps + 1):
                    self.assertIsNone(
                        tracker.observe(
                            _transition(
                                bin_id=self.bin_id,
                                step=step,
                                **active_override,
                            )
                        )
                    )
                outcome = tracker.observe(
                    _transition(
                        bin_id=self.bin_id,
                        step=active_steps + 1,
                        done=True,
                        termination_terms=("time_out",),
                    )
                )
                assert outcome is not None
                self.assertEqual(outcome.primary_reason, "timeout")
                self.assertIn(diagnostic, outcome.diagnostic_flags)

    def test_recoverable_signal_must_remain_active_at_timeout_to_be_primary(
        self,
    ) -> None:
        cases = (
            (
                "base_orientation_instability",
                10,
                {"abs_roll_rad": 0.9},
            ),
            (
                "slip",
                5,
                {"foot_planar_speeds_mps": (0.8, 0.0)},
            ),
            (
                "insufficient_clearance",
                3,
                {
                    "foot_contacts": (False, True),
                    "foot_surface_clearances_m": (0.0, 0.05),
                },
            ),
        )
        for reason, active_steps, active_override in cases:
            with self.subTest(signal=reason):
                tracker = self._tracker()
                outcome = None
                for step in range(1, active_steps + 1):
                    outcome = tracker.observe(
                        _transition(
                            bin_id=self.bin_id,
                            step=step,
                            done=step == active_steps,
                            termination_terms=("time_out",)
                            if step == active_steps
                            else (),
                            **active_override,
                        )
                    )
                assert outcome is not None
                self.assertEqual(outcome.primary_reason, reason)

    def _tracker(self) -> StairEpisodeTracker:
        return StairEpisodeTracker(
            self.protocol,
            self.semantics,
            env_index=0,
            bin_id=self.bin_id,
        )

    def test_split_summary_requires_exact_env_and_bin_closure(self) -> None:
        per_seed = [_complete_seed_outcomes(self.protocol) for _ in range(3)]

        summary = build_stair_split_summary(
            self.protocol,
            semantics=self.semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_dev",
            per_seed_outcomes=per_seed,
        )

        self.assertEqual(summary["score_stats"]["count"], 3)
        self.assertEqual(summary["score_stats"]["mean"], 1.0)
        self.assertEqual(summary["total_episode_count"], 168)
        self.assertEqual(summary["classification_warning_count"], 0)

        duplicate_env = list(per_seed[0])
        duplicate_env[1] = replace(duplicate_env[1], env_index=0)
        with self.assertRaisesRegex(ValueError, "env identity coverage"):
            build_stair_split_summary(
                self.protocol,
                semantics=self.semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_dev",
                per_seed_outcomes=[duplicate_env, per_seed[1], per_seed[2]],
            )

        unknown_diagnostic = list(per_seed[0])
        unknown_diagnostic[0] = replace(
            unknown_diagnostic[0],
            diagnostic_flags=(
                "first_valid_top_arrival_achieved",
                "private_probe",
            ),
        )
        with self.assertRaisesRegex(ValueError, "undeclared diagnostic"):
            build_stair_split_summary(
                self.protocol,
                semantics=self.semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_dev",
                per_seed_outcomes=[
                    unknown_diagnostic,
                    per_seed[1],
                    per_seed[2],
                ],
            )

        missing_arrival = list(per_seed[0])
        missing_arrival[0] = replace(
            missing_arrival[0],
            diagnostic_flags=(),
        )
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "stable-top success requires first valid top arrival evidence",
        ):
            build_stair_split_summary(
                self.protocol,
                semantics=self.semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_dev",
                per_seed_outcomes=[
                    missing_arrival,
                    per_seed[1],
                    per_seed[2],
                ],
            )

    def test_post_arrival_diagnostic_requires_arrival_evidence(self) -> None:
        with self.assertRaisesRegex(
            StairEpisodeEvidenceError,
            "post-arrival safety failure requires top arrival evidence",
        ):
            StairEpisodeOutcome(
                env_index=0,
                bin_id=self.bin_id,
                episode_steps=1,
                primary_reason="timeout",
                success=False,
                diagnostic_flags=("post_arrival_safety_failure_detected",),
                max_progress_m=self.goal_m,
                progress_fraction=1.0,
                active_terminal_terms=("time_out",),
            )


def _transition(
    *,
    bin_id: str,
    step: int,
    progress_m: float = 0.0,
    lateral_m: float = 0.0,
    root_height_m: float = 0.9,
    root_clearance_m: float = 0.9,
    abs_roll_rad: float = 0.0,
    abs_pitch_rad: float = 0.0,
    done: bool = False,
    termination_terms: tuple[str, ...] = (),
    forbidden_contact: bool = False,
    evaluation_cell_id: str | None = None,
    foot_contacts: tuple[bool, bool] = (True, True),
    foot_progress_m: tuple[float, float] = (4.35, 4.35),
    foot_surface_levels: tuple[int, int] = (0, 0),
    foot_lateral_m: tuple[float, float] = (0.0, 0.0),
    foot_planar_speeds_mps: tuple[float, float] = (0.0, 0.0),
    foot_surface_clearances_m: tuple[float, float] = (0.05, 0.05),
    foot_upward_forces_n: tuple[float, float] = (100.0, 100.0),
) -> StairEnvTransition:
    return StairEnvTransition(
        env_index=0,
        bin_id=bin_id,
        episode_step=step,
        progress_m=progress_m,
        lateral_m=lateral_m,
        base_height_m=root_height_m,
        base_clearance_m=root_clearance_m,
        abs_roll_rad=abs_roll_rad,
        abs_pitch_rad=abs_pitch_rad,
        forbidden_contact=forbidden_contact,
        support_contacts=foot_contacts,
        support_progress_m=foot_progress_m,
        support_surface_levels=foot_surface_levels,
        support_lateral_m=foot_lateral_m,
        support_planar_speeds_mps=foot_planar_speeds_mps,
        support_surface_clearances_m=foot_surface_clearances_m,
        support_upward_forces_n=foot_upward_forces_n,
        done=done,
        termination_terms=termination_terms,
        evaluation_cell_id=evaluation_cell_id,
    )


def _complete_seed_outcomes(protocol: StairProtocolV1) -> list[StairEpisodeOutcome]:
    outcomes: list[StairEpisodeOutcome] = []
    for env_index in range(56):
        bin_id = str(protocol.eval_bins[env_index // 8]["bin_id"])
        outcomes.append(
            StairEpisodeOutcome(
                env_index=env_index,
                bin_id=bin_id,
                episode_steps=20,
                primary_reason="success",
                success=True,
                diagnostic_flags=("first_valid_top_arrival_achieved",),
                max_progress_m=4.35,
                progress_fraction=1.0,
                active_terminal_terms=("time_out",),
            )
        )
    return outcomes


if __name__ == "__main__":
    unittest.main()

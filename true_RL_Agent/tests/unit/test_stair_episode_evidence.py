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
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        )
        cls.bin_id = str(cls.protocol.eval_bins[0]["bin_id"])
        cls.goal_m = float(
            cls.protocol.to_dict()["terrain"]["goal"][
                "progress_goal_m_from_origin"
            ]
        )

    def test_terminal_transition_state_is_counted_before_reset(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                progress_m=0.8,
                done=True,
                termination_terms=("base_contact",),
                base_contact=True,
            )
        )

        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.max_progress_m, 0.8)
        self.assertEqual(outcome.primary_reason, "base_contact")

    def test_progress_is_floored_at_the_protocol_course_start(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                progress_m=-0.08,
                done=True,
                termination_terms=("base_contact",),
                base_contact=True,
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

    def test_simultaneous_timeout_and_contact_follow_frozen_priority(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)

        outcome = tracker.observe(
            _transition(
                bin_id=self.bin_id,
                step=1,
                done=True,
                termination_terms=("time_out", "base_contact"),
                base_contact=True,
            )
        )

        assert outcome is not None
        self.assertEqual(outcome.primary_reason, "base_contact")
        self.assertEqual(outcome.active_terminal_terms, ("time_out", "base_contact"))

    def test_unknown_manager_term_fails_closed(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)
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

    def test_success_requires_full_top_hold_and_uses_terminal_step(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)
        outcome = None
        for step in range(1, 21):
            outcome = tracker.observe(
                _transition(
                    bin_id=self.bin_id,
                    step=step,
                    progress_m=self.goal_m,
                    done=step == 20,
                    termination_terms=("time_out",) if step == 20 else (),
                )
            )

        assert outcome is not None
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.primary_reason, "success")
        self.assertEqual(outcome.episode_steps, 20)

    def test_failure_signal_streaks_are_not_single_step_diagnostics(self) -> None:
        tracker = StairEpisodeTracker(self.protocol, env_index=0, bin_id=self.bin_id)
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

    def test_split_summary_requires_exact_env_and_bin_closure(self) -> None:
        per_seed = [_complete_seed_outcomes(self.protocol) for _ in range(3)]

        summary = build_stair_split_summary(
            self.protocol,
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
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_dev",
                per_seed_outcomes=[duplicate_env, per_seed[1], per_seed[2]],
            )

        unknown_diagnostic = list(per_seed[0])
        unknown_diagnostic[0] = replace(
            unknown_diagnostic[0],
            diagnostic_flags=("private_probe",),
        )
        with self.assertRaisesRegex(ValueError, "undeclared diagnostic"):
            build_stair_split_summary(
                self.protocol,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_dev",
                per_seed_outcomes=[
                    unknown_diagnostic,
                    per_seed[1],
                    per_seed[2],
                ],
            )


def _transition(
    *,
    bin_id: str,
    step: int,
    progress_m: float = 0.0,
    root_clearance_m: float = 0.9,
    done: bool = False,
    termination_terms: tuple[str, ...] = (),
    base_contact: bool = False,
) -> StairEnvTransition:
    return StairEnvTransition(
        env_index=0,
        bin_id=bin_id,
        episode_step=step,
        progress_m=progress_m,
        root_clearance_m=root_clearance_m,
        abs_pitch_rad=0.0,
        base_contact=base_contact,
        foot_contacts=(True, True),
        foot_planar_speeds_mps=(0.0, 0.0),
        swing_foot_clearances_m=(0.1, 0.1),
        done=done,
        termination_terms=termination_terms,
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
                diagnostic_flags=(),
                max_progress_m=4.35,
                progress_fraction=1.0,
                active_terminal_terms=("time_out",),
            )
        )
    return outcomes


if __name__ == "__main__":
    unittest.main()

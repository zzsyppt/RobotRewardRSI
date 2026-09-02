from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.acceptance_protocol import (  # noqa: E402
    acceptance_delta_met,
    acceptance_threshold,
    guard_regression_threshold,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)


class StairAcceptanceProtocolTests(unittest.TestCase):
    def test_historical_r4_noise_is_allowed_but_r5_regression_is_not(self) -> None:
        protocol = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        ).acceptance_protocol
        r3 = [0.9821428571428571, 1.0, 1.0, 0.9821428571428571, 1.0]
        r4 = [
            0.9821428571428571,
            0.9642857142857143,
            0.9464285714285714,
            0.9821428571428571,
            0.9642857142857143,
        ]
        r5 = [
            0.875,
            0.8928571428571429,
            0.875,
            0.9107142857142857,
            0.8214285714285714,
        ]
        r3_objective = [
            0.9285714285714286,
            0.9392857142857143,
            0.9321428571428572,
            0.9142857142857143,
            0.9321428571428572,
        ]
        r4_objective = [
            0.9535714285714286,
            0.9571428571428572,
            0.9392857142857143,
            0.9464285714285714,
            0.95,
        ]

        r4_threshold = guard_regression_threshold(
            protocol,
            candidate_guard_score=statistics.fmean(r4),
            best_guard_score=statistics.fmean(r3),
            candidate_guard_score_samples=r4,
            best_guard_score_samples=r3,
        )
        r5_threshold = guard_regression_threshold(
            protocol,
            candidate_guard_score=statistics.fmean(r5),
            best_guard_score=statistics.fmean(r3),
            candidate_guard_score_samples=r5,
            best_guard_score_samples=r3,
        )

        self.assertAlmostEqual(r4_threshold.threshold, 0.036071428571428574)
        self.assertGreaterEqual(
            statistics.fmean(r4),
            statistics.fmean(r3) - r4_threshold.threshold,
        )
        self.assertAlmostEqual(r5_threshold.threshold, 0.04624453295839122)
        self.assertLess(
            statistics.fmean(r5),
            statistics.fmean(r3) - r5_threshold.threshold,
        )
        objective_threshold = acceptance_threshold(
            protocol,
            candidate_score=statistics.fmean(r4_objective),
            best_score=statistics.fmean(r3_objective),
            candidate_score_samples=r4_objective,
            best_score_samples=r3_objective,
        )
        self.assertAlmostEqual(objective_threshold.threshold, 0.014142857142857146)
        self.assertTrue(
            acceptance_delta_met(
                protocol,
                statistics.fmean(r4_objective) - statistics.fmean(r3_objective),
                objective_threshold.threshold,
            )
        )

    def test_historical_r0_r9_counterfactual_has_no_high_score_dead_zone(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        protocol = config.acceptance_protocol
        self.assertEqual(protocol.headroom_reduction_fraction, 0.2)
        self.assertEqual(protocol.paired_standard_error_multiplier, 1.5)
        self.assertEqual(
            config.protocol_v1.acceptance_protocol.protocol_hash,
            "4e860ba981315580dba49ab0385a6b6826c8987288a1e98b8b4bbcacfe3f5917",
        )

        best = [
            0.5892857142857143,
            0.5892857142857143,
            0.5892857142857143,
            0.5892857142857143,
            0.5357142857142857,
        ]
        candidates = {
            3: [
                0.9285714285714286,
                0.8928571428571429,
                0.8928571428571429,
                0.8571428571428571,
                0.9107142857142857,
            ],
            4: [
                0.875,
                0.9107142857142857,
                0.9464285714285714,
                0.9107142857142857,
                0.9464285714285714,
            ],
            5: [
                0.9107142857142857,
                0.9107142857142857,
                0.875,
                0.9285714285714286,
                0.9642857142857143,
            ],
            6: [
                0.875,
                0.8571428571428571,
                0.875,
                0.8928571428571429,
                0.8214285714285714,
            ],
            7: [
                0.8392857142857143,
                0.7857142857142857,
                0.8392857142857143,
                0.875,
                0.8392857142857143,
            ],
            8: [
                1.0,
                0.9642857142857143,
                0.9821428571428571,
                0.9821428571428571,
                0.9821428571428571,
            ],
            9: [1.0, 1.0, 0.9821428571428571, 0.9821428571428571, 1.0],
        }
        accepted_rounds: list[int] = []
        for round_number, candidate in candidates.items():
            candidate_mean = statistics.fmean(candidate)
            best_mean = statistics.fmean(best)
            threshold = acceptance_threshold(
                protocol,
                candidate_score=candidate_mean,
                best_score=best_mean,
                candidate_score_samples=candidate,
                best_score_samples=best,
            ).threshold
            if acceptance_delta_met(
                protocol,
                candidate_mean - best_mean,
                threshold,
            ):
                accepted_rounds.append(round_number)
                best = candidate

        self.assertEqual(accepted_rounds, [3, 8])

        perfect = [1.0] * 5
        perfect_delta = statistics.fmean(perfect) - statistics.fmean(best)
        perfect_threshold = acceptance_threshold(
            protocol,
            candidate_score=statistics.fmean(perfect),
            best_score=statistics.fmean(best),
            candidate_score_samples=perfect,
            best_score_samples=best,
        ).threshold
        self.assertTrue(
            acceptance_delta_met(protocol, perfect_delta, perfect_threshold)
        )


if __name__ == "__main__":
    unittest.main()

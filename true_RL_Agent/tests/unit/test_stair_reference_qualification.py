from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.unit.test_stair_public_projection import (  # noqa: E402
    _source_preflight,
    _summary,
    _train_summary,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (  # noqa: E402
    build_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (  # noqa: E402
    project_stair_public_result,
)
from true_rl_agent.tasks.stair_climbing.reference_qualification import (  # noqa: E402
    qualify_stair_reference_public_result,
)


class StairReferenceQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )

    def test_final_v2_gate_is_derived_from_canonical_public_accept(self) -> None:
        projected = self._project(_summary(self.config, "public_accept"))
        result = qualify_stair_reference_public_result(
            config=self.config,
            public_result=projected,
        )

        self.assertTrue(result.passed)
        self.assertEqual(
            result.gate["protocol_hash"],
            self.config.task_protocol_hash,
        )
        self.assertIn(
            "reference_capability_gate",
            result.public_result.metrics["acceptance"]["required_pass_gates"],
        )
        task_gate = result.private_evidence["task_gate"]
        self.assertEqual(task_gate["observed_macro_mean_success_rate"], 0.5)
        self.assertGreater(
            task_gate["observed_macro_mean_max_progress_m"],
            task_gate["minimum_macro_mean_max_progress_m"],
        )

    def test_gate_fails_without_relaxing_frozen_thresholds(self) -> None:
        projected = self._project(self._failed_public_accept())
        result = qualify_stair_reference_public_result(
            config=self.config,
            public_result=projected,
        )

        self.assertFalse(result.passed)
        task_gate = result.private_evidence["task_gate"]
        contract = self.config.protocol_v2.to_public_dict()[
            "reference_capability_gate"
        ]
        self.assertEqual(
            task_gate["minimum_macro_mean_success_rate"],
            contract["minimum_macro_mean_success_rate"],
        )
        self.assertEqual(
            task_gate["minimum_macro_mean_max_progress_m"],
            contract["minimum_macro_mean_max_progress_m"],
        )
        self.assertFalse(task_gate["threshold_relaxation_allowed"])

    def test_rejects_post_projection_seed_or_bin_drift(self) -> None:
        projected = self._project(_summary(self.config, "public_accept"))
        metrics = copy.deepcopy(dict(projected.metrics))
        metrics["public_accept"]["seed_summaries"][0]["bin_success_rates"][
            "stair_010"
        ] = 0.0
        forged = PublicEvalResult(
            status=projected.status,
            public_score=projected.public_score,
            metrics=metrics,
        )

        with self.assertRaisesRegex(ValueError, "success rate|success counts|score mean"):
            qualify_stair_reference_public_result(
                config=self.config,
                public_result=forged,
            )

    def test_accepts_canonical_float_reconstruction_ulp_drift(self) -> None:
        projected = self._project(_summary(self.config, "public_accept"))
        metrics = copy.deepcopy(dict(projected.metrics))
        score_stats = metrics["public_accept"]["score_stats"]
        score_stats["standard_error"] = math.nextafter(
            score_stats["standard_error"],
            math.inf,
        )
        reconstructed = PublicEvalResult(
            status=projected.status,
            public_score=projected.public_score,
            metrics=metrics,
        )

        result = qualify_stair_reference_public_result(
            config=self.config,
            public_result=reconstructed,
        )

        self.assertTrue(result.passed)

    def _project(self, public_accept: dict[str, object]) -> PublicEvalResult:
        return project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=float(
                    public_accept["anchor"]["score_stats"]["mean"]
                ),
                metrics={
                    "environment_source_preflight": _source_preflight(self.config),
                    "stair_evaluation": {
                        "public_dev": _summary(self.config, "public_dev"),
                        "public_accept": public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        )

    def _failed_public_accept(self) -> dict[str, object]:
        per_seed = []
        for _ in range(self.config.public_accept_seed_count):
            outcomes = []
            for env_index, cell in enumerate(
                self.config.evaluation_profile.private_cell_assignment()
            ):
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.bin_id,
                        episode_steps=600,
                        primary_reason="timeout",
                        success=False,
                        diagnostic_flags=("goal_not_reached_detected",),
                        max_progress_m=0.1,
                        progress_fraction=0.01,
                        active_terminal_terms=("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        anchor_ids = {
            cell.cell_id for cell in self.config.evaluation_profile.anchor_cells
        }
        anchor_outcomes = [
            [
                outcome
                for outcome in seed_outcomes
                if outcome.evaluation_cell_id in anchor_ids
            ]
            for seed_outcomes in per_seed
        ]
        return {
            "anchor": build_stair_split_summary(
                self.config.protocol_v1,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                per_seed_outcomes=anchor_outcomes,
            ),
            "generalization": build_stair_generalization_summary(
                profile=self.config.evaluation_profile,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                per_seed_outcomes=per_seed,
            ),
        }


if __name__ == "__main__":
    unittest.main()

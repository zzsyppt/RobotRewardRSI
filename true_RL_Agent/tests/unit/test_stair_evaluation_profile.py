from __future__ import annotations

import copy
import statistics
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.evaluation.sealed_validation import (  # noqa: E402
    validate_sealed_rollout_evidence,
)
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_evaluation_summary,
    build_stair_sealed_rollout_evidence,
    canonical_stair_sealed_summary,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (  # noqa: E402
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (  # noqa: E402
    build_stair_generalization_summary,
    validate_stair_generalization_summary,
)


class StairEvaluationProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        cls.profile = cls.config.evaluation_profile

    def test_profile_preserves_anchor_and_covers_equal_weight_grid(self) -> None:
        profile = self.profile
        assignment = profile.private_cell_assignment()

        self.assertEqual(len(profile.cells), 35)
        self.assertEqual(len(assignment), 280)
        self.assertEqual(profile.episodes_per_seed_per_cell, 8)
        self.assertEqual(
            Counter(cell.cell_id for cell in assignment),
            {cell.cell_id: 8 for cell in profile.cells},
        )
        self.assertEqual(
            [cell.tread_depth_m for cell in profile.cells[:7]],
            [0.4] * 7,
        )
        self.assertEqual(
            tuple(cell.cell_id for cell in profile.anchor_cells),
            tuple(cell.cell_id for cell in profile.cells[:7]),
        )
        self.assertEqual(
            profile.profile_hash,
            "86d8eed0c8b9a8c32af49c69a81d983b3ec826abec408d5664717f61e16939d5",
        )
        self.assertEqual(
            profile.evaluation_protocol.protocol_hash,
            "9daa940351fd8557fb3fab314fc952dfd3fc47c749c303615258800bade2abd0",
        )
        self.assertNotIn("max_public_to_sealed_score_drop", profile.to_dict())
        self.assertEqual(
            self.config.acceptance_protocol.max_public_to_sealed_score_drop,
            0.05,
        )

    def test_geometry_dependent_goal_and_top_length_share_one_cell_source(self) -> None:
        for cell in self.profile.cells:
            self.assertAlmostEqual(
                cell.progress_goal_m,
                0.75 + 8 * cell.tread_depth_m + 0.4,
            )
            self.assertAlmostEqual(
                cell.top_platform_length_m,
                8.0 - 1.5 - 8 * cell.tread_depth_m,
            )
        self.assertEqual(
            [cell.episode_horizon_control_steps for cell in self.profile.cells[::7]],
            [600, 490, 545, 655, 710],
        )
        self.assertEqual(
            [cell.episode_horizon_s for cell in self.profile.cells[::7]],
            [12.0, 9.8, 10.9, 13.1, 14.2],
        )
        self.assertEqual(self.profile.max_episode_horizon_control_steps, 710)
        self.assertEqual(self.profile.max_episode_horizon_s, 14.2)

    def test_profile_hash_rejects_unbound_geometry_change(self) -> None:
        payload = self.profile.to_dict()
        payload["top_platform_goal_offset_m"] = 0.41

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            StairEvaluationProfile.from_mapping(
                payload,
                base_protocol=self.config.protocol_v1,
            )

    def test_generalization_summary_rejects_cell_tamper(self) -> None:
        summary = self._summary()
        changed = copy.deepcopy(summary)
        first_seed = changed["seed_summaries"][0]
        first_cell = self.profile.cells[0].cell_id
        first_seed["cell_success_rates"][first_cell] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            "cell_success_rates|height_success_rates|seed macro",
        ):
            validate_stair_generalization_summary(
                profile=self.profile,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(summary)
        first_cell = self.profile.cells[0].cell_id
        changed["aggregate_success_rate_by_cell"][first_cell] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "aggregate_success_rate_by_cell",
        ):
            validate_stair_generalization_summary(
                profile=self.profile,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(summary)
        first_cell = self.profile.cells[0].cell_id
        changed["primary_outcome_counts_by_cell"][first_cell]["success"] += 1
        changed["primary_outcome_counts_by_cell"][first_cell]["timeout"] -= 1
        with self.assertRaisesRegex(ValueError, "per-cell success outcome"):
            validate_stair_generalization_summary(
                profile=self.profile,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(summary)
        seed_counts = changed["seed_summaries"][0][
            "primary_outcome_counts_by_cell"
        ][first_cell]
        seed_counts["timeout"] -= 1
        seed_counts["forbidden_contact"] += 1
        with self.assertRaisesRegex(ValueError, "per-seed primary outcomes"):
            validate_stair_generalization_summary(
                profile=self.profile,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

    def test_canonical_builder_emits_anchor_and_generalization_from_one_cohort(
        self,
    ) -> None:
        pair = build_stair_evaluation_summary(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_accept",
            per_seed_outcomes=self._per_seed_outcomes(),
            evaluation_profile=self.profile,
        )

        self.assertEqual(set(pair), {"anchor", "generalization"})
        self.assertEqual(pair["anchor"]["total_episode_count"], 5 * 56)
        self.assertEqual(
            pair["generalization"]["total_episode_count"],
            5 * 280,
        )
        self.assertEqual(
            pair["anchor"]["seed_macro_scores"],
            pair["generalization"]["seed_macro_scores"],
        )

    def test_nonmonotonic_success_is_diagnostic_not_a_score_gate(self) -> None:
        summary = self._summary()
        canonical = validate_stair_generalization_summary(
            profile=self.profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_accept",
            summary=summary,
        )

        diagnostics = canonical["difficulty_diagnostics"]
        self.assertGreater(diagnostics["inversion_count"], 0)
        self.assertEqual(diagnostics["acceptance_role"], "diagnostic_only")
        self.assertAlmostEqual(canonical["score_stats"]["mean"], 2 / 7)

    def test_sealed_gate_rejects_public_cohort_overfit_above_absolute_floor(
        self,
    ) -> None:
        public_summary = self._uniform_summary(
            split="public_accept",
            successes_per_cell=8,
        )
        regressed_sealed = self._uniform_summary(
            split="sealed_validation",
            successes_per_cell=7,
        )
        evidence = self._sealed_evidence(
            public_summary=public_summary,
            sealed_summary=regressed_sealed,
        )

        self.assertEqual(evidence["status"], "fail")
        self.assertEqual(
            evidence["reason"],
            "stair_generalization_holdout_public_score_regression",
        )
        matched_public = self._uniform_summary(
            split="public_accept",
            successes_per_cell=7,
        )
        matched = self._sealed_evidence(
            public_summary=matched_public,
            sealed_summary=regressed_sealed,
        )
        self.assertEqual(matched["status"], "pass")
        self.assertEqual(
            matched["reason"],
            "stair_generalization_holdout_public_consistency_met",
        )

    def test_sealed_evidence_hashes_the_canonical_recomputable_summary(self) -> None:
        public_summary = self._uniform_summary(
            split="public_accept",
            successes_per_cell=7,
        )
        sealed_summary = self._uniform_summary(
            split="sealed_validation",
            successes_per_cell=7,
        )
        canonical = canonical_stair_sealed_summary(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            summary=sealed_summary,
            evaluation_profile=self.profile,
        )
        evidence = self._sealed_evidence(
            public_summary=public_summary,
            sealed_summary=canonical,
        )

        self.assertEqual(
            evidence["task_evidence_hash"],
            canonical_json_hash(canonical),
        )
        self.assertAlmostEqual(
            evidence["score_stats"]["mean"],
            statistics.fmean(canonical["seed_macro_scores"]),
        )
        tampered = copy.deepcopy(canonical)
        tampered["seed_macro_scores"][0] = 0.0
        validation = validate_sealed_rollout_evidence(
            evidence,
            expected_task_evidence_hash=canonical_json_hash(tampered),
        )
        self.assertFalse(validation.complete)
        self.assertIn(
            "sealed_validation task_evidence_hash binding mismatch",
            validation.reasons,
        )

    def test_seed_cell_primary_counts_bind_failure_distribution_into_hash(self) -> None:
        seed_count = self.config.sealed_validation_seed_count
        first = self._per_seed_outcomes(seed_count=seed_count)
        second = [list(outcomes) for outcomes in first]
        failure_index = next(
            index for index, outcome in enumerate(first[0]) if not outcome.success
        )
        first[1][failure_index] = replace(
            first[1][failure_index],
            primary_reason="forbidden_contact",
            active_terminal_terms=("base_contact",),
        )
        second[0][failure_index] = replace(
            second[0][failure_index],
            primary_reason="forbidden_contact",
            active_terminal_terms=("base_contact",),
        )
        second[1][failure_index] = replace(
            second[1][failure_index],
            primary_reason="timeout",
            active_terminal_terms=("time_out",),
        )

        first_summary = build_stair_generalization_summary(
            profile=self.profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="sealed_validation",
            per_seed_outcomes=first,
        )
        second_summary = build_stair_generalization_summary(
            profile=self.profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="sealed_validation",
            per_seed_outcomes=second,
        )

        self.assertEqual(
            first_summary["primary_outcome_counts"],
            second_summary["primary_outcome_counts"],
        )
        self.assertEqual(
            first_summary["primary_outcome_counts_by_cell"],
            second_summary["primary_outcome_counts_by_cell"],
        )
        self.assertNotEqual(
            first_summary["seed_summaries"],
            second_summary["seed_summaries"],
        )
        self.assertNotEqual(
            canonical_json_hash(
                canonical_stair_sealed_summary(
                    self.config.protocol_v1,
                    semantics=self.config.evaluation_semantics,
                    evaluation_protocol=self.config.evaluation_protocol(),
                    summary=first_summary,
                    evaluation_profile=self.profile,
                )
            ),
            canonical_json_hash(
                canonical_stair_sealed_summary(
                    self.config.protocol_v1,
                    semantics=self.config.evaluation_semantics,
                    evaluation_protocol=self.config.evaluation_protocol(),
                    summary=second_summary,
                    evaluation_profile=self.profile,
                )
            ),
        )

    def test_legacy_v3_summary_remains_canonical_but_cannot_mint_new_evidence(
        self,
    ) -> None:
        current = self._uniform_summary(
            split="sealed_validation",
            successes_per_cell=7,
        )
        legacy = copy.deepcopy(current)
        legacy["schema_version"] = (
            "true_rl_agent.task.stair_climbing.generalization_summary.v3"
        )
        for seed in legacy["seed_summaries"]:
            seed.pop("primary_outcome_counts_by_cell")

        canonical = canonical_stair_sealed_summary(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            summary=legacy,
            evaluation_profile=self.profile,
        )
        self.assertEqual(canonical, legacy)
        self.assertEqual(
            canonical_json_hash(canonical),
            canonical_json_hash(legacy),
        )
        self.assertEqual(
            canonical_json_hash(canonical),
            "a280d21ba4b393a061bbf22fb736b2e145f819569b1dd54e618e337dde8e7dd5",
        )

        public_summary = self._uniform_summary(
            split="public_accept",
            successes_per_cell=7,
        )
        with self.assertRaisesRegex(ValueError, "current summary schema"):
            self._sealed_evidence(
                public_summary=public_summary,
                sealed_summary=legacy,
            )

    def _summary(self) -> dict[str, object]:
        return build_stair_generalization_summary(
            profile=self.profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_accept",
            per_seed_outcomes=self._per_seed_outcomes(),
        )

    def _uniform_summary(
        self,
        *,
        split: str,
        successes_per_cell: int,
    ) -> dict[str, object]:
        per_seed = []
        seed_count = {
            "public_dev": self.config.public_seed_count,
            "public_accept": self.config.public_accept_seed_count,
            "sealed_validation": self.config.sealed_validation_seed_count,
        }[split]
        for _ in range(seed_count):
            outcomes = []
            for env_index, cell in enumerate(
                self.profile.private_cell_assignment()
            ):
                success = env_index % self.profile.episodes_per_seed_per_cell < (
                    successes_per_cell
                )
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.bin_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success" if success else "timeout",
                        success=success,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",)
                            if success
                            else ()
                        ),
                        max_progress_m=cell.progress_goal_m if success else 0.5,
                        progress_fraction=1.0 if success else 0.1,
                        active_terminal_terms=(() if success else ("time_out",)),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        return build_stair_generalization_summary(
            profile=self.profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split=split,
            per_seed_outcomes=per_seed,
        )

    def _sealed_evidence(
        self,
        *,
        public_summary: dict[str, object],
        sealed_summary: dict[str, object],
    ) -> dict[str, object]:
        return build_stair_sealed_rollout_evidence(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            summary=sealed_summary,
            checkpoint_hash="sha256:" + "1" * 64,
            evaluation_binding_hash="2" * 64,
            evaluation_subject=ExperimentSubject(
                kind="candidate",
                subject_id="candidate-1",
            ),
            subject_manifest_hash="3" * 64,
            program_source_checksum="sha256:" + "4" * 64,
            training_binding_hash="5" * 64,
            capability_profile_hash="6" * 64,
            acceptance_protocol=self.config.acceptance_protocol,
            evaluation_profile=self.profile,
            public_objective_summary=public_summary,
        )

    def _per_seed_outcomes(
        self,
        *,
        seed_count: int | None = None,
    ) -> list[list[StairEpisodeOutcome]]:
        per_seed = []
        for _ in range(seed_count or self.config.public_accept_seed_count):
            outcomes = []
            for env_index, cell in enumerate(self.profile.private_cell_assignment()):
                success = cell.height_m >= 0.35
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.bin_id,
                        episode_steps=600,
                        primary_reason="success" if success else "timeout",
                        success=success,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",)
                            if success
                            else ("goal_not_reached_detected",)
                        ),
                        max_progress_m=(cell.progress_goal_m if success else 0.5),
                        progress_fraction=1.0 if success else 0.1,
                        active_terminal_terms=(() if success else ("time_out",)),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        return per_seed


if __name__ == "__main__":
    unittest.main()

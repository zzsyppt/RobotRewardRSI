from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.phase_d_run import (  # noqa: E402
    stair_phase_d_pilot_seed_root,
)
from true_rl_agent.tasks.stair_climbing.reference_pilot import (  # noqa: E402
    assert_stair_pilot_formal_bindings_disjoint,
    build_stair_pilot_config,
    build_stair_pilot_evaluation_binding,
    build_stair_pilot_training_binding,
    build_stair_reference_capability_evidence,
    stair_pilot_training_seed_domain_hash,
    validate_stair_reference_capability_evidence,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_PILOT_TRAIN_NAMESPACE,
    StairPrivateTrainingSeedResolver,
)


class StairPhaseDPilotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"
        )

    def test_pilot_config_and_bindings_use_the_private_pilot_domain(self) -> None:
        pilot = build_stair_pilot_config(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"
        )
        self.assertEqual(pilot.train_max_iterations, 2000)
        self.assertEqual(
            pilot.budget_steps,
            pilot.train_num_envs
            * pilot.train_steps_per_iteration
            * pilot.train_max_iterations,
        )
        training_binding = build_stair_pilot_training_binding(pilot)
        evaluation_binding = build_stair_pilot_evaluation_binding(pilot)
        self.assertEqual(training_binding.budget_steps, 196_608_000)
        self.assertEqual(
            evaluation_binding.seed_cohort_hash,
            pilot.seed_resolver().namespace_cohort_hash(
                STAIR_PILOT_EVAL_NAMESPACE
            ),
        )
        self.assertTrue(stair_pilot_training_seed_domain_hash(pilot))
        assert_stair_pilot_formal_bindings_disjoint(pilot)

        wrong_domain = copy.copy(pilot)
        object.__setattr__(wrong_domain, "train_seed", 42)
        with self.assertRaisesRegex(ValueError, "outside the pilot cohort"):
            build_stair_pilot_training_binding(wrong_domain)

    def test_pilot_config_rejects_budget_override(self) -> None:
        pilot = build_stair_pilot_config(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"
        )
        with self.assertRaisesRegex(ValueError, "fixed V2 pilot budget"):
            replace(pilot, budget_steps=pilot.budget_steps + 1)

    def test_evidence_layout_consumes_the_protocol_iteration_count(self) -> None:
        class Config:
            train_max_iterations = 123

        self.assertEqual(
            stair_phase_d_pilot_seed_root(
                output_root=Path("run-root"),
                config=Config(),
            ),
            Path("run-root") / "pilot-123",
        )

    def test_training_and_evaluation_namespaces_are_distinct(self) -> None:
        training = StairPrivateTrainingSeedResolver(
            base_seed=42,
            task_protocol_hash=self.config.protocol_v2.protocol_hash,
        )
        training.assert_namespaces_disjoint()
        pilot_seed = training.derive_seed(namespace=STAIR_PILOT_TRAIN_NAMESPACE)
        formal_seed = training.derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
        self.assertNotEqual(pilot_seed, formal_seed)
        self.assertNotEqual(
            training.namespace_binding_hash(
                namespace=STAIR_PILOT_TRAIN_NAMESPACE
            ),
            training.namespace_binding_hash(
                namespace=STAIR_FORMAL_TRAIN_NAMESPACE
            ),
        )

        evaluation = self.config.seed_resolver()
        self.assertNotEqual(
            evaluation.namespace_cohort_hash(STAIR_PILOT_EVAL_NAMESPACE),
            evaluation.namespace_cohort_hash(STAIR_FORMAL_EVAL_NAMESPACE),
        )
        pilot_record = evaluation.public_seed_record(
            split="public_accept",
            run_index=1,
            namespace=STAIR_PILOT_EVAL_NAMESPACE,
        )
        formal_record = evaluation.public_seed_record(
            split="public_accept",
            run_index=1,
            namespace=STAIR_FORMAL_EVAL_NAMESPACE,
        )
        self.assertNotEqual(
            pilot_record["seed_cohort_hash"],
            formal_record["seed_cohort_hash"],
        )

    def test_capability_gate_passes_at_exact_floors(self) -> None:
        outcomes = _outcomes(
            self.config,
            total_low_bin_successes=6,
            low_bin_progress=0.5,
        )
        evidence = self._build(outcomes)
        gate = evidence["gate"]
        self.assertEqual(gate["status"], "pass")
        self.assertAlmostEqual(gate["observed_macro_mean_success_rate"], 0.05)
        self.assertAlmostEqual(gate["observed_macro_mean_max_progress_m"], 0.5)

    def test_capability_gate_requires_both_floors(self) -> None:
        cases = (
            (0, 0.6, "success_floor_met"),
            (12, 0.49, "progress_floor_met"),
        )
        for successes, progress, failed_check in cases:
            with self.subTest(failed_check=failed_check):
                evidence = self._build(
                    _outcomes(
                        self.config,
                        total_low_bin_successes=successes,
                        low_bin_progress=progress,
                    )
                )
                self.assertEqual(evidence["gate"]["status"], "fail")
                self.assertFalse(evidence["gate"][failed_check])

    def test_validator_rejects_forged_measurement_even_with_new_hash(self) -> None:
        evidence = self._build(
            _outcomes(
                self.config,
                total_low_bin_successes=12,
                low_bin_progress=0.6,
            )
        )
        forged = copy.deepcopy(evidence)
        forged["seed_measurements"][0]["low_step_macro_success_rate"] = 1.0
        forged["evidence_hash"] = canonical_json_hash(
            {key: value for key, value in forged.items() if key != "evidence_hash"}
        )
        with self.assertRaisesRegex(ValueError, "success macro mismatch"):
            self._validate(forged)

    def test_validator_rejects_nonfinite_progress_and_missing_coverage(self) -> None:
        evidence = self._build(
            _outcomes(
                self.config,
                total_low_bin_successes=12,
                low_bin_progress=0.6,
            )
        )
        forged = copy.deepcopy(evidence)
        forged["seed_measurements"][0]["mean_max_progress_m_by_bin"][
            "stair_010"
        ] = float("nan")
        forged["evidence_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self._validate(forged)

        missing = _outcomes(
            self.config,
            total_low_bin_successes=12,
            low_bin_progress=0.6,
        )
        missing[0] = tuple(
            item for item in missing[0] if item.bin_id != "stair_010"
        )
        with self.assertRaisesRegex(ValueError, "episode count"):
            self._build(missing)

    def _build(self, outcomes):
        return build_stair_reference_capability_evidence(
            self.config.protocol_v1,
            protocol_v2=self.config.protocol_v2,
            evaluation_protocol=self.config.evaluation_protocol(),
            per_seed_outcomes=outcomes,
        )

    def _validate(self, evidence):
        return validate_stair_reference_capability_evidence(
            self.config.protocol_v1,
            protocol_v2=self.config.protocol_v2,
            evaluation_protocol=self.config.evaluation_protocol(),
            evidence=evidence,
        )


def _outcomes(
    config: StairClimbingTaskConfig,
    *,
    total_low_bin_successes: int,
    low_bin_progress: float,
) -> list[tuple[StairEpisodeOutcome, ...]]:
    assignment = config.seed_resolver().private_bin_assignment()
    low_bins = {"stair_010", "stair_015", "stair_020"}
    remaining_successes = total_low_bin_successes
    result: list[tuple[StairEpisodeOutcome, ...]] = []
    for _seed_index in range(config.public_accept_seed_count):
        seed_outcomes = []
        for env_index, bin_id in enumerate(assignment):
            success = bin_id in low_bins and remaining_successes > 0
            if success:
                remaining_successes -= 1
            seed_outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=bin_id,
                    episode_steps=600,
                    primary_reason="success" if success else "timeout",
                    success=success,
                    diagnostic_flags=(),
                    max_progress_m=(
                        low_bin_progress if bin_id in low_bins else 0.25
                    ),
                    progress_fraction=0.5 if bin_id in low_bins else 0.2,
                    active_terminal_terms=("time_out",),
                )
            )
        result.append(tuple(seed_outcomes))
    if remaining_successes:
        raise AssertionError("test requested more low-bin successes than available")
    return result


if __name__ == "__main__":
    unittest.main()

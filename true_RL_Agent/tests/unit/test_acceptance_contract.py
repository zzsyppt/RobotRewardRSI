from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_CAPABILITY_PROFILE_HASH,
    TEST_EVALUATION_BINDING,
    TEST_EVALUATION_PROTOCOL,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SESSION_GENESIS_HASH,
    TEST_SUBJECT_MANIFEST_HASH,
    TEST_TRAINING_BINDING_HASH,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.decision import decide_candidate
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.acceptance_protocol import (
    GUARDED_PAIRED_HEADROOM_DELTA_RULE,
    NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE,
    PAIRED_HEADROOM_DELTA_RULE,
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    PUBLIC_SCORE_BINDING_GUARD_MEAN,
    AcceptanceProtocol,
    NamedGuardSpec,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult


class AcceptanceContractTests(unittest.TestCase):
    def test_score_source_must_resolve_the_exact_declared_path(self) -> None:
        base = _guarded_protocol()
        protocol = AcceptanceProtocol(
            protocol_id="test_nested_guarded_acceptance_v1",
            score_source="forged.acceptance_objective.score_stats.mean",
            min_score_count=base.min_score_count,
            absolute_accept_floor=base.absolute_accept_floor,
            min_delta=base.min_delta,
            public_score_binding=base.public_score_binding,
            decision_rule=base.decision_rule,
            score_ceiling=base.score_ceiling,
            headroom_reduction_fraction=base.headroom_reduction_fraction,
            paired_standard_error_multiplier=(
                base.paired_standard_error_multiplier
            ),
            guard_score_source=base.guard_score_source,
            guard_absolute_accept_floor=base.guard_absolute_accept_floor,
            guard_max_regression=base.guard_max_regression,
        )
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        record = _guarded_record(
            objective_score=0.8,
            guard_score=0.5,
            protocol=protocol,
            binding=binding,
        )

        validation = validate_acceptance_evidence(
            record["public_result"]["metrics"],
            acceptance_protocol=protocol,
            public_score=0.5,
        )

        self.assertFalse(validation.valid)
        self.assertEqual(validation.reject_reason, "required_score_missing")

    def test_guard_blocks_objective_improvement_that_regresses_anchor(self) -> None:
        protocol = _guarded_protocol()
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        baseline = _guarded_record(
            objective_score=0.5,
            guard_score=0.5,
            protocol=protocol,
            binding=binding,
        )
        current_best = CurrentBestEvalStats.from_public_result_mapping(
            round_id="round-acceptance",
            candidate_id="candidate-acceptance",
            public_result=baseline["public_result"],
            acceptance_protocol=protocol,
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=TEST_TRAINING_BINDING_HASH,
            capability_profile_hash=TEST_CAPABILITY_PROFILE_HASH,
            checkpoint_hash="sha256:" + "d" * 64,
            expected_evaluation_binding=binding,
        )
        regressed = _guarded_record(
            objective_score=0.9,
            guard_score=0.49,
            protocol=protocol,
            binding=binding,
        )

        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level",
                True,
                "code_level_candidate",
            ),
            public_result=PublicEvalResult(**regressed["public_result"]),
            current_best=current_best,
            acceptance_protocol=protocol,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "guard_score_regression")
        self.assertGreater(decision.comparison["delta"], 0.0)
        self.assertFalse(decision.comparison["guard_met"])

        improved = _guarded_record(
            objective_score=0.8,
            guard_score=0.5,
            protocol=protocol,
            binding=binding,
        )
        accepted = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level",
                True,
                "code_level_candidate",
            ),
            public_result=PublicEvalResult(**improved["public_result"]),
            current_best=current_best,
            acceptance_protocol=protocol,
        )
        improved["decision"] = accepted.to_public_dict()
        improved["evaluation_binding"] = binding.to_public_dict()
        improved["jobs"]["eval"]["evaluation_binding"] = (
            binding.to_public_dict()
        )
        ledger_validation = validate_decision_record(
            improved,
            acceptance_protocol=protocol,
            expected_evaluation_binding=binding,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )

        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.reason, "guarded_paired_headroom_delta")
        self.assertTrue(ledger_validation.valid, ledger_validation.reasons)

    def test_named_guards_are_independent_and_replay_bound(self) -> None:
        protocol = _named_guard_protocol()
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        baseline = _named_guard_record(
            objective_score=0.5,
            guard_scores={"j_ood": 0.5, "c0_id": 0.5, "c1_id": 0.5},
            protocol=protocol,
            binding=binding,
        )
        current_best = CurrentBestEvalStats.from_public_result_mapping(
            round_id="round-named-guard",
            candidate_id="candidate-acceptance",
            public_result=baseline["public_result"],
            acceptance_protocol=protocol,
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=TEST_TRAINING_BINDING_HASH,
            capability_profile_hash=TEST_CAPABILITY_PROFILE_HASH,
            checkpoint_hash="sha256:" + "d" * 64,
            expected_evaluation_binding=binding,
        )

        regressed = _named_guard_record(
            objective_score=0.7,
            guard_scores={"j_ood": 0.5, "c0_id": 0.39, "c1_id": 0.5},
            protocol=protocol,
            binding=binding,
        )
        rejected = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level", True, "code_level_candidate"
            ),
            public_result=PublicEvalResult(**regressed["public_result"]),
            current_best=current_best,
            acceptance_protocol=protocol,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "named_guard_score_regression")
        self.assertFalse(rejected.comparison["named_guards"]["c0_id"]["met"])

        improved = _named_guard_record(
            objective_score=0.7,
            guard_scores={"j_ood": 0.51, "c0_id": 0.49, "c1_id": 0.5},
            protocol=protocol,
            binding=binding,
        )
        accepted = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level", True, "code_level_candidate"
            ),
            public_result=PublicEvalResult(**improved["public_result"]),
            current_best=current_best,
            acceptance_protocol=protocol,
        )
        improved["decision"] = accepted.to_public_dict()
        improved["evaluation_binding"] = binding.to_public_dict()
        improved["jobs"]["eval"]["evaluation_binding"] = binding.to_public_dict()
        replay = validate_decision_record(
            improved,
            acceptance_protocol=protocol,
            expected_evaluation_binding=binding,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.reason, "named_guards_paired_headroom_delta")
        self.assertTrue(replay.valid, replay.reasons)
        for guard in protocol.named_guards:
            self.assertIsInstance(
                accepted.to_public_dict()["comparison"]["named_guards"][
                    guard.name
                ]["best_score_stats"]["score_samples"],
                list,
            )
        persisted = json.loads(json.dumps([improved]))
        acceptance_replay = replay_acceptance_ledger(
            persisted,
            initial_current_best=current_best,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )
        acceptance_replay.raise_for_continuation()

        missing = copy.deepcopy(improved)
        missing["public_result"]["metrics"]["acceptance"]["named_guards"].pop(
            "c1_id"
        )
        missing_validation = validate_acceptance_evidence(
            missing["public_result"]["metrics"],
            acceptance_protocol=protocol,
            public_score=missing["public_result"]["public_score"],
        )
        self.assertFalse(missing_validation.valid)
        self.assertIn("acceptance_named_guard_set_mismatch", missing_validation.reasons)

        tampered = copy.deepcopy(improved)
        tampered["public_result"]["metrics"]["acceptance"]["named_guards"][
            "c0_id"
        ]["score_samples"][0] = 0.1
        tampered_replay = validate_decision_record(
            tampered,
            acceptance_protocol=protocol,
            expected_evaluation_binding=binding,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        self.assertFalse(tampered_replay.valid)

    def test_paired_acceptance_samples_are_bound_to_declared_stats(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        protocol = _paired_protocol()
        metrics["acceptance"]["min_delta"] = protocol.min_delta
        metrics["acceptance"][
            "acceptance_protocol_hash"
        ] = protocol.protocol_hash
        metrics["acceptance"]["score_samples"] = [0.4, 0.5, 0.6]

        validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=protocol,
            public_score=0.5,
        )

        self.assertTrue(validation.valid, validation.reasons)
        self.assertEqual(validation.score_samples, (0.4, 0.5, 0.6))
        self.assertEqual(
            validation.score_stats["score_samples"],
            [0.4, 0.5, 0.6],
        )

        for samples in ([0.4, 0.5], [0.4, 0.5, 0.9], [0.4, 0.5, 1.1]):
            with self.subTest(samples=samples):
                changed = copy.deepcopy(metrics)
                changed["acceptance"]["score_samples"] = samples
                rejected = validate_acceptance_evidence(
                    changed,
                    acceptance_protocol=protocol,
                    public_score=0.5,
                )
                self.assertFalse(rejected.valid)
                self.assertIn(
                    "acceptance_score_samples_untrusted",
                    rejected.reasons,
                )

    def test_paired_protocol_bootstrap_uses_floor_without_reference_samples(
        self,
    ) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        protocol = _paired_protocol()
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_BINDING.eval_protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        acceptance = metrics["acceptance"]
        acceptance["min_delta"] = protocol.min_delta
        acceptance["absolute_accept_floor"] = protocol.absolute_accept_floor
        acceptance["acceptance_protocol_hash"] = protocol.protocol_hash
        acceptance["score_samples"] = [0.4, 0.5, 0.6]
        metrics["evaluation_binding"] = binding.to_public_dict()
        metrics["sealed_validation"]["evaluation_binding_hash"] = (
            binding.binding_hash
        )

        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level",
                True,
                "code_level_candidate",
            ),
            public_result=PublicEvalResult(**record["public_result"]),
            current_best=_bootstrap(protocol, binding),
            acceptance_protocol=protocol,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "absolute_accept_floor")
        self.assertEqual(decision.comparison["mode"], "bootstrap")
        self.assertNotIn("threshold_components", decision.comparison)

    def test_baseline_requires_eval_gates_but_not_candidate_training_budget(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["required_pass_gates"] = [
            "accept_evidence_gate",
            "sealed_validation_gate",
            "task_behavior_gate",
        ]
        metrics.pop("budget_accounting")
        metrics["task_behavior_gate"] = {"status": "pass"}

        validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=record["public_result"]["public_score"],
            expected_sealed_seed_count=7,
            expected_checkpoint_hash="sha256:" + "d" * 64,
            evidence_subject_kind="baseline",
        )

        self.assertTrue(validation.valid, validation.reasons)
        self.assertNotIn("budget_accounting", validation.required_gates)
        self.assertEqual(validation.gate_statuses["task_behavior_gate"], "pass")

        metrics.pop("task_behavior_gate")
        missing_task_gate = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=record["public_result"]["public_score"],
            expected_sealed_seed_count=7,
            expected_checkpoint_hash="sha256:" + "d" * 64,
            evidence_subject_kind="baseline",
        )
        self.assertFalse(missing_task_gate.valid)
        self.assertIn("task_behavior_gate", missing_task_gate.failed_required_gates)

    def test_eval_metrics_cannot_override_task_owned_bootstrap_floor(self) -> None:
        for label, floor in {
            "missing": "missing",
            "none": None,
            "string": "0.05",
            "negative": -1.0,
            "valid": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
        }.items():
            with self.subTest(label=label):
                record = valid_accepted_ledger_record(
                    round_id=f"round-{label}",
                    candidate_id=f"candidate-{label}",
                    public_score=0.01,
                )
                acceptance = record["public_result"]["metrics"]["acceptance"]
                if floor == "missing":
                    acceptance.pop("absolute_accept_floor")
                else:
                    acceptance["absolute_accept_floor"] = floor
                decision = decide_candidate(
                    surface=CandidateSurfaceAssessment(
                        "code_level", True, "code_level_candidate"
                    ),
                    public_result=PublicEvalResult(**record["public_result"]),
                    current_best=_bootstrap(
                        TEST_ACCEPTANCE_PROTOCOL,
                        TEST_EVALUATION_BINDING,
                    ),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )

                self.assertFalse(decision.accepted)
                self.assertEqual(
                    decision.reason,
                    (
                        "absolute_accept_floor_not_met"
                        if label == "valid"
                        else "acceptance_protocol_mismatch"
                    ),
                )
    def test_counts_must_be_positive_and_do_not_form_canonical_stats(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["min_score_count"] = -1
        metrics["acceptance"]["score_stats"]["count"] = -1
        metrics["public_accept"]["score_stats"]["count"] = -1

        validation = validate_acceptance_evidence(metrics)

        self.assertFalse(validation.valid)
        self.assertIn("min_score_count_invalid", validation.reasons)
        self.assertIn("accept_score_count_invalid", validation.reasons)
        self.assertIsNone(validation.min_score_count)
        self.assertIsNone(validation.score_count)
        self.assertEqual(validation.score_stats, {})

    def test_min_count_one_allows_missing_standard_error_everywhere(self) -> None:
        record = _record()
        protocol = AcceptanceProtocol(
            protocol_id="test_acceptance_count_one_v1",
            score_source=TEST_ACCEPTANCE_PROTOCOL.score_source,
            min_score_count=1,
            absolute_accept_floor=0.05,
            min_delta=0.05,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        )
        eval_protocol = EvaluationProtocol(
            eval_protocol_version="test_eval_count_one_v1",
            public_seed_count=1,
            public_accept_seed_count=1,
            sealed_validation_seed_count=7,
            eval_num_envs=64,
        )
        binding = EvaluationBindingV1(
            eval_protocol_hash=eval_protocol.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        initial_current_best = _bootstrap(protocol, binding)
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["min_score_count"] = 1
        metrics["acceptance"]["acceptance_protocol_hash"] = protocol.protocol_hash
        record["decision"]["comparison"]["acceptance_protocol_hash"] = protocol.protocol_hash
        record["evaluation_binding"] = binding.to_public_dict()
        metrics["evaluation_binding"] = binding.to_public_dict()
        record["decision"]["comparison"][
            "candidate_evaluation_binding_hash"
        ] = binding.binding_hash
        record["decision"]["comparison"][
            "current_best"
        ] = initial_current_best.comparison_identity()
        for stats in (
            metrics["acceptance"]["score_stats"],
            metrics["public_accept"]["score_stats"],
        ):
            stats["count"] = 1
            stats["eval_protocol_hash"] = eval_protocol.protocol_hash
            stats.pop("standard_error", None)
        sealed_evidence = metrics["sealed_validation"]
        sealed_evidence["eval_protocol_hash"] = eval_protocol.protocol_hash
        sealed_evidence["evaluation_binding_hash"] = binding.binding_hash
        sealed_evidence["score_stats"][
            "eval_protocol_hash"
        ] = eval_protocol.protocol_hash
        record["jobs"]["eval"] = {
            **record["jobs"]["eval"],
            **eval_protocol.to_public_dict(),
            "evaluation_binding": binding.to_public_dict(),
        }
        record["decision"]["comparison"]["candidate_score_stats"] = copy.deepcopy(
            metrics["acceptance"]["score_stats"]
        )

        validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=protocol,
        )

        self.assertTrue(validation.valid, validation.reasons)
        self.assertEqual(validation.score_count, 1)
        self.assertEqual(validation.min_score_count, 1)
        self.assertEqual(validation.score_mean, 0.50)
        self.assertIsNone(validation.standard_error)
        self.assertNotIn("standard_error", validation.score_stats)

        ledger_validation = validate_decision_record(
            record,
            acceptance_protocol=protocol,
        )
        self.assertTrue(ledger_validation.valid_policy_accept, ledger_validation.reasons)
        current_best = CurrentBestEvalStats.from_ledger_records(
            [record],
            initial_current_best=initial_current_best,
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )
        self.assertEqual(current_best.mode, "accepted")
        self.assertTrue(current_best.has_full_eval_stats())

    def test_missing_acceptance_score_stats_has_one_canonical_reject_reason(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"].pop("score_stats")
        public_result = PublicEvalResult(**record["public_result"])

        evidence = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level", True, "code_level_candidate"
            ),
            public_result=public_result,
            current_best=_bootstrap(
                TEST_ACCEPTANCE_PROTOCOL,
                TEST_EVALUATION_BINDING,
            ),
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        record["decision"] = decision.to_public_dict()
        ledger_validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertEqual(evidence.reject_reason, "required_score_missing")
        self.assertEqual(evidence.score_stats, {})
        self.assertEqual(decision.reason, evidence.reject_reason)
        self.assertTrue(ledger_validation.valid, ledger_validation.reasons)

    def test_public_score_is_canonical_only_when_bound_to_acceptance_mean(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]

        valid = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=0.50,
        )
        mismatched = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=0.99,
        )

        self.assertTrue(valid.public_score_binding_valid, valid.reasons)
        self.assertEqual(valid.canonical_public_score, 0.50)
        self.assertFalse(mismatched.public_score_binding_valid)
        self.assertIsNone(mismatched.canonical_public_score)
        self.assertIn("public_score_binding_mismatch", mismatched.reasons)
        self.assertEqual(mismatched.reject_reason, "public_score_binding_mismatch")

    def test_decision_comparison_preserves_fractional_score_evidence(self) -> None:
        public_score = 1.0 / 140.0
        record = valid_accepted_ledger_record(
            round_id="round-fractional-score",
            candidate_id="candidate-fractional-score",
            public_score=public_score,
        )
        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level", True, "code_level_candidate"
            ),
            public_result=PublicEvalResult(**record["public_result"]),
            current_best=_bootstrap(
                TEST_ACCEPTANCE_PROTOCOL,
                TEST_EVALUATION_BINDING,
            ),
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        record["decision"] = decision.to_public_dict()

        validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertEqual(
            decision.comparison["candidate_score"],
            public_score,
        )
        self.assertTrue(validation.valid, validation.reasons)


def _record() -> dict[str, object]:
    return copy.deepcopy(
        valid_accepted_ledger_record(
            round_id="round-acceptance",
            candidate_id="candidate-acceptance",
            public_score=0.50,
        )
    )


def _bootstrap(
    protocol: AcceptanceProtocol,
    binding: EvaluationBindingV1,
) -> CurrentBestEvalStats:
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=0.0,
        acceptance_protocol=protocol,
        evaluation_binding=binding,
        anchor_id="acceptance-contract-bootstrap",
    )


def _paired_protocol() -> AcceptanceProtocol:
    return AcceptanceProtocol(
        protocol_id="test_paired_acceptance_v1",
        score_source=TEST_ACCEPTANCE_PROTOCOL.score_source,
        min_score_count=3,
        absolute_accept_floor=0.05,
        min_delta=0.0,
        public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        decision_rule=PAIRED_HEADROOM_DELTA_RULE,
        score_ceiling=1.0,
        headroom_reduction_fraction=0.2,
        paired_standard_error_multiplier=1.5,
    )


def _guarded_protocol() -> AcceptanceProtocol:
    return AcceptanceProtocol(
        protocol_id="test_guarded_acceptance_v1",
        score_source="acceptance_objective.score_stats.mean",
        min_score_count=3,
        absolute_accept_floor=0.2,
        min_delta=0.0,
        public_score_binding=PUBLIC_SCORE_BINDING_GUARD_MEAN,
        decision_rule=GUARDED_PAIRED_HEADROOM_DELTA_RULE,
        score_ceiling=1.0,
        headroom_reduction_fraction=0.2,
        paired_standard_error_multiplier=1.5,
        guard_score_source="public_accept.score_stats.mean",
        guard_absolute_accept_floor=0.2,
        guard_max_regression=0.0,
    )


def _named_guard_protocol() -> AcceptanceProtocol:
    return AcceptanceProtocol(
        protocol_id="test_named_guard_acceptance_v1",
        score_source="unified_score.j_primary.score_stats.mean",
        min_score_count=3,
        absolute_accept_floor=0.2,
        min_delta=0.0,
        public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        decision_rule=NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE,
        score_ceiling=1.0,
        headroom_reduction_fraction=0.2,
        paired_standard_error_multiplier=1.5,
        named_guards=tuple(
            NamedGuardSpec(
                name=name,
                score_source=f"unified_score.{name}.score_stats.mean",
                absolute_accept_floor=0.1,
                max_regression=0.05,
                regression_margin=0.02,
            )
            for name in ("j_ood", "c0_id", "c1_id")
        ),
        max_public_to_sealed_score_drop=0.05,
    )


def _guarded_record(
    *,
    objective_score: float,
    guard_score: float,
    protocol: AcceptanceProtocol,
    binding: EvaluationBindingV1,
) -> dict[str, object]:
    record = _record()
    result = record["public_result"]
    result["public_score"] = guard_score
    metrics = result["metrics"]

    def score_stats(value: float) -> dict[str, object]:
        return {
            "count": 3,
            "mean": value,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
        }

    objective_stats = score_stats(objective_score)
    guard_stats = score_stats(guard_score)
    metrics["acceptance_objective"] = {"score_stats": objective_stats}
    metrics["public_accept"] = {"score_stats": guard_stats}
    metrics["evaluation_binding"] = binding.to_public_dict()
    metrics["acceptance"].update(
        {
            "score_source": protocol.score_source,
            "public_score_binding": protocol.public_score_binding,
            "score_stats": objective_stats,
            "guard_score_source": protocol.guard_score_source,
            "guard_score_stats": guard_stats,
            "guard_absolute_accept_floor": protocol.guard_absolute_accept_floor,
            "guard_max_regression": protocol.guard_max_regression,
            "absolute_accept_floor": protocol.absolute_accept_floor,
            "min_delta": protocol.min_delta,
            "min_score_count": protocol.min_score_count,
            "acceptance_protocol_hash": protocol.protocol_hash,
            "score_samples": [objective_score] * 3,
        }
    )
    metrics["sealed_validation"]["evaluation_binding_hash"] = (
        binding.binding_hash
    )
    return record


def _named_guard_record(
    *,
    objective_score: float,
    guard_scores: dict[str, float],
    protocol: AcceptanceProtocol,
    binding: EvaluationBindingV1,
) -> dict[str, object]:
    record = _record()
    result = record["public_result"]
    result["public_score"] = objective_score
    metrics = result["metrics"]

    def score_stats(value: float) -> dict[str, object]:
        return {
            "count": 3,
            "mean": value,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
        }

    objective_stats = score_stats(objective_score)
    unified_score = {
        "j_primary": {
            "score_stats": objective_stats,
            "seed_macro_scores": [objective_score] * 3,
        }
    }
    named_guards = {}
    for guard in protocol.named_guards:
        score = guard_scores[guard.name]
        stats = score_stats(score)
        unified_score[guard.name] = {
            "score_stats": stats,
            "seed_macro_scores": [score] * 3,
        }
        named_guards[guard.name] = {
            **guard.to_public_dict(),
            "score_stats": stats,
            "score_samples": [score] * 3,
        }
    required_pass_gates = list(metrics["acceptance"]["required_pass_gates"])
    metrics["unified_score"] = unified_score
    metrics["evaluation_binding"] = binding.to_public_dict()
    metrics["acceptance"] = {
        "score_source": protocol.score_source,
        "public_score_binding": protocol.public_score_binding,
        "score_stats": objective_stats,
        "score_samples": [objective_score] * 3,
        "named_guards": named_guards,
        "min_score_count": protocol.min_score_count,
        "required_pass_gates": required_pass_gates,
        "absolute_accept_floor": protocol.absolute_accept_floor,
        "min_delta": protocol.min_delta,
        "max_public_to_sealed_score_drop": (
            protocol.max_public_to_sealed_score_drop
        ),
        "acceptance_protocol_hash": protocol.protocol_hash,
    }
    metrics["sealed_validation"]["evaluation_binding_hash"] = binding.binding_hash
    return record


if __name__ == "__main__":
    unittest.main()

"""Early comparison agrees with the existing complete acceptance decision."""

import unittest

from tests.unit import test_acceptance_contract as fixtures
from true_rl_agent.agent_core.decision import decide_candidate
from true_rl_agent.agent_core.generation_early_rejection import score_rejection_reason
from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.public_accept_scores import PublicAcceptScores
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.derived_statistics import sample_statistics


class EarlyRejectionTests(unittest.TestCase):
    def test_guard_headroom_equality_and_paired_uncertainty_match_full_decisions(self):
        for named in (False, True):
            protocol = fixtures._named_guard_protocol() if named else fixtures._guarded_protocol()
            binding = EvaluationBindingV1(eval_protocol_hash=fixtures.TEST_EVALUATION_PROTOCOL.protocol_hash,
                acceptance_protocol_hash=protocol.protocol_hash,
                task_protocol_hash=fixtures.TEST_EVALUATION_BINDING.task_protocol_hash,
                seed_cohort_hash=fixtures.TEST_EVALUATION_BINDING.seed_cohort_hash)

            def record(score, guard):
                if named:
                    return fixtures._named_guard_record(objective_score=score,
                        guard_scores={"j_ood": 0.5, "c0_id": guard, "c1_id": 0.5},
                        protocol=protocol, binding=binding)
                return fixtures._guarded_record(objective_score=score, guard_score=guard,
                    protocol=protocol, binding=binding)

            baseline = record(0.5, 0.5)
            parent = CurrentBestEvalStats.from_public_result_mapping(
                round_id="round-early", candidate_id="candidate-acceptance",
                public_result=baseline["public_result"], acceptance_protocol=protocol,
                subject_manifest_hash=fixtures.TEST_SUBJECT_MANIFEST_HASH,
                program_source_checksum=fixtures.TEST_PROGRAM_SOURCE_CHECKSUM,
                training_binding_hash=fixtures.TEST_TRAINING_BINDING_HASH,
                capability_profile_hash=fixtures.TEST_CAPABILITY_PROFILE_HASH,
                checkpoint_hash="sha256:" + "d" * 64, expected_evaluation_binding=binding)
            cases = [("equality", record(0.6, 0.5)), ("improved", record(0.7, 0.5)),
                     ("guard_regression", record(0.9, 0.3))]
            noisy = record(2 / 3, 0.5)
            metrics = noisy["public_result"]["metrics"]
            samples = [0.3, 0.7, 1.0]
            stats = {**sample_statistics(samples, statistics_version=None),
                "eval_protocol_hash": binding.eval_protocol_hash}
            source = metrics["unified_score"]["j_primary"] if named else metrics["acceptance_objective"]
            source.update(score_stats=stats, seed_macro_scores=samples)
            metrics["acceptance"].update(score_stats=stats, score_samples=samples)
            if named:
                noisy["public_result"]["public_score"] = stats["mean"]
            cases.append(("paired_noise", noisy))
            for name, candidate in cases:
                with self.subTest(named=named, case=name):
                    result = PublicEvalResult(**candidate["public_result"])
                    validation = validate_acceptance_evidence(result.metrics,
                        acceptance_protocol=protocol, public_score=result.public_score)
                    self.assertTrue(validation.valid, validation.reasons)
                    scores = PublicAcceptScores(**{field: getattr(validation, field)
                        for field in PublicAcceptScores.__dataclass_fields__})
                    early_reason = score_rejection_reason(scores.to_dict(), parent=parent, protocol=protocol)
                    full = decide_candidate(surface=fixtures.CandidateSurfaceAssessment(
                        "code_level", True, "code_level_candidate"), public_result=result,
                        current_best=parent, acceptance_protocol=protocol)
                    self.assertEqual(early_reason, None if full.accepted else full.reason)
                    self.assertEqual(early_reason is None, name == "improved")
                    self.assertNotIn("sealed_validation", scores.to_dict())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import make_evaluation_binding
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.sealed_validation import (
    build_sealed_rollout_evidence,
    validate_sealed_rollout_evidence,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.job_identity import (
    JobIdentity,
    validate_job_summary_identity,
)
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
)
from true_rl_agent.safety.eval_seal import build_trusted_eval_seal
from true_rl_agent.tasks.base import TaskInvariantSpec

_CANDIDATE_CHECKSUM = "sha256:" + "2" * 64
_CANDIDATE_MANIFEST_HASH = "c" * 64
_PROGRAM_SOURCE_CHECKSUM = "sha256:" + "3" * 64
_TRAINING_BINDING_HASH = "4" * 64
_CAPABILITY_PROFILE_HASH = "5" * 64


class EvaluationBindingTests(unittest.TestCase):
    def _checkpoint_safety_report(self, checkpoint_hash: str) -> dict:
        return {
            "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
            "status": "pass",
            "checkpoint_hash": checkpoint_hash,
            "expected_checkpoint_hash": checkpoint_hash,
            "file_exists": True,
            "byte_read_status": "pass",
            "weights_only_load_status": "pass",
            "checkpoint_hash_match_status": "pass",
            "loader_error_class": None,
            "loader_error_summary": None,
            "reason": "checkpoint_safe_to_load",
        }

    def _score_stats(self, protocol_hash: str) -> dict:
        return {
            "count": 7,
            "mean": 0.5,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": protocol_hash,
        }

    def test_subject_identity_rejects_paths_and_noncanonical_mappings(self) -> None:
        for subject_id in (
            "",
            " candidate",
            "candidate ",
            ".",
            "..",
            "../candidate",
            "a/b",
            "a\\b",
            "x://y",
            "candidate id",
            "candidate:id",
            "Candidate",
            "a.",
            "nul",
            "con.txt",
            "a" * 129,
        ):
            with self.subTest(subject_id=subject_id):
                with self.assertRaises(ValueError):
                    ExperimentSubject.candidate(subject_id)
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            ExperimentSubject.from_mapping(
                {
                    "schema_version": "true_rl_agent.evaluation.subject.v1",
                    "kind": "candidate",
                    "id": "candidate-001",
                    "candidate_path": "forbidden",
                }
            )

    def test_candidate_subject_deserialization_uses_shared_candidate_id_contract(self) -> None:
        for subject_id in ("Candidate", "a.", "nul", "con.txt"):
            with self.subTest(subject_id=subject_id):
                with self.assertRaises(ValueError):
                    ExperimentSubject.from_mapping(
                        {
                            "schema_version": "true_rl_agent.experiment.subject.v2",
                            "kind": "candidate",
                            "id": subject_id,
                        }
                    )

        self.assertEqual(
            ExperimentSubject.baseline("Official-Baseline").subject_id,
            "Official-Baseline",
        )

    def test_binding_hash_covers_every_comparison_domain_constituent(self) -> None:
        binding = make_evaluation_binding()
        public = binding.to_public_dict()
        for field in (
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "task_protocol_hash",
            "seed_cohort_hash",
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(public)
                tampered[field] = "f" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "evaluation binding hash mismatch",
                ):
                    EvaluationBindingV1.from_mapping(tampered)
                changed = replace(binding, **{field: "f" * 64})
                self.assertNotEqual(changed.binding_hash, binding.binding_hash)

    def test_job_specs_keep_candidate_and_baseline_checkpoint_roles_disjoint(
        self,
    ) -> None:
        binding = make_evaluation_binding()
        candidate_checkpoint = CheckpointRef(
            "artifact://candidate.pt",
            _CANDIDATE_CHECKSUM,
            "eval",
            sealed=True,
        )
        baseline_checkpoint = CheckpointRef(
            "baseline://official.pt",
            "sha256:" + "1" * 64,
            "warm_start",
            sealed=True,
        )
        candidate = EvalJobSpec(
            job_id="candidate-eval",
            task_id="task",
            subject=ExperimentSubject.candidate("candidate-001"),
            subject_manifest_hash=_CANDIDATE_MANIFEST_HASH,
            program_source_checksum=_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=_TRAINING_BINDING_HASH,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint=candidate_checkpoint,
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=binding,
            execution_device="cuda:0",
        )
        baseline = BaselineEvalJobSpec(
            job_id="baseline-eval",
            task_id="task",
            subject=ExperimentSubject.baseline("official-baseline"),
            subject_manifest_hash="d" * 64,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint=baseline_checkpoint,
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=binding,
            execution_device="cuda:0",
        )
        self.assertEqual(candidate.evaluation_subject().kind, "candidate")
        self.assertEqual(
            candidate.evaluation_manifest_hash(),
            _CANDIDATE_MANIFEST_HASH,
        )
        self.assertEqual(baseline.evaluation_subject().kind, "baseline")
        self.assertEqual(baseline.evaluation_manifest_hash(), "d" * 64)

        with self.assertRaisesRegex(ValueError, "role='eval'"):
            replace(candidate, checkpoint=baseline_checkpoint)
        with self.assertRaisesRegex(ValueError, "role='warm_start'"):
            replace(baseline, checkpoint=candidate_checkpoint)
        with self.assertRaisesRegex(ValueError, "subject_manifest_hash"):
            JobIdentity(
                job_id="baseline-eval",
                task_id="task",
                subject=ExperimentSubject.baseline("official-baseline"),
            )

    def test_eval_identity_rejects_legacy_or_cross_kind_fields(self) -> None:
        candidate = JobIdentity(
            job_id="candidate-eval",
            task_id="task",
            subject=ExperimentSubject.candidate("candidate-001"),
        )
        baseline = JobIdentity(
            job_id="baseline-eval",
            task_id="task",
            subject=ExperimentSubject.baseline("official-baseline"),
            subject_manifest_hash="d" * 64,
        )
        for expected, extra in (
            (candidate, {"candidate_id": "candidate-001"}),
            (candidate, {"baseline_manifest_hash": "d" * 64}),
            (baseline, {"candidate_id": "candidate-001"}),
            (baseline, {"baseline_id": "official-baseline"}),
            (
                baseline,
                {
                    "evaluation_subject": ExperimentSubject.candidate(
                        "forged-candidate"
                    ).to_public_dict()
                },
            ),
            (
                baseline,
                {
                    "subject": ExperimentSubject.candidate(
                        "forged-candidate"
                    ).to_public_dict()
                },
            ),
        ):
            with self.subTest(kind=expected.subject.kind, extra=extra):
                with self.assertRaisesRegex(ValueError, "must not carry"):
                    validate_job_summary_identity(
                        {**expected.to_public_dict(), **extra},
                        expected=expected,
                        summary_name="eval summary",
                    )

    def test_eval_job_counts_reject_boolean_values(self) -> None:
        binding = make_evaluation_binding()
        candidate = EvalJobSpec(
            job_id="candidate-eval",
            task_id="task",
            subject=ExperimentSubject.candidate("candidate-001"),
            subject_manifest_hash=_CANDIDATE_MANIFEST_HASH,
            program_source_checksum=_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=_TRAINING_BINDING_HASH,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint=CheckpointRef(
                "artifact://candidate.pt",
                _CANDIDATE_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=binding,
            execution_device="cuda:0",
        )
        baseline = BaselineEvalJobSpec(
            job_id="baseline-eval",
            task_id="task",
            subject=ExperimentSubject.baseline("official-baseline"),
            subject_manifest_hash="d" * 64,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint=CheckpointRef(
                "baseline://official.pt",
                "sha256:" + "1" * 64,
                "warm_start",
                sealed=True,
            ),
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=binding,
            execution_device="cuda:0",
        )
        for spec in (candidate, baseline):
            for field_name in (
                "public_seed_count",
                "public_accept_seed_count",
                "sealed_validation_seed_count",
                "eval_num_envs",
            ):
                with self.subTest(
                    kind=spec.evaluation_subject().kind,
                    field=field_name,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be a positive integer",
                    ):
                        replace(spec, **{field_name: True})

    def test_sealed_evidence_binds_subject_and_baseline_manifest(self) -> None:
        binding = make_evaluation_binding()
        protocol_hash = binding.eval_protocol_hash
        baseline = ExperimentSubject.baseline("official-baseline")
        manifest_hash = "d" * 64
        evidence = build_sealed_rollout_evidence(
            score_stats=self._score_stats(protocol_hash),
            expected_seed_count=7,
            eval_protocol_hash=protocol_hash,
            evaluation_binding_hash=binding.binding_hash,
            evaluation_subject=baseline,
            subject_manifest_hash=manifest_hash,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint_hash="sha256:" + "1" * 64,
            task_validation_status="pass",
            reason="holdout_passed",
        )
        validation = validate_sealed_rollout_evidence(
            evidence,
            expected_evaluation_binding_hash=binding.binding_hash,
            expected_evaluation_subject=baseline,
            expected_subject_manifest_hash=manifest_hash,
            expected_capability_profile_hash=_CAPABILITY_PROFILE_HASH,
        )
        self.assertTrue(validation.complete, validation.reasons)

        forged = copy.deepcopy(evidence)
        forged["evaluation_subject"]["id"] = "other-baseline"
        self.assertFalse(
            validate_sealed_rollout_evidence(
                forged,
                expected_evaluation_subject=baseline,
                expected_subject_manifest_hash=manifest_hash,
                expected_capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            ).complete
        )
        with self.assertRaisesRegex(
            ValueError,
            "subject_manifest_hash is required",
        ):
            build_sealed_rollout_evidence(
                score_stats=self._score_stats(protocol_hash),
                expected_seed_count=7,
                eval_protocol_hash=protocol_hash,
                evaluation_binding_hash=binding.binding_hash,
                evaluation_subject=baseline,
                capability_profile_hash=_CAPABILITY_PROFILE_HASH,
                checkpoint_hash="sha256:" + "1" * 64,
                task_validation_status="pass",
                reason="holdout_passed",
            )

    def test_trusted_seal_includes_baseline_manifest_binding(self) -> None:
        checkpoint_hash = "sha256:" + "1" * 64
        eval_spec = BaselineEvalJobSpec(
            job_id="baseline-eval",
            task_id="task",
            subject=ExperimentSubject.baseline("official-baseline"),
            subject_manifest_hash="d" * 64,
            capability_profile_hash=_CAPABILITY_PROFILE_HASH,
            checkpoint=CheckpointRef(
                "baseline://official.pt",
                checkpoint_hash,
                "warm_start",
                sealed=True,
            ),
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=make_evaluation_binding(),
            execution_device="cuda:0",
        )

        seal = build_trusted_eval_seal(
            eval_spec=eval_spec,
            task_invariants=TaskInvariantSpec({"fixed": True}),
            checkpoint_safety_report=self._checkpoint_safety_report(
                checkpoint_hash
            ),
            candidate_isolation_status="pass",
        ).to_public_dict()

        self.assertEqual(seal["subject_kind"], "baseline")
        self.assertEqual(seal["subject_id"], "official-baseline")
        self.assertEqual(seal["subject_manifest_hash"], "d" * 64)


if __name__ == "__main__":
    unittest.main()

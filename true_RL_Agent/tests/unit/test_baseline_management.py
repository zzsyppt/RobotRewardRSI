from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    BaselineSourceBinding,
    CheckpointProvenance,
    validate_baseline_compatibility,
)
from true_rl_agent.runner.jobs import dry_run_train_command
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec


class BaselineManagementTests(unittest.TestCase):
    def _manifest(self, *, sealed: bool = True, backend_id: str = "isaac_lab_rsl_rl") -> BaselineManifest:
        return BaselineManifest(
            baseline_id="push-warm-start-v1",
            task_id="unitree_push_recovery",
            checkpoint=CheckpointRef(
                uri="baseline://push/policy.pt",
                checksum=f"sha256:{'1' * 64}",
                role="warm_start",
                sealed=sealed,
            ),
            provenance=CheckpointProvenance(
                source_run_id="seed-run-001",
                source_task_id="unitree_push_recovery",
                created_by="runner",
            ),
            compatibility=BaselineCompatibilitySpec(
                backend_id=backend_id,
                robot="unitree_g1",
                observation_schema_hash="obs-hash",
                action_schema_hash="act-hash",
                policy_architecture="rsl_rl_ppo_mlp",
            ),
        )

    def test_random_initialization_needs_no_checkpoint(self) -> None:
        init = PolicyInitialization.random()

        self.assertIsNone(init.baseline)

    def test_baseline_manifest_builds_warm_start_policy_initialization(self) -> None:
        manifest = self._manifest()

        init = manifest.to_policy_initialization()

        self.assertEqual(init.kind, "warm_start")
        self.assertEqual(init.baseline.baseline_id, "push-warm-start-v1")
        self.assertEqual(init.baseline.checkpoint.role, "warm_start")
        self.assertEqual(init.baseline.checkpoint.uri, "baseline://push/policy.pt")
        self.assertEqual(
            init.baseline.baseline_manifest_hash,
            manifest.manifest_hash,
        )

    def test_source_binding_participates_in_runtime_manifest_identity(self) -> None:
        manifest = self._manifest()
        bound = replace(
            manifest,
            source_binding=BaselineSourceBinding(
                source_schema_version="task.baseline.v1",
                source_manifest_hash="2" * 64,
                trust_anchor_hash="3" * 64,
                admission_evidence_hash="4" * 64,
            ),
            manifest_hash=None,
        )

        self.assertNotEqual(bound.manifest_hash, manifest.manifest_hash)
        self.assertEqual(
            bound.to_public_dict()["source_binding"]["source_manifest_hash"],
            "2" * 64,
        )

    def test_baseline_manifest_keeps_local_checkpoint_path_private(self) -> None:
        manifest = self._manifest()
        with_private_path = replace(
            manifest,
            checkpoint=replace(
                manifest.checkpoint,
                metadata={"local_path": "D:/private/policy.pt"},
            ),
            manifest_hash=None,
        )

        self.assertNotIn(
            "local_path",
            str(with_private_path.to_public_dict()),
        )
        for local_uri in ("D:/private/policy.pt", "/private/policy.pt", "file:///private/policy.pt"):
            with self.subTest(local_uri=local_uri):
                with self.assertRaisesRegex(
                    ValueError,
                    "non-local artifact URI",
                ):
                    replace(
                        manifest,
                        checkpoint=replace(
                            manifest.checkpoint,
                            uri=local_uri,
                        ),
                        manifest_hash=None,
                    )

    def test_baseline_compatibility_checks_backend_robot_obs_action_policy_and_reference(self) -> None:
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec({"fall_threshold": 0.30}),
        )
        backend = BackendSpec(
            backend_id="isaac_lab_rsl_rl",
            supported_tasks=frozenset({"unitree_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            supports_baseline_eval=True,
        )
        manifest = self._manifest()

        report = validate_baseline_compatibility(
            manifest=manifest,
            task_spec=task,
            backend_spec=backend,
            robot="unitree_g1",
            observation_schema_hash="obs-hash",
            action_schema_hash="act-hash",
            policy_architecture="rsl_rl_ppo_mlp",
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.checkpoint_reference_status, "pass")

    def test_incompatible_backend_or_unsealed_checkpoint_fails_before_training(self) -> None:
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec({"fall_threshold": 0.30}),
        )
        backend = BackendSpec(
            backend_id="other_backend",
            supported_tasks=frozenset({"unitree_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
        )

        report = validate_baseline_compatibility(
            manifest=self._manifest(sealed=False),
            task_spec=task,
            backend_spec=backend,
            robot="unitree_g1",
            observation_schema_hash="obs-hash",
            action_schema_hash="act-hash",
            policy_architecture="rsl_rl_ppo_mlp",
        )

        self.assertFalse(report.compatible)
        self.assertIn("backend_id_mismatch", report.reasons)
        self.assertIn("baseline_checkpoint_not_sealed", report.reasons)

    def test_warm_start_train_job_exposes_init_checkpoint_separately_from_eval_checkpoint(self) -> None:
        manifest = self._manifest()
        init = manifest.to_policy_initialization()
        candidate_artifact = make_verified_candidate_artifact("candidate-001")
        train_spec = TrainJobSpec(
            job_id="train-warm",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("candidate-001"),
            subject_manifest_hash=candidate_artifact.subject_manifest_hash,
            program_source_checksum=candidate_artifact.program_source_checksum,
            policy_initialization=init,
            training_binding=make_training_binding(init),
            budget_steps=128,
            candidate_artifact=candidate_artifact,
        )
        eval_spec = EvalJobSpec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("candidate-001"),
            subject_manifest_hash=candidate_artifact.subject_manifest_hash,
            program_source_checksum=candidate_artifact.program_source_checksum,
            training_binding_hash=str(make_training_binding(init).binding_hash),
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=CheckpointRef(
                uri="artifact://round/checkpoints/best.pt",
                checksum="sha256:" + "2" * 64,
                role="eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )

        command = dry_run_train_command(train_spec)

        self.assertEqual(train_spec.training_start_checkpoint().uri, "baseline://push/policy.pt")
        self.assertNotEqual(train_spec.training_start_checkpoint().uri, eval_spec.checkpoint.uri)
        self.assertIn("--warm-start baseline://push/policy.pt", command)


if __name__ == "__main__":
    unittest.main()

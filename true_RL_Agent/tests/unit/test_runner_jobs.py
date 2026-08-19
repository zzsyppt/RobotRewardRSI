from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.artifacts import RunnerArtifactLayout
from true_rl_agent.runner.budget import (
    BudgetViolation,
    build_budget_accounting,
    enforce_fixed_budget,
)
from true_rl_agent.runner.checkpoint import (
    build_checkpoint_safety_report,
    seal_checkpoint,
    validate_checkpoint_safety_report,
)
from true_rl_agent.runner.dry_run import DryRunAuditRunner, DryRunCandidateMetadata
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.isolation import IsolationPolicy
from true_rl_agent.runner.jobs import dry_run_eval_command, dry_run_train_command
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
)
from true_rl_agent.safety.eval_seal import build_trusted_eval_seal
from true_rl_agent.safety.invariants import enforce_task_invariants
from true_rl_agent.tasks.base import TaskInvariantSpec

_BASELINE_CHECKSUM = "sha256:" + "b" * 64
_EVAL_CHECKSUM = "sha256:" + "e" * 64
_OTHER_CHECKSUM = "sha256:" + "0" * 64
_RAW_CHECKSUM = "sha256:" + "a" * 64


def _train_job_spec(*, candidate_id: str, **kwargs) -> TrainJobSpec:
    initialization = kwargs["policy_initialization"]
    budget_steps = kwargs["budget_steps"]
    candidate_artifact = make_verified_candidate_artifact(candidate_id)
    kwargs.setdefault("execution_device", "cuda:0")
    return TrainJobSpec(
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=candidate_artifact.subject_manifest_hash,
        program_source_checksum=candidate_artifact.program_source_checksum,
        training_binding=make_training_binding(
            initialization,
            budget_steps=budget_steps,
        ),
        candidate_artifact=candidate_artifact,
        **kwargs,
    )


def _eval_job_spec(*, candidate_id: str, **kwargs) -> EvalJobSpec:
    kwargs.setdefault("execution_device", "cuda:0")
    return EvalJobSpec(
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
        program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
        training_binding_hash="6" * 64,
        capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
        **kwargs,
    )


class RunnerJobTests(unittest.TestCase):
    def _checkpoint_safety_report(self, *, status: str = "pass", reason: str = "checkpoint_safe_to_load") -> dict:
        report = {
            "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
            "status": status,
            "checkpoint_hash": _EVAL_CHECKSUM,
            "expected_checkpoint_hash": _EVAL_CHECKSUM,
            "file_exists": True,
            "byte_read_status": "pass",
            "weights_only_load_status": "pass" if status == "pass" else "fail",
            "checkpoint_hash_match_status": "pass",
            "loader_error_class": None if status == "pass" else "RuntimeError",
            "loader_error_summary": None if status == "pass" else "RuntimeError",
            "reason": reason,
        }
        return report

    def test_dry_run_commands_distinguish_random_and_warm_start_train_jobs(self) -> None:
        random_spec = _train_job_spec(
            job_id="train-random",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
        )
        baseline_checkpoint = CheckpointRef(
            uri="baseline://push/policy.pt",
            checksum=_BASELINE_CHECKSUM,
            role="warm_start",
            sealed=True,
        )
        warm_spec = _train_job_spec(
            job_id="train-warm",
            task_id="unitree_push_recovery",
            candidate_id="candidate-002",
            policy_initialization=PolicyInitialization.warm_start(
                BaselinePolicyRef(
                    baseline_id="push-baseline",
                    task_id="unitree_push_recovery",
                    checkpoint=baseline_checkpoint,
                )
            ),
            budget_steps=128,
        )

        random_command = dry_run_train_command(random_spec)
        warm_command = dry_run_train_command(warm_spec)

        self.assertIn("--init random", random_command)
        self.assertIn("--device cuda:0", random_command)
        self.assertNotIn("--warm-start", random_command)
        self.assertIn("--init warm_start", warm_command)
        self.assertIn("--warm-start baseline://push/policy.pt", warm_command)

    def test_eval_dry_run_command_uses_eval_checkpoint_only(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
            execution_device="cuda:0",
        )

        command = dry_run_eval_command(eval_spec)

        self.assertIn("--checkpoint artifact://round/best.pt", command)
        self.assertIn(f"--checkpoint-sha256 {_EVAL_CHECKSUM}", command)
        self.assertIn("--subject-kind candidate", command)
        self.assertIn("--subject-id candidate-001", command)
        self.assertNotIn("--candidate-id", command)
        self.assertNotIn("--candidate candidate-001", command)
        self.assertIn("--public-seeds 4", command)
        self.assertIn("--device cuda:0", command)
        self.assertNotIn("trusted_seed_count", command)
        self.assertEqual(eval_spec.to_public_dict()["sealed_validation_seed_count"], 8)
        self.assertIn("eval_protocol_hash", eval_spec.to_public_dict())
        self.assertNotIn("trusted_seed_count", eval_spec.to_public_dict())

    def test_eval_binding_rejects_checkpoint_execution_device_drift(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-device-lineage",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=None,
            checkpoint_output_plan=CheckpointOutputPlan(
                "artifact://round/planned.pt"
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            execution_device="cuda:1",
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        checkpoint = CheckpointRef(
            uri="artifact://round/sealed.pt",
            checksum=_EVAL_CHECKSUM,
            role="eval",
            sealed=True,
            metadata={
                "training_subject": eval_spec.subject.to_public_dict(),
                "subject_manifest_hash": eval_spec.subject_manifest_hash,
                "program_source_checksum": eval_spec.program_source_checksum,
                "training_binding_hash": eval_spec.training_binding_hash,
                "capability_profile_hash": eval_spec.capability_profile_hash,
                "execution_device": "cuda:0",
            },
        )

        with self.assertRaisesRegex(ValueError, "execution_device"):
            bind_eval_spec_to_checkpoint(eval_spec, checkpoint)

    def test_bound_eval_requires_exact_checkpoint_gpu_process_binding(self) -> None:
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="9" * 64,
        )
        eval_spec = _eval_job_spec(
            job_id="eval-gpu-lineage",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=None,
            checkpoint_output_plan=CheckpointOutputPlan(
                "artifact://round/planned.pt"
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            execution_device="cuda:0",
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
            gpu_process_binding=binding,
        )
        lineage = {
            "training_subject": eval_spec.subject.to_public_dict(),
            "subject_manifest_hash": eval_spec.subject_manifest_hash,
            "program_source_checksum": eval_spec.program_source_checksum,
            "training_binding_hash": eval_spec.training_binding_hash,
            "capability_profile_hash": eval_spec.capability_profile_hash,
            "execution_device": eval_spec.execution_device,
        }
        checkpoint = CheckpointRef(
            uri="artifact://round/sealed.pt",
            checksum=_EVAL_CHECKSUM,
            role="eval",
            sealed=True,
            metadata={**lineage, "gpu_process_binding": binding.to_dict()},
        )

        bound = bind_eval_spec_to_checkpoint(eval_spec, checkpoint)
        self.assertEqual(bound.gpu_process_binding, binding)

        with self.assertRaisesRegex(ValueError, "GPU process binding"):
            bind_eval_spec_to_checkpoint(
                replace(eval_spec, gpu_process_binding=None),
                checkpoint,
            )

        invalid_bindings = (
            None,
            {
                **binding.to_dict(),
                "unexpected": "field",
            },
            GpuProcessBinding(
                host_execution_device="cuda:2",
                process_execution_device="cuda:0",
                resource_binding_hash=binding.resource_binding_hash,
            ).to_dict(),
        )
        for checkpoint_binding in invalid_bindings:
            with self.subTest(checkpoint_binding=checkpoint_binding):
                with self.assertRaisesRegex(ValueError, "GPU process binding"):
                    bind_eval_spec_to_checkpoint(
                        eval_spec,
                        replace(
                            checkpoint,
                            metadata={
                                **lineage,
                                "gpu_process_binding": checkpoint_binding,
                            },
                        ),
                    )

    def test_eval_binding_carries_verified_initial_policy_pairing_hash(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-initial-policy-lineage",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=None,
            checkpoint_output_plan=CheckpointOutputPlan(
                "artifact://round/planned.pt"
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            execution_device="cuda:1",
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        pairing = InitialPolicyStatePairing(
            expected_initial_policy_state_hash="1" * 64,
            actual_initial_policy_state_hash="1" * 64,
            match=True,
        )
        checkpoint = CheckpointRef(
            uri="artifact://round/sealed.pt",
            checksum=_EVAL_CHECKSUM,
            role="eval",
            sealed=True,
            metadata={
                "training_subject": eval_spec.subject.to_public_dict(),
                "subject_manifest_hash": eval_spec.subject_manifest_hash,
                "program_source_checksum": eval_spec.program_source_checksum,
                "training_binding_hash": eval_spec.training_binding_hash,
                "capability_profile_hash": eval_spec.capability_profile_hash,
                "execution_device": "cuda:1",
                "initial_policy_state_pairing": pairing.to_public_dict(),
            },
        )

        bound = bind_eval_spec_to_checkpoint(eval_spec, checkpoint)

        self.assertEqual(
            bound.initial_policy_state_pairing_hash,
            pairing.pairing_hash,
        )
        self.assertEqual(
            bound.to_public_dict()["initial_policy_state_pairing_hash"],
            pairing.pairing_hash,
        )

    def test_dry_run_runner_rejects_unbound_checkpoint_plan(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-planned",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=None,
            checkpoint_output_plan=CheckpointOutputPlan(
                "artifact://round/planned.pt"
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        runner = DryRunAuditRunner(
            {
                "candidate-001": DryRunCandidateMetadata(
                    changed_components=(),
                    public_score=0.0,
                )
            },
            eval_result_builder=lambda *_: self.fail(
                "planned eval must not reach the result builder"
            ),
            train_summary_builder=lambda *_: {},
        )

        with self.assertRaisesRegex(ValueError, "checkpoint bytes have not been bound"):
            runner.run_eval(eval_spec)

    def test_dry_run_runner_rechecks_warm_start_checkpoint_state(self) -> None:
        checkpoint = CheckpointRef(
            uri="baseline://push/policy.pt",
            checksum=_BASELINE_CHECKSUM,
            role="warm_start",
            sealed=True,
        )
        baseline_spec = BaselineEvalJobSpec(
            job_id="baseline-eval",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.baseline("push-baseline"),
            subject_manifest_hash="c" * 64,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=checkpoint,
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
            execution_device="cuda:0",
        )
        train_spec = _train_job_spec(
            job_id="train-warm",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            policy_initialization=PolicyInitialization.warm_start(
                BaselinePolicyRef(
                    baseline_id="push-baseline",
                    task_id="unitree_push_recovery",
                    checkpoint=checkpoint,
                )
            ),
            budget_steps=128,
        )
        runner = DryRunAuditRunner(
            {
                "candidate-001": DryRunCandidateMetadata(
                    changed_components=(),
                    public_score=0.0,
                )
            },
            eval_result_builder=lambda *_: self.fail(
                "tampered checkpoint must not reach eval builder"
            ),
            train_summary_builder=lambda *_: self.fail(
                "tampered checkpoint must not reach train builder"
            ),
            baseline_eval_result_builder=lambda *_: self.fail(
                "tampered checkpoint must not reach baseline builder"
            ),
        )
        object.__setattr__(checkpoint, "sealed", False)

        with self.assertRaises(ValueError):
            runner.run_baseline_eval(baseline_spec)
        with self.assertRaises(ValueError):
            runner.run_train(train_spec)

    def test_eval_job_accepts_trusted_seed_count_only_as_legacy_constructor_alias(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-legacy-alias",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            trusted_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )

        self.assertEqual(eval_spec.sealed_validation_seed_count, 8)
        self.assertEqual(eval_spec.to_public_dict()["sealed_validation_seed_count"], 8)
        self.assertNotIn("trusted_seed_count", eval_spec.to_public_dict())

        with self.assertRaisesRegex(ValueError, "legacy alias does not match"):
            _eval_job_spec(
                job_id="eval-legacy-mismatch",
                task_id="unitree_push_recovery",
                candidate_id="candidate-001",
                checkpoint=CheckpointRef(
                    "artifact://round/best.pt",
                    _EVAL_CHECKSUM,
                    "eval",
                    sealed=True,
                ),
                public_seed_count=4,
                sealed_validation_seed_count=8,
                trusted_seed_count=9,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=4,
                    sealed_validation_seed_count=8,
                ),
            )

    def test_fixed_budget_check_blocks_budget_drift(self) -> None:
        spec = _train_job_spec(
            job_id="train-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
        )

        enforce_fixed_budget(spec, expected_budget_steps=128)
        with self.assertRaises(BudgetViolation):
            enforce_fixed_budget(spec, expected_budget_steps=256)

        for invalid in (True, 128.0, "128"):
            with self.subTest(expected_budget_steps=invalid):
                with self.assertRaisesRegex(
                    BudgetViolation,
                    "expected_budget_steps must be a positive integer",
                ):
                    enforce_fixed_budget(
                        spec,
                        expected_budget_steps=invalid,
                    )

        for invalid in (True, 128.0, "128"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "budget_steps must be a positive integer",
                ):
                    _train_job_spec(
                        job_id="train-invalid-budget",
                        task_id="unitree_push_recovery",
                        candidate_id="candidate-001",
                        policy_initialization=PolicyInitialization.random(),
                        budget_steps=invalid,
                    )

    def test_budget_accounting_requires_planned_env_steps_to_match_budget(self) -> None:
        accounting = build_budget_accounting(
            budget_steps=49_152_000,
            num_envs=4096,
            num_steps_per_env=24,
            max_iterations=500,
        )

        self.assertEqual(accounting["status"], "pass")
        self.assertEqual(accounting["planned_env_steps"], 49_152_000)
        self.assertEqual(accounting["config_derived_total_timesteps"], 49_152_000)

        with self.assertRaisesRegex(BudgetViolation, "planned env steps mismatch"):
            build_budget_accounting(
                budget_steps=64,
                num_envs=4096,
                num_steps_per_env=24,
                max_iterations=500,
            )

        for invalid in (True, 4096.0, "4096"):
            with self.subTest(num_envs=invalid):
                with self.assertRaisesRegex(
                    BudgetViolation,
                    "num_envs must be a positive integer",
                ):
                    build_budget_accounting(
                        budget_steps=49_152_000,
                        num_envs=invalid,
                        num_steps_per_env=24,
                        max_iterations=500,
                    )

    def test_artifact_layout_builds_session_round_train_and_eval_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunnerArtifactLayout(root=Path(tmp), session_id="session-001")

            self.assertEqual(layout.session_dir.name, "session-001")
            self.assertEqual(layout.round_dir("round-001").name, "round-001")
            self.assertEqual(layout.train_dir("round-001").name, "train")
            self.assertEqual(layout.eval_dir("round-001").name, "eval")

    def test_isolation_policy_reserves_dry_run_subprocess_and_rootfs_modes(self) -> None:
        self.assertEqual(IsolationPolicy().kind, "dry_run")
        self.assertTrue(IsolationPolicy(kind="subprocess", subprocess_boundary=True).subprocess_boundary)
        self.assertEqual(IsolationPolicy(kind="rootfs", rootfs="/runner-rootfs").rootfs, "/runner-rootfs")
        with self.assertRaises(ValueError):
            IsolationPolicy(kind="rootfs")

    def test_checkpoint_sealing_is_runner_owned(self) -> None:
        train_output = CheckpointRef(
            uri="artifact://round/train/raw.pt",
            checksum=_RAW_CHECKSUM,
            role="train_output",
        )

        sealed = seal_checkpoint(train_output, sealed_uri="artifact://round/checkpoints/best.pt")

        self.assertEqual(sealed.role, "eval")
        self.assertTrue(sealed.sealed)
        self.assertEqual(sealed.checksum, _RAW_CHECKSUM)

    def test_planned_checkpoint_is_not_a_byte_addressed_checkpoint(self) -> None:
        plan = CheckpointOutputPlan("artifact://round/checkpoints/best.pt")
        eval_spec = _eval_job_spec(
            job_id="eval-planned",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=None,
            checkpoint_output_plan=plan,
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )

        public = eval_spec.to_public_dict()

        self.assertNotIn("checkpoint", public)
        self.assertEqual(public["checkpoint_plan"], plan.to_public_dict())
        self.assertNotIn("checksum", public["checkpoint_plan"])
        with self.assertRaisesRegex(ValueError, "have not been bound"):
            eval_spec.execution_checkpoint()
        with self.assertRaisesRegex(ValueError, "have not been bound"):
            dry_run_eval_command(eval_spec)

    def test_checkpoint_refs_and_executable_eval_specs_fail_closed(self) -> None:
        for malformed in (
            "sha256:not-a-digest",
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    CheckpointRef(
                        uri="artifact://round/checkpoints/best.pt",
                        checksum=malformed,
                        role="eval",
                        sealed=True,
                    )

        with self.assertRaisesRegex(ValueError, "requires a sealed checkpoint"):
            _eval_job_spec(
                job_id="eval-unsealed",
                task_id="unitree_push_recovery",
                candidate_id="candidate-001",
                checkpoint=CheckpointRef(
                    "artifact://round/checkpoints/best.pt",
                    _EVAL_CHECKSUM,
                    "eval",
                ),
                public_seed_count=4,
                sealed_validation_seed_count=8,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=4,
                    sealed_validation_seed_count=8,
                ),
            )

    def test_trusted_eval_seal_rechecks_checkpoint_sealed_state(self) -> None:
        checkpoint = CheckpointRef(
            "artifact://round/checkpoints/best.pt",
            _EVAL_CHECKSUM,
            "eval",
            sealed=True,
        )
        eval_spec = _eval_job_spec(
            job_id="eval-seal-tamper",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=checkpoint,
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        object.__setattr__(checkpoint, "sealed", False)

        with self.assertRaisesRegex(ValueError, "sealed checkpoint"):
            build_trusted_eval_seal(
                eval_spec=eval_spec,
                task_invariants=TaskInvariantSpec({}),
                checkpoint_safety_report=self._checkpoint_safety_report(),
                candidate_isolation_status="pass",
            )

    def test_checkpoint_safety_report_records_byte_level_load_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint bytes")
            expected_hash = "sha256:" + hashlib.sha256(b"checkpoint bytes").hexdigest()

            report = build_checkpoint_safety_report(
                checkpoint,
                expected_checkpoint_hash=expected_hash,
                weights_only_loader=lambda path: {"loaded": path.name},
            )

        public_report = report.to_public_dict()
        self.assertEqual(report.status, "pass")
        self.assertTrue(report.file_exists)
        self.assertEqual(report.checkpoint_hash, expected_hash)
        self.assertEqual(report.expected_checkpoint_hash, expected_hash)
        self.assertEqual(report.byte_read_status, "pass")
        self.assertEqual(report.weights_only_load_status, "pass")
        self.assertIsNone(report.loader_error_class)
        self.assertEqual(validate_checkpoint_safety_report(public_report), public_report)
        text = json.dumps(public_report, sort_keys=True)
        self.assertNotIn(str(checkpoint), text)
        self.assertNotIn("checkpoint bytes", text)

    def test_checkpoint_safety_report_fails_on_expected_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint bytes")

            report = build_checkpoint_safety_report(
                checkpoint,
                expected_checkpoint_hash="sha256:" + "0" * 64,
                weights_only_loader=lambda path: {"loaded": True},
            )

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.reason, "checkpoint_hash_mismatch")
        self.assertEqual(report.weights_only_load_status, "pass")
        with self.assertRaisesRegex(ValueError, "checkpoint safety failed"):
            validate_checkpoint_safety_report(report.to_public_dict())

    def test_checkpoint_safety_report_validation_rejects_missing_or_failing_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_safety_report must be an object"):
            validate_checkpoint_safety_report(None)

        failing_report = {
            "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
            "status": "fail",
            "checkpoint_hash": None,
            "expected_checkpoint_hash": "sha256:" + "1" * 64,
            "file_exists": False,
            "byte_read_status": "fail",
            "weights_only_load_status": "fail",
            "checkpoint_hash_match_status": "not_checked",
            "loader_error_class": "FileNotFoundError",
            "loader_error_summary": "FileNotFoundError",
            "reason": "checkpoint_missing",
        }

        with self.assertRaisesRegex(ValueError, "checkpoint safety failed"):
            validate_checkpoint_safety_report(failing_report)

    def test_checkpoint_safety_report_validation_requires_expected_eval_checkpoint_hash(self) -> None:
        report = dict(self._checkpoint_safety_report())

        with self.assertRaisesRegex(ValueError, "report hash does not match"):
            validate_checkpoint_safety_report(
                report,
                expected_checkpoint_hash=_OTHER_CHECKSUM,
            )

        report = dict(self._checkpoint_safety_report())
        report["expected_checkpoint_hash"] = _OTHER_CHECKSUM
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint hash does not match its expected hash",
        ):
            validate_checkpoint_safety_report(
                report,
                expected_checkpoint_hash=_EVAL_CHECKSUM,
            )

    def test_checkpoint_safety_report_rejects_malformed_digest(self) -> None:
        report = dict(self._checkpoint_safety_report())
        report["checkpoint_hash"] = "sha256:not-a-digest"
        report["expected_checkpoint_hash"] = "sha256:not-a-digest"

        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            validate_checkpoint_safety_report(
                report,
                expected_checkpoint_hash="sha256:not-a-digest",
            )

        report = dict(self._checkpoint_safety_report())
        report["expected_checkpoint_hash"] = None
        report["checkpoint_hash_match_status"] = "not_checked"

        with self.assertRaisesRegex(ValueError, "expected hash does not match"):
            validate_checkpoint_safety_report(
                report,
                expected_checkpoint_hash=_EVAL_CHECKSUM,
            )

    def test_task_invariant_violation_fails_before_training(self) -> None:
        expected = TaskInvariantSpec({"fall_threshold": 0.30, "episode_length": 512})
        changed = TaskInvariantSpec({"fall_threshold": 0.20, "episode_length": 512})

        with self.assertRaises(ValueError):
            enforce_task_invariants(expected=expected, actual=changed)

    def test_trusted_eval_seal_records_hashes_without_candidate_code(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        invariants = TaskInvariantSpec({"fall_threshold": 0.30})

        seal = build_trusted_eval_seal(
            eval_spec=eval_spec,
            task_invariants=invariants,
            checkpoint_safety_report=self._checkpoint_safety_report(),
            candidate_isolation_status="pass",
        )

        self.assertEqual(len(seal.eval_config_hash), 64)
        self.assertEqual(seal.checkpoint_hash, _EVAL_CHECKSUM)
        self.assertEqual(len(seal.task_invariant_hash), 64)
        self.assertFalse(seal.candidate_code_loaded)
        self.assertEqual(seal.checkpoint_safety_status, "pass")
        self.assertEqual(seal.candidate_isolation_status, "pass")
        self.assertEqual(len(seal.checkpoint_safety_report_hash), 64)
        self.assertEqual(len(seal.seal_hash), 64)
        self.assertEqual(eval_spec.to_public_dict()["sealed_validation_seed_count"], 8)

    def test_trusted_eval_seal_requires_checkpoint_safety_report(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                sealed_validation_seed_count=8,
            ),
        )
        invariants = TaskInvariantSpec({"fall_threshold": 0.30})

        with self.assertRaisesRegex(ValueError, "checkpoint_safety_report must be an object"):
            build_trusted_eval_seal(
                eval_spec=eval_spec,
                task_invariants=invariants,
                checkpoint_safety_report=None,
                candidate_isolation_status="pass",
            )
        with self.assertRaisesRegex(ValueError, "checkpoint safety failed"):
            build_trusted_eval_seal(
                eval_spec=eval_spec,
                task_invariants=invariants,
                checkpoint_safety_report=self._checkpoint_safety_report(
                    status="fail",
                    reason="checkpoint_weights_only_load_failed",
                ),
                candidate_isolation_status="pass",
            )
        with self.assertRaisesRegex(ValueError, "candidate isolation failed"):
            build_trusted_eval_seal(
                eval_spec=eval_spec,
                task_invariants=invariants,
                checkpoint_safety_report=self._checkpoint_safety_report(),
                candidate_isolation_status="fail",
            )
        drifted_report = dict(self._checkpoint_safety_report())
        drifted_report["expected_checkpoint_hash"] = None
        drifted_report["checkpoint_hash_match_status"] = "not_checked"
        with self.assertRaisesRegex(ValueError, "expected hash does not match"):
            build_trusted_eval_seal(
                eval_spec=eval_spec,
                task_invariants=invariants,
                checkpoint_safety_report=drifted_report,
                candidate_isolation_status="pass",
            )

    def test_trusted_eval_seal_hash_tracks_public_accept_and_eval_env_protocol(self) -> None:
        invariants = TaskInvariantSpec({"fall_threshold": 0.30})
        base_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            public_accept_seed_count=3,
            eval_num_envs=64,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=3,
                sealed_validation_seed_count=8,
                eval_num_envs=64,
            ),
        )
        changed_accept_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            public_accept_seed_count=4,
            eval_num_envs=64,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=4,
                sealed_validation_seed_count=8,
                eval_num_envs=64,
            ),
        )
        changed_env_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            public_accept_seed_count=3,
            eval_num_envs=128,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=3,
                sealed_validation_seed_count=8,
                eval_num_envs=128,
            ),
        )
        changed_protocol_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            public_accept_seed_count=3,
            eval_num_envs=64,
            eval_protocol_version="fixed_public_accept_sealed_validation_v2",
            evaluation_binding=make_evaluation_binding(
                eval_protocol_version="fixed_public_accept_sealed_validation_v2",
                public_seed_count=4,
                public_accept_seed_count=3,
                sealed_validation_seed_count=8,
                eval_num_envs=64,
            ),
        )
        changed_device_spec = replace(base_spec, execution_device="cuda:1")

        base_report = self._checkpoint_safety_report()
        base_hash = build_trusted_eval_seal(
            eval_spec=base_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).eval_config_hash
        changed_accept_hash = build_trusted_eval_seal(
            eval_spec=changed_accept_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).eval_config_hash
        changed_env_hash = build_trusted_eval_seal(
            eval_spec=changed_env_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).eval_config_hash
        changed_protocol_hash = build_trusted_eval_seal(
            eval_spec=changed_protocol_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).eval_config_hash
        changed_device_hash = build_trusted_eval_seal(
            eval_spec=changed_device_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).eval_config_hash
        base_seal_hash = build_trusted_eval_seal(
            eval_spec=base_spec,
            task_invariants=invariants,
            checkpoint_safety_report=base_report,
            candidate_isolation_status="pass",
        ).seal_hash

        self.assertNotEqual(base_hash, changed_accept_hash)
        self.assertNotEqual(base_hash, changed_env_hash)
        self.assertNotEqual(base_hash, changed_protocol_hash)
        self.assertNotEqual(base_hash, changed_device_hash)
        self.assertEqual(
            build_trusted_eval_seal(
                eval_spec=base_spec,
                task_invariants=invariants,
                checkpoint_safety_report=base_report,
                candidate_isolation_status="pass",
            ).to_public_dict()["execution_device"],
            "cuda:0",
        )
        changed_reason_report = dict(base_report)
        changed_reason_report["reason"] = "checkpoint_safe_to_load_after_retry"
        changed_report_hash = build_trusted_eval_seal(
            eval_spec=base_spec,
            task_invariants=invariants,
            checkpoint_safety_report=changed_reason_report,
            candidate_isolation_status="pass",
        ).seal_hash
        self.assertNotEqual(base_seal_hash, changed_report_hash)

    def test_eval_job_protocol_hash_matches_public_protocol_projection(self) -> None:
        eval_spec = _eval_job_spec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef(
                "artifact://round/best.pt",
                _EVAL_CHECKSUM,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            public_accept_seed_count=2,
            eval_num_envs=64,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=2,
                sealed_validation_seed_count=8,
                eval_num_envs=64,
            ),
        )
        expected = EvaluationProtocol(
            eval_protocol_version=eval_spec.eval_protocol_version,
            public_seed_count=4,
            public_accept_seed_count=2,
            sealed_validation_seed_count=8,
            eval_num_envs=64,
        )

        self.assertEqual(eval_spec.to_public_dict()["eval_protocol_hash"], expected.protocol_hash)


if __name__ == "__main__":
    unittest.main()

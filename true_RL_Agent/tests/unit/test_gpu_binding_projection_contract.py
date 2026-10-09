from __future__ import annotations

import tempfile
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.agent_core.generation_contract import GenerationLaneAssignment
from true_rl_agent.agent_core.session_genesis import (
    _job_spec_to_private_dict,
    _reference_eval_artifact_payload,
)
from true_rl_agent.agent_core.trial_receipts import _train_result_to_private_mapping
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.process_handle import (
    OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
)
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.runner.gpu_process_binding import (
    GpuRuntimeBinding,
    validate_gpu_runtime_summary_evidence,
)


_PUBLIC_BINDING_FIELDS = {
    "process_execution_device",
    "resource_binding_hash",
}
_PRIVATE_BINDING_FIELDS = _PUBLIC_BINDING_FIELDS | {"host_execution_device"}


class GpuBindingProjectionContractTests(unittest.TestCase):
    def test_persisted_binding_projections_have_explicit_public_private_fields(
        self,
    ) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        checkpoint_path = Path(temporary_directory.name) / "best.pt"
        checkpoint_path.write_bytes(b"gpu projection checkpoint")
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="9" * 64,
        )
        candidate = make_verified_candidate_artifact("gpu-projection-contract")
        subject = ExperimentSubject.candidate(candidate.candidate_id)
        initialization = PolicyInitialization.random()
        training_binding = make_training_binding(initialization)
        train_spec = TrainJobSpec(
            job_id="gpu-projection-train",
            task_id="gpu_projection_task",
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            policy_initialization=initialization,
            training_binding=training_binding,
            budget_steps=training_binding.budget_steps,
            execution_device="cuda:0",
            candidate_artifact=candidate,
            gpu_process_binding=binding,
        )
        checkpoint_metadata = {
            "training_subject": subject.to_public_dict(),
            "subject_manifest_hash": candidate.subject_manifest_hash,
            "program_source_checksum": candidate.program_source_checksum,
            "training_binding_hash": training_binding.binding_hash,
            "capability_profile_hash": training_binding.capability_profile_hash,
            "execution_device": "cuda:0",
            "gpu_process_binding": binding.to_dict(),
            "local_path": str(checkpoint_path),
        }
        checkpoint = CheckpointRef(
            uri="artifact://gpu-projection/best.pt",
            checksum="sha256:" + "8" * 64,
            role="eval",
            sealed=True,
            metadata=checkpoint_metadata,
        )
        train_result = TrainJobResult(
            job_id=train_spec.job_id,
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            training_binding_hash=training_binding.binding_hash,
            capability_profile_hash=training_binding.capability_profile_hash,
            execution_device="cuda:0",
            checkpoint=checkpoint,
            gpu_process_binding=binding,
            summary={"gpu_process_binding": binding.to_dict()},
        )
        evaluation_binding = make_evaluation_binding()
        eval_spec = EvalJobSpec(
            job_id="gpu-projection-eval",
            task_id="gpu_projection_task",
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            training_binding_hash=training_binding.binding_hash,
            capability_profile_hash=training_binding.capability_profile_hash,
            checkpoint=checkpoint,
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=evaluation_binding,
            execution_device="cuda:0",
            gpu_process_binding=binding,
        )
        baseline_spec = BaselineEvalJobSpec(
            job_id="gpu-projection-baseline",
            task_id="gpu_projection_task",
            subject=ExperimentSubject.baseline("gpu-projection-baseline"),
            subject_manifest_hash="7" * 64,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=CheckpointRef(
                uri="baseline://gpu-projection/policy.pt",
                checksum="sha256:" + "6" * 64,
                role="warm_start",
                sealed=True,
            ),
            public_seed_count=3,
            sealed_validation_seed_count=7,
            evaluation_binding=evaluation_binding,
            execution_device="cuda:0",
            gpu_process_binding=binding,
        )
        lane = GenerationLaneAssignment(
            lane_id="lane-a",
            execution_device="cuda:0",
            resource_binding_hash=binding.resource_binding_hash,
            lane_intent_hash="5" * 64,
            gpu_process_binding=binding,
        )
        phase_result = EvaluationPhaseResult(
            job_id="gpu-projection-public-dev",
            task_id=eval_spec.task_id,
            phase=EvaluationPhase.PUBLIC_DEV,
            evaluation_plan_hash=canonical_json_hash(eval_spec.to_public_dict()),
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            training_binding_hash=training_binding.binding_hash,
            capability_profile_hash=training_binding.capability_profile_hash,
            checkpoint_hash=checkpoint.checksum,
            eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=evaluation_binding.binding_hash,
            execution_device="cuda:0",
            resource_binding_hash=binding.resource_binding_hash,
            task_evidence={},
            execution_cost={},
            environment_source_preflight={},
            backend_execution_environment_verification={},
            checkpoint_safety_report={},
            gpu_process_binding=binding.to_dict(),
        )
        reference_private_artifact = _reference_eval_artifact_payload(
            session_id="gpu-projection-session",
            backend_id="gpu-projection-backend",
            reference=SimpleNamespace(
                manifest=SimpleNamespace(to_public_dict=lambda: {}),
                source_text="reference source",
            ),
            reference_artifact=SimpleNamespace(
                to_public_dict=lambda: {},
                path=str(Path(temporary_directory.name) / "reference"),
            ),
            train_spec=train_spec,
            train_result=train_result,
            eval_spec=eval_spec,
            public_result=SimpleNamespace(to_public_dict=lambda: {}),
            trusted_eval_seal=SimpleNamespace(to_public_dict=lambda: {}),
            qualification=SimpleNamespace(gate={}, private_evidence={}),
        )
        process_identity = {
            "schema_version": OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
            "process_id": 1234,
            "process_group_id": 1234,
        }
        runtime = GpuRuntimeBinding(
            gpu_process_binding=binding,
            physical_gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            renderer_active_gpu_ordinal=1,
        )
        with patch(
            "true_rl_agent.runner.gpu_process_binding."
            "current_owned_process_identity",
            return_value=process_identity,
        ):
            runner_summary = runtime.summary_fields(
                compute_context_observation=None,
            )

        public_projections = {
            "binding": binding.to_public_dict(),
            "train_job": train_spec.to_public_dict()["gpu_process_binding"],
            "eval_job": eval_spec.to_public_dict()["gpu_process_binding"],
            "baseline_job": baseline_spec.to_public_dict()["gpu_process_binding"],
            "train_result": train_result.to_public_dict()["gpu_process_binding"],
            "generation_manifest_lane": lane.to_public_dict()[
                "gpu_process_binding"
            ],
        }
        private_projections = {
            "binding": binding.to_dict(),
            "reference_train_job": _job_spec_to_private_dict(train_spec)[
                "gpu_process_binding"
            ],
            "reference_eval_job": _job_spec_to_private_dict(eval_spec)[
                "gpu_process_binding"
            ],
            "reference_train_result": reference_private_artifact["train_result"][
                "gpu_process_binding"
            ],
            "baseline_eval_job": _job_spec_to_private_dict(baseline_spec)[
                "gpu_process_binding"
            ],
            "generation_manifest_receipt_lane": lane.to_dict()[
                "gpu_process_binding"
            ],
            "trial_train_receipt": _train_result_to_private_mapping(train_result)[
                "gpu_process_binding"
            ],
            "checkpoint_metadata": checkpoint.metadata["gpu_process_binding"],
            "phase_result_receipt": phase_result.to_private_dict()[
                "gpu_process_binding"
            ],
            "stair_runner_summary": runner_summary["gpu_process_binding"],
        }

        for label, projection in public_projections.items():
            with self.subTest(visibility="public", artifact=label):
                self.assertEqual(set(projection), _PUBLIC_BINDING_FIELDS)
                self.assertEqual(projection, binding.to_public_dict())
                with self.assertRaisesRegex(ValueError, "fields mismatch"):
                    GpuProcessBinding.from_mapping(projection)
        for label, projection in private_projections.items():
            with self.subTest(visibility="private", artifact=label):
                self.assertEqual(set(projection), _PRIVATE_BINDING_FIELDS)
                self.assertEqual(projection, binding.to_dict())
                self.assertEqual(GpuProcessBinding.from_mapping(projection), binding)

        strict_summary_readers = {
            "subprocess_runner": (
                SubprocessRunner._validated_gpu_process_binding_from_summary(
                    runner_summary,
                    expected_binding=binding,
                )
            ),
            "stair_runtime": GpuProcessBinding.from_mapping(
                validate_gpu_runtime_summary_evidence(
                    runner_summary,
                    expected_binding=binding,
                    expected_physical_gpu_uuid=runtime.physical_gpu_uuid,
                    require_compute_context_observation=False,
                )["gpu_process_binding"]
            ),
        }
        for label, loaded_binding in strict_summary_readers.items():
            with self.subTest(strict_reader=label):
                self.assertEqual(loaded_binding, binding)

        # Historical receipt bytes keep their original wire schema. Decoding
        # only changes the in-memory container key, never the hashed evidence.
        historical = copy.deepcopy(runner_summary)
        historical_evidence = historical.pop("gpu_runtime_evidence")
        historical_evidence["schema_version"] = (
            "true_rl_agent.task.stair_climbing.gpu_runtime_evidence.v2"
        )
        historical_evidence["external_compute_context_observation"] = {
            "schema_version": "true_rl_agent.task.stair_climbing.gpu_compute_context_observation.v1",
            "source": "nvidia-smi-query-compute-apps",
            "monotonic_ns": 42,
            "process_id": 1234,
            "rows": [{"gpu_uuid": runtime.physical_gpu_uuid, "used_gpu_memory_mib": 10}],
        }
        historical["stair_gpu_runtime_evidence"] = historical_evidence
        original_hash = canonical_json_hash(historical)
        decoded = validate_gpu_runtime_summary_evidence(
            historical, expected_binding=binding,
            expected_physical_gpu_uuid=runtime.physical_gpu_uuid,
            require_compute_context_observation=True,
        )
        self.assertEqual(decoded["gpu_runtime_evidence"], historical_evidence)
        self.assertEqual(canonical_json_hash(historical), original_hash)


if __name__ == "__main__":
    unittest.main()

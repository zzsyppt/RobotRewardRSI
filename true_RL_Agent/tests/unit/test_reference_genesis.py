from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    make_test_candidate_package,
    make_verified_candidate_artifact,
)
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FAKE_CAPABILITY_PROFILE,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationSessionContract,
    publish_generation_session_contract,
)
from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_V5,
    NON_PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.live_loop import run_live_loop
from true_rl_agent.agent_core.session_genesis import (
    GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
    IMPORTED_REFERENCE_SESSION_GENESIS_SCHEMA_VERSION,
    LEGACY_SESSION_GENESIS_SCHEMA_VERSION,
    PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
    SESSION_GENESIS_CREATION_LEASE_FILENAME,
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
    SessionGenesisCreationInProgressError,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.agent_core.source_request_limits import CURRENT_BEST_SOURCE_ID
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseReceiptBinding,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import canonical_json_bytes, canonical_json_text
from true_rl_agent.memory.context_builder import PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompting.source_catalog import build_prompt_source_context
from true_rl_agent.reference_artifact import materialize_reference_program_artifact
from true_rl_agent.runner.checkpoint import file_sha256, seal_checkpoint
from true_rl_agent.runner.checkpoint_preflight import (
    validate_checkpoint_process_input,
)
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.specs import (
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.reference_qualification import (
    REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
    ReferenceQualificationResult,
)
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)
from true_rl_agent.training_programs import resolve_reference_program_artifact


class _ReferenceTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="fake_push_recovery",
            name="Fake Reference Task",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={"random": "reference_program"},
            candidate_capability_profile=FAKE_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec(
                {
                    "episode_length": 512,
                    "reference_fixture": "fixed_v1",
                }
            ),
            acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
        )

    def qualify_reference_result(self, *, public_result, train_summary):
        self.project_public_result(
            public_result=public_result,
            train_summary=train_summary,
        )
        evidence = {
            "schema_version": "fake.reference_qualification.v1",
            "result": "fixture-pass",
        }
        evidence["evidence_hash"] = canonical_json_hash(evidence)
        gate = {
            "schema_version": REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
            "status": "pass",
            "protocol_hash": self.task_spec.resolved_task_protocol_hash(),
            "evidence_hash": evidence["evidence_hash"],
        }
        metrics = dict(public_result.metrics)
        acceptance = dict(metrics["acceptance"])
        acceptance["required_pass_gates"] = [
            *acceptance["required_pass_gates"],
            "reference_capability_gate",
        ]
        metrics["acceptance"] = acceptance
        metrics["reference_capability_gate"] = gate
        return ReferenceQualificationResult(
            public_result=PublicEvalResult(
                status=public_result.status,
                public_score=public_result.public_score,
                metrics=metrics,
            ),
            gate=gate,
            private_evidence=evidence,
        )


class _FailingReferenceTaskPlugin(_ReferenceTaskPlugin):
    def qualify_reference_result(self, *, public_result, train_summary):
        passed = super().qualify_reference_result(
            public_result=public_result,
            train_summary=train_summary,
        )
        evidence = {
            "schema_version": "fake.reference_qualification.v1",
            "result": "fixture-fail",
        }
        evidence["evidence_hash"] = canonical_json_hash(evidence)
        gate = {
            "schema_version": REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
            "status": "fail",
            "protocol_hash": self.task_spec.resolved_task_protocol_hash(),
            "evidence_hash": evidence["evidence_hash"],
        }
        metrics = dict(passed.public_result.metrics)
        metrics["reference_capability_gate"] = gate
        return ReferenceQualificationResult(
            public_result=PublicEvalResult(
                status=passed.public_result.status,
                public_score=passed.public_result.public_score,
                metrics=metrics,
            ),
            gate=gate,
            private_evidence=evidence,
        )


class _WrongProtocolReferenceTaskPlugin(_ReferenceTaskPlugin):
    def qualify_reference_result(self, *, public_result, train_summary):
        qualified = super().qualify_reference_result(
            public_result=public_result,
            train_summary=train_summary,
        )
        gate = {**dict(qualified.gate), "protocol_hash": "b" * 64}
        metrics = dict(qualified.public_result.metrics)
        metrics["reference_capability_gate"] = gate
        return ReferenceQualificationResult(
            public_result=PublicEvalResult(
                status=qualified.public_result.status,
                public_score=qualified.public_result.public_score,
                metrics=metrics,
            ),
            gate=gate,
            private_evidence=qualified.private_evidence,
        )


class _ReferenceBackend(FakeBackendAdapter):
    def materialize_reference_jobs(
        self,
        *,
        session_id,
        task_spec,
        reference,
        artifact_root,
        policy_initialization,
        evaluation_binding=None,
        gpu_process_binding=None,
        train_attempt_scope=None,
        eval_attempt_scope=None,
    ):
        binding = evaluation_binding or self.evaluation_binding_for(task_spec)
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=binding,
        )
        artifact = materialize_reference_program_artifact(
            reference,
            artifact_root=artifact_root,
        )
        training_binding, agent_json = _reference_training_binding()
        subject = ExperimentSubject.reference(reference.manifest.reference_id)
        run_root = Path(artifact_root).parent / "runner" / session_id
        train_summary = run_root / "train" / "summary.json"
        eval_summary = run_root / "eval" / "summary.json"
        checkpoint_output = run_root / "checkpoints" / "best.pt"
        checkpoint_plan = CheckpointOutputPlan(
            uri=f"artifact://{session_id}/checkpoints/best.pt"
        )
        train_argv = _reference_train_argv(
            session_id=session_id,
            task_id=task_spec.task_id,
            subject=subject,
            manifest_hash=str(reference.manifest.manifest_hash),
            source_checksum=reference.manifest.source_checksum,
            artifact_path=artifact.path,
            artifact_hash=artifact.artifact_hash,
            binding=training_binding,
            agent_json=agent_json,
            summary_path=train_summary,
            checkpoint_output=checkpoint_output,
            gpu_process_binding=gpu_process_binding,
        )
        train_spec = TrainJobSpec(
            job_id=f"{session_id}-train",
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=str(reference.manifest.manifest_hash),
            program_source_checksum=reference.manifest.source_checksum,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            budget_steps=training_binding.budget_steps,
            execution_device="cuda:0",
            gpu_process_binding=gpu_process_binding,
            reference_artifact=artifact,
            attempt_scope=train_attempt_scope,
            metadata={
                "cli_argv": train_argv,
                "cli_command": " ".join(train_argv),
                "summary_path": str(train_summary),
                "checkpoint_output_path": str(checkpoint_output),
                "sealed_checkpoint_uri": checkpoint_plan.uri,
                "fake_public_score": 0.25,
            },
        )
        eval_argv = _reference_eval_argv(
            session_id=session_id,
            task_id=task_spec.task_id,
            subject=subject,
            manifest_hash=str(reference.manifest.manifest_hash),
            source_checksum=reference.manifest.source_checksum,
            training_binding=training_binding,
            capability_profile_hash=reference.manifest.capability_profile_hash,
            evaluation_binding=binding,
            checkpoint_uri=checkpoint_plan.uri,
            summary_path=eval_summary,
            gpu_process_binding=gpu_process_binding,
        )
        eval_spec = EvalJobSpec(
            job_id=f"{session_id}-eval",
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=str(reference.manifest.manifest_hash),
            program_source_checksum=reference.manifest.source_checksum,
            training_binding_hash=str(training_binding.binding_hash),
            capability_profile_hash=reference.manifest.capability_profile_hash,
            checkpoint=None,
            checkpoint_output_plan=checkpoint_plan,
            public_seed_count=4,
            public_accept_seed_count=3,
            sealed_validation_seed_count=8,
            eval_num_envs=64,
            evaluation_binding=binding,
            execution_device="cuda:0",
            gpu_process_binding=gpu_process_binding,
            metadata={
                "cli_argv": eval_argv,
                "cli_command": " ".join(eval_argv),
                "summary_path": str(eval_summary),
                "fake_public_score": 0.25,
            },
            attempt_scope=eval_attempt_scope,
        )
        return BackendJobBundle(
            train_spec=train_spec,
            eval_spec=eval_spec,
            task_invariants=task_spec.invariants,
        )

    def materialize_eval_phase(
        self,
        *,
        eval_plan,
        phase,
        attempt_scope=None,
    ):
        phase = EvaluationPhase.parse(phase)
        return EvalPhaseJobSpec(
            job_id=f"{eval_plan.job_id}-{phase.value}",
            evaluation_plan=eval_plan,
            phase=phase,
            metadata={"phase": phase.value},
            attempt_scope=attempt_scope,
        )


class _CountingReferenceBackend(_ReferenceBackend):
    def __init__(self) -> None:
        self.materialize_reference_calls = 0
        self.reference_jobs = None

    def materialize_reference_jobs(self, **kwargs):
        self.materialize_reference_calls += 1
        self.reference_jobs = super().materialize_reference_jobs(**kwargs)
        return self.reference_jobs


class _ReferenceRunner(FakeRunner):
    def __init__(self) -> None:
        self.train_calls = 0
        self.eval_calls = 0

    def run_train(self, train_spec):
        self.train_calls += 1
        checkpoint_path = Path(train_spec.metadata["checkpoint_output_path"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(
            b"fake-reference-checkpoint:" + train_spec.subject.subject_id.encode()
        )
        checkpoint_hash = file_sha256(checkpoint_path)
        assert checkpoint_hash is not None
        checkpoint_metadata = {
            "local_path": str(checkpoint_path.resolve()),
            "training_subject": train_spec.subject.to_public_dict(),
            "subject_manifest_hash": train_spec.subject_manifest_hash,
            "program_source_checksum": train_spec.program_source_checksum,
            "training_binding_hash": train_spec.training_binding.binding_hash,
            "capability_profile_hash": (
                train_spec.training_binding.capability_profile_hash
            ),
            "execution_device": train_spec.execution_device,
        }
        if train_spec.gpu_process_binding is not None:
            checkpoint_metadata["gpu_process_binding"] = (
                train_spec.gpu_process_binding.to_dict()
            )
        raw = CheckpointRef(
            uri=str(checkpoint_path),
            checksum=checkpoint_hash,
            role="train_output",
            metadata=checkpoint_metadata,
        )
        checkpoint = seal_checkpoint(
            raw,
            sealed_uri=str(train_spec.metadata["sealed_checkpoint_uri"]),
        )
        from true_rl_agent.runner.base import TrainJobResult

        return TrainJobResult(
            job_id=train_spec.job_id,
            subject=train_spec.subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=str(train_spec.training_binding.binding_hash),
            capability_profile_hash=(
                train_spec.training_binding.capability_profile_hash
            ),
            execution_device=train_spec.execution_device,
            gpu_process_binding=train_spec.gpu_process_binding,
            checkpoint=checkpoint,
            summary={
                "formal_training_executed": True,
                "training": {
                    "initial_policy_state_hash": canonical_json_hash(
                        {"seed": 42, "policy": "fake-reference"}
                    ),
                    "budget_accounting": {"status": "pass"},
                },
                "runtime_activation_summary": {"status": "pass"},
            },
        )

    def run_eval(self, eval_spec):
        self.eval_calls += 1
        return super().run_eval(eval_spec)


class _PhasedReferenceRunner(_ReferenceRunner):
    def __init__(self) -> None:
        super().__init__()
        self.phase_calls: list[EvaluationPhase] = []
        self.projection_calls = 0

    def run_eval_phase(self, eval_spec):
        self.phase_calls.append(eval_spec.phase)
        return EvaluationPhaseResult(
            job_id=eval_spec.job_id,
            task_id=eval_spec.task_id,
            phase=eval_spec.phase,
            evaluation_plan_hash=eval_spec.evaluation_plan_hash,
            subject=eval_spec.subject,
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            checkpoint_hash=eval_spec.execution_checkpoint().checksum,
            eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            execution_device=eval_spec.execution_device,
            resource_binding_hash=(
                eval_spec.attempt_scope.resource_binding_hash
                if eval_spec.attempt_scope is not None
                else None
            ),
            task_evidence={"phase": eval_spec.phase.value},
            execution_cost={"seed_units": 1},
            environment_source_preflight={"status": "pass"},
            backend_execution_environment_verification={"status": "pass"},
            checkpoint_safety_report={"status": "pass"},
            gpu_process_binding=(
                eval_spec.gpu_process_binding.to_dict()
                if eval_spec.gpu_process_binding is not None
                else None
            ),
        )

    def project_assembled_eval_summary(self, eval_spec, assembly):
        self.projection_calls += 1
        if not isinstance(assembly, EvaluationAssembly):
            raise TypeError("test reference assembly is invalid")
        return FakeRunner.run_eval(self, eval_spec)


class _ReferencePhasedAdapter:
    def __init__(self, gate_status=None):
        self.gate_status = gate_status

    def project_reference_public_dev_gate(self, *, phase_result):
        return None if self.gate_status is None else {"status": self.gate_status}

    def assemble_reference_full_summary(
        self,
        *,
        eval_plan,
        phase_results,
        phase_receipt_refs,
    ):
        by_phase = {result.phase: result for result in phase_results}
        bindings = []
        for phase in EvaluationPhase:
            result = by_phase[phase]
            ref = phase_receipt_refs[phase]
            phase_spec = EvalPhaseJobSpec(
                job_id=result.job_id,
                evaluation_plan=eval_plan,
                phase=phase,
            )
            bindings.append(
                EvaluationPhaseReceiptBinding(
                    phase=phase,
                    job_id=result.job_id,
                    path=ref["path"],
                    checksum=ref["checksum"],
                    size_bytes=ref["size_bytes"],
                    ref_schema_version=ref["schema_version"],
                    job_spec_hash=canonical_json_hash(
                        phase_spec.to_public_dict()
                    ),
                    phase_result_hash=canonical_json_hash(
                        result.to_private_dict()
                    ),
                    resource_binding_hash=result.resource_binding_hash,
                )
            )
        reference_id = eval_plan.subject.subject_id
        return EvaluationAssembly(
            evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
            selected_trial_id=reference_id,
            selection={
                "schema_version": "fake.reference_assembly.v1",
                "selected_trial_id": reference_id,
                "acceptance_authority": False,
            },
            phase_receipts=tuple(bindings),
            summary={"status": "pass"},
        )


class _LiveReferenceBackend(_ReferenceBackend):
    """Keep candidate jobs on the exact binding used by the fake R0."""

    def materialize_jobs(self, **kwargs):
        jobs = super().materialize_jobs(**kwargs)
        training_binding, _ = _reference_training_binding()
        return replace(
            jobs,
            train_spec=replace(
                jobs.train_spec,
                training_binding=training_binding,
                budget_steps=training_binding.budget_steps,
            ),
            eval_spec=replace(
                jobs.eval_spec,
                training_binding_hash=training_binding.binding_hash,
            ),
        )


class _ReferencePromptAdapter:
    task_id = "fake_push_recovery"

    def build_prompt_sections(self, *, public_records, public_context):
        _ = public_records
        evidence = public_context.get("initial_anchor_public_evidence")
        if not isinstance(evidence, dict):
            return {
                "public_feedback_summary": {
                    "schema_version": "fake.reference_prompt_feedback.v1",
                    "status": "not_available",
                }
            }
        result = evidence.get("public_result")
        public_score = result.get("public_score") if isinstance(result, dict) else None
        return {
            "public_feedback_summary": {
                "schema_version": "fake.reference_prompt_feedback.v1",
                "status": "available",
                "anchor_kind": evidence.get("anchor_kind"),
                "public_result_hash": evidence.get("public_result_hash"),
                "public_score": public_score,
            }
        }

    def prompt_public_record_scrub_policy(self):
        return {"blocked_key_parts": (), "blocked_text_parts": ()}


class _LiveReferenceTaskPlugin(_ReferenceTaskPlugin):
    def candidate_surface_policy(self):
        return FakeTaskPlugin().candidate_surface_policy()

    def prompt_adapter(self):
        return _ReferencePromptAdapter()


class _LiveReferenceRunner(_ReferenceRunner):
    def run_train(self, train_spec):
        if train_spec.subject.kind == "reference":
            return super().run_train(train_spec)
        return FakeRunner.run_train(self, train_spec)


class _CapturingCandidateProvider:
    def __init__(self, candidates) -> None:
        self._candidates = list(candidates)
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload):
        self.payloads.append(prompt_payload)
        if not self._candidates:
            raise AssertionError("candidate provider was called after replay resume")
        return self._candidates.pop(0)


class _BlockingReferenceRunner(_ReferenceRunner):
    def __init__(self) -> None:
        super().__init__()
        self.train_started = Event()
        self.allow_train = Event()

    def run_train(self, train_spec):
        self.train_started.set()
        if not self.allow_train.wait(timeout=5):
            raise TimeoutError("test reference training was not released")
        return super().run_train(train_spec)


def _checksum(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reference_training_binding() -> tuple[TrainingBinding, str]:
    policy_hash = canonical_json_hash({"policy": "fake-reference-v1"})
    optimizer_hash = canonical_json_hash({"optimizer": "fake-reference-v1"})
    agent_payload = {
        "schema_version": "fake.reference_agent_configuration.v1",
        "policy_architecture_hash": policy_hash,
        "optimizer_training_profile_hash": optimizer_hash,
        "hidden_sizes": [32, 32],
    }
    agent_hash = canonical_json_hash(agent_payload)
    agent_json = canonical_json_text(
        {**agent_payload, "agent_configuration_hash": agent_hash}
    )
    binding = TrainingBinding(
        initialization_mode="random",
        policy_architecture_hash=policy_hash,
        agent_configuration_hash=agent_hash,
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(42),
        optimizer_training_profile_hash=optimizer_hash,
        budget_steps=128,
        private_training_seed_resolver_hash=fixed_training_seed_resolver_hash(42),
        capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
    )
    return binding, agent_json


def _gpu_process_binding_argv(
    binding: GpuProcessBinding | None,
) -> list[str]:
    if binding is None:
        return []
    return [
        "--gpu-host-execution-device",
        binding.host_execution_device,
        "--gpu-process-execution-device",
        binding.process_execution_device,
        "--gpu-resource-binding-hash",
        binding.resource_binding_hash,
    ]


def _reference_train_argv(
    *,
    session_id: str,
    task_id: str,
    subject: ExperimentSubject,
    manifest_hash: str,
    source_checksum: str,
    artifact_path: str,
    artifact_hash: str,
    binding: TrainingBinding,
    agent_json: str,
    summary_path: Path,
    checkpoint_output: Path,
    gpu_process_binding: GpuProcessBinding | None = None,
) -> list[str]:
    return [
        "fake-python",
        "fake-train.py",
        "--job-id",
        f"{session_id}-train",
        "--round",
        session_id,
        "--task-id",
        task_id,
        "--subject-kind",
        subject.kind,
        "--subject-id",
        subject.subject_id,
        "--subject-manifest-hash",
        manifest_hash,
        "--program-source-checksum",
        source_checksum,
        "--reference-artifact",
        artifact_path,
        "--reference-artifact-hash",
        artifact_hash,
        "--budget-steps",
        str(binding.budget_steps),
        "--init",
        "random",
        "--training-binding-hash",
        str(binding.binding_hash),
        "--policy-architecture-hash",
        binding.policy_architecture_hash,
        "--optimizer-training-profile-hash",
        binding.optimizer_training_profile_hash,
        "--agent-configuration-hash",
        binding.agent_configuration_hash,
        "--agent-configuration-json",
        agent_json,
        "--initialization-seed-schedule-hash",
        binding.initialization_seed_schedule_hash,
        "--capability-profile-hash",
        binding.capability_profile_hash,
        "--summary-path",
        str(summary_path),
        "--checkpoint-output",
        str(checkpoint_output),
        "--seed",
        "42",
        "--device",
        "cuda:0",
        *_gpu_process_binding_argv(gpu_process_binding),
    ]


def _reference_eval_argv(
    *,
    session_id: str,
    task_id: str,
    subject: ExperimentSubject,
    manifest_hash: str,
    source_checksum: str,
    training_binding: TrainingBinding,
    capability_profile_hash: str,
    evaluation_binding,
    checkpoint_uri: str,
    summary_path: Path,
    gpu_process_binding: GpuProcessBinding | None = None,
) -> list[str]:
    protocol = EvaluationProtocol(
        eval_protocol_version=DEFAULT_EVAL_PROTOCOL_VERSION,
        public_seed_count=4,
        public_accept_seed_count=3,
        sealed_validation_seed_count=8,
        eval_num_envs=64,
    )
    return [
        "fake-python",
        "fake-eval.py",
        "--job-id",
        f"{session_id}-eval",
        "--round",
        session_id,
        "--task-id",
        task_id,
        "--subject-kind",
        subject.kind,
        "--subject-id",
        subject.subject_id,
        "--subject-manifest-hash",
        manifest_hash,
        "--program-source-checksum",
        source_checksum,
        "--training-binding-hash",
        str(training_binding.binding_hash),
        "--checkpoint",
        checkpoint_uri,
        "--public-seeds",
        "4",
        "--public-accept-seeds",
        "3",
        "--sealed-validation-seeds",
        "8",
        "--eval-protocol-version",
        protocol.eval_protocol_version,
        "--task-protocol-hash",
        evaluation_binding.task_protocol_hash,
        "--acceptance-protocol-hash",
        evaluation_binding.acceptance_protocol_hash,
        "--seed-cohort-hash",
        evaluation_binding.seed_cohort_hash,
        "--evaluation-binding-hash",
        evaluation_binding.binding_hash,
        "--capability-profile-hash",
        capability_profile_hash,
        "--summary-path",
        str(summary_path),
        "--num-envs",
        "64",
        "--seed",
        "0",
        "--device",
        "cuda:0",
        *_gpu_process_binding_argv(gpu_process_binding),
    ]


def _reference_program_for_task(
    tmp_path: Path,
    task: _ReferenceTaskPlugin,
):
    source = (
        "from true_rl_agent.candidates.components import TrainingComponents\n\n"
        "def build_training_components() -> TrainingComponents:\n"
        "    return TrainingComponents(name='official-reference-r0')\n"
    )
    source_path = tmp_path / "official_reference_components.py"
    source_path.write_bytes(source.encode("utf-8"))
    return source, resolve_reference_program_artifact(
        source_path,
        reference_id="official-reference-r0",
        task_id=task.task_spec.task_id,
        authorship_origin="official_upstream",
        upstream_project="official-locomotion",
        upstream_revision="revision-1",
        upstream_source_path="rewards/components.py",
        capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
    )


def _reference_fixture(tmp_path: Path, *, scientific: bool = False) -> dict[str, object]:
    task = _ReferenceTaskPlugin()
    backend = _ReferenceBackend()
    runner = _ReferenceRunner()
    binding = backend.evaluation_binding_for(task.task_spec)
    source, reference = _reference_program_for_task(tmp_path, task)
    genesis_identity_snapshot = {"fixture": "reference-session-v1"}
    if scientific:
        genesis_identity_snapshot = {
            "schema_version": "true_rl_agent.runtime_composition_snapshot.v3",
            "scientific": {"fixture": "reference-session-v1", "budget_steps": 128},
            "execution_placement": {"project_root": str(tmp_path), "execution_device": "cuda:0"},
        }
    genesis = prepare_session_genesis(
        output_dir=tmp_path,
        session_id="reference-session",
        task_plugin=task,
        backend_adapter=backend,
        runner=runner,
        policy_initialization=PolicyInitialization.random(),
        genesis_identity_snapshot=genesis_identity_snapshot,
        reference_program_artifact=reference,
    )
    return {
        "task": task,
        "backend": backend,
        "runner": runner,
        "binding": binding,
        "source": source,
        "reference": reference,
        "manifest": reference.manifest,
        "training_binding": genesis.artifact.anchor.training_binding,
        "genesis_identity_snapshot": genesis_identity_snapshot,
        "private_path": Path(genesis.artifact.anchor.private_eval_artifact.uri),
        "genesis": genesis,
    }


def _publish_test_generation_contract(
    output_dir: Path,
    *,
    session_id: str,
    task: TaskPlugin,
    backend: FakeBackendAdapter,
):
    contract = GenerationSessionContract(
        schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        session_mode="development_generation_pilot",
        session_id=session_id,
        task_id=task.task_spec.task_id,
        backend_id=backend.backend_id,
        max_generations=1,
        max_trials=2,
        max_train_jobs=2,
        bound_contract_hashes={
            "authoring_policy": "1" * 64,
            "continuation_policy": "2" * 64,
            "evaluation_cost": "3" * 64,
            "evaluation_protocol": "4" * 64,
            "exploration_projector": "5" * 64,
            "final_confirmation_plan": "6" * 64,
            "provider_policy": "7" * 64,
            "resource_plan": "8" * 64,
            "retry_policy": "9" * 64,
            "selector": "a" * 64,
            "session_cost": "b" * 64,
        },
    )
    return publish_generation_session_contract(output_dir, contract)


def _load_reference_genesis(
    tmp_path: Path,
    fixture: dict[str, object],
):
    return prepare_session_genesis(
        output_dir=tmp_path,
        session_id="reference-session",
        task_plugin=fixture["task"],
        backend_adapter=fixture["backend"],
        runner=fixture["runner"],
        policy_initialization=PolicyInitialization.random(),
        genesis_identity_snapshot=fixture["genesis_identity_snapshot"],
        reference_program_artifact=fixture["reference"],
    )


def _accepted_candidate_record(
    fixture: dict[str, object],
    *,
    genesis_hash: str,
    current_best,
) -> dict[str, object]:
    candidate_id = "candidate-after-reference"
    candidate_artifact = make_verified_candidate_artifact(candidate_id)
    candidate_manifest_hash = candidate_artifact.subject_manifest_hash
    candidate_source_checksum = candidate_artifact.program_source_checksum
    checkpoint = CheckpointRef(
        uri="artifact://candidate-after-reference/eval.pt",
        checksum=_checksum("candidate-checkpoint-bytes"),
        role="eval",
        sealed=True,
    )
    training_binding = fixture["training_binding"]
    train_spec = TrainJobSpec(
        job_id="round-001-train",
        task_id=fixture["task"].task_spec.task_id,
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=candidate_manifest_hash,
        program_source_checksum=candidate_source_checksum,
        policy_initialization=PolicyInitialization.random(),
        training_binding=training_binding,
        budget_steps=training_binding.budget_steps,
        execution_device="cuda:0",
        candidate_artifact=candidate_artifact,
    )
    eval_spec = EvalJobSpec(
        job_id="round-001-eval",
        task_id=fixture["task"].task_spec.task_id,
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=candidate_manifest_hash,
        program_source_checksum=candidate_source_checksum,
        training_binding_hash=str(training_binding.binding_hash),
        capability_profile_hash=training_binding.capability_profile_hash,
        checkpoint=checkpoint,
        public_seed_count=4,
        public_accept_seed_count=3,
        sealed_validation_seed_count=8,
        eval_num_envs=64,
        evaluation_binding=fixture["binding"],
        execution_device="cuda:0",
        metadata={"fake_public_score": 0.50},
    )
    public_result = fixture["runner"].run_eval(eval_spec)
    raw_score_stats = public_result.metrics["acceptance"]["score_stats"]
    score_stats = {
        key: raw_score_stats[key]
        for key in (
            "count",
            "mean",
            "std",
            "standard_error",
            "eval_protocol_hash",
        )
    }
    required_gates = list(
        public_result.metrics["acceptance"]["required_pass_gates"]
    )
    delta = 0.50 - current_best.score_for_comparison()
    return {
        "schema_version": LEDGER_RECORD_SCHEMA_V5,
        "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
        "round_id": "round-001",
        "parent_program_source_checksum": current_best.program_source_checksum,
        "session_genesis_hash": genesis_hash,
        "ledger_epoch_hash": primary_ledger_epoch_hash(genesis_hash),
        "evaluation_binding": fixture["binding"].to_public_dict(),
        "candidate": {
            "candidate_id": candidate_id,
            "changed_components": ["reward_callable"],
        },
        "verifier": {"status": "pass"},
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "jobs": {
            "train": train_spec.to_public_dict(),
            "eval": eval_spec.to_public_dict(),
        },
        "train_result": {"job_id": train_spec.job_id},
        "public_result": public_result.to_public_dict(),
        "decision": {
            "outcome": "policy_improvement_accept",
            "accepted": True,
            "reason": "noise_aware_delta",
            "surface_classification": "code_level",
            "public_score": 0.50,
            "comparison": {
                "mode": "reference",
                "score_source": FAKE_ACCEPTANCE_PROTOCOL.score_source,
                "acceptance_protocol_hash": (
                    FAKE_ACCEPTANCE_PROTOCOL.protocol_hash
                ),
                "candidate_score": 0.50,
                "current_best_score": current_best.score_for_comparison(),
                "delta": delta,
                "threshold": FAKE_ACCEPTANCE_PROTOCOL.min_delta,
                "delta_met": True,
                "absolute_accept_floor": (
                    FAKE_ACCEPTANCE_PROTOCOL.absolute_accept_floor
                ),
                "candidate_score_stats": score_stats,
                "best_score_stats": dict(current_best.public_accept_stats),
                "candidate_evaluation_binding_hash": (
                    fixture["binding"].binding_hash
                ),
                "current_best": current_best.comparison_identity(),
                "public_score": 0.50,
            },
            "consumed_gates": required_gates,
            "gate_statuses": {gate: "pass" for gate in required_gates},
        },
    }


def _run_reference_live_case(
    output_dir: Path,
    *,
    session_id: str,
    first_score: float,
) -> dict[str, object]:
    task = _LiveReferenceTaskPlugin()
    backend = _LiveReferenceBackend()
    runner = _LiveReferenceRunner()
    source, reference = _reference_program_for_task(output_dir, task)
    genesis = prepare_session_genesis(
        output_dir=output_dir,
        session_id=session_id,
        task_plugin=task,
        backend_adapter=backend,
        runner=runner,
        policy_initialization=PolicyInitialization.random(),
        genesis_identity_snapshot={"fixture": "reference-live-v1"},
        reference_program_artifact=reference,
    )
    candidates = [
        make_test_candidate_package(
            f"{session_id}-r1",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": first_score, "verifier_status": "pass"},
        ),
        make_test_candidate_package(
            f"{session_id}-r2",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.20, "verifier_status": "pass"},
        ),
    ]
    provider = _CapturingCandidateProvider(candidates)
    summary = run_live_loop(
        output_dir=output_dir,
        rounds=2,
        session_id=session_id,
        task_plugin=task,
        backend_adapter=backend,
        runner=runner,
        candidate_admission_policy=FakeVerifier(),
        provider=provider,
        policy_initialization=PolicyInitialization.random(),
        session_genesis=genesis,
    )
    records = load_ledger(output_dir / "ledger.jsonl")
    resume_provider = _CapturingCandidateProvider([])
    resumed = run_live_loop(
        output_dir=output_dir,
        rounds=2,
        session_id=session_id,
        task_plugin=task,
        backend_adapter=backend,
        runner=runner,
        candidate_admission_policy=FakeVerifier(),
        provider=resume_provider,
        policy_initialization=PolicyInitialization.random(),
        session_genesis=genesis,
    )
    return {
        "genesis": genesis,
        "reference": reference,
        "reference_source": source,
        "provider": provider,
        "resume_provider": resume_provider,
        "records": records,
        "summary": summary,
        "resumed": resumed,
    }


class ReferenceGenesisContractTests(unittest.TestCase):
    def test_scientific_import_owns_checkpoint_and_preserves_origin_on_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            fixture = _reference_fixture(source_dir, scientific=True)
            source = fixture["genesis"]
            origin_bytes = source.artifact_path.read_bytes()
            target = root / "new-session"
            contract_ref = _publish_test_generation_contract(
                target,
                session_id="import-owned",
                task=fixture["task"],
                backend=fixture["backend"],
            )
            snapshot = deepcopy(fixture["genesis_identity_snapshot"])
            snapshot["execution_placement"] = {
                "project_root": "/relocated/project",
                "execution_device": "cuda:1",
            }
            kwargs = dict(
                output_dir=target,
                session_id="import-owned",
                task_plugin=fixture["task"],
                backend_adapter=fixture["backend"],
                runner=fixture["runner"],
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot=snapshot,
                reference_program_artifact=fixture["reference"],
                generation_session_contract_ref=contract_ref,
                source_reference_genesis_dir=source_dir,
                source_reference_session_id="reference-session",
            )
            imported = prepare_session_genesis(**kwargs)
            self.assertEqual(
                imported.artifact.runtime_snapshot_hash,
                source.artifact.runtime_snapshot_hash,
            )
            local_checkpoint = Path(
                imported.reference_eval_plan().checkpoint.metadata["local_path"]
            )
            self.assertEqual(local_checkpoint, target / "reference_checkpoint.pt")
            self.assertEqual(
                file_sha256(local_checkpoint), source.artifact.anchor.checkpoint_hash
            )
            self.assertEqual(imported.artifact.anchor, source.artifact.anchor)
            for original in {
                Path(source.artifact.anchor.checkpoint_artifact.uri),
                source_dir / "reference_checkpoint.pt",
            }:
                original.chmod(0o600)
                original.unlink()
            reloaded = prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(
                reloaded.reference_eval_plan().checkpoint.metadata["local_path"],
                str(local_checkpoint),
            )
            self.assertEqual(
                reloaded.session_genesis_hash, imported.session_genesis_hash
            )
            self.assertEqual(source.artifact_path.read_bytes(), origin_bytes)
            self.assertEqual(
                (fixture["runner"].train_calls, fixture["runner"].eval_calls), (1, 1)
            )
            changed = deepcopy(snapshot)
            changed["scientific"]["budget_steps"] += 1
            with self.assertRaisesRegex(ValueError, "identity drift"):
                prepare_session_genesis(
                    **{**kwargs, "genesis_identity_snapshot": changed},
                    require_existing=True,
                )
            local_checkpoint.chmod(0o600)
            content = local_checkpoint.read_bytes()
            local_checkpoint.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            with self.assertRaisesRegex(ValueError, "checkpoint copy bytes changed"):
                prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(
                (fixture["runner"].train_calls, fixture["runner"].eval_calls), (1, 1)
            )
            self.assertEqual(source.artifact_path.read_bytes(), origin_bytes)

    def test_reference_eval_plan_restores_private_checkpoint_for_process_preflight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _reference_fixture(Path(tmp))
            plan = fixture["genesis"].reference_eval_plan()
            self.assertIsNotNone(plan)
            assert plan is not None
            checkpoint = plan.execution_checkpoint()
            private_payload = json.loads(fixture["private_path"].read_text("utf-8"))
            expected_checkpoint = private_payload["train_result"]["checkpoint"]

            self.assertEqual(
                checkpoint.to_public_dict(include_metadata=True),
                expected_checkpoint,
            )
            self.assertEqual(
                plan.to_public_dict(),
                private_payload["eval_job"]["spec"],
            )
            local_path = Path(checkpoint.metadata["local_path"]).resolve(strict=True)
            self.assertEqual(
                validate_checkpoint_process_input(
                    checkpoint,
                    argv=plan.metadata["cli_argv"],
                    checkpoint_path_option="--checkpoint",
                    checkpoint_checksum_option="--checkpoint-sha256",
                    resolved_local_path=local_path,
                ),
                local_path,
            )

    def test_legacy_reference_genesis_reads_posix_private_uris_on_any_host(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _reference_fixture(Path(tmp))
            payload = fixture["genesis"].artifact.to_dict()

        payload["schema_version"] = LEGACY_SESSION_GENESIS_SCHEMA_VERSION
        payload.pop("backend_execution_environment")
        payload["anchor"]["checkpoint_artifact"]["uri"] = (
            "/data/yuyangliu/history/reference_checkpoint.pt"
        )
        payload["anchor"]["private_eval_artifact"]["uri"] = (
            "/data/yuyangliu/history/reference_eval_artifact.json"
        )
        payload["session_genesis_hash"] = canonical_json_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "session_genesis_hash"
            }
        )

        historical = SessionGenesisArtifact.from_mapping(payload)
        self.assertEqual(
            historical.anchor.checkpoint_artifact.uri,
            "/data/yuyangliu/history/reference_checkpoint.pt",
        )
        self.assertEqual(
            historical.anchor.private_eval_artifact.uri,
            "/data/yuyangliu/history/reference_eval_artifact.json",
        )
    def test_reference_genesis_anchors_fresh_epoch_without_private_relocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_root = root / "formal-v2"
            epoch_root = root / "verified-v5"
            genesis_root.mkdir()
            task = _LiveReferenceTaskPlugin()
            backend = _LiveReferenceBackend()
            runner = _LiveReferenceRunner()
            _, reference = _reference_program_for_task(genesis_root, task)
            genesis = prepare_session_genesis(
                output_dir=genesis_root,
                session_id="reference-epoch-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "reference-epoch-v1"},
                reference_program_artifact=reference,
            )
            legacy_ledger = genesis_root / "ledger.jsonl"
            legacy_ledger.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "true_rl_agent.ledger_record.v2",
                        "session_genesis_hash": (
                            genesis.session_genesis_hash
                        ),
                        "decision": {"accepted": False},
                    },
                    trailing_newline=True,
                )
            )
            original_files = {
                path.relative_to(genesis_root): path.read_bytes()
                for path in genesis_root.rglob("*")
                if path.is_file()
            }
            provider = _CapturingCandidateProvider(
                [
                    make_test_candidate_package(
                        "reference-epoch-candidate",
                        change_set=CandidateChangeSet(
                            frozenset({"reward_callable"})
                        ),
                        metadata={
                            "fake_public_score": 0.50,
                            "verifier_status": "pass",
                        },
                    )
                ]
            )

            summary = run_live_loop(
                output_dir=epoch_root,
                rounds=1,
                session_id="reference-epoch-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
                session_genesis_dir=genesis_root,
                ledger_epoch_id="phase-f-v5-rerun",
            )
            final_files = {
                path.relative_to(genesis_root): path.read_bytes()
                for path in genesis_root.rglob("*")
                if path.is_file()
            }
            records = load_ledger(epoch_root / "ledger.jsonl")

        self.assertEqual(final_files, original_files)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["session_genesis_hash"], genesis.session_genesis_hash)
        self.assertEqual(summary["ledger_epoch"]["epoch_id"], "phase-f-v5-rerun")
        self.assertEqual(
            provider.payloads[0]["candidate_lineage"][
                "parent_program_source_checksum"
            ],
            reference.manifest.source_checksum,
        )
        self.assertIn(
            "official-reference-r0",
            json.dumps(provider.payloads[0]["source_payloads"]),
        )

    def _assert_live_training_fairness(self, fixture) -> None:
        genesis = fixture["genesis"]
        expected_initial = genesis.expected_initial_policy_state_hash
        expected_binding = genesis.initial_current_best.training_binding_hash
        for record in fixture["records"]:
            train_job = record["jobs"]["train"]
            pairing = record["train_result"]["initial_policy_state_pairing"]
            self.assertEqual(
                train_job["expected_initial_policy_state_hash"],
                expected_initial,
            )
            self.assertEqual(
                train_job["training_binding"]["binding_hash"],
                expected_binding,
            )
            self.assertEqual(
                pairing["expected_initial_policy_state_hash"],
                expected_initial,
            )
            self.assertEqual(
                pairing["actual_initial_policy_state_hash"],
                expected_initial,
            )
            self.assertTrue(pairing["match"])

    def test_reload_does_not_rerun_training_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            runner = fixture["runner"]
            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))

            reloaded = _load_reference_genesis(tmp_path, fixture)

            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))
            self.assertEqual(
                reloaded.session_genesis_hash,
                fixture["genesis"].session_genesis_hash,
            )

    def test_imported_reference_genesis_creates_fresh_session_without_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "qualified-reference"
            target_dir = root / "fresh-generation"
            source_dir.mkdir()
            fixture = _reference_fixture(source_dir)
            task = fixture["task"]
            backend = fixture["backend"]
            runner = fixture["runner"]
            source = fixture["genesis"]
            source_files = {
                path.relative_to(source_dir): path.read_bytes()
                for path in source_dir.rglob("*")
                if path.is_file()
            }
            target_session_id = "fresh-generation-session"
            contract_ref = _publish_test_generation_contract(
                target_dir,
                session_id=target_session_id,
                task=task,
                backend=backend,
            )
            kwargs = {
                "output_dir": target_dir,
                "session_id": target_session_id,
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "policy_initialization": PolicyInitialization.random(),
                "genesis_identity_snapshot": fixture["genesis_identity_snapshot"],
                "reference_program_artifact": fixture["reference"],
                "generation_session_contract_ref": contract_ref,
                "source_reference_genesis_dir": source_dir,
                "source_reference_session_id": "reference-session",
            }

            imported = prepare_session_genesis(**kwargs)

            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))
            self.assertEqual(
                imported.artifact.schema_version,
                IMPORTED_REFERENCE_SESSION_GENESIS_SCHEMA_VERSION,
            )
            self.assertEqual(imported.artifact.session_id, target_session_id)
            self.assertNotEqual(
                imported.session_genesis_hash,
                source.session_genesis_hash,
            )
            self.assertEqual(imported.artifact.anchor, source.artifact.anchor)
            self.assertEqual(
                imported.initial_current_best,
                source.initial_current_best,
            )
            import_ref = imported.reference_genesis_import_ref
            self.assertIsNotNone(import_ref)
            self.assertEqual(import_ref.source_session_id, "reference-session")
            self.assertEqual(
                import_ref.source_session_genesis_hash,
                source.session_genesis_hash,
            )
            self.assertEqual(
                Path(import_ref.source_session_genesis_uri),
                source.artifact_path,
            )
            self.assertEqual(
                import_ref.source_session_genesis_checksum,
                file_sha256(source.artifact_path),
            )
            self.assertFalse(
                (target_dir / "reference_eval_artifact.json").exists()
            )
            self.assertEqual(
                imported.reference_eval_artifact_path(),
                source.reference_eval_artifact_path(),
            )
            self.assertEqual(
                imported.reference_eval_plan(),
                source.reference_eval_plan(),
            )
            self.assertEqual(
                imported.reference_program_source(),
                fixture["source"],
            )
            self.assertEqual(
                source_files,
                {
                    path.relative_to(source_dir): path.read_bytes()
                    for path in source_dir.rglob("*")
                    if path.is_file()
                },
            )
            target_genesis_bytes = (
                target_dir / SESSION_GENESIS_FILENAME
            ).read_bytes()

            reloaded = prepare_session_genesis(**kwargs, require_existing=True)

            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))
            self.assertEqual(
                reloaded.session_genesis_hash,
                imported.session_genesis_hash,
            )
            self.assertEqual(
                (target_dir / SESSION_GENESIS_FILENAME).read_bytes(),
                target_genesis_bytes,
            )
            self.assertEqual(
                source_files,
                {
                    path.relative_to(source_dir): path.read_bytes()
                    for path in source_dir.rglob("*")
                    if path.is_file()
                },
            )
            environment_bound = replace(
                imported.artifact,
                backend_execution_environment={"manifest_hash": "e" * 64},
                session_genesis_hash=None,
            )
            environment_round_trip = SessionGenesisArtifact.from_mapping(
                environment_bound.to_dict()
            )
            self.assertEqual(
                dict(environment_round_trip.backend_execution_environment),
                {"manifest_hash": "e" * 64},
            )

    def test_imported_reference_genesis_rejects_import_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "qualified-reference"
            first_target_dir = root / "first-generation"
            second_target_dir = root / "second-generation"
            source_dir.mkdir()
            fixture = _reference_fixture(source_dir)
            first_session_id = "first-generation-session"
            first_contract_ref = _publish_test_generation_contract(
                first_target_dir,
                session_id=first_session_id,
                task=fixture["task"],
                backend=fixture["backend"],
            )
            prepare_session_genesis(
                output_dir=first_target_dir,
                session_id=first_session_id,
                task_plugin=fixture["task"],
                backend_adapter=fixture["backend"],
                runner=fixture["runner"],
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot=fixture["genesis_identity_snapshot"],
                reference_program_artifact=fixture["reference"],
                generation_session_contract_ref=first_contract_ref,
                source_reference_genesis_dir=source_dir,
                source_reference_session_id="reference-session",
            )
            second_session_id = "second-generation-session"
            second_contract_ref = _publish_test_generation_contract(
                second_target_dir,
                session_id=second_session_id,
                task=fixture["task"],
                backend=fixture["backend"],
            )

            with self.assertRaisesRegex(ValueError, "import chains are forbidden"):
                prepare_session_genesis(
                    output_dir=second_target_dir,
                    session_id=second_session_id,
                    task_plugin=fixture["task"],
                    backend_adapter=fixture["backend"],
                    runner=fixture["runner"],
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot=(
                        fixture["genesis_identity_snapshot"]
                    ),
                    reference_program_artifact=fixture["reference"],
                    generation_session_contract_ref=second_contract_ref,
                    source_reference_genesis_dir=first_target_dir,
                    source_reference_session_id=first_session_id,
                )

            self.assertFalse(
                (second_target_dir / SESSION_GENESIS_FILENAME).exists()
            )

    def test_imported_reference_genesis_rejects_source_checkpoint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "qualified-reference"
            target_dir = root / "fresh-generation"
            source_dir.mkdir()
            fixture = _reference_fixture(source_dir)
            checkpoint_path = Path(
                fixture["genesis"].artifact.anchor.checkpoint_artifact.uri
            )
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"drift")
            target_session_id = "fresh-generation-session"
            contract_ref = _publish_test_generation_contract(
                target_dir,
                session_id=target_session_id,
                task=fixture["task"],
                backend=fixture["backend"],
            )

            with self.assertRaisesRegex(ValueError, "checkpoint.*changed"):
                prepare_session_genesis(
                    output_dir=target_dir,
                    session_id=target_session_id,
                    task_plugin=fixture["task"],
                    backend_adapter=fixture["backend"],
                    runner=fixture["runner"],
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot=(
                        fixture["genesis_identity_snapshot"]
                    ),
                    reference_program_artifact=fixture["reference"],
                    generation_session_contract_ref=contract_ref,
                    source_reference_genesis_dir=source_dir,
                    source_reference_session_id="reference-session",
                )

            self.assertFalse((target_dir / SESSION_GENESIS_FILENAME).exists())

    def test_gpu_bound_reference_v3_round_trips_private_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_id = "gpu-reference-session"
            task = _ReferenceTaskPlugin()
            backend = _CountingReferenceBackend()
            runner = _ReferenceRunner()
            _, reference = _reference_program_for_task(output_dir, task)
            snapshot = {"fixture": "gpu-reference-session-v1"}
            resource_binding_hash = "d" * 64
            gpu_process_binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash=resource_binding_hash,
            )
            session_cost_hash = "a" * 64
            contract = GenerationSessionContract(
                schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
                session_mode="development_generation_pilot",
                session_id=session_id,
                task_id=task.task_spec.task_id,
                backend_id=backend.backend_id,
                max_generations=1,
                max_trials=2,
                max_train_jobs=2,
                bound_contract_hashes={
                    "authoring_policy": "1" * 64,
                    "continuation_policy": "2" * 64,
                    "evaluation_cost": "3" * 64,
                    "evaluation_protocol": "4" * 64,
                    "exploration_projector": "e" * 64,
                    "final_confirmation_plan": "5" * 64,
                    "provider_policy": "6" * 64,
                    "resource_plan": "7" * 64,
                    "retry_policy": "8" * 64,
                    "selector": "9" * 64,
                    "session_cost": session_cost_hash,
                },
            )
            contract_ref = publish_generation_session_contract(
                output_dir,
                contract,
            )
            common_scope = {
                "journal_root": output_dir / "job_attempts",
                "session_id": session_id,
                "genesis_identity_hash": session_genesis_identity_hash(snapshot),
                "session_contract_hash": contract.contract_hash,
                "lane_id": "lane-a",
                "invocation_segment_id": "session-genesis",
                "budget_contract_hash": session_cost_hash,
                "resource_binding_hash": resource_binding_hash,
            }
            train_scope = PreGenesisExecutionScope(
                **common_scope,
                attempt_budget_units=2,
                max_infrastructure_retries=1,
            )
            eval_scope = PreGenesisExecutionScope(
                **common_scope,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
            )
            kwargs = {
                "output_dir": output_dir,
                "session_id": session_id,
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "policy_initialization": PolicyInitialization.random(),
                "genesis_identity_snapshot": snapshot,
                "reference_program_artifact": reference,
                "generation_session_contract_ref": contract_ref,
                "gpu_process_binding": gpu_process_binding,
                "reference_train_attempt_scope": train_scope,
                "reference_eval_attempt_scope": eval_scope,
            }

            created = prepare_session_genesis(**kwargs)
            private_path = output_dir / "reference_eval_artifact.json"
            genesis_path = output_dir / SESSION_GENESIS_FILENAME
            private_bytes = private_path.read_bytes()
            genesis_bytes = genesis_path.read_bytes()
            payload = json.loads(private_bytes)

            self.assertEqual(
                payload["schema_version"],
                GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
            )
            for raw_binding in (
                payload["train_job"]["spec"]["gpu_process_binding"],
                payload["eval_job"]["spec"]["gpu_process_binding"],
                payload["train_result"]["gpu_process_binding"],
            ):
                self.assertEqual(raw_binding, gpu_process_binding.to_dict())
            self.assertEqual(backend.materialize_reference_calls, 1)
            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))

            loaded = prepare_session_genesis(**kwargs, require_existing=True)

            self.assertEqual(backend.materialize_reference_calls, 1)
            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))
            self.assertEqual(
                loaded.session_genesis_hash,
                created.session_genesis_hash,
            )
            self.assertEqual(private_path.read_bytes(), private_bytes)
            self.assertEqual(genesis_path.read_bytes(), genesis_bytes)
            self.assertIsNotNone(backend.reference_jobs)
            for spec in (
                backend.reference_jobs.train_spec,
                backend.reference_jobs.eval_spec,
                loaded.reference_eval_plan(),
            ):
                self.assertEqual(spec.gpu_process_binding, gpu_process_binding)
                self.assertEqual(spec.execution_device, "cuda:0")
                public_binding = spec.to_public_dict()["gpu_process_binding"]
                self.assertEqual(
                    public_binding,
                    gpu_process_binding.to_public_dict(),
                )
                self.assertNotIn("host_execution_device", public_binding)

    def test_unscoped_phased_reference_round_trips_three_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = _ReferenceTaskPlugin()
            backend = _CountingReferenceBackend()
            runner = _PhasedReferenceRunner()
            _, reference = _reference_program_for_task(output_dir, task)
            kwargs = {
                "output_dir": output_dir,
                "session_id": "unscoped-phased-reference-session",
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "policy_initialization": PolicyInitialization.random(),
                "genesis_identity_snapshot": {
                    "fixture": "unscoped-phased-reference-session-v1"
                },
                "reference_program_artifact": reference,
                "reference_phased_evaluation_adapter": _ReferencePhasedAdapter(),
            }

            created = prepare_session_genesis(**kwargs)
            payload = json.loads(
                (output_dir / "reference_eval_artifact.json").read_text("utf-8")
            )
            self.assertEqual(
                payload["schema_version"],
                PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
            )
            self.assertEqual(
                {item["phase"] for item in payload["evaluation_assembly"]["phase_receipts"]},
                {phase.value for phase in EvaluationPhase},
            )
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))
            self.assertEqual(runner.eval_calls, 0)
            self.assertEqual(runner.projection_calls, 1)

            loaded = prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(loaded.session_genesis_hash, created.session_genesis_hash)
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))

    def test_reference_dev_gate_stops_before_holdouts_and_reuses_failed_receipt(self):
        from unittest.mock import patch
        from true_rl_agent.agent_core.session_genesis import _run_reference_phased_evaluation

        for status in ("pass", "fail"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                task, backend, runner = _ReferenceTaskPlugin(), _CountingReferenceBackend(), _PhasedReferenceRunner()
                _, reference = _reference_program_for_task(output_dir, task)
                kwargs = dict(output_dir=output_dir, session_id="dev-gated-reference", task_plugin=task,
                    backend_adapter=backend, runner=runner, policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixture": "dev-gated-reference"}, reference_program_artifact=reference,
                    reference_phased_evaluation_adapter=_ReferencePhasedAdapter(status))
                with patch("true_rl_agent.agent_core.session_genesis._run_reference_phased_evaluation",
                           wraps=_run_reference_phased_evaluation) as invoked:
                    if status == "fail":
                        with self.assertRaisesRegex(ValueError, "task-owned public-dev gate"):
                            prepare_session_genesis(**kwargs)
                    else:
                        prepare_session_genesis(**kwargs)
                gate_path = output_dir / "reference_public_dev_gate.json"
                original = (gate_path.read_bytes(), gate_path.stat().st_mtime_ns)
                self.assertEqual(json.loads(original[0])["task_gate"]["status"], status)
                self.assertEqual(runner.phase_calls, [EvaluationPhase.PUBLIC_DEV] if status == "fail" else list(EvaluationPhase))
                self.assertEqual(runner.train_calls, 1)
                if status == "fail":
                    self.assertFalse((output_dir / SESSION_GENESIS_FILENAME).exists())
                    with self.assertRaisesRegex(ValueError, "task-owned public-dev gate"):
                        _run_reference_phased_evaluation(**invoked.call_args.kwargs)
                    self.assertEqual(runner.phase_calls, [EvaluationPhase.PUBLIC_DEV])
                    self.assertEqual((gate_path.read_bytes(), gate_path.stat().st_mtime_ns), original)

    def test_gpu_bound_phased_reference_round_trips_three_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_id = "gpu-phased-reference-session"
            task = _ReferenceTaskPlugin()
            backend = _CountingReferenceBackend()
            runner = _PhasedReferenceRunner()
            adapter = _ReferencePhasedAdapter()
            _, reference = _reference_program_for_task(output_dir, task)
            snapshot = {"fixture": "gpu-phased-reference-session-v1"}
            resource_binding_hash = "d" * 64
            gpu_process_binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash=resource_binding_hash,
            )
            session_cost_hash = "a" * 64
            contract = GenerationSessionContract(
                schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
                session_mode="development_generation_pilot",
                session_id=session_id,
                task_id=task.task_spec.task_id,
                backend_id=backend.backend_id,
                max_generations=1,
                max_trials=2,
                max_train_jobs=2,
                bound_contract_hashes={
                    "authoring_policy": "1" * 64,
                    "continuation_policy": "2" * 64,
                    "evaluation_cost": "3" * 64,
                    "evaluation_protocol": "4" * 64,
                    "exploration_projector": "e" * 64,
                    "final_confirmation_plan": "5" * 64,
                    "provider_policy": "6" * 64,
                    "resource_plan": "7" * 64,
                    "retry_policy": "8" * 64,
                    "selector": "9" * 64,
                    "session_cost": session_cost_hash,
                },
            )
            contract_ref = publish_generation_session_contract(output_dir, contract)
            common_scope = {
                "journal_root": output_dir / "job_attempts",
                "session_id": session_id,
                "genesis_identity_hash": session_genesis_identity_hash(snapshot),
                "session_contract_hash": contract.contract_hash,
                "lane_id": "lane-a",
                "budget_contract_hash": session_cost_hash,
                "resource_binding_hash": resource_binding_hash,
            }
            train_scope = PreGenesisExecutionScope(
                **common_scope,
                invocation_segment_id="session-genesis",
                attempt_budget_units=2,
                max_infrastructure_retries=1,
            )
            phase_scopes = {
                phase: PreGenesisExecutionScope(
                    **common_scope,
                    invocation_segment_id=f"session-genesis-{phase.value}",
                    attempt_budget_units=1,
                    max_infrastructure_retries=0,
                )
                for phase in EvaluationPhase
            }
            kwargs = {
                "output_dir": output_dir,
                "session_id": session_id,
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "policy_initialization": PolicyInitialization.random(),
                "genesis_identity_snapshot": snapshot,
                "reference_program_artifact": reference,
                "generation_session_contract_ref": contract_ref,
                "gpu_process_binding": gpu_process_binding,
                "reference_train_attempt_scope": train_scope,
                "reference_phased_evaluation_adapter": adapter,
                "reference_eval_phase_attempt_scopes": phase_scopes,
            }

            created = prepare_session_genesis(**kwargs)
            private_path = output_dir / "reference_eval_artifact.json"
            payload = json.loads(private_path.read_text("utf-8"))
            self.assertEqual(
                payload["schema_version"],
                PHASED_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
            )
            self.assertEqual(
                {item["phase"] for item in payload["evaluation_assembly"]["phase_receipts"]},
                {phase.value for phase in EvaluationPhase},
            )
            self.assertEqual(runner.train_calls, 1)
            self.assertEqual(runner.eval_calls, 0)
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))
            self.assertEqual(runner.projection_calls, 1)

            loaded = prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(
                loaded.session_genesis_hash,
                created.session_genesis_hash,
            )
            self.assertEqual(runner.train_calls, 1)
            self.assertEqual(runner.eval_calls, 0)
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))
            self.assertEqual(runner.projection_calls, 1)

            first_receipt = Path(
                payload["evaluation_assembly"]["phase_receipts"][0]["path"]
            )
            first_receipt.write_bytes(first_receipt.read_bytes() + b" ")
            with self.assertRaisesRegex(
                ValueError,
                "checksum mismatch|not canonical JSON",
            ):
                prepare_session_genesis(**kwargs, require_existing=True)

    def test_require_existing_rejects_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "missing-session"
            task = _ReferenceTaskPlugin()
            backend = _CountingReferenceBackend()
            runner = _ReferenceRunner()
            _, reference = _reference_program_for_task(Path(tmp), task)

            with self.assertRaisesRegex(FileNotFoundError, "Genesis is required"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="reference-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixture": "existing-only-v1"},
                    reference_program_artifact=reference,
                    require_existing=True,
                )

            self.assertFalse(output_dir.exists())
            self.assertEqual(backend.materialize_reference_calls, 0)
            self.assertEqual((runner.train_calls, runner.eval_calls), (0, 0))

    def test_concurrent_reference_creator_cannot_materialize_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = _ReferenceTaskPlugin()
            winner_backend = _CountingReferenceBackend()
            loser_backend = _CountingReferenceBackend()
            winner_runner = _BlockingReferenceRunner()
            loser_runner = _ReferenceRunner()
            _, reference = _reference_program_for_task(output_dir, task)
            identity = {"fixture": "reference-concurrent-v1"}

            def create_winner():
                return prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="reference-concurrent",
                    task_plugin=task,
                    backend_adapter=winner_backend,
                    runner=winner_runner,
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot=identity,
                    reference_program_artifact=reference,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(create_winner)
                self.assertTrue(winner_runner.train_started.wait(timeout=5))
                checkpoint_path = (
                    output_dir
                    / "runner"
                    / "reference-concurrent"
                    / "checkpoints"
                    / "best.pt"
                )
                self.assertFalse(checkpoint_path.exists())

                with self.assertRaises(SessionGenesisCreationInProgressError):
                    prepare_session_genesis(
                        output_dir=output_dir,
                        session_id="reference-concurrent",
                        task_plugin=task,
                        backend_adapter=loser_backend,
                        runner=loser_runner,
                        policy_initialization=PolicyInitialization.random(),
                        genesis_identity_snapshot=identity,
                        reference_program_artifact=reference,
                    )
                self.assertEqual(loser_backend.materialize_reference_calls, 0)
                self.assertEqual((loser_runner.train_calls, loser_runner.eval_calls), (0, 0))
                self.assertFalse(checkpoint_path.exists())

                winner_runner.allow_train.set()
                winner = future.result(timeout=10)

            checkpoint_bytes = checkpoint_path.read_bytes()
            self.assertEqual(winner_backend.materialize_reference_calls, 1)
            self.assertEqual((winner_runner.train_calls, winner_runner.eval_calls), (1, 1))
            self.assertTrue(
                (
                    output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME
                ).exists()
            )

            loaded = prepare_session_genesis(
                output_dir=output_dir,
                session_id="reference-concurrent",
                task_plugin=task,
                backend_adapter=loser_backend,
                runner=loser_runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot=identity,
                reference_program_artifact=reference,
            )
            self.assertEqual(
                loaded.session_genesis_hash,
                winner.session_genesis_hash,
            )
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)
            self.assertEqual(loser_backend.materialize_reference_calls, 0)
            self.assertEqual((loser_runner.train_calls, loser_runner.eval_calls), (0, 0))

    def test_checkpoint_byte_drift_is_rejected_on_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            checkpoint_path = Path(
                fixture["genesis"].artifact.anchor.checkpoint_artifact.uri
            )
            checkpoint_path.write_bytes(b"forged-reference-checkpoint")

            with self.assertRaisesRegex(
                ValueError,
                "checkpoint bytes changed after Genesis",
            ):
                _load_reference_genesis(tmp_path, fixture)

    def test_failed_capability_gate_creates_no_genesis_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task = _FailingReferenceTaskPlugin()
            backend = _ReferenceBackend()
            runner = _ReferenceRunner()
            _, reference = _reference_program_for_task(tmp_path, task)

            with self.assertRaisesRegex(ValueError, "capability gate"):
                prepare_session_genesis(
                    output_dir=tmp_path,
                    session_id="failed-reference-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixture": "failed-reference-v1"},
                    reference_program_artifact=reference,
                )

            self.assertEqual((runner.train_calls, runner.eval_calls), (1, 1))
            self.assertFalse((tmp_path / SESSION_GENESIS_FILENAME).exists())
            self.assertFalse((tmp_path / "reference_eval_artifact.json").exists())
            self.assertFalse((tmp_path / "ledger.jsonl").exists())
            failure = json.loads(
                (tmp_path / "reference_qualification_failure.json").read_text("utf-8")
            )
            self.assertEqual(failure["session_id"], "failed-reference-session")
            self.assertEqual(failure["reference_manifest_hash"], reference.manifest.manifest_hash)
            self.assertEqual(failure["qualification_gate"]["status"], "fail")
            evidence = failure["qualification_evidence"]
            self.assertEqual(evidence["result"], "fixture-fail")
            self.assertEqual(
                failure["qualification_gate"]["evidence_hash"],
                canonical_json_hash({k: v for k, v in evidence.items() if k != "evidence_hash"}),
            )
            self.assertEqual(
                failure["evaluation_binding"]["task_protocol_hash"],
                task.task_spec.resolved_task_protocol_hash(),
            )
            from true_rl_agent.agent_core.session_genesis import _reject_partial_session_without_genesis
            with self.assertRaisesRegex(ValueError, "reference_qualification_failure.json"):
                _reject_partial_session_without_genesis(
                    tmp_path, session_id="failed-reference-session", allow_pre_genesis_state=True)

    def test_wrong_qualification_protocol_fails_before_genesis_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task = _WrongProtocolReferenceTaskPlugin()
            backend = _ReferenceBackend()
            runner = _ReferenceRunner()
            _, reference = _reference_program_for_task(tmp_path, task)

            with self.assertRaisesRegex(ValueError, "task protocol mismatch"):
                prepare_session_genesis(
                    output_dir=tmp_path,
                    session_id="wrong-protocol-reference-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={
                        "fixture": "wrong-protocol-reference-v1"
                    },
                    reference_program_artifact=reference,
                )

            self.assertFalse((tmp_path / SESSION_GENESIS_FILENAME).exists())
            self.assertFalse((tmp_path / "reference_eval_artifact.json").exists())
            self.assertFalse((tmp_path / "ledger.jsonl").exists())
            self.assertFalse((tmp_path / "reference_qualification_failure.json").exists())

    def test_restores_current_best_and_source_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)

            self.assertEqual(genesis.initial_current_best.mode, "reference")
            self.assertEqual(genesis.initial_current_best.anchor_kind, "reference")
            self.assertTrue(genesis.initial_current_best.has_full_eval_stats())
            self.assertEqual(genesis.reference_program_source(), fixture["source"])
            anchor_evidence = genesis.initial_anchor_public_evidence()
            self.assertEqual(anchor_evidence["anchor_kind"], "reference_program")
            self.assertEqual(
                anchor_evidence["public_result"],
                genesis.artifact.anchor.to_dict()["public_result"],
            )
            self.assertEqual(
                canonical_json_hash(anchor_evidence["public_result"]),
                anchor_evidence["public_result_hash"],
            )
            builder = PublicContextBuilder(
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                initial_current_best=genesis.initial_current_best,
                initial_anchor_public_evidence=anchor_evidence,
            )
            self.assertEqual(builder.initial_anchor_public_evidence, anchor_evidence)
            self.assertEqual(anchor_evidence["training_runtime"]["checkpoint_hash"],
                             genesis.initial_current_best.checkpoint_hash)
            tampered = json.loads(json.dumps(anchor_evidence))
            tampered["public_result"]["public_score"] = 0.99
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                PublicContextBuilder(
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    initial_current_best=genesis.initial_current_best,
                    initial_anchor_public_evidence=tampered,
                )
            tampered["public_result_hash"] = canonical_json_hash(
                tampered["public_result"]
            )
            with self.assertRaisesRegex(ValueError, "does not restore current-best"):
                PublicContextBuilder(
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    initial_current_best=genesis.initial_current_best,
                    initial_anchor_public_evidence=tampered,
                )

            source_context = build_prompt_source_context(
                [],
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                initial_current_best=genesis.initial_current_best,
                session_genesis_hash=genesis.session_genesis_hash,
                initial_reference_program_source=genesis.reference_program_source(),
            )
            self.assertEqual(
                source_context["candidate_lineage"]["status"],
                "reference_current_best",
            )
            current = source_context["candidate_lineage"]["current_best"]
            self.assertEqual(current["reference_id"], "official-reference-r0")
            self.assertEqual(
                current["current_head_program_source_checksum"],
                fixture["manifest"].source_checksum,
            )
            self.assertEqual(
                source_context["candidate_lineage"][
                    "parent_program_source_checksum"
                ],
                fixture["manifest"].source_checksum,
            )
            payload = next(
                item
                for item in source_context["source_payloads"]
                if item["logical_id"] == CURRENT_BEST_SOURCE_ID
            )
            self.assertEqual(payload["role"], "reference_program_anchor")
            self.assertEqual(payload["content"], fixture["source"])

    def test_first_candidate_advances_without_sequence_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)
            record = _accepted_candidate_record(
                fixture,
                genesis_hash=genesis.session_genesis_hash,
                current_best=genesis.initial_current_best,
            )

            validation = validate_decision_record(
                record,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                expected_session_genesis_hash=genesis.session_genesis_hash,
                expected_evaluation_binding=fixture["binding"],
            )
            replay = replay_acceptance_ledger(
                [record],
                initial_current_best=genesis.initial_current_best,
                expected_session_genesis_hash=genesis.session_genesis_hash,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
            )

            self.assertTrue(validation.valid_policy_accept, validation.reasons)
            self.assertEqual(
                replay.entries[0].current_best_before.mode,
                "reference",
            )
            self.assertEqual(replay.entries[0].current_best_after.mode, "accepted")
            self.assertEqual(replay.current_best.anchor_kind, "candidate")
            self.assertEqual(replay.current_best.accepted_round_id, "round-001")
            self.assertEqual(
                replay.current_best.accepted_candidate_id,
                "candidate-after-reference",
            )
            self.assertEqual(
                record["parent_program_source_checksum"],
                genesis.initial_current_best.program_source_checksum,
            )

    def test_live_r1_accept_advances_r2_parent_and_replays_without_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _run_reference_live_case(
                Path(tmp),
                session_id="reference-live-accept",
                first_score=0.60,
            )

        records = fixture["records"]
        provider = fixture["provider"]
        resume_provider = fixture["resume_provider"]
        genesis = fixture["genesis"]
        self.assertEqual(len(records), 2)
        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(resume_provider.payloads, [])
        r0_feedback = provider.payloads[0]["public_feedback_summary"]
        self.assertEqual(r0_feedback["status"], "available")
        self.assertEqual(r0_feedback["anchor_kind"], "reference_program")
        self.assertEqual(
            r0_feedback["public_result_hash"],
            genesis.initial_anchor_public_evidence()["public_result_hash"],
        )
        self.assertNotIn(
            "initial_anchor_public_evidence",
            json.dumps(provider.payloads[0], sort_keys=True),
        )
        self.assertTrue(records[0]["decision"]["accepted"])
        r1_source = records[0]["jobs"]["eval"]["program_source_checksum"]
        r2_lineage = provider.payloads[1]["candidate_lineage"]
        self.assertEqual(r2_lineage["status"], "accepted_current_best")
        self.assertEqual(r2_lineage["parent_program_source_checksum"], r1_source)
        self.assertEqual(records[1]["parent_program_source_checksum"], r1_source)
        self._assert_live_training_fairness(fixture)
        self.assertNotEqual(
            r1_source,
            genesis.initial_current_best.program_source_checksum,
        )
        self.assertTrue(all(fixture["summary"]["checks"].values()))
        self.assertTrue(all(fixture["resumed"]["checks"].values()))

    def test_live_r1_deny_keeps_r0_as_r2_parent_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _run_reference_live_case(
                Path(tmp),
                session_id="reference-live-deny",
                first_score=0.20,
            )

        records = fixture["records"]
        provider = fixture["provider"]
        resume_provider = fixture["resume_provider"]
        genesis = fixture["genesis"]
        r0_source = genesis.initial_current_best.program_source_checksum
        self.assertEqual(len(records), 2)
        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(resume_provider.payloads, [])
        self.assertFalse(records[0]["decision"]["accepted"])
        r2_lineage = provider.payloads[1]["candidate_lineage"]
        self.assertEqual(r2_lineage["status"], "reference_current_best")
        self.assertEqual(r2_lineage["parent_program_source_checksum"], r0_source)
        self.assertEqual(records[1]["parent_program_source_checksum"], r0_source)
        self._assert_live_training_fairness(fixture)
        self.assertTrue(all(fixture["summary"]["checks"].values()))
        self.assertTrue(all(fixture["resumed"]["checks"].values()))

    def test_rejects_private_eval_job_or_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            private_path = fixture["private_path"]
            payload = json.loads(private_path.read_text(encoding="utf-8"))
            payload["eval_job"]["spec"]["job_id"] = "forged-eval-job"
            private_path.write_bytes(
                canonical_json_bytes(payload, trailing_newline=True)
            )
            private_checksum = file_sha256(private_path)
            self.assertIsNotNone(private_checksum)
            genesis_path = tmp_path / SESSION_GENESIS_FILENAME
            genesis_payload = json.loads(genesis_path.read_text(encoding="utf-8"))
            genesis_payload["anchor"]["private_eval_artifact"]["checksum"] = (
                private_checksum
            )
            genesis_payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in genesis_payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path.write_bytes(
                canonical_json_bytes(genesis_payload, trailing_newline=True)
            )

            with self.assertRaisesRegex(ValueError, "--job-id"):
                _load_reference_genesis(tmp_path, fixture)

    def test_rejects_public_execution_device_drift_on_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            private_path = fixture["private_path"]
            private_payload = json.loads(private_path.read_text(encoding="utf-8"))
            public_result = private_payload["public_result"]
            public_result["metrics"]["execution_device"] = "cuda:1"
            private_path.write_bytes(
                canonical_json_bytes(private_payload, trailing_newline=True)
            )

            genesis_path = tmp_path / SESSION_GENESIS_FILENAME
            genesis_payload = json.loads(genesis_path.read_text(encoding="utf-8"))
            genesis_payload["anchor"]["public_result"] = public_result
            genesis_payload["anchor"]["public_result_hash"] = canonical_json_hash(
                public_result
            )
            genesis_payload["anchor"]["private_eval_artifact"]["checksum"] = (
                file_sha256(private_path)
            )
            genesis_payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in genesis_payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path.write_bytes(
                canonical_json_bytes(genesis_payload, trailing_newline=True)
            )

            with self.assertRaisesRegex(
                ValueError,
                "public execution device mismatch",
            ):
                _load_reference_genesis(tmp_path, fixture)

    def test_source_projection_rejects_unbound_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)

            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                build_prompt_source_context(
                    [],
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    initial_current_best=genesis.initial_current_best,
                    session_genesis_hash=genesis.session_genesis_hash,
                    initial_reference_program_source="forged source",
                )


if __name__ == "__main__":
    unittest.main()

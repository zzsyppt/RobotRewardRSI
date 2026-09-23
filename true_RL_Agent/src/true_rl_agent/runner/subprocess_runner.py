"""Runner-owned subprocess execution boundary."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from true_rl_agent.agent_core.public_safety import (
    redact_secret_key_markers,
    sanitize_untrusted_public_diagnostic,
)
from true_rl_agent.backends.base import (
    BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
)
from true_rl_agent.composition.cell_manifest import (
    ResolvedCellManifestV1,
    load_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.evaluation.public_metric_contract import (
    RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION,
    EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION,
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    MAX_EVALUATION_PHASE_RECEIPT_BYTES,
    EvaluationAssembly,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.result_contract import (
    validate_public_eval_result_fields,
    validate_result_status,
)
from true_rl_agent.evaluation.sealed_validation import (
    SEALED_VALIDATION_EVIDENCE_KEY,
    build_sealed_validation_gate,
    validate_sealed_rollout_evidence,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.base import (
    CandidateRuntimeError,
    Runner,
    RunnerExecutionError,
    RunnerFailureKind,
    TrainJobResult,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import (
    seal_checkpoint,
    validate_checkpoint_safety_report,
    validate_train_checkpoint_artifact,
)
from true_rl_agent.runner.checkpoint_preflight import (
    validate_checkpoint_process_input,
)
from true_rl_agent.runner.eval_isolation import (
    CANDIDATE_CONTAMINATION_FIELDS,
    build_candidate_isolation_gate,
    reject_candidate_contamination_flags,
    reject_candidate_path_arguments,
)
from true_rl_agent.runner.evidence import validate_evidence_projection_bindings
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.gpu_lease import (
    GpuLeaseHolder,
    GpuLeaseManager,
    GpuLeaseUnavailableError,
    PreGenesisGpuLeaseHolder,
)
from true_rl_agent.runner.job_attempts import (
    JobAttemptExecutionScope,
    JobAttemptJournal,
    JobAttemptRecoveryRequiredError,
    JobAttemptScope,
    JobAttemptTerminalResultError,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.runner.argv_contract import (
    option_occurrences,
    unique_option_value,
)
from true_rl_agent.runner.job_identity import (
    ALTERNATE_JOB_IDENTITY_FIELDS,
    JobIdentity,
    validate_job_summary_identity,
)
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    EvaluationJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
    require_eval_phase_result_matches_spec,
    require_positive_integer,
)
from true_rl_agent.runner.process_handle import (
    OwnedProcessCleanupIncomplete,
    OwnedProcessLaunchError,
    OwnedProcessResult,
    OwnedProcessTimeout,
    run_owned_process,
    validate_owned_process_identity,
)


@dataclass(frozen=True)
class _SummaryJobCompletion:
    """Validated structured summary paired with its process completion status."""

    summary: dict[str, Any]
    status: str
    returncode: int
    gpu_process_binding: GpuProcessBinding | None = None
    owned_process_identity: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _GpuJobExecutionContext:
    """Trusted, immutable host-to-child mapping for one owned GPU process."""

    gpu_process_binding: GpuProcessBinding
    physical_id: str

    @property
    def environment_overrides(self) -> dict[str, str | None]:
        return {
            "CUDA_VISIBLE_DEVICES": self.physical_id,
            "NVIDIA_VISIBLE_DEVICES": None,
        }


_EVAL_CHECKPOINT_OPTIONS = ("--checkpoint", "--checkpoint-sha256")
_TRAIN_CHECKPOINT_OPTIONS = (
    "--init-checkpoint",
    "--init-checkpoint-sha256",
)
_ExecutionResultT = TypeVar(
    "_ExecutionResultT",
    TrainJobResult,
    PublicEvalResult,
    EvaluationPhaseResult,
)
AttemptAdmissionHook = Callable[
    [JobAttemptExecutionScope, str, str, int],
    int | None,
]
EvalCheckpointAdmissionHook = Callable[
    [CheckpointRef, ResolvedCellManifestV1, tuple[str, ...]], None
]


@dataclass(frozen=True)
class SubprocessRunner(Runner):
    """Execute materialized train/eval jobs through structured argv only."""

    allowed_executables: tuple[str, ...]
    timeout_s: int = 3600
    cwd: Path | None = None
    termination_grace_s: int = 5
    gpu_lease_manager: GpuLeaseManager | None = None
    attempt_admission_hook: AttemptAdmissionHook | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    eval_checkpoint_admission_hook: EvalCheckpointAdmissionHook | None = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.allowed_executables, tuple)
            or not self.allowed_executables
            or any(
                not isinstance(executable, str) or not executable.strip()
                for executable in self.allowed_executables
            )
        ):
            raise ValueError("allowed_executables must be a non-empty tuple of strings")
        require_positive_integer(self.timeout_s, "runner timeout_s")
        require_positive_integer(
            self.termination_grace_s,
            "runner termination_grace_s",
        )
        if self.cwd is not None and not isinstance(self.cwd, Path):
            raise TypeError("runner cwd must be a Path when provided")
        if self.gpu_lease_manager is not None and not isinstance(
            self.gpu_lease_manager,
            GpuLeaseManager,
        ):
            raise TypeError("gpu_lease_manager must be a GpuLeaseManager")
        if self.attempt_admission_hook is not None and not callable(
            self.attempt_admission_hook
        ):
            raise TypeError("attempt_admission_hook must be callable when provided")
        if self.eval_checkpoint_admission_hook is not None and not callable(
            self.eval_checkpoint_admission_hook
        ):
            raise TypeError("eval_checkpoint_admission_hook must be callable when provided")

    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        return self._execute_with_attempts(
            train_spec,
            job_kind="train",
            operation=lambda log_dir, execution_context, timeout_s: (
                self._run_train_once(
                    train_spec,
                    process_log_dir=log_dir,
                    attempt_timeout_s=timeout_s,
                    execution_context=execution_context,
                )
            ),
            recover_completed=lambda execution_context: self._run_train_once(
                train_spec,
                process_log_dir=None,
                reuse_existing_summary=True,
                execution_context=execution_context,
            ),
        )

    def recover_completed_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        """Validate retained training evidence without launching or retrying a job."""
        return self._run_train_once(
            train_spec, process_log_dir=None, reuse_existing_summary=True,
        )

    def _run_train_once(
        self,
        train_spec: TrainJobSpec,
        *,
        process_log_dir: Path | None,
        attempt_timeout_s: int | None = None,
        reuse_existing_summary: bool = False,
        execution_context: _GpuJobExecutionContext | None = None,
    ) -> TrainJobResult:
        argv = self._validated_argv(
            train_spec.metadata,
            job_id=train_spec.job_id,
        )
        try:
            validate_train_job_argv(train_spec, argv)
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{train_spec.job_id}: train argv binding failed: {exc}"
            ) from exc
        training_start_checkpoint = train_spec.training_start_checkpoint()
        expected_checkpoint_path = self._required_local_output_path(
            train_spec.metadata,
            key="checkpoint_output_path",
            job_id=train_spec.job_id,
        )
        completion = self._execute_summary_job(
            train_spec.metadata,
            argv=argv,
            identity=train_spec.job_identity(),
            summary_name="train summary",
            stale_output_paths=(("checkpoint_output_path", expected_checkpoint_path),),
            checkpoint=training_start_checkpoint,
            checkpoint_path_option=_TRAIN_CHECKPOINT_OPTIONS[0],
            checkpoint_checksum_option=_TRAIN_CHECKPOINT_OPTIONS[1],
            process_log_dir=process_log_dir,
            attempt_timeout_s=attempt_timeout_s,
            reuse_existing_summary=reuse_existing_summary,
            execution_context=execution_context,
            reused_gpu_process_binding=(train_spec.gpu_process_binding if reuse_existing_summary else None),
        )
        summary = completion.summary
        actual_gpu_process_binding = completion.gpu_process_binding
        self._validate_backend_execution_environment_summary(summary, train_spec)
        self._validate_execution_device_summary(summary, train_spec)
        self._validate_train_binding_summary(summary, train_spec)
        if completion.status == "fail":
            self._raise_failed_train_summary(
                completion,
                train_spec=train_spec,
            )
        initial_policy_state_pairing = self._initial_policy_state_pairing(
            summary,
            train_spec=train_spec,
        )
        checkpoint = self._checkpoint_from_summary(summary, job_id=train_spec.job_id)
        checkpoint = self._validate_train_checkpoint_output(
            checkpoint,
            expected_path=expected_checkpoint_path,
            metadata=train_spec.metadata,
            job_id=train_spec.job_id,
        )
        checkpoint = CheckpointRef(
            uri=checkpoint.uri,
            checksum=checkpoint.checksum,
            role=checkpoint.role,
            sealed=checkpoint.sealed,
            metadata={
                **dict(checkpoint.metadata),
                "training_subject": train_spec.subject.to_public_dict(),
                "subject_manifest_hash": train_spec.subject_manifest_hash,
                "program_source_checksum": train_spec.program_source_checksum,
                "training_binding_hash": train_spec.training_binding.binding_hash,
                "capability_profile_hash": (
                    train_spec.training_binding.capability_profile_hash
                ),
                "execution_device": (
                    actual_gpu_process_binding.process_execution_device
                    if actual_gpu_process_binding is not None
                    else train_spec.execution_device
                ),
                **(
                    {
                        "resolved_cell_manifest_hash": summary[
                            "resolved_cell_manifest_hash"
                        ],
                        "resolved_cell_identity": dict(
                            summary["resolved_cell_identity"]
                        ),
                    }
                    if "resolved_cell_manifest_hash" in summary
                    else {}
                ),
                **(
                    {"gpu_process_binding": (actual_gpu_process_binding.to_dict())}
                    if actual_gpu_process_binding is not None
                    else {}
                ),
                **(
                    {
                        "initial_policy_state_pairing": (
                            initial_policy_state_pairing.to_public_dict()
                        )
                    }
                    if initial_policy_state_pairing is not None
                    else {}
                ),
            },
        )
        sealed_uri = train_spec.metadata.get("sealed_checkpoint_uri", checkpoint.uri)
        if not isinstance(sealed_uri, str) or not sealed_uri:
            raise RunnerExecutionError(
                f"{train_spec.job_id}: sealed_checkpoint_uri must be a non-empty string"
            )
        summary = self._project_formal_training_completion(
            summary, train_spec=train_spec, argv=argv, checkpoint=checkpoint
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            subject=train_spec.subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=str(train_spec.training_binding.binding_hash),
            capability_profile_hash=(
                train_spec.training_binding.capability_profile_hash
            ),
            execution_device=(
                actual_gpu_process_binding.process_execution_device
                if actual_gpu_process_binding is not None
                else train_spec.execution_device
            ),
            checkpoint=seal_checkpoint(checkpoint, sealed_uri=sealed_uri),
            gpu_process_binding=actual_gpu_process_binding,
            initial_policy_state_pairing=initial_policy_state_pairing,
            summary=summary,
        )

    def _project_formal_training_completion(
        self,
        summary: Mapping[str, Any],
        *,
        train_spec: TrainJobSpec,
        argv: list[str],
        checkpoint: CheckpointRef,
    ) -> Mapping[str, Any]:
        """Recognize complete runner evidence without rewriting its source file.

        Explicit producer declarations retain their exact historical result.
        Missing declarations are projected only after the ordinary train path
        has validated identity, backend, ABI, binding and checkpoint bytes.
        Incomplete evidence remains unqualified at the existing Genesis gate.
        """
        if "formal_training_executed" in summary or summary.get("status") != "pass":
            return summary
        if "dry_run" in summary and summary["dry_run"] is not False:
            return summary
        training = summary.get("training")
        if not isinstance(training, Mapping):
            return summary
        try:
            expected = build_budget_accounting(
                budget_steps=train_spec.budget_steps,
                num_envs=int(unique_option_value(argv, "--num-envs")),
                num_steps_per_env=int(unique_option_value(argv, "--num-steps-per-env")),
                max_iterations=int(unique_option_value(argv, "--max-iterations")),
            )
            for name in ("budget_steps", "num_envs", "num_steps_per_env", "max_iterations"):
                if type(training.get(name)) is not int or training[name] != expected[name]:
                    return summary
            if training.get("budget_accounting") != expected:
                return summary
            final_iteration = training.get("latest_checkpoint_iteration")
            if type(final_iteration) is not int or final_iteration != expected["max_iterations"] - 1:
                return summary
            require_sha256_digest(training.get("initial_policy_state_hash"), "initial policy state hash")
        except (TypeError, ValueError):
            return summary
        source_path = self._summary_path(train_spec.metadata, job_id=train_spec.job_id)
        source_bytes = source_path.read_bytes()
        if strict_json_object_loads(source_bytes.decode("utf-8"), context="completed train summary") != dict(summary):
            raise RunnerExecutionError(f"{train_spec.job_id}: completed train summary changed during validation")
        return {
            **summary,
            "formal_training_executed": True,
            "runner_training_completion": {
                "schema_version": "true_rl_agent.runner.training_completion.v1",
                "source_summary_path": str(source_path),
                "source_summary_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
                "source_formal_training_declaration": "missing",
                "job_spec_hash": canonical_json_hash(train_spec.to_public_dict()),
                "training_binding_hash": train_spec.training_binding.binding_hash,
                "budget_accounting_hash": canonical_json_hash(expected),
                "final_checkpoint_iteration": final_iteration,
                "checkpoint_sha256": checkpoint.checksum,
            },
        }

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        return self._execute_with_attempts(
            eval_spec,
            job_kind="eval",
            operation=lambda log_dir, execution_context, timeout_s: (
                self._run_evaluation(
                    eval_spec,
                    process_log_dir=log_dir,
                    attempt_timeout_s=timeout_s,
                    execution_context=execution_context,
                )
            ),
            recover_completed=lambda execution_context: self._run_evaluation(
                eval_spec,
                process_log_dir=None,
                reuse_existing_summary=True,
                execution_context=execution_context,
            ),
        )

    def run_eval_phase(
        self,
        eval_spec: EvalPhaseJobSpec,
    ) -> EvaluationPhaseResult:
        return self._execute_with_attempts(
            eval_spec,
            job_kind="eval_phase",
            operation=lambda log_dir, execution_context, timeout_s: (
                self._run_evaluation_phase(
                    eval_spec,
                    process_log_dir=log_dir,
                    attempt_timeout_s=timeout_s,
                    execution_context=execution_context,
                )
            ),
            recover_completed=lambda execution_context: self._run_evaluation_phase(
                eval_spec,
                process_log_dir=None,
                reuse_existing_summary=True,
                execution_context=execution_context,
            ),
        )

    def project_assembled_eval_summary(
        self,
        eval_spec: EvalJobSpec,
        assembly: EvaluationAssembly,
    ) -> PublicEvalResult:
        """Project a selection- and receipt-bound task assembly."""

        if not isinstance(eval_spec, EvalJobSpec):
            raise TypeError("assembled eval summary requires an EvalJobSpec")
        if not isinstance(assembly, EvaluationAssembly):
            raise TypeError("assembled eval summary requires an EvaluationAssembly")
        expected_plan_hash = canonical_json_hash(eval_spec.to_public_dict())
        if assembly.evaluation_plan_hash != expected_plan_hash:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: assembled eval plan binding mismatch"
            )
        self._validate_assembled_phase_receipts(
            eval_spec=eval_spec,
            assembly=assembly,
        )
        summary = assembly.summary
        checkpoint = eval_spec.execution_checkpoint()
        try:
            validate_job_summary_identity(
                summary,
                expected=eval_spec.job_identity(),
                summary_name="assembled eval summary",
            )
            validate_result_status(
                summary.get("status"),
                field_name="assembled eval summary status",
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{eval_spec.job_id}: {exc}") from exc
        self._validate_backend_execution_environment_summary(summary, eval_spec)
        self._validate_execution_device_summary(summary, eval_spec)
        self._validate_eval_protocol_binding(summary, eval_spec)
        return self._public_result_from_summary(
            summary,
            job_id=eval_spec.job_id,
            expected_checkpoint_hash=checkpoint.checksum,
            expected_eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            expected_sealed_seed_count=eval_spec.sealed_validation_seed_count,
            evaluation_binding=eval_spec.evaluation_binding,
            evaluation_subject=eval_spec.subject.to_public_dict(),
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            expected_execution_device=eval_spec.execution_device,
        )

    def _validate_assembled_phase_receipts(
        self,
        *,
        eval_spec: EvalJobSpec,
        assembly: EvaluationAssembly,
    ) -> None:
        """Recheck the three immutable receipts named by the assembly token."""

        contract_hash: str | None = None
        try:
            for binding in assembly.phase_receipts:
                if (
                    binding.ref_schema_version
                    != EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION
                ):
                    raise ValueError("eval phase receipt ref schema mismatch")
                payload_bytes = read_bounded_regular_file(
                    Path(binding.path),
                    max_bytes=MAX_EVALUATION_PHASE_RECEIPT_BYTES,
                    label="assembled eval phase receipt",
                )
                if len(payload_bytes) != binding.size_bytes:
                    raise ValueError("eval phase receipt size mismatch")
                checksum = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
                if checksum != binding.checksum:
                    raise ValueError("eval phase receipt checksum mismatch")
                payload = strict_json_object_loads(
                    payload_bytes,
                    context="assembled eval phase receipt",
                )
                if payload_bytes != canonical_json_bytes(
                    payload,
                    trailing_newline=True,
                ):
                    raise ValueError("eval phase receipt is not canonical JSON")
                if set(payload) != {
                    "schema_version",
                    "contract_hash",
                    "job_spec_hash",
                    "phase_result",
                    "phase_result_hash",
                } or payload.get("schema_version") != (
                    EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION
                ):
                    raise ValueError("eval phase receipt contract mismatch")
                if contract_hash is None:
                    contract_hash = require_sha256_digest(
                        payload.get("contract_hash"),
                        "eval phase receipt contract_hash",
                    )
                elif payload.get("contract_hash") != contract_hash:
                    raise ValueError("assembled eval phase receipts mix contracts")
                phase_spec = EvalPhaseJobSpec(
                    job_id=binding.job_id,
                    evaluation_plan=eval_spec,
                    phase=binding.phase,
                )
                expected_job_spec_hash = canonical_json_hash(
                    phase_spec.to_public_dict()
                )
                if (
                    binding.job_spec_hash != expected_job_spec_hash
                    or payload.get("job_spec_hash") != expected_job_spec_hash
                ):
                    raise ValueError("eval phase receipt job spec mismatch")
                phase_payload = payload.get("phase_result")
                phase_result_hash = canonical_json_hash(phase_payload)
                if (
                    binding.phase_result_hash != phase_result_hash
                    or payload.get("phase_result_hash") != phase_result_hash
                ):
                    raise ValueError("eval phase receipt result hash mismatch")
                result = EvaluationPhaseResult.from_private_mapping(phase_payload)
                require_eval_phase_result_matches_spec(
                    result,
                    phase_spec,
                    expected_resource_binding_hash=binding.resource_binding_hash,
                )
        except (OSError, TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: assembled eval receipt binding failed: {exc}"
            ) from exc

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        return self._execute_with_attempts(
            eval_spec,
            job_kind="baseline_eval",
            operation=lambda log_dir, execution_context, timeout_s: (
                self._run_evaluation(
                    eval_spec,
                    process_log_dir=log_dir,
                    attempt_timeout_s=timeout_s,
                    execution_context=execution_context,
                )
            ),
            recover_completed=lambda execution_context: self._run_evaluation(
                eval_spec,
                process_log_dir=None,
                reuse_existing_summary=True,
                execution_context=execution_context,
            ),
        )

    def _run_evaluation(
        self,
        eval_spec: EvaluationJobSpec,
        *,
        process_log_dir: Path | None,
        attempt_timeout_s: int | None = None,
        reuse_existing_summary: bool = False,
        execution_context: _GpuJobExecutionContext | None = None,
    ) -> PublicEvalResult:
        try:
            checkpoint = eval_spec.execution_checkpoint()
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval checkpoint is not executable: {exc}"
            ) from exc
        argv = self._validated_argv(
            eval_spec.metadata,
            job_id=eval_spec.job_id,
        )
        try:
            validate_eval_job_argv(eval_spec, argv)
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval argv binding failed: {exc}"
            ) from exc
        completion = self._execute_summary_job(
            eval_spec.metadata,
            argv=argv,
            identity=eval_spec.job_identity(),
            summary_name="eval summary",
            reject_eval_candidate_paths=True,
            checkpoint=checkpoint,
            checkpoint_path_option=_EVAL_CHECKPOINT_OPTIONS[0],
            checkpoint_checksum_option=_EVAL_CHECKPOINT_OPTIONS[1],
            process_log_dir=process_log_dir,
            attempt_timeout_s=attempt_timeout_s,
            reuse_existing_summary=reuse_existing_summary,
            execution_context=execution_context,
        )
        summary = completion.summary
        self._validate_backend_execution_environment_summary(summary, eval_spec)
        self._validate_execution_device_summary(summary, eval_spec)
        self._validate_eval_protocol_binding(summary, eval_spec)
        return self._public_result_from_summary(
            summary,
            job_id=eval_spec.job_id,
            expected_checkpoint_hash=checkpoint.checksum,
            expected_eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            expected_sealed_seed_count=eval_spec.sealed_validation_seed_count,
            evaluation_binding=eval_spec.evaluation_binding,
            evaluation_subject=eval_spec.evaluation_subject().to_public_dict(),
            subject_manifest_hash=eval_spec.evaluation_manifest_hash(),
            program_source_checksum=(
                eval_spec.program_source_checksum
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            training_binding_hash=(
                eval_spec.training_binding_hash
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            capability_profile_hash=eval_spec.capability_profile_hash,
            expected_execution_device=eval_spec.execution_device,
        )

    def _run_evaluation_phase(
        self,
        eval_spec: EvalPhaseJobSpec,
        *,
        process_log_dir: Path | None,
        attempt_timeout_s: int | None = None,
        reuse_existing_summary: bool = False,
        execution_context: _GpuJobExecutionContext | None = None,
    ) -> EvaluationPhaseResult:
        try:
            checkpoint = eval_spec.execution_checkpoint()
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval checkpoint is not executable: {exc}"
            ) from exc
        argv = self._validated_argv(eval_spec.metadata, job_id=eval_spec.job_id)
        try:
            validate_eval_job_argv(eval_spec, argv)
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval phase argv binding failed: {exc}"
            ) from exc
        completion = self._execute_summary_job(
            eval_spec.metadata,
            argv=argv,
            identity=eval_spec.job_identity(),
            summary_name="eval phase summary",
            reject_eval_candidate_paths=True,
            checkpoint=checkpoint,
            checkpoint_path_option=_EVAL_CHECKPOINT_OPTIONS[0],
            checkpoint_checksum_option=_EVAL_CHECKPOINT_OPTIONS[1],
            process_log_dir=process_log_dir,
            attempt_timeout_s=attempt_timeout_s,
            reuse_existing_summary=reuse_existing_summary,
            execution_context=execution_context,
        )
        summary = completion.summary
        self._validate_backend_execution_environment_summary(summary, eval_spec)
        self._validate_execution_device_summary(summary, eval_spec)
        self._validate_eval_protocol_binding(summary, eval_spec)
        return self._phase_result_from_summary(
            summary,
            eval_spec=eval_spec,
            expected_checkpoint_hash=checkpoint.checksum,
            actual_gpu_process_binding=completion.gpu_process_binding,
        )

    def _execute_with_attempts(
        self,
        spec: TrainJobSpec | EvaluationJobSpec,
        *,
        job_kind: str,
        operation: Callable[
            [Path | None, _GpuJobExecutionContext | None, int],
            _ExecutionResultT,
        ],
        recover_completed: Callable[
            [_GpuJobExecutionContext | None],
            _ExecutionResultT,
        ]
        | None = None,
    ) -> _ExecutionResultT:
        scope = spec.attempt_scope
        gpu_process_binding = getattr(spec, "gpu_process_binding", None)
        if scope is None:
            if gpu_process_binding is not None:
                raise RunnerExecutionError(
                    f"{spec.job_id}: GPU process binding requires an attempt scope",
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                )
            return operation(None, None, self.timeout_s)
        if not isinstance(scope, JobAttemptExecutionScope):
            raise TypeError("job attempt scope is invalid")

        resource_binding_hash: str | None = None
        lease_execution_device: str | None = None
        execution_context: _GpuJobExecutionContext | None = None
        if spec.execution_device.startswith("cuda:"):
            if self.gpu_lease_manager is None:
                raise RunnerExecutionError(
                    f"{spec.job_id}: CUDA attempt scope requires a GPU lease manager",
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                )
            try:
                if gpu_process_binding is not None:
                    if not isinstance(gpu_process_binding, GpuProcessBinding):
                        raise ValueError("GPU process binding type is invalid")
                    if (
                        spec.execution_device
                        != gpu_process_binding.process_execution_device
                    ):
                        raise ValueError(
                            "process execution device does not match job spec"
                        )
                    lease_execution_device = gpu_process_binding.host_execution_device
                else:
                    lease_execution_device = spec.execution_device
                identity = self.gpu_lease_manager.identity_for(lease_execution_device)
                if (
                    gpu_process_binding is not None
                    and identity.identity_hash
                    != gpu_process_binding.resource_binding_hash
                ):
                    raise ValueError(
                        "GPU process binding does not match trusted device identity"
                    )
            except ValueError as exc:
                raise RunnerExecutionError(
                    f"{spec.job_id}: GPU resource binding is invalid",
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                ) from exc
            assert identity is not None
            resource_binding_hash = identity.identity_hash
            if gpu_process_binding is not None:
                if scope.resource_binding_hash != resource_binding_hash:
                    raise RunnerExecutionError(
                        f"{spec.job_id}: attempt scope GPU binding does not match "
                        "the trusted process binding",
                        failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                    )
                execution_context = _GpuJobExecutionContext(
                    gpu_process_binding=gpu_process_binding,
                    physical_id=identity.physical_id,
                )
        elif gpu_process_binding is not None:
            raise RunnerExecutionError(
                f"{spec.job_id}: CPU job cannot declare a GPU process binding",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID,
            )
        elif scope.resource_binding_hash is not None:
            raise RunnerExecutionError(
                f"{spec.job_id}: CPU job cannot declare a GPU resource binding",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID,
            )

        try:
            journal = JobAttemptJournal(
                scope=scope,
                logical_job_id=spec.job_id,
                job_kind=job_kind,
                job_input_hash=self._job_input_hash(spec, job_kind=job_kind),
                environment_hash=self._job_environment_hash(spec),
                execution_device=spec.execution_device,
                resource_binding_hash=resource_binding_hash,
            )
            reusable_result = journal.load_reusable_output_result()
            if reusable_result is not None:
                if recover_completed is None:
                    raise ValueError(
                        "completed attempt output has no typed recovery projector"
                    )
                recovered = recover_completed(execution_context)
                if self._attempt_output_hash(recovered) != reusable_result.payload.get(
                    "output_hash"
                ):
                    raise ValueError("completed attempt output hash mismatch")
                return recovered
            attempt_index, previous_result = journal.next_runnable_attempt()
        except JobAttemptTerminalResultError as exc:
            payload = exc.result.payload
            try:
                if (
                    payload.get("outcome") != "infrastructure_failure"
                    or payload.get("retryable") is not False
                    or payload.get("output_hash") is not None
                ):
                    raise ValueError(
                        "terminal attempt is not a reusable infrastructure failure"
                    )
                failure_kind = RunnerFailureKind(payload.get("failure_kind"))
                terminal_error = RunnerExecutionError(
                    f"{spec.job_id}: prior terminal infrastructure failure",
                    failure_kind=failure_kind,
                )
                if not terminal_error.retryable_infrastructure_failure:
                    raise ValueError(
                        "terminal infrastructure failure kind is not resumable"
                    )
            except (TypeError, ValueError) as validation_error:
                raise RunnerExecutionError(
                    f"{spec.job_id}: attempt evidence contract is invalid",
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                ) from validation_error
            raise terminal_error from exc
        except JobAttemptRecoveryRequiredError as exc:
            raise RunnerExecutionError(
                f"{spec.job_id}: requires recovery inspection",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID,
            ) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{spec.job_id}: attempt evidence contract is invalid",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID,
            ) from exc
        attempt_timeout_s = self._admit_attempt(
            scope=scope,
            logical_job_id=spec.job_id,
            job_kind=job_kind,
            attempt_index=attempt_index,
        )
        try:
            reservation = journal.reserve(
                attempt_index,
                previous_result=previous_result,
            )
            if not reservation.created:
                context = "initial attempt" if attempt_index == 0 else "retry"
                raise ValueError(f"{context} reservation raced another writer")
        except (OSError, TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{spec.job_id}: attempt evidence contract is invalid",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID,
            ) from exc
        lease_context = nullcontext(None)
        if resource_binding_hash is not None:
            assert self.gpu_lease_manager is not None
            assert lease_execution_device is not None
            if isinstance(scope, PreGenesisExecutionScope):
                holder = PreGenesisGpuLeaseHolder(
                    session_id=scope.session_id,
                    genesis_identity_hash=scope.genesis_identity_hash,
                    session_contract_hash=scope.session_contract_hash,
                    lane_id=scope.lane_id,
                    invocation_segment_id=scope.invocation_segment_id,
                    logical_job_id=spec.job_id,
                )
            else:
                assert isinstance(scope, JobAttemptScope)
                holder = GpuLeaseHolder(
                    session_genesis_hash=scope.session_genesis_hash,
                    generation_id=scope.generation_id,
                    lane_id=scope.lane_id,
                    invocation_segment_id=scope.invocation_segment_id,
                    logical_job_id=spec.job_id,
                )
            try:
                lease_context = self.gpu_lease_manager.acquire(
                    lease_execution_device,
                    holder=holder,
                )
            except GpuLeaseUnavailableError as exc:
                journal.finish(
                    reservation,
                    outcome="infrastructure_failure",
                    failure_kind=RunnerFailureKind.RESOURCE_UNAVAILABLE.value,
                    retryable=False,
                    output_hash=None,
                    returncode=None,
                )
                raise RunnerExecutionError(
                    f"{spec.job_id}: GPU resource is unavailable",
                    failure_kind=RunnerFailureKind.RESOURCE_UNAVAILABLE,
                ) from exc

        with lease_context:
            for current_attempt_index in range(
                attempt_index,
                scope.max_infrastructure_retries + 1,
            ):
                if current_attempt_index > attempt_index:
                    assert previous_result is not None
                    attempt_timeout_s = self._admit_attempt(
                        scope=scope,
                        logical_job_id=spec.job_id,
                        job_kind=job_kind,
                        attempt_index=current_attempt_index,
                    )
                    try:
                        reservation = journal.reserve(
                            current_attempt_index,
                            previous_result=previous_result,
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        raise RunnerExecutionError(
                            f"{spec.job_id}: retry evidence contract is invalid",
                            failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                        ) from exc
                    if not reservation.created:
                        raise RunnerExecutionError(
                            f"{spec.job_id}: existing retry requires recovery inspection",
                            failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                        )
                log_dir = journal.root / f"attempt_{current_attempt_index:02d}.logs"
                lease_wall_started_ns = (
                    time.perf_counter_ns()
                    if resource_binding_hash is not None
                    else None
                )
                try:
                    result = operation(
                        log_dir,
                        execution_context,
                        attempt_timeout_s,
                    )
                except CandidateRuntimeError as exc:
                    journal.finish(
                        reservation,
                        outcome="candidate_failure",
                        failure_kind=type(exc).__name__,
                        retryable=False,
                        output_hash=None,
                        returncode=None,
                        resource_lease_wall_duration_ns=(
                            self._resource_lease_wall_duration_ns(lease_wall_started_ns)
                        ),
                    )
                    raise
                except RunnerExecutionError as exc:
                    retryable = (
                        exc.retryable_infrastructure_failure
                        and current_attempt_index < scope.max_infrastructure_retries
                    )
                    outcome = self._attempt_error_outcome(exc)
                    previous_result = journal.finish(
                        reservation,
                        outcome=outcome,
                        failure_kind=exc.failure_kind.value,
                        retryable=retryable,
                        output_hash=None,
                        returncode=None,
                        resource_lease_wall_duration_ns=(
                            self._resource_lease_wall_duration_ns(lease_wall_started_ns)
                        ),
                    )
                    if retryable:
                        continue
                    raise
                except KeyboardInterrupt:
                    journal.finish(
                        reservation,
                        outcome="interrupted",
                        failure_kind=RunnerFailureKind.INTERRUPTED.value,
                        retryable=False,
                        output_hash=None,
                        returncode=None,
                        resource_lease_wall_duration_ns=(
                            self._resource_lease_wall_duration_ns(lease_wall_started_ns)
                        ),
                    )
                    raise
                except BaseException as exc:
                    journal.finish(
                        reservation,
                        outcome="unclassified_failure",
                        failure_kind=type(exc).__name__,
                        retryable=False,
                        output_hash=None,
                        returncode=None,
                        resource_lease_wall_duration_ns=(
                            self._resource_lease_wall_duration_ns(lease_wall_started_ns)
                        ),
                    )
                    raise
                outcome = (
                    "scientific_failure"
                    if isinstance(result, PublicEvalResult) and result.status == "fail"
                    else "completed"
                )
                journal.finish(
                    reservation,
                    outcome=outcome,
                    failure_kind=None,
                    retryable=False,
                    output_hash=self._attempt_output_hash(result),
                    returncode=None,
                    resource_lease_wall_duration_ns=(
                        self._resource_lease_wall_duration_ns(lease_wall_started_ns)
                    ),
                )
                return result
        raise RuntimeError("attempt loop exited without a terminal result")

    def _admit_attempt(
        self,
        *,
        scope: JobAttemptExecutionScope,
        logical_job_id: str,
        job_kind: str,
        attempt_index: int,
    ) -> int:
        hook = self.attempt_admission_hook
        if hook is None:
            return self.timeout_s
        timeout_s = hook(scope, logical_job_id, job_kind, attempt_index)
        if timeout_s is None:
            return self.timeout_s
        require_positive_integer(timeout_s, "attempt admission timeout_s")
        return min(self.timeout_s, timeout_s)

    @staticmethod
    def _attempt_output_hash(result: _ExecutionResultT) -> str:
        return canonical_json_hash(
            result.to_private_dict()
            if isinstance(result, EvaluationPhaseResult)
            else result.to_public_dict()
        )

    @staticmethod
    def _resource_lease_wall_duration_ns(started_ns: int | None) -> int | None:
        if started_ns is None:
            return None
        return max(1, time.perf_counter_ns() - started_ns)

    @staticmethod
    def _attempt_error_outcome(error: RunnerExecutionError) -> str:
        if error.failure_kind in {
            RunnerFailureKind.EXECUTABLE_NOT_FOUND,
            RunnerFailureKind.CONTRACT_INVALID,
            RunnerFailureKind.SUMMARY_INVALID,
        }:
            return "contract_failure"
        if error.failure_kind == RunnerFailureKind.INTERRUPTED:
            return "interrupted"
        return "infrastructure_failure"

    @staticmethod
    def _job_input_hash(
        spec: TrainJobSpec | EvaluationJobSpec,
        *,
        job_kind: str,
    ) -> str:
        metadata = spec.metadata
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.runner.logical_job_input.v1",
                "job_kind": job_kind,
                "job_spec": spec.to_public_dict(),
                "cli_argv": metadata.get("cli_argv"),
                "summary_path": metadata.get("summary_path"),
                "checkpoint_output_path": metadata.get("checkpoint_output_path"),
                "work_dir": metadata.get("work_dir"),
            }
        )

    @staticmethod
    def _job_environment_hash(spec: TrainJobSpec | EvaluationJobSpec) -> str:
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.runner.job_environment.v1",
                "backend_id": spec.metadata.get("backend_id"),
                "backend_execution_environment_manifest_hash": spec.metadata.get(
                    "backend_execution_environment_manifest_hash"
                ),
            }
        )

    def _validate_checkpoint_before_launch(
        self,
        *,
        checkpoint: CheckpointRef | None,
        metadata: Mapping[str, Any],
        job_id: str,
        argv: list[str],
        checkpoint_path_option: str,
        checkpoint_checksum_option: str,
    ) -> None:
        raw_path = (
            checkpoint.metadata.get("local_path") if checkpoint is not None else None
        )
        resolved = (
            self._resolved_job_path(raw_path, metadata=metadata)
            if isinstance(raw_path, str) and raw_path and "://" not in raw_path
            else None
        )
        try:
            validate_checkpoint_process_input(
                checkpoint,
                argv=argv,
                checkpoint_path_option=checkpoint_path_option,
                checkpoint_checksum_option=checkpoint_checksum_option,
                resolved_local_path=resolved,
            )
            manifest_occurrences = option_occurrences(
                argv,
                "--resolved-cell-manifest-hash",
            )
            if checkpoint is not None and manifest_occurrences:
                expected_cell_hash = unique_option_value(
                    argv,
                    "--resolved-cell-manifest-hash",
                )
                manifest_path = unique_option_value(
                    argv,
                    "--resolved-cell-manifest-path",
                )
                manifest = load_resolved_cell_manifest(
                    manifest_path,
                    expected_manifest_hash=expected_cell_hash,
                )
                checkpoint_cell_hash = checkpoint.metadata.get(
                    "resolved_cell_manifest_hash"
                )
                predecessor_manifest_hash = metadata.get(
                    "training_predecessor_manifest_hash"
                )
                allowed_checkpoint_manifest_hashes = {expected_cell_hash}
                if predecessor_manifest_hash is not None:
                    allowed_checkpoint_manifest_hashes.add(
                        require_sha256_digest(
                            predecessor_manifest_hash,
                            "training predecessor manifest hash",
                        )
                    )
                if (
                    checkpoint_cell_hash is None
                    and not manifest.uses_legacy_training_binding_v1
                ):
                    raise ValueError(
                        "checkpoint resolved cell manifest hash is required"
                    )
                if (
                    self.eval_checkpoint_admission_hook is not None
                    and checkpoint_path_option == _EVAL_CHECKPOINT_OPTIONS[0]
                ):
                    # A composition-owned task policy validates cross-version
                    # admission; artifact metadata cannot extend its lineage.
                    self.eval_checkpoint_admission_hook(checkpoint, manifest, tuple(argv))
                    return
                if (
                    checkpoint_cell_hash is not None
                    and checkpoint_cell_hash
                    not in allowed_checkpoint_manifest_hashes
                ):
                    raise ValueError("checkpoint resolved cell manifest hash mismatch")
        except ValueError as exc:
            raise RunnerExecutionError(
                f"{job_id}: checkpoint preflight failed: {exc}"
            ) from exc

    def _execute_summary_job(
        self,
        metadata: Mapping[str, Any],
        *,
        argv: list[str],
        identity: JobIdentity,
        summary_name: str,
        reject_eval_candidate_paths: bool = False,
        stale_output_paths: tuple[tuple[str, Path], ...] = (),
        checkpoint: CheckpointRef | None,
        checkpoint_path_option: str,
        checkpoint_checksum_option: str,
        process_log_dir: Path | None,
        attempt_timeout_s: int | None = None,
        reuse_existing_summary: bool = False,
        execution_context: _GpuJobExecutionContext | None = None,
        reused_gpu_process_binding: GpuProcessBinding | None = None,
    ) -> _SummaryJobCompletion:
        """Interpret process exit and a fresh structured summary as one contract."""

        self._validate_checkpoint_before_launch(
            checkpoint=checkpoint,
            metadata=metadata,
            job_id=identity.job_id,
            argv=argv,
            checkpoint_path_option=checkpoint_path_option,
            checkpoint_checksum_option=checkpoint_checksum_option,
        )
        if reject_eval_candidate_paths:
            try:
                reject_candidate_path_arguments(argv)
            except ValueError as exc:
                raise RunnerExecutionError(f"{identity.job_id}: {exc}") from exc
        if reuse_existing_summary:
            # Recovery validates the original job's binding without acquiring a
            # new lease or pretending its completed process is being launched.
            expected_binding = (
                execution_context.gpu_process_binding if execution_context is not None
                else reused_gpu_process_binding
            )
            try:
                summary = self._read_summary(metadata, job_id=identity.job_id)
                validate_job_summary_identity(
                    summary,
                    expected=identity,
                    summary_name=summary_name,
                )
                self._validate_resolved_cell_summary(
                    summary,
                    argv=argv,
                    job_id=identity.job_id,
                )
                status = validate_result_status(
                    summary.get("status"),
                    field_name=f"{summary_name} status",
                )
                actual_gpu_process_binding = (
                    self._validated_gpu_process_binding_from_summary(
                        summary,
                        expected_binding=expected_binding,
                    )
                )
                owned_process_identity = (
                    self._validated_owned_process_identity_from_summary(
                        summary,
                        expected_binding=expected_binding,
                        expected_process_id=None,
                    )
                )
            except (RunnerExecutionError, ValueError) as exc:
                raise RunnerExecutionError(
                    f"{identity.job_id}: reusable {summary_name} is invalid: {exc}",
                    failure_kind=RunnerFailureKind.SUMMARY_INVALID,
                ) from exc
            return _SummaryJobCompletion(
                summary=summary,
                status=status,
                returncode=0,
                gpu_process_binding=actual_gpu_process_binding,
                owned_process_identity=owned_process_identity,
            )
        self._clear_existing_summary(metadata, job_id=identity.job_id)
        for output_name, output_path in stale_output_paths:
            self._clear_existing_output(
                output_path,
                output_name=output_name,
                job_id=identity.job_id,
            )
        completed = self._run_job(
            argv,
            metadata,
            job_id=identity.job_id,
            process_log_dir=process_log_dir,
            attempt_timeout_s=attempt_timeout_s,
            execution_context=execution_context,
        )
        try:
            summary = self._read_summary(metadata, job_id=identity.job_id)
            validate_job_summary_identity(
                summary,
                expected=identity,
                summary_name=summary_name,
            )
            self._validate_resolved_cell_summary(
                summary,
                argv=argv,
                job_id=identity.job_id,
            )
            status = validate_result_status(
                summary.get("status"),
                field_name=f"{summary_name} status",
            )
            actual_gpu_process_binding = (
                self._validated_gpu_process_binding_from_summary(
                    summary,
                    expected_binding=(execution_context.gpu_process_binding if execution_context is not None else None),
                )
            )
            owned_process_identity = (
                self._validated_owned_process_identity_from_summary(
                    summary,
                    expected_binding=(execution_context.gpu_process_binding if execution_context is not None else None),
                    expected_process_id=(completed.pid if os.name != "nt" else None),
                )
            )
        except (RunnerExecutionError, ValueError) as exc:
            failure_kind = (
                exc.failure_kind
                if isinstance(exc, RunnerExecutionError)
                else RunnerFailureKind.SUMMARY_INVALID
            )
            error = self._summary_contract_error(
                job_id=identity.job_id,
                completed=completed,
                summary_error=str(exc),
                failure_kind=failure_kind,
            )
            raise error from exc
        if completed.returncode != 0 and status != "fail":
            raise self._summary_contract_error(
                job_id=identity.job_id,
                completed=completed,
                summary_error=(
                    f"{summary_name} status is pass but subprocess exited nonzero"
                ),
                failure_kind=RunnerFailureKind.SUMMARY_INVALID,
            )
        return _SummaryJobCompletion(
            summary=summary,
            status=status,
            returncode=int(completed.returncode),
            gpu_process_binding=actual_gpu_process_binding,
            owned_process_identity=owned_process_identity,
        )

    @staticmethod
    def _validated_gpu_process_binding_from_summary(
        summary: Mapping[str, Any],
        *,
        expected_binding: GpuProcessBinding | None,
    ) -> GpuProcessBinding | None:
        if expected_binding is None:
            if summary.get("gpu_process_binding") is not None:
                raise ValueError("unbound summary cannot claim a GPU process binding")
            return None
        actual = GpuProcessBinding.from_mapping(summary.get("gpu_process_binding"))
        if actual != expected_binding:
            raise ValueError(
                "summary GPU process binding does not match trusted job binding"
            )
        return actual

    @staticmethod
    def _validated_owned_process_identity_from_summary(
        summary: Mapping[str, Any],
        *,
        expected_binding: GpuProcessBinding | None,
        expected_process_id: int | None,
    ) -> Mapping[str, Any] | None:
        raw = summary.get("owned_process_identity")
        if expected_binding is None:
            if raw is not None:
                raise ValueError(
                    "unbound summary cannot claim a runner-owned process identity"
                )
            return None
        return validate_owned_process_identity(
            raw,
            expected_process_id=expected_process_id,
            require_process_group_leader=True,
        )

    def _run_job(
        self,
        argv: list[str],
        metadata: Mapping[str, Any],
        *,
        job_id: str,
        process_log_dir: Path | None,
        attempt_timeout_s: int | None,
        execution_context: _GpuJobExecutionContext | None,
    ) -> OwnedProcessResult:
        work_dir = self._work_dir(metadata)
        if process_log_dir is None:
            process_log_dir = (
                self._summary_path(metadata, job_id=job_id).parent / "runner_process"
            )
        timeout_s = self.timeout_s if attempt_timeout_s is None else attempt_timeout_s
        require_positive_integer(timeout_s, "attempt timeout_s")
        try:
            completed = run_owned_process(
                argv,
                cwd=work_dir,
                timeout_s=timeout_s,
                log_dir=process_log_dir,
                termination_grace_s=self.termination_grace_s,
                environment_overrides=(
                    execution_context.environment_overrides
                    if execution_context is not None
                    else None
                ),
            )
        except OwnedProcessCleanupIncomplete as exc:
            raise RunnerExecutionError(
                f"{job_id}: subprocess process-group cleanup remained incomplete",
                failure_kind=RunnerFailureKind.INTERRUPTED,
            ) from exc
        except OwnedProcessTimeout as exc:
            raise RunnerExecutionError(
                f"{job_id}: subprocess timed out after {timeout_s}s",
                failure_kind=RunnerFailureKind.TIMEOUT,
            ) from exc
        except OwnedProcessLaunchError as exc:
            cause = exc.__cause__
            if isinstance(cause, FileNotFoundError):
                raise RunnerExecutionError(
                    f"{job_id}: executable not found: {argv[0]}",
                    failure_kind=RunnerFailureKind.EXECUTABLE_NOT_FOUND,
                ) from exc
            failure_kind = (
                RunnerFailureKind.LAUNCH_TRANSIENT
                if self._is_transient_launch_error(cause)
                else RunnerFailureKind.LAUNCH_FAILED
            )
            raise RunnerExecutionError(
                f"{job_id}: subprocess launch failed: "
                f"{type(cause).__name__ if cause is not None else type(exc).__name__}",
                failure_kind=failure_kind,
            ) from exc
        return completed

    def _summary_contract_error(
        self,
        *,
        job_id: str,
        completed: OwnedProcessResult,
        summary_error: str,
        failure_kind: RunnerFailureKind,
    ) -> RunnerExecutionError:
        job_prefix = f"{job_id}: "
        if summary_error.startswith(job_prefix):
            summary_error = summary_error[len(job_prefix) :]
        process_context = ""
        if completed.returncode != 0:
            process_context = (
                f"subprocess exited {completed.returncode}; "
                f"stdout={self._safe_output(completed.stdout_tail)}; "
                f"stderr={self._safe_output(completed.stderr_tail)}; "
            )
        return RunnerExecutionError(
            f"{job_id}: {process_context}structured summary invalid: {summary_error}",
            failure_kind=failure_kind,
        )

    @staticmethod
    def _is_transient_launch_error(error: BaseException | None) -> bool:
        if not isinstance(error, OSError):
            return False
        transient_errno = {11, 12, 16, 26}
        transient_winerror = {8, 14, 1450}
        return (
            error.errno in transient_errno
            or getattr(
                error,
                "winerror",
                None,
            )
            in transient_winerror
        )

    def _validated_argv(self, metadata: Mapping[str, Any], *, job_id: str) -> list[str]:
        argv = metadata.get("cli_argv")
        if not isinstance(argv, list):
            raise RunnerExecutionError(f"{job_id}: cli_argv must be list[str]")
        if not argv:
            raise RunnerExecutionError(f"{job_id}: cli_argv must not be empty")
        if not all(isinstance(part, str) for part in argv):
            raise RunnerExecutionError(f"{job_id}: cli_argv entries must be strings")
        executable = argv[0]
        if executable not in self.allowed_executables:
            raise RunnerExecutionError(
                f"{job_id}: executable is not allowlisted: {executable}"
            )
        cli_command = metadata.get("cli_command")
        if cli_command is not None:
            if not isinstance(cli_command, str) or cli_command != " ".join(argv):
                raise RunnerExecutionError(
                    f"{job_id}: cli_command must be derived from cli_argv"
                )
        return list(argv)

    def _work_dir(self, metadata: Mapping[str, Any]) -> Path | None:
        raw_work_dir = metadata.get("work_dir")
        if raw_work_dir is None:
            return self.cwd
        if not isinstance(raw_work_dir, str):
            raise RunnerExecutionError("work_dir must be a string path")
        return Path(raw_work_dir)

    def _resolved_job_path(
        self,
        raw_path: str,
        *,
        metadata: Mapping[str, Any],
    ) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return Path(os.path.abspath(path))
        work_dir = self._work_dir(metadata)
        base = Path.cwd() if work_dir is None else work_dir
        if not base.is_absolute():
            base = Path.cwd() / base
        return Path(os.path.abspath(base / path))

    def _read_summary(
        self, metadata: Mapping[str, Any], *, job_id: str
    ) -> dict[str, Any]:
        summary_path = self._summary_path(metadata, job_id=job_id)
        if summary_path.is_symlink():
            raise RunnerExecutionError(
                f"{job_id}: summary_path must not be a symlink: {summary_path}",
                failure_kind=RunnerFailureKind.SUMMARY_INVALID,
            )
        if not summary_path.exists():
            raise RunnerExecutionError(
                f"{job_id}: summary file was not created: {summary_path}",
                failure_kind=RunnerFailureKind.SUMMARY_MISSING,
            )
        try:
            summary_text = summary_path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file is not valid UTF-8: {summary_path}",
                failure_kind=RunnerFailureKind.SUMMARY_INVALID,
            ) from exc
        except OSError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file could not be read: {summary_path}",
                failure_kind=RunnerFailureKind.SUMMARY_INVALID,
            ) from exc
        try:
            return strict_json_object_loads(
                summary_text,
                context=f"{job_id} summary file",
            )
        except ValueError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file is not valid JSON: {summary_path}",
                failure_kind=RunnerFailureKind.SUMMARY_INVALID,
            ) from exc

    @staticmethod
    def _validate_resolved_cell_summary(
        summary: Mapping[str, Any],
        *,
        argv: list[str],
        job_id: str,
    ) -> None:
        path_occurrences = option_occurrences(
            argv,
            "--resolved-cell-manifest-path",
        )
        hash_occurrences = option_occurrences(
            argv,
            "--resolved-cell-manifest-hash",
        )
        if not path_occurrences and not hash_occurrences:
            if (
                summary.get("resolved_cell_manifest_hash") is not None
                or summary.get("resolved_cell_identity") is not None
            ):
                raise ValueError(
                    f"{job_id}: unbound summary claims resolved cell identity"
                )
            return
        path = unique_option_value(argv, "--resolved-cell-manifest-path")
        manifest_hash = unique_option_value(
            argv,
            "--resolved-cell-manifest-hash",
        )
        manifest = load_resolved_cell_manifest(
            path,
            expected_manifest_hash=manifest_hash,
        )
        if summary.get("resolved_cell_manifest_hash") != manifest_hash:
            raise ValueError(f"{job_id}: summary resolved cell manifest hash mismatch")
        identity = summary.get("resolved_cell_identity")
        if not isinstance(identity, Mapping) or dict(identity) != (
            manifest.summary_identity()
        ):
            raise ValueError(f"{job_id}: summary resolved cell identity mismatch")

    def _summary_path(self, metadata: Mapping[str, Any], *, job_id: str) -> Path:
        raw_summary_path = metadata.get("summary_path")
        if not isinstance(raw_summary_path, str) or not raw_summary_path:
            raise RunnerExecutionError(f"{job_id}: summary_path is required")
        return self._resolved_job_path(raw_summary_path, metadata=metadata)

    def _clear_existing_summary(
        self, metadata: Mapping[str, Any], *, job_id: str
    ) -> None:
        summary_path = self._summary_path(metadata, job_id=job_id)
        if summary_path.is_symlink():
            raise RunnerExecutionError(
                f"{job_id}: summary_path must not be a symlink: {summary_path}"
            )
        if not summary_path.exists():
            return
        if not summary_path.is_file():
            raise RunnerExecutionError(
                f"{job_id}: summary_path exists but is not a file: {summary_path}"
            )
        try:
            summary_path.unlink()
        except OSError as exc:
            raise RunnerExecutionError(
                f"{job_id}: failed to remove stale summary file: {summary_path}"
            ) from exc

    def _clear_existing_output(
        self,
        path: Path,
        *,
        output_name: str,
        job_id: str,
    ) -> None:
        if path.is_symlink():
            raise RunnerExecutionError(
                f"{job_id}: {output_name} must not be a symlink: {path}"
            )
        if not path.exists():
            return
        if not path.is_file():
            raise RunnerExecutionError(
                f"{job_id}: {output_name} exists but is not a file: {path}"
            )
        try:
            path.unlink()
        except OSError as exc:
            raise RunnerExecutionError(
                f"{job_id}: failed to remove stale {output_name}: {path}"
            ) from exc

    def _required_local_output_path(
        self,
        metadata: Mapping[str, Any],
        *,
        key: str,
        job_id: str,
    ) -> Path:
        raw_path = metadata.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            raise RunnerExecutionError(
                f"{job_id}: {key} is required for local subprocess train"
            )
        if "://" in raw_path:
            raise RunnerExecutionError(f"{job_id}: {key} must be a local file path")
        return self._resolved_job_path(raw_path, metadata=metadata)

    def _validate_eval_protocol_binding(
        self,
        summary: Mapping[str, Any],
        eval_spec: EvaluationJobSpec,
    ) -> None:
        protocol = summary.get("evaluation_protocol")
        if not isinstance(protocol, Mapping):
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval summary missing evaluation_protocol object"
            )
        expected_protocol = eval_spec.evaluation_protocol().to_public_dict()
        for key, expected in expected_protocol.items():
            actual = protocol.get(key)
            if isinstance(expected, int):
                matches = (
                    isinstance(actual, int)
                    and not isinstance(actual, bool)
                    and actual == expected
                )
            else:
                matches = actual == expected
            if not matches:
                raise RunnerExecutionError(
                    f"{eval_spec.job_id}: eval summary binding mismatch for evaluation_protocol.{key}: "
                    f"expected={expected!r} actual={actual!r}"
                )
        try:
            actual_binding = EvaluationBindingV1.from_mapping(
                summary.get("evaluation_binding")
            )
        except ValueError as exc:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: invalid evaluation binding: {exc}"
            ) from exc
        if actual_binding != eval_spec.evaluation_binding:
            raise RunnerExecutionError(
                f"{eval_spec.job_id}: eval summary evaluation binding mismatch"
            )

    def _validate_train_checkpoint_output(
        self,
        checkpoint: CheckpointRef,
        *,
        expected_path: Path,
        metadata: Mapping[str, Any],
        job_id: str,
    ) -> CheckpointRef:
        raw_declared_path = checkpoint.metadata.get("local_path", checkpoint.uri)
        if (
            not isinstance(raw_declared_path, str)
            or not raw_declared_path
            or "://" in raw_declared_path
        ):
            raise RunnerExecutionError(
                f"{job_id}: train summary checkpoint must declare a local output path"
            )
        declared_path = self._resolved_job_path(raw_declared_path, metadata=metadata)
        try:
            return validate_train_checkpoint_artifact(
                checkpoint,
                expected_path=expected_path,
                declared_path=declared_path,
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{job_id}: {exc}") from exc

    def _raise_failed_train_summary(
        self,
        completion: _SummaryJobCompletion,
        *,
        train_spec: TrainJobSpec,
    ) -> None:
        if completion.status != "fail":
            return
        job_id = train_spec.job_id
        summary = completion.summary
        if completion.returncode == 0:
            raise RunnerExecutionError(
                f"{job_id}: failed train summary requires a nonzero subprocess exit"
            )
        if train_spec.subject.kind != "candidate":
            raise RunnerExecutionError(
                f"{job_id}: reference training failure is infrastructure-owned"
            )
        contract = train_spec.candidate_failure_contract
        if contract is None:
            raise RunnerExecutionError(
                f"{job_id}: train failure is not bound to a candidate failure contract"
            )
        if summary.get("errors") not in (None, []):
            raise RunnerExecutionError(
                f"{job_id}: candidate failure summary cannot contain raw errors"
            )
        if summary.get("checkpoint") is not None:
            raise RunnerExecutionError(
                f"{job_id}: candidate failure summary cannot contain a checkpoint"
            )
        try:
            evidence = contract.validate_evidence(summary.get("candidate_failure"))
            rule = contract.rule_for(
                stage=evidence.failure_stage,
                reason_code=evidence.reason_code,
            )
            if rule.requires_initial_policy_state:
                self._initial_policy_state_pairing(
                    summary,
                    train_spec=train_spec,
                )
            else:
                self._validate_preinitial_policy_failure(
                    summary,
                    train_spec=train_spec,
                )
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{job_id}: invalid candidate failure evidence: {exc}"
            ) from exc
        details: dict[str, Any] = {
            "status": "fail",
            "subprocess_returncode": completion.returncode,
            "candidate_failure_contract": contract.to_public_dict(),
            "candidate_failure": evidence.to_public_dict(),
        }
        training = summary.get("training")
        if isinstance(training, Mapping) and "numeric_evidence" in training:
            from true_rl_agent.evaluation.public_projection import project_public_training_sampling
            details["training_sampling"] = project_public_training_sampling(training["numeric_evidence"])
        raise CandidateRuntimeError(
            f"{job_id}: candidate program failed during "
            f"{evidence.failure_stage} ({evidence.reason_code})",
            public_details=details,
        )

    @staticmethod
    def _validate_preinitial_policy_failure(
        summary: Mapping[str, Any],
        *,
        train_spec: TrainJobSpec,
    ) -> None:
        training = summary.get("training")
        policy_fields = {
            "initial_policy_state_hash",
            "initial_policy_state_match",
        }
        if isinstance(training, Mapping) and policy_fields.intersection(training):
            raise ValueError(
                "pre-initialization candidate failure cannot claim actual policy state"
            )
        expected = train_spec.expected_initial_policy_state_hash
        if expected is None:
            if isinstance(training, Mapping) and (
                "expected_initial_policy_state_hash" in training
            ):
                raise ValueError(
                    "candidate failure unexpectedly declares an initial policy anchor"
                )
            return
        if not isinstance(training, Mapping):
            raise ValueError(
                "candidate failure is missing the expected initial policy anchor"
            )
        if training.get("expected_initial_policy_state_hash") != expected:
            raise ValueError(
                "candidate failure expected initial policy state hash mismatch"
            )

    def _checkpoint_from_summary(
        self, summary: Mapping[str, Any], *, job_id: str
    ) -> CheckpointRef:
        checkpoint = summary.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RunnerExecutionError(
                f"{job_id}: train summary missing checkpoint object"
            )
        try:
            parsed = CheckpointRef(
                uri=checkpoint["uri"],
                checksum=checkpoint["checksum"],
                role=checkpoint["role"],
                sealed=checkpoint.get("sealed", False),
                metadata=checkpoint.get("metadata", {}),
            )
            if parsed.role != "train_output":
                raise ValueError("train summary checkpoint role must be train_output")
            if parsed.sealed:
                raise ValueError(
                    "train summary checkpoint must be unsealed before runner sealing"
                )
            return parsed
        except KeyError as exc:
            raise RunnerExecutionError(
                f"{job_id}: train summary checkpoint missing {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{job_id}: invalid train summary checkpoint: {exc}"
            ) from exc

    def _phase_result_from_summary(
        self,
        summary: Mapping[str, Any],
        *,
        eval_spec: EvalPhaseJobSpec,
        expected_checkpoint_hash: str,
        actual_gpu_process_binding: GpuProcessBinding | None = None,
    ) -> EvaluationPhaseResult:
        job_id = eval_spec.job_id
        if summary.get("schema_version") != EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION:
            raise RunnerExecutionError(f"{job_id}: eval phase summary schema mismatch")
        if summary.get("status") != "pass" or summary.get("errors") != []:
            raise RunnerExecutionError(
                f"{job_id}: eval phase did not produce complete evidence"
            )
        if summary.get("evaluation_phase") != eval_spec.phase.value:
            raise RunnerExecutionError(
                f"{job_id}: eval phase summary identity mismatch"
            )
        if summary.get("evaluation_plan_hash") != eval_spec.evaluation_plan_hash:
            raise RunnerExecutionError(f"{job_id}: eval phase plan binding mismatch")
        task_evidence = summary.get("task_evidence")
        if not isinstance(task_evidence, Mapping):
            raise RunnerExecutionError(
                f"{job_id}: eval phase task evidence must be an object"
            )
        if canonical_json_hash(task_evidence) != summary.get("task_evidence_hash"):
            raise RunnerExecutionError(
                f"{job_id}: eval phase task evidence hash mismatch"
            )
        execution_cost = summary.get("evaluation_execution_cost")
        source_preflight = summary.get("environment_source_preflight")
        environment_verification = summary.get(
            "backend_execution_environment_verification"
        )
        for field_name, value in (
            ("evaluation_execution_cost", execution_cost),
            ("environment_source_preflight", source_preflight),
            (
                "backend_execution_environment_verification",
                environment_verification,
            ),
        ):
            if not isinstance(value, Mapping):
                raise RunnerExecutionError(
                    f"{job_id}: eval phase {field_name} must be an object"
                )
        try:
            reject_candidate_contamination_flags(summary, source="eval phase summary")
            checkpoint_safety_report = validate_checkpoint_safety_report(
                summary.get("checkpoint_safety_report"),
                expected_checkpoint_hash=expected_checkpoint_hash,
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{job_id}: {exc}") from exc
        return EvaluationPhaseResult(
            job_id=job_id,
            task_id=eval_spec.task_id,
            phase=eval_spec.phase,
            evaluation_plan_hash=eval_spec.evaluation_plan_hash,
            subject=eval_spec.subject,
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            checkpoint_hash=expected_checkpoint_hash,
            eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            execution_device=(
                actual_gpu_process_binding.process_execution_device
                if actual_gpu_process_binding is not None
                else eval_spec.execution_device
            ),
            resource_binding_hash=(
                actual_gpu_process_binding.resource_binding_hash
                if actual_gpu_process_binding is not None
                else (
                    eval_spec.attempt_scope.resource_binding_hash
                    if eval_spec.attempt_scope is not None
                    else None
                )
            ),
            task_evidence=task_evidence,
            execution_cost=execution_cost,
            environment_source_preflight=source_preflight,
            backend_execution_environment_verification=environment_verification,
            checkpoint_safety_report=checkpoint_safety_report,
            gpu_process_binding=(
                actual_gpu_process_binding.to_dict()
                if actual_gpu_process_binding is not None
                else None
            ),
        )

    def _public_result_from_summary(
        self,
        summary: Mapping[str, Any],
        *,
        job_id: str,
        expected_checkpoint_hash: str,
        expected_eval_protocol_hash: str,
        expected_evaluation_binding_hash: str,
        expected_sealed_seed_count: int,
        evaluation_binding: EvaluationBindingV1,
        evaluation_subject: Mapping[str, Any],
        subject_manifest_hash: str | None,
        program_source_checksum: str | None,
        training_binding_hash: str | None,
        capability_profile_hash: str,
        expected_execution_device: str,
    ) -> PublicEvalResult:
        try:
            fields = validate_public_eval_result_fields(
                status=summary.get("status"),
                public_score=summary.get("public_score"),
                metrics=summary.get("metrics"),
                field_name="eval summary",
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{job_id}: {exc}") from exc
        status = fields.status
        public_score = fields.public_score
        metrics = fields.metrics
        reserved_metric_collisions = sorted(
            set(metrics)
            & (
                (
                    RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
                    - frozenset(CANDIDATE_CONTAMINATION_FIELDS)
                )
                | frozenset(ALTERNATE_JOB_IDENTITY_FIELDS)
            )
        )
        if reserved_metric_collisions:
            raise RunnerExecutionError(
                f"{job_id}: eval summary metrics must not contain "
                f"{reserved_metric_collisions[0]}"
            )
        try:
            reject_candidate_contamination_flags(metrics, source="eval summary metrics")
            reject_candidate_contamination_flags(summary, source="eval summary")
            checkpoint_safety_report = validate_checkpoint_safety_report(
                summary.get("checkpoint_safety_report"),
                expected_checkpoint_hash=expected_checkpoint_hash,
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{job_id}: {exc}") from exc
        public_metrics = dict(metrics)
        public_metrics["evaluation_binding"] = evaluation_binding.to_public_dict()
        public_metrics["evaluation_subject"] = dict(evaluation_subject)
        if subject_manifest_hash is not None:
            public_metrics["subject_manifest_hash"] = subject_manifest_hash
        if program_source_checksum is not None:
            public_metrics["program_source_checksum"] = program_source_checksum
        if training_binding_hash is not None:
            public_metrics["training_binding_hash"] = training_binding_hash
        public_metrics["capability_profile_hash"] = capability_profile_hash
        public_metrics["execution_device"] = expected_execution_device
        public_metrics["checkpoint_safety_status"] = checkpoint_safety_report["status"]
        public_metrics["checkpoint_safety_report"] = checkpoint_safety_report
        for key in CANDIDATE_CONTAMINATION_FIELDS:
            if key in summary:
                value = summary[key]
                public_metrics[key] = value
        rollout_evidence = summary.get("sealed_validation_evidence")
        raw_task_evidence = summary.get("sealed_validation_summary")
        if raw_task_evidence is not None and not isinstance(raw_task_evidence, Mapping):
            raise RunnerExecutionError(
                f"{job_id}: eval summary sealed_validation_summary must be an object"
            )
        declared_task_evidence_hash = (
            rollout_evidence.get("task_evidence_hash")
            if isinstance(rollout_evidence, Mapping)
            else None
        )
        if (declared_task_evidence_hash is not None) != (raw_task_evidence is not None):
            raise RunnerExecutionError(
                f"{job_id}: sealed task evidence hash and summary must be present together"
            )
        if raw_task_evidence is not None:
            raw_task_score_stats = raw_task_evidence.get("score_stats")
            declared_score_stats = (
                rollout_evidence.get("score_stats")
                if isinstance(rollout_evidence, Mapping)
                else None
            )
            if (
                not isinstance(raw_task_score_stats, Mapping)
                or not isinstance(declared_score_stats, Mapping)
                or dict(raw_task_score_stats) != dict(declared_score_stats)
            ):
                raise RunnerExecutionError(
                    f"{job_id}: sealed task summary score_stats differ from public evidence"
                )
        expected_task_evidence_hash = (
            canonical_json_hash(raw_task_evidence)
            if raw_task_evidence is not None
            else None
        )
        rollout_validation = validate_sealed_rollout_evidence(
            rollout_evidence,
            expected_seed_count=expected_sealed_seed_count,
            expected_eval_protocol_hash=expected_eval_protocol_hash,
            expected_evaluation_binding_hash=expected_evaluation_binding_hash,
            expected_evaluation_subject=ExperimentSubject.from_mapping(
                evaluation_subject
            ),
            expected_subject_manifest_hash=subject_manifest_hash,
            expected_program_source_checksum=program_source_checksum,
            expected_training_binding_hash=training_binding_hash,
            expected_capability_profile_hash=capability_profile_hash,
            expected_checkpoint_hash=expected_checkpoint_hash,
            expected_task_evidence_hash=expected_task_evidence_hash,
        )
        if not rollout_validation.complete:
            detail = ", ".join(rollout_validation.reasons)
            raise RunnerExecutionError(
                f"{job_id}: sealed validation rollout evidence invalid: {detail}"
            )
        raw_evidence_manifest = summary.get("evidence_manifest")
        if raw_evidence_manifest is not None:
            if not isinstance(raw_evidence_manifest, list):
                raise RunnerExecutionError(
                    f"{job_id}: eval summary evidence_manifest must be a list"
                )
            try:
                public_metrics["evidence_manifest"] = (
                    validate_evidence_projection_bindings(
                        raw_evidence_manifest,
                        projections={
                            "metrics.checkpoint_safety_report": checkpoint_safety_report,
                            f"metrics.{SEALED_VALIDATION_EVIDENCE_KEY}": rollout_evidence,
                        },
                    )
                )
            except ValueError as exc:
                raise RunnerExecutionError(f"{job_id}: {exc}") from exc
        supplied_gate = summary.get("sealed_validation_gate")
        if not isinstance(supplied_gate, Mapping):
            raise RunnerExecutionError(
                f"{job_id}: eval summary sealed_validation_gate must be an object"
            )
        isolation_gate = build_candidate_isolation_gate(
            candidate_code_loaded=summary.get("candidate_code_loaded_in_eval") is True,
            candidate_package_loaded=summary.get("candidate_package_loaded_in_eval")
            is True,
            candidate_path_argument_present=False,
        )
        canonical_gate = build_sealed_validation_gate(
            rollout_evidence=rollout_evidence,
            checkpoint_safety_status=str(checkpoint_safety_report["status"]),
            checkpoint_hash_match_status=str(
                checkpoint_safety_report["checkpoint_hash_match_status"]
            ),
            candidate_isolation_gate=isolation_gate,
            expected_eval_protocol_hash=expected_eval_protocol_hash,
            expected_evaluation_binding_hash=expected_evaluation_binding_hash,
        )
        if dict(supplied_gate) != canonical_gate:
            raise RunnerExecutionError(
                f"{job_id}: sealed_validation_gate does not match runner-derived evidence"
            )
        public_metrics[SEALED_VALIDATION_EVIDENCE_KEY] = dict(rollout_evidence)
        public_metrics["sealed_validation_gate"] = canonical_gate
        unexpected_runner_metrics = (
            set(public_metrics) - set(metrics) - RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
        )
        if unexpected_runner_metrics:
            raise RunnerExecutionError(
                f"{job_id}: runner projected undeclared public metric(s): "
                + ", ".join(sorted(unexpected_runner_metrics))
            )
        try:
            return PublicEvalResult(
                status=status,
                public_score=public_score,
                metrics=public_metrics,
            )
        except ValueError as exc:
            raise RunnerExecutionError(
                f"{job_id}: invalid public eval result: {exc}"
            ) from exc

    @staticmethod
    def _validate_backend_execution_environment_summary(
        summary: Mapping[str, Any],
        spec: TrainJobSpec | EvaluationJobSpec,
    ) -> None:
        expected_hash = spec.metadata.get("backend_execution_environment_manifest_hash")
        receipt = summary.get("backend_execution_environment_verification")
        if expected_hash is None:
            if receipt is not None:
                raise RunnerExecutionError(
                    f"{spec.job_id}: unexpected backend environment verification"
                )
            return
        expected_hash = require_sha256_digest(
            expected_hash,
            "job backend execution environment manifest hash",
        )
        if not isinstance(receipt, Mapping):
            raise RunnerExecutionError(
                f"{spec.job_id}: backend environment verification is missing"
            )
        expected_fields = {
            "schema_version",
            "backend_id",
            "status",
            "expected_manifest_hash",
            "actual_manifest_hash",
        }
        if set(receipt) != expected_fields:
            raise RunnerExecutionError(
                f"{spec.job_id}: backend environment verification fields mismatch"
            )
        expected_backend_id = spec.metadata.get("backend_id")
        if (
            receipt.get("schema_version")
            != BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION
            or receipt.get("backend_id") != expected_backend_id
            or receipt.get("status") != "pass"
            or receipt.get("expected_manifest_hash") != expected_hash
            or receipt.get("actual_manifest_hash") != expected_hash
        ):
            raise RunnerExecutionError(
                f"{spec.job_id}: backend environment verification mismatch"
            )

    @staticmethod
    def _validate_execution_device_summary(
        summary: Mapping[str, Any],
        spec: TrainJobSpec | EvaluationJobSpec,
    ) -> None:
        if summary.get("execution_device") != spec.execution_device:
            raise RunnerExecutionError(
                f"{spec.job_id}: summary execution_device does not match job spec"
            )

    @staticmethod
    def _validate_train_binding_summary(
        summary: Mapping[str, Any],
        train_spec: TrainJobSpec,
    ) -> None:
        expected = {
            "training_binding_hash": train_spec.training_binding.binding_hash,
            "policy_architecture_hash": (
                train_spec.training_binding.policy_architecture_hash
            ),
            "optimizer_training_profile_hash": (
                train_spec.training_binding.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": (
                train_spec.training_binding.agent_configuration_hash
            ),
            "initialization_seed_schedule_hash": (
                train_spec.training_binding.initialization_seed_schedule_hash
            ),
            "capability_profile_hash": (
                train_spec.training_binding.capability_profile_hash
            ),
            "program_source_checksum": train_spec.program_source_checksum,
        }
        candidate = train_spec.candidate_artifact
        if candidate is not None and candidate.execution_authorization is not None:
            authorization = candidate.execution_authorization
            expected.update(
                {
                    "candidate_verification_hash": candidate.verification_hash,
                    "candidate_admission_policy_id": (
                        authorization.admission_policy_id
                    ),
                    "candidate_admission_policy_hash": (
                        authorization.admission_policy_hash
                    ),
                    "candidate_execution_authorization_schema_version": (
                        authorization.payload_schema_version
                    ),
                    "candidate_execution_authorization_hash": (
                        authorization.authorization_hash
                    ),
                }
            )
        failure_contract = train_spec.candidate_failure_contract
        if failure_contract is None:
            if "candidate_failure_contract_hash" in summary:
                raise RunnerExecutionError(
                    f"{train_spec.job_id}: unexpected candidate failure contract hash"
                )
        else:
            expected["candidate_failure_contract_hash"] = failure_contract.contract_hash
        for field_name, expected_value in expected.items():
            if summary.get(field_name) != expected_value:
                raise RunnerExecutionError(
                    f"{train_spec.job_id}: train summary binding mismatch for "
                    f"{field_name}"
                )

    @staticmethod
    def _initial_policy_state_pairing(
        summary: Mapping[str, Any],
        *,
        train_spec: TrainJobSpec,
    ) -> InitialPolicyStatePairing | None:
        expected = train_spec.expected_initial_policy_state_hash
        if expected is None:
            return None
        try:
            pairing = InitialPolicyStatePairing.from_training_summary(
                summary,
                expected_initial_policy_state_hash=expected,
            )
            pairing.require_match()
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{train_spec.job_id}: initial policy state pairing failed: {exc}"
            ) from exc
        return pairing

    @staticmethod
    def _safe_output(output: str | None) -> str:
        if not output:
            return ""
        redacted = redact_secret_key_markers(output)
        redacted = re.sub(
            r"(?i)(?<![A-Za-z0-9])(OPENAI_API_KEY|API_KEY|TOKEN|SECRET)\s*=\s*\S+",
            r"\1=REDACTED",
            redacted,
        )
        compact = " ".join(redacted.split())
        compact = sanitize_untrusted_public_diagnostic(compact)
        if not isinstance(compact, str):
            raise TypeError("subprocess diagnostic sanitizer must return text")
        return compact[:1200]

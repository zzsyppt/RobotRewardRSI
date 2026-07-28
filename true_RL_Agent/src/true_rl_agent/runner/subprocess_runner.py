"""Runner-owned subprocess execution boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.public_result import PublicEvalResult
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
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.runner.base import (
    CandidateRuntimeError,
    Runner,
    RunnerExecutionError,
    TrainJobResult,
)
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
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
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
    EvaluationJobSpec,
    TrainJobSpec,
    require_positive_integer,
)


@dataclass(frozen=True)
class _SummaryJobCompletion:
    """Validated structured summary paired with its process completion status."""

    summary: dict[str, Any]
    status: str
    returncode: int


_EVAL_CHECKPOINT_OPTIONS = ("--checkpoint", "--checkpoint-sha256")
_TRAIN_CHECKPOINT_OPTIONS = (
    "--init-checkpoint",
    "--init-checkpoint-sha256",
)


@dataclass(frozen=True)
class SubprocessRunner(Runner):
    """Execute materialized train/eval jobs through structured argv only."""

    allowed_executables: tuple[str, ...]
    timeout_s: int = 3600
    cwd: Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.allowed_executables, tuple)
            or not self.allowed_executables
            or any(
                not isinstance(executable, str) or not executable.strip()
                for executable in self.allowed_executables
            )
        ):
            raise ValueError(
                "allowed_executables must be a non-empty tuple of strings"
            )
        require_positive_integer(self.timeout_s, "runner timeout_s")
        if self.cwd is not None and not isinstance(self.cwd, Path):
            raise TypeError("runner cwd must be a Path when provided")

    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
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
        )
        summary = completion.summary
        self._validate_train_binding_summary(summary, train_spec)
        self._raise_failed_train_summary(completion, job_id=train_spec.job_id)
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
            },
        )
        sealed_uri = train_spec.metadata.get("sealed_checkpoint_uri", checkpoint.uri)
        if not isinstance(sealed_uri, str) or not sealed_uri:
            raise RunnerExecutionError(
                f"{train_spec.job_id}: sealed_checkpoint_uri must be a non-empty string"
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
            checkpoint=seal_checkpoint(checkpoint, sealed_uri=sealed_uri),
            summary=summary,
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        return self._run_evaluation(eval_spec)

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        return self._run_evaluation(eval_spec)

    def _run_evaluation(
        self,
        eval_spec: EvaluationJobSpec,
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
        )
        summary = completion.summary
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
            checkpoint.metadata.get("local_path")
            if checkpoint is not None
            else None
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
        )
        try:
            summary = self._read_summary(metadata, job_id=identity.job_id)
            validate_job_summary_identity(
                summary,
                expected=identity,
                summary_name=summary_name,
            )
            status = validate_result_status(
                summary.get("status"),
                field_name=f"{summary_name} status",
            )
        except (RunnerExecutionError, ValueError) as exc:
            error = self._summary_contract_error(
                job_id=identity.job_id,
                completed=completed,
                summary_error=str(exc),
            )
            raise error from exc
        if completed.returncode != 0 and status != "fail":
            raise self._summary_contract_error(
                job_id=identity.job_id,
                completed=completed,
                summary_error=(
                    f"{summary_name} status is pass but subprocess exited nonzero"
                ),
            )
        return _SummaryJobCompletion(
            summary=summary,
            status=status,
            returncode=int(completed.returncode),
        )

    def _run_job(
        self,
        argv: list[str],
        metadata: Mapping[str, Any],
        *,
        job_id: str,
    ) -> subprocess.CompletedProcess[str]:
        work_dir = self._work_dir(metadata)
        try:
            completed = subprocess.run(
                argv,
                cwd=work_dir,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise RunnerExecutionError(f"{job_id}: executable not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerExecutionError(f"{job_id}: subprocess timed out after {self.timeout_s}s") from exc
        except OSError as exc:
            raise RunnerExecutionError(
                f"{job_id}: subprocess launch failed: {type(exc).__name__}"
            ) from exc
        return completed

    def _summary_contract_error(
        self,
        *,
        job_id: str,
        completed: subprocess.CompletedProcess[str],
        summary_error: str,
    ) -> RunnerExecutionError:
        job_prefix = f"{job_id}: "
        if summary_error.startswith(job_prefix):
            summary_error = summary_error[len(job_prefix) :]
        process_context = ""
        if completed.returncode != 0:
            process_context = (
                f"subprocess exited {completed.returncode}; "
                f"stdout={self._safe_output(completed.stdout)}; "
                f"stderr={self._safe_output(completed.stderr)}; "
            )
        return RunnerExecutionError(
            f"{job_id}: {process_context}structured summary invalid: {summary_error}"
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
            raise RunnerExecutionError(f"{job_id}: executable is not allowlisted: {executable}")
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

    def _read_summary(self, metadata: Mapping[str, Any], *, job_id: str) -> dict[str, Any]:
        summary_path = self._summary_path(metadata, job_id=job_id)
        if summary_path.is_symlink():
            raise RunnerExecutionError(f"{job_id}: summary_path must not be a symlink: {summary_path}")
        if not summary_path.exists():
            raise RunnerExecutionError(f"{job_id}: summary file was not created: {summary_path}")
        try:
            summary_text = summary_path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file is not valid UTF-8: {summary_path}"
            ) from exc
        except OSError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file could not be read: {summary_path}"
            ) from exc
        try:
            return strict_json_object_loads(
                summary_text,
                context=f"{job_id} summary file",
            )
        except ValueError as exc:
            raise RunnerExecutionError(
                f"{job_id}: summary file is not valid JSON: {summary_path}"
            ) from exc

    def _summary_path(self, metadata: Mapping[str, Any], *, job_id: str) -> Path:
        raw_summary_path = metadata.get("summary_path")
        if not isinstance(raw_summary_path, str) or not raw_summary_path:
            raise RunnerExecutionError(f"{job_id}: summary_path is required")
        return self._resolved_job_path(raw_summary_path, metadata=metadata)

    def _clear_existing_summary(self, metadata: Mapping[str, Any], *, job_id: str) -> None:
        summary_path = self._summary_path(metadata, job_id=job_id)
        if summary_path.is_symlink():
            raise RunnerExecutionError(f"{job_id}: summary_path must not be a symlink: {summary_path}")
        if not summary_path.exists():
            return
        if not summary_path.is_file():
            raise RunnerExecutionError(f"{job_id}: summary_path exists but is not a file: {summary_path}")
        try:
            summary_path.unlink()
        except OSError as exc:
            raise RunnerExecutionError(f"{job_id}: failed to remove stale summary file: {summary_path}") from exc

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
            raise RunnerExecutionError(f"{job_id}: {key} is required for local subprocess train")
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
            raise RunnerExecutionError(f"{eval_spec.job_id}: eval summary missing evaluation_protocol object")
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
        if not isinstance(raw_declared_path, str) or not raw_declared_path or "://" in raw_declared_path:
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
        job_id: str,
    ) -> None:
        if completion.status != "fail":
            return
        summary = completion.summary
        errors = self._public_errors(summary.get("errors"))
        details: dict[str, Any] = {
            "status": "fail",
            "subprocess_returncode": completion.returncode,
        }
        if errors:
            details["errors"] = errors
        error_text = self._safe_output(json.dumps(errors, sort_keys=True))
        raise CandidateRuntimeError(
            f"{job_id}: train summary status fail; "
            f"subprocess_returncode={completion.returncode}; errors={error_text}",
            public_details=details,
        )

    def _checkpoint_from_summary(self, summary: Mapping[str, Any], *, job_id: str) -> CheckpointRef:
        checkpoint = summary.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RunnerExecutionError(f"{job_id}: train summary missing checkpoint object")
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
                raise ValueError("train summary checkpoint must be unsealed before runner sealing")
            return parsed
        except KeyError as exc:
            raise RunnerExecutionError(f"{job_id}: train summary checkpoint missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise RunnerExecutionError(
                f"{job_id}: invalid train summary checkpoint: {exc}"
            ) from exc

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
        for reserved_key in (
            "checkpoint_safety_report",
            "checkpoint_safety_status",
            "evidence_manifest",
            SEALED_VALIDATION_EVIDENCE_KEY,
            "sealed_validation_gate",
            "evaluation_binding",
            "subject_manifest_hash",
            "program_source_checksum",
            "training_binding_hash",
            "capability_profile_hash",
            *ALTERNATE_JOB_IDENTITY_FIELDS,
        ):
            if reserved_key in metrics:
                raise RunnerExecutionError(f"{job_id}: eval summary metrics must not contain {reserved_key}")
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
        public_metrics["checkpoint_safety_status"] = checkpoint_safety_report["status"]
        public_metrics["checkpoint_safety_report"] = checkpoint_safety_report
        for key in CANDIDATE_CONTAMINATION_FIELDS:
            if key in summary:
                value = summary[key]
                public_metrics[key] = value
        rollout_evidence = summary.get("sealed_validation_evidence")
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
        )
        if not rollout_validation.complete:
            detail = ", ".join(rollout_validation.reasons)
            raise RunnerExecutionError(
                f"{job_id}: sealed validation rollout evidence invalid: {detail}"
            )
        raw_evidence_manifest = summary.get("evidence_manifest")
        if raw_evidence_manifest is not None:
            if not isinstance(raw_evidence_manifest, list):
                raise RunnerExecutionError(f"{job_id}: eval summary evidence_manifest must be a list")
            try:
                public_metrics["evidence_manifest"] = validate_evidence_projection_bindings(
                    raw_evidence_manifest,
                    projections={
                        "metrics.checkpoint_safety_report": checkpoint_safety_report,
                        f"metrics.{SEALED_VALIDATION_EVIDENCE_KEY}": rollout_evidence,
                    },
                )
            except ValueError as exc:
                raise RunnerExecutionError(f"{job_id}: {exc}") from exc
        supplied_gate = summary.get("sealed_validation_gate")
        if not isinstance(supplied_gate, Mapping):
            raise RunnerExecutionError(f"{job_id}: eval summary sealed_validation_gate must be an object")
        isolation_gate = build_candidate_isolation_gate(
            candidate_code_loaded=summary.get("candidate_code_loaded_in_eval") is True,
            candidate_package_loaded=summary.get("candidate_package_loaded_in_eval") is True,
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
        try:
            return PublicEvalResult(
                status=status,
                public_score=public_score,
                metrics=public_metrics,
            )
        except ValueError as exc:
            raise RunnerExecutionError(f"{job_id}: invalid public eval result: {exc}") from exc

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
        for field_name, expected_value in expected.items():
            if summary.get(field_name) != expected_value:
                raise RunnerExecutionError(
                    f"{train_spec.job_id}: train summary binding mismatch for "
                    f"{field_name}"
                )

    @classmethod
    def _public_errors(cls, errors: Any) -> list[dict[str, str]]:
        if not isinstance(errors, list):
            return []
        public_errors: list[dict[str, str]] = []
        for raw in errors[:5]:
            if not isinstance(raw, Mapping):
                continue
            item: dict[str, str] = {}
            if "type" in raw:
                item["type"] = cls._safe_output(str(raw["type"]))
            if "message" in raw:
                item["message"] = cls._safe_output(str(raw["message"]))
            if item:
                public_errors.append(item)
        return public_errors

    @staticmethod
    def _safe_output(output: str | None) -> str:
        if not output:
            return ""
        redacted = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", output)
        redacted = re.sub(r"(?i)(OPENAI_API_KEY|API_KEY|TOKEN|SECRET)=\S+", r"\1=REDACTED", redacted)
        redacted = re.sub(r"(?i)(Authorization:\s*Bearer\s+)\S+", r"\1REDACTED", redacted)
        compact = " ".join(redacted.split())
        return compact[:1200]

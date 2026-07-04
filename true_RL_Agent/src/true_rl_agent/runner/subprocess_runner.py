"""Runner-owned subprocess execution boundary."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.runner.base import CandidateRuntimeError, Runner, RunnerExecutionError, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec


@dataclass(frozen=True)
class SubprocessRunner(Runner):
    """Execute materialized train/eval jobs through structured argv only."""

    allowed_executables: tuple[str, ...]
    timeout_s: int = 3600
    cwd: Path | None = None

    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        self._run_job(train_spec.metadata, job_id=train_spec.job_id)
        summary = self._read_summary(train_spec.metadata, job_id=train_spec.job_id)
        self._raise_failed_train_summary(summary, job_id=train_spec.job_id)
        checkpoint = self._checkpoint_from_summary(summary, job_id=train_spec.job_id)
        sealed_uri = str(train_spec.metadata.get("sealed_checkpoint_uri", checkpoint.uri))
        return TrainJobResult(
            job_id=train_spec.job_id,
            checkpoint=seal_checkpoint(checkpoint, sealed_uri=sealed_uri),
            summary=summary,
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        self._run_job(eval_spec.metadata, job_id=eval_spec.job_id)
        summary = self._read_summary(eval_spec.metadata, job_id=eval_spec.job_id)
        return self._public_result_from_summary(summary, job_id=eval_spec.job_id)

    def _run_job(self, metadata: Mapping[str, Any], *, job_id: str) -> None:
        argv = self._validated_argv(metadata, job_id=job_id)
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

        if completed.returncode != 0:
            raise RunnerExecutionError(
                f"{job_id}: subprocess exited {completed.returncode}; "
                f"stdout={self._safe_output(completed.stdout)}; "
                f"stderr={self._safe_output(completed.stderr)}"
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
        return list(argv)

    def _work_dir(self, metadata: Mapping[str, Any]) -> Path | None:
        raw_work_dir = metadata.get("work_dir")
        if raw_work_dir is None:
            return self.cwd
        if not isinstance(raw_work_dir, str):
            raise RunnerExecutionError("work_dir must be a string path")
        return Path(raw_work_dir)

    def _read_summary(self, metadata: Mapping[str, Any], *, job_id: str) -> dict[str, Any]:
        raw_summary_path = metadata.get("summary_path")
        if not isinstance(raw_summary_path, str) or not raw_summary_path:
            raise RunnerExecutionError(f"{job_id}: summary_path is required")
        summary_path = Path(raw_summary_path)
        if not summary_path.exists():
            raise RunnerExecutionError(f"{job_id}: summary file was not created: {summary_path}")
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunnerExecutionError(f"{job_id}: summary file is not valid JSON: {summary_path}") from exc
        if not isinstance(data, dict):
            raise RunnerExecutionError(f"{job_id}: summary file must contain a JSON object")
        return data

    def _raise_failed_train_summary(self, summary: Mapping[str, Any], *, job_id: str) -> None:
        if summary.get("status") != "fail":
            return
        errors = self._public_errors(summary.get("errors"))
        details: dict[str, Any] = {"status": "fail"}
        if errors:
            details["errors"] = errors
        error_text = self._safe_output(json.dumps(errors, sort_keys=True))
        raise CandidateRuntimeError(
            f"{job_id}: train summary status fail; errors={error_text}",
            public_details=details,
        )

    def _checkpoint_from_summary(self, summary: Mapping[str, Any], *, job_id: str) -> CheckpointRef:
        checkpoint = summary.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RunnerExecutionError(f"{job_id}: train summary missing checkpoint object")
        try:
            return CheckpointRef(
                uri=str(checkpoint["uri"]),
                checksum=str(checkpoint["checksum"]),
                role=str(checkpoint["role"]),
                sealed=bool(checkpoint.get("sealed", False)),
                metadata=dict(checkpoint.get("metadata", {})),
            )
        except KeyError as exc:
            raise RunnerExecutionError(f"{job_id}: train summary checkpoint missing {exc.args[0]}") from exc

    def _public_result_from_summary(self, summary: Mapping[str, Any], *, job_id: str) -> PublicEvalResult:
        status = summary.get("status")
        if not isinstance(status, str):
            raise RunnerExecutionError(f"{job_id}: eval summary missing status")
        raw_score = summary.get("public_score")
        if not isinstance(raw_score, int | float):
            raise RunnerExecutionError(f"{job_id}: eval summary missing numeric public_score")
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise RunnerExecutionError(f"{job_id}: eval summary metrics must be an object")
        public_metrics = dict(metrics)
        for key in ("candidate_code_loaded_in_eval", "candidate_package_loaded"):
            if key in summary:
                value = summary[key]
                if not isinstance(value, bool):
                    raise RunnerExecutionError(f"{job_id}: eval summary {key} must be boolean")
                public_metrics[key] = value
        return PublicEvalResult(status=status, public_score=float(raw_score), metrics=public_metrics)

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

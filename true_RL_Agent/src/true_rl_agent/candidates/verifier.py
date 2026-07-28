"""Candidate verifier interface for replay and fake-round tests."""

from __future__ import annotations

import ast
import multiprocessing as mp
import queue as queue_module
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.artifact import (
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    SealedCandidateArtifactRef,
    VerifiedCandidateArtifactRef,
    seal_candidate_package,
    verify_sealed_candidate_artifact,
)
from true_rl_agent.candidates.loader import (
    load_candidate_manifest,
    load_training_components,
)
from true_rl_agent.candidates.package_contract import (
    PROVIDER_AUTHORED_CANDIDATE_FILES,
)
from true_rl_agent.candidates.schedule_contract import (
    allowed_schedule_keys as candidate_allowed_schedule_keys,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.training_programs import (
    CandidateProgramIdentity,
    resolve_candidate_program_identity,
)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)
    program_identity: CandidateProgramIdentity | None = None
    candidate_artifact_hash: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "reason": self.reason,
            "details": dict(self.details),
        }
        if self.program_identity is not None:
            result["verified_program_identity"] = {
                "candidate_id": self.program_identity.candidate_id,
                "subject_manifest_hash": self.program_identity.manifest_hash,
                "program_source_checksum": self.program_identity.source_checksum,
                "program_source_size_bytes": self.program_identity.source_size_bytes,
                "candidate_artifact_hash": self.candidate_artifact_hash,
            }
        return result


@dataclass(frozen=True)
class CandidateArtifactVerification:
    result: VerificationResult
    verified_artifact: VerifiedCandidateArtifactRef | None = None

    @property
    def passed(self) -> bool:
        return self.result.status == "pass" and self.verified_artifact is not None

    def require_verified(self) -> VerifiedCandidateArtifactRef:
        if self.verified_artifact is None:
            raise ValueError(
                "candidate artifact verification failed: " + self.result.reason
            )
        return self.verified_artifact


class CandidateVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    ) -> VerificationResult:
        raise NotImplementedError


def seal_and_verify_candidate(
    candidate: CandidatePackageRef,
    *,
    artifact_root: str | Path,
    verifier: CandidateVerifier,
) -> CandidateArtifactVerification:
    """Create the only execution capability accepted by production backends."""

    try:
        sealed = seal_candidate_package(candidate, artifact_root=artifact_root)
    except (OSError, TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_sealing_failed",
                {"issues": [f"artifact_sealing_failed:{type(exc).__name__}"]},
            )
        )
    return verify_candidate_artifact(sealed, verifier=verifier)


def verify_candidate_artifact(
    artifact: SealedCandidateArtifactRef,
    *,
    verifier: CandidateVerifier,
) -> CandidateArtifactVerification:
    try:
        sealed_before = verify_sealed_candidate_artifact(artifact)
    except (OSError, TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_invalid",
                {"issues": [f"artifact_invalid:{type(exc).__name__}"]},
            )
        )

    result = verifier.verify(sealed_before)
    if result.status != "pass":
        return CandidateArtifactVerification(result=result)
    if not _verified_artifact_matches(result, sealed_before):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verified_identity_mismatch",
                {"issues": ["verified_identity_does_not_match_sealed_artifact"]},
            )
        )
    try:
        sealed_after = verify_sealed_candidate_artifact(sealed_before)
    except (OSError, TypeError, ValueError) as exc:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_changed_during_verification",
                {"issues": [f"artifact_changed:{type(exc).__name__}"]},
            )
        )
    if sealed_after != sealed_before:
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_artifact_changed_during_verification",
                {"issues": ["artifact_reference_changed"]},
            )
        )

    effective_change_set = candidate_change_set_from_verification(
        fallback=sealed_after.change_set,
        details=result.details,
    )
    bound_artifact = replace(
        sealed_after,
        change_set=effective_change_set,
    )
    try:
        verification_hash = canonical_json_hash(result.to_public_dict())
    except (TypeError, ValueError):
        return CandidateArtifactVerification(
            result=VerificationResult(
                "fail",
                "candidate_verification_evidence_invalid",
            )
        )
    return CandidateArtifactVerification(
        result=result,
        verified_artifact=VerifiedCandidateArtifactRef._validated(
            sealed_artifact=bound_artifact,
            verification_policy=result.reason,
            verification_hash=verification_hash,
        ),
    )


class StaticCandidateVerifier(CandidateVerifier):
    DEFAULT_COMPONENT_LOAD_TIMEOUT_S = 2.0

    provider_authored_files = frozenset(PROVIDER_AUTHORED_CANDIDATE_FILES)
    runner_internal_files = frozenset(
        {CANDIDATE_ARTIFACT_MANIFEST_FILENAME}
    )
    forbidden_tokens = (
        "import os",
        "subprocess",
        "open(",
        "optimizer",
        "checkpoint_selector",
        "eval_config",
        "success_metric",
        "runner.",
        "env.",
    )
    allowed_imports = {
        "__future__": frozenset({"annotations"}),
        "true_rl_agent.candidates.components": frozenset(
            {"DiagnosticTerm", "RewardInputs", "RewardTerm", "ScheduleTerm", "TrainingComponents"}
        ),
    }
    forbidden_names = frozenset(
        {
            "env",
            "runner",
            "optimizer",
            "checkpoint_selector",
            "eval_config",
            "success_metric",
        }
    )
    forbidden_calls = frozenset(
        {
            "__import__",
            "compile",
            "eval",
            "exec",
            "globals",
            "input",
            "locals",
            "open",
            "vars",
        }
    )
    forbidden_call_roots = frozenset({"os", "pathlib", "shutil", "socket", "subprocess", "sys"})
    allowed_schedule_keys = frozenset(candidate_allowed_schedule_keys())

    def __init__(self, *, component_load_timeout_s: float = DEFAULT_COMPONENT_LOAD_TIMEOUT_S) -> None:
        if component_load_timeout_s <= 0:
            raise ValueError("component_load_timeout_s must be positive")
        self.component_load_timeout_s = float(component_load_timeout_s)

    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef | Path,
    ) -> VerificationResult:
        issues: list[str] = []
        try:
            candidate_dir = self._candidate_dir(candidate)
        except ValueError as exc:
            return VerificationResult("fail", "candidate_path_invalid", {"issues": [str(exc)]})
        allowed_files = self.provider_authored_files
        if isinstance(candidate, SealedCandidateArtifactRef):
            allowed_files = allowed_files | self.runner_internal_files
        for path in candidate_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(candidate_dir).as_posix()
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if rel not in allowed_files:
                issues.append(f"forbidden_file:{rel}")
                continue
            if path.suffix == ".py":
                text = path.read_text(encoding="utf-8").lstrip("\ufeff")
                for token in self.forbidden_tokens:
                    if token in text:
                        issues.append(f"forbidden_token:{rel}:{token}")
                issues.extend(self._audit_python_source(rel, text))
        try:
            manifest = load_candidate_manifest(candidate_dir)
            if isinstance(candidate, CandidatePackageRef) and manifest.candidate_id != candidate.candidate_id:
                issues.append("manifest_candidate_id_mismatch")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"manifest_load_failed:{type(exc).__name__}")
        if issues:
            return VerificationResult("fail", "static_candidate_verifier_failed", {"issues": issues})
        try:
            if isinstance(candidate, SealedCandidateArtifactRef):
                verify_sealed_candidate_artifact(candidate)
            identity_before = resolve_candidate_program_identity(
                candidate_dir,
                expected_candidate_id=(
                    candidate.candidate_id
                    if isinstance(
                        candidate,
                        CandidatePackageRef | SealedCandidateArtifactRef,
                    )
                    else manifest.candidate_id
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "fail",
                "candidate_program_identity_invalid",
                {"issues": [f"program_identity_invalid:{type(exc).__name__}"]},
            )
        loaded = _load_components_summary_with_timeout(
            candidate_dir=candidate_dir,
            allowed_schedule_keys=self.allowed_schedule_keys,
            timeout_s=self.component_load_timeout_s,
        )
        if loaded["status"] == "timeout":
            return VerificationResult(
                "fail",
                "training_components_load_timeout",
                {"issues": ["component_load_timeout"]},
            )
        if loaded["status"] == "fail":
            return VerificationResult(
                "fail",
                "training_components_load_failed",
                {"issues": [f"component_load_failed:{loaded['error_type']}"]},
            )
        component_issues = list(loaded["component_issues"])
        if component_issues:
            return VerificationResult(
                "fail",
                "training_components_invalid",
                {"issues": component_issues},
            )
        if loaded["is_noop"]:
            return VerificationResult("fail", "candidate_has_no_effective_components")
        try:
            if isinstance(candidate, SealedCandidateArtifactRef):
                verify_sealed_candidate_artifact(candidate)
            identity_after = resolve_candidate_program_identity(
                candidate_dir,
                expected_candidate_id=identity_before.candidate_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "fail",
                "candidate_program_identity_changed",
                {"issues": [f"program_identity_changed:{type(exc).__name__}"]},
            )
        if identity_after != identity_before:
            return VerificationResult(
                "fail",
                "candidate_program_identity_changed",
                {"issues": ["program_identity_changed_during_verification"]},
            )
        return VerificationResult(
            "pass",
            "static_candidate_verifier_passed",
            {"derived_change_set": list(loaded["derived_change_set"])},
            program_identity=identity_after,
            candidate_artifact_hash=(
                candidate.artifact_hash
                if isinstance(candidate, SealedCandidateArtifactRef)
                else None
            ),
        )

    def _candidate_dir(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef | Path,
    ) -> Path:
        if isinstance(candidate, CandidatePackageRef | SealedCandidateArtifactRef):
            if "://" in candidate.path:
                raise ValueError("candidate_path_must_be_local_directory")
            return Path(candidate.path)
        return Path(candidate)

    def _audit_python_source(self, rel: str, text: str) -> list[str]:
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            return [f"syntax_error:{rel}:{exc.lineno}"]

        issues: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, ast.ImportFrom):
                issues.extend(self._audit_import_from(rel, node))
                continue
            if isinstance(node, ast.FunctionDef):
                continue
            issues.append(f"forbidden_top_level:{rel}:{type(node).__name__}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                issues.append(f"forbidden_import:{rel}:import")
            elif isinstance(node, ast.ImportFrom):
                issues.extend(self._audit_import_from(rel, node))
            elif isinstance(node, ast.Name) and node.id in self.forbidden_names:
                issues.append(f"forbidden_name:{rel}:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in self.forbidden_names:
                issues.append(f"forbidden_attribute:{rel}:{node.attr}")
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in self.forbidden_calls:
                    issues.append(f"forbidden_call:{rel}:{call_name}")
                call_root = call_name.split(".", 1)[0] if call_name else ""
                if call_root in self.forbidden_call_roots:
                    issues.append(f"forbidden_call:{rel}:{call_name}")
        return sorted(set(issues))

    def _audit_import_from(self, rel: str, node: ast.ImportFrom) -> list[str]:
        if node.level != 0 or node.module not in self.allowed_imports:
            return [f"forbidden_import:{rel}:{node.module or ''}"]
        allowed_names = self.allowed_imports[node.module]
        issues = []
        for alias in node.names:
            if alias.name not in allowed_names:
                issues.append(f"forbidden_import_name:{rel}:{node.module}.{alias.name}")
            if alias.asname is not None:
                issues.append(f"forbidden_import_alias:{rel}:{alias.name}")
        return issues

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""


def candidate_change_set_from_verification(
    *,
    fallback: CandidateChangeSet,
    details: Mapping[str, Any],
) -> CandidateChangeSet:
    raw = details.get("derived_change_set")
    if not isinstance(raw, list | tuple) or not all(
        isinstance(item, str) for item in raw
    ):
        return fallback
    return CandidateChangeSet(frozenset(raw))


def _verified_artifact_matches(
    verification: VerificationResult,
    artifact: SealedCandidateArtifactRef,
) -> bool:
    identity = verification.program_identity
    return (
        identity is not None
        and identity.candidate_id == artifact.candidate_id
        and identity.manifest_hash == artifact.subject_manifest_hash
        and identity.source_checksum == artifact.program_source_checksum
        and identity.source_size_bytes == artifact.program_source_size_bytes
        and verification.candidate_artifact_hash == artifact.artifact_hash
    )


def static_verifier_prompt_rules() -> dict[str, Any]:
    """Return the prompt-safe static rules enforced by StaticCandidateVerifier."""

    return {
        "schema_version": "true_rl_agent.static_candidate_verifier_prompt_rules.v1",
        "rule_source": "StaticCandidateVerifier",
        "allowed_files": sorted(StaticCandidateVerifier.provider_authored_files),
        "allowed_imports": {
            module: sorted(names)
            for module, names in sorted(StaticCandidateVerifier.allowed_imports.items())
        },
        "blocked_text_fragments": [
            _prompt_safe_verifier_rule_text(item)
            for item in StaticCandidateVerifier.forbidden_tokens
        ],
        "blocked_names": [
            _prompt_safe_verifier_rule_text(item)
            for item in sorted(StaticCandidateVerifier.forbidden_names)
        ],
        "blocked_calls": sorted(StaticCandidateVerifier.forbidden_calls),
        "blocked_call_roots": sorted(StaticCandidateVerifier.forbidden_call_roots),
        "allowed_top_level_ast_nodes": [
            "module_docstring",
            "ImportFrom",
            "FunctionDef",
        ],
        "allowed_schedule_keys": sorted(StaticCandidateVerifier.allowed_schedule_keys),
        "additional_files_allowed": False,
        "import_aliases_allowed": False,
        "relative_imports_allowed": False,
    }


def _prompt_safe_verifier_rule_text(value: str) -> str:
    replacements = {
        "checkpoint_selector": "checkpoint policy selector surface",
        "eval_config": "eval configuration surface",
        "runner.": "runner attribute access",
        "env.": "environment attribute access",
        "success_metric": "success metric surface",
    }
    return replacements.get(value, value)


def _load_components_summary_with_timeout(
    *,
    candidate_dir: Path,
    allowed_schedule_keys: frozenset[str],
    timeout_s: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_component_summary_worker,
        args=(str(candidate_dir), tuple(sorted(allowed_schedule_keys)), result_queue),
    )
    process.daemon = True
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1.0)
        return {"status": "timeout"}
    try:
        result = result_queue.get_nowait()
    except queue_module.Empty:
        return {"status": "fail", "error_type": "WorkerExit"}
    if isinstance(result, dict):
        return result
    return {"status": "fail", "error_type": "WorkerResultInvalid"}


def _component_summary_worker(
    candidate_dir: str,
    allowed_schedule_keys: tuple[str, ...],
    result_queue: Any,
) -> None:
    try:
        components = load_training_components(Path(candidate_dir))
        allowed = frozenset(allowed_schedule_keys)
        result_queue.put(
            {
                "status": "pass",
                "component_issues": _component_issues(components, allowed),
                "is_noop": components.is_noop(),
                "derived_change_set": sorted(components.to_change_set().changed_components),
            }
        )
    except Exception as exc:  # noqa: BLE001 - verifier exposes only the public error type.
        result_queue.put({"status": "fail", "error_type": type(exc).__name__})


def _component_issues(components: Any, allowed_schedule_keys: frozenset[str]) -> list[str]:
    issues: list[str] = []
    for reward in components.reward_terms:
        if not callable(reward.fn):
            issues.append(f"non_callable_reward_term:{reward.name}")
    for diagnostic in components.diagnostics:
        if not callable(diagnostic.fn):
            issues.append(f"non_callable_diagnostic_term:{diagnostic.name}")
    for schedule in components.schedules:
        if schedule.key not in allowed_schedule_keys:
            issues.append(f"schedule_key_not_allowlisted:{schedule.name}:{schedule.key}")
    return issues

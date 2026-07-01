"""Candidate verifier interface for replay and fake-round tests."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.loader import load_candidate_manifest, load_training_components
from true_rl_agent.candidates.schema import CandidatePackageRef


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "details": dict(self.details)}


class CandidateVerifier(ABC):
    @abstractmethod
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        raise NotImplementedError


class StaticCandidateVerifier:
    allowed_files = frozenset(
        {
            "candidate_manifest.yaml",
            "training_code/components.py",
            "rationale.md",
            "diagnostics.json",
            "operation_diff.json",
        }
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
            {"DiagnosticTerm", "RewardTerm", "ScheduleTerm", "TrainingComponents"}
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
    allowed_schedule_keys = frozenset({"reward_weight_scale"})

    def verify(self, candidate_dir: Path) -> VerificationResult:
        issues: list[str] = []
        for path in candidate_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(candidate_dir).as_posix()
            if rel not in self.allowed_files:
                issues.append(f"forbidden_file:{rel}")
                continue
            if path.suffix == ".py":
                text = path.read_text(encoding="utf-8")
                for token in self.forbidden_tokens:
                    if token in text:
                        issues.append(f"forbidden_token:{rel}:{token}")
                issues.extend(self._audit_python_source(rel, text))
        try:
            load_candidate_manifest(candidate_dir)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"manifest_load_failed:{type(exc).__name__}")
        if issues:
            return VerificationResult("fail", "static_candidate_verifier_failed", {"issues": issues})
        try:
            components = load_training_components(candidate_dir)
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(
                "fail",
                "training_components_load_failed",
                {"issues": [f"component_load_failed:{type(exc).__name__}"]},
            )
        component_issues = self._audit_loaded_components(components)
        if component_issues:
            return VerificationResult(
                "fail",
                "training_components_invalid",
                {"issues": component_issues},
            )
        if components.is_noop():
            return VerificationResult("fail", "candidate_has_no_effective_components")
        return VerificationResult("pass", "static_candidate_verifier_passed")

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

    def _audit_loaded_components(self, components: Any) -> list[str]:
        issues: list[str] = []
        for reward in components.reward_terms:
            if not callable(reward.fn):
                issues.append(f"non_callable_reward_term:{reward.name}")
        for diagnostic in components.diagnostics:
            if not callable(diagnostic.fn):
                issues.append(f"non_callable_diagnostic_term:{diagnostic.name}")
        for schedule in components.schedules:
            if schedule.key not in self.allowed_schedule_keys:
                issues.append(f"schedule_key_not_allowlisted:{schedule.name}:{schedule.key}")
        return issues

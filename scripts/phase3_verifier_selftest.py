#!/usr/bin/env python3
"""Self-test the Phase 3 static verifier with synthetic candidate patches."""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import (
    load_verifier_config,
    load_verifier_config_from_git,
    run_static_verification,
    verify_static_documents,
)
from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - self-test uses parser fallback.


def _run_case(
    name: str,
    base_docs: dict[str, Any],
    current_docs: dict[str, Any],
    changed_files: list[str],
    settings: dict[str, Any],
    *,
    expect_pass: bool,
    expected_code: str | None = None,
) -> dict[str, Any]:
    result = verify_static_documents(
        base_docs=base_docs,
        current_docs=current_docs,
        changed_files=changed_files,
        settings=settings,
        check_imports=False,
    )
    passed = result["status"] == "pass"
    codes = {issue["code"] for issue in result["issues"]}
    ok = passed == expect_pass and (expected_code is None or expected_code in codes)
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "verifier_status": result["status"],
        "issue_codes": sorted(codes),
        "expected_pass": expect_pass,
        "expected_code": expected_code,
    }


def _run_git_command(repo: Path, args: list[str]) -> None:
    completed = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")


def _run_git_integration_case(repo_root: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Exercise the real git diff + git show base:path path."""

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="phase3-verifier-git-") as tmp:
        repo = Path(tmp)
        shutil.copytree(repo_root / "configs", repo / "configs")
        _run_git_command(repo, ["init"])
        _run_git_command(repo, ["add", "configs"])
        _run_git_command(
            repo,
            [
                "-c",
                "user.email=phase3-verifier@example.invalid",
                "-c",
                "user.name=Phase3 Verifier",
                "commit",
                "-m",
                "base",
            ],
        )

        phase2_path = repo / "configs/phase2_baseline.yaml"
        phase3_path = repo / "configs/phase3_verifier.yaml"

        phase2_text = phase2_path.read_text(encoding="utf-8")
        phase2_path.write_text(phase2_text.replace("    seed: 1001", "    seed: 9999", 1), encoding="utf-8")

        # Simulate a malicious candidate deleting all verifier guards in the
        # worktree. The verifier must still load rules from HEAD.
        phase3_path.write_text(
            "schema_version: tampered\n"
            "diff_policy:\n"
            "  ignored_globs: []\n"
            "  allowed_globs:\n"
            "    - '**'\n"
            "  forbidden_globs: []\n"
            "structured_files:\n"
            "  yaml_globs: []\n"
            "  json_globs: []\n"
            "protected_keys: {}\n",
            encoding="utf-8",
        )
        trusted = load_verifier_config_from_git(repo, "HEAD", "configs/phase3_verifier.yaml")
        result = run_static_verification(repo, trusted, base_ref="HEAD", include_untracked=False, check_imports=False)
        codes = {issue["code"] for issue in result["issues"]}
        cases.append(
            {
                "name": "git_path_rejects_tampered_verifier_config_and_eval_seed",
                "status": "pass"
                if result["status"] == "fail"
                and "forbidden_file_changed" in codes
                and "protected_key_changed" in codes
                else "fail",
                "verifier_status": result["status"],
                "issue_codes": sorted(codes),
            }
        )

        _run_git_command(repo, ["checkout", "--", "configs/phase2_baseline.yaml", "configs/phase3_verifier.yaml"])
        phase2_path.unlink()
        result = run_static_verification(repo, trusted, base_ref="HEAD", include_untracked=False, check_imports=False)
        codes = {issue["code"] for issue in result["issues"]}
        cases.append(
            {
                "name": "git_path_rejects_deleted_protected_config",
                "status": "pass"
                if result["status"] == "fail"
                and "protected_file_missing" in codes
                else "fail",
                "verifier_status": result["status"],
                "issue_codes": sorted(codes),
            }
        )
    return cases


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    settings = load_verifier_config(repo_root / "configs/phase3_verifier.yaml")
    base_docs = {
        "configs/phase1_push_recovery.yaml": _load_structured_file(repo_root / "configs/phase1_push_recovery.yaml"),
        "configs/phase2_baseline.yaml": _load_structured_file(repo_root / "configs/phase2_baseline.yaml"),
        "configs/phase3_verifier.yaml": settings,
    }

    cases: list[dict[str, Any]] = []

    legal_docs = copy.deepcopy(base_docs)
    legal_docs["configs/phase2_baseline.yaml"]["reward_weights"]["alive"] = 0.30
    cases.append(
        _run_case(
            "legal_reward_weight_patch",
            base_docs,
            legal_docs,
            ["configs/phase2_baseline.yaml"],
            settings,
            expect_pass=True,
        )
    )

    eval_seed_docs = copy.deepcopy(base_docs)
    eval_seed_docs["configs/phase2_baseline.yaml"]["evaluation"]["public"]["seed"] = 9999
    cases.append(
        _run_case(
            "reject_eval_seed_patch",
            base_docs,
            eval_seed_docs,
            ["configs/phase2_baseline.yaml"],
            settings,
            expect_pass=False,
            expected_code="protected_key_changed",
        )
    )

    nan_docs = copy.deepcopy(base_docs)
    nan_docs["configs/phase2_baseline.yaml"]["reward_weights"]["alive"] = math.nan
    cases.append(
        _run_case(
            "reject_nan_reward_weight",
            base_docs,
            nan_docs,
            ["configs/phase2_baseline.yaml"],
            settings,
            expect_pass=False,
            expected_code="non_finite_numeric",
        )
    )

    numeric_string_docs = copy.deepcopy(base_docs)
    numeric_string_docs["configs/phase2_baseline.yaml"]["reward_weights"]["energy"] = "-2e-05"
    cases.append(
        _run_case(
            "reject_numeric_string_reward_weight",
            base_docs,
            numeric_string_docs,
            ["configs/phase2_baseline.yaml"],
            settings,
            expect_pass=False,
            expected_code="numeric_string_config_value",
        )
    )

    forbidden_docs = copy.deepcopy(base_docs)
    cases.append(
        _run_case(
            "reject_protected_eval_script",
            base_docs,
            forbidden_docs,
            ["scripts/phase1_push_recovery_eval.py"],
            settings,
            expect_pass=False,
            expected_code="forbidden_file_changed",
        )
    )

    budget_docs = copy.deepcopy(base_docs)
    budget_docs["configs/phase2_baseline.yaml"]["training"]["max_iterations"] = 1000
    cases.append(
        _run_case(
            "reject_training_budget_patch",
            base_docs,
            budget_docs,
            ["configs/phase2_baseline.yaml"],
            settings,
            expect_pass=False,
            expected_code="protected_key_changed",
        )
    )

    cases.extend(_run_git_integration_case(repo_root, settings))

    result = {
        "schema_version": "humanoid_agent.phase3.verifier.selftest.v1",
        "status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
        "cases": cases,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

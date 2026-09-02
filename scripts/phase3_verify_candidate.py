#!/usr/bin/env python3
"""Run the Phase 3 verifier for a candidate patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import (
    load_verifier_config,
    load_verifier_config_from_git,
    run_smoke_verification,
    run_static_verification,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--verifier-config", default="configs/phase3_verifier.yaml")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--mode", choices=("static", "smoke", "full"), default="static")
    parser.add_argument("--run-dir", default="runs/phase3_verifier/latest")
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--exclude-untracked", action="store_true")
    parser.add_argument(
        "--trust-worktree-verifier-config",
        action="store_true",
        help="Load verifier policy from the worktree instead of base-ref. Use only for bootstrapping or smoke-only debugging.",
    )
    return parser


def _combine(static: dict[str, Any] | None, smoke: dict[str, Any] | None) -> dict[str, Any]:
    issues = []
    checks = []
    for result in (static, smoke):
        if not result:
            continue
        issues.extend(result.get("issues", []))
        checks.extend(result.get("checks", []))
    return {
        "schema_version": "humanoid_agent.phase3.verifier.summary.v1",
        "status": "fail" if any(issue.get("level") == "error" for issue in issues) else "pass",
        "mode": "full",
        "checks": checks,
        "issues": issues,
        "static": static,
        "smoke": smoke,
    }


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.mode in {"static", "full"} and not args.trust_worktree_verifier_config:
        settings = load_verifier_config_from_git(repo_root, args.base_ref, args.verifier_config)
    else:
        settings = load_verifier_config(repo_root / args.verifier_config)

    static_result = None
    smoke_result = None
    if args.mode in {"static", "full"}:
        static_result = run_static_verification(
            repo_root,
            settings,
            base_ref=args.base_ref,
            include_untracked=not args.exclude_untracked,
        )
    if args.mode in {"smoke", "full"}:
        smoke_result = run_smoke_verification(repo_root, settings, run_dir=repo_root / args.run_dir)

    if args.mode == "static":
        result = static_result
    elif args.mode == "smoke":
        result = smoke_result
    else:
        result = _combine(static_result, smoke_result)

    assert result is not None
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        if not summary_path.is_absolute():
            summary_path = repo_root / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

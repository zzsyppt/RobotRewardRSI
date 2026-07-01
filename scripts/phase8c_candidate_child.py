#!/usr/bin/env python3
"""Phase 8C isolated child process for loading candidate components.

This script is intentionally small: the parent runner launches it in a restricted
process environment.  The child applies Landlock before importing candidate
code, then writes a public structural summary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolation import (  # noqa: E402
    IsolationError,
    apply_landlock,
    default_landlock_read_paths,
)
from humanoid_agent.code_sandbox.loader import CandidateLoadError, load_training_components  # noqa: E402
from humanoid_agent.cold_start_lab.recipe_adapter import fast_validate_components  # noqa: E402
from humanoid_agent.cold_start_lab.training_adapter import reward_inputs_contract  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-schema-version", default="humanoid_agent.cold_start.candidate.v1")
    parser.add_argument("--allow-llm-authored", action="store_true")
    parser.add_argument("--require-noop-components", action="store_true")
    parser.add_argument("--fast-validation-config", default=None)
    parser.add_argument("--landlock", action="store_true")
    return parser


def _summary(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8c.candidate_child.summary.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **kwargs,
    }


def main() -> int:
    args = _build_parser().parse_args()
    candidate_dir = Path(args.candidate_dir).resolve()
    summary_file = Path(args.summary_file).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    isolation: dict[str, Any] = {
        "landlock_requested": bool(args.landlock),
        "landlock": {"enabled": False},
        "cwd": os.getcwd(),
    }
    runtime = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    try:
        if args.landlock:
            isolation["landlock"] = apply_landlock(
                read_paths=default_landlock_read_paths(repo_root=repo_root, candidate_dir=candidate_dir),
                read_write_paths=[run_dir],
            )

        manifest, components = load_training_components(
            candidate_dir,
            expected_schema_version=args.expected_schema_version,
            allow_llm_authored=args.allow_llm_authored,
            task_spec={"phase": "8C-isolated-child"},
            base_env_cfg=None,
            base_agent_cfg=None,
        )
        if args.require_noop_components and not components.is_noop():
            raise CandidateLoadError("Phase 8C selftest required no-op components")
        fast_validation = None
        errors: list[dict[str, Any]] = []
        status = "pass"
        if args.fast_validation_config:
            fast_cfg = json.loads(Path(args.fast_validation_config).read_text(encoding="utf-8"))
            fast_validation = fast_validate_components(
                components,
                contract=reward_inputs_contract(),
                num_envs=int(fast_cfg.get("num_envs", 4)),
                num_joints=int(fast_cfg.get("num_joints", 29)),
                num_actions=int(fast_cfg.get("num_actions", 29)),
                progress_points=fast_cfg.get("progress_points", [0.0, 0.5, 1.0]),
                allowed_schedule_keys=fast_cfg.get("allowed_schedule_keys", None) or (
                    "push_magnitude_scale",
                    "push_start_fraction",
                    "push_interval_s",
                    "domain_randomization_scale",
                    "reward_weight_scale",
                ),
                prefer_torch=bool(fast_cfg.get("prefer_torch", True)),
            )
            if fast_validation.get("status") != "pass":
                status = "reject"
                errors.append({"type": "DynamicFastValidationFailed", "message": "candidate failed isolated dynamic fast validation"})

        summary = _summary(
            status,
            isolation=isolation,
            runtime=runtime,
            candidate=manifest.to_public_summary(),
            components=components.to_public_summary(),
            fast_validation=fast_validation,
            errors=errors,
        )
        exit_code = 0 if status == "pass" else 1
    except Exception as exc:  # noqa: BLE001 - child must return structured rejection.
        summary = _summary(
            "reject",
            isolation=isolation,
            runtime=runtime,
            candidate={"path": str(candidate_dir)},
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )
        if isinstance(exc, IsolationError):
            summary["status"] = "fail"
        exit_code = 1

    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_file), "status": summary["status"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

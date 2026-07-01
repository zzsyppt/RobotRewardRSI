#!/usr/bin/env python3
"""Phase 8E-1 trusted evaluation-process probe.

The trusted eval process consumes only a checkpoint and runner-owned public
metadata.  It must never receive or import a candidate package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.public_eval import (  # noqa: E402
    build_phase8e_public_result,
    load_json_mapping,
)
from humanoid_agent.immutable_runner.safe_checkpoint import checkpoint_safety_report  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--candidate-dir", default=None, help="Forbidden; present only for negative tests.")
    parser.add_argument(
        "--public-result-file",
        default=None,
        help="Write a Phase 8E public_result JSON. Requires at least one --public-dev-summary.",
    )
    parser.add_argument(
        "--public-dev-summary",
        action="append",
        default=[],
        help="Trusted public-dev rollout summary JSON. May be supplied multiple times.",
    )
    parser.add_argument(
        "--public-accept-summary",
        action="append",
        default=[],
        help="Trusted public-accept rollout summary JSON. May be supplied multiple times.",
    )
    parser.add_argument(
        "--budget-summary",
        default=None,
        help="Runner-owned train/budget summary used to populate the budget_accounting gate.",
    )
    return parser


def _summary(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.trusted_eval_probe.summary.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **kwargs,
    }


def _candidate_module_loaded() -> bool:
    loaded_names = set(sys.modules)
    return any(name == "training_code" or name.startswith("training_code.") for name in loaded_names)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_file": str(path), "status": summary["status"]}, ensure_ascii=False))


def _real_public_eval_requested(args: argparse.Namespace) -> bool:
    return bool(args.public_result_file or args.public_dev_summary or args.public_accept_summary)


def main() -> int:
    args = _build_parser().parse_args()
    summary_file = Path(args.summary_file).resolve()
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()

    if args.candidate_dir:
        summary = _summary(
            "reject",
            process_role="trusted_eval_process",
            reason="candidate_dir_argument_forbidden",
            candidate_code_loaded=_candidate_module_loaded(),
            inputs={"checkpoint": str(checkpoint), "candidate_dir_supplied": True},
        )
        _write_summary(summary_file, summary)
        return 1

    if not checkpoint.is_file():
        summary = _summary(
            "reject",
            process_role="trusted_eval_process",
            reason="checkpoint_missing",
            candidate_code_loaded=_candidate_module_loaded(),
            inputs={"checkpoint": str(checkpoint), "candidate_dir_supplied": False},
        )
        _write_summary(summary_file, summary)
        return 1

    # AUTHORITATIVE checkpoint-safety gate.  This process never loads candidate
    # code, so a candidate cannot monkeypatch this check (unlike the advisory
    # check in the training process).  A candidate-tampered checkpoint pickle
    # would execute under RSL-RL's weights_only=False loader, so we refuse to
    # accept any checkpoint that does not load under weights_only=True.
    safety = checkpoint_safety_report(checkpoint)
    if not safety["safe_load"]:
        summary = _summary(
            "reject",
            process_role="trusted_eval_process",
            reason="checkpoint_failed_safe_load_gate",
            candidate_code_loaded=_candidate_module_loaded(),
            checkpoint_safety=safety,
            inputs={"checkpoint": str(checkpoint), "candidate_dir_supplied": False},
        )
        _write_summary(summary_file, summary)
        return 1

    if _real_public_eval_requested(args):
        if not args.public_result_file:
            summary = _summary(
                "reject",
                process_role="trusted_eval_process",
                reason="public_result_file_required",
                candidate_code_loaded=_candidate_module_loaded(),
                checkpoint_safety=safety,
            )
            _write_summary(summary_file, summary)
            return 1
        if not args.public_dev_summary:
            summary = _summary(
                "reject",
                process_role="trusted_eval_process",
                reason="public_dev_summary_required",
                candidate_code_loaded=_candidate_module_loaded(),
                checkpoint_safety=safety,
            )
            _write_summary(summary_file, summary)
            return 1
        public_result_file = Path(args.public_result_file).resolve()
        public_result_file.parent.mkdir(parents=True, exist_ok=True)
        public_result = build_phase8e_public_result(
            public_dev_summaries=[load_json_mapping(path) for path in args.public_dev_summary],
            public_accept_summaries=[load_json_mapping(path) for path in args.public_accept_summary],
            budget_summary=load_json_mapping(args.budget_summary) if args.budget_summary else None,
        )
        public_result_file.write_text(json.dumps(public_result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        summary = _summary(
            "pass",
            process_role="trusted_eval_process",
            candidate_code_allowed=False,
            candidate_code_loaded=_candidate_module_loaded(),
            checkpoint_safety=safety,
            inputs={
                "checkpoint": str(checkpoint),
                "candidate_dir_supplied": False,
                "checkpoint_only": True,
            },
            eval_contract={
                "frozen_observation_interface": True,
                "frozen_action_interface": True,
                "immutable_success_metric": True,
                "candidate_code_executed": False,
                "public_result_only": True,
            },
            public_eval={
                "status": public_result["status"],
                "public_result_file": str(public_result_file),
                "public_dev_runs": len(args.public_dev_summary),
                "public_accept_runs": len(args.public_accept_summary),
                "budget_gate_status": public_result["budget_accounting"]["status"],
                "motion_quality_gate_status": public_result["motion_quality_gate"]["status"],
                "ood_stress_gate_status": public_result["ood_stress_gate"]["status"],
            },
            errors=[],
        )
        _write_summary(summary_file, summary)
        return 0

    summary = _summary(
        "pass",
        process_role="trusted_eval_process",
        candidate_code_allowed=False,
        candidate_code_loaded=_candidate_module_loaded(),
        checkpoint_safety=safety,
        inputs={
            "checkpoint": str(checkpoint),
            "candidate_dir_supplied": False,
            "checkpoint_only": True,
        },
        eval_contract={
            "frozen_observation_interface": True,
            "frozen_action_interface": True,
            "immutable_success_metric": True,
            "candidate_code_executed": False,
            "public_result_only": True,
        },
        public_eval_placeholder={
            "status": "not_run",
            "reason": "Phase8E-1 probes the trusted-eval boundary only; formal eval is added later.",
        },
        errors=[],
    )
    _write_summary(summary_file, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

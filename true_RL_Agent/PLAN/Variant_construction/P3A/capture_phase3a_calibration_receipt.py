"""Run the pre-registered C1 terrain calibration on one fixed clean policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3a_calibration_receipt.v1"
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_text_hash(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _captured_head(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _resolve_irregular_cell(project_root: Path) -> tuple[Any, Any]:
    from true_rl_agent.tasks.stair_climbing.config import (
        STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        STAIR_PROTOCOL_V2_RELATIVE_PATH,
        StairClimbingTaskConfig,
    )
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import (
        resolve_stair_cell_manifest,
    )

    protocol_path = project_root / STAIR_PROTOCOL_V2_RELATIVE_PATH
    provisional = StairClimbingTaskConfig.defaults(
        protocol_path,
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
    )
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        config=provisional,
        plugin=StairClimbingTaskPlugin(provisional),
    )
    config = StairClimbingTaskConfig.defaults(
        protocol_path,
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
    )
    final_manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        config=config,
        plugin=StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest),
    )
    if final_manifest.to_dict() != manifest.to_dict():
        raise RuntimeError("C1 manifest changed after resolved identity application")
    return config, final_manifest


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve(strict=True)
    if output.exists():
        raise FileExistsError("Phase 3A calibration receipt path must be fresh")
    sys.path.insert(0, str(project_root / "src"))

    from true_rl_agent.backends.isaac_lab_execution_environment import (
        build_current_isaac_lab_execution_environment_manifest,
        verify_current_isaac_lab_execution_environment,
    )
    from true_rl_agent.runner.checkpoint import build_checkpoint_safety_report
    from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
        execute_stair_evaluation,
        preflight_stair_execution_source_manifest,
    )
    from true_rl_agent.tasks.stair_climbing.seed_policy import (
        STAIR_CALIBRATION_EVAL_NAMESPACE,
    )

    config, manifest = _resolve_irregular_cell(project_root)
    protocol = config.irregular_calibration_protocol
    checkpoint_report = build_checkpoint_safety_report(
        checkpoint,
        expected_checkpoint_hash=protocol.anchor_checkpoint_sha256,
    ).to_public_dict()
    if checkpoint_report["status"] != "pass":
        raise ValueError(str(checkpoint_report["reason"]))
    source_preflight = preflight_stair_execution_source_manifest(config)
    backend_manifest = build_current_isaac_lab_execution_environment_manifest()
    backend_verification = verify_current_isaac_lab_execution_environment(
        str(backend_manifest.manifest_hash)
    )

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=args.device)
    _ = launcher.app
    execution = execute_stair_evaluation(
        config=config,
        source_manifest_preflight=source_preflight,
        backend_execution_environment_verification=backend_verification,
        checkpoint_path=checkpoint,
        device=args.device,
        seed_namespace=STAIR_CALIBRATION_EVAL_NAMESPACE,
        splits=("calibration",),
        resolved_cell_manifest=manifest,
    )
    summary = execution.public_summaries()["calibration"]
    qualification = protocol.qualify_summary(
        suite=config.ordered_evaluation_suite,
        semantics=config.evaluation_semantics,
        summary=summary,
        checkpoint_sha256=execution.checkpoint_hash,
    )
    source_paths = (
        "configs/robots/unitree_g1_37dof_isaaclab_v2.json",
        "src/true_rl_agent/robots/defaults.py",
        "src/true_rl_agent/tasks/stair_climbing/irregular_calibration.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_summary.py",
        "src/true_rl_agent/tasks/stair_climbing/seed_policy.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "configs/tasks/stair_g1_37dof_irregular_calibration_protocol_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_evaluation_suite_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_acceptance_protocol_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_reference_gate_v1.json",
        "PLAN/Variant_construction/P3A/capture_phase3a_calibration_receipt.py",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "captured_repository_head": _captured_head(project_root),
        "head_identity_semantics": "provenance only; normalized source bytes bind execution",
        "cell_variant_id": manifest.cell_variant_id,
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "resolved_cell_task_protocol_hash": (
            manifest.resolved_cell_task_protocol_hash
        ),
        "calibration_protocol_hash": protocol.protocol_hash,
        "checkpoint_safety_report": checkpoint_report,
        "execution_identity": execution.execution_identity(),
        "execution_cost": execution.execution_cost_artifact(),
        "calibration_summary": summary,
        "terrain_qualification": qualification,
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": (
            "pass"
            if qualification["terrain_profile_status"] == "pass"
            else "fail"
        ),
    }
    receipt["receipt_hash"] = _canonical_hash(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_hash": receipt["receipt_hash"],
                "terrain_qualification": qualification,
                "execution_cost": execution.execution_cost_artifact(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        receipt = _capture(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    os._exit(0 if receipt["status"] == "pass" else 2)


if __name__ == "__main__":
    raise SystemExit(main())

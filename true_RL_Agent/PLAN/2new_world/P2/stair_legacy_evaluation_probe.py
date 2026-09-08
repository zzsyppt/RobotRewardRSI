"""Read-only P2 comparison using the frozen pre-P2 checkout and interpreter.

This is acceptance evidence, not a production launcher or training fallback.
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / "src"))
from isaaclab.app import AppLauncher
from true_rl_agent.backends.isaac_lab_execution_environment import (
    build_current_isaac_lab_execution_environment_manifest,
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    execute_stair_evaluation,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE

filename = (
    "stair_unified_protocol_v2.json"
    if args.robot == "g1"
    else "stair_go2_12dof_unified_protocol_v2.json"
)
config = StairClimbingTaskConfig.defaults(args.source_root / "configs/tasks" / filename)
cell = resolve_stair_cell_manifest(
    cell_variant_id=config.cell_variant_id,
    config=config,
    plugin=StairClimbingTaskPlugin(config),
)
config = replace(
    config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash
)
environment = build_current_isaac_lab_execution_environment_manifest()
verification = verify_current_isaac_lab_execution_environment(environment.manifest_hash)
preflight = preflight_stair_execution_source_manifest(config)
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(
    headless=True,
    device=args.device,
    multi_gpu=False,
    kit_args="--/log/file=" + str((args.output / "result.kit.log").resolve()),
).app
try:
    result = execute_stair_evaluation(
        config=config,
        source_manifest_preflight=preflight,
        backend_execution_environment_verification=verification,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
        splits=("public_dev",),
        resolved_cell_manifest=cell,
    )
    report = {
        "status": "PASS",
        "robot": args.robot,
        "environment": environment.to_dict(),
        "execution": result.execution_identity(),
        "summary": result.public_summaries()["public_dev"],
        "vector_env_step_calls": dict(result.split_vector_env_step_calls),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", flush=True)
finally:
    app.close()

"""P2 diagnostic and full-budget runs through the production Stair executor."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher
from true_rl_agent.backends.isaac_lab_execution_environment import (
    build_current_isaac_lab_execution_environment_manifest,
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.runner.isaac_app_lifecycle import (
    bind_isaac_log_path,
    close_isaac_app_on_success,
)
from true_rl_agent.runner.native_video import (
    add_native_video_arguments,
    native_video_options_from_args,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    execute_stair_training,
    execute_stair_training_smoke,
    execute_stair_evaluation,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument(
    "--mode", choices=("smoke", "train", "eval", "profile"), required=True
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
add_native_video_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
video = native_video_options_from_args(args)
args.output.mkdir(parents=True, exist_ok=True)
filename = (
    "stair_unified_protocol_v2.json"
    if args.robot == "g1"
    else "stair_go2_12dof_unified_protocol_v2.json"
)
config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / filename)
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
official = resolve_official_stair_reference_program(
    config, project_root=ROOT, resolved_cell_manifest=cell
)
report = {
    "mode": args.mode,
    "robot": args.robot,
    "device": args.device,
    "environment": environment.to_dict(),
    "cell": cell.to_dict(),
}
bind_isaac_log_path(args, args.output / "result.json")
app = AppLauncher(args, multi_gpu=False).app
code = 1
try:
    if args.mode == "eval":
        result = execute_stair_evaluation(
            config=config,
            source_manifest_preflight=preflight,
            backend_execution_environment_verification=verification,
            checkpoint_path=args.checkpoint,
            device=args.device,
            seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            splits=("public_dev",),
            resolved_cell_manifest=cell,
            native_video=video,
        )
        report["summary"] = result.public_summaries()["public_dev"]
        report["vector_env_step_calls"] = dict(result.split_vector_env_step_calls)
    else:
        run = (
            execute_stair_training
            if args.mode == "train"
            else execute_stair_training_smoke
        )
        kwargs = dict(
            config=config,
            source_manifest_preflight=preflight,
            agent_configuration=config.training_agent_configuration(),
            reward_program=official.reward_program,
            device=args.device,
            run_dir=args.output / "run",
            checkpoint_output=args.output / "best.pt",
            run_name=args.output.name,
            resolved_cell_manifest=cell,
        )
        if args.mode == "profile":
            import cProfile
            from unittest.mock import patch
            from rsl_rl.runners import OnPolicyRunner
            from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
                _execute_stair_training,
                _StairTrainingWorkload,
            )

            kwargs.pop("source_manifest_preflight")
            kwargs.update(
                expected_initial_policy_state_hash=None,
                workload=_StairTrainingWorkload(
                    kind="phase_d_smoke",
                    num_envs=config.train_num_envs,
                    num_steps_per_env=config.train_steps_per_iteration,
                    max_iterations=2,
                    budget_steps=config.train_num_envs
                    * config.train_steps_per_iteration
                    * 2,
                    checkpoint_save_interval=1,
                ),
            )
            profiler = cProfile.Profile()
            learn = OnPolicyRunner.learn

            def profiled_learn(*a, **k):
                return profiler.runcall(learn, *a, **k)

            with patch.object(OnPolicyRunner, "learn", profiled_learn):
                result = _execute_stair_training(**kwargs)
            profiler.dump_stats(str(args.output / "runtime.pstats"))
        else:
            result = run(**kwargs)
        report["training_summary"] = result.to_summary_fields()
        if result.status != "pass":
            raise RuntimeError(str(result.to_summary_fields()))
    report["status"] = "PASS"
    code = 0
except Exception:
    report["status"] = "FAIL"
    report["error"] = traceback.format_exc()
    traceback.print_exc()
finally:
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}), flush=True
    )
    sys.exit(close_isaac_app_on_success(app, exit_code=code))

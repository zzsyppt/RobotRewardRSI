"""Dump actual task environment configs for the old/new backend audit."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:1")
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / "src"))
from isaaclab.app import AppLauncher

args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(headless=True, device=args.device, multi_gpu=False,
    kit_args="--/log/file=" + str(args.output / "config.kit.log")).app
code = 1
try:
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
    from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
    from true_rl_agent.tasks.stair_climbing.reference_program import resolve_official_stair_reference_program
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import build_stair_train_env_cfg, build_stair_eval_env_cfg

    def normalize(value):
        if isinstance(value, dict):
            return {str(k): normalize(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)):
            return [normalize(v) for v in value]
        if callable(value):
            return value.__module__ + ":" + value.__qualname__
        if isinstance(value, (str,int,float,bool)) or value is None:
            return value
        return str(value)

    for robot in ("g1", "go2"):
        filename = "stair_unified_protocol_v2.json" if robot == "g1" else "stair_go2_12dof_unified_protocol_v2.json"
        config = StairClimbingTaskConfig.defaults(args.source_root / "configs/tasks" / filename)
        cell = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
        config = replace(config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash)
        projection = stair_runtime_projection_from_manifest(cell)
        official = resolve_official_stair_reference_program(config, project_root=args.source_root, resolved_cell_manifest=cell)
        train = build_stair_train_env_cfg(None, seed=config.train_seed, device=args.device, reward_program=official.reward_program,
            unified_protocol=config.unified_protocol, runtime_projection=projection)
        evaluation = build_stair_eval_env_cfg(None, seed=config.train_seed, device=args.device, runtime_projection=projection,
            unified_protocol=config.unified_protocol, ordered_evaluation_suite=config.ordered_evaluation_suite, evaluation_split="public_dev")
        for mode,cfg in (("train",train),("eval",evaluation)):
            (args.output / f"{robot}_{mode}.json").write_text(json.dumps(normalize(cfg.to_dict()),indent=2,sort_keys=True)+"\n")
    print("PASS",flush=True)
    code=0
finally:
    if code==0:
        app.close()
    sys.exit(code)

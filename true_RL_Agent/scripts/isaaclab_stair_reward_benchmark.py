#!/usr/bin/env python3
"""Measure one isolated Stair reward-runtime benchmark case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_performance import (  # noqa: E402
    STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION,
    STAIR_REWARD_BENCHMARK_CASES,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    case_action = parser.add_argument("--case", choices=STAIR_REWARD_BENCHMARK_CASES)
    output_action = parser.add_argument("--output")
    AppLauncher.add_app_launcher_args(parser)
    case_action.required = True
    output_action.required = True
    parser.set_defaults(headless=True)
    return parser


args = _build_parser().parse_args()
config = StairClimbingTaskConfig.defaults(
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
)
official = resolve_official_stair_reference_program(
    config,
    project_root=PROJECT_ROOT,
)
simulation_app = AppLauncher(args).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import (  # noqa: E402
    build_stair_env_cfg,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (  # noqa: E402
    get_stair_complete_reward_term,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (  # noqa: E402
    StairEvalRecorderManagerCfg,
    build_stair_train_env_cfg,
    register_production_stair_envs,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_TRAIN_GYM_ID,
)


def _build_case_cfg() -> object:
    protocol = config.protocol_v1.to_dict()
    if args.case == "official":
        cfg = build_stair_env_cfg(
            protocol,
            mode="train",
            seed=config.train_seed,
            device=str(args.device),
        )
        cfg.recorders = None
        return cfg
    cfg = build_stair_train_env_cfg(
        protocol,
        seed=config.train_seed,
        device=str(args.device),
        reward_program=official.reward_program,
    )
    if args.case == "dispatcher_with_eval_recorder":
        cfg.recorders = StairEvalRecorderManagerCfg()
    return cfg


def main() -> int:
    env = None
    output_path = Path(args.output)
    report: dict[str, object]
    try:
        performance = config.protocol_v2.complete_reward_runtime.to_public_dict()[
            "performance_gate"
        ]
        num_envs = int(performance["num_envs"])
        warmup_steps = int(performance["warmup_control_steps"])
        measured_steps = int(performance["measured_control_steps"])
        register_production_stair_envs()
        env_cfg = _build_case_cfg()
        env_cfg.scene.num_envs = num_envs
        env = gym.make(STAIR_TRAIN_GYM_ID, cfg=env_cfg, render_mode=None)
        env.reset()
        base = env.unwrapped
        action_dim = int(
            config.protocol_v1.to_dict()["policy_compatibility"]["action"][
                "total_dim"
            ]
        )
        actions = torch.zeros(
            (num_envs, action_dim),
            dtype=torch.float32,
            device=base.device,
        )
        for _ in range(warmup_steps):
            env.step(actions)
        torch.cuda.synchronize(device=base.device)
        started = time.perf_counter()
        for _ in range(measured_steps):
            env.step(actions)
        torch.cuda.synchronize(device=base.device)
        elapsed = time.perf_counter() - started
        term = None if args.case == "official" else get_stair_complete_reward_term(base)
        runtime = {} if term is None else term.runtime_summary()
        program = None if term is None else term.program.to_public_dict()
        active_terms = list(base.reward_manager.active_terms)
        if args.case == "official":
            expected_terms = set(
                config.protocol_v2.complete_reward_runtime.to_public_dict()[
                    "reference_equivalence"
                ]["active_term_names"]
            ) | set(
                config.protocol_v2.complete_reward_runtime.to_public_dict()[
                    "reference_equivalence"
                ]["zero_weight_term_names"]
            )
            if set(active_terms) != expected_terms:
                raise RuntimeError("official benchmark reward term set drifted")
        elif active_terms != ["complete_reward"]:
            raise RuntimeError("dispatcher benchmark has multiple reward authorities")
        recorder_enabled = env_cfg.recorders is not None
        expected_recorder = args.case == "dispatcher_with_eval_recorder"
        if recorder_enabled is not expected_recorder:
            raise RuntimeError("benchmark recorder state differs from its case")
        report = {
            "schema_version": STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION,
            "status": "pass",
            "case": args.case,
            "protocol_v2_hash": config.protocol_v2.protocol_hash,
            "num_envs": num_envs,
            "warmup_control_steps": warmup_steps,
            "measured_control_steps": measured_steps,
            "elapsed_seconds": round(elapsed, 8),
            "env_steps_per_second": round(num_envs * measured_steps / elapsed, 3),
            "reward_manager_active_terms": active_terms,
            "training_recorder_enabled": recorder_enabled,
            "program": program,
            "runtime": runtime,
            "hardware_fingerprint": {
                "device": str(base.device),
                "gpu_name": torch.cuda.get_device_name(base.device),
                "torch": torch.__version__,
                "simulation_dt_s": float(base.physics_dt),
                "control_dt_s": float(base.step_dt),
            },
        }
        report["report_hash"] = canonical_json_hash(report)
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema_version": STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION,
            "status": "fail",
            "case": args.case,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "status": report["status"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    exit_code = close_isaac_app_on_success(
        simulation_app,
        exit_code=main(),
    )
    os._exit(exit_code)

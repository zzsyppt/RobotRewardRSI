#!/usr/bin/env python3
"""Train one complete Stair reward program with Isaac Lab and RSL-RL."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.runner.checkpoint import file_sha256  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    StairTrainEntrypointContext,
    validate_stair_train_invocation,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_training import (  # noqa: E402
    configure_stair_rsl_rl_agent,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    QualifiedStairRewardProgram,
    qualify_stair_candidate_reward_program,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
    STAIR_TRAIN_GYM_ID,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    required_actions = []
    for option in (
        "--job-id",
        "--round",
        "--task-id",
        "--subject-kind",
        "--subject-id",
        "--subject-manifest-hash",
        "--program-source-checksum",
        "--candidate",
        "--candidate-artifact-hash",
        "--init",
        "--training-binding-hash",
        "--policy-architecture-hash",
        "--optimizer-training-profile-hash",
        "--agent-configuration-hash",
        "--agent-configuration-json",
        "--initialization-seed-schedule-hash",
        "--capability-profile-hash",
        "--summary-path",
        "--checkpoint-output",
        "--profile",
        "--runtime-dynamics-hash",
        "--config",
    ):
        required_actions.append(parser.add_argument(option))
    for option in (
        "--budget-steps",
        "--num-envs",
        "--num-steps-per-env",
        "--max-iterations",
        "--seed",
    ):
        required_actions.append(parser.add_argument(option, type=int))
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-checkpoint-sha256", default=None)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    else:
        AppLauncher.add_app_launcher_args(parser)
    for action in required_actions:
        action.required = True
    parser.set_defaults(headless=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        context = validate_stair_train_invocation(args, project_root=PROJECT_ROOT)
        if args.headless is not True:
            raise ValueError("stair production train must be headless")
        reward_program = _load_and_qualify_candidate(context)
        summary_path = _local_output_path(args.summary_path, "summary_path")
        checkpoint_output = _local_output_path(
            args.checkpoint_output,
            "checkpoint_output",
        )
    except (TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    exit_code = _run_training(
        args=args,
        context=context,
        reward_program=reward_program,
        summary_path=summary_path,
        checkpoint_output=checkpoint_output,
    )
    return close_isaac_app_on_success(
        simulation_app,
        exit_code=exit_code,
    )


def _load_and_qualify_candidate(
    context: StairTrainEntrypointContext,
) -> QualifiedStairRewardProgram:
    return qualify_stair_candidate_reward_program(
        artifact=context.candidate_artifact,
        contract=context.config.protocol_v2.complete_reward_runtime,
    )


def _run_training(
    *,
    args: argparse.Namespace,
    context: StairTrainEntrypointContext,
    reward_program: QualifiedStairRewardProgram,
    summary_path: Path,
    checkpoint_output: Path,
) -> int:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
        G1RoughPPORunnerCfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
        get_stair_complete_reward_term,
        run_stair_reward_activation_preflight,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_train_env_cfg,
        register_production_stair_envs,
    )

    attempt_id = uuid.uuid4().hex
    run_dir = summary_path.parent / "run" / f"attempt-{attempt_id}"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = None
    complete_term = None
    formal_training_executed = False
    activation_preflight: dict[str, Any] | None = None
    try:
        if checkpoint_output.exists():
            raise FileExistsError(
                "checkpoint output must be fresh for this Stair training attempt"
            )
        draft = context.config.protocol_v2.to_public_dict()
        execution = draft["training_execution"]
        torch.backends.cuda.matmul.allow_tf32 = bool(
            execution["torch_matmul_allow_tf32"]
        )
        torch.backends.cudnn.allow_tf32 = bool(execution["cudnn_allow_tf32"])
        torch.backends.cudnn.deterministic = bool(
            execution["cudnn_deterministic"]
        )
        torch.backends.cudnn.benchmark = bool(execution["cudnn_benchmark"])

        register_production_stair_envs()
        env_cfg = build_stair_train_env_cfg(
            context.config.protocol_v1.to_dict(),
            seed=context.config.train_seed,
            device=str(args.device),
            reward_program=reward_program,
        )
        if env_cfg.recorders is not None:
            raise RuntimeError("Stair training must not install a recorder")
        agent_cfg = G1RoughPPORunnerCfg()
        effective_agent_execution = configure_stair_rsl_rl_agent(
            agent_cfg,
            configuration=context.agent_configuration,
            seed=context.config.train_seed,
            device=str(args.device),
            num_steps_per_env=context.config.train_steps_per_iteration,
            max_iterations=context.config.train_max_iterations,
            save_interval=int(
                execution["checkpoint_save_interval_iterations"]
            ),
            run_name=f"true_rl_agent_{args.round}",
            experiment_name="true_rl_agent_stair",
        )

        env = gym.make(STAIR_TRAIN_GYM_ID, cfg=env_cfg, render_mode=None)
        env.reset()
        base_env = env.unwrapped
        if list(base_env.reward_manager.active_terms) != ["complete_reward"]:
            raise RuntimeError("Stair RewardManager contains non-authoritative terms")
        term_cfg = base_env.reward_manager.get_term_cfg("complete_reward")
        if float(term_cfg.weight) != 1.0:
            raise RuntimeError("Stair complete reward manager weight drifted")
        complete_term = get_stair_complete_reward_term(base_env)
        activation_preflight = run_stair_reward_activation_preflight(
            env,
            reward_program=reward_program,
        )

        wrapped_env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )
        log_dir = run_dir / "rsl_rl"
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        formal_training_executed = True
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=bool(
                execution["init_at_random_episode_length"]
            ),
        )
        latest_checkpoint = _find_latest_checkpoint(log_dir)
        if latest_checkpoint is None:
            raise RuntimeError("Stair RSL-RL training produced no checkpoint")
        if execution["checkpoint_selector"] != "final_iteration":
            raise RuntimeError("unsupported Stair checkpoint selector")
        shutil.copy2(latest_checkpoint, checkpoint_output)
        source_hash = file_sha256(latest_checkpoint)
        checkpoint_hash = file_sha256(checkpoint_output)
        if source_hash is None or checkpoint_hash is None:
            raise RuntimeError("Stair checkpoint bytes could not be hashed")
        if source_hash != checkpoint_hash:
            raise RuntimeError("Stair checkpoint copy hash mismatch")
        runtime_activation = complete_term.runtime_summary()
        if runtime_activation["dispatcher_calls"] <= 0:
            raise RuntimeError("Stair complete reward dispatcher never executed")
        summary = _status_summary(
            "pass",
            args=args,
            context=context,
            reward_program=reward_program,
            attempt_id=attempt_id,
            formal_training_executed=True,
            checkpoint={
                "uri": str(checkpoint_output),
                "checksum": checkpoint_hash,
                "role": "train_output",
                "metadata": {
                    "local_path": str(checkpoint_output),
                    "source_latest_checkpoint": str(latest_checkpoint),
                    "source_latest_checkpoint_sha256": source_hash,
                },
            },
            runtime_activation_summary={
                **runtime_activation,
                "runtime_status": STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
                "activation_preflight": activation_preflight,
                "reward_manager_active_terms": list(
                    base_env.reward_manager.active_terms
                ),
                "reward_manager_term_weight": float(term_cfg.weight),
                "training_recorder_enabled": False,
            },
            training={
                "budget_accounting": dict(context.budget_accounting),
                "num_envs": context.config.train_num_envs,
                "num_steps_per_env": context.config.train_steps_per_iteration,
                "max_iterations": context.config.train_max_iterations,
                "budget_steps": context.config.budget_steps,
                "device": str(agent_cfg.device),
                "log_dir": str(log_dir),
                "latest_checkpoint_iteration": _checkpoint_iteration(
                    latest_checkpoint
                ),
                "execution_contract": dict(execution),
                "effective_agent_execution": effective_agent_execution,
            },
            errors=[],
        )
    except Exception as exc:  # noqa: BLE001 - one failed guard fails the job
        runtime_summary = (
            complete_term.runtime_summary() if complete_term is not None else None
        )
        summary = _status_summary(
            "fail",
            args=args,
            context=context,
            reward_program=reward_program,
            attempt_id=attempt_id,
            formal_training_executed=formal_training_executed,
            runtime_activation_summary=runtime_summary,
            activation_preflight=activation_preflight,
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    _write_json(summary_path, summary)
    print(
        json.dumps(
            {"summary_path": str(summary_path), "status": summary["status"]},
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


def _status_summary(
    status: str,
    *,
    args: argparse.Namespace,
    context: StairTrainEntrypointContext,
    reward_program: QualifiedStairRewardProgram,
    attempt_id: str,
    formal_training_executed: bool,
    **payload: Any,
) -> dict[str, Any]:
    binding = context.training_binding
    return {
        **payload,
        "schema_version": "true_rl_agent.task.stair_climbing.train_summary.v1",
        **context.identity.to_public_dict(),
        "training_binding_hash": binding.binding_hash,
        "policy_architecture_hash": binding.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            binding.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": binding.agent_configuration_hash,
        "initialization_seed_schedule_hash": (
            binding.initialization_seed_schedule_hash
        ),
        "capability_profile_hash": binding.capability_profile_hash,
        "program_source_checksum": reward_program.program_source_checksum,
        "reward_program_qualification_hash": reward_program.qualification_hash,
        "status": status,
        "formal_training_executed": formal_training_executed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "cwd": os.getcwd(),
            "round_id": args.round,
            "train_attempt_id": attempt_id,
            "seed_policy": "runner_owned_not_public",
        },
    }


def _find_latest_checkpoint(log_dir: Path) -> Path | None:
    checkpoints = sorted(
        log_dir.rglob("model_*.pt"),
        key=lambda path: (
            _checkpoint_iteration(path),
            path.stat().st_mtime,
        ),
    )
    return checkpoints[-1] if checkpoints else None


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _local_output_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field_name} must be a local path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

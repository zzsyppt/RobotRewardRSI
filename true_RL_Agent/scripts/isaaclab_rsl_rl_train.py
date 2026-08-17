#!/usr/bin/env python3
"""Train a push recovery candidate with IsaacLab + RSL-RL."""

from __future__ import annotations

import argparse
import inspect
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

from isaaclab.app import AppLauncher  # type: ignore[import-not-found]  # noqa: E402

from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.backends.rsl_rl_training_config import (  # noqa: E402
    RslRlAgentConfiguration,
)
from true_rl_agent.candidates.artifact import (  # noqa: E402
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.schema import CandidateChangeSet  # noqa: E402
from true_rl_agent.evaluation.binding import (
    require_sha256_checksum,  # noqa: E402
    require_sha256_digest,  # noqa: E402
)
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.runner.job_identity import JobIdentity  # noqa: E402
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.push_recovery.candidate_failure import (  # noqa: E402
    PUSH_CANDIDATE_FAILURE_CONTRACT,
    PushCandidateProgramFailure,
)
from true_rl_agent.training_binding import (  # noqa: E402
    initialization_seed_schedule_hash,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--subject-kind", required=True, choices=("candidate",))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subject-manifest-hash", required=True)
    parser.add_argument("--program-source-checksum", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-artifact-hash", required=True)
    parser.add_argument("--budget-steps", type=int, required=True)
    parser.add_argument("--init", required=True, choices=("random", "warm_start"))
    parser.add_argument("--training-binding-hash", required=True)
    parser.add_argument("--policy-architecture-hash", required=True)
    parser.add_argument("--optimizer-training-profile-hash", required=True)
    parser.add_argument("--agent-configuration-hash", required=True)
    parser.add_argument("--agent-configuration-json", required=True)
    parser.add_argument("--initialization-seed-schedule-hash", required=True)
    parser.add_argument("--capability-profile-hash", required=True)
    parser.add_argument("--candidate-failure-contract-hash", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-checkpoint-sha256", default=None)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--checkpoint-output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="strong_push")
    parser.add_argument("--runtime-dynamics-hash", required=True)
    parser.add_argument(
        "--backend-execution-environment-manifest-hash",
        default=None,
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--num-steps-per-env",
        "--train-steps-per-iteration",
        dest="num_steps_per_env",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--candidate-diagnostic-interval-steps", type=int, default=24)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


args_cli = _build_parser().parse_args()
args_cli.device = require_execution_device(
    args_cli.device,
    "push train execution device",
)
job_identity = JobIdentity(
    job_id=args_cli.job_id,
    task_id=args_cli.task_id,
    subject=ExperimentSubject(
        kind=args_cli.subject_kind,
        subject_id=args_cli.subject_id,
    ),
    subject_manifest_hash=args_cli.subject_manifest_hash,
)
program_source_checksum = require_sha256_checksum(
    args_cli.program_source_checksum,
    "program source checksum",
)
training_binding_hash = require_sha256_digest(
    args_cli.training_binding_hash,
    "training binding hash",
)
capability_profile_hash = require_sha256_digest(
    args_cli.capability_profile_hash,
    "capability profile hash",
)
candidate_failure_contract_hash = require_sha256_digest(
    args_cli.candidate_failure_contract_hash,
    "push candidate failure contract hash",
)
if candidate_failure_contract_hash != PUSH_CANDIDATE_FAILURE_CONTRACT.contract_hash:
    raise ValueError("push candidate failure contract hash mismatch")
policy_architecture_hash = require_sha256_digest(
    args_cli.policy_architecture_hash,
    "policy architecture hash",
)
optimizer_training_profile_hash = require_sha256_digest(
    args_cli.optimizer_training_profile_hash,
    "optimizer training profile hash",
)
agent_configuration_hash = require_sha256_digest(
    args_cli.agent_configuration_hash,
    "agent configuration hash",
)
seed_schedule_hash = require_sha256_digest(
    args_cli.initialization_seed_schedule_hash,
    "initialization seed schedule hash",
)
training_agent_configuration = RslRlAgentConfiguration.from_json(
    args_cli.agent_configuration_json
)
if training_agent_configuration.policy_architecture_hash != policy_architecture_hash:
    raise ValueError("policy architecture hash does not match agent configuration")
if (
    training_agent_configuration.optimizer_training_profile_hash
    != optimizer_training_profile_hash
):
    raise ValueError(
        "optimizer training profile hash does not match agent configuration"
    )
if training_agent_configuration.agent_configuration_hash != agent_configuration_hash:
    raise ValueError("agent configuration hash does not match payload")
if initialization_seed_schedule_hash(args_cli.seed) != seed_schedule_hash:
    raise ValueError("training seed does not match initialization seed schedule")
if "://" in args_cli.candidate:
    raise ValueError("train execution requires a local candidate artifact")
candidate_dir = Path(args_cli.candidate).resolve(strict=True)
candidate_execution_artifact = load_sealed_candidate_artifact(
    candidate_dir,
    change_set=CandidateChangeSet(frozenset()),
    expected_candidate_id=job_identity.subject.subject_id,
    expected_artifact_hash=require_sha256_digest(
        args_cli.candidate_artifact_hash,
        "candidate artifact hash",
    ),
    expected_subject_manifest_hash=job_identity.subject_manifest_hash,
    expected_program_source_checksum=program_source_checksum,
)
backend_execution_environment_verification = (
    verify_current_isaac_lab_execution_environment(
        args_cli.backend_execution_environment_manifest_hash
    )
    if args_cli.backend_execution_environment_manifest_hash is not None
    else None
)
train_attempt_id = uuid.uuid4().hex
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # type: ignore[import-not-found]  # noqa: E402
import isaaclab_tasks  # type: ignore[import-not-found]  # noqa: F401,E402
import torch  # type: ignore[import-not-found]  # noqa: E402
import unitree_rl_lab.tasks  # type: ignore[import-not-found]  # noqa: F401,E402
from isaaclab.envs import (  # type: ignore[import-not-found]  # noqa: E402
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml  # type: ignore[import-not-found]  # noqa: E402
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,  # type: ignore[import-not-found]  # noqa: E402
)
from isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,  # type: ignore[import-not-found]  # noqa: E402
)
from rsl_rl.runners import (
    OnPolicyRunner,  # type: ignore[import-not-found]  # noqa: E402
)
from unitree_rl_lab.utils.export_deploy_cfg import (
    export_deploy_cfg,  # type: ignore[import-not-found]  # noqa: E402
)
from unitree_rl_lab.utils.parser_cfg import (
    parse_env_cfg,  # type: ignore[import-not-found]  # noqa: E402
)

from true_rl_agent.runner.budget import build_budget_accounting  # noqa: E402
from true_rl_agent.tasks.push_recovery.isaac_config import (  # noqa: E402
    config_for_profile,
    load_runtime_config,
    public_runtime_dynamics,
    sha256_file,
)
from true_rl_agent.tasks.push_recovery.isaac_overlay import (
    apply_push_recovery_overlay,  # noqa: E402
)
from true_rl_agent.tasks.push_recovery.runtime_adapter import (  # noqa: E402
    CandidateRewardVecEnvWrapper,
    CandidateRuntimeAdapter,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main() -> int:
    summary_path = Path(args_cli.summary_path)
    checkpoint_output = Path(args_cli.checkpoint_output)
    run_root = Path(args_cli.run_dir) if args_cli.run_dir else summary_path.parent / "run"
    run_dir = run_root / f"attempt-{train_attempt_id}"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = None
    candidate_adapter = None
    candidate_summary = None
    try:
        if checkpoint_output.exists():
            raise FileExistsError(
                f"checkpoint output must be fresh for this training attempt: {checkpoint_output}"
            )
        if args_cli.init == "warm_start" and not args_cli.init_checkpoint:
            raise ValueError("--init-checkpoint is required for warm_start")
        if args_cli.init == "warm_start" and not args_cli.init_checkpoint_sha256:
            raise ValueError("--init-checkpoint-sha256 is required for warm_start")
        if args_cli.init == "warm_start":
            actual_init_checksum = f"sha256:{sha256_file(args_cli.init_checkpoint)}"
            if actual_init_checksum != args_cli.init_checkpoint_sha256:
                raise ValueError(
                    "warm-start checkpoint checksum mismatch: "
                    f"expected={args_cli.init_checkpoint_sha256!r} "
                    f"actual={actual_init_checksum!r}"
                )
        config = config_for_profile(load_runtime_config(args_cli.config), args_cli.profile)
        runtime_dynamics = public_runtime_dynamics(config)
        if runtime_dynamics["runtime_dynamics_hash"] != args_cli.runtime_dynamics_hash:
            raise ValueError(
                "runtime dynamics hash mismatch: "
                f"expected={args_cli.runtime_dynamics_hash!r} "
                f"actual={runtime_dynamics['runtime_dynamics_hash']!r}"
            )
        task = args_cli.task or config["task"]["gym_id"]
        args_cli.task = task
        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="env_cfg_entry_point")
        apply_push_recovery_overlay(env_cfg, config, args_cli)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        training_agent_configuration.apply_and_verify(agent_cfg)
        if getattr(agent_cfg, "experiment_name", "") == "":
            agent_cfg.experiment_name = task.lower().replace("-", "_")
        agent_cfg.seed = args_cli.seed
        if hasattr(agent_cfg, "device") and args_cli.device is not None:
            agent_cfg.device = args_cli.device
        if args_cli.num_steps_per_env is not None:
            if int(args_cli.num_steps_per_env) <= 0:
                raise ValueError("--num-steps-per-env must be positive")
            setattr(agent_cfg, "num_steps_per_env", int(args_cli.num_steps_per_env))
        num_steps_per_env = int(getattr(agent_cfg, "num_steps_per_env", 1) or 1)
        if args_cli.num_steps_per_env is not None and num_steps_per_env != int(args_cli.num_steps_per_env):
            raise RuntimeError(
                "agent_cfg.num_steps_per_env did not match requested runner-owned "
                f"value: requested={args_cli.num_steps_per_env}, effective={num_steps_per_env}"
            )
        max_iterations = args_cli.max_iterations
        if max_iterations is None:
            max_iterations = max(1, int(args_cli.budget_steps) // max(1, args_cli.num_envs * num_steps_per_env))
        agent_cfg.max_iterations = int(max_iterations)
        budget_accounting = build_budget_accounting(
            budget_steps=int(args_cli.budget_steps),
            num_envs=int(args_cli.num_envs),
            num_steps_per_env=num_steps_per_env,
            max_iterations=int(agent_cfg.max_iterations),
        )
        agent_cfg.run_name = f"true_rl_agent_{args_cli.round}"

        env = gym.make(task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if (
            str(env.unwrapped.device) != args_cli.device
            or str(agent_cfg.device) != args_cli.device
        ):
            raise RuntimeError(
                "push training device does not match the requested execution device"
            )
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        planned_training_steps = int(agent_cfg.max_iterations) * num_steps_per_env
        candidate_adapter = CandidateRuntimeAdapter.from_candidate_dir(
            candidate_dir,
            num_envs=args_cli.num_envs,
            planned_training_steps=planned_training_steps,
            diagnostic_interval_steps=args_cli.candidate_diagnostic_interval_steps,
            expected_candidate_id=job_identity.subject.subject_id,
            expected_subject_manifest_hash=job_identity.subject_manifest_hash,
            expected_program_source_checksum=program_source_checksum,
            expected_candidate_artifact_hash=(
                candidate_execution_artifact.artifact_hash
            ),
        )
        candidate_summary = candidate_adapter.public_summary()
        wrapped_env = CandidateRewardVecEnvWrapper(wrapped_env, candidate_adapter)

        log_dir = run_dir / "rsl_rl"
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if args_cli.init == "warm_start":
            runner.load(args_cli.init_checkpoint)

        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        export_deploy_cfg(wrapped_env.unwrapped, str(log_dir))
        shutil.copy(inspect.getfile(env_cfg.__class__), params_dir / Path(inspect.getfile(env_cfg.__class__)).name)

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        latest_checkpoint = find_latest_checkpoint(log_dir)
        if latest_checkpoint is None:
            raise RuntimeError(f"no RSL-RL model checkpoint found under {log_dir}")
        shutil.copy2(latest_checkpoint, checkpoint_output)
        checkpoint_hash = sha256_file(checkpoint_output)
        source_hash = sha256_file(latest_checkpoint)
        if source_hash != checkpoint_hash:
            raise RuntimeError("checkpoint copy hash mismatch")
        runtime_activation = candidate_adapter.runtime_summary()
        summary = status_summary(
            "pass",
            checkpoint={
                "uri": str(checkpoint_output),
                "checksum": f"sha256:{checkpoint_hash}",
                "role": "train_output",
                "metadata": {
                    "local_path": str(checkpoint_output),
                    "source_latest_checkpoint": str(latest_checkpoint),
                    "source_latest_checkpoint_sha256": f"sha256:{source_hash}",
                },
            },
            task={"gym_id": task, "profile": args_cli.profile},
            candidate=candidate_summary,
            runtime_activation_summary=runtime_activation,
            training={
                "num_envs": args_cli.num_envs,
                "budget_steps": args_cli.budget_steps,
                "num_steps_per_env": num_steps_per_env,
                "max_iterations": int(agent_cfg.max_iterations),
                "planned_env_steps": budget_accounting["planned_env_steps"],
                "config_derived_total_timesteps": budget_accounting["config_derived_total_timesteps"],
                "budget_accounting": budget_accounting,
                "device": str(agent_cfg.device),
                "log_dir": str(log_dir),
                "checkpoint_output": str(checkpoint_output),
                "latest_checkpoint_iteration": checkpoint_iteration(latest_checkpoint),
                "policy_architecture_hash": policy_architecture_hash,
                "optimizer_training_profile_hash": (
                    optimizer_training_profile_hash
                ),
                "agent_configuration_hash": agent_configuration_hash,
                "initialization_seed_schedule_hash": seed_schedule_hash,
            },
            configuration={
                "config_path": str(Path(args_cli.config).resolve()) if args_cli.config else None,
                "active_profile": config.get("active_profile"),
                "runtime_dynamics_hash": runtime_dynamics["runtime_dynamics_hash"],
                "standing_command": config["standing_command"],
                "push": config["push"],
            },
            errors=[],
        )
    except PushCandidateProgramFailure as exc:
        summary = status_summary(
            "fail",
            candidate=candidate_summary,
            runtime_activation_summary=(
                candidate_adapter.runtime_summary()
                if candidate_adapter is not None
                else None
            ),
            candidate_failure=exc.evidence.to_public_dict(),
            errors=[],
        )
    except Exception as exc:  # noqa: BLE001
        summary = status_summary(
            "fail",
            candidate=candidate_summary,
            runtime_activation_summary=candidate_adapter.runtime_summary() if candidate_adapter is not None else None,
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "status": summary["status"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def status_summary(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        **kwargs,
        "schema_version": "true_rl_agent.push_recovery.train_summary.v1",
        **job_identity.to_public_dict(),
        "training_binding_hash": training_binding_hash,
        "policy_architecture_hash": policy_architecture_hash,
        "optimizer_training_profile_hash": optimizer_training_profile_hash,
        "agent_configuration_hash": agent_configuration_hash,
        "initialization_seed_schedule_hash": seed_schedule_hash,
        "capability_profile_hash": capability_profile_hash,
        "candidate_failure_contract_hash": candidate_failure_contract_hash,
        "program_source_checksum": program_source_checksum,
        "execution_device": str(args_cli.device),
        **(
            {
                "backend_execution_environment_verification": dict(
                    backend_execution_environment_verification
                )
            }
            if backend_execution_environment_verification is not None
            else {}
        ),
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "round_id": args_cli.round,
            "train_attempt_id": train_attempt_id,
            "seed_policy": "runner_owned_not_public",
        },
    }


def find_latest_checkpoint(log_dir: Path) -> Path | None:
    checkpoints = sorted(log_dir.rglob("model_*.pt"), key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime))
    return checkpoints[-1] if checkpoints else None


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    exit_code = main()
    exit_code = close_isaac_app_on_success(
        simulation_app,
        exit_code=exit_code,
    )
    os._exit(exit_code)

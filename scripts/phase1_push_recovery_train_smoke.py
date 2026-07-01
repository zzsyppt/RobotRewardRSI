#!/usr/bin/env python3
"""Short PPO training smoke for the Phase 1 standing push-recovery overlay."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

_ORIGINAL_JSON_DUMPS = json.dumps
_ORIGINAL_OS_CLOSE = os.close
_ORIGINAL_OS_EXIT = os._exit
_ORIGINAL_OS_MAKEDIRS = os.makedirs
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_REPLACE = os.replace
_ORIGINAL_OS_WRITE = os.write


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--profile", default="no_push")
    parser.add_argument("--phase2-config", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument(
        "--runner-artifacts-dir",
        default=None,
        help="Runner-owned writable artifact directory. Required by Phase 8E isolation probes to keep authoritative outputs out of candidate_output.",
    )
    parser.add_argument("--candidate-dir", default=None, help="Optional Phase 8E candidate package loaded inside the isolated training process.")
    parser.add_argument("--candidate-schema-version", default="humanoid_agent.cold_start.candidate.v1")
    parser.add_argument("--candidate-diagnostic-interval-steps", type=int, default=24)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from humanoid_agent.immutable_runner.checkpoints import (  # noqa: E402
    checkpoint_iteration,
    find_latest_checkpoint,
    list_model_checkpoints,
)
from humanoid_agent.immutable_runner.safe_checkpoint import checkpoint_safety_report  # noqa: E402
from humanoid_agent.cold_start_lab.curriculum_runtime import (  # noqa: E402
    Phase8ECompiledCurriculumRuntime,
    Phase8EVecEnvCurriculumWrapper,
)
from humanoid_agent.cold_start_lab.push_signal import (  # noqa: E402
    install_phase8e_push_signal_probe,
    push_signal_public_summary,
)
from humanoid_agent.cold_start_lab.training_adapter import Phase8ETrainingAdapter, Phase8EVecEnvRewardWrapper  # noqa: E402
from humanoid_agent.phase1_push_recovery.config import config_for_profile, load_config, sha256_file  # noqa: E402
from humanoid_agent.phase1_push_recovery.isaac_overlay import apply_phase1_overlay  # noqa: E402
from humanoid_agent.phase2_baseline.overrides import apply_phase2_baseline_overrides  # noqa: E402


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _status_summary(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase1.push_recovery.train_smoke.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "seed": args_cli.seed,
        },
        **kwargs,
    }


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_attr_path(obj: Any, path: tuple[str, ...]) -> int | None:
    current = obj
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    try:
        return int(current)
    except (TypeError, ValueError):
        return None


def _first_int_attr_path(obj: Any, *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        value = _int_attr_path(obj, path)
        if value is not None:
            return value
    return None


def _training_accounting(agent_cfg: Any, *, num_envs: int, max_iterations: int) -> dict[str, Any]:
    """Return runner-owned structured training-budget accounting.

    Phase 8B has not yet installed a VecEnv.step runtime counter.  The total
    timesteps here are therefore config-derived from trusted runner inputs, not
    a direct observation.  True runtime counters are a Phase 8D/8E hardening
    item.
    """

    num_steps_per_env = _int_attr(agent_cfg, "num_steps_per_env")
    env_steps_per_iteration = num_envs * num_steps_per_env if num_steps_per_env is not None else None
    config_derived_total_timesteps = env_steps_per_iteration * max_iterations if env_steps_per_iteration is not None else None
    num_learning_epochs = _first_int_attr_path(
        agent_cfg,
        ("algorithm", "num_learning_epochs"),
        ("num_learning_epochs",),
    )
    num_mini_batches = _first_int_attr_path(
        agent_cfg,
        ("algorithm", "num_mini_batches"),
        ("num_mini_batches",),
    )
    estimated_optimizer_steps = (
        max_iterations * num_learning_epochs * num_mini_batches
        if num_learning_epochs is not None and num_mini_batches is not None
        else None
    )
    return {
        "source": "runner_owned_training_script",
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env,
        "env_steps_per_iteration": env_steps_per_iteration,
        "config_derived_total_timesteps": config_derived_total_timesteps,
        "total_timesteps_semantics": "config_derived_not_runtime_observed",
        "policy_update_iterations": max_iterations,
        "num_learning_epochs": num_learning_epochs,
        "num_mini_batches": num_mini_batches,
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "estimated_gradient_steps": estimated_optimizer_steps,
        "estimated_forward_backward_calls": estimated_optimizer_steps,
    }


def _write_authoritative_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the runner summary through captured low-level primitives.

    Phase 8E candidates execute in this training process until the later
    out-of-process code-level runner lands.  Capturing the low-level write path
    before candidate loading prevents ordinary Python monkeypatches/atexit hooks
    from changing the summary after the runner decides pass/fail.
    """

    payload = (_ORIGINAL_JSON_DUMPS(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _ORIGINAL_OS_MAKEDIRS(str(path.parent), exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    fd = _ORIGINAL_OS_OPEN(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        _ORIGINAL_OS_WRITE(fd, payload)
    finally:
        _ORIGINAL_OS_CLOSE(fd)
    _ORIGINAL_OS_REPLACE(str(temp_path), str(path))


def _write_status_line(summary_file: Path, status: str) -> None:
    line = _ORIGINAL_JSON_DUMPS({"summary_file": str(summary_file), "status": status}, ensure_ascii=False) + "\n"
    _ORIGINAL_OS_WRITE(1, line.encode("utf-8"))


def _seal_checkpoint(source: Path | None, target_dir: Path) -> Path | None:
    """Copy the selected checkpoint into a runner-owned sealed directory."""

    if source is None:
        return None
    _ORIGINAL_OS_MAKEDIRS(str(target_dir), exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    source_hash = sha256_file(source)
    target_hash = sha256_file(target)
    if source_hash != target_hash:
        raise RuntimeError(f"sealed checkpoint hash mismatch: {source} -> {target}")
    return target


def main() -> int:
    run_dir = Path(args_cli.run_dir)
    summary_path = Path(args_cli.summary_file)
    runner_artifacts_dir = Path(args_cli.runner_artifacts_dir) if args_cli.runner_artifacts_dir else (
        run_dir / "runner_artifacts" if args_cli.candidate_dir else run_dir
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    runner_artifacts_dir.mkdir(parents=True, exist_ok=True)
    env = None
    phase8e_adapter = None
    phase8e_adapter_summary = None
    phase8e_curriculum_runtime = None
    phase8e_push_signal_probe = None

    try:
        config = config_for_profile(load_config(args_cli.config), args_cli.profile)
        task = args_cli.task or config["task"]["gym_id"]
        args_cli.task = task

        env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="env_cfg_entry_point")
        apply_phase1_overlay(env_cfg, config, args_cli)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        phase2_override_report = None
        phase2_config_snapshot = None
        if args_cli.phase2_config:
            phase2_config_snapshot = load_config(args_cli.phase2_config)
            phase2_override_report = apply_phase2_baseline_overrides(env_cfg, agent_cfg, phase2_config_snapshot)
        if getattr(agent_cfg, "experiment_name", "") == "":
            agent_cfg.experiment_name = task.lower().replace("-", "_")
        agent_cfg.seed = args_cli.seed
        agent_cfg.max_iterations = args_cli.max_iterations
        agent_cfg.run_name = f"phase1_train_smoke_{args_cli.profile}"
        if hasattr(agent_cfg, "device"):
            agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

        num_steps_per_env = _int_attr(agent_cfg, "num_steps_per_env") or 1
        phase8e_curriculum_runtime = Phase8ECompiledCurriculumRuntime.from_training_config(
            config,
            planned_training_steps=args_cli.max_iterations * num_steps_per_env,
        )
        log_dir = runner_artifacts_dir / "rsl_rl"
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        if args_cli.candidate_dir:
            phase8e_push_signal_probe = install_phase8e_push_signal_probe(env)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if phase8e_curriculum_runtime.enabled:
            wrapped_env = Phase8EVecEnvCurriculumWrapper(wrapped_env, phase8e_curriculum_runtime)
        if args_cli.candidate_dir:
            phase8e_adapter = Phase8ETrainingAdapter.from_candidate_dir(
                args_cli.candidate_dir,
                num_envs=args_cli.num_envs,
                planned_training_steps=args_cli.max_iterations * num_steps_per_env,
                expected_schema_version=args_cli.candidate_schema_version,
                task_spec={
                    "phase": "8E-3-training-loop-adapter",
                    "task": task,
                    "profile": args_cli.profile,
                    "candidate_receives_runner_objects": False,
                },
                diagnostic_interval_steps=args_cli.candidate_diagnostic_interval_steps,
            )
            phase8e_adapter_summary = phase8e_adapter.public_summary()
            wrapped_env = Phase8EVecEnvRewardWrapper(wrapped_env, phase8e_adapter)
        env = wrapped_env
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        export_deploy_cfg(wrapped_env.unwrapped, str(log_dir))
        shutil.copy(inspect.getfile(env_cfg.__class__), params_dir / Path(inspect.getfile(env_cfg.__class__)).name)

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        checkpoints = list_model_checkpoints(log_dir)
        source_latest_checkpoint = find_latest_checkpoint(log_dir)
        latest_checkpoint = _seal_checkpoint(source_latest_checkpoint, runner_artifacts_dir / "sealed_checkpoints")
        summary = _status_summary(
            "pass" if latest_checkpoint is not None else "fail",
            task={"gym_id": task, "profile": args_cli.profile},
            training={
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations,
                "device": args_cli.device,
                "log_dir": str(log_dir),
                "runner_artifacts_dir": str(runner_artifacts_dir),
                "checkpoint_count": len(checkpoints),
                "source_latest_checkpoint": str(source_latest_checkpoint) if source_latest_checkpoint is not None else None,
                "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
                "latest_checkpoint_iteration": checkpoint_iteration(latest_checkpoint) if latest_checkpoint is not None else None,
                "latest_checkpoint_sha256": sha256_file(latest_checkpoint) if latest_checkpoint is not None else None,
                # ADVISORY ONLY: this process executes candidate code, so a
                # candidate can monkeypatch the safety helper and forge this
                # field.  The authoritative checkpoint-safety verdict is produced
                # by the trusted eval boundary (which never loads candidate code)
                # via immutable_runner.safe_checkpoint.  Never gate eval/accept on
                # this child-reported value.
                "child_reported_checkpoint_safe_load": (
                    checkpoint_safety_report(latest_checkpoint)["safe_load"] if latest_checkpoint is not None else False
                ),
            },
            training_accounting=_training_accounting(
                agent_cfg,
                num_envs=args_cli.num_envs,
                max_iterations=args_cli.max_iterations,
            ),
            phase8e_candidate_components=phase8e_adapter_summary,
            phase8e_candidate_runtime=phase8e_adapter.runtime_summary() if phase8e_adapter is not None else None,
            phase8e_compiled_curriculum=(
                phase8e_curriculum_runtime.public_summary() if phase8e_curriculum_runtime is not None else None
            ),
            phase8e_compiled_curriculum_runtime=(
                phase8e_curriculum_runtime.runtime_summary() if phase8e_curriculum_runtime is not None else None
            ),
            phase8e_push_signal_probe=phase8e_push_signal_probe,
            phase8e_push_signal_runtime=push_signal_public_summary(env) if args_cli.candidate_dir else None,
            configuration={
                "config_path": str(Path(args_cli.config).resolve()),
                "config_sha256": sha256_file(args_cli.config),
                "phase2_config_path": str(Path(args_cli.phase2_config).resolve()) if args_cli.phase2_config else None,
                "phase2_config_sha256": sha256_file(args_cli.phase2_config) if args_cli.phase2_config else None,
                "phase2_override_report": phase2_override_report,
                "standing_command": config["standing_command"],
                "push": config["push"],
            },
            errors=[],
        )
    except BaseException as exc:  # noqa: BLE001
        summary = _status_summary(
            "fail",
            phase8e_candidate_components=phase8e_adapter_summary,
            phase8e_candidate_runtime=phase8e_adapter.runtime_summary() if phase8e_adapter is not None else None,
            phase8e_compiled_curriculum=(
                phase8e_curriculum_runtime.public_summary() if phase8e_curriculum_runtime is not None else None
            ),
            phase8e_compiled_curriculum_runtime=(
                phase8e_curriculum_runtime.runtime_summary() if phase8e_curriculum_runtime is not None else None
            ),
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    _write_authoritative_summary(summary_path, summary)
    _write_status_line(summary_path, str(summary["status"]))
    exit_code = 0 if summary["status"] == "pass" else 1
    try:
        simulation_app.close()
    except Exception:
        pass
    return exit_code


if __name__ == "__main__":
    _ORIGINAL_OS_EXIT(main())

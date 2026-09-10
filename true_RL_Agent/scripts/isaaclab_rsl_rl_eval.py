#!/usr/bin/env python3
"""Trusted push recovery eval for a runner-sealed checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    EvaluationBindingV1,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.protocol import (  # noqa: E402
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.evaluation.sealed_validation import (  # noqa: E402
    build_sealed_validation_gate,
)
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.runner.checkpoint import build_checkpoint_safety_report  # noqa: E402
from true_rl_agent.runner.eval_isolation import (  # noqa: E402
    build_candidate_isolation_gate,
    reject_candidate_path_arguments,
)
from true_rl_agent.runner.evidence import evidence_artifact_from_json  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    bind_isaac_log_path,
    close_isaac_app_on_success,
)
from true_rl_agent.backends.isaac_lab_arrays import as_torch  # noqa: E402
from true_rl_agent.backends.isaac_lab_runtime import require_physx_success  # noqa: E402
from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint  # noqa: E402
from true_rl_agent.robots.unitree_locomotion import (  # noqa: E402
    instantiate_g1_29_environment_cfg, instantiate_g1_29_runner_cfg,
)
from true_rl_agent.runner.job_identity import JobIdentity  # noqa: E402
from true_rl_agent.runner.gpu_process_binding import (  # noqa: E402
    add_gpu_process_binding_arguments,
    attach_gpu_runtime_evidence,
    capture_gpu_compute_context,
    validate_gpu_process_binding_invocation,
)
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.push_recovery.seed_policy import (  # noqa: E402
    SEED_POLICY_NAME,
    build_seed_cohort_hash,
    derive_runner_seed,
    public_seed_record,
)
from true_rl_agent.composition.cell_manifest import (  # noqa: E402
    add_resolved_cell_manifest_arguments,
    load_optional_resolved_cell_manifest_from_args,
    resolved_cell_summary_fields,
)


def evaluation_protocol_from_args(args: argparse.Namespace) -> EvaluationProtocol:
    return EvaluationProtocol(
        eval_protocol_version=str(args.eval_protocol_version),
        public_seed_count=int(args.public_seeds),
        public_accept_seed_count=int(args.public_accept_seeds),
        sealed_validation_seed_count=int(args.sealed_validation_seeds),
        eval_num_envs=int(args.num_envs),
    )


def evaluation_binding_from_args(
    args: argparse.Namespace,
) -> EvaluationBindingV1:
    binding = EvaluationBindingV1(
        eval_protocol_hash=evaluation_protocol_from_args(args).protocol_hash,
        acceptance_protocol_hash=str(args.acceptance_protocol_hash),
        task_protocol_hash=str(args.task_protocol_hash),
        seed_cohort_hash=str(args.seed_cohort_hash),
    )
    if binding.binding_hash != args.evaluation_binding_hash:
        raise ValueError("evaluation binding hash mismatch")
    return binding


try:
    reject_candidate_path_arguments(sys.argv[1:])
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc

from isaaclab.app import AppLauncher  # type: ignore[import-not-found]  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--subject-kind",
        required=True,
        choices=("candidate", "baseline", "reference"),
    )
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subject-manifest-hash", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--public-seeds", type=int, required=True)
    parser.add_argument("--public-accept-seeds", type=int, default=3)
    parser.add_argument("--sealed-validation-seeds", type=int, default=7)
    parser.add_argument("--eval-protocol-version", default=DEFAULT_EVAL_PROTOCOL_VERSION)
    parser.add_argument("--task-protocol-hash", required=True)
    parser.add_argument("--base-task-protocol-hash", default=None)
    parser.add_argument("--acceptance-protocol-hash", required=True)
    parser.add_argument("--seed-cohort-hash", required=True)
    parser.add_argument("--evaluation-binding-hash", required=True)
    parser.add_argument(
        "--evaluation-phase",
        choices=("public_dev", "public_accept", "sealed_validation"),
    )
    parser.add_argument("--evaluation-plan-hash")
    parser.add_argument("--training-binding-hash", default=None)
    parser.add_argument("--program-source-checksum", default=None)
    parser.add_argument("--capability-profile-hash", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="strong_push")
    parser.add_argument("--runtime-dynamics-hash", required=True)
    parser.add_argument(
        "--backend-execution-environment-manifest-hash",
        default=None,
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--policy", default="checkpoint", choices=("checkpoint", "random", "zero"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--status-file", default=None)
    add_resolved_cell_manifest_arguments(parser)
    add_gpu_process_binding_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


args_cli = _build_parser().parse_args()
try:
    if bool(args_cli.evaluation_phase) != bool(args_cli.evaluation_plan_hash):
        raise ValueError("evaluation phase and plan hash must be supplied together")
    args_cli.device = require_execution_device(
        args_cli.device,
        "push eval execution device",
    )
    gpu_runtime_binding = validate_gpu_process_binding_invocation(
        args_cli, environment=os.environ,
    )
    args_cli.evaluation_subject = ExperimentSubject(
        kind=args_cli.subject_kind,
        subject_id=args_cli.subject_id,
    )
    args_cli.evaluation_job_identity = JobIdentity(
        job_id=args_cli.job_id,
        task_id=args_cli.task_id,
        subject=args_cli.evaluation_subject,
        subject_manifest_hash=args_cli.subject_manifest_hash,
    )
    args_cli.capability_profile_hash = require_sha256_digest(
        args_cli.capability_profile_hash,
        "capability profile hash",
    )
    if args_cli.evaluation_subject.kind in {"candidate", "reference"}:
        args_cli.program_source_checksum = require_sha256_checksum(
            args_cli.program_source_checksum,
            "program source checksum",
        )
        args_cli.training_binding_hash = require_sha256_digest(
            args_cli.training_binding_hash,
            "training binding hash",
        )
    elif (
        args_cli.training_binding_hash is not None
        or args_cli.program_source_checksum is not None
    ):
        raise ValueError("baseline eval cannot carry program lineage")
    args_cli.evaluation_binding = evaluation_binding_from_args(args_cli)
    if (
        args_cli.resolved_cell_manifest_path is not None
        or args_cli.resolved_cell_manifest_hash is not None
    ) and args_cli.base_task_protocol_hash is None:
        raise ValueError("resolved cell manifest requires --base-task-protocol-hash")
    args_cli.resolved_cell_manifest = load_optional_resolved_cell_manifest_from_args(
        args_cli,
        expected_task_id=args_cli.task_id,
        expected_base_task_protocol_hash=(
            args_cli.base_task_protocol_hash
            if args_cli.base_task_protocol_hash is not None
            else args_cli.task_protocol_hash
        ),
    )
    actual_seed_cohort_hash = build_seed_cohort_hash(
        base_seed=int(args_cli.seed),
        evaluation_protocol=evaluation_protocol_from_args(args_cli),
    )
    if actual_seed_cohort_hash != args_cli.seed_cohort_hash:
        raise ValueError("seed cohort hash does not match runtime seed policy")
    args_cli.backend_execution_environment_verification = (
        verify_current_isaac_lab_execution_environment(
            args_cli.backend_execution_environment_manifest_hash
        )
        if args_cli.backend_execution_environment_manifest_hash is not None
        else None
    )
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc
bind_isaac_log_path(args_cli, args_cli.summary_path)
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
from isaaclab.utils.math import (
    euler_xyz_from_quat,  # type: ignore[import-not-found]  # noqa: E402
)
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,  # type: ignore[import-not-found]  # noqa: E402
)
from rsl_rl.runners import (
    OnPolicyRunner,  # type: ignore[import-not-found]  # noqa: E402
)

from true_rl_agent.tasks.push_recovery.diagnostics import (
    build_diagnostic_hints,  # noqa: E402
)
from true_rl_agent.tasks.push_recovery.failure_classifier import (  # noqa: E402
    classify_episode,
    summarize_condition_flags,
    summarize_failures,
)
from true_rl_agent.tasks.push_recovery.isaac_config import (  # noqa: E402
    config_for_profile,
    load_runtime_config,
    public_runtime_dynamics,
    sha256_file,
)
from true_rl_agent.tasks.push_recovery.isaac_overlay import (  # noqa: E402
    apply_eval_overrides,
    apply_push_recovery_overlay,
    as_pair,
)
from true_rl_agent.tasks.push_recovery.metrics import (  # noqa: E402
    build_push_public_result_from_summaries,
    build_push_sealed_rollout_evidence,
)


def main() -> int:
    summary_path = Path(args_cli.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args_cli.status_file:
        Path(args_cli.status_file).parent.mkdir(parents=True, exist_ok=True)
    env = None
    base_env = None
    checkpoint_safety_report = build_checkpoint_safety_report(
        Path(args_cli.checkpoint),
        expected_checkpoint_hash=args_cli.checkpoint_sha256,
    ).to_public_dict()
    try:
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
        if args_cli.policy == "checkpoint":
            assert_safe_checkpoint(Path(args_cli.checkpoint), checkpoint_safety_report=checkpoint_safety_report)
        env_cfg = instantiate_g1_29_environment_cfg(device=args_cli.device, num_envs=args_cli.num_envs)
        apply_push_recovery_overlay(env_cfg, config, args_cli)
        apply_eval_overrides(env_cfg, config)
        base_env = gym.make(task, cfg=env_cfg, render_mode=None)
        if isinstance(base_env.unwrapped, DirectMARLEnv):
            base_env = multi_agent_to_single_agent(base_env)
        if str(base_env.unwrapped.device) != args_cli.device:
            raise RuntimeError(
                "push evaluation device does not match the requested execution device"
            )

        inference_policy = None
        checkpoint_loading = None
        constructed_learner = None
        if args_cli.policy == "checkpoint":
            from true_rl_agent.robots.rsl_rl_runtime import (
                G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION,
            )

            agent_cfg = instantiate_g1_29_runner_cfg()
            if hasattr(agent_cfg, "device") and args_cli.device is not None:
                agent_cfg.device = args_cli.device
            if str(agent_cfg.device) != args_cli.device:
                raise RuntimeError(
                    "push policy device does not match the requested execution device"
                )
            env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            constructed_learner = (
                G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION.verify_runner(runner)
            )
            checkpoint_loading = load_inference_checkpoint(runner, args_cli.checkpoint)
            inference_policy = runner.get_inference_policy(device=env.unwrapped.device)
        else:
            env = base_env

        summary = run_public_eval(
            env,
            config,
            args_cli,
            inference_policy=inference_policy,
            checkpoint_safety_report=checkpoint_safety_report,
        )
        summary["checkpoint_loading"] = checkpoint_loading
        summary["constructed_learner"] = constructed_learner
        summary = attach_gpu_runtime_evidence(
            summary, gpu_runtime_binding,
            compute_context_observation=(
                capture_gpu_compute_context(gpu_runtime_binding)
                if gpu_runtime_binding is not None else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        summary = {
            "schema_version": "true_rl_agent.push_recovery.eval_summary.v1",
            **eval_summary_binding(args_cli),
            "execution_device": str(args_cli.device),
            "status": "fail",
            "public_score": 0.0,
            "metrics": {"success_rate": 0.0, "failure_breakdown": {}, "motion_quality": {}},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_code_loaded_in_eval": False,
            "candidate_package_loaded": False,
            "candidate_package_loaded_in_eval": False,
            "checkpoint_safety_report": checkpoint_safety_report,
            "checkpoint_safety_status": checkpoint_safety_report["status"],
            "evidence_manifest": build_eval_evidence_manifest(checkpoint_safety_report),
            "run": {
                "cwd": os.getcwd(),
                "round_id": args_cli.round,
                "task_id": args_cli.task_id,
                "subject_kind": args_cli.evaluation_subject.kind,
                "subject_id": args_cli.evaluation_subject.subject_id,
                "seed_policy": SEED_POLICY_NAME,
                "public_seed_count": args_cli.public_seeds,
                "public_accept_seed_count": args_cli.public_accept_seeds,
                "sealed_validation_seed_count": args_cli.sealed_validation_seeds,
                "eval_num_envs": args_cli.num_envs,
            },
            "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        }
    finally:
        if env is not None:
            env.close()
        elif base_env is not None:
            base_env.close()

    summary.update(resolved_cell_summary_fields(args_cli.resolved_cell_manifest))
    if summary.get("status") == "pass":
        require_physx_success(summary_path.with_suffix(".kit.log"))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args_cli.status_file:
        Path(args_cli.status_file).write_text(str(summary.get("status", "fail")) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "status": summary.get("status")}, sort_keys=True))
    return 0 if summary.get("status") == "pass" else 1


def run_public_eval(
    env: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
    inference_policy: Any = None,
    checkpoint_safety_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint_safety_report is None:
        raise ValueError("checkpoint_safety_report is required")
    evaluation_protocol = evaluation_protocol_from_args(args)
    if args.evaluation_phase is not None:
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary

        phase = EvaluationPhase.parse(args.evaluation_phase)
        seed_count = {
            EvaluationPhase.PUBLIC_DEV: evaluation_protocol.public_seed_count,
            EvaluationPhase.PUBLIC_ACCEPT: evaluation_protocol.public_accept_seed_count,
            EvaluationPhase.SEALED_VALIDATION: evaluation_protocol.sealed_validation_seed_count,
        }[phase]
        started_ns = time.perf_counter_ns()
        summaries = [
            run_rollout(
                env, config, args, inference_policy=inference_policy,
                split_name=phase.value, run_index=index + 1,
            )
            for index in range(seed_count)
        ]
        return build_push_phase_summary(
            phase=phase,
            evaluation_plan_hash=args.evaluation_plan_hash,
            seed_summaries=summaries,
            expected_seed_count=seed_count,
            wall_time_ns=time.perf_counter_ns() - started_ns,
            identity=eval_summary_binding(args),
            execution_device=str(args.device),
            checkpoint_safety_report=checkpoint_safety_report,
            runtime_dynamics_hash=public_runtime_dynamics(config)["runtime_dynamics_hash"],
        )
    public_dev_summaries = [
        run_rollout(env, config, args, inference_policy=inference_policy, split_name="public_dev", run_index=index + 1)
        for index in range(evaluation_protocol.public_seed_count)
    ]
    public_accept_summaries = [
        run_rollout(env, config, args, inference_policy=inference_policy, split_name="public_accept", run_index=index + 1)
        for index in range(evaluation_protocol.public_accept_seed_count)
    ]
    sealed_validation_summaries = [
        run_rollout(
            env,
            config,
            args,
            inference_policy=inference_policy,
            split_name="sealed_validation",
            run_index=index + 1,
        )
        for index in range(evaluation_protocol.sealed_validation_seed_count)
    ]
    sealed_validation_evidence = build_push_sealed_rollout_evidence(
        summaries=sealed_validation_summaries,
        evaluation_protocol=evaluation_protocol,
        checkpoint_hash=str(args.checkpoint_sha256),
        evaluation_binding_hash=args.evaluation_binding.binding_hash,
        evaluation_subject=args.evaluation_subject,
        subject_manifest_hash=args.subject_manifest_hash,
        program_source_checksum=args.program_source_checksum,
        training_binding_hash=args.training_binding_hash,
        capability_profile_hash=args.capability_profile_hash,
    )
    sealed_validation_gate = build_eval_sealed_validation_gate(
        checkpoint_safety_report,
        rollout_evidence=sealed_validation_evidence,
        evaluation_protocol=evaluation_protocol,
        evaluation_binding=args.evaluation_binding,
    )
    projected = build_push_public_result_from_summaries(
        public_dev_summaries=public_dev_summaries,
        public_accept_summaries=public_accept_summaries,
        budget_summary=None,
        min_public_accept_count=evaluation_protocol.public_accept_seed_count,
        evaluation_protocol=evaluation_protocol,
    )
    first_summary = public_accept_summaries[0] if public_accept_summaries else public_dev_summaries[0]
    summary = {
        "schema_version": "true_rl_agent.push_recovery.eval_summary.v1",
        **eval_summary_binding(args),
        "execution_device": str(args.device),
        "status": projected.status,
        "public_score": projected.public_score,
        "metrics": dict(projected.metrics),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "checkpoint_safety_report": checkpoint_safety_report,
        "checkpoint_safety_status": checkpoint_safety_report["status"],
        "sealed_validation_evidence": sealed_validation_evidence,
        "sealed_validation_gate": sealed_validation_gate,
        "evidence_manifest": build_eval_evidence_manifest(
            checkpoint_safety_report,
            sealed_validation_evidence=sealed_validation_evidence,
        ),
        "run": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "round_id": args.round,
            "task_id": args.task_id,
            "subject_kind": args.evaluation_subject.kind,
            "subject_id": args.evaluation_subject.subject_id,
            "seed_policy": SEED_POLICY_NAME,
            "public_seed_count": evaluation_protocol.public_seed_count,
            "public_accept_seed_count": evaluation_protocol.public_accept_seed_count,
            "sealed_validation_seed_count": evaluation_protocol.sealed_validation_seed_count,
            "eval_num_envs": evaluation_protocol.eval_num_envs,
        },
        "task": first_summary.get("task", {}),
        "policy": first_summary.get("policy", {}),
        "configuration": first_summary.get("configuration", {}),
        "environment": first_summary.get("environment", {}),
        "evidence": first_summary.get("evidence", {}),
        "aggregate": first_summary.get("aggregate", {}),
        "secondary_metrics": first_summary.get("secondary_metrics", {}),
        "per_episode": first_summary.get("per_episode", []),
        "sealed_validation_rollouts": sealed_validation_summaries,
        "reset_extras_keys": first_summary.get("reset_extras_keys", []),
        "errors": projected.metrics.get("errors", []),
    }
    summary["diagnostic_hints"] = build_diagnostic_hints(summary)
    return summary


def run_rollout(
    env: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
    inference_policy: Any = None,
    split_name: str = "public_eval",
    run_index: int = 1,
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    device = torch.device(unwrapped.device)
    num_envs = int(unwrapped.num_envs)
    action_dim = action_dim_from_env(env)
    step_dt = float(unwrapped.step_dt)
    max_steps = args.max_steps if args.max_steps is not None else int(round(config["environment"]["episode_length_s"] / step_dt))
    max_steps = int(max_steps)
    split_seed_count = {
        "public_accept": args.public_accept_seeds,
        "sealed_validation": args.sealed_validation_seeds,
    }.get(split_name, args.public_seeds)
    seed_record = public_seed_record(
        split_name=split_name,
        run_index=run_index,
        split_seed_count=split_seed_count,
    )
    runner_seed = derive_runner_seed(base_seed=args.seed, split_name=split_name, run_index=run_index)
    set_runner_seed(runner_seed)

    policy_defaults = config.get("policy_defaults", {})
    if args.action_scale is not None:
        action_scale = float(args.action_scale)
    elif args.policy == "random":
        action_scale = float(policy_defaults.get("random_action_scale", 1.0))
    else:
        action_scale = float(policy_defaults.get("zero_action_scale", 0.0))

    obs, reset_extras = reset_env(env, seed=runner_seed)
    observation_finite = all_finite_tree(obs)
    reward_finite = True

    robot = unwrapped.scene["robot"]
    initial_xy = robot.data.root_pos_w[:, :2].clone()
    inf = torch.full((num_envs,), float("inf"), device=device)
    neg_inf = torch.full((num_envs,), float("-inf"), device=device)
    min_base_height = inf.clone()
    max_pitch = neg_inf.clone()
    min_pitch = inf.clone()
    max_abs_pitch = torch.zeros(num_envs, device=device)
    max_abs_roll = torch.zeros(num_envs, device=device)
    final_xy_drift = torch.zeros(num_envs, device=device)
    final_planar_speed = torch.zeros(num_envs, device=device)
    cumulative_reward = torch.zeros(num_envs, device=device)
    done_step = torch.full((num_envs,), max_steps, dtype=torch.long, device=device)
    terminated_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    timed_out_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)

    max_pre_push_speed = torch.zeros(num_envs, device=device)
    max_post_push_speed = torch.zeros(num_envs, device=device)
    push_interval = as_pair(config["push"]["interval_range_s"])
    push_start_s = push_interval[0]
    push_evidence_cfg = config["push"]["evidence"]
    pre_window_s = float(push_evidence_cfg["pre_window_s"])
    post_window_s = float(push_evidence_cfg["post_window_s"])
    single_step_window_s = as_pair(push_evidence_cfg.get("single_step_window_s", [push_start_s, push_start_s]))
    max_push_step_velocity_delta = torch.zeros(num_envs, device=device)
    push_step_of_max_delta = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    nonzero_push_configured = has_nonzero_push(config)

    last_action = torch.zeros((num_envs, action_dim), device=device)
    action_l2_sum = torch.zeros(num_envs, device=device)
    action_rate_l2_sum = torch.zeros(num_envs, device=device)
    action_samples = torch.zeros(num_envs, device=device)

    first_step_seen = False
    vector_env_step_calls = 0
    for step in range(max_steps):
        time_s = step * step_dt
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        if len(active_idx) > 0:
            root_pos = robot.data.root_pos_w
            root_quat = as_torch(robot.data.root_quat_w)
            root_vel = robot.data.root_lin_vel_w
            pre_step_root_lin_vel_xy = root_vel[:, :2].clone()
            roll, pitch, _ = euler_xyz_from_quat(root_quat)
            planar_speed = torch.linalg.vector_norm(root_vel[:, :2], dim=1)
            drift = torch.linalg.vector_norm(root_pos[:, :2] - initial_xy, dim=1)
            min_base_height[active_idx] = torch.minimum(min_base_height[active_idx], root_pos[active_idx, 2])
            max_pitch[active_idx] = torch.maximum(max_pitch[active_idx], pitch[active_idx])
            min_pitch[active_idx] = torch.minimum(min_pitch[active_idx], pitch[active_idx])
            max_abs_pitch[active_idx] = torch.maximum(max_abs_pitch[active_idx], torch.abs(pitch[active_idx]))
            max_abs_roll[active_idx] = torch.maximum(max_abs_roll[active_idx], torch.abs(roll[active_idx]))
            final_xy_drift[active_idx] = drift[active_idx]
            final_planar_speed[active_idx] = planar_speed[active_idx]
            if time_s <= max(push_start_s, pre_window_s):
                max_pre_push_speed[active_idx] = torch.maximum(max_pre_push_speed[active_idx], planar_speed[active_idx])
            if push_start_s <= time_s <= push_start_s + post_window_s:
                max_post_push_speed[active_idx] = torch.maximum(max_post_push_speed[active_idx], planar_speed[active_idx])
        else:
            pre_step_root_lin_vel_xy = robot.data.root_lin_vel_w[:, :2].clone()

        if inference_policy is not None:
            with torch.inference_mode():
                actions = inference_policy(obs)
        else:
            actions = make_actions(args.policy, num_envs, action_dim, device, action_scale)
        action_norm = torch.linalg.vector_norm(actions, dim=1)
        action_rate = torch.linalg.vector_norm(actions - last_action, dim=1)
        if not first_step_seen:
            action_rate = torch.zeros_like(action_rate)
            first_step_seen = True
        last_action = actions

        active_float = active.to(dtype=torch.float32)
        action_l2_sum += action_norm * active_float
        action_rate_l2_sum += action_rate * active_float
        action_samples += active_float

        active_before_step = active.clone()
        obs, reward, terminated, truncated, _extras = step_env(env, actions)
        vector_env_step_calls += 1
        observation_finite = observation_finite and all_finite_tree(obs)
        reward_finite = reward_finite and bool(torch.isfinite(reward).all().item())
        cumulative_reward += reward * active_float

        if single_step_window_s[0] <= time_s <= single_step_window_s[1]:
            post_step_root_lin_vel_xy = robot.data.root_lin_vel_w[:, :2]
            step_delta = torch.linalg.vector_norm(post_step_root_lin_vel_xy - pre_step_root_lin_vel_xy, dim=1)
            update_mask = active_before_step & (step_delta > max_push_step_velocity_delta)
            max_push_step_velocity_delta[update_mask] = step_delta[update_mask]
            push_step_of_max_delta[update_mask] = step + 1

        done = (terminated | truncated) & active
        if bool(done.any().item()):
            done_step[done] = step + 1
            terminated_once |= terminated & done
            timed_out_once |= truncated & done
            active[done] = False
        if not bool(active.any().item()):
            break

    active_idx = active.nonzero(as_tuple=False).squeeze(-1)
    if len(active_idx) > 0:
        root_pos = robot.data.root_pos_w
        root_quat = as_torch(robot.data.root_quat_w)
        root_vel = robot.data.root_lin_vel_w
        roll, pitch, _ = euler_xyz_from_quat(root_quat)
        drift = torch.linalg.vector_norm(root_pos[:, :2] - initial_xy, dim=1)
        planar_speed = torch.linalg.vector_norm(root_vel[:, :2], dim=1)
        min_base_height[active_idx] = torch.minimum(min_base_height[active_idx], root_pos[active_idx, 2])
        max_pitch[active_idx] = torch.maximum(max_pitch[active_idx], pitch[active_idx])
        min_pitch[active_idx] = torch.minimum(min_pitch[active_idx], pitch[active_idx])
        max_abs_pitch[active_idx] = torch.maximum(max_abs_pitch[active_idx], torch.abs(pitch[active_idx]))
        max_abs_roll[active_idx] = torch.maximum(max_abs_roll[active_idx], torch.abs(roll[active_idx]))
        final_xy_drift[active_idx] = drift[active_idx]
        final_planar_speed[active_idx] = planar_speed[active_idx]

    thresholds = config["success_thresholds"]
    recovery_window_speed_limit = float(config.get("behavior_gate_thresholds", {}).get("max_final_planar_speed_mps", 5.0))
    max_records = int(config.get("metrics", {}).get("max_per_episode_records", 24))
    per_episode: list[dict[str, Any]] = []
    labels: list[str] = []
    classifications: list[dict[str, Any]] = []
    successes = 0
    for env_id in range(num_envs):
        samples = max(1.0, float_value(action_samples[env_id]))
        metrics = {
            "env_id": env_id,
            "episode_steps": int(done_step[env_id].detach().cpu().item()),
            "terminated": bool(terminated_once[env_id].detach().cpu().item()),
            "timed_out": bool(timed_out_once[env_id].detach().cpu().item()),
            "min_base_height_m": round_float(float_value(min_base_height[env_id])),
            "max_pitch_rad": round_float(float_value(max_pitch[env_id])),
            "min_pitch_rad": round_float(float_value(min_pitch[env_id])),
            "max_abs_pitch_rad": round_float(float_value(max_abs_pitch[env_id])),
            "max_abs_roll_rad": round_float(float_value(max_abs_roll[env_id])),
            "final_xy_drift_m": round_float(float_value(final_xy_drift[env_id])),
            "final_planar_speed_mps": round_float(float_value(final_planar_speed[env_id])),
            "recovery_window_success": bool(float_value(final_planar_speed[env_id]) <= recovery_window_speed_limit),
            "return": round_float(float_value(cumulative_reward[env_id])),
            "mean_action_l2": round_float(float_value(action_l2_sum[env_id]) / samples),
            "mean_action_rate_l2": round_float(float_value(action_rate_l2_sum[env_id]) / samples),
            "pre_push_peak_speed_mps": round_float(float_value(max_pre_push_speed[env_id])),
            "post_push_peak_speed_mps": round_float(float_value(max_post_push_speed[env_id])),
            "observed_push_delta_mps": round_float(float_value(max_post_push_speed[env_id] - max_pre_push_speed[env_id])),
        }
        classification = classify_episode(metrics, thresholds)
        classifications.append(classification)
        metrics["failure_label"] = classification["label"]
        metrics["success"] = classification["success"]
        labels.append(classification["label"])
        successes += int(classification["success"])
        if len(per_episode) < max_records:
            per_episode.append(metrics)

    failure_summary = summarize_failures(labels)
    condition_flag_summary = summarize_condition_flags(classifications)
    success_rate = round_float(successes / num_envs)
    single_step_deltas = max_push_step_velocity_delta.detach().cpu()
    min_single_step_delta = float(push_evidence_cfg["min_single_step_delta_mps"])
    min_single_step_fraction = float(push_evidence_cfg["min_env_fraction_with_single_step_delta"])
    single_step_fraction = float((single_step_deltas >= min_single_step_delta).to(dtype=torch.float32).mean().item())
    push_gate_pass = single_step_fraction >= min_single_step_fraction if nonzero_push_configured else None
    recovery_window_success_rate = float((final_planar_speed <= recovery_window_speed_limit).to(dtype=torch.float32).mean().item())

    secondary_metrics = {
        "mean_episode_steps": round_float(mean([float(int(done_step[i].detach().cpu().item())) for i in range(num_envs)])),
        "mean_return": round_float(mean([float_value(cumulative_reward[i]) for i in range(num_envs)])),
        "mean_action_l2": round_float(mean([float_value(action_l2_sum[i]) / max(1.0, float_value(action_samples[i])) for i in range(num_envs)])),
        "mean_action_rate_l2": round_float(mean([float_value(action_rate_l2_sum[i]) / max(1.0, float_value(action_samples[i])) for i in range(num_envs)])),
        "mean_min_base_height_m": round_float(float(min_base_height.mean().item())),
        "mean_max_abs_pitch_rad": round_float(float(max_abs_pitch.mean().item())),
        "mean_max_abs_roll_rad": round_float(float(max_abs_roll.mean().item())),
        "mean_final_xy_drift_m": round_float(float(final_xy_drift.mean().item())),
        "mean_final_planar_speed_mps": round_float(float(final_planar_speed.mean().item())),
        "recovery_window_success_rate": round_float(recovery_window_success_rate),
    }
    summary = {
        "schema_version": "true_rl_agent.push_recovery.eval_summary.v1",
        "status": "pass",
        "public_score": float(success_rate),
        "metrics": {
            "success_rate": float(success_rate),
            "failure_breakdown": failure_summary,
            "motion_quality": secondary_metrics,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "run": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "round_id": args.round,
            "subject_kind": args.evaluation_subject.kind,
            "subject_id": args.evaluation_subject.subject_id,
            "seed_policy": seed_record["seed_policy"],
            "public_seed_count": args.public_seeds,
            "public_accept_seed_count": args.public_accept_seeds,
            "split": seed_record["split"],
            "run_index": seed_record["run_index"],
            "split_seed_count": seed_record["split_seed_count"],
        },
        "task": {
            "gym_id": args.task,
            "robot": config["task"]["robot"],
            "implementation_strategy": config["task"]["implementation_strategy"],
            "vendor_env": config["task"]["vendor_env"],
        },
        "policy": {"kind": args.policy, "checkpoint": args.checkpoint, "action_scale": action_scale, "action_dim": action_dim},
        "configuration": {
            "config_path": str(Path(args.config).resolve()) if args.config else None,
            "config_sha256": (
                sha256_file(args.config)
                if args.config and Path(args.config).exists()
                else None
            ),
            "active_profile": config.get("active_profile"),
            "runtime_dynamics_hash": public_runtime_dynamics(config)["runtime_dynamics_hash"],
            "standing_command": config["standing_command"],
            "push": config["push"],
            "success_thresholds": thresholds,
        },
        "environment": {"num_envs": num_envs, "device": str(device), "step_dt": step_dt, "max_steps": max_steps},
        "execution_cost": {
            "evaluation_env_count": num_envs,
            "vector_env_step_calls": vector_env_step_calls,
            "closed_first_episode_control_steps": int(done_step.sum().item()),
        },
        "evidence": {
            "reset_ok": True,
            "observation_finite": observation_finite,
            "reward_finite": reward_finite,
            "push_effect": {
                "mechanism": config["push"]["mechanism"],
                "nonzero_push_configured": nonzero_push_configured,
                "expected_interval_range_s": list(push_interval),
                "single_step_window_s": list(single_step_window_s),
                "mean_single_step_delta_mps": round_float(float(single_step_deltas.mean().item())),
                "max_single_step_delta_mps": round_float(float(single_step_deltas.max().item())),
                "min_single_step_delta_mps": round_float(min_single_step_delta),
                "env_fraction_with_single_step_delta": round_float(single_step_fraction),
                "min_env_fraction_with_single_step_delta": round_float(min_single_step_fraction),
                "gate_pass": push_gate_pass,
            },
            "recovery_window": {
                "max_final_planar_speed_mps": round_float(recovery_window_speed_limit),
                "success_rate": round_float(recovery_window_success_rate),
            },
        },
        "aggregate": {
            "episodes": num_envs,
            "success_count": successes,
            "success_rate": float(success_rate),
            "failure_breakdown": failure_summary,
            "condition_flag_breakdown": condition_flag_summary,
        },
        "secondary_metrics": secondary_metrics,
        "per_episode": per_episode,
        "reset_extras_keys": sorted(reset_extras.keys()) if isinstance(reset_extras, dict) else [],
        "errors": [],
    }
    summary["diagnostic_hints"] = build_diagnostic_hints(summary)
    return summary


def build_eval_sealed_validation_gate(
    checkpoint_safety_report: dict[str, Any],
    *,
    rollout_evidence: dict[str, Any],
    evaluation_protocol: EvaluationProtocol,
    evaluation_binding: EvaluationBindingV1,
) -> dict[str, Any]:
    return build_sealed_validation_gate(
        rollout_evidence=rollout_evidence,
        checkpoint_safety_status=str(checkpoint_safety_report["status"]),
        checkpoint_hash_match_status=str(
            checkpoint_safety_report["checkpoint_hash_match_status"]
        ),
        candidate_isolation_gate=build_candidate_isolation_gate(),
        expected_eval_protocol_hash=evaluation_protocol.protocol_hash,
        expected_evaluation_binding_hash=evaluation_binding.binding_hash,
    )


def eval_summary_binding(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **args.evaluation_job_identity.to_public_dict(),
        "evaluation_protocol": evaluation_protocol_from_args(args).to_public_dict(),
        "evaluation_binding": evaluation_binding_from_args(args).to_public_dict(),
        **(
            {
                "backend_execution_environment_verification": dict(
                    args.backend_execution_environment_verification
                )
            }
            if args.backend_execution_environment_verification is not None
            else {}
        ),
    }


def build_eval_evidence_manifest(
    checkpoint_safety_report: dict[str, Any],
    *,
    sealed_validation_evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    checkpoint_artifact = evidence_artifact_from_json(
        artifact_id="checkpoint-safety-report",
        kind="checkpoint_safety_report",
        payload=checkpoint_safety_report,
        producer="isaaclab_rsl_rl_eval",
        visibility="public_projection_only",
        public_projection_ref="metrics.checkpoint_safety_report",
    )
    artifacts = [checkpoint_artifact.to_public_dict()]
    if sealed_validation_evidence is not None:
        sealed_artifact = evidence_artifact_from_json(
            artifact_id="sealed-validation-evidence",
            kind="sealed_validation_rollout_evidence",
            payload=sealed_validation_evidence,
            producer="isaaclab_rsl_rl_eval",
            visibility="public_projection_only",
            public_projection_ref="metrics.sealed_validation",
        )
        artifacts.append(sealed_artifact.to_public_dict())
    return artifacts


def assert_safe_checkpoint(path: Path, *, checkpoint_safety_report: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if checkpoint_safety_report["status"] != "pass":
        raise RuntimeError(f"checkpoint safety failed: {checkpoint_safety_report['reason']}")


def all_finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(all_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite_tree(item) for item in value)
    return True


def float_value(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def make_actions(policy: str, num_envs: int, action_dim: int, device: torch.device, scale: float) -> torch.Tensor:
    if policy == "zero":
        return torch.zeros((num_envs, action_dim), device=device)
    if policy == "random":
        return (2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0) * scale
    raise ValueError(f"unsupported policy for direct action generation: {policy}")


def action_dim_from_env(env: gym.Env) -> int:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_manager"):
        return int(unwrapped.action_manager.total_action_dim)
    return int(gym.spaces.flatdim(unwrapped.single_action_space))


def set_runner_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_env(env: Any, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        seed_method = getattr(env, "seed", None)
        if not callable(seed_method) or not _try_seed_method(seed_method, int(seed)):
            unwrapped_seed_method = getattr(getattr(env, "unwrapped", None), "seed", None)
            if callable(unwrapped_seed_method):
                _try_seed_method(unwrapped_seed_method, int(seed))
        return env.reset()


def _try_seed_method(seed_method: Any, seed: int) -> bool:
    try:
        seed_method(seed)
        return True
    except TypeError:
        try:
            seed_method(seed=seed)
            return True
        except TypeError:
            return False


def step_env(env: Any, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    step_result = env.step(actions)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, extras = step_result
        return obs, reward, terminated.bool(), truncated.bool(), extras
    if len(step_result) == 4:
        obs, reward, dones, extras = step_result
        dones_bool = dones.bool()
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
        truncated = torch.zeros_like(dones_bool, dtype=torch.bool) if time_outs is None else time_outs.bool()
        terminated = dones_bool & ~truncated
        return obs, reward, terminated, truncated, extras
    raise RuntimeError(f"unexpected env.step return length: {len(step_result)}")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def has_nonzero_push(config: dict[str, Any]) -> bool:
    velocity_range = config["push"]["velocity_range_mps"]
    return any(abs(float(bound)) > 1.0e-9 for axis in ("x", "y") for bound in velocity_range[axis])


if __name__ == "__main__":
    exit_code = main()
    exit_code = close_isaac_app_on_success(
        simulation_app,
        exit_code=exit_code,
    )
    os._exit(exit_code)

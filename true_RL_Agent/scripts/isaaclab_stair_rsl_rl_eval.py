#!/usr/bin/env python3
"""Evaluate a runner-sealed checkpoint on the fixed Stair protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.sealed_validation import (  # noqa: E402
    build_sealed_validation_gate,
)
from true_rl_agent.runner.checkpoint import (  # noqa: E402
    build_checkpoint_safety_report,
)
from true_rl_agent.runner.eval_isolation import (  # noqa: E402
    build_candidate_isolation_gate,
)
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    StairEvalEntrypointContext,
    validate_stair_eval_invocation,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_sealed_rollout_evidence,
)


def _build_parser(app_launcher_class: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "--job-id",
        "--round",
        "--task-id",
        "--subject-kind",
        "--subject-id",
        "--subject-manifest-hash",
        "--program-source-checksum",
        "--training-binding-hash",
        "--checkpoint",
        "--checkpoint-sha256",
        "--eval-protocol-version",
        "--task-protocol-hash",
        "--acceptance-protocol-hash",
        "--seed-cohort-hash",
        "--evaluation-binding-hash",
        "--capability-profile-hash",
        "--summary-path",
        "--profile",
        "--runtime-dynamics-hash",
        "--config",
    ):
        parser.add_argument(option, required=True)
    for option in (
        "--public-seeds",
        "--public-accept-seeds",
        "--sealed-validation-seeds",
        "--num-envs",
        "--seed",
    ):
        parser.add_argument(option, type=int, required=True)
    app_launcher_class.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    app_launcher_class: Any | None = None,
) -> int:
    if app_launcher_class is None:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        app_launcher_class = AppLauncher

    args = _build_parser(app_launcher_class).parse_args(argv)
    try:
        context = validate_stair_eval_invocation(args, project_root=PROJECT_ROOT)
    except (TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    checkpoint_report = build_checkpoint_safety_report(
        context.checkpoint_path,
        expected_checkpoint_hash=context.checkpoint_hash,
    ).to_public_dict()
    if checkpoint_report["status"] != "pass":
        print(str(checkpoint_report["reason"]), file=sys.stderr)
        return 2

    app_launcher = app_launcher_class(args)
    simulation_app = app_launcher.app
    try:
        summary = _run_eval(
            args,
            context=context,
            checkpoint_report=checkpoint_report,
        )
        status_code = 0
    except Exception as exc:  # noqa: BLE001
        summary = _failure_summary(
            context,
            checkpoint_report=checkpoint_report,
            error=exc,
        )
        status_code = 1

    summary_path = _local_output_path(args.summary_path, "summary_path")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": summary["status"], "summary_path": str(summary_path)},
            sort_keys=True,
        ),
        flush=True,
    )
    return close_isaac_app_on_success(
        simulation_app,
        exit_code=status_code,
    )


def _run_eval(
    args: argparse.Namespace,
    *,
    context: StairEvalEntrypointContext,
    checkpoint_report: Mapping[str, Any],
) -> dict[str, Any]:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab_tasks  # type: ignore[import-not-found]  # noqa: F401
    from isaaclab_rl.rsl_rl import (  # type: ignore[import-not-found]
        RslRlVecEnvWrapper,
    )
    from isaaclab_tasks.utils.parse_cfg import (  # type: ignore[import-not-found]
        load_cfg_from_registry,
    )
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

    from true_rl_agent.tasks.stair_climbing.evaluation_runtime import (
        run_all_stair_splits,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import (
        STAIR_EVAL_GYM_ID,
    )

    register_production_stair_envs()
    if not isinstance(args.device, str) or not args.device:
        raise ValueError("stair eval device must be a non-empty string")
    device = args.device
    env_cfg = build_stair_eval_env_cfg(
        context.config.protocol_v1.to_dict(),
        seed=context.config.seed_resolver().base_seed,
        device=device,
    )
    base_env = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg, render_mode=None)
    env = None
    try:
        agent_cfg = load_cfg_from_registry(
            "Isaac-Velocity-Rough-G1-v0",
            "rsl_rl_cfg_entry_point",
        )
        context.config.training_agent_configuration().apply_and_verify(agent_cfg)
        agent_cfg.seed = context.config.train_seed
        if hasattr(agent_cfg, "device"):
            agent_cfg.device = device
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(context.checkpoint_path))
        inference_policy = runner.get_inference_policy(device=env.unwrapped.device)
        summaries = run_all_stair_splits(
            env=env,
            inference_policy=inference_policy,
            protocol=context.config.protocol_v1,
            seed_resolver=context.config.seed_resolver(),
        )
    finally:
        target = env if env is not None else base_env
        try:
            target.close()
        except Exception:
            pass
    evaluation_protocol = context.config.evaluation_protocol()
    sealed_evidence = build_stair_sealed_rollout_evidence(
        context.config.protocol_v1,
        evaluation_protocol=evaluation_protocol,
        summary=summaries["sealed_validation"],
        checkpoint_hash=context.checkpoint_hash,
        evaluation_binding_hash=context.evaluation_binding.binding_hash,
        evaluation_subject=context.evaluation_subject,
        subject_manifest_hash=context.subject_manifest_hash,
        program_source_checksum=context.program_source_checksum,
        training_binding_hash=context.training_binding_hash,
        capability_profile_hash=context.capability_profile_hash,
    )
    isolation_gate = build_candidate_isolation_gate()
    sealed_gate = build_sealed_validation_gate(
        rollout_evidence=sealed_evidence,
        checkpoint_safety_status=str(checkpoint_report["status"]),
        checkpoint_hash_match_status=str(
            checkpoint_report["checkpoint_hash_match_status"]
        ),
        candidate_isolation_gate=isolation_gate,
        expected_eval_protocol_hash=evaluation_protocol.protocol_hash,
        expected_evaluation_binding_hash=context.evaluation_binding.binding_hash,
    )
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.eval_summary.v1",
        **context.identity.to_public_dict(),
        "evaluation_protocol": evaluation_protocol.to_public_dict(),
        "evaluation_binding": context.evaluation_binding.to_public_dict(),
        "status": "pass",
        "public_score": float(summaries["public_accept"]["score_stats"]["mean"]),
        "metrics": {
            "stair_evaluation": {
                "public_dev": summaries["public_dev"],
                "public_accept": summaries["public_accept"],
            }
        },
        "checkpoint_safety_report": dict(checkpoint_report),
        "sealed_validation_evidence": sealed_evidence,
        "sealed_validation_gate": sealed_gate,
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [],
    }


def _failure_summary(
    context: StairEvalEntrypointContext,
    *,
    checkpoint_report: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.eval_summary.v1",
        **context.identity.to_public_dict(),
        "evaluation_protocol": context.config.evaluation_protocol().to_public_dict(),
        "evaluation_binding": context.evaluation_binding.to_public_dict(),
        "status": "fail",
        "public_score": 0.0,
        "metrics": {"stair_evaluation": {}},
        "checkpoint_safety_report": dict(checkpoint_report),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [{"type": type(error).__name__, "message": str(error)}],
    }


def _local_output_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field_name} must be a local path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

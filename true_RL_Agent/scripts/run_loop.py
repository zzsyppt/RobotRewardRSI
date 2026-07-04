from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.live_loop import run_live_loop
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
)
from true_rl_agent.audit import run_loop_audit
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter, IsaacLabRslRlBackendConfig
from true_rl_agent.candidates.verifier import StaticCandidateVerifier
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery import PushRecoveryTaskConfig, PushRecoveryTaskPlugin, PushTaskMemoryAdapter


DEFAULT_PUSH_TASK_CONFIG = PushRecoveryTaskConfig()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    output_dir = args.output_dir or _default_output_dir("live" if args.mode == "live" else "audit")
    if args.mode == "live":
        summary = _run_live(args, output_dir)
        print(json.dumps(summary["checks"], indent=2, sort_keys=True))
        return 0 if all(summary["checks"].values()) else 1

    summary = run_loop_audit(output_dir=output_dir, rounds=args.rounds, session_id=args.session_id)
    print(json.dumps(summary["checks"], indent=2, sort_keys=True))
    return 0 if all(summary["checks"].values()) else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the true_rl_agent loop.")
    parser.add_argument("--mode", choices=("dry-run-audit", "live"), default="dry-run-audit")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--provider-base-url", default="https://api.deepseek.com")
    parser.add_argument("--provider-endpoint", default="/chat/completions")
    parser.add_argument("--provider-model", default="deepseek-v4-pro")
    parser.add_argument("--provider-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--provider-env-file", type=Path, default=None)
    parser.add_argument("--provider-timeout-s", type=int, default=180)
    parser.add_argument("--provider-max-tokens", type=int, default=8192)
    parser.add_argument("--provider-max-attempts", type=int, default=2)
    parser.add_argument("--isaac-python", default="/data/yuyangliu/envs/isaaclab/bin/python")
    parser.add_argument("--train-entrypoint", default="scripts/isaaclab_rsl_rl_train.py")
    parser.add_argument("--eval-entrypoint", default="scripts/isaaclab_rsl_rl_eval.py")
    parser.add_argument("--budget-steps", type=int, default=DEFAULT_PUSH_TASK_CONFIG.budget_steps)
    parser.add_argument("--public-seed-count", type=int, default=DEFAULT_PUSH_TASK_CONFIG.public_seed_count)
    parser.add_argument("--public-accept-seed-count", type=int, default=DEFAULT_PUSH_TASK_CONFIG.public_accept_seed_count)
    parser.add_argument("--trusted-seed-count", type=int, default=7)
    parser.add_argument("--train-profile", default="strong_push")
    parser.add_argument("--eval-profile", default="strong_push")
    parser.add_argument("--train-num-envs", type=int, default=DEFAULT_PUSH_TASK_CONFIG.train_num_envs)
    parser.add_argument("--train-max-iterations", type=int, default=DEFAULT_PUSH_TASK_CONFIG.train_max_iterations)
    parser.add_argument(
        "--train-steps-per-iteration",
        type=int,
        default=DEFAULT_PUSH_TASK_CONFIG.train_steps_per_iteration,
    )
    parser.add_argument("--eval-num-envs", type=int, default=DEFAULT_PUSH_TASK_CONFIG.eval_num_envs)
    parser.add_argument("--runner-timeout-s", type=int, default=DEFAULT_PUSH_TASK_CONFIG.runner_timeout_s)
    parser.add_argument("--initial-best-score", type=float, default=0.0)
    parser.add_argument("--policy-accept-delta", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.mode == "live":
        _validate_training_budget_args(parser, args)
    return args


def _validate_training_budget_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    planned_env_steps = args.train_num_envs * args.train_steps_per_iteration * args.train_max_iterations
    if args.budget_steps != planned_env_steps:
        parser.error(
            "training budget mismatch: "
            f"--budget-steps={args.budget_steps} but "
            f"--train-num-envs * --train-steps-per-iteration * --train-max-iterations={planned_env_steps}"
        )


def _run_live(args: argparse.Namespace, output_dir: Path) -> dict:
    session_id = args.session_id or f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    provider = OpenAICompatibleCandidateProvider(_build_provider_config(args, output_dir))
    backend = IsaacLabRslRlBackendAdapter(_build_backend_config(args, output_dir))
    runner = _build_runner(args)
    task_config = _build_task_config(args)
    return run_live_loop(
        output_dir=output_dir,
        rounds=args.rounds,
        session_id=session_id,
        task_plugin=PushRecoveryTaskPlugin(task_config),
        backend_adapter=backend,
        runner=runner,
        verifier=StaticCandidateVerifier(),
        provider=provider,
        policy_initialization=PolicyInitialization.random(),
        initial_best_score=args.initial_best_score,
        policy_accept_delta=args.policy_accept_delta,
        task_memory_adapter=PushTaskMemoryAdapter(),
        max_provider_attempts=args.provider_max_attempts,
    )


def _build_provider_config(args: argparse.Namespace, output_dir: Path) -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        model=args.provider_model,
        output_dir=output_dir / "candidates",
        base_url=args.provider_base_url,
        endpoint=args.provider_endpoint,
        api_key_env=args.provider_api_key_env,
        env_file=args.provider_env_file,
        max_tokens=args.provider_max_tokens,
        timeout_s=args.provider_timeout_s,
    )


def _build_backend_config(args: argparse.Namespace, output_dir: Path) -> IsaacLabRslRlBackendConfig:
    return IsaacLabRslRlBackendConfig(
        isaac_lab_python=args.isaac_python,
        train_entrypoint=args.train_entrypoint,
        eval_entrypoint=args.eval_entrypoint,
        default_budget_steps=args.budget_steps,
        public_seed_count=args.public_seed_count,
        trusted_seed_count=args.trusted_seed_count,
        public_accept_seed_count=args.public_accept_seed_count,
        artifact_root=str(output_dir / "runner"),
        train_profile=args.train_profile,
        eval_profile=args.eval_profile,
        train_num_envs=args.train_num_envs,
        train_max_iterations=args.train_max_iterations,
        train_steps_per_iteration=args.train_steps_per_iteration,
        eval_num_envs=args.eval_num_envs,
        work_dir=str(PROJECT_ROOT),
    )


def _build_task_config(args: argparse.Namespace) -> PushRecoveryTaskConfig:
    return PushRecoveryTaskConfig(
        budget_steps=args.budget_steps,
        train_num_envs=args.train_num_envs,
        train_max_iterations=args.train_max_iterations,
        train_steps_per_iteration=args.train_steps_per_iteration,
        eval_num_envs=args.eval_num_envs,
        public_seed_count=args.public_seed_count,
        public_accept_seed_count=args.public_accept_seed_count,
        runner_timeout_s=args.runner_timeout_s,
    )


def _build_runner(args: argparse.Namespace) -> SubprocessRunner:
    return SubprocessRunner(
        allowed_executables=(args.isaac_python,),
        timeout_s=args.runner_timeout_s,
        cwd=PROJECT_ROOT,
    )


def _default_output_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / f"{prefix}_{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())

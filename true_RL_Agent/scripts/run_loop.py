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

from true_rl_agent.agent_core.live_loop import (
    DEFAULT_MAX_PROVIDER_ATTEMPTS,
    run_live_loop,
)
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
)
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
from true_rl_agent.candidates.verifier import StaticCandidateVerifier
from true_rl_agent.composition import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionRegistry,
    resolve_task_composition,
)
from true_rl_agent.composition.cli import (
    add_task_runtime_arguments,
    add_task_selection_argument,
    select_task_composition_provider,
)
from true_rl_agent.composition.defaults import build_application_composition_registry


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    output_dir = args.output_dir or _default_output_dir("live" if args.mode == "live" else "audit")
    composition = _resolve_composition(args, output_dir)
    if args.mode == "live":
        summary = _run_live(args, output_dir, composition)
        print(json.dumps(summary["checks"], indent=2, sort_keys=True))
        return 0 if all(summary["checks"].values()) else 1

    provider = args._composition_provider
    summary = provider.run_audit(
        composition=composition,
        output_dir=output_dir,
        rounds=args.rounds,
        session_id=args.session_id,
    )
    print(json.dumps(summary["checks"], indent=2, sort_keys=True))
    return 0 if all(summary["checks"].values()) else 1


def _parse_args(
    argv: list[str] | None = None,
    *,
    registry: TaskCompositionRegistry | None = None,
) -> argparse.Namespace:
    registry = registry or build_application_composition_registry()
    provider = select_task_composition_provider(argv, registry)
    task_defaults = provider.cli_defaults()
    parser = argparse.ArgumentParser(description="Run the true_rl_agent loop.")
    add_task_selection_argument(
        parser,
        registry=registry,
        provider=provider,
    )
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
    parser.add_argument("--provider-max-attempts", type=int, default=DEFAULT_MAX_PROVIDER_ATTEMPTS)
    add_task_runtime_arguments(parser, task_defaults)
    parser.add_argument("--initial-best-score", type=float, default=0.0)
    parser.add_argument(
        "--policy-accept-delta",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility check only; it cannot override the "
            "task-owned acceptance protocol."
        ),
    )
    args = parser.parse_args(argv)
    args._composition_provider = provider
    args._composition_registry = registry
    return args


def _run_live(
    args: argparse.Namespace,
    output_dir: Path,
    composition: ResolvedTaskComposition,
) -> dict:
    session_id = args.session_id or f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if composition.genesis_anchor_kind != "bootstrap" and args.initial_best_score != 0.0:
        raise ValueError(
            "--initial-best-score is only valid for a bootstrap Genesis"
        )
    session_genesis = prepare_session_genesis(
        output_dir=output_dir,
        session_id=session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        policy_initialization=composition.policy_initialization,
        runtime_snapshot=composition.runtime_snapshot(),
        baseline_manifest=composition.baseline_manifest,
        reference_program_manifest=composition.reference_program_manifest,
        bootstrap_initial_best_score=(
            args.initial_best_score
            if composition.genesis_anchor_kind == "bootstrap"
            else None
        ),
    )
    provider = OpenAICompatibleCandidateProvider(_build_provider_config(args, output_dir))
    return run_live_loop(
        output_dir=output_dir,
        rounds=args.rounds,
        session_id=session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        verifier=StaticCandidateVerifier(),
        provider=provider,
        policy_initialization=composition.policy_initialization,
        session_genesis=session_genesis,
        policy_accept_delta=args.policy_accept_delta,
        task_memory_adapter=composition.task_memory_adapter,
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


def _resolve_composition(args: argparse.Namespace, output_dir: Path):
    provider = getattr(args, "_composition_provider", None)
    if provider is None:
        registry = build_application_composition_registry()
        provider = registry.get(args.task_id)
    return resolve_task_composition(
        provider=provider,
        values=vars(args),
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )


def _default_output_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / f"{prefix}_{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())

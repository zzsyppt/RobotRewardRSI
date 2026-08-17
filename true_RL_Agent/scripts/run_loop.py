from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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
from true_rl_agent.agent_core.provider_limits import (
    DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS,
    DEFAULT_PROVIDER_MODEL,
    DEFAULT_PROVIDER_REASONING_EFFORT,
    DEFAULT_PROVIDER_REASONING_MODE,
    PROVIDER_REASONING_EFFORTS,
    PROVIDER_REASONING_MODES,
)
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
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

_DEEPSEEK_ADAPTIVE_REASONING_MODELS = frozenset(
    {"deepseek-v4-flash", "deepseek-v4-pro"}
)


def main(
    argv: list[str] | None = None,
    *,
    registry: TaskCompositionRegistry | None = None,
    require_existing_genesis: bool = False,
    live_summary_validator: Callable[[dict], None] | None = None,
) -> int:
    if live_summary_validator is not None and not callable(live_summary_validator):
        raise TypeError("live_summary_validator must be callable")
    args = _parse_args(argv, registry=registry)

    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    output_dir = args.output_dir or _default_output_dir("live" if args.mode == "live" else "audit")
    composition = _resolve_composition(args, output_dir)
    if args.mode == "live":
        summary = _run_live(
            args,
            output_dir,
            composition,
            require_existing_genesis=require_existing_genesis,
            live_summary_validator=live_summary_validator,
        )
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
    parser.add_argument(
        "--session-genesis-dir",
        type=Path,
        default=None,
        help=(
            "Read an immutable existing Genesis from this directory while "
            "writing a fresh ledger epoch under --output-dir."
        ),
    )
    parser.add_argument(
        "--ledger-epoch-id",
        default=None,
        help="Required identity when --session-genesis-dir differs from --output-dir.",
    )
    parser.add_argument("--provider-base-url", default="https://api.deepseek.com")
    parser.add_argument("--provider-endpoint", default="/chat/completions")
    parser.add_argument("--provider-model", default=DEFAULT_PROVIDER_MODEL)
    parser.add_argument("--provider-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--provider-env-file", type=Path, default=None)
    parser.add_argument("--provider-timeout-s", type=int, default=180)
    parser.add_argument(
        "--provider-max-tokens",
        type=int,
        default=DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--provider-reasoning-mode",
        choices=PROVIDER_REASONING_MODES,
        default=None,
        help=(
            "Provider reasoning policy. Adaptive keeps reasoning for new designs "
            "and disables it only when retrying a truncated, already-materialized "
            "failed draft. When omitted, the official DeepSeek V4 chat endpoint "
            "uses adaptive; other OpenAI-compatible endpoints use provider_default."
        ),
    )
    parser.add_argument(
        "--provider-reasoning-effort",
        choices=PROVIDER_REASONING_EFFORTS,
        default=DEFAULT_PROVIDER_REASONING_EFFORT,
        help=(
            "Reasoning intensity for thinking-enabled provider requests. "
            "The project default is max; disabled-thinking repairs omit it."
        ),
    )
    parser.add_argument("--provider-max-attempts", type=int, default=DEFAULT_MAX_PROVIDER_ATTEMPTS)
    add_task_runtime_arguments(parser, task_defaults)
    provider.add_cli_arguments(parser)
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
    if args.provider_reasoning_mode is None:
        args.provider_reasoning_mode = _default_cli_provider_reasoning_mode(
            base_url=args.provider_base_url,
            endpoint=args.provider_endpoint,
            model=args.provider_model,
        )
    args._composition_provider = provider
    args._composition_registry = registry
    return args


def _default_cli_provider_reasoning_mode(
    *,
    base_url: str,
    endpoint: str,
    model: str,
) -> str:
    """Choose a safe CLI default without changing the provider library ABI."""

    try:
        parsed = urlsplit(base_url.strip())
        port = parsed.port
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_PROVIDER_REASONING_MODE
    official_deepseek_origin = (
        parsed.scheme.lower() == "https"
        and parsed.hostname == "api.deepseek.com"
        and port in {None, 443}
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
        and parsed.path.rstrip("/") in {"", "/v1"}
    )
    normalized_endpoint = "/" + endpoint.strip().lstrip("/")
    official_chat_endpoint = normalized_endpoint == "/chat/completions"
    supported_model = model.strip().lower() in _DEEPSEEK_ADAPTIVE_REASONING_MODELS
    if official_deepseek_origin and official_chat_endpoint and supported_model:
        return "adaptive"
    return DEFAULT_PROVIDER_REASONING_MODE


def _run_live(
    args: argparse.Namespace,
    output_dir: Path,
    composition: ResolvedTaskComposition,
    *,
    require_existing_genesis: bool = False,
    live_summary_validator: Callable[[dict], None] | None = None,
) -> dict:
    if require_existing_genesis and args.session_id is None:
        raise ValueError("an existing Genesis live run requires --session-id")
    session_id = args.session_id or f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    session_genesis_dir = args.session_genesis_dir or output_dir
    if composition.genesis_anchor_kind != "bootstrap" and args.initial_best_score != 0.0:
        raise ValueError(
            "--initial-best-score is only valid for a bootstrap Genesis"
        )
    bootstrap_initial_best_score = None
    if composition.genesis_anchor_kind == "bootstrap":
        bootstrap_initial_best_score = (
            composition.bootstrap_initial_best_score
            if composition.bootstrap_initial_best_score is not None
            else args.initial_best_score
        )
    session_genesis = prepare_session_genesis(
        output_dir=session_genesis_dir,
        session_id=session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        policy_initialization=composition.policy_initialization,
        genesis_identity_snapshot=composition.genesis_identity_snapshot(),
        baseline_manifest=composition.baseline_manifest,
        reference_program_artifact=composition.reference_program_artifact,
        bootstrap_initial_best_score=(
            bootstrap_initial_best_score
        ),
        bootstrap_initial_policy_state_hash=(
            composition.bootstrap_initial_policy_state_hash
        ),
        require_existing=require_existing_genesis,
        bind_backend_execution_environment=True,
    )
    provider = OpenAICompatibleCandidateProvider(_build_provider_config(args, output_dir))
    return run_live_loop(
        output_dir=output_dir,
        rounds=args.rounds,
        session_id=session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        candidate_admission_policy=composition.candidate_admission_policy,
        provider=provider,
        policy_initialization=composition.policy_initialization,
        session_genesis=session_genesis,
        session_genesis_dir=session_genesis_dir,
        ledger_epoch_id=args.ledger_epoch_id,
        policy_accept_delta=args.policy_accept_delta,
        task_memory_adapter=composition.task_memory_adapter,
        authoring_capability_readiness=(
            composition.authoring_capability_readiness
        ),
        max_provider_attempts=args.provider_max_attempts,
        completion_validator=live_summary_validator,
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
        reasoning_mode=args.provider_reasoning_mode,
        reasoning_effort=args.provider_reasoning_effort,
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

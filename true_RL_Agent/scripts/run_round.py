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

from true_rl_agent.composition import (
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
    provider = args._composition_provider
    output_dir = args.output_dir or _default_output_dir("round")
    composition = _resolve_composition(args, output_dir)
    summary = provider.run_audit(
        composition=composition,
        output_dir=output_dir,
        rounds=1,
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
    parser = argparse.ArgumentParser(description="Run one true_rl_agent dry-run audit round.")
    add_task_selection_argument(
        parser,
        registry=registry,
        provider=provider,
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    add_task_runtime_arguments(parser, provider.cli_defaults())
    args = parser.parse_args(argv)
    args._composition_provider = provider
    args._composition_registry = registry
    return args


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

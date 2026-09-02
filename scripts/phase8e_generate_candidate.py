#!/usr/bin/env python3
"""Generate and stage a Phase 8E candidate package from an LLM provider.

This command implements Phase 8E-6-3 only: prompt construction, provider call,
candidate package landing, and optional Phase 8D verification.  It does not
launch training, evaluation, decision, or ledger inheritance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e import Phase8EError  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_provider import Phase8EProviderError, run_phase8e_provider_candidate  # noqa: E402
from humanoid_agent.phase5_llm_agent.providers import ProviderError  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start/phase8e_loop.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lab-dir", default=None, help="Stable experiment dir for prompt history/accepted recipe lineage.")
    parser.add_argument("--current-recipe-dir", default=None, help="Optional current recipe override for prompt construction.")
    parser.add_argument("--provider", default=None, help="Provider name from config.provider; defaults to config.provider.default.")
    parser.add_argument("--replay-response", default=None, help="Replay JSON response file for offline provider testing.")
    parser.add_argument("--allow-api-spend", action="store_true", help="Required for networked openai-compatible providers.")
    parser.add_argument("--skip-verify", action="store_true", help="Only stage the package; skip Phase 8D. Intended for parser tests.")
    parser.add_argument("--require-server-enforcement", action="store_true", help="Require hard isolation in Phase 8D dynamic checks.")
    parser.add_argument("--python-executable", default=None, help="Python executable for Phase 8D isolated dynamic checks.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        summary = run_phase8e_provider_candidate(
            config_path=Path(args.config).resolve(),
            run_dir=Path(args.run_dir).resolve(),
            lab_dir=Path(args.lab_dir).resolve() if args.lab_dir else None,
            current_recipe_dir=Path(args.current_recipe_dir).resolve() if args.current_recipe_dir else None,
            provider_name=args.provider,
            replay_response=Path(args.replay_response).resolve() if args.replay_response else None,
            allow_api_spend=args.allow_api_spend,
            verify=not args.skip_verify,
            require_server_enforcement=args.require_server_enforcement,
            python_executable=args.python_executable,
        )
    except (Phase8EProviderError, Phase8EError, ProviderError, ValueError, json.JSONDecodeError) as exc:
        print(f"phase8e_provider_error={exc}", file=sys.stderr)
        return 1
    print(json.dumps({"summary_file": str(Path(args.run_dir).resolve() / "phase8e_provider_candidate_summary.json"), "status": summary["status"]}, ensure_ascii=False))
    return 0 if summary["status"] in {"pass", "staged_unverified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

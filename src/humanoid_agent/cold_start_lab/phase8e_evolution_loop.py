"""Phase 8E multi-round self-evolution loop.

This is a thin sequential driver over :func:`run_phase8e_full_round`.  Rounds are
sequential by construction: every round shares one ``lab_dir`` so the provider
prompt inherits the accumulated public ledger and the latest accepted recipe,
which is how round N+1 "sees" round N's public feedback.  Policy weights still
cold-start each round; only recipe code is inherited.  The runner keeps owning
budget, eval, checkpoint selection, and decisions for every round.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .phase8e_full_round import run_phase8e_full_round


EVOLUTION_LOOP_SUMMARY_SCHEMA = "humanoid_agent.phase8e.evolution_loop.summary.v1"

FullRoundRunner = Callable[..., tuple[int, dict[str, Any]]]


class Phase8EEvolutionLoopError(RuntimeError):
    """Raised when the evolution loop cannot start safely."""


def run_phase8e_evolution_loop(
    *,
    config_path: Path,
    run_root: Path,
    lab_dir: Path | None = None,
    max_rounds: int = 3,
    provider_name: str | None = None,
    replay_response: Path | None = None,
    allow_api_spend: bool = False,
    require_server_enforcement: bool = True,
    python_executable: str | None = None,
    stop_after_no_improve: int | None = None,
    full_round_runner: FullRoundRunner = run_phase8e_full_round,
) -> tuple[int, dict[str, Any]]:
    """Run up to ``max_rounds`` self-evolution rounds, then stop.

    A round that rejects/blocks is a normal outcome and does not stop the loop;
    the loop stops at ``max_rounds`` (or early if ``stop_after_no_improve`` is set
    and that many consecutive rounds produce no new best public-accept mean).
    """

    if max_rounds < 1:
        raise Phase8EEvolutionLoopError("max_rounds must be >= 1")
    if stop_after_no_improve is not None and stop_after_no_improve < 1:
        raise Phase8EEvolutionLoopError("stop_after_no_improve must be >= 1 when set")

    started = time.monotonic()
    created_at_utc = _now()
    config_path = config_path.resolve()
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # A shared lab_dir is REQUIRED for cross-round inheritance/feedback; without
    # it each round's provider prompt would start from an empty ledger.
    effective_lab_dir = (lab_dir.resolve() if lab_dir is not None else run_root / "lab")
    effective_lab_dir.mkdir(parents=True, exist_ok=True)

    rounds: list[dict[str, Any]] = []
    accepted_rounds = 0
    best_accept_mean: float | None = None
    no_improve_streak = 0
    stop_reason = "max_rounds_reached"

    for index in range(1, max_rounds + 1):
        run_dir = run_root / f"round_{index:03d}"
        exit_code, summary = full_round_runner(
            config_path=config_path,
            run_dir=run_dir,
            lab_dir=effective_lab_dir,
            provider_name=provider_name,
            replay_response=replay_response,
            allow_api_spend=allow_api_spend,
            append_ledger=True,
            inherit_if_accepted=True,
            require_server_enforcement=require_server_enforcement,
            python_executable=python_executable,
        )
        status = str(summary.get("status", "unknown"))
        accepted = status == "accept"
        accept_mean = _round_public_accept_mean(summary)
        improved = False
        if accepted:
            accepted_rounds += 1
            if accept_mean is not None and (best_accept_mean is None or accept_mean > best_accept_mean):
                best_accept_mean = accept_mean
                improved = True
        no_improve_streak = 0 if improved else no_improve_streak + 1

        rounds.append(
            {
                "round": index,
                "run_dir": str(run_dir),
                "status": status,
                "accepted": accepted,
                "exit_code": int(exit_code),
                "public_accept_mean": accept_mean,
                "formal_training_executed": summary.get("formal_training_executed"),
                "failure_classification": summary.get("failure_classification"),
                "summary_file": summary.get("summary_file"),
            }
        )

        if stop_after_no_improve is not None and no_improve_streak >= stop_after_no_improve:
            stop_reason = f"no_improve_for_{stop_after_no_improve}_rounds"
            break

    aggregate = {
        "schema_version": EVOLUTION_LOOP_SUMMARY_SCHEMA,
        "status": "completed",
        "created_at_utc": created_at_utc,
        "config_path": str(config_path),
        "run_root": str(run_root),
        "lab_dir": str(effective_lab_dir),
        "max_rounds": max_rounds,
        "rounds_executed": len(rounds),
        "accepted_rounds": accepted_rounds,
        "best_public_accept_mean": best_accept_mean,
        "stop_reason": stop_reason,
        "policy_init": "random_each_round_cold_start",
        "recipe_inheritance": "accepted_recipe_via_shared_lab_dir",
        "rounds": rounds,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "completed_at_utc": _now(),
    }
    summary_path = run_root / "phase8e_evolution_loop_summary.json"
    aggregate["summary_file"] = str(summary_path)
    _write_json(summary_path, aggregate)
    return 0, aggregate


def _round_public_accept_mean(summary: Mapping[str, Any]) -> float | None:
    """Best-effort read of a round's public-accept mean for reporting only.

    The authoritative number lives in the nested linked/decision summary; if it
    cannot be read we return None and the loop simply lacks an improvement signal
    for that round (control flow then relies on ``max_rounds``).
    """

    stages = summary.get("stages") if isinstance(summary.get("stages"), Mapping) else {}
    linked = stages.get("linked_round") if isinstance(stages.get("linked_round"), Mapping) else {}
    summary_file = linked.get("summary_file") if isinstance(linked, Mapping) else None
    if not isinstance(summary_file, str) or not Path(summary_file).is_file():
        return None
    try:
        linked_summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))
        decision = linked_summary["round_summary"]["decision"]
        detail = decision.get("detail", {}) if isinstance(decision, Mapping) else {}
        value = detail.get("candidate_public_accept_mean")
        return float(value) if value is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

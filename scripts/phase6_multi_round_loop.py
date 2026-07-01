#!/usr/bin/env python3
"""Run or plan the Phase 6 multi-round public-only LLM improvement loop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - CLI YAML fallback.
from humanoid_agent.phase5_llm_agent.runner import run_phase5_iteration
from humanoid_agent.phase6_loop.decision import build_phase6_public_decision, public_feedback_from_decision, write_json
from humanoid_agent.phase6_loop.feedback import build_phase6_feedback_summary, assert_public_only_payload
from humanoid_agent.phase6_loop.summaries import load_json, subset_public_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6-config", default="configs/phase6_loop.yaml")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--replay-response", default=None)
    parser.add_argument("--allow-api-spend", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually call the provider and launch training/eval commands.")
    parser.add_argument("--skip-train-eval", action="store_true", help="Only generate proposals; useful for API plumbing tests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    phase6_config_path = Path(args.phase6_config)
    if not phase6_config_path.is_absolute():
        phase6_config_path = REPO_ROOT / phase6_config_path
    cfg = _load_structured_file(phase6_config_path)
    seed_policy = cfg["seed_policy"]
    initial_seeds = [int(seed) for seed in seed_policy["initial_train_seeds"]]
    expanded_seeds = [int(seed) for seed in seed_policy["expanded_train_seeds"]]
    rounds = int(args.rounds or cfg["loop"].get("min_rounds", 3))
    rounds = min(rounds, int(cfg["loop"].get("max_rounds", rounds)))

    run_root = Path(args.run_root or cfg["loop"]["run_root"])
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_phase6_loop")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = Path(args.baseline_summary or cfg["loop"]["baseline_summary"])
    if not baseline_summary_path.is_absolute():
        baseline_summary_path = REPO_ROOT / baseline_summary_path
    baseline_source = load_json(baseline_summary_path)
    best_summary = subset_public_summary(baseline_source, seeds=initial_seeds, summary_path=baseline_summary_path)
    best_summary_path = run_dir / "current_best" / "round_00_baseline_public_3seed_summary.json"
    write_json(best_summary_path, best_summary)

    current_best_phase1 = _resolve_repo_path(cfg["configs"]["baseline_phase1"])
    current_best_phase2 = _resolve_repo_path(cfg["configs"]["baseline_phase2"])
    current_best_full_summary_path: Path | None = baseline_summary_path
    accepted_count = 0
    history: list[dict[str, Any]] = []
    rounds_out: list[dict[str, Any]] = []
    provider = args.provider or cfg["loop"].get("provider")

    for round_index in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        feedback = build_phase6_feedback_summary(
            best_public_summary=best_summary,
            history=history,
            include_history_rounds=int(cfg.get("feedback", {}).get("include_history_rounds", 3)),
        )
        assert_public_only_payload(feedback)
        feedback_file = round_dir / "prompt_public_feedback_summary.json"
        write_json(feedback_file, feedback)

        proposal_dir = round_dir / "proposal"
        proposal_cmd = _proposal_command(
            feedback_file=feedback_file,
            phase1_config=current_best_phase1,
            phase2_config=current_best_phase2,
            phase5_config=_resolve_repo_path(cfg["configs"]["phase5_agent"]),
            verifier_config=_resolve_repo_path(cfg["configs"]["verifier"]),
            output_dir=proposal_dir,
            provider=provider,
            allow_api_spend=args.allow_api_spend,
            replay_response=Path(args.replay_response) if args.replay_response else None,
        )
        round_out: dict[str, Any] = {
            "round": round_index,
            "status": "planned" if not args.execute else "started",
            "best_phase1_config": str(current_best_phase1),
            "best_phase2_config": str(current_best_phase2),
            "best_summary": str(best_summary_path),
            "feedback_summary": str(feedback_file),
            "proposal_command": proposal_cmd,
        }

        if not args.execute:
            round_out["note"] = "Dry run only; use --execute to call the provider and launch training/eval."
            rounds_out.append(round_out)
            continue

        proposal = run_phase5_iteration(
            repo_root=REPO_ROOT,
            summary_file=feedback_file,
            phase1_config_file=current_best_phase1,
            phase2_config_file=current_best_phase2,
            phase5_config_file=_resolve_repo_path(cfg["configs"]["phase5_agent"]),
            verifier_config_file=_resolve_repo_path(cfg["configs"]["verifier"]),
            output_dir=proposal_dir,
            provider_name=provider,
            replay_response=Path(args.replay_response) if args.replay_response else None,
            allow_api_spend=args.allow_api_spend,
        )
        round_out["proposal_summary"] = proposal
        if proposal.get("status") != "pass":
            round_out["status"] = "proposal_failed"
            rounds_out.append(round_out)
            history.append(_failed_decision(round_index, proposal))
            continue

        if args.skip_train_eval:
            round_out["status"] = "proposal_only_pass"
            rounds_out.append(round_out)
            continue

        try:
            candidate_summary_path = _run_public_eval(
                candidate_dir=proposal_dir,
                phase1_config=Path(proposal["candidate_phase1_config"]),
                phase2_config=Path(proposal["candidate_phase2_config"]),
                seeds=initial_seeds,
                run_name="public_3seed_eval",
            )
            final_candidate_summary_path = candidate_summary_path
            expanded_used = False
            decision_path = round_dir / "public_decision_3seed.json"
            decision = _write_decision(
                best_summary_path=best_summary_path,
                candidate_summary_path=candidate_summary_path,
                operation_diff=Path(proposal["operation_diff"]),
                seed_policy=seed_policy,
                output_file=decision_path,
            )
            round_out["public_3seed_summary"] = str(candidate_summary_path)
            round_out["public_3seed_decision"] = str(decision_path)
        except Exception as exc:  # noqa: BLE001 - keep unattended multi-round loop alive.
            decision_path = round_dir / "public_decision_eval_failed.json"
            decision = _eval_failed_decision(
                round_index=round_index,
                proposal=proposal,
                stage="candidate_public_3seed_eval_or_decision",
                exc=exc,
            )
            write_json(decision_path, decision)
            history.append(decision)
            round_out["status"] = "eval_failed_rejected"
            round_out["final_decision"] = "reject"
            round_out["public_eval_failure_decision"] = str(decision_path)
            round_out["public_eval_failure"] = _exception_digest(exc)
            rounds_out.append(round_out)
            continue

        if decision.get("needs_more_seeds"):
            try:
                best5_summary_path = _best_expanded_summary(
                    round_dir=round_dir,
                    current_best_phase1=current_best_phase1,
                    current_best_phase2=current_best_phase2,
                    current_best_full_summary_path=current_best_full_summary_path,
                    expanded_seeds=expanded_seeds,
                )
                candidate5_summary_path = _run_public_eval(
                    candidate_dir=proposal_dir,
                    phase1_config=Path(proposal["candidate_phase1_config"]),
                    phase2_config=Path(proposal["candidate_phase2_config"]),
                    seeds=expanded_seeds,
                    run_name="public_5seed_eval",
                )
                decision_path = round_dir / "public_decision_5seed.json"
                decision = _write_decision(
                    best_summary_path=best5_summary_path,
                    candidate_summary_path=candidate5_summary_path,
                    operation_diff=Path(proposal["operation_diff"]),
                    seed_policy=seed_policy,
                    output_file=decision_path,
                )
                round_out["public_5seed_best_summary"] = str(best5_summary_path)
                round_out["public_5seed_candidate_summary"] = str(candidate5_summary_path)
                round_out["public_5seed_decision"] = str(decision_path)
                final_candidate_summary_path = candidate5_summary_path
                expanded_used = True
            except Exception as exc:  # noqa: BLE001 - failed disambiguation rejects candidate, not whole loop.
                decision_path = round_dir / "public_decision_5seed_eval_failed.json"
                decision = _eval_failed_decision(
                    round_index=round_index,
                    proposal=proposal,
                    stage="candidate_public_5seed_eval_or_decision",
                    exc=exc,
                    prior_decision=decision,
                )
                write_json(decision_path, decision)
                history.append(decision)
                round_out["status"] = "seed_expansion_failed_rejected"
                round_out["final_decision"] = "reject"
                round_out["public_5seed_failure_decision"] = str(decision_path)
                round_out["public_5seed_failure"] = _exception_digest(exc)
                rounds_out.append(round_out)
                continue

        history.append(decision)
        if decision.get("accepted"):
            accepted_count += 1
            current_best_phase1, current_best_phase2 = _promote_best(
                run_dir=run_dir,
                round_index=round_index,
                candidate_phase1=Path(proposal["candidate_phase1_config"]),
                candidate_phase2=Path(proposal["candidate_phase2_config"]),
            )
            current_best_full_summary_path = final_candidate_summary_path if expanded_used else None
            if expanded_used:
                best_summary = subset_public_summary(
                    load_json(final_candidate_summary_path),
                    seeds=initial_seeds,
                    summary_path=final_candidate_summary_path,
                )
                best_summary_path = run_dir / "current_best" / f"round_{round_index:02d}_public_3seed_summary.json"
                write_json(best_summary_path, best_summary)
            else:
                best_summary = load_json(final_candidate_summary_path)
                best_summary_path = final_candidate_summary_path
            round_out["status"] = "accepted"
            round_out["promoted_phase1_config"] = str(current_best_phase1)
            round_out["promoted_phase2_config"] = str(current_best_phase2)
            if _should_reevaluate_best(cfg, accepted_count):
                try:
                    best_summary_path = _run_public_eval(
                        candidate_dir=_current_best_input_dir(run_dir, current_best_phase1, current_best_phase2),
                        phase1_config=current_best_phase1,
                        phase2_config=current_best_phase2,
                        seeds=initial_seeds,
                        run_name=f"best_recheck_after_accept_{accepted_count:02d}",
                    )
                    best_summary = load_json(best_summary_path)
                    current_best_full_summary_path = None
                    round_out["best_recheck_summary"] = str(best_summary_path)
                except Exception as exc:  # noqa: BLE001 - keep accepted best; record failed recheck.
                    round_out["best_recheck_status"] = "failed_kept_existing_best"
                    round_out["best_recheck_failure"] = _exception_digest(exc)
        else:
            round_out["status"] = "rejected"

        round_out["final_decision"] = decision.get("decision")
        rounds_out.append(round_out)

    loop_summary = {
        "schema_version": "humanoid_agent.phase6.loop.run.v1",
        "status": "pass" if args.execute else "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "run_dir": str(run_dir),
        "start_from": cfg["loop"].get("start_from"),
        "baseline_summary": str(baseline_summary_path),
        "initial_train_seeds": initial_seeds,
        "expanded_train_seeds": expanded_seeds,
        "rounds": rounds_out,
        "security_notes": [
            "Phase 6 starts from the Phase 2 baseline summary/configs.",
            "Per-round feedback uses public-only projected summaries and operation-only diffs.",
            "Raw accept/reject files and held-out diagnostics are not used as Agent prompt input.",
        ],
    }
    write_json(run_dir / "phase6_loop_summary.json", loop_summary)
    print(json.dumps(loop_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if loop_summary["status"] in {"pass", "planned"} else 1


def _proposal_command(
    *,
    feedback_file: Path,
    phase1_config: Path,
    phase2_config: Path,
    phase5_config: Path,
    verifier_config: Path,
    output_dir: Path,
    provider: str | None,
    allow_api_spend: bool,
    replay_response: Path | None,
) -> list[str]:
    command = [
        "python",
        "scripts/phase5_llm_single_iteration.py",
        "--summary-file",
        str(feedback_file),
        "--phase1-config",
        str(phase1_config),
        "--phase2-config",
        str(phase2_config),
        "--phase5-config",
        str(phase5_config),
        "--verifier-config",
        str(verifier_config),
        "--output-dir",
        str(output_dir),
    ]
    if provider:
        command.extend(["--provider", provider])
    if allow_api_spend:
        command.append("--allow-api-spend")
    if replay_response:
        command.extend(["--replay-response", str(replay_response)])
    return command


def _run_public_eval(
    *,
    candidate_dir: Path,
    phase1_config: Path,
    phase2_config: Path,
    seeds: list[int],
    run_name: str,
) -> Path:
    env = os.environ.copy()
    env["CANDIDATE_DIR"] = str(candidate_dir)
    env["CANDIDATE_PHASE1"] = str(phase1_config)
    env["CANDIDATE_PHASE2"] = str(phase2_config)
    env["SEEDS"] = " ".join(str(seed) for seed in seeds)
    env["RUN_NAME"] = run_name
    completed = subprocess.run(
        ["bash", "scripts/phase6_run_candidate_public_eval.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Phase 6 public eval failed with exit code {completed.returncode}")
    return candidate_dir / run_name / "summary.json"


def _best_expanded_summary(
    *,
    round_dir: Path,
    current_best_phase1: Path,
    current_best_phase2: Path,
    current_best_full_summary_path: Path | None,
    expanded_seeds: list[int],
) -> Path:
    if current_best_full_summary_path is not None:
        try:
            summary = subset_public_summary(load_json(current_best_full_summary_path), seeds=expanded_seeds, summary_path=current_best_full_summary_path)
            output = round_dir / "best_public_5seed_summary.json"
            write_json(output, summary)
            return output
        except ValueError:
            pass
    return _run_public_eval(
        candidate_dir=_current_best_input_dir(round_dir, current_best_phase1, current_best_phase2),
        phase1_config=current_best_phase1,
        phase2_config=current_best_phase2,
        seeds=expanded_seeds,
        run_name="best_public_5seed_eval",
    )


def _write_decision(
    *,
    best_summary_path: Path,
    candidate_summary_path: Path,
    operation_diff: Path,
    seed_policy: dict[str, Any],
    output_file: Path,
) -> dict[str, Any]:
    decision = build_phase6_public_decision(
        best_summary=load_json(best_summary_path),
        candidate_summary=load_json(candidate_summary_path),
        best_summary_path=best_summary_path,
        candidate_summary_path=candidate_summary_path,
        operation_diff=load_json(operation_diff),
        seed_policy=seed_policy,
    )
    assert_public_only_payload(public_feedback_from_decision(decision))
    write_json(output_file, decision)
    return decision


def _promote_best(*, run_dir: Path, round_index: int, candidate_phase1: Path, candidate_phase2: Path) -> tuple[Path, Path]:
    best_dir = run_dir / "current_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    phase1_out = best_dir / f"round_{round_index:02d}_phase1_push_recovery.yaml"
    phase2_out = best_dir / f"round_{round_index:02d}_phase2_baseline.yaml"
    shutil.copy2(candidate_phase1, phase1_out)
    shutil.copy2(candidate_phase2, phase2_out)
    return phase1_out, phase2_out


def _current_best_input_dir(parent: Path, phase1_config: Path, phase2_config: Path) -> Path:
    input_dir = parent / "current_best_eval_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(phase1_config, input_dir / "candidate_phase1_push_recovery.yaml")
    shutil.copy2(phase2_config, input_dir / "candidate_phase2_baseline.yaml")
    return input_dir


def _failed_decision(round_index: int, proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase6.public_decision.v1",
        "round": round_index,
        "decision": "reject",
        "accepted": False,
        "needs_more_seeds": False,
        "decision_basis": {"public_eval_only": True, "proposal_failed": True},
        "operation_diff": {},
        "proposal_errors": proposal.get("errors", []),
    }


def _eval_failed_decision(
    *,
    round_index: int,
    proposal: dict[str, Any],
    stage: str,
    exc: Exception,
    prior_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operation_diff = {}
    operation_diff_path = proposal.get("operation_diff")
    if operation_diff_path:
        try:
            operation_diff = load_json(Path(operation_diff_path))
        except Exception:  # noqa: BLE001 - failure decision must still be writable.
            operation_diff = {"status": "unavailable"}
    decision_basis: dict[str, Any] = {
        "public_eval_only": True,
        "eval_failed": True,
        "failed_stage": stage,
        "failure_policy": "reject_candidate_and_continue_loop",
        "failure": _exception_digest(exc),
    }
    if prior_decision is not None:
        decision_basis["prior_3seed_decision"] = {
            "decision": prior_decision.get("decision"),
            "needs_more_seeds": prior_decision.get("needs_more_seeds"),
            "public_delta": prior_decision.get("decision_basis", {}).get("public_delta"),
            "acceptance_threshold": prior_decision.get("decision_basis", {}).get("acceptance_threshold"),
        }
    return {
        "schema_version": "humanoid_agent.phase6.public_decision.v1",
        "round": round_index,
        "decision": "reject",
        "accepted": False,
        "needs_more_seeds": False,
        "decision_basis": decision_basis,
        "operation_diff": operation_diff,
        "feedback_safe": {
            "public_only": True,
            "contains_heldout_results": False,
            "contains_eval_seed_config": False,
        },
    }


def _exception_digest(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _should_reevaluate_best(cfg: dict[str, Any], accepted_count: int) -> bool:
    every = int(cfg.get("loop", {}).get("best_reevaluate_every_accepts", 2))
    return every > 0 and accepted_count > 0 and accepted_count % every == 0


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())

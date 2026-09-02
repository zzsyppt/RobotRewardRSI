"""Phase 8E-6-2 linked orchestration from sealed checkpoint to decision.

This module is intentionally a thin trusted-side wiring layer.  It does not
launch LLM generation or long-horizon training.  It consumes runner-owned
training artifacts, proves the checkpoint path/safety contract, asks the
trusted eval probe to produce ``public_result``, then delegates accept/reject
and ledger handling to :func:`run_phase8e_round`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.immutable_runner.budget import build_budget_accounting
from humanoid_agent.immutable_runner.safe_checkpoint import checkpoint_safety_report
from humanoid_agent.immutable_runner.train_start_paths import resolve_train_start_checkpoint_path

from .phase8e import Phase8EError, run_phase8e_round


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRUSTED_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "phase8e_trusted_eval_probe.py"
LINKED_SUMMARY_SCHEMA = "humanoid_agent.phase8e.linked_round.summary.v1"


class Phase8ELinkedError(RuntimeError):
    """Raised when a linked Phase 8E round cannot continue safely."""


def resolve_train_start_checkpoint_for_eval(
    training_summary: Mapping[str, Any],
    *,
    runner_artifacts_root: Path,
    validate_safe_load: bool = True,
) -> dict[str, Any]:
    """Resolve and optionally safe-load-check the checkpoint used by eval.

    ``training_summary`` is untrusted data from a child process.  The only path
    accepted here is the isolated sealed-checkpoint path defined by
    ``train_start_paths``.  The returned host path is derived from the
    runner-owned ``runner_artifacts_root`` supplied by the orchestrator, never
    from candidate output.
    """

    child_summary = _child_summary(training_summary)
    latest_checkpoint = _latest_train_start_checkpoint(child_summary)
    if not latest_checkpoint:
        raise Phase8ELinkedError("training summary does not contain probes.train_start.latest_checkpoint")

    host_checkpoint = resolve_train_start_checkpoint_path(latest_checkpoint, runner_artifacts_root)
    if host_checkpoint is None:
        raise Phase8ELinkedError(f"train-start checkpoint path is outside the runner sealed-checkpoint contract: {latest_checkpoint}")

    host_checkpoint = host_checkpoint.resolve(strict=False)
    if not host_checkpoint.is_file():
        raise Phase8ELinkedError(f"resolved train-start checkpoint is missing: {host_checkpoint}")

    if validate_safe_load:
        safety = checkpoint_safety_report(host_checkpoint)
        if not safety.get("safe_load"):
            raise Phase8ELinkedError(f"resolved train-start checkpoint failed safe-load gate: {safety.get('error')}")
    else:
        safety = {
            "path": str(host_checkpoint),
            "safe_load": None,
            "sha256": None,
            "error": None,
            "skipped": True,
            "reason": "validate_safe_load=False",
        }

    return {
        "isolated_path": latest_checkpoint,
        "host_path": str(host_checkpoint),
        "runner_artifacts_root": str(runner_artifacts_root.resolve(strict=False)),
        "checkpoint_safety": safety,
    }


def run_phase8e_linked_round(
    *,
    config_path: Path,
    run_dir: Path,
    candidate_dir: Path,
    train_summary_path: Path,
    runner_artifacts_root: Path,
    public_dev_summary_paths: Sequence[Path],
    public_accept_summary_paths: Sequence[Path] | None = None,
    curriculum_lineage_path: Path | None = None,
    append_ledger: bool = False,
    inherit_if_accepted: bool = False,
    require_server_enforcement: bool = False,
    lab_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the Phase 8E linked eval/decision path for an existing training artifact."""

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_summary_path = train_summary_path.resolve()
    train_summary = load_mapping(train_summary_path)
    checkpoint_for_eval = resolve_train_start_checkpoint_for_eval(
        train_summary,
        runner_artifacts_root=runner_artifacts_root.resolve(),
        validate_safe_load=True,
    )

    trusted_eval_dir = run_dir / "trusted_eval"
    trusted_eval_summary_path = trusted_eval_dir / "trusted_eval_probe_summary.json"
    public_result_path = trusted_eval_dir / "public_result.json"
    runner_budget_summary_path = trusted_eval_dir / "runner_budget_accounting.json"
    runner_budget_summary = build_runner_owned_budget_summary(config_path=config_path, training_summary=train_summary)
    _write_json(runner_budget_summary_path, runner_budget_summary)
    trusted_eval = _run_trusted_eval_probe(
        checkpoint_path=Path(checkpoint_for_eval["host_path"]),
        summary_file=trusted_eval_summary_path,
        public_result_file=public_result_path,
        public_dev_summary_paths=[Path(path).resolve() for path in public_dev_summary_paths],
        public_accept_summary_paths=[Path(path).resolve() for path in public_accept_summary_paths or []],
        budget_summary_path=runner_budget_summary_path,
    )
    if trusted_eval.get("status") != "pass":
        linked = _linked_summary(
            status="reject",
            config_path=config_path,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            train_summary_path=train_summary_path,
            budget_summary_path=runner_budget_summary_path,
            checkpoint_for_eval=checkpoint_for_eval,
            trusted_eval=trusted_eval,
            public_result_path=public_result_path,
            round_summary=None,
            phase8e_exit_code=1,
        )
        _write_json(run_dir / "phase8e_linked_round_summary.json", linked)
        raise Phase8ELinkedError(f"trusted eval probe failed: {trusted_eval.get('reason', trusted_eval.get('status'))}")

    exit_code, round_summary = run_phase8e_round(
        config_path=config_path.resolve(),
        run_dir=run_dir / "phase8e_round",
        candidate_dir=candidate_dir.resolve(),
        public_result_path=public_result_path,
        curriculum_lineage_path=curriculum_lineage_path.resolve() if curriculum_lineage_path is not None else None,
        append_ledger=append_ledger,
        inherit_if_accepted=inherit_if_accepted,
        require_server_enforcement=require_server_enforcement,
        lab_dir=lab_dir.resolve() if lab_dir is not None else None,
    )
    linked = _linked_summary(
        status=str(round_summary.get("status", "unknown")),
        config_path=config_path,
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        train_summary_path=train_summary_path,
        budget_summary_path=runner_budget_summary_path,
        checkpoint_for_eval=checkpoint_for_eval,
        trusted_eval=trusted_eval,
        public_result_path=public_result_path,
        round_summary=round_summary,
        phase8e_exit_code=exit_code,
    )
    _write_json(run_dir / "phase8e_linked_round_summary.json", linked)
    return exit_code, linked


def build_runner_owned_budget_summary(*, config_path: Path, training_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Build a trusted budget gate input from immutable runner config plus child evidence.

    Raw train summaries are not trusted budget gates.  They may contain useful
    child-reported counters, but the accepted budget values must come from the
    runner-owned Phase 8B config.  This function turns the child summary into
    evidence, cross-checks it against that config, and returns the only budget
    summary that linked trusted eval should consume.
    """

    config = load_mapping(config_path.resolve())
    budget_config_path = _resolve_project_path(config["runner"]["budget_config_path"])
    budget_config = load_mapping(budget_config_path)
    runner_config = dict(budget_config["runner"])
    budget_contract = dict(budget_config["budget_contract"])
    num_envs = int(runner_config["num_envs"])
    max_iterations = int(runner_config["max_iterations"])
    steps_per_iteration = int(budget_contract["steps_per_iteration"])
    config_derived_total = num_envs * max_iterations * steps_per_iteration
    evidence = _budget_evidence_from_training_summary(training_summary)
    # Config-derived budget values are runner-owned: the runner sets num_envs and
    # max_iterations on the training CLI and steps_per_iteration via the env, so it
    # derives these from its own immutable Phase 8B config rather than requiring the
    # candidate-executing child to echo them.  The child evidence is used only to
    # cross-check the values it DOES report, so a forged child is still rejected.
    runner_accounting = {
        "config_derived_total_timesteps": config_derived_total,
        "num_steps_per_env": steps_per_iteration,
        "policy_update_iterations": max_iterations,
    }
    accounting = build_budget_accounting(
        runner_config=runner_config,
        budget_contract=budget_contract,
        train_summary={"training_accounting": runner_accounting},
        elapsed_seconds=_optional_float(evidence.get("elapsed_seconds")),
        latest_checkpoint=_latest_train_start_checkpoint(_child_summary(training_summary)),
    )
    accounting["num_steps_per_env"] = accounting["steps_per_iteration"]
    accounting["budget_config_path"] = str(budget_config_path)
    accounting["budget_source"] = "runner_owned_phase8b_budget_config"
    accounting["child_evidence"] = evidence
    accounting["violations"].extend(
        _reported_budget_field_violations(
            evidence,
            expected={
                "num_envs": num_envs,
                "max_iterations": max_iterations,
                "num_steps_per_env": steps_per_iteration,
                "config_derived_total_timesteps": config_derived_total,
            },
        )
    )
    accounting["status"] = "pass" if not accounting["violations"] else "fail"
    return {
        "schema_version": "humanoid_agent.phase8e.linked_runner_budget_summary.v1",
        "status": accounting["status"],
        "budget_config_path": str(budget_config_path),
        "budget_accounting": accounting,
    }


def _run_trusted_eval_probe(
    *,
    checkpoint_path: Path,
    summary_file: Path,
    public_result_file: Path,
    public_dev_summary_paths: Sequence[Path],
    public_accept_summary_paths: Sequence[Path],
    budget_summary_path: Path,
) -> dict[str, Any]:
    if not public_dev_summary_paths:
        raise Phase8ELinkedError("at least one public_dev rollout summary is required for linked eval")
    command = [
        sys.executable,
        str(TRUSTED_EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint_path),
        "--summary-file",
        str(summary_file),
        "--public-result-file",
        str(public_result_file),
        "--budget-summary",
        str(budget_summary_path),
    ]
    for path in public_dev_summary_paths:
        command.extend(["--public-dev-summary", str(path)])
    for path in public_accept_summary_paths:
        command.extend(["--public-accept-summary", str(path)])

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") if not existing_pythonpath else str(PROJECT_ROOT / "src") + os.pathsep + existing_pythonpath
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)
    summary = load_mapping(summary_file) if summary_file.is_file() else {}
    summary["summary_file"] = str(summary_file.resolve())
    summary["public_result_file"] = str(public_result_file.resolve())
    summary["command_result"] = {
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0 and summary.get("status") == "pass":
        summary["status"] = "reject"
        summary["reason"] = "trusted_eval_process_failed_after_pass_summary"
    if result.returncode != 0 and "reason" not in summary:
        summary["reason"] = "trusted_eval_process_failed"
    return summary


def _child_summary(training_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    child = training_summary.get("child_summary")
    return child if isinstance(child, Mapping) else training_summary


def _latest_train_start_checkpoint(child_summary: Mapping[str, Any]) -> str | None:
    probes = child_summary.get("probes")
    if not isinstance(probes, Mapping):
        return None
    train_start = probes.get("train_start")
    if not isinstance(train_start, Mapping):
        return None
    latest = train_start.get("latest_checkpoint")
    return latest if isinstance(latest, str) and latest else None


def _budget_evidence_from_training_summary(training_summary: Mapping[str, Any]) -> dict[str, Any]:
    child_summary = _child_summary(training_summary)
    probes = child_summary.get("probes") if isinstance(child_summary.get("probes"), Mapping) else {}
    train_start = probes.get("train_start") if isinstance(probes.get("train_start"), Mapping) else {}
    sources = _budget_sources(training_summary, train_start)
    return {
        "config_derived_total_timesteps": _first_int(sources, "config_derived_total_timesteps"),
        "num_envs": _first_int(sources, "num_envs"),
        "max_iterations": _first_int(sources, "max_iterations"),
        "num_steps_per_env": _first_int(sources, "num_steps_per_env", "steps_per_iteration"),
        "policy_update_iterations": _first_int(sources, "policy_update_iterations"),
        "estimated_optimizer_steps": _first_int(sources, "estimated_optimizer_steps"),
        "estimated_gradient_steps": _first_int(sources, "estimated_gradient_steps"),
        "estimated_forward_backward_calls": _first_int(sources, "estimated_forward_backward_calls"),
        "elapsed_seconds": _first_float(sources, "elapsed_seconds"),
        "reported_values": _reported_budget_values(sources),
        "source": "runner_child_summary_cross_checked_against_phase8b_budget_config",
    }


def _reported_budget_field_violations(evidence: Mapping[str, Any], *, expected: Mapping[str, int]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    reported_values = evidence.get("reported_values") if isinstance(evidence.get("reported_values"), Mapping) else {}
    for key, expected_value in expected.items():
        reports = reported_values.get(key) if isinstance(reported_values.get(key), list) else []
        # A child that omits a runner-owned config-derived value is fine (the
        # runner fills it from its immutable config); only a child that REPORTS a
        # value conflicting with the runner config is a forgery violation.
        for report in reports:
            if not isinstance(report, Mapping):
                continue
            actual = report.get("value")
            if int(actual) != int(expected_value):
                violations.append(
                    {
                        "key": key,
                        "source": report.get("source"),
                        "actual": int(actual),
                        "expected": int(expected_value),
                    }
                )
    return violations


def _budget_sources(training_summary: Mapping[str, Any], train_start: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    top_training_accounting = training_summary.get("training_accounting") if isinstance(training_summary.get("training_accounting"), Mapping) else {}
    top_budget = training_summary.get("budget_accounting") if isinstance(training_summary.get("budget_accounting"), Mapping) else {}
    return [
        ("training_accounting", top_training_accounting),
        ("child_summary.probes.train_start", train_start),
        ("budget_accounting", top_budget),
    ]


def _reported_budget_values(sources: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    aliases = {
        "config_derived_total_timesteps": ("config_derived_total_timesteps",),
        "num_envs": ("num_envs",),
        "max_iterations": ("max_iterations",),
        "num_steps_per_env": ("num_steps_per_env", "steps_per_iteration"),
    }
    for canonical, keys in aliases.items():
        for source_name, source in sources:
            value = _first_int([(source_name, source)], *keys)
            if value is not None:
                out.setdefault(canonical, []).append({"source": source_name, "value": value})
    return out


def _first_int(sources: Sequence[tuple[str, Mapping[str, Any]]], *keys: str) -> int | None:
    for _source_name, source in sources:
        for key in keys:
            if key in source:
                try:
                    return int(source[key])
                except (TypeError, ValueError):
                    return None
    return None


def _first_float(sources: Sequence[tuple[str, Mapping[str, Any]]], key: str) -> float | None:
    for _source_name, source in sources:
        if key in source:
            return _optional_float(source[key])
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _linked_summary(
    *,
    status: str,
    config_path: Path,
    run_dir: Path,
    candidate_dir: Path,
    train_summary_path: Path,
    budget_summary_path: Path,
    checkpoint_for_eval: Mapping[str, Any],
    trusted_eval: Mapping[str, Any],
    public_result_path: Path,
    round_summary: Mapping[str, Any] | None,
    phase8e_exit_code: int,
) -> dict[str, Any]:
    return {
        "schema_version": LINKED_SUMMARY_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "formal_training_executed": False,
        "linked_training_artifact_consumed": True,
        "linked_eval_executed": trusted_eval.get("status") == "pass",
        "train_summary_path": str(train_summary_path),
        "budget_summary_path": str(budget_summary_path.resolve()),
        "checkpoint_for_eval": dict(checkpoint_for_eval),
        "trusted_eval": {
            "status": trusted_eval.get("status"),
            "summary_file": trusted_eval.get("command_result", {}).get("summary_file") or trusted_eval.get("summary_file"),
            "candidate_code_loaded": trusted_eval.get("candidate_code_loaded"),
            "checkpoint_safe_load": (trusted_eval.get("checkpoint_safety") or {}).get("safe_load")
            if isinstance(trusted_eval.get("checkpoint_safety"), Mapping)
            else None,
            "public_eval": trusted_eval.get("public_eval"),
            "command_result": trusted_eval.get("command_result"),
        },
        "public_result_path": str(public_result_path.resolve()),
        "phase8e_exit_code": phase8e_exit_code,
        "round_summary_file": str((run_dir / "phase8e_round" / "phase8e_round_summary.json").resolve()) if round_summary is not None else None,
        "round_summary": dict(round_summary) if round_summary is not None else None,
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

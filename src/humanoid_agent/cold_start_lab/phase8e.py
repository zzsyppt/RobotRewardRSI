"""Phase 8E single-agent code-level cold-start loop orchestration.

Prompt construction, prompt-facing projections, and source-pack assembly live in
dedicated sibling modules.  This file keeps the historical public API while
owning the one-round orchestration, decision, ledger, and inheritance flow.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.code_sandbox.verifier import verify_candidate_package, write_summary
from humanoid_agent.cold_start_lab.curriculum_compiler import project_curriculum_lineage_bundle
from humanoid_agent.cold_start_lab.phase8e_common import (
    PROJECT_ROOT,
    SOURCE_PACK_FORBIDDEN_TOKENS,
    SOURCE_PACK_MAX_TEXT_BYTES,
    Phase8EError,
    assert_prompt_facing_payload,
    assert_public_only_payload,
)
from humanoid_agent.cold_start_lab.phase8e_projection import (
    _prompt_history_public_result,
    latest_accepted_record,
    project_candidate_package,
    project_ledger_record,
    project_public_result,
    project_verifier_summary,
)
from humanoid_agent.cold_start_lab.phase8e_prompt import (
    NAMED_COMPONENT_COLLECTION_FIELDS,
    PROMPT_FACING_COMPONENT_CLASSES_TO_IMPORT,
    PROMPT_FACING_NAMED_COMPONENT_COLLECTION_FIELDS,
    PROMPT_FACING_TRAINING_COMPONENT_FIELDS,
    TRAINING_COMPONENT_FIELDS,
    build_candidate_self_audit_checklist,
    build_phase8e_prompt_package,
    messages_to_text,
    prompt_facing_candidate_allowed_controls,
    prompt_facing_control_effectiveness_contract,
    prompt_facing_reward_inputs_contract,
    prompt_facing_reward_inputs_field_expansion_report,
)
from humanoid_agent.cold_start_lab.phase8e_source_pack import (
    ALLOWED_TOP_LEVEL_CANDIDATE_FILES,
    KNOWN_GOOD_REWARD_TERMS_SOURCE,
    build_minimal_source_pack,
    build_source_access_policy,
    known_good_reward_examples_metadata,
)


def run_phase8e_round(
    *,
    config_path: Path,
    run_dir: Path,
    candidate_dir: Path | None = None,
    current_recipe_dir: Path | None = None,
    public_result_path: Path | None = None,
    curriculum_lineage_path: Path | None = None,
    append_ledger: bool = False,
    inherit_if_accepted: bool = False,
    require_server_enforcement: bool = False,
    lab_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Prepare one Phase 8E round and optionally verify one candidate.

    No formal training is launched here.  If ``public_result_path`` is omitted,
    a verifier-passing candidate is marked ``pending_training`` rather than
    accepted.

    ``lab_dir`` is the stable experiment-level home for the ledger and immutable
    accepted-recipe snapshots; it must be the same across rounds for inheritance
    to work.  ``run_dir`` only holds per-round artifacts.
    """

    config = load_mapping(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_lab_dir = lab_dir.resolve() if lab_dir is not None else _lab_dir(config, run_dir)
    effective_lab_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = _ledger_path(config, effective_lab_dir)
    ledger = load_ledger(ledger_path)
    current_recipe = resolve_current_recipe(config, ledger, current_recipe_dir=current_recipe_dir)
    prompt_package = build_phase8e_prompt_package(config=config, ledger=ledger, current_recipe_dir=current_recipe)
    _write_prompt_files(config, run_dir, prompt_package)

    verifier_summary: dict[str, Any] | None = None
    public_result: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    inherited_recipe_dir: Path | None = None
    candidate_projection: dict[str, Any] | None = None
    curriculum_lineage: dict[str, Any] | None = _load_curriculum_lineage(curriculum_lineage_path, config=config)
    curriculum_lineage_public = project_curriculum_lineage_bundle(curriculum_lineage)

    if candidate_dir is not None:
        candidate_dir = candidate_dir.resolve()
        verifier_run_dir = run_dir / "phase8d_verifier"
        verifier_summary = verify_candidate_package(
            candidate_dir=candidate_dir,
            config_path=_resolve_project_path(config["verifier"]["config_path"]),
            run_dir=verifier_run_dir,
            require_server_enforcement=require_server_enforcement,
        )
        write_summary(verifier_run_dir / "phase8d_verifier_summary.json", verifier_summary)
        if verifier_summary.get("status") == "pass":
            candidate_projection = project_candidate_package(candidate_dir, include_public_artifacts=True)
        else:
            candidate_projection = {
                "candidate_dir": str(candidate_dir),
                "manifest": None,
                "fingerprint": verifier_summary.get("fingerprint"),
                "operation_diff": None,
                "diagnostics": None,
            }

        if public_result_path is not None:
            public_result = load_mapping(public_result_path.resolve())
            assert_public_only_payload(public_result, config=config, context="public_result")

        current_best_record = latest_accepted_record(ledger)
        decision = build_phase8e_decision(
            config=config,
            verifier_summary=verifier_summary,
            public_result=public_result,
            current_best_record=current_best_record,
            candidate_dir=candidate_dir,
        )
        if curriculum_lineage_public is not None:
            detail = dict(decision.get("detail", {})) if isinstance(decision.get("detail"), dict) else {}
            detail["curriculum_lineage"] = curriculum_lineage_public
            decision["detail"] = detail

        if append_ledger:
            if decision.get("accepted") is True:
                if not inherit_if_accepted:
                    raise Phase8EError("accepted candidate ledger append requires inherit_if_accepted for immutable recipe snapshot")
                if curriculum_lineage is None:
                    raise Phase8EError("accepted candidate ledger append requires runner curriculum_lineage_path")
                candidate_id = (candidate_projection or {}).get("manifest", {}).get("candidate_id")
                inherited_recipe_dir = _snapshot_accepted_recipe(
                    config=config,
                    lab_dir=effective_lab_dir,
                    ledger=ledger,
                    candidate_dir=candidate_dir,
                    candidate_id=candidate_id,
                    curriculum_lineage=curriculum_lineage,
                )
            append_ledger_record(
                ledger_path,
                build_ledger_record(
                    config,
                    candidate_projection,
                    verifier_summary,
                    decision,
                    inherited_recipe_dir=inherited_recipe_dir,
                    curriculum_lineage_public=curriculum_lineage_public,
                ),
            )
            ledger = load_ledger(ledger_path)

    status = "prepared"
    if decision is not None:
        status = str(decision["decision"])
    elif verifier_summary is not None:
        status = "verified" if verifier_summary.get("status") == "pass" else "reject"

    summary = {
        "schema_version": str(config.get("summary_schema_version", "humanoid_agent.phase8e.loop.summary.v1")),
        "phase": str(config.get("phase", "8E-single-agent-code-level-cold-start-loop")),
        "status": status,
        "created_at_utc": _now(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir),
        "formal_training_executed": False,
        "prompt": {
            "messages_file": str(run_dir / str(config["paths"].get("prompt_messages_filename", "phase8e_prompt_messages.json"))),
            "text_file": str(run_dir / str(config["paths"].get("prompt_text_filename", "phase8e_prompt.txt"))),
            "public_only": True,
        },
        "current_recipe": project_candidate_package(current_recipe) if current_recipe is not None else None,
        "candidate": candidate_projection,
        "verifier": project_verifier_summary(verifier_summary) if verifier_summary is not None else None,
        "public_result": project_public_result(public_result) if public_result is not None else None,
        "decision": decision,
        "curriculum_lineage": curriculum_lineage_public,
        "ledger": {
            "path": str(ledger_path),
            "records": len(ledger),
            "appended": bool(append_ledger and candidate_projection is not None),
        },
        "inheritance": {
            "requested": inherit_if_accepted,
            "snapshot_created": inherited_recipe_dir is not None,
            "accepted_recipe_snapshot_dir": str(inherited_recipe_dir) if inherited_recipe_dir is not None else None,
            "lab_dir": str(effective_lab_dir),
        },
    }
    assert_public_only_payload(summary, config=config, context="phase8e_round_summary")
    summary_file = run_dir / str(config["paths"].get("round_summary_filename", "phase8e_round_summary.json"))
    _write_json(summary_file, summary)
    print(json.dumps({"summary_file": str(summary_file), "status": status}, ensure_ascii=False))
    return (1 if status == "reject" else 0), summary


def resolve_current_recipe(config: dict[str, Any], ledger: list[dict[str, Any]], *, current_recipe_dir: Path | None = None) -> Path | None:
    if current_recipe_dir is not None:
        return current_recipe_dir.resolve()
    record = latest_accepted_record(ledger)
    if record is not None:
        # Prefer the immutable accepted-recipe snapshot so the inheritance chain
        # cannot be mutated by editing the original candidate directory.
        inherited = record.get("inherited_recipe")
        if isinstance(inherited, dict):
            snapshot_dir = inherited.get("dir")
            if isinstance(snapshot_dir, str) and snapshot_dir and Path(snapshot_dir).is_dir():
                return Path(snapshot_dir).resolve()
        candidate_dir = record.get("candidate", {}).get("candidate_dir")
        if isinstance(candidate_dir, str) and candidate_dir and Path(candidate_dir).is_dir():
            return Path(candidate_dir).resolve()
    initial = config.get("initial_recipe", {})
    path = initial.get("path") if isinstance(initial, dict) else None
    if isinstance(path, str) and path:
        return _resolve_project_path(path)
    return None


def build_phase8e_decision(
    *,
    config: dict[str, Any],
    verifier_summary: dict[str, Any],
    public_result: dict[str, Any] | None,
    current_best_record: dict[str, Any] | None,
    candidate_dir: Path,
) -> dict[str, Any]:
    acceptance = config.get("acceptance", {})
    if verifier_summary.get("status") != "pass":
        return _decision(
            "reject",
            accepted=False,
            reason="phase8d_verifier_failed",
            candidate_dir=candidate_dir,
            verifier_summary=verifier_summary,
        )
    if public_result is None:
        return _decision(
            "pending_training",
            accepted=False,
            reason="no_public_result_supplied",
            candidate_dir=candidate_dir,
            verifier_summary=verifier_summary,
        )

    projected = project_public_result(public_result)
    if projected.get("status") != "pass":
        return _decision("reject", accepted=False, reason="public_result_status_not_pass", candidate_dir=candidate_dir, verifier_summary=verifier_summary, public_result=projected)
    failed_gates = _failed_required_gates(projected, acceptance)
    if failed_gates:
        return _decision("reject", accepted=False, reason="required_gate_failed", candidate_dir=candidate_dir, verifier_summary=verifier_summary, public_result=projected, detail={"failed_gates": failed_gates})

    accept_stats = _stats(projected.get("public_accept"))
    dev_stats = _stats(projected.get("public_dev"))
    if accept_stats is None:
        if _absolute_floor_met(dev_stats, acceptance):
            return _decision("promising", accepted=False, reason="dev_absolute_floor_met_but_public_accept_missing", candidate_dir=candidate_dir, verifier_summary=verifier_summary, public_result=projected)
        return _decision("reject", accepted=False, reason="public_accept_missing_and_dev_floor_not_met", candidate_dir=candidate_dir, verifier_summary=verifier_summary, public_result=projected)

    best_stats = None
    if current_best_record is not None:
        best_stats = _stats(current_best_record.get("public_result", {}).get("public_accept"))
    near_zero = float(acceptance.get("near_zero_success_threshold", 0.02))
    absolute_accept_floor = float(acceptance.get("absolute_accept_success_floor", 0.05))
    candidate_mean = _mean(accept_stats)
    best_mean = _mean(best_stats)
    if best_mean is None or best_mean <= near_zero:
        accepted = candidate_mean is not None and candidate_mean >= absolute_accept_floor
        return _decision(
            "accept" if accepted else "reject",
            accepted=accepted,
            reason="absolute_accept_floor" if accepted else "absolute_accept_floor_not_met",
            candidate_dir=candidate_dir,
            verifier_summary=verifier_summary,
            public_result=projected,
            detail={"candidate_public_accept_mean": candidate_mean, "absolute_accept_success_floor": absolute_accept_floor},
        )

    delta = None if candidate_mean is None else candidate_mean - best_mean
    pooled_se = _pooled_standard_error(accept_stats, best_stats)
    min_delta = float(acceptance.get("public_min_delta", 0.05))
    threshold = max(min_delta, pooled_se if pooled_se is not None else min_delta)
    accepted = delta is not None and delta > threshold
    return _decision(
        "accept" if accepted else "reject",
        accepted=accepted,
        reason="noise_aware_delta" if accepted else "noise_aware_delta_not_met",
        candidate_dir=candidate_dir,
        verifier_summary=verifier_summary,
        public_result=projected,
        detail={
            "candidate_public_accept_mean": candidate_mean,
            "best_public_accept_mean": best_mean,
            "delta": delta,
            "pooled_standard_error": pooled_se,
            "acceptance_threshold": threshold,
        },
    )


def build_ledger_record(
    config: dict[str, Any],
    candidate: dict[str, Any],
    verifier_summary: dict[str, Any],
    decision: dict[str, Any],
    *,
    inherited_recipe_dir: Path | None = None,
    curriculum_lineage_public: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": "humanoid_agent.phase8e.ledger_record.v1",
        "created_at_utc": _now(),
        "record_type": "phase8e_candidate",
        "task_id": config.get("task", {}).get("task_id"),
        "candidate": candidate,
        "verifier": project_verifier_summary(verifier_summary),
        "decision": decision,
        "public_result": decision.get("public_result"),
        "curriculum_lineage": curriculum_lineage_public,
        "inherited_recipe": {"dir": str(inherited_recipe_dir), "immutable_snapshot": True} if inherited_recipe_dir is not None else None,
        "feedback_safe": {
            "public_only": True,
            "private_eval_used": False,
            "eval_seed_config_included": False,
        },
    }
    assert_public_only_payload(record, config=config, context="phase8e_ledger_record")
    return record


def append_ledger_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase8EError(f"ledger line {line_number} is not valid JSON: {path}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def copy_accepted_recipe(candidate_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        raise Phase8EError(f"refusing to overwrite inherited recipe directory: {target_dir}")
    shutil.copytree(candidate_dir, target_dir, symlinks=False)


def _decision(
    decision: str,
    *,
    accepted: bool,
    reason: str,
    candidate_dir: Path,
    verifier_summary: dict[str, Any],
    public_result: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.public_decision.v1",
        "created_at_utc": _now(),
        "decision": decision,
        "accepted": accepted,
        "reason": reason,
        "candidate_dir": str(candidate_dir.resolve()),
        "verifier": project_verifier_summary(verifier_summary),
        "public_result": public_result,
        "detail": detail or {},
        "feedback_safe": {
            "public_only": True,
            "private_eval_used": False,
            "eval_seed_config_included": False,
        },
    }


def _absolute_floor_met(stats: dict[str, Any] | None, acceptance: dict[str, Any]) -> bool:
    if stats is None:
        return False
    seed_results = stats.get("seed_results", [])
    floor = float(acceptance.get("absolute_dev_seed_floor", 0.03))
    min_count = int(acceptance.get("absolute_dev_min_passing_runs", 2))
    if isinstance(seed_results, list) and seed_results:
        count = sum(1 for item in seed_results if isinstance(item, dict) and float(item.get("success_rate", 0.0)) >= floor)
        return count >= min_count
    mean = _mean(stats)
    return mean is not None and mean >= floor


def _failed_required_gates(projected: dict[str, Any], acceptance: dict[str, Any]) -> list[str]:
    failed = []
    for name in acceptance.get("required_pass_gates", []):
        gate = projected.get(name)
        if not isinstance(gate, dict) or gate.get("status") != "pass":
            failed.append(str(name))
    return failed


def _stats(split: Any) -> dict[str, Any] | None:
    if not isinstance(split, dict):
        return None
    stats = split.get("success_rate_stats")
    if not isinstance(stats, dict):
        return None
    out = dict(stats)
    out["seed_results"] = split.get("seed_results", [])
    return out


def _mean(stats: dict[str, Any] | None) -> float | None:
    if not stats:
        return None
    value = stats.get("mean")
    return None if value is None else float(value)


def _pooled_standard_error(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    cand = candidate.get("standard_error")
    base = baseline.get("standard_error")
    if cand is None or base is None:
        return None
    return math.sqrt(float(cand) ** 2 + float(base) ** 2)


def _write_prompt_files(config: dict[str, Any], run_dir: Path, prompt_package: dict[str, Any]) -> None:
    paths = config["paths"]
    _write_json(run_dir / str(paths.get("prompt_messages_filename", "phase8e_prompt_messages.json")), prompt_package["messages"])
    (run_dir / str(paths.get("prompt_text_filename", "phase8e_prompt.txt"))).write_text(prompt_package["prompt_text"], encoding="utf-8")
    _write_prompt_payload_json(
        run_dir / str(paths.get("prompt_payload_filename", "phase8e_prompt_payload.json")),
        prompt_package["payload"],
    )


def _lab_dir(config: dict[str, Any], run_dir: Path) -> Path:
    """Stable experiment-level home for the ledger and accepted-recipe snapshots.

    Multi-round inheritance needs the ledger and immutable snapshots to persist
    across rounds, so they live in ``paths.lab_dir`` when set.  Falling back to
    ``run_dir`` keeps single-directory usage working but cannot support
    multi-round inheritance across separate per-round run directories.
    """

    lab = config.get("paths", {}).get("lab_dir")
    if isinstance(lab, str) and lab:
        return _resolve_project_path(lab)
    return run_dir


def _ledger_path(config: dict[str, Any], lab_dir: Path) -> Path:
    ledger = str(config.get("paths", {}).get("ledger_filename", "phase8e_ledger.jsonl"))
    path = Path(ledger)
    return path if path.is_absolute() else lab_dir / path


def _snapshot_accepted_recipe(
    *,
    config: dict[str, Any],
    lab_dir: Path,
    ledger: list[dict[str, Any]],
    candidate_dir: Path,
    candidate_id: str | None,
    curriculum_lineage: dict[str, Any] | None = None,
) -> Path:
    """Copy an accepted candidate to an immutable, ordered snapshot directory."""

    accepted_dirname = str(config.get("paths", {}).get("accepted_recipes_dirname", "accepted_recipes"))
    index = sum(
        1
        for record in ledger
        if isinstance(record.get("decision"), dict) and record["decision"].get("accepted") is True
    )
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(candidate_id or "candidate"))
    snapshot_dir = lab_dir / accepted_dirname / f"{index:04d}_{safe_id}"
    public_summary = None
    if isinstance(curriculum_lineage, dict):
        public_summary = project_curriculum_lineage_bundle(curriculum_lineage)
        if public_summary is None:
            raise Phase8EError("accepted curriculum lineage must include a projectable public_summary")
    copy_accepted_recipe(candidate_dir, snapshot_dir)
    if isinstance(curriculum_lineage, dict):
        curriculum_dir = snapshot_dir / "runner_curriculum_snapshot"
        _write_json(curriculum_dir / "curriculum_lineage.json", curriculum_lineage)
        _write_json(curriculum_dir / "public_summary.json", public_summary)
        raw = curriculum_lineage.get("raw_curriculum_design")
        if raw is not None:
            _write_json(curriculum_dir / "raw_curriculum_design.json", raw)
        effective = curriculum_lineage.get("effective_curriculum")
        if effective is not None:
            _write_json(curriculum_dir / "effective_curriculum.json", effective)
    return snapshot_dir


def _load_curriculum_lineage(path: Path | None, *, config: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    lineage = load_mapping(path.resolve())
    public_summary = project_curriculum_lineage_bundle(lineage)
    if public_summary is None:
        raise Phase8EError(f"curriculum_lineage_path is not projectable to public lineage: {path.resolve()}")
    assert_public_only_payload(public_summary, config=config, context="curriculum_lineage_public")
    return lineage


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_prompt_payload_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

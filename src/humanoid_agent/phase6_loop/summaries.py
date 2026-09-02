"""Build Phase 6 public-only summaries from training/eval artifacts."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.phase1_push_recovery.config import sha256_file
from humanoid_agent.phase2_baseline.overrides import apply_phase2_baseline_overrides  # noqa: F401 - import smoke coverage
from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - reuse YAML fallback.
from humanoid_agent.phase6_loop.stats import stats


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def subset_public_summary(summary: dict[str, Any], *, seeds: list[int], summary_path: Path | None = None) -> dict[str, Any]:
    """Return a summary containing only the requested training/eval seed rows.

    This is used to compare Phase 6 candidates against the same paired seed
    subset without regenerating baseline checkpoints.
    """

    wanted = set(int(seed) for seed in seeds)
    rows = [copy.deepcopy(row) for row in summary.get("seed_results", []) if int(row.get("seed", -1)) in wanted]
    if len(rows) != len(wanted):
        found = sorted(int(row.get("seed")) for row in rows)
        raise ValueError(f"Summary does not contain requested seeds. requested={seeds}, found={found}")
    public_values = [float(row["public_success_rate"]) for row in rows if row.get("public_success_rate") is not None]
    public_flags = [row.get("public_condition_flag_breakdown") for row in rows if isinstance(row.get("public_condition_flag_breakdown"), dict)]
    out = copy.deepcopy(summary)
    out["seed_results"] = rows
    out["public_eval"]["success_rate_stats"] = stats(public_values)
    out["public_eval"]["condition_flag_breakdown"] = _aggregate_condition_flags(public_flags)
    out["status"] = "pass" if all(row.get("train_status") == "pass" and row.get("public_eval_status") == "pass" for row in rows) else "fail"
    out["schema_version"] = "humanoid_agent.phase6.public_subset_summary.v1"
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_summary"] = str(summary_path) if summary_path else summary.get("run_dir")
    out.setdefault("config", {}).setdefault("training", {})["seeds"] = [int(seed) for seed in seeds]
    _project_public_config(out)
    out.setdefault("decision", {})["phase2_workflow_complete"] = out["status"] == "pass"
    out.setdefault("decision", {})["phase6_public_only_subset"] = True
    out.pop("iid_hidden_eval", None)
    for row in out["seed_results"]:
        for key in list(row):
            if key.startswith("iid_hidden"):
                row.pop(key, None)
            if key == "timing_seconds":
                row[key].pop("iid_hidden_eval", None)
    return out


def collect_public_candidate_summary(
    *,
    run_dir: Path,
    phase1_config: Path,
    phase2_config: Path,
    seeds: list[int],
) -> dict[str, Any]:
    phase2_cfg = _load_structured_file(phase2_config)
    training_cfg = copy.deepcopy(phase2_cfg["training"])
    training_cfg["seeds"] = [int(seed) for seed in seeds]
    public_eval_cfg = _public_eval_projection(phase2_cfg.get("evaluation", {}))
    seed_results: list[dict[str, Any]] = []
    public_values: list[float] = []
    public_flags: list[dict[str, Any]] = []
    total_gpu_hours = 0.0

    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        train_summary = _load_or_missing(seed_dir / "train_summary.json")
        public_summary = _load_or_missing(seed_dir / "public_eval_summary.json")
        train_seconds = _read_float(seed_dir / "train.seconds")
        public_seconds = _read_float(seed_dir / "public_eval.seconds")
        gpu_hours = (train_seconds + public_seconds) / 3600.0
        total_gpu_hours += gpu_hours
        public_rate = public_summary.get("aggregate", {}).get("success_rate")
        if public_rate is not None:
            public_values.append(float(public_rate))
        flags = public_summary.get("aggregate", {}).get("condition_flag_breakdown")
        if isinstance(flags, dict):
            public_flags.append(flags)
        seed_results.append(
            {
                "seed": int(seed),
                "train_status": train_summary.get("status"),
                "public_eval_status": public_summary.get("status"),
                "checkpoint": train_summary.get("training", {}).get("latest_checkpoint"),
                "timing_seconds": {"train": train_seconds, "public_eval": public_seconds},
                "gpu_hours_estimate": gpu_hours,
                "public_success_rate": public_rate,
                "public_failure_breakdown": public_summary.get("aggregate", {}).get("failure_breakdown"),
                "public_condition_flag_breakdown": flags,
            }
        )

    workflow_complete = all(row["train_status"] == "pass" and row["public_eval_status"] == "pass" for row in seed_results)
    return {
        "schema_version": "humanoid_agent.phase6.public_candidate_summary.v1",
        "status": "pass" if workflow_complete else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "config": {
            "phase1_config": str(phase1_config.resolve()),
            "phase1_config_sha256": sha256_file(phase1_config),
            "phase2_config": str(phase2_config.resolve()),
            "phase2_config_sha256": sha256_file(phase2_config),
            "baseline": phase2_cfg.get("baseline", {}),
            "training": training_cfg,
            "public_evaluation_contract": public_eval_cfg,
        },
        "seed_results": seed_results,
        "public_eval": {
            "success_rate_stats": stats(public_values),
            "target_success_rate_range": public_eval_cfg.get("target_success_rate_range"),
            "target_range_met": _target_range_met(public_values, public_eval_cfg.get("target_success_rate_range")),
            "condition_flag_breakdown": _aggregate_condition_flags(public_flags),
        },
        "compute": {
            "gpu_hours_estimate_total": total_gpu_hours,
            "gpu_hours_estimate_per_seed_mean": total_gpu_hours / len(seeds) if seeds else 0.0,
            "accounting_method": "public-only wall_clock_seconds_times_one_gpu_per_run",
        },
        "decision": {
            "phase2_workflow_complete": workflow_complete,
            "phase6_public_only_candidate": True,
            "notes": [
                "Phase 6 per-round summaries are public-only. Held-out diagnostics are not collected here.",
                "Use the same paired seed subset for candidate and comparator summaries.",
            ],
        },
    }


def _target_range_met(values: list[float], target_range: Any) -> bool | None:
    if not values or not isinstance(target_range, list) or len(target_range) != 2:
        return None
    mean_value = sum(values) / len(values)
    return float(target_range[0]) <= mean_value <= float(target_range[1])


def _project_public_config(summary: dict[str, Any]) -> None:
    config = summary.get("config")
    if not isinstance(config, dict):
        return
    evaluation = config.pop("evaluation", None)
    if isinstance(evaluation, dict):
        config["public_evaluation_contract"] = _public_eval_projection(evaluation)


def _public_eval_projection(evaluation: dict[str, Any]) -> dict[str, Any]:
    public = evaluation.get("public", {}) if isinstance(evaluation, dict) else {}
    return {
        "profile": public.get("profile") if isinstance(public, dict) else None,
        "num_envs": public.get("num_envs") if isinstance(public, dict) else None,
        "max_steps": public.get("max_steps") if isinstance(public, dict) else None,
        "target_success_rate_range": evaluation.get("target_success_rate_range") if isinstance(evaluation, dict) else None,
        "eval_seed_removed": True,
        "heldout_config_removed": True,
    }


def _load_or_missing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "errors": [{"type": "MissingFile", "message": str(path)}]}
    return load_json(path)


def _read_float(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0.0


def _aggregate_condition_flags(breakdowns: list[dict[str, Any]]) -> dict[str, Any]:
    if not breakdowns:
        return {"total": 0, "counts": {}, "rates": {}, "rate_stats_by_flag": {}, "diagnostic_only": True}
    flag_names = sorted({name for item in breakdowns for name in item.get("counts", {})})
    counts = {name: 0 for name in flag_names}
    total = 0
    rates_by_flag: dict[str, list[float]] = {name: [] for name in flag_names}
    for item in breakdowns:
        item_total = int(item.get("total", 0) or 0)
        total += item_total
        for name in flag_names:
            counts[name] += int(item.get("counts", {}).get(name, 0) or 0)
            if name in item.get("rates", {}):
                rates_by_flag[name].append(float(item["rates"][name]))
    return {
        "total": total,
        "counts": counts,
        "rates": {name: (counts[name] / total if total else 0.0) for name in flag_names},
        "rate_stats_by_flag": {name: stats(values) for name, values in rates_by_flag.items()},
        "nonexclusive": True,
        "diagnostic_only": True,
    }

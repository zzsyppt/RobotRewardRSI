#!/usr/bin/env python3
"""Collect Phase 2 baseline variance training and eval outputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from humanoid_agent.phase1_push_recovery.config import load_config, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "errors": [{"type": "MissingFile", "message": str(path)}]}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_float(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0.0


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "standard_error": None, "values": []}
    if len(values) == 1:
        return {"count": 1, "mean": values[0], "std": 0.0, "standard_error": 0.0, "values": values}
    sample_std = stdev(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "std": sample_std,
        "standard_error": sample_std / math.sqrt(len(values)),
        "values": values,
    }


def _success_rate(summary: dict[str, Any]) -> float | None:
    value = summary.get("aggregate", {}).get("success_rate")
    return None if value is None else float(value)


def _condition_flag_breakdown(summary: dict[str, Any]) -> dict[str, Any] | None:
    value = summary.get("aggregate", {}).get("condition_flag_breakdown")
    return value if isinstance(value, dict) else None


def _aggregate_condition_flags(breakdowns: list[dict[str, Any]]) -> dict[str, Any]:
    if not breakdowns:
        return {"total": 0, "counts": {}, "rates": {}, "rate_stats_by_flag": {}}

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
        "rate_stats_by_flag": {name: _stats(values) for name, values in rates_by_flag.items()},
        "nonexclusive": True,
        "diagnostic_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase1-config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--phase2-config", default="configs/phase2_baseline.yaml")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    phase2_cfg = load_config(args.phase2_config)
    training_cfg = phase2_cfg["training"]
    eval_cfg = phase2_cfg["evaluation"]
    seeds = [int(seed) for seed in training_cfg["seeds"]]

    seed_results: list[dict[str, Any]] = []
    public_success: list[float] = []
    iid_success: list[float] = []
    public_condition_flags: list[dict[str, Any]] = []
    iid_condition_flags: list[dict[str, Any]] = []
    total_gpu_hours = 0.0

    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        train_summary = _load_json(seed_dir / "train_summary.json")
        public_summary = _load_json(seed_dir / "public_eval_summary.json")
        iid_summary = _load_json(seed_dir / "iid_hidden_eval_summary.json")
        train_seconds = _read_float(seed_dir / "train.seconds")
        public_seconds = _read_float(seed_dir / "public_eval.seconds")
        iid_seconds = _read_float(seed_dir / "iid_hidden_eval.seconds")
        gpu_hours = (train_seconds + public_seconds + iid_seconds) / 3600.0
        total_gpu_hours += gpu_hours

        public_rate = _success_rate(public_summary)
        iid_rate = _success_rate(iid_summary)
        if public_rate is not None:
            public_success.append(public_rate)
        if iid_rate is not None:
            iid_success.append(iid_rate)
        public_flags = _condition_flag_breakdown(public_summary)
        iid_flags = _condition_flag_breakdown(iid_summary)
        if public_flags is not None:
            public_condition_flags.append(public_flags)
        if iid_flags is not None:
            iid_condition_flags.append(iid_flags)

        seed_results.append(
            {
                "seed": seed,
                "train_status": train_summary.get("status"),
                "public_eval_status": public_summary.get("status"),
                "iid_hidden_eval_status": iid_summary.get("status"),
                "checkpoint": train_summary.get("training", {}).get("latest_checkpoint"),
                "timing_seconds": {
                    "train": train_seconds,
                    "public_eval": public_seconds,
                    "iid_hidden_eval": iid_seconds,
                },
                "gpu_hours_estimate": gpu_hours,
                "public_success_rate": public_rate,
                "iid_hidden_success_rate": iid_rate,
                "public_failure_breakdown": public_summary.get("aggregate", {}).get("failure_breakdown"),
                "iid_hidden_failure_breakdown": iid_summary.get("aggregate", {}).get("failure_breakdown"),
                "public_condition_flag_breakdown": public_flags,
                "iid_hidden_condition_flag_breakdown": iid_flags,
            }
        )

    public_stats = _stats(public_success)
    iid_stats = _stats(iid_success)
    target_low, target_high = [float(x) for x in eval_cfg["target_success_rate_range"]]
    public_mean = public_stats["mean"]
    public_target_met = public_mean is not None and target_low <= float(public_mean) <= target_high
    workflow_complete = all(
        item["train_status"] == "pass"
        and item["public_eval_status"] == "pass"
        and item["iid_hidden_eval_status"] == "pass"
        for item in seed_results
    )

    output = {
        "schema_version": "humanoid_agent.phase2.baseline.summary.v1",
        "status": "pass" if workflow_complete else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "config": {
            "phase1_config": str(Path(args.phase1_config).resolve()),
            "phase1_config_sha256": sha256_file(args.phase1_config),
            "phase2_config": str(Path(args.phase2_config).resolve()),
            "phase2_config_sha256": sha256_file(args.phase2_config),
            "baseline": phase2_cfg["baseline"],
            "training": training_cfg,
            "evaluation": eval_cfg,
        },
        "seed_results": seed_results,
        "public_eval": {
            "success_rate_stats": public_stats,
            "target_success_rate_range": [target_low, target_high],
            "target_range_met": public_target_met,
            "condition_flag_breakdown": _aggregate_condition_flags(public_condition_flags),
        },
        "iid_hidden_eval": {
            "success_rate_stats": iid_stats,
            "condition_flag_breakdown": _aggregate_condition_flags(iid_condition_flags),
        },
        "compute": {
            "gpu_hours_estimate_total": total_gpu_hours,
            "gpu_hours_estimate_per_seed_mean": total_gpu_hours / len(seeds) if seeds else 0.0,
            "accounting_method": "wall_clock_seconds_times_one_gpu_per_run",
        },
        "decision": {
            "phase2_workflow_complete": workflow_complete,
            "difficulty_calibration_needed": workflow_complete and not public_target_met,
            "notes": [
                "Phase 2 baseline variance requires workflow completion, public/iid summaries, and compute accounting.",
                "If target_range_met is false, do not hide it; tune difficulty or budget in a later calibrated run.",
            ],
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_file": str(run_dir / "summary.json"), "status": output["status"]}, ensure_ascii=False))
    return 0 if workflow_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

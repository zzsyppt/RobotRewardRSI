#!/usr/bin/env python3
"""Noise-aware Phase 5 accept/reject decision.

This script is intentionally a Phase 5 endpoint, not a Phase 6 loop driver.  It
compares one candidate summary against the current best/baseline summary using
the protected public evaluation only.  IID-hidden results, if present in the
summaries, are copied as diagnostics but are not used for the decision.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLIC_MIN_DELTA = 0.05
POOLED_SE_MULTIPLIER = 1.0

PROTECTED_TRAINING_KEYS = (
    "task",
    "profile",
    "seeds",
    "num_envs",
    "max_iterations",
    "device",
    "timeout_seconds_per_seed",
)
PROTECTED_EVAL_KEYS = (
    "public",
    "iid_hidden",
    "target_success_rate_range",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _stats(summary: dict[str, Any], section: str) -> dict[str, Any]:
    stats = summary.get(section, {}).get("success_rate_stats", {})
    if not isinstance(stats, dict):
        return {"count": 0, "mean": None, "std": None, "standard_error": None, "values": []}
    return {
        "count": int(stats.get("count") or 0),
        "mean": _optional_float(stats.get("mean")),
        "std": _optional_float(stats.get("std")),
        "standard_error": _optional_float(stats.get("standard_error")),
        "values": [float(value) for value in stats.get("values", [])],
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    if candidate["mean"] is None or baseline["mean"] is None:
        return None
    return float(candidate["mean"]) - float(baseline["mean"])


def _pooled_standard_error(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    if candidate["standard_error"] is None or baseline["standard_error"] is None:
        return None
    return math.sqrt(float(candidate["standard_error"]) ** 2 + float(baseline["standard_error"]) ** 2)


def _protected_contract_issues(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    base_config = baseline.get("config", {})
    cand_config = candidate.get("config", {})
    base_training = base_config.get("training", {})
    cand_training = cand_config.get("training", {})
    base_eval = base_config.get("evaluation", {})
    cand_eval = cand_config.get("evaluation", {})

    for key in PROTECTED_TRAINING_KEYS:
        if base_training.get(key) != cand_training.get(key):
            issues.append(
                {
                    "type": "protected_training_mismatch",
                    "key": key,
                    "baseline": base_training.get(key),
                    "candidate": cand_training.get(key),
                }
            )
    for key in PROTECTED_EVAL_KEYS:
        if base_eval.get(key) != cand_eval.get(key):
            issues.append(
                {
                    "type": "protected_evaluation_mismatch",
                    "key": key,
                    "baseline": base_eval.get(key),
                    "candidate": cand_eval.get(key),
                }
            )
    return issues


def _paired_seed_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base_by_seed = {item.get("seed"): item for item in baseline.get("seed_results", [])}
    deltas: list[dict[str, Any]] = []
    for cand_item in candidate.get("seed_results", []):
        seed = cand_item.get("seed")
        base_item = base_by_seed.get(seed)
        if not isinstance(base_item, dict):
            continue
        base_rate = base_item.get("public_success_rate")
        cand_rate = cand_item.get("public_success_rate")
        if base_rate is None or cand_rate is None:
            continue
        deltas.append(
            {
                "seed": seed,
                "baseline_public_success_rate": float(base_rate),
                "candidate_public_success_rate": float(cand_rate),
                "delta": float(cand_rate) - float(base_rate),
            }
        )
    return deltas


def _summary_digest(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "summary_file": str(path),
        "schema_version": summary.get("schema_version"),
        "status": summary.get("status"),
        "run_dir": summary.get("run_dir"),
        "phase2_workflow_complete": summary.get("decision", {}).get("phase2_workflow_complete"),
        "phase1_config": summary.get("config", {}).get("phase1_config"),
        "phase1_config_sha256": summary.get("config", {}).get("phase1_config_sha256"),
        "phase2_config": summary.get("config", {}).get("phase2_config"),
        "phase2_config_sha256": summary.get("config", {}).get("phase2_config_sha256"),
    }


def build_decision(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_path: Path,
    candidate_path: Path,
    proposal_summary: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_baseline = _stats(baseline_summary, "public_eval")
    public_candidate = _stats(candidate_summary, "public_eval")
    iid_baseline = _stats(baseline_summary, "iid_hidden_eval")
    iid_candidate = _stats(candidate_summary, "iid_hidden_eval")
    public_delta = _mean_delta(public_candidate, public_baseline)
    pooled_se = _pooled_standard_error(public_candidate, public_baseline)
    threshold = None if pooled_se is None else max(PUBLIC_MIN_DELTA, POOLED_SE_MULTIPLIER * pooled_se)
    iid_delta = _mean_delta(iid_candidate, iid_baseline)
    protected_issues = _protected_contract_issues(baseline_summary, candidate_summary)
    workflow_ok = (
        baseline_summary.get("status") == "pass"
        and candidate_summary.get("status") == "pass"
        and bool(baseline_summary.get("decision", {}).get("phase2_workflow_complete"))
        and bool(candidate_summary.get("decision", {}).get("phase2_workflow_complete"))
    )
    decision_issues: list[dict[str, Any]] = []
    if not workflow_ok:
        decision_issues.append(
            {
                "type": "workflow_incomplete",
                "baseline_status": baseline_summary.get("status"),
                "candidate_status": candidate_summary.get("status"),
                "baseline_workflow_complete": baseline_summary.get("decision", {}).get("phase2_workflow_complete"),
                "candidate_workflow_complete": candidate_summary.get("decision", {}).get("phase2_workflow_complete"),
            }
        )
    decision_issues.extend(protected_issues)
    if public_delta is None or threshold is None:
        decision_issues.append({"type": "missing_public_statistics"})

    accepted = bool(workflow_ok and not protected_issues and public_delta is not None and threshold is not None and public_delta > threshold)
    decision = "accept" if accepted else "reject"
    if decision_issues:
        rationale = "reject: workflow/protected-contract/statistics issue"
    elif accepted:
        rationale = "accept: public mean improvement exceeds max(5pp, 1.0*pooled_SE)"
    else:
        rationale = "reject: public mean improvement does not exceed noise-aware threshold"

    return {
        "schema_version": "humanoid_agent.phase5.accept_reject.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase5_single_candidate",
        "decision": decision,
        "accepted": accepted,
        "rationale": rationale,
        "decision_basis": {
            "public_eval_only": True,
            "iid_hidden_used_for_decision": False,
            "rule": "accept iff candidate_public_mean - baseline_public_mean > max(0.05, 1.0 * pooled_standard_error)",
            "public_min_delta": PUBLIC_MIN_DELTA,
            "pooled_se_multiplier": POOLED_SE_MULTIPLIER,
            "public_delta": public_delta,
            "pooled_standard_error": pooled_se,
            "acceptance_threshold": threshold,
        },
        "baseline": {
            **_summary_digest(baseline_summary, baseline_path),
            "public_success_rate_stats": public_baseline,
            "iid_hidden_success_rate_stats": iid_baseline,
        },
        "candidate": {
            **_summary_digest(candidate_summary, candidate_path),
            "public_success_rate_stats": public_candidate,
            "iid_hidden_success_rate_stats": iid_candidate,
        },
        "public_seed_deltas": _paired_seed_deltas(baseline_summary, candidate_summary),
        "iid_hidden_diagnostic": {
            "baseline_mean": iid_baseline["mean"],
            "candidate_mean": iid_candidate["mean"],
            "delta": iid_delta,
            "diagnostic_only": True,
            "not_exposed_to_agent": True,
        },
        "protected_contract": {
            "status": "pass" if not protected_issues else "fail",
            "checked_training_keys": list(PROTECTED_TRAINING_KEYS),
            "checked_evaluation_keys": list(PROTECTED_EVAL_KEYS),
            "issues": protected_issues,
        },
        "proposal": {
            "run_summary": proposal_summary or {},
            "analysis": analysis or {},
        },
        "issues": decision_issues,
        "carry_forward": [
            "This is a single Phase 5 candidate decision, not a Phase 6 multi-round loop.",
            "IID-hidden diagnostics are recorded for audit only and must not be fed into the next LLM prompt.",
        ],
    }


def _selftest() -> int:
    baseline = _fake_summary([0.50, 0.52, 0.48, 0.51, 0.49])
    good = _fake_summary([0.70, 0.72, 0.69, 0.71, 0.73])
    noisy = _fake_summary([0.54, 0.52, 0.50, 0.51, 0.53])
    good_decision = build_decision(
        baseline_summary=baseline,
        candidate_summary=good,
        baseline_path=Path("baseline.json"),
        candidate_path=Path("good.json"),
    )
    noisy_decision = build_decision(
        baseline_summary=baseline,
        candidate_summary=noisy,
        baseline_path=Path("baseline.json"),
        candidate_path=Path("noisy.json"),
    )
    checks = [
        {"name": "large_improvement_accepts", "status": "pass" if good_decision["decision"] == "accept" else "fail"},
        {"name": "small_delta_rejects", "status": "pass" if noisy_decision["decision"] == "reject" else "fail"},
    ]
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    print(json.dumps({"status": status, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


def _fake_summary(public_values: list[float]) -> dict[str, Any]:
    count = len(public_values)
    mean = sum(public_values) / count
    variance = sum((value - mean) ** 2 for value in public_values) / (count - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(count)
    seeds = [101, 102, 103, 104, 105]
    training = {
        "task": "Unitree-G1-29dof-Velocity",
        "profile": "phase2_train",
        "seeds": seeds,
        "num_envs": 512,
        "max_iterations": 500,
        "device": "cuda:0",
        "timeout_seconds_per_seed": 1500,
    }
    evaluation = {
        "public": {"profile": "phase2_public", "seed": 1001, "num_envs": 128, "max_steps": 220},
        "iid_hidden": {"profile": "phase2_iid_hidden", "seed": 2001, "num_envs": 128, "max_steps": 220},
        "target_success_rate_range": [0.30, 0.70],
    }
    return {
        "schema_version": "selftest",
        "status": "pass",
        "run_dir": "selftest",
        "config": {"training": training, "evaluation": evaluation},
        "decision": {"phase2_workflow_complete": True},
        "public_eval": {
            "success_rate_stats": {
                "count": count,
                "mean": mean,
                "std": std,
                "standard_error": se,
                "values": public_values,
            }
        },
        "iid_hidden_eval": {
            "success_rate_stats": {
                "count": count,
                "mean": mean,
                "std": std,
                "standard_error": se,
                "values": public_values,
            }
        },
        "seed_results": [
            {"seed": seed, "public_success_rate": value, "iid_hidden_success_rate": value}
            for seed, value in zip(seeds, public_values, strict=True)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--proposal-summary", type=Path, default=None)
    parser.add_argument("--analysis-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    required = {
        "--baseline-summary": args.baseline_summary,
        "--candidate-summary": args.candidate_summary,
        "--output-file": args.output_file,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    baseline_summary = _load_json(args.baseline_summary)
    candidate_summary = _load_json(args.candidate_summary)
    proposal_summary = _load_json(args.proposal_summary) if args.proposal_summary and args.proposal_summary.exists() else None
    analysis = _load_json(args.analysis_file) if args.analysis_file and args.analysis_file.exists() else None
    decision = build_decision(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        baseline_path=args.baseline_summary,
        candidate_path=args.candidate_summary,
        proposal_summary=proposal_summary,
        analysis=analysis,
    )
    _write_json(args.output_file, decision)
    print(json.dumps({"decision_file": str(args.output_file), "decision": decision["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

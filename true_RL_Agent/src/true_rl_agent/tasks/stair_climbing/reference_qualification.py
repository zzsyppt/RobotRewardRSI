"""Final V2 reference capability qualification owned by the Stair task."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.reference_qualification import (
    REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
    ReferenceQualificationResult,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    validate_stair_split_summary,
)

STAIR_REFERENCE_QUALIFICATION_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reference_qualification_evidence.v1"
)


def qualify_stair_reference_public_result(
    *,
    config: StairClimbingTaskConfig,
    public_result: PublicEvalResult,
) -> ReferenceQualificationResult:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair reference qualification requires task config")
    if not isinstance(public_result, PublicEvalResult):
        raise TypeError("Stair reference qualification requires a public result")
    contract = _gate_contract(config)
    metrics = dict(public_result.metrics)
    split_name = str(contract["source_split"])
    split = _mapping(metrics.get(split_name), "Stair reference source split")
    canonical_split = validate_stair_split_summary(
        config.protocol_v1,
        semantics=config.evaluation_semantics,
        evaluation_protocol=config.evaluation_protocol(),
        split=split_name,
        summary=split,
    )
    split = canonical_split
    seed_count = _positive_int(contract["seed_count"], "Stair reference seed count")
    if split.get("seed_count") != seed_count:
        raise ValueError("Stair reference source split seed count mismatch")
    if split.get("complete_seven_bin_coverage") is not True:
        raise ValueError("Stair reference source split coverage is incomplete")
    episodes_per_bin = _positive_int(
        split.get("episodes_per_seed_per_bin"),
        "Stair reference episodes per seed per bin",
    )

    low_bins = tuple(str(item) for item in contract["low_step_bin_ids"])
    raw_seed_summaries = split.get("seed_summaries")
    if not isinstance(raw_seed_summaries, list) or len(raw_seed_summaries) != seed_count:
        raise ValueError("Stair reference seed summaries are incomplete")
    seed_measurements = []
    for expected_index, raw_seed in enumerate(raw_seed_summaries, start=1):
        seed = _mapping(raw_seed, "Stair reference seed summary")
        if (
            seed.get("seed_index") != expected_index
            or seed.get("complete_bin_coverage") is not True
        ):
            raise ValueError("Stair reference seed coverage mismatch")
        rates = _mapping(
            seed.get("bin_success_rates"),
            "Stair reference seed success rates",
        )
        selected = {
            bin_id: _rate(rates.get(bin_id), f"Stair reference success {bin_id}")
            for bin_id in low_bins
        }
        for value in selected.values():
            count = value * episodes_per_bin
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError("Stair reference success rate is not count-derived")
        seed_measurements.append(
            {
                "seed_index": expected_index,
                "success_rate_by_bin": selected,
                "low_step_macro_success_rate": statistics.fmean(
                    selected.values()
                ),
            }
        )
    observed_success = statistics.fmean(
        item["low_step_macro_success_rate"] for item in seed_measurements
    )

    progress = _mapping(
        split.get("progress_summary"),
        "Stair reference progress summary",
    )
    progress_by_bin = _mapping(
        progress.get("mean_max_progress_m_by_bin"),
        "Stair reference progress by bin",
    )
    selected_progress = {
        bin_id: _nonnegative_number(
            progress_by_bin.get(bin_id),
            f"Stair reference progress {bin_id}",
        )
        for bin_id in low_bins
    }
    observed_progress = statistics.fmean(selected_progress.values())
    success_floor = _rate(
        contract["minimum_macro_mean_success_rate"],
        "Stair reference success floor",
    )
    progress_floor = _nonnegative_number(
        contract["minimum_macro_mean_max_progress_m"],
        "Stair reference progress floor",
    )
    success_met = observed_success >= success_floor
    progress_met = observed_progress >= progress_floor
    passed = success_met and progress_met
    task_gate = {
        "schema_version": str(contract["schema_version"]),
        "status": "pass" if passed else "fail",
        "predicate": str(contract["predicate"]),
        "minimum_macro_mean_success_rate": success_floor,
        "minimum_macro_mean_max_progress_m": progress_floor,
        "observed_macro_mean_success_rate": observed_success,
        "observed_macro_mean_max_progress_m": observed_progress,
        "success_floor_met": success_met,
        "progress_floor_met": progress_met,
        "threshold_relaxation_allowed": False,
        "reason": (
            "frozen_reference_capability_floor_met"
            if passed
            else "frozen_reference_capability_floor_not_met"
        ),
    }
    evidence = {
        "schema_version": STAIR_REFERENCE_QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
        "task_protocol_hash": config.task_protocol_hash,
        "source_split": split_name,
        "gate_contract": dict(contract),
        "coverage": {
            "seed_count": seed_count,
            "episodes_per_seed_per_bin": episodes_per_bin,
            "low_step_bin_ids": list(low_bins),
            "complete_seven_bin_coverage": True,
        },
        "seed_success_measurements": seed_measurements,
        "progress_measurement": {
            "mean_max_progress_m_by_bin": selected_progress,
            "low_step_macro_max_progress_m": observed_progress,
            "aggregation_equivalence": (
                "per_seed_then_cross_seed_equals_aggregate_bin_macro_under_"
                "fixed_equal_episode_coverage"
            ),
        },
        "task_gate": task_gate,
    }
    evidence["evidence_hash"] = canonical_json_hash(evidence)
    public_gate = {
        "schema_version": REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
        "status": task_gate["status"],
        "protocol_hash": config.task_protocol_hash,
        "evidence_hash": evidence["evidence_hash"],
    }
    acceptance = dict(_mapping(metrics.get("acceptance"), "Stair acceptance"))
    required = acceptance.get("required_pass_gates")
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise ValueError("Stair acceptance required gates are invalid")
    if "reference_capability_gate" in required:
        raise ValueError("Stair reference capability gate is already projected")
    acceptance["required_pass_gates"] = [*required, "reference_capability_gate"]
    metrics["acceptance"] = acceptance
    metrics["reference_capability_gate"] = public_gate
    qualified_public = PublicEvalResult(
        status=public_result.status,
        public_score=public_result.public_score,
        metrics=metrics,
    )
    return ReferenceQualificationResult(
        public_result=qualified_public,
        gate=public_gate,
        private_evidence=evidence,
    )


def _gate_contract(config: StairClimbingTaskConfig) -> Mapping[str, Any]:
    contract = _mapping(
        config.protocol_v2.to_public_dict().get("reference_capability_gate"),
        "Stair reference capability contract",
    )
    expected = {
        "schema_version",
        "source_split",
        "seed_count",
        "low_step_bin_ids",
        "bin_coverage",
        "aggregation",
        "predicate",
        "minimum_macro_mean_success_rate",
        "minimum_macro_mean_max_progress_m",
        "threshold_relaxation_after_pilot_allowed",
    }
    if set(contract) != expected:
        raise ValueError("Stair reference capability contract fields mismatch")
    if contract["source_split"] != "public_accept":
        raise ValueError("Stair reference capability source split mismatch")
    if contract["threshold_relaxation_after_pilot_allowed"] is not False:
        raise ValueError("Stair reference capability threshold is mutable")
    low_bins = contract["low_step_bin_ids"]
    if not isinstance(low_bins, list) or not low_bins or len(low_bins) != len(set(low_bins)):
        raise ValueError("Stair reference capability low bins are invalid")
    return contract


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field_name} must be finite and non-negative")
    return float(value)


def _rate(value: Any, field_name: str) -> float:
    parsed = _nonnegative_number(value, field_name)
    if parsed > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return parsed


__all__ = [
    "STAIR_REFERENCE_QUALIFICATION_EVIDENCE_SCHEMA_VERSION",
    "qualify_stair_reference_public_result",
]

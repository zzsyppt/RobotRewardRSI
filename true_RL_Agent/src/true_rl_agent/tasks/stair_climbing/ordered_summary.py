"""Task-owned evidence aggregation for explicit non-grid Stair cells."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    STAIR_ORDERED_RUNNABLE_COHORTS,
    StairOrderedEvaluationSuite,
)

STAIR_ORDERED_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.ordered_evaluation_summary.v2"
)
_STAIR_ORDERED_LEGACY_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.ordered_evaluation_summary.v1"
)


def build_ordered_stair_summary(
    *,
    suite: StairOrderedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    split: str,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> dict[str, Any]:
    """Aggregate one split with equal cell weight and no grid projection."""

    _require_inputs(suite=suite, semantics=semantics, split=split)
    expected_assignment = suite.private_cell_assignment(split)
    expected_seed_count = _split_seed_count(suite, split)
    seeds = tuple(tuple(outcomes) for outcomes in per_seed_outcomes)
    if len(seeds) != expected_seed_count:
        raise ValueError("Stair ordered summary seed count mismatch")
    reasons = ("success", *tuple(semantics.to_dict()["episode_outcomes"]["failure_priority"]))
    diagnostic_flags = tuple(
        semantics.to_dict()["episode_outcomes"]["diagnostic_flags"]
    )
    cell_ids = tuple(cell.cell_id for cell in suite.runtime_cells(split))
    episodes = suite.episodes_per_seed_per_cell
    seed_summaries: list[dict[str, Any]] = []
    aggregate_counts = {cell_id: Counter({reason: 0 for reason in reasons}) for cell_id in cell_ids}
    aggregate_diagnostics = {
        cell_id: Counter({flag: 0 for flag in diagnostic_flags})
        for cell_id in cell_ids
    }
    for seed_index, outcomes in enumerate(seeds, start=1):
        if len(outcomes) != len(expected_assignment):
            raise ValueError("Stair ordered summary env count mismatch")
        counts = {cell_id: Counter({reason: 0 for reason in reasons}) for cell_id in cell_ids}
        diagnostics = {
            cell_id: Counter({flag: 0 for flag in diagnostic_flags})
            for cell_id in cell_ids
        }
        progress = {cell_id: [] for cell_id in cell_ids}
        for index, (outcome, cell) in enumerate(
            zip(outcomes, expected_assignment, strict=True)
        ):
            if not isinstance(outcome, StairEpisodeOutcome):
                raise TypeError("Stair ordered summary outcome is invalid")
            if (
                outcome.env_index != index
                or outcome.evaluation_cell_id != cell.cell_id
                or outcome.bin_id != cell.cell_id
            ):
                raise ValueError("Stair ordered summary outcome/cell identity drifted")
            if outcome.primary_reason not in reasons:
                raise ValueError("Stair ordered summary primary reason is undeclared")
            if outcome.success != (outcome.primary_reason == "success"):
                raise ValueError("Stair ordered summary success/reason mismatch")
            counts[cell.cell_id][outcome.primary_reason] += 1
            unknown_flags = set(outcome.diagnostic_flags) - set(diagnostic_flags)
            if unknown_flags:
                raise ValueError("Stair ordered summary diagnostic flag is undeclared")
            diagnostics[cell.cell_id].update(outcome.diagnostic_flags)
            progress[cell.cell_id].append(float(outcome.progress_fraction))
            aggregate_counts[cell.cell_id][outcome.primary_reason] += 1
            aggregate_diagnostics[cell.cell_id].update(outcome.diagnostic_flags)
        if any(sum(value.values()) != episodes for value in counts.values()):
            raise ValueError("Stair ordered per-seed cell coverage mismatch")
        success_by_cell = {
            cell_id: counts[cell_id]["success"] / episodes for cell_id in cell_ids
        }
        seed_summaries.append(
            {
                "seed_index": seed_index,
                "macro_score": statistics.fmean(success_by_cell.values()),
                "success_rate_by_cell": success_by_cell,
                "mean_progress_fraction_by_cell": {
                    cell_id: statistics.fmean(progress[cell_id])
                    for cell_id in cell_ids
                },
                "primary_outcome_counts": _sum_counts(counts, reasons=reasons),
                "primary_outcome_counts_by_cell": {
                    cell_id: dict(counts[cell_id]) for cell_id in cell_ids
                },
                "diagnostic_flag_counts_by_cell": {
                    cell_id: dict(diagnostics[cell_id]) for cell_id in cell_ids
                },
            }
        )
    aggregate_denominator = expected_seed_count * episodes
    aggregate_success = {
        cell_id: aggregate_counts[cell_id]["success"] / aggregate_denominator
        for cell_id in cell_ids
    }
    aggregate_progress_mean = {
        cell_id: statistics.fmean(
            float(seed["mean_progress_fraction_by_cell"][cell_id])
            for seed in seed_summaries
        )
        for cell_id in cell_ids
    }
    macro_scores = [float(seed["macro_score"]) for seed in seed_summaries]
    summary = {
        "schema_version": STAIR_ORDERED_SUMMARY_SCHEMA_VERSION,
        "ordered_suite_hash": suite.suite_hash,
        "split": split,
        "split_cohort_hash": suite.cohort_hash(split),
        "cell_aggregation": "equal_weight_mean_over_explicit_cells",
        "cell_ids": list(cell_ids),
        "episodes_per_seed_per_cell": episodes,
        "seed_count": expected_seed_count,
        "total_episode_count": expected_seed_count * len(cell_ids) * episodes,
        "seed_macro_scores": macro_scores,
        "score_stats": _score_stats(
            macro_scores,
            eval_protocol_hash=suite.evaluation_protocol.protocol_hash,
        ),
        "aggregate_success_rate_by_cell": aggregate_success,
        "aggregate_mean_progress_fraction_by_cell": aggregate_progress_mean,
        "primary_outcome_counts": _sum_counts(aggregate_counts, reasons=reasons),
        "primary_outcome_counts_by_cell": {
            cell_id: dict(aggregate_counts[cell_id]) for cell_id in cell_ids
        },
        "diagnostic_flag_counts_by_cell": {
            cell_id: dict(aggregate_diagnostics[cell_id]) for cell_id in cell_ids
        },
        "factor_diagnostics": _factor_diagnostics(
            suite=suite,
            split=split,
            success_by_cell=aggregate_success,
            progress_by_cell=aggregate_progress_mean,
        ),
        "seed_summaries": seed_summaries,
    }
    return validate_ordered_stair_summary(
        suite=suite,
        semantics=semantics,
        split=split,
        summary=summary,
    )


def validate_ordered_stair_summary(
    *,
    suite: StairOrderedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    split: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all redundant aggregates and return canonical JSON data."""

    _require_inputs(suite=suite, semantics=semantics, split=split)
    if not isinstance(summary, Mapping):
        raise TypeError("Stair ordered summary must be an object")
    schema_version = summary.get("schema_version")
    if schema_version not in {
        STAIR_ORDERED_SUMMARY_SCHEMA_VERSION,
        _STAIR_ORDERED_LEGACY_SUMMARY_SCHEMA_VERSION,
    }:
        raise ValueError("Stair ordered summary schema mismatch")
    has_diagnostic_counts = schema_version == STAIR_ORDERED_SUMMARY_SCHEMA_VERSION
    expected_fields = {
        "schema_version",
        "ordered_suite_hash",
        "split",
        "split_cohort_hash",
        "cell_aggregation",
        "cell_ids",
        "episodes_per_seed_per_cell",
        "seed_count",
        "total_episode_count",
        "seed_macro_scores",
        "score_stats",
        "aggregate_success_rate_by_cell",
        "aggregate_mean_progress_fraction_by_cell",
        "primary_outcome_counts",
        "primary_outcome_counts_by_cell",
        "factor_diagnostics",
        "seed_summaries",
    }
    if has_diagnostic_counts:
        expected_fields.add("diagnostic_flag_counts_by_cell")
    if set(summary) != expected_fields:
        raise ValueError("Stair ordered summary fields mismatch")
    identity = {
        "ordered_suite_hash": suite.suite_hash,
        "split": split,
        "split_cohort_hash": suite.cohort_hash(split),
        "cell_aggregation": "equal_weight_mean_over_explicit_cells",
        "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
        "seed_count": _split_seed_count(suite, split),
    }
    for key, expected in identity.items():
        if summary.get(key) != expected:
            raise ValueError(f"Stair ordered summary {key} mismatch")
    cell_ids = tuple(cell.cell_id for cell in suite.runtime_cells(split))
    if summary.get("cell_ids") != list(cell_ids):
        raise ValueError("Stair ordered summary cell order mismatch")
    expected_total = len(cell_ids) * suite.episodes_per_seed_per_cell * identity["seed_count"]
    if summary.get("total_episode_count") != expected_total:
        raise ValueError("Stair ordered summary total episode count mismatch")
    reasons = ("success", *tuple(semantics.to_dict()["episode_outcomes"]["failure_priority"]))
    diagnostic_flags = tuple(
        semantics.to_dict()["episode_outcomes"]["diagnostic_flags"]
    )
    seeds = summary.get("seed_summaries")
    if not isinstance(seeds, list) or len(seeds) != identity["seed_count"]:
        raise ValueError("Stair ordered summary seed summaries mismatch")
    aggregate_counts = {cell_id: Counter({reason: 0 for reason in reasons}) for cell_id in cell_ids}
    aggregate_diagnostics = {
        cell_id: Counter({flag: 0 for flag in diagnostic_flags})
        for cell_id in cell_ids
    }
    seed_scores: list[float] = []
    aggregate_progress_accumulator = {cell_id: [] for cell_id in cell_ids}
    for seed_index, seed in enumerate(seeds, start=1):
        _validate_seed_summary(
            seed,
            seed_index=seed_index,
            cell_ids=cell_ids,
            reasons=reasons,
            diagnostic_flags=diagnostic_flags,
            episodes=suite.episodes_per_seed_per_cell,
            has_diagnostic_counts=has_diagnostic_counts,
        )
        counts_by_cell = seed["primary_outcome_counts_by_cell"]
        success_by_cell = seed["success_rate_by_cell"]
        progress_by_cell = seed["mean_progress_fraction_by_cell"]
        expected_success = {
            cell_id: counts_by_cell[cell_id]["success"] / suite.episodes_per_seed_per_cell
            for cell_id in cell_ids
        }
        if success_by_cell != expected_success:
            raise ValueError("Stair ordered seed success rates mismatch")
        expected_macro = statistics.fmean(expected_success.values())
        if not _close(seed["macro_score"], expected_macro):
            raise ValueError("Stair ordered seed macro score mismatch")
        seed_scores.append(float(seed["macro_score"]))
        for cell_id in cell_ids:
            aggregate_counts[cell_id].update(counts_by_cell[cell_id])
            if has_diagnostic_counts:
                aggregate_diagnostics[cell_id].update(
                    seed["diagnostic_flag_counts_by_cell"][cell_id]
                )
            aggregate_progress_accumulator[cell_id].append(float(progress_by_cell[cell_id]))
    if summary.get("seed_macro_scores") != seed_scores:
        raise ValueError("Stair ordered macro score sequence mismatch")
    if summary.get("score_stats") != _score_stats(
        seed_scores,
        eval_protocol_hash=suite.evaluation_protocol.protocol_hash,
    ):
        raise ValueError("Stair ordered score statistics mismatch")
    expected_counts_by_cell = {cell_id: dict(aggregate_counts[cell_id]) for cell_id in cell_ids}
    if summary.get("primary_outcome_counts_by_cell") != expected_counts_by_cell:
        raise ValueError("Stair ordered aggregate per-cell outcomes mismatch")
    if summary.get("primary_outcome_counts") != _sum_counts(aggregate_counts, reasons=reasons):
        raise ValueError("Stair ordered aggregate outcomes mismatch")
    if has_diagnostic_counts:
        expected_diagnostics_by_cell = {
            cell_id: dict(aggregate_diagnostics[cell_id]) for cell_id in cell_ids
        }
        if summary.get("diagnostic_flag_counts_by_cell") != expected_diagnostics_by_cell:
            raise ValueError("Stair ordered aggregate diagnostic counts mismatch")
    denominator = identity["seed_count"] * suite.episodes_per_seed_per_cell
    expected_success = {
        cell_id: aggregate_counts[cell_id]["success"] / denominator for cell_id in cell_ids
    }
    if summary.get("aggregate_success_rate_by_cell") != expected_success:
        raise ValueError("Stair ordered aggregate success rates mismatch")
    expected_progress = {
        cell_id: statistics.fmean(aggregate_progress_accumulator[cell_id])
        for cell_id in cell_ids
    }
    if summary.get("aggregate_mean_progress_fraction_by_cell") != expected_progress:
        raise ValueError("Stair ordered aggregate progress mismatch")
    expected_factors = _factor_diagnostics(
        suite=suite,
        split=split,
        success_by_cell=expected_success,
        progress_by_cell=expected_progress,
    )
    if summary.get("factor_diagnostics") != expected_factors:
        raise ValueError("Stair ordered factor diagnostics mismatch")
    return _copy_json(summary)


def build_ordered_stair_prompt_feedback(
    *,
    suite: StairOrderedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose bounded public-dev ordered-stratum diagnostics to the next LLM round."""

    canonical = validate_ordered_stair_summary(
        suite=suite,
        semantics=semantics,
        split="public_dev",
        summary=summary,
    )
    total = int(canonical["total_episode_count"])
    counts = canonical["primary_outcome_counts"]
    failure_rates = {
        reason: int(count) / total
        for reason, count in counts.items()
        if reason != "success"
    }
    dominant_reason, dominant_rate = max(
        failure_rates.items(),
        key=lambda item: (item[1], item[0]),
    )
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.prompt_feedback.v9",
        "source_split": "public_dev",
        "mean_success_rate_by_cell": dict(
            canonical["aggregate_success_rate_by_cell"]
        ),
        "generalization_summary": {
            "public_dev_objective_score": float(
                canonical["score_stats"]["mean"]
            ),
            "ordered_suite_hash": suite.suite_hash,
            "split_cohort_hash": suite.cohort_hash("public_dev"),
            "success_rate_by_cell": dict(
                canonical["aggregate_success_rate_by_cell"]
            ),
            "factor_diagnostics": dict(canonical["factor_diagnostics"]),
        },
        "primary_failure_breakdown": failure_rates,
        "dominant_failure": {
            "reason": dominant_reason,
            "rate": dominant_rate,
        },
        "diagnostic_flag_rates": _diagnostic_flag_rates(canonical),
        "stable_success_summary": {
            "stable_top_success_rate": float(canonical["score_stats"]["mean"]),
            "valid_top_arrival_rate": None,
            "top_arrival_without_stable_success_rate": None,
        },
        "progress_summary": {
            "mean_progress_fraction": statistics.fmean(
                canonical["aggregate_mean_progress_fraction_by_cell"].values()
            ),
            "mean_progress_fraction_by_cell": dict(
                canonical["aggregate_mean_progress_fraction_by_cell"]
            ),
        },
    }


def _validate_seed_summary(
    seed: Any,
    *,
    seed_index: int,
    cell_ids: tuple[str, ...],
    reasons: tuple[str, ...],
    diagnostic_flags: tuple[str, ...],
    episodes: int,
    has_diagnostic_counts: bool,
) -> None:
    fields = {
        "seed_index",
        "macro_score",
        "success_rate_by_cell",
        "mean_progress_fraction_by_cell",
        "primary_outcome_counts",
        "primary_outcome_counts_by_cell",
    }
    if has_diagnostic_counts:
        fields.add("diagnostic_flag_counts_by_cell")
    if not isinstance(seed, Mapping) or set(seed) != fields or seed.get("seed_index") != seed_index:
        raise ValueError("Stair ordered seed summary fields mismatch")
    for key in ("success_rate_by_cell", "mean_progress_fraction_by_cell", "primary_outcome_counts_by_cell"):
        if not isinstance(seed.get(key), Mapping) or set(seed[key]) != set(cell_ids):
            raise ValueError(f"Stair ordered seed {key} cell set mismatch")
    counts = seed["primary_outcome_counts_by_cell"]
    for cell_id in cell_ids:
        if set(counts[cell_id]) != set(reasons) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts[cell_id].values()
        ) or sum(counts[cell_id].values()) != episodes:
            raise ValueError("Stair ordered seed per-cell outcome counts mismatch")
        for mapping_name in ("success_rate_by_cell", "mean_progress_fraction_by_cell"):
            value = seed[mapping_name][cell_id]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Stair ordered seed {mapping_name} values are invalid")
    if seed.get("primary_outcome_counts") != _sum_counts(counts, reasons=reasons):
        raise ValueError("Stair ordered seed aggregate outcomes mismatch")
    if has_diagnostic_counts:
        diagnostics = seed["diagnostic_flag_counts_by_cell"]
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(cell_ids):
            raise ValueError("Stair ordered seed diagnostic cell set mismatch")
        for cell_id in cell_ids:
            cell_diagnostics = diagnostics[cell_id]
            if (
                not isinstance(cell_diagnostics, Mapping)
                or set(cell_diagnostics) != set(diagnostic_flags)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value > episodes
                    for value in cell_diagnostics.values()
                )
            ):
                raise ValueError("Stair ordered seed diagnostic counts mismatch")


def _diagnostic_flag_rates(summary: Mapping[str, Any]) -> dict[str, float]:
    counts_by_cell = summary.get("diagnostic_flag_counts_by_cell")
    if not isinstance(counts_by_cell, Mapping) or not counts_by_cell:
        return {}
    total = int(summary["total_episode_count"])
    first = next(iter(counts_by_cell.values()))
    return {
        str(flag): sum(int(counts[flag]) for counts in counts_by_cell.values()) / total
        for flag in first
    }


def _factor_diagnostics(
    *,
    suite: StairOrderedEvaluationSuite,
    split: str,
    success_by_cell: Mapping[str, float],
    progress_by_cell: Mapping[str, float],
) -> dict[str, Any]:
    cells = suite.cells_for_cohort(split)
    factor_names = tuple(next(iter(cells)).factor_levels)
    result: dict[str, Any] = {}
    for factor in factor_names:
        levels: dict[str, Any] = {}
        ordered_levels = tuple(
            dict.fromkeys(cell.factor_levels[factor] for cell in cells)
        )
        for level in ordered_levels:
            members = [cell for cell in cells if cell.factor_levels[factor] == level]
            if not members:
                continue
            levels[level] = {
                "cell_count": len(members),
                "cell_ids": [cell.cell_id for cell in members],
                "success_rate": statistics.fmean(success_by_cell[cell.cell_id] for cell in members),
                "mean_progress_fraction": statistics.fmean(progress_by_cell[cell.cell_id] for cell in members),
            }
        result[factor] = levels
    return result


def _sum_counts(
    counts_by_cell: Mapping[str, Mapping[str, int]],
    *,
    reasons: tuple[str, ...],
) -> dict[str, int]:
    return {
        reason: sum(int(counts[reason]) for counts in counts_by_cell.values())
        for reason in reasons
    }


def _score_stats(
    values: Sequence[float],
    *,
    eval_protocol_hash: str,
) -> dict[str, Any]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "std": sample_std,
        "standard_error": sample_std / math.sqrt(len(values)),
        "eval_protocol_hash": eval_protocol_hash,
    }


def _require_inputs(
    *,
    suite: StairOrderedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    split: str,
) -> None:
    if not isinstance(suite, StairOrderedEvaluationSuite):
        raise TypeError("Stair ordered summary requires an ordered suite")
    if not isinstance(semantics, StairEvaluationSemantics):
        raise TypeError("Stair ordered summary requires evaluation semantics")
    if split not in STAIR_ORDERED_RUNNABLE_COHORTS:
        raise ValueError("Stair ordered summary split is invalid")


def _split_seed_count(suite: StairOrderedEvaluationSuite, split: str) -> int:
    return suite.seed_count(split)


def _close(left: Any, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, int | float)
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(dict(value), allow_nan=False))


__all__ = [
    "STAIR_ORDERED_SUMMARY_SCHEMA_VERSION",
    "build_ordered_stair_prompt_feedback",
    "build_ordered_stair_summary",
    "validate_ordered_stair_summary",
]

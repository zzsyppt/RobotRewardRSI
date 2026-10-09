"""Strict aggregation for the frozen Stair height-by-tread evaluation grid."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.tasks.stair_climbing.episode_evidence import (
    StairEpisodeOutcome,
    validate_stair_success_diagnostic_relations,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)

STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.generalization_summary.v4"
)
_LEGACY_STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.generalization_summary.v3"
)


def build_stair_generalization_summary(
    *,
    profile: StairEvaluationProfile,
    semantics: StairEvaluationSemantics,
    evaluation_protocol: EvaluationProtocol,
    split: str,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> dict[str, Any]:
    """Build one equal-cell macro score per seed from closed episode outcomes."""

    seed_count = _split_seed_count(evaluation_protocol, split)
    if len(per_seed_outcomes) != seed_count:
        raise ValueError(f"{split} requires {seed_count} complete Stair seeds")
    cells = profile.cells
    cell_ids = tuple(cell.cell_id for cell in cells)
    episodes_per_cell = profile.episodes_per_seed_per_cell
    episode_count_per_seed = len(cells) * episodes_per_cell
    height_ids = tuple(str(cell.bin_id) for cell in profile.anchor_cells)
    tread_keys = tuple(_tread_key(value) for value in profile.tread_depths_m)
    primary_reasons = tuple(semantics.outcomes["primary_reasons"])
    diagnostic_flags = tuple(semantics.outcomes["diagnostic_flags"])

    all_outcomes: list[StairEpisodeOutcome] = []
    seed_summaries: list[dict[str, Any]] = []
    seed_scores: list[float] = []
    total_success_count = 0
    for seed_index, raw_outcomes in enumerate(per_seed_outcomes, start=1):
        outcomes = list(raw_outcomes)
        _validate_outcome_cohort(
            outcomes,
            cell_ids=cell_ids,
            episodes_per_cell=episodes_per_cell,
            expected_episode_count=episode_count_per_seed,
            primary_reasons=primary_reasons,
            diagnostic_flags=diagnostic_flags,
        )
        seed_primary_counts_by_cell = {
            cell_id: {
                reason: sum(
                    1
                    for outcome in outcomes
                    if outcome.evaluation_cell_id == cell_id
                    and outcome.primary_reason == reason
                )
                for reason in primary_reasons
            }
            for cell_id in cell_ids
        }
        cell_rates = {
            cell_id: seed_primary_counts_by_cell[cell_id]["success"]
            / episodes_per_cell
            for cell_id in cell_ids
        }
        height_rates = {
            bin_id: statistics.fmean(
                cell_rates[cell.cell_id] for cell in cells if cell.bin_id == bin_id
            )
            for bin_id in height_ids
        }
        tread_rates = {
            tread_key: statistics.fmean(
                cell_rates[cell.cell_id]
                for cell in cells
                if _tread_key(cell.tread_depth_m) == tread_key
            )
            for tread_key in tread_keys
        }
        macro = statistics.fmean(cell_rates.values())
        seed_scores.append(macro)
        total_success_count += sum(
            counts["success"] for counts in seed_primary_counts_by_cell.values()
        )
        seed_summaries.append(
            {
                "seed_index": seed_index,
                "complete_cell_coverage": True,
                "episode_count": episode_count_per_seed,
                "primary_outcome_counts_by_cell": seed_primary_counts_by_cell,
                "cell_success_rates": cell_rates,
                "height_success_rates": height_rates,
                "tread_success_rates": tread_rates,
                "macro_success_rate": macro,
            }
        )
        all_outcomes.extend(outcomes)

    score_stats = _score_stats(seed_scores, evaluation_protocol)
    aggregate_by_height = {
        bin_id: statistics.fmean(
            summary["height_success_rates"][bin_id] for summary in seed_summaries
        )
        for bin_id in height_ids
    }
    aggregate_by_tread = {
        tread: statistics.fmean(
            summary["tread_success_rates"][tread] for summary in seed_summaries
        )
        for tread in tread_keys
    }
    aggregate_by_cell = {
        cell_id: statistics.fmean(
            summary["cell_success_rates"][cell_id] for summary in seed_summaries
        )
        for cell_id in cell_ids
    }
    primary_counts_by_cell = {
        cell_id: {
            reason: sum(
                seed["primary_outcome_counts_by_cell"][cell_id][reason]
                for seed in seed_summaries
            )
            for reason in primary_reasons
        }
        for cell_id in cell_ids
    }
    primary_counts = {
        reason: sum(
            counts[reason] for counts in primary_counts_by_cell.values()
        )
        for reason in primary_reasons
    }
    diagnostic_counts = Counter(
        flag for item in all_outcomes for flag in item.diagnostic_flags
    )
    total = len(all_outcomes)
    progress_by_tread = {
        tread: statistics.fmean(
            item.max_progress_m
            for item in all_outcomes
            if _tread_key(profile.cell(str(item.evaluation_cell_id)).tread_depth_m)
            == tread
        )
        for tread in tread_keys
    }
    return {
        "schema_version": STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION,
        "evaluation_profile_hash": profile.profile_hash,
        "evaluation_semantics_hash": semantics.protocol_hash,
        "split": split,
        "seed_count": seed_count,
        "cell_count": len(cells),
        "episodes_per_seed_per_cell": episodes_per_cell,
        "complete_height_tread_coverage": True,
        "seed_summaries": seed_summaries,
        "seed_macro_scores": seed_scores,
        "score_stats": score_stats,
        "aggregate_success_rate_by_cell": aggregate_by_cell,
        "aggregate_success_rate_by_height": aggregate_by_height,
        "aggregate_success_rate_by_tread": aggregate_by_tread,
        "difficulty_diagnostics": _difficulty_diagnostics(
            profile,
            aggregate_by_cell,
        ),
        "primary_outcome_counts": {
            reason: primary_counts[reason] for reason in primary_reasons
        },
        "primary_outcome_counts_by_cell": primary_counts_by_cell,
        "primary_failure_breakdown": {
            reason: primary_counts[reason] / total
            for reason in primary_reasons
            if reason != "success"
        },
        "diagnostic_flag_rates": {
            flag: int(diagnostic_counts.get(flag, 0)) / total
            for flag in diagnostic_flags
        },
        "progress_summary": {
            "mean_max_progress_m": statistics.fmean(
                item.max_progress_m for item in all_outcomes
            ),
            "mean_max_progress_m_by_tread": progress_by_tread,
        },
        "classification_warning_count": 0,
        "total_episode_count": total,
        "total_success_count": total_success_count,
    }


def validate_stair_generalization_summary(
    *,
    profile: StairEvaluationProfile,
    semantics: StairEvaluationSemantics,
    evaluation_protocol: EvaluationProtocol,
    split: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all score-bearing aggregates from the public seed summaries."""

    expected_fields = {
        "schema_version",
        "evaluation_profile_hash",
        "evaluation_semantics_hash",
        "split",
        "seed_count",
        "cell_count",
        "episodes_per_seed_per_cell",
        "complete_height_tread_coverage",
        "seed_summaries",
        "seed_macro_scores",
        "score_stats",
        "aggregate_success_rate_by_cell",
        "aggregate_success_rate_by_height",
        "aggregate_success_rate_by_tread",
        "difficulty_diagnostics",
        "primary_outcome_counts",
        "primary_outcome_counts_by_cell",
        "primary_failure_breakdown",
        "diagnostic_flag_rates",
        "progress_summary",
        "classification_warning_count",
        "total_episode_count",
        "total_success_count",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_fields:
        raise ValueError("Stair generalization summary fields mismatch")
    schema_version = summary.get("schema_version")
    if schema_version not in {
        STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION,
        _LEGACY_STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION,
    }:
        raise ValueError("Stair generalization summary schema_version mismatch")
    has_seed_primary_counts = (
        schema_version == STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION
    )
    identity = {
        "schema_version": schema_version,
        "evaluation_profile_hash": profile.profile_hash,
        "evaluation_semantics_hash": semantics.protocol_hash,
        "split": split,
        "seed_count": _split_seed_count(evaluation_protocol, split),
        "cell_count": len(profile.cells),
        "episodes_per_seed_per_cell": profile.episodes_per_seed_per_cell,
        "complete_height_tread_coverage": True,
        "classification_warning_count": 0,
    }
    for key, expected in identity.items():
        if summary.get(key) != expected:
            raise ValueError(f"Stair generalization summary {key} mismatch")

    seed_count = int(identity["seed_count"])
    episodes_per_cell = profile.episodes_per_seed_per_cell
    episode_count_per_seed = len(profile.cells) * episodes_per_cell
    total = seed_count * episode_count_per_seed
    if summary.get("total_episode_count") != total:
        raise ValueError("Stair generalization total episode count mismatch")
    raw_seeds = summary.get("seed_summaries")
    if not isinstance(raw_seeds, list) or len(raw_seeds) != seed_count:
        raise ValueError("Stair generalization seed summaries are incomplete")

    cell_ids = tuple(cell.cell_id for cell in profile.cells)
    height_ids = tuple(cell.bin_id for cell in profile.anchor_cells)
    tread_keys = tuple(_tread_key(value) for value in profile.tread_depths_m)
    primary_reasons = tuple(semantics.outcomes["primary_reasons"])
    canonical_seeds: list[dict[str, Any]] = []
    seed_scores: list[float] = []
    total_success = 0
    for seed_index, raw_seed in enumerate(raw_seeds, start=1):
        expected_seed_fields = {
            "seed_index",
            "complete_cell_coverage",
            "episode_count",
            "cell_success_rates",
            "height_success_rates",
            "tread_success_rates",
            "macro_success_rate",
        }
        if has_seed_primary_counts:
            expected_seed_fields.add("primary_outcome_counts_by_cell")
        if not isinstance(raw_seed, Mapping) or set(raw_seed) != expected_seed_fields:
            raise ValueError("Stair generalization seed summary fields mismatch")
        if (
            raw_seed.get("seed_index") != seed_index
            or raw_seed.get("complete_cell_coverage") is not True
            or raw_seed.get("episode_count") != episode_count_per_seed
        ):
            raise ValueError("Stair generalization seed summary identity mismatch")
        seed_primary_by_cell: dict[str, dict[str, int]] | None = None
        if has_seed_primary_counts:
            raw_seed_primary = _mapping(
                raw_seed.get("primary_outcome_counts_by_cell")
            )
            if set(raw_seed_primary) != set(cell_ids):
                raise ValueError(
                    "Stair generalization seed per-cell primary outcome fields mismatch"
                )
            seed_primary_by_cell = {}
            for cell_id in cell_ids:
                counts = _count_mapping(
                    raw_seed_primary[cell_id],
                    expected_keys=primary_reasons,
                    field_name=(
                        "seed_summaries.primary_outcome_counts_by_cell."
                        f"{cell_id}"
                    ),
                )
                if sum(counts.values()) != episodes_per_cell:
                    raise ValueError(
                        "Stair generalization seed per-cell primary outcome total mismatch"
                    )
                seed_primary_by_cell[cell_id] = counts
            cell_rates = {
                cell_id: seed_primary_by_cell[cell_id]["success"]
                / episodes_per_cell
                for cell_id in cell_ids
            }
            _require_rate_mapping_close(
                raw_seed.get("cell_success_rates"),
                cell_rates,
                "cell_success_rates",
            )
        else:
            cell_rates = _rate_mapping(
                raw_seed.get("cell_success_rates"),
                expected_keys=cell_ids,
                episodes_per_rate=episodes_per_cell,
                field_name="cell_success_rates",
            )
        height_rates = {
            bin_id: statistics.fmean(
                cell_rates[cell.cell_id]
                for cell in profile.cells
                if cell.bin_id == bin_id
            )
            for bin_id in height_ids
        }
        tread_rates = {
            tread: statistics.fmean(
                cell_rates[cell.cell_id]
                for cell in profile.cells
                if _tread_key(cell.tread_depth_m) == tread
            )
            for tread in tread_keys
        }
        _require_rate_mapping_close(
            raw_seed.get("height_success_rates"),
            height_rates,
            "height_success_rates",
        )
        _require_rate_mapping_close(
            raw_seed.get("tread_success_rates"),
            tread_rates,
            "tread_success_rates",
        )
        macro = statistics.fmean(cell_rates.values())
        _require_close(raw_seed.get("macro_success_rate"), macro, "seed macro")
        seed_scores.append(macro)
        total_success += (
            sum(counts["success"] for counts in seed_primary_by_cell.values())
            if seed_primary_by_cell is not None
            else sum(round(rate * episodes_per_cell) for rate in cell_rates.values())
        )
        canonical_seed = {
            "seed_index": seed_index,
            "complete_cell_coverage": True,
            "episode_count": episode_count_per_seed,
            "cell_success_rates": cell_rates,
            "height_success_rates": height_rates,
            "tread_success_rates": tread_rates,
            "macro_success_rate": macro,
        }
        if seed_primary_by_cell is not None:
            canonical_seed["primary_outcome_counts_by_cell"] = seed_primary_by_cell
        canonical_seeds.append(canonical_seed)

    raw_seed_scores = summary.get("seed_macro_scores")
    if not isinstance(raw_seed_scores, list) or len(raw_seed_scores) != seed_count:
        raise ValueError("Stair generalization seed macro scores are incomplete")
    for declared, expected in zip(raw_seed_scores, seed_scores, strict=True):
        _require_close(declared, expected, "seed macro score")
    expected_stats = _score_stats(seed_scores, evaluation_protocol)
    _validate_score_stats(summary.get("score_stats"), expected_stats)

    aggregate_by_height = {
        bin_id: statistics.fmean(
            seed["height_success_rates"][bin_id] for seed in canonical_seeds
        )
        for bin_id in height_ids
    }
    aggregate_by_tread = {
        tread: statistics.fmean(
            seed["tread_success_rates"][tread] for seed in canonical_seeds
        )
        for tread in tread_keys
    }
    aggregate_by_cell = {
        cell_id: statistics.fmean(
            seed["cell_success_rates"][cell_id] for seed in canonical_seeds
        )
        for cell_id in cell_ids
    }
    _require_rate_mapping_close(
        summary.get("aggregate_success_rate_by_cell"),
        aggregate_by_cell,
        "aggregate_success_rate_by_cell",
    )
    _require_rate_mapping_close(
        summary.get("aggregate_success_rate_by_height"),
        aggregate_by_height,
        "aggregate_success_rate_by_height",
    )
    _require_rate_mapping_close(
        summary.get("aggregate_success_rate_by_tread"),
        aggregate_by_tread,
        "aggregate_success_rate_by_tread",
    )
    expected_diagnostics = _difficulty_diagnostics(profile, aggregate_by_cell)
    if dict(_mapping(summary.get("difficulty_diagnostics"))) != expected_diagnostics:
        raise ValueError("Stair difficulty diagnostics mismatch")
    if summary.get("total_success_count") != total_success:
        raise ValueError("Stair generalization success count mismatch")

    primary = _count_mapping(
        summary.get("primary_outcome_counts"),
        expected_keys=primary_reasons,
        field_name="primary_outcome_counts",
    )
    if sum(primary.values()) != total or primary.get("success") != total_success:
        raise ValueError("Stair generalization primary outcome counts mismatch")
    raw_primary_by_cell = _mapping(summary.get("primary_outcome_counts_by_cell"))
    if set(raw_primary_by_cell) != set(cell_ids):
        raise ValueError(
            "Stair generalization per-cell primary outcome fields mismatch"
        )
    episodes_per_aggregate_cell = seed_count * episodes_per_cell
    primary_by_cell: dict[str, dict[str, int]] = {}
    for cell_id in cell_ids:
        cell_counts = _count_mapping(
            raw_primary_by_cell[cell_id],
            expected_keys=primary_reasons,
            field_name=f"primary_outcome_counts_by_cell.{cell_id}",
        )
        if sum(cell_counts.values()) != episodes_per_aggregate_cell:
            raise ValueError(
                "Stair generalization per-cell primary outcome total mismatch"
            )
        expected_cell_success = round(
            aggregate_by_cell[cell_id] * episodes_per_aggregate_cell
        )
        if cell_counts.get("success") != expected_cell_success:
            raise ValueError(
                "Stair generalization per-cell success outcome mismatch"
            )
        primary_by_cell[cell_id] = cell_counts
    if has_seed_primary_counts:
        expected_primary_by_cell = {
            cell_id: {
                reason: sum(
                    seed["primary_outcome_counts_by_cell"][cell_id][reason]
                    for seed in canonical_seeds
                )
                for reason in primary_reasons
            }
            for cell_id in cell_ids
        }
        if primary_by_cell != expected_primary_by_cell:
            raise ValueError(
                "Stair generalization per-seed primary outcomes do not match aggregates"
            )
        expected_primary = {
            reason: sum(
                counts[reason] for counts in expected_primary_by_cell.values()
            )
            for reason in primary_reasons
        }
        if primary != expected_primary:
            raise ValueError(
                "Stair generalization per-seed primary outcome total mismatch"
            )
    elif any(
        sum(counts[reason] for counts in primary_by_cell.values())
        != primary[reason]
        for reason in primary_reasons
    ):
        raise ValueError(
            "Stair generalization per-cell primary outcome aggregate mismatch"
        )
    failure_rates = _validated_fraction_mapping(
        summary.get("primary_failure_breakdown"),
        expected_counts={
            key: value for key, value in primary.items() if key != "success"
        },
        total=total,
        field_name="primary_failure_breakdown",
    )
    diagnostics = _validated_count_derived_rates(
        summary.get("diagnostic_flag_rates"),
        expected_keys=tuple(semantics.outcomes["diagnostic_flags"]),
        total=total,
        field_name="diagnostic_flag_rates",
    )
    progress = _mapping(summary.get("progress_summary"))
    if set(progress) != {"mean_max_progress_m", "mean_max_progress_m_by_tread"}:
        raise ValueError("Stair generalization progress summary fields mismatch")
    progress_by_tread = _nonnegative_mapping(
        progress.get("mean_max_progress_m_by_tread"),
        expected_keys=tread_keys,
        field_name="mean_max_progress_m_by_tread",
    )
    mean_progress = _nonnegative_number(
        progress.get("mean_max_progress_m"),
        "mean_max_progress_m",
    )
    _require_close(
        mean_progress, statistics.fmean(progress_by_tread.values()), "mean progress"
    )

    return {
        **identity,
        "seed_summaries": canonical_seeds,
        "seed_macro_scores": seed_scores,
        "score_stats": expected_stats,
        "aggregate_success_rate_by_cell": aggregate_by_cell,
        "aggregate_success_rate_by_height": aggregate_by_height,
        "aggregate_success_rate_by_tread": aggregate_by_tread,
        "difficulty_diagnostics": expected_diagnostics,
        "primary_outcome_counts": primary,
        "primary_outcome_counts_by_cell": primary_by_cell,
        "primary_failure_breakdown": failure_rates,
        "diagnostic_flag_rates": diagnostics,
        "progress_summary": {
            "mean_max_progress_m": mean_progress,
            "mean_max_progress_m_by_tread": progress_by_tread,
        },
        "total_episode_count": total,
        "total_success_count": total_success,
    }


def _validate_outcome_cohort(
    outcomes: Sequence[StairEpisodeOutcome],
    *,
    cell_ids: tuple[str, ...],
    episodes_per_cell: int,
    expected_episode_count: int,
    primary_reasons: tuple[str, ...],
    diagnostic_flags: tuple[str, ...],
) -> None:
    if len(outcomes) != expected_episode_count:
        raise ValueError("Stair generalization episode coverage is incomplete")
    env_indexes = [item.env_index for item in outcomes]
    if env_indexes != list(range(expected_episode_count)):
        raise ValueError("Stair generalization env identity coverage is incomplete")
    counts = Counter(item.evaluation_cell_id for item in outcomes)
    if set(counts) != set(cell_ids) or any(
        counts[cell_id] != episodes_per_cell for cell_id in cell_ids
    ):
        raise ValueError("Stair generalization cell coverage is incomplete")
    if any(item.success != (item.primary_reason == "success") for item in outcomes):
        raise ValueError("Stair generalization success flag differs from reason")
    if any(item.primary_reason not in primary_reasons for item in outcomes):
        raise ValueError("Stair generalization primary reason is undeclared")
    if any(
        not set(item.diagnostic_flags).issubset(diagnostic_flags) for item in outcomes
    ):
        raise ValueError("Stair generalization diagnostic flag is undeclared")
    validate_stair_success_diagnostic_relations(
        outcomes,
        declared_diagnostic_flags=diagnostic_flags,
    )


def _difficulty_diagnostics(
    profile: StairEvaluationProfile,
    aggregate_by_cell: Mapping[str, float],
) -> dict[str, Any]:
    inversion_count = 0
    max_inversion = 0.0
    adjacent_pair_count = 0
    for tread in profile.tread_depths_m:
        cells = sorted(
            (
                cell
                for cell in profile.cells
                if math.isclose(
                    cell.tread_depth_m,
                    tread,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            key=lambda cell: cell.height_m,
        )
        for easier, harder in zip(cells, cells[1:]):
            adjacent_pair_count += 1
            inversion = (
                aggregate_by_cell[harder.cell_id] - aggregate_by_cell[easier.cell_id]
            )
            if inversion > 1.0e-12:
                inversion_count += 1
                max_inversion = max(max_inversion, inversion)
    return {
        "adjacent_height_pair_count": adjacent_pair_count,
        "inversion_count": inversion_count,
        "inversion_rate": inversion_count / adjacent_pair_count,
        "maximum_inversion": max_inversion,
        "acceptance_role": "diagnostic_only",
    }


def _score_stats(
    seed_scores: Sequence[float],
    evaluation_protocol: EvaluationProtocol,
) -> dict[str, Any]:
    mean = statistics.fmean(seed_scores)
    std = statistics.stdev(seed_scores) if len(seed_scores) > 1 else 0.0
    return {
        "count": len(seed_scores),
        "mean": mean,
        "std": std,
        "standard_error": std / math.sqrt(len(seed_scores)),
        "eval_protocol_hash": evaluation_protocol.protocol_hash,
    }


def _validate_score_stats(value: Any, expected: Mapping[str, Any]) -> None:
    validation = validate_score_stats(
        value,
        minimum_count=int(expected["count"]),
        require_std=True,
        require_standard_error=True,
        require_protocol_hash=True,
    )
    if not validation.valid or validation.count != expected["count"]:
        raise ValueError("Stair generalization score stats are invalid")
    for key in ("mean", "std", "standard_error"):
        _require_close(getattr(validation, key), expected[key], f"score_stats.{key}")
    if validation.eval_protocol_hash != expected["eval_protocol_hash"]:
        raise ValueError("Stair generalization eval protocol hash mismatch")


def _rate_mapping(
    value: Any,
    *,
    expected_keys: Sequence[str],
    episodes_per_rate: int,
    field_name: str,
) -> dict[str, float]:
    mapping = _mapping(value)
    if set(mapping) != set(expected_keys):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    result = {key: _bounded_rate(mapping[key], field_name) for key in expected_keys}
    for rate in result.values():
        if not math.isclose(
            rate * episodes_per_rate,
            round(rate * episodes_per_rate),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"Stair generalization {field_name} is not count-derived")
    return result


def _require_rate_mapping_close(
    value: Any,
    expected: Mapping[str, float],
    field_name: str,
) -> None:
    mapping = _mapping(value)
    if set(mapping) != set(expected):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    for key, expected_value in expected.items():
        _require_close(mapping[key], expected_value, f"{field_name}.{key}")


def _count_mapping(
    value: Any,
    *,
    expected_keys: Sequence[str],
    field_name: str,
) -> dict[str, int]:
    mapping = _mapping(value)
    if set(mapping) != set(expected_keys):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    result: dict[str, int] = {}
    for key in expected_keys:
        raw = mapping[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"Stair generalization {field_name} counts are invalid")
        result[key] = raw
    return result


def _validated_fraction_mapping(
    value: Any,
    *,
    expected_counts: Mapping[str, int],
    total: int,
    field_name: str,
) -> dict[str, float]:
    mapping = _mapping(value)
    if set(mapping) != set(expected_counts):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    result: dict[str, float] = {}
    for key, count in expected_counts.items():
        rate = _bounded_rate(mapping[key], field_name)
        _require_close(rate, count / total, f"{field_name}.{key}")
        result[key] = rate
    return result


def _validated_count_derived_rates(
    value: Any,
    *,
    expected_keys: Sequence[str],
    total: int,
    field_name: str,
) -> dict[str, float]:
    mapping = _mapping(value)
    if set(mapping) != set(expected_keys):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    result: dict[str, float] = {}
    for key in expected_keys:
        rate = _bounded_rate(mapping[key], field_name)
        if not math.isclose(rate * total, round(rate * total), abs_tol=1.0e-9):
            raise ValueError(f"Stair generalization {field_name} is not count-derived")
        result[key] = rate
    return result


def _nonnegative_mapping(
    value: Any,
    *,
    expected_keys: Sequence[str],
    field_name: str,
) -> dict[str, float]:
    mapping = _mapping(value)
    if set(mapping) != set(expected_keys):
        raise ValueError(f"Stair generalization {field_name} keys mismatch")
    return {
        key: _nonnegative_number(mapping[key], f"{field_name}.{key}")
        for key in expected_keys
    }


def _split_seed_count(protocol: EvaluationProtocol, split: str) -> int:
    try:
        return {
            "public_dev": protocol.public_seed_count,
            "public_accept": protocol.public_accept_seed_count,
            "sealed_validation": protocol.sealed_validation_seed_count,
        }[split]
    except KeyError as exc:
        raise ValueError(f"unknown Stair evaluation split: {split}") from exc


def _tread_key(value: float) -> str:
    return f"{float(value):.2f}"


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stair generalization value must be an object")
    return value


def _bounded_rate(value: Any, field_name: str) -> float:
    parsed = _finite_number(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"Stair generalization {field_name} must be in [0, 1]")
    return parsed


def _nonnegative_number(value: Any, field_name: str) -> float:
    parsed = _finite_number(value, field_name)
    if parsed < 0.0:
        raise ValueError(f"Stair generalization {field_name} must be non-negative")
    return parsed


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Stair generalization {field_name} must be finite")
    return float(value)


def _require_close(value: Any, expected: float, field_name: str) -> None:
    actual = _finite_number(value, field_name)
    if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(f"Stair generalization {field_name} mismatch")


__all__ = [
    "STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION",
    "build_stair_generalization_summary",
    "validate_stair_generalization_summary",
]

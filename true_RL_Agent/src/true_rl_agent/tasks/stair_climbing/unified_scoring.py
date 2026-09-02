"""Task-owned ID/OOD score aggregation for unified Stair evaluation."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.binding import canonical_json_hash, is_sha256_digest
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.ordered_summary import (
    build_ordered_stair_prompt_feedback,
    build_ordered_stair_summary,
    validate_ordered_stair_summary,
)
from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
    StairUnifiedEvaluationSuite,
)

STAIR_UNIFIED_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.unified_evaluation_summary.v1"
)
STAIR_OBJECTIVE_FAMILY_MASK_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.objective_family_mask.v2"
)
UNIFIED_STAIR_PROMPT_OBJECTIVE_SCORE_SOURCE = (
    "prompt_feedback.generalization_summary.public_dev_objective_score"
)


def unified_stair_prompt_named_guard_score_source(name: str) -> str:
    """Return the provider-visible public-dev path for one named guard."""

    if not isinstance(name, str) or not name:
        raise ValueError("unified Stair prompt named guard requires a name")
    return f"prompt_feedback.generalization_summary.named_guard_scores.{name}"


def build_unified_stair_summary(
    *,
    suite: StairUnifiedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    split: str,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> dict[str, Any]:
    """Aggregate episode -> cell -> stratum -> seed without geometry weighting."""

    if not isinstance(suite, StairUnifiedEvaluationSuite):
        raise TypeError("unified Stair summary requires its evaluation suite")
    base = build_ordered_stair_summary(
        suite=suite,
        semantics=semantics,
        split=split,
        per_seed_outcomes=per_seed_outcomes,
    )
    return _build_unified_summary_from_ordered(
        suite=suite,
        split=split,
        ordered_summary=base,
    )


def validate_unified_stair_summary(
    *,
    suite: StairUnifiedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    split: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("unified Stair summary must be an object")
    ordered = validate_ordered_stair_summary(
        suite=suite,
        semantics=semantics,
        split=split,
        summary=summary.get("ordered_summary"),
    )
    expected = _build_unified_summary_from_ordered(
        suite=suite,
        split=split,
        ordered_summary=ordered,
    )
    if dict(summary) != expected:
        raise ValueError("unified Stair derived score summary mismatch")
    return expected


def build_reference_objective_family_mask(
    *,
    suite: StairUnifiedEvaluationSuite,
    public_accept_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the OOD families supported by the reference policy."""

    if not isinstance(suite, StairUnifiedEvaluationSuite):
        raise TypeError("objective family mask requires its evaluation suite")
    if (
        public_accept_summary.get("schema_version")
        != STAIR_UNIFIED_SUMMARY_SCHEMA_VERSION
        or public_accept_summary.get("split") != "public_accept"
        or public_accept_summary.get("ordered_suite_hash") != suite.suite_hash
    ):
        raise ValueError("objective family mask requires canonical public_accept")
    rule = suite.qualification_rule
    minimum_score = float(rule["minimum_reference_score"])
    ordered = public_accept_summary["ordered_summary"]
    success_by_cell = ordered["aggregate_success_rate_by_cell"]
    family_measurements: dict[str, Any] = {}
    qualified: list[str] = []
    declared_families = [str(name) for name in suite.score_contract["families"]]
    for family in declared_families:
        member_strata = suite.score_contract["families"][family]
        cell_ids = [
            cell_id
            for stratum in member_strata
            for cell_id in public_accept_summary["stratum_cell_ids"].get(stratum, ())
        ]
        if not cell_ids:
            raise ValueError(f"objective family {family} has no public_accept cells")
        score = float(public_accept_summary["family_scores"][family])
        success = statistics.fmean(float(success_by_cell[cell]) for cell in cell_ids)
        # A family below the reference floor cannot provide a useful bounded
        # optimization signal.  A ceiling family is different: it is already a
        # demonstrated capability and must remain in J_OOD so a candidate cannot
        # destroy it for free.
        is_qualified = score >= minimum_score
        family_measurements[family] = {
            "score": score,
            "macro_success_rate": success,
            "cell_count": len(cell_ids),
            "qualified": is_qualified,
        }
        if is_qualified:
            qualified.append(family)
    payload = {
        "schema_version": STAIR_OBJECTIVE_FAMILY_MASK_SCHEMA_VERSION,
        "ordered_suite_hash": suite.suite_hash,
        "source_cohort": "public_accept",
        "source_cohort_hash": suite.cohort_hash("public_accept"),
        "qualification_rule_hash": canonical_json_hash(dict(rule)),
        "declared_ood_families": declared_families,
        "qualified_ood_families": qualified,
        "family_measurements": family_measurements,
    }
    payload["objective_family_mask_hash"] = canonical_json_hash(payload)
    return payload


def validate_objective_family_mask(
    *,
    suite: StairUnifiedEvaluationSuite,
    mask: Mapping[str, Any],
    public_accept_summary: Mapping[str, Any] | None = None,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Validate one reference-derived family mask and optionally replay it."""

    if not isinstance(require_nonempty, bool):
        raise TypeError("objective family mask require_nonempty must be boolean")
    if not isinstance(mask, Mapping):
        raise ValueError("objective family mask must be an object")
    expected_fields = {
        "schema_version",
        "ordered_suite_hash",
        "source_cohort",
        "source_cohort_hash",
        "qualification_rule_hash",
        "declared_ood_families",
        "qualified_ood_families",
        "family_measurements",
        "objective_family_mask_hash",
    }
    if set(mask) != expected_fields:
        raise ValueError("objective family mask fields mismatch")
    canonical = dict(mask)
    observed_hash = canonical.pop("objective_family_mask_hash")
    if not is_sha256_digest(observed_hash) or canonical_json_hash(canonical) != observed_hash:
        raise ValueError("objective family mask hash mismatch")
    declared = [str(name) for name in suite.score_contract["families"]]
    qualified = mask.get("qualified_ood_families")
    if (
        mask.get("schema_version") != STAIR_OBJECTIVE_FAMILY_MASK_SCHEMA_VERSION
        or mask.get("ordered_suite_hash") != suite.suite_hash
        or mask.get("source_cohort") != "public_accept"
        or mask.get("source_cohort_hash") != suite.cohort_hash("public_accept")
        or mask.get("qualification_rule_hash")
        != canonical_json_hash(dict(suite.qualification_rule))
        or mask.get("declared_ood_families") != declared
        or not isinstance(qualified, list)
        or (require_nonempty and not qualified)
        or len(qualified) != len(set(qualified))
        or any(name not in declared for name in qualified)
    ):
        raise ValueError("objective family mask contract mismatch")
    measurements = mask.get("family_measurements")
    if not isinstance(measurements, Mapping) or set(measurements) != set(declared):
        raise ValueError("objective family mask measurements mismatch")
    rule = suite.qualification_rule
    expected_qualified: list[str] = []
    for family in declared:
        measurement = measurements[family]
        if not isinstance(measurement, Mapping) or set(measurement) != {
            "score",
            "macro_success_rate",
            "cell_count",
            "qualified",
        }:
            raise ValueError("objective family mask measurement fields mismatch")
        score = measurement["score"]
        success = measurement["macro_success_rate"]
        cell_count = measurement["cell_count"]
        observed_qualified = measurement["qualified"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or isinstance(success, bool)
            or not isinstance(success, int | float)
            or not math.isfinite(float(success))
            or not 0.0 <= float(success) <= 1.0
            or isinstance(cell_count, bool)
            or not isinstance(cell_count, int)
            or cell_count <= 0
            or not isinstance(observed_qualified, bool)
        ):
            raise ValueError("objective family mask measurement is invalid")
        expected = float(score) >= float(rule["minimum_reference_score"])
        if observed_qualified != expected:
            raise ValueError("objective family mask qualification predicate mismatch")
        if expected:
            expected_qualified.append(family)
    if qualified != expected_qualified:
        raise ValueError("objective family mask qualified family list mismatch")
    if public_accept_summary is not None:
        replayed = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=public_accept_summary,
        )
        if canonical_json_hash(replayed) != canonical_json_hash(dict(mask)):
            raise ValueError("objective family mask differs from reference replay")
    return dict(mask)


def objective_family_mask_from_public_result(
    *,
    suite: StairUnifiedEvaluationSuite,
    public_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the one frozen objective mask carried by reference public evidence."""

    if not isinstance(public_result, Mapping):
        raise ValueError("unified Stair reference public result is unavailable")
    metrics = public_result.get("metrics")
    mask = metrics.get("objective_family_mask") if isinstance(metrics, Mapping) else None
    if not isinstance(mask, Mapping):
        raise ValueError("unified Stair reference public result lacks its objective mask")
    return validate_objective_family_mask(
        suite=suite,
        mask=_thaw_json_value(mask),
    )


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(item) for item in value]
    return value


def build_unified_score_projection(
    summary: Mapping[str, Any],
    *,
    objective_family_mask: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose paired seed vectors at the exact v6 acceptance source paths."""

    if summary.get("schema_version") != STAIR_UNIFIED_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unified Stair score projection summary is invalid")
    raw_seeds = summary.get("seed_summaries")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("unified Stair score projection has no paired seeds")
    if objective_family_mask is None:
        qualified_families = list(summary.get("family_scores", {}))
        mask_hash = None
        aggregation_mode = "pre_reference_all_declared_families"
    else:
        qualified_families, mask_hash = _mask_projection_identity(
            summary=summary,
            mask=objective_family_mask,
        )
        aggregation_mode = "frozen_reference_qualified_families"
    j_ood_scores = [
        statistics.fmean(float(seed["family_scores"][name]) for name in qualified_families)
        for seed in raw_seeds
    ]
    j_id_scores = [float(seed["j_id"]) for seed in raw_seeds]
    j_primary_scores = [
        (2.0 * j_ood + j_id) / 3.0
        for j_ood, j_id in zip(j_ood_scores, j_id_scores, strict=True)
    ]
    sources = {
        "j_primary": j_primary_scores,
        "j_ood": j_ood_scores,
        "c0_id": [
            float(seed["named_guard_scores"]["c0_id"]) for seed in raw_seeds
        ],
        "c1_id": [
            float(seed["named_guard_scores"]["c1_id"]) for seed in raw_seeds
        ],
    }
    eval_protocol_hash = str(
        summary["ordered_summary"]["score_stats"]["eval_protocol_hash"]
    )
    projection = {
        name: {
            "score_stats": _score_stats(
                values,
                eval_protocol_hash=eval_protocol_hash,
            ),
            "seed_macro_scores": list(values),
        }
        for name, values in sources.items()
    }
    projection.update(
        {
            "aggregation_mode": aggregation_mode,
            "qualified_ood_families": qualified_families,
            "objective_family_mask_hash": mask_hash,
        }
    )
    return projection


def build_unified_reference_score_projection(
    summary: Mapping[str, Any],
    *,
    objective_family_mask: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a reference diagnostically when every OOD family is below floor."""

    if objective_family_mask.get("qualified_ood_families"):
        return build_unified_score_projection(
            summary,
            objective_family_mask=objective_family_mask,
        )
    projection = build_unified_score_projection(summary)
    projection.update(
        {
            "aggregation_mode": (
                "diagnostic_all_declared_families_empty_reference_mask"
            ),
            "qualified_ood_families": [],
            "objective_family_mask_hash": objective_family_mask.get(
                "objective_family_mask_hash"
            ),
        }
    )
    return projection


def build_unified_stair_prompt_feedback(
    *,
    suite: StairUnifiedEvaluationSuite,
    semantics: StairEvaluationSemantics,
    summary: Mapping[str, Any],
    objective_family_mask: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose task-owned stratum evidence without flattening OOD claims."""

    canonical = validate_unified_stair_summary(
        suite=suite,
        semantics=semantics,
        split="public_dev",
        summary=summary,
    )
    ordered = canonical["ordered_summary"]
    score_projection = (
        build_unified_reference_score_projection(
            canonical,
            objective_family_mask=objective_family_mask,
        )
        if objective_family_mask is not None
        and not objective_family_mask.get("qualified_ood_families")
        else build_unified_score_projection(
            canonical,
            objective_family_mask=objective_family_mask,
        )
    )
    named_guard_names = tuple(
        str(name) for name in suite.score_contract["named_guards"]
    )
    feedback = build_ordered_stair_prompt_feedback(
        suite=suite,
        semantics=semantics,
        summary=ordered,
    )
    success = ordered["aggregate_success_rate_by_cell"]
    progress = ordered["aggregate_mean_progress_fraction_by_cell"]
    outcome_counts = ordered["primary_outcome_counts_by_cell"]
    diagnostic_counts = ordered.get("diagnostic_flag_counts_by_cell")
    per_cell_episode_count = int(ordered["seed_count"]) * int(
        ordered["episodes_per_seed_per_cell"]
    )
    stratum_diagnostics = {}
    for stratum, cell_ids in canonical["stratum_cell_ids"].items():
        stratum_counts = {
            reason: sum(int(outcome_counts[cell][reason]) for cell in cell_ids)
            for reason in next(iter(outcome_counts.values()))
        }
        stratum_total = sum(stratum_counts.values())
        stratum_diagnostics[stratum] = {
            "cell_count": len(cell_ids),
            "cell_ids": list(cell_ids),
            "score": float(canonical["aggregate_stratum_scores"][stratum]),
            "success_rate": statistics.fmean(float(success[cell]) for cell in cell_ids),
            "mean_progress_fraction": statistics.fmean(
                float(progress[cell]) for cell in cell_ids
            ),
            "primary_failure_breakdown": {
                reason: count / stratum_total
                for reason, count in stratum_counts.items()
                if reason != "success"
            },
            "diagnostic_flag_rates": (
                {
                    flag: sum(
                        int(diagnostic_counts[cell][flag]) for cell in cell_ids
                    )
                    / (len(cell_ids) * per_cell_episode_count)
                    for flag in next(iter(diagnostic_counts.values()))
                }
                if isinstance(diagnostic_counts, Mapping) and diagnostic_counts
                else {}
            ),
        }
    feedback["generalization_summary"] = {
        "public_dev_objective_score": float(
            score_projection["j_primary"]["score_stats"]["mean"]
        ),
        "j_id": float(canonical["j_id"]),
        "j_ood": float(score_projection["j_ood"]["score_stats"]["mean"]),
        "named_guard_scores": {
            name: float(score_projection[name]["score_stats"]["mean"])
            for name in named_guard_names
        },
        "family_scores": dict(canonical["family_scores"]),
        "qualified_ood_families": list(
            score_projection["qualified_ood_families"]
        ),
        "objective_family_mask_hash": score_projection[
            "objective_family_mask_hash"
        ],
        "ordered_suite_hash": suite.suite_hash,
        "split_cohort_hash": suite.cohort_hash("public_dev"),
        "success_rate_by_cell": dict(success),
        "factor_diagnostics": dict(ordered["factor_diagnostics"]),
        "stratum_diagnostics": stratum_diagnostics,
    }
    return feedback


def _build_unified_summary_from_ordered(
    *,
    suite: StairUnifiedEvaluationSuite,
    split: str,
    ordered_summary: Mapping[str, Any],
) -> dict[str, Any]:
    cells = suite.cells_for_cohort(split)
    cells_by_stratum: dict[str, list[str]] = {}
    for cell in cells:
        cells_by_stratum.setdefault(cell.stratum_id, []).append(cell.cell_id)
    minimum = int(
        suite.qualification_rule["minimum_cells_per_stratum_per_cohort"]
    )
    if split != "calibration" and any(
        len(cell_ids) < minimum for cell_ids in cells_by_stratum.values()
    ):
        raise ValueError("unified Stair stratum cell minimum is not met")

    seed_views: list[dict[str, Any]] = []
    for seed in ordered_summary["seed_summaries"]:
        cell_scores = {
            cell_id: 0.5 * float(seed["success_rate_by_cell"][cell_id])
            + 0.5 * float(seed["mean_progress_fraction_by_cell"][cell_id])
            for cell_id in seed["success_rate_by_cell"]
        }
        stratum_scores = {
            stratum: statistics.fmean(cell_scores[cell_id] for cell_id in cell_ids)
            for stratum, cell_ids in cells_by_stratum.items()
        }
        seed_views.append(
            {
                "seed_index": int(seed["seed_index"]),
                "cell_scores": cell_scores,
                "stratum_scores": stratum_scores,
                **_provisional_objectives(
                    stratum_scores=stratum_scores,
                    score_contract=suite.score_contract,
                ),
            }
        )
    aggregate_strata = {
        stratum: statistics.fmean(
            float(seed["stratum_scores"][stratum]) for seed in seed_views
        )
        for stratum in cells_by_stratum
    }
    objective = _provisional_objectives(
        stratum_scores=aggregate_strata,
        score_contract=suite.score_contract,
    )
    return {
        "schema_version": STAIR_UNIFIED_SUMMARY_SCHEMA_VERSION,
        "ordered_summary": dict(ordered_summary),
        "ordered_suite_hash": suite.suite_hash,
        "split": split,
        "aggregation_order": "episode_to_cell_to_stratum_to_paired_seed",
        "cell_score_formula": "0.5*success_rate+0.5*goal_progress_fraction",
        "stratum_cell_ids": cells_by_stratum,
        "aggregate_stratum_scores": aggregate_strata,
        "seed_summaries": seed_views,
        **objective,
    }


def _score_stats(
    values: Sequence[float],
    *,
    eval_protocol_hash: str,
) -> dict[str, Any]:
    canonical = [float(value) for value in values]
    count = len(canonical)
    mean = statistics.fmean(canonical)
    std = statistics.stdev(canonical) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "standard_error": std / (count**0.5),
        "values": canonical,
        "eval_protocol_hash": eval_protocol_hash,
    }


def _mask_projection_identity(
    *,
    summary: Mapping[str, Any],
    mask: Mapping[str, Any],
) -> tuple[list[str], str]:
    if not isinstance(mask, Mapping):
        raise ValueError("objective family mask must be an object")
    observed_hash = mask.get("objective_family_mask_hash")
    payload = dict(mask)
    payload.pop("objective_family_mask_hash", None)
    qualified = mask.get("qualified_ood_families")
    available = summary.get("family_scores")
    if (
        mask.get("schema_version") != STAIR_OBJECTIVE_FAMILY_MASK_SCHEMA_VERSION
        or mask.get("ordered_suite_hash") != summary.get("ordered_suite_hash")
        or not is_sha256_digest(observed_hash)
        or canonical_json_hash(payload) != observed_hash
        or not isinstance(qualified, list)
        or not qualified
        or len(qualified) != len(set(qualified))
        or not isinstance(available, Mapping)
        or any(name not in available for name in qualified)
    ):
        raise ValueError("objective family mask cannot project this summary")
    return [str(name) for name in qualified], str(observed_hash)


def _provisional_objectives(
    *,
    stratum_scores: Mapping[str, float],
    score_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not {"c0_id", "c1_id"} <= set(stratum_scores):
        return {
            "qualification_status": "calibration_only",
            "family_scores": {},
            "j_id": None,
            "j_ood": None,
            "j_primary": None,
            "named_guard_scores": {},
        }
    j_id = 0.5 * float(stratum_scores["c0_id"]) + 0.5 * float(
        stratum_scores["c1_id"]
    )
    family_scores: dict[str, float] = {}
    for family, members in score_contract["families"].items():
        available = [float(stratum_scores[name]) for name in members if name in stratum_scores]
        if available:
            family_scores[str(family)] = statistics.fmean(available)
    declared_families = tuple(str(name) for name in score_contract["families"])
    ood_values = [
        family_scores[family]
        for family in declared_families
        if family in family_scores
    ]
    if not ood_values:
        raise ValueError("unified Stair formal summary has no OOD family")
    j_ood = statistics.fmean(ood_values)
    j_primary = (2.0 * j_ood + j_id) / 3.0
    return {
        "qualification_status": "pre_reference_all_declared_strata",
        "family_scores": family_scores,
        "j_id": j_id,
        "j_ood": j_ood,
        "j_primary": j_primary,
        "named_guard_scores": {
            "j_ood": j_ood,
            "c0_id": float(stratum_scores["c0_id"]),
            "c1_id": float(stratum_scores["c1_id"]),
        },
    }


__all__ = [
    "STAIR_OBJECTIVE_FAMILY_MASK_SCHEMA_VERSION",
    "STAIR_UNIFIED_SUMMARY_SCHEMA_VERSION",
    "build_reference_objective_family_mask",
    "build_unified_score_projection",
    "build_unified_stair_prompt_feedback",
    "build_unified_stair_summary",
    "objective_family_mask_from_public_result",
    "validate_objective_family_mask",
    "validate_unified_stair_summary",
]

"""Pure task-owned aggregation contracts for the frozen stair protocol."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.evaluation.sealed_validation import build_sealed_rollout_evidence
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.tasks.stair_climbing.episode_evidence import (
    StairEpisodeOutcome,
    validate_stair_success_diagnostic_relations,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import StairEvaluationProfile
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION,
    build_stair_generalization_summary,
    validate_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1

STAIR_SPLIT_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.split_summary.v2"
)


def build_stair_evaluation_summary(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics,
    evaluation_protocol: EvaluationProtocol,
    split: str,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
    evaluation_profile: StairEvaluationProfile | None = None,
) -> dict[str, Any]:
    """Build the canonical legacy summary or active anchor/objective pair."""

    if evaluation_profile is None:
        return build_stair_split_summary(
            protocol,
            semantics=semantics,
            evaluation_protocol=evaluation_protocol,
            split=split,
            per_seed_outcomes=per_seed_outcomes,
        )
    anchor_ids = {cell.cell_id for cell in evaluation_profile.anchor_cells}
    anchor_outcomes = [
        [
            outcome
            for outcome in seed_outcomes
            if outcome.evaluation_cell_id in anchor_ids
        ]
        for seed_outcomes in per_seed_outcomes
    ]
    return {
        "anchor": build_stair_split_summary(
            protocol,
            semantics=semantics,
            evaluation_protocol=evaluation_protocol,
            split=split,
            per_seed_outcomes=anchor_outcomes,
        ),
        "generalization": build_stair_generalization_summary(
            profile=evaluation_profile,
            semantics=semantics,
            evaluation_protocol=evaluation_protocol,
            split=split,
            per_seed_outcomes=per_seed_outcomes,
        ),
    }


_LEGACY_STAIR_SPLIT_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.split_summary.v1"
)

_LEGACY_SPLIT_SUMMARY_FIELDS = {
    "schema_version",
    "split",
    "seed_count",
    "bin_count",
    "episodes_per_seed_per_bin",
    "complete_seven_bin_coverage",
    "seed_summaries",
    "seed_macro_scores",
    "score_stats",
    "primary_outcome_counts",
    "primary_failure_breakdown",
    "diagnostic_flag_rates",
    "progress_summary",
    "classification_warning_count",
    "total_episode_count",
}
_SPLIT_SUMMARY_FIELDS = {
    *_LEGACY_SPLIT_SUMMARY_FIELDS,
    "evaluation_semantics_hash",
}
_SEED_SUMMARY_FIELDS = {
    "seed_index",
    "complete_bin_coverage",
    "episode_count",
    "bin_success_rates",
    "macro_success_rate",
}
_SCORE_STATS_FIELDS = {
    "count",
    "mean",
    "std",
    "standard_error",
    "eval_protocol_hash",
}


def aggregate_stair_seed_scores(
    protocol: StairProtocolV1,
    *,
    split: str,
    per_seed_success_counts: Sequence[Mapping[str, int]],
    evaluation_protocol: EvaluationProtocol | None = None,
) -> dict[str, Any]:
    """Apply seven-bin macro averaging, then compute statistics across seeds."""

    payload = protocol.to_dict()
    split_cfg = payload["evaluation"]["splits"].get(split)
    if not isinstance(split_cfg, Mapping):
        raise ValueError(f"unknown stair evaluation split: {split}")
    evaluation_protocol = evaluation_protocol or protocol.evaluation_protocol
    expected_seed_count = _split_seed_count(evaluation_protocol, split)
    if expected_seed_count != int(split_cfg["seed_count"]):
        raise ValueError(f"{split} seed count differs from frozen Stair split")
    if len(per_seed_success_counts) != expected_seed_count:
        raise ValueError(
            f"{split} requires {expected_seed_count} complete seed results"
        )
    episodes_per_bin = int(split_cfg["episodes_per_seed_per_bin"])
    bin_ids = tuple(
        item["bin_id"] for item in payload["terrain"]["evaluation_bins"]
    )
    seed_scores: list[float] = []
    for seed_index, counts in enumerate(per_seed_success_counts):
        if set(counts) != set(bin_ids):
            raise ValueError(
                f"{split} seed {seed_index} must contain every frozen bin exactly once"
            )
        rates: list[float] = []
        for bin_id in bin_ids:
            success_count = counts[bin_id]
            if (
                isinstance(success_count, bool)
                or not isinstance(success_count, int)
                or success_count < 0
                or success_count > episodes_per_bin
            ):
                raise ValueError(
                    f"{split} seed {seed_index} has invalid success count for {bin_id}"
                )
            rates.append(success_count / episodes_per_bin)
        seed_scores.append(statistics.fmean(rates))

    mean = statistics.fmean(seed_scores)
    sample_std = statistics.stdev(seed_scores) if len(seed_scores) > 1 else 0.0
    standard_error = sample_std / math.sqrt(len(seed_scores))
    return {
        "seed_macro_scores": seed_scores,
        "score_stats": {
            "count": len(seed_scores),
            "mean": mean,
            "std": sample_std,
            "standard_error": standard_error,
            "eval_protocol_hash": evaluation_protocol.protocol_hash,
        },
    }


def build_stair_split_summary(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics | None,
    evaluation_protocol: EvaluationProtocol,
    split: str,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> dict[str, Any]:
    """Validate closed evidence and project one public-safe split summary."""

    payload = protocol.to_dict()
    split_cfg = payload["evaluation"]["splits"].get(split)
    if not isinstance(split_cfg, Mapping):
        raise ValueError(f"unknown stair evaluation split: {split}")
    expected_seed_count = _split_seed_count(evaluation_protocol, split)
    if expected_seed_count != int(split_cfg["seed_count"]):
        raise ValueError(f"{split} seed count differs from frozen Stair split")
    if len(per_seed_outcomes) != expected_seed_count:
        raise ValueError(f"{split} requires {expected_seed_count} complete seeds")

    bin_ids = tuple(str(item["bin_id"]) for item in protocol.eval_bins)
    episodes_per_bin = int(split_cfg["episodes_per_seed_per_bin"])
    outcomes_contract = _outcomes_contract(protocol, semantics)
    primary_reasons = tuple(outcomes_contract["primary_reasons"])
    diagnostic_flags = tuple(outcomes_contract["diagnostic_flags"])
    per_seed_success_counts: list[dict[str, int]] = []
    all_outcomes: list[StairEpisodeOutcome] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed_index, raw_outcomes in enumerate(per_seed_outcomes, start=1):
        outcomes = list(raw_outcomes)
        if len(outcomes) != len(bin_ids) * episodes_per_bin:
            raise ValueError(
                f"{split} seed {seed_index} episode count does not cover seven bins"
            )
        env_indexes = [item.env_index for item in outcomes]
        if set(env_indexes) != set(range(len(outcomes))) or len(env_indexes) != len(
            set(env_indexes)
        ):
            raise ValueError(
                f"{split} seed {seed_index} env identity coverage is incomplete"
            )
        counts_by_bin = Counter(item.bin_id for item in outcomes)
        if set(counts_by_bin) != set(bin_ids) or any(
            counts_by_bin[bin_id] != episodes_per_bin for bin_id in bin_ids
        ):
            raise ValueError(
                f"{split} seed {seed_index} bin coverage is incomplete"
            )
        primary_counts = Counter(item.primary_reason for item in outcomes)
        assert_primary_outcome_counts(
            protocol,
            semantics=semantics,
            total_episode_count=len(outcomes),
            primary_counts={
                reason: int(primary_counts.get(reason, 0))
                for reason in primary_reasons
            },
        )
        if any(item.success != (item.primary_reason == "success") for item in outcomes):
            raise ValueError("stair outcome success flag differs from primary reason")
        if any(
            not set(item.diagnostic_flags).issubset(diagnostic_flags)
            for item in outcomes
        ):
            raise ValueError("stair outcome contains an undeclared diagnostic flag")
        validate_stair_success_diagnostic_relations(
            outcomes,
            declared_diagnostic_flags=diagnostic_flags,
        )
        success_counts = {
            bin_id: sum(
                int(item.success) for item in outcomes if item.bin_id == bin_id
            )
            for bin_id in bin_ids
        }
        per_seed_success_counts.append(success_counts)
        all_outcomes.extend(outcomes)
        seed_summaries.append(
            {
                "seed_index": seed_index,
                "complete_bin_coverage": True,
                "episode_count": len(outcomes),
                "bin_success_rates": {
                    bin_id: success_counts[bin_id] / episodes_per_bin
                    for bin_id in bin_ids
                },
                "macro_success_rate": statistics.fmean(
                    success_counts[bin_id] / episodes_per_bin
                    for bin_id in bin_ids
                ),
            }
        )

    scores = aggregate_stair_seed_scores(
        protocol,
        split=split,
        per_seed_success_counts=per_seed_success_counts,
        evaluation_protocol=evaluation_protocol,
    )
    total = len(all_outcomes)
    primary_counts = Counter(item.primary_reason for item in all_outcomes)
    diagnostic_counts = Counter(
        flag for item in all_outcomes for flag in item.diagnostic_flags
    )
    bin_progress = {
        bin_id: statistics.fmean(
            item.max_progress_m for item in all_outcomes if item.bin_id == bin_id
        )
        for bin_id in bin_ids
    }
    return {
        "schema_version": _split_summary_schema_version(semantics),
        **(
            {"evaluation_semantics_hash": semantics.protocol_hash}
            if semantics is not None
            else {}
        ),
        "split": split,
        "seed_count": expected_seed_count,
        "bin_count": len(bin_ids),
        "episodes_per_seed_per_bin": episodes_per_bin,
        "complete_seven_bin_coverage": True,
        "seed_summaries": seed_summaries,
        **scores,
        "primary_outcome_counts": {
            reason: int(primary_counts.get(reason, 0))
            for reason in primary_reasons
        },
        "primary_failure_breakdown": {
            reason: int(primary_counts.get(reason, 0)) / total
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
            "mean_max_progress_m_by_bin": bin_progress,
        },
        "classification_warning_count": 0,
        "total_episode_count": total,
    }


def validate_stair_split_summary(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics | None,
    evaluation_protocol: EvaluationProtocol,
    split: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every public aggregate from one strict split summary."""

    if not isinstance(protocol, StairProtocolV1):
        raise TypeError("stair summary validator requires StairProtocolV1")
    if not isinstance(evaluation_protocol, EvaluationProtocol):
        raise TypeError("stair summary validator requires EvaluationProtocol")
    expected_fields = (
        _SPLIT_SUMMARY_FIELDS
        if semantics is not None
        else _LEGACY_SPLIT_SUMMARY_FIELDS
    )
    if not isinstance(summary, Mapping) or set(summary) != expected_fields:
        raise ValueError("stair split summary fields mismatch")
    if summary.get("schema_version") != _split_summary_schema_version(semantics):
        raise ValueError("stair split summary schema mismatch")
    if (
        semantics is not None
        and summary.get("evaluation_semantics_hash") != semantics.protocol_hash
    ):
        raise ValueError("stair split summary evaluation semantics mismatch")
    if summary.get("split") != split:
        raise ValueError("stair split summary identity mismatch")

    payload = protocol.to_dict()
    split_cfg = payload["evaluation"]["splits"].get(split)
    if not isinstance(split_cfg, Mapping):
        raise ValueError(f"unknown stair evaluation split: {split}")
    seed_count = _split_seed_count(evaluation_protocol, split)
    episodes_per_bin = int(split_cfg["episodes_per_seed_per_bin"])
    bin_ids = tuple(str(item["bin_id"]) for item in protocol.eval_bins)
    episode_count_per_seed = len(bin_ids) * episodes_per_bin
    total_episode_count = seed_count * episode_count_per_seed

    if summary.get("seed_count") != seed_count:
        raise ValueError("stair split summary seed count mismatch")
    if summary.get("bin_count") != len(bin_ids):
        raise ValueError("stair split summary bin count mismatch")
    if summary.get("episodes_per_seed_per_bin") != episodes_per_bin:
        raise ValueError("stair split summary episodes-per-bin mismatch")
    if summary.get("complete_seven_bin_coverage") is not True:
        raise ValueError("stair split summary does not prove seven-bin coverage")
    if summary.get("total_episode_count") != total_episode_count:
        raise ValueError("stair split summary total episode count mismatch")
    if summary.get("classification_warning_count") != 0:
        raise ValueError("stair split summary contains classification warnings")

    raw_seed_summaries = summary.get("seed_summaries")
    if not isinstance(raw_seed_summaries, list) or len(raw_seed_summaries) != seed_count:
        raise ValueError("stair split summary seed summaries are incomplete")
    seed_macro_scores: list[float] = []
    total_success_count = 0
    canonical_seed_summaries: list[dict[str, Any]] = []
    for expected_index, raw_seed in enumerate(raw_seed_summaries, start=1):
        if not isinstance(raw_seed, Mapping) or set(raw_seed) != _SEED_SUMMARY_FIELDS:
            raise ValueError("stair seed summary fields mismatch")
        if raw_seed.get("seed_index") != expected_index:
            raise ValueError("stair seed summary index mismatch")
        if raw_seed.get("complete_bin_coverage") is not True:
            raise ValueError("stair seed summary bin coverage is incomplete")
        if raw_seed.get("episode_count") != episode_count_per_seed:
            raise ValueError("stair seed summary episode count mismatch")
        raw_rates = raw_seed.get("bin_success_rates")
        if not isinstance(raw_rates, Mapping) or set(raw_rates) != set(bin_ids):
            raise ValueError("stair seed summary bin success rates mismatch")
        canonical_rates: dict[str, float] = {}
        for bin_id in bin_ids:
            rate = _bounded_rate(raw_rates[bin_id], f"{split}.{bin_id} success rate")
            success_count = rate * episodes_per_bin
            rounded_count = round(success_count)
            if not math.isclose(
                success_count,
                rounded_count,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("stair bin success rate is not episode-count derived")
            canonical_rates[bin_id] = rate
            total_success_count += int(rounded_count)
        macro = statistics.fmean(canonical_rates.values())
        declared_macro = _finite_number(
            raw_seed.get("macro_success_rate"),
            "stair seed macro success rate",
        )
        _require_close(declared_macro, macro, "stair seed macro success rate")
        seed_macro_scores.append(macro)
        canonical_seed_summaries.append(
            {
                "seed_index": expected_index,
                "complete_bin_coverage": True,
                "episode_count": episode_count_per_seed,
                "bin_success_rates": canonical_rates,
                "macro_success_rate": macro,
            }
        )

    raw_seed_scores = summary.get("seed_macro_scores")
    if not isinstance(raw_seed_scores, list) or len(raw_seed_scores) != seed_count:
        raise ValueError("stair split summary seed macro scores are incomplete")
    for declared, expected in zip(raw_seed_scores, seed_macro_scores):
        _require_close(
            _finite_number(declared, "stair seed macro score"),
            expected,
            "stair seed macro score",
        )

    expected_mean = statistics.fmean(seed_macro_scores)
    expected_std = statistics.stdev(seed_macro_scores) if seed_count > 1 else 0.0
    expected_standard_error = expected_std / math.sqrt(seed_count)
    raw_stats = summary.get("score_stats")
    if not isinstance(raw_stats, Mapping) or set(raw_stats) != _SCORE_STATS_FIELDS:
        raise ValueError("stair split summary score_stats fields mismatch")
    stats = validate_score_stats(
        raw_stats,
        minimum_count=seed_count,
        require_std=True,
        require_standard_error=True,
        require_protocol_hash=True,
    )
    if not stats.valid or stats.count != seed_count:
        raise ValueError("stair split summary score_stats are invalid")
    if stats.eval_protocol_hash != evaluation_protocol.protocol_hash:
        raise ValueError("stair split summary evaluation protocol hash mismatch")
    _require_close(stats.mean, expected_mean, "stair score mean")
    _require_close(stats.std, expected_std, "stair score sample std")
    _require_close(
        stats.standard_error,
        expected_standard_error,
        "stair score standard error",
    )

    outcomes_contract = _outcomes_contract(protocol, semantics)
    primary_reasons = tuple(outcomes_contract["primary_reasons"])
    raw_primary = summary.get("primary_outcome_counts")
    if not isinstance(raw_primary, Mapping) or set(raw_primary) != set(primary_reasons):
        raise ValueError("stair primary outcome fields mismatch")
    primary_counts = {
        reason: _nonnegative_int(raw_primary[reason], f"stair primary count {reason}")
        for reason in primary_reasons
    }
    assert_primary_outcome_counts(
        protocol,
        semantics=semantics,
        total_episode_count=total_episode_count,
        primary_counts=primary_counts,
    )
    if primary_counts["success"] != total_success_count:
        raise ValueError("stair success counts differ from seven-bin rates")

    failure_reasons = tuple(reason for reason in primary_reasons if reason != "success")
    raw_failures = summary.get("primary_failure_breakdown")
    if not isinstance(raw_failures, Mapping) or set(raw_failures) != set(failure_reasons):
        raise ValueError("stair primary failure breakdown fields mismatch")
    failure_breakdown: dict[str, float] = {}
    for reason in failure_reasons:
        rate = _bounded_rate(raw_failures[reason], f"stair failure rate {reason}")
        _require_close(
            rate,
            primary_counts[reason] / total_episode_count,
            f"stair failure rate {reason}",
        )
        failure_breakdown[reason] = rate

    diagnostic_flags = tuple(outcomes_contract["diagnostic_flags"])
    raw_diagnostics = summary.get("diagnostic_flag_rates")
    if not isinstance(raw_diagnostics, Mapping) or set(raw_diagnostics) != set(
        diagnostic_flags
    ):
        raise ValueError("stair diagnostic flag rate fields mismatch")
    diagnostic_rates: dict[str, float] = {}
    for flag in diagnostic_flags:
        rate = _bounded_rate(raw_diagnostics[flag], f"stair diagnostic rate {flag}")
        diagnostic_count = rate * total_episode_count
        if not math.isclose(
            diagnostic_count,
            round(diagnostic_count),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("stair diagnostic rate is not episode-count derived")
        diagnostic_rates[flag] = rate

    raw_progress = summary.get("progress_summary")
    if not isinstance(raw_progress, Mapping) or set(raw_progress) != {
        "mean_max_progress_m",
        "mean_max_progress_m_by_bin",
    }:
        raise ValueError("stair progress summary fields mismatch")
    raw_progress_by_bin = raw_progress.get("mean_max_progress_m_by_bin")
    if not isinstance(raw_progress_by_bin, Mapping) or set(raw_progress_by_bin) != set(
        bin_ids
    ):
        raise ValueError("stair progress-by-bin fields mismatch")
    progress_by_bin = {
        bin_id: _nonnegative_number(
            raw_progress_by_bin[bin_id],
            f"stair progress {bin_id}",
        )
        for bin_id in bin_ids
    }
    mean_progress = _nonnegative_number(
        raw_progress.get("mean_max_progress_m"),
        "stair mean max progress",
    )
    _require_close(
        mean_progress,
        statistics.fmean(progress_by_bin.values()),
        "stair mean max progress",
    )

    return {
        "schema_version": _split_summary_schema_version(semantics),
        **(
            {"evaluation_semantics_hash": semantics.protocol_hash}
            if semantics is not None
            else {}
        ),
        "split": split,
        "seed_count": seed_count,
        "bin_count": len(bin_ids),
        "episodes_per_seed_per_bin": episodes_per_bin,
        "complete_seven_bin_coverage": True,
        "seed_summaries": canonical_seed_summaries,
        "seed_macro_scores": seed_macro_scores,
        "score_stats": {
            "count": seed_count,
            "mean": expected_mean,
            "std": expected_std,
            "standard_error": expected_standard_error,
            "eval_protocol_hash": evaluation_protocol.protocol_hash,
        },
        "primary_outcome_counts": primary_counts,
        "primary_failure_breakdown": failure_breakdown,
        "diagnostic_flag_rates": diagnostic_rates,
        "progress_summary": {
            "mean_max_progress_m": mean_progress,
            "mean_max_progress_m_by_bin": progress_by_bin,
        },
        "classification_warning_count": 0,
        "total_episode_count": total_episode_count,
    }


def assert_primary_outcome_counts(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics | None,
    total_episode_count: int,
    primary_counts: Mapping[str, int],
    classification_warning_count: int = 0,
) -> None:
    """Enforce one and only one primary terminal reason per episode."""

    if (
        isinstance(total_episode_count, bool)
        or not isinstance(total_episode_count, int)
        or total_episode_count <= 0
    ):
        raise ValueError("total_episode_count must be a positive integer")
    if (
        isinstance(classification_warning_count, bool)
        or not isinstance(classification_warning_count, int)
        or classification_warning_count < 0
    ):
        raise ValueError("classification_warning_count must be non-negative")
    reasons = tuple(_outcomes_contract(protocol, semantics)["primary_reasons"])
    if set(primary_counts) != set(reasons):
        raise ValueError("primary_counts must contain every primary reason exactly once")
    for reason in reasons:
        count = primary_counts[reason]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"primary count must be non-negative: {reason}")
    if sum(primary_counts.values()) != total_episode_count:
        raise ValueError("primary outcome count invariant failed")
    if classification_warning_count:
        raise ValueError("classification warnings block stair evaluation pass")


def build_stair_sealed_rollout_evidence(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics,
    evaluation_protocol: EvaluationProtocol,
    summary: Mapping[str, Any],
    checkpoint_hash: str,
    evaluation_binding_hash: str,
    evaluation_subject: ExperimentSubject,
    subject_manifest_hash: str,
    program_source_checksum: str,
    training_binding_hash: str,
    capability_profile_hash: str,
    acceptance_protocol: AcceptanceProtocol,
    evaluation_profile: StairEvaluationProfile | None = None,
    public_objective_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind validated sealed Stair aggregates without exposing task details."""

    if not isinstance(acceptance_protocol, AcceptanceProtocol):
        raise TypeError("Stair sealed evidence requires active acceptance protocol")
    canonical = canonical_stair_sealed_summary(
        protocol,
        semantics=semantics,
        evaluation_protocol=evaluation_protocol,
        summary=summary,
        evaluation_profile=evaluation_profile,
    )
    if (
        evaluation_profile is not None
        and canonical.get("schema_version")
        != STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION
    ):
        raise ValueError(
            "new Stair generalization sealed evidence requires current summary schema"
        )
    score_stats = dict(canonical["score_stats"])
    public_mean: float | None = None
    if evaluation_profile is not None:
        if public_objective_summary is None:
            raise ValueError(
                "Stair generalization sealed evidence requires public objective summary"
            )
        public_objective = validate_stair_generalization_summary(
            profile=evaluation_profile,
            semantics=semantics,
            evaluation_protocol=evaluation_protocol,
            split="public_accept",
            summary=public_objective_summary,
        )
        public_mean = float(public_objective["score_stats"]["mean"])
    elif public_objective_summary is not None:
        raise ValueError(
            "legacy Stair sealed evidence cannot accept a generalization objective"
        )
    passed, reason = evaluate_stair_sealed_score_gate(
        sealed_mean=float(score_stats["mean"]),
        public_mean=public_mean,
        acceptance_protocol=acceptance_protocol,
        evaluation_profile=evaluation_profile,
    )
    return build_sealed_rollout_evidence(
        score_stats=score_stats,
        expected_seed_count=evaluation_protocol.sealed_validation_seed_count,
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
        evaluation_subject=evaluation_subject,
        checkpoint_hash=checkpoint_hash,
        task_validation_status="pass" if passed else "fail",
        reason=reason,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        task_evidence_hash=canonical_json_hash(canonical),
    )


def canonical_stair_sealed_summary(
    protocol: StairProtocolV1,
    *,
    semantics: StairEvaluationSemantics,
    evaluation_protocol: EvaluationProtocol,
    summary: Mapping[str, Any],
    evaluation_profile: StairEvaluationProfile | None = None,
) -> dict[str, Any]:
    """Return the validated canonical summary persisted for sealed replay."""

    if evaluation_profile is not None:
        return validate_stair_generalization_summary(
            profile=evaluation_profile,
            semantics=semantics,
            evaluation_protocol=evaluation_protocol,
            split="sealed_validation",
            summary=summary,
        )
    return validate_stair_split_summary(
        protocol,
        semantics=semantics,
        evaluation_protocol=evaluation_protocol,
        split="sealed_validation",
        summary=summary,
    )


def evaluate_stair_sealed_score_gate(
    *,
    sealed_mean: float,
    public_mean: float | None,
    acceptance_protocol: AcceptanceProtocol,
    evaluation_profile: StairEvaluationProfile | None,
) -> tuple[bool, str]:
    """Return the single task-owned verdict for Stair sealed aggregate scores."""

    if not isinstance(acceptance_protocol, AcceptanceProtocol):
        raise TypeError("Stair sealed score gate requires active acceptance protocol")
    sealed_score = _finite_number(sealed_mean, "Stair sealed score mean")
    floor_met = sealed_score >= acceptance_protocol.absolute_accept_floor
    public_consistency_met = True
    if evaluation_profile is not None:
        if public_mean is None:
            raise ValueError(
                "Stair generalization sealed score gate requires public score mean"
            )
        public_score = _finite_number(public_mean, "Stair public score mean")
        max_score_drop = acceptance_protocol.max_public_to_sealed_score_drop
        if max_score_drop is None:
            raise ValueError(
                "Stair generalization sealed score gate requires an acceptance-owned "
                "public-to-sealed tolerance"
            )
        public_consistency_met = sealed_score >= (
            public_score - max_score_drop
        )
    elif public_mean is not None:
        raise ValueError("legacy Stair sealed score gate cannot accept a public score")

    reason_prefix = (
        "stair_generalization_holdout"
        if evaluation_profile is not None
        else "stair_holdout"
    )
    if not floor_met:
        reason = f"{reason_prefix}_floor_not_met"
    elif not public_consistency_met:
        reason = f"{reason_prefix}_public_score_regression"
    elif evaluation_profile is not None:
        reason = f"{reason_prefix}_public_consistency_met"
    else:
        reason = f"{reason_prefix}_floor_met"
    return floor_met and public_consistency_met, reason


def evaluate_unified_stair_sealed_score_gate(
    *,
    sealed_mean: float,
    public_mean: float,
    acceptance_protocol: AcceptanceProtocol,
    evaluation_subject: ExperimentSubject,
) -> tuple[bool, str]:
    """Apply reference telemetry or candidate acceptance semantics by subject."""

    if not isinstance(acceptance_protocol, AcceptanceProtocol):
        raise TypeError("unified Stair sealed score gate requires active acceptance protocol")
    if not isinstance(evaluation_subject, ExperimentSubject):
        raise TypeError("unified Stair sealed score gate requires an experiment subject")
    sealed_score = _finite_number(sealed_mean, "unified Stair sealed score mean")
    public_score = _finite_number(public_mean, "unified Stair public score mean")
    if evaluation_subject.kind == "reference":
        return True, "unified_stair_reference_sealed_telemetry_complete"

    max_drop = acceptance_protocol.max_public_to_sealed_score_drop
    if max_drop is None:
        raise ValueError("unified Stair acceptance lacks sealed consistency tolerance")
    floor_met = sealed_score >= acceptance_protocol.absolute_accept_floor
    consistency_met = sealed_score >= public_score - max_drop
    if not floor_met:
        return False, "unified_stair_sealed_absolute_floor_failed"
    if not consistency_met:
        return False, "unified_stair_sealed_public_consistency_failed"
    return True, "unified_stair_sealed_primary_complete"


def _outcomes_contract(
    protocol: StairProtocolV1,
    semantics: StairEvaluationSemantics | None,
) -> Mapping[str, Any]:
    # None is retained only for replaying pre-migration V1/V2 artifacts.
    if semantics is None:
        return protocol.to_dict()["episode_outcomes"]
    return semantics.to_dict()["episode_outcomes"]


def _split_summary_schema_version(
    semantics: StairEvaluationSemantics | None,
) -> str:
    if semantics is None:
        return _LEGACY_STAIR_SPLIT_SUMMARY_SCHEMA_VERSION
    return STAIR_SPLIT_SUMMARY_SCHEMA_VERSION


def _split_seed_count(protocol: EvaluationProtocol, split: str) -> int:
    try:
        return {
            "public_dev": protocol.public_seed_count,
            "public_accept": protocol.public_accept_seed_count,
            "sealed_validation": protocol.sealed_validation_seed_count,
        }[split]
    except KeyError as exc:
        raise ValueError(f"unknown stair evaluation split: {split}") from exc


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _nonnegative_number(value: Any, field_name: str) -> float:
    parsed = _finite_number(value, field_name)
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _bounded_rate(value: Any, field_name: str) -> float:
    parsed = _finite_number(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_close(actual: float | None, expected: float, field_name: str) -> None:
    if actual is None or not math.isclose(
        actual,
        expected,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{field_name} does not match recomputed evidence")


__all__ = [
    "STAIR_SPLIT_SUMMARY_SCHEMA_VERSION",
    "aggregate_stair_seed_scores",
    "assert_primary_outcome_counts",
    "build_stair_evaluation_summary",
    "build_stair_sealed_rollout_evidence",
    "build_stair_split_summary",
    "canonical_stair_sealed_summary",
    "evaluate_stair_sealed_score_gate",
    "evaluate_unified_stair_sealed_score_gate",
    "validate_stair_split_summary",
]

"""Shared structural validation for public score-stat summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.evaluation.derived_statistics import DERIVED_STATISTICS_VERSION


@dataclass(frozen=True)
class ScoreStatsValidation:
    valid: bool
    reasons: tuple[str, ...]
    count: int | None = None
    mean: float | None = None
    std: float | None = None
    standard_error: float | None = None
    eval_protocol_hash: str | None = None
    canonical: Mapping[str, Any] = field(default_factory=dict)


def validate_score_stats(
    value: Any,
    *,
    minimum_count: int,
    require_std: bool,
    require_standard_error: bool,
    require_protocol_hash: bool,
) -> ScoreStatsValidation:
    if not isinstance(value, Mapping) or not value:
        return ScoreStatsValidation(False, ("score_stats_missing",))
    reasons: list[str] = []
    version = value.get("statistics_version")
    if version not in {None, DERIVED_STATISTICS_VERSION}:
        reasons.append("score_stats_version_invalid")
    count = _positive_int(value.get("count"))
    if count is None:
        reasons.append("score_stats_count_invalid")
    elif count < max(1, minimum_count):
        reasons.append("score_stats_count_insufficient")
    mean = _finite_number(value.get("mean"))
    if mean is None:
        reasons.append("score_stats_mean_invalid")
    std = _nonnegative_number(value.get("std"))
    if require_std and std is None:
        reasons.append("score_stats_std_invalid")
    standard_error = _nonnegative_number(value.get("standard_error"))
    if require_standard_error and standard_error is None:
        reasons.append("score_stats_standard_error_invalid")
    protocol_hash = value.get("eval_protocol_hash")
    if not isinstance(protocol_hash, str) or not protocol_hash:
        protocol_hash = None
        if require_protocol_hash:
            reasons.append("score_stats_eval_protocol_hash_missing")
    canonical: dict[str, Any] = {}
    if not reasons:
        canonical = {"count": count, "mean": mean}
        if version is not None:
            canonical["statistics_version"] = version
        if std is not None:
            canonical["std"] = std
        if standard_error is not None:
            canonical["standard_error"] = standard_error
        if protocol_hash is not None:
            canonical["eval_protocol_hash"] = protocol_hash
    deduped = tuple(dict.fromkeys(reasons))
    return ScoreStatsValidation(
        valid=not deduped,
        reasons=deduped,
        count=count,
        mean=mean,
        std=std,
        standard_error=standard_error,
        eval_protocol_hash=protocol_hash,
        canonical=canonical,
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _nonnegative_number(value: Any) -> float | None:
    parsed = _finite_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


__all__ = ["ScoreStatsValidation", "validate_score_stats"]

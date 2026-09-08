"""Derived-statistics v1: exact binary inputs, fixed decimal square root.

Raw evidence is never rounded or rehashed through this module. Callers select
this contract through their versioned summary schema; legacy readers keep the
arithmetic declared by their original format.
"""

from __future__ import annotations

import math
import statistics
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from typing import Iterable

DERIVED_STATISTICS_VERSION = "true_rl_agent.derived_statistics.v1"


def sample_mean(
    values: Iterable[float],
    *,
    statistics_version: str | None = DERIVED_STATISTICS_VERSION,
) -> float:
    """Round an exact mean once; None preserves historical fmean arithmetic."""
    samples = tuple(float(value) for value in values)
    if statistics_version is None:
        return statistics.fmean(samples)
    if statistics_version != DERIVED_STATISTICS_VERSION:
        raise ValueError("unsupported derived statistics version")
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("derived statistics require nonempty finite samples")
    return float(sum(Fraction(value) for value in samples) / len(samples))


def sample_statistics(
    values: Iterable[float],
    *,
    statistics_version: str | None = DERIVED_STATISTICS_VERSION,
) -> dict[str, int | float | str]:
    samples = tuple(float(value) for value in values)
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("derived statistics require nonempty finite samples")
    count = len(samples)
    if statistics_version is None:
        std = statistics.stdev(samples) if count > 1 else 0.0
        return {
            "count": count,
            "mean": statistics.fmean(samples),
            "std": std,
            "standard_error": std / math.sqrt(count),
        }
    if statistics_version != DERIVED_STATISTICS_VERSION:
        raise ValueError("unsupported derived statistics version")
    exact = tuple(Fraction(value) for value in samples)
    mean = sum(exact) / count
    variance = (
        sum((value - mean) ** 2 for value in exact) / (count - 1)
        if count > 1
        else Fraction(0)
    )
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        decimal_variance = Decimal(variance.numerator) / Decimal(variance.denominator)
        std = float(decimal_variance.sqrt())
        error = float((decimal_variance / count).sqrt())
    return {
        "count": count,
        "mean": float(mean),
        "std": std,
        "standard_error": error,
        "statistics_version": DERIVED_STATISTICS_VERSION,
    }


def paired_standard_error(
    values: Iterable[float],
    *,
    statistics_version: str | None = DERIVED_STATISTICS_VERSION,
) -> float:
    return float(
        sample_statistics(values, statistics_version=statistics_version)[
            "standard_error"
        ]
    )

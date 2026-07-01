"""Small statistics helpers for Phase 6 public-only comparisons."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any


def stats(values: list[float]) -> dict[str, Any]:
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


def pooled_standard_error(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    cand_se = candidate.get("standard_error")
    base_se = baseline.get("standard_error")
    if cand_se is None or base_se is None:
        return None
    return math.sqrt(float(cand_se) ** 2 + float(base_se) ** 2)

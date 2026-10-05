"""Qualified periods of current contact-marker readouts, in seconds per foot."""
from __future__ import annotations

import numpy as np

from .contracts import NumericEpisode

CYCLE_CONFIG = {
    "complete_cycles_per_window": 4,
    "population_cv_max": 0.15, "median_relative_deviation_max": 0.20,
    "debounce_max_run_samples": [1, 2], "boundary_context_samples": 3,
    "roundoff_tolerance": 1e-12,
}


def _runs(values: np.ndarray, steps: np.ndarray) -> list[tuple[int, int]]:
    cuts = np.flatnonzero((values[1:] != values[:-1]) | (np.diff(steps) != 1)) + 1
    return list(zip([0, *cuts.tolist()], [*cuts.tolist(), len(values)]))


def _onsets(values: np.ndarray, steps: np.ndarray) -> list[int]:
    return (np.flatnonzero((values[:-1] == 0) & (values[1:] == 1) & (np.diff(steps) == 1)) + 1).tolist()


def _foot(values: np.ndarray, episode: NumericEpisode) -> list[float]:
    steps = episode.steps
    runs = _runs(values, steps)
    variants = []
    for width in CYCLE_CONFIG["debounce_max_run_samples"]:
        changed = values.copy()
        for a, b in runs:
            if (0 < a and b < len(values) and b - a <= width and values[a] >= 0
                    and values[a - 1] == values[b] == 1 - values[a]
                    and steps[a] == steps[a - 1] + 1 and steps[b] == steps[b - 1] + 1):
                changed[a:b] = values[a - 1]
        variants.append(changed)
    periods_s = []
    for lo, hi in _runs((values >= 0).astype(np.int8), steps):
        if values[lo] < 0:
            continue
        indices = [lo + k for k in _onsets(values[lo:hi], steps[lo:hi])]
        qualified = set()
        count = CYCLE_CONFIG["complete_cycles_per_window"]
        for k in range(len(indices) - count):
            idx = indices[k:k + count + 1]
            a, b = idx[0], idx[-1]
            periods = np.diff(steps[idx]).astype(np.float64)
            cv = float(np.std(periods) / np.mean(periods))
            median = float(np.median(periods))
            deviation = float(np.max(np.abs(periods - median)) / median)
            tol = CYCLE_CONFIG["roundoff_tolerance"]
            if (cv > CYCLE_CONFIG["population_cv_max"] + tol
                    or deviation > CYCLE_CONFIG["median_relative_deviation_max"] + tol):
                continue
            context = CYCLE_CONFIG["boundary_context_samples"]
            if not (lo <= a-context and b+context <= hi
                    and np.all(values[a-context:a] == 0) and np.all(values[b:b+context] == 1)):
                continue
            if any(np.any(variant[a-context:b+context] != values[a-context:b+context]) for variant in variants):
                continue
            qualified.update(range(k, k + count))
        # Overlapping qualified windows contribute each raw cycle only once.
        periods_s.extend(int(steps[indices[k+1]] - steps[indices[k]]) * episode.dt_s for k in sorted(qualified))
    return periods_s


def measure_contact_cycles(episode: NumericEpisode) -> dict[str, list[float]]:
    """Return unique qualified inter-onset periods; no physical gait inference."""
    threshold = float(episode.task["contact_threshold_n"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("contact threshold must be finite and nonnegative")
    entities = episode.entities["support"]
    forces = np.asarray(episode.arrays.get("support_contact_forces_w",
                        np.full((len(episode.steps), len(entities), 3), np.nan)), dtype=np.float64)
    if forces.shape != (len(episode.steps), len(entities), 3):
        raise ValueError("cycle force dimensions differ from source entity binding")
    markers = np.where(np.isfinite(forces).all(axis=2), (np.linalg.norm(forces, axis=2) > threshold).astype(np.int8), -1)
    return {entity: _foot(markers[:, i], episode) for i, entity in enumerate(entities)}

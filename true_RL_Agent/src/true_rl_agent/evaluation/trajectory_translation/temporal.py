"""Readout-sequence intervals and relations, without interpolation or debounce.

    No interval here establishes continuous physical contact. History buffer
    slots never add observations to the control clock.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .contracts import NumericEpisode, PredicateTrack


def contiguous_slices(steps: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open, valid, consecutive-step islands."""
    result: list[tuple[int, int]] = []
    first = None
    for i in range(len(steps)):
        if first is not None and (not valid[i] or steps[i] != steps[i - 1] + 1):
            result.append((first, i))
            first = None
        if valid[i] and first is None:
            first = i
    if first is not None:
        result.append((first, len(steps)))
    return result


def interval_rows(episode: NumericEpisode, track: PredicateTrack) -> list[dict[str, Any]]:
    """Keep all constant runs, including false and unknown, and one-sample runs."""
    values, steps, times = track.values, episode.steps, episode.times
    if len(values) != len(steps) or track.episode_id != episode.episode_id:
        raise ValueError("predicate/episode identity or length mismatch")
    if track.config.get("not_applicable_count", 0):
        values = values.copy()
        values[:track.config["not_applicable_count"]] = -2
    cuts = [0, *(np.flatnonzero((np.diff(values) != 0) | ~episode.adjacent) + 1).tolist(), len(values)]
    rows = []
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        state = int(values[start])
        left = "observed_transition"
        right = "observed_transition"
        if start == 0:
            left = "observation_start"
        elif steps[start] != steps[start - 1] + 1:
            left = "step_gap"
        elif values[start - 1] == -2 or state == -2:
            left = "insufficient_initial_history"
        elif values[start - 1] == -1 or state == -1:
            left = "unknown_sample"
        if stop == len(steps):
            right = "observation_end"
        elif steps[stop] != steps[stop - 1] + 1:
            right = "step_gap"
        elif values[stop] == -2 or state == -2:
            right = "insufficient_initial_history"
        elif values[stop] == -1 or state == -1:
            right = "unknown_sample"
        rows.append({"index_half_open": [start, stop],
                     "steps_half_open": [int(steps[start]), int(steps[stop - 1]) + 1],
                     "state": {-2: "not_applicable", -1: "unknown", 0: "false", 1: "true"}[state],
                     "sample_count": stop - start,
                     "sample_cell_duration_proxy_s": (stop - start) * episode.dt_s,
                     "observed_span_s": float(times[stop - 1] - times[start]),
                     "first_observed_s": float(times[start]), "last_observed_s": float(times[stop - 1]),
                     "confirmation_time_s": float(times[start]) if state >= 0 else None,
                     "onset_range_s": [float(times[start - 1]), float(times[start])] if left == "observed_transition" else None,
                     "end_range_s": [float(times[stop - 1]), float(times[stop])] if right == "observed_transition" else None,
                     "left_censored": left != "observed_transition", "right_censored": right != "observed_transition",
                     "left_boundary": left, "right_boundary": right,
                     "complete_run": left == right == "observed_transition",
                     "physical_continuity": "unknown", "boundary_clock": "recorder_readout"})
    return rows

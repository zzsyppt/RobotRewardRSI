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
        elif values[start - 1] == -1 or state == -1:
            left = "unknown_sample"
        if stop == len(steps):
            right = "observation_end"
        elif steps[stop] != steps[stop - 1] + 1:
            right = "step_gap"
        elif values[stop] == -1 or state == -1:
            right = "unknown_sample"
        rows.append({"index_half_open": [start, stop],
                     "steps_half_open": [int(steps[start]), int(steps[stop - 1]) + 1],
                     "state": {-1: "unknown", 0: "false", 1: "true"}[state],
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


def compose_track(episode: NumericEpisode, track: PredicateTrack) -> list[dict[str, Any]]:
    records = []
    lookback = int(track.config.get("lookback_samples", 0))
    for row in interval_rows(episode, track):
        start, stop = row["index_half_open"]
        # Include both bracketing operands when available, without inferring a
        # transition across missing samples or step gaps.
        records.append(episode.record(track.key, "state_interval", track.entity_ids,
                       track.channels, row, unit="s", frame="recorder_readout",
                       config=track.config, start=max(0, start - 1 - lookback), stop=min(len(episode.steps), stop + 1)))
    for i in range(1, len(track.values)):
        before, after = int(track.values[i - 1]), int(track.values[i])
        if episode.adjacent[i - 1] and before >= 0 and after >= 0 and before != after:
            records.append(episode.record(track.key + ".transition", "event", track.entity_ids,
                track.channels, {"before": bool(before), "after": bool(after), "step": int(episode.steps[i]),
                                 "readout_range_s": episode.times[i - 1:i + 1].tolist(),
                                 "confirmation_time_s": float(episode.times[i]),
                                 "physical_event_range_s": None},
                unit="s", frame="recorder_readout", config=track.config, start=max(0, i - 1 - lookback), stop=i + 1))
    return records


def not_observed(episode: NumericEpisode, track: PredicateTrack, start: int, stop: int) -> str:
    """No true record within [start, stop); a single true is decisive counterevidence."""
    if not 0 <= start < stop <= len(episode.steps):
        raise ValueError("query interval outside episode")
    if track.episode_id != episode.episode_id or len(track.values) != len(episode.steps):
        raise ValueError("predicate/episode identity or length mismatch")
    values = track.values[start:stop]
    if np.any(values == 1):
        return "false"
    if np.any(values == -1) or not episode.adjacent[start:stop - 1].all():
        return "unknown"
    return "true"


def conjoin_same_entity(left: PredicateTrack, right: PredicateTrack, key: str) -> PredicateTrack:
    if (left.entity_ids != right.entity_ids or left.episode_id != right.episode_id
            or len(left.values) != len(right.values)):
        raise ValueError("same-entity conjunction requires identical entities and clock")
    values = np.where((left.values == 0) | (right.values == 0), 0,
                      np.where((left.values == 1) & (right.values == 1), 1, -1))
    return PredicateTrack(left.episode_id, key, left.entity_ids, tuple(dict.fromkeys((*left.channels, *right.channels))),
                          values, {"operation": "three_valued_and", "operands": [left.config, right.config],
                                   "lookback_samples": max(left.config.get("lookback_samples", 0),
                                                           right.config.get("lookback_samples", 0))})


def before_ranges(left: list[float] | None, right: list[float] | None) -> str:
    """Strict order of readout brackets; overlapping boundaries are unknown."""
    if left is None or right is None:
        return "unknown"
    if left[1] < right[0]:
        return "true"
    if left[0] > right[1]:
        return "false"
    return "unknown"


def relate_states(episode: NumericEpisode, left: dict[str, Any], right: dict[str, Any],
                  relation: str) -> dict[str, Any]:
    """A bounded relation between two true, source-bound readout intervals."""
    for record in (left, right):
        if (record["episode_id"] != episode.episode_id or record["record_type"] != "state_interval"
                or record["value"]["state"] != "true"):
            raise ValueError("relations require true intervals in the same episode")
    a, b = left["value"], right["value"]
    if relation == "onset_before":
        status = before_ranges(a["onset_range_s"], b["onset_range_s"])
    elif relation == "overlap":
        status = "true" if max(a["first_observed_s"], b["first_observed_s"]) <= min(a["last_observed_s"], b["last_observed_s"]) else "false"
    elif relation == "during_observed_records":
        status = "true" if (b["first_observed_s"] <= a["first_observed_s"] and a["last_observed_s"] <= b["last_observed_s"]) else "false"
    else:
        raise ValueError("unsupported state relation")
    return episode.record(left["key"] + "." + relation + "." + right["key"], "temporal_relation",
        tuple(dict.fromkeys((*left["entity_ids"], *right["entity_ids"]))),
        tuple(dict.fromkeys((*left["evidence_ref"]["channels"], *right["evidence_ref"]["channels"]))),
        {"relation": relation, "status": status, "operand_record_ids": [left["record_id"], right["record_id"]],
         "physical_relation": "unknown"}, unit="bool", frame="recorder_readout",
        config={"operator_version": "readout_relation.v1"},
        start=min(a["index_half_open"][0], b["index_half_open"][0]),
        stop=max(a["index_half_open"][1], b["index_half_open"][1]))


def pair_onsets(episode: NumericEpisode, left: PredicateTrack,
                right: PredicateTrack) -> list[dict[str, Any]]:
    """Every left onset takes nearest right onset, ties earlier; never across gaps.

    The pairing is directional, permits reuse, and has no invented latency cutoff.
    Every search cites its complete joint-valid island, adjacent boundary rows,
    and operand lookback. Unpaired references retain the same negative-search
    evidence; absent joint coverage is established at the reference edge itself.
    """
    if (left.episode_id != episode.episode_id or right.episode_id != episode.episode_id
            or len(left.values) != len(right.values) or len(left.values) != len(episode.steps)):
        raise ValueError("onset tracks must share the episode clock")
    def onsets(track):
        return np.flatnonzero((track.values[:-1] == 0) & (track.values[1:] == 1) & episode.adjacent) + 1
    lhs, rhs = onsets(left), onsets(right)
    islands = contiguous_slices(episode.steps, (left.values >= 0) & (right.values >= 0))
    lookback = int(max(left.config.get("lookback_samples", 0), right.config.get("lookback_samples", 0)))
    pairs = []
    for i in lhs:
        island = next(((a, b) for a, b in islands if a < i < b), None)
        candidates = [] if island is None else [int(j) for j in rhs if island[0] < j < island[1]]
        j = min(candidates, key=lambda j: (abs(int(episode.steps[j]) - int(episode.steps[i])), j)) if candidates else None
        lr = episode.times[i - 1:i + 1].tolist()
        rr = episode.times[j - 1:j + 1].tolist() if j is not None else None
        if island is None:
            # One of the two reference-edge rows lacks joint coverage. Those
            # operands prove that no candidate search can be made for this edge.
            start, stop = int(i) - 1, int(i) + 1
            search_island = None
        else:
            a, b = island
            start, stop = a - 1, min(len(episode.steps), b + 1)
            search_island = {"index_half_open": [a, b],
                             "steps_half_open": [int(episode.steps[a]), int(episode.steps[b - 1]) + 1]}
        pairs.append(episode.record(left.key + ".nearest_onset." + right.key, "temporal_relation",
            (*left.entity_ids, *right.entity_ids), tuple(dict.fromkeys((*left.channels, *right.channels))),
            {"relation": "nearest_readout_onset", "reference_step": int(episode.steps[i]),
             "target_step": int(episode.steps[j]) if j is not None else None,
             "status": "paired" if j is not None else "unpaired_no_target_in_joint_coverage",
             "search_island": search_island, "candidate_count": len(candidates),
             "delta_confirmation_s": float(episode.times[j] - episode.times[i]) if j is not None else None,
             "delta_range_s": [rr[0] - lr[1], rr[1] - lr[0]] if rr is not None else None,
             "before": before_ranges(lr, rr), "physical_order": "unknown"},
            unit="s", frame="recorder_readout", config={"pairing": "nearest_target_tie_earlier_reuse_allowed",
                "operator_version": "nearest_readout_onset.v2",
                "search_coverage": "complete_joint_valid_island_plus_boundaries_and_operand_lookback",
                "left": left.config, "right": right.config},
            start=max(0, start - lookback), stop=stop))
    return pairs

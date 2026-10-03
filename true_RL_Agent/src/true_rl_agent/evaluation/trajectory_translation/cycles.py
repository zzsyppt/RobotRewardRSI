"""Additive v3 M06: local rhythms of current contact-marker readouts.

The frozen v2 measurement/card path is deliberately unchanged. No physical
contact, gait, success classifier, or sensor-history interpolation lives here.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import CATALOG_VERSION as BASE_CATALOG_VERSION, NumericEpisode, stable_id

CATALOG_VERSION = "trajectory.p0.measurements.v3"
EXTRACTOR_VERSION = "trajectory.contact_cycles.v1"
CYCLE_CONFIG = {
    "version": EXTRACTOR_VERSION, "catalog_version": CATALOG_VERSION,
    "contact_threshold_n": 1.0, "complete_cycles_per_window": 4,
    "population_cv_max": 0.15, "median_relative_deviation_max": 0.20,
    "debounce_max_run_samples": [1, 2], "boundary_context_samples": 3,
    "phase_resultant_min": 0.9, "roundoff_tolerance": 1e-12,
    "threshold_source": "development_calibration:predeclared_resolution_and_variation_rule",
    "window_policy": "all_consecutive_four_cycle_windows_local_only",
    "debounce_policy": "independent_single_pass_bounded_raw_runs",
    "phase_policy": "one_target_onset_per_half_open_reference_cycle_both_qualified",
}


def _runs(values: np.ndarray, steps: np.ndarray) -> list[tuple[int, int]]:
    cuts = np.flatnonzero((values[1:] != values[:-1]) | (np.diff(steps) != 1)) + 1
    return list(zip([0, *cuts.tolist()], [*cuts.tolist(), len(values)]))


def _onsets(values: np.ndarray, steps: np.ndarray) -> list[int]:
    return (np.flatnonzero((values[:-1] == 0) & (values[1:] == 1) & (np.diff(steps) == 1)) + 1).tolist()


def _foot(values: np.ndarray, episode: NumericEpisode, entity: str) -> dict:
    steps, dt = episode.steps, episode.dt_s
    runs = _runs(values, steps)
    variants, sensitivity = [], []
    for width in CYCLE_CONFIG["debounce_max_run_samples"]:
        changed = values.copy()
        for a, b in runs:
            if (0 < a and b < len(values) and b - a <= width and values[a] >= 0
                    and values[a - 1] == values[b] == 1 - values[a]
                    and steps[a] == steps[a - 1] + 1 and steps[b] == steps[b - 1] + 1):
                changed[a:b] = values[a - 1]
        variants.append(changed)
        sensitivity.append({"max_run_samples": width, "max_run_cell_proxy_s": width * dt,
                            "changed_steps": steps[changed != values].tolist(),
                            "onset_steps": steps[_onsets(changed, steps)].tolist(),
                            "transitions": [{"step": int(steps[k]), "before": int(changed[k-1]), "after": int(changed[k])}
                                            for k in range(1, len(steps)) if steps[k] == steps[k-1]+1
                                            and changed[k-1] >= 0 and changed[k] >= 0 and changed[k-1] != changed[k]]})
    islands = []
    for a, b in _runs((values >= 0).astype(np.int8), steps):
        if values[a] >= 0:
            islands.append((a, b))
    onsets, cycles, windows, tails = [], [], [], []
    for island, (lo, hi) in enumerate(islands):
        indices = [lo + k for k in _onsets(values[lo:hi], steps[lo:hi])]
        onsets.extend(indices)
        tails.append({"island": island, "indices_half_open": [lo, hi],
                      "prefix_indices_half_open": [lo, indices[0] if indices else hi],
                      "suffix_indices_half_open": [indices[-1], hi] if indices else None,
                      "initial_state": int(values[lo]), "final_state": int(values[hi - 1])})
        offset = len(cycles)
        for a, b in zip(indices, indices[1:]):
            cycles.append({"island": island, "indices_half_open": [a, b],
                           "onset_steps": [int(steps[a]), int(steps[b])],
                           "raw_inter_onset_interval_s": int(steps[b] - steps[a]) * dt})
        count = CYCLE_CONFIG["complete_cycles_per_window"]
        for k in range(len(indices) - count):
            idx = indices[k:k + count + 1]
            a, b = idx[0], idx[-1]
            periods = np.diff(steps[idx]).astype(np.float64)
            cv = float(np.std(periods) / np.mean(periods))
            median = float(np.median(periods))
            deviation = float(np.max(np.abs(periods - median)) / median)
            reasons = []
            tol = CYCLE_CONFIG["roundoff_tolerance"]
            if cv > CYCLE_CONFIG["population_cv_max"] + tol:
                reasons.append("period_cv")
            if deviation > CYCLE_CONFIG["median_relative_deviation_max"] + tol:
                reasons.append("period_median_deviation")
            context = CYCLE_CONFIG["boundary_context_samples"]
            bounded = lo <= a - context and b + context <= hi
            if not bounded or not (np.all(values[a-context:a] == 0) and np.all(values[b:b+context] == 1)):
                reasons.append("boundary_context_insufficient")
            changed = [width for width, variant in zip(CYCLE_CONFIG["debounce_max_run_samples"], variants)
                       if np.any(variant[max(lo, a-context):min(hi, b+context)]
                                 != values[max(lo, a-context):min(hi, b+context)])]
            if changed:
                reasons.append("debounce_sensitive")
            windows.append({"window_id": stable_id("cycle-window", episode.episode_id, entity, CYCLE_CONFIG, steps[idx].tolist()),
                            "island": island, "cycle_indices": list(range(offset + k, offset + k + count)),
                            "onset_steps": steps[idx].tolist(), "population_cv": cv,
                            "median_relative_deviation": deviation, "sensitive_variants": changed,
                            "qualification": "qualified" if not reasons else "not_qualified", "reasons": reasons,
                            "median_period_s": median * dt if not reasons else None})
    qualified = sorted({c for w in windows if not w["reasons"] for c in w["cycle_indices"]})
    covered = np.zeros(max(0, len(steps) - 1), dtype=bool)
    for c in qualified:
        a, b = cycles[c]["indices_half_open"]
        covered[a:b] = True
    valid_edges = (values[:-1] >= 0) & (values[1:] >= 0) & (np.diff(steps) == 1)
    return {"onsets": [{"index": k, "step": int(steps[k]),
                         "readout_range_s": [float((steps[k]-1)*dt), float(steps[k]*dt)]} for k in onsets],
            "runs": [{"indices_half_open": [a, b], "state": int(values[a]),
                      "left_censored": bool(a == 0 or steps[a] != steps[a-1]+1 or values[a-1] < 0),
                      "right_censored": bool(b == len(steps) or steps[b] != steps[b-1]+1 or values[b] < 0)}
                     for a, b in runs if values[a] >= 0],
            "islands_and_unclosed_parts": tails, "complete_cycles": cycles, "windows": windows,
            "debounce_sensitivity": sensitivity, "qualified_cycle_indices": qualified,
            "counts": {"total_rows": len(steps), "known_rows": int(np.count_nonzero(values >= 0)),
                       "unknown_rows": int(np.count_nonzero(values < 0)),
                       "missing_steps": int(steps[-1] - steps[0] + 1 - len(steps)),
                       "known_adjacent_edges": int(valid_edges.sum()), "covered_unique_edges": int(covered.sum()),
                       "complete_cycles": len(cycles), "candidate_windows": len(windows),
                       "qualified_windows": sum(not w["reasons"] for w in windows),
                       "qualified_unique_cycles": len(qualified),
                       "islands_with_fewer_than_four_cycles": sum(max(0, len([c for c in cycles if c['island'] == i])) < 4
                                                                  for i in range(len(islands)))},
            "rejected_window_reasons": dict(Counter(r for w in windows for r in w["reasons"])),
            "known_readout_span_s": int(valid_edges.sum()) * dt, "qualified_readout_span_s": int(covered.sum()) * dt,
            "qualified_span_fraction": float(covered.sum() / valid_edges.sum()) if valid_edges.any() else None,
            "status": "qualified_local_windows" if qualified else "unknown_rhythm",
            "scope": "local readout windows only; uncovered spans are not evidence of nonperiodic physical motion"}


def measure_contact_cycles(episode: NumericEpisode) -> list[dict[str, Any]]:
    """Current-only source-linked extension; v1/v2 archives have equal capability."""
    threshold = float(episode.task["contact_threshold_n"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("contact threshold must be finite and nonnegative")
    config = {**CYCLE_CONFIG, "contact_threshold_n": threshold}
    entities = episode.entities["support"]
    forces = np.asarray(episode.arrays.get("support_contact_forces_w",
                        np.full((len(episode.steps), len(entities), 3), np.nan)), dtype=np.float64)
    if forces.shape != (len(episode.steps), len(entities), 3):
        raise ValueError("cycle force dimensions differ from source entity binding")
    markers = np.where(np.isfinite(forces).all(axis=2), (np.linalg.norm(forces, axis=2) > threshold).astype(np.int8), -1)

    def record(key, bound, value):
        ref = episode.reference(("support_contact_forces_w",))
        return {"record_type": "measurement", "key": key, "entity_ids": list(bound),
                "record_id": stable_id("cycle-evidence", episode.episode_id, config, key, bound, value),
                "episode_id": episode.episode_id, "catalog_version": CATALOG_VERSION,
                "base_catalog_version": BASE_CATALOG_VERSION, "extractor_version": EXTRACTOR_VERSION,
                "measurement_config": config, "unit": "s", "frame": "recorder_readout",
                "evidence_ref": ref, "value": value}

    feet = [record("contact_cycles", (entity,), _foot(markers[:, i], episode, entity)) for i, entity in enumerate(entities)]
    return feet

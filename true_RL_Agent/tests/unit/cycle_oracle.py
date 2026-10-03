"""Independent scalar oracle: no production translator imports or helpers.

Deliberately duplicate the *frozen specification*, not the numpy implementation.
Input is the original force vectors and integer step sequence.
"""
from collections import Counter
from itertools import groupby
import math
import statistics


def expected(steps, forces, dt):
    steps = [int(k) for k in steps]
    markers = [[int(math.sqrt(sum(float(x)**2 for x in vector)) > 1)
                if all(math.isfinite(float(x)) for x in vector) else -1 for vector in row] for row in forces]
    feet = []
    for foot in range(len(forces[0])):
        values = [row[foot] for row in markers]
        islands, current = [], []
        for i, (k, value) in enumerate(zip(steps, values)):
            if current and (k != steps[current[-1]] + 1 or value < 0):
                islands.append(current)
                current = []
            if value >= 0:
                current.append(i)
        if current:
            islands.append(current)
        filtered = [list(values), list(values)]
        onset_groups, runs = [], []
        for island in islands:
            groups = [list(items) for _, items in groupby(island, key=lambda i: values[i])]
            for j, indices in enumerate(groups):
                runs.append({"indices_half_open": [indices[0], indices[-1]+1], "state": values[indices[0]],
                             "left_censored": j == 0, "right_censored": j == len(groups)-1})
                if 0 < j < len(groups)-1:
                    for width in (1, 2):
                        if len(indices) <= width:
                            for i in indices:
                                filtered[width-1][i] = 1-values[i]
            onset_groups.append([i for i in island[1:] if values[i] == 1 and values[i-1] == 0])
        onsets = [i for group in onset_groups for i in group]
        cycles, windows = [], []
        for island_index, (island, events) in enumerate(zip(islands, onset_groups)):
            offset = len(cycles)
            cycles.extend([(a, b) for a, b in zip(events, events[1:])])
            for pos in range(max(0, len(events)-4)):
                boundaries = events[pos:pos+5]
                a, b = boundaries[0], boundaries[-1]
                periods = [steps[boundaries[j+1]]-steps[boundaries[j]] for j in range(4)]
                mu, med = sum(periods)/4, statistics.median(periods)
                cv = math.sqrt(sum((x-mu)**2 for x in periods)/4)/mu
                dev = max(abs(x-med) for x in periods)/med
                reasons = []
                if cv > .15 + 1e-12:
                    reasons.append("period_cv")
                if dev > .20 + 1e-12:
                    reasons.append("period_median_deviation")
                if not (a-3 >= island[0] and b+2 <= island[-1]
                        and values[a-3:a] == [0, 0, 0] and values[b:b+3] == [1, 1, 1]):
                    reasons.append("boundary_context_insufficient")
                indices = range(max(island[0], a-3), min(island[-1]+1, b+3))
                sensitive = [w for w in (1, 2) if any(filtered[w-1][i] != values[i] for i in indices)]
                if sensitive:
                    reasons.append("debounce_sensitive")
                windows.append({"onset_steps": [steps[i] for i in boundaries],
                                "cycle_indices": list(range(offset+pos, offset+pos+4)), "reasons": reasons,
                                "population_cv": cv, "median_relative_deviation": dev,
                                "median_period_s": med*dt if not reasons else None, "sensitive_variants": sensitive})
        eligible = sorted(set(c for w in windows if not w["reasons"] for c in w["cycle_indices"]))
        edges = set(i for c in eligible for i in range(*cycles[c]))
        valid_edges = sum(values[i-1] >= 0 and values[i] >= 0 and steps[i] == steps[i-1]+1 for i in range(1, len(steps)))
        feet.append({"onset_steps": [steps[i] for i in onsets], "cycles": [[steps[a], steps[b]] for a, b in cycles],
                     "runs": runs, "windows": windows, "qualified_cycle_indices": eligible,
                     "covered_unique_edges": len(edges), "known_adjacent_edges": valid_edges,
                     "qualified_span_fraction": len(edges)/valid_edges if valid_edges else None,
                     "known_rows": sum(v >= 0 for v in values), "unknown_rows": sum(v < 0 for v in values),
                     "rejected_window_reasons": dict(Counter(r for w in windows for r in w["reasons"])),
                     "debounce_sensitivity": [{"max_run_samples": w, "changed_steps": [k for k, a, b in zip(steps, values, filtered[w-1]) if a != b],
                         "onset_steps": [steps[i] for i in range(1, len(steps)) if steps[i] == steps[i-1]+1
                                         and filtered[w-1][i-1] == 0 and filtered[w-1][i] == 1],
                         "transitions": [{"step": steps[i], "before": filtered[w-1][i-1], "after": filtered[w-1][i]}
                             for i in range(1,len(steps)) if steps[i] == steps[i-1]+1 and filtered[w-1][i] >= 0
                             and filtered[w-1][i-1] >= 0 and filtered[w-1][i] != filtered[w-1][i-1]]} for w in (1, 2)]})
    return feet


def assert_close(actual, wanted, path=""):
    if isinstance(wanted, dict):
        for key, value in wanted.items():
            assert key in actual, (path, key)
            assert_close(actual[key], value, path+"/"+key)
    elif isinstance(wanted, list):
        assert len(actual) == len(wanted), (path, len(actual), len(wanted))
        for i, (a, b) in enumerate(zip(actual, wanted)):
            assert_close(a, b, path+"/"+str(i))
    elif isinstance(wanted, float):
        assert actual is not None and math.isclose(actual, wanted, rel_tol=1e-10, abs_tol=1e-12), (path, actual, wanted)
    else:
        assert actual == wanted, (path, actual, wanted)


def check_records(records, steps, forces, dt):
    # Independent frozen semantic literals: numeric agreement cannot establish
    # the meaning of a relabeled unit/frame, even with a recomputed record ID.
    for record in records:
        for key, value in {"record_type": "measurement", "unit": "s",
                           "frame": "recorder_readout", "catalog_version": "trajectory.p0.measurements.v3",
                           "base_catalog_version": "trajectory.p0.measurements.v2",
                           "extractor_version": "trajectory.contact_cycles.v1"}.items():
            assert record.get(key) == value, ("metadata", key, record.get(key), value)
    feet = expected(steps, forces, dt)
    observed_feet = [r for r in records if r["key"] == "contact_cycles"]
    observed_pairs = [r for r in records if r["key"] == "contact_phase"]
    assert len(observed_feet) == len(feet) and not observed_pairs
    for record, expected_foot in zip(observed_feet, feet):
        v = record["value"]
        for key in ("runs", "windows", "qualified_cycle_indices", "qualified_span_fraction", "rejected_window_reasons", "debounce_sensitivity"):
            assert_close(v[key], expected_foot[key], key)
        assert_close([o["step"] for o in v["onsets"]], expected_foot["onset_steps"], "onsets")
        assert_close([c["onset_steps"] for c in v["complete_cycles"]], expected_foot["cycles"], "cycles")
        for c, (a, b) in zip(v["complete_cycles"], expected_foot["cycles"]):
            assert_close(c["raw_inter_onset_interval_s"], (b-a)*dt)
        counts = {key: expected_foot[key] for key in ("covered_unique_edges", "known_adjacent_edges", "known_rows", "unknown_rows")}
        counts.update(total_rows=len(steps), missing_steps=int(steps[-1]-steps[0]+1-len(steps)),
                      complete_cycles=len(expected_foot["cycles"]), candidate_windows=len(expected_foot["windows"]),
                      qualified_windows=sum(not w["reasons"] for w in expected_foot["windows"]),
                      qualified_unique_cycles=len(expected_foot["qualified_cycle_indices"]))
        assert_close(v["counts"], counts, "counts")
        assert_close(v["known_readout_span_s"], counts["known_adjacent_edges"]*dt)
        assert_close(v["qualified_readout_span_s"], counts["covered_unique_edges"]*dt)
        good_ids = {w["window_id"] for w in v["windows"] if not w["reasons"]}
        assert len(good_ids) == counts["qualified_windows"]
    return feet

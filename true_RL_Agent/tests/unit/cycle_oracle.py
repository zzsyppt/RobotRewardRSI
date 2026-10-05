"""Independent scalar oracle: no production translator imports or helpers.

Check the local-window rule with scalar operations, independently of numpy.
Input is the original force vectors and integer step sequence.
"""
from itertools import groupby
import math
import statistics


def expected(steps, forces, dt, threshold):
    steps = [int(k) for k in steps]
    markers = [[int(math.sqrt(sum(float(x)**2 for x in vector)) > threshold)
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
        onset_groups = []
        for island in islands:
            groups = [list(items) for _, items in groupby(island, key=lambda i: values[i])]
            for j, indices in enumerate(groups):
                if 0 < j < len(groups)-1:
                    for width in (1, 2):
                        if len(indices) <= width:
                            for i in indices:
                                filtered[width-1][i] = 1-values[i]
            onset_groups.append([i for i in island[1:] if values[i] == 1 and values[i-1] == 0])
        cycles, windows = [], []
        for island, events in zip(islands, onset_groups):
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
                windows.append({"cycle_indices": list(range(offset+pos, offset+pos+4)), "reasons": reasons})
        eligible = sorted(set(c for w in windows if not w["reasons"] for c in w["cycle_indices"]))
        feet.append([(steps[cycles[c][1]]-steps[cycles[c][0]])*dt for c in eligible])
    return feet


def check_periods(periods, entities, steps, forces, dt, threshold):
    assert set(periods) == set(entities)
    for entity, wanted in zip(entities, expected(steps, forces, dt, threshold), strict=True):
        actual = periods[entity]
        assert len(actual) == len(wanted), (entity, actual, wanted)
        for a, b in zip(actual, wanted):
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (entity, a, b)

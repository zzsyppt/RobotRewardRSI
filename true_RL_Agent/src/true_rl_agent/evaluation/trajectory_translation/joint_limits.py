"""Recorded joint coordinates relative to identified bounds, never effective physical limits."""
from __future__ import annotations

import numpy as np

from .contracts import NumericEpisode

INSIDE_EDGE_BAND_FRACTION = 0.05


def measure_runtime_joint_limits(episode: NumericEpisode, buffers: dict) -> dict:
    """Buffers already follow episode joint columns; select only the environment.

    The caller validates metadata names, units and axes once per archive. No
    resolved articulation ID is applied a second time, and soft gets no edge band.
    """
    raw = np.asarray(episode.arrays["joint_pos"])
    q = raw.astype(float)
    result = {}
    for kind, table in buffers.items():
        bounds = table[episode.env_index]
        values = {}
        for column, name in enumerate(episode.entities["joints"]):
            lower, upper = bounds[column]
            valid = np.isfinite(q[:, column])
            v = q[valid, column]
            # Only the exact error of encoding a boundary in q's recorded dtype.
            tol = np.abs(bounds[column].astype(raw.dtype).astype(float)-bounds[column])
            low, high = int((v < lower-tol[0]).sum()), int((v > upper+tol[1]).sum())
            values[name] = {"label": name+"相对启动"+kind+"缓冲区", "unit": "rad",
                "lower": float(lower), "upper": float(upper), "boundary_tolerance_rad": tol.tolist(),
                "valid": int(valid.sum()), "unknown": int((~valid).sum()),
                "lower_outside": low, "upper_outside": high,
                "minimum_margin": float(np.minimum(v-lower,upper-v).min()) if len(v) else None}
        result[kind] = values
    return result


def measure_joint_limits(episode: NumericEpisode, bounds: dict, *, q_unit: str) -> dict[str, dict]:
    """Bounds are named, audited radian joint coordinates; absent mapping is unknown."""
    if q_unit != "rad":
        raise ValueError("joint q must use qualified rad coordinates")
    names = episode.entities["joints"]
    raw = np.asarray(episode.arrays.get("joint_pos", np.full((len(episode.steps), len(names)), np.nan)))
    q = raw.astype(float)
    if q.shape != (len(episode.steps), len(names)):
        raise ValueError("joint_pos shape differs from named joint columns")
    result = {}
    for column, name in enumerate(names):
        bound = bounds.get(name)
        qualified = bound is not None and bound.get("coordinate_status") == "qualified_nominal"
        if qualified:
            if bound["unit"] != "rad":
                raise ValueError("nominal bounds must use rad")
            lower, upper = bound["lower"], bound["upper"]
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("nominal bounds must be finite and ordered")
        valid = np.isfinite(q[:, column]) & qualified
        rows = np.flatnonzero(valid)
        total, known = len(q), len(rows)
        value = {"counts": {"total_rows": total, "finite_rows": known, "unknown_rows": total-known,
                            "inside_edge_band_rows": 0, "outside_rows": 0,
                            "lower_edge_rows": 0, "upper_edge_rows": 0,
                            "lower_outside_rows": 0, "upper_outside_rows": 0, "roundoff_boundary_rows": 0},
                 "minimum_signed_margin_rad": None}
        if known:
            values = q[rows, column]
            width = upper-lower
            low_distance, high_distance = values-lower, upper-values
            margin = np.minimum(low_distance, high_distance)
            band = INSIDE_EDGE_BAND_FRACTION*width
            # Only representational error at the two exact nominal boundaries.
            lo_tol = abs(float(np.asarray(lower, dtype=raw.dtype))-lower)
            hi_tol = abs(float(np.asarray(upper, dtype=raw.dtype))-upper)
            inside = (values >= lower-lo_tol) & (values <= upper+hi_tol)
            lower_near = inside & (values <= lower+band)
            upper_near = inside & (values >= upper-band)
            near = lower_near | upper_near
            outside = ~inside
            near_n, outside_n = int(near.sum()), int(outside.sum())
            value["minimum_signed_margin_rad"] = float(margin.min())
            value["counts"].update(inside_edge_band_rows=near_n, outside_rows=outside_n,
                lower_edge_rows=int(lower_near.sum()), upper_edge_rows=int(upper_near.sum()),
                lower_outside_rows=int((values < lower-lo_tol).sum()), upper_outside_rows=int((values > upper+hi_tol).sum()),
                roundoff_boundary_rows=int(((margin < 0) & inside).sum()))
        result[name] = value
    return result

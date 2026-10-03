"""Recorded joint coordinates relative to qualified nominal bounds, never effective limits."""
from __future__ import annotations

import numpy as np

from .contracts import NumericEpisode, stable_id

CATALOG_VERSION = "trajectory.p0.measurements.v4"
BASE_CATALOG_VERSION = "trajectory.p0.measurements.v2"
CONFIG = {"version": "nominal_joint_limits.v1", "inside_edge_band_fraction": 0.05,
          "threshold_source": "predeclared_descriptive_band_not_safety_limit",
          "effective_limits": "unknown_not_recorded"}


def measure_joint_limits(episode: NumericEpisode, bounds: dict, *, q_unit: str, asset_id: str) -> list[dict]:
    """Bounds are named, audited radian joint coordinates; absent mapping is unknown."""
    if q_unit != "rad":
        raise ValueError("joint q must use qualified rad coordinates")
    names = episode.entities["joints"]
    raw = np.asarray(episode.arrays.get("joint_pos", np.full((len(episode.steps), len(names)), np.nan)))
    q = raw.astype(float)
    if q.shape != (len(episode.steps), len(names)):
        raise ValueError("joint_pos shape differs from named joint columns")
    records = []
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
        value = {"status": "measured" if known else "unknown",
                 "unknown_reason": None if known else ("no_finite_q" if qualified else "unproven_coordinate_mapping"),
                 "counts": {"total_rows": total, "finite_rows": known, "unknown_rows": total-known,
                            "inside_edge_band_rows": 0, "outside_rows": 0,
                            "lower_edge_rows": 0, "upper_edge_rows": 0,
                            "lower_outside_rows": 0, "upper_outside_rows": 0, "roundoff_boundary_rows": 0},
                 "missing_steps": int(episode.steps[-1]-episode.steps[0]+1-total),
                 "nominal_bounds_rad": [lower, upper] if qualified else None,
                 "q_range_rad": None, "position_fraction_range": None, "range_occupancy_fraction": None,
                 "minimum_signed_margin_rad": None, "inside_edge_band_fraction": None, "outside_fraction": None,
                 "sample_cell_seconds": {"finite": known*episode.dt_s, "inside_edge_band": 0.0, "outside": 0.0},
                 "witnesses": {}, "effective_limit_status": "unknown_not_recorded"}
        if known:
            values = q[rows, column]
            width = upper-lower
            low_distance, high_distance = values-lower, upper-values
            margin = np.minimum(low_distance, high_distance)
            position = low_distance/width
            band = CONFIG["inside_edge_band_fraction"]*width
            # Only representational error at the two exact nominal boundaries.
            lo_tol = abs(float(np.asarray(lower, dtype=raw.dtype))-lower)
            hi_tol = abs(float(np.asarray(upper, dtype=raw.dtype))-upper)
            inside = (values >= lower-lo_tol) & (values <= upper+hi_tol)
            lower_near = inside & (values <= lower+band)
            upper_near = inside & (values >= upper-band)
            near = lower_near | upper_near
            outside = ~inside
            near_n, outside_n = int(near.sum()), int(outside.sum())
            value.update(q_range_rad=[float(values.min()), float(values.max())],
                         position_fraction_range=[float(position.min()), float(position.max())],
                         range_occupancy_fraction=float((values.max()-values.min())/width),
                         minimum_signed_margin_rad=float(margin.min()),
                         inside_edge_band_fraction=near_n/known, outside_fraction=outside_n/known)
            value["boundary_roundoff_tolerance_rad"] = [lo_tol, hi_tol]
            value["counts"].update(inside_edge_band_rows=near_n, outside_rows=outside_n,
                lower_edge_rows=int(lower_near.sum()), upper_edge_rows=int(upper_near.sum()),
                lower_outside_rows=int((values < lower-lo_tol).sum()), upper_outside_rows=int((values > upper+hi_tol).sum()),
                roundoff_boundary_rows=int(((margin < 0) & inside).sum()))
            value["sample_cell_seconds"].update(inside_edge_band=near_n*episode.dt_s, outside=outside_n*episode.dt_s)
            for label, index in (("q_min", int(values.argmin())), ("q_max", int(values.argmax())),
                                 ("minimum_margin", int(margin.argmin()))):
                row = int(rows[index])
                value["witnesses"][label] = {"step": int(episode.steps[row]), "q_rad": float(values[index]),
                    "lower_distance_rad": float(low_distance[index]), "upper_distance_rad": float(high_distance[index]),
                    "signed_margin_rad": float(margin[index]), "position_fraction": float(position[index]),
                    "joint_column": column, "source": episode.reference(("joint_pos",), row, row+1)}
        record = episode.record("nominal_joint_limits", "measurement", (name,), ("joint_pos",), value,
                                unit="rad; fraction; sampled s", frame="nominal_usd_joint_coordinate", config=CONFIG)
        record.update(catalog_version=CATALOG_VERSION, base_catalog_version=BASE_CATALOG_VERSION,
                      extractor_version="trajectory.nominal_joint_limits.v1", asset_id=asset_id)
        record.pop("record_id")
        record["record_id"] = stable_id("nominal-limit-evidence", record)
        records.append(record)
    return records

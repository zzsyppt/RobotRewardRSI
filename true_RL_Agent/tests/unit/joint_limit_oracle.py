"""Independent scalar M09 oracle: no production measurement/asset loader imports."""
import math
import re
import struct


def asset_bounds(asset):
    """Re-read authored USDA float fields, independently of Sdf-extracted joint table."""
    result = {}
    for name, body in re.findall(r'def PhysicsRevoluteJoint "([^"]+)".*?\{(.*?)\n        \}', asset["root_usda"], re.S):
        lo, hi = [struct.unpack("f", struct.pack("f", float(re.search(r'physics:'+field+r' = ([-\d.]+)', body)[1])))[0]
                  for field in ("lowerLimit", "upperLimit")]
        assert re.search(r'physics:axis = "([XYZ])"', body)[1] == "X"
        rotations = [re.search(r'physics:localRot'+str(i)+r' = \(([^)]+)\)', body)[1] for i in (0, 1)]
        assert rotations[0] == rotations[1]
        result[name] = (lo*math.pi/180, hi*math.pi/180)
    assert len(result) == 12
    by_name = {j["name"]: j for j in asset["joints"]}
    assert set(by_name) == set(result)
    for name, (lo, hi) in result.items():
        j = by_name[name]
        assert abs(lo-math.radians(j["physics:lowerLimit"])) < 1e-14
        assert abs(hi-math.radians(j["physics:upperLimit"])) < 1e-14
        assert j["physics:localRot0"] == j["physics:localRot1"]
        for b in j["body_authored_transforms"].values():
            assert b["xformOp:orient"] == [1, 0, 0, 0] and b["xformOp:scale"] == [1, 1, 1]
        # Both zero-pose anchors coincide within the authored float32 precision.
        b0, b1 = [j["body_authored_transforms"]["physics:body"+str(i)] for i in (0, 1)]
        for axis in range(3):
            assert abs(b0["xformOp:translate"][axis]+j["physics:localPos0"][axis]
                       -b1["xformOp:translate"][axis]-j["physics:localPos1"][axis]) < 1e-7
    return result


def expected(values, steps, lower, upper, dt):
    samples = [(i, float(v)) for i, v in enumerate(values) if math.isfinite(v)]
    n = len(samples)
    counts = {"total_rows": len(values), "finite_rows": n, "unknown_rows": len(values)-n,
              "inside_edge_band_rows": 0, "outside_rows": 0,
              "lower_edge_rows":0,"upper_edge_rows":0,"lower_outside_rows":0,"upper_outside_rows":0,"roundoff_boundary_rows":0}
    fields = {k: None for k in ("q_range_rad", "position_fraction_range", "range_occupancy_fraction",
              "minimum_signed_margin_rad", "inside_edge_band_fraction", "outside_fraction")}
    witnesses = {}
    if n:
        width = upper-lower
        margins = [min(q-lower, upper-q) for _, q in samples]
        counts["inside_edge_band_rows"] = sum(lower <= q <= upper and (q <= lower+.05*width or q >= upper-.05*width)
                                               for _, q in samples)
        counts["outside_rows"] = sum(m < 0 for m in margins)
        counts["lower_edge_rows"] = sum(lower <= q <= lower+.05*width for _,q in samples)
        counts["upper_edge_rows"] = sum(upper-.05*width <= q <= upper for _,q in samples)
        counts["lower_outside_rows"] = sum(q < lower for _,q in samples)
        counts["upper_outside_rows"] = sum(q > upper for _,q in samples)
        qlo, qhi = min(q for _, q in samples), max(q for _, q in samples)
        fields.update(q_range_rad=[qlo, qhi], position_fraction_range=[(qlo-lower)/width, (qhi-lower)/width],
                      range_occupancy_fraction=(qhi-qlo)/width, minimum_signed_margin_rad=min(margins),
                      inside_edge_band_fraction=counts["inside_edge_band_rows"]/n,
                      outside_fraction=counts["outside_rows"]/n)
        for label, index in (("q_min", next(i for i, (_, q) in enumerate(samples) if q == qlo)),
                             ("q_max", next(i for i, (_, q) in enumerate(samples) if q == qhi)),
                             ("minimum_margin", margins.index(min(margins)))):
            row, q = samples[index]
            witnesses[label] = {"row": row, "step": int(steps[row]), "q_rad": q,
                                "lower_distance_rad": q-lower, "upper_distance_rad": upper-q,
                                "signed_margin_rad": min(q-lower, upper-q), "position_fraction": (q-lower)/width}
    return {"counts": counts, **fields, "witnesses": witnesses,
            "sample_cell_seconds": {"finite": n*dt, "inside_edge_band": counts["inside_edge_band_rows"]*dt,
                                    "outside": counts["outside_rows"]*dt}}


def close(actual, wanted):
    if isinstance(wanted, dict):
        assert actual.keys() == wanted.keys(), (actual.keys(), wanted.keys())
        for key in wanted:
            close(actual[key], wanted[key])
    elif isinstance(wanted, (list, tuple)):
        assert len(actual) == len(wanted)
        for a, b in zip(actual, wanted):
            close(a, b)
    elif isinstance(wanted, float):
        assert abs(actual-wanted) <= 1e-12, (actual, wanted)
    else:
        assert actual == wanted, (actual, wanted)


def check_record(record, values, steps, lower, upper, dt, name):
    assert record["entity_ids"] == [name], "metadata entity"
    assert record["unit"] == "rad; fraction; sampled s", "metadata unit"
    assert record["frame"] == "nominal_usd_joint_coordinate", "metadata frame"
    assert record["catalog_version"] == "trajectory.p0.measurements.v4"
    assert record["base_catalog_version"] == "trajectory.p0.measurements.v2"
    assert record["extractor_version"] == "trajectory.nominal_joint_limits.v1"
    assert record["measurement_config"] == {"version": "nominal_joint_limits.v1", "inside_edge_band_fraction": .05,
        "threshold_source": "predeclared_descriptive_band_not_safety_limit", "effective_limits": "unknown_not_recorded"}
    wanted = expected(values, steps, lower, upper, dt)
    observed = record["value"]
    for key in wanted:
        if key != "witnesses":
            close(observed[key], wanted[key])
    close(observed["nominal_bounds_rad"], [lower, upper])
    assert observed["status"] == ("measured" if wanted["counts"]["finite_rows"] else "unknown")
    assert observed["unknown_reason"] == (None if wanted["counts"]["finite_rows"] else "no_finite_q")
    assert observed["effective_limit_status"] == "unknown_not_recorded"
    assert observed["missing_steps"] == int(steps[-1]-steps[0]+1-len(steps))
    assert observed["witnesses"].keys() == wanted["witnesses"].keys()
    for label, witness in wanted["witnesses"].items():
        close({k: observed["witnesses"][label][k] for k in witness if k != "row"},
              {k: v for k, v in witness.items() if k != "row"})
    return wanted

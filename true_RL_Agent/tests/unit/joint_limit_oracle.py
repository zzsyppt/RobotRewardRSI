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


def expected(values, lower, upper):
    """Scalar oracle for float64 samples; float32 boundaries have separate examples."""
    samples = [float(v) for v in values if math.isfinite(v)]
    n = len(samples)
    counts = {"total_rows": len(values), "finite_rows": n, "unknown_rows": len(values)-n,
              "inside_edge_band_rows": 0, "outside_rows": 0,
              "lower_edge_rows":0,"upper_edge_rows":0,"lower_outside_rows":0,"upper_outside_rows":0,"roundoff_boundary_rows":0}
    margin = None
    if n:
        width = upper-lower
        margins = [min(q-lower, upper-q) for q in samples]
        counts["inside_edge_band_rows"] = sum(lower <= q <= upper and (q <= lower+.05*width or q >= upper-.05*width)
                                               for q in samples)
        counts["outside_rows"] = sum(m < 0 for m in margins)
        counts["lower_edge_rows"] = sum(lower <= q <= lower+.05*width for q in samples)
        counts["upper_edge_rows"] = sum(upper-.05*width <= q <= upper for q in samples)
        counts["lower_outside_rows"] = sum(q < lower for q in samples)
        counts["upper_outside_rows"] = sum(q > upper for q in samples)
        margin = min(margins)
    return {"counts": counts, "minimum_signed_margin_rad": margin}


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

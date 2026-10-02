"""Category-wide source arithmetic for old calibration data; no score imports."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / "review_calibration_r1/materials/evidence.json"


def function_parts(source):
    return {node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def main():
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    out = {"purpose": "Independent source arithmetic across both supplied programs; not scores or physical observations.",
           "evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(), "samples": {}}
    for name, sample in data.items():
        parent, candidate = function_parts(sample["parent_source"]), function_parts(sample["candidate_source"])
        out["samples"][name] = {"common_function_comparison": [
            {"name": key, "source_bytes_equal": parent[key] == candidate[key],
             "ast_equal": ast.dump(ast.parse(parent[key]), include_attributes=False) ==
                          ast.dump(ast.parse(candidate[key]), include_attributes=False)}
            for key in sorted(parent.keys() & candidate.keys())]}
    source_x = data["sample-x"]["candidate_source"]
    assert "release = (tilt / 0.4).clamp(min=0.0, max=1.0) * 0.75" in source_x
    assert "keep = 1.0 - release" in source_x
    limits = data["sample-x"]["author_visible_evidence"]["public_feedback_summary"]["public_dev"]["behavior"]["measurement_contract"]["stable_limits"]
    out["samples"]["sample-x"].update(stable_limits_as_supplied=limits,
        damping_scale_grid=[{"roll_rad": roll, "pitch_rad": pitch,
            "tilt_rad": math.hypot(roll, pitch),
            "keep": 1 - .75 * min(math.hypot(roll, pitch) / .4, 1)}
            for roll, pitch in itertools.product((0., .1, .2, .4), repeat=2)])
    source_y = data["sample-y"]["candidate_source"]
    for text in ("nose_up_edge = 0.55", "nose_down_edge = 0.20", "in_band_scale = 0.25", "out_band_linear = 1.0", "out_band_quadratic = 3.0"):
        assert text in source_y
    values = []
    for pitch in (-1., -.75, -.55, -.2, 0., .2, .55, .75, 1.):
        inside = min(.2, max(-.55, pitch))
        excess = max(-pitch - .55, 0) + max(pitch - .2, 0)
        candidate = .25 * inside**2 + excess + 3 * excess**2
        values.append({"projected_gravity_x": pitch, "projected_gravity_y": 0.,
                       "parent_unweighted_cost": pitch**2, "candidate_unweighted_cost": candidate,
                       "candidate_to_parent_ratio": None if pitch == 0 else candidate / pitch**2})
    contribution = -2.5 * math.sin(.53)**2
    out["samples"]["sample-y"].update(pitch_cost_grid_at_roll_zero=values,
        measured_angle_formula_grid=[{"pitch_rad": .53, "roll_rad": roll,
            "pitch_contribution": contribution,
            "full_parent_rate": -2.5 * (math.sin(.53)**2 + math.sin(roll)**2 * math.cos(.53)**2)}
            for roll in (0., .1, .2, .3)],
        formula_scope="Grid values are synthetic arithmetic states, not measured samples. Projected gravity uses the supplied full Euler formula; actual synchronized roll/pitch availability must be judged from the supplied record.")
    path = BASE / "review_calibration_r2/materials/arithmetic_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(out["samples"]), "scores_read": False, "api_calls": 0}))


if __name__ == "__main__":
    main()

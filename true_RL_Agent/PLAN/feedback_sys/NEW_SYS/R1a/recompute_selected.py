"""Independent raw-array/USDA/triangle-oracle check of two selected K1 claims.

Adapted by the sole writer from the read-only Astra ultra audit command. Does
not invoke production measurement or aggregation functions.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.unit.joint_limit_oracle import asset_bounds, check_record as check_joint
from tests.unit.terrain_mesh_oracle import mesh_height, check_record as check_terrain, check_first_extrema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    started, cpu = time.perf_counter(), time.process_time()
    budgets = ROOT / "outputs/r1a/budgets_final/artifacts"
    entry = next(b for b in json.loads((budgets / "budgets.json").read_bytes())["budgets"]
                 if b["source"] == "k1" and b["budget_bytes"] == 10240)
    reference = entry["source_manifest"]
    manifest_path = Path(reference["manifest_path"])
    assert sha(manifest_path) == reference["manifest_sha256"]
    raw = json.loads(manifest_path.read_bytes())
    measurement_path = Path(entry["measurements_reference"]["manifest_path"])
    assert sha(measurement_path) == entry["measurements_reference"]["manifest_sha256"]
    measurement = json.loads(measurement_path.read_bytes())
    for name, artifact in measurement["artifacts"].items():
        assert sha(measurement_path.parent / name) == artifact["sha256"]
    summary = json.loads((measurement_path.parent / "summary.json").read_bytes())
    joint = next(f for f in entry["selected_new"] if f["measurement"] == "joint")
    terrain = next(f for f in entry["selected_new"] if f["measurement"] == "terrain" and f["group"].startswith("c2_rough/"))
    assert joint["operands"] == summary["joint"]["groups"][joint["group"]]
    assert terrain["operands"] == summary["terrain"]["groups"][terrain["group"]]
    asset = ROOT / "src/true_rl_agent/tasks/stair_climbing/go2_nominal_usd.json"
    assert sha(asset) == "b8a1ab9846cca1b282258cd9b8c871d5f6746bd9bbb85e3d38a34921d90614bb"
    low, high = asset_bounds(json.loads(asset.read_bytes()))[joint["group"]]
    width = high-low
    history_path = ROOT / "outputs/p2c3/historical_02/stdout.json"
    assert sha(history_path) == "84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f"
    history = next(s for s in json.loads(history_path.read_bytes())["sources"] if s["manifest_sha256"] == reference["manifest_sha256"])
    cells = {c["cell_id"]: c for c in history["cells"]}
    envs = [i for i, c in enumerate(history["env_cell_ids"]) if cells[c]["regime"] == "c2_rough"]
    projection = raw["metadata"]["runtime_projection"]
    joint_col = projection["joint_names"].index(joint["group"])
    foot_col = projection["resolved_roles"]["support_effector"].index(terrain["group"].split(".")[-1])
    qs, steps, locators, points = (defaultdict(list) for _ in range(4))
    origins, raw_bytes = {}, 0
    for chunk in raw["chunks"]:
        path = manifest_path.parent / chunk["file"]
        assert sha(path) == chunk["sha256"]
        raw_bytes += path.stat().st_size
        run_index = chunk["metadata"]["run_index"]
        with np.load(path, allow_pickle=False) as arrays:
            valid, step = arrays["valid"], arrays["episode_length"]
            q, point = arrays["joint_pos"][:, :, joint_col].astype(float), arrays["support_pos_w"][:, :, foot_col].astype(float)
            if chunk["metadata"]["static_present"]:
                for env in envs:
                    cell = cells[history["env_cell_ids"][env]]
                    assert int(arrays["terrain_types"][env]) == cell["terrain_type"]
                    np.testing.assert_allclose(arrays["terrain_env_origins_w"][env], cell["expected_origin_w"], rtol=0, atol=1e-4)
                    origins[run_index, env] = arrays["terrain_env_origins_w"][env].astype(float)
            for env in range(valid.shape[1]):
                ix = np.flatnonzero(valid[:, env]); key = run_index, env
                if not len(ix):
                    continue
                qs[key].append(q[ix, env]); steps[key].append(step[ix, env])
                locators[key].extend((chunk["file"], chunk["sha256"], int(i)) for i in ix)
                if env in envs:
                    points[key].append(point[ix, env])
    qs, steps, points = ({k: np.concatenate(v) for k, v in rows.items()} for rows in (qs, steps, points))
    counts = dict.fromkeys(("total_rows", "finite_rows", "unknown_rows", "inside_edge_band_rows", "outside_rows"), 0)
    margins, occupancies, worst = [], [], None
    for key, q in qs.items():
        finite = np.isfinite(q); values = q[finite]; margin = np.minimum(values-low, high-values)
        edge = (values >= low) & (values <= high) & ((values <= low+.05*width) | (values >= high-.05*width))
        for name, number in zip(counts, (len(q), len(values), (~finite).sum(), edge.sum(), ((values < low) | (values > high)).sum())):
            counts[name] += int(number)
        margins.append(float(margin.min())); occupancies.append(float(np.ptp(values)/width))
        ix = int(np.flatnonzero(finite)[np.argmin(margin)])
        if worst is None or margin.min() < worst[0]:
            worst = float(margin.min()), key, ix, float(q[ix])
    assert counts == joint["operands"]["counts"]
    assert abs(np.median(margins)-joint["operands"]["episode_equal"]["minimum_signed_margin_rad"]["median"]) < 1e-12
    assert abs(np.median(occupancies)-joint["operands"]["episode_equal"]["range_occupancy_fraction"]["median"]) < 1e-12
    terrain_counts = dict.fromkeys(("total", "known", "unknown", "boundary_band", "outside_patch"), 0)
    results, means, boundary = {}, [], None
    for cell_id in sorted({history["env_cell_ids"][env] for env in envs}):
        keys = sorted(k for k in points if history["env_cell_ids"][k[1]] == cell_id)
        origin = origins[keys[0]]
        assert all(np.array_equal(origins[k], origin) for k in keys)
        heights, differences, statuses = mesh_height(np.concatenate([points[k] for k in keys]), origin, cells[cell_id])
        offset = 0
        for key in keys:
            end = offset+len(points[key]); h, d, s = heights[offset:end], differences[offset:end], statuses[offset:end]
            results[key] = h, d, s; known = s == 0
            for name, number in zip(terrain_counts, (len(s), known.sum(), (~known).sum(), (s == 5).sum(), (s == 4).sum())):
                terrain_counts[name] += int(number)
            if known.any():
                means.append(float(d[known].mean()))
            if boundary is None and (s == 5).any():
                boundary = key, int(np.flatnonzero(s == 5)[0])
            offset = end
    assert all(terrain_counts[k] == terrain["operands"]["counts"][k] for k in terrain_counts)
    assert abs(np.median(means)-terrain["operands"]["episode_equal"]["mean"]["median"]) < 1e-10
    assert len(means) == terrain["operands"]["known_episodes"] == 108
    joint_key, terrain_key = worst[1], boundary[0]
    joint_record = terrain_record = None
    with (measurement_path.parent / "measurements.jsonl").open(encoding="utf-8") as stream:
        for lines, line in enumerate(stream, 1):
            row = json.loads(line); key = row["cycle"]["run_index"], row["cycle"]["env_index"]
            if key == joint_key:
                joint_record = next(r for r in row["joint"]["records"] if r["entity_ids"] == [joint["group"]])
            if key == terrain_key:
                terrain_record = next(r for r in row["terrain"]["records"] if r["entity_ids"] == [terrain["group"].split("/", 1)[1]])
            if joint_record is not None and terrain_record is not None:
                break
    check_joint(joint_record, qs[joint_key], steps[joint_key], low, high, raw["metadata"]["seeds"][joint_key[0]-1]["step_dt"], joint["group"])
    h, d, s = results[terrain_key]
    check_terrain(terrain_record, points[terrain_key], h, d, s, raw["metadata"]["seeds"][terrain_key[0]-1]["step_dt"], terrain["group"].split("/", 1)[1])
    check_first_extrema(terrain_record, h, d, steps[terrain_key])
    def locator(key, ix, channel, column):
        file, digest, row = locators[key][ix]
        return {"manifest_sha256": reference["manifest_sha256"], "run": key[0], "env": key[1],
            "runner_seed": raw["metadata"]["seeds"][key[0]-1]["runner_seed"], "step": int(steps[key][ix]),
            "file": file, "sha256": digest, "row_zero_based": row, "channel": channel, "column": column}
    outside = locator(joint_key, worst[2], "joint_pos", joint_col)
    boundary_locator = locator(terrain_key, boundary[1], "support_pos_w", foot_col)
    for witness, expected in ((joint_record["value"]["witnesses"]["minimum_margin"]["source"], outside),
                              (terrain_record["value"]["first_by_status"]["boundary_band"]["source_ref"], boundary_locator)):
        assert witness == {"manifest_sha256": expected["manifest_sha256"], "run_index": expected["run"],
            "channels": [expected["channel"]], "steps_half_open": [expected["step"], expected["step"]+1],
            "chunks": [{"file": expected["file"], "sha256": expected["sha256"],
                        "rows_half_open": [expected["row_zero_based"], expected["row_zero_based"]+1], "env_index": expected["env"]}]}
    known_ix = int(np.flatnonzero(s == 0)[0])
    result = {"status": "PASS_independent_targeted_raw_recompute", "source": "K1 historical, not new rollout",
        "joint": {"claim_id": joint["id"], "bounds_rad": [low, high], "counts": counts, "episodes": len(qs),
            "median_min_margin_rad": float(np.median(margins)), "median_occupancy": float(np.median(occupancies)),
            "counterexample": {"locator": outside, "q_rad": worst[3], "margin_rad": worst[0]}, "effective_limits": "unknown"},
        "terrain": {"claim_id": terrain["id"], "counts": terrain_counts, "episodes": len(means),
            "median_mean_difference_m": float(np.median(means)), "origin_world_m": origins[terrain_key].tolist(),
            "known": {"locator": locator(terrain_key, known_ix, "support_pos_w", foot_col),
                      "point_world_m": points[terrain_key][known_ix].tolist(), "height_m": float(h[known_ix]), "difference_m": float(d[known_ix])},
            "counterexample": {"locator": boundary_locator, "point_world_m": points[terrain_key][boundary[1]].tolist(),
                               "height_m": None, "status": "boundary_band", "unknown_not_zero": bool(np.isnan(h[boundary[1]]))},
            "runtime_surface": "unknown"},
        "cost": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
            "raw_chunks": len(raw["chunks"]), "raw_compressed_bytes": raw_bytes, "measurement_lines_scanned": lines,
            "external_model_calls": 0, "GPU_runs": 0, "training_runs": 0, "tokens": "unknown"}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()

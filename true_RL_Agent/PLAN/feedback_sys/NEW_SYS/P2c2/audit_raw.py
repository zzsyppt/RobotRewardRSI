"""Independent raw q / authored USD / report audit; no production imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.unit.joint_limit_oracle import asset_bounds, check_record, close


def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identifier(kind, *parts):
    return kind+":"+hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    def percentile(p):
        if not ordered: return None
        index = (len(ordered)-1)*p
        i = int(index)
        return ordered[i]+(ordered[min(i+1, len(ordered)-1)]-ordered[i])*(index-i)
    return {"count": len(values), "quantile_method": "linear", "minimum": min(values) if values else None,
            "maximum": max(values) if values else None, "mean": sum(values)/len(values) if values else None,
            "q25": percentile(.25), "median": percentile(.5), "q75": percentile(.75)}


def raw_episodes(source):
    path = ROOT/source["manifest_relative_path"]
    assert sha(path) == source["manifest_sha256"]
    raw = read(path)
    assert raw["identity"] == source["identity"] and raw["identity"]["splits"] == ["public_dev"]
    locks = {c["file"]: c["sha256"] for c in source["chunks"]}
    assert set(locks) == {c["file"] for c in raw["chunks"]}
    for run, seed in enumerate(raw["metadata"]["seeds"], 1):
        assert seed["split"] == "public_dev" and seed["run_index"] == run
        chunks = [c for c in raw["chunks"] if c["metadata"]["run_index"] == run]
        parts = {k: [] for k in ("joint_pos", "valid", "episode_length")}
        for chunk in chunks:
            assert sha(path.parent/chunk["file"]) == chunk["sha256"] == locks[chunk["file"]]
            with np.load(path.parent/chunk["file"], allow_pickle=False) as npz:
                for key in parts: parts[key].append(npz[key].copy())
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        for env, n in enumerate(seed["episode_steps"]):
            assert arrays["valid"][:, env].tolist() == [True]*n+[False]*(len(arrays["valid"])-n)
            assert arrays["episode_length"][:n, env].tolist() == list(range(1, n+1))
            refs = [{"file": c["file"], "sha256": c["sha256"], "env_index": env,
                     "rows_half_open": [0, min(c["metadata"]["steps"], n-c["metadata"]["step_start"]+1)]}
                    for c in chunks if c["metadata"]["step_start"] <= n]
            yield raw, run, env, arrays["joint_pos"][:n, env], seed["step_dt"], refs, chunks


def audit(source, directory):
    started = time.perf_counter()
    asset_path = ROOT/"src/true_rl_agent/tasks/stair_climbing/go2_nominal_usd.json"
    bounds = asset_bounds(read(asset_path))
    build, report = read(directory/"manifest.json"), read(directory/"report.json")
    for name, info in build["artifacts"].items():
        assert sha(directory/name) == info["sha256"] and (directory/name).stat().st_size == info["bytes"]
    assert report["binding"] == build["binding"]
    assert report["binding"]["source_manifest_sha256"] == source["manifest_sha256"]
    assert report["binding"]["identity"] == source["identity"]
    assert report["binding"]["asset"]["asset_extraction_sha256"] == sha(asset_path)
    assert report["catalog_version"] == "trajectory.p0.measurements.v4"
    assert report["base_catalog_version"] == "trajectory.p0.measurements.v2"
    assert report["status"] == "offline_not_production_delivery" and report["effective_limits"] == "unknown_not_recorded"
    totals, examples, count, observations = {}, {}, 0, 0
    with (directory/"joint_limits.jsonl").open(encoding="utf-8") as stream:
        for raw, run, env, values, dt, refs, chunks in raw_episodes(source):
            row = json.loads(next(stream))
            names = raw["metadata"]["runtime_projection"]["joint_names"]
            assert row["joint_names"] == names == report["binding"]["asset"]["joint_names"]
            assert report["binding"]["asset"]["closure"] == raw["metadata"]["resolved_cell_manifest"]["robot_profile"]["asset_closure"]
            assert (row["run_index"], row["env_index"], row["dt_s"]) == (run, env, dt)
            eid = identifier("episode", source["manifest_sha256"], raw["identity"]["job_id"], run, env)
            assert row["episode_id"] == eid and row["runner_seed"] == raw["metadata"]["seeds"][run-1]["runner_seed"]
            assert len(row["records"]) == len(names)
            for col, (name, record) in enumerate(zip(names, row["records"], strict=True)):
                lower, upper = bounds[name]
                wanted = check_record(record, values[:, col], list(range(1, len(values)+1)), lower, upper, dt, name)
                assert record["asset_id"] == sha(asset_path)
                assert record["episode_id"] == eid
                assert record["record_id"] == identifier("nominal-limit-evidence", {k: v for k, v in record.items() if k != "record_id"})
                expected_ref = {"manifest_sha256": source["manifest_sha256"], "run_index": run,
                    "channels": ["joint_pos"], "steps_half_open": [1, len(values)+1], "chunks": refs}
                assert record["evidence_ref"] == expected_ref
                for label, w in wanted["witnesses"].items():
                    step = w["step"]
                    c = next(c for c in chunks if c["metadata"]["step_start"] <= step < c["metadata"]["step_start"]+c["metadata"]["steps"])
                    idx = step-c["metadata"]["step_start"]
                    got = record["value"]["witnesses"][label]
                    assert got["joint_column"] == col
                    assert got["source"] == {**expected_ref, "steps_half_open": [step, step+1], "chunks": [
                        {"file": c["file"], "sha256": c["sha256"], "env_index": env, "rows_half_open": [idx, idx+1]}]}
                totals.setdefault(name, []).append(wanted)
                # Earliest (run,env,step,column) per predeclared category, not a pretty-example selection.
                width = upper-lower
                for idx, q in enumerate(values[:, col]):
                    q = float(q)
                    if not np.isfinite(q): continue
                    category = ("outside" if q < lower or q > upper else "inside_edge_band"
                                if q <= lower+.05*width or q >= upper-.05*width else "ordinary_inside")
                    key = [run, env, idx+1, col]
                    if category not in examples or key < examples[category]["key"]:
                        c = next(c for c in chunks if c["metadata"]["step_start"] <= idx+1 < c["metadata"]["step_start"]+c["metadata"]["steps"])
                        examples[category] = {"key": key, "joint_name": name, "manifest_sha256": source["manifest_sha256"],
                            "chunk": c["file"], "chunk_sha256": c["sha256"], "row": idx+1-c["metadata"]["step_start"],
                            "raw_q_rad_all_joint_columns": values[idx].tolist(), "nominal_bounds_rad": [lower, upper],
                            "signed_margin_rad": min(q-lower, upper-q), "position_fraction": (q-lower)/width,
                            "episode_measurement": record}
                count += 1
            observations += len(values)
        assert next(stream, None) is None
    assert report["summary"]["episode_count"] == build["episode_count"] == count//12
    expected_claims = []
    for name, values in totals.items():
        group = report["summary"]["groups"][name]
        c = {k: sum(v["counts"][k] for v in values) for k in values[0]["counts"]}
        seconds = {k: sum(v["sample_cell_seconds"][k] for v in values) for k in values[0]["sample_cell_seconds"]}
        close(group["counts"], c); close(group["sample_cell_seconds"], seconds)
        assert group["episodes"] == len(values)
        assert group["known_episodes"] == sum(v["counts"]["finite_rows"] > 0 for v in values)
        for metric in ("minimum_signed_margin_rad", "range_occupancy_fraction", "inside_edge_band_fraction", "outside_fraction"):
            close(group["episode_equal"][metric], distribution([v[metric] for v in values if v[metric] is not None]))
        for metric in ("inside_edge_band", "outside"):
            close(group["pooled_sample_fraction"][metric], c[metric+"_rows"]/c["finite_rows"])
            close(group["pooled_sample_cell_time_fraction"][metric], seconds[metric]/seconds["finite"])
        margin = group["episode_equal"]["minimum_signed_margin_rad"]["median"]
        occupancy = group["episode_equal"]["range_occupancy_fraction"]["median"]
        text = (f"{name}: {c['inside_edge_band_rows']}/{c['finite_rows']} finite recorded q samples lie inside the nominal USD 5% edge band; "
                f"{c['outside_rows']}/{c['finite_rows']} lie outside nominal bounds. "
                f"Episode-equal median minimum signed margin {margin:.6g}rad; median range occupancy {occupancy:.6g}. "
                f"Unknown samples {c['unknown_rows']}/{c['total_rows']}. Effective runtime limits unknown; no saturation or cause claim.")
        expected_claims.append({"id": identifier("nominal-limit-claim", report["binding"], report["measurement_config"], name, group), "text": text})
    assert report["optional_statements"] == expected_claims
    base = read(ROOT/"outputs/p2a"/source["source_id"]/"public.json")
    preview, receipt = read(directory/"preview.json"), read(directory/"preview_receipt.json")
    assert receipt["combined_bytes"] == (directory/"preview.json").stat().st_size <= 8192
    assert (preview.get("base_report", preview)) == base
    included = receipt["included_ids"]
    assert preview.get("optional_nominal_joint_limit_statements", []) == [c for c in expected_claims if c["id"] in included]
    assert {x["id"] for x in receipt["omitted"]} | set(included) == {c["id"] for c in expected_claims}
    result = {"status": "PASS", "source": source["source_id"], "episodes": count//12, "records_checked": count,
              "valid_recorded_rows": observations, "joint_observations": observations*12,
              "absent_example_categories": sorted({"outside", "inside_edge_band", "ordinary_inside"}-examples.keys()),
              "raw_examples": examples, "all_statements": expected_claims, "preview_receipt": receipt,
              "wall_seconds": time.perf_counter()-started}
    with (directory/"audit.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("raw_examples", "all_statements", "preview_receipt")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    sources = read(Path(__file__).parents[1]/"P2a/source_lock.json")["sources"]
    audit(next(s for s in sources if s["source_id"] == a.source), a.directory)

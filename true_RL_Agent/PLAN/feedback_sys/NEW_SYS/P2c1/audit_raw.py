"""Independent public Stair NPZ oracle; imports no production cycle/adapter code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.unit.cycle_oracle import check_records, assert_close


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values):
    ordered = sorted(values)
    def q(p):
        if not ordered:
            return None
        index = (len(ordered)-1)*p
        lo, hi = int(index), min(int(index)+1, len(ordered)-1)
        return ordered[lo]*(hi-index)+ordered[hi]*(index-lo) if hi != lo else ordered[lo]
    return {"minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "count": len(values), "q25": q(.25), "median": q(.5), "q75": q(.75),
            "mean": sum(values)/len(values) if values else None}


def raw_episodes(source):
    path = ROOT / source["manifest_relative_path"]
    assert sha(path) == source["manifest_sha256"]
    manifest = read(path)
    assert manifest["identity"] == source["identity"]
    assert manifest["identity"]["splits"] == ["public_dev"]
    locks = {c["file"]: c["sha256"] for c in source["chunks"]}
    assert set(locks) == {c["file"] for c in manifest["chunks"]}
    for run, seed in enumerate(manifest["metadata"]["seeds"], 1):
        assert seed["split"] == "public_dev"
        chunks = [c for c in manifest["chunks"] if c["metadata"]["run_index"] == run]
        pieces = {name: [] for name in ("valid", "episode_length", "support_contact_forces_w")}
        for chunk in chunks:
            assert sha(path.parent/chunk["file"]) == locks[chunk["file"]] == chunk["sha256"]
            with np.load(path.parent/chunk["file"], allow_pickle=False) as archive:
                for name in pieces:
                    pieces[name].append(archive[name].copy())
        joined = {k: np.concatenate(v) for k,v in pieces.items()}
        for env, n in enumerate(seed["episode_steps"]):
            assert joined["valid"][:,env].tolist() == [True]*n + [False]*(len(joined["valid"])-n)
            steps = joined["episode_length"][:n,env]
            assert steps.tolist() == list(range(1,n+1))
            refs = []
            for chunk in chunks:
                first = chunk["metadata"]["step_start"]
                stop = min(n-first+1, chunk["metadata"]["steps"])
                if stop > 0:
                    refs.append({"file": chunk["file"], "sha256": chunk["sha256"], "rows_half_open": [0,stop], "env_index": env})
            yield manifest, run, env, steps, joined["support_contact_forces_w"][:n,env], seed["step_dt"], refs


def audit(source, directory):
    started = time.perf_counter()
    manifest, report = read(directory/"manifest.json"), read(directory/"report.json")
    for name, info in manifest["artifacts"].items():
        assert sha(directory/name) == info["sha256"]
        assert (directory/name).stat().st_size == info["bytes"]
    assert report["catalog_version"] == "trajectory.p0.measurements.v3"
    assert report["base_catalog_version"] == "trajectory.p0.measurements.v2"
    assert report["binding"]["manifest_sha256"] == source["manifest_sha256"]
    rows = iter((json.loads(line) for line in (directory/"cycles.jsonl").read_text(encoding="utf-8").splitlines()))
    aggregate, pair_aggregate, examples = {}, {}, {}
    count = records_checked = 0
    for raw, run, env, steps, forces, dt, refs in raw_episodes(source):
        row = next(rows)
        assert (row["run_index"], row["env_index"], row["dt_s"]) == (run,env,dt)
        assert row["runner_seed"] == raw["metadata"]["seeds"][run-1]["runner_seed"]
        expected_feet, expected_pairs = check_records(row["records"], steps, forces, dt)
        names = ["robot_0.support."+x for x in raw["metadata"]["runtime_projection"]["resolved_roles"]["support_effector"]]
        foot_records = row["records"][:len(names)]
        for record in row["records"]:
            records_checked += 1
            assert record["catalog_version"] == report["catalog_version"]
            assert record["measurement_config"] == report["measurement_config"]
            assert record["evidence_ref"] == {"manifest_sha256": source["manifest_sha256"], "run_index": run,
                "channels": ["support_contact_forces_w"], "steps_half_open": [1,len(steps)+1], "chunks": refs}
        for i, (record, expectation) in enumerate(zip(foot_records, expected_feet)):
            assert record["entity_ids"] == [names[i]]
            aggregate.setdefault(names[i], []).append(expectation)
            for window in record["value"]["windows"]:
                categories = (["qualified"] if not window["reasons"] else window["reasons"])
                for category in categories:
                    if category not in examples:
                        a, b = window["onset_steps"][0], window["onset_steps"][-1]
                        lo, hi = max(0,a-4), min(len(steps),b+2)
                        examples[category] = {"run_index": run, "env_index": env, "entity": names[i],
                            "measurement_record_id": record["record_id"], "source_ref": record["evidence_ref"],
                            "window": window, "raw_steps": steps[lo:hi].tolist(),
                            "raw_current_force_xyz_n": forces[lo:hi,i].tolist(), "dt_s": dt}
            if not expectation["windows"] and "insufficient_cycles" not in examples:
                examples["insufficient_cycles"] = {"run_index": run, "env_index": env, "entity": names[i],
                    "measurement_record_id": record["record_id"], "source_ref": record["evidence_ref"],
                    "onset_steps": expectation["onset_steps"], "raw_steps": steps.tolist(),
                    "raw_current_force_xyz_n": forces[:,i].tolist(), "dt_s": dt}
        for record, expected_pair in zip(row["records"][len(names):], expected_pairs):
            pair_aggregate.setdefault(tuple(record["entity_ids"]), []).append(expected_pair)
            left, right = [next(r for r in foot_records if r["entity_ids"] == [name]) for name in record["entity_ids"]]
            assert record["value"]["reference_record_id"] == left["record_id"]
            assert record["value"]["target_record_id"] == right["record_id"]
            for window in record["value"]["windows"]:
                assert any(w["window_id"] == window["reference_window_id"] and not w["reasons"] for w in left["value"]["windows"])
                for ci, wid, target_step in zip(window["target_cycle_indices"], window["target_qualification_window_ids"],
                                                [x[0] for x in window["target_onset_steps_by_cycle"] if len(x)==1]):
                    if ci is not None:
                        assert right["value"]["complete_cycles"][ci]["onset_steps"][0] == target_step
                        assert any(w["window_id"] == wid and not w["reasons"] and ci in w["cycle_indices"] for w in right["value"]["windows"])
                if not window["reasons"] and "qualified_phase" not in examples:
                    examples["qualified_phase"] = {"run_index": run, "env_index": env,
                        "record": record, "reference_foot_record": left, "target_foot_record": right,
                        "raw_steps": steps.tolist(), "raw_current_force_xyz_n": forces.tolist(), "dt_s": dt}
        count += 1
    assert next(rows, None) is None and count == manifest["episode_count"] == report["summary"]["episode_count"]
    for entity, values in aggregate.items():
        group = report["summary"]["groups"]["contact_cycles:"+entity]
        periods = [statistics.median([(v["cycles"][c][1]-v["cycles"][c][0])*dt for c in v["qualified_cycle_indices"]])
                   for v in values if v["qualified_cycle_indices"]]
        fractions = [v["qualified_span_fraction"] for v in values if v["qualified_span_fraction"] is not None]
        edges = sum(v["known_adjacent_edges"] for v in values)
        counts = {key: sum(v[key] for v in values) for key in ("covered_unique_edges", "known_adjacent_edges", "known_rows", "unknown_rows")}
        counts.update(complete_cycles=sum(len(v["cycles"]) for v in values),
                      qualified_unique_cycles=sum(len(v["qualified_cycle_indices"]) for v in values),
                      candidate_windows=sum(len(v["windows"]) for v in values),
                      qualified_windows=sum(sum(not w["reasons"] for w in v["windows"]) for v in values))
        reasons = Counter()
        for v in values:
            reasons.update(v["rejected_window_reasons"])
        assert_close(group, {"episodes": count, "episodes_with_qualified_windows": len(periods),
            "unknown_rhythm_episodes": count-len(periods), "counts": counts,
            "episode_median_qualified_cycle_period_s": quantiles(periods),
            "episode_equal_qualified_span_fraction": quantiles(fractions),
            "pooled_qualified_span_fraction": counts["covered_unique_edges"]/edges if edges else None,
            "rejected_window_reasons": dict(reasons)}, "cohort/"+entity)
    for pair, values in pair_aggregate.items():
        assert_close(report["summary"]["groups"]["contact_phase:"+",".join(pair)], {
            "episodes": count, "candidate_windows": sum(len(v) for v in values),
            "qualified_windows": sum(sum(not w["reasons"] for w in v) for v in values),
            "episodes_with_qualified_phase": sum(any(not w["reasons"] for w in v) for v in values)})
    if (directory/"preview.json").exists():
        preview, receipt = read(directory/"preview.json"), read(directory/"preview_receipt.json")
        base = read(ROOT/"outputs/p2a"/source["source_id"]/"public.json")
        assert preview.get("base_report", preview) == base
        size = len(json.dumps(preview,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8"))
        assert size <= 8192 and size == receipt["combined_bytes"] == (directory/"preview.json").stat().st_size
        assert set(receipt["included_ids"]) | {r["id"] for r in receipt["omitted"]} == {s["id"] for s in report["optional_statements"]}
    result = {"status": "PASS", "source_id": source["source_id"], "episodes": count, "measurement_records_checked": records_checked,
        "chunk_sha256_checked": len(source["chunks"]), "source_lock_sha256": sha(Path(__file__).parents[1]/"P2a/source_lock.json"),
        "source_manifest_sha256": source["manifest_sha256"], "supplement_manifest_sha256": sha(directory/"manifest.json"),
        "found_example_categories": list(examples), "wall_seconds": time.perf_counter()-started,
        "oracle": "independent scalar formulas + direct locked NPZ; no production cycle/adapter imports"}
    (directory/"raw_examples.json").write_text(json.dumps(examples,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (directory/"oracle.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["k1","k2","k3"], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    lock = read(Path(__file__).parents[1]/"P2a/source_lock.json")
    source = next(s for s in lock["sources"] if s["source_id"] == args.source)
    print(json.dumps(audit(source,args.directory),indent=2))

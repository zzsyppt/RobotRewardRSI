"""Independent K4 raw-array oracle. No imports from the translator or its reducer.

This audit checks archived observations, not unrecorded forces/contact or a new
policy. Exact event metadata is an input; its physical truth is not remeasured.
"""
from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/p2b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert np.isclose(a, b, rtol=1e-12, atol=1e-12), (a, b)


def scalar(values, metric, total):
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    assert metric["valid_samples"] == len(valid)
    assert metric["unknown_samples"] == total - len(valid)
    for name, operation in (("mean", np.mean), ("minimum", np.min), ("maximum", np.max), ("sum", np.sum)):
        close(metric[name], float(operation(valid)) if len(valid) else None)


def distribution(values, actual):
    assert actual["count"] == len(values)
    for name, op in (("mean", np.mean), ("median", np.median), ("minimum", np.min), ("maximum", np.max)):
        close(actual[name], float(op(values)) if len(values) else None)
    for name, q in (("q25", .25), ("q75", .75)):
        close(actual[name], float(np.quantile(values, q)) if len(values) else None)


def main():
    started = time.perf_counter()
    lock = json.loads(Path(__file__).with_name("source_lock.json").read_bytes())
    assert sha(Path(__file__).with_name("FREEZE_ZH.md")) == lock["freeze_sha256"]
    report = {"status": "PASS", "oracle_imports_translator": False, "groups": [], "model_calls": 0, "gpu_runs": 0}
    traces = []
    for group in lock["groups"]:
        output = OUT / group["source_id"]
        card = json.loads((output / "policy_card.json").read_bytes())
        records = [json.loads(line) for line in (output / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        by_key = {(r["value"]["run_index"], r["value"]["env_index"]): r for r in records}
        oracle = {}
        sample_bins = defaultdict(list)
        raw_chunks = {}
        chunk_bytes = 0
        for source in group["sources"]:
            path = ROOT / source["manifest_path"]
            assert sha(path) == source["manifest_sha256"]
            manifest = json.loads(path.read_bytes())
            assert manifest["identity"] == source["identity"]
            d = manifest["metadata"]
            chunks, locators = [], []
            for c in manifest["chunks"]:
                p = path.parent / c["file"]
                assert sha(p) == c["sha256"] and p.stat().st_size == c["stored_bytes"]
                chunk_bytes += p.stat().st_size
                with np.load(p, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                raw_chunks[(source["manifest_sha256"], c["file"])] = arrays
                chunks.append(arrays)
                locators += [{"file": c["file"], "sha256": c["sha256"], "row": j} for j in range(len(arrays["pose"]))]
            arrays = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
            run = manifest["identity"]["run_index"]
            timing, params = d["time"], d["parameters"]
            force = "force" in d
            for env in range(d["num_envs"]):
                valid = arrays["valid"][:, env]
                a = {k: v[valid, env] for k, v in arrays.items() if k != "valid"}
                steps, pose = a["control_step"], a["pose"]
                n = len(steps)
                original = d["runtime_episode_records"][env]
                record = by_key[(run, env)]
                v = record["value"]
                assert v["coverage"]["observed_samples"] == n == original["episode_steps"]
                assert v["coverage"]["unobserved_samples"] == 0
                assert np.array_equal(steps, np.arange(1, n + 1))
                assert v["task_outcome"] == original["failure_label"]
                assert v["stratum"] == original["stratum"]
                outcome = "success" if original["success"] else "failure"
                assert v["outcome_class"] == outcome
                sample_hash = hashlib.sha256(f"p2b1|{source['manifest_sha256']}|{run}|{env}".encode()).hexdigest()
                sample_bins[(original["stratum"], outcome)].append((sample_hash, record["record_id"]))
                cond = []
                # Direct comparisons in archived dtype, separate from production functions.
                for col, pkey, upper, absolute in [(3, "recovery_velocity_mps", True, False),
                    (4, "recovery_angular_velocity_radps", True, False), (1, "recovery_tilt_rad", True, True),
                    (2, "recovery_tilt_rad", True, True), (0, "recovery_height_m", False, False)]:
                    operand = abs(pose[:, col]) if absolute else pose[:, col]
                    threshold = np.array(params[pkey], dtype=pose.dtype)
                    cond.append(operand <= threshold if upper else operand >= threshold)
                condition = np.asarray(cond).T
                stable = condition.all(axis=1)
                assert np.array_equal(condition, a["conditions"])
                assert v["stability"]["true_samples"] == int(stable.sum())
                assert v["stability"]["known_samples"] == n and v["stability"]["unknown_samples"] == 0
                assert v["final"]["stable"] == bool(stable[-1])
                assert v["final"]["failure_flag"] == bool(a["failure"][-1])
                assert v["final"]["timeout_flag"] == bool(a["timeout"][-1])
                for col, key in enumerate(d["pose_fields"]):
                    scalar(pose[:, col], v["pose"][key], n)
                    close(v["final"]["pose"][key], float(pose[-1, col]))
                close(v["speed_integral_proxy_m"], float(pose[:, 3].astype(float).sum()) * timing["step_dt_s"])
                if "planar_motion" in a:
                    for col, key in enumerate(d["planar_motion_fields"]):
                        scalar(a["planar_motion"][:, col], v["signed_xy"][key], n)
                        close(v["signed_xy"][key]["final"], float(a["planar_motion"][-1, col]))
                else:
                    assert v["signed_xy"] is None
                runs = [(bool(state), list(rows)) for state, rows in groupby(range(n), key=lambda j: stable[j])]
                assert len(v["stability"]["intervals"]) == len(runs)
                for interval, (state, indices) in zip(v["stability"]["intervals"], runs):
                    assert interval["state"] == ("true" if state else "false")
                    assert interval["steps_half_open"] == [indices[0] + 1, indices[-1] + 2]
                    assert interval["sample_count"] == len(indices)
                    close(interval["sample_cell_duration_proxy_s"], len(indices) * timing["step_dt_s"])
                    close(interval["observed_span_s"], (len(indices) - 1) * timing["step_dt_s"])
                planned = d["planned_push_counts"][env]
                recovered_steps = []
                for slot, r in enumerate(v["recoveries"]):
                    if force:
                        schedule = np.array(timing["event_scheduled_steps"])
                        complete = (a["actual_substeps"] == d["force"]["expected_substeps"]) & (a["event_steps"] == schedule) & (a["event_ends"] == schedule + d["force"]["pulse_steps"])
                        eligible = (complete & a["event_verified"])[:, :planned].all(axis=1)
                        verified = ~a["unplanned_wrench"] & a["event_verified"][:, :planned].all(axis=1)
                        anchor = int(schedule[planned - 1] + d["force"]["pulse_steps"])
                        anchor_seen = bool(complete[-1, :planned].all())
                    else:
                        anchor = int(a["event_steps"][-1, slot])
                        eligible = a["event_steps"][:, slot] >= 0
                        verified = a["event_verified"][:, slot]
                        anchor_seen = anchor >= 0
                    mask = eligible & (steps > anchor) & (steps <= anchor + timing["recovery_window_steps"])
                    count, dwell_max, found = int(mask.sum()), 0, None
                    # Independent run grouping; no production dwell state machine.
                    for is_stable, rows in groupby(range(n), key=lambda j: bool(mask[j] and stable[j])):
                        indices = list(rows)
                        if not is_stable or found is not None:
                            continue
                        qualifying = [j for j in indices[timing["required_dwell_steps"] - 1:] if verified[j]]
                        if qualifying:
                            found = int(steps[qualifying[0]])
                            dwell_max = max(dwell_max, qualifying[0] - indices[0] + 1)
                        else:
                            dwell_max = max(dwell_max, len(indices))
                    status = ("recovered" if found is not None else "anchor_not_observed" if not anchor_seen else
                        "invalid_injection" if not verified[-1] else "full_window_without_dwell" if count == timing["recovery_window_steps"] else "terminal_before_window_end")
                    assert r["status"] == status and r["recovery_step"] == found, (run, env, slot, r, status, found)
                    assert r["anchor_step"] == (anchor if anchor_seen else None)
                    assert r["scheduled_anchor_step"] == (anchor if force else timing["event_scheduled_steps"][slot])
                    assert r["required_dwell_steps"] == timing["required_dwell_steps"]
                    close(r["latency_s"], (found - anchor) * timing["step_dt_s"] if found is not None else None)
                    assert r["max_dwell_before_latch_steps"] == dwell_max
                    assert r["observed_window_samples"] == count
                    assert r["expected_window_samples"] == timing["recovery_window_steps"]
                    assert r["unobserved_window_samples"] == timing["recovery_window_steps"] - count
                    assert r["unknown_stability_samples"] == 0
                    assert r["stable_window_samples"] == int(stable[mask].sum())
                    assert r["failed_condition_samples"] == dict(zip(d["condition_fields"], (~condition[mask]).sum(axis=0).tolist()))
                    assert r["right_censored"] == (found is None and status != "full_window_without_dwell")
                    assert int(a["observed_recovery_step"][-1, slot]) == (-1 if found is None else found)
                    interruptions = []
                    for state, rows in groupby(range(n), key=lambda j: bool(mask[j] and stable[j])):
                        indices = list(rows)
                        following = indices[-1] + 1
                        if (state and following < n and mask[following] and not stable[following]
                                and (found is None or steps[following] < found)):
                            interruptions.append({"step": int(steps[following]), "preceding_dwell_steps": len(indices), "state": "false"})
                    assert r["interruptions"] == interruptions
                    recovered_steps.append(found)
                any_recovered = any(x is not None for x in recovered_steps)
                known_absent = bool(v["recoveries"]) and all(r["status"] == "full_window_without_dwell" for r in v["recoveries"])
                assert v["any_recovered"] == (True if any_recovered else False if known_absent else None)
                assert v["final_unstable_after_recovery"] == ((not bool(stable[-1])) if any_recovered else None)
                for event in v["events"]:
                    if event["windows"] is None:
                        assert not any(x["event_index"] == event["slot"] for x in original["force_events" if force else "push_events"])
                        continue
                    raw_event = next(x for x in original["force_events" if force else "push_events"] if x["event_index"] == event["slot"])
                    start = raw_event["start_control_step" if force else "control_step"]
                    end = raw_event["end_control_step"] if force else start
                    post_end = end + timing["recovery_window_steps"]
                    if event["slot"] + 1 < planned:
                        post_end = min(post_end, timing["event_scheduled_steps"][event["slot"] + 1])
                    assert event["vector_world"] == raw_event["force_world_n" if force else "observed_delta_v_world_mps"]
                    assert event["windows"]["pre"]["steps_inclusive"] == [max(1, start - int(np.ceil(.5 / timing["step_dt_s"])) + 1), start]
                    assert event["windows"]["post"]["steps_inclusive"] == [end + 1, post_end]
                    if force:
                        assert event["windows"]["during"]["steps_inclusive"] == [start + 1, end]
                        assert event["delivered_impulse_world_ns"] == raw_event["delivered_impulse_world_ns"]
                    for name, window in event["windows"].items():
                        if window is None:
                            assert not force and name == "during"
                            continue
                        lo, hi = window["steps_inclusive"]
                        expected = max(0, hi - lo + 1)
                        mask = (steps >= lo) & (steps <= hi)
                        assert window["observed_samples"] == int(mask.sum())
                        assert window["expected_samples"] == expected
                        assert window["stable_samples"] == int(stable[mask].sum())
                        for col, key in enumerate(d["pose_fields"]):
                            scalar(pose[mask, col], window["pose"][key], expected)
                    direction = event["root_along_disturbance"]
                    if "planar_motion" in a:
                        low, high = event["windows"]["post"]["steps_inclusive"]
                        mask = (steps >= low) & (steps <= high)
                        vec = np.array(event["vector_world"][:2], dtype=float)
                        dots = a["planar_motion"][mask, :2].astype(float) @ (vec / np.linalg.norm(vec))
                        close(direction["mean_velocity_mps"], float(dots.mean()) if len(dots) else None)
                    else:
                        assert direction["mean_velocity_mps"] is None and direction["valid_samples"] == 0
                oracle[record["record_id"]] = {"speed_mean": float(pose[:, 3].astype(float).mean()), "pose": pose,
                    "stable": stable, "outcome": outcome, "stratum": original["stratum"],
                    "n": n, "locators": [locators[i] for i in np.flatnonzero(valid)], "a": a, "record": record}
                # Resolve each raw span instead of accepting a syntactically valid citation.
                ref = record["evidence_ref"]
                cited_steps = []
                for span in ref["chunks"]:
                    raw = raw_chunks[(source["manifest_sha256"], span["file"])]
                    begin, end = span["rows_half_open"]
                    assert span["env_index"] == env
                    assert raw["valid"][begin:end, env].all()
                    cited_steps.extend(raw["control_step"][begin:end, env].tolist())
                assert cited_steps == steps.tolist()
        selected = [min(rows)[1] for _, rows in sorted(sample_bins.items())]
        assert card["selection"]["representatives"] == selected
        assert len(oracle) == 192
        for name, facet in card["cohorts"].items():
            candidates = [row for row in oracle.values() if name == "all" or
                          (name.startswith("outcome:") and row["outcome"] == name.split(":")[1]) or
                          (name.startswith("stratum:") and row["stratum"] == name.split(":")[1])]
            assert facet["total_episodes"] == len(candidates)
            assert set(facet["record_ids"]) == {r["record"]["record_id"] for r in candidates}
            distribution([r["speed_mean"] for r in candidates], facet["pose"]["planar_speed_mps"]["episode_means"])
            for col, key in enumerate(records[0]["value"]["pose"]):
                # JSON key order is sorted, so select the semantic source column explicitly.
                column = ["height_m", "roll_rad", "pitch_rad", "planar_speed_mps", "angular_speed_radps", "xy_drift_m"].index(key)
                distribution([float(r["pose"][:, column].astype(float).mean()) for r in candidates], facet["pose"][key]["episode_means"])
                assert facet["pose"][key]["valid_samples"] == sum(r["n"] for r in candidates)
            distribution([float(r["stable"].mean()) for r in candidates], facet["stability"]["episode_equal"])
            assert facet["stability"]["pooled_true_samples"] == sum(int(r["stable"].sum()) for r in candidates)
            assert facet["stability"]["pooled_known_samples"] == sum(r["n"] for r in candidates)
            statuses = Counter(rr["status"] for r in candidates for rr in r["record"]["value"]["recoveries"])
            assert facet["recovery_units"]["status_counts"] == dict(statuses)
            latencies = [rr["latency_s"] for r in candidates for rr in r["record"]["value"]["recoveries"] if rr["latency_s"] is not None]
            distribution(latencies, facet["recovery_units"]["latency_s_recovered_only"])
            for key in ("any_recovered", "final_stable", "final_unstable_after_recovery"):
                flags = [r["record"]["value"]["final"]["stable"] if key == "final_stable" else r["record"]["value"][key] for r in candidates]
                assert facet[key] == {"true": sum(f is True for f in flags), "valid": sum(f is not None for f in flags),
                                      "unknown": sum(f is None for f in flags), "total": len(flags)}
        public = card["public_report"]
        public_ids = {c["id"] for c in public["claims"]}
        assert len(json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()) <= 12288
        assert all(c["id"] in public_ids for c in card["claims"] if c["mandatory"])
        assert public_ids | {x["id"] for x in card["projection_audit"]["omitted"]} == {c["id"] for c in card["claims"]}
        for claim in card["claims"]:
            assert all(ref in oracle for ref in claim["record_ids"])
        example_ids = list(dict.fromkeys(selected + [w["record_id"] for w in card["selection"]["witnesses"].values() if w]))
        for key in example_ids:
            o = oracle[key]
            v = o["record"]["value"]
            steps = {1, o["n"]}
            for event in v["events"]:
                if event["windows"] is not None:
                    steps.update((event["start_step"], event["end_step"], event["end_step"] + 1))
            steps.update(r["recovery_step"] for r in v["recoveries"] if r["recovery_step"] is not None)
            traces.append({"group": group["source_id"], "run": v["run_index"], "env": v["env_index"],
                "record_id": key, "selected_by_stratum_outcome": key in selected,
                "raw_rows": [{"step": step, **o["locators"][step - 1], "env": v["env_index"],
                    "pose": o["pose"][step - 1].tolist(), "stable": bool(o["stable"][step - 1]),
                    "planar_motion": o["a"]["planar_motion"][step - 1].tolist() if "planar_motion" in o["a"] else None}
                    for step in sorted(steps) if 1 <= step <= o["n"]],
                "measurements": {k: v[k] for k in ("pose", "recoveries", "final", "task_outcome")},
                "sentences": [{"id": c["id"], "text": c["text"], "public": c["id"] in public_ids}
                    for c in card["claims"] if c["record_ids"] == [key]]})
        report["groups"].append({"source_id": group["source_id"], "episodes": len(oracle),
            "valid_samples": sum(o["n"] for o in oracle.values()), "npz_bytes": chunk_bytes,
            "selected_representatives": len(selected), "representatives_plus_witnesses": len(example_ids),
            "full_claims": len(card["claims"]), "public_claims": len(public_ids),
            "public_bytes": card["projection_audit"]["bytes"], "cohorts": len(card["cohorts"]),
            "stability": card["cohorts"]["all"]["stability"], "recovery": card["cohorts"]["all"]["recovery_units"]})
    report["wall_seconds"] = time.perf_counter() - started
    for name, value in (("independent_oracle.json", report), ("raw_to_sentence.json", traces)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

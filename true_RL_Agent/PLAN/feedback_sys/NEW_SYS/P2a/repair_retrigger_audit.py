"""Targeted correction: scan current force only, then seek three existing timelines.

No production imports, new measurement artifacts, or all-metric recomputation.
The original necessary ledger, source lock and all three cards stay untouched.
"""
import json
from pathlib import Path
import time
import numpy as np
import audit_replay as audit

ROOT = audit.ROOT
OUT = ROOT / "outputs/p2a_retrigger"


def order(w):
    return w["run"], w["env"], w["on_step"], w["entity"]


def current_episodes(source):
    path = ROOT / source["manifest_relative_path"]
    audit.old.check(audit.old.sha(path), source["manifest_sha256"], "raw source hash")
    m = audit.read(path)
    audit.old.check(m["identity"], source["identity"], "source identity")
    locks = {c["file"]: c["sha256"] for c in source["chunks"]}
    for seed in m["metadata"]["seeds"]:
        pieces = {k: [] for k in ("valid", "episode_length", "support_contact_forces_w")}
        locations = []
        for c in m["chunks"]:
            if c["metadata"]["run_index"] != seed["run_index"]: continue
            audit.old.check(audit.old.sha(path.parent / c["file"]), locks[c["file"]], "chunk hash")
            with np.load(path.parent / c["file"], allow_pickle=False) as z:
                for key in pieces: pieces[key].append(z[key])
                locations.extend({"file": c["file"], "sha256": c["sha256"], "row_zero_based": i}
                                 for i in range(len(z["valid"])))
        joined = {k: np.concatenate(v) for k, v in pieces.items()}
        for env, n in enumerate(seed["episode_steps"]):
            audit.old.check(joined["valid"][:, env].tolist(), [True]*n+[False]*(len(locations)-n), "valid prefix")
            steps = joined["episode_length"][:n, env]
            audit.old.check(steps.tolist(), list(range(1, n+1)), "actual step sequence")
            yield m, seed["run_index"], env, seed["step_dt"], steps, joined["support_contact_forces_w"][:n, env], locations[:n]


def existing_trace(source, selected):
    replay = audit.read(audit.OUT / source["source_id"] / "replay.json")
    cm_path = Path(replay["card_reference"]["manifest_path"])
    cm = audit.read(cm_path)
    with cm_path.with_name("evidence.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            s = json.loads(line)
            if s["record_type"] == "episode_summary" and (s["run_index"], s["env_index"]) == (selected["run"], selected["env"]):
                break
        else: raise AssertionError("selected episode summary absent")
    timeline = s["timeline_ref"]
    path = Path(cm["source_files"][timeline["file_id"]]["evidence_path"])
    with path.open("rb") as stream:
        stream.seek(timeline["byte_offset"])
        records = [json.loads(line) for line in stream.read(timeline["byte_length"]).splitlines()]
    key = selected["entity"] + ".marker"
    measurement = next(r for r in records if r["record_type"] == "measurement" and r["key"] == key)
    verified = []
    for label, state in (("off_before", "false"), ("on", "true"), ("off_after", "false")):
        expected = {**selected[label], "state": state}
        interval = next(r for r in records if r["record_type"] == "state_interval" and r["key"] == key
                        and r["value"]["steps_half_open"] == expected["steps_half_open"])
        audit.old.check(interval["value"], expected, label + " emitted interval")
        compact_interval = next(v for v in measurement["value"]["intervals"] if v["steps_half_open"] == expected["steps_half_open"])
        audit.old.check(compact_interval, expected, label + " measurement interval")
        verified.append(interval)
    for step, before, after in ((selected["on_step"], False, True), (selected["on_step"]+1, True, False)):
        event = next(r for r in records if r["record_type"] == "event" and r["key"] == key + ".transition"
                     and r["value"]["step"] == step)
        audit.old.check(event["value"], {"step": step, "before": before, "after": after,
            "confirmation_time_s": step*selected["dt_s"], "physical_event_range_s": None}, "observed transition")
        verified.append(event)
    chunks = {c["file"]: c for c in audit.read(ROOT / source["manifest_relative_path"])["chunks"]}
    for r in verified:
        ref = r["evidence_ref"]
        audit.old.check(ref["manifest_sha256"], source["manifest_sha256"], "reference source")
        audit.old.check(ref["run_index"], selected["run"], "reference run")
        audit.old.check(ref["channels"], ["support_contact_forces_w"], "reference channel")
        observed_steps = []
        for span in ref["chunks"]:
            c = chunks[span["file"]]
            audit.old.check(span["sha256"], c["sha256"], "reference chunk hash")
            audit.old.check(span["env_index"], selected["env"], "reference environment")
            first, stop = span["rows_half_open"]
            observed_steps.extend(c["metadata"]["step_start"]+i for i in range(first, stop))
        audit.old.check(observed_steps, list(range(*ref["steps_half_open"])), "reference rows cover stated steps")
    return {"episode_summary_id": s["record_id"], "timeline_ref": timeline,
            "numeric_bytes_read": timeline["byte_length"], "measurement_id": measurement["record_id"],
            "three_intervals_two_transitions_checked": True, "records": verified}


def main():
    started = time.perf_counter()
    original = audit.read(audit.OUT / "necessary_raw.json")
    index = audit.read(Path(__file__).with_name("verification.json"))
    # Stream hash only: retain previous full independent numerical checks.
    preserved = []
    for source_index in index["sources"]:
        for ref in source_index["files"]:
            audit.old.check(audit.old.sha(ROOT / ref["path"]), ref["sha256"], "original production artifact unchanged")
            preserved.append(ref)
    results = []
    for source in audit.read(audit.LOCK)["sources"]:
        name = source["source_id"]
        pre = next(s for s in original["sources"] if s["source_id"] == name)
        old_pair = tuple(pre["counterexamples"]["single_current_retrigger"])
        candidates = []; old_short = None; selected = None; anchors = None; scanned = rows = 0
        for m, run, env, dt, steps, forces, locations in current_episodes(source):
            entities = ["robot_0.support." + e for e in m["metadata"]["runtime_projection"]["resolved_roles"]["support_effector"]]
            witnesses = audit.single_current_on_witnesses(steps, forces, entities, dt)
            if (run, env) == old_pair:
                old_short = witnesses
                assert witnesses and not any(w["qualifies_long_off"] for w in witnesses)
            for w in witnesses:
                if not w["qualifies_long_off"]: continue
                row = {"run": run, "env": env, **w}
                candidates.append(row)
                if selected is None or order(row) < order(selected):
                    selected = row
                    start, stop = w["off_before"]["steps_half_open"][0], w["off_after"]["steps_half_open"][1]
                    entity_index = w["entity_index"]
                    anchors = [{"step": int(step), **locations[i], "env_index": env,
                        "force_xyz_n": forces[i, entity_index].astype(float).tolist(),
                        "force_norm_n": float(np.linalg.norm(forces[i, entity_index].astype(float))),
                        "current_marker": bool(np.linalg.norm(forces[i, entity_index].astype(float)) > 1)}
                        for i, step in enumerate(steps) if max(1, start-1) <= step <= stop]
            scanned += 1; rows += len(steps)
        candidates.sort(key=order)
        assert selected == candidates[0]
        old_ids = sorted(pre["examples"])
        selected_id = f"r{selected['run']}.e{selected['env']}"
        new_ids = sorted(set(old_ids) | {selected_id})
        corrected_cases = dict(pre["counterexamples"])
        corrected_cases["isolated_current_on"] = corrected_cases.pop("single_current_retrigger")
        corrected_cases["single_current_retrigger"] = [selected["run"], selected["env"]]
        result = {"source_id": name, "manifest_sha256": source["manifest_sha256"],
            "scanned_episodes": scanned, "scanned_valid_rows": rows,
            "candidate_count": len(candidates), "all_qualified_candidates_sorted": candidates,
            "ordering": ["run", "env", "on_step", "entity stable string ID"],
            "selected_global_minimum": selected, "selected_raw_anchors": anchors,
            "old_short_on": {"run": old_pair[0], "env": old_pair[1], "witnesses": old_short,
                "classification": "valid isolated current-on; original long-off classification invalid",
                "max_adjacent_off_sum_s": max(w["off_before"]["sample_cell_duration_proxy_s"]+w["off_after"]["sample_cell_duration_proxy_s"] for w in old_short)},
            "sample_relationship": {"original_sample_ids": old_ids, "corrected_union_sample_ids": new_ids,
                "added_sample_ids": sorted(set(new_ids)-set(old_ids)), "removed_sample_ids": [],
                "selected_already_in_original_samples": selected_id in old_ids,
                "qualified_candidates_in_original_samples": [list(order(w)) for w in candidates if f"r{w['run']}.e{w['env']}" in old_ids]},
            "corrected_counterexamples": corrected_cases,
            "existing_production_trace": existing_trace(source, selected)}
        results.append(result)
        print(json.dumps({"source": name, "candidates": len(candidates), "minimum": selected,
            "old_samples": len(old_ids), "corrected_union": len(new_ids)}, ensure_ascii=False), flush=True)
    result = {"schema_version": "trajectory.p2a.retrigger_audit_correction.v1", "passed": True,
        "qualification": audit.RETRIGGER_QUALIFICATION,
        "qualification_status": "clarified after independent review; targeted audit correction, not predeclared or unseen confirmation",
        "source_lock_sha256": audit.old.sha(audit.LOCK),
        "original_necessary_raw_sha256": audit.old.sha(audit.OUT / "necessary_raw.json"),
        "original_audit_sha256": {n: audit.old.sha(audit.OUT / (n + "_audit.json")) for n in ("k1", "k2", "k3")},
        "preserved_production_artifacts": preserved, "sources": results,
        "scope": "only current-force selector and three existing episode timelines checked; no production rebuild or all-metric rerun",
        "wall_seconds": time.perf_counter()-started}
    audit.write(OUT / "correction.json", result)
    print("PASS targeted retrigger correction", result["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()

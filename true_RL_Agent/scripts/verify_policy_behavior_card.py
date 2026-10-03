#!/usr/bin/env python3
"""Independent P0-frozen raw-array and all-claim audit of an offline P1b card.

Only stdlib and NumPy. Does not import the producer, templates, or P0 oracle.
This numerical/trace audit complements a separate review of actual wording.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"source_lock.json": "c3f0f643eaead24df15da92c5247375c1a32c8d383639186587a02131d7fb412",
          "necessary_facts.json": "ce3a6115793eb4e182f3bb5641f163dc9d7858e3f4682b9ee99806ab637d7c74",
          "e1_selection.json": "4f45e1f9138a322a4f369a6ea3a287dc50c7de74519b4af5dca8208edeb777fc"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not expected.keys() <= actual.keys():
            raise AssertionError(label + ": missing fields")
        for k, v in expected.items():
            check(actual[k], v, label + "." + str(k))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(label + ": list length")
        for i, v in enumerate(expected):
            check(actual[i], v, label + f"[{i}]")
    elif isinstance(expected, float):
        if actual is None or not math.isclose(actual, expected, abs_tol=1e-8, rel_tol=1e-10):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def quantiles(values):
    x = np.asarray(values, dtype=float)
    return {"count": len(values), "minimum": float(min(x)) if len(x) else None,
            "maximum": float(max(x)) if len(x) else None,
            "median": float(np.percentile(x, 50)) if len(x) else None,
            "q25": float(np.percentile(x, 25)) if len(x) else None,
            "q75": float(np.percentile(x, 75)) if len(x) else None,
            "mean": float(sum(x) / len(x)) if len(x) else None}


def runs(values, dt, state):
    result = []
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[stop] == values[start]:
            stop += 1
        if values[start] == state:
            left = bool(state == -1 or start == 0 or values[start - 1] == -1)
            right = bool(state == -1 or stop == len(values) or values[stop] == -1)
            result.append({"steps_half_open": [start + 1, stop + 1], "sample_count": stop - start,
                           "sample_cell_duration_proxy_s": (stop - start) * dt,
                           "observed_span_s": (stop - start - 1) * dt,
                           "left_censored": left, "right_censored": right, "complete_run": not (left or right)})
        start = stop
    return result


def run_stats(values, dt, state):
    r = runs(values, dt, state)
    return {"count": len(r), "complete_count": sum(x["complete_run"] for x in r),
            "left_censored_count": sum(x["left_censored"] for x in r),
            "right_censored_count": sum(x["right_censored"] for x in r),
            "censored_count": sum(not x["complete_run"] for x in r),
            "complete_observed_spans_s": [x["observed_span_s"] for x in r if x["complete_run"]],
            "longest": max(r, key=lambda x: x["sample_count"], default=None),
            "shortest_samples": min((x["sample_count"] for x in r), default=None)}


def raw_episodes(source):
    """Read each locked chunk once per source; keep arrays for one run at a time."""
    path = ROOT / source["manifest_relative_path"]
    check(sha(path), source["manifest_sha256"], "raw manifest hash")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    check(manifest["identity"], source["identity"], "frozen identity")
    check(manifest["identity"]["splits"], ["public_dev"], "permission")
    locks = {c["file"]: c["sha256"] for c in source["chunks"]}
    for run_index, seed in enumerate(manifest["metadata"]["seeds"], 1):
        check(seed["split"], "public_dev", "seed permission")
        chunks = [c for c in manifest["chunks"] if c["metadata"]["run_index"] == run_index]
        pieces = {}
        origin = None
        for chunk in chunks:
            file = path.parent / chunk["file"]
            check(sha(file), locks[chunk["file"]], "raw chunk hash")
            with np.load(file, allow_pickle=False) as archive:
                for name in archive.files:
                    if name == "terrain_env_origins_w":
                        origin = archive[name].copy()
                    elif name != "terrain_types":
                        pieces.setdefault(name, []).append(archive[name].copy())
        joined = {name: np.concatenate(parts) for name, parts in pieces.items()}
        for env, outcome in enumerate(seed["outcomes"]):
            n = seed["episode_steps"][env]
            valid = joined["valid"][:, env]
            check(valid.tolist(), [True] * n + [False] * (len(valid) - n), "valid prefix")
            arrays = {name: values[:n, env].astype(float) for name, values in joined.items()}
            check(arrays["episode_length"].tolist(), [float(i) for i in range(1, n + 1)], "step sequence")
            yield manifest, run_index, env, seed["step_dt"], origin[env].astype(float), outcome, arrays


def audit_episode(s, arrays, dt, origin, outcome, entities):
    """Independently check every compact metric against frozen raw readouts."""
    n = len(arrays["episode_length"])
    check(s["task"]["outcome"], outcome, "archived outcome")
    check(s["task"]["cell_id"], outcome["evaluation_cell_id"], "cell")
    stratum = outcome["bin_id"].removeprefix("public_dev.").rsplit("_level_", 1)[0]
    check(s["task"]["stratum_id"], stratum, "stratum")
    check(s["coverage"]["valid_rows"], n, "rows")
    p = arrays["base_pos_w"]
    q = arrays["base_quat_w"]
    force = arrays["support_contact_forces_w"]
    speed = np.sqrt(np.square(arrays["support_lin_vel_w"][:, :, :2]).sum(axis=2))
    # This real-data gate is deliberately limited to the frozen finite archive.
    # Missing-channel/gap semantics are separately covered by synthetic gates.
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise AssertionError("unexpected nonfinite frozen channel " + name)
    norm_q = np.sqrt(np.square(q).sum(axis=1))
    if np.any(abs(norm_q - 1) > 1e-4):
        raise AssertionError("quaternion qualification")
    q = q / norm_q[:, None]
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    norm = np.sqrt(np.square(force).sum(axis=2))
    marker = norm > 1
    low = np.full(n, -1, dtype=int)
    width = math.ceil(.5 / dt)
    x = p[:, 0] - origin[0]
    for j in range(width, n):
        low[j] = int(max(x[j-width:j+1]) - x[j-width] < .02)
    moving = (speed >= .05).any(axis=1)
    tracks = {"low_progress": low, "body_up_negative": (up < 0).astype(int)}
    for key, component in (("low_progress_any_foot_moving", moving), ("low_progress_all_feet_low_speed", ~moving)):
        # Scalar truth table, independent of producer np.where implementation.
        tracks[key] = np.array([0 if a == 0 or not b else (1 if a == 1 and b else -1)
                                for a, b in zip(low, component, strict=True)], dtype=int)
    metrics = s["metrics"]
    checked = set()

    def scalar(key, value, valid=n, total=n):
        check(metrics[key], {"value": value, "valid_samples": valid, "unknown_samples": total-valid,
                            "total_samples": total, "unobserved_control_steps": 0}, key)
        checked.add(key)

    def ratio(key, numerator, denominator, valid=n, value="compute"):
        expected = numerator / denominator if abs(denominator) > 1e-12 else None
        if value != "compute":
            expected = value
        check(metrics[key], {"value": expected, "numerator": numerator, "denominator": denominator,
                             "valid_samples": valid, "unknown_samples": n-valid,
                             "total_samples": n, "unobserved_control_steps": 0}, key)
        checked.add(key)

    def track(key, values):
        true, false, unknown = (int((values == state).sum()) for state in (1, 0, -1))
        ratio(key, true * dt, (true + false) * dt, true + false)
        check(metrics[key]["presence"], True if true else (None if unknown else False), key + ".presence")
        check(metrics[key]["runs"], run_stats(values, dt, 1), key + ".runs")
        if metrics[key]["onset_count"] is not None:
            check(metrics[key]["onset_count"], int(((values[1:] == 1) & (values[:-1] == 0)).sum()), key + ".onsets")
            check(metrics[key]["offset_count"], int(((values[1:] == 0) & (values[:-1] == 1)).sum()), key + ".offsets")
        if "off_runs" in metrics[key]:
            check(metrics[key]["off_runs"], run_stats(values, dt, 0), key + ".off_runs")

    scalar("net_progress_m", float(x[-1] - x[0]))
    scalar("complete_path_xy_m", float(np.sqrt(np.square(np.diff(p[:, :2], axis=0)).sum(axis=1)).sum()), n-1, n-1)
    drawdown = np.maximum.accumulate(x) - x
    scalar("max_drawdown_m", float(max(drawdown)))
    scalar("minimum_up_z", float(min(up)))
    scalar("terminal_time_s", n * dt)
    for key, values in tracks.items():
        track(key, values)
    body_offsets = arrays["support_pos_w"] - p[:, None, :]
    relative = np.empty_like(body_offsets)
    for j, (w, qx, qy, qz) in enumerate(q):
        rot = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-w*qz), 2*(qx*qz+w*qy)],
                        [2*(qx*qy+w*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-w*qx)],
                        [2*(qx*qz-w*qy), 2*(qy*qz+w*qx), 1-2*(qx*qx+qy*qy)]])
        relative[j] = body_offsets[j] @ rot
    for i, entity in enumerate(entities["support"]):
        track(entity + ".marker", marker[:, i].astype(int))
        vals = ((~marker[:, i]) & (speed[:, i] < .05)).astype(int)
        track(entity + ".low_speed_off", vals)
        longest = run_stats(vals, dt, 1)["longest"]
        scalar(entity + ".low_speed_off.long_observed", bool(longest and longest["sample_count"] * dt >= .5))
        if "support_contact_forces_w_history" in arrays:
            history = (np.sqrt(np.square(arrays["support_contact_forces_w_history"][:, :, i]).sum(axis=2)) > 1).any(axis=1)
            track(entity + ".history_diagnostic", history.astype(int))
            ratio(entity + ".history_diagnostic.history_on_current_off", int((history & ~marker[:, i]).sum()) * dt, n * dt)
        else:
            track(entity + ".history_diagnostic", np.full(n, -1))
            ratio(entity + ".history_diagnostic.history_on_current_off", 0.0, 0.0, 0)
            check(metrics[entity + ".history_diagnostic.history_on_current_off"]["presence"], None, "missing history presence")
        for field, channel in (("norm_share", norm), ("signed_fz_share", force[:, :, 2])):
            sums = channel.sum(axis=0)
            den = float(sum(sums))
            value = float(sums[i] / den) if abs(den) > 1e-12 else None
            ratio(entity + "." + field, float(sums[i] * dt), float(sum(sums * dt)), value=value)
            check(metrics[entity + "." + field]["sample_force_sum_n"], float(sums[i]), "force sum")
        for axis, name in enumerate("xyz"):
            scalar(entity + ".relative_" + name + "_range_m", float(np.ptp(relative[:, i, axis])))
    codes = sum(marker[:, i].astype(int) * 2**i for i in range(marker.shape[1]))
    for code in range(2**marker.shape[1]):
        count = int((codes == code).sum())
        ratio("combination." + str(code), count * dt, n * dt)
        check(metrics["combination." + str(code)]["presence"], count > 0, "combination presence")
    for i, entity in enumerate(entities["joints"]):
        scalar(entity + ".pose.range_rad", float(np.ptp(arrays["joint_pos"][:, i])))
    for i, entity in enumerate(entities["forbidden"]):
        current = np.sqrt(np.square(arrays["forbidden_contact_forces_w"][:, i]).sum(axis=1)) > 1
        track(entity + ".current_marker", current.astype(int))
        if "forbidden_contact_forces_w_history" in arrays:
            history = (np.sqrt(np.square(arrays["forbidden_contact_forces_w_history"][:, :, i]).sum(axis=2)) > 1).any(axis=1)
            track(entity + ".history_diagnostic", history.astype(int))
            ratio(entity + ".history_diagnostic.history_on_current_off", int((history & ~current).sum()) * dt, n * dt)
        else:
            track(entity + ".history_diagnostic", np.full(n, -1))
            ratio(entity + ".history_diagnostic.history_on_current_off", 0.0, 0.0, 0)
            check(metrics[entity + ".history_diagnostic.history_on_current_off"]["presence"], None, "missing history presence")
    if checked != set(metrics):
        raise AssertionError("unverified metric keys: " + str(set(metrics) - checked))
    # Independently check the newly introduced activity intervals, not just counts.
    for key in ("low_progress_any_foot_moving", "low_progress_all_feet_low_speed"):
        record = s["activity"]["value"][key]
        vals = tracks[key]
        check(record, {"true_count": int((vals == 1).sum()), "false_count": int((vals == 0).sum()),
                       "unknown_count": int((vals == -1).sum())}, key)
        for state, label in ((1, "true"), (0, "false"), (-1, "unknown")):
            check([r for r in record["intervals"] if r["state"] == label], runs(vals, dt, state), key + label)
        longest = max(runs(vals, dt, 1), key=lambda r: r["sample_count"], default=None)
        if longest is None:
            check(record["longest_true_diagnostics"], None, key + ".diagnostics")
        else:
            a, b = longest["steps_half_open"]
            high = speed[a-1:b-1] >= .05
            check(record["longest_true_diagnostics"], {
                "interval": longest, "entity_ids": entities["support"],
                "xy_ge_threshold_counts": high.sum(axis=0).tolist(),
                "valid_counts": [b-a] * len(entities["support"]),
                "same_entity_ge_threshold_every_record": high.all(axis=0).tolist(),
            }, key + ".diagnostics")
    return {"metrics_checked": len(checked), "activity": {k: [int((v == state).sum()) for state in (1, 0, -1)]
            for k, v in tracks.items() if k.startswith("low_progress")}}


def audit_groups(groups, summaries, primary_sha):
    primary = [s for s in summaries.values() if s["source_manifest_sha256"] == primary_sha]
    checks = 0
    for group in groups.values():
        name = group["group"]
        expected = primary
        if name in ("success", "failure"):
            expected = [s for s in primary if s["task"]["outcome"]["success"] == (name == "success")]
        elif name.startswith("outcome:"):
            expected = [s for s in primary if s["task"]["outcome"]["primary_reason"] == name.split(":", 1)[1]]
        elif name != "all":
            facet, _, result = name.partition("/")
            field, value = facet.split(":", 1)
            expected = [s for s in primary if s["task"][field] == value
                        and (not result or s["task"]["outcome"]["success"] == (result == "success"))]
        check(sorted(group["member_ids"]), sorted(s["record_id"] for s in expected), "member identity " + name)
        check(group["episode_count"], len(expected), "group count")
        check(group["recorded_rows"], sum(s["coverage"]["valid_rows"] for s in expected), "group rows")
        check(group["outcome_counts"], dict(Counter(s["task"]["outcome"]["primary_reason"] for s in expected)), "outcome counts")
        for key, actual in group["metrics"].items():
            rows = [s["metrics"][key] for s in expected if key in s["metrics"]]
            values = [r["value"] for r in rows if r["value"] is not None]
            check(actual, {"total_episode_count": len(expected), "processed_episode_count": len(rows),
                           "valid_episode_count": len(values), "unknown_episode_count": len(rows)-len(values),
                           "unprocessed_episode_count": len(expected)-len(rows),
                           "valid_samples": sum(r["valid_samples"] for r in rows),
                           "unknown_samples": sum(r["unknown_samples"] for r in rows),
                           "total_processed_samples": sum(r["total_samples"] for r in rows),
                           "unobserved_control_steps": sum(r["unobserved_control_steps"] for r in rows),
                           "episode_equal": quantiles(values)}, name + ":" + key)
            if rows and rows[0]["kind"] == "ratio":
                num, den = sum(r["numerator"] for r in rows), sum(r["denominator"] for r in rows)
                check(actual["pooled"], {"numerator": num, "denominator": den,
                                       "value": num/den if abs(den)>1e-12 else None}, key + " pooled")
            if rows and ("presence" in rows[0] or rows[0]["kind"] == "presence"):
                states = [r["presence"] if "presence" in r else r["value"] for r in rows]
                yes, no, unknown = (sum(v is state for v in states) for state in (True, False, None))
                check(actual["observed_presence"], {"numerator": yes, "valid_denominator": yes+no,
                    "unknown_episode_count": unknown, "unprocessed_episode_count": len(expected)-len(rows),
                    "total_episode_count": len(expected)}, key + " presence")
            for run_key in ("runs", "off_runs"):
                if rows and run_key in rows[0]:
                    data = [r[run_key] for r in rows]
                    totals = {k: sum(r[k] for r in data) for k in ("count", "complete_count", "left_censored_count", "right_censored_count", "censored_count")}
                    totals["complete_observed_span_s"] = quantiles([v for r in data for v in r["complete_observed_spans_s"]])
                    check(actual[run_key], totals, key + " runs")
            checks += 1
    return checks


def follow(record, path):
    for part in path:
        record = record[part]
    return record


def audit_references(summaries, records, files, source_manifests):
    total = 0
    opened = {k: Path(v["evidence_path"]).open("rb") for k, v in files.items()}
    try:
        for summary in summaries.values():
            for key, pointer in summary["references"].items():
                if pointer.get("local"):
                    records[pointer["record_id"]] = summary["activity"]
                    continue
                f = opened[pointer["file_id"]]
                f.seek(pointer["byte_offset"])
                r = json.loads(f.read(pointer["byte_length"]))
                check(r["episode_id"], summary["episode_id"], "ref episode")
                if key == "header":
                    check(r["source"]["manifest_sha256"], summary["source_manifest_sha256"], "header source")
                else:
                    check(r["record_id"], pointer["record_id"], "ref record ID")
                    check(r["key"], key, "ref measurement key")
                    ref = r["evidence_ref"]
                    check(ref["manifest_sha256"], summary["source_manifest_sha256"], "ref source")
                    check(ref["run_index"], summary["run_index"], "ref run")
                    chunks = {c["file"]: c for c in source_manifests[ref["manifest_sha256"]]["chunks"]}
                    steps = []
                    for span in ref["chunks"]:
                        chunk = chunks[span["file"]]
                        check(span["sha256"], chunk["sha256"], "ref chunk hash")
                        check(span["env_index"], summary["env_index"], "ref environment")
                        check(chunk["metadata"]["run_index"], summary["run_index"], "chunk run")
                        a, b = span["rows_half_open"]
                        if not 0 <= a < b <= chunk["metadata"]["steps"]:
                            raise AssertionError("reference row bounds")
                        steps.extend(range(chunk["metadata"]["step_start"] + a, chunk["metadata"]["step_start"] + b))
                    check(steps, list(range(*ref["steps_half_open"])), "ref step/row exact mapping")
                # Retain only records that the final claims use (pre-created keys).
                if pointer["record_id"] in records:
                    records[pointer["record_id"]] = r
                total += 1
    finally:
        for f in opened.values():
            f.close()
    return total


def necessary_coverage(necessary, card, summaries):
    basic = {"source_entity_clock": ["episode_scope"], "archived_result_and_endpoint": ["episode_scope", "episode_terminal"],
             "root_progress": ["episode_root"], "body_up": ["episode_attitude"], "force_semantics": ["episode_force"],
             "contact_combinations": ["episode_combinations"], "body_frame_anchors": ["episode_relative"],
             "short_terminal_events": ["episode_terminal", "episode_base_runs"],
             "history_and_missing_channels": ["episode_history"], "required_inverted_counterexample": ["inverted_before", "episode_low_speed", "episode_attitude"],
             "required_return_counterexample": ["successful_return"]}
    coverage = []
    for episode in necessary["episodes"]:
        ref = episode["raw_reference"]
        matches = [s for s in summaries.values() if s["source_manifest_sha256"] == ref["manifest_sha256"]
                   and (s["run_index"], s["env_index"]) == (ref["run_index"], ref["env_index"])]
        if len(matches) != 1:
            raise AssertionError("necessary episode missing")
        summary = matches[0]
        for fact in episode["facts"]:
            template = fact["template"]
            target = basic.get(template)
            if target is None:
                target = ["episode_marker"] if template.endswith(".markers") else ["episode_low_speed"]
            claims = [c for c in card["claims"] if c["scope"] == summary["display_name"] and c["template_id"] in target
                      and (not template.endswith((".markers", ".low_speed_off")) or c["literals"].get("entity", "").endswith(template.split(".")[0]))]
            if not claims or not set(target) <= {c["template_id"] for c in claims}:
                raise AssertionError("necessary fact omitted from reading: " + episode["episode_id"] + "." + template)
            coverage.append({"fact_id": episode["episode_id"] + "." + template,
                             "claim_ids": [c["claim_id"] for c in claims], "status": "core_rendered_with_expandable_detail",
                             "detail_note": "Raw normalized-quaternion diagnostics, all transition rows, force sums and secondary diagnostics remain in referenced P1a records; not every scalar is restated in Chinese."})
    coverage.append({"fact_id": necessary["cohort_facts"][0]["fact_id"], "status": "core_rendered_with_expandable_detail",
                     "claim_ids": [c["claim_id"] for c in card["claims"] if c["scope"] in ("all", "success", "failure") and c["template_id"] in ("cohort_scope", "ratio")],
                     "detail_note": "Full combination histogram and sample force sums reside in independently recomputed cohort metrics; main view renders marker/force shares, denominators and results."})
    check(len(coverage), 247, "necessary count")
    return coverage


def verify(directory: Path) -> dict:
    started = time.perf_counter()
    p0 = ROOT / "PLAN/feedback_sys/NEW_SYS/P0"
    for name, digest in FROZEN.items():
        check(sha(p0/name), digest, "frozen " + name)
    lock = json.loads((p0/"source_lock.json").read_text(encoding="utf-8"))
    necessary = json.loads((p0/"necessary_facts.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    check(manifest["complete"], True, "completed build")
    for path, digest in manifest["code_sha256"].items():
        check(sha(ROOT/path), digest, "executed production code " + path)
    for name, item in manifest["artifacts"].items():
        check(sha(directory/name), item["sha256"], "artifact hash")
        check((directory/name).stat().st_size, item["bytes"], "artifact bytes")
    card = json.loads((directory/"policy_card.json").read_text(encoding="utf-8"))
    records, summaries, groups = {}, {}, {}
    for line in (directory/"evidence.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["record_type"] == "claim":
            continue
        records[r["record_id"]] = r
        if r["record_type"] == "episode_summary":
            summaries[r["episode_id"]] = r
        elif r["record_type"] == "cohort_summary":
            groups[r["group"]] = r
    by_source = {s["source_manifest_sha256"] for s in summaries.values()}
    raw_reports, raw_manifests = [], {}
    for source in lock["sources"]:
        if source["manifest_sha256"] not in by_source:
            continue
        selected = {(s["run_index"], s["env_index"]): s for s in summaries.values() if s["source_manifest_sha256"] == source["manifest_sha256"]}
        for raw_manifest, run, env, dt, origin, outcome, arrays in raw_episodes(source):
            raw_manifests[source["manifest_sha256"]] = raw_manifest
            if (run, env) not in selected:
                continue
            s = selected[(run, env)]
            entities = manifest["primary_binding"]["entities"]
            report = audit_episode(s, arrays, dt, origin, outcome, entities)
            raw_reports.append({"episode_id": s["episode_id"], "display_name": s["display_name"], **report})
    check(len(raw_reports), len(summaries), "raw episode audit count")
    aggregate_checks = audit_groups(groups, summaries, manifest["primary_binding"]["manifest_sha256"])
    for file_id, info in manifest["source_files"].items():
        check(sha(info["manifest_path"]), info["manifest_sha256"], "input manifest")
        check(sha(info["evidence_path"]), file_id, "input evidence")
    references = audit_references(summaries, records, manifest["source_files"], raw_manifests)
    claim_reports = []
    md = (directory/"policy_card_zh.md").read_text(encoding="utf-8")
    primary_ids = card["policy_summary_claim_ids"]
    if len(primary_ids) > 15 or len(set(primary_ids)) != len(primary_ids) or not set(primary_ids) <= {c["claim_id"] for c in card["claims"]}:
        raise AssertionError("policy summary is not a small selection of original claims")
    for c in card["claims"]:
        resolved = {name: follow(records[o["record_id"]], o["path"]) for name, o in c["operands"].items()}
        if not resolved or c["text"] not in md or md.count(c["claim_id"]) != 1:
            raise AssertionError("sentence/evidence/reading trace missing")
        if any(p in c["text"] for p in ("D:\\", "C:\\", ".npz", "evidence_path", "manifest_path")):
            raise AssertionError("internal path in sentence")
        claim_reports.append({"claim_id": c["claim_id"], "text": c["text"], "scope": c["scope"],
                              "template_id": c["template_id"], "operands": c["operands"],
                              "numerical_trace": "resolved; cohort metrics checked against raw arrays and independent aggregation; P1a statements additionally require existing 247/309 gate",
                              "wording_review": "separate independent semantic review required; not established by template identity"})
    coverage = necessary_coverage(necessary, card, summaries)
    return {"schema_version": "trajectory.p1b.independent_audit.v1", "passed": True,
            "scope": "frozen P0 raw arrays, all compact metrics/facets, all sentence operands and necessary-fact reading mapping; separate semantic review",
            "producer_or_oracle_imports": False, "source_manifest_and_chunk_hashes_verified": True,
            "raw_episodes": len(raw_reports), "metric_checks": sum(r["metrics_checked"] for r in raw_reports),
            "aggregate_metric_checks": aggregate_checks, "references_checked": references,
            "claim_count": len(claim_reports), "necessary_fact_count": len(coverage),
            "raw_reports": raw_reports, "claim_reports": claim_reports, "necessary_coverage": coverage,
            "frozen_inputs": FROZEN, "card_manifest_sha256": sha(directory/"manifest.json"),
            "verifier_sha256": sha(Path(__file__)), "wall_seconds": time.perf_counter()-started,
            "not_verified": ["final author serialization P1c", "visual", "cross-policy generalization", "closed-loop benefit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.input)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "raw_episodes", "metric_checks", "aggregate_metric_checks", "references_checked", "claim_count", "necessary_fact_count", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()

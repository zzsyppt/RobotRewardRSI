"""Offline independent development calculation; never imports production reducers.

The earlier raw reader supplies checked NPZ arrays, not behavior labels. This
script derives the new signatures, comparison domains and coverage anew. It is
development material and cannot admit a new confirmation cohort by itself.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


SEED = "p3new-20260929"
MIN_SUCCESS = 3
MATCH_FRACTION = .5


def bin_seconds(value):
    return 0 if value <= .5 else 1 if value <= 2 else 2


def rank(row, task):
    return hashlib.sha256(f"{SEED}|{task}|{row['id']}".encode()).hexdigest(), row["id"]


def signature_stair(row, curve, rules):
    terminal = row["steps"]
    first = {}
    sustained = row["sustained_tilt_intervals"]
    if sustained:
        first["tilt"] = min(x[0] + rules["tilt_steps"] - 1 for x in sustained)
    if row["low_progress_intervals"]:
        first["low_progress"] = row["low_progress_intervals"][0][0]
    slow = [s["confirmed_step"] for spans in row["same_support_unloaded_low_motion"].values()
            for s in spans]
    if slow:
        first["unloaded_low_motion"] = min(slow)
    negative = np.flatnonzero(curve["up_z"] < 0)
    if negative.size:
        first["up_z_negative"] = int(curve["steps"][negative[0]])
    first = {k: v for k, v in first.items() if v < terminal}
    return tuple(sorted((k, bin_seconds((terminal-v)*row["dt"])) for k, v in first.items())), first


def stair_residuals(rows, curves, rules):
    signatures = {r["id"]: signature_stair(r, curves[r["id"]], rules) for r in rows}
    success = defaultdict(list)
    for r in rows:
        if r["success"]:
            success[r["cell"], r["run"]].append(r)
    records = []
    for r in rows:
        if r["success"]:
            continue
        controls = success[r["cell"], r["run"]]
        signature, confirmed = signatures[r["id"]]
        matching = sum(signatures[c["id"]][0] == signature for c in controls)
        state = ("comparison_missing" if len(controls) < MIN_SUCCESS else
                 "rule_negative" if not signature or matching/len(controls) >= MATCH_FRACTION else
                 "rule_positive")
        records.append({"episode_id": r["id"], "stratum": r["scope"],
                        "domain": [r["outcome"], r["cell"], r["run"]],
                        "state": state, "signature": signature, "confirmed": confirmed,
                        "matched_success_n": len(controls), "matching_signature_n": matching})
    return records


def push_signature(event, dt):
    first = {name: spans[0][0] for name, spans in event["violation_intervals"].items() if spans}
    return tuple(sorted((k, bin_seconds((v-event["anchor_step"])*dt)) for k, v in first.items()))


def push_residuals(rows):
    controls = defaultdict(list)
    for row in rows:
        executed = [e for e in row["events"] if e["status"] == "executed"]
        if row["success"]:
            for e in executed:
                if e["verified"] and e.get("first_step") is not None:
                    controls[row["scope"], len(executed), e["event"]].append(push_signature(e, row["dt"]))
    records = []
    for row in rows:
        if row["success"]:
            continue
        executed = [e for e in row["events"] if e["status"] == "executed"]
        windows = []
        for e in executed:
            domain = (row["scope"], len(executed), e["event"])
            pool = controls[domain]
            signature = push_signature(e, row["dt"])
            matching = sum(s == signature for s in pool)
            state = ("comparison_missing" if not e["verified"] or e.get("first_step") is None
                     or len(pool) < MIN_SUCCESS else
                     "rule_negative" if matching/len(pool) >= MATCH_FRACTION else "rule_positive")
            windows.append({"domain": domain, "state": state, "signature": signature,
                            "matched_success_n": len(pool), "matching_signature_n": matching})
        state = ("rule_positive" if any(w["state"] == "rule_positive" for w in windows) else
                 "comparison_missing" if not windows or any(w["state"] == "comparison_missing" for w in windows)
                 else "rule_negative")
        records.append({"episode_id": row["id"], "stratum": row["scope"],
                        "state": state, "windows": windows})
    return records


def coverage(rows, records, task):
    groups = defaultdict(list)
    for r in rows:
        groups[r["scope"], bool(r["success"])].append(r)
    baseline = {min(pool, key=lambda r: rank(r, task))["id"] for pool in groups.values()}
    eligible = {r["episode_id"] for r in records if r["state"] != "rule_positive"}
    candidates = [r for r in rows if r["id"] in eligible]
    supplement = {r["id"] for r in sorted(candidates, key=lambda r: rank(r, task))[:4]}
    picked = baseline | supplement
    selected = []
    for r in sorted((r for r in rows if r["id"] in picked), key=lambda r: rank(r, task)):
        n = r["steps"]
        anchors = [1, 1+(n-1)//2, n]
        windows = sorted(set((max(1, a-25), min(n, a+25)) for a in anchors))
        covered = set()
        for a, b in windows:
            covered.update(range(a, b+1))
        selected.append({"episode_id": r["id"], "windows": windows,
                         "baseline": r["id"] in baseline, "residual_supplement": r["id"] in supplement,
                         "covered_steps": len(covered), "total_steps": n,
                         "fraction": len(covered)/n})
    return {"baseline_ids": sorted(baseline), "residual_supplement_ids": sorted(supplement),
            "eligible_residual_ids": sorted(eligible), "union_residual_ids": sorted(eligible & picked),
            "unselected_residual_ids": sorted(eligible-picked), "selected": selected,
            "stratum_outcome_coverage": [
                {"stratum": scope, "success": outcome, "selected_n": sum(r["id"] in picked for r in pool),
                 "total_n": len(pool)} for (scope, outcome), pool in sorted(groups.items())],
            "time_fraction_median": float(np.median([r["fraction"] for r in selected])),
            "time_fraction_min": min(r["fraction"] for r in selected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stair", "push"], required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--raw-reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_raw_material_reader", args.raw_reader)
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    if args.task == "stair":
        rows, curves, _ = raw.stair_rows(args.manifest[0], args.rules)
        rules = json.loads(args.rules.read_text(encoding="utf-8"))["semantics"]["thresholds"]
        records = stair_residuals(rows, curves, rules)
    else:
        rows, curves, _ = raw.push_rows(args.manifest)
        records = push_residuals(rows)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate raw episode identity")
    result = {"schema_version": "feedback.p3new.independent_raw_residuals.v1",
              "purpose": "development_only_not_confirmation_admission", "task": args.task,
              "source_refs": [{"path": str(p), "sha256": raw.sha(p)} for p in [*args.manifest, args.raw_reader]],
              "episodes": len(rows), "failures": len(records), "states": dict(Counter(r["state"] for r in records)),
              "residuals": records, "sampling": coverage(rows, records, args.task)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("task", "episodes", "failures", "states")}))


if __name__ == "__main__":
    main()

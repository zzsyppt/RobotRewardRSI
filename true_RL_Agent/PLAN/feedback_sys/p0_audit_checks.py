"""Independent re-checks used by the P0 audit (2026-09-23). Read-only; writes only p0_audit_checks.json next to itself.

Reproduces: Go2 v3 public-dev fact distributions, existing-flag discrimination, per-cell failure concentration,
formal v2 -> current v3 contract-hash diff, formal checkpoint counts, Push attempt_08 case inventory,
token-estimator fit, and Stair replay cost from saved receipts. No simulator, GPU, provider or training.
"""
from __future__ import annotations
import collections, glob, json, os, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P0 = ROOT / "PLAN/feedback_sys/P0"
out = {}

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

# 1. Go2 v3 public-dev facts
facts = []
def walk(v):
    if isinstance(v, dict):
        if {"primary_reason", "episode_steps", "progress_fraction", "diagnostic_flags"} <= set(v):
            facts.append(v); return
        for c in v.values(): walk(c)
    elif isinstance(v, list):
        for c in v: walk(c)
go2 = load(ROOT / "PLAN/2new_world/P3/results/cohort_alignment_20260909/go2_12dof_qualification/evaluation/public_dev/summary.json")
walk(go2)
def q(xs):
    return [round(statistics.quantiles(xs, n=10)[i], 3) for i in (0, 4, 8)] if len(xs) > 2 else xs
prim = collections.Counter(f["primary_reason"] for f in facts)
flags = collections.Counter(fl for f in facts for fl in f["diagnostic_flags"])
per_primary = {}
for pr in prim:
    g = [f for f in facts if f["primary_reason"] == pr]
    per_primary[pr] = {"n": len(g), "steps_p10_p50_p90": q([f["episode_steps"] for f in g]),
        "steps_min_max": [min(f["episode_steps"] for f in g), max(f["episode_steps"] for f in g)],
        "progress_p10_p50_p90": q([f["progress_fraction"] for f in g]),
        "flag_rates": {fl: round(sum(fl in f["diagnostic_flags"] for f in g) / len(g), 4) for fl in flags}}
cells = collections.defaultdict(collections.Counter)
seeds = go2["task_evidence"]["summary"]["ordered_summary"]["seed_summaries"]
for s in seeds:
    for cell, lst in s["episode_facts_by_cell"].items():
        for f in lst: cells[cell][f["primary_reason"]] += 1
out["go2_v3_public_dev"] = {"episodes": len(facts), "primary": dict(prim), "flag_totals": dict(flags),
    "per_primary": per_primary, "per_cell_primary": {c: dict(v) for c, v in sorted(cells.items())},
    "forbidden_contact_progress_deciles": dict(sorted(collections.Counter(min(int(f["progress_fraction"] * 10), 9)
        for f in facts if f["primary_reason"] == "forbidden_contact").items()))}

# 2. formal v2 vs current v3 contracts
raw = P0 / "inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba"
m2 = load(raw / "resolved_cell_manifest.json"); m3 = load(P0 / "stair_current_identity.json")["go2"]["resolved_manifest"]
a, b = m2["contract_hashes"], m3["contract_hashes"]
out["formal_v2_vs_current_v3"] = {"same": sorted(k for k in a if a.get(k) == b.get(k)),
    "different": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)),
    "robot_profile_same": m2["robot_profile"] == m3["robot_profile"], "runtime_adapter_same": m2["runtime_adapter"] == m3["runtime_adapter"],
    "resolved_roles_same": m2["resolved_roles"] == m3["resolved_roles"]}
f8 = load(raw / "runner/formal-go2-stair-15round-20260901-c6ab8ba-generation-008-lane-a/train/summary.json")
p2 = load(ROOT / "PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/train/summary.json")
out["training_binding"] = {"formal_gen008a": f8["training_binding_hash"], "p2_reference": p2["training_binding_hash"],
    "formal_cell": f8["resolved_cell_identity"]["cell_variant_id"], "p2_reference_cell": p2["resolved_cell_identity"]["cell_variant_id"],
    "same_resolved_cell_manifest": f8["resolved_cell_manifest_hash"] == p2["resolved_cell_manifest_hash"]}
rb = load(P0 / "remote_baseline.json")
best = [c for c in rb["checkpoints"] if c["path"].endswith("best.pt")]
out["formal_checkpoints"] = {"best_pt": len(best), "distinct_best_hashes": len({c["sha256"] for c in best}), "all_exist": all(c["exists"] for c in best)}

# 3. Push attempt_08 case inventory (local per-episode facts)
def find_all(v, key, acc):
    if isinstance(v, dict):
        for k, c in v.items():
            if k == key: acc.append(c)
            find_all(c, key, acc)
    elif isinstance(v, list):
        for c in v: find_all(c, key, acc)
push = {}
for p in sorted(glob.glob(str(ROOT / "PLAN/2new_world/Go2_push/attempt_08/session/runner/*/eval/public_dev/summary.json"))):
    d = load(p); pe = []; find_all(d, "per_episode", pe)
    labels = collections.Counter(e.get("failure_label") for lst in pe if isinstance(lst, list) for e in lst if isinstance(e, dict))
    push[Path(p).parts[-4]] = {"episodes": sum(labels.values()), "labels": dict(labels)}
ledger = [json.loads(l) for l in (ROOT / "PLAN/2new_world/Go2_push/attempt_08/session/ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
out["push_attempt_08"] = {"runner_public_dev_summaries": push, "generation_outcomes": {r["generation_id"][-3:]: r["generation_outcome"] for r in ledger},
    "train_jobs": load(ROOT / "PLAN/2new_world/Go2_push/attempt_08/session/generation_session_report.json")["cost"]["actual"]["train_jobs"]}

# 4. token estimator fit on the nine measured inputs
cal = load(ROOT / "PLAN/2new_world/P5/audit_capacity/result.json"); pm = load(P0 / "prompt_measurements.json")
pts = [(fr["prompt_bytes"], fr["usage"]["prompt_tokens"]) for fr in cal["frames"].values()]
pts += [(fr["payload_bytes"] + fr["system_text_bytes"], fr["usage"]["prompt_tokens"]) for fr in pm["frames"]]
n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts); icpt = my - slope * mx
out["token_estimator"] = {"n": n, "linear_fit": {"intercept": round(icpt), "slope": round(slope, 4),
    "max_abs_rel_err": round(max(abs((icpt + slope * x - y) / y) for x, y in pts), 4)},
    # Old estimator parameters were removed after the audit. Original receipt
    # reservations remain historical facts; do not reconstruct a retired model.
    "historical_receipt_reservation_overhead": {
        fr["name"]: round(load(P0 / "provider_receipts" / (fr["name"] + ".json"))["estimated_input_reserved"] /
                          fr["usage"]["prompt_tokens"] - 1, 3) for fr in pm["frames"]},
    "tokens_per_byte_min_max": [round(min(y / x for x, y in pts), 4), round(max(y / x for x, y in pts), 4)]}

# 5. Stair replay cost from saved receipts
p6 = load(ROOT / "PLAN/2new_world/P6/stair_dev_evidence_01/go2_process/process.json")
out["stair_public_dev_replay_cost"] = {"p3_qualification_split_wall_s": go2["evaluation_execution_cost"]["splits"]["public_dev"]["wall_time_s"],
    "p6_regression_split_wall_s": load(ROOT / "PLAN/2new_world/P6/stair_dev_evidence_01/go2/result.json")["evaluation_execution_cost"]["splits"]["public_dev"]["wall_time_s"],
    "p6_regression_process_wall_s": round(p6["ended_at_unix"] - p6["started_at_unix"], 1), "envs_per_seed": 288, "seeds": 3}

(Path(__file__).with_suffix(".json")).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: (v if k in ("formal_checkpoints", "training_binding", "token_estimator", "stair_public_dev_replay_cost") else "...") for k, v in out.items()}, ensure_ascii=False, indent=1))

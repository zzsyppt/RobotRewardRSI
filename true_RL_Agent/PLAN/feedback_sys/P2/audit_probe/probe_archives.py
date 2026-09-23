"""Read-only P2 audit probes on the real archives (server CPU)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_current/src")
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.transition_decoder import _roll_pitch_from_wxyz
R = Path("/data/yuyangliu/AGENT/feedback_p2_20260923")
out = {}

# ---- E: provenance: parent archive identity keys
ref = json.load(open(R/"r2_parent_on/public_dev_result.json"))["task_evidence"]["numeric_evidence"]
man = load_numeric_evidence_manifest(ref)
out["parent_identity_keys"] = sorted(man["identity"].keys())
out["parent_metadata_keys"] = sorted(man["metadata"].keys())

# ---- C1: per-array raw bytes and padding fraction
per_array = {}
for chunk in man["chunks"]:
    for name, spec in chunk["arrays"].items():
        n = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        per_array[name] = per_array.get(name, 0) + n
out["parent_per_array_raw_bytes"] = dict(sorted(per_array.items(), key=lambda kv: -kv[1]))
out["parent_raw_total"] = man["raw_bytes"]; out["parent_stored_total"] = man["stored_chunk_bytes"]
valid_total = frames_total = 0
seeds = man["metadata"]["seeds"]
# choose forbidden_contact episodes in run 1
outcomes = seeds[0]["outcomes"]
out["outcome_keys"] = sorted(outcomes[0].keys())
fc = [(i, o) for i, o in enumerate(outcomes) if o.get("primary_reason") == "forbidden_contact"]
succ = [(i, o) for i, o in enumerate(outcomes) if o.get("primary_reason") == "success"]
out["run1_forbidden_count"] = len(fc); out["run1_success_count"] = len(succ)
target_env, target = fc[0]
T = int(target["episode_steps"]); W = 50
series = {k: [] for k in ("step","pitch_deg","roll_deg","forbidden_force_N","support_force_N","support_speed_mps","base_z","done","valid")}
static_seen = False
t0 = time.time()
for meta, arrays in iter_numeric_evidence(ref):
    frames_total += arrays["valid"].shape[0]*arrays["valid"].shape[1]
    valid_total += int(arrays["valid"].sum())
    if meta["run_index"] != 1: continue
    start = meta["step_start"]
    for s in range(meta["steps"]):
        step = start + s
        if step < T - W or step > T + 2: continue
        q = arrays["base_quat_w"][s, target_env].tolist()
        roll, pitch = _roll_pitch_from_wxyz(q)
        series["step"].append(step)
        series["pitch_deg"].append(round(math.degrees(pitch),1)); series["roll_deg"].append(round(math.degrees(roll),1))
        series["forbidden_force_N"].append([round(float(np.linalg.norm(v)),1) for v in arrays["forbidden_contact_forces_w"][s, target_env]])
        series["support_force_N"].append([round(float(np.linalg.norm(v)),0) for v in arrays["support_contact_forces_w"][s, target_env]])
        series["support_speed_mps"].append([round(float(np.linalg.norm(v[:2])),2) for v in arrays["support_lin_vel_w"][s, target_env]])
        series["base_z"].append(round(float(arrays["base_pos_w"][s, target_env, 2]),3))
        series["done"].append(bool(arrays["done_mask"][s, target_env])); series["valid"].append(bool(arrays["valid"][s, target_env]))
out["iter_seconds"] = round(time.time()-t0,1)
out["padding_fraction"] = 1 - valid_total/frames_total
out["probe_episode"] = {"run_index":1, "env":target_env, "outcome":target, "last_window":series}

# ---- D: training sample autocorrelation
def train_probe(manifest_path):
    ref = {"schema_version":"true_rl_agent.numeric_evidence.ref.v1","manifest_path":str(manifest_path),
           "manifest_sha256": __import__("hashlib").sha256(Path(manifest_path).read_bytes()).hexdigest()}
    m = load_numeric_evidence_manifest(ref)
    cov = m["metadata"]; terms = cov["semantics"]["terms"]
    samples = []
    for meta, arrays in iter_numeric_evidence(ref):
        if meta["kind"] != "training_sample": continue
        samples.append((meta["window"], meta["training_control_step"], {k: v for k, v in arrays.items() if k.startswith("raw_term_")}))
    res = {"planned": cov["planned_control_steps"], "windows": [(w["name"], w["start_step"], w["end_step"], w["status"]) for w in cov["windows"]], "terms": {}}
    for i, term in enumerate(terms):
        key = f"raw_term_{i:03d}"
        by_window = {}
        for w, step, arr in samples:
            by_window.setdefault(w, []).append(arr[key].astype(np.float64))
        info = {}
        for w, rows in by_window.items():
            X = np.stack(rows)  # [steps, envs]
            step_std = float(X.std(axis=0).mean())   # variation over consecutive steps, per env
            env_std = float(X.std(axis=1).mean())    # variation across envs, per step
            lag1 = None
            if X.shape[0] > 2:
                a, b = X[:-1].ravel(), X[1:].ravel()
                if a.std() > 0 and b.std() > 0: lag1 = float(np.corrcoef(a, b)[0,1])
            info[w] = {"mean": round(float(X.mean()),5), "std_over_steps_per_env": round(step_std,5), "std_over_envs_per_step": round(env_std,5), "lag1_autocorr": None if lag1 is None else round(lag1,3), "nonzero_frac": round(float((X!=0).mean()),3)}
        res["terms"][term["name"]] = info
    return res
out["train_push_go2"] = train_probe(R/"r5_train_push_go2_on/run/attempt-03177bc7139c47e0bab46317a45ff3e6/training_numeric_evidence/manifest.json")
out["train_stair_go2"] = train_probe(R/"r6_train_stair_go2_on/run/training_numeric_evidence/manifest.json")
Path("/data/yuyangliu/p2_audit/probe_archives.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("parent_identity_keys","parent_per_array_raw_bytes","parent_raw_total","parent_stored_total","padding_fraction","run1_forbidden_count","run1_success_count","iter_seconds","outcome_keys")}, indent=1))
print("PROBE_EPISODE", json.dumps(out["probe_episode"]["outcome"]))
pe = out["probe_episode"]["last_window"]
for i in range(len(pe["step"])):
    print(pe["step"][i], pe["valid"][i], pe["done"][i], "pitch", pe["pitch_deg"][i], "roll", pe["roll_deg"][i], "z", pe["base_z"][i], "forb", pe["forbidden_force_N"][i], "sup", pe["support_force_N"][i], "supv", pe["support_speed_mps"][i])
print("TRAIN_PUSH_GO2", json.dumps(out["train_push_go2"], indent=1))
print("TRAIN_STAIR_GO2", json.dumps(out["train_stair_go2"], indent=1))

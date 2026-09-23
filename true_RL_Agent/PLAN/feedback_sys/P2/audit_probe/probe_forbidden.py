"""Does the stored forbidden-body force vector carry the contact that terminated the episode?"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_current/src")
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
R = Path("/data/yuyangliu/AGENT/feedback_p2_20260923")
out = {}
for name in ("r2_parent_on", "r2_gen8_on", "r2_g1_on"):
    ref = json.load(open(R/name/"public_dev_result.json"))["task_evidence"]["numeric_evidence"]
    man = load_numeric_evidence_manifest(ref)
    seeds = man["metadata"]["seeds"]
    # terminal step per (run, env) and primary reason
    info = {}
    for si, seed in enumerate(seeds):
        for env, o in enumerate(seed["outcomes"]):
            info[(si+1, env)] = {"T": int(o["episode_steps"]), "reason": o["primary_reason"]}
    stats = {"forbidden_episodes": 0, "forb_force_nonzero_at_terminal": 0, "forb_force_nonzero_within_last5": 0,
             "manager_term_true_at_terminal": 0, "any_support_contact_at_terminal": 0, "max_forb_force_terminal_N": [],
             "nonforbidden_episodes_with_forb_force_nonzero_anywhere": 0}
    term_order = None
    forb_any = {}
    for meta, arrays in iter_numeric_evidence(ref):
        run = meta["run_index"]; start = meta["step_start"]
        term_order = meta["termination_term_order"]
        bc = term_order.index("base_contact") if "base_contact" in term_order else None
        fnorm = np.linalg.norm(arrays["forbidden_contact_forces_w"], axis=-1).max(axis=-1)  # [steps, env]
        snorm = np.linalg.norm(arrays["support_contact_forces_w"], axis=-1)  # [steps, env, nsup]
        for env in range(fnorm.shape[1]):
            T = info[(run, env)]["T"]; reason = info[(run, env)]["reason"]
            valid = arrays["valid"][:, env]
            if (fnorm[:, env][valid] > 0).any():
                forb_any[(run, env)] = True
            if reason != "forbidden_contact": continue
            if start <= T < start + meta["steps"]:
                s = T - start
                stats["forbidden_episodes"] += 1
                stats["forb_force_nonzero_at_terminal"] += int(fnorm[s, env] > 0)
                stats["max_forb_force_terminal_N"].append(float(fnorm[s, env]))
                lo = max(0, s-4)
                stats["forb_force_nonzero_within_last5"] += int((fnorm[lo:s+1, env] > 0).any())
                if bc is not None:
                    stats["manager_term_true_at_terminal"] += int(bool(arrays[f"termination_{bc}"][s, env]))
                stats["any_support_contact_at_terminal"] += int((snorm[s, env] > 1.0).any())
    stats["nonforbidden_episodes_with_forb_force_nonzero_anywhere"] = sum(1 for k, v in forb_any.items() if info[k]["reason"] != "forbidden_contact")
    stats["forbidden_episodes_with_forb_force_nonzero_anywhere"] = sum(1 for k, v in forb_any.items() if info[k]["reason"] == "forbidden_contact")
    m = stats.pop("max_forb_force_terminal_N")
    stats["forb_force_terminal_N_quantiles"] = [round(float(x),1) for x in np.quantile(m, [0, 0.5, 0.9, 1.0])] if m else None
    stats["termination_term_order"] = term_order
    out[name] = stats
    print(name, json.dumps(stats), flush=True)
Path("/data/yuyangliu/p2_audit/probe_forbidden.json").write_text(json.dumps(out, indent=1))

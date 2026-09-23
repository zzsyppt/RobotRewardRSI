"""Recompute every real P2 archive with the CURRENT local source tree (read-only)."""
import json, sys, time, hashlib
from pathlib import Path
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_fix/src")
import torch; torch.set_num_threads(4)
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
from true_rl_agent.tasks.push_recovery.numeric_capture import recompute_push_numeric_evidence
R = Path("/data/yuyangliu/AGENT/feedback_p2_20260923")
report = {"source": "/data/yuyangliu/p2_audit/src_fix", "stair": {}, "push": {}}
for name in ("r2_parent_on", "r2_gen8_on", "r2_g1_on"):
    res = json.load(open(R/name/"public_dev_result.json"))
    ref = res["task_evidence"]["numeric_evidence"]
    t0 = time.time()
    try:
        rc = recompute_stair_numeric_evidence(ref)
        ok = rc["summary"] == res["task_evidence"]["summary"]
        report["stair"][name] = {"status": "PASS" if ok else "SUMMARY_MISMATCH", "episodes": rc["episode_count"], "seconds": round(time.time()-t0,1)}
    except Exception as exc:
        report["stair"][name] = {"status": "ERROR", "error": repr(exc), "seconds": round(time.time()-t0,1)}
    print(name, report["stair"][name], flush=True)
for name in ("r2_push_g1_v2_on", "r2_push_g1_v3_on", "r2_push_go2_v2_on", "r2_push_go2_v3_on"):
    refs = json.load(open(R/name/"numeric_references.json"))
    for ref in refs:
        t0 = time.time()
        try:
            rc = recompute_push_numeric_evidence(ref)
            key = f"{name}/run{rc['identity']['run_index']}"
            report["push"][key] = {"status": "PASS" if rc["matches_runtime"] else "MISMATCH", "checks": rc["checks"], "episodes": rc["episodes"], "seconds": round(time.time()-t0,1)}
        except Exception as exc:
            report["push"][ref["manifest_path"]] = {"status": "ERROR", "error": repr(exc)}
        print(key if 'key' in dir() else ref["manifest_path"], report["push"].get(key), flush=True)
Path("/data/yuyangliu/p2_audit/recompute_fix_v1.json").write_text(json.dumps(report, indent=1))
print("DONE", json.dumps({k: {kk: vv["status"] for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)}))

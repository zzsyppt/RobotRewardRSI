"""Recompute the NEW v2 archives (audit_20260924 root) with the fixed source tree (read-only)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_fix/src")
import torch; torch.set_num_threads(4)
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
from true_rl_agent.tasks.push_recovery.numeric_capture import recompute_push_numeric_evidence
R = Path("/data/yuyangliu/AGENT/feedback_p2_audit_20260924")
report = {"source": "/data/yuyangliu/p2_audit/src_fix", "stair": {}, "push": {}}
for name in ("stair_parent_on", "stair_g1_on"):
    res = json.load(open(R/name/"public_dev_result.json"))
    ref = res["task_evidence"]["numeric_evidence"]
    t0 = time.time()
    try:
        rc = recompute_stair_numeric_evidence(ref)
        ok = rc["summary"] == res["task_evidence"]["summary"]
        report["stair"][name] = {"status": "PASS" if ok else "SUMMARY_MISMATCH", "episodes": rc["episode_count"], "profile": rc.get("profile"),
                                 "contact_force_history": rc.get("contact_force_history"), "seconds": round(time.time()-t0,1)}
    except Exception as exc:
        report["stair"][name] = {"status": "ERROR", "error": repr(exc), "seconds": round(time.time()-t0,1)}
    print(name, report["stair"][name], flush=True)
for name in ("push_g1_v3_on", "push_go2_v2_on"):
    for ref in json.load(open(R/name/"numeric_references.json")):
        t0 = time.time(); key = ref["manifest_path"]
        try:
            rc = recompute_push_numeric_evidence(ref)
            key = f"{name}/run{rc['identity']['run_index']}"
            report["push"][key] = {"status": "PASS" if rc["matches_runtime"] else "MISMATCH", "checks": rc["checks"], "episodes": rc["episodes"],
                                   "direction_status": rc.get("direction_status"), "seconds": round(time.time()-t0,1)}
        except Exception as exc:
            report["push"][key] = {"status": "ERROR", "error": repr(exc)}
        print(key, report["push"][key], flush=True)
Path("/data/yuyangliu/p2_audit/recompute_fix_v2.json").write_text(json.dumps(report, indent=1, default=str))
print("DONE_V2", json.dumps({k: {kk: vv["status"] for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)}))

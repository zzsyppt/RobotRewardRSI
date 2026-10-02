import json, sys, io
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
z = np.load("stair_trunk.npz"); T = z["data"].astype(np.float64); pts = json.loads(str(z["points"])); eps = json.loads(str(z["episodes"]))
ref = {(e["run"], e["env"]): e for e in json.load(open("stair_modes_out.json"))["episodes"]}
def form(e):
    r = ref[(e["run"], e["env"])]
    if r["reason"] == "success": return "success"
    if r["inverted"]: return "A_rollover"
    if r["reason"] == "timeout": return "B_timeout"
    if r["reason"] == "forbidden_contact": return "C_contact_advancing" if r["stall_s"] < 0.5 else "D_contact_after_stall"
    return "E_other"
print("points:", pts)
def q(xs):
    xs = np.sort(np.asarray(xs)); n = len(xs)
    return f"n={n} min {xs[0]:.3f} q05 {xs[int(.05*n)]:.3f} q25 {xs[n//4]:.3f} med {np.median(xs):.3f} q75 {xs[3*n//4]:.3f}"
G = {}
for e in eps:
    G.setdefault(form(e), []).append(e)
bottom = [i for i, p in enumerate(pts) if p.startswith("bottom") or p.startswith("head")]
for fm in ("success", "C_contact_advancing", "D_contact_after_stall", "B_timeout", "A_rollover"):
    g = G[fm]
    end_min, ep_min, arg_end, ep_arg, neg_any = [], [], [], [], []
    last5_min = []
    for e in g:
        d = T[e["start"]:e["start"] + e["steps"]]
        end_min.append(d[-1].min()); arg_end.append(pts[int(d[-1].argmin())])
        m = d.min(1); ep_min.append(m.min()); ep_arg.append(pts[int(d[int(m.argmin())].argmin())])
        last5_min.append(d[-5:].min())
        neg_any.append((m < 0).mean())
    print(f"== {fm} n={len(g)}")
    print("   min clearance over all sample points at the last step:", q(end_min))
    print("   min over last 5 steps:", q(last5_min), "| share of episodes with any point below surface in last 5 steps:", f"{np.mean(np.array(last5_min) < 0):.2f}")
    print("   closest point at last step:", dict(Counter(arg_end).most_common(6)))
    print("   episode minimum (any step):", q(ep_min), "| share of episodes ever below surface:", f"{np.mean(np.array(ep_min) < 0):.2f}", "| fraction of steps below surface, median:", f"{np.median(neg_any):.4f}")
    print("   point at the episode minimum:", dict(Counter(ep_arg).most_common(6)))

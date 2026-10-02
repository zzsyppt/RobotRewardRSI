import json, sys, io, pickle
import numpy as np
from collections import Counter
from generic_describe import kmedoids, silhouette
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = pickle.load(open("stair_generic.pkl", "rb")); names = P["names"]; F = len(names); eps = P["eps"]; fails = P["fails"]; V = P["V"]
ref = {(e["run"], e["env"]): e for e in json.load(open("stair_modes_out.json"))["episodes"]}
def form(e):
    r = ref[(e["run"], e["env"])]
    if r["inverted"]: return "A_rollover"
    if r["reason"] == "timeout": return "B_timeout"
    if r["reason"] == "forbidden_contact": return "C_contact_advancing" if r["stall_s"] < 0.5 else "D_contact_after_stall"
    return "E_other"
forms = np.array([form(eps[i]) for i in fails]); reason = np.array([eps[i]["reason"] for i in fails])
Sall, Slast = V[:, :2 * F], V[:, 2 * F:]          # scores / tau (>=1 means beyond the 95% success level)
lab = lambda j: f"{names[j % F]} {'above' if j < F else 'below'}"
for fm in ("A_rollover", "B_timeout", "C_contact_advancing", "D_contact_after_stall"):
    m = forms == fm
    fl = (Slast[m] > 1).mean(0); order = np.argsort(-fl)[:8]
    print(f"== {fm} n={m.sum()}: flagged feature-signs in the last 2 s, median count {np.median((Slast[m] > 1).sum(1)):.0f}; none flagged in {np.mean((Slast[m] > 1).sum(1) == 0):.2f}")
    print("    ", "; ".join(f"{lab(j)} {fl[j]:.2f}" for j in order))
    fa = (Sall[m] > 1).mean(0); order = np.argsort(-fa)[:6]
    print("     whole episode:", "; ".join(f"{lab(j)} {fa[j]:.2f}" for j in order))
# hierarchical grouping: termination reason first (generic), then binary flagged pattern in last 2 s with Jaccard-like distance
B = np.concatenate([(Slast > 1), (Sall > 1)], 1).astype(float)
print("\nHierarchical: split by termination reason, then k-medoids on flagged patterns")
for rs in ("forbidden_contact", "timeout", "base_orientation_instability"):
    m = np.nonzero(reason == rs)[0]
    if len(m) < 10:
        continue
    for k in (2, 3):
        l, med, cost = min((kmedoids(B[m], k, seed=s) for s in range(10)), key=lambda r: r[2])
        print(f"  {rs} n={len(m)} k={k} silhouette {silhouette(B[m], l):.2f}:", [dict(Counter(forms[m][l == j]).most_common()) for j in range(k)])

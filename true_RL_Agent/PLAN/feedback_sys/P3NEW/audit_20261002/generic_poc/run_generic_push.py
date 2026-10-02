import json, sys, io
import numpy as np
from collections import Counter
from generic_describe import phase_bins, build_envelope, deviation, excursions, kmedoids, silhouette
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
z = np.load("push_steps.npz"); D = z["data"].astype(np.float64); cols = json.loads(str(z["columns"])); eps = json.loads(str(z["episodes"]))
C = {c: i for i, c in enumerate(cols)}
n = len(eps)
ep = np.zeros(len(D), int); step = np.zeros(len(D), int); phase = np.zeros(len(D))
vpar = np.zeros(len(D)); vperp = np.zeros(len(D))
for i, e in enumerate(eps):
    s, T = e["start"], e["steps"]; sl = slice(s, s + T)
    ep[sl] = i; step[sl] = np.arange(1, T + 1)
    anchors = e["anchors"]; t = np.arange(1, T + 1)
    last = np.full(T, anchors[0])
    for a in anchors:
        last[t > a] = a
    phase[sl] = t - last                                        # steps since the most recent executed disturbance (negative before the first)
    dv = np.array(e["delta_v"][0][:2]); u = dv / (np.linalg.norm(dv) + 1e-9)
    vpar[sl] = D[sl, C["vx"]] * u[0] + D[sl, C["vy"]] * u[1]; vperp[sl] = -D[sl, C["vx"]] * u[1] + D[sl, C["vy"]] * u[0]
names = ["height", "roll", "pitch", "planar_speed", "angular_speed", "speed_along_push", "speed_across_push"]
X = np.column_stack([D[:, C["height"]], D[:, C["roll"]], D[:, C["pitch"]], D[:, C["planar_speed"]], D[:, C["angular_speed"]], vpar, vperp])
cond = np.array([e["stratum"] for e in eps]); success = np.array([e["reason"] == "success_recovered" for e in eps]); run = np.array([e["run"] for e in eps])
reason = np.array([e["reason"] for e in eps])
NB = 40; pb = phase_bins(phase, NB, -70, 345)
# split-half check of the false-flag rate
envA = build_envelope(X, ep, pb, cond, success, run <= 2, n_bins=NB, min_steps=150)
scale = np.nanmedian(envA["hi"] - envA["lo"], axis=(0, 1)); scale[scale <= 0] = 1.0
devA = deviation(X, pb, envA, scale)
def flagged(dev, idx):
    out = np.zeros((len(idx), 2 * X.shape[1]), bool)
    for r, i in enumerate(idx):
        e = eps[i]; ex = excursions(dev[e["start"]:e["start"] + e["steps"]])
        for f, sign, on, ln, pk in ex:
            out[r, 2 * f + (0 if sign > 0 else 1)] = True
    return out
held = np.nonzero(success & (run == 3))[0]; fails = np.nonzero(~success)[0]
fh = flagged(devA, held); ff = flagged(devA, fails)
print(f"PUSH generic envelope (fit: successes of runs 1-2, n={int((success & (run <= 2)).sum())}); held-out successes n={len(held)}, failures n={len(fails)}")
print(f"  episodes with any sustained excursion: held-out successes {fh.any(1).mean():.2f}, failures {ff.any(1).mean():.2f}")
for lab in ("not_recovered", "fall_forward", "fall_side", "height_collapse"):
    m = reason[fails] == lab
    print(f"    {lab:16s} n={m.sum():3d} any excursion {ff[m].any(1).mean():.2f}")
print("  per feature/sign: share of failures flagged vs held-out successes flagged")
for f, nm in enumerate(names):
    for k, sg in ((0, "above"), (1, "below")):
        a, b = ff[:, 2 * f + k].mean(), fh[:, 2 * f + k].mean()
        if a > 0.05 or b > 0.05:
            print(f"    {nm:18s} {sg:5s} failures {a:.2f}  held-out successes {b:.2f}   not_recovered {ff[reason[fails]=='not_recovered', 2*f+k].mean():.2f}")
# final envelope from all successes; signatures for failures
env = build_envelope(X, ep, pb, cond, success, np.ones(n, bool), n_bins=NB, min_steps=150)
dev = deviation(X, pb, env, scale)
V = []
for i in fails:
    e = eps[i]; d = dev[e["start"]:e["start"] + e["steps"]]
    pos = np.clip(d, 0, None); neg = np.clip(-d, 0, None)
    V.append(np.concatenate([np.minimum(pos.max(0), 5), np.minimum(neg.max(0), 5), (pos > 0).mean(0) * 5, (neg > 0).mean(0) * 5]))
V = np.array(V)
print("  clustering failures on deviation signatures:")
for k in (2, 3, 4, 5):
    best = min((kmedoids(V, k, seed=s) for s in range(8)), key=lambda r: r[2])
    lab = best[0]
    print(f"   k={k} silhouette {silhouette(V, lab):.2f}", [dict(Counter(reason[fails][lab == j]).most_common()) for j in range(k)])

import json, sys, io, pickle
import numpy as np
from collections import Counter, defaultdict
from generic_describe import phase_bins, build_envelope, deviation, kmedoids, silhouette
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
z = np.load("stair_steps.npz"); D = z["data"].astype(np.float64); cols = json.loads(str(z["columns"])); eps = json.loads(str(z["episodes"]))
C = {c: i for i, c in enumerate(cols)}; n = len(eps); N = len(D)
ep = np.zeros(N, int); tstep = np.zeros(N, int)
for i, e in enumerate(eps):
    ep[e["start"]:e["start"] + e["steps"]] = i; tstep[e["start"]:e["start"] + e["steps"]] = np.arange(e["steps"])
dt = 0.02
def diff_in_episode(v, k=5):
    out = np.zeros_like(v)
    for e in eps:
        s, T = e["start"], e["steps"]; x = v[s:s + T]
        d = np.zeros(T); d[k:] = (x[k:] - x[:-k]) / (k * dt); d[:k] = d[k] if T > k else 0.0
        out[s:s + T] = d
    return out
qw, qx, qy, qz = (D[:, C[c]] for c in ("qw", "qx", "qy", "qz"))
bx = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz)], 1)
bz = np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1)
heading = np.arctan2(bx[:, 1], bx[:, 0]); c, s = np.cos(-heading), np.sin(-heading)
lean_f = c * bz[:, 0] - s * bz[:, 1]; lean_l = s * bz[:, 0] + c * bz[:, 1]
tilt = np.arccos(np.clip(bz[:, 2], -1, 1))
fdx = D[:, [C[f"foot_dx_{k}"] for k in range(4)]]; fdy = D[:, [C[f"foot_dy_{k}"] for k in range(4)]]
flong = c[:, None] * fdx - s[:, None] * fdy                                     # foot offset along body heading
fvert = D[:, [C[f"foot_z_{k}"] for k in range(4)]] - D[:, [C["base_height"]]]
fz = D[:, [C[f"foot_fz_{k}"] for k in range(4)]]; loaded = fz > 1.0
fclr = D[:, [C[f"foot_clr_{k}"] for k in range(4)]]; fspd = D[:, [C[f"foot_speed_{k}"] for k in range(4)]]
feat = {
    "forward_speed": diff_in_episode(D[:, C["progress"]]),
    "lateral_speed": diff_in_episode(D[:, C["lateral"]]),
    "vertical_speed": diff_in_episode(D[:, C["base_height"]]),
    "lean_forward": lean_f, "lean_left": lean_l, "up_axis_z": bz[:, 2],
    "heading": heading, "yaw_rate": diff_in_episode(np.unwrap(heading)), "tilt_rate": diff_in_episode(tilt),
    "base_clearance": D[:, C["base_clearance"]], "course_offset": D[:, C["lateral"]],
    "support_count": loaded.sum(1).astype(float), "total_vertical_force": fz.sum(1),
    "min_foot_clearance": fclr.min(1), "max_foot_clearance": fclr.max(1),
    "max_loaded_foot_speed": np.where(loaded, fspd, 0).max(1),
    "front_feet_ahead": flong[:, :2].mean(1), "hind_feet_ahead": flong[:, 2:].mean(1),
    "front_feet_below_base": fvert[:, :2].mean(1), "hind_feet_below_base": fvert[:, 2:].mean(1),
    "left_right_height_diff": fvert[:, [0, 2]].mean(1) - fvert[:, [1, 3]].mean(1),
}
for k in range(12):
    feat[f"joint_{k}"] = D[:, C[f"joint_{k}"]]
# generic multi-timescale rates and a dwell clock derived only from the task's phase variable
feat["forward_speed_0p5s"] = diff_in_episode(D[:, C["progress"]], 25)
feat["forward_speed_2s"] = diff_in_episode(D[:, C["progress"]], 100)
stall = np.zeros(N)
for e in eps:
    s0, T = e["start"], e["steps"]; x = D[s0:s0 + T, C["progress"]]; rm = -1e9; last = 0
    for i in range(T):
        if x[i] > rm + 1e-4:
            rm, last = x[i], i
        stall[s0 + i] = (i - last) * dt
feat["time_since_progress"] = stall
names = list(feat); X = np.column_stack([feat[k] for k in names]); F = len(names)
goal = np.array([e["goal_m"] for e in eps])[ep]
phase = D[:, C["progress"]] / goal
cond = np.array([e["stratum"] for e in eps]); reason = np.array([e["reason"] for e in eps]); success = reason == "success"; run = np.array([e["run"] for e in eps])
NB = 24; pb = phase_bins(phase, NB, -0.05, 1.1)
def sustained(dev_ep, k=5):
    """Largest deviation held for k consecutive steps, per feature and sign -> [2F]."""
    T = dev_ep.shape[0]
    if T < k:
        return np.zeros(2 * dev_ep.shape[1])
    out = []
    for sign in (1, -1):
        v = np.clip(dev_ep * sign, 0, None)
        w = np.lib.stride_tricks.sliding_window_view(v, k, axis=0).min(-1)
        out.append(w.max(0))
    return np.concatenate(out)
def scores(dev, idx, last=None):
    S = np.zeros((len(idx), 2 * F))
    for r, i in enumerate(idx):
        e = eps[i]; d = dev[e["start"]:e["start"] + e["steps"]]
        S[r] = sustained(d if last is None else d[-last:])
    return S
# three-way split: fit run 1, calibrate run 2, test run 3
env1 = build_envelope(X, ep, pb, cond, success, run == 1, n_bins=NB)
scale = np.nanmedian(env1["hi"] - env1["lo"], axis=(0, 1)); scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
dev1 = deviation(X, pb, env1, scale)
cal = np.nonzero(success & (run == 2))[0]; tst = np.nonzero(success & (run == 3))[0]; fails = np.nonzero(~success)[0]
tau = np.quantile(scores(dev1, cal), 0.95, axis=0) + 1e-6
St, Sf = scores(dev1, tst), scores(dev1, fails)
ft, ff = St > tau, Sf > tau
print(f"STAIR generic envelope: fit {int((success & (run == 1)).sum())} successes (run 1), calibrate {len(cal)} (run 2), test {len(tst)} (run 3); failures {len(fails)}; features {F}")
print(f"  flagged feature-signs per episode: test successes median {np.median(ft.sum(1)):.0f} (mean {ft.sum(1).mean():.1f}); failures median {np.median(ff.sum(1)):.0f} (mean {ff.sum(1).mean():.1f})")
print(f"  episodes with >=3 flagged feature-signs: test successes {np.mean(ft.sum(1) >= 3):.2f}; failures {np.mean(ff.sum(1) >= 3):.2f}")
order = np.argsort(-(ff.mean(0) - ft.mean(0)))
print("  most discriminating feature-signs (failures flagged / held-out successes flagged):")
for j in order[:14]:
    print(f"    {names[j % F]:24s} {'above' if j < F else 'below'}  {ff[:, j].mean():.2f} / {ft[:, j].mean():.2f}")
# descriptive pass: fit runs 1-2, calibrate run 3
env = build_envelope(X, ep, pb, cond, success, run <= 2, n_bins=NB)
dev = deviation(X, pb, env, scale)
tau2 = np.quantile(scores(dev, tst), 0.95, axis=0) + 1e-6
Sall = np.minimum(scores(dev, fails) / tau2, 6); Slast = np.minimum(scores(dev, fails, last=100) / tau2, 6)
V = np.concatenate([Sall, Slast], 1)
pickle.dump({"names": names, "dev": dev.astype(np.float32), "tau": tau2, "eps": eps, "fails": fails, "X": X.astype(np.float32), "env": env, "pb": pb, "V": V, "scale": scale}, open("stair_generic.pkl", "wb"))
# reference forms from the task-specific audit rules
ref = {(e["run"], e["env"]): e for e in json.load(open("stair_modes_out.json"))["episodes"]}
def form(e):
    r = ref[(e["run"], e["env"])]
    if r["inverted"]: return "A_rollover"
    if r["reason"] == "timeout": return "B_timeout"
    if r["reason"] == "forbidden_contact": return "C_contact_advancing" if r["stall_s"] < 0.5 else "D_contact_after_stall"
    return "E_other"
forms = np.array([form(eps[i]) for i in fails])
print("  clustering failures on generic deviation signatures (no outcome label, no task rule):")
for k in (3, 4, 5, 6):
    lab, med, cost = min((kmedoids(V, k, seed=s) for s in range(10)), key=lambda r: r[2])
    purity = sum(Counter(forms[lab == j]).most_common(1)[0][1] for j in range(k) if (lab == j).any()) / len(fails)
    print(f"   k={k} silhouette {silhouette(V, lab):.2f} purity vs audit forms {purity:.2f}")
    for j in range(k):
        print("      ", int((lab == j).sum()), dict(Counter(forms[lab == j]).most_common()), "| reasons", dict(Counter(reason[fails][lab == j]).most_common()))
    if k == 4:
        pickle.dump({"lab": lab, "med": med}, open("stair_generic_k4.pkl", "wb"))

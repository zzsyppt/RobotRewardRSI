# Audit probe (local, CPU, no model call): what generic contact / gait / support descriptors say about the sealed
# R4 Stair cohort. Inputs are two read-only server dumps:
#   stair_steps.npz     <- P3NEW/audit_20261002/generic_poc/server_stair_dump.py
#   stair_contacts.npz  <- server_stair_contacts.py (production contact flags: foot force norm > task threshold)
# Nothing here is task-specific except the phase variable (progress) and the terrain level index used for reporting.
import json, sys, io
import numpy as np
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
z = np.load("stair_steps.npz"); D = z["data"].astype(np.float64); cols = json.loads(str(z["columns"])); eps = json.loads(str(z["episodes"]))
zc = np.load("stair_contacts.npz"); Dc = zc["data"].astype(np.float64); epc = json.loads(str(zc["episodes"]))
assert [(e["run"], e["env"], e["steps"]) for e in eps] == [(e["run"], e["env"], e["steps"]) for e in epc]
C = {c: i for i, c in enumerate(cols)}; dt = 0.02; FEET = ["FL", "FR", "RL", "RR"]; OUT = {}
n_by = Counter(e["reason"] for e in eps); print("episodes", len(eps), dict(n_by))

def runs(m):
    out = []; s = 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i] != m[s]:
            out.append((s, i - s, m[s])); s = i
    return out

def ep(e):
    s = slice(e["start"], e["start"] + e["steps"]); X = D[s]; Xc = Dc[s]
    qw, qx, qy, qz = (X[:, C[c]] for c in ("qw", "qx", "qy", "qz"))
    heading = np.arctan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz)); c, s_ = np.cos(-heading), np.sin(-heading)
    dx = X[:, [C[f"foot_dx_{k}"] for k in range(4)]]; dy = X[:, [C[f"foot_dy_{k}"] for k in range(4)]]
    return dict(T=e["steps"], L=Xc[:, 0:4] > 0.5, fn=Xc[:, 4:8], fz=Xc[:, 8:12], prog=X[:, C["progress"]], upz=1 - 2 * (qx * qx + qy * qy),
                heading=heading, fwd=c[:, None] * dx - s_[:, None] * dy, left=s_[:, None] * dx + c[:, None] * dy,
                clr=X[:, [C[f"foot_clr_{k}"] for k in range(4)]], spd=X[:, [C[f"foot_speed_{k}"] for k in range(4)]],
                lvl=X[:, [C[f"foot_level_{k}"] for k in range(4)]], forb=X[:, C["forbidden_force"]], bclr=X[:, C["base_clearance"]], bh=X[:, C["base_height"]])

E = [ep(e) for e in eps]
succ = [i for i, e in enumerate(eps) if e["reason"] == "success"]

# ------------------------------------------------------------------ A. how the successful episodes move
L = np.concatenate([E[i]["L"] for i in succ]); on = np.concatenate([E[i]["lvl"].max(1) > 0 for i in succ])
fn = np.concatenate([E[i]["fn"] for i in succ])
per_duty = np.array([E[i]["L"].mean(0) for i in succ])
code = (L * np.array([8, 4, 2, 1])).sum(1); cnt = Counter(code.tolist())
swing, period, phase = [], [[] for _ in range(4)], [[] for _ in range(4)]
for i in succ:
    M = E[i]["L"]; td = []
    for k in range(4):
        r = runs(M[:, k]); swing.extend(n * dt for (s0, n, v) in r[1:-1] if not v)
        t = np.flatnonzero(M[1:, k] & ~M[:-1, k]) + 1
        t = t[np.concatenate([[True], np.diff(t) > 5])] if len(t) else t          # ignore re-contact within 0.1 s
        td.append(t); period[k].extend((np.diff(t) * dt).tolist())
    for k in range(1, 4):
        for a, b in zip(td[0][:-1], td[0][1:]):
            ins = td[k][(td[k] >= a) & (td[k] < b)]
            if len(ins) == 1 and b - a > 10:
                phase[k].append((ins[0] - a) / (b - a))
swing = np.array(swing)
fwd_s = np.nanmedian(np.array([E[i]["fwd"][E[i]["lvl"].max(1) > 0].mean(0) for i in succ if (E[i]["lvl"].max(1) > 0).sum() > 20]), 0)
left_s = np.nanmedian(np.array([E[i]["left"][E[i]["lvl"].max(1) > 0].mean(0) for i in succ if (E[i]["lvl"].max(1) > 0).sum() > 20]), 0)
head = np.degrees(np.concatenate([E[i]["heading"][E[i]["lvl"].max(1) > 0] for i in succ]))
circ = lambda p: (float(np.angle(np.mean(np.exp(2j * np.pi * np.array(p)))) / (2 * np.pi) % 1), float(abs(np.mean(np.exp(2j * np.pi * np.array(p))))))
A = dict(episodes=len(succ), steps=int(len(L)),
         duty_stairs=L[on].mean(0).round(3).tolist(), duty_flat=L[~on].mean(0).round(3).tolist(),
         impulse_share=(fn.mean(0) / fn.mean(0).sum()).round(3).tolist(),
         hind_left_lowest_duty_share=float(np.mean(per_duty.argmin(1) == 2)), hind_left_duty_below_half_share=float(np.mean(per_duty[:, 2] < 0.5)),
         modes={format(m, "04b"): round(c / len(code), 3) for m, c in cnt.most_common(6)}, flight=round(cnt[0] / len(code), 3),
         agreement={f"{FEET[a]}-{FEET[b]}": round(float(np.mean(L[:, a] == L[:, b])), 2) for a in range(4) for b in range(a + 1, 4)},
         touchdown_period_s=[round(float(np.median(p)), 2) for p in period],
         touchdown_phase_vs_FL={FEET[k]: [round(v, 2) for v in circ(phase[k])] for k in range(1, 4)},
         swing_s_q25_50_75=np.quantile(swing, [.25, .5, .75]).round(2).tolist(), swing_shorter_than_quarter_second=round(float(np.mean(swing < 0.25)), 3),
         foot_forward_of_trunk_m=fwd_s.round(2).tolist(), foot_left_of_trunk_m=left_s.round(2).tolist(),
         heading_deg_q10_50_90=np.quantile(head, [.1, .5, .9]).round(1).tolist())
OUT["A_success_gait"] = A
print("\nA. successful episodes (%d, %d steps), feet FL FR RL RR" % (A["episodes"], A["steps"]))
for k, v in A.items():
    if k not in ("episodes", "steps"):
        print("   %-34s %s" % (k, v))

# ------------------------------------------------------------------ B. what the time-outs are doing; single-foot unloaded runs
UP = 0.82
tout = Counter(); held = defaultdict(list)
for i, e in enumerate(eps):
    X = E[i]
    ok = (X["upz"] >= UP) & (X["forb"] <= 1.0); best = (0, -1, 0)
    for k in range(4):
        for (s0, n, v) in runs((~X["L"][:, k]) & ok):
            if v and n > best[0]:
                best = (n, k, s0)
    if best[0] * dt >= 1.0:
        n, k, s0 = best; seg = slice(s0, s0 + n); oth = [j for j in range(4) if j != k]
        held[e["reason"]].append(dict(run=e["run"], env=e["env"], stratum=e["stratum"], foot=FEET[k], dur=n * dt, to_end=(X["T"] - s0 - n) * dt,
                                      clr=float(X["clr"][seg, k].mean()), clr_loaded=float(np.median(X["clr"][:, k][X["L"][:, k]])) if X["L"][:, k].any() else None,
                                      spd=float(X["spd"][seg, k].mean()), dprog=float(X["prog"][s0 + n - 1] - X["prog"][s0]),
                                      others_td=float(sum(sum(1 for (a, b, v) in runs(X["L"][seg, j])[1:] if v) for j in oth) / (n * dt)),
                                      levels=X["lvl"][s0 + n - 1].astype(int).tolist()))
    if e["reason"] == "timeout":
        M = X["L"][-150:]; duty = M.mean(0); td = sum(sum(1 for (a, b, v) in runs(M[:, k])[1:] if v) for k in range(4)) / 3.0
        dprog = X["prog"][-1] - X["prog"][-150]
        if duty.min() < 0.10 and np.sort(duty)[1] > 0.5: f = "standing, one %s foot unloaded" % ("hind" if duty.argmin() >= 2 else "front")
        elif td < 1.0 and duty.min() > 0.8: f = "standing on four feet"
        elif td >= 1.0 and abs(dprog) < 0.10: f = "stepping without progress"
        elif abs(dprog) >= 0.10: f = "still advancing"
        else: f = "other"
        tout[f] += 1
long2 = {r: [h for h in held[r] if h["dur"] >= 2.0] for r in held}
B = dict(timeouts_last_3s=dict(tout),
         upright_single_foot_unloaded_ge_1s={r: f"{len(held[r])}/{n_by[r]}" for r in n_by},
         upright_single_foot_unloaded_ge_2s={r: f"{len(long2.get(r, []))}/{n_by[r]}" for r in n_by})
all2 = [h for r in long2 for h in long2[r]]
B["ge_2s_runs"] = dict(n=len(all2), hind=sum(h["foot"] in ("RL", "RR") for h in all2), median_dur_s=float(np.median([h["dur"] for h in all2])),
                       last_until_end=sum(h["to_end"] < 0.1 for h in all2),
                       foot_at_surface=sum(h["clr"] < 0.04 for h in all2), foot_slow=sum(h["spd"] <= 0.1 for h in all2),
                       no_progress=sum(abs(h["dprog"]) < 0.05 for h in all2), others_not_stepping=sum(h["others_td"] < 1.0 for h in all2),
                       front_on_first_step_hind_on_ground=sum(h["levels"] == [1, 1, 0, 0] for h in all2))
OUT["B_stall"] = B; OUT["B_runs_ge_2s"] = all2
print("\nB. time-outs and single-foot unloaded runs")
for k, v in B.items():
    print("   %-36s %s" % (k, v))

# ------------------------------------------------------------------ C. weight-support accounting
by = defaultdict(list)
for i in succ:
    by[eps[i]["stratum"]].append(float(E[i]["fz"].sum(1).mean()))
W = float(np.median(by["c0_id"]))
Cc = dict(weight_proxy_N=round(W, 1), mean_foot_vertical_force_by_stratum_N={k: round(float(np.median(v)), 1) for k, v in sorted(by.items())})
static = []
for i, e in enumerate(eps):
    X = E[i]
    if X["T"] < 300: continue
    seg = slice(X["T"] - 250, X["T"])
    if abs(X["prog"][-1] - X["prog"][-250]) < 0.05 and np.abs(np.diff(X["bh"][seg])).max() / dt < 0.25 and X["upz"][seg].min() > UP:
        static.append(dict(run=e["run"], env=e["env"], stratum=e["stratum"], reason=e["reason"], feet_share=float(X["fz"][seg].sum(1).mean() / W),
                           base_force_max=float(X["forb"][seg].max()), feet_loaded=int((X["L"][seg].mean(0) > 0.9).sum())))
Cc["static_last_5s_episodes"] = len(static)
Cc["static_feet_share_below_0p7"] = [(s["run"], s["env"], s["stratum"], round(s["feet_share"], 2), s["base_force_max"], s["feet_loaded"]) for s in static if s["feet_share"] < 0.7]
Cc["static_feet_share_quantiles_rest"] = np.quantile([s["feet_share"] for s in static if s["feet_share"] >= 0.7], [.1, .5, .9]).round(2).tolist()
OUT["C_support"] = Cc
print("\nC. weight-support accounting")
for k, v in Cc.items():
    print("   %-40s %s" % (k, v))

# ------------------------------------------------------------------ D. full-coverage regime timeline from state alone
def regimes(X, win=50):
    T = X["T"]; M = X["L"]; lab = np.empty(T, dtype=object); td = np.zeros((T, 4)); td[1:] = (M[1:] & ~M[:-1]); half = win // 2
    c_td = np.vstack([np.zeros((1, 4)), np.cumsum(td, 0)]); c_L = np.vstack([np.zeros((1, 4)), np.cumsum(M, 0)])
    for t in range(T):
        a, b = max(0, t - half), min(T, t + half)
        rate = (c_td[b] - c_td[a]).sum() / ((b - a) * dt); duty = (c_L[b] - c_L[a]) / (b - a); v = (X["prog"][b - 1] - X["prog"][a]) / ((b - a) * dt)
        if X["upz"][t] < 0.5: lab[t] = "tipped over"
        elif X["forb"][t] > 1.0: lab[t] = "trunk loaded"
        elif duty.min() == 0 and np.sort(duty)[1] > 0.5: lab[t] = "standing, %s foot unloaded" % FEET[int(duty.argmin())]
        elif rate >= 4 and v > 0.1: lab[t] = "stepping and advancing"
        elif rate >= 4: lab[t] = "stepping without advancing"
        elif duty.min() > 0.8 and abs(v) <= 0.1: lab[t] = "standing on four feet"
        elif v > 0.1: lab[t] = "advancing with few steps"
        else: lab[t] = "unclassified"
    out = []
    for (s0, n, v) in runs(lab):
        if out and (n * dt < 0.3 or out[-1][2] == v):
            out[-1] = (out[-1][0], out[-1][1] + n, out[-1][2])
        else:
            out.append((s0, n, v))
    return out
R = [regimes(X) for X in E]
ns = np.array([len(r) for r in R[:]]); ok_ = np.array([e["reason"] == "success" for e in eps])
share = defaultdict(Counter)
for i, r in enumerate(R):
    for (s0, n, v) in r:
        share["success" if ok_[i] else "failure"]["standing, one foot unloaded" if "foot unloaded" in v else v] += n
Dd = dict(regimes_per_episode=dict(success_median=float(np.median(ns[ok_])), success_q90=float(np.quantile(ns[ok_], .9)),
                                   failure_median=float(np.median(ns[~ok_])), failure_q90=float(np.quantile(ns[~ok_], .9))),
          time_share={g: {k: round(v / sum(share[g].values()), 3) for k, v in share[g].most_common()} for g in share})
OUT["D_timeline"] = Dd
print("\nD. regime timeline (every step labelled, no model call)")
for k, v in Dd.items():
    print("   %-22s %s" % (k, v))
for (run, env) in [(2, 68), (1, 129), (1, 204), (1, 134)]:
    i = next(j for j, e in enumerate(eps) if e["run"] == run and e["env"] == env); e = eps[i]; X = E[i]
    print(f"   r{run}.e{env} [{e['stratum']}, {e['reason']}, {X['T'] * dt:.1f} s, progress {X['prog'][-1]:.2f}/{e['goal_m']:.2f} m]")
    for (s0, n, v) in R[i]:
        seg = slice(s0, s0 + n)
        print(f"      {s0 * dt:5.1f}-{(s0 + n) * dt:5.1f} s  {v:30s} progress {X['prog'][s0]:+.2f} -> {X['prog'][s0 + n - 1]:+.2f} m; feet carry {X['fz'][seg].sum(1).mean() / W:.2f} of weight; upright min {X['upz'][seg].min():.2f}")
json.dump(OUT, open("stair_behavior_probe_out.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ E. two descriptor checks tied to reward logic
# (1) air time at touchdown versus the 0.5 s threshold used by the anchor's feet_air_time term (control-step sampling)
# (2) hind knee (calf joint) held at its extension limit (Go2 URDF upper limit -0.8378 rad; not present in the robot manifest)
air = []
for i in succ:
    M = E[i]["L"]
    for k in range(4):
        air.extend(n * dt for (s0, n, v) in runs(M[:, k])[1:-1] if not v)
air = np.array(air)
J = D[:, [C[f"joint_{k}"] for k in range(8, 12)]]; CAP = -0.8378
def at_limit(sel, part):
    out = []
    for i, e in enumerate(eps):
        if not sel(e): continue
        X = J[e["start"]:e["start"] + e["steps"]]; lv = E[i]["lvl"].max(1)
        m = (lv > 0) if part == "stairs" else (lv == 0) if part == "flat" else (np.arange(len(X)) >= len(X) - 150)
        if m.sum() >= 10: out.append((X[m] > CAP - 0.02).mean(0))
    return np.array(out)
ls, lf, lt = at_limit(lambda e: e["reason"] == "success", "stairs"), at_limit(lambda e: e["reason"] == "success", "flat"), at_limit(lambda e: e["reason"] == "timeout", "last")
Ee = dict(touchdowns=int(len(air)), air_time_s_median=float(np.median(air)), air_time_ge_half_second_share=float(np.mean(air >= 0.5)),
          mean_air_minus_threshold_s=float(np.mean(air - 0.5)),
          calf_at_extension_limit_share_success_stairs=np.median(ls, 0).round(2).tolist(), calf_at_extension_limit_share_success_flat=np.median(lf, 0).round(2).tolist(),
          calf_at_extension_limit_share_timeouts_last3s=np.median(lt, 0).round(2).tolist(),
          timeouts_with_a_hind_calf_at_limit_over_90pct=f"{int((lt[:, 2:] > 0.9).any(1).sum())}/{len(lt)}")
OUT["E_reward_links"] = Ee
print("\nE. descriptor checks tied to reward logic")
for k, v in Ee.items():
    print("   %-48s %s" % (k, v))
json.dump(OUT, open("stair_behavior_probe_out.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ F. time-outs: where they stop x what they do (last 3 s)
tab = Counter(); mid = []; adv_levels = []
for i, e in enumerate(eps):
    if e["reason"] != "timeout": continue
    X = E[i]; M = X["L"][-150:]; duty = M.mean(0); td = sum(sum(1 for (a, b, v) in runs(M[:, k])[1:] if v) for k in range(4)) / 3.0
    dprog = X["prog"][-1] - X["prog"][-150]; lv = X["lvl"][-1].astype(int).tolist(); share = float(X["fz"][-150:].sum(1).mean() / W)
    if abs(dprog) >= 0.10: where, what = "still advancing", "-"; adv_levels.append(lv)
    else:
        where = "front feet on step 1, hind feet on ground" if lv == [1, 1, 0, 0] else ("rough-surface cell, mid-stairs" if e["stratum"] == "c2_mechanism_ood" else "elsewhere")
        if duty.min() < 0.10 and np.sort(duty)[1] > 0.5: what = "three feet loaded, one %s foot unloaded" % ("hind" if duty.argmin() >= 2 else "front")
        elif td < 1.0 and duty.min() > 0.8: what = "four feet loaded, no stepping"
        elif td >= 1.0: what = "stepping in place"
        else: what = "other"
        if where != "front feet on step 1, hind feet on ground":
            mid.append((e["run"], e["env"], e["stratum"], lv, what, round(share, 2), float(X["forb"][-150:].max())))
    tab[(where, what)] += 1
print("\nF. time-outs: where x what (last 3 s)")
for k, v in sorted(tab.items(), key=lambda kv: -kv[1]):
    print("   %2d  %-42s %s" % (v, k[0], k[1]))
print("   still advancing, foot levels at the end:", adv_levels)
print("   not at the first step: (run, env, stratum, foot levels, form, feet share of weight, max base force N)")
for m in mid: print("      ", m)
OUT["F_timeouts"] = dict(table={f"{k[0]} | {k[1]}": v for k, v in tab.items()}, not_at_first_step=mid, still_advancing_levels=adv_levels)
json.dump(OUT, open("stair_behavior_probe_out.json", "w"), indent=1, default=float)

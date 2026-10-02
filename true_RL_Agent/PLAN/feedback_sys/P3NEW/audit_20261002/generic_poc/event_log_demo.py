import json, sys, io, pickle
import numpy as np
from generic_describe import excursions
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = pickle.load(open("stair_generic.pkl", "rb")); names = P["names"]; dev = P["dev"]; eps = P["eps"]; X = P["X"]; env = P["env"]; pb = P["pb"]; tau = P["tau"]; F = len(names)
z = np.load("stair_trunk.npz"); T = z["data"].astype(np.float64); pts = json.loads(str(z["points"]))
def log(run, envi, max_lines=12):
    i = next(k for k, e in enumerate(eps) if e["run"] == run and e["env"] == envi); e = eps[i]; s, n = e["start"], e["steps"]
    d = dev[s:s + n]; ex = excursions(d, min_len=5, min_peak=0.0)
    # keep excursions whose sustained peak exceeds the calibrated level, merge by feature/sign keeping the earliest onset that runs into the last 3 s
    keep = {}
    for f, sign, on, ln, pk in ex:
        j = f if sign > 0 else F + f
        if pk <= tau[j] or on + ln < n - 150:
            continue
        if (f, sign) not in keep or on < keep[(f, sign)][0]:
            keep[(f, sign)] = (on, ln, pk)
    rows = sorted(((on, f, sign, ln, pk) for (f, sign), (on, ln, pk) in keep.items()))
    print(f"episode r{run}.e{envi}: {e['reason']}, {n} steps ({n*0.02:.2f} s), cell {e['cell'].split('.')[-1]}")
    for on, f, sign, ln, pk in rows[:max_lines]:
        k = s + on; ci = env["cidx"][k]; lo, hi = env["lo"][ci, pb[k], f], env["hi"][ci, pb[k], f]
        seg = X[s + on:s + on + ln, f]; val = seg.max() if sign > 0 else seg.min()
        print(f"  {(n - on) * 0.02:5.2f} s before end: {names[f]} {'above' if sign > 0 else 'below'} the success range for {ln * 0.02:.2f} s "
              f"(reaches {val:+.2f}; successes at this phase {lo:+.2f}..{hi:+.2f})")
    if len(rows) > max_lines:
        print(f"  ... {len(rows) - max_lines} more features leave the success range later")
    if not rows:
        print("  no feature leaves the success range in the last 3 s")
    last = T[s + n - 1]; k = int(last.argmin())
    print(f"  at the end: closest trunk point {pts[k]} is {last[k]:.3f} m above the surface; forward speed (0.5 s) {X[s+n-1, names.index('forward_speed_0p5s')]:+.2f} m/s; feet loaded {int(X[s+n-1, names.index('support_count')])}")
log(2, 68); print(); log(2, 175); print(); log(1, 129)

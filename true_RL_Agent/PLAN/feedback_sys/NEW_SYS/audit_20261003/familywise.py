# Audit probe (local): episode-level (family-wise) false-flag rate of the success-envelope PoC, and what an
# episode-level calibration buys. Re-uses run_generic_stair.py definitions up to the three-way split.
import json, sys, io, re
src = open("run_generic_stair.py", encoding="utf-8").read()
head = src.split("# descriptive pass")[0]
head = head.replace('print(f"STAIR generic', 'pass  # print(f"STAIR generic').replace('print(f"  flagged', 'pass  # print(f"  flagged').replace('print(f"  episodes with', 'pass  # ')
head = head.split("order = np.argsort")[0]
exec(head)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8") if not isinstance(sys.stdout, io.TextIOWrapper) else sys.stdout
print("features", F, "feature-signs", 2 * F)
print("per feature-sign false-flag on held-out successes: median %.3f max %.3f" % (np.median(ft.mean(0)), ft.mean(0).max()))
for m in (1, 2, 3, 5):
    print(f"  held-out successes with >= {m} flagged feature-signs: {np.mean(ft.sum(1) >= m):.3f} | failures: {np.mean(ff.sum(1) >= m):.3f}")
# episode-level statistic: number of flagged feature-signs, and max normalised score
Sc = scores(dev1, cal)
def stat_count(S): return (S > tau).sum(1)
def stat_max(S): return (S / tau).max(1)
ref = {(e["run"], e["env"]): e for e in json.load(open("stair_modes_out.json"))["episodes"]}
def form(e):
    r = ref[(e["run"], e["env"])]
    if r["inverted"]: return "A_rollover"
    if r["reason"] == "timeout": return "B_timeout"
    if r["reason"] == "forbidden_contact": return "C_contact_advancing" if r["stall_s"] < 0.5 else "D_contact_after_stall"
    return "E_other"
forms = np.array([form(eps[i]) for i in fails])
# split calibration successes in two halves: one sets per-feature tau, the other the episode-level threshold
rng = np.random.default_rng(0); perm = rng.permutation(len(cal)); a, b = perm[: len(cal) // 2], perm[len(cal) // 2:]
tau_a = np.quantile(Sc[a], 0.95, axis=0) + 1e-6
for name, fn in (("count of flagged feature-signs", lambda S: (S > tau_a).sum(1)), ("max normalised score", lambda S: (S / tau_a).max(1))):
    thr = np.quantile(fn(Sc[b]), 0.95)
    fs = np.mean(fn(St) > thr); fa = fn(Sf) > thr
    print(f"episode-level statistic = {name}: threshold {thr:.2f}; held-out success false-flag {fs:.3f}; failures flagged {fa.mean():.3f}; by form",
          {k: f"{fa[forms == k].mean():.2f} (n {int((forms == k).sum())})" for k in sorted(set(forms))})

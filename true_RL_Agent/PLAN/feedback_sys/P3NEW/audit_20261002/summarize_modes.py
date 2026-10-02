"""Summarise the two read-only server probes (stair_modes_out.json, push_probe_out.json).

Audit artefact, 2026-10-02. The four Stair failure forms below are a deliberately
crude decision list used to show feasibility; they were not checked against video.
"""
import json, statistics as st, sys, io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).parent


def stair_form(e):
    if e["inverted"]:
        return "A_stall_then_rollover"
    if e["reason"] == "timeout":
        return "B_timeout_stalled"
    if e["reason"] == "forbidden_contact":
        return "C_base_contact_while_advancing" if e["stall_s"] < 0.5 else "D_base_contact_after_stall"
    return "E_other"


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


eps = json.loads((HERE / "stair_modes_out.json").read_text())["episodes"]
fail = [e for e in eps if e["reason"] != "success"]
succ = [e for e in eps if e["reason"] == "success"]
forms = defaultdict(list)
for e in fail:
    forms[stair_form(e)].append(e)
print(f"STAIR episodes {len(eps)}, failures {len(fail)}")
for name in sorted(forms):
    g = forms[name]
    print(f"  {name}: {len(g)} ({len(g)/len(fail):.0%}); strata {dict(Counter(e['stratum'] for e in g).most_common())}")
    print(f"     stall_s median {med([e['stall_s'] for e in g])}; advance in last 0.5 s median {med([e['adv_last_0p5s'] for e in g])} m; "
          f"most nose-up pitch on stairs median {med([e['pitch_min_on_stairs'] for e in g])} rad; "
          f"tilt>60deg direction {dict(Counter(e['tilt60_dir'] for e in g))}")
    print(f"     front-feet level at end {dict(sorted(Counter(e['front_level_end'] for e in g).items()))}; hind-feet level at end {dict(sorted(Counter(e['hind_level_end'] for e in g).items()))}")
print(f"  success: most nose-up pitch on stairs median {med([e['pitch_min_on_stairs'] for e in succ])} rad; "
      f"base clearance median-on-stairs median {med([e['base_clr_median_on_stairs'] for e in succ])} m")

push = json.loads((HERE / "push_probe_out.json").read_text())
groups = defaultdict(list)
for e in push:
    groups[e["label"]].append(e)
print(f"PUSH episodes {len(push)}: {dict(Counter(e['label'] for e in push))}")
print(f"  before any push: mean planar speed median {med([e['pre_speed_mean'] for e in push])} m/s; "
      f"episodes exceeding the speed bound before the push {sum((e['pre_frac_above'] or 0) > 0 for e in push)}/{len(push)}")
for name in ("success_recovered", "not_recovered"):
    g = groups[name]
    late = [e for e in g if e.get("late_steps")]
    print(f"  {name}: n={len(g)}; upright through every post-push step {sum(bool(e.get('upright_all')) for e in g)}; "
          f"after the last recovery window: exceed the speed bound again {sum(e['late_frac_above'] > 0 for e in late)}/{len(late)}, "
          f"time above bound median {med([e['late_frac_above'] for e in late])}, peak speed median {med([e['late_max'] for e in late])} m/s")

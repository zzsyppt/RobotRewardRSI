"""Apply the lead reviewer's explicit, hand-decided adjustments to the verified reviews.

Every edit below was decided by reading the item; this script only applies them and
recounts fact_support from the claims list (arithmetic). No judgement is automated.
"""
import json
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = json.loads(SRC.read_text(encoding="utf-8"))

H1 = ("lead reviewer (rule H1): correct fact carried forward from the author's own attempt-1 input of the "
      "same authoring episode, visible here as its prior source/rationale; counted as grounded.")
H2 = ("lead reviewer (rule H2): the number is the visible cell-mean progress; calling it a 'stall' is loose "
      "wording, not a fabricated observation (same treatment as other items citing mean progress).")
H3 = ("lead reviewer (rule H3): misreading of the guard/selector rule that does not bear on the diagnosis or "
      "the mechanism; kept as a contradicted claim, not a hard error.")


def claim_idx(v, needle):
    hits = [i for i, c in enumerate(v["claims"]) if needle in c["claim"]]
    if len(hits) > 1:
        hits = [i for i in hits if v["claims"][i]["type"] in ("measured_fact", "code_fact")]
    if len(hits) != 1:
        raise SystemExit(f"{v['blind_id']}: claim '{needle}' matched {hits}")
    return hits[0]


def drop_he(v, needle, htype):
    before = len(v["hard_errors"])
    v["hard_errors"] = [h for h in v["hard_errors"] if not (h["type"] == htype and needle in h["quote"])]
    if len(v["hard_errors"]) != before - 1:
        raise SystemExit(f"{v['blind_id']}: hard error '{needle}' ({htype}) not removed exactly once")


def note(v, i, text):
    c = v["claims"][i]
    c["note"] = (c["note"] + " | " if c["note"] else "") + text


def replace_notes(v, old, new):
    if old not in v["reviewer_notes"]:
        raise SystemExit(f"{v['blind_id']}: note text not found: {old}")
    v["reviewer_notes"] = v["reviewer_notes"].replace(old, new)


EDITS = {}


def edit(pid):
    def deco(fn):
        EDITS[pid] = fn
        return fn
    return deco


@edit("proposal-047")
def _(v):
    for needle in ("all eight other terms are carried over unchanged",
                   "forbidden_contact is the dominant failure at 0.131 overall",
                   "concentrated at c1_conditional_frontier 0.324",
                   "c1_conditional_frontier L4/L5 success 17/36"):
        i = claim_idx(v, needle)
        v["claims"][i]["grounded"] = True
        note(v, i, H1)
    replace_notes(v, "按 grounding 规则不算 grounded，事实支撑为 6/11",
                  "按统一口径（同一作者任务首轮输入中确有、并在本帧以作者上一版源码可见的正确数字计为有依据）计为 grounded，事实支撑为 10/11")
    for h in v["hard_errors"]:
        if "joint_ood L4/L5 mean progress" in h["quote"] and h["type"] == "contradicted_fact":
            h["type"] = "numeric"
    return ["H1: 4 carried-forward facts set grounded=True",
            "H3 boundary: level mislabel of visible data retyped contradicted_fact -> numeric (still a hard error)"]


@edit("proposal-042")
def _(v):
    i = claim_idx(v, "height_ood level_5 stalls at progress 0.589")
    v["claims"][i].update(truth="supported", grounded=True)
    note(v, i, H2)
    drop_he(v, "height_ood level_5 stalls at progress 0.589", "fabricated_observation")
    j = claim_idx(v, "so the 0.05 paired non-regression cap holds")
    note(v, j, H3)
    drop_he(v, "so the 0.05 paired non-regression cap holds", "contradicted_fact")
    return ["H2: 0.589 mean-progress claim -> supported/grounded, fabricated_observation removed",
            "H3: guard-rule misreading kept as contradicted claim, contradicted_fact hard error removed"]


@edit("proposal-024")
def _(v):
    replace_notes(v, "根部到足心约 0.06-0.09 m", "根部到足心的竖直距离估计远低于 0.21 m（仅凭录像粗估，未实测）")
    return ["notes: video-based height estimate marked as a rough estimate"]


H4 = ("lead reviewer (rule H4): a qualitative size word ('near zero') applied to small but non-zero visible "
      "counts, with no wrong number stated, is an imprecise claim, not a numeric hard error.")


@edit("proposal-041")
def _(v):
    for needle in ("Exactly one component mechanism changes relative to the reference",
                   "stationary_v3 taxes all sub-band velocity from zero",
                   "Public failure evidence: 51/192 not_recovered",
                   "planar_speed failing for 3291 not_recovered window steps",
                   "plus 34 episodes unstable after all recoveries latched",
                   "while roll and pitch bands fail in only 722 and 401 steps",
                   "raises high-stratum and repeated-stratum success_recovered above the reference"):
        i = claim_idx(v, needle)
        v["claims"][i]["grounded"] = True
        note(v, i, H1)
    replace_notes(v, "因此按本指南判为未落地（7/17）",
                  "按统一口径（同一作者任务首轮输入中确有、并在本帧以作者上一版理由可见的正确数字计为有依据）计为有依据（14/17）")
    return ["H1: 7 carried-forward facts set grounded=True"]


@edit("proposal-027")
def _(v):
    i = claim_idx(v, "while roll/pitch/height failures are near zero")
    note(v, i, H4)
    drop_he(v, "while roll/pitch/height failures are near zero", "numeric")
    return ["H4: 'near zero' wording kept as contradicted claim, numeric hard error removed"]


@edit("proposal-052")
def _(v):
    i = claim_idx(v, "Strata degrade with push count")
    note(v, i, H3.replace("guard/selector rule", "stratum design"))
    drop_he(v, "Strata degrade with push count", "contradicted_fact")
    return ["H3: peripheral stratum-design misreading kept as contradicted claim, hard error removed"]


for pid, fn in EDITS.items():
    v = R[pid]["verify"]
    changes = fn(v)
    v.setdefault("lead_reviewer_adjustments", []).extend(changes)

for pid, rec in R.items():
    v = rec.get("verify")
    if not v:
        continue
    facts = [c for c in v["claims"] if c["type"] in ("measured_fact", "code_fact")]
    recount = {"grounded": sum(1 for c in facts if c["grounded"] is True), "total": len(facts)}
    if recount != v["fact_support"]:
        v.setdefault("lead_reviewer_adjustments", []).append(
            f"fact_support recounted from claims: {v['fact_support']} -> {recount}")
        v["fact_support"] = recount

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("edited:", sorted(EDITS))

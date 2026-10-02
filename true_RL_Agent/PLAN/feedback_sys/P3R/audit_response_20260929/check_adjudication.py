"""Small offline arithmetic and scope checks, never modifies historical scores."""
import hashlib
import itertools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EXPORT = BASE / "results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    source = (EXPORT / "proposal-001.py").read_text(encoding="utf-8")
    assert "release = (tilt / 0.4).clamp(min=0.0, max=1.0) * 0.75" in source
    assert "keep = 1.0 - release" in source
    assert "return 1.0 - 0.25 * keep * (planar_cost + angular_cost)" in source
    # Both velocity ratios are .5, all postural bounds can hold at these points.
    costs = .5 * .5**2 + .5 * .5**2
    fixtures = []
    for roll, pitch in [(0, 0), (.1, 0), (.2, 0), (.2, .2)]:
        keep = 1 - .75 * min(math.hypot(roll, pitch) / .4, 1)
        fixtures.append({"roll_rad": roll, "pitch_rad": pitch,
                         "inside_inclusive_task_tilt_band": abs(roll) <= .2 and abs(pitch) <= .2,
                         "planar_scaled_speed": .04, "angular_scaled_speed": .25,
                         "keep": keep, "parent_rate": 1 - .25 * costs,
                         "candidate_rate": 1 - .25 * keep * costs})
    original = BASE.parent / "P3/results/author_final_bundle/author_blind_review"
    sheet = read(original / "reviewer/review_sheet.json")
    mapping = {r["blind_id"]: r for r in read(original / "withhold_from_reviewer.json")["mapping"]}
    proposals = {r["blind_id"]: r for r in sheet["proposals"]}
    seen, rows = set(), []
    hard_types = {"numeric", "fabricated_observation", "contradicted_fact", "causal_overreach", "code_misdescription"}
    for ident in sorted(proposals):
        m, p = mapping[ident], proposals[ident]
        if m["result_path"] in seen:
            continue
        seen.add(m["result_path"])
        rows.append({"case": m["case_id"], "arm": m["arm"],
                     "grounded": p["fact_support"]["grounded"], "total": p["fact_support"]["total"],
                     "hard_errors": sum(e["type"] in hard_types for e in p["hard_errors"])})
    paired = []
    for case in sorted({r["case"] for r in rows}):
        groups = {arm: [r for r in rows if r["case"] == case and r["arm"] == arm] for arm in ("A", "T")}
        rates = {a: sum(r["grounded"] for r in g) / sum(r["total"] for r in g) for a, g in groups.items()}
        hard = {a: sum(r["hard_errors"] for r in g) for a, g in groups.items()}
        paired.append({"case": case, "fact_rate_difference_T_minus_A": rates["T"] - rates["A"],
                       "hard_error_difference_T_minus_A": hard["T"] - hard["A"]})
    def exact_signflip(values):
        observed = abs(sum(values))
        return sum(abs(sum(v*s for v, s in zip(values, signs))) >= observed-1e-12
                   for signs in itertools.product((-1, 1), repeat=len(values))) / (2**len(values))
    out = {"schema_version": "feedback_p3r.audit_adjudication_arithmetic.v1", "api_calls": 0,
           "001_source_sha256": hashlib.sha256((EXPORT / "proposal-001.py").read_bytes()).hexdigest(),
           "001_method": "Independent scalar arithmetic with exact formula text checked; synthetic valid stable-band states, not measured trajectories or training outcomes",
           "001_fixtures": fixtures,
           "001_interruption_scope": {"not_recovered_total": 57, "episodes": 14, "planned_windows": 28,
                                      "per_episode": 57/14, "per_window": 57/28},
           "p3_original_scores_unchanged": True, "p3_unique_answers": len(rows), "p3_case_pairs": paired,
           "p3_case_level_exact_signflip_p": {k: exact_signflip([r[k] for r in paired]) for k in (
               "fact_rate_difference_T_minus_A", "hard_error_difference_T_minus_A")},
           "statistical_scope": "12 fixed questions from 2 observation packages/policy clusters; case exchangeability is an assumption, not 12 independent environments. Audit bootstrap samples whole answers independently by arm, without keeping paired cases/source clusters. No generalization or causal modality claim."}
    (Path(__file__).parent / "adjudication_arithmetic.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"stable_band_keep": [r["keep"] for r in fixtures],
                      "unique_answers": len(rows), "case_exact_p": out["p3_case_level_exact_signflip_p"]}))


if __name__ == "__main__":
    main()

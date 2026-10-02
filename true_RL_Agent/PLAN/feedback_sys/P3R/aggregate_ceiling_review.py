"""Aggregate sealed R.0-b model reviews after root-authorized unblinding only.

Reads the existing scores; no API, self-grading, training or new acceptance rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

VARIANTS = ("A", "T0", "Tc-num", "Tc-vis")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def unique_items(items, count):
    indexed = {item["proposal_id"]: item for item in items}
    if len(items) != count or len(indexed) != count:
        raise ValueError(f"expected {count} unique proposals; duplicates or missing terminals")
    return indexed


def qualifies(item, *, has_candidate):
    errors, hard = item["machine_errors"], item["hard_errors"]
    if not isinstance(hard, list):
        raise ValueError("hard_errors must be the original review's list")
    if errors is None:
        if has_candidate or hard:
            raise ValueError("unresolved error count cannot qualify a candidate or hide hard errors")
    elif type(errors) is not int or errors < 0 or errors != len(hard):
        raise ValueError("machine_errors must equal the number of original hard_errors")
    mechanism, action = item["mechanism_matches_registered_failure"], item["action_relevance_0_2"]
    if not has_candidate:
        if mechanism not in {None, "not_applicable"} or action is not None:
            raise ValueError("no-candidate terminals must not receive invented semantic scores")
        return False
    if item["machine_checks_complete"] is not True:
        raise ValueError("candidate fact/code checks are not complete")
    if mechanism not in {"yes", "no", "unjudgeable"} or type(action) is not int or action not in {0, 1, 2}:
        raise ValueError("review score is outside the frozen rule")
    return errors == 0 and mechanism == "yes" and action == 2


def summarize(items):
    variants = {}
    for variant in VARIANTS:
        rows = [row for row in items if row["variant"] == variant]
        passed = [row for row in rows if row["qualifies"]]
        cases = sorted({row["case_id"] for row in passed})
        variants[variant] = {"terminal_denominator": len(rows),
            "candidate_count": sum(row["has_candidate"] for row in rows),
            "terminal_outcomes": dict(Counter(row["terminal"] for row in rows)),
            "qualifying_proposals": [row["proposal_id"] for row in passed],
            "qualifying_proposal_count": len(passed), "qualifying_cases": cases,
            "distinct_qualifying_cases": len(cases), "effective": len(cases) >= 2}
    numeric, visual = (variants[name]["effective"] for name in ("Tc-num", "Tc-vis"))
    decision = ("CONTINUE_NUMERIC_PROCESS_TRACK" if numeric else
                "LIMITED_VISUAL_TRACK" if visual else "PAUSE_CONDITION_EXPANSION_STRONG_MODEL_NOT_TESTED")
    return {"variants": variants, "development_branch": decision,
        "branch_rule": "Tc-num effective: continue; only Tc-vis effective: bounded visual work; neither: pause condition expansion. A/T0 do not grant either Tc gate.",
        "stronger_author_probe": "NOT_ENABLED_NOT_TESTED",
        "formal_p3r_value_gate": "NOT_EVALUATED", "training_benefit": "NOT_EVALUATED"}


def aggregate(args):
    full_reviewer = args.full_export / "reviewer"
    mapping_path = args.full_export / "withhold_mapping.json"
    sheet_path = full_reviewer / "review_sheet.json"
    mapping = unique_items(read(mapping_path), 32)
    sheet = read(sheet_path)
    if sheet["profile"] != "model_review":
        raise ValueError("full export is not the frozen model_review profile")
    exported = unique_items(sheet["items"], 32)
    if mapping.keys() != exported.keys():
        raise ValueError("full export and mapping do not cover the same 32 proposals")
    cases = {row["case_id"] for row in mapping.values()}
    if len(cases) != 4:
        raise ValueError("expected four cases")
    slots = Counter((row["case_id"], row["variant"], row["repetition"]) for row in mapping.values())
    if slots != Counter({(case, variant, repetition): 1 for case in cases for variant in VARIANTS for repetition in (1, 2)}):
        raise ValueError("frozen mapping must have eight per case and two per case/variant")
    reviews, refs, reviewed_cases = {}, [], set()
    if len(args.reviews) != 4:
        raise ValueError("exactly four sealed case reviews are required")
    for path in args.reviews:
        review = read(path)
        if review["profile"] != "model_review" or review["status"] != "COMPLETED":
            raise ValueError("all four model reviews must be sealed COMPLETED before unblinding")
        case_items = unique_items(review["items"], 8)
        case_ids = {row["case_id"] for row in case_items.values()}
        if len(case_ids) != 1 or not case_ids <= cases or reviewed_cases & case_ids:
            raise ValueError("case reviews must cover each registered case once")
        reviewed_cases.update(case_ids)
        refs.append(file_ref(path))
        for ident, item in case_items.items():
            if ident in reviews or ident not in mapping:
                raise ValueError("review proposal coverage is duplicated or unexpected")
            original, bound = exported[ident], mapping[ident]
            if item["case_id"] != bound["case_id"] or item["case_id"] != original["case_id"] or item["terminal"] != original["terminal"]:
                raise ValueError("review case or terminal differs from the frozen full export")
            artifact_refs = {}
            for suffix in (".py", "-initial.json", "-seen.json", "-parent.py"):
                name = ident + suffix
                case_file, full_file = path.parent / name, full_reviewer / name
                if case_file.read_bytes() != full_file.read_bytes():
                    raise ValueError(f"reviewed artifact differs from full export: {name}")
                artifact_refs[name] = file_ref(full_file)["sha256"]
            has_candidate = bool((full_reviewer / f"{ident}.py").read_text(encoding="utf-8").strip())
            qualified = qualifies(item, has_candidate=has_candidate)
            reviews[ident] = {"proposal_id": ident, "case_id": item["case_id"], "variant": bound["variant"],
                "repetition": bound["repetition"], "terminal": item["terminal"], "has_candidate": has_candidate,
                "machine_checks_complete": item["machine_checks_complete"], "machine_errors": item["machine_errors"],
                "hard_errors": item["hard_errors"], "mechanism_matches_registered_failure": item["mechanism_matches_registered_failure"],
                "action_relevance_0_2": item["action_relevance_0_2"], "qualifies": qualified,
                "artifact_sha256": artifact_refs}
    if reviews.keys() != mapping.keys() or reviewed_cases != cases:
        raise ValueError("sealed reviews must cover all 32 terminals")
    items = [reviews[key] for key in sorted(reviews)]
    return {"schema_version": "feedback_p3r.ceiling_review_aggregate.v1", "status": "COMPLETE",
        "profile": "model_review", "api_calls": 0, "scores_recomputed": False,
        "review_refs": refs, "frozen_rules_ref": file_ref(args.rules), "mapping_ref": file_ref(mapping_path),
        "full_export_sheet_ref": file_ref(sheet_path), "aggregator_ref": file_ref(Path(__file__)),
        "terminal_denominator": 32, "case_denominator": 4,
        "candidate_count": sum(row["has_candidate"] for row in items),
        "qualification_rule": "candidate exists; action2; mechanism yes; fact/code checks complete; original hard errors0; >=2 distinct cases makes a variant effective",
        **summarize(items), "items": items,
        "limitations": ["Model review, not human adjudication or formal R.4 value acceptance.",
            "The two samples of one case are not independent cases; all failures remain in denominators.",
            "The stronger author was not enabled or tested; lack of Tc effect cannot establish its ineffectiveness."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs=4, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "profile", "terminal_denominator", "variants", "development_branch")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

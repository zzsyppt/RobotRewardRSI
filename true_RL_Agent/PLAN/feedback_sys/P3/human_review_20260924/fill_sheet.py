"""Transfer the consolidated per-item reviews into review_sheet.json (data entry only)."""
import hashlib
import json
import sys
from pathlib import Path

reviews = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sheet_path = Path(sys.argv[2])
filled_at = sys.argv[3]
guide_sha = sys.argv[4]
sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
before_sha = hashlib.sha256(sheet_path.read_bytes()).hexdigest()

for item in sheet["proposals"]:
    pid = item["blind_id"]
    rec = reviews[pid]
    v, first = rec["verify"], rec["review"]
    assert v["blind_id"] == pid and v["case_id"] == item["case_id"], pid
    item["claims"] = v["claims"]
    item["hard_errors"] = v["hard_errors"]
    item["action_relevance"] = v["action_relevance"]
    item["testability"] = v["testability"]
    item["authorization"] = v["authorization"]
    item["restraint"] = v["restraint"]
    item["reviewer_notes"] = v["reviewer_notes"]
    item["fact_support"] = v["fact_support"]
    item["score_reasons"] = {
        "action_relevance": v["action_relevance_reason"],
        "testability": v["testability_reason"],
        "restraint": v["restraint_reason"],
    }
    item["mechanism_matches_code"] = v["mechanism_matches_code"]
    item["focus_addressed"] = v["focus_addressed"]
    item["input_kind"] = v["input_kind"]
    item["behavior_block_visible"] = v["behavior_block_visible"]
    item["behavior_evidence_used"] = v["behavior_evidence_used"]
    item["uncertain_items"] = v["uncertain_items"]
    item["review_trace"] = {
        "first_pass": {
            "action_relevance": first["action_relevance"],
            "testability": first["testability"],
            "restraint": first["restraint"],
            "hard_error_types": [h["type"] for h in first["hard_errors"]],
            "fact_support": first["fact_support"],
        },
        "verifier_changes": v.get("changes") or [],
        "lead_reviewer_adjustments": v.get("lead_reviewer_adjustments") or [],
    }

sheet["reviewer"] = {
    "name": "Claude (claude-opus-5-5), acting as the single expert reviewer requested by the user",
    "procedure": (
        "Per item: one independent same-model review under a fixed calibration guide, then one adversarial "
        "verification pass; the lead reviewer read every verified item, applied four uniform consistency rules "
        "(H1-H4) and recorded each adjustment in review_trace. Duplicate items were reviewed independently and "
        "left as produced, so they measure test-retest consistency of this reviewer setup, not inter-rater agreement."
    ),
    "calibration_guide_sha256": guide_sha,
    "consistency_rules": {
        "H1": "Correct facts carried forward from the author's own attempt-1 input of the same authoring episode "
              "and visible in the repair frame as its prior source/rationale count as grounded.",
        "H2": "A visible cell-mean progress value described as a 'stall' counts as a supported number; the wording is noted.",
        "H3": "Misreadings of evaluation/selector/guard rules or stratum design that do not bear on the diagnosis or "
              "mechanism are contradicted claims, not hard errors. Misreadings of data values (wrong column, cell, "
              "unit, overlap treated as exclusive) remain numeric hard errors.",
        "H4": "Qualitative size words ('near zero') for small non-zero visible counts, with no wrong number stated, "
              "are imprecise claims, not numeric hard errors.",
    },
    "additional_truth": (
        "Reviewer-run native simulator replays of the same checkpoints, cohorts and source snapshots "
        "(Stair: 6 forbidden-contact, 1 timeout, 1 success episode; Push: 2 not_recovered, 1 fall_forward). "
        "Recorded outcomes matched the evidence files exactly. Frame grids and sidecars: "
        "PLAN/feedback_sys/P3/review_video_20260924/."
    ),
    "blinding_note": (
        "Arm labels and the withheld mapping were not opened before the sheet was completed. Blinding is nominal: "
        "grounding checks require reading seen-*.json, which shows whether a behavior block was present. One content "
        "search incidentally displayed a provider_candidates path matching proposal-018 (a lane directory name, no arm label)."
    ),
    "filled_at": filled_at,
}
sheet["status"] = "COMPLETED"
sheet_path.write_text(json.dumps(sheet, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
after_sha = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
print(json.dumps({"before_sha256": before_sha, "after_sha256": after_sha, "items": len(sheet["proposals"])}))
